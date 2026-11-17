"""New reward labels on the unchanged three-step proximal control scheduler."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import time

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.fix3_roles import CenteringReward
from numerical_experiment.picar_solver_repair.student import SolverDevelopmentRun
from . import VERSION
from .config import CONDITIONS, power
from .math import episode_labels


class _RewardStore:
    """Only the historical constructor sees its original empty scorer schema."""

    def __init__(self, store, condition, legacy_scorer):
        self.store, self.condition = store, condition
        self.legacy_scorer = legacy_scorer
        self.constructing = True
        self.saved = None

    def __getattr__(self, name):
        return getattr(self.store, name)

    def load_checkpoint(self, **kwargs):
        saved = self.store.load_checkpoint(**kwargs)
        if saved is None or not self.constructing:
            return saved
        self.saved = saved
        return {**saved, "scorer": copy.deepcopy(self.legacy_scorer)}

    def checkpoint(self, state, *, metrics=(), progress=None, **kwargs):
        return self.store.checkpoint(state,
            metrics=[{**r, "condition": self.condition} for r in metrics],
            progress={**(progress or {}), "condition": self.condition,
                "reward_condition": self.condition, "evaluation_scoring_pending": state.get("evaluation_scoring_pending"),
                "evaluations": [{"checkpoint": p["checkpoint"], "difficulty": p["difficulty"],
                    "episodes": len(p["rows"]), "scored": sum(bool(r.get("reward_scoring")) for r in p["rows"])}
                    for p in state.get("evaluations", [])]}, **kwargs)


def assign_rewards(episode, scores, condition, edges, config, *, bonuses=None):
    rows = episode["intervals"]
    if len(scores) != len(rows) or any(s is None for s in scores):
        raise ValueError("all post-action images must be scored before episode labels")
    if bonuses is None:
        bonuses = [{"success": 10., "collision": -5., "deadline": -1., "timeout": -1.}.get(r["end_reason"], 0.) for r in rows]
    elapsed = torch.tensor([r["elapsed_s"] for r in rows], dtype=torch.float64)
    q = torch.tensor([s["score"] for s in scores], dtype=torch.float64)
    terminal = torch.tensor([r["terminated"] for r in rows])
    bonus = torch.tensor(bonuses, dtype=torch.float64)
    derived = {c: episode_labels(elapsed, q, bonus, terminal, c, edges.double().cpu(),
        gamma=config.gamma, nominal_period=config.nominal_period_s) for c in CONDITIONS}
    actual = derived[condition]
    episode.update(visual_scores=copy.deepcopy(scores), terminal_bonuses=list(bonuses),
        stage_rewards=actual["coded_stage_rewards"].tolist(), returns=actual["coded_returns"].tolist(),
        raw_stage_rewards=actual["raw_stage_rewards"].tolist(), raw_returns=actual["raw_returns"].tolist(),
        return_bins=actual["bins"].tolist(), reward_condition=condition, reward_scoring=True,
        counterfactual_rewards={c: {k: v.tolist() for k, v in values.items()} for c, values in derived.items()})
    return actual


def validate_initial_panels(supplied, evaluations, asset_id):
    expected = {p["difficulty"]: p for p in evaluations}
    if (not asset_id or len(supplied) != len(expected)
            or {p["difficulty"] for p in supplied} != set(expected)):
        raise ValueError("reward initial panels must match asset and difficulties")
    result = []
    for name, spec in expected.items():
        panel = next(p for p in supplied if p["difficulty"] == name)
        if (panel.get("initial_asset_id") != asset_id or panel.get("checkpoint") != "initial"
                or panel.get("acquired_count") != 0 or panel.get("seeds") != list(spec["seeds"])
                or canonical_json(panel.get("config")) != canonical_json(asdict(spec["config"]))
                or [r["seed"] for r in panel["rows"]] != list(spec["seeds"])):
            raise ValueError("reward initial panel configuration/seed identity differs")
        if any(not r.get("complete") or r.get("truncated") or not r.get("evaluation")
               or r.get("learning") is not False or r.get("explore") is not False
               or r.get("reward_scoring") is not True for r in panel["rows"]):
            raise ValueError("reward initial panels must be complete scored frozen evaluations")
        result.append(copy.deepcopy(panel))
    return result


class RewardDevelopmentRun(SolverDevelopmentRun):
    @property
    def condition(self):
        return self.reward_condition

    def __init__(self, model, config, sim_config, store, stopped, *, condition, scorer,
                 initial_evaluations=None, **kwargs):
        power(condition)
        adapter = _RewardStore(store, condition, CenteringReward(model.encoder).state_dict())
        # Reuse of scored panels is validated here, not by the old unscored guard.
        super().__init__(model, config, sim_config, adapter, stopped, solver_recipe="proximal_3", **kwargs)
        adapter.constructing = False
        self.reward_condition = condition
        self.scorer = scorer
        if adapter.saved is not None:
            self.scorer.load_state_dict(adapter.saved["scorer"])
        reused = (validate_initial_panels(initial_evaluations, self.evaluations, kwargs.get("initial_asset_id"))
                  if initial_evaluations is not None else None)
        marker = {"version": VERSION, "condition": condition, "scorer": scorer.provenance,
            "initial_evaluations_sha256": hashlib.sha256(canonical_json(reused).encode()).hexdigest()
                if reused is not None else None}
        if adapter.saved is not None and canonical_json(self.s.get("reward_repair")) != canonical_json(marker):
            raise ValueError("reward pilot resume identity differs")
        self.s["reward_repair"] = marker
        if reused is not None and adapter.saved is None:
            self.s["evaluations"] = reused
            self.s["videos"] = [v for p in reused for r in p["rows"] for v in r.get("videos", [])]
            self.s["initial_evaluation_source"] = {"initial_asset_id": kwargs.get("initial_asset_id"),
                "initial_evaluations_sha256": marker["initial_evaluations_sha256"]}
            self.s["stage"] = "collect"

    def label_episode(self):
        s, current = self.s, self.s["current"]
        if current["label_complete"]:
            return
        if current["truncated"]:
            return super().label_episode()
        start = time.monotonic()
        for item in s["pending"]:
            if item["score"] is None:
                tick = time.monotonic()
                try:
                    item["score"] = self.scorer.score(item["post"], item["pan"])
                except BaseException:
                    self.save()
                    raise
                s["timings"]["scoring_seconds"] += time.monotonic() - tick
                self.save()
        values = assign_rewards(current, [p["score"] for p in s["pending"]], self.reward_condition,
            self.learner.edges, self.config.learning, bonuses=[p["bonus"] for p in s["pending"]])
        for index, item in enumerate(s["pending"]):
            action = item["action"]
            if action is None or item["history"] is None:
                continue
            record = {"episode_id": current["episode_id"], "history": item["history"],
                "arrival": item["arrival"], "baseline": action["baseline"],
                "logits": action["logits"].clone(), "bin": int(values["bins"][index]),
                "return": float(values["coded_returns"][index]), "raw_return": float(values["raw_returns"][index]),
                "learned": False}
            if any(r["arrival"] == record["arrival"] for r in s["eligible"]):
                raise ValueError("episode was already admitted")
            s["ready"].append(record["arrival"])
            s["eligible"].append(record)
        current["label_complete"] = True
        s["timings"]["sleep_seconds"] += time.monotonic() - start
        self.save()

    def evaluate(self, checkpoint):
        self.s["evaluation_scoring_pending"] = checkpoint
        self.save()
        super().evaluate(checkpoint)
        self._finish_evaluation_scores()

    def _finish_evaluation_scores(self):
        checkpoint = self.s.get("evaluation_scoring_pending")
        if checkpoint is None:
            return
        for panel in self.s["evaluations"]:
            if panel["checkpoint"] != checkpoint:
                continue
            for episode in panel["rows"]:
                if episode.get("reward_scoring"):
                    continue
                scores = episode.setdefault("visual_scores", [None] * len(episode["intervals"]))
                for index, row in enumerate(episode["intervals"]):
                    if scores[index] is None:
                        before = time.monotonic()
                        try:
                            scores[index] = self.scorer.score_rgb(self.store.frames.get(row["frame_ref"]))
                        except BaseException:
                            self.save()
                            raise
                        self.s["timings"]["scoring_seconds"] += time.monotonic() - before
                        self.save()
                assign_rewards(episode, scores, self.reward_condition, self.learner.edges, self.config.learning)
                self.save()
        self.s.pop("evaluation_scoring_pending", None)
        self.save()

    def collect(self):
        self._finish_evaluation_scores()
        return super().collect()

    def run(self):
        # Final-evaluation completion can precede score completion on a halt.
        if self.s["stage"] == "done":
            self._finish_evaluation_scores()
        result = super().run()
        return {**result, "kind": "reward_pilot", "condition": self.reward_condition,
            "reward_repair": self.s["reward_repair"], "reward_scoring": True,
            "limitations": ["Two-seed development reward comparison; not a confirmatory or sample-savings study.",
                "FD is disabled in both conditions; all STM and LTM parameters remain trainable.",
                "Each reward has a separately fitted initialization and matching equal-weight Fisher.",
                "Recorded counterfactual reward arithmetic is not a counterfactual rollout."]}
