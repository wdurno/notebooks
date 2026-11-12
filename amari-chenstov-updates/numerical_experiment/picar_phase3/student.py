"""Phase 3 development scheduling over the frozen learning/control machinery.

The FD=0 construction view exists only to pass Fix3's historical arm guard.
Every committed identity, learner configuration and optimizer event uses the
real Phase 3 configuration. Likelihood/initial-asset compatibility is caller-owned.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import math
import time

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.fix3_student import _batch
from numerical_experiment.picar_phase2.student import ReadinessRun, _coverage, _terminal_diagnostics
from src.seeding import derive_seed_map


VERSION = "plan19-phase3-runtime-v1"
SCHEMA = 1


class _DevelopmentStore:
    def __init__(self, store, config, safe_config, marker):
        self.store, self.config, self.safe_config, self.marker = store, config, safe_config, marker
        self.constructing, self.resumed = True, False

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _validate(self, identity):
        if (canonical_json(identity.get("phase3")) != canonical_json(self.marker)
                or canonical_json(identity.get("config")) != canonical_json(self.config)):
            raise ValueError("Phase 3 resume identity differs")

    def load_checkpoint(self, **kwargs):
        saved = self.store.load_checkpoint(**kwargs)
        if saved is None:
            return None
        self._validate(saved["identity"])
        self.resumed = True
        if not self.constructing:
            return saved
        identity = {**saved["identity"], "config": self.safe_config}
        identity.pop("phase3")
        return {**saved, "identity": identity}

    def checkpoint(self, state, *, metrics=(), progress=None, **kwargs):
        self._validate(state["identity"])
        condition = "stm_fd" if self.config["learning"]["fd_strength"] > 0 else "replay_ewc"
        return self.store.checkpoint(state,
            metrics=[{**r, "phase": 3, "condition": condition} for r in metrics],
            progress={**(progress or {}), "phase": 3,
                      "status": f"Phase 3 {state['identity']['mode']}: {state['stage']}"}, **kwargs)


def _initial_panels(evaluations, supplied, asset_identity):
    """Only complete, frozen panels from the same explicitly identified asset."""
    if supplied is None:
        return None
    if not isinstance(asset_identity, str) or not asset_identity:
        raise ValueError("initial panel reuse requires initial_asset_id")
    supplied = copy.deepcopy(list(supplied))
    expected = {p["difficulty"]: p for p in evaluations}
    if (not expected or len(expected) != len(evaluations) or len(supplied) != len(expected)
            or {p["difficulty"] for p in supplied} != set(expected)):
        raise ValueError("initial panel difficulties must exactly match evaluation specifications")
    by_name = {p["difficulty"]: p for p in supplied}
    for name, spec in expected.items():
        panel, seeds = by_name[name], list(spec["seeds"])
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or panel.get("checkpoint") != "initial" or panel.get("acquired_count") != 0
                or panel.get("initial_asset_id") != asset_identity
                or panel.get("seeds") != seeds
                or canonical_json(panel.get("config")) != canonical_json(asdict(spec["config"]))
                or [r["seed"] for r in panel["rows"]] != seeds):
            raise ValueError("initial panel identity/configuration/seeds differ")
        if any(not r.get("complete") or r.get("truncated") or not r.get("evaluation")
               or r.get("learning") is not False or r.get("explore") is not False
               or r.get("reward_scoring") is not False or r.get("actor") != "learned_student"
               for r in panel["rows"]):
            raise ValueError("initial panels must be complete frozen student evaluations")
    return [by_name[p["difficulty"]] for p in evaluations]


class DevelopmentRun(ReadinessRun):
    def __init__(self, model, config, sim_config, store, stopped, *, mode="pilot", seed,
                 max_transitions, initial=None, teacher_variant="centering", evaluations=(),
                 initial_evaluations=None, initial_asset_id=None):
        config.learning.validate()  # Validate real FD strength before constructing the safe view.
        evaluations = copy.deepcopy(list(evaluations))
        reused = _initial_panels(evaluations, initial_evaluations, initial_asset_id)
        if reused is not None and mode != "pilot":
            raise ValueError("initial panel reuse is pilot-only")
        marker = {"version": VERSION, "schema_version": SCHEMA, "phase": 3,
            "mechanism": "development", "confirmatory": False,
            "model_provenance": copy.deepcopy(getattr(model, "provenance", None)),
            "initial_asset_id": initial_asset_id,
            "initial_evaluations_sha256": hashlib.sha256(canonical_json(reused).encode()).hexdigest()
                if reused is not None else None,
            "midpoint": "first_drained_complete_episode_boundary_at_or_after_half_budget",
            "bootstrap_nll_steps_per_record": 1}
        canonical_json(marker)
        safe = replace(config, learning=replace(config.learning, fd_strength=0.))
        adapter = _DevelopmentStore(store, asdict(config), asdict(safe), marker)
        super().__init__(model, safe, sim_config, adapter, stopped, mode=mode, seed=seed,
            max_transitions=max_transitions, initial=initial, teacher_variant=teacher_variant,
            evaluations=evaluations)
        adapter.constructing = False
        self.store, self.config, self.learner.config = adapter, config, config.learning
        self.identity.update(config=asdict(config), phase3=marker)
        self.s.update(identity=self.identity, phase=3, phase3=marker,
                      mechanism="development", confirmatory=False)
        self.s.pop("phase2", None)
        self.s.setdefault("fit_metrics", [])
        self.s.setdefault("midpoint_done", False)
        for group in self.learner.optimizer.param_groups:
            if adapter.resumed and group["lr"] != config.learning.learning_rate:
                raise ValueError("Phase 3 checkpoint Adam learning rate differs from configuration")
            group["lr"] = config.learning.learning_rate
        if reused is not None and not adapter.resumed:
            self.s["evaluations"] = reused
            self.s["videos"] = [v for p in reused for r in p["rows"] for v in r["videos"]]
            self.s["initial_evaluation_source"] = {k: marker[k] for k in
                ("initial_asset_id", "initial_evaluations_sha256")}
            self.s["stage"] = "collect"

    @property
    def condition(self):
        return "stm_fd" if self.config.learning.fd_strength > 0 else "replay_ewc"

    def work_one(self, session=None):
        s = self.s
        item = next(x for x in s["eligible"] if x["arrival"] == s["ready"][0])
        replay = [x for x in s["eligible"] if x["learned"] and x is not item]
        replay_state = self.replay_rng.get_state()
        try:
            other = replay[int(torch.randint(len(replay), (), generator=self.replay_rng))] if replay else None
            kwargs = {"new": _batch(item, s["eligible"]),
                "replay": _batch(other, s["eligible"]) if other else None,
                "acquired_count": s["arrival_offset"] + s["acquired_count"],
                "steps": self.config.learning.optimizer_steps}
            metrics = self.learner.fit(**kwargs) if session is None else session.update(cooperative=True, **kwargs)
        except BaseException:
            self.replay_rng.set_state(replay_state)
            raise
        s["fit_metrics"].append({**metrics, "phase": 3, "condition": self.condition,
            "path": "sleep" if session is None else "online", "arrival": item["arrival"],
            "replay_arrival": other["arrival"] if other else None,
            "acquired_count": kwargs["acquired_count"], "model_version": self.learner.version,
            "optimizer_steps": kwargs["steps"], "optimizer_step_count": self.learner.update_count,
            "fd_strength": self.config.learning.fd_strength, "learning_rate": self.config.learning.learning_rate})
        s["timings"]["fit_seconds"] += metrics["update_seconds"]
        s["online_updates" if session is not None else "sleep_updates"] += 1
        item["learned"] = True
        s["ready"].pop(0)
        return metrics

    def collect(self):
        s = self.s
        reserve = math.ceil(self.sim_config.deadline_s / self.sim_config.control_interval_s) + 2
        while True:
            if (self.mode == "pilot" and not s["midpoint_done"] and s["current"] is None
                    and "incomplete_episode" not in s
                    and 2 * s["acquired_count"] >= self.identity["max_transitions"]):
                s.update(stage="midpoint_drain", midpoint_acquired_count=s["acquired_count"])
                self.save()
                return
            if s["acquired_count"] >= self.identity["max_transitions"] and s["current"] is None:
                break
            if s["current"] is None and self.identity["max_transitions"] - s["acquired_count"] < reserve:
                s["unused_horizon_reserve"] = self.identity["max_transitions"] - s["acquired_count"]
                break
            episode = len(s["episodes"])
            seed = derive_seed_map(self.seeds["layouts"], (f"episode_{episode}",))[f"episode_{episode}"]
            self.episode(self.sim_config, seed=seed,
                         exposure=0. if self.mode == "bootstrap" else float(s["acquired_count"]))
            self.label_episode()
            current = s["current"]
            s["episodes"].append({k: v for k, v in current.items() if k != "frames"})
            if current["truncated"]:
                s["incomplete_episode"] = {"pending": s["pending"], "session": s["session"],
                    "history": s["history"], "episode": copy.deepcopy(current)}
            s["current"], s["session"], s["history"], s["pending"] = None, None, None, []
            self.save(metrics=[{"phase": 3, "condition": self.condition, "replica": self.identity["seed"],
                "step": s["acquired_count"], "episode": episode, "end_reason": current["end_reason"],
                "success": current["success"], "new_transitions": s["acquired_count"],
                "retired_transitions": self.learner.retired_count, "model_version": self.learner.version}])
            if current["truncated"]:
                break
        s["stage"] = "distill" if self.mode == "bootstrap" else "drain"
        self.save()

    def drain(self):
        s = self.s
        midpoint = s["stage"] == "midpoint_drain"
        # A committed fit must finish its FIFO retirements before another fit,
        # including recovery from the save immediately after that fit.
        while s["ready"] or s.get("drain_retire_pending", False):
            if s.get("drain_retire_pending", False):
                while len(s["eligible"]) > self.config.learning.replay_capacity and s["eligible"][0]["learned"]:
                    self.retire_one()
                    self.save()
                s["drain_retire_pending"] = False
            else:
                before = time.monotonic()
                self.work_one()
                s["timings"]["sleep_seconds"] += time.monotonic() - before
                s["drain_retire_pending"] = True
                self.save()
        while len(s["eligible"]) > self.config.learning.replay_capacity:
            self.retire_one()
            self.save()
        s["stage"] = "midpoint_evaluation" if midpoint else (
            "calibration" if self.mode == "bootstrap" else "final_evaluation")
        self.save()

    def evaluate(self, checkpoint):
        if checkpoint not in {"initial", "midpoint", "final"}:
            raise ValueError("checkpoint must be initial, midpoint or final")
        s = self.s
        if s["pending"] or s["ready"] or (s["current"] and not s["current"]["evaluation"]):
            raise RuntimeError("evaluation requires a drained training episode boundary")
        for panel in self.evaluations:
            found = next((e for e in s["evaluations"] if e["checkpoint"] == checkpoint
                          and e["difficulty"] == panel["difficulty"]), None)
            if found is None:
                found = {"checkpoint": checkpoint, "difficulty": panel["difficulty"], "rows": [],
                    "config": asdict(panel["config"]), "seeds": list(panel["seeds"]),
                    "acquired_count": s["acquired_count"],
                    "initial_asset_id": self.identity["phase3"]["initial_asset_id"]}
                s["evaluations"].append(found)
            if found["acquired_count"] != s["acquired_count"]:
                raise ValueError("cannot reuse an evaluation checkpoint at a different acquisition count")
            for seed in panel["seeds"][len(found["rows"]):]:
                before = time.monotonic()
                video = seed == panel["seeds"][0]
                self.episode(panel["config"], seed=seed, exposure=0., evaluation=True, video=video)
                current = s["current"]
                if current["truncated"]:
                    raise RuntimeError("evaluation infrastructure truncation is not a task failure")
                terminal = _terminal_diagnostics(s["session"])
                clips = self.render_video(current, checkpoint, panel["difficulty"]) if video else []
                row = {k: v for k, v in current.items() if k not in {"frames", "label_complete"}}
                row.update(explore=False, learning=False, reward_scoring=False, actor="learned_student",
                    videos=clips, terminal_diagnostics=terminal, coverage=_coverage(current["intervals"]),
                    acquired_count=s["acquired_count"],
                    physical_seconds=sum(x["elapsed_s"] for x in current["intervals"]))
                found["rows"].append(row)
                s["videos"].extend(clips)
                s["session"], s["history"], s["current"], s["pending"] = None, None, None, []
                s["timings"]["evaluation_seconds"] += time.monotonic() - before
                self.save()
        if checkpoint == "midpoint":
            s["midpoint_done"] = True
        s["stage"] = "done" if checkpoint == "final" else "collect"
        self.save()

    def run(self):
        while self.s["stage"] != "done":
            stage = self.s["stage"]
            if stage.endswith("_evaluation"):
                self.evaluate(stage.removesuffix("_evaluation"))
            elif stage in {"initial_fisher", "drain", "midpoint_drain"}:
                self.drain()
            elif stage in {"collect", "distill", "initial_nll", "calibration"}:
                getattr(self, stage)()
            else:
                raise ValueError(f"unknown Phase 3 checkpoint stage: {stage}")
        summary = super().run()
        summary.pop("phase2", None)
        metrics = self.s["fit_metrics"]
        return {**summary, "kind": f"phase3_{self.mode}", "phase": 3, "phase3": self.identity["phase3"],
            "mechanism": "development", "confirmatory": False, "condition": self.condition,
            "fd_strength": self.config.learning.fd_strength, "learning_rate": self.config.learning.learning_rate,
            "optimizer_steps_per_event": self.config.learning.optimizer_steps,
            "learner_optimizer_step_count": self.learner.update_count,
            "online_optimizer_steps": sum(m["optimizer_steps"] for m in metrics if m["path"] == "online"),
            "sleep_optimizer_steps": sum(m["optimizer_steps"] for m in metrics if m["path"] == "sleep"),
            "fit_metrics": metrics, "midpoint_acquired_count": self.s.get("midpoint_acquired_count"),
            "initial_evaluation_source": self.s.get("initial_evaluation_source"),
            "fd_enabled_fit_seconds": sum(m["update_seconds"] for m in metrics
                if m["fd_strength"] > 0 and m["replay_arrival"] is not None), "fd_overhead_measured": False,
            "limitations": ["Development only; no confirmatory efficacy or matched-retention claim.",
                "Bootstrap NLL retains one step per record; pilot events use the configured step budget.",
                "FD-enabled fit time includes the entire update, not isolated FD overhead.",
                "Capped/infrastructure-truncated episodes never receive invented returns."]}
