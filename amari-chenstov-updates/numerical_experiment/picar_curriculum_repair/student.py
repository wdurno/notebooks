"""Persistent Fix 4 trajectory; evaluation never advances or trains its state.

The control loop is versioned here because the historical loop truncates an
episode at its transition budget and discards full predictive outputs.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import time

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.control import ControlSession
from numerical_experiment.picar.fix2_data import TASK, history_batch, observation_row
from numerical_experiment.picar.fix3_student import _Run, _History
from numerical_experiment.picar.models import batch_digest
from numerical_experiment.picar.runtime import ModelAccess
from numerical_experiment.picar_phase2.student import _terminal_diagnostics, _coverage
from numerical_experiment.picar_phase2_rescue.data import _history as restore_history
from numerical_experiment.picar_reward_repair.student import RewardDevelopmentRun, assign_rewards
from numerical_experiment.picar_solver_repair.learner import SolverLearner
from src.continual import picar_math as pm
from src.robotics.worker import SimulatorWorker
from src.seeding import derive_seed_map
from . import VERSION
from .config import CHECKPOINTS, panels, stage_at


class PredictiveLearner(SolverLearner):
    @torch.no_grad()
    def choose_action(self, history, *, explore=True):
        start = time.monotonic()
        output, _, metadata = self.model.predict(history)
        mean = output["action_mean"][0]
        noise = torch.randn(mean.shape, device=self.device, dtype=mean.dtype,
            generator=self.generators["exploration"]) if explore else torch.zeros_like(mean)
        z = mean + self.config.exploration_std * noise
        baseline = (output["return_logits"].softmax(-1) * pm.return_bin_values(self.edges)).sum(-1)[0]
        return {"logits": z.cpu(), "probabilities": torch.cat((z, z.new_zeros(1))).softmax(-1).cpu(),
            "baseline": float(baseline), "model_version": self.version,
            "inference_seconds": time.monotonic()-start, "metadata": metadata,
            "prediction": {key: value[0].detach().cpu().tolist() for key, value in output.items()
                           if key in {"return_logits", "action_mean"}}}


class _Store:
    def __init__(self, store, method):
        self.store, self.method = store, method

    def __getattr__(self, name):
        return getattr(self.store, name)

    def checkpoint(self, state, *, metrics=(), progress=None, **kwargs):
        n = state["acquired_count"]
        total = state["identity"]["sim_config"]["curriculum_total_intervals"]
        curriculum = stage_at(n*2048/total)
        curriculum = {**curriculum, "start": curriculum["start"]*total/2048,
            "end": curriculum["end"]*total/2048, "within": curriculum["within"]*total/2048,
            "compressed_diagnostic": total != 2048}
        preview = []
        for panel in state["evaluations"]:
            episodes = [e for e in panel["rows"] if e.get("predictive_scoring")]
            if len(episodes) != len(panel["seeds"]):
                continue
            preview.append({"target": panel["target"], "environment": panel["difficulty"],
                "acquired_count": panel["acquired_count"], "episodes": len(episodes),
                "success": sum(e["success"] for e in episodes)/len(episodes),
                "raw_return": sum(e["raw_returns"][0] for e in episodes)/len(episodes)})
        return self.store.checkpoint(state, metrics=[{**r, "condition": self.method} for r in metrics],
            progress={**(progress or {}), "version": VERSION, "condition": self.method,
                "stage": state["stage"], "curriculum": curriculum, "acquired_count": n,
                "events": len(state.get("fit_metrics", [])),
                "evaluation_checkpoint": state.get("evaluation_checkpoint"),
                "reference_cursor": state.get("reference_cursor", 0),
                "checkpoint_preview": preview,
                "reference_preview": [{"target": p["target"], "environment": environment,
                    "joint_nll": sum(r["joint_nll"] for r in p["rows"] if r["environment"] == environment)
                        /sum(r["environment"] == environment for r in p["rows"]),
                    "records": sum(r["environment"] == environment for r in p["rows"])}
                    for p in state.get("reference_scores", [])
                    for environment in sorted({r["environment"] for r in p["rows"]})],
                "evaluations": [{"checkpoint": p["checkpoint"], "difficulty": p["difficulty"],
                    "episodes": len(p["rows"]), "scored": sum(r.get("reward_scoring", False) for r in p["rows"])}
                    for p in state["evaluations"]],
                "preview_training": [{"n": m["acquired_count"]-state["arrival_offset"],
                    "data": m["inner_steps"][0]["data"]} for m in state.get("fit_metrics", [])[-32:]]}, **kwargs)


class CurriculumRun(RewardDevelopmentRun):
    def __init__(self, *args, checkpoints=CHECKPOINTS, reference_records=(), reference_id=None,
                 reference_mode=False, **kwargs):
        cfg = args[1] if len(args) > 1 else kwargs["config"]
        method = "stm_fd" if cfg.learning.fd_strength > 0 else "replay_ewc"
        positional = list(args)
        if len(positional) > 3:
            positional[3] = _Store(positional[3], method)
        else:
            kwargs["store"] = _Store(kwargs["store"], method)
        self.method = method
        self.reference_mode = reference_mode
        self.references = list(reference_records)
        self.checkpoints = tuple(checkpoints)
        if (not self.checkpoints or self.checkpoints[0] != 0
                or any(a >= b for a, b in zip(self.checkpoints, self.checkpoints[1:]))):
            raise ValueError("checkpoints must increase from zero")
        super().__init__(*positional, condition="width6_linear", **kwargs)
        self.learner.__class__ = PredictiveLearner
        marker = {"version": VERSION, "checkpoints": list(checkpoints), "method": method,
            "reference_id": reference_id, "reference_mode": reference_mode,
            "evaluation": "fixed_eight_interim_twelve_final_A_Binitial_Bfinal",
            "finish_episode": True}
        if self.s.get("fix4") is not None and self.s["fix4"] != marker:
            raise ValueError("Fix 4 resume identity differs")
        self.s.setdefault("fix4", marker)
        self.s.setdefault("evaluated_targets", [0] if self.s.get("initial_evaluation_source") else [])
        self.s.setdefault("reference_scores", [])
        self.s.setdefault("reference_episodes", [])

    @property
    def condition(self):
        return self.method

    def episode(self, sim_config, *, seed, exposure, evaluation=False, video=False):
        s = self.s
        if s["current"] is not None and s["current"]["complete"]:
            return
        history = _History(self.config, sim_config, s["history"])
        access = ModelAccess()
        access.version = self.learner.version
        worker = SimulatorWorker(sim_config, run_id=f"fix4-{self.identity['seed']}")
        session = ControlSession(worker, access, pending_capacity=512, own_worker=True,
            learner=self.learner, history_builder=history, explore=self.reference_mode or not evaluation)
        try:
            if s["session"] is not None:
                session.restore(s["session"])
                if not s["session"]["terminal"]:
                    session.resume()
            else:
                s["pending"] = []
                ordinal = len(s["reference_episodes"]) if self.reference_mode else len(s["episodes"])
                s["current"] = {"episode_id": f"{'reference' if self.reference_mode else 'pilot'}:{ordinal}",
                    "seed": seed, "reset_exposure": exposure, "complete": False,
                    "evaluation": evaluation, "frames": [], "intervals": [],
                    "end_reason": None, "success": None, "label_complete": False}
                session.start(seed=seed, exposure_intervals=exposure, training=not evaluation)
                s["current"]["layout"] = worker.diagnostics()["layout"]
                self.save(session, history)
            current = s["current"]
            # Budget admission is at episode boundaries; never synthesize a
            # timeout or discard a successful suffix at the observation cap.
            while True:
                try:
                    interval = session.next_interval(timeout=120.)
                except StopIteration:
                    if not current["complete"]:
                        raise RuntimeError("ended session lacks a terminal interval")
                    break
                result, action = interval.result, interval.action
                observation = result.observation
                with history.lock:
                    causal = None if action is None or interval.policy_observation is None else copy.deepcopy(
                        history.state["histories"][interval.policy_observation.observation_id])
                row = {"index": interval.index, "fallback": interval.fallback_reason,
                    "elapsed_s": result.reward_components.elapsed_s, "end_reason": result.end_reason,
                    "timing": interval.timing, "terminated": result.terminated, "truncated": result.truncated,
                    "model_version": interval.model_version, "controls": asdict(observation.executed_controls),
                    "policy_observation_id": None if interval.policy_observation is None else interval.policy_observation.observation_id,
                    "probabilities": None if action is None else action["probabilities"].tolist(),
                    "logits": None if action is None else action["logits"].tolist(),
                    "baseline": None if action is None else action["baseline"],
                    "prediction": None if action is None else copy.deepcopy(action["prediction"])}
                post_row = observation_row(observation, previous_time=interval.observation.capture_time_s)
                row["frame_ref"] = self.store.frames.put(post_row["rgb"])
                if causal is not None:
                    pixels = causal["images"][0].detach().cpu().mul(255).round().clamp(0, 255).byte()
                    row.update(context_refs=[self.store.frames.put(frame.permute(1, 2, 0).numpy()) for frame in pixels],
                        context_actions=causal["executed_actions"][0].tolist(),
                        context_elapsed=causal["elapsed"][0, :, 0].tolist(),
                        context_decision_times=causal["decision_time"][0].tolist(),
                        context_start_time=float(causal["start_time"][0]), contexts=copy.deepcopy(causal["contexts"][0]),
                        context_status_events=copy.deepcopy(causal["status_events"][0]),
                        episode_deadline=float(causal["episode_deadline"][0]), history_digest=batch_digest(causal))
                current["intervals"].append(row)
                if evaluation:
                    if video:
                        current["frames"].append(worker.snapshot()["backend"])
                else:
                    s["acquired_count"] += 1
                    row["arrival"] = s["arrival_offset"] + s["acquired_count"] - 1
                    s["intervals"].append({**row, "episode": len(s["episodes"])})
                    key = interval.fallback_reason or "policy"
                    s["fallback_counts"][key] = s["fallback_counts"].get(key, 0)+1
                    post = history_batch([post_row], contexts=[{"text": TASK, "issued_at": 0.}],
                        deadline=history.state["deadline"] or sim_config.deadline_s,
                        width=sim_config.width, height=sim_config.height)
                    s["pending"].append({"elapsed": result.reward_components.elapsed_s,
                        "bonus": result.reward_components.terminal_bonus, "terminal": result.terminated,
                        "truncated": result.truncated, "post": post, "pan": observation.executed_controls.pan_rad,
                        "action": copy.deepcopy(action), "history": causal, "arrival": row["arrival"], "score": None})
                if result.terminated or result.truncated:
                    session.pause(reason="episode_boundary")
                    current.update(complete=True, end_reason=result.end_reason,
                        success=result.end_reason == "success", truncated=result.truncated)
                    self.save(session, history)
                    break
                if not evaluation and s["ready"]:
                    self.work_one(session)
                    if len(s["eligible"]) > self.config.learning.replay_capacity and s["eligible"][0]["learned"]:
                        self.retire_one(access)
                self.save(session, history)
        finally:
            session.close(timeout=120.)

    def collect(self):
        s = self.s
        while True:
            due = [n for n in self.checkpoints if n not in s["evaluated_targets"] and n <= s["acquired_count"]]
            if due and s["current"] is None:
                s.update(stage="checkpoint_drain", evaluation_checkpoint=due[0])
                self.save()
                return
            if s["acquired_count"] >= self.identity["max_transitions"] and s["current"] is None:
                s.update(stage="checkpoint_drain", evaluation_checkpoint=self.identity["max_transitions"])
                self.save()
                return
            ordinal = len(s["episodes"])
            seed = derive_seed_map(self.seeds["layouts"], (f"episode_{ordinal}",))[f"episode_{ordinal}"]
            self.episode(self.sim_config, seed=seed, exposure=float(s["acquired_count"]))
            if s["current"]["truncated"]:
                raise RuntimeError("infrastructure truncation is not a learning failure")
            self.label_episode()
            current = s["current"]
            s["episodes"].append({k: v for k, v in current.items() if k != "frames"})
            s["current"], s["session"], s["history"], s["pending"] = None, None, None, []
            self.save(metrics=[{"phase": 3, "condition": self.method, "replica": self.identity["seed"],
                "step": s["acquired_count"], "episode": ordinal, "end_reason": current["end_reason"],
                "success": current["success"], "new_transitions": s["acquired_count"],
                "retired_transitions": self.learner.retired_count, "model_version": self.learner.version}])

    def drain(self):
        self.s["stage"] = "drain"
        super().drain()
        self.s["stage"] = "checkpoint_evaluation"
        self.save()

    def _score_episode(self, episode):
        from .predictive import score_prediction
        scores = episode.setdefault("visual_scores", [None]*len(episode["intervals"]))
        for i, row in enumerate(episode["intervals"]):
            if scores[i] is None:
                before = time.monotonic()
                scores[i] = self.scorer.score_rgb(self.store.frames.get(row["frame_ref"]))
                self.s["timings"]["scoring_seconds"] += time.monotonic()-before
                self.save()
        assign_rewards(episode, scores, "width6_linear", self.learner.edges, self.config.learning)
        episode["predictive_scores"] = []
        for i, row in enumerate(episode["intervals"]):
            if row.get("prediction") is not None:
                episode["predictive_scores"].append({"index": i, **score_prediction(row["prediction"],
                    label=episode["return_bins"][i], action=row["logits"], baseline=row["baseline"],
                    bin_values=pm.return_bin_values(self.learner.edges).cpu(),
                    temperature=self.config.learning.temperature, action_variance=self.config.learning.action_variance)})
        episode.update(reward_scoring=True, predictive_scoring=True,
            predictive_missing_intervals=len(episode["intervals"])-len(episode["predictive_scores"]))
        self.save()

    def _score_reference(self, target):
        from .predictive import evaluate_records
        s = self.s
        found = next((r for r in s["reference_scores"] if r["target"] == target), None)
        if found is None:
            found = {"target": target, "acquired_count": s["acquired_count"], "rows": []}
            s["reference_scores"].append(found)
        for record in self.references[len(found["rows"]):]:
            row = evaluate_records(self.learner, [record])[0]
            found["rows"].append({**row, "environment": record["environment"],
                "record_id": record["record_id"], "episode_id": record["episode_id"]})
            self.save()

    def evaluate(self, checkpoint):
        s = self.s
        target = 0 if checkpoint == "initial" else int(checkpoint)
        label = f"n{target}"
        if s["pending"] or s["ready"] or (s["current"] and not s["current"]["evaluation"]):
            raise RuntimeError("held-out evaluation requires drained training")
        specifications = panels(final=target == self.checkpoints[-1]) if self.evaluations else []
        # For tests and short diagnostics the caller supplies smaller fixed panels.
        if self.evaluations and self.identity["max_transitions"] != 2048:
            specifications = self.evaluations
        for spec in specifications:
            found = next((p for p in s["evaluations"] if p["checkpoint"] == label
                          and p["difficulty"] == spec["difficulty"]), None)
            if found is None:
                found = {"checkpoint": label, "target": target, "difficulty": spec["difficulty"], "rows": [],
                    "config": asdict(spec["config"]), "seeds": list(spec["seeds"]),
                    "acquired_count": s["acquired_count"], "initial_asset_id": self.identity["phase3"]["initial_asset_id"]}
                s["evaluations"].append(found)
            for seed in spec["seeds"][len(found["rows"]):]:
                before = time.monotonic()
                video = seed == spec["seeds"][0] and target in {0, 1024, self.checkpoints[-1]}
                self.episode(spec["config"], seed=seed, exposure=0., evaluation=True, video=video)
                current = s["current"]
                if current["truncated"]:
                    raise RuntimeError("truncated held-out episode is not a failure label")
                clips = self.render_video(current, label, spec["difficulty"]) if video else []
                row = {k: v for k, v in current.items() if k not in {"frames", "label_complete"}}
                row.update(explore=False, learning=False, actor="learned_student", videos=clips,
                    terminal_diagnostics=_terminal_diagnostics(s["session"]), coverage=_coverage(current["intervals"]),
                    acquired_count=s["acquired_count"], physical_seconds=sum(r["elapsed_s"] for r in current["intervals"]))
                found["rows"].append(row)
                s["videos"].extend(clips)
                s["current"], s["session"], s["history"], s["pending"] = None, None, None, []
                s["timings"]["evaluation_seconds"] += time.monotonic()-before
                self.save()
            for episode in found["rows"]:
                if not episode.get("predictive_scoring"):
                    self._score_episode(episode)
        self._score_reference(target)
        if target not in s["evaluated_targets"]:
            s["evaluated_targets"].append(target)
        s["stage"] = "done" if target == self.checkpoints[-1] else "collect"
        self.save()

    def collect_references(self, environment, *, target_histories=32, min_episodes=8, interval_cap=256):
        s = self.s
        while True:
            episodes = s["reference_episodes"]
            captured = sum(len(e["intervals"]) for e in episodes)
            usable = sum(len(e.get("predictive_scores", [])) for e in episodes)
            if s["current"] is None and ((usable >= target_histories and len(episodes) >= min_episodes)
                    or captured >= interval_cap):
                break
            reserve = int(self.sim_config.deadline_s/self.sim_config.control_interval_s)+1
            if s["current"] is None and interval_cap-captured < reserve:
                break
            # Admit only complete episodes that fit the reference cap; a
            # shortfall is reported, never silently filled with partial labels.
            ordinal = len(episodes)
            seed = derive_seed_map(self.seeds["layouts"], (f"reference_{ordinal}",))[f"reference_{ordinal}"]
            self.episode(self.sim_config, seed=seed, exposure=0., evaluation=True)
            current = s["current"]
            if current["truncated"]:
                raise RuntimeError("reference labels require complete episodes")
            self._score_episode(current)
            episodes.append({k: v for k, v in current.items() if k != "frames"})
            s["current"], s["session"], s["history"], s["pending"] = None, None, None, []
            self.save()
        records = []
        for j, episode in enumerate(s["reference_episodes"]):
            for i, row in enumerate(episode["intervals"]):
                if row.get("prediction") is not None and row.get("context_refs"):
                    records.append({"environment": environment, "record_id": f"{environment}:{j}:{i}",
                        "episode_id": f"{environment}:{j}", "history": restore_history(row, self.store.frames,
                            self.config, self.sim_config.calibration), "bin": episode["return_bins"][i],
                        "baseline": row["baseline"], "logits": torch.tensor(row["logits"]), "arrival": len(records)})
        s["reference_records"] = records
        s["stage"] = "done"
        self.save()
        return {"environment": environment, "histories": len(records), "episodes": len(s["reference_episodes"]),
            "captured": captured, "interval_cap": interval_cap, "overshoot": max(0, captured-interval_cap),
            "shortfall": len(records) < target_histories or len(s["reference_episodes"]) < min_episodes,
            "new_training_observations": 0, "episode_rows": s["reference_episodes"], "timings": s["timings"]}

    def run(self):
        while self.s["stage"] != "done":
            stage = self.s["stage"]
            if stage == "initial_evaluation":
                self.evaluate("initial")
            elif stage == "collect":
                self.collect()
            elif stage in {"checkpoint_drain", "drain", "final_evaluation"}:
                self.drain()
            elif stage == "checkpoint_evaluation":
                self.evaluate(str(self.s["evaluation_checkpoint"]))
            else:
                raise ValueError(f"unknown Fix 4 stage: {stage}")
        result = _Run.run(self)
        metrics = self.s["fit_metrics"]
        if self.s["ready"] or self.s["pending"] or any(not r["learned"] for r in self.s["eligible"]):
            raise RuntimeError("completed trajectory has unfinished labels/learning")
        return {**result, "kind": "curriculum_trajectory", "version": VERSION, "condition": self.method,
            "fit_metrics": metrics, "fd_strength": self.config.learning.fd_strength,
            "ridge": self.config.learning.ridge, "reference_scores": self.s["reference_scores"],
            "evaluated_targets": self.s["evaluated_targets"], "fix4": self.s["fix4"],
            "overshoot": max(0, self.s["acquired_count"]-self.identity["max_transitions"]),
            "optimizer_exposures": sum(m["optimizer_exposures"] for m in metrics),
            "fd_evaluations": sum(m["fd_evaluations"] for m in metrics),
            "first_step_ewc_events": sum(m["first_step_ewc_proposal_difference_norm"] > 0 for m in metrics),
            "limitations": ["Two-seed paired development, not confirmation or retention equivalence.",
                "Fixed-reference NLL is off-policy fit, not demonstrated navigation or sample savings.",
                "Own deterministic actions give a Gaussian normalizer, not imitation accuracy."]}
