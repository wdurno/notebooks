"""Real-time coached acquisition and unchanged rank-64 FD/AWAC fitting."""

import copy
from dataclasses import asdict
import io
import math
from pathlib import Path
import time

import torch

from src.continual.picar_math import discounted_returns, time_discounts
from src.robotics.simulator import SimulatorConfig
from src.robotics.worker import SimulatorWorker
from src.seeding import derive_component_seed
from numerical_experiment.picar.fix2_data import history_batch, observation_row
from numerical_experiment.picar.runtime import ModelAccess
from numerical_experiment.picar.storage import _atomic_bytes
from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar_head_awac.config import simulator
from numerical_experiment.picar_head_awac.data import append_history, pack_history, unpack_history
from numerical_experiment.picar_head_awac.trajectory import Trajectory
from numerical_experiment.picar_head_awac.likelihood import data_loss
from numerical_experiment.picar_head_rank.numerics import block_diagnostics
from .coach import CoachingWindows, chassis_yaw, guidance_schedule
from .history import History
from .scoring import reward_components
from .config import seed as component_seed
from .roles import CoachingTeacher as CenteringTeacher
from .session import CoachingSession
from .transitions import execution_segments, decision_returns


def snapshot_at_capture(snapshot, at):
    """Trim the original event journal to an already captured physics tick."""
    from src.robotics.simulator import PyBulletBackend
    value = copy.deepcopy(snapshot)
    target = round(at / value["config"]["physics_dt_s"]) - value["clock_start_ticks"]
    if target < 0:
        raise ValueError("camera capture predates simulator reset")
    ticks, events = 0, []
    for event in value["events"]:
        if ticks >= target:
            break
        if event["kind"] == "advance":
            count = min(event["physics_steps"], target - ticks)
            events.append({**event, "physics_steps": count})
            ticks += count
        else:
            events.append(event)
    if ticks != target:
        raise ValueError("saved simulator journal does not contain the camera capture")
    value["events"] = events
    value["observations"] = [r for r in value["observations"] if r[1] <= at + 1e-8]
    value["last_result_ticks"] = min(value["last_result_ticks"], target)
    value.pop("sha256")
    value["sha256"] = PyBulletBackend._snapshot_hash(value)
    return value


def difficulty_group(difficulty):
    for name, lo, hi in (("initial", 0., .1), ("medium", .4, .6), ("final", .9, 1.)):
        if lo - 1e-8 <= difficulty <= hi + 1e-8:
            return name
    return None


def episode_summary(episode):
    excluded = {"rows", "frames", "coaching_audits", "coaching_windows", "compliance"}
    summary = {k: v for k, v in episode.items() if k not in excluded}
    summary["scored_guidance_windows"] = len(episode.get("compliance", {}))
    return summary


class CoachingTrajectory(Trajectory):
    def __init__(self, *args, model, compliance=None, adherence_enabled=False, condition, phase, collection=False,
                 initial_training_ids=(), **kwargs):
        super().__init__(*args, evaluation_episodes=8, initial_episodes=4, midpoint_episodes=4,
                         development=phase == "development", **kwargs)
        self.model, self.compliance, self.condition, self.phase = model, compliance, condition, phase
        if type(adherence_enabled) is not bool:
            raise ValueError("adherence_enabled must be Boolean")
        self.adherence_enabled = adherence_enabled
        self.reward_mode = "task_and_adherence" if self.adherence_enabled else "task_only"
        if self.adherence_enabled and compliance is None:
            raise ValueError("adherence-enabled trajectories require a scorer")
        recorded_mode = self.s.get("reward_mode")
        if recorded_mode != self.reward_mode and (recorded_mode is not None or self.store._current is not None):
            raise ValueError("trajectory reward mode differs from recovery state")
        self.s["reward_mode"] = self.reward_mode
        self.collection = collection
        self.teacher = CenteringTeacher(model.encoder) if collection else None
        self.coach = None
        self.initial_training_ids = set(initial_training_ids)
        if collection and self.s["stage"] == "evaluate" and not self.s["evaluations"]:
            self.s["stage"] = "collect"
        self.s.setdefault("initial_records", [])
        self.s.setdefault("media_counts", {})
        self.s.setdefault("coaching_metrics", [])
        self.s.setdefault("archive_metrics", [])
        self.s.setdefault("action_metrics", [])

    def save(self, session=None, history=None):
        if session is not None:
            if self.s["current"] is not None:
                self.s["current"]["collection_active_seconds"] = self._episode_prior + time.monotonic() - self._episode_started
            snapshot = session.snapshot(timeout=120., capture_external=lambda: {
                "history": history.snapshot(), "learner": self.learner.state_dict(),
                "coach": self.coach.state_dict()})
            self.s.update(session=snapshot, history=snapshot["external"]["history"],
                          learner=snapshot["external"]["learner"], coach=snapshot["external"]["coach"])
        else:
            self.s["learner"] = self.learner.state_dict()
            if self.coach is not None:
                self.s["coach"] = self.coach.state_dict()
        current = self.s["current"]
        preview = [{"n": p["n"], "environment": p["environment"], "episodes": len(p["episodes"]),
                    "success": sum(e["success"] for e in p["episodes"]) / len(p["episodes"]),
                    "raw_return": sum(e["task_return"] for e in p["episodes"]) / len(p["episodes"])}
                   for p in self.s["evaluations"] if p["episodes"]]
        self.store.checkpoint(self.s, progress={"stage": self.s["stage"], "n": self.s["n"],
            "total": self.total, "condition": self.condition, "phase": self.phase,
            "fits": len(self.s["metrics"]), "evaluation_preview": preview,
            "captured_in_episode": len(current["rows"]) if current else 0,
            "episodes": len(self.s["episodes"]), "latest_fit": self.s["metrics"][-1] if self.s["metrics"] else None,
            "latest_episode": self.s["episodes"][-1] if self.s["episodes"] else None})
        self.check()

    def _advance_coach(self, worker, history, *, acquired=None, terminal=False, transition_id=None):
        diagnostic = worker.diagnostics()
        now = diagnostic["simulator_time_s"]
        result = self.coach.advance(now, self.s["n"] if acquired is None else acquired, diagnostic, worker.config,
            terminal=terminal, transition_id=transition_id,
            publication_clock=None if terminal else lambda: worker.diagnostics()["simulator_time_s"])
        context = self.coach.policy_context(worker.diagnostics()["simulator_time_s"], scheduled_expiry=True)
        identities = [{k: c[k] for k in ("instruction_id", "text", "available_at_s")}
                      for window in self.coach.windows for c in window["instructions"]]
        history.update(context["instructions"], context["statuses"], instruction_ids=identities)
        if self.s["current"] is not None:
            self.s["current"].setdefault("coaching_events", []).extend(result["events"])
            self.s["current"].setdefault("measured_yaw", []).append({"time_s": now, "yaw_rad": chassis_yaw(diagnostic)})
        return result

    def episode(self, seed, *, environment=None, video=True):
        s = self.s
        self._episode_started = time.monotonic()
        self._episode_prior = s["current"].get("collection_active_seconds", 0.) if s["current"] else 0.
        evaluation = environment is not None
        cfg = simulator(s["n"], self.total, fixed="initial_A" if self.collection else environment)
        if s["current"] is not None:
            cfg = SimulatorConfig.from_dict(s["current"]["simulator_config"])
        history = History(self.cfg, cfg, s["history"])
        self.coach = CoachingWindows(derive_component_seed(self.seed, f"coach:{s['episode_counter']}"),
            condition="U" if evaluation else self.condition, total=self.total, initialization=self.collection)
        if s.get("coach") is not None:
            self.coach.load_state_dict(s["coach"])
        access = ModelAccess()
        access.version = self.learner.version
        worker = SimulatorWorker(cfg, run_id=f"coaching-{self.condition}-{self.seed}")

        def before_submit(session, observation):
            opened = session._open
            if opened and opened["action"] is not None:
                action = opened["action"]
                self.coach.record_action(opened["receipt"], generated_at_s=action["generated_at_s"],
                    decision_time_s=float(action["causal"]["decision_time"][0, -1]),
                    exposed_instruction_ids=action["instruction_ids"],
                    source="teacher" if self.collection else "student")
            acquired = s["current"]["n"] if evaluation else s["current"]["n"] + session._index
            self._advance_coach(worker, history, acquired=acquired)
            # Capture AFTER publication: the inference job and its receipt then
            # identify exactly the image/history that could see the new cue.
            return worker.observe()

        def infer(observation, version):
            causal = history(observation)
            exposed = list(causal.get("coaching_instruction_ids", [[]])[0])
            if self.collection:
                target = self.teacher.distribution(causal, probability_floor=self.cfg.learning.teacher_floor)
                return {"logits": target.reference_logits[0].cpu(), "probabilities": target.executed_probabilities[0].cpu(),
                    "teacher_probabilities": target.executed_probabilities[0].cpu(),
                    "original_probabilities": target.original_probabilities[0].cpu(),
                    "teacher_diagnostic": copy.deepcopy(self.teacher.last_diagnostic),
                    "feature": None, "causal": causal, "model_version": version,
                    "instruction_ids": exposed, "generated_at_s": worker.diagnostics()["simulator_time_s"]}
            feature = self.features(causal)
            z = self.learner.choose(feature, explore=not evaluation)
            return {"logits": z, "probabilities": torch.cat((z, torch.zeros(1))).softmax(-1),
                    "feature": feature, "causal": causal, "model_version": version,
                    "instruction_ids": exposed, "generated_at_s": worker.diagnostics()["simulator_time_s"]}

        session = CoachingSession(worker, access, infer=infer, pending_capacity=512, own_worker=True,
                                  before_submit=before_submit)
        try:
            if s["session"] is not None:
                session.restore(s["session"])
                if not s["session"]["terminal"]:
                    session.resume()
            else:
                schedule = guidance_schedule(s["n"], self.total, condition=self.condition, initialization=self.collection)
                if environment:
                    schedule = {**schedule, "difficulty": 0. if environment == "initial_A" else 1., "coaching_probability": 0.}
                s["current"] = {"seed": seed, "evaluation": evaluation, "environment": environment,
                    "phase": self.phase, "condition": self.condition, "n": s["n"], "rows": [], "scores": [],
                    "reward_mode": self.reward_mode, "adherence_enabled": self.adherence_enabled,
                    "complete": False, "frames": [], "coaching_events": [], "compliance": {},
                    "episode_id": f"{'eval' if evaluation else 'collect' if self.collection else 'train'}:{seed}:{s['episode_counter']}",
                    "simulator_config": asdict(cfg), "difficulty": schedule["difficulty"], "schedule": schedule}
                initial_observation = session.start(seed=seed, exposure_intervals=float(s["n"]), training=not evaluation)
                initial_image = observation_row(initial_observation)
                s["current"].update(initial_frame_ref=self.store.frames.put(initial_image["rgb"]),
                                     initial_frame_time=initial_observation.capture_time_s,
                                     initial_controls=initial_image["action"])
                s["current"]["layout"] = worker.diagnostics()["layout"]
                self.save(session, history)
            episode = s["current"]
            if episode["complete"]:
                return episode
            while True:
                interval = session.next_interval(timeout=120.)
                result, action = interval.result, interval.action
                post = observation_row(result.observation, previous_time=interval.observation.capture_time_s)
                row_id = episode["episode_id"] + f":{len(episode['rows'])}"
                row = {"id": row_id, "frame_ref": self.store.frames.put(post["rgb"]),
                    "elapsed": result.reward_components.elapsed_s, "bonus": result.reward_components.terminal_bonus,
                    "terminal": result.terminated, "truncated": result.truncated, "end_reason": result.end_reason,
                    "fallback": interval.fallback_reason, "timing": interval.timing,
                    "start_time": interval.receipt.received_at_s, "end_time": result.observation.capture_time_s,
                    "receipt": asdict(interval.receipt), "executed_controls": asdict(interval.receipt.executed_controls),
                    "requested_controls": asdict(interval.command.controls),
                    "action": None if action is None else action["logits"].cpu(),
                    "feature": None if action is None else action["feature"],
                    "coaching_events": copy.deepcopy(episode["coaching_events"])}
                if action is not None:
                    causal = action["causal"]
                    row["generated_at"] = float(causal["decision_time"][0, -1])
                    row["inference_finished_at_s"] = action["generated_at_s"]
                    row["instruction_ids_seen"] = action["instruction_ids"]
                    contexts, statuses = history.context_at(post["time"])
                    nxt = append_history({**causal, "contexts": [contexts], "status_events": [statuses]},
                                         post, self.cfg.model.context_length)
                    row.update(history=pack_history(causal, self.store.frames),
                               next_history=pack_history(nxt, self.store.frames), next_feature=None)
                    if self.collection:
                        row.update(teacher_probabilities=action["teacher_probabilities"],
                                   original_probabilities=action["original_probabilities"],
                                   teacher_diagnostic=action["teacher_diagnostic"])
                if not evaluation:
                    row["arrival"] = s["n"]
                    s["n"] += 1
                episode["rows"].append(row)
                episode["scores"].append(None)
                # Store all original physical states for later outcome-based selection.
                saved_snapshot = snapshot_at_capture(worker.snapshot()["backend"], post["time"])
                episode["frames"].append(saved_snapshot)
                row["execution_segments"] = execution_segments(saved_snapshot, post["time"] - row["elapsed"], post["time"])
                row["observed_controls"] = post["action"]
                if result.terminated or result.truncated:
                    session.pause(reason="episode_boundary")
                    self._advance_coach(worker, history, acquired=s["current"]["n"] + session._index,
                                        terminal=True, transition_id=row_id)
                    episode.update(complete=True, success=result.end_reason == "success", end_reason=result.end_reason)
                    self.save(session, history)
                    break
                self.save(session, history)
            return episode
        finally:
            session.close(timeout=120.)

    def _compliance(self):
        ep = self.s["current"]
        if not self.adherence_enabled or ep["evaluation"] or self.condition == "U":
            return
        for window in self.coach.windows:
            if not window["assessable"] or window["window_id"] in ep["compliance"]:
                continue
            close = window["closed_at_s"]
            closing = next((r for r in ep["rows"] if r["end_time"] >= close - 1e-5), ep["rows"][-1])
            # An expiry may occur between two acquisitions. Bind to its containing interval.
            window["closing_transition_id"] = closing["id"]
            cues = window["instructions"]
            if not cues:
                continue
            start = window["start_s"]
            controls = []
            tick_s = ep["simulator_config"]["physics_dt_s"]
            receipts = {round(r["start_time"] / tick_s): r for r in ep["rows"] if r["receipt"]["accepted"]}
            for row in ep["rows"]:
                for segment in row["execution_segments"]:
                    lo, hi = max(start, segment["start_time"]), min(close, segment["end_time"])
                    if hi <= lo:
                        continue
                    owner = receipts.get(round(segment["command_at_s"] / tick_s)) if segment["command_at_s"] is not None else None
                    imposed = owner is None or owner["action"] is None
                    controls.append({"start_time": lo, "end_time": hi, "executed": segment["executed"],
                        "fallback": imposed, "source": segment["source"],
                        "instruction_ids_seen": [] if imposed else owner.get("instruction_ids_seen", [])})
            if not controls:
                continue
            before_row = max((r for r in ep["rows"] if r["end_time"] <= start),
                             key=lambda r: r["end_time"], default=None)
            after_row = max((r for r in ep["rows"] if r["end_time"] <= close + 1e-5),
                            key=lambda r: r["end_time"], default=None)
            if after_row is None or after_row["end_time"] <= start:
                continue
            evidence = {"window_id": window["window_id"], "start_time": start, "end_time": close,
                "status": window["close_reason"], "instructions": [
                    {"instruction_id": c["instruction_id"], "text": c["text"], "issued_at": c["issued_at_s"],
                     "available_at": c["available_at_s"], "completed_at": c["completed_at_s"],
                     "ended_at": c["ended_at_s"], "status": c["status"]}
                    for c in cues], "controls": controls,
                "eligible_instruction_ids": list(window["eligible_instruction_ids"]),
                "image_times": {"before": before_row["end_time"] if before_row else ep["initial_frame_time"],
                                "after": after_row["end_time"]}}
            tick = time.monotonic()
            before_ref = before_row["frame_ref"] if before_row else ep["initial_frame_ref"]
            judgment = self.compliance.score(evidence, self.store.frames.get(before_ref),
                                              self.store.frames.get(after_row["frame_ref"]))
            receipt = self.coach.claim_reward(window["window_id"], judgment["judgment"], transition_id=closing["id"])
            ep["compliance"][window["window_id"]] = {"judgment": judgment, "receipt": receipt,
                                                      "evidence": evidence, "closing_row": closing["id"]}
            self.s["timings"]["scoring"] += time.monotonic() - tick
            self.save()

    def label(self):
        ep = self.s["current"]
        if not ep["complete"] or ep["rows"][-1]["truncated"]:
            raise RuntimeError("unfinished/infrastructure-truncated episode cannot receive task returns")
        if self.coach is None:
            self.coach = CoachingWindows(derive_component_seed(self.seed, f"coach:{self.s['episode_counter']}"),
                condition="U" if ep["evaluation"] else self.condition, total=self.total, initialization=self.collection)
            self.coach.load_state_dict(self.s["coach"])
        for i, row in enumerate(ep["rows"]):
            if ep["scores"][i] is None:
                tick = time.monotonic()
                ep["scores"][i] = self.scorer.score_rgb(self.store.frames.get(row["frame_ref"]))
                self.s["timings"]["scoring"] += time.monotonic() - tick
                self.save()
        if self.adherence_enabled:
            self._compliance()
        elif ep.get("compliance"):
            raise ValueError("task-only episode contains adherence outcomes")
        dt = torch.tensor([r["elapsed"] for r in ep["rows"]], dtype=torch.float64)
        q = torch.tensor([r["score"] for r in ep["scores"]], dtype=torch.float64)
        bonus = torch.tensor([r["bonus"] for r in ep["rows"]], dtype=torch.float64)
        guidance = torch.zeros_like(dt)
        row_indices = {r["id"]: i for i, r in enumerate(ep["rows"])}
        for item in ep["compliance"].values():
            guidance[row_indices[item["closing_row"]]] += item["receipt"]["raw"]
        parts = reward_components(dt, q, bonus, guidance)
        terminal = torch.tensor([r["terminal"] for r in ep["rows"]])
        discount = time_discounts(dt)
        returns = discounted_returns(parts["total"], discount, terminal)
        task_returns = discounted_returns(parts["task"], discount, terminal)
        ep.update(raw_stage_rewards=parts["total"].tolist(), task_stage_rewards=parts["task"].tolist(),
            guidance_stage_rewards=guidance.tolist(), coded_stage_rewards=parts["encoded"].tolist(),
            raw_returns=returns.tolist(), coded_returns=(returns / 10).tolist(), raw_return=float(returns[0]),
            task_return=float(task_returns[0]), guidance_return=float(returns[0] - task_returns[0]),
            coded_return=float(returns[0] / 10), raw_qwen_mean=float(q.mean()),
            coaching_windows=copy.deepcopy(self.coach.windows),
            coaching_audits=copy.deepcopy(self.coach.state_dict()["audits"]))
        for i, row in enumerate(ep["rows"]):
            row["reward_parts"] = {"task": float(parts["task"][i]), "guidance": float(guidance[i]),
                                   "raw": float(parts["total"][i]), "encoded": float(parts["encoded"][i])}
        records = []
        decisions = decision_returns(parts["encoded"], discount, terminal,
                                     torch.tensor([r["action"] is not None for r in ep["rows"]]))
        for decision in decisions:
            i, last = decision["first"], decision["last"]
            row, endpoint = ep["rows"][i], ep["rows"][last]
            if "post_history" not in endpoint:
                raw_rows = [{"rgb": self.store.frames.get(ep["initial_frame_ref"]),
                    "time": ep["initial_frame_time"], "elapsed": 0., "action": ep["initial_controls"]}]
                raw_rows.extend({"rgb": self.store.frames.get(r["frame_ref"]), "time": r["end_time"],
                    "elapsed": r["elapsed"], "action": r["observed_controls"]} for r in ep["rows"][:last + 1])
                context = self.coach.policy_context(endpoint["end_time"], scheduled_expiry=True)
                causal = history_batch(raw_rows, contexts=context["instructions"], statuses=context["statuses"],
                    deadline=ep["initial_frame_time"] + ep["simulator_config"]["deadline_s"],
                    length=self.cfg.model.context_length, width=ep["simulator_config"]["width"],
                    height=ep["simulator_config"]["height"])
                # Persistent task is part of every public causal history.
                from numerical_experiment.picar.fix2_data import TASK
                causal["contexts"][0].insert(0, {"text": TASK, "issued_at": 0.})
                endpoint["post_history"] = pack_history(causal, self.store.frames)
            if not self.collection and endpoint.get("post_feature") is None:
                endpoint["post_feature"] = self.features(unpack_history(endpoint["post_history"], self.store.frames))
                self.save()
            record = {"id": row["id"], "episode_id": ep["episode_id"], "feature": row["feature"],
                "next_feature": endpoint.get("post_feature"), "action": row["action"],
                "reward": decision["reward"].float(), "return": (returns[i] / 10).float(),
                "discount": decision["discount"].float(), "arrival": row.get("arrival", 0),
                "visits": 0, "history": row["history"], "next_history": endpoint["post_history"],
                "physical_rows": last - i + 1}
            if self.collection:
                record["teacher_probabilities"] = row["teacher_probabilities"]
            records.append(record)
        if ep["evaluation"] and records:
            before = copy.deepcopy(self.learner.state_dict())
            try:
                data, design, _ = self.learner.design(records)
                output = self.learner.coordinates.predict(self.learner.vector, data["feature"], data["action"])
                ep["heldout_nll"] = float(data_loss(output, data["action"], design)["nll"].mean().detach())
            finally:
                self.learner.load_state_dict(before)
        return records

    def finish_episode(self):
        ep = self.s["current"]
        if ep is not None:
            self.s["timings"]["collection"] += ep.get("collection_active_seconds", 0.)
            ep["difficulty_start"] = ep["difficulty"]
            # Layout difficulty is fixed at episode reset; the acquisition
            # schedule advances between resets, not within a sampled layout.
            ep["difficulty_end"] = ep["difficulty"]
            ep["guidance_status"] = "coached" if any(w["instructions"] for w in self.coach.windows) else "uncoached"
            ep["difficulty_group"] = difficulty_group(ep["difficulty"])
            ep["cohort"] = self.phase
            key = difficulty_group(ep["difficulty"])
            outcome = "success" if ep["success"] else "failure"
            media_key = f"{key}:{outcome}"
            preselected = len(self.s["episodes"]) == 0 if not ep["evaluation"] else ep.get("panel_index") == 0
            gallery = key is not None and self.s["media_counts"].get(media_key, 0) < 3
            from .video import save_episode
            ep["media_source"] = save_episode(self.store, ep, condition=self.condition,
                                              phase=self.phase, checkpoint=ep["n"])
            if (gallery or preselected) and not ep.get("videos"):
                from .video import render_video
                ep["videos"] = render_video(self.store, ep, condition=self.condition, phase=self.phase,
                    checkpoint=ep["n"], selection="preselected" if preselected else "outcome_gallery")
                if gallery:
                    self.s["media_counts"][media_key] = self.s["media_counts"].get(media_key, 0) + 1
            relative = "episodes/" + hashlib_id(ep["episode_id"]) + ".pt"
            path = self.store.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = io.BytesIO()
            torch.save(ep, payload)
            _atomic_bytes(path, payload.getvalue())
            ep["episode_artifact"] = str(path)
            if not ep["evaluation"]:
                from numerical_experiment.picar_head_action_report import control_summary
                action_summary = control_summary([(i, r["elapsed"], tuple(r["executed_controls"].values()))
                                                  for i, r in enumerate(ep["rows"])])
                segments = [segment for r in ep["rows"] for segment in r["execution_segments"]]
                executed = control_summary([(i, s["end_time"] - s["start_time"], tuple(s["executed"]))
                                            for i, s in enumerate(segments)])
                action_summary.update(moving_seconds=executed["moving_seconds"],
                                      mean_abs_steering_deg=executed["mean_abs_steering_deg"],
                                      executed_turn_bouts=executed["turn_bouts"])
                yaw = [r["yaw_rad"] for r in ep.get("measured_yaw", [])]
                action_summary["measured_net_yaw_change_deg"] = math.degrees(math.atan2(
                    math.sin(yaw[-1] - yaw[0]), math.cos(yaw[-1] - yaw[0]))) if yaw else None
                action_summary["measured_yaw_change_deg"] = sum(math.degrees(abs(math.atan2(math.sin(b-a), math.cos(b-a))))
                    for a, b in zip(yaw, yaw[1:])) if yaw else None
                action_summary["time_above_turn_threshold_s"] = sum(s["end_time"] - s["start_time"] for s in segments
                    if abs(s["executed"][2]) > math.radians(5.) and abs(s["executed"][3]) > .04)
                self.s["action_metrics"].append({**action_summary, "n": ep["n"],
                    "episode_id": ep["episode_id"], "difficulty": ep["difficulty"], "success": ep["success"],
                    "complete_history": True})
                self.s["coaching_metrics"].append({"episode_id": ep["episode_id"], "n": ep["n"],
                    "windows": len(self.coach.windows), "enabled": sum(w["enabled"] for w in self.coach.windows),
                    "cues": sum(len(w["instructions"]) for w in self.coach.windows),
                    "assessable": sum(w["assessable"] for w in self.coach.windows),
                    "abstentions": sum(w["abstentions"] for w in self.coach.windows),
                    "active_instruction_seconds": sum(max(0., (c["ended_at_s"] or w["closed_at_s"]) - c["available_at_s"])
                        for w in self.coach.windows for c in w["instructions"]),
                    "scored": len(ep["compliance"]), "task_return": ep["task_return"],
                    "adherence_enabled": self.adherence_enabled,
                    "guidance_return": ep["guidance_return"], "difficulty": ep["difficulty"]})
            self.s["archive_metrics"].append({"n": self.s["n"], **block_diagnostics(self.learner.archive)})
        self.s.update(current=None, session=None, history=None, coach=None)
        self.s["episode_counter"] += 1
        self.coach = None
        if self.features:
            self.features.clear()

    def trim(self):
        """Retire old context by physical time, including valid fallback covariates."""
        from numerical_experiment.picar.fix2_pipeline import eligible_history
        replay = self.s["replay"]
        cutoffs = {}
        for record in replay:
            at = float(record["history"]["decision_time"][0, -1])
            cutoffs[record["episode_id"]] = min(cutoffs.get(record["episode_id"], at), at)
        for record in replay:
            for raw_key, feature_key in (("history", "feature"), ("next_history", "next_feature")):
                raw = unpack_history(record[raw_key], self.store.frames)
                times = raw["decision_time"][0].tolist()
                allowed = {at for at in times if at >= cutoffs[record["episode_id"]]}
                trimmed = eligible_history(raw, allowed)
                if trimmed["images"].shape[1] != raw["images"].shape[1]:
                    record[raw_key] = pack_history(trimmed, self.store.frames)
                    record[feature_key] = self.features(trimmed)
        self.features.clear()

    def evaluate(self):
        n = self.s["target"]
        count = 8 if n == self.total else 4
        for environment in ("initial_A", "moderate_A"):
            panel = next((p for p in self.s["evaluations"] if p["n"] == n and p["environment"] == environment), None)
            if panel is None:
                panel = {"n": n, "acquired_count": self.s["n"], "environment": environment, "episodes": []}
                self.s["evaluations"].append(panel)
            for i in range(len(panel["episodes"]), count):
                layout_seed = component_seed(f"evaluation:{self.phase == 'development'}:{environment}:{i}")
                self.episode(layout_seed, environment=environment)
                self.s["current"]["panel_index"] = i
                self.label()
                ep = self.s["current"]
                self.finish_episode()
                panel["episodes"].append(episode_summary(ep))
                self.save()
        self.s["stage"] = "done" if n == self.total else "collect"
        self.save()

    def run(self):
        while self.s["stage"] != "done":
            stage = self.s["stage"]
            if stage == "evaluate":
                self.evaluate()
            elif stage == "collect":
                next_checkpoint = next((n for n in (256, 512, 768, self.total)
                    if self.s["n"] >= n and not any(p["n"] == n for p in self.s["evaluations"])), None)
                if self.s["current"] is None and self.s["n"] >= self.total and self.collection:
                    self.s["stage"] = "done"
                elif self.s["current"] is None and next_checkpoint is not None and not self.collection:
                    self.s.update(stage="evaluate", target=next_checkpoint)
                else:
                    layout_seed = derive_component_seed(self.seed, f"episode:{len(self.s['episodes'])}")
                    self.episode(layout_seed)
                    self.s["stage"] = "label"
                self.save()
            elif stage == "label":
                records = self.label()
                if self.collection:
                    self.s["initial_records"].extend(records)
                else:
                    known = {r["id"] for r in self.s["replay"]} | set(self.s["retired_ids"]) | self.initial_training_ids
                    if any(r["id"] in known for r in records):
                        raise ValueError("duplicate observed record admission")
                    self.s["replay"].extend(records)
                    self.s["ready"].extend(r["id"] for r in records)
                ep = self.s["current"]
                self.finish_episode()
                self.s["episodes"].append(episode_summary(ep))
                self.s["stage"] = "collect" if self.collection else "fit"
                self.save()
            elif stage == "fit":
                self.work()
                self.s["stage"] = "collect"
                self.save()
            else:
                raise ValueError("unknown coaching trajectory stage: " + stage)
        return {"method": self.learner.method, "condition": self.condition, "phase": self.phase,
            "reward_mode": self.reward_mode, "adherence_enabled": self.adherence_enabled,
            "seed": self.seed, "n": self.s["n"], "metrics": self.s["metrics"],
            "episodes": self.s["episodes"], "evaluations": self.s["evaluations"],
            "records": len(self.s["initial_records"]), "coaching_metrics": self.s["coaching_metrics"],
            "action_metrics": self.s["action_metrics"],
            "archive_metrics": self.s["archive_metrics"], "archive": self.learner.archive.diagnostics(),
            "retired": len(self.s["retired_ids"]), "timings": self.s["timings"]}


def hashlib_id(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:20]
