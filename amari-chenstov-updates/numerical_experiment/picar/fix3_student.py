"""Bounded Fix 3 bootstrap and Replay/EWC pilot, with transactional recovery.

Only completed physical episodes receive full-return labels. The independent
control/physics clocks keep running during learner work and checkpoint I/O.
Bootstrap KL uses recorded executed distributions, never renewed teacher calls.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
import random
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.continual import picar_math as pm
from src.seeding import derive_seed_map
from src.robotics.worker import SimulatorWorker
from .config import canonical_json
from .control import ControlSession
from .fix2_data import TASK, history_batch, observation_row
from .fix2_pipeline import eligible_history
from .fix3_roles import CenteringReward, CenteringTeacher
from .learner import LearningBatch, MemoryLearner
from .models import batch_digest
from .runtime import ModelAccess


SCHEMA = 1
KL_PASSES = 1
NLL_PASSES = 1


def _rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class _History:
    """Inference-owned history, captured under ControlSession's quiescence."""

    def __init__(self, config, sim_config, state=None):
        self.config, self.sim_config = config, sim_config
        self.lock = threading.RLock()
        self.state = copy.deepcopy(state) if state else {
            "live": [], "histories": {}, "deadline": None, "statuses": []}

    def __call__(self, observation):
        with self.lock:
            s = self.state
            if s["deadline"] is None:
                s["deadline"] = observation.capture_time_s + self.sim_config.deadline_s
            live = s["live"]
            if not live or observation.capture_time_s > live[-1]["time"]:
                live.append(observation_row(observation, previous_time=live[-1]["time"] if live else None))
            live[:] = live[-self.config.model.context_length:]
            history = history_batch(live, contexts=[{"text": TASK, "issued_at": 0.}],
                deadline=s["deadline"], length=self.config.model.context_length,
                statuses=[x for x in s["statuses"] if x["available_at"] <= observation.capture_time_s],
                width=self.sim_config.width, height=self.sim_config.height)
            s["histories"][observation.observation_id] = history
            return history

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)


def _batch(item, eligible):
    # Times alone are not identities: clocks may reset between episodes.
    available = {float(r["history"]["decision_time"][0, -1]) for r in eligible
                 if r["episode_id"] == item["episode_id"]}
    history = eligible_history(item["history"], available)
    return LearningBatch([history], torch.tensor([item["baseline"]]),
        torch.tensor([item["bin"]]), item["logits"].reshape(1, 7),
        torch.tensor([item["arrival"]]))


def _trim_eligible(eligible):
    # Remove retired pixels, not just their endpoints, from surviving contexts.
    for item in eligible:
        item["history"] = _batch(item, eligible).histories[0]


def _completed_labels(pending, config, edges):
    if not pending or not pending[-1]["terminal"] or any(p["truncated"] for p in pending):
        raise ValueError("full returns require a completed, nontruncated episode")
    elapsed = torch.tensor([p["elapsed"] for p in pending], dtype=torch.float64)
    terminal = torch.tensor([p["terminal"] for p in pending])
    bonus = torch.tensor([p["bonus"] for p in pending], dtype=torch.float64)
    scores = torch.tensor([p["score"]["score"] for p in pending], dtype=torch.float64)
    rewards = pm.dense_visual_rewards(elapsed, scores, bonus, terminal)
    discounts = pm.time_discounts(elapsed, config.gamma, config.nominal_period_s)
    returns = pm.discounted_returns(rewards, discounts, terminal)
    return rewards, returns, pm.bin_returns(returns, edges.double().cpu())


@torch.no_grad()
def calibrate_exploration(learner, eligible, calibration, *, seed):
    """Frozen common calibration; a separate CPU generator never touches actor RNG."""
    if not eligible:
        raise ValueError("exploration calibration requires eligible bootstrap histories")
    indices = sorted({0, len(eligible) // 2, len(eligible) - 1})
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((len(indices), 256, 7), generator=generator, dtype=torch.float64)
    means = [learner.choose_action(_batch(eligible[i], eligible).histories[0],
                                   explore=False)["logits"].double() for i in indices]
    rows = []
    for std in (.1, .5, 1., 2., 4.):
        contexts = []
        for i, mean, probes in zip(indices, means, noise):
            z = mean + std * probes
            q = torch.cat((z, z.new_zeros(256, 1)), -1).softmax(-1)
            controls = [calibration.map_distribution(row) for row in q]
            drive = float(np.std([c.drive_mps for c in controls]))
            steer = math.degrees(float(np.std([c.steering_rad for c in controls])))
            contexts.append({"arrival": eligible[i]["arrival"], "drive_sd_mps": drive,
                "steering_sd_degrees": steer, "passes": drive >= .02 and steer >= 2.})
        rows.append({"std": std, "contexts": contexts,
                     "passes": sum(x["passes"] for x in contexts) > len(contexts) / 2})
    passing = [r["std"] for r in rows if r["passes"]]
    return {"selected_exploration_std": min(passing) if passing else 4.,
        "criterion_met": bool(passing), "seed": seed, "probes_per_context": 256,
        "selection": "smallest_majority_pass_else_largest_with_explicit_failure",
        "rows": rows, "frozen_initial": True, "rollouts": 0}


class _Run:
    def __init__(self, model, config, sim_config, store, stopped, *, mode, seed,
                 max_transitions, initial=None, teacher_variant="centering", evaluations=()):
        config.learning.validate()
        if config.learning.fd_strength != 0 or config.learning.retired_archive_replay:
            raise ValueError("Fix 3 is Replay/EWC only, with no retired archive replay")
        if not sim_config.realtime or type(max_transitions) is not int or max_transitions < 1:
            raise ValueError("a positive transition budget and independent realtime clock are required")
        self.model, self.config, self.sim_config = model, config, sim_config
        self.store, self.stopped, self.mode = store, stopped, mode
        self.started = time.monotonic()
        self.seeds = replace(config, seed=seed).seeds(0)
        self.learner = MemoryLearner(model, config.learning, self.seeds)
        self.scorer = CenteringReward(model.encoder)
        self.teacher = CenteringTeacher(model.encoder, variant=teacher_variant) if mode == "bootstrap" else None
        self.replay_rng = torch.Generator().manual_seed(self.seeds["replay"])
        self.evaluations = list(evaluations)
        self.identity = {"schema": SCHEMA, "mode": mode, "seed": seed,
            "config": asdict(config), "sim_config": asdict(sim_config),
            "max_transitions": max_transitions, "teacher_variant": teacher_variant,
            "evaluations": [{**x, "config": asdict(x["config"])} for x in self.evaluations]}
        saved = store.load_checkpoint(trusted=True)
        if saved:
            if canonical_json(saved["identity"]) != canonical_json(self.identity):
                raise ValueError("Fix 3 resume identity differs")
            self.s = saved
            self.learner.load_state_dict(saved["learner"])
            self.scorer.load_state_dict(saved["scorer"])
            self.replay_rng.set_state(saved["replay_rng"])
            _restore_rng(saved["rng"], self.learner.device)
        else:
            if mode == "pilot":
                if not initial or initial.get("stage") != "done" or not initial.get("distill_receipt"):
                    raise ValueError("pilot needs a completed bootstrap checkpoint with KL receipt")
                self.learner.load_state_dict(initial["learner"])
                for name, generator in self.learner.generators.items():
                    generator.manual_seed(self.seeds[name])
            self.s = {"identity": self.identity, "schema_version": SCHEMA,
                "stage": "initial_evaluation" if mode == "pilot" else "collect",
                "seeds": self.seeds, "acquired_count": 0, "episodes": [], "evaluations": [],
                "eligible": copy.deepcopy(initial["eligible"]) if initial else [],
                "retired": list(initial["retired"]) if initial else [], "ready": [],
                "arrival_offset": initial["acquired_count"] if initial else 0,
                "distill_receipt": copy.deepcopy(initial["distill_receipt"]) if initial else None,
                "selected_exploration_std": config.learning.exploration_std,
                "exploration_calibration": copy.deepcopy(initial.get("exploration_calibration")) if initial else None,
                "distinct_bootstrap_samples": initial.get("distinct_bootstrap_samples", 0) if initial else 0,
                "session": None, "history": None, "current": None, "pending": [],
                "intervals": [], "videos": [], "fallback_counts": {},
                "online_updates": 0, "sleep_updates": 0, "kl_step": 0, "kl_losses": [],
                "kl_optimizer": None, "nll_step": 0,
                "timings": {"fit_seconds": 0., "fisher_seconds": 0., "scoring_seconds": 0.,
                    "kl_seconds": 0., "checkpoint_seconds": 0., "sleep_seconds": 0.,
                    "evaluation_seconds": 0., "active_seconds": 0.}}
        self.base_active = self.s["timings"]["active_seconds"]
        self.model.distillation_receipt = copy.deepcopy(self.s["distill_receipt"])
        self.model.eval()

    def save(self, session=None, history=None, *, metrics=()):
        started = time.monotonic()
        if session is not None:
            snapshot = session.snapshot(timeout=120., capture_external=history.snapshot)
            self.s["history"] = snapshot.pop("external")
            self.s["session"] = snapshot
            # Student RNG/parameters must be the same instant as its prediction.
            self.s["learner"] = snapshot["learner"] or self.learner.state_dict()
        else:
            self.s["learner"] = self.learner.state_dict()
        self.s["scorer"] = self.scorer.state_dict()
        self.s["replay_rng"] = self.replay_rng.get_state()
        self.s["rng"] = _rng_state(self.learner.device)
        self.s["timings"]["active_seconds"] = self.base_active + time.monotonic() - self.started
        self.store.checkpoint(self.s, metrics=metrics, progress={"phase": 1,
            "status": f"Fix 3 {self.mode}: {self.s['stage']}",
            "counts": {"step": self.s["acquired_count"], "target": self.identity["max_transitions"]}})
        self.s["timings"]["checkpoint_seconds"] += time.monotonic() - started
        self.stopped()

    def work_one(self, session=None):
        s = self.s
        item = next(x for x in s["eligible"] if x["arrival"] == s["ready"][0])
        replay = [x for x in s["eligible"] if x["learned"] and x is not item]
        other = replay[int(torch.randint(len(replay), (), generator=self.replay_rng))] if replay else None
        kwargs = {"new": _batch(item, s["eligible"]),
            "replay": _batch(other, s["eligible"]) if other else None,
            "acquired_count": s["arrival_offset"] + s["acquired_count"], "steps": 1}
        if session is None:
            metrics = self.learner.fit(**kwargs)
        else:
            metrics = session.update(cooperative=True, **kwargs)
        s["timings"]["fit_seconds"] += metrics["update_seconds"]
        s["online_updates" if session else "sleep_updates"] += 1
        item["learned"] = True
        s["ready"].pop(0)
        return metrics

    def retire_one(self, access=None):
        s = self.s
        old = s["eligible"][0]
        if not old["learned"] or old["arrival"] in s["retired"]:
            raise RuntimeError("FIFO retirement must happen once, after fitting")
        if access is None:
            metrics = self.learner.retire(_batch(old, s["eligible"]))
        else:
            with access.update():
                metrics = self.learner.retire(_batch(old, s["eligible"]))
        s["timings"]["fisher_seconds"] += metrics["fisher_seconds"]
        s["retired"].append(old["arrival"])
        s["eligible"].pop(0)
        _trim_eligible(s["eligible"])

    def episode(self, sim_config, *, seed, exposure, evaluation=False, video=False):
        s = self.s
        if s["current"] is not None and s["current"]["complete"]:
            return
        history = _History(self.config, sim_config, s["history"])
        access = ModelAccess()
        access.version = self.learner.version

        def teacher_action(observation, version):
            before = time.monotonic()
            batch = history(observation)
            target = self.teacher.distribution(batch, probability_floor=self.config.learning.teacher_floor)
            # The bootstrap likelihood uses the declared fixed zero baseline.
            # Targets and finite reference logits are recorded before execution.
            return {"probabilities": target.executed_probabilities[0].cpu(),
                "logits": target.reference_logits[0].cpu(), "baseline": 0.,
                "original_probabilities": target.original_probabilities[0].cpu(),
                "target_kind": target.target_kind, "probability_floor": target.probability_floor,
                "diagnostic": copy.deepcopy(self.teacher.last_diagnostic),
                "model_version": version, "inference_seconds": time.monotonic() - before}

        callback = self.mode == "bootstrap" and not evaluation
        worker = SimulatorWorker(sim_config, run_id=f"fix3-{self.mode}-{self.identity['seed']}")
        kwargs = {"infer": teacher_action} if callback else {
            "learner": self.learner, "history_builder": history, "explore": not evaluation}
        session = ControlSession(worker, access, pending_capacity=512, own_worker=True, **kwargs)
        try:
            if s["session"] is not None:
                session.restore(s["session"])
                if not s["session"]["terminal"]:
                    session.resume()
            else:
                s["pending"] = []
                s["current"] = {"episode_id": f"{self.mode}:{len(s['episodes'])}",
                    "seed": seed, "reset_exposure": exposure, "complete": False,
                    "evaluation": evaluation, "frames": [], "intervals": [],
                    "end_reason": None, "success": None, "label_complete": False}
                session.start(seed=seed, exposure_intervals=exposure, training=not evaluation)
                s["current"]["layout"] = worker.diagnostics()["layout"]
                self.save(session, history)
            current = s["current"]
            while evaluation or s["acquired_count"] < self.identity["max_transitions"]:
                try:
                    interval = session.next_interval(timeout=120.)
                except StopIteration:
                    if not current["complete"]:
                        raise RuntimeError("ended control session has no recorded terminal interval")
                    break
                result = interval.result
                observation = result.observation
                action = interval.action
                with history.lock:
                    causal = None if action is None or interval.policy_observation is None else copy.deepcopy(
                        history.state["histories"][interval.policy_observation.observation_id])
                row = {"index": interval.index, "fallback": interval.fallback_reason,
                    "elapsed_s": result.reward_components.elapsed_s, "end_reason": result.end_reason,
                    "timing": interval.timing, "terminated": result.terminated,
                    "truncated": result.truncated, "model_version": interval.model_version,
                    "controls": asdict(observation.executed_controls),
                    "policy_observation_id": None if interval.policy_observation is None else
                        interval.policy_observation.observation_id,
                    "probabilities": None if action is None else action["probabilities"].tolist(),
                    "logits": None if action is None else action["logits"].tolist(),
                    "baseline": None if action is None else action["baseline"],
                    "teacher_diagnostic": action.get("diagnostic") if action else None}
                post_row = observation_row(observation, previous_time=interval.observation.capture_time_s)
                row["frame_ref"] = self.store.frames.put(post_row["rgb"])
                if causal is not None:
                    pixels = causal["images"][0].detach().cpu().mul(255).round().clamp(0, 255).byte()
                    row.update(context_refs=[self.store.frames.put(frame.permute(1, 2, 0).numpy())
                                             for frame in pixels],
                        context_actions=causal["executed_actions"][0].tolist(),
                        context_elapsed=causal["elapsed"][0, :, 0].tolist(),
                        context_decision_times=causal["decision_time"][0].tolist(),
                        context_start_time=float(causal["start_time"][0]),
                        contexts=copy.deepcopy(causal["contexts"][0]),
                        context_status_events=copy.deepcopy(causal["status_events"][0]),
                        episode_deadline=float(causal["episode_deadline"][0]),
                        history_digest=batch_digest(causal))
                current["intervals"].append(row)
                if evaluation:
                    if video:
                        # A snapshot later reconstructs BOTH views at one physical
                        # instant, avoiding mislabeled asynchronous top-down frames.
                        current["frames"].append(worker.snapshot()["backend"])
                else:
                    s["acquired_count"] += 1
                    row["arrival"] = s["arrival_offset"] + s["acquired_count"] - 1
                    s["intervals"].append({**row, "episode": len(s["episodes"])})
                    key = interval.fallback_reason or "policy"
                    s["fallback_counts"][key] = s["fallback_counts"].get(key, 0) + 1
                    post = history_batch([post_row], contexts=[{"text": TASK, "issued_at": 0.}],
                        deadline=history.state["deadline"] or sim_config.deadline_s,
                        width=sim_config.width, height=sim_config.height)
                    s["pending"].append({"elapsed": result.reward_components.elapsed_s,
                        "bonus": result.reward_components.terminal_bonus, "terminal": result.terminated,
                        "truncated": result.truncated, "post": post,
                        "pan": observation.executed_controls.pan_rad,
                        "action": copy.deepcopy(action), "history": causal,
                        "arrival": row["arrival"], "score": None})
                if result.terminated or result.truncated:
                    session.pause(reason="episode_boundary")
                    current.update(complete=True, end_reason=result.end_reason,
                        success=result.end_reason == "success", truncated=result.truncated)
                    self.save(session, history)
                    break
                if not evaluation and self.mode == "pilot" and s["ready"]:
                    self.work_one(session)
                    if len(s["eligible"]) > self.config.learning.replay_capacity and s["eligible"][0]["learned"]:
                        self.retire_one(access)
                self.save(session, history)
            if not current["complete"]:
                current.update(complete=True, end_reason="transition_cap_incomplete", success=None,
                               truncated=True)
                # Preserve the open physical interval and all queued captures.
                # They are recovery evidence, never fabricated failure returns.
                self.save(session, history)
                current["unconsumed_intervals"] = len(s["session"]["pending"])
        finally:
            session.close(timeout=120.)

    def label_episode(self):
        s, current = self.s, self.s["current"]
        if current["label_complete"]:
            return
        if current["truncated"]:
            current["label_complete"] = True
            current["unlabeled_intervals"] = len(s["pending"])
            self.save()
            return
        before = time.monotonic()
        for item in s["pending"]:
            if item["score"] is None:
                tick = time.monotonic()
                item["score"] = self.scorer.score(item["post"], item["pan"])
                s["timings"]["scoring_seconds"] += time.monotonic() - tick
                self.save()
        rewards, returns, bins = _completed_labels(s["pending"], self.config.learning, self.learner.edges)
        for i, item in enumerate(s["pending"]):
            action = item["action"]
            if action is None or item["history"] is None:
                continue
            record = {"episode_id": current["episode_id"], "history": item["history"],
                "arrival": item["arrival"], "baseline": action["baseline"],
                "logits": action["logits"].clone(), "bin": int(bins[i]),
                "return": float(returns[i]), "learned": False}
            if self.mode == "bootstrap":
                record.update(executed_probabilities=action["probabilities"].reshape(1, 8).clone(),
                    original_probabilities=action["original_probabilities"].reshape(1, 8).clone(),
                    target_kind=action["target_kind"], probability_floor=action["probability_floor"])
            else:
                s["ready"].append(record["arrival"])
            s["eligible"].append(record)
        current.update(label_complete=True, returns=returns.tolist(), stage_rewards=rewards.tolist(),
            visual_scores=[p["score"] for p in s["pending"]],
            terminal_bonuses=[p["bonus"] for p in s["pending"]])
        s["timings"]["sleep_seconds"] += time.monotonic() - before
        self.save()

    def collect(self):
        s = self.s
        reserve = math.ceil(self.sim_config.deadline_s / self.sim_config.control_interval_s) + 2
        while s["acquired_count"] < self.identity["max_transitions"] or s["current"] is not None:
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
                # Keep a separate recovery payload for the unfinished episode.
                s["incomplete_episode"] = {"pending": s["pending"], "session": s["session"],
                    "history": s["history"], "episode": copy.deepcopy(current)}
            s["current"], s["session"], s["history"], s["pending"] = None, None, None, []
            self.save(metrics=[{"phase": 1, "condition": "replay_ewc", "replica": self.identity["seed"],
                "step": s["acquired_count"], "episode": episode, "end_reason": current["end_reason"],
                "success": current["success"], "new_transitions": s["acquired_count"],
                "retired_transitions": self.learner.retired_count, "model_version": self.learner.version}])
            if current["truncated"]:
                break
        s["stage"] = "distill" if self.mode == "bootstrap" else "drain"
        self.save()

    def distill(self):
        s = self.s
        records = s["eligible"]
        if len(records) <= self.config.learning.replay_capacity:
            raise RuntimeError("bootstrap has too few completed policy samples for disjoint Fisher/replay")
        if self.model.distillation_receipt is not None:
            raise ValueError("one-time recorded KL cannot be repeated")
        optimizer = torch.optim.Adam(self.learner.params.values(),
                                     lr=self.config.learning.learning_rate, weight_decay=0.)
        if s["kl_optimizer"] is not None:
            optimizer.load_state_dict(copy.deepcopy(s["kl_optimizer"]))
        steps = KL_PASSES * len(records)
        while s["kl_step"] < steps:
            before = time.monotonic()
            record = records[s["kl_step"] % len(records)]
            mean = self.model(record["history"])["action_mean"]
            logits = torch.cat((mean, mean.new_zeros(mean.shape[0], 1)), -1)
            loss = F.kl_div(logits.log_softmax(-1), record["executed_probabilities"].to(logits),
                            reduction="batchmean")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite recorded bootstrap KL")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in self.learner.params.values()):
                raise FloatingPointError("nonfinite bootstrap KL gradient")
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.learner.params.values()):
                raise FloatingPointError("nonfinite bootstrap KL parameter")
            self.learner.version += 1
            self.model.mark_published()
            s["kl_step"] += 1
            s["kl_losses"].append(float(loss.detach()))
            s["kl_optimizer"] = copy.deepcopy(optimizer.state_dict())
            s["timings"]["kl_seconds"] += time.monotonic() - before
            self.save()
        digests = [batch_digest(r["history"]) for r in records]
        s["distill_receipt"] = {"stage": "one_time_recorded_teacher_kl", "steps": steps,
            "batch_size": 1, "batches": len(records), "learning_rate": self.config.learning.learning_rate,
            "target_kinds": sorted({r["target_kind"] for r in records}),
            "initial_kl": s["kl_losses"][0], "last_preupdate_kl": s["kl_losses"][-1],
            "history_digests": digests, "resume_schema": SCHEMA,
            "teacher_provenance": self.teacher.provenance, "bootstrap_baseline": 0.}
        self.model.distillation_receipt = copy.deepcopy(s["distill_receipt"])
        s["distinct_bootstrap_samples"] = len(set(digests))
        s["stage"] = "initial_nll"
        self.save()

    def initial_nll(self):
        s = self.s
        records = s["eligible"]
        while s["nll_step"] < NLL_PASSES * len(records):
            record = records[s["nll_step"] % len(records)]
            metrics = self.learner.fit(new=_batch(record, records), replay=None,
                                      acquired_count=s["acquired_count"], steps=1)
            s["timings"]["fit_seconds"] += metrics["update_seconds"]
            record["learned"] = True
            s["nll_step"] += 1
            self.save()
        s["stage"] = "initial_fisher"
        self.save()

    def drain(self):
        s = self.s
        while s["ready"]:
            before = time.monotonic()
            self.work_one()
            s["timings"]["sleep_seconds"] += time.monotonic() - before
            self.save()
            while len(s["eligible"]) > self.config.learning.replay_capacity and s["eligible"][0]["learned"]:
                self.retire_one()
                self.save()
        while len(s["eligible"]) > self.config.learning.replay_capacity:
            self.retire_one()
            self.save()
        s["stage"] = "calibration" if self.mode == "bootstrap" else "final_evaluation"
        self.save()

    def calibration(self):
        seed = derive_seed_map(self.identity["seed"], ("exploration_calibration",))["exploration_calibration"]
        result = calibrate_exploration(self.learner, self.s["eligible"], self.sim_config.calibration, seed=seed)
        self.s["exploration_calibration"] = result
        self.s["selected_exploration_std"] = result["selected_exploration_std"]
        self.s["stage"] = "done"
        self.save()

    def evaluate(self, checkpoint):
        s = self.s
        for panel in self.evaluations:
            found = next((e for e in s["evaluations"] if e["checkpoint"] == checkpoint
                          and e["difficulty"] == panel["difficulty"]), None)
            if found is None:
                found = {"checkpoint": checkpoint, "difficulty": panel["difficulty"], "rows": []}
                s["evaluations"].append(found)
            for seed in panel["seeds"][len(found["rows"]):]:
                before = time.monotonic()
                # Preselected first easy layout, identical before and after.
                video = panel is self.evaluations[0] and seed == panel["seeds"][0]
                self.episode(panel["config"], seed=seed, exposure=0., evaluation=True, video=video)
                current = s["current"]
                if current["truncated"]:
                    raise RuntimeError("evaluation infrastructure truncation is not a task failure")
                clips = self.render_video(current, checkpoint, panel["difficulty"]) if video else []
                row = {k: v for k, v in current.items() if k not in {"frames", "label_complete"}}
                row.update(explore=False, learning=False, reward_scoring=False,
                    videos=clips, physical_seconds=sum(x["elapsed_s"] for x in current["intervals"]))
                found["rows"].append(row)
                s["videos"].extend(clips)
                s["session"], s["history"], s["current"], s["pending"] = None, None, None, []
                s["timings"]["evaluation_seconds"] += time.monotonic() - before
                self.save()
        s["stage"] = "collect" if checkpoint == "initial" else "done"
        self.save()

    def render_video(self, current, checkpoint, difficulty):
        from PIL import Image
        from src.robotics.simulator import PyBulletBackend, SimulatorConfig
        from .report import assemble_video

        snapshots = current["frames"]
        if not snapshots:
            return []
        rows = []
        config = SimulatorConfig.from_dict(snapshots[0]["config"])
        with PyBulletBackend(config, run_id=snapshots[0]["run_id"]) as renderer:
            for snapshot in snapshots:
                renderer.restore(snapshot)
                timestamp = renderer.time_s
                if rows and timestamp <= rows[-1]["time_s"]:
                    continue
                paths = []
                for rgb in (renderer.render(), renderer.render_topdown()):
                    ref = self.store.frames.put(rgb)
                    path = self.store.root / (ref[7:] + ".png")
                    if not path.exists():
                        Image.fromarray(rgb).save(path)
                    paths.append(str(path))
                rows.append({"time_s": timestamp, "ego_time_s": timestamp, "topdown_time_s": timestamp,
                    "ego_path": paths[0], "topdown_path": paths[1], "instruction": TASK,
                    "outcome": renderer.diagnostics()["end_reason"] or "running"})
        path = self.store.root / f"student-{checkpoint}-{difficulty}-{current['seed']}-{time.time_ns()}.mp4"
        result = assemble_video(rows, path, seed=current["seed"], fps=5.)
        return [{**result, "checkpoint": checkpoint, "difficulty": difficulty,
                 "layout_id": current["seed"], "view": "ego+topdown",
                 "views": "synchronized_reconstruction_of_recorded_physics_snapshots"}]

    def run(self):
        while self.s["stage"] != "done":
            stage = self.s["stage"]
            if stage == "initial_evaluation":
                self.evaluate("initial")
            elif stage == "collect":
                self.collect()
            elif stage == "distill":
                self.distill()
            elif stage == "initial_nll":
                self.initial_nll()
            elif stage in ("initial_fisher", "drain"):
                self.drain()
            elif stage == "calibration":
                self.calibration()
            elif stage == "final_evaluation":
                self.evaluate("final")
            else:
                raise ValueError(f"unknown Fix 3 checkpoint stage: {stage}")
        s = self.s
        return {"schema_version": SCHEMA, "kind": f"fix3_{self.mode}", "seed": self.identity["seed"],
            "condition": "replay_ewc", "acquired_count": s["acquired_count"],
            "transitions": s["acquired_count"], "episodes": s["episodes"], "evaluations": s["evaluations"],
            "timings": s["timings"], "retired_count": self.learner.retired_count,
            "retiredcount": self.learner.retired_count, "fallback_counts": s["fallback_counts"],
            "fallbackcounts": s["fallback_counts"], "videos": s["videos"],
            "distinct_bootstrap_samples": s["distinct_bootstrap_samples"],
            "selected_exploration_std": s["selected_exploration_std"],
            "exploration_calibration": s["exploration_calibration"],
            "distill_receipt": s["distill_receipt"], "online_updates": s["online_updates"],
            "sleep_updates": s["sleep_updates"], "pending_updates": len(s["ready"]),
            "eligible_count": len(s["eligible"]), "incomplete_episode": "incomplete_episode" in s,
            "unused_horizon_reserve": s.get("unused_horizon_reserve", 0),
            "checkpoint_asset": "RunStore.load_checkpoint(trusted=True)",
            "bootstrap_batch_size": 1, "retired_archive_replay": False,
            "limitations": ["Development pilot; no FD comparison or confirmatory efficacy claim.",
                "Labeling and final backlog drain are charged episode-boundary work.",
                "Capped/infrastructure-truncated episodes never receive invented returns."]}


def bootstrap(model, config, sim_config, store, stopped, *, teacher_variant="centering", max_transitions=300):
    """Return JSON summary; leave the reusable learner asset in store's checkpoint."""
    sim_config = replace(sim_config, curriculum_exposure_override=0., domain_override="A")
    return _Run(model, config, sim_config, store, stopped, mode="bootstrap", seed=config.seed,
        max_transitions=max_transitions, teacher_variant=teacher_variant).run()


def pilot(model, config, initial, sim_config, store, stopped, *, seed, max_transitions=320,
          evaluation_configs=None):
    """One independent seed from the shared trusted bootstrap, no teacher queries."""
    if evaluation_configs is None:
        from .fix3_rollout import evaluation_configs as fixed_panels
        evaluation_configs = fixed_panels()
    config = replace(config, learning=replace(config.learning,
        exploration_std=initial["selected_exploration_std"]))
    return _Run(model, config, sim_config, store, stopped, mode="pilot", seed=seed,
        max_transitions=max_transitions, initial=initial, evaluations=evaluation_configs).run()

