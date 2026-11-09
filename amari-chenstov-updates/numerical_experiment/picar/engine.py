"""Transactional acceptance runner using the production numerical components.

Phase 0's tiny, accelerated experiment is explicitly non-scientific. Real-model
calibration uses qwen_acceptance; no physical backend can be selected here.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.continual import picar_math as pm
from src.robotics.actions import Calibration
from src.robotics.simulator import PyBulletBackend, SimulatorConfig
from src.seeding import derive_component_seed
from .config import PicarConfig
from .learner import LearningBatch, MemoryLearner
from .models import build_tiny_model
from .runtime import ComputeLedger, StopController, source_fingerprint
from .schemas import TransitionRecord, metric_row
from .storage import FIFOReplay, RunStore
from .teacher import DeterministicFakeTeacher, distill_once


INSTRUCTION = "Go around the obstacle, approach the red ball, and stop."


def execution_seeds(config: PicarConfig, replica: int, *, bootstrap=False) -> dict[str, int]:
    phase = "picar.bootstrap" if bootstrap else "picar.online"
    return {name: derive_component_seed(seed, phase)
            for name, seed in config.seeds(0 if bootstrap else replica).items()}


def _parameter_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state["parameters"].items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SmokeEngine:
    """Recoverable numerical/pipeline acceptance, never a robot capability run.

    The pending episode is bounded separately from C. Histories are reconstructed
    from eligible observations only, masking predecessors at the FIFO boundary.
    No archived observation is fetched to improve a training history for free.
    """

    def __init__(self, config: PicarConfig, store: RunStore, *, condition: str,
                 replica: int, bootstrap: bool = False, initial: dict | None = None):
        if config.mode != "smoke" or config.model.kind != "tiny":
            raise ValueError("accelerated acceptance only; use calibrated real-time stages for research")
        self.config, self.store, self.condition, self.replica = config, store, condition, replica
        self.bootstrap = bootstrap
        self.seeds = execution_seeds(config, replica, bootstrap=bootstrap)
        mc = config.model
        torch.set_num_threads(config.runtime.threads)
        self.model = build_tiny_model(seed=config.seeds(0)["initialization"],
            hidden_size=mc.hidden_size, return_bins=mc.return_bins, top_layers=mc.top_layers,
            ltm_rank=mc.ltm_rank, stm_rank=mc.stm_rank, context_length=mc.context_length,
            dtype=getattr(torch, config.runtime.dtype), device=config.runtime.device)
        self.learner = MemoryLearner(self.model, config.learning, self.seeds)
        self.replay = FIFOReplay(config.learning.replay_capacity)
        self.replay_rng = random.Random(self.seeds["replay"])
        self.backend = PyBulletBackend(SimulatorConfig(
            width=config.image_width, height=config.image_height,
            control_interval_s=config.simulation_period_s,
            deadline_s=config.simulation_period_s * config.episode_steps,
            total_training_intervals=config.transitions, realtime=False), run_id="plan19-smoke")
        self.teacher = DeterministicFakeTeacher(test_only=True)
        self.ledger = ComputeLedger(config.runtime.wall_clock_limit_s)
        self.acquired = 0
        self.online_count = 0
        self.replay_exposures = 0
        self.episode = 0
        self.stage = "collect"
        self.history = []
        self.pending = []
        self.ready_queue = []
        self.bootstrap_records = []
        self.distillation = None
        self.initial_hash = None
        self.step = 0
        self.observation = None
        restored = store.load_checkpoint(trusted=True)
        if restored is not None:
            self.load_state_dict(restored)
        elif initial is not None:
            self.learner.load_state_dict(initial["learner"])
            for name, generator in self.learner.generators.items():
                generator.manual_seed(self.seeds[name])
            self.replay.load_state_dict(initial["replay"])
            self.acquired = initial["acquired"]
            self.initial_hash = _parameter_hash(initial["learner"])
            self.distillation = initial["distillation"]
            self.model.distillation_receipt = copy.deepcopy(self.distillation)
            self._reset()
        else:
            self._reset()

    def _reset(self):
        self.observation = self.backend.reset(seed=self.seeds["layouts"] + self.episode,
            exposure_intervals=0 if self.bootstrap else self.online_count)
        self.history = []

    def _frame(self, observation):
        rgb = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
            observation.height, observation.width, 3)
        return self.store.frames.put(rgb)

    def _append_history(self):
        ref = self._frame(self.observation)
        previous_time = self.history[-1]["time"] if self.history else self.observation.capture_time_s
        self.history.append({"ref": ref, "arrival": self.acquired,
            "action": tuple(asdict(self.observation.executed_controls).values()),
            "elapsed": self.observation.capture_time_s - previous_time,
            "time": self.observation.capture_time_s})
        self.history = self.history[-self.config.model.context_length:]
        return ref

    def _history_batch(self, refs, actions, elapsed, start, events):
        images = torch.stack([torch.from_numpy(self.store.frames.get(ref).copy()).permute(2, 0, 1)
                              for ref in refs]).to(self.learner.dtype) / 255
        return {"images": images[None],
            "executed_actions": torch.tensor(actions, dtype=self.learner.dtype)[None],
            "elapsed": torch.tensor(elapsed, dtype=self.learner.dtype)[None, :, None],
            "contexts": [list(events)], "start_time": torch.tensor([start], dtype=self.learner.dtype)}

    def _live_batch(self):
        history = self.history
        return self._history_batch([x["ref"] for x in history], [x["action"] for x in history],
            [x["elapsed"] for x in history], history[0]["time"] - history[0]["elapsed"],
            [{"text": INSTRUCTION, "issued_at": 0.}])

    def _batch(self, records, *, all_bootstrap=False):
        if not records:
            return None
        available = {r.arrival_index for r in (*self.replay.eligible(), *self.replay.pending(),
                                               *self.ready_queue, *records)}
        histories = []
        for record in records:
            keep = [i for i, n in enumerate(record.context_arrivals) if all_bootstrap or n in available]
            # Eligibility is a suffix: never bridge a missing recurrent predecessor.
            first = keep[-1]
            while first > 0 and first - 1 in keep:
                first -= 1
            start = record.context_start_s + sum(record.context_elapsed[:first])
            histories.append(self._history_batch(record.context_refs[first:], record.action_history[first:],
                record.context_elapsed[first:], start, record.instruction_events))
        return LearningBatch(histories, torch.tensor([r.baseline for r in records]),
            torch.tensor([r.return_bin for r in records]),
            torch.tensor([r.action_logits for r in records], dtype=self.learner.dtype),
            torch.tensor([r.arrival_index for r in records]))

    def state_dict(self):
        return {"schema_version": 1, "learner": self.learner.state_dict(),
            "replay": self.replay.state_dict(), "replay_rng": self.replay_rng.getstate(),
            "simulator": self.backend.snapshot(), "observation": self.observation,
            "ledger": self.ledger.state_dict(), "acquired": self.acquired,
            "online_count": self.online_count, "replay_exposures": self.replay_exposures,
            "episode": self.episode, "stage": self.stage, "history": copy.deepcopy(self.history),
            "pending": [r.to_dict() for r in self.pending],
            "ready_queue": [r.to_dict() for r in self.ready_queue],
            "bootstrap_records": [r.to_dict() for r in self.bootstrap_records],
            "distillation": self.distillation, "initial_hash": self.initial_hash, "step": self.step}

    def load_state_dict(self, state):
        if state["schema_version"] != 1:
            raise ValueError("incompatible runner checkpoint")
        self.learner.load_state_dict(state["learner"])
        self.replay.load_state_dict(state["replay"])
        self.replay_rng.setstate(state["replay_rng"])
        self.backend.restore(state["simulator"])
        self.observation = state["observation"]
        self.ledger = ComputeLedger(**state["ledger"])
        for name in ("acquired", "online_count", "replay_exposures", "episode", "stage", "history",
                     "distillation", "initial_hash", "step"):
            setattr(self, name, copy.deepcopy(state[name]))
        for name in ("pending", "ready_queue", "bootstrap_records"):
            setattr(self, name, [TransitionRecord.from_dict(r) for r in state[name]])
        self.model.distillation_receipt = copy.deepcopy(self.distillation)

    def _collect(self):
        current_ref = self._append_history()
        history = self._live_batch()
        start = time.monotonic()
        if self.bootstrap:
            distribution = self.teacher.distribution(history, probability_floor=self.config.learning.teacher_floor)
            z, q = distribution.reference_logits[0], distribution.executed_probabilities[0]
            # Frozen pre-action critic is evaluated even for teacher actions.
            action = self.learner.choose_action(history, explore=False)
            action.update(logits=z, probabilities=q)
        else:
            action = self.learner.choose_action(history)
        inference = time.monotonic() - start
        control = Calibration().map_distribution(action["probabilities"].tolist())
        result = self.backend.step(control, training=not self.bootstrap)
        elapsed = result.reward_components.elapsed_s
        discount = float(pm.time_discounts(torch.tensor([elapsed], dtype=torch.float64),
            self.config.learning.gamma, self.config.learning.nominal_period_s)[0])
        # Fake evaluator is identically zero, explicitly smoke-only. Production
        # must substitute frozen RGB potential differences, not privileged state.
        reward = float(pm.shaped_rewards(torch.tensor([elapsed], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.tensor([result.reward_components.terminal_bonus], dtype=torch.float64),
            torch.tensor([result.terminated]), self.config.learning.gamma,
            self.config.learning.nominal_period_s)[0])
        record = TransitionRecord(transition_id=f"{self.condition}:{self.replica}:{self.acquired}",
            episode_id=f"{self.condition}:{self.replica}:episode-{self.episode}",
            arrival_index=self.acquired, observation_ref=current_ref, next_observation_ref=self._frame(result.observation),
            context_refs=tuple(x["ref"] for x in self.history), instruction=INSTRUCTION,
            action_logits=tuple(action["logits"].tolist()), action_probabilities=tuple(action["probabilities"].tolist()),
            requested_control=asdict(control), executed_control=asdict(result.receipt.executed_controls),
            baseline=action["baseline"], model_version=action["model_version"], elapsed_s=elapsed,
            reward=reward, discount=discount, terminated=result.terminated, truncated=result.truncated,
            end_reason=result.end_reason, receipt=asdict(result.receipt),
            instruction_events=({"text": INSTRUCTION, "issued_at": 0.},),
            action_history=tuple(x["action"] for x in self.history),
            context_arrivals=tuple(x["arrival"] for x in self.history),
            context_elapsed=tuple(x["elapsed"] for x in self.history),
            context_start_s=self.history[0]["time"] - self.history[0]["elapsed"],
            behavior_kind="fake_teacher" if self.bootstrap else "gaussian_exploration")
        self.replay.add_pending(record)
        self.pending.append(record)
        self.observation = result.observation
        self.acquired += 1
        if not self.bootstrap:
            self.online_count += 1
        if len(self.pending) > self.config.episode_steps + 1:
            raise RuntimeError("pending-label queue exceeded its declared finite horizon")
        if result.terminated:
            self.stage = "labels"
        return [record], {"inference_seconds": inference, "success": int(result.end_reason == "success"),
            "collision": int(result.end_reason == "collision"), "end_reason": result.end_reason,
            "drift_alpha": self.backend.alpha}

    def _labels(self):
        rows = self.pending
        returns = pm.discounted_returns(torch.tensor([r.reward for r in rows], dtype=torch.float64),
            torch.tensor([r.discount for r in rows], dtype=torch.float64),
            torch.tensor([r.terminated for r in rows]), torch.tensor([r.truncated for r in rows]))
        bins = pm.bin_returns(returns, self.learner.edges.double().cpu())
        ready = [r.labeled(float(value), int(label)) for r, value, label in zip(rows, returns, bins)]
        for record in ready:
            self.replay.finalize(record)
        self.pending = []
        self.ready_queue = ready
        if self.bootstrap:
            self.bootstrap_records.extend(ready)
            self.ready_queue = []
            self.episode += 1
            if self.acquired >= self.config.bootstrap_transitions:
                self.stage = "distill"
            else:
                self._reset()
                self.stage = "collect"
        else:
            self.stage = "update"
        return ready, {}

    def _update(self):
        record = self.ready_queue.pop(0)
        excluded = {r.transition_id for r in self.ready_queue} | {record.transition_id}
        pool = [r for r in self.replay.eligible() if r.transition_id not in excluded]
        replay = self.replay_rng.sample(pool, min(len(pool), self.config.learning.replay_batch))
        values = self.learner.fit(new=self._batch([record]), replay=self._batch(replay), acquired_count=self.acquired)
        self.replay_exposures += len(replay)
        if not self.ready_queue:
            self.stage = "retire"
        return [], {"data_nll": values["data"], "ewc": values["ewc"], "ridge": values["ridge"],
            "fd": values["fd"], "total_objective": values["total"], "update_seconds": values["update_seconds"]}

    def _retire(self):
        ids = self.replay.retirement_candidates()
        values = {}
        if ids:
            rows = [r for r in self.replay.eligible() if r.transition_id in ids]
            values = self.learner.retire(self._batch(rows, all_bootstrap=self.bootstrap))
            self.replay.retire(ids, transaction_id=f"retire-{self.step}")
        if self.bootstrap:
            self.initial_hash = _parameter_hash(self.learner.state_dict())
            self.stage = "done"
        else:
            self.episode += 1
            if self.online_count >= self.config.transitions:
                self.stage = "done"
            elif self.config.learning.sleep_every and self.online_count % self.config.learning.sleep_every == 0:
                self.stage = "sleep"
            else:
                self.stage = "collect"
                self._reset()
        return [], {"fisher_seconds": values.get("fisher_seconds", 0.)}

    def advance(self, *, fault=None):
        stage = self.stage
        if stage == "done":
            return
        records, values = [], {}
        with self.ledger.charge():
            if stage == "collect":
                records, values = self._collect()
            elif stage == "labels":
                records, values = self._labels()
            elif stage == "distill":
                demonstrations = [{"batch": self._batch([r], all_bootstrap=True).histories[0],
                    "executed_probabilities": torch.tensor([r.action_probabilities]),
                    "domain": "A", "target_kind": "soft"} for r in self.bootstrap_records]
                self.distillation = distill_once(self.model, demonstrations, steps=len(demonstrations))
                self.stage = "joint"
            elif stage == "joint":
                # Bootstrap is shared, A-only, and precedes archive construction.
                self.learner.fit(new=self._batch(self.bootstrap_records, all_bootstrap=True), replay=None,
                                 acquired_count=self.acquired)
                self.stage = "retire"
            elif stage == "update":
                records, values = self._update()
            elif stage == "retire":
                records, values = self._retire()
            elif stage == "sleep":
                rows = self.replay.sample(min(len(self.replay), self.config.learning.sleep_batch), rng=self.replay_rng)
                self.learner.fit(new=None, replay=self._batch(rows), acquired_count=self.acquired,
                                 steps=self.config.learning.sleep_steps)
                self.replay_exposures += len(rows) * self.config.learning.sleep_steps
                self._reset()
                self.stage = "collect"
            else:
                raise RuntimeError(f"unknown execution stage: {stage}")
        self.step += 1
        row = metric_row(phase="phase0_bootstrap" if self.bootstrap else "phase0_smoke",
            condition=self.condition, replica=self.replica, step=self.step, episode=self.episode,
            new_transitions=self.online_count, retired_transitions=self.learner.retired_count,
            replay_exposures=self.replay_exposures, model_version=self.learner.version,
            active_seconds=self.ledger.charged_s, smoke_only=True, **values)
        if fault:
            fault(stage, "before_commit")
        self.store.checkpoint(self.state_dict(), records=records, metrics=[row],
            transaction_id=f"event-{self.step}", progress={"stage": self.stage, "step": self.step,
                "phase": 0, "new_transitions": self.online_count, "target": self.config.transitions,
                "active_seconds": self.ledger.charged_s, "smoke_only": True})
        if fault:
            fault(stage, "after_commit")

    def run(self, stop: StopController, *, max_events: int | None = None, fault=None):
        if not self.store.progress():
            self.store.checkpoint(self.state_dict(), transaction_id="initial", progress={"stage": self.stage})
        count = 0
        while self.stage != "done":
            if stop.check() or max_events is not None and count >= max_events:
                return False
            self.advance(fault=fault)
            count += 1
        self.store.complete({"phase": 0, "smoke_only": True, "condition": self.condition,
            "replica": self.replica, "new_transitions": self.online_count,
            "bootstrap_transitions": self.acquired if self.bootstrap else self.acquired - self.online_count,
            "active_seconds": self.ledger.charged_s, "initial_parameter_sha256": self.initial_hash,
            "final_parameter_sha256": _parameter_hash(self.learner.state_dict()),
            "retired_transitions": self.learner.retired_count, "replay_exposures": self.replay_exposures,
            "context_policy": "mask_at_fifo_eligibility_boundary", "distillation": self.distillation,
            "model_provenance": self.model.encoder.provenance,
            "inventory": self.model.trainable_inventory()})
        return True

    def close(self):
        self.backend.close()


def run_smoke(config: PicarConfig, root: Path, *, resume=False, archive_root=None,
              max_events: int | None = None, fault=None) -> dict:
    config.validate()
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    kwargs = dict(resume=resume, local_quota_bytes=config.runtime.local_quota_bytes,
        archive_quota_bytes=config.runtime.archive_quota_bytes, reserve_bytes=config.runtime.reserve_bytes,
        archive_root=archive_root)
    identity = {"config": config.to_dict(), "source_sha256": source_fingerprint()}
    try:
        with RunStore(root / "bootstrap", {**identity, "condition": "bootstrap", "replica": 0,
            "component_seeds": execution_seeds(config, 0, bootstrap=True),
            "model_initialization_seed": config.seeds(0)["initialization"]}, **kwargs) as common:
            if not common.completed:
                engine = SmokeEngine(config, common, condition="bootstrap", replica=0, bootstrap=True)
                try:
                    if not engine.run(stop, max_events=max_events, fault=fault):
                        return {"status": "paused", "unit": "bootstrap"}
                finally:
                    engine.close()
            initial = common.load_checkpoint(trusted=True)
            for replica in range(config.replicas):
                for condition in config.conditions:
                    unit = f"replica-{replica:03d}/{condition}"
                    with RunStore(root / unit, {**identity, "config": config.for_condition(condition).to_dict(),
                        "condition": condition, "replica": replica,
                        "component_seeds": execution_seeds(config, replica),
                        "model_initialization_seed": config.seeds(0)["initialization"]}, **kwargs) as store:
                        if store.completed:
                            continue
                        if not store.progress():
                            for record in common.records():
                                for ref in (record.observation_ref, record.next_observation_ref, *record.context_refs):
                                    store.frames.put(common.frames.get(ref))
                        engine = SmokeEngine(config.for_condition(condition), store, condition=condition,
                                             replica=replica, initial=initial)
                        try:
                            if not engine.run(stop, max_events=max_events, fault=fault):
                                return {"status": "paused", "unit": unit}
                        finally:
                            engine.close()
        return {"status": "completed", "phase": 0, "smoke_only": True}
    finally:
        stop.close()
