"""Paired scheduling diagnostics; no claim of trained student task skill."""

from __future__ import annotations

from dataclasses import asdict, replace
import threading
import time

import numpy as np
import torch

from .control import ControlSession
from .learner import LearningBatch, MemoryLearner
from .phase1 import _histories, resolved
from .phase1_roles import _batch
from .runtime import BoundedWorkQueue, ModelAccess
from src.robotics.worker import SimulatorWorker
from src.robotics.simulator import SimulatorConfig


def control_pilot(model, recipe, initial, store, stopped, *, period=1.25, cooperative=False):
    learner = MemoryLearner(model, resolved(recipe).learning, resolved(recipe).seeds(0))
    learner.load_state_dict(initial)
    history = _histories(recipe)
    # A cold logical event, not repeated-image cache throughput.
    model.encoder.clear_caches()
    learner.choose_action(history, explore=False)
    model.encoder.clear_caches()
    rows, guard = [], threading.RLock()
    def history_builder(observation):
        row = {"rgb": np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
            observation.height, observation.width, 3).copy(), "time": observation.capture_time_s,
            "elapsed": period, "action": tuple(asdict(observation.executed_controls).values())}
        with guard:
            if not rows or row["time"] > rows[-1]["time"]:
                rows.append(row)
            visible = [r for r in rows if r["time"] <= observation.capture_time_s]
            rows[:] = rows[-recipe.context_length * 3:]
        return _batch(visible, recipe)
    worker = SimulatorWorker(SimulatorConfig(width=recipe.image_width, height=recipe.image_height,
        control_interval_s=period, realtime=True), run_id="fix1-control")
    access = ModelAccess()
    session = ControlSession(worker, access, learner=learner, history_builder=history_builder,
                             explore=False, pending_capacity=128, own_worker=True)
    intervals, updates = [], []
    deferred = BoundedWorkQueue(8)
    max_deferred, background_seconds = 0, 0.
    try:
        session.start(seed=191107, training=False)
        while True:
            try:
                interval = session.next_interval(timeout=30)
            except StopIteration:
                break
            intervals.append({"index": interval.index, "fallback": interval.fallback_reason,
                "timing": interval.timing, "end_reason": interval.result.end_reason,
                "physical_seconds": interval.result.observation.physical_time_s})
            if len(intervals) in (4, 10) and not interval.result.terminated:
                replay = LearningBatch([history], torch.zeros(1), torch.tensor([10]),
                                       torch.full((1, 7), .1), torch.tensor([0]))
                fresh = LearningBatch([{**history, "elapsed": history["elapsed"] + .125}],
                    torch.zeros(1), torch.tensor([10]), torch.full((1, 7), .1), torch.tensor([128]))
                t0 = time.monotonic()
                values = session.update(new=fresh, replay=replay, acquired_count=256, cooperative=cooperative)
                torch.cuda.synchronize()
                updates.append({"wall_seconds": time.monotonic() - t0, "losses": values})
                # Ready background work is bounded and subordinate to control.
                # Its synthetic labels are cost probes, never study evidence.
                if not deferred.offer(replay):
                    raise RuntimeError("deferred Fisher diagnostic queue exhausted")
                max_deferred = max(max_deferred, len(deferred))
            if interval.result.terminated:
                break
            stopped()
        clock = worker.diagnostics()
        # Drain only after active motion has ended. This demonstrates bounded
        # accounting, not proof that every future label can meet an online SLA.
        while len(deferred):
            job = deferred.take()
            t0 = time.monotonic()
            with access.update():
                learner.retire(job)
            torch.cuda.synchronize()
            background_seconds += time.monotonic() - t0
    finally:
        session.close()
    eligible = [r for r in intervals if r["fallback"] != "startup"]
    misses = sum(r["fallback"] is not None for r in eligible)
    fraction = misses / len(eligible) if eligible else 1.
    store.checkpoint({"done": True}, progress={"phase": 1, "status": "control pilot complete",
                                              "counts": {"step": len(intervals)}})
    return {"period_s": period, "context_length": recipe.context_length, "cooperative": cooperative,
        "fallback_count": misses, "eligible_intervals": len(eligible), "fallback_fraction": fraction,
        "deadline_pass": fraction <= .1, "intervals": intervals, "updates": updates,
        "model_access": access.diagnostics(), "clock": clock,
        "background": {"capacity": 8, "maximum_backlog": max_deferred,
                       "remaining": len(deferred), "drain_seconds": background_seconds},
        "limitations": ["Two diagnostic optimizer events; synthetic targets measure cost, not learning.",
            "Deferred Fisher jobs drain at the episode boundary, not under a proven sustained-arrival load.",
            "A shorter context/control period is not selected without matching controllability and competence checks."]}
