"""Multi-episode Fix 2 timing with real Gaussian actions and completed labels.

This is a bounded diagnostic, not bootstrap or a learning-effect trajectory.
An incomplete timing attempt restarts from paired initialization; its journal
evidence remains. Completed units are never replayed or modified.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import statistics
import threading
import time

import torch

from src.continual import picar_math as pm
from src.robotics.simulator import SimulatorConfig
from src.robotics.worker import SimulatorWorker
from .control import ControlSession
from .dense_roles import GeneratedVisualReward
from .fix2_data import TASK, history_batch, observation_row
from .learner import LearningBatch, MemoryLearner
from .runtime import ModelAccess


def eligible_history(history, available_times):
    """Keep the contiguous eligible suffix; never bridge a retired predecessor."""
    times = history["decision_time"][0].tolist()
    first = len(times) - 1
    while first > 0 and times[first - 1] in available_times:
        first -= 1
    result = {**history}
    for name in ("images", "executed_actions", "elapsed", "decision_time"):
        result[name] = history[name][:, first:]
    result["start_time"] = history["start_time"] + history["elapsed"][:, :first, 0].sum(1)
    return result


def pipeline(model, config, initial, *, condition, rubric, store, stopped, episodes=3, period=1.):
    started = time.monotonic()
    learner = MemoryLearner(model, config.for_condition(condition).learning, config.seeds(0))
    learner.load_state_dict(initial)
    scorer = GeneratedVisualReward(model.encoder, rubric=rubric)
    access = ModelAccess()
    guard = threading.RLock()
    live, histories = [], {}
    clock = {"deadline": None}
    def build_history(observation):
        with guard:
            if clock["deadline"] is None:
                clock["deadline"] = observation.capture_time_s + 20.
            if not live or observation.capture_time_s > live[-1]["time"]:
                live.append(observation_row(observation, previous_time=live[-1]["time"] if live else None))
            live[:] = live[-4:]
            data = history_batch(live, contexts=[{"text": TASK, "issued_at": 0.}],
                deadline=clock["deadline"], width=160, height=120)
            histories[observation.observation_id] = data
            return data
    worker = SimulatorWorker(SimulatorConfig(realtime=True, width=160, height=120,
        control_interval_s=period, task_version="ball-contact-v1"), run_id=f"fix2-pipeline-{condition}")
    session = ControlSession(worker, access, learner=learner, history_builder=build_history,
                             explore=True, pending_capacity=128, own_worker=True)
    eligible, ready, retired, intervals, episode_rows = [], [], [], [], []
    acquisition = update_count = online_updates = sleep_updates = 0
    max_backlog = 0
    timings = {"scoring_seconds": 0., "fit_seconds": 0., "retirement_seconds": 0., "sleep_seconds": 0.}
    fit_times, fisher_times, score_times, actor_times, fit_histories = [], [], [], [], []
    def batch(item):
        available = {float(r["history"]["decision_time"][0, -1]) for r in eligible}
        history = eligible_history(item["history"], available)
        fit_histories.append(history["images"].shape[1])
        return LearningBatch([history], torch.tensor([item["baseline"]]),
            torch.tensor([item["bin"]]), item["logits"][None], torch.tensor([item["arrival"]]))
    def work_one(*, active):
        nonlocal update_count, online_updates, sleep_updates
        item = ready[0]
        replay_item = next((r for r in reversed(eligible) if r.get("learned") and r is not item), None)
        before = time.monotonic()
        session.update(new=batch(item), replay=None if replay_item is None else batch(replay_item),
                       acquired_count=acquisition, cooperative=True)
        timings["fit_seconds"] += time.monotonic() - before
        fit_times.append(time.monotonic() - before)
        update_count += 1
        online_updates += int(active)
        sleep_updates += int(not active)
        ready.pop(0)
        item["learned"] = True
        while len(eligible) > config.learning.replay_capacity and eligible[0].get("learned"):
            old = eligible[0]
            before = time.monotonic()
            with access.update():
                learner.retire(batch(old))
            timings["retirement_seconds"] += time.monotonic() - before
            fisher_times.append(time.monotonic() - before)
            retired.append(old["arrival"])
            eligible.pop(0)
    previous = store.load_checkpoint(trusted=True)
    attempt = 1 + (previous.get("attempt", 0) if previous else 0)
    store.checkpoint({"attempt": attempt, "stage": "starting-paired-timing-attempt"},
        progress={"phase": 1, "status": f"Fix 2 {condition} pipeline starting", "counts": {"step": 0}})
    try:
        for episode in range(episodes):
            with guard:
                live.clear()
                histories.clear()
                clock["deadline"] = None
            initial_observation = session.start(seed=192601 + episode, exposure_intervals=0., training=False)
            with guard:
                clock["deadline"] = initial_observation.capture_time_s + 20.
            pending, previous_post = [], initial_observation.capture_time_s
            while True:
                try:
                    interval = session.next_interval(timeout=60)
                except StopIteration:
                    break
                acquisition += 1
                intervals.append({"episode": episode, "fallback": interval.fallback_reason,
                    "end_reason": interval.result.end_reason, "timing": interval.timing,
                    "elapsed_s": interval.result.reward_components.elapsed_s})
                observation = interval.result.observation
                post = history_batch([observation_row(observation, previous_time=previous_post)],
                    contexts=[{"text": TASK, "issued_at": 0.}], deadline=clock["deadline"],
                    width=320, height=240)
                previous_post = observation.capture_time_s
                item = {"elapsed": interval.result.reward_components.elapsed_s,
                    "bonus": interval.result.reward_components.terminal_bonus,
                    "terminal": interval.result.terminated, "post": post, "action": interval.action,
                    "arrival": acquisition - 1, "history": None}
                if interval.action is not None and interval.policy_observation is not None:
                    item["history"] = histories[interval.policy_observation.observation_id]
                    actor_times.append(interval.action["inference_seconds"])
                pending.append(item)
                max_backlog = max(max_backlog, len(pending) + len(ready))
                if len(pending) + len(ready) > 128:
                    raise RuntimeError("bounded label/update queue exhausted")
                if ready and not interval.result.terminated:
                    work_one(active=True)
                stopped()
                if interval.result.terminated:
                    break
            session.pause(reason="episode_boundary")
            # Wait for the final inference lease before resetting/checkpointing.
            session.snapshot()
            if not pending or not pending[-1]["terminal"]:
                raise RuntimeError("timing episode has incomplete returns")
            scoring_start = time.monotonic()
            scores = []
            for item in pending:
                score_started = time.monotonic()
                with access.update():
                    scores.append(float(scorer.scores(item["post"])[0]))
                score_times.append(time.monotonic() - score_started)
                stopped()
            timings["scoring_seconds"] += time.monotonic() - scoring_start
            elapsed = torch.tensor([p["elapsed"] for p in pending], dtype=torch.float64)
            terminal = torch.tensor([p["terminal"] for p in pending])
            bonus = torch.tensor([p["bonus"] for p in pending], dtype=torch.float64)
            rewards = pm.dense_visual_rewards(elapsed, torch.tensor(scores, dtype=torch.float64), bonus, terminal)
            discounts = pm.time_discounts(elapsed, config.learning.gamma, config.learning.nominal_period_s)
            returns = pm.discounted_returns(rewards, discounts, terminal)
            labels = pm.bin_returns(returns, learner.edges.double().cpu())
            for i, item in enumerate(pending):
                if item["action"] is not None and item["history"] is not None:
                    record = {"history": item["history"], "baseline": item["action"]["baseline"],
                        "logits": item["action"]["logits"].clone(), "arrival": item["arrival"],
                        "bin": int(labels[i]), "return": float(returns[i]), "learned": False}
                    eligible.append(record)
                    ready.append(record)
            episode_rows.append({"episode": episode, "intervals": len(pending),
                "end_reason": intervals[-1]["end_reason"], "visual_scores": scores,
                "stage_rewards": rewards.tolist(), "terminal_bonuses": bonus.tolist(),
                "returns": returns.tolist(), "pending_updates": len(ready)})
            max_backlog = max(max_backlog, len(ready))
            store.checkpoint({"attempt": attempt, "stage": "episode-boundary", "learner": learner.state_dict(),
                "episodes": episode_rows, "intervals": intervals, "completed_episode_count": episode + 1},
                progress={"phase": 1, "status": f"Fix 2 {condition} episode labeled",
                          "counts": {"step": episode + 1, "target": episodes}})
        sleep_start = time.monotonic()
        while ready:
            work_one(active=False)
            stopped()
        timings["sleep_seconds"] = time.monotonic() - sleep_start
        final_clock = worker.diagnostics()
    finally:
        session.close()
    eligible_intervals = [x for x in intervals if x["fallback"] != "startup"]
    misses = sum(x["fallback"] is not None for x in eligible_intervals)
    return {"condition": condition, "episodes": episode_rows, "intervals": intervals,
        "fallback_count": misses, "eligible_intervals": len(eligible_intervals),
        "fallback_fraction": misses / max(1, len(eligible_intervals)), "updates": update_count,
        "online_updates": online_updates, "sleep_updates": sleep_updates, "retired_records": len(retired),
        "maximum_label_backlog": max_backlog, "pending_after_drain": len(ready),
        "pipeline_seconds": time.monotonic() - started, "timings": timings,
        "model_access": access.diagnostics(), "clock": final_clock, "attempt": attempt,
        "task_version": "ball-contact-v1", "reward_version": "dense-visual-v1",
        "return_bins": model.return_bins, "retired_archive_replay": False,
        "operation_medians": {"inference_seconds": statistics.median(actor_times) if actor_times else None,
            "fit_seconds": statistics.median(fit_times) if fit_times else None,
            "fisher_seconds": statistics.median(fisher_times) if fisher_times else None,
            "evaluator_seconds": statistics.median(score_times) if score_times else None},
        "training_history_lengths": {"minimum": min(fit_histories) if fit_histories else 0,
            "maximum": max(fit_histories) if fit_histories else 0,
            "mean": statistics.mean(fit_histories) if fit_histories else 0},
        "limitations": ["Untrained paired initialization; this measures pipeline cost, not task skill.",
            "Full-return labels arrive at episode end; scoring and final backlog drain are charged stopped work.",
            "Incomplete timing attempts restart from paired initialization; prior journal traces are retained.",
            "This cycle with stopped labeling/sleep is not proof of continuous one-update-per-observation service."]}
