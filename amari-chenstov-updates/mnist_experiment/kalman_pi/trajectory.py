"""Accepted-arrival checkpoints for causal adaptive-action trajectories."""

from __future__ import annotations

import math
import time

import torch

from mnist_experiment.diagonal_lfu.curvature import build_model, observation
from mnist_experiment.diagonal_lfu.learner import evaluate, fit_batch
from mnist_experiment.diagonal_lfu.trajectory import cpu_tree, rng_state, restore_rng, initialize
from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit
from .observations import observed_scores, combine_moments
from .policies import Controller, PolicySpec, concentration


def calibrate(store, budget, cfg, unit_id, initial, assets, *, pulse=lambda **kw: None):
    """Reuse initial training labels; no new observations enter the nominal q."""
    count = len(assets["calibration_y"])
    metadata = {"kind": "calibration", "observations": count,
                "role": "already_seen_initialization_training_labels"}
    path = store.begin(unit_id, metadata)
    if path is None:
        return store.load_checkpoint(unit_id)
    model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
    theta = initial["theta"].to(cfg.device)
    state = store.load_checkpoint(unit_id)
    if state is None:
        state = {"count": 0, "mean": torch.zeros_like(theta, dtype=torch.float64, device="cpu"),
                 "m2": torch.zeros_like(theta, dtype=torch.float64, device="cpu"), "wall_seconds": 0.}
    base, started = state["wall_seconds"], time.perf_counter()
    try:
        for offset in range(state["count"], count, cfg.chunk_size):
            budget.check()
            x = assets["calibration_x"][offset:min(count, offset + cfg.chunk_size)].to(cfg.device)
            y = assets["calibration_y"][offset:min(count, offset + cfg.chunk_size)].to(cfg.device)
            values = observed_scores(model, layout, theta, x, y, chunk_size=cfg.chunk_size, check=budget.check).cpu()
            state["count"], state["mean"], state["m2"] = combine_moments(
                state["count"], state["mean"], state["m2"], values)
            state["wall_seconds"] = base + time.perf_counter() - started
            store.save_checkpoint(unit_id, state)
            pulse(active_unit=unit_id, phase="observer calibration", step=state["count"], total_steps=count)
        if count < 2:
            raise ValueError("observer calibration requires at least two initial examples")
        state["variance"] = state["m2"] / (count - 1)
        store.save_checkpoint(unit_id, state)
        store.write_json(unit_id, "summary.json", {**metadata, "wall_seconds": state["wall_seconds"],
                         "mean_score_norm": float(state["mean"].norm()),
                         "score_variance_trace": float(state["variance"].sum())})
        store.complete(unit_id)
        return state
    except (StopRequested, ResourceLimit):
        if not (store.unit_path(unit_id) / "COMPLETED").exists():
            state["wall_seconds"] = base + time.perf_counter() - started
            store.save_checkpoint(unit_id, state, finalizing=True)
        raise


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    return value


def run_trajectory(store, budget, cfg, unit_id, metadata, initial, assets, stream,
                   train_x, train_y, *, calibration=None, shadow_specs=(), pulse=lambda **kw: None):
    path = store.begin(unit_id, metadata)
    if path is None:
        return store.load_checkpoint(unit_id)
    spec = PolicySpec.parse(metadata["condition"])
    model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
    state = store.load_checkpoint(unit_id)
    if state is None:
        state = {"theta": initial["theta"], "previous": None, "diagonal": initial["diagonal"],
                 "step": 0, "q": 1 / cfg.initialization_size, "log_replacement_mass": 0.,
                 "metrics": [], "step_metrics": [], "shadow_metrics": {s.condition: [] for s in shadow_specs},
                 "wall_seconds": 0., "rng": rng_state(), "controller": None, "shadows": {},
                 "costs": {k: 0. for k in ("solver_seconds", "curvature_seconds", "controller_seconds",
                                           "evaluation_seconds", "checkpoint_seconds")}}
    else:
        restore_rng(state["rng"])
    theta = state["theta"].to(cfg.device)
    diagonal = state["diagonal"].to(cfg.device)
    previous = None if state["previous"] is None else state["previous"].to(cfg.device)
    c, ridge = metadata["c"], metadata["ridge"]

    def make_controller(policy, saved):
        value = Controller(policy, cfg, c, device_tree(saved, cfg.device) if saved else None)
        if not saved and policy.movement:
            if calibration is None:
                raise ValueError("movement policy requires an initialization-score calibration")
            value.initialize(theta, diagonal, calibration["variance"].to(cfg.device))
        return value

    controller = make_controller(spec, state["controller"])
    shadows = {s.condition: make_controller(s, state["shadows"].get(s.condition)) for s in shadow_specs}
    observers = ([controller] if spec.movement else []) + list(shadows.values())
    eval_x = assets["eval_x"].to(device=cfg.device, dtype=theta.dtype)
    eval_y = assets["eval_y"].to(cfg.device)
    indices = stream["ids"].reshape(-1, cfg.samples_per_step)
    count = len(indices)
    base, started = state["wall_seconds"], time.perf_counter()
    refresh = metadata.get("anchor_refresh", cfg.anchor_refresh)

    def metric(step, detail=None):
        tick = time.perf_counter()
        values = evaluate(model, layout, theta, eval_x, eval_y, assets["non9_weights"], check=budget.check)
        state["costs"]["evaluation_seconds"] += time.perf_counter() - tick
        frozen = state["metrics"][0] if state["metrics"] else values
        row = {"step": step, "arrivals": step * cfg.samples_per_step,
               "p": float(stream["p"][max(0, step - 1)]),
               "segment": str(stream["segments"][max(0, step - 1)]),
               "observed_nines": int((train_y[indices[:step].reshape(-1)] == 9).sum()),
               "expected_nines": float(stream["expected_cumulative_nines"][step - 1]) if step else 0.,
               "unique_ids": int(indices[:step].unique().numel()),
               "initial_nines": initial["initial_observed_nines"],
               "q": state["q"], "effective_size": 1 / state["q"],
               "log_replacement_mass": state["log_replacement_mass"],
               "replacement_mass": math.exp(state["log_replacement_mass"]),
               "wall_seconds": base + time.perf_counter() - started,
               "frozen_nine_recall": frozen["nine_recall"], "frozen_old_accuracy": frozen["old_accuracy"],
               **values, **(detail or {})}
        state["metrics"].append(row)

    def checkpoint(finalizing=False):
        tick = time.perf_counter()
        state.update(theta=theta.detach().cpu(), diagonal=diagonal.detach().cpu(),
                     previous=None if previous is None else previous.detach().cpu(),
                     controller=cpu_tree(controller.state),
                     shadows={name: cpu_tree(obs.state) for name, obs in shadows.items()},
                     rng=rng_state(), wall_seconds=base + time.perf_counter() - started)
        store.save_checkpoint(unit_id, state, finalizing=finalizing)
        state["costs"]["checkpoint_seconds"] += time.perf_counter() - tick

    accepted = True
    before = None
    try:
        if not state["metrics"]:
            metric(0)
            checkpoint()
        for step in range(state["step"], count):
            budget.check()
            before = (previous, theta, diagonal, state["q"], state["log_replacement_mass"],
                      cpu_tree(controller.state), {k: cpu_tree(v.state) for k, v in shadows.items()},
                      len(state["metrics"]), len(state["step_metrics"]),
                      {k: len(v) for k, v in state["shadow_metrics"].items()}, dict(state["costs"]))
            accepted = False
            # No current batch is read until all actual/shadow actions are frozen.
            tick = time.perf_counter()
            action = controller.choose(state["q"], diagonal, cfg.samples_per_step, step)
            shadow_actions = {name: obs.choose(state["q"], diagonal, cfg.samples_per_step, step)
                              for name, obs in shadows.items()}
            state["costs"]["controller_seconds"] += time.perf_counter() - tick
            pi = action["applied_pi"]
            ids = indices[step]
            x, y = train_x[ids].to(cfg.device), train_y[ids].to(cfg.device)
            tick = time.perf_counter()
            fresh = observation(model, layout, theta, x, chunk_size=cfg.chunk_size, check=budget.check) if step else diagonal
            diagonal_next = (1 - pi) * diagonal + pi * fresh if step else diagonal
            curvature_seconds = time.perf_counter() - tick
            state["costs"]["curvature_seconds"] += curvature_seconds
            tick = time.perf_counter()
            scores = None
            observer_diagnostics, shadow_diagnostics = {}, {}
            if observers:
                scores = observed_scores(model, layout, observers[0].state["anchor"], x, y,
                                         chunk_size=cfg.chunk_size, check=budget.check)
                if spec.movement:
                    observer_diagnostics = controller.assimilate(scores, step)
                shadow_diagnostics = {name: obs.assimilate(scores, step) for name, obs in shadows.items()}
            state["costs"]["controller_seconds"] += time.perf_counter() - tick
            candidate, solve = fit_batch(model, layout, theta, diagonal_next, x.to(theta.dtype), y,
                                         c=pi, ridge_ratio=ridge, max_iter=cfg.max_iter, max_eval=cfg.max_eval,
                                         history_size=cfg.history_size, learning_rate=cfg.learning_rate,
                                         tolerance_grad=cfg.tolerance_grad, tolerance_change=cfg.tolerance_change,
                                         check=budget.check)
            state["costs"]["solver_seconds"] += solve["solver_seconds"]
            previous, theta, diagonal = theta, candidate, diagonal_next
            tick = time.perf_counter()
            if observers and refresh and (step + 1) % refresh == 0:
                anchor_diagonal = observation(model, layout, theta, x, chunk_size=cfg.chunk_size, check=budget.check)
                new_scores = observed_scores(model, layout, theta, x, y, chunk_size=cfg.chunk_size, check=budget.check)
                if spec.movement:
                    observer_diagnostics.update(controller.reanchor(theta, anchor_diagonal, scores, new_scores, diagonal))
                for name, obs in shadows.items():
                    shadow_diagnostics[name].update(obs.reanchor(theta, anchor_diagonal, scores, new_scores, diagonal))
            state["costs"]["controller_seconds"] += time.perf_counter() - tick
            detail = {"step": step + 1, "curvature_seconds": curvature_seconds, **solve, **action, **observer_diagnostics}
            state["step_metrics"].append(detail)
            for name in shadows:
                state["shadow_metrics"][name].append({"step": step + 1, "arrivals": (step + 1) * cfg.samples_per_step,
                    "segment": str(stream["segments"][step]), **shadow_actions[name], **shadow_diagnostics[name]})
            state["step"], state["q"] = step + 1, concentration(state["q"], pi, cfg.samples_per_step)
            state["log_replacement_mass"] += math.log1p(-pi)
            boundary = step + 1 == count or stream["segments"][step + 1] != stream["segments"][step]
            if (step + 1) % cfg.eval_every == 0 or boundary:
                metric(step + 1, detail)
            checkpoint()
            accepted = True
            pulse(active_unit=unit_id, phase=metadata["stage"], step=step + 1, total_steps=count,
                  accepted_metrics=state["metrics"], accepted_shadow_metrics=state["shadow_metrics"])
        wall = base + time.perf_counter() - started
        calibration_seconds = calibration["wall_seconds"] if spec.movement else 0.
        summary = {"wall_seconds": wall + calibration_seconds, "online_wall_seconds": wall,
                   "steps": count, "initial_nines": initial["initial_observed_nines"],
                   "online_nines": state["metrics"][-1]["observed_nines"],
                   "unique_online_ids": int(indices.unique().numel()),
                   "final_nine_recall": state["metrics"][-1]["nine_recall"],
                   "final_old_accuracy": state["metrics"][-1]["old_accuracy"],
                   "mean_closures": sum(v["closure_count"] for v in state["step_metrics"]) / count,
                   "boundary_fraction": sum(v["boundary_low"] or v["boundary_high"] for v in state["step_metrics"]) / count,
                   "fallback_fraction": sum(bool(v["fallback"]) for v in state["step_metrics"]) / count,
                   "costs": {**state["costs"], "calibration_seconds": calibration_seconds,
                             "wall_seconds": wall + calibration_seconds}}
        store.write_json(unit_id, "metrics.json", state["metrics"])
        fields = list(dict.fromkeys(key for row in state["metrics"] for key in row))
        store.write_csv(unit_id, "metrics.csv", state["metrics"], fieldnames=fields)
        store.write_json(unit_id, "summary.json", summary)
        if shadows:
            store.write_json(unit_id, "shadow_metrics.json", state["shadow_metrics"])
        store.complete(unit_id, metadata=summary)
        return state
    except (StopRequested, ResourceLimit):
        if (store.unit_path(unit_id) / "COMPLETED").exists():
            raise
        if not accepted and before is not None:
            previous, theta, diagonal, state["q"], state["log_replacement_mass"], control_state, shadow_states, nm, ns, nsh, costs = before
            controller.state = device_tree(control_state, cfg.device)
            for name, obs in shadows.items():
                obs.state = device_tree(shadow_states[name], cfg.device)
                del state["shadow_metrics"][name][nsh[name]:]
            state["step"] = ns
            state["costs"] = costs
            del state["metrics"][nm:]
            del state["step_metrics"][ns:]
        checkpoint(finalizing=True)
        raise
