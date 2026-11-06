"""Checkpointed pretraining and causal accepted-arrival regression paths.

The batch callback returns tensors for an immutable named panel and positional
indices. It performs no sampling; all source IDs and transforms were sealed.
"""

from __future__ import annotations

import math
import resource
import time

import torch

from mnist_experiment.diagonal_lfu.trajectory import cpu_tree, rng_state, restore_rng
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.kalman_pi.observations import combine_moments
from mnist_experiment.kalman_pi.policies import concentration
from .policies import Controller, PolicySpec
from mnist_experiment.kalman_pi.trajectory import device_tree

from .learner import evaluate, fit_batch, mean_nll
from .model import build_model, observation, observed_scores


def initialize(store, budget, cfg, uid, metadata, batch, *, pulse=lambda **kw: None):
    if store.begin(uid, metadata) is None:
        return store.load_checkpoint(uid)
    model, layout = build_model(metadata["model_seed"], device=cfg.device,
                                dtype=getattr(torch, cfg.dtype))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)
    generator = torch.Generator().manual_seed(metadata["order_seed"])
    state = store.load_checkpoint(uid)
    if state is None:
        state = {"epoch": 0, "offset": 0, "order": None, "history": [],
            "best_loss": None, "best_model": None, "phase": "fit", "wall_seconds": 0.,
            "reference_offset": 0, "epochs_target": cfg.initial_epochs}
        with torch.no_grad():
            state["constant_mean"] = batch("initialization", None)["targets"].mean(0).cpu()
    else:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        restore_rng(state["rng"])
    started, base = time.perf_counter(), state["wall_seconds"]

    def checkpoint(finalizing=False):
        state.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     generator=generator.get_state(), rng=rng_state(),
                     wall_seconds=base + time.perf_counter() - started)
        store.save_checkpoint(uid, state, finalizing=finalizing)

    try:
        while state["phase"] == "fit" and state["epoch"] < state["epochs_target"]:
            if state["order"] is None:
                state["order"] = torch.randperm(cfg.initialization_size, generator=generator)
            for offset in range(state["offset"], cfg.initialization_size, cfg.initial_batch_size):
                budget.check()
                indices = state["order"][offset:offset + cfg.initial_batch_size]
                values = batch("initialization", indices)
                optimizer.zero_grad(set_to_none=True)
                loss = mean_nll(model(values["inputs"]), values["targets"])
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite registration pretraining")
                optimizer.step()
                state["offset"] = offset + len(indices)
                checkpoint()
                pulse(active_unit=uid, phase="7A/7B: pretraining", step=state["epoch"] + 1,
                      batch_offset=state["offset"], total_steps=state["epochs_target"])
            squared, zero, constant_squared, count = 0., 0., 0., 0
            with torch.no_grad():
                for offset in range(0, cfg.validation_size, 256):
                    budget.check()
                    values = batch("validation", torch.arange(offset, min(cfg.validation_size, offset + 256)))
                    output = model(values["inputs"])
                    squared += float((output.double() - values["targets"].double()).square().sum())
                    zero += float(values["targets"].double().square().sum())
                    constant_squared += float((values["targets"].double() - state["constant_mean"].to(cfg.device)).square().sum())
                    count += len(output)
            validation_risk = squared / (3 * count)
            state["history"].append({"epoch": state["epoch"] + 1, "validation_risk": validation_risk,
                                     "zero_risk": zero / (3 * count),
                                     "constant_mean_risk": constant_squared / (3 * count)})
            if state["best_loss"] is None or validation_risk < state["best_loss"]:
                state["best_loss"], state["best_model"] = validation_risk, cpu_tree(model.state_dict())
            state["epoch"] += 1
            state["offset"], state["order"] = 0, None
            if (state["epoch"] == cfg.initial_epochs and state["best_loss"] >= .95 * zero / (3 * count)
                    and cfg.initial_max_epochs > cfg.initial_epochs):
                state["epochs_target"] = cfg.initial_max_epochs
            checkpoint()
        if state["phase"] == "fit":
            model.load_state_dict(state["best_model"])
            state.update(theta=layout.flatten_module(model, detach=True).cpu(),
                         fisher_sum=torch.zeros(layout.total_numel, dtype=torch.float64), phase="fisher")
            checkpoint()
        theta = state["theta"].to(cfg.device)
        for offset in range(state["reference_offset"], cfg.reference_size, cfg.chunk_size):
            budget.check()
            indices = torch.arange(offset, min(cfg.reference_size, offset + cfg.chunk_size))
            values = batch("reference", indices)
            diagonal = observation(model, layout, theta, values["inputs"],
                                   chunk_size=cfg.chunk_size, check=budget.check)
            state["fisher_sum"] += diagonal.cpu() * len(indices)
            state["reference_offset"] = offset + len(indices)
            checkpoint()
            pulse(active_unit=uid, phase="initial regression Fisher", step=state["reference_offset"],
                  total_steps=cfg.reference_size)
        state.update(diagonal=state["fisher_sum"] / cfg.reference_size,
                     parameter_layout=layout.metadata(), phase="completed")
        checkpoint()
        store.write_json(uid, "summary.json", {"history": state["history"], "best_validation_risk": state["best_loss"],
            "epochs": state["epoch"], "wall_seconds": state["wall_seconds"], "parameter_count": 448,
            "initialization_labels": cfg.initialization_size, "reference_inputs": cfg.reference_size})
        store.complete(uid)
        return state
    except (ResourceLimit, StopRequested):
        # An optimizer batch is atomic for this checkpointed initialization.
        checkpoint(finalizing=True)
        raise


def calibrate(store, budget, cfg, uid, metadata, initial, batch, *, pulse=lambda **kw: None):
    if store.begin(uid, metadata) is None:
        return store.load_checkpoint(uid)
    model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
    theta = initial["theta"].to(cfg.device)
    state = store.load_checkpoint(uid)
    if state is None:
        state = {"count": 0, "mean": torch.zeros(448, dtype=torch.float64),
                 "m2": torch.zeros(448, dtype=torch.float64), "wall_seconds": 0.}
    started, base = time.perf_counter(), state["wall_seconds"]
    for offset in range(state["count"], cfg.calibration_size, cfg.chunk_size):
        budget.check()
        indices = torch.arange(offset, min(cfg.calibration_size, offset + cfg.chunk_size))
        values = batch("calibration", indices)
        scores = observed_scores(model, layout, theta, values["inputs"], values["targets"],
                                 chunk_size=cfg.chunk_size, check=budget.check).cpu()
        state["count"], state["mean"], state["m2"] = combine_moments(
            state["count"], state["mean"], state["m2"], scores)
        state["wall_seconds"] = base + time.perf_counter() - started
        store.save_checkpoint(uid, state)
        pulse(active_unit=uid, phase="regression observer calibration", step=state["count"],
              total_steps=cfg.calibration_size)
    state["variance"] = state["m2"] / (state["count"] - 1)
    store.save_checkpoint(uid, state)
    store.write_json(uid, "summary.json", {"count": state["count"], "additional_labels": 0,
        "variance_trace": float(state["variance"].sum()), "wall_seconds": state["wall_seconds"]})
    store.complete(uid)
    return state


def run_trajectory(store, budget, cfg, uid, metadata, initial, batch, stream, class_weights,
                   *, calibration=None, shadow_specs=(), frozen_reference=None, pulse=lambda **kw: None):
    if store.begin(uid, metadata) is None:
        return store.load_checkpoint(uid)
    model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
    spec = PolicySpec.parse(metadata["condition"])
    saved = store.load_checkpoint(uid)
    state = saved or {"theta": initial["theta"], "diagonal": initial["diagonal"],
        "step": 0, "q": 1 / cfg.initialization_size, "log_replacement_mass": 0.,
        "metrics": [], "step_metrics": [], "shadow_metrics": {s.condition: [] for s in shadow_specs},
        "controller": None, "shadows": {}, "rng": rng_state(), "wall_seconds": 0., "audit_states": [],
        "costs": {k: 0. for k in ("solver_seconds", "curvature_seconds", "controller_seconds",
                                   "evaluation_seconds", "checkpoint_seconds", "render_seconds", "shadow_seconds")}}
    if saved:
        restore_rng(state["rng"])
    theta, diagonal = state["theta"].to(cfg.device), state["diagonal"].to(cfg.device)

    def make_controller(policy, previous):
        controller = Controller(policy, cfg, cfg.fixed_pi, device_tree(previous, cfg.device) if previous else None)
        if policy.movement and not previous:
            if calibration is None:
                raise ValueError("movement policies need already-seen-label noise calibration")
            controller.initialize(initial["theta"].to(cfg.device), initial["diagonal"].to(cfg.device),
                                  calibration["variance"].to(cfg.device))
        return controller

    controller = make_controller(spec, state["controller"])
    shadows = {s.condition: make_controller(s, state["shadows"].get(s.condition)) for s in shadow_specs}
    count = len(stream["segments"])
    started, base = time.perf_counter(), state["wall_seconds"]
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def metric(step, detail=None):
        began = time.perf_counter()
        row = {"step": step, "arrivals": step * cfg.samples_per_step,
            "segment": stream["segments"][max(0, step - 1)],
            "q": state["q"], "effective_size": 1 / state["q"],
            "replacement_mass": math.exp(state["log_replacement_mass"]),
            "wall_seconds": base + time.perf_counter() - started}
        roles = [("old", "old"), ("shifted", "shifted"),
                 ("current", f"current:{metadata['schedule']}:{step}")]
        if metadata["schedule"] == "stationary":
            roles += [("shifted25", "shifted25"), ("shifted45", "shifted45")]
        predictions = {}
        for label, role in roles:
            panel_key = batch.plan(role)["hash"] if hasattr(batch, "plan") else role
            if panel_key not in predictions:
                values = batch(role, None)
                predictions[panel_key] = evaluate(model, layout, theta, values["inputs"], values["targets"],
                                    values["digits"], class_weights, check=budget.check)
            measures = predictions[panel_key]
            row.update({f"{label}_{key}": value for key, value in measures.items()})
        if frozen_reference is not None:
            frozen = frozen_reference[str(step)]
            row.update({f"frozen_current_{key}": value for key, value in frozen.items()})
        state["costs"]["evaluation_seconds"] += time.perf_counter() - began
        state["metrics"].append({**row, **(detail or {})})

    def checkpoint():
        began = time.perf_counter()
        state.update(theta=theta.detach().cpu(), diagonal=diagonal.detach().cpu(),
            controller=cpu_tree(controller.state), shadows={k: cpu_tree(v.state) for k, v in shadows.items()},
            rng=rng_state(), wall_seconds=base + time.perf_counter() - started)
        store.save_checkpoint(uid, state)
        state["costs"]["checkpoint_seconds"] += time.perf_counter() - began

    if not state["metrics"]:
        metric(0)
        checkpoint()
    for step in range(state["step"], count):
        budget.check()
        began = time.perf_counter()
        action = controller.choose(state["q"], diagonal, cfg.samples_per_step, step, theta)
        state["costs"]["controller_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        shadow_actions = {name: obs.choose(state["q"], diagonal, cfg.samples_per_step, step, theta)
                          for name, obs in shadows.items()}
        state["costs"]["shadow_seconds"] += time.perf_counter() - began
        # Only the frozen actions precede this access to current inputs/targets.
        began = time.perf_counter()
        values = batch(metadata["schedule"], torch.arange(step * cfg.samples_per_step,
            (step + 1) * cfg.samples_per_step))
        state["costs"]["render_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        fresh = observation(model, layout, theta, values["inputs"],
                            chunk_size=cfg.chunk_size, check=budget.check)
        next_diagonal = (1 - action["applied_pi"]) * diagonal + action["applied_pi"] * fresh
        state["costs"]["curvature_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        diagnostics, shadow_diagnostics = {}, {}
        observers = ([controller] if spec.movement else []) + list(shadows.values())
        if observers:
            scores = observed_scores(model, layout, observers[0].state["anchor"], values["inputs"],
                                     values["targets"], chunk_size=cfg.chunk_size, check=budget.check)
            state["costs"]["controller_seconds" if spec.movement else "shadow_seconds"] += time.perf_counter() - began
            began = time.perf_counter()
            if spec.movement:
                diagnostics = controller.assimilate(scores, step)
            state["costs"]["controller_seconds"] += time.perf_counter() - began
            began = time.perf_counter()
            shadow_diagnostics = {name: observer.assimilate(scores, step) for name, observer in shadows.items()}
            state["costs"]["shadow_seconds"] += time.perf_counter() - began
        candidate, solve = fit_batch(model, layout, theta, next_diagonal, values["inputs"], values["targets"],
            pi=action["applied_pi"], ridge=metadata["ridge"], max_iter=cfg.max_iter, max_eval=cfg.max_eval,
            learning_rate=cfg.learning_rate, history_size=cfg.history_size, tolerance_grad=cfg.tolerance_grad,
            tolerance_change=cfg.tolerance_change, check=budget.check)
        audit_steps = (0, metadata.get("hold_steps", cfg.hold_steps) + metadata.get("ramp_steps", cfg.ramp_steps) // 2,
                       metadata.get("hold_steps", cfg.hold_steps) + metadata.get("ramp_steps", cfg.ramp_steps) - 1)
        if metadata.get("solver_audit") and step in audit_steps:
            state["audit_states"].append(cpu_tree({"step": step, "theta": theta,
                "diagonal": next_diagonal, "inputs": values["inputs"], "targets": values["targets"],
                "pi": action["applied_pi"], "ridge": metadata["ridge"], "solve": solve,
                "accepted": candidate, "observer_anchor": initial["theta"],
                "anchor_diagonal": initial["diagonal"]}))
        theta, diagonal = candidate, next_diagonal
        state["costs"]["solver_seconds"] += solve["solver_seconds"]
        detail = {**action, **diagnostics, **solve}
        state["step_metrics"].append({"step": step + 1, "segment": stream["segments"][step], **detail})
        for name in shadows:
            state["shadow_metrics"][name].append({"step": step + 1,
                "segment": stream["segments"][step], **shadow_actions[name], **shadow_diagnostics[name]})
        state["step"] = step + 1
        state["q"] = concentration(state["q"], action["applied_pi"], cfg.samples_per_step)
        state["log_replacement_mass"] += math.log1p(-action["applied_pi"])
        boundary = step + 1 == count or stream["segments"][step + 1] != stream["segments"][step]
        if (step + 1) % cfg.eval_every == 0 or boundary:
            metric(step + 1, detail)
        checkpoint()
        pulse(active_unit=uid, step=step + 1, total_steps=count, accepted_metrics=state["metrics"],
              accepted_shadow_metrics=state["shadow_metrics"])
    online_wall = base + time.perf_counter() - started
    summary = {"online_wall_seconds": online_wall, "wall_seconds": online_wall,
        "steps": count, "learning_seconds": sum(state["costs"][k] for k in
             ("solver_seconds", "curvature_seconds", "controller_seconds", "render_seconds")),
        "costs": dict(state["costs"]), "final_old_risk": state["metrics"][-1]["old_risk"],
        "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cpu_rss_scope": "process lifetime high-water mark",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cfg.device == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cfg.device == "cuda" else 0,
        "final_shifted_risk": state["metrics"][-1]["shifted_risk"],
        "final_current_risk": state["metrics"][-1]["current_risk"],
        "boundary_fraction": sum(r["boundary_low"] or r["boundary_high"] for r in state["step_metrics"]) / count,
        "fallback_fraction": sum(bool(r["fallback"]) for r in state["step_metrics"]) / count}
    store.write_json(uid, "metrics.json", state["metrics"])
    store.write_csv(uid, "metrics.csv", state["metrics"])
    store.write_json(uid, "steps.json", state["step_metrics"])
    store.write_json(uid, "shadow_metrics.json", state["shadow_metrics"])
    store.write_json(uid, "summary.json", summary)
    store.complete(uid)
    return state

