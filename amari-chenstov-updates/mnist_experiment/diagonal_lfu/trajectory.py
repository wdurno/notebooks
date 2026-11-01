"""Checkpointed initialization and accepted-arrival Plan 16 trajectories."""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit
from .curvature import build_model, observation, observe_update, logits
from .learner import evaluate, fit_batch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    torch.set_rng_state(value["cpu"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def initialize(store, budget, cfg, unit_id, assets, replica, cohort, *, pulse=lambda **kw: None):
    metadata = {"kind": "initialization", "replica": replica, "cohort": cohort,
                "seeds": assets["seeds"], "parameter_count": 487}
    path = store.begin(unit_id, metadata)
    if path is None:
        return store.load_checkpoint(unit_id)
    dtype = getattr(torch, cfg.dtype)
    seed = assets["seeds"]["initialization_model"]
    model, layout = build_model(seed, device=cfg.device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)
    generator = torch.Generator().manual_seed(assets["seeds"]["initialization_order"])
    state = store.load_checkpoint(unit_id)
    if state is None:
        state = {"epoch": 0, "offset": 0, "order": None, "best_loss": float("inf"),
                 "best_model": None, "history": [], "wall_seconds": 0., "phase": "fit"}
    else:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        restore_rng(state["rng"])
    x = assets["init_x"].to(device=cfg.device, dtype=dtype)
    y = assets["init_y"].to(cfg.device)
    vx = assets["validation_x"].to(device=cfg.device, dtype=dtype)
    vy = assets["validation_y"].to(cfg.device)
    base, invocation = state["wall_seconds"], time.perf_counter()

    def checkpoint(finalizing=False):
        state.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     generator=generator.get_state(), rng=rng_state(),
                     wall_seconds=base + time.perf_counter() - invocation)
        store.save_checkpoint(unit_id, state, finalizing=finalizing)

    try:
        while state["phase"] == "fit" and state["epoch"] < cfg.initial_epochs:
            if state["order"] is None:
                state["order"] = torch.randperm(len(y), generator=generator)
            for offset in range(state["offset"], len(y), cfg.initial_batch_size):
                budget.check()
                batch = state["order"][offset:offset + cfg.initial_batch_size].to(cfg.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
                if not torch.isfinite(loss):
                    raise FloatingPointError("initialization loss is nonfinite")
                state["offset"] = offset + len(batch)
                if (offset // cfg.initial_batch_size) % 8 == 0:
                    checkpoint()
                pulse(active_unit=unit_id, phase="initialization", step=state["epoch"])
            with torch.no_grad():
                validation = float(F.cross_entropy(model(vx), vy))
            state["history"].append({"epoch": state["epoch"] + 1, "validation_nll": validation})
            if validation < state["best_loss"]:
                state["best_loss"] = validation
                state["best_model"] = cpu_tree(model.state_dict())
            state["epoch"] += 1
            state["offset"], state["order"] = 0, None
            checkpoint()
        if state["phase"] == "fit":
            model.load_state_dict(state["best_model"])
            state["theta"] = layout.flatten_module(model, detach=True).cpu()
            state["fisher_sum"] = torch.zeros(487, dtype=torch.float64)
            state["reference_offset"], state["phase"] = 0, "fisher"
            checkpoint()
        reference = assets["reference_x"]
        theta = state["theta"].to(cfg.device)
        for offset in range(state["reference_offset"], len(reference), cfg.chunk_size):
            budget.check()
            chunk = reference[offset:offset + cfg.chunk_size].to(cfg.device)
            measured = observation(model, layout, theta, chunk, chunk_size=cfg.chunk_size, check=budget.check)
            state["fisher_sum"] += measured.cpu() * len(chunk)
            state["reference_offset"] = offset + len(chunk)
            checkpoint()
            pulse(active_unit=unit_id, phase="initial Fisher", step=offset)
        state["diagonal"] = state["fisher_sum"] / len(reference)
        state["phase"] = "completed"
        state["parameter_layout"] = layout.metadata()
        state["initial_observed_nines"] = int((assets["init_y"] == 9).sum())
        checkpoint()
        store.write_json(unit_id, "summary.json", {"replica": replica, "cohort": cohort,
                         "validation_nll": state["best_loss"], "wall_seconds": state["wall_seconds"],
                         "initial_observed_nines": state["initial_observed_nines"],
                         "fisher_trace": float(state["diagonal"].sum()), "parameter_count": 487})
        store.complete(unit_id)
        return state
    except (StopRequested, ResourceLimit):
        if not (store.unit_path(unit_id) / "COMPLETED").exists():
            checkpoint(finalizing=True)
        raise


def run_trajectory(store, budget, cfg, unit_id, metadata, initial, assets, stream,
                   train_x, train_y, *, pulse=lambda **kw: None):
    path = store.begin(unit_id, metadata)
    if path is None:
        return store.load_checkpoint(unit_id)
    model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
    state = store.load_checkpoint(unit_id)
    if state is None:
        theta = initial["theta"].to(cfg.device)
        state = {"theta": theta.cpu(), "previous": None, "diagonal": initial["diagonal"],
                 "step": 0, "metrics": [], "step_metrics": [], "wall_seconds": 0.,
                 "q": 1 / cfg.initialization_size, "rng": rng_state()}
    else:
        restore_rng(state["rng"])
    theta = state["theta"].to(cfg.device)
    previous = None if state["previous"] is None else state["previous"].to(cfg.device)
    diagonal = state["diagonal"].to(cfg.device)
    c, ridge = metadata["c"], metadata["ridge"]
    condition = metadata["condition"]
    ex = assets["eval_x"].to(device=cfg.device, dtype=theta.dtype)
    ey = assets["eval_y"].to(cfg.device)
    weights = assets["non9_weights"]
    indices = stream["ids"]
    if indices.ndim == 1:
        indices = indices.reshape(-1, cfg.samples_per_step)
    count = len(indices)
    base, invocation = state["wall_seconds"], time.perf_counter()

    def metric(step, extra=None):
        values = evaluate(model, layout, theta, ex, ey, weights, check=budget.check)
        row = {"step": step, "arrivals": step * cfg.samples_per_step,
               "p": float(stream["p"][max(0, step - 1)]),
               "segment": str(stream["segments"][max(0, step - 1)]),
               "expected_nines": float(stream["expected_cumulative_nines"][step - 1]) if step else 0.,
               "observed_nines": int((train_y[indices[:step].reshape(-1)] == 9).sum()),
               "initial_nines": initial["initial_observed_nines"],
               "wall_seconds": base + time.perf_counter() - invocation,
               **values, **(extra or {})}
        if state["metrics"]:
            row["frozen_nine_recall"] = state["metrics"][0]["nine_recall"]
            row["frozen_old_accuracy"] = state["metrics"][0]["old_accuracy"]
        else:
            row["frozen_nine_recall"], row["frozen_old_accuracy"] = row["nine_recall"], row["old_accuracy"]
        state["metrics"].append(row)

    def checkpoint(finalizing=False):
        state.update(theta=theta.detach().cpu(), previous=None if previous is None else previous.detach().cpu(),
                     diagonal=diagonal.detach().cpu(), rng=rng_state(),
                     wall_seconds=base + time.perf_counter() - invocation)
        store.save_checkpoint(unit_id, state, finalizing=finalizing)

    if not state["metrics"]:
        metric(0)
        checkpoint()
    valid_transition = True
    before = None
    try:
        for step in range(state["step"], count):
            before = (previous, theta, diagonal, state["step"], state["q"],
                      len(state["metrics"]), len(state["step_metrics"]))
            budget.check()
            ids = indices[step]
            x = train_x[ids].to(device=cfg.device, dtype=theta.dtype)
            y = train_y[ids].to(cfg.device)
            tick = time.perf_counter()
            if step == 0:
                diag_next = diagonal
                health = {"negative_fraction": 0., "negative_mass": 0., "raw_min": float(diagonal.min()),
                          "fisher_trace": float(diagonal.sum()), "correction_norm": 0., "fresh_norm": 0.}
            else:
                diag_next, health = observe_update(model, layout, previous, theta, diagonal, x,
                                                   condition=condition, c=c, chunk_size=cfg.chunk_size,
                                                   check=budget.check)
            curvature_seconds = time.perf_counter() - tick
            candidate, solve = fit_batch(model, layout, theta, diag_next, x, y, c=c,
                                         ridge_ratio=ridge, max_iter=cfg.max_iter,
                                         max_eval=cfg.max_eval, history_size=cfg.history_size,
                                         learning_rate=cfg.learning_rate, tolerance_grad=cfg.tolerance_grad,
                                         tolerance_change=cfg.tolerance_change, check=budget.check)
            # All fields below are committed together; interruption before here redoes one arrival.
            valid_transition = False
            previous, theta, diagonal = theta, candidate, diag_next
            state["step"] = step + 1
            state["q"] = (1 - c)**2 * state["q"] + c**2 / cfg.samples_per_step
            detail = {"step": step + 1, "curvature_seconds": curvature_seconds, **health, **solve}
            state["step_metrics"].append(detail)
            boundary = step + 1 == count or str(stream["segments"][step + 1]) != str(stream["segments"][step])
            if (step + 1) % cfg.eval_every == 0 or boundary:
                metric(step + 1, detail)
            valid_transition = True
            checkpoint()
            pulse(active_unit=unit_id, phase=metadata["stage"], step=step + 1, total_steps=count,
                  accepted_metrics=state["metrics"])
        summary = {"wall_seconds": state["wall_seconds"], "steps": count,
                   "mean_closures": sum(v["closure_count"] for v in state["step_metrics"]) / count,
                   "mean_negative_fraction": sum(v["negative_fraction"] for v in state["step_metrics"]) / count,
                   "final_nine_recall": state["metrics"][-1]["nine_recall"],
                   "final_old_accuracy": state["metrics"][-1]["old_accuracy"],
                   "initial_nines": initial["initial_observed_nines"],
                   "online_nines": state["metrics"][-1]["observed_nines"]}
        store.write_json(unit_id, "metrics.json", state["metrics"])
        store.write_csv(unit_id, "metrics.csv", state["metrics"], fieldnames=list(state["metrics"][-1]))
        store.write_json(unit_id, "summary.json", summary)
        store.complete(unit_id, metadata=summary)
        return state
    except (StopRequested, ResourceLimit):
        if (store.unit_path(unit_id) / "COMPLETED").exists():
            raise
        if not valid_transition:
            previous, theta, diagonal, state["step"], state["q"], nm, ns = before
            del state["metrics"][nm:]
            del state["step_metrics"][ns:]
        checkpoint(finalizing=True)
        raise
