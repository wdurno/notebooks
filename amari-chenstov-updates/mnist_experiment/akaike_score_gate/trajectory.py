"""Paired causal trajectories with fixed episode anchors and accepted-state resume."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call

from .artifacts import UnitStore, file_hash
from .budget import WorkBudget
from .config import canonical_hash, canonical_policy
from .evidence import weights, effective_sample_size, retention_precision, consolidate
from .fisher import model_fisher, represent
from .gate import evaluate_gate
from .model import build_model, active_indices
from .optimizer import fit_restricted

REQUIRED = ("summary.json", "metrics.json", "gates.json", "snapshots.pt", "final_state.pt",
            "pr_curves.json", "pr_curves.pt")


def cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def scalar_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return float(value) if value.numel() == 1 else None
    if isinstance(value, dict):
        return {k: scalar_tree(v) for k, v in value.items()
                if not isinstance(v, Tensor) or v.numel() == 1}
    if isinstance(value, (tuple, list)):
        return [scalar_tree(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weighted_gradient(model, layout, inputs, targets, a, indices=None) -> tuple[Tensor, float]:
    """Mean loss gradient g=-s, differentiating only the requested suffix."""
    if indices is None:
        indices = torch.arange(layout.total_numel, device=inputs.device)
    params, variables, names = {}, [], []
    for spec in layout.specs:
        param = dict(model.named_parameters())[spec.name].detach()
        selected = bool(((indices >= spec.start) & (indices < spec.stop)).any())
        if selected:
            param = param.clone().requires_grad_(True)
            variables.append(param)
            names.append(spec)
        params[spec.name] = param
    logits = functional_call(model, (params, dict(model.named_buffers())), (inputs,))
    loss = a.to(inputs.device).double() @ torch.nn.functional.cross_entropy(
        logits, targets, reduction="none").double()
    result = torch.zeros(layout.total_numel, device=inputs.device, dtype=torch.float64)
    if variables:
        gradients = torch.autograd.grad(loss, variables)
        for spec, gradient in zip(names, gradients):
            result[spec.start:spec.stop] = gradient.reshape(-1).double()
    return result, float(loss.detach())


def operation_key(profile: dict, operation: str, width: int, n: int,
                  *, depth: int = 1, ewc: str = "lr", gate: str = "lr",
                  family: str | None = None) -> str:
    buckets = profile["buckets"]
    if isinstance(buckets, dict):
        buckets = buckets[family] if family is not None else sorted({v for rows in buckets.values() for v in rows})
    candidates = [int(b) for b in buckets if int(b) >= n]
    if not candidates:
        raise ValueError(f"buffer size {n} exceeds frozen cost profile")
    bucket = min(candidates)
    base = f"{operation}:w{width}"
    if operation in {"fit", "probe", "gradient", "preview"}:
        base += f":d{depth}"
    base += f":n{bucket}"
    if operation in {"fit", "probe", "archive"}:
        base += f":e{ewc}"
    if operation == "probe":
        base += f":g{gate}"
    if base not in profile["charges"]:
        raise ValueError(f"missing frozen operation charge {base}")
    return base


def _auc(rows: list[dict], field: str, stop: int | None = None) -> float:
    values = rows if stop is None else rows[:stop + 1]
    xs = torch.tensor([r["observations"] for r in values], dtype=torch.float64)
    ys = torch.tensor([r[field] for r in values], dtype=torch.float64)
    if len(values) < 2 or xs[-1] <= xs[0]:
        return float(ys[-1])
    return float(torch.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def _snapshot(state, model, layout, inputs, targets, a, positions, batch_sizes, step):
    return cpu_tree({"step": step, "model": model.state_dict(),
                     "theta": layout.flatten_module(model, detach=True),
                     "anchor": state["anchor"], "archive": state["archive"],
                     "precision": state["precision"], "inputs": inputs,
                     "targets": targets, "weights": a,
                     "batch_positions": positions, "batch_sizes": batch_sizes,
                     "position": positions[-1], "history": state["history"]})


def ensure_trajectory(store: UnitStore, phase: str, index: int, environment: dict,
                      policy: dict, asset_path: Path, *, width: int, retention: dict,
                      profile: dict | None = None, budget_tier: str = "unlimited",
                      snapshot_steps: tuple[int, ...] = (), episode_arrivals: int | None = None,
                      resume: bool = True) -> Path:
    from .evaluation import evaluate

    study = store.study
    policy = canonical_policy(policy)
    episodes = study.episode_arrivals if episode_arrivals is None else episode_arrivals
    unit = store.unit(phase, "trajectory", index, environment=environment["name"],
                      policy=policy["name"], detail={"policy": policy, "width": width,
                      "retention": retention, "asset_hash": file_hash(asset_path / "config.json"),
                      "profile_hash": None if profile is None else canonical_hash(profile),
                      "budget_tier": budget_tier, "snapshot_steps": list(snapshot_steps),
                      "episode_arrivals": episodes, "metric_origin": "post_common_warmup",
                      "auc_rule": "trapezoidal_postarrival_including_zero_exposure"})
    session = store.begin(unit, REQUIRED, resume=resume)
    if session is None:
        return store.completed(unit, REQUIRED)
    device = torch.device(study.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, study.dtype)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    model, layout = build_model(study.seed(f"{phase}:{environment['family']}:model", index),
                                width=width, device=device, dtype=dtype)
    model.load_state_dict(assets["initial_model"])
    layout.assert_metadata(assets["parameter_layout"])
    stream = assets["streams"][environment["name"]]
    positions = [float(p) for p in stream["positions"]]
    total_steps, m = len(positions), int(environment["samples"])
    archive_samples = policy.get("archive_samples", study.score_samples)
    eval_x, eval_y = assets["evaluation_inputs"], assets["evaluation_targets"]
    checkpoint = session.path / "checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        budget = WorkBudget.from_state_dict(state["budget"])
        if profile is not None and dict(budget.charges) != profile["charges"]:
            raise RuntimeError("resume would change the frozen operation costs")
    else:
        archive = represent(assets["initial_fisher"].to(device), kind=policy["ewc"], rank=policy["rank"])
        precision, kappa = retention_precision(archive, retention["omega"], retention["ridge_ratio"])
        allowance = None if profile is None or budget_tier == "unlimited" else (
            profile["b0"][environment["family"]] * (2 if budget_tier == "principal" else 1))
        charges = {"unlimited": 1e-12} if profile is None else profile["charges"]
        budget = WorkBudget(allowance, charges)
        initial = evaluate(model, eval_x, eval_y, family=environment["family"], position=0., device=device)
        baseline = {"step": -1, "position": 0., "observations": 0,
                    "expected_nines": 0., "realized_nines": 0, "online_seconds": 0.,
                    "charged_seconds": 0., "selected_depth": 0, "active_parameters": 0,
                    "n_eff": 0., "d_eff": None, "gate_status": "baseline", **initial}
        state = {"step": 0, "stage": "arrival", "episode_start": 0,
                 "depth": 1, "previous_depth": 1, "anchor": layout.flatten_module(model, detach=True),
                 "archive": archive, "precision": precision, "kappa": kappa, "ledger": {},
                 "history": [], "metrics": [baseline], "gates": [], "snapshots": [],
                 "online_seconds": 0., "fit_seconds": 0., "probe_seconds": 0.,
                 "archive_seconds": 0., "evaluation_seconds": 0., "prequential_nll_sum": 0.,
                 "fit_calls": 0, "unresolved_fits": 0, "gate_denials": 0, "d_eff": None,
                 "last_fit": {}, "pr_curves": [], "started_unix": time.time()}

    def save():
        if device.type == "cuda":
            state["peak_cuda_bytes"] = max(state.get("peak_cuda_bytes", 0),
                                            torch.cuda.max_memory_allocated(device))
        state["model"] = cpu_tree(model.state_dict())
        state["budget"] = budget.state_dict()
        session.write_torch("checkpoint.pt", cpu_tree(state))

    def key(operation, n, depth=1):
        if profile is None:
            return "unlimited"
        return operation_key(profile, operation, width, n, depth=depth,
                             ewc=policy["ewc"], gate=policy["gate"] or "lr",
                             family=environment["family"])

    def timed(operation, callback):
        synchronize(device)
        start = time.perf_counter()
        result = callback()
        synchronize(device)
        elapsed = time.perf_counter() - start
        state["online_seconds"] += elapsed
        state[operation + "_seconds"] = state.get(operation + "_seconds", 0.) + elapsed
        return result

    while state["step"] < total_steps:
        store.limit.check()
        # A completed consolidation has already cleared the old episode.
        # Advance before rebuilding a buffer when resuming at that boundary.
        if state["stage"] == "advance":
            state["previous_depth"] = state["depth"]
            state["step"] += 1
            state["stage"] = "arrival"
            save()
            continue
        step = state["step"]
        start = state["episode_start"]
        for name in ("archive", "precision", "anchor"):
            state[name] = state[name].to(device)
        x = stream["inputs"][start:step + 1].flatten(0, 1).to(device=device, dtype=dtype)
        y = stream["targets"][start:step + 1].flatten().to(device)
        a = weights([m] * (step - start + 1), policy["half_life"], device=device)
        n_eff = effective_sample_size(a)
        if state["stage"] == "arrival":
            state["last_fit"] = {}
            budget.grant_arrival()
            # Prequential evaluation is made before this arrival changes parameters.
            pq_key = key("prequential", m)
            pq_admitted = budget.spend(pq_key)
            if not pq_admitted:
                state["gate_denials"] += 1
            def prequential():
                with torch.no_grad():
                    return float(torch.nn.functional.cross_entropy(model(x[-m:]), y[-m:]))
            if pq_admitted:
                state["prequential_nll"] = timed("prequential", prequential)
            else:
                # Research evaluation of the unchanged prediction is still
                # available, but a policy unable to predict cannot adapt here.
                evaluation_start = time.perf_counter()
                state["prequential_nll"] = prequential()
                synchronize(device)
                state["evaluation_seconds"] += time.perf_counter() - evaluation_start
            end = min(start + episodes, total_steps)
            archive_key = key("archive", (end - start) * m)
            reserve_ok = (policy["depth"] == 0 or state.get("archive_reserved", False)
                          or budget.reserve("archive", archive_key,
                                            units=archive_samples / study.score_samples))
            state["archive_reserved"] = reserve_ok
            state["prequential_nll_sum"] += m * state["prequential_nll"]
            state["probe_scheduled"] = step == start or step % policy["cadence"] == 0
            state["depth"] = (policy["depth"] if policy["depth"] is not None else
                              (1 if state["probe_scheduled"] else state["previous_depth"]))
            state["gate_status"] = "fixed_depth" if policy["gate"] is None else "not_scheduled"
            state["d_eff"] = None
            if step in snapshot_steps:
                state["snapshots"].append(_snapshot(state, model, layout, x, y, a,
                    positions[start:step + 1], [m] * (step - start + 1), step))
            state["stage"] = "fit" if state["depth"] and reserve_ok and pq_admitted else "evaluate"
            if not pq_admitted:
                state["gate_status"] = "budget_denied_prediction"
            if not reserve_ok:
                state["gate_status"] = "budget_denied_archive_reservation"
            save()
        if state["stage"] == "fit":
            indices, _ = active_indices(layout, state["depth"])
            fit = timed("fit", lambda: fit_restricted(
                model, layout, x, y, a, state["anchor"], state["precision"], indices.to(device),
                max_iter=retention.get("max_iter", study.max_iter),
                max_eval=retention.get("max_eval", study.max_eval),
                tolerance_grad=study.tolerance_grad, tolerance_change=study.tolerance_change,
                residual_tolerance=retention.get("residual_tolerance", study.tolerance_grad),
                budget=budget, charge_key=key("fit", len(y), state["depth"])))
            state["last_fit"] = scalar_tree(fit)
            state["fit_calls"] += 1
            state["unresolved_fits"] += int(not fit["converged"])
            state["gate_status"] = fit["status"]
            state["stage"] = ("gate" if fit["converged"] and policy["gate"] is not None
                              and state["probe_scheduled"] and state["depth"] < 6 else "evaluate")
            save()
            store.limit.check()
        if state["stage"] == "gate":
            depth = state["depth"]
            probe_units = (profile or {}).get("probe_units", policy["score_samples"] / study.score_samples)
            if not budget.spend(key("probe", len(y), depth), units=probe_units):
                state["gate_status"] = "budget_denied_probe"
                state["gate_denials"] += 1
                state["stage"] = "evaluate"
            else:
                ai, _ = active_indices(layout, depth)
                ci, _ = active_indices(layout, depth + 1)
                bi = ci[~torch.isin(ci, ai)]
                seed = study.seed(f"{phase}:{environment['family']}:gate:{step}:{depth}", index)
                def probe():
                    g, _ = weighted_gradient(model, layout, x, y, a, ci.to(device))
                    h, diagnostics = model_fisher(model, layout, x, a,
                        sample_size=policy["score_samples"], seed=seed, chunk_size=study.fisher_chunk_size)
                    h = represent(h, kind=policy["gate"], rank=policy.get("gate_rank", policy["rank"]))
                    theta = layout.flatten_module(model, detach=True).double()
                    r = g + state["precision"] @ (theta - state["anchor"].double())
                    result = evaluate_gate(g, r, h, state["precision"], h / n_eff,
                        ai, bi, c=policy["c"], floor_relative=study.solve_floor_relative,
                        restricted_tolerance=retention.get("residual_tolerance", study.tolerance_grad))
                    result["score_diagnostics"] = scalar_tree(diagnostics)
                    return result
                gate = timed("probe", probe)
                state["gates"].append({"step": step, "depth": depth, "seed": seed,
                                       "n_eff": n_eff, **scalar_tree(gate)})
                state["d_eff"] = gate["d_eff_expanded"] if gate["open"] else gate["d_eff_active"]
                state["gate_status"] = gate["status"]
                if gate["open"]:
                    state["depth"] += 1
                    state["stage"] = "fit"
                else:
                    state["stage"] = "evaluate"
            save()
            store.limit.check()
            if state["stage"] == "fit":
                continue
        if state["stage"] == "evaluate":
            synchronize(device)
            begin = time.perf_counter()
            metrics = evaluate(model, eval_x, eval_y, family=environment["family"],
                               position=positions[step], device=device,
                               return_pr_curve=environment["family"] == "mixture" and step == total_steps - 1)
            curve = metrics.pop("nine_pr_curve", None)
            if curve is not None:
                state["pr_curves"].append(cpu_tree({"step": step, "observations": (step+1)*m, **curve}))
            synchronize(device)
            state["evaluation_seconds"] += time.perf_counter() - begin
            count = (step + 1) * m
            actual_nines = int((stream["targets"][:step + 1] == 9).sum())
            expected_nines = (m * sum(positions[:step + 1]) if environment["family"] == "mixture"
                              else count * float((stream["targets"] == 9).double().mean()))
            indices = active_indices(layout, state["depth"])[0] if state["depth"] else []
            state["metrics"].append({"step": step, "position": positions[step],
                "observations": count, "expected_nines": expected_nines,
                "realized_nines": actual_nines, "selected_depth": state["depth"],
                "active_parameters": len(indices), "n_eff": n_eff, "d_eff": state["d_eff"],
                "d_eff_scope": "last_probe_preview_selected_coordinates_not_postfit_curvature",
                "gate_status": state["gate_status"], "prequential_nll": state["prequential_nll"],
                "online_seconds": state["online_seconds"], "charged_seconds": budget.spent,
                "budget_remaining": None if budget.unlimited else budget.remaining,
                "fit_residual": state["last_fit"].get("restricted_residual_max"),
                "archive_trace": float(state["archive"].trace()), "kappa": state["kappa"],
                **metrics})
            state["stage"] = "consolidate" if step + 1 == min(start + episodes, total_steps) else "advance"
            save()
        if state["stage"] == "consolidate":
            if policy["depth"] != 0 and state["archive_reserved"]:
                budget.spend_reserved("archive")
                seed = study.seed(f"{phase}:{environment['family']}:archive:{start}", index)
                def update_archive():
                    fresh, diagnostic = model_fisher(model, layout, x, a,
                        sample_size=archive_samples, seed=seed, chunk_size=study.fisher_chunk_size)
                    old = cpu_tree(state["archive"])
                    update = consolidate(state["archive"], layout.flatten_module(model, detach=True), fresh,
                        omega=retention["omega"], arrivals=step - start + 1,
                        episode_id=str(start), observation_ids=list(range(start*m, (step+1)*m)),
                        ledger=state["ledger"], kind=policy["ewc"], rank=policy["rank"],
                        ridge_ratio=retention["ridge_ratio"])
                    if snapshot_steps:
                        state["history"].append(cpu_tree({"model": model.state_dict(),
                            "inputs": x, "targets": y, "weights": a, "eta": update["eta"],
                            "dense_new_fisher": fresh, "archive_before": old,
                            "batch_positions": positions[start:step+1], "batch_sizes": [m]*(step-start+1),
                            "seed": seed, "diagnostics": scalar_tree(diagnostic)}))
                    for name in ("archive", "anchor", "precision", "kappa", "ledger"):
                        state[name] = update[name]
                timed("archive", update_archive)
            elif policy["depth"] != 0:
                # No fit was admitted without this reservation. Retain the old
                # archive and mark the unprocessed episode, never update for free.
                state["metrics"][-1]["gate_status"] = "budget_denied_consolidation"
                state["gate_denials"] += 1
            state["archive_reserved"] = False
            state["episode_start"] = step + 1
            state["stage"] = "advance"
            state["metrics"][-1]["online_seconds"] = state["online_seconds"]
            state["metrics"][-1]["charged_seconds"] = budget.spent
            save()
        if state["stage"] == "advance":
            state["previous_depth"] = state["depth"]
            state["step"] += 1
            state["stage"] = "arrival"
            save()

    rows = state["metrics"]
    summary = {"phase": phase, "kind": "trajectory", "environment": environment["name"],
        "family": environment["family"], "replica_index": index, "policy": policy["name"],
        "width": width, "budget_tier": budget_tier, "development": phase != "phase5",
        "total_observations": total_steps*m, "expected_nines": rows[-1]["expected_nines"],
        "realized_nines": rows[-1]["realized_nines"], "warmup_observations": 8 if environment["family"] == "mixture" else 0,
        "nll_auc": _auc(rows, "current_nll"), "retention_nll": _auc(rows, "retention_nll"),
        "initial_retention_nll": rows[0]["retention_nll"],
        "retention_change": _auc(rows, "retention_nll") - rows[0]["retention_nll"],
        "accuracy_auc": _auc(rows, "current_accuracy"), "final_nll": rows[-1]["current_nll"],
        "prequential_nll": state["prequential_nll_sum"] / (total_steps*m),
        "mean_active_parameters": sum(r["active_parameters"] for r in rows[1:]) / total_steps,
        "mean_depth": sum(r["selected_depth"] for r in rows[1:]) / total_steps,
        "online_seconds": state["online_seconds"], "charged_seconds": budget.spent,
        "fit_seconds": state["fit_seconds"], "probe_seconds": state["probe_seconds"],
        "archive_seconds": state["archive_seconds"], "evaluation_seconds": state["evaluation_seconds"],
        "fit_calls": state["fit_calls"], "unresolved_fits": state["unresolved_fits"],
        "gate_denials": state["gate_denials"], "gate_events": len(state["gates"]),
        "gate_openings": sum(g["open"] for g in state["gates"]),
        "peak_cuda_bytes": max(state.get("peak_cuda_bytes", 0), torch.cuda.max_memory_allocated(device))
                           if device.type == "cuda" else 0,
        "peak_cuda_scope": "trajectory process allocations, including research evaluation",
        "exposure_origin": "post_common_warmup", "status": "completed"}
    for fraction in (.25, .5):
        stop = next(i for i, row in enumerate(rows) if row["observations"] >= total_steps*m*fraction)
        summary[f"early_{int(100*fraction)}_nll_auc"] = _auc(rows, "current_nll", stop)
        summary[f"early_{int(100*fraction)}_observations"] = rows[stop]["observations"]
    session.write_json("summary.json", scalar_tree(summary))
    session.write_json("metrics.json", scalar_tree(rows))
    session.write_json("gates.json", state["gates"])
    pr_rows = []
    for curve in state["pr_curves"]:
        count = len(curve["precision"])
        selected = torch.linspace(0, count-1, min(count, 201)).round().long().unique()
        for j in selected.tolist():
            pr_rows.append({"step": curve["step"], "observations": curve["observations"],
                "threshold": scalar_tree(float(curve["thresholds"][j])),
                "precision": scalar_tree(float(curve["precision"][j])),
                "recall": float(curve["recall"][j]),
                "precision_defined": bool(curve["precision_defined"][j]),
                "false_positive_rate": float(curve["false_positive_rate"][j])})
    session.write_json("pr_curves.json", pr_rows)
    session.write_torch("pr_curves.pt", state["pr_curves"])
    session.write_torch("snapshots.pt", state["snapshots"])
    session.write_torch("final_state.pt", cpu_tree({"model": model.state_dict(),
        "archive": state["archive"], "anchor": state["anchor"], "budget": budget.state_dict(),
        "ledger": state["ledger"], "history": state["history"]}))
    return store.finish(session, REQUIRED)
