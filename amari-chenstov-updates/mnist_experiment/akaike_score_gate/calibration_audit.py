"""Phase 3 variance calibration and Phase 3/4 one/two-layer shadow audits.

Repeated histories contain real observed labels from the independent reference
pool. Model-generated labels are used only for Fisher integration. Gradient
noise is evaluated at one common restricted state, while every repeated fit
starts at the saved pre-fit state with the same anchor and precision.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.derivatives import per_sample_derivatives
from src.mnist_model import mnist_nll
from src.seeding import derive_component_seed

from . import fisher_audit as fa
from .evidence import effective_sample_size
from .fisher import represent
from .gate import _response, evaluate_gate
from .model import active_indices


CALIBRATION_SCHEMA_VERSION = 1
REQUIRED = fa.REQUIRED


def observed_covariance(gradients: Tensor, weights: Tensor, batch_sizes) -> dict:
    """Estimate V = sum_i a_i^2 Cov(g_i), centering *within* each arrival.

    Each arrival is an iid stratum at its own scheduled environment. A positive
    weight singleton cannot estimate its covariance; it is reported missing,
    never assigned zero or pooled with a different environment's mean.
    """
    gradients = torch.as_tensor(gradients).detach().double()
    weights = torch.as_tensor(weights, device=gradients.device).detach().double()
    effective_sample_size(weights)
    weights = weights / weights.sum()
    sizes = tuple(batch_sizes)
    if (gradients.ndim != 2 or gradients.shape[0] != len(weights)
            or not torch.isfinite(gradients).all() or not sizes
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes)
            or sum(sizes) != len(weights)):
        raise ValueError("per-sample gradients, weights and arrival sizes must align")
    total = gradients.new_zeros((gradients.shape[1],) * 2)
    means, unavailable, start = [], [], 0
    for batch, size in enumerate(sizes):
        values, a = gradients[start:start + size], weights[start:start + size]
        center = values.mean(0)
        means.append(center)
        if size == 1 and bool(a.square().sum() > 0):
            unavailable.append(batch)
        elif size > 1:
            centered = values - center
            total += a.square().sum() * (centered.mT @ centered) / (size - 1)
        start += size
    return {"covariance": None if unavailable else (total + total.mT) / 2,
            "mean_gradient": weights @ gradients, "arrival_mean_gradients": torch.stack(means),
            "missing_covariance_arrivals": unavailable,
            "status": "singleton_strata_unresolved" if unavailable else "available",
            "centering": "within_arrival_not_across_drifting_means",
            "sum_squared_weights": float(weights.square().sum()),
            "ess": effective_sample_size(weights), "observed_labels": True}


def replicated_covariance(values: Tensor) -> dict:
    """Unbiased covariance across complete independent-history realizations."""
    values = torch.as_tensor(values).detach().double()
    if values.ndim != 2 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("at least two finite vector-valued replications are required")
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.mT @ centered / (len(values) - 1)
    # The trace jackknife avoids R additional dense covariance constructions.
    jackknife_se = None
    if len(values) > 2:
        squared = centered.square().sum(1)
        leave_one_out = (squared.sum() - len(values) / (len(values) - 1) * squared) / (len(values) - 2)
        jackknife_se = float(((len(values) - 1) / len(values) *
                              (leave_one_out - leave_one_out.mean()).square().sum()).sqrt())
    return {"mean": mean, "covariance": (covariance + covariance.mT) / 2,
            "replications": len(values), "denominator": len(values) - 1,
            "trace": float(covariance.trace()), "trace_jackknife_standard_error": jackknife_se,
            "maximum_resolved_rank": min(len(values) - 1, values.shape[1]),
            "uncertainty_scope": "conditional_history_resampling_not_final_replica_uncertainty"}


def response_covariance(h: Tensor, p: Tensor, v: Tensor, indices, *, floor_relative=1e-8) -> dict:
    """Supported K_CC^-1 V_CC K_CC^-1, with no invented null-space information."""
    h, p, v = (fa._square(matrix) for matrix in (h, p, v))
    if h.shape != p.shape or h.shape != v.shape:
        raise ValueError("curvature, precision and gradient covariance layouts differ")
    indices = torch.as_tensor(indices, device=h.device)
    if (indices.ndim != 1 or indices.dtype != torch.long or not indices.numel()
            or (indices < 0).any() or (indices >= len(h)).any()
            or indices.unique().numel() != indices.numel()):
        raise ValueError("response coordinates must be unique nonempty int64 indices")
    take = lambda matrix: matrix[indices[:, None], indices[None, :]]
    response, diagnostics = _response(take(h + p), floor_relative, damp=False)
    covariance = response @ take(v) @ response
    return {"covariance": (covariance + covariance.mT) / 2,
            "optimism": float(torch.sum(response * take(v).mT)),
            "indices": indices, "solve_diagnostics": diagnostics,
            "supported_subspace_only": diagnostics["rank"] < len(indices),
            "numerical_floor_is_not_statistical_ridge": True}


def sample_history(inputs: Tensor, targets: Tensor, snapshot: dict, *, family: str,
                   seed: int, rotation=None) -> dict:
    """Independent actual-label draws at every original arrival's position/size."""
    positions, _ = fa.weighted_history(snapshot["batch_positions"], snapshot["batch_sizes"], snapshot["weights"])
    batches, labels, draws = [], [], []
    for arrival, (position, size) in enumerate(zip(positions.tolist(), snapshot["batch_sizes"], strict=True)):
        draw_seed = derive_component_seed(seed, f"arrival:{arrival}")
        x, metadata = fa.sample_reference_inputs(
            inputs, targets, family=family, batch_positions=[position], batch_sizes=[size],
            weights=torch.full((size,), 1 / size, dtype=torch.float64),
            sample_size=size, seed=draw_seed, instantaneous=position, rotation=rotation)
        batches.append(x)
        labels.append(targets.detach().cpu()[metadata["pool_indices"]])
        draws.append(metadata)
    return {"inputs": torch.cat(batches), "targets": torch.cat(labels),
            "weights": snapshot["weights"].detach().cpu().clone(),
            "batch_positions": list(snapshot["batch_positions"]), "batch_sizes": list(snapshot["batch_sizes"]),
            "position": snapshot["position"], "seed": seed, "arrival_draws": draws,
            "sampling": "independent_with_replacement_from_finite_reference_pool",
            "training_labels": "actual_reference_pool_labels_not_model_generated"}


def _observed(ctx, key, state, snapshot):
    def compute():
        ctx.load(state)
        x, y, a = ctx.data(snapshot)
        pieces = []
        for start in range(0, len(y), ctx.study.fisher_chunk_size):
            stop = min(len(y), start + ctx.study.fisher_chunk_size)
            piece = ctx.cache.get(f"{key}/gradients/{start}-{stop}", lambda: per_sample_derivatives(
                ctx.model, x[start:stop], y[start:stop], mnist_nll, ctx.layout,
                strategy="vmap").gradients.double())
            pieces.append(piece)
        return observed_covariance(torch.cat(pieces), a.cpu(), snapshot["batch_sizes"])
    try:
        return ctx.cache.get(key, compute)
    finally:
        ctx.load(ctx.snapshot["model"])


def _base(ctx):
    depth = ctx.case["frontier"]
    fitted = ctx.fit("base/restricted", ctx.snapshot["model"], depth, ctx.precision)
    bank = ctx.cache.get("base/model_bank", lambda: ctx.estimate(
        "base/model_bank", fitted["model"], ctx.policy.get("score_samples", ctx.study.score_samples)))
    kind = ctx.policy.get("gate") or "lr"
    h = represent(bank["matrix"], kind=kind, rank=ctx.policy.get("gate_rank", ctx.rank))
    observed = _observed(ctx, "base/observed", fitted["model"], ctx.snapshot)
    return {"restricted": fitted, "model_bank": bank, "H": h,
            "V_model": h / ctx.ess, "V_model_dense_same_bank": bank["matrix"] / ctx.ess,
            "V_observed": observed["covariance"], "observed": observed,
            "gradient": observed["mean_gradient"], "gate_representation": kind,
            "gradient_evaluation_state": "common_saved_buffer_restricted_fit",
            "repeated_fit_start": "saved_prefit_snapshot_not_common_restricted_fit"}


def _gate(ctx, fitted, g, h, v, depth, expansion, *, namespace):
    a, _ = active_indices(ctx.layout, depth)
    joined, _ = active_indices(ctx.layout, depth + expansion)
    b = joined[~torch.isin(joined, a)]
    if v is None:
        return {"status": "covariance_unavailable", "gain": None, "open": None,
                "delta_optimism": None, "risk_reduction": None, "c_sweep": {}, "scope": namespace}
    device = ctx.device
    g, h, v = (value.to(device=device, dtype=torch.float64) for value in (g, h, v))
    r = g + ctx.precision @ (fitted["theta"].to(device).double() - ctx.anchor.double())
    tolerance = ctx.fit_policy.get("residual_tolerance", ctx.study.tolerance_grad)
    result = evaluate_gate(g, r, h, ctx.precision, v, a.to(device), b.to(device),
                           c=ctx.policy.get("c", 1.), floor_relative=ctx.study.solve_floor_relative,
                           restricted_tolerance=tolerance)
    if not fitted["fit"]["converged"] or result["status"] == "restricted_fit_unresolved":
        result.update(status="restricted_fit_unresolved", open=None)
        for key in ("gain", "gain_expanded", "gain_active_only", "risk_reduction"):
            result[key] = None
    resolved = result["status"] in {"open", "closed"}
    result["c_sweep"] = {str(c): result["gain"] > c * max(result["delta_optimism"], 0.) if resolved else None
                         for c in (.5, 1., 2., 4.)}
    result["scope"] = namespace
    return fa._cpu(result)


def _evaluation(ctx, key, state):
    result = ctx.evaluation(key, state)
    stream = ctx.assets["streams"][ctx.case["environment"]["name"]]
    positions = stream["positions"]
    step = ctx.case["step"]
    if step >= len(positions) or float(positions[step]) != float(ctx.snapshot["position"]):
        raise ValueError("snapshot position does not match its declared stream step")
    next_position = float(positions[step + 1]) if step + 1 < len(positions) else None
    try:
        result["next"] = None if next_position is None else ctx.cache.get(
            f"{key}/next", lambda: ctx.losses(state, position=next_position))
    finally:
        ctx.load(ctx.snapshot["model"])
    return {**result, "next_position": next_position}


def _gains(ctx, before, after):
    result = fa._shadow_gains(ctx, before, after)
    result["next"] = None if before["next"] is None else fa.paired_panel_gain(
        before["next"], after["next"], family=ctx.family, position=before["next_position"])
    return result


def _newest(ctx, base):
    size = ctx.snapshot["batch_sizes"][-1]
    snapshot = {"inputs": ctx.snapshot["inputs"][-size:], "targets": ctx.snapshot["targets"][-size:],
                "weights": ctx.snapshot["weights"][-size:] / ctx.snapshot["weights"][-size:].sum(),
                "batch_positions": [ctx.snapshot["position"]], "batch_sizes": [size],
                "position": ctx.snapshot["position"]}
    fitted = base["restricted"]
    observed = _observed(ctx, "newest/observed", fitted["model"], snapshot)
    bank = ctx.cache.get("newest/model_bank", lambda: ctx.estimate(
        "newest/model_bank", fitted["model"], ctx.policy.get("score_samples", ctx.study.score_samples),
        snapshot=snapshot))
    h = represent(bank["matrix"], kind=base["gate_representation"],
                  rank=ctx.policy.get("gate_rank", ctx.rank))
    ess = effective_sample_size(snapshot["weights"])
    depth = ctx.case["frontier"]
    gates = {}
    for expansion in range(1, min(2, 6 - depth) + 1):
        gates[expansion] = {
            "curvature_noise_only": _gate(ctx, fitted, base["gradient"], h, h / ess, depth, expansion,
                                           namespace="full_episode_gradient_newest_curvature_noise"),
            "all_newest": _gate(ctx, fitted, observed["mean_gradient"], h, h / ess, depth, expansion,
                                  namespace="newest_gradient_curvature_noise_at_full_episode_fit"),
        }
    return {"gates": gates, "observed": observed, "model_bank": bank, "ess": ess,
            "evidence_count": size, "episode_anchor_and_precision_unchanged": True,
            "no_refit_or_archive_consolidation": True,
            "limitation": "Newest-score gates require stationarity at the fixed full-episode fitted state; otherwise unresolved."}


def _shadows(ctx, base, *, empirical_v=None):
    depth, fitted = ctx.case["frontier"], base["restricted"]
    before = _evaluation(ctx, "shadow/restricted_evaluation", fitted["model"])
    rows, records = [], {}
    covariances = {"model": base["V_model"], "observed": base["V_observed"]}
    if empirical_v is not None:
        covariances["repeated_history"] = empirical_v
    for expansion in range(1, min(2, 6 - depth) + 1):
        key = f"shadow/expand{expansion}"
        expanded = ctx.fit(f"{key}/fit", fitted["model"], depth + expansion, ctx.precision)
        after = _evaluation(ctx, f"{key}/evaluation", expanded["model"])
        gains = _gains(ctx, before, after)
        gates = {}
        for name, covariance in covariances.items():
            gate = ctx.cache.get(f"{key}/gate/{name}", lambda: _gate(
                ctx, fitted, base["gradient"], base["H"], covariance, depth, expansion,
                namespace="saved_episode_objective_fixed_archive"))
            resolved = gate["status"] in {"open", "closed"}
            shadow_resolved = fitted["fit"]["converged"] and expanded["fit"]["converged"]
            next_gain = None if gains["next"] is None else gains["next"]["gain"]
            training_gain = fitted["fit"]["data_loss_after"] - expanded["fit"]["data_loss_after"]
            row = {"metric_schema_version": CALIBRATION_SCHEMA_VERSION, "category": "expansion_shadow",
                   "step": ctx.case["step"], "frontier": depth, "expansion_layers": expansion,
                   "covariance": name, "gate_status": gate["status"], "gate_open": gate["open"],
                   "forecast_gain": gate["gain"], "delta_optimism": gate["delta_optimism"],
                   "forecast_risk_reduction": gate["risk_reduction"], "c_sweep": gate["c_sweep"],
                   "restricted_fit_converged": fitted["fit"]["converged"],
                   "expanded_fit_converged": expanded["fit"]["converged"],
                   "realized_weighted_training_gain": training_gain, "realized_gains": gains,
                   "realized_next_gain": next_gain, "next_position": before["next_position"],
                   "training_minus_next_gain": None if next_gain is None else training_gain - next_gain,
                   "forecast_minus_realized_training_gain": None if gate["gain"] is None else gate["gain"] - training_gain,
                   "next_gain_regret": (max(-next_gain, 0.) if gate["open"] else max(next_gain, 0.))
                   if resolved and shadow_resolved and next_gain is not None else None,
                   "optimism_scope": "next_condition_drift_and_selection_not_a_pure_optimism_estimate",
                   "research_only": True}
            rows.append(row)
            gates[name] = gate
        records[expansion] = {"fit": expanded, "evaluation": after, "gains": gains, "gates": gates}
    blocked = None
    if 1 in records and 2 in records:
        first, second = (records[k]["gates"]["model"] for k in (1, 2))
        if all(g["status"] in {"open", "closed"} for g in (first, second)):
            blocked = not first["open"] and bool(second["open"])
    newest = ctx.cache.get("newest/diagnostic", lambda: _newest(ctx, base))
    for expansion, gates in newest["gates"].items():
        for scope, gate in gates.items():
            rows.append({"metric_schema_version": CALIBRATION_SCHEMA_VERSION,
                         "category": "newest_batch_gate", "scope": scope,
                         "step": ctx.case["step"], "frontier": depth, "expansion_layers": expansion,
                         "gate_status": gate["status"], "gate_open": gate["open"], "forecast_gain": gate["gain"],
                         "delta_optimism": gate["delta_optimism"], "ess": newest["ess"],
                         "episode_anchor_and_precision_unchanged": True, "research_only": True})
    return {"rows": rows, "restricted_evaluation": before, "expansions": records,
            "newest_batch": newest, "two_layer_open_after_one_layer_closed": blocked,
            "frontier_source": "caller_declared_case_at_driver_prefit_state",
            "snapshot_stage": "driver_arrival_prefit",
            "next_condition_available": before["next_position"] is not None}


def _repetition(ctx, base, repeat):
    key = f"history/{repeat}"
    history = ctx.cache.get(f"{key}/sample", lambda: sample_history(
        ctx.assets["reference_inputs"], ctx.assets["reference_targets"], ctx.snapshot,
        family=ctx.family, seed=ctx.seed(f"observed_history:{repeat}"), rotation=ctx.rotation))
    observed = _observed(ctx, f"{key}/observed", base["restricted"]["model"], history)
    fitted = ctx.fit(f"{key}/fit", ctx.snapshot["model"], ctx.case["frontier"], ctx.precision,
                     snapshot=history)
    active, _ = active_indices(ctx.layout, ctx.case["frontier"])
    frozen = torch.ones(ctx.layout.total_numel, dtype=torch.bool)
    frozen[active] = False
    if not torch.equal(fitted["theta"][frozen], ctx.snapshot["theta"][frozen]):
        raise RuntimeError("restricted history fit changed frozen coordinates")
    return {"repeat": repeat, "seed": history["seed"], "mean_gradient": observed["mean_gradient"],
            "observed_covariance": observed["covariance"], "observed_status": observed["status"],
            "arrival_mean_gradients": observed["arrival_mean_gradients"],
            "theta": fitted["theta"], "fit": fitted["fit"],
            "realized_nines": int((history["targets"] == 9).sum()),
            "sample_count": len(history["targets"]), "ess": effective_sample_size(history["weights"])}


def _calibration(ctx, base, replications):
    gradients = replicated_covariance(torch.stack([r["mean_gradient"] for r in replications]))
    parameters = replicated_covariance(torch.stack([r["theta"] for r in replications]))
    active, _ = active_indices(ctx.layout, ctx.case["frontier"])
    empirical_theta = parameters["covariance"][active[:, None], active[None, :]]
    covariances = {"model": base["V_model"], "model_dense_same_bank": base["V_model_dense_same_bank"],
                   "observed": base["V_observed"], "repeated_history": gradients["covariance"]}
    complete_observed = all(r["observed_covariance"] is not None for r in replications)
    if complete_observed:
        covariances["mean_observed_over_histories"] = torch.stack([r["observed_covariance"] for r in replications]).mean(0)
    rows, responses = [], {}
    for name, covariance in covariances.items():
        if covariance is None:
            rows.append({"metric_schema_version": CALIBRATION_SCHEMA_VERSION,
                         "category": "covariance_calibration", "covariance": name,
                         "status": "singleton_strata_unresolved", "step": ctx.case["step"],
                         "frontier": ctx.case["frontier"]})
            continue
        response = response_covariance(base["H"], ctx.precision.cpu(), covariance, active,
                                       floor_relative=ctx.study.solve_floor_relative)
        responses[name] = response
        rows.append({"metric_schema_version": CALIBRATION_SCHEMA_VERSION,
                     "category": "covariance_calibration", "covariance": name,
                     "status": "available", "step": ctx.case["step"], "frontier": ctx.case["frontier"],
                     "gradient_noise_error": fa.matrix_metrics(covariance, gradients["covariance"]),
                     "parameter_covariance_error": fa.matrix_metrics(response["covariance"], empirical_theta),
                     "predicted_optimism": response["optimism"],
                     "supported_subspace_only": response["supported_subspace_only"],
                     "restricted_reference_converged": base["restricted"]["fit"]["converged"],
                     "all_history_fits_converged": all(r["fit"]["converged"] for r in replications),
                     "includes_all_assigned_history_fits": True,
                     "finite_covariance_reference": True, "research_only": True})
    return {"rows": rows, "empirical_mean_gradient": gradients,
            "empirical_fitted_parameters": parameters, "response_predictions": responses,
            "observed_covariance_complete_for_all_histories": complete_observed,
            "gradient_state_fixed_across_histories": True,
            "between_arrival_mean_drift_not_pooled": True}


def _prepare(store, index, case, asset_path, driver_path, *, resume, kind, phase="phase3"):
    depth = case["frontier"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 6:
        raise ValueError("calibration frontier must be an integer in 1 through 6")
    return fa._prepare(store, index, case, asset_path, driver_path,
                       resume=resume, phase=phase, kind=kind)


def _finish(store, session, ctx, rows, diagnostics, *, repeats, phase="phase3"):
    failures = sum(not r["fit"]["converged"] for r in diagnostics.get("histories", []))
    base = diagnostics["base"]
    limitations = [
        "Finite reference-pool resampling is conditional on the frozen archive and starting state, not final-replica evidence.",
        "Observed covariance is centered within arrivals; singleton strata cannot estimate it.",
        "A shared local curvature is a response approximation; retained-parameter uncertainty and nonlinear selection are not corrected.",
        "Dense structured realizations do not establish scalable policy cost.",
    ]
    if failures or not base["restricted"]["fit"]["converged"]:
        limitations.append("Unresolved fits are retained; covariance includes their accepted states and unresolved gates have no forecast gain.")
    if not diagnostics["shadows"]["next_condition_available"]:
        limitations.append("This is a terminal snapshot; matched next-condition gain is unavailable.")
    summary = {"metric_schema_version": CALIBRATION_SCHEMA_VERSION, "case": ctx.case,
               "width": ctx.width, "namespace": ctx.namespace, "phase": phase,
               "history_repeats": repeats, "unresolved_history_fits": failures,
               "restricted_reference_converged": base["restricted"]["fit"]["converged"],
               "all_history_fits_unresolved": failures == repeats if repeats else None,
               "model_score_bank_inputs": base["model_bank"]["input_count"],
               "score_draws_do_not_increase_evidence": True, "ess": ctx.ess,
               "row_count": len(rows), "research_only": True, "smoke": ctx.study.smoke,
               "snapshot_stage": "driver_arrival_prefit", "frontier_source": "case.frontier",
               "source_driver_depth": ctx.policy.get("depth"),
               "parameter_layout": ctx.layout.metadata(), "limitations": limitations}
    session.write_json("audit_metrics.json", fa._json(rows))
    session.write_json("summary.json", fa._json(summary))
    session.write_torch("diagnostics.pt", fa._cpu(diagnostics))
    return store.finish(session, (*REQUIRED, *ctx.cache.required()))


def ensure_calibration_audit(store, index, case, asset_path, driver_path, *, resume=True) -> Path:
    """One assigned Phase 3 state: 32 fixed-schedule history fits plus shadows.

    The parent ledger assigns 16 states (four conditions by four replicas).
    This runner neither selects additional states nor treats the 32 histories
    as independent experimental replicas. Smoke uses its frozen smaller count.
    """
    completed, session, ctx = _prepare(store, index, case, asset_path, driver_path,
                                       resume=resume, kind="calibration_audit")
    if completed is not None:
        return completed
    if ctx.study.history_variance_repeats < 2:
        raise ValueError("history variance requires at least two replications")
    base = ctx.cache.get("base", lambda: _base(ctx))
    histories = [ctx.cache.get(f"history/{i}/complete", lambda: _repetition(ctx, base, i))
                 for i in range(ctx.study.history_variance_repeats)]
    calibration = ctx.cache.get("calibration", lambda: _calibration(ctx, base, histories))
    shadows = ctx.cache.get("shadows", lambda: _shadows(
        ctx, base, empirical_v=calibration["empirical_mean_gradient"]["covariance"]))
    history_rows = [{"metric_schema_version": CALIBRATION_SCHEMA_VERSION, "category": "history_fit",
                     "repeat": r["repeat"], "seed": r["seed"], "converged": r["fit"]["converged"],
                     "restricted_residual_max": r["fit"]["restricted_residual_max"],
                     "realized_nines": r["realized_nines"], "sample_count": r["sample_count"],
                     "ess": r["ess"], "observed_covariance_status": r["observed_status"]}
                    for r in histories]
    return _finish(store, session, ctx, history_rows + calibration["rows"] + shadows["rows"],
                   {"base": base, "histories": histories, "calibration": calibration, "shadows": shadows},
                   repeats=len(histories))


def ensure_shadow_audit(store, index, case, asset_path, driver_path, *, resume=True, phase="phase3") -> Path:
    """Preassigned Phase 3 or bounded Phase 4 repair shadows, without history fits."""
    completed, session, ctx = _prepare(store, index, case, asset_path, driver_path,
                                       resume=resume, kind="shadow_audit", phase=phase)
    if completed is not None:
        return completed
    base = ctx.cache.get("base", lambda: _base(ctx))
    shadows = ctx.cache.get("shadows", lambda: _shadows(ctx, base))
    return _finish(store, session, ctx, shadows["rows"], {"base": base, "shadows": shadows},
                   repeats=0, phase=phase)
