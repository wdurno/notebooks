"""Bounded, artifact-free Phase 1 audits and initial gradient-strength brackets.

The caller supplies shared pre-solve checkpoints, including matured replay.
This module never creates data, writes files, changes the model, or runs a
trajectory. Its only optimization is the learner's bounded ordinary shadow
proposal for C. Returned dictionaries contain JSON scalars, lists, and mappings.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from src.seeding import derive_component_seed

from .curvature import exact_directional_values, exact_trace, jvp_values, kl_values, step_kl_values
from .learner import FitSettings, StopRequested, fit_step, prepare_objective
from .control import ResourceLimit, StopRequested as RuntimeStop
from .model import memory_indices


AUDIT_SCHEMA_VERSION = 1
PROBE_LADDER = (1, 2, 4, 8)
METHODS = ("jvp", "finite_difference", "directional")


def _stop(callback: Callable | None) -> None:
    if callback is not None and callback():
        raise StopRequested("numerical audit stopped between bounded checks")


def _sync(theta: Tensor) -> None:
    if theta.is_cuda:
        torch.cuda.synchronize(theta.device)


def _evaluate(function: Callable, theta: Tensor, check_stop: Callable | None) -> dict:
    _stop(check_stop)
    _sync(theta)
    started = time.perf_counter()
    with torch.enable_grad():
        point = theta.detach().requires_grad_()
        values = function(point)
        gradient = torch.autograd.grad(values.mean(), point)[0]
    _sync(theta)
    if not torch.isfinite(values).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite value or derivative in numerical audit")
    return {"values": values.detach(), "gradient": gradient.detach(),
            "seconds": time.perf_counter() - started}


def _vector_agreement(actual: Tensor, expected: Tensor, *, cosine: bool = False) -> dict:
    actual, expected = actual.flatten(), expected.flatten()
    norm = float(expected.norm())
    absolute = float((actual - expected).norm())
    near_zero = norm <= 1e-10
    relative = None if near_zero else absolute / norm
    product = float(actual.norm()) * norm
    alignment = None if product <= 1e-20 else float((actual @ expected) / product)
    passed = absolute <= 1e-9 if near_zero else relative < .02
    if cosine and not near_zero:
        passed = passed and alignment is not None and alignment > .99
    return {"reference_norm": norm, "absolute_error": absolute, "relative_error": relative,
            "cosine": alignment, "near_zero_reference": near_zero, "passed": bool(passed)}


def _agreement(actual: dict, expected: dict, ltm: Tensor, stm: Tensor) -> dict:
    values = _vector_agreement(actual["values"], expected["values"])
    gradients = {
        name: _vector_agreement(actual["gradient"][indices], expected["gradient"][indices], cosine=True)
        for name, indices in (("ltm", ltm), ("stm", stm))
    }
    return {"values": values, "gradients": gradients,
            "passed": values["passed"] and all(row["passed"] for row in gradients.values())}


def _gradient_stencil(function: Callable, theta: Tensor, gradient: Tensor,
                      ltm: Tensor, stm: Tensor, check_stop: Callable | None) -> dict:
    rows = {}
    for name, indices in (("ltm", ltm), ("stm", stm)):
        _stop(check_stop)
        norm = float(gradient[indices].norm())
        if norm <= 1e-10:
            rows[name] = {"near_zero_gradient": True, "passed": None, "gradient_norm": norm}
            continue
        direction = torch.zeros_like(theta)
        direction[indices] = gradient[indices] / norm
        epsilon = 1e-4
        with torch.no_grad():
            plus = function(theta + epsilon * direction).mean()
            minus = function(theta - epsilon * direction).mean()
        numerical = float((plus - minus) / (2 * epsilon))
        analytic = float(gradient @ direction)
        error = abs(numerical - analytic)
        rows[name] = {"near_zero_gradient": False, "gradient_norm": norm,
                      "coordinate_step": epsilon, "analytic": analytic, "finite_difference": numerical,
                      "absolute_error": error, "relative_error": error / abs(analytic),
                      "passed": math.isfinite(numerical) and error <= 1e-8 + .02 * abs(analytic)}
    return {"blocks": rows, "passed": all(row["passed"] is not False for row in rows.values()),
            "nondegenerate_blocks": sum(not row["near_zero_gradient"] for row in rows.values())}


def _norms(gradient: Tensor, ltm: Tensor, stm: Tensor) -> dict:
    return {"total": float(gradient.norm()), "ltm": float(gradient[ltm].norm()),
            "stm": float(gradient[stm].norm())}


def _bracket(auxiliary: Tensor, data: Tensor, ltm: Tensor, stm: Tensor,
             ratios: Sequence[float]) -> dict:
    auxiliary_norms, data_norms = _norms(auxiliary, ltm, stm), _norms(data, ltm, stm)
    reason = ("near_zero_data_gradient" if data_norms["total"] <= 1e-10 else
              "near_zero_auxiliary_gradient" if auxiliary_norms["total"] <= 1e-10 else None)
    scale = None if reason else data_norms["total"] / auxiliary_norms["total"]
    rows = []
    for ratio in ratios:
        coefficient = None if scale is None else float(ratio * scale)
        rows.append({"target_ratio": float(ratio), "coefficient": coefficient,
                     "block_ratios": {
                         name: None if coefficient is None or data_norms[name] <= 1e-10
                         else coefficient * auxiliary_norms[name] / data_norms[name]
                         for name in ("ltm", "stm")}})
    return {"calibrated": reason is None, "reason": reason, "unit_ratio_coefficient": scale,
            "auxiliary_gradient_norms": auxiliary_norms, "data_gradient_norms": data_norms,
            "bracket": rows}


def audit(
    model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
    replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any, anchor: Tensor,
    settings: FitSettings | Mapping, seed: int, *, cfg: Any = None,
    direction: Tensor | None = None, checkpoint_label: str | None = None,
    check_stop: Callable | None = None,
) -> dict:
    """Audit one shared checkpoint; return compact JSON without I/O.

    Settings use the learner's FitSettings contract. Optional cfg supplies
    ``strength_ratios`` (default .03/.1/.3) and ``diagnostic_probes`` (at least
    16). Audits use float64 functional parameters, without changing the model.
    The ordinary shadow proposal uses the supplied execution dtype and at most
    five closures. An optional direction must be a detached STM proposal from
    the same checkpoint; its provenance is labeled caller-supplied.

    Stable scheduler fields: ``valid_methods``, ``methods``, ``selected_probes``,
    ``selected_fd_step``, ``strengths`` (method -> coefficients in ratio order),
    ``strength_ratios``, ``calibration``, and ``calibration_signature``. None
    coefficients mean insufficient calibration evidence, never a zero penalty.
    High stochastic trace error does not make a method mathematically invalid.
    """
    _stop(check_stop)
    started = time.perf_counter()
    settings = settings if isinstance(settings, FitSettings) else FitSettings(**dict(settings))
    settings.validate()
    if settings.num_probes not in PROBE_LADDER:
        raise ValueError("numerical calibration requires num_probes in (1, 2, 4, 8)")
    option = lambda name, default: (cfg.get(name, default) if isinstance(cfg, Mapping)
                                    else getattr(cfg, name, default))
    ratios = tuple(float(value) for value in option("strength_ratios", (.03, .1, .3)))
    if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("strength ratios must be finite and positive")
    diagnostic_count = max(16, int(option("diagnostic_probes", 16)))
    if replay_x.shape[0] != replay_y.numel() or ages.shape != replay_y.shape:
        raise ValueError("replay inputs, labels, and age weights must agree")
    if replay_y.numel() == 0:
        raise ValueError("audit requires a shared checkpoint with nonempty replay")
    if not torch.isfinite(ages).all() or ((ages < 0) | (ages > 1)).any():
        raise ValueError("ages must be fixed squared weights in [0, 1]")
    point, inputs = theta.detach().double(), replay_x.detach().double()
    ltm, stm = (index.to(point.device) for index in memory_indices(layout))
    seeds = {name: derive_component_seed(seed, name) for name in
             ("training_curvature_probes", "independent_diagnostic_probes", "fallback_direction")}

    def probes(count, name):
        generator = torch.Generator(device="cpu").manual_seed(seeds[name])
        return (2 * torch.randint(2, (count, stm.numel()), generator=generator) - 1).to(point)

    bank = probes(8, "training_curvature_probes")
    selected = bank[:settings.num_probes]
    independent = probes(diagnostic_count, "independent_diagnostic_probes")
    exact = _evaluate(lambda p: exact_trace(model, layout, p, inputs, stm,
                                           chunk_size=settings.chunk_size), point, check_stop)
    reference = _evaluate(lambda p: exact_directional_values(
        model, layout, p, inputs, selected, chunk_size=settings.chunk_size), point, check_stop)
    result = {"schema_version": AUDIT_SCHEMA_VERSION, "checkpoint_label": checkpoint_label,
              "dtype": "float64", "execution_dtype": str(theta.dtype), "seed": seed, "seeds": seeds,
              "replay_count": len(replay_y), "mean_age_weight": float(ages.double().mean()),
              "has_mature_replay": bool((ages >= .5).any()),
              "exact_trace_mean": float(exact["values"].mean()),
              "exact_trace_seconds": exact["seconds"], "methods": {}, "probe_ladder": [],
              "selected_probes": settings.num_probes, "selected_fd_step": settings.fd_step,
              "strength_ratios": list(ratios), "calibration": {}, "strengths": {}}

    def error_record(error):
        return {"valid": False, "status": "unavailable_numerics", "error":
                f"{type(error).__name__}: {error}"}

    try:
        for count in PROBE_LADDER:
            row = _evaluate(lambda p: jvp_values(model, layout, p, inputs, bank[:count]), point, check_stop)
            averaged = {**row, "values": row["values"].mean(0)}
            diagnostic = _agreement(averaged, exact, ltm, stm)
            means = row["values"].mean(-1)
            relative_se = (None if count == 1 or abs(float(exact["values"].mean())) <= 1e-10 else
                           float(means.std() / math.sqrt(count) / exact["values"].mean().abs()))
            result["probe_ladder"].append({"probes": count, "seconds": row["seconds"],
                                            "trace_sampling_error": diagnostic,
                                            "estimated_relative_standard_error": relative_se})
            if count == settings.num_probes:
                training = row
        same_direction = _agreement(training, reference, ltm, stm)
        stencil = _gradient_stencil(lambda p: jvp_values(model, layout, p, inputs, selected),
                                    point, training["gradient"], ltm, stm, check_stop)
        result["methods"]["jvp"] = {"valid": same_direction["passed"] and stencil["passed"],
                                     "status": "checked", "agreement": same_direction,
                                     "gradient_check": stencil, "seconds": training["seconds"]}
        row = _evaluate(lambda p: jvp_values(model, layout, p, inputs, independent), point, check_stop)
        result["independent_trace"] = {
            "probes": diagnostic_count, "mean": float(row["values"].mean()),
            "agreement": _agreement({**row, "values": row["values"].mean(0)}, exact, ltm, stm),
            "seconds": row["seconds"], "used_for_mathematical_validity": False}
    except (StopRequested, RuntimeStop, ResourceLimit):
        raise
    except (RuntimeError, FloatingPointError) as error:
        result["methods"]["jvp"] = error_record(error)

    fd_candidates = []
    for h in (settings.fd_step / 2, settings.fd_step, settings.fd_step * 2):
        try:
            row = _evaluate(lambda p: kl_values(model, layout, p, inputs, selected, h), point, check_stop)
            agreement = _agreement(row, reference, ltm, stm)
            fd_candidates.append({"h": h, "full_perturbation_norm": h * math.sqrt(stm.numel()),
                                  "agreement": agreement, "seconds": row["seconds"],
                                  "valid": agreement["passed"]})
        except (StopRequested, RuntimeStop, ResourceLimit):
            raise
        except (RuntimeError, FloatingPointError) as error:
            fd_candidates.append({"h": h, **error_record(error)})
    valid_steps = [row["h"] for row in fd_candidates if row["valid"]]
    if valid_steps:
        chosen_h = min(valid_steps, key=lambda h: abs(math.log(h / settings.fd_step)))
        result["selected_fd_step"] = chosen_h
        function = lambda p: kl_values(model, layout, p, inputs, selected, chosen_h)
        row = _evaluate(function, point, check_stop)
        stencil = _gradient_stencil(function, point, row["gradient"], ltm, stm, check_stop)
        result["methods"]["finite_difference"] = {"valid": stencil["passed"], "status": "checked",
                                                   "candidates": fd_candidates, "gradient_check": stencil}
    else:
        result["methods"]["finite_difference"] = {"valid": False, "status": "needs_fd_calibration",
                                                   "candidates": fd_candidates}

    direction_source = "caller_supplied_shadow"
    shadow = {}
    if direction is None:
        direction_source = "ordinary_optimizer_shadow"
        shadow_settings = dataclasses.replace(settings, lambda_f=0., lambda_d=0.,
                                               max_eval=settings.shadow_max_eval,
                                               max_iter=min(settings.max_iter, settings.shadow_max_eval))
        proposed, shadow = fit_step(model, layout, theta, new_x, new_y, replay_x, replay_y,
                                    ages, fisher, anchor, "replay_ewc", shadow_settings, seed, check_stop)
        direction = (proposed - theta)[stm].detach()
    else:
        direction = torch.as_tensor(direction, device=point.device).detach()
    if direction.shape != (stm.numel(),) or not torch.isfinite(direction).all():
        raise ValueError("direction must be a finite STM-coordinate shadow proposal")
    direction = direction.double()
    original_norm = float(direction.norm())
    if original_norm > settings.direction_max_norm:
        direction = direction * (settings.direction_max_norm / original_norm)
    supported = float(direction.norm()) > settings.direction_min_norm
    unit = direction / direction.norm() if supported else probes(1, "fallback_direction")[0] / math.sqrt(stm.numel())
    local_delta = unit * min(1e-3, settings.direction_max_norm)
    result["direction"] = {"source": direction_source, "uncapped_norm": original_norm,
                            "norm": float(direction.norm()), "supported": supported,
                            "local_check_norm": float(local_delta.norm()),
                            "fallback_for_math_check_only": not supported,
                            "shadow": shadow}
    try:
        directional_reference = _evaluate(lambda p: exact_directional_values(
            model, layout, p, inputs, unit[None], chunk_size=settings.chunk_size)[0], point, check_stop)
        candidates = []
        radii = dict.fromkeys(min(radius, settings.direction_max_norm)
                              for radius in (1e-3, 5e-4, 2.5e-4, 1e-4))
        for radius in radii:
            function = lambda p, delta=unit * radius: step_kl_values(
                model, layout, p, inputs, delta, min_norm=0.)
            row = _evaluate(function, point, check_stop)
            agreement = _agreement(row, directional_reference, ltm, stm)
            candidates.append({"radius": radius, "agreement": agreement})
            if agreement["passed"]:
                break
        result["direction"].update(local_check_norm=radius, local_check_candidates=candidates)
        stencil = _gradient_stencil(function, point, row["gradient"], ltm, stm, check_stop)
        result["methods"]["directional"] = {"valid": agreement["passed"] and stencil["passed"],
                                             "status": "checked", "agreement": agreement,
                                             "gradient_check": stencil,
                                             "actual_direction_supported": supported}
        if supported:
            actual = _evaluate(lambda p: step_kl_values(model, layout, p, inputs, direction,
                                                       min_norm=settings.direction_min_norm), point, check_stop)
            result["direction"]["finite_radius_locality"] = _agreement(actual, directional_reference, ltm, stm)
            result["direction"]["finite_radius_locality_is_validity_test"] = False
    except (StopRequested, RuntimeStop, ResourceLimit):
        raise
    except (RuntimeError, FloatingPointError) as error:
        result["methods"]["directional"] = error_record(error)

    calibrated_settings = dataclasses.replace(settings, lambda_f=1., lambda_d=1.,
                                               fd_step=result["selected_fd_step"])
    common = (model, layout, point, new_x.double(), new_y, inputs, replay_y, ages, fisher, anchor)
    ordinary, _ = prepare_objective(*common, "replay_ewc", calibrated_settings, seed, check_stop)
    data = _evaluate(lambda p: ordinary(p)[1]["data_loss"], point, check_stop)
    for name in (*METHODS, "distillation"):
        _stop(check_stop)
        if name != "distillation" and not result["methods"][name]["valid"]:
            result["calibration"][name] = {"calibrated": False, "reason": "method_not_numerically_valid"}
            result["strengths"][name] = [None] * len(ratios)
            continue
        try:
            objective, _ = prepare_objective(*common, name, calibrated_settings, seed,
                                              check_stop, direction=direction if name == "directional" else None)
            auxiliary = _evaluate(lambda p: objective(p)[1]["auxiliary"], point, check_stop)
            bracket = _bracket(auxiliary["gradient"], data["gradient"], ltm, stm, ratios)
            if name == "directional" and not supported:
                bracket.update(calibrated=False, reason="unsupported_actual_direction")
            result["calibration"][name] = bracket
            result["strengths"][name] = [row["coefficient"] for row in bracket["bracket"]]
            if name == "distillation":
                stencil = _gradient_stencil(lambda p: objective(p)[1]["auxiliary"], point,
                                            auxiliary["gradient"], ltm, stm, check_stop)
                result["methods"][name] = {"valid": stencil["passed"] and
                                           float(auxiliary["gradient"][stm].norm()) == 0.,
                                           "status": "checked", "gradient_check": stencil,
                                           "stm_auxiliary_gradient_norm": float(auxiliary["gradient"][stm].norm())}
        except (StopRequested, RuntimeStop, ResourceLimit):
            raise
        except (RuntimeError, FloatingPointError) as error:
            result["methods"][name] = error_record(error)
            result["calibration"][name] = {"calibrated": False, "reason": "unsupported_auxiliary_gradient"}
            result["strengths"][name] = [None] * len(ratios)
    result["valid_methods"] = [name for name in METHODS if result["methods"][name]["valid"]]
    result["calibration_signature"] = {
        "num_probes": result["selected_probes"], "fd_step": result["selected_fd_step"],
        "direction_max_norm": settings.direction_max_norm, "replay_weight": settings.replay_weight,
        "strength_ratios": list(ratios)}
    result["status"] = "passed" if len(result["valid_methods"]) == len(METHODS) else "partial"
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def aggregate_audits(audits: Sequence[Mapping]) -> dict:
    """Pool nondegenerate checkpoint brackets, never change a checked protocol.

    The caller must rerun calibration under common selected settings when
    checkpoint-specific FD choices differ. A near-zero young checkpoint does
    not veto strengths identified at matured states. Method validity comes
    from mathematical checks, not trace-sampling errors or predictive results.
    """
    if not audits:
        raise ValueError("at least one audit is required")
    signature = audits[0]["calibration_signature"]
    if any(row["schema_version"] != AUDIT_SCHEMA_VERSION for row in audits):
        raise ValueError("incompatible numerical audit schema")
    if any(row["calibration_signature"] != signature for row in audits):
        raise ValueError("calibration settings differ; rerun under one common numerical contract")
    ratios = signature["strength_ratios"]
    methods, strengths = {}, {}
    for name in (*METHODS, "distillation"):
        valid = all(row["methods"][name]["valid"] for row in audits)
        eligible = [row["calibration"][name]["unit_ratio_coefficient"] for row in audits
                    if row["calibration"][name].get("calibrated")]
        scale = statistics.median(eligible) if valid and eligible else None
        methods[name] = {"valid": valid, "calibrated_checkpoints": len(eligible),
                         "unit_ratio_coefficient": scale}
        strengths[name] = [None if scale is None else ratio * scale for ratio in ratios]
    return {"schema_version": AUDIT_SCHEMA_VERSION, "checkpoint_count": len(audits),
            "has_mature_replay": any(row["has_mature_replay"] for row in audits),
            "calibration_signature": signature, "selected_probes": signature["num_probes"],
            "selected_fd_step": signature["fd_step"], "strength_ratios": ratios,
            "methods": methods, "strengths": strengths,
            "valid_methods": [name for name in METHODS if methods[name]["valid"]],
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in audits)}
