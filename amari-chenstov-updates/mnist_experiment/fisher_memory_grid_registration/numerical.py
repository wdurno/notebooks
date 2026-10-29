"""Development-only FD checks and gradient-scale calibration on fixed snapshots.

This audit does not optimize prediction or tune against held-out outcomes.
The common probe draw is independent of training probes. Both float64 and
production-dtype FD values/gradients must match the float64 JVP reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from mnist_experiment.fisher_memory.learner import _stop
from src.seeding import derive_component_seed

from .curvature import fd_values, gaussian_nll, jvp_values
from .model import flat_prediction, memory_indices


NUMERICAL_SCHEMA_VERSION = 2
AUDIT_PROBE_COMPONENT = "registration_fd_numerical_audit_probes"


class CalibrationError(ValueError):
    """A failed calibration with JSON-safe diagnostic records for publication."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def _norm(value: Tensor) -> float | None:
    result = float(value.detach().double().norm())
    return result if math.isfinite(result) else None


def _relative_error(actual: Tensor, reference: Tensor) -> float | None:
    numerator, denominator = _norm(actual - reference), _norm(reference)
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _gradient(value: Tensor, theta: Tensor) -> Tensor:
    if not value.requires_grad:
        return torch.zeros_like(theta)
    result, = torch.autograd.grad(value, theta, allow_unused=True)
    return torch.zeros_like(theta) if result is None else result.detach()


def _identity(snapshot: Mapping, index: int) -> dict:
    seed = snapshot["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("snapshot seed must be an integer in [0, 2**63)")
    return {"snapshot_index": index, "snapshot_seed": seed,
            "probe_seed": derive_component_seed(seed, AUDIT_PROBE_COMPONENT),
            "n": int(snapshot["n"]), "step": int(snapshot["step"])}


def _prepare(snapshot: Mapping, layout, device: torch.device) -> dict:
    if "layout" in snapshot:
        layout.assert_metadata(snapshot["layout"])
    result = {key: torch.as_tensor(snapshot[key]).detach().to(device=device, dtype=torch.float64).clone()
              for key in ("theta", "new_x", "new_y", "replay_x", "replay_y", "ages")}
    if result["theta"].shape != (layout.total_numel,):
        raise ValueError("snapshot theta does not match the parameter layout")
    for prefix in ("new", "replay"):
        x, y = result[f"{prefix}_x"], result[f"{prefix}_y"]
        if y.ndim != 2 or y.shape[1] != 54 or x.shape != (len(y), 2, 84, 84):
            raise ValueError("snapshot pairs require matching physical-pixel targets")
    if not len(result["new_y"]) or result["ages"].shape != (len(result["replay_y"]),):
        raise ValueError("snapshot needs new data and one age weight per replay observation")
    if ((result["ages"] < 0) | (result["ages"] > 1)).any():
        raise ValueError("snapshot ages must be squared arrival weights in [0,1]")
    result["theta"].requires_grad_()
    return result


def _production_dtype(cfg, snapshot: Mapping) -> torch.dtype:
    value = getattr(cfg, "dtype", None)
    if value is None:
        value = torch.as_tensor(snapshot["theta"]).dtype
    if isinstance(value, str):
        value = {"float32": torch.float32, "float64": torch.float64}.get(value)
    if value not in (torch.float32, torch.float64):
        raise ValueError("production FD dtype must be float32 or float64")
    return value


def _comparison(approximate, gradient, reference, reference_gradient, ltm, stm, tolerance):
    value_error = _relative_error(approximate, reference)
    gradient_error = _relative_error(gradient, reference_gradient)
    finite = (bool(torch.isfinite(approximate).all()) and bool(torch.isfinite(gradient).all())
              and value_error is not None and gradient_error is not None)
    passed = finite and value_error <= tolerance and gradient_error <= tolerance
    return {
        "passed": passed, "status": "passed" if passed else "failed",
        "reason": (None if passed else "relative_error_exceeds_tolerance" if finite
                   else "nonfinite_or_unresolved_fd_comparison"),
        "value_relative_error": value_error, "gradient_relative_error": gradient_error,
        "fd_value_norm": _norm(approximate), "fd_gradient_norm": _norm(gradient),
        "fd_ltm_gradient_norm": _norm(gradient[ltm]), "fd_stm_gradient_norm": _norm(gradient[stm]),
    }


def calibrate_fd(cfg, snapshots: Sequence[Mapping], model, layout,
                 check_stop: Callable = lambda: None) -> dict:
    """Select a stable FD step and a finite, positive gradient-scale lambda0.

    Check h/2, h and 2h against exact JVP values and the full theta gradient of
    mean(age * probe_mean(curvature) / d_S). Each step is evaluated in float64
    and in cfg.dtype (falling back to the snapshot theta dtype). Both must pass
    every viable snapshot against the same float64 JVP reference. Prefer h;
    if only the equally close neighbors pass, prefer h/2. Production gradients
    remain in their native dtype until comparison; double accumulation cannot
    repair a low-precision forward or backward computation.

    Lambda0 is the median ||grad(0.5 new_NLL + 0.5 replay_NLL)|| divided by
    ||grad(unit age-weighted FD)|| at the selected step, using float64 for this
    scale estimate after both precisions pass. No retention term,
    denominator epsilon, predictive selection, or silent clipping is used.
    Empty, unaged, nonfinite, and zero-gradient snapshots are recorded as
    skipped. A failure raises CalibrationError with the accumulated diagnostics.
    """
    _stop(check_stop)
    h, tolerance, count = cfg.fd_step, cfg.numerical_tolerance, cfg.probes
    if (isinstance(h, bool) or not math.isfinite(h) or h <= 0
            or not math.isfinite(2 * h) or h / 2 == 0):
        raise ValueError("FD calibration needs a finite positive representable h/2, h, and 2h")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("numerical tolerance must be finite and positive")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("FD audit probe count must be a positive integer")
    steps = [h / 2, h, 2 * h]
    ltm, stm = memory_indices(layout)
    device = next(model.parameters()).device
    ltm, stm = ltm.to(device), stm.to(device)
    result = {
        "schema_version": NUMERICAL_SCHEMA_VERSION, "selected_step": None, "lambda0": None,
        "requested_step": h, "candidate_steps": steps, "numerical_tolerance": tolerance,
        "audit_dtype": "float64", "probe_component": AUDIT_PROBE_COMPONENT,
        "production_dtype": None, "production_dtypes": [], "production_verified": False,
        "gradient_ratio_dtype": "float64",
        "num_probes": count, "stm_dimension": len(stm), "numerical_rows": [],
        "gradient_ratios": [], "viable_snapshots": 0, "valid_gradient_ratios": 0,
        "step_preference": "original, then closest multiplicative distance, smaller on ties",
    }
    viable = []

    def skipped(identity, reason):
        for step in steps:
            result["numerical_rows"].append({**identity, "fd_step": step,
                                             "status": "skipped", "reason": reason,
                                             "passed": False, "float64_passed": False,
                                             "float64_status": "skipped", "float64_reason": reason,
                                             "production_passed": False, "production_status": "skipped",
                                             "production_reason": reason,
                                             "production_value_relative_error": None,
                                             "production_gradient_relative_error": None,
                                             "production_fd_value_norm": None,
                                             "production_fd_gradient_norm": None,
                                             "production_fd_ltm_gradient_norm": None,
                                             "production_fd_stm_gradient_norm": None})
        result["gradient_ratios"].append({**identity, "status": "skipped", "reason": reason,
                                          "fd_step": None, "ratio": None})

    for index, snapshot in enumerate(snapshots):
        _stop(check_stop)
        identity = _identity(snapshot, index)
        production_dtype = _production_dtype(cfg, snapshot)
        identity["production_dtype"] = str(production_dtype).removeprefix("torch.")
        result["production_dtypes"] = sorted(set(result["production_dtypes"]) | {identity["production_dtype"]})
        result["production_dtype"] = (result["production_dtypes"][0]
                                      if len(result["production_dtypes"]) == 1 else "mixed")
        values = _prepare(snapshot, layout, device)
        if not len(values["replay_y"]):
            skipped(identity, "empty_replay")
            continue
        if not all(torch.isfinite(value).all() for value in values.values()):
            skipped(identity, "nonfinite_snapshot")
            continue
        ages = values["ages"]
        if not ages.count_nonzero():
            skipped(identity, "no_aged_replay")
            continue
        theta, x = values["theta"], values["replay_x"]
        generator = torch.Generator(device="cpu").manual_seed(identity["probe_seed"])
        probes = (2 * torch.randint(2, (count, len(stm)), generator=generator) - 1).to(theta)
        _stop(check_stop)
        exact = jvp_values(model, layout, theta, x, probes)
        _stop(check_stop)
        exact_grad = _gradient((ages * exact.mean(0) / len(stm)).mean(), theta)
        exact = exact.detach()
        if (not torch.isfinite(exact).all() or not torch.isfinite(exact_grad).all()
                or _norm(exact) is None or _norm(exact_grad) is None):
            skipped(identity, "nonfinite_jvp_reference")
            continue
        if _norm(exact_grad) == 0:
            skipped(identity, "zero_jvp_penalty_gradient")
            continue
        production_theta = theta.detach().to(dtype=production_dtype).clone().requires_grad_()
        production_x = x.to(dtype=production_dtype)
        production_probes = probes.to(dtype=production_dtype)
        rows = {}
        for step in steps:
            _stop(check_stop)
            approximate = fd_values(model, layout, theta, x, probes, step)
            _stop(check_stop)
            approximate_grad = _gradient((ages * approximate.mean(0) / len(stm)).mean(), theta)
            approximate = approximate.detach()
            double_check = _comparison(approximate, approximate_grad, exact, exact_grad, ltm, stm, tolerance)
            if production_dtype == torch.float64:
                production_check = dict(double_check)
            else:
                _stop(check_stop)
                production_values = fd_values(model, layout, production_theta, production_x,
                                               production_probes, step)
                _stop(check_stop)
                production_gradient = _gradient(
                    (ages * production_values.mean(0) / len(stm)).mean(), production_theta
                )
                production_check = _comparison(production_values.detach(), production_gradient,
                                               exact, exact_grad, ltm, stm, tolerance)
            passed = double_check["passed"] and production_check["passed"]
            row = {
                **identity, "fd_step": step, **double_check,
                **{f"production_{key}": value for key, value in production_check.items()},
                "float64_passed": double_check["passed"], "float64_status": double_check["status"],
                "float64_reason": double_check["reason"], "passed": passed,
                "status": "passed" if passed else "failed",
                "reason": (None if passed else double_check["reason"] if not double_check["passed"]
                           else "production_" + production_check["reason"]),
                "jvp_value_norm": _norm(exact), "jvp_gradient_norm": _norm(exact_grad),
                "jvp_ltm_gradient_norm": _norm(exact_grad[ltm]),
                "jvp_stm_gradient_norm": _norm(exact_grad[stm]),
            }
            result["numerical_rows"].append(row)
            rows[step] = row
        viable.append((identity, values, rows))

    _stop(check_stop)
    result["viable_snapshots"] = len(viable)
    if not viable:
        raise CalibrationError(
            "FD calibration has no viable aged-replay snapshot: inspect skipped/nonfinite/zero-gradient diagnostics.",
            result,
        )
    passing = [step for step in steps if all(rows[step]["passed"] for _, _, rows in viable)]
    if not passing:
        raise CalibrationError(
            "No FD step among h/2, h, 2h passes value AND theta-gradient tolerance in both float64 "
            "and production dtype on every viable snapshot.",
            result,
        )
    selected = min(passing, key=lambda step: (abs(math.log2(step / h)), step))
    result["selected_step"] = selected
    result["production_verified"] = True
    ratios = []
    for identity, values, rows in viable:
        _stop(check_stop)
        theta = values["theta"]
        new = gaussian_nll(flat_prediction(model, layout, theta, values["new_x"]), values["new_y"]).mean()
        _stop(check_stop)
        replay = gaussian_nll(flat_prediction(model, layout, theta, values["replay_x"]), values["replay_y"]).mean()
        _stop(check_stop)
        ordinary_gradient = _gradient(.5 * (new + replay), theta)
        numerator = _norm(ordinary_gradient)
        denominator = rows[selected]["fd_gradient_norm"]
        ratio = None
        if (not torch.isfinite(ordinary_gradient).all() or numerator is None
                or denominator is None):
            reason = "nonfinite_gradient_norm"
        elif numerator == 0:
            reason = "zero_data_gradient"
        elif denominator == 0:
            reason = "zero_fd_penalty_gradient"
        else:
            ratio = numerator / denominator
            reason = None if math.isfinite(ratio) and ratio > 0 else "nonfinite_or_underflowed_ratio"
            if reason is not None:
                ratio = None
        result["gradient_ratios"].append({
            **identity, "fd_step": selected, "status": "valid" if reason is None else "skipped",
            "reason": reason, "ratio": ratio, "data_gradient_norm": numerator,
            "fd_gradient_norm": denominator, "data_ltm_gradient_norm": _norm(ordinary_gradient[ltm]),
            "data_stm_gradient_norm": _norm(ordinary_gradient[stm]),
            "fd_ltm_gradient_norm": rows[selected]["fd_ltm_gradient_norm"],
            "fd_stm_gradient_norm": rows[selected]["fd_stm_gradient_norm"],
        })
        if ratio is not None:
            ratios.append(ratio)
    result["gradient_ratios"].sort(key=lambda row: row["snapshot_index"])
    result["valid_gradient_ratios"] = len(ratios)
    _stop(check_stop)
    if not ratios:
        raise CalibrationError(
            "FD calibration has no finite positive gradient ratio; degenerate denominators are not epsilon-clamped.",
            result,
        )
    # This midpoint avoids overflow of the sum and underflow from halving
    # two positive subnormal ratios independently.
    ordered = sorted(ratios)
    middle = len(ordered) // 2
    result["lambda0"] = (ordered[middle] if len(ordered) % 2 else
                         ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2)
    return result
