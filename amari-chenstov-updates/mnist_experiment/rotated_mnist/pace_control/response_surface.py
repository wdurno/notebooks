"""Paired analysis for the Plan 10 Phase 1d response surface."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from src.seeding import derive_component_seed

from .response_surface_config import ResponseSurfaceConfig


def _quantile(values: Tensor, probability: float) -> float:
    return float(torch.quantile(values.to(dtype=torch.float64), probability))


def simultaneous_mean_intervals(
    samples: Tensor,
    *,
    confidence_level: float,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Return paired max-t-style intervals for a matrix of mean contrasts."""

    values = torch.as_tensor(samples, dtype=torch.float64, device="cpu")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("interval samples must have shape (replicate >= 2, contrast)")
    if not torch.isfinite(values).all():
        raise ValueError("interval samples must be finite")
    if not 0.0 < confidence_level < 1.0 or bootstrap_replicates < 100:
        raise ValueError("invalid bootstrap interval controls")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        values.shape[0],
        (bootstrap_replicates, values.shape[0]),
        generator=generator,
    )
    point = values.mean(dim=0)
    bootstrap = values[indices].mean(dim=1)
    centered = bootstrap - point
    upper_critical = _quantile(centered.max(dim=1).values, confidence_level)
    lower_critical = _quantile((-centered).max(dim=1).values, confidence_level)
    lower = point - lower_critical
    upper = point + upper_critical
    return {
        "mean": point.tolist(),
        "simultaneous_lower": lower.tolist(),
        "simultaneous_upper": upper.tolist(),
        "lower_critical": lower_critical,
        "upper_critical": upper_critical,
        "confidence_level": confidence_level,
        "bootstrap_replicates": bootstrap_replicates,
        "complete_replicates": int(values.shape[0]),
    }


def regret_interval(
    nll: Tensor,
    pi_values: Sequence[float],
    *,
    fixed_pi: float,
    confidence_level: float,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = torch.as_tensor(nll, dtype=torch.float64, device="cpu")
    if values.ndim != 2 or values.shape[1] != len(pi_values):
        raise ValueError("NLL values must have one column per pi")
    try:
        fixed_index = tuple(pi_values).index(fixed_pi)
    except ValueError as exc:
        raise ValueError("fixed pi must belong to the diagnostic grid") from exc
    differences = values[:, fixed_index].unsqueeze(1) - values
    intervals = simultaneous_mean_intervals(
        differences,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    point = torch.tensor(intervals["mean"], dtype=torch.float64)
    lower = torch.tensor(intervals["simultaneous_lower"], dtype=torch.float64)
    upper = torch.tensor(intervals["simultaneous_upper"], dtype=torch.float64)
    intervals.update(
        {
            "pi_values": list(pi_values),
            "point_regret": max(0.0, float(point.max())),
            "lower_regret": max(0.0, float(lower.max())),
            "upper_regret": max(0.0, float(upper.max())),
        }
    )
    return intervals


def _complete_matrix(
    records: Sequence[Mapping[str, Any]], pi_values: tuple[float, ...]
) -> tuple[Tensor, list[int]]:
    by_replicate: dict[int, dict[float, float]] = defaultdict(dict)
    for row in records:
        if row.get("failure") is None and row.get("next_nll") is not None:
            by_replicate[int(row["replicate_index"])][float(row["pi"])] = float(
                row["next_nll"]
            )
    complete = sorted(
        replicate
        for replicate, values in by_replicate.items()
        if set(values) == set(pi_values)
    )
    if not complete:
        return torch.empty((0, len(pi_values)), dtype=torch.float64), []
    matrix = torch.tensor(
        [[by_replicate[replicate][pi] for pi in pi_values] for replicate in complete],
        dtype=torch.float64,
    )
    return matrix, complete


def _crossing_intervals(
    complete_by_group: Mapping[tuple[int, float], Tensor],
    anchor_directions: Mapping[int, int],
    in_bound_paces: tuple[float, ...],
    config: ResponseSurfaceConfig,
) -> dict[str, Any]:
    low_index = config.pi_values.index(config.crossing_low_pi)
    high_index = config.pi_values.index(config.crossing_high_pi)
    output = {}
    for direction in (-1, 1):
        anchors = sorted(
            anchor
            for anchor, value in anchor_directions.items()
            if value == direction
        )
        columns = []
        complete_count = min(
            complete_by_group[(anchor, pace)].shape[0]
            for anchor in anchors
            for pace in in_bound_paces
        )
        for pace in in_bound_paces:
            by_anchor = []
            for anchor in anchors:
                values = complete_by_group[(anchor, pace)][:complete_count]
                by_anchor.append(values[:, low_index] - values[:, high_index])
            columns.append(torch.stack(by_anchor).mean(dim=0))
        samples = torch.stack(columns, dim=1)
        intervals = simultaneous_mean_intervals(
            samples,
            confidence_level=config.confidence_level,
            bootstrap_replicates=config.bootstrap_replicates,
            seed=derive_component_seed(
                config.replica_seed, f"phase1d_crossing_bootstrap:{direction}"
            ),
        )
        lower = intervals["simultaneous_lower"]
        upper = intervals["simultaneous_upper"]
        witnesses = [
            {
                "lower_pace": in_bound_paces[left],
                "higher_pace": in_bound_paces[right],
            }
            for left in range(len(in_bound_paces))
            for right in range(left + 1, len(in_bound_paces))
            if upper[left] < 0.0 and lower[right] > 0.0
        ]
        output[str(direction)] = {
            **intervals,
            "pace_degrees": list(in_bound_paces),
            "low_pi": config.crossing_low_pi,
            "high_pi": config.crossing_high_pi,
            "anchor_steps": anchors,
            "resolved": bool(witnesses),
            "witnesses": witnesses,
        }
    return output


def analyze_response_surface(
    records: Iterable[Mapping[str, Any]],
    anchor_contracts: Sequence[Mapping[str, Any]],
    config: ResponseSurfaceConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = [dict(row) for row in records]
    anchor_directions = {
        int(anchor["step"]): int(anchor["direction_to_next"])
        for anchor in anchor_contracts
    }
    expected_fits = (
        len(config.anchor_steps)
        * len(config.pace_degrees)
        * config.replicate_count
        * len(config.pi_values)
    )
    if len(rows) != expected_fits:
        raise ValueError(
            f"response surface has {len(rows)} fits; expected {expected_fits}"
        )
    failed_fits = sum(row.get("failure") is not None for row in rows)
    failure_fraction = failed_fits / expected_fits
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["anchor_step"]), float(row["pace_degrees"]))].append(row)

    surface = []
    complete_by_group = {}
    margin = config.practical_nll_margin
    for anchor in config.anchor_steps:
        for pace in config.pace_degrees:
            matrix, complete = _complete_matrix(grouped[(anchor, pace)], config.pi_values)
            if matrix.shape[0] < 2:
                raise ValueError("response-surface group has fewer than two complete replicates")
            complete_by_group[(anchor, pace)] = matrix
            means = matrix.mean(dim=0)
            minimum = float(means.min())
            competitive = [
                pi
                for pi, value in zip(config.pi_values, means.tolist(), strict=True)
                if value <= minimum + margin
            ]
            interval = regret_interval(
                matrix,
                config.pi_values,
                fixed_pi=config.fixed_pi,
                confidence_level=config.confidence_level,
                bootstrap_replicates=config.bootstrap_replicates,
                seed=derive_component_seed(
                    config.replica_seed,
                    f"phase1d_regret_bootstrap:anchor={anchor}:pace={pace}",
                ),
            )
            best_index = int(torch.argmin(means))
            surface.append(
                {
                    "anchor_step": anchor,
                    "direction_to_next": anchor_directions[anchor],
                    "pace_degrees": pace,
                    "within_physical_bounds": (
                        config.physical_pace_minimum
                        <= pace
                        <= config.physical_pace_maximum
                    ),
                    "strictly_inside_physical_bounds": (
                        config.physical_pace_minimum
                        < pace
                        < config.physical_pace_maximum
                    ),
                    "complete_replicate_indices": complete,
                    "mean_nll_by_pi": {
                        str(pi): float(value)
                        for pi, value in zip(config.pi_values, means, strict=True)
                    },
                    "empirical_best_pi": config.pi_values[best_index],
                    "epsilon_competitive_pi": competitive,
                    "fixed_pi_competitive_point": config.fixed_pi in competitive,
                    "regret": interval,
                }
            )

    in_bound_paces = tuple(
        pace
        for pace in config.pace_degrees
        if config.physical_pace_minimum <= pace <= config.physical_pace_maximum
    )
    by_anchor = {
        anchor: [
            item
            for item in surface
            if item["anchor_step"] == anchor and item["within_physical_bounds"]
        ]
        for anchor in config.anchor_steps
    }
    for values in by_anchor.values():
        values.sort(key=lambda item: item["pace_degrees"])

    monotone = []
    for anchor, values in by_anchor.items():
        for previous, current in zip(values, values[1:], strict=False):
            passed = not (
                max(current["epsilon_competitive_pi"])
                < min(previous["epsilon_competitive_pi"])
            )
            monotone.append(
                {
                    "anchor_step": anchor,
                    "lower_pace": previous["pace_degrees"],
                    "higher_pace": current["pace_degrees"],
                    "nondecreasing_or_tied": passed,
                }
            )
    monotone_fraction = sum(
        item["nondecreasing_or_tied"] for item in monotone
    ) / len(monotone)

    anchor_upper_competitive = {
        anchor: any(item["regret"]["upper_regret"] <= margin for item in values)
        for anchor, values in by_anchor.items()
    }
    anchor_not_rejected = {
        anchor: any(item["regret"]["lower_regret"] <= margin for item in values)
        for anchor, values in by_anchor.items()
    }
    anchor_point_competitive = {
        anchor: any(item["regret"]["point_regret"] <= margin for item in values)
        for anchor, values in by_anchor.items()
    }
    anchor_interior_competitive = {
        anchor: any(
            item["strictly_inside_physical_bounds"]
            and item["regret"]["upper_regret"] <= margin
            for item in values
        )
        for anchor, values in by_anchor.items()
    }
    anchor_interior_point_competitive = {
        anchor: any(
            item["strictly_inside_physical_bounds"]
            and item["regret"]["point_regret"] <= margin
            for item in values
        )
        for anchor, values in by_anchor.items()
    }
    crossing = _crossing_intervals(
        complete_by_group,
        anchor_directions,
        in_bound_paces,
        config,
    )
    resolved_both_directions = all(
        value["resolved"] for value in crossing.values()
    )

    gate_checks = {
        "every_anchor_fixed_pi_competitive": all(anchor_upper_competitive.values()),
        "resolved_crossing_both_directions": resolved_both_directions,
        "minimizer_nondecreasing": (
            monotone_fraction >= config.minimum_monotone_fraction
        ),
        "interior_crossings_sufficient": (
            sum(anchor_interior_competitive.values())
            >= config.minimum_interior_anchor_count
        ),
        "fit_failures_acceptable": (
            failure_fraction < config.maximum_fit_failure_fraction
        ),
    }
    if config.stage == "smoke":
        gate_pass = failure_fraction == 0.0
        expansion_recommended = False
    elif config.stage == "coarse":
        decisive_direction_failures = {
            str(direction): all(
                not anchor_not_rejected[anchor]
                for anchor, value in anchor_directions.items()
                if value == direction
            )
            for direction in (-1, 1)
        }
        gate_checks = {
            "no_direction_decisively_rejected": not any(
                decisive_direction_failures.values()
            ),
            "fit_failures_acceptable": gate_checks["fit_failures_acceptable"],
        }
        gate_pass = all(gate_checks.values())
        expansion_recommended = False
    else:
        gate_pass = all(gate_checks.values())
        qualitative = (
            gate_checks["resolved_crossing_both_directions"]
            and gate_checks["minimizer_nondecreasing"]
            and sum(anchor_interior_point_competitive.values())
            >= config.minimum_interior_anchor_count
            and gate_checks["fit_failures_acceptable"]
        )
        uncertainty_only = (
            not gate_checks["every_anchor_fixed_pi_competitive"]
            and all(anchor_not_rejected.values())
            and all(anchor_point_competitive.values())
        )
        expansion_recommended = config.stage == "full" and qualitative and uncertainty_only

    diagnostics = {
        "monotone_comparisons": monotone,
        "crossing_contrasts": crossing,
        "anchor_upper_competitive": {
            str(key): value for key, value in anchor_upper_competitive.items()
        },
        "anchor_not_rejected": {
            str(key): value for key, value in anchor_not_rejected.items()
        },
        "anchor_point_competitive": {
            str(key): value for key, value in anchor_point_competitive.items()
        },
        "anchor_interior_competitive": {
            str(key): value for key, value in anchor_interior_competitive.items()
        },
    }
    summary = {
        "stage": config.stage,
        "anchor_count": len(config.anchor_steps),
        "pace_count": len(config.pace_degrees),
        "pi_count": len(config.pi_values),
        "replicate_count": config.replicate_count,
        "expected_fit_count": expected_fits,
        "failed_fit_count": failed_fits,
        "fit_failure_fraction": failure_fraction,
        "monotone_fraction": monotone_fraction,
        "resolved_crossing_both_directions": resolved_both_directions,
        "upper_competitive_anchor_count": sum(anchor_upper_competitive.values()),
        "point_competitive_anchor_count": sum(anchor_point_competitive.values()),
        "interior_upper_competitive_anchor_count": sum(
            anchor_interior_competitive.values()
        ),
        "gate_checks": gate_checks,
        "gate_pass": gate_pass,
        "expansion_recommended": expansion_recommended,
    }
    if not all(
        math.isfinite(float(row["next_nll"]))
        for row in rows
        if row.get("next_nll") is not None
    ):
        raise ValueError("response surface contains nonfinite NLL")
    return surface, {**diagnostics, "summary": summary}, summary
