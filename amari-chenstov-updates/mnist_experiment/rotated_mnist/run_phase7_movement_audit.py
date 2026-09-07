"""Execute the artifact-only Plan 7 movement-premium calibration audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.controller import (
    decomposed_fixed_batch_pi,
    effective_size_update,
    tracked_q_covariance_pi,
)

from .phase6_artifacts import load_completed_phase6_oracle
from .phase7_artifacts import load_completed_phase7_audit
from .phase7_movement_artifacts import (
    MOVEMENT_REQUIRED_ARTIFACTS,
    MovementPremiumAuditStore,
)
from .phase7_movement_audit import (
    log_ratio,
    marginal_pi,
    movement_premium,
    summarize_error_breakdowns,
    summarize_movement_rows,
    transition_key,
)
from .phase7_movement_config import (
    MovementPremiumAuditConfig,
    load_movement_premium_audit_config,
)
from .phase8_artifacts import load_completed_phase8_run


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "cache/mnist_experiment/rotated_mnist/phase7/movement_premium_audit"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _population_lookup(config, coefficient_audit, oracle):
    if coefficient_audit.config.oracle_run_id != oracle.config.run_id:
        raise ValueError("coefficient audit and oracle ancestry differ")
    rows = [
        row
        for row in coefficient_audit.coefficient_rows
        if row["condition"] == coefficient_audit.config.primary_condition
        and int(row["rank"]) == config.oracle_rank
    ]
    expected = sum(
        len(oracle.oracle_estimates[schedule])
        for schedule in config.schedule_kinds
    )
    if len(rows) != expected:
        raise ValueError("coefficient audit does not cover the oracle path")

    lookup = {}
    maximum_reproduction_error = 0.0
    for row in rows:
        schedule = str(row["schedule"])
        step = int(row["step"])
        try:
            details = oracle.oracle_estimates[schedule][str(step)][
                str(config.oracle_sample_size)
            ][str(config.oracle_rank)][str(config.oracle_replicate_count)]
        except KeyError as exc:
            raise ValueError("selected population oracle estimate is absent") from exc
        comparisons = (
            (row["population_signal_corrected"], details["signal"]),
            (row["old_covariance_shape_risk"], details["old_covariance_shape_risk"]),
            (row["new_covariance_shape_risk"], details["new_covariance_shape_risk"]),
        )
        maximum_reproduction_error = max(
            maximum_reproduction_error,
            *(abs(float(left) - float(right)) for left, right in comparisons),
        )
        if any(not _close(float(left), float(right)) for left, right in comparisons):
            raise ValueError("Plan 7 coefficients do not reproduce Plan 6")
        key = transition_key(
            schedule,
            float(row["angle_degrees"]),
            float(row["next_angle_degrees"]),
        )
        if key in lookup:
            raise ValueError("population oracle transition is not unique")
        lookup[key] = {
            "signal": float(details["signal"]),
            "signal_raw": float(details["signal_raw"]),
            "signal_noise_correction": float(details["signal_noise_correction"]),
            "old_scale": float(details["old_covariance_shape_risk"]),
            "new_scale": float(details["new_covariance_shape_risk"]),
            "precision_pass": bool(details["precision_pass"]),
            "six_standard_error_half_width": float(
                details["six_standard_error_half_width"]
            ),
            "source_step": step,
        }
    return lookup, maximum_reproduction_error


def _validate_source_compatibility(config, coefficient_audit, oracle, plan8_runs):
    oracle_schedules = oracle.source_contract["schedule_hashes"]
    if coefficient_audit.source_contract["schedule_hashes"] != oracle_schedules:
        raise ValueError("coefficient and oracle schedule ancestry differ")
    if plan8_runs["single_lap"].source_contract["schedule_hashes"] != oracle_schedules:
        raise ValueError("single-lap and oracle schedule ancestry differ")

    parameter_counts = {
        int(oracle.reference_contract["parameter_count"]),
        *(int(run.source_contract["parameter_count"]) for run in plan8_runs.values()),
    }
    if parameter_counts != {512}:
        raise ValueError("movement-audit parameter layouts differ")
    initial_state_hashes = {
        str(run.source_contract["initial_state_hash"])
        for run in plan8_runs.values()
    }
    if len(initial_state_hashes) != 1:
        raise ValueError("Plan 8 movement paths do not share an initial state")
    partition_hashes = {
        str(oracle.source_contract["source_partition_hash"]),
        *(str(run.source_contract["partition_hash"]) for run in plan8_runs.values()),
    }
    if len(partition_hashes) != 1:
        raise ValueError("movement-audit data partitions differ")
    for run in plan8_runs.values():
        if run.config.controller.cold_start_steps != config.cold_start_steps:
            raise ValueError("Plan 8 cold-start contract differs from audit")
    return {
        "oracle_schedule_hashes": oracle_schedules,
        "plan8_schedule_hashes": {
            name: run.source_contract["schedule_hashes"]
            for name, run in plan8_runs.items()
        },
        "parameter_count": parameter_counts.pop(),
        "initial_state_hash": initial_state_hashes.pop(),
        "partition_hash": partition_hashes.pop(),
        "cold_start_steps": config.cold_start_steps,
    }


def _movement_rows(
    config: MovementPremiumAuditConfig,
    population: dict[tuple[str, str, str], dict[str, Any]],
    runs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    missing = []
    uses: Counter[tuple[str, str, str]] = Counter()
    maximum_action_error = 0.0
    maximum_plugin_error = 0.0
    maximum_q_error = 0.0

    for design, run in runs.items():
        if run.config.source_kind != design:
            raise ValueError("Plan 8 design and source kind differ")
        if run.run_summary["samples_per_step"] != config.deployed_batch_size:
            raise ValueError("Plan 8 batch size differs from movement audit")
        for schedule in config.schedule_kinds:
            trajectory = run.trajectory_metrics[schedule][config.condition]
            previous_direction = None
            reversal_step = None
            for index, row in enumerate(trajectory[:-1]):
                following = trajectory[index + 1]
                direction = int(row["direction_to_next"])
                expected_direction = (
                    1
                    if float(following["angle_degrees"])
                    > float(row["angle_degrees"])
                    else -1
                )
                if direction != expected_direction:
                    raise ValueError("transition direction and angles differ")
                if (
                    previous_direction is not None
                    and direction != 0
                    and direction != previous_direction
                ):
                    reversal_step = int(row["step"])
                if direction != 0:
                    previous_direction = direction
                steps_since_reversal = (
                    None
                    if reversal_step is None
                    else int(row["step"]) - reversal_step
                )
                key = transition_key(
                    schedule,
                    float(row["angle_degrees"]),
                    float(following["angle_degrees"]),
                )
                if key not in population:
                    missing.append(key)
                    continue
                uses[key] += 1
                reference = population[key]
                controller = row["controller"]
                acceptance = row["controller_acceptance"]
                if controller is None or acceptance is None:
                    raise ValueError("Plan 8 transition lacks controller state")
                q = 1.0 / float(controller["effective_size"])
                online_signal = float(controller["signal_squared"])
                online_scale = float(controller["trace_estimate"])
                fallback = bool(controller["unsupported_scale_fallback"])
                cold_start = bool(controller["cold_start_active"])
                clipped = bool(
                    controller["lower_bound_active"]
                    or controller["upper_bound_active"]
                )
                covariance_pi = tracked_q_covariance_pi(
                    q, config.deployed_batch_size
                )
                if fallback:
                    online_rho = None
                    online_instantaneous_pi = covariance_pi
                else:
                    online_rho = movement_premium(
                        online_signal,
                        online_scale,
                        config.deployed_batch_size,
                    )
                    online_instantaneous_pi = marginal_pi(
                        online_signal,
                        q,
                        online_scale,
                        online_scale,
                        config.deployed_batch_size,
                    )
                    maximum_plugin_error = max(
                        maximum_plugin_error,
                        abs(online_instantaneous_pi - float(controller["plugin_pi"])),
                    )

                population_signal = reference["signal"]
                old_scale = reference["old_scale"]
                new_scale = reference["new_scale"]
                population_rho = movement_premium(
                    population_signal,
                    new_scale,
                    config.deployed_batch_size,
                )
                population_pi = marginal_pi(
                    population_signal,
                    q,
                    old_scale,
                    new_scale,
                    config.deployed_batch_size,
                )
                population_covariance_pi = marginal_pi(
                    0.0,
                    q,
                    old_scale,
                    new_scale,
                    config.deployed_batch_size,
                )
                population_signal_online_scale_pi = (
                    covariance_pi
                    if fallback
                    else marginal_pi(
                        population_signal,
                        q,
                        online_scale,
                        online_scale,
                        config.deployed_batch_size,
                    )
                )
                online_signal_population_scale_pi = marginal_pi(
                    online_signal,
                    q,
                    old_scale,
                    new_scale,
                    config.deployed_batch_size,
                )

                discounted_rho = float(
                    controller["discounted_movement_premium"]
                )
                reconstructed_raw = (
                    float(controller["cold_start_pi"])
                    if cold_start
                    else covariance_pi
                    if fallback
                    else decomposed_fixed_batch_pi(
                        discounted_rho,
                        q,
                        config.deployed_batch_size,
                    )
                )
                bounded = min(
                    max(reconstructed_raw, run.config.controller.pi_min),
                    run.config.controller.pi_max,
                )
                maximum_action_error = max(
                    maximum_action_error,
                    abs(bounded - float(controller["applied_pi"])),
                )
                q_after = float(acceptance["state_after"]["q"])
                expected_q_after = effective_size_update(
                    q,
                    float(controller["applied_pi"]),
                    config.deployed_batch_size,
                )
                maximum_q_error = max(
                    maximum_q_error, abs(q_after - expected_q_after)
                )

                output.append(
                    {
                        "design": design,
                        "schedule": schedule,
                        "step": int(row["step"]),
                        "oracle_source_step": reference["source_step"],
                        "angle_degrees": float(row["angle_degrees"]),
                        "next_angle_degrees": float(following["angle_degrees"]),
                        "direction": direction,
                        "leg_id": int(row["leg_id"]),
                        "reversal_step": reversal_step,
                        "steps_since_reversal": steps_since_reversal,
                        "reversal_window": (
                            steps_since_reversal is not None
                            and steps_since_reversal < config.reversal_window_steps
                        ),
                        "cumulative_angular_degrees": float(
                            row["cumulative_angular_degrees"]
                        ),
                        "q": q,
                        "effective_size": float(controller["effective_size"]),
                        "cold_start": cold_start,
                        "fallback": fallback,
                        "clipped": clipped,
                        "eligible": not cold_start and not fallback and not clipped,
                        "applied_pi": float(controller["applied_pi"]),
                        "online_instantaneous_pi": online_instantaneous_pi,
                        "tracked_q_covariance_pi": covariance_pi,
                        "population_covariance_pi": population_covariance_pi,
                        "population_marginal_pi": population_pi,
                        "population_signal_online_scale_pi": (
                            population_signal_online_scale_pi
                        ),
                        "online_signal_population_scale_pi": (
                            online_signal_population_scale_pi
                        ),
                        "online_signal": online_signal,
                        "population_signal": population_signal,
                        "population_signal_raw": reference["signal_raw"],
                        "population_signal_noise_correction": reference[
                            "signal_noise_correction"
                        ],
                        "online_scale": online_scale,
                        "population_old_scale": old_scale,
                        "population_new_scale": new_scale,
                        "online_movement_premium": online_rho,
                        "discounted_movement_premium": discounted_rho,
                        "population_movement_premium": population_rho,
                        "signal_log_ratio": log_ratio(
                            online_signal,
                            population_signal,
                            floor=config.ratio_floor,
                        ),
                        "scale_log_ratio": log_ratio(
                            online_scale,
                            new_scale,
                            floor=config.ratio_floor,
                        ),
                        "oracle_precision_pass": reference["precision_pass"],
                        "oracle_six_standard_error_half_width": reference[
                            "six_standard_error_half_width"
                        ],
                    }
                )
    if missing:
        raise ValueError(f"unmatched population transitions: {missing[:3]}")
    diagnostics = {
        "row_count": len(output),
        "unique_population_transition_count": len(uses),
        "reused_population_transition_count": sum(count - 1 for count in uses.values()),
        "maximum_action_reconstruction_error": maximum_action_error,
        "maximum_plugin_reconstruction_error": maximum_plugin_error,
        "maximum_q_reconstruction_error": maximum_q_error,
        "all_transitions_matched": True,
    }
    return output, diagnostics


def _source_contract(
    paths: dict[str, Path],
    coefficient_audit,
    oracle,
    plan8_runs: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    files = {
        "oracle": ("config.json", "manifest.json", "oracle_estimates.json"),
        "coefficient": ("config.json", "manifest.json", "coefficient_rows.json"),
        "single_lap": ("config.json", "manifest.json", "trajectory_metrics.json"),
        "double_lap": ("config.json", "manifest.json", "trajectory_metrics.json"),
    }
    run_ids = {
        "oracle": oracle.config.run_id,
        "coefficient": coefficient_audit.config.run_id,
        **{name: run.config.run_id for name, run in plan8_runs.items()},
    }
    return {
        "run_ids": run_ids,
        "paths": {name: str(path) for name, path in paths.items()},
        "config_hashes": {
            "oracle": oracle.config.config_hash,
            "coefficient": coefficient_audit.config.config_hash,
            **{
                name: run.config.config_hash for name, run in plan8_runs.items()
            },
        },
        "file_sha256": {
            name: {filename: _sha256(paths[name] / filename) for filename in names}
            for name, names in files.items()
        },
        "source_artifacts_mutated": False,
        "historical_q_reused": False,
        "compatibility": compatibility,
    }


def main() -> None:
    arguments = _arguments()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_movement_premium_audit_config(arguments.config)
    paths = {
        "oracle": _resolve(repo_root, config.oracle_run_path),
        "coefficient": _resolve(repo_root, config.coefficient_run_path),
        "single_lap": _resolve(repo_root, config.single_lap_run_path),
        "double_lap": _resolve(repo_root, config.double_lap_run_path),
    }
    oracle = load_completed_phase6_oracle(paths["oracle"])
    coefficient_audit = load_completed_phase7_audit(paths["coefficient"])
    plan8_runs = {
        "single_lap": load_completed_phase8_run(paths["single_lap"]),
        "double_lap": load_completed_phase8_run(paths["double_lap"]),
    }
    actual_ids = (
        oracle.config.run_id,
        coefficient_audit.config.run_id,
        plan8_runs["single_lap"].config.run_id,
        plan8_runs["double_lap"].config.run_id,
    )
    expected_ids = (
        config.oracle_run_id,
        config.coefficient_run_id,
        config.single_lap_run_id,
        config.double_lap_run_id,
    )
    if actual_ids != expected_ids:
        raise ValueError("loaded movement-audit run IDs differ from configuration")
    compatibility = _validate_source_compatibility(
        config, coefficient_audit, oracle, plan8_runs
    )

    started = time.perf_counter()
    population, coefficient_error = _population_lookup(
        config, coefficient_audit, oracle
    )
    rows, diagnostics = _movement_rows(config, population, plan8_runs)
    summaries, classification = summarize_movement_rows(
        rows, attribution_tolerance=config.attribution_tolerance
    )
    breakdowns = summarize_error_breakdowns(
        rows, reversal_window_steps=config.reversal_window_steps
    )
    diagnostics["maximum_population_coefficient_reproduction_error"] = (
        coefficient_error
    )
    diagnostics["all_finite"] = all(
        not isinstance(value, float) or math.isfinite(value)
        for row in rows
        for value in row.values()
        if value is not None
    )
    if not diagnostics["all_finite"]:
        raise ValueError("movement audit produced nonfinite values")

    store = MovementPremiumAuditStore(arguments.output_root)
    session = store.begin(config, repo_root, resume=arguments.resume)
    session.write_json(
        "source_contract.json",
        _source_contract(
            paths, coefficient_audit, oracle, plan8_runs, compatibility
        ),
    )
    session.write_json("movement_rows.json", rows)
    summary = {
        "config_hash": config.config_hash,
        "row_count": len(rows),
        "summaries": summaries,
        "error_breakdowns": breakdowns,
        "evidence_classification": classification,
        "operational_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    session.write_json("audit_summary.json", summary)
    completed = session.complete(required=MOVEMENT_REQUIRED_ARTIFACTS)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(completed),
                "classification": classification,
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
