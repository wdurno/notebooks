"""Screen EDR memory scales on one fixed-EWC trajectory without retraining."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from src.controller import (
    ControllerState,
    DiscountedRiskState,
    accept_controller_step,
    decide_controller,
    decide_discounted_risk_controller,
)
from src.hybrid import blend_archive_fisher
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from .phase4_artifacts import load_completed_phase4_run
from .phase5_artifacts import (
    PHASE5_SENSITIVITY_REQUIRED,
    RotatedPhase5RunStore,
)
from .phase5_config import RotatedPhase5Config, load_phase5_config
from .phase5_reconstruct import (
    SOURCE_CONDITION,
    _controller_config,
    _region,
    _resolve_source,
    _sha256,
)
from .run_phase3 import _fresh_fisher


ACTION_HALF_LIVES = (1.0, 2.0, 4.0, 8.0)
TREND_HALF_LIVES_DEGREES = (1.875, 3.75, 7.5)
DYNAMIC_KNOTS = (40, 60)
SAME_SPEED_KNOTS = (20, 80)
KNOT_WINDOW = 4
RESPONSE_HORIZON = 8


@dataclasses.dataclass
class _SensitivityState:
    controller: ControllerState
    discounted: DiscountedRiskState
    rows: list[dict[str, Any]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/phase5/sensitivity"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _setting_name(action_half_life: float, trend_half_life: float) -> str:
    action = f"{action_half_life:g}".replace(".", "p")
    trend = f"{trend_half_life:g}".replace(".", "p")
    return f"hpi{action}__hphi{trend}"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires observations")
    return statistics.fmean(values)


def _knot_response(
    actions: list[float], boundary: int, *, window: int, horizon: int
) -> dict[str, Any]:
    if boundary < window or boundary + horizon >= len(actions):
        raise ValueError("knot response window exceeds the action trajectory")
    before = actions[boundary - window : boundary]
    after = actions[boundary + 1 : boundary + 1 + window]
    knot_action = actions[boundary]
    departures = [
        abs(actions[index] - knot_action)
        for index in range(boundary + 1, boundary + 1 + horizon)
    ]
    maximum = max(departures)
    return {
        "step": boundary,
        "pre_mean": _mean(before),
        "knot_action": knot_action,
        "post_mean": _mean(after),
        "signed_post_minus_pre": _mean(after) - _mean(before),
        "absolute_post_pre_shift": abs(_mean(after) - _mean(before)),
        "maximum_predictable_departure": maximum,
        "maximum_departure_lag_steps": departures.index(maximum) + 1,
    }


def summarize_sensitivity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) <= max((*DYNAMIC_KNOTS, *SAME_SPEED_KNOTS)) + RESPONSE_HORIZON:
        raise ValueError("sensitivity trajectory is too short for knot diagnostics")
    actions = [float(row["applied_pi"]) for row in rows]
    cold_count = sum(bool(row["cold_start_active"]) for row in rows)
    post_cold = actions[cold_count:]
    if len(post_cold) < 3:
        raise ValueError("sensitivity trajectory lacks post-cold-start actions")
    dynamic = [
        _knot_response(
            actions, knot, window=KNOT_WINDOW, horizon=RESPONSE_HORIZON
        )
        for knot in DYNAMIC_KNOTS
    ]
    same_speed = [
        _knot_response(
            actions, knot, window=KNOT_WINDOW, horizon=RESPONSE_HORIZON
        )
        for knot in SAME_SPEED_KNOTS
    ]
    dynamic_response = _mean(
        [item["absolute_post_pre_shift"] for item in dynamic]
    )
    same_speed_response = _mean(
        [item["absolute_post_pre_shift"] for item in same_speed]
    )
    first_differences = [
        post_cold[index] - post_cold[index - 1]
        for index in range(1, len(post_cold))
    ]
    second_differences = [
        first_differences[index] - first_differences[index - 1]
        for index in range(1, len(first_differences))
    ]
    regions = {
        region: [
            float(row["applied_pi"])
            for row in rows
            if row["region"] == region and not row["cold_start_active"]
        ]
        for region in ("first_ascent", "return", "second_ascent")
    }
    regional_means = {name: _mean(values) for name, values in regions.items()}
    return {
        "action_half_life_steps": float(rows[0]["action_half_life_steps"]),
        "trend_half_life_degrees": float(rows[0]["trend_half_life_degrees"]),
        "cold_start_steps": cold_count,
        "cold_release_step": cold_count,
        "post_cold_action_min": min(post_cold),
        "post_cold_action_mean": _mean(post_cold),
        "post_cold_action_max": max(post_cold),
        "post_cold_action_span": max(post_cold) - min(post_cold),
        "post_cold_floor_fraction": _mean(
            [float(row["lower_bound_active"]) for row in rows[cold_count:]]
        ),
        "post_cold_ceiling_fraction": _mean(
            [float(row["upper_bound_active"]) for row in rows[cold_count:]]
        ),
        "total_variation": sum(abs(value) for value in first_differences),
        "mean_absolute_first_difference": _mean(
            [abs(value) for value in first_differences]
        ),
        "mean_absolute_second_difference": _mean(
            [abs(value) for value in second_differences]
        ),
        "regional_action_means": regional_means,
        "regional_action_mean_span": (
            max(regional_means.values()) - min(regional_means.values())
        ),
        "dynamic_knot_responses": dynamic,
        "same_speed_knot_responses": same_speed,
        "mean_dynamic_knot_response": dynamic_response,
        "mean_same_speed_knot_response": same_speed_response,
        "dynamic_to_same_speed_response_ratio": dynamic_response
        / max(same_speed_response, 1e-12),
        "all_finite": all(
            math.isfinite(float(row[name]))
            for row in rows
            for name in (
                "applied_pi",
                "instantaneous_old_risk",
                "instantaneous_new_risk",
                "old_risk_moment",
                "new_risk_moment",
            )
        ),
    }


def sensitivity_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_name = _setting_name(4.0, 7.5)
    baseline = summaries[baseline_name]
    comparisons = []
    candidates = []
    for action_half_life in (1.0, 2.0):
        name = _setting_name(action_half_life, 7.5)
        candidate = summaries[name]
        checks = {
            "dynamic_response_increased_by_50_percent": (
                candidate["mean_dynamic_knot_response"]
                >= 1.5 * baseline["mean_dynamic_knot_response"]
            ),
            "roughness_at_most_doubled": (
                candidate["mean_absolute_second_difference"]
                <= 2.0 * baseline["mean_absolute_second_difference"]
            ),
            "dynamic_reference_ratio_not_reduced": (
                candidate["dynamic_to_same_speed_response_ratio"]
                >= baseline["dynamic_to_same_speed_response_ratio"]
            ),
            "material_dynamic_response": (
                candidate["mean_dynamic_knot_response"] >= 0.003
            ),
            "not_floor_driven": candidate["post_cold_floor_fraction"] < 0.90,
            "all_finite": candidate["all_finite"],
        }
        passed = all(checks.values())
        comparisons.append(
            {
                "setting": name,
                "checks": checks,
                "passes": passed,
                "dynamic_response_ratio_to_baseline": (
                    candidate["mean_dynamic_knot_response"]
                    / max(baseline["mean_dynamic_knot_response"], 1e-12)
                ),
                "roughness_ratio_to_baseline": (
                    candidate["mean_absolute_second_difference"]
                    / max(baseline["mean_absolute_second_difference"], 1e-12)
                ),
            }
        )
        if passed:
            candidates.append(name)
    return {
        "recommendation": "reopen" if candidates else "retain_stop",
        "baseline_setting": baseline_name,
        "closed_loop_candidates": candidates,
        "shorter_action_half_life_comparisons": comparisons,
        "selection_uses_predictive_outcomes": False,
        "trend_cold_start_sensitivity_is_descriptive_only": True,
        "gate_semantics": (
            "tests whether shorter action memory reveals knot-aligned response "
            "without an unacceptable roughness increase"
        ),
    }


def run_phase5_sensitivity(
    config: RotatedPhase5Config,
    *,
    output_root: str | Path,
    repo_root: str | Path,
    resume: bool = False,
) -> Path:
    root = Path(repo_root)
    source_path = _resolve_source(config, root)
    source = load_completed_phase4_run(source_path)
    if (
        source.config.run_id != config.source_phase4_run_id
        or source.config.config_hash != config.source_phase4_config_hash
        or source.config.replica_id != config.replica_id
    ):
        raise ValueError("Phase 5 sensitivity source identity differs")
    schedule = source.stream_plan.schedule
    if schedule.num_transitions != 100:
        raise ValueError("Phase 5 sensitivity requires the 100-transition path")

    device = resolve_device(source.config.runtime.device)
    training_dtype = resolve_dtype(source.config.runtime.dtype)
    matrix_dtype = resolve_dtype(source.config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=source.config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = RotatedPhase5RunStore(
        output_root, run_kind="phase5_edr_sensitivity"
    ).begin(config, source.config, root, resume=resume)
    started = time.perf_counter()

    stream = torch.load(
        source.path / "stream_tensors.pt", map_location="cpu", weights_only=True
    )
    trajectories = torch.load(
        source.path / "trajectories.pt", map_location="cpu", weights_only=True
    )
    model_states = torch.load(
        source.path / "model_states.pt", map_location="cpu", weights_only=True
    )
    initial_fisher_artifact = torch.load(
        source.path / "initial_fisher.pt", map_location="cpu", weights_only=True
    )
    parameters = trajectories[SOURCE_CONDITION]["parameters"]
    displacements = trajectories[SOURCE_CONDITION]["displacements"]
    model, layout = build_canonical_model(
        derive_component_seed(
            source.config.replica_seed, "plan5_model_initialization"
        ),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(model_states[SOURCE_CONDITION]["initial"])
    representation = representation_from_artifact(
        initial_fisher_artifact["representation"], device=device
    )
    if not isinstance(representation, LowRankDiagonalFisher):
        raise ValueError("Phase 5 sensitivity requires low-rank plus diagonal Fisher")
    fisher = representation.to(device=device, dtype=matrix_dtype)

    states: dict[str, _SensitivityState] = {}
    fixed_configs = {}
    edr_configs = {}
    for action_half_life in ACTION_HALF_LIVES:
        for trend_half_life in TREND_HALF_LIVES_DEGREES:
            name = _setting_name(action_half_life, trend_half_life)
            states[name] = _SensitivityState(
                controller=ControllerState.initialize(
                    layout.total_numel,
                    source.config.data.initialization_size,
                    dtype=matrix_dtype,
                    device="cpu",
                ),
                discounted=DiscountedRiskState(),
                rows=[],
            )
            fixed_configs[name] = _controller_config(
                config,
                policy="fixed_unified",
                fixed_pi=0.05,
                trend_half_life_degrees=trend_half_life,
            )
            edr_configs[name] = _controller_config(
                config,
                policy="discounted_risk",
                fixed_pi=config.controller.cold_start_pi,
                trend_half_life_degrees=trend_half_life,
                action_half_life_steps=action_half_life,
            )

    maximum_trace_relative_error = 0.0
    progress = tqdm(
        range(schedule.num_transitions),
        desc="Phase 5 EDR sensitivity",
        unit="step",
    )
    for step in progress:
        predictable_fisher = fisher
        pending = {}
        for name, state in states.items():
            fixed = decide_controller(
                state.controller,
                fixed_configs[name],
                batch_size=source.config.data.samples_per_step,
                fisher=predictable_fisher,
            )
            edr = decide_discounted_risk_controller(
                state.controller,
                state.discounted,
                edr_configs[name],
                batch_size=source.config.data.samples_per_step,
                fisher=predictable_fisher,
            )
            action_half_life = edr_configs[name].action_half_life_steps
            trend_half_life = edr_configs[name].trend_half_life_p
            assert action_half_life is not None and trend_half_life is not None
            state.rows.append(
                {
                    "step": step,
                    "angle_degrees": schedule.angles_degrees[step],
                    "next_angle_degrees": schedule.angles_degrees[step + 1],
                    "region": _region(step, schedule.transitions_per_arrow),
                    "predictable_fisher_source_step": None if step == 0 else step - 1,
                    "action_half_life_steps": action_half_life,
                    "trend_half_life_degrees": trend_half_life,
                    "applied_pi": edr.controller.applied_pi,
                    "unclipped_pi": edr.unclipped_pi,
                    "cold_start_active": edr.controller.cold_start_active,
                    "lower_bound_active": edr.controller.lower_bound_active,
                    "upper_bound_active": edr.controller.upper_bound_active,
                    "instantaneous_old_risk": edr.instantaneous_old_risk,
                    "instantaneous_new_risk": edr.instantaneous_new_risk,
                    "old_risk_moment": edr.state.old_risk_moment,
                    "new_risk_moment": edr.state.new_risk_moment,
                    "edr_gain": edr.gain,
                }
            )
            pending[name] = (fixed, edr)

        layout.copy_vector_to_module(
            model, parameters[step].to(device=device, dtype=training_dtype)
        )
        fresh = _fresh_fisher(
            model,
            layout,
            stream["inputs"][step].to(device=device, dtype=training_dtype),
            stream["targets"][step].to(device=device),
            matrix_dtype=matrix_dtype,
        )
        if step > 0:
            update = blend_archive_fisher(
                fisher,
                fresh,
                blend_gain=0.05,
                rank=source.config.fisher.rank,
                lanczos_seed=derive_component_seed(
                    source.config.replica_seed,
                    f"plan5_phase4_ewc_lanczos:step={step}",
                ),
            )
            fisher = update.representation
            reconstructed_trace = update.candidate_trace
        else:
            reconstructed_trace = float(fisher.diagonal_vector().sum())
        source_trace = float(
            source.trajectory_metrics[SOURCE_CONDITION][step]["fisher_update"][
                "candidate_trace"
            ]
        )
        relative_error = abs(reconstructed_trace - source_trace) / max(
            abs(source_trace), 1e-12
        )
        maximum_trace_relative_error = max(
            maximum_trace_relative_error, relative_error
        )

        delta_degrees = abs(
            schedule.angles_degrees[step + 1] - schedule.angles_degrees[step]
        )
        for name, state in states.items():
            fixed, edr = pending[name]
            trend_half_life = edr_configs[name].trend_half_life_p
            assert trend_half_life is not None
            acceptance = accept_controller_step(
                state.controller,
                fixed,
                displacements[step].to(dtype=matrix_dtype),
                batch_size=source.config.data.samples_per_step,
                delta_p=delta_degrees,
                half_life_p=trend_half_life,
                fisher=predictable_fisher,
            )
            state.controller = acceptance.state
            state.discounted = edr.state
            state.rows[-1]["trend_gain"] = acceptance.gain
            state.rows[-1]["residual_risk_energy"] = acceptance.residual_squared

    summaries = {
        name: summarize_sensitivity_rows(state.rows)
        for name, state in states.items()
    }
    gate = sensitivity_gate(summaries)
    gate["maximum_fisher_trace_relative_error"] = maximum_trace_relative_error
    gate["fisher_reconstruction_exact"] = maximum_trace_relative_error <= 1e-6
    if not gate["fisher_reconstruction_exact"]:
        gate["recommendation"] = "retain_stop"
        gate["closed_loop_candidates"] = []

    source_mapping = {
        "path": str(source.path),
        "run_id": source.config.run_id,
        "config_hash": source.config.config_hash,
        "stream_plan_hash": source.stream_plan.content_hash,
        "schedule_hash": schedule.content_hash,
        "files": {
            name: _sha256(source.path / name)
            for name in (
                "config.json",
                "manifest.json",
                "stream_tensors.pt",
                "initial_fisher.pt",
                "trajectories.pt",
            )
        },
    }
    setting_names = list(states)
    session.write_json("source.json", source_mapping)
    session.write_json(
        "sensitivity_trajectories.json",
        {name: states[name].rows for name in setting_names},
    )
    session.write_json("sensitivity_summary.json", summaries)
    session.write_json("gate.json", gate)
    session.write_json(
        "run_summary.json",
        {
            "settings": setting_names,
            "action_half_lives_steps": list(ACTION_HALF_LIVES),
            "trend_half_lives_degrees": list(TREND_HALF_LIVES_DEGREES),
            "dynamic_knots": list(DYNAMIC_KNOTS),
            "same_speed_knots": list(SAME_SPEED_KNOTS),
            "knot_window": KNOT_WINDOW,
            "response_horizon": RESPONSE_HORIZON,
            "trajectory_rows_per_setting": schedule.num_transitions,
            "setting_count": len(setting_names),
            "source_phase4_run_id": source.config.run_id,
            "recommendation": gate["recommendation"],
            "wall_time_seconds": time.perf_counter() - started,
            "predictive_outcomes_used": False,
        },
    )
    return session.complete(PHASE5_SENSITIVITY_REQUIRED)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase5_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_phase5_sensitivity(
        config,
        output_root=arguments.output_root,
        repo_root=repo_root,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {"run_id": config.run_id, "path": str(path), "status": "completed"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
