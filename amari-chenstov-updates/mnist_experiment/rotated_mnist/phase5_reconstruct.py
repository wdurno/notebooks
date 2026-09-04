"""Reconstruct predictable EDR coefficients from a completed Phase 4 path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from src.config import ControllerConfig
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
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from .phase4_artifacts import load_completed_phase4_run
from .phase5_artifacts import (
    PHASE5_RECONSTRUCTION_REQUIRED,
    RotatedPhase5RunStore,
)
from .phase5_config import RotatedPhase5Config, load_phase5_config
from .run_phase3 import _fresh_fisher


SOURCE_CONDITION = "ewc_fixed_pi005"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "cache/mnist_experiment/rotated_mnist/phase5/reconstruction"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_source(config: RotatedPhase5Config, repo_root: Path) -> Path:
    path = Path(config.source_phase4_run)
    return path if path.is_absolute() else repo_root / path


def _controller_config(
    config: RotatedPhase5Config,
    *,
    policy: str,
    fixed_pi: float,
    trend_half_life_degrees: float | None = None,
    action_half_life_steps: float | None = None,
) -> ControllerConfig:
    value = ControllerConfig(
        policy=policy,
        fixed_pi=fixed_pi,
        pi_min=config.controller.pi_min,
        pi_max=config.controller.pi_max,
        trend_half_life_p=(
            config.controller.trend_half_life_degrees
            if trend_half_life_degrees is None
            else trend_half_life_degrees
        ),
        trace_epsilon=config.controller.trace_epsilon,
        oracle_mode="none",
        reference_optimum_artifact=None,
        risk_metric="fisher",
        action_half_life_steps=(
            (
                config.controller.action_half_life_steps
                if action_half_life_steps is None
                else action_half_life_steps
            )
            if policy == "discounted_risk"
            else None
        ),
        damping=None,
        epsilon=None,
    )
    value.validate()
    return value


def _region(step: int, transitions_per_arrow: int) -> str:
    if step < 2 * transitions_per_arrow:
        return "first_ascent"
    if step < 3 * transitions_per_arrow:
        return "return"
    return "second_ascent"


def edr_reconstruction_gate(
    rows: list[dict[str, Any]],
    *,
    pi_min: float,
    prospective_pi: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("EDR gate requires coefficient rows")
    post_cold = [row for row in rows if not row["cold_start_active"]]
    if not post_cold:
        raise ValueError("EDR gate requires post-cold-start recommendations")
    actions = [float(row["applied_pi"]) for row in post_cold]
    regional = {
        region: [
            float(row["applied_pi"])
            for row in post_cold
            if row["region"] == region
        ]
        for region in ("first_ascent", "return", "second_ascent")
    }
    regional_means = {
        region: (sum(values) / len(values) if values else None)
        for region, values in regional.items()
    }
    finite_regional = [value for value in regional_means.values() if value is not None]
    action_span = max(actions) - min(actions)
    regional_mean_span = max(finite_regional) - min(finite_regional)
    floor_fraction = sum(value <= pi_min + 1e-12 for value in actions) / len(actions)
    mean_abs_from_prospective = sum(
        abs(value - prospective_pi) for value in actions
    ) / len(actions)
    checks = {
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
        "predictable_source_only": all(
            row["predictable_fisher_source_step"] == row["step"] - 1
            for row in rows[1:]
        )
        and rows[0]["predictable_fisher_source_step"] is None,
        "not_floor_driven": floor_fraction < 0.90,
        "action_span_at_least_001": action_span >= 0.01,
        "regional_mean_span_at_least_0005": regional_mean_span >= 0.005,
        "differs_from_fixed_005": mean_abs_from_prospective >= 0.005,
    }
    return {
        "recommendation": "proceed" if all(checks.values()) else "stop",
        "checks": checks,
        "post_cold_update_count": len(post_cold),
        "post_cold_floor_fraction": floor_fraction,
        "post_cold_action_min": min(actions),
        "post_cold_action_mean": sum(actions) / len(actions),
        "post_cold_action_max": max(actions),
        "post_cold_action_span": action_span,
        "post_cold_mean_absolute_difference_from_fixed_005": (
            mean_abs_from_prospective
        ),
        "regional_action_means": regional_means,
        "regional_action_mean_span": regional_mean_span,
        "gate_semantics": (
            "technical actuation gate only; no counterfactual predictive claim"
        ),
    }


def reconstruct_phase5(
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
        raise ValueError("Phase 5 source identity differs from its configuration")
    if source.config.rotation.knots_degrees != (0.0, 15.0, 30.0, 0.0, 15.0, 30.0):
        raise ValueError("Phase 5 source does not use the repeated rotation path")

    device = resolve_device(source.config.runtime.device)
    training_dtype = resolve_dtype(source.config.runtime.dtype)
    matrix_dtype = resolve_dtype(source.config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=source.config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = RotatedPhase5RunStore(
        output_root, run_kind="phase5_edr_reconstruction"
    ).begin(config, source.config, root, resume=resume)
    started = time.perf_counter()

    stream = torch.load(
        source.path / "stream_tensors.pt", map_location="cpu", weights_only=True
    )
    trajectories = torch.load(
        source.path / "trajectories.pt", map_location="cpu", weights_only=True
    )
    states = torch.load(
        source.path / "model_states.pt", map_location="cpu", weights_only=True
    )
    initial_fisher_artifact = torch.load(
        source.path / "initial_fisher.pt", map_location="cpu", weights_only=True
    )
    parameters = trajectories[SOURCE_CONDITION]["parameters"]
    displacements = trajectories[SOURCE_CONDITION]["displacements"]
    schedule = source.stream_plan.schedule
    if (
        parameters.shape[0] != schedule.num_points
        or displacements.shape[0] != schedule.num_transitions
    ):
        raise RuntimeError("Phase 4 source trajectory has incompatible dimensions")

    model, layout = build_canonical_model(
        derive_component_seed(
            source.config.replica_seed, "plan5_model_initialization"
        ),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(states[SOURCE_CONDITION]["initial"])
    representation = representation_from_artifact(
        initial_fisher_artifact["representation"], device=device
    )
    if not isinstance(representation, LowRankDiagonalFisher):
        raise ValueError("Phase 5 source Fisher must be low-rank plus diagonal")
    fisher = representation.to(device=device, dtype=matrix_dtype)

    controller_state = ControllerState.initialize(
        layout.total_numel,
        source.config.data.initialization_size,
        dtype=matrix_dtype,
        device="cpu",
    )
    discounted_state = DiscountedRiskState()
    fixed_config = _controller_config(config, policy="fixed_unified", fixed_pi=0.05)
    edr_config = _controller_config(
        config,
        policy="discounted_risk",
        fixed_pi=config.controller.cold_start_pi,
    )
    rows: list[dict[str, Any]] = []
    fisher_rows: list[dict[str, Any]] = []
    maximum_trace_relative_error = 0.0
    for step in tqdm(
        range(schedule.num_transitions),
        desc="Phase 5 EDR reconstruction",
        unit="step",
    ):
        predictable_fisher = fisher
        state_before = controller_state
        fixed_decision = decide_controller(
            state_before,
            fixed_config,
            batch_size=source.config.data.samples_per_step,
            fisher=predictable_fisher,
        )
        edr = decide_discounted_risk_controller(
            state_before,
            discounted_state,
            edr_config,
            batch_size=source.config.data.samples_per_step,
            fisher=predictable_fisher,
        )
        decision_mapping = {
            **edr.controller.mapping(extended=True),
            **edr.mapping(),
        }
        rows.append(
            {
                "step": step,
                "angle_degrees": schedule.angles_degrees[step],
                "next_angle_degrees": schedule.angles_degrees[step + 1],
                "angular_distance_degrees": abs(
                    schedule.angles_degrees[step + 1]
                    - schedule.angles_degrees[step]
                ),
                "region": _region(step, schedule.transitions_per_arrow),
                "predictable_fisher_source_step": None if step == 0 else step - 1,
                "fixed_source_pi": fixed_decision.applied_pi,
                "applied_pi": edr.controller.applied_pi,
                "unclipped_pi": edr.unclipped_pi,
                "cold_start_active": edr.controller.cold_start_active,
                "lower_bound_active": edr.controller.lower_bound_active,
                "upper_bound_active": edr.controller.upper_bound_active,
                "instantaneous_old_risk": edr.instantaneous_old_risk,
                "instantaneous_new_risk": edr.instantaneous_new_risk,
                "old_risk_moment": edr.state.old_risk_moment,
                "new_risk_moment": edr.state.new_risk_moment,
                "controller_state_before": state_before.scalar_mapping(
                    config.controller.trace_epsilon, risk_metric="fisher"
                ),
                "decision": decision_mapping,
            }
        )

        layout.copy_vector_to_module(
            model, parameters[step].to(device=device, dtype=training_dtype)
        )
        inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
        targets = stream["targets"][step].to(device=device)
        fresh = _fresh_fisher(
            model,
            layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        )
        source_update = source.trajectory_metrics[SOURCE_CONDITION][step][
            "fisher_update"
        ]
        if step == 0:
            reconstructed_trace = float(fisher.diagonal_vector().sum())
            lanczos = None
        else:
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
            lanczos = update.lanczos.mapping()
        source_trace = float(source_update["candidate_trace"])
        trace_relative_error = abs(reconstructed_trace - source_trace) / max(
            abs(source_trace), 1e-12
        )
        maximum_trace_relative_error = max(
            maximum_trace_relative_error, trace_relative_error
        )
        fisher_rows.append(
            {
                "step": step,
                "source_candidate_trace": source_trace,
                "reconstructed_candidate_trace": reconstructed_trace,
                "candidate_trace_relative_error": trace_relative_error,
                "fresh_trace": float(torch.trace(fresh)),
                "realized_rank": fisher.rank,
                "lanczos": lanczos,
            }
        )

        delta_degrees = rows[-1]["angular_distance_degrees"]
        acceptance = accept_controller_step(
            state_before,
            fixed_decision,
            displacements[step].to(dtype=matrix_dtype),
            batch_size=source.config.data.samples_per_step,
            delta_p=delta_degrees,
            half_life_p=config.controller.trend_half_life_degrees,
            fisher=predictable_fisher,
        )
        controller_state = acceptance.state
        discounted_state = edr.state
        rows[-1]["controller_acceptance"] = {
            "gain": acceptance.gain,
            "residual_risk_energy": acceptance.residual_squared,
            "residual_euclidean_squared": acceptance.residual_euclidean_squared,
            "scale_observation": acceptance.scale_observation,
        }
        rows[-1]["controller_state_after"] = controller_state.scalar_mapping(
            config.controller.trace_epsilon, risk_metric="fisher"
        )

    gate = edr_reconstruction_gate(
        rows,
        pi_min=config.controller.pi_min,
        prospective_pi=config.controller.cold_start_pi,
    )
    gate["fisher_trace_reconstruction_tolerance"] = 1e-6
    gate["maximum_fisher_trace_relative_error"] = maximum_trace_relative_error
    gate["checks"]["fisher_trace_reconstruction"] = (
        maximum_trace_relative_error <= 1e-6
    )
    gate["recommendation"] = (
        "proceed" if all(gate["checks"].values()) else "stop"
    )
    source_mapping = {
        "path": str(source.path),
        "run_id": source.config.run_id,
        "config_hash": source.config.config_hash,
        "stream_plan_hash": source.stream_plan.content_hash,
        "schedule_hash": source.stream_plan.schedule.content_hash,
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
    session.write_json("source.json", source_mapping)
    session.write_json("coefficient_trajectory.json", rows)
    session.write_json(
        "fisher_reconstruction.json",
        {
            "source_condition": SOURCE_CONDITION,
            "maximum_candidate_trace_relative_error": (
                maximum_trace_relative_error
            ),
            "all_rank_eight": all(row["realized_rank"] == 8 for row in fisher_rows),
            "rows": fisher_rows,
        },
    )
    session.write_json("gate.json", gate)
    session.write_json(
        "run_summary.json",
        {
            "trajectory_rows": len(rows),
            "parameter_count": layout.total_numel,
            "source_condition": SOURCE_CONDITION,
            "source_phase4_run_id": source.config.run_id,
            "recommendation": gate["recommendation"],
            "wall_time_seconds": time.perf_counter() - started,
            "counterfactual_predictive_claim": False,
        },
    )
    return session.complete(PHASE5_RECONSTRUCTION_REQUIRED)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase5_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = reconstruct_phase5(
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
