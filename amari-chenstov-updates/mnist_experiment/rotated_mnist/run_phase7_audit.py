"""Create the immutable artifact-only Plan 7 coefficient audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from src.initialization import state_dict_hash
from src.representations import representation_from_artifact

from .artifacts import _read_json
from .phase5_single_lap_artifacts import load_completed_single_lap_run
from .phase6_artifacts import (
    load_completed_phase6_debias,
    load_completed_phase6_oracle,
)
from .phase6_oracle import angle_key
from .phase7_artifacts import PHASE7_REQUIRED_ARTIFACTS, Phase7RunStore
from .phase7_audit import (
    anchor_decomposition,
    risk_recommendations,
    stationary_weight_concentration,
    summarize_coefficient_rows,
    weight_concentration_update,
)
from .phase7_config import Phase7AuditConfig, load_phase7_audit_config
from .transform import tensor_content_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_q_recursion(
    rows: tuple[dict[str, Any], ...], batch_size: int
) -> float:
    errors = []
    for step in range(len(rows) - 2):
        current = rows[step]["controller"]
        following = rows[step + 1]["controller"]
        if current is None or following is None:
            raise ValueError("source trajectory lacks a controller state")
        q = 1.0 / float(current["effective_size"])
        expected = weight_concentration_update(
            q, float(current["applied_pi"]), batch_size
        )
        observed = 1.0 / float(following["effective_size"])
        errors.append(abs(expected - observed))
    return max(errors, default=0.0)


def _branch_diagnostics(
    config: Phase7AuditConfig,
    source,
    oracle,
    trajectories: dict[str, Any],
    model_states: dict[str, Any],
    references: dict[str, Any],
    fisher_artifacts: dict[str, Any],
) -> dict[str, Any]:
    state_hashes = {
        schedule: {
            condition: state_dict_hash(model_states[schedule][condition]["initial"])
            for condition in config.conditions
        }
        for schedule in config.schedule_kinds
    }
    parameter_hashes = {
        schedule: {
            condition: tensor_content_hash(
                trajectories[schedule][condition]["parameters"][0]
            )
            for condition in config.conditions
        }
        for schedule in config.schedule_kinds
    }
    state_values = {
        value for schedule in state_hashes.values() for value in schedule.values()
    }
    parameter_values = {
        value
        for schedule in parameter_hashes.values()
        for value in schedule.values()
    }
    first_reference = min(
        oracle.reference_metrics, key=lambda row: float(row["angle_degrees"])
    )
    zero_key = angle_key(0.0)
    source_initial = trajectories[config.schedule_kinds[0]][
        config.primary_condition
    ]["parameters"][0].to(torch.float64)
    reference_initial = references["parameters"][zero_key].to(torch.float64)
    initial_anchor_energy = {}
    for rank in config.ranks:
        fisher = representation_from_artifact(
            fisher_artifacts[zero_key][str(rank)]["representation"], device="cpu"
        ).to(dtype=torch.float64)
        error = source_initial - reference_initial
        initial_anchor_energy[str(rank)] = float(fisher.quadratic(error))
    trajectory_shapes = {
        schedule: {
            condition: list(
                trajectories[schedule][condition]["parameters"].shape
            )
            for condition in config.conditions
        }
        for schedule in config.schedule_kinds
    }
    return {
        "gate": "pass_with_reference_offset",
        "parameter_count": int(source_initial.numel()),
        "initial_state_hashes": state_hashes,
        "initial_parameter_hashes": parameter_hashes,
        "all_initial_states_identical": len(state_values) == 1,
        "all_initial_parameter_vectors_identical": len(parameter_values) == 1,
        "trajectory_parameter_shapes": trajectory_shapes,
        "reference_ancestry": {
            "source_run_id": oracle.reference_contract["source_run_id"],
            "source_condition": config.primary_condition,
            "source_initializer_state_hash": next(iter(state_values)),
            "byte_level_reference_pre_fit_state_recorded": False,
            "evidence": (
                "the Plan 6 reference contract names the source run and the "
                "runner initializes its reference branch from the source condition"
            ),
        },
        "reference_zero_degree_fit": {
            "initial_validation_nll": float(
                first_reference["fit"]["initial_validation_nll"]
            ),
            "best_validation_nll": float(
                first_reference["fit"]["best_validation_nll"]
            ),
            "best_epoch": int(first_reference["fit"]["best_epoch"]),
            "epochs_completed": int(first_reference["fit"]["epochs_completed"]),
            "validation_nll_reduction": float(
                first_reference["fit"]["initial_validation_nll"]
                - first_reference["fit"]["best_validation_nll"]
            ),
        },
        "initial_anchor_energy_by_rank": initial_anchor_energy,
        "interpretation_limit": (
            "anchor energy includes additional zero-degree optimization on the "
            "reference branch and cannot be attributed wholly to policy lag"
        ),
        "schedule_hashes": source.run_summary["schedule_hashes"],
    }


def _coefficient_rows(
    config: Phase7AuditConfig,
    source,
    debias,
    oracle,
    trajectories: dict[str, Any],
    references: dict[str, Any],
    fisher_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_key = str(config.oracle_sample_size)
    replicate_key = str(config.oracle_replicate_count)
    reference_sample_size = oracle.config.reference.fit_sample_size
    output = []
    for schedule in config.schedule_kinds:
        primary_oracles = oracle.oracle_estimates[schedule]
        debiased = {
            int(row["step"]): row
            for row in debias.recommendations[schedule][
                f"variance_scale_{config.variance_scale:g}"
            ]
        }
        for condition in config.conditions:
            source_rows = source.trajectory_metrics[schedule][condition]
            parameters = trajectories[schedule][condition]["parameters"].to(
                torch.float64
            )
            if parameters.shape != (len(source_rows), 512):
                raise ValueError("source trajectory parameter shape is invalid")
            q_error = _validate_q_recursion(
                source_rows, config.deployed_batch_size
            )
            if q_error > 1e-12:
                raise ValueError("stored q recursion does not reproduce")
            for step in range(len(source_rows) - 1):
                source_row = source_rows[step]
                following = source_rows[step + 1]
                controller = source_row["controller"]
                if controller is None:
                    raise ValueError("source transition lacks a controller decision")
                left_key = angle_key(float(source_row["angle_degrees"]))
                right_key = angle_key(float(following["angle_degrees"]))
                population_displacement = (
                    references["parameters"][right_key]
                    - references["parameters"][left_key]
                ).to(torch.float64)
                anchor_error = (
                    parameters[step] - references["parameters"][left_key]
                ).to(torch.float64)
                q = 1.0 / float(controller["effective_size"])
                applied_pi = float(controller["applied_pi"])
                for rank in config.ranks:
                    rank_key = str(rank)
                    details = primary_oracles[str(step)][sample_key][rank_key][
                        replicate_key
                    ]
                    if condition == config.primary_condition and not math.isclose(
                        q, float(details["q"]), rel_tol=0.0, abs_tol=1e-14
                    ):
                        raise ValueError("Plan 6 and source q values differ")
                    fisher = representation_from_artifact(
                        fisher_artifacts[left_key][rank_key]["representation"],
                        device="cpu",
                    ).to(dtype=torch.float64)
                    decomposition = anchor_decomposition(
                        population_displacement, anchor_error, fisher
                    )
                    population_correction = float(
                        details["signal_noise_correction"]
                    )
                    population_signal = max(
                        0.0,
                        decomposition.population_signal - population_correction,
                    )
                    if not math.isclose(
                        population_signal,
                        float(details["signal"]),
                        rel_tol=1e-9,
                        abs_tol=1e-10,
                    ):
                        raise ValueError("population signal does not reproduce Plan 6")
                    conditional_reference_correction = (
                        float(details["new_covariance_shape_risk"])
                        / reference_sample_size
                    )
                    conditional_signal = max(
                        0.0,
                        decomposition.conditional_signal
                        - conditional_reference_correction,
                    )
                    recommendations = risk_recommendations(
                        q=q,
                        old_covariance_shape=float(
                            details["old_covariance_shape_risk"]
                        ),
                        new_covariance_shape=float(
                            details["new_covariance_shape_risk"]
                        ),
                        deployed_batch_size=config.deployed_batch_size,
                        population_signal=population_signal,
                        conditional_signal=conditional_signal,
                    )
                    parameter_count = fisher.shape[0]
                    efficient_old = q * parameter_count
                    efficient_new = parameter_count / config.deployed_batch_size
                    efficient_marginal_pi = (
                        population_signal + efficient_old
                    ) / (population_signal + efficient_old + efficient_new)
                    efficient_conditional_pi = conditional_signal / (
                        conditional_signal + efficient_new
                    )
                    if condition == config.primary_condition and not math.isclose(
                        recommendations.centered_marginal,
                        float(details["pi"]),
                        rel_tol=1e-9,
                        abs_tol=1e-10,
                    ):
                        raise ValueError("centered marginal oracle does not reproduce")
                    online_old = float(controller["old_covariance_risk"])
                    online_new = float(controller["new_covariance_risk"])
                    online_denominator = online_old + online_new
                    online_covariance = (
                        None
                        if online_denominator == 0.0
                        else online_old / online_denominator
                    )
                    debias_row = (
                        debiased[step]
                        if condition == config.primary_condition
                        else None
                    )
                    stationary_q = stationary_weight_concentration(
                        applied_pi, config.deployed_batch_size
                    )
                    output.append(
                        {
                            "schedule": schedule,
                            "condition": condition,
                            "step": step,
                            "rank": rank,
                            "angle_degrees": float(source_row["angle_degrees"]),
                            "next_angle_degrees": float(
                                following["angle_degrees"]
                            ),
                            "cumulative_angular_degrees": float(
                                source_row["cumulative_angular_degrees"]
                            ),
                            "leg_id": int(source_row["leg_id"]),
                            "cold_start": bool(controller["cold_start_active"]),
                            "q": q,
                            "effective_size": 1.0 / q,
                            "applied_pi": applied_pi,
                            "stationary_q_at_applied_pi": stationary_q,
                            "q_minus_stationary_q": q - stationary_q,
                            "stationary_covariance_pi": applied_pi / 2.0,
                            "population_signal_raw": decomposition.population_signal,
                            "population_signal_correction": population_correction,
                            "population_signal_corrected": population_signal,
                            "anchor_error_energy": decomposition.anchor_error_energy,
                            "cross_term": decomposition.cross_term,
                            "conditional_signal_raw": decomposition.conditional_signal,
                            "conditional_reference_correction": (
                                conditional_reference_correction
                            ),
                            "conditional_signal_corrected": conditional_signal,
                            "decomposition_identity_error": (
                                decomposition.identity_error
                            ),
                            "old_covariance_shape_risk": float(
                                details["old_covariance_shape_risk"]
                            ),
                            "new_covariance_shape_risk": float(
                                details["new_covariance_shape_risk"]
                            ),
                            "old_shape_fraction_of_parameter_count": float(
                                details["old_covariance_shape_risk"]
                            )
                            / parameter_count,
                            "new_shape_fraction_of_parameter_count": float(
                                details["new_covariance_shape_risk"]
                            )
                            / parameter_count,
                            "old_covariance_risk": (
                                recommendations.old_covariance_risk
                            ),
                            "new_covariance_risk": (
                                recommendations.new_covariance_risk
                            ),
                            "covariance_only_pi": recommendations.covariance_only,
                            "centered_marginal_pi": (
                                recommendations.centered_marginal
                            ),
                            "realized_conditional_pi": (
                                recommendations.realized_conditional
                            ),
                            "efficient_mle_centered_marginal_pi": (
                                efficient_marginal_pi
                            ),
                            "efficient_mle_conditional_pi": (
                                efficient_conditional_pi
                            ),
                            "online_signal": float(controller["signal_energy"]),
                            "online_uncertainty_scale": float(
                                controller["uncertainty_scale_estimate"]
                            ),
                            "online_covariance_only_pi": online_covariance,
                            "online_plugin_pi": float(controller["plugin_pi"]),
                            "debiased_online_pi": (
                                None
                                if debias_row is None
                                or debias_row["debiased_unclipped_pi"] is None
                                else float(debias_row["debiased_unclipped_pi"])
                            ),
                            "debiased_online_signal": (
                                None
                                if debias_row is None
                                else float(debias_row["debiased_signal_energy"])
                            ),
                            "trend_variance_correction": (
                                None
                                if debias_row is None
                                else float(debias_row["scaled_variance_correction"])
                            ),
                        }
                    )
    return output


def _classify(
    summaries: list[dict[str, Any]], branch: dict[str, Any], config: Phase7AuditConfig
) -> dict[str, Any]:
    primary = [
        row
        for row in summaries
        if row["condition"] == config.primary_condition
        and int(row["rank"]) == config.primary_rank
    ]
    marginal_mae = sum(float(row["online_to_marginal_mae"]) for row in primary)
    conditional_mae = sum(float(row["online_to_conditional_mae"]) for row in primary)
    efficient_conditional_mae = sum(
        float(row["online_to_efficient_mle_conditional_mae"])
        for row in primary
    )
    population = sum(float(row["mean_population_signal"]) for row in primary)
    anchor = sum(float(row["mean_anchor_error_energy"]) for row in primary)
    reference_offset = float(
        branch["reference_zero_degree_fit"]["validation_nll_reduction"]
    )
    if anchor > 10.0 * max(population, 1e-15) and reference_offset > 0.01:
        classification = "mixed_reference_offset_and_online_signal_inflation"
        reason = (
            "the realized-anchor target is dominated by a reference branch that "
            "was substantially optimized beyond the shared initializer, while the "
            "online statistic remains above the centered population target"
        )
    elif conditional_mae < 0.8 * marginal_mae:
        classification = "realized_conditional_target_supported"
        reason = "the online recommendation is materially closer to the anchor target"
    elif marginal_mae < 0.8 * conditional_mae:
        classification = "centered_marginal_target_closer"
        reason = "the online recommendation is materially closer to the marginal target"
    else:
        classification = "mixed_or_inconclusive"
        reason = "the stored trajectory does not separate the two target interpretations"
    return {
        "classification": classification,
        "reason": reason,
        "primary_online_to_marginal_mae_sum": marginal_mae,
        "primary_online_to_conditional_mae_sum": conditional_mae,
        "primary_online_to_efficient_mle_conditional_mae_sum": (
            efficient_conditional_mae
        ),
        "primary_anchor_to_population_mean_energy_ratio": anchor
        / max(population, 1e-15),
        "reference_zero_degree_validation_nll_reduction": reference_offset,
        "mean_local_covariance_shape_fraction_of_parameter_count": sum(
            float(row["mean_old_shape_fraction_of_parameter_count"])
            for row in primary
        )
        / len(primary),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "cache/mnist_experiment/rotated_mnist/phase7/coefficient_audit"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(config_path: Path, output_root: Path, *, resume: bool) -> Path:
    config = load_phase7_audit_config(config_path)
    repo_root = Path(__file__).parents[2]
    source = load_completed_single_lap_run(repo_root / config.source_run_path)
    debias = load_completed_phase6_debias(repo_root / config.debias_run_path)
    oracle = load_completed_phase6_oracle(repo_root / config.oracle_run_path)
    if (
        source.config.run_id != config.source_run_id
        or debias.config.run_id != config.debias_run_id
        or oracle.config.run_id != config.oracle_run_id
        or debias.config.source_run_id != source.config.run_id
        or oracle.config.source_run_id != source.config.run_id
        or source.config.data.samples_per_step != config.deployed_batch_size
        or source.config.schedule_kinds != config.schedule_kinds
    ):
        raise ValueError("Plan 7 input artifacts violate the frozen contract")

    trajectories = _load_torch(source.path / "trajectories.pt")
    model_states = _load_torch(source.path / "model_states.pt")
    references = _load_torch(oracle.path / "reference_states.pt")
    fisher_artifacts = _load_torch(oracle.path / "reference_fishers.pt")
    branch = _branch_diagnostics(
        config,
        source,
        oracle,
        trajectories,
        model_states,
        references,
        fisher_artifacts,
    )
    if (
        branch["all_initial_states_identical"] is not True
        or branch["all_initial_parameter_vectors_identical"] is not True
        or branch["parameter_count"] != oracle.run_summary["parameter_count"]
    ):
        raise ValueError("Plan 7 parameter branches are incompatible")
    rows = _coefficient_rows(
        config,
        source,
        debias,
        oracle,
        trajectories,
        references,
        fisher_artifacts,
    )
    summaries = summarize_coefficient_rows(
        rows, live_start_step=source.config.controller.cold_start_steps
    )
    classification = _classify(summaries, branch, config)

    session = Phase7RunStore(output_root).begin(config, repo_root, resume=resume)
    session.write_json(
        "source_contract.json",
        {
            "source_run_id": source.config.run_id,
            "debias_run_id": debias.config.run_id,
            "oracle_run_id": oracle.config.run_id,
            "source_config_hash": source.config.config_hash,
            "debias_config_hash": debias.config.config_hash,
            "oracle_config_hash": oracle.config.config_hash,
            "source_manifest_sha256": _sha256(source.path / "manifest.json"),
            "source_trajectories_sha256": _sha256(
                source.path / "trajectories.pt"
            ),
            "debias_rows_sha256": _sha256(
                debias.path / "debiased_recommendations.json"
            ),
            "oracle_rows_sha256": _sha256(
                oracle.path / "oracle_estimates.json"
            ),
            "reference_states_sha256": _sha256(
                oracle.path / "reference_states.pt"
            ),
            "reference_fishers_sha256": _sha256(
                oracle.path / "reference_fishers.pt"
            ),
            "schedule_hashes": source.run_summary["schedule_hashes"],
            "transition_alignment": "row t uses the pre-transition state and target t+1",
        },
    )
    session.write_json("branch_diagnostics.json", branch)
    session.write_json("coefficient_rows.json", rows)
    session.write_json(
        "audit_summary.json",
        {
            "run_kind": "phase7_artifact_only_anchor_coefficient_audit",
            "config_hash": config.config_hash,
            "row_count": len(rows),
            "live_start_step": source.config.controller.cold_start_steps,
            "summaries": summaries,
            "evidence_classification": classification,
            "limitations": [
                branch["interpretation_limit"],
                "one trajectory cannot identify the replica mean anchor error",
                "online and oracle quadratics use different Fisher representations",
                "the local oracle remains a surrogate rather than predictive policy regret",
            ],
        },
    )
    return session.complete(required=PHASE7_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    path = run(arguments.config, arguments.output_root, resume=arguments.resume)
    print(json.dumps({"path": str(path), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
