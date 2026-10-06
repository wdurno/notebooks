"""Plan 13 Phase 5 optimizer-by-EWC parity diagnostic.

The diagnostic reuses immutable digit-9-mixture assets and holds Plan 13's
prior-archive proposal ordering fixed.  It crosses the Plan 13 Armijo solver
with the historical strong-Wolfe L-BFGS solver, with and without EWC.
"""

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

from src.config import OptimizerConfig
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.mnist_data import MixtureStreamPlan
from src.representations import LowRankDiagonalFisher
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.run_phase8 import _finite_tree

from .artifacts import UnitStore, file_hash
from .config import Plan13Study, canonical_hash
from .environments.digit9_mixture import MIXTURE_PI, evaluate_mixture
from .mixture import _evaluation_assets, _load_initial_state
from .optimizer import fixed_budget_update
from .rotation import runtime
from .trajectory import _fresh_fisher


DEFAULT_CONFIG = Path("mnist_experiment/continual_subgd/configs/default.json")
DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
RUN_REQUIRED = ("metrics.json", "summary.json", "checks.json", "final_state.pt")
ANALYSIS_REQUIRED = ("summary.json", "checks.json")
DIAGNOSTIC_SCHEMA_VERSION = "plan13-phase5-optimizer-parity-v1"
HISTORICAL_ENDPOINT_RECALL = 0.77


@dataclasses.dataclass(frozen=True)
class DiagnosticCondition:
    name: str
    optimizer: str
    use_ewc: bool


CONDITIONS = (
    DiagnosticCondition("armijo_ewc", "armijo", True),
    DiagnosticCondition("armijo_current_only", "armijo", False),
    DiagnosticCondition("lbfgs_ewc", "lbfgs", True),
    DiagnosticCondition("lbfgs_current_only", "lbfgs", False),
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_asset(
    root: Path,
    study: Plan13Study,
    index: int,
) -> tuple[Path, str]:
    phase_root = root / "digit9_mixture" / "phase5"
    candidates = []
    for path in phase_root.glob(
        f"digit9_mixture__phase5__mixture_assets__{index:05d}__*"
    ):
        if not (path / "COMPLETED").is_file():
            continue
        config = _read_json(path / "config.json")
        if (
            config.get("study_hash") == study.config_hash
            and config.get("kind") == "mixture_assets"
            and config.get("index") == index
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one completed Phase 5 asset for replica {index}, "
            f"found {len(candidates)}"
        )
    asset_path = candidates[0]
    integrity = _read_json(asset_path / "integrity.json")
    digest = file_hash(asset_path / "assets.pt")
    if integrity.get("assets.pt") != digest:
        raise RuntimeError(f"source asset failed integrity validation: {asset_path}")
    return asset_path, digest


def _run_unit(
    store: UnitStore,
    index: int,
    condition: DiagnosticCondition,
    *,
    steps: int,
    asset_digest: str,
) -> dict[str, Any]:
    return store.unit(
        "phase5_diagnostic",
        "optimizer_parity",
        index,
        environment="digit9_mixture",
        condition=condition.name,
        detail={
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "steps": steps,
            "optimizer": condition.optimizer,
            "use_ewc": condition.use_ewc,
            "proposal_fisher_timing": "prior_archive",
            "source_asset_sha256": asset_digest,
        },
    )


def _lbfgs_config(store: UnitStore, condition: DiagnosticCondition) -> OptimizerConfig:
    learner = store.study.protocol.learner
    return OptimizerConfig(
        name="lbfgs",
        learning_rate=learner.learning_rate,
        inner_steps=learner.inner_steps,
        ewc_strength=1.0 if condition.use_ewc else 0.0,
        lbfgs_history_size=learner.lbfgs_history_size,
        lbfgs_max_eval_factor=learner.lbfgs_max_eval_factor,
        lbfgs_tolerance_grad=learner.lbfgs_tolerance_grad,
        lbfgs_tolerance_change=learner.lbfgs_tolerance_change,
        lbfgs_line_search_fn=learner.lbfgs_line_search_fn,
    )


def _optimizer_step(
    store: UnitStore,
    condition: DiagnosticCondition,
    model: torch.nn.Module,
    layout: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    fisher: LowRankDiagonalFisher,
    lbfgs_optimizer: torch.optim.Optimizer | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    training_fisher = fisher.to(device=inputs.device, dtype=inputs.dtype)
    strength = mixture_ewc_strength(MIXTURE_PI) if condition.use_ewc else 0.0
    if condition.optimizer == "armijo":
        result = fixed_budget_update(
            model,
            layout,
            inputs,
            targets,
            training_fisher,
            lambda gradient: gradient,
            strength=strength,
            kappa=0.0,
            inner_steps=store.study.inner_steps,
            learning_rate=store.study.learning_rate,
            max_backtracks=store.study.max_backtracks,
        )
        mapping = result.mapping()
        mapping["relative_final_gradient_norm"] = result.final_gradient_norm / max(
            result.initial_gradient_norm,
            torch.finfo(inputs.dtype).tiny,
        )
        mapping["objective_decrease"] = result.objective_before - result.objective_after
        mapping["optimizer_iterations"] = result.accepted_steps
        mapping["optimizer_function_evaluations"] = result.function_evaluations
        return result.displacement, mapping
    if condition.optimizer != "lbfgs" or lbfgs_optimizer is None:
        raise RuntimeError(f"unsupported diagnostic optimizer: {condition.optimizer}")
    result = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        training_fisher,
        _lbfgs_config(store, condition),
        lbfgs_optimizer,
        adaptation_weight=MIXTURE_PI,
    )
    return result.displacement, result.metrics_mapping()


def run_condition(
    store: UnitStore,
    index: int,
    condition: DiagnosticCondition,
    *,
    steps: int,
    resume: bool,
) -> Path:
    asset_path, asset_digest = _source_asset(store.root, store.study, index)
    unit = _run_unit(
        store,
        index,
        condition,
        steps=steps,
        asset_digest=asset_digest,
    )
    session = store.begin(unit, RUN_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, RUN_REQUIRED)
        assert completed is not None
        return completed

    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    if not 0 < steps < len(plan.p_values):
        raise ValueError("diagnostic steps must leave one endpoint for evaluation")
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout, fisher = _load_initial_state(store, assets, index)
    evaluation_inputs, evaluation_targets = _evaluation_assets(
        assets,
        device=device,
        dtype=training_dtype,
    )
    lbfgs_optimizer = None
    if condition.optimizer == "lbfgs":
        lbfgs_optimizer = build_optimizer(model, _lbfgs_config(store, condition))

    rows: list[dict[str, Any]] = []
    displacements = []
    started = time.perf_counter()
    for step in range(steps):
        p = float(plan.p_values[step])
        evaluation = evaluate_mixture(model, evaluation_inputs, evaluation_targets, p)
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
        fresh, _ = _fresh_fisher(
            model,
            layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        )
        archive_trace_before = float(fisher.diagonal_vector().sum())
        step_started = time.perf_counter()
        displacement, optimizer = _optimizer_step(
            store,
            condition,
            model,
            layout,
            inputs,
            targets,
            fisher,
            lbfgs_optimizer,
        )
        step_seconds = time.perf_counter() - step_started
        displacements.append(displacement.to(dtype=matrix_dtype).cpu())
        update = blend_archive_fisher(
            fisher,
            fresh,
            blend_gain=MIXTURE_PI,
            rank=8,
            lanczos_seed=derive_component_seed(
                store.study.seed("phase5_diagnostic:replica", index),
                f"{condition.name}:archive:{step}",
            ),
        )
        fisher = update.representation
        rows.append(
            {
                "stage": "pre_update",
                "step": step,
                "p": p,
                "replica_index": index,
                "condition": condition.name,
                "optimizer_kind": condition.optimizer,
                "use_ewc": condition.use_ewc,
                "archive_trace_before": archive_trace_before,
                "archive_trace_after": float(fisher.diagonal_vector().sum()),
                "optimizer_wall_seconds": step_seconds,
                "optimizer": optimizer,
                **evaluation,
            }
        )

    endpoint_p = float(plan.p_values[steps])
    endpoint = {
        "stage": "post_burn_endpoint",
        "step": steps,
        "p": endpoint_p,
        "replica_index": index,
        "condition": condition.name,
        "optimizer_kind": condition.optimizer,
        "use_ewc": condition.use_ewc,
        "archive_trace": float(fisher.diagonal_vector().sum()),
        **evaluate_mixture(model, evaluation_inputs, evaluation_targets, endpoint_p),
    }
    summary = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "phase5_diagnostic",
        "environment": "digit9_mixture",
        "replica_index": index,
        "condition": dataclasses.asdict(condition),
        "steps": steps,
        "samples_per_step": plan.samples_per_step,
        "proposal_fisher_timing": "prior_archive",
        "source_asset": str(asset_path),
        "source_asset_sha256": asset_digest,
        "endpoint": endpoint,
        "median_displacement_norm": statistics.median(
            float(row["optimizer"]["displacement_norm"]) for row in rows
        ),
        "median_data_loss_decrease": statistics.median(
            float(row["optimizer"]["data_loss_before"])
            - float(row["optimizer"]["data_loss_after"])
            for row in rows
        ),
        "median_relative_final_gradient_norm": statistics.median(
            float(row["optimizer"]["relative_final_gradient_norm"]) for row in rows
        ),
        "median_function_evaluations": statistics.median(
            float(row["optimizer"]["optimizer_function_evaluations"]) for row in rows
        ),
        "median_backtracking_rejections": statistics.median(
            float(row["optimizer"]["backtracking_rejections"]) for row in rows
        ),
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    displacement_tensor = torch.stack(displacements)
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(endpoint) and _finite_tree(summary),
        "displacements_finite": bool(torch.isfinite(displacement_tensor).all()),
        "endpoint_is_first_treatment_coordinate": steps == 32 and math.isclose(endpoint_p, 32 / 99),
        "source_asset_integrity_verified": True,
        "proposal_fisher_timing": "prior_archive",
    }
    if not all(
        value
        for key, value in checks.items()
        if key not in {"proposal_fisher_timing", "endpoint_is_first_treatment_coordinate"}
    ):
        raise RuntimeError(f"optimizer diagnostic checks failed: {checks}")
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    session.write_torch(
        "final_state.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "displacements": displacement_tensor,
            "parameter_layout": layout.metadata(),
        },
    )
    return store.finish(session, RUN_REQUIRED)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _analysis_unit(
    store: UnitStore,
    replicas: tuple[int, ...],
    steps: int,
    run_hashes: list[str],
) -> dict[str, Any]:
    return store.unit(
        "phase5_diagnostic",
        "optimizer_parity_analysis",
        0,
        environment="digit9_mixture",
        detail={
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "replicas": list(replicas),
            "steps": steps,
            "conditions": [condition.name for condition in CONDITIONS],
            "run_summary_hashes": run_hashes,
        },
    )


def analyze(
    store: UnitStore,
    replicas: tuple[int, ...],
    *,
    steps: int,
    resume: bool,
) -> Path:
    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    run_hashes = []
    for index in replicas:
        _, asset_digest = _source_asset(store.root, store.study, index)
        for condition in CONDITIONS:
            unit = _run_unit(
                store,
                index,
                condition,
                steps=steps,
                asset_digest=asset_digest,
            )
            path = store.completed(unit, RUN_REQUIRED)
            if path is None:
                raise RuntimeError(f"missing diagnostic run: replica {index}, {condition.name}")
            summaries[index, condition.name] = _read_json(path / "summary.json")
            run_hashes.append(file_hash(path / "summary.json"))
    unit = _analysis_unit(store, replicas, steps, run_hashes)
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed

    condition_summary = []
    for condition in CONDITIONS:
        values = [summaries[index, condition.name] for index in replicas]
        endpoints = [value["endpoint"] for value in values]
        condition_summary.append(
            {
                "condition": condition.name,
                "optimizer": condition.optimizer,
                "use_ewc": condition.use_ewc,
                "replicas": len(values),
                "mean_endpoint_nine_ovr_recall": _mean(
                    [float(value["nine_ovr_recall"]) for value in endpoints]
                ),
                "mean_endpoint_nine_ovr_balanced_accuracy": _mean(
                    [float(value["nine_ovr_balanced_accuracy"]) for value in endpoints]
                ),
                "mean_endpoint_nine_ovr_specificity": _mean(
                    [float(value["nine_ovr_specificity"]) for value in endpoints]
                ),
                "mean_endpoint_current_accuracy": _mean(
                    [float(value["current_accuracy"]) for value in endpoints]
                ),
                "mean_endpoint_p0_accuracy": _mean(
                    [float(value["p0_accuracy"]) for value in endpoints]
                ),
                "mean_median_displacement_norm": _mean(
                    [float(value["median_displacement_norm"]) for value in values]
                ),
                "mean_median_data_loss_decrease": _mean(
                    [float(value["median_data_loss_decrease"]) for value in values]
                ),
                "mean_median_relative_final_gradient_norm": _mean(
                    [float(value["median_relative_final_gradient_norm"]) for value in values]
                ),
                "mean_median_function_evaluations": _mean(
                    [float(value["median_function_evaluations"]) for value in values]
                ),
                "mean_wall_time_seconds": _mean(
                    [float(value["total_wall_time_seconds"]) for value in values]
                ),
            }
        )
    by_name = {row["condition"]: row for row in condition_summary}
    armijo_ewc = by_name["armijo_ewc"]
    lbfgs_ewc = by_name["lbfgs_ewc"]
    recall_gain = (
        lbfgs_ewc["mean_endpoint_nine_ovr_recall"]
        - armijo_ewc["mean_endpoint_nine_ovr_recall"]
    )
    optimizer_failure_supported = (
        recall_gain >= 0.20
        and lbfgs_ewc["mean_endpoint_nine_ovr_recall"] >= 0.50
    )
    historical_parity_reached = (
        lbfgs_ewc["mean_endpoint_nine_ovr_recall"]
        >= HISTORICAL_ENDPOINT_RECALL - 0.10
    )
    conclusion = (
        "optimizer_failure_supported"
        if optimizer_failure_supported
        else "optimizer_failure_not_yet_isolated"
    )
    summary = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "replicas": list(replicas),
        "steps": steps,
        "endpoint_p": steps / 99,
        "proposal_fisher_timing": "prior_archive",
        "condition_summary": condition_summary,
        "lbfgs_minus_armijo_ewc_endpoint_recall": recall_gain,
        "historical_endpoint_recall_reference": HISTORICAL_ENDPOINT_RECALL,
        "historical_parity_reached": historical_parity_reached,
        "conclusion": conclusion,
        "interpretation": (
            "L-BFGS recovers digit-9 acquisition while holding the Plan 13 "
            "objective, data, and Fisher ordering fixed."
            if optimizer_failure_supported
            else "The optimizer swap alone does not recover digit-9 acquisition; "
            "test historical Fisher ordering and Fisher-scale differences next."
        ),
    }
    checks = {
        "all_finite": _finite_tree(summary),
        "complete_factorial": len(condition_summary) == 4,
        "replica_count": len(replicas),
        "run_count": len(summaries),
        "analysis_hash": canonical_hash(summary),
    }
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, ANALYSIS_REQUIRED)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    replicas = tuple(arguments.replicas)
    if len(set(replicas)) != len(replicas) or any(index < 1 for index in replicas):
        raise ValueError("diagnostic replicas must be unique positive integers")
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    for index in replicas:
        for condition in CONDITIONS:
            path = run_condition(
                store,
                index,
                condition,
                steps=arguments.steps,
                resume=arguments.resume,
            )
            print(f"[phase5-diagnostic] replica {index:02d} {condition.name}: {path}")
    analysis_path = analyze(
        store,
        replicas,
        steps=arguments.steps,
        resume=arguments.resume,
    )
    print(f"[phase5-diagnostic] analysis: {analysis_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "CONDITIONS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticCondition",
    "analyze",
    "run_condition",
]
