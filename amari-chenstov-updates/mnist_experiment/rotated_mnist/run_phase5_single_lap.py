"""Run the paired slow-trend single-lap EDR retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .phase5_single_lap_artifacts import (
    SINGLE_LAP_REQUIRED_ARTIFACTS,
    RotatedSingleLapRunStore,
)
from .phase5_single_lap_config import (
    SINGLE_LAP_EDR_CONDITION,
    SINGLE_LAP_FIXED_PI,
    SINGLE_LAP_SCHEDULES,
    RotatedSlowSingleLapConfig,
    load_single_lap_config,
)
from .run_phase5_double_lap import run_double_lap


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("cache/mnist_experiment/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/phase5/single_lap"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def classify_single_lap_retry(
    summaries: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    cold_only = any(
        summaries[schedule][SINGLE_LAP_EDR_CONDITION].get(
            "action_statistics_include_cold_start", False
        )
        for schedule in SINGLE_LAP_SCHEDULES
    )
    comparisons = {}
    for schedule in SINGLE_LAP_SCHEDULES:
        values = summaries[schedule]
        best_fixed = max(
            SINGLE_LAP_FIXED_PI,
            key=lambda name: values[name]["environment_accuracy_auc"],
        )
        edr = values[SINGLE_LAP_EDR_CONDITION]
        fixed05 = values["fixed_pi005"]
        comparisons[schedule] = {
            "best_fixed_condition": best_fixed,
            "best_fixed_environment_accuracy_auc": values[best_fixed][
                "environment_accuracy_auc"
            ],
            "edr_environment_accuracy_auc": edr["environment_accuracy_auc"],
            "edr_minus_fixed005_environment_accuracy_auc": (
                edr["environment_accuracy_auc"]
                - fixed05["environment_accuracy_auc"]
            ),
            "edr_gap_to_best_fixed_environment_accuracy_auc": (
                edr["environment_accuracy_auc"]
                - values[best_fixed]["environment_accuracy_auc"]
            ),
            "edr_minus_fixed005_environment_nll_auc": (
                edr["environment_nll_auc"] - fixed05["environment_nll_auc"]
            ),
            "edr_minus_fixed005_final_upright_accuracy": (
                edr["final_upright_environment_accuracy"]
                - fixed05["final_upright_environment_accuracy"]
            ),
            "edr_minus_fixed005_final_worst_class_recall": (
                edr["final_worst_class_recall"]
                - fixed05["final_worst_class_recall"]
            ),
        }
    no_secondary_regression = all(
        comparisons[schedule]["edr_minus_fixed005_environment_nll_auc"] <= 0.10
        and comparisons[schedule][
            "edr_minus_fixed005_final_upright_accuracy"
        ]
        >= -0.02
        and comparisons[schedule][
            "edr_minus_fixed005_final_worst_class_recall"
        ]
        >= -0.02
        for schedule in SINGLE_LAP_SCHEDULES
    )
    stable_actions = all(
        summaries[schedule][SINGLE_LAP_EDR_CONDITION]["action_min"] >= 0.01
        and summaries[schedule][SINGLE_LAP_EDR_CONDITION]["action_max"] <= 0.10
        for schedule in SINGLE_LAP_SCHEDULES
    )
    responsive_actions = any(
        summaries[schedule][SINGLE_LAP_EDR_CONDITION]["action_span"] >= 0.005
        for schedule in SINGLE_LAP_SCHEDULES
    )
    automatic = (
        stable_actions
        and no_secondary_regression
        and all(
            comparisons[schedule][
                "edr_gap_to_best_fixed_environment_accuracy_auc"
            ]
            >= -0.01
            for schedule in SINGLE_LAP_SCHEDULES
        )
    )
    prospective = (
        stable_actions
        and no_secondary_regression
        and all(
            comparisons[schedule][
                "edr_minus_fixed005_environment_accuracy_auc"
            ]
            >= -0.01
            for schedule in SINGLE_LAP_SCHEDULES
        )
        and any(
            comparisons[schedule][
                "edr_minus_fixed005_environment_accuracy_auc"
            ]
            > 0.0
            for schedule in SINGLE_LAP_SCHEDULES
        )
    )
    diagnostic = (
        stable_actions
        and responsive_actions
        and no_secondary_regression
        and all(
            comparisons[schedule][
                "edr_minus_fixed005_environment_accuracy_auc"
            ]
            >= -0.03
            for schedule in SINGLE_LAP_SCHEDULES
        )
    )
    classification = (
        "integration_only"
        if cold_only
        else "automatic_selection_value"
        if automatic
        else "prospective_value"
        if prospective
        else "diagnostic_only"
        if diagnostic
        else "stop"
    )
    return {
        "classification": classification,
        "comparisons": comparisons,
        "stable_action_bracket": stable_actions,
        "responsive_actions": responsive_actions,
        "no_material_secondary_regression": no_secondary_regression,
        "development_replica_only": True,
        "action_statistics_include_cold_start": cold_only,
    }


def run_single_lap(
    config: RotatedSlowSingleLapConfig,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    return run_double_lap(
        config,
        data_root=data_root,
        output_root=output_root,
        repo_root=repo_root,
        download=download,
        resume=resume,
        run_store=RotatedSingleLapRunStore(output_root),
        required_artifacts=SINGLE_LAP_REQUIRED_ARTIFACTS,
        classifier=classify_single_lap_retry,
        progress_description="Phase 5 slow single lap",
        seed_namespace="plan5_single_lap",
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_single_lap_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_single_lap(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=repo_root,
        download=arguments.download,
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
