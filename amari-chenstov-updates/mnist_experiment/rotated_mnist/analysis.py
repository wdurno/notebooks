"""Read-only lightweight summaries for rotated-MNIST notebooks."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from .audit_config import RotatedAuditConfig
from .artifacts import (
    AUDIT_REQUIRED_ARTIFACTS,
    LoadedRotatedRun,
    RotatedArtifactError,
    RotatedIncompleteRunError,
    load_completed_run,
)
from .phase3_artifacts import (
    LoadedRotatedPhase3Run,
    load_completed_phase3_run,
)
from .phase4_artifacts import (
    LoadedRotatedPhase4Run,
    load_completed_phase4_run,
)


@dataclasses.dataclass(frozen=True)
class LoadedRotatedAudit:
    path: Path
    config: RotatedAuditConfig
    manifest: dict[str, Any]
    initialization_metrics: dict[str, Any]
    zero_shot_metrics: tuple[dict[str, Any], ...]
    reference_metrics: tuple[dict[str, Any], ...]
    path_metrics: dict[str, Any]
    audit_summary: dict[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedArtifactError(f"could not read {path}: {exc}") from exc


def discover_completed_runs(root: str | Path) -> tuple[Path, ...]:
    root_path = Path(root)
    incomplete_root = root_path / ".incomplete"
    incomplete = (
        tuple(path for path in incomplete_root.iterdir() if path.is_dir())
        if incomplete_root.is_dir()
        else ()
    )
    if incomplete:
        names = ", ".join(sorted(path.name for path in incomplete))
        raise RotatedIncompleteRunError(
            f"rotated-MNIST run root contains incomplete runs: {names}"
        )
    if not root_path.exists():
        return ()
    paths = tuple(
        sorted(
            path
            for path in root_path.iterdir()
            if path.is_dir() and path.name != ".incomplete"
        )
    )
    for path in paths:
        if not (path / "COMPLETED").is_file():
            raise RotatedArtifactError(
                f"rotated-MNIST run lacks COMPLETED marker: {path}"
            )
    return paths


def load_completed_summaries(root: str | Path) -> tuple[LoadedRotatedRun, ...]:
    return tuple(load_completed_run(path) for path in discover_completed_runs(root))


def load_completed_phase3_runs(
    root: str | Path,
) -> tuple[LoadedRotatedPhase3Run, ...]:
    return tuple(
        load_completed_phase3_run(path) for path in discover_completed_runs(root)
    )


def load_completed_phase4_runs(
    root: str | Path,
) -> tuple[LoadedRotatedPhase4Run, ...]:
    return tuple(
        load_completed_phase4_run(path) for path in discover_completed_runs(root)
    )


def phase3_summary_rows(run: LoadedRotatedPhase3Run) -> list[dict[str, Any]]:
    summaries = run.run_summary["condition_summaries"]
    return [
        {
            "condition": condition,
            **summaries[condition],
        }
        for condition in run.config.conditions
    ]


def phase3_trajectory_rows(run: LoadedRotatedPhase3Run) -> list[dict[str, Any]]:
    return [
        dict(row)
        for condition in run.config.conditions
        for row in run.trajectory_metrics[condition]
    ]


def phase3_paired_contrast(run: LoadedRotatedPhase3Run) -> dict[str, float]:
    summaries = run.run_summary["condition_summaries"]
    current = summaries["current_only"]
    ewc = summaries["ewc_fixed_pi005"]
    return {
        "ewc_minus_current_environment_accuracy_auc": (
            float(ewc["environment_accuracy_auc"])
            - float(current["environment_accuracy_auc"])
        ),
        "ewc_minus_current_environment_nll_auc": (
            float(ewc["environment_nll_auc"])
            - float(current["environment_nll_auc"])
        ),
        "ewc_minus_current_final_environment_accuracy": (
            float(ewc["final_current_environment_accuracy"])
            - float(current["final_current_environment_accuracy"])
        ),
        "ewc_minus_current_final_upright_accuracy": (
            float(ewc["final_upright_environment_accuracy"])
            - float(current["final_upright_environment_accuracy"])
        ),
    }


def phase4_summary_rows(run: LoadedRotatedPhase4Run) -> list[dict[str, Any]]:
    summaries = run.run_summary["condition_summaries"]
    return [
        {"condition": condition, **summaries[condition]}
        for condition in run.config.conditions
    ]


def phase4_trajectory_rows(run: LoadedRotatedPhase4Run) -> list[dict[str, Any]]:
    return [
        dict(row)
        for condition in run.config.conditions
        for row in run.trajectory_metrics[condition]
    ]


def phase4_revisit_rows(run: LoadedRotatedPhase4Run) -> list[dict[str, Any]]:
    transitions = run.config.rotation.transitions_per_arrow
    output = []
    for condition in run.config.conditions:
        rows = run.trajectory_metrics[condition]
        first = rows[: 2 * transitions + 1]
        second = rows[3 * transitions : 5 * transitions + 1]
        if len(first) != len(second):
            raise RotatedArtifactError("Phase 4 ascent lengths differ")
        for first_row, second_row in zip(first, second, strict=True):
            if first_row["angle_degrees"] != second_row["angle_degrees"]:
                raise RotatedArtifactError("Phase 4 matched ascent angles differ")
            output.append(
                {
                    "condition": condition,
                    "angle_degrees": first_row["angle_degrees"],
                    "first_step": first_row["step"],
                    "second_step": second_row["step"],
                    "first_environment_accuracy": first_row[
                        "current_environment_accuracy"
                    ],
                    "second_environment_accuracy": second_row[
                        "current_environment_accuracy"
                    ],
                    "environment_accuracy_revisit_lift": (
                        second_row["current_environment_accuracy"]
                        - first_row["current_environment_accuracy"]
                    ),
                    "first_environment_nll": first_row["current_nll"],
                    "second_environment_nll": second_row["current_nll"],
                    "environment_nll_revisit_change": (
                        second_row["current_nll"] - first_row["current_nll"]
                    ),
                    "first_worst_class_recall": first_row[
                        "current_worst_class_recall"
                    ],
                    "second_worst_class_recall": second_row[
                        "current_worst_class_recall"
                    ],
                }
            )
    return output


def phase4_applied_contrasts(
    run: LoadedRotatedPhase4Run,
) -> list[dict[str, float | str]]:
    summaries = run.run_summary["condition_summaries"]
    pairs = (
        ("ewc_fixed_pi005", "current_only", "EWC - current only"),
        ("replay_b032", "current_only", "Replay B32 - current only"),
        (
            "hybrid_b032_fixed_pi005",
            "replay_b032",
            "Hybrid B32 - Replay B32",
        ),
        (
            "hybrid_b032_fixed_pi005",
            "replay_unbounded",
            "Hybrid B32 - unbounded replay",
        ),
    )
    fields = (
        "environment_accuracy_auc",
        "environment_nll_auc",
        "first_ascent_environment_accuracy_auc",
        "second_ascent_environment_accuracy_auc",
        "final_current_environment_accuracy",
        "final_upright_environment_accuracy",
    )
    output = []
    for treatment, comparator, label in pairs:
        output.append(
            {
                "contrast": label,
                **{
                    f"delta_{field}": (
                        float(summaries[treatment][field])
                        - float(summaries[comparator][field])
                    )
                    for field in fields
                },
            }
        )
    return output


def summary_rows(runs: tuple[LoadedRotatedRun, ...]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.config.run_id,
            "replica_id": run.config.replica_id,
            "condition": run.run_summary["condition"],
            "num_points": run.run_summary["num_points"],
            "samples_per_step": run.run_summary["samples_per_step"],
            "final_current_nine_ovr_accuracy": run.run_summary[
                "final_current_nine_ovr_accuracy"
            ],
            "final_current_environment_accuracy": run.run_summary[
                "final_current_environment_accuracy"
            ],
        }
        for run in runs
    ]


def discover_completed_audits(root: str | Path) -> tuple[Path, ...]:
    root_path = Path(root)
    incomplete_root = root_path / ".incomplete"
    incomplete = (
        tuple(path for path in incomplete_root.iterdir() if path.is_dir())
        if incomplete_root.is_dir()
        else ()
    )
    if incomplete:
        names = ", ".join(sorted(path.name for path in incomplete))
        raise RotatedIncompleteRunError(
            f"rotated-MNIST audit root contains incomplete runs: {names}"
        )
    if not root_path.exists():
        return ()
    paths = tuple(
        sorted(
            path
            for path in root_path.iterdir()
            if path.is_dir() and path.name != ".incomplete"
        )
    )
    for path in paths:
        if not (path / "COMPLETED").is_file():
            raise RotatedArtifactError(f"rotated audit lacks COMPLETED marker: {path}")
    return paths


def load_completed_audit(path: str | Path) -> LoadedRotatedAudit:
    audit_path = Path(path)
    if not (audit_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"rotated audit is incomplete: {audit_path}")
    for name in AUDIT_REQUIRED_ARTIFACTS:
        if not (audit_path / name).is_file():
            raise RotatedArtifactError(f"completed rotated audit is missing {name}")
    config = RotatedAuditConfig.from_mapping(_read_json(audit_path / "config.json"))
    manifest = _read_json(audit_path / "manifest.json")
    if manifest.get("status") != "completed":
        raise RotatedArtifactError("completed rotated audit manifest is not complete")
    if manifest.get("run_kind") != "phase2_learnability_audit":
        raise RotatedArtifactError("artifact is not a Phase 2 learnability audit")
    if manifest.get("config_hash") != config.config_hash:
        raise RotatedArtifactError("rotated audit config hash differs from manifest")
    return LoadedRotatedAudit(
        path=audit_path,
        config=config,
        manifest=manifest,
        initialization_metrics=_read_json(audit_path / "initialization_metrics.json"),
        zero_shot_metrics=tuple(_read_json(audit_path / "zero_shot_metrics.json")),
        reference_metrics=tuple(_read_json(audit_path / "reference_metrics.json")),
        path_metrics=_read_json(audit_path / "path_metrics.json"),
        audit_summary=_read_json(audit_path / "audit_summary.json"),
    )


def load_completed_audits(root: str | Path) -> tuple[LoadedRotatedAudit, ...]:
    return tuple(load_completed_audit(path) for path in discover_completed_audits(root))


def load_transform_examples(audit: LoadedRotatedAudit) -> dict[str, torch.Tensor]:
    value = torch.load(
        audit.path / "transform_examples.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = {"angles_degrees", "labels", "source_indices", "images"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RotatedArtifactError("rotated transform-example artifact is invalid")
    images = value["images"]
    if images.ndim != 5 or images.shape[1:] != (10, 1, 28, 28):
        raise RotatedArtifactError("rotated transform-example tensor has invalid shape")
    return value
