"""Artifact-only inventory and trajectory adapters for Plan 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .plan2 import Plan2Bundle, plan2_status_rows
from .results_analysis import (
    AnalysisArtifactError,
    load_phase8_controller_run,
    phase8_controller_rows,
)
from .classification_backfill import default_classification_backfill_root

PLAN2_PROFILE = "plan2-low-data"
PLAN2_SUPERSEDED_BUNDLE_IDS = frozenset(
    {
        # Replaced before execution by a treatment-only extension so completed
        # schema-7 controls could be reused without mutable regeneration.
        "plan2-low-data__r0001-r0005__967e09afc995",
    }
)
PLAN2_M128_ANCHORS = {
    "no-ewc-pi100": ("adaptation-screen", "fixed-100-no-ewc", "control"),
    "fixed-ewc-pi010": (
        "lfu-isolation",
        "fixed-010-no-lfu",
        "treatment",
    ),
    "adaptive-ewc-h010": (
        "lfu-isolation",
        "adaptive-h010-no-lfu",
        "treatment",
    ),
}
PLAN2_CONDITION_SEMANTICS = {
    "no-ewc-pi100": {
        "original_learning_process": "current batch only",
        "auxiliary_fisher_process": (
            "instantaneous empirical Fisher after initialization"
        ),
    },
    "fixed-ewc-pi010": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "fixed-ewc-pi005": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h010": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h005": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h020": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h040": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h020-pimin001": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
    "adaptive-ewc-h020-pimin010": {
        "original_learning_process": "current batch plus EWC-compressed history",
        "auxiliary_fisher_process": "recursive Fisher summary",
    },
}


def _condition_semantics(condition: str) -> dict[str, str]:
    try:
        return dict(PLAN2_CONDITION_SEMANTICS[condition])
    except KeyError as error:
        raise AnalysisArtifactError(
            f"Plan 2 condition has no process-semantics declaration: {condition}"
        ) from error


def plan2_condition_semantics_rows() -> list[dict[str, str]]:
    """Describe learner and auxiliary-Fisher memory without editing artifacts."""

    return [
        {"condition": condition, **_condition_semantics(condition)}
        for condition in PLAN2_CONDITION_SEMANTICS
    ]


def _accepted_bundles(bundles: Sequence[Plan2Bundle]) -> list[Plan2Bundle]:
    """Exclude immutable intentions explicitly superseded before execution."""

    return [
        bundle
        for bundle in bundles
        if bundle.bundle_id not in PLAN2_SUPERSEDED_BUNDLE_IDS
    ]


def _deduplicated_entries(bundles: Sequence[Plan2Bundle]) -> list[dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        for source in bundle.entries:
            entry = dict(source)
            run_id = str(entry["run_id"])
            existing = by_run_id.get(run_id)
            if existing is None:
                entry["bundle_ids"] = [bundle.bundle_id]
                entry["spec_name"] = bundle.manifest["spec_name"]
                entry["declared_control_run_ids"] = (
                    []
                    if entry["control_run_id"] is None
                    else [entry["control_run_id"]]
                )
                entry["external_control_pairing"] = (
                    entry["control_run_id"] is None
                )
                by_run_id[run_id] = entry
                continue
            comparable = {
                key: value
                for key, value in existing.items()
                if key
                not in {
                    "bundle_ids",
                    "spec_name",
                    "control_run_id",
                    "declared_control_run_ids",
                    "external_control_pairing",
                }
            }
            source_comparable = {
                key: value for key, value in entry.items() if key != "control_run_id"
            }
            if comparable != source_comparable:
                raise AnalysisArtifactError(
                    f"Plan 2 run {run_id} has conflicting bundle metadata"
                )
            if existing["spec_name"] != bundle.manifest["spec_name"]:
                raise AnalysisArtifactError(
                    f"Plan 2 run {run_id} has conflicting specification names"
                )
            existing["bundle_ids"].append(bundle.bundle_id)
            control_run_id = entry["control_run_id"]
            if (
                control_run_id is not None
                and control_run_id not in existing["declared_control_run_ids"]
            ):
                existing["declared_control_run_ids"].append(control_run_id)
            existing["external_control_pairing"] = bool(
                existing["external_control_pairing"]
                or control_run_id is None
            )
            existing["control_run_id"] = (
                None
                if existing["external_control_pairing"]
                or len(existing["declared_control_run_ids"]) != 1
                else existing["declared_control_run_ids"][0]
            )
    return sorted(
        by_run_id.values(),
        key=lambda row: (
            row["replica_index"],
            row["samples_per_step"],
            row["kind"] != "control",
            row["condition"],
        ),
    )


def plan2_inventory_rows(
    bundles: Sequence[Plan2Bundle], repo_root: str | Path
) -> list[dict[str, Any]]:
    """Return deduplicated run intentions and dependency states."""

    accepted_bundles = _accepted_bundles(bundles)
    status_by_run: dict[str, dict[str, Any]] = {}
    for bundle in accepted_bundles:
        for status in plan2_status_rows(bundle, repo_root):
            run_id = str(status["run_id"])
            existing = status_by_run.get(run_id)
            state = (
                status["run_state"],
                status["derived_bundle_state"],
                status["reference_state"],
            )
            if existing is not None and state != existing["state"]:
                raise AnalysisArtifactError(
                    f"Plan 2 run {run_id} has conflicting artifact states"
                )
            status_by_run[run_id] = {"status": status, "state": state}

    rows = []
    for entry in _deduplicated_entries(accepted_bundles):
        status = status_by_run[entry["run_id"]]["status"]
        rows.append(
            {
                **entry,
                **_condition_semantics(str(entry["condition"])),
                "bundle_count": len(entry["bundle_ids"]),
                "run_state": status["run_state"],
                "derived_bundle_state": status["derived_bundle_state"],
                "reference_state": status["reference_state"],
                "run_path": status["run_path"],
                "config_path": status["config_path"],
            }
        )
    return rows


def plan2_progress_rows(
    inventory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize launch readiness and completion by condition and m."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in inventory_rows:
        grouped[(int(row["samples_per_step"]), str(row["condition"]))].append(row)
    results = []
    for (sample_size, condition), rows in grouped.items():
        results.append(
            {
                "samples_per_step": sample_size,
                "condition": condition,
                "expected_replicas": len(rows),
                "completed_runs": sum(
                    row["run_state"] == "completed" for row in rows
                ),
                "dependencies_ready": all(
                    row["derived_bundle_state"] == "completed"
                    and row["reference_state"] == "completed"
                    for row in rows
                ),
            }
        )
    return sorted(results, key=lambda row: (row["samples_per_step"], row["condition"]))


def plan2_trajectory_rows(
    bundles: Sequence[Plan2Bundle], repo_root: str | Path
) -> list[dict[str, Any]]:
    """Load completed Plan 2 scalar trajectories without model artifacts."""

    rows = []
    for entry in plan2_inventory_rows(bundles, repo_root):
        if entry["run_state"] != "completed":
            continue
        run = load_phase8_controller_run(entry["run_path"])
        for source in phase8_controller_rows(
            [run],
            classification_backfill_root=default_classification_backfill_root(
                repo_root
            ),
        ):
            rows.append(
                {
                    **source,
                    **_condition_semantics(str(entry["condition"])),
                    "profile": PLAN2_PROFILE,
                    "cell": entry["condition"],
                    "condition": entry["condition"],
                    "kind": entry["kind"],
                    "replica_index": entry["replica_index"],
                    "replica_seed": entry["replica_seed"],
                    "replica_bundle_id": entry["replica_bundle_id"],
                    "control_run_id": entry["control_run_id"],
                    "bundle_ids": entry["bundle_ids"],
                    "evidence_source": "plan2_nested_stream",
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["samples_per_step"],
            row["cell"],
            row["replica_index"],
            row["method"],
            row["step"],
        ),
    )


def plan2_m128_anchor_rows(
    phase9_rows: Sequence[Mapping[str, Any]],
    *,
    replica_indices: Sequence[int] = (1, 2, 3),
) -> list[dict[str, Any]]:
    """Relabel the three compatible Phase 9 conditions as the m=128 anchor."""

    requested = tuple(replica_indices)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("replica_indices must be unique and nonempty")
    selected: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    reverse = {
        (profile, cell): condition
        for condition, (profile, cell, _) in PLAN2_M128_ANCHORS.items()
    }
    for row in phase9_rows:
        condition = reverse.get((row.get("profile"), row.get("cell")))
        replica = int(row["replica_index"])
        if condition is not None and replica in requested:
            if int(row["samples_per_step"]) != 128:
                raise AnalysisArtifactError(
                    f"Phase 9 anchor {condition} does not use m=128"
                )
            selected[(replica, condition)].append(row)

    missing = [
        f"replica-{replica:04d}:{condition}"
        for replica in requested
        for condition in PLAN2_M128_ANCHORS
        if not selected[(replica, condition)]
    ]
    if missing:
        raise AnalysisArtifactError(
            "completed Plan 2 m=128 anchors are unavailable: " + ", ".join(missing)
        )

    control_runs = {
        replica: str(selected[(replica, "no-ewc-pi100")][0]["run_id"])
        for replica in requested
    }
    results = []
    for replica in requested:
        expected_grid = None
        for condition, (_, _, kind) in PLAN2_M128_ANCHORS.items():
            condition_rows = selected[(replica, condition)]
            p_grid = tuple(float(row["p"]) for row in condition_rows)
            if expected_grid is None:
                expected_grid = p_grid
            elif p_grid != expected_grid:
                raise AnalysisArtifactError(
                    f"unaligned m=128 p grid for replica {replica}"
                )
            for source in condition_rows:
                results.append(
                    {
                        **source,
                        **_condition_semantics(condition),
                        "profile": PLAN2_PROFILE,
                        "cell": condition,
                        "condition": condition,
                        "kind": kind,
                        "control_run_id": (
                            None if kind == "control" else control_runs[replica]
                        ),
                        "evidence_source": "phase9_m128_anchor",
                    }
                )
    return sorted(
        results,
        key=lambda row: (
            row["samples_per_step"],
            row["cell"],
            row["replica_index"],
            row["method"],
            row["step"],
        ),
    )


def combined_plan2_trajectory_rows(
    plan2_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine nested-stream evidence and the compatible m=128 anchor."""

    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in (*anchor_rows, *plan2_rows):
        row = dict(source)
        key = (
            row["condition"],
            row["replica_index"],
            row["samples_per_step"],
            row["method"],
            row["step"],
        )
        if key in indexed:
            raise AnalysisArtifactError(f"duplicate Plan 2 trajectory point: {key}")
        indexed[key] = row
    return sorted(
        indexed.values(),
        key=lambda row: (
            row["samples_per_step"],
            row["cell"],
            row["replica_index"],
            row["method"],
            row["step"],
        ),
    )
