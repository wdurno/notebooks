import pytest

from src.plan2_analysis import (
    _accepted_bundles,
    _deduplicated_entries,
    combined_plan2_trajectory_rows,
    plan2_condition_semantics_rows,
    plan2_m128_anchor_rows,
    plan2_progress_rows,
)
from src.plan2 import Plan2Bundle
from src.results_analysis import AnalysisArtifactError


ANCHORS = {
    "no-ewc-pi100": ("adaptation-screen", "fixed-100-no-ewc"),
    "fixed-ewc-pi010": ("lfu-isolation", "fixed-010-no-lfu"),
    "adaptive-ewc-h010": ("lfu-isolation", "adaptive-h010-no-lfu"),
}


def _phase9_rows(replicas=(1, 2)):
    rows = []
    for replica in replicas:
        for condition, (profile, cell) in ANCHORS.items():
            for step, p_value in enumerate((0.0, 1.0)):
                rows.append(
                    {
                        "profile": profile,
                        "cell": cell,
                        "replica_index": replica,
                        "replica_id": f"replica-{replica:04d}",
                        "run_id": f"r{replica}:{condition}",
                        "method": "low_rank_diagonal_r8",
                        "step": step,
                        "p": p_value,
                        "samples_per_step": 128,
                    }
                )
    return rows


def test_m128_anchor_mapping_is_complete_and_redefines_pairing() -> None:
    rows = plan2_m128_anchor_rows(_phase9_rows(), replica_indices=(1, 2))

    assert len(rows) == 12
    assert {row["condition"] for row in rows} == set(ANCHORS)
    assert {row["samples_per_step"] for row in rows} == {128}
    assert {row["evidence_source"] for row in rows} == {"phase9_m128_anchor"}
    no_ewc = [row for row in rows if row["condition"] == "no-ewc-pi100"]
    assert {
        row["original_learning_process"] for row in no_ewc
    } == {"current batch only"}
    assert {
        row["auxiliary_fisher_process"] for row in no_ewc
    } == {"instantaneous empirical Fisher after initialization"}
    for replica in (1, 2):
        control = f"r{replica}:no-ewc-pi100"
        treatment_rows = [
            row
            for row in rows
            if row["replica_index"] == replica and row["kind"] == "treatment"
        ]
        assert {row["control_run_id"] for row in treatment_rows} == {control}


def test_condition_semantics_keep_learning_and_fisher_memory_distinct() -> None:
    rows = {
        row["condition"]: row for row in plan2_condition_semantics_rows()
    }

    assert rows["no-ewc-pi100"]["original_learning_process"] == (
        "current batch only"
    )
    assert rows["no-ewc-pi100"]["auxiliary_fisher_process"] == (
        "instantaneous empirical Fisher after initialization"
    )
    assert rows["fixed-ewc-pi010"]["auxiliary_fisher_process"] == (
        "recursive Fisher summary"
    )
    assert rows["adaptive-ewc-h010"]["auxiliary_fisher_process"] == (
        "recursive Fisher summary"
    )
    assert rows["adaptive-ewc-h005"]["auxiliary_fisher_process"] == (
        "recursive Fisher summary"
    )
    assert rows["adaptive-ewc-h040"]["original_learning_process"] == (
        "current batch plus EWC-compressed history"
    )


def test_m128_anchor_mapping_fails_clearly_when_condition_is_missing() -> None:
    incomplete = [
        row
        for row in _phase9_rows(replicas=(1,))
        if row["cell"] != "fixed-010-no-lfu"
    ]
    with pytest.raises(AnalysisArtifactError, match="anchors are unavailable"):
        plan2_m128_anchor_rows(incomplete, replica_indices=(1,))


def test_plan2_progress_reports_dependencies_separately_from_completion() -> None:
    inventory = [
        {
            "samples_per_step": 1,
            "condition": "no-ewc-pi100",
            "run_state": "completed",
            "derived_bundle_state": "completed",
            "reference_state": "completed",
        },
        {
            "samples_per_step": 1,
            "condition": "no-ewc-pi100",
            "run_state": "missing",
            "derived_bundle_state": "completed",
            "reference_state": "completed",
        },
    ]

    assert plan2_progress_rows(inventory) == [
        {
            "samples_per_step": 1,
            "condition": "no-ewc-pi100",
            "expected_replicas": 2,
            "completed_runs": 1,
            "dependencies_ready": True,
        }
    ]


def test_combined_plan2_rows_reject_duplicate_points() -> None:
    row = {
        "condition": "no-ewc-pi100",
        "cell": "no-ewc-pi100",
        "replica_index": 1,
        "samples_per_step": 1,
        "method": "low_rank_diagonal_r8",
        "step": 0,
    }
    with pytest.raises(AnalysisArtifactError, match="duplicate"):
        combined_plan2_trajectory_rows([row], [row])


def test_deduplication_allows_external_pairing_to_supersede_bundle_control() -> None:
    common = {
        "run_id": "adaptive-run",
        "condition": "adaptive-ewc-h005",
        "replica_index": 1,
        "samples_per_step": 8,
        "kind": "treatment",
    }
    internal = Plan2Bundle(
        path=None,
        manifest={
            "bundle_id": "internal",
            "spec_name": "plan2-low-data",
            "entries": [{**common, "control_run_id": "unused-control"}],
        },
    )
    extension = Plan2Bundle(
        path=None,
        manifest={
            "bundle_id": "extension",
            "spec_name": "plan2-low-data",
            "entries": [{**common, "control_run_id": None}],
        },
    )

    assert _deduplicated_entries((internal, extension)) == [
        {
            **common,
            "control_run_id": None,
            "bundle_ids": ["internal", "extension"],
            "spec_name": "plan2-low-data",
            "declared_control_run_ids": ["unused-control"],
            "external_control_pairing": True,
        }
    ]


def test_accepted_bundles_exclude_prelaunch_superseded_intention() -> None:
    superseded = Plan2Bundle(
        path=None,
        manifest={
            "bundle_id": "plan2-low-data__r0001-r0005__967e09afc995",
            "entries": [],
        },
    )
    accepted = Plan2Bundle(
        path=None,
        manifest={"bundle_id": "accepted", "entries": []},
    )

    assert _accepted_bundles((superseded, accepted)) == [accepted]
