from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.plan3 import (
    Plan3Error,
    _phase3_profiles,
    _phase4_new_profiles,
    _phase5_new_profiles,
    _phase6_profiles,
    load_plan3_spec,
    parse_stage_names,
    preview_plan3,
    validate_plan3_handoff_manifests,
)


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "mnist_experiment" / "plan3_profiles.json"


def test_plan3_preview_is_staged_and_reuses_frozen_replicas() -> None:
    spec = load_plan3_spec(SPEC_PATH)
    preview = preview_plan3(spec)

    assert preview["planning_only"] is True
    assert preview["replica_indices"] == [6, 7, 8, 9, 10]
    assert preview["source_anchor_bundle"].endswith("065c98b1061d")
    assert preview["confirmation_bundle"].endswith("7bb69d3a9424")
    assert preview["new_runs"] == 90
    assert preview["reused_condition_runs"] == 40
    assert preview["estimated_wall_hours"] > 0
    assert preview["estimated_storage_gib"] > 0
    assert [row["stage"] for row in preview["stages"]] == [
        "replay-screen",
        "memory-hybrid",
        "lfu-isolation",
        "deployment-frontier",
    ]


def test_replay_screen_counts_repeated_work_but_not_new_control_runs() -> None:
    spec = load_plan3_spec(SPEC_PATH)
    preview = preview_plan3(
        spec,
        stage_names=("replay-screen",),
        replica_indices=(6,),
    )
    rows = {row["condition"]: row for row in preview["conditions"]}

    assert preview["new_runs"] == 4
    assert preview["reused_condition_runs"] == 2
    assert rows["current-only"]["optimizer_observations_per_run"] == 792
    assert rows["replay-b008"]["optimizer_observations_per_run"] == 1576
    assert rows["replay-b032"]["optimizer_observations_per_run"] == 3880
    assert rows["replay-b128"]["optimizer_observations_per_run"] == 12376
    assert rows["replay-unbounded"]["optimizer_observations_per_run"] == 39600
    assert rows["current-only"]["new_runs"] == 0
    assert rows["replay-unbounded"]["new_runs"] == 1


def test_pure_replay_costs_use_the_phase1_production_pilot() -> None:
    spec = load_plan3_spec(SPEC_PATH)
    preview = preview_plan3(spec, stage_names=("replay-screen",))
    rows = {row["condition"]: row for row in preview["conditions"]}

    assert rows["replay-b032"]["estimated_seconds_per_new_run"] == pytest.approx(
        59.25900611499992
    )
    assert preview["estimated_wall_hours"] == pytest.approx(0.5163024774312721)
    assert preview["estimated_storage_gib"] == pytest.approx(
        20 * 1_700_000 / 2**30
    )


def test_plan3_memory_preview_separates_ewc_and_replay_payloads() -> None:
    spec = load_plan3_spec(SPEC_PATH)
    preview = preview_plan3(spec, stage_names=("replay-screen",))
    rows = {row["condition"]: row for row in preview["conditions"]}

    assert rows["current-only"]["estimated_incremental_persistent_bytes"] == 0
    assert rows["ewc-fixed005-no-lfu"][
        "estimated_incremental_persistent_bytes"
    ] == 512 * (8 + 2) * 4
    assert rows["replay-b032"][
        "estimated_incremental_persistent_bytes"
    ] == 24 + 32 * 800
    assert rows["replay-unbounded"][
        "estimated_incremental_persistent_bytes"
    ] == 24 + 800 * 800
    assert preview["derived_memory_contract"] == {
        "ewc_summary_bytes": 20480,
        "observation_payload_bytes": 800,
        "replay_fixed_metadata_bytes": 24,
        "memory_matched_replay_budget": 25,
    }


def test_phase3_smokes_cover_archive_transition_boundaries_on_both_devices() -> None:
    profiles = _phase3_profiles()

    assert len(profiles) == 8
    assert {(row["device"], row["capacity"]) for row in profiles} == {
        (device, capacity)
        for device in ("cpu", "cuda")
        for capacity in (0, 8, 25, "unbounded")
    }
    assert {
        row["capacity"]: row["expected_archived_online_events"]
        for row in profiles
        if row["device"] == "cpu"
    } == {0: 16, 8: 8, 25: 7, "unbounded": 0}


def test_phase4_adds_only_memory_matched_replay_and_three_hybrids() -> None:
    profiles = _phase4_new_profiles()

    assert [(row["runner"], row["capacity"]) for row in profiles] == [
        ("replay", 25),
        ("hybrid", 8),
        ("hybrid", 25),
        ("hybrid", 32),
    ]
    assert profiles[1]["comparison"] == "replay-b008"
    assert profiles[2]["comparison"] == "replay-memory-matched"
    assert profiles[3]["comparison"] == "replay-selected"


def test_phase5_isolates_lfu_without_crossing_replay_capacities() -> None:
    profiles = _phase5_new_profiles()

    assert [(row["runner"], row["method"]) for row in profiles] == [
        ("controller", "ac_only"),
        ("controller", "full_lfu"),
        ("hybrid", "full_lfu"),
    ]
    assert profiles[0]["comparison"] == "ewc-fixed005-no-lfu"
    assert profiles[1]["comparison"] == "ewc-fixed005-no-lfu"
    assert profiles[2]["comparison"] == "hybrid-selected-no-lfu"


def test_phase6_expands_the_explicit_oracle_free_deployment_frontier() -> None:
    profiles = _phase6_profiles()

    assert [row["condition"] for row in profiles] == [
        "deployment-current-only",
        "deployment-ewc-fixed005",
        "deployment-hybrid-b032-fixed005",
        "deployment-ewc-adaptive-h020",
        "deployment-hybrid-b032-adaptive-h020",
        "deployment-replay-b032",
        "deployment-replay-unbounded",
    ]
    assert [(row["runner"], row["capacity"]) for row in profiles] == [
        ("replay", 0),
        ("hybrid", 0),
        ("hybrid", 32),
        ("hybrid", 0),
        ("hybrid", 32),
        ("replay", 32),
        ("replay", "unbounded"),
    ]


def test_plan3_stage_selection_rejects_unknown_or_duplicate_names() -> None:
    spec = load_plan3_spec(SPEC_PATH)

    assert parse_stage_names("replay-screen,lfu-isolation", spec) == (
        "replay-screen",
        "lfu-isolation",
    )
    with pytest.raises(Plan3Error, match="unknown"):
        parse_stage_names("not-a-stage", spec)
    with pytest.raises(Plan3Error, match="unique"):
        parse_stage_names("replay-screen,replay-screen", spec)


def test_plan3_spec_rejects_unknown_named_comparison(tmp_path: Path) -> None:
    mapping = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    mapping["stages"][0]["conditions"][1]["comparison"] = "missing-control"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(Plan3Error, match="unknown comparisons"):
        load_plan3_spec(path)


def _handoff_manifests():
    spec = load_plan3_spec(SPEC_PATH)
    anchors = []
    confirmation = []
    conditions = (
        "no-ewc-pi100",
        "fixed-ewc-pi005",
        "adaptive-ewc-h020",
        "fixed-ewc-pi010",
    )
    for replica in spec.replica_indices:
        master_id = f"master-{replica}"
        derived_id = f"derived-{replica}"
        anchor_run = f"anchor-{replica}"
        anchors.append(
            {
                "replica_index": replica,
                "is_oracle_anchor": True,
                "replica_bundle_id": master_id,
                "run_id": anchor_run,
                "config_hash": f"anchor-hash-{replica}",
            }
        )
        control_run = f"run-{replica}-no-ewc-pi100"
        for condition in conditions:
            confirmation.append(
                {
                    "replica_index": replica,
                    "condition": condition,
                    "samples_per_step": 8,
                    "master_replica_bundle_id": master_id,
                    "replica_bundle_id": derived_id,
                    "reference_source_run_id": anchor_run,
                    "run_id": f"run-{replica}-{condition}",
                    "config_hash": f"hash-{replica}-{condition}",
                    "control_run_id": (
                        None if condition == "no-ewc-pi100" else control_run
                    ),
                }
            )
    source = {
        "bundle_id": spec.source_anchor_bundle,
        "selection": {"replica_indices": list(spec.replica_indices)},
        "entries": anchors,
    }
    confirmed = {
        "bundle_id": spec.confirmation_bundle,
        "selection": {
            "replica_indices": list(spec.replica_indices),
            "samples_per_step": [8],
        },
        "entries": confirmation,
    }
    return spec, source, confirmed


def test_plan3_handoff_freezes_paired_replica_identities() -> None:
    spec, source, confirmation = _handoff_manifests()

    rows = validate_plan3_handoff_manifests(spec, source, confirmation)

    assert len(rows) == 5
    assert rows[0]["master_replica_bundle_id"] == "master-6"
    assert rows[0]["derived_replica_bundle_id"] == "derived-6"
    assert rows[0]["control_run_id"] == "run-6-no-ewc-pi100"
    assert rows[0]["fixed_ewc_run_id"] == "run-6-fixed-ewc-pi005"


def test_plan3_handoff_rejects_reference_mismatch() -> None:
    spec, source, confirmation = _handoff_manifests()
    confirmation["entries"][0]["reference_source_run_id"] = "wrong-anchor"

    with pytest.raises(Plan3Error, match="reference path mismatch"):
        validate_plan3_handoff_manifests(spec, source, confirmation)
