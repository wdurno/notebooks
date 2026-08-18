from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.plan3 import (
    Plan3Error,
    load_plan3_spec,
    parse_stage_names,
    preview_plan3,
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
    assert preview["new_runs"] == 75
    assert preview["reused_condition_runs"] == 35
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
    ] == 32 * 792
    assert rows["replay-unbounded"][
        "estimated_incremental_persistent_bytes"
    ] == 800 * 792


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
