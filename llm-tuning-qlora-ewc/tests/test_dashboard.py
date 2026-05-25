import json

import pandas as pd
import pytest

from src.run_case1 import configured_ewc_combinations
from src.dashboard import (
    available_columns,
    duplicate_combo_summary,
    headline_scorecard,
    headline_grid_row,
    latest_metrics_by_combo,
    load_metrics,
    load_predictions,
    metadata_by_id,
    prediction_path,
    slice_retention,
)
from src.data.io import JsonlExample, write_jsonl


def test_metrics_loader_marks_latest_repeated_combo(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {"ewc_n0": 64, "ewc_rank": 4, "ewc_lambda": 1.0, "target_em": 0.5},
        {"ewc_n0": 16, "ewc_rank": 0, "ewc_lambda": 1.0, "target_em": 1.0},
        {"ewc_n0": 64, "ewc_rank": 4, "ewc_lambda": 1.0, "target_em": 0.875},
    ]
    write_jsonl(metrics_path, rows)

    df = load_metrics(metrics_path)
    latest = latest_metrics_by_combo(df)
    duplicates = duplicate_combo_summary(df)

    assert list(df["run_index"]) == [0, 1, 2]
    assert not bool(df.loc[0, "is_latest_for_combo"])
    assert bool(df.loc[2, "is_latest_for_combo"])
    assert len(latest) == 2
    assert latest.loc[latest["ewc_n0"] == 64, "target_em"].item() == 0.875
    assert duplicates[["ewc_n0", "ewc_rank", "ewc_lambda", "run_count"]].to_dict("records") == [
        {"ewc_n0": 64, "ewc_rank": 4, "ewc_lambda": 1.0, "run_count": 2}
    ]


def test_load_predictions_falls_back_to_generated_metadata(tmp_path):
    run_dir = tmp_path / "outputs"
    run_dir.mkdir()
    path = prediction_path(run_dir, "task_a_after", 8, 2, 1.0)
    write_jsonl(
        path,
        [
            {
                "id": "example_1",
                "target": "ROUTE_BLUE_01",
                "parsed_prediction": "ROUTE_BLUE_01",
                "exact_match": 1.0,
            }
        ],
    )
    examples = [
        JsonlExample(
            id="example_1",
            task_family="rule_transform",
            task_id="task_a",
            split="eval_heldout",
            prompt="prompt",
            target="ROUTE_BLUE_01",
            answer_key="ROUTE_BLUE_01",
            metadata={
                "rule_id": "route_AB_HIGH_SOUTH_STANDARD",
                "relation_to_next_task": "direct_conflict",
                "frequency_bucket": "common",
                "composition_type": "seen",
            },
        )
    ]

    df = load_predictions(run_dir, "task_a_after", 8, 2, 1.0, metadata_by_id(examples))

    assert df.loc[0, "rule_id"] == "route_AB_HIGH_SOUTH_STANDARD"
    assert df.loc[0, "relation_to_next_task"] == "direct_conflict"
    assert df.loc[0, "legacy_conflict_group"] == "conflicting"


def test_slice_retention_reports_before_after_delta_and_counts():
    before = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 1.0},
        ]
    )
    after = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "shared", "exact_match": 0.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 0.0},
        ]
    )

    sliced = slice_retention(before, after, "relation_to_next_task")
    shared = sliced[sliced["relation_to_next_task"] == "shared"].iloc[0]
    near = sliced[sliced["relation_to_next_task"] == "near_conflict"].iloc[0]

    assert shared["before_em"] == 1.0
    assert shared["after_em"] == 0.5
    assert shared["forgetting_delta"] == 0.5
    assert shared["n"] == 2
    assert near["after_em"] == 0.0


def test_missing_optional_columns_are_safe():
    df = pd.DataFrame([{"target_em": 1.0}])

    assert available_columns(df, ["ewc_n0", "target_em", "task_a_after_em"]) == ["target_em"]
    assert slice_retention(df, df, "relation_to_next_task").empty


def test_headline_scorecard_uses_rich_prediction_slices():
    metric_row = pd.Series({"eval_original_em": 0.25, "run_wall_seconds": 120.0, "peak_vram_mb": 7000.0})
    task_a_before = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 1.0},
            {"relation_to_next_task": "rare_rule", "exact_match": 1.0},
        ]
    )
    task_a_after = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 0.0},
            {"relation_to_next_task": "rare_rule", "exact_match": 1.0},
        ]
    )
    task_b_after = pd.DataFrame(
        [
            {"relation_to_next_task": "direct_conflict", "composition_type": "seen", "exact_match": 1.0},
            {"relation_to_next_task": "shared", "composition_type": "seen", "exact_match": 1.0},
            {"relation_to_next_task": "heldout_composition", "composition_type": "heldout_triple", "exact_match": 0.0},
        ]
    )

    scorecard = headline_scorecard(metric_row, task_a_before, task_a_after, task_b_after)

    scores = dict(zip(scorecard["metric"], scorecard["score"], strict=True))
    counts = dict(zip(scorecard["metric"], scorecard["n"], strict=True))
    assert scores["Target learning"] == 1.0
    assert counts["Target learning"] == 2
    assert scores["Target heldout composition"] == 0.0
    assert scores["Protected retention"] == 2 / 3
    assert scorecard.loc[scorecard["metric"] == "Protected retention", "delta"].item() == pytest.approx(1 / 3)
    assert scores["Expected overwrite"] == 1.0
    assert scores["Base retention"] == 0.25


def test_headline_grid_row_is_one_row_per_combo():
    metric_row = pd.Series(
        {
            "run_index": 3,
            "ewc_n0": 64,
            "ewc_rank": 4,
            "ewc_lambda": 1.0,
            "eval_original_em": 0.25,
            "run_wall_seconds": 120.0,
            "peak_vram_mb": 7000.0,
        }
    )
    task_a_before = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 1.0},
        ]
    )
    task_a_after = pd.DataFrame(
        [
            {"relation_to_next_task": "shared", "exact_match": 1.0},
            {"relation_to_next_task": "near_conflict", "exact_match": 0.0},
        ]
    )
    task_b_after = pd.DataFrame(
        [
            {"relation_to_next_task": "direct_conflict", "composition_type": "seen", "exact_match": 1.0},
            {"relation_to_next_task": "heldout_composition", "composition_type": "heldout_triple", "exact_match": 0.0},
        ]
    )

    row = headline_grid_row(metric_row, task_a_before, task_a_after, task_b_after)

    assert row["run_index"] == 3
    assert row["ewc_n0"] == 64
    assert row["target_seen_em"] == 1.0
    assert row["target_heldout_composition_em"] == 0.0
    assert row["protected_retention_em"] == 0.5
    assert row["protected_retention_drop"] == 0.5
    assert row["near_conflict_retention_em"] == 0.0
    assert row["expected_overwrite_em"] == 1.0


def test_configured_ewc_combinations_prefers_explicit_grid():
    config = {
        "ewc": {
            "n0_values": [0, 16],
            "rank_values": [0, 2],
            "lambda_values": [0.0, 1.0],
            "combinations": [
                {"n0": 0, "rank": 0, "lambda": 0.0},
                {"n0": 16, "rank": 2, "lambda": 1.0},
            ],
        }
    }

    assert configured_ewc_combinations(config) == [(0, 0, 0.0), (16, 2, 1.0)]
