from __future__ import annotations

import pytest

from src.plan3_analysis import (
    ALL_METRICS,
    _mean_interval,
    _phase7_interval,
    _json_sha256,
    _normalize_control_rows,
    _normalize_hybrid_rows,
    _normalize_replay_rows,
    _normalized_auc,
)


def _metric_values(value: float) -> dict[str, float]:
    return {metric: value for metric in ALL_METRICS}


def test_normalized_auc_uses_only_the_declared_pre_half_path() -> None:
    rows = [
        {"p": 0.0, "metric": 0.0},
        {"p": 0.25, "metric": 0.25},
        {"p": 0.49, "metric": 0.49},
        {"p": 0.5, "metric": 100.0},
    ]

    assert _normalized_auc(rows, "metric") == pytest.approx(0.245)


def test_control_and_replay_schemas_normalize_to_the_same_rows() -> None:
    control_rows = [
        {
            "step": step,
            "p": step / 99,
            **{f"before_{metric}": step / 100 for metric in ALL_METRICS},
        }
        for step in range(100)
    ]
    replay_rows = [
        {
            "step": step,
            "p": step / 99,
            "classification": _metric_values(step / 100),
        }
        for step in range(100)
    ]

    normalized_control = _normalize_control_rows(
        {"phase8_metric_schema_version": 8, "condition_steps": control_rows}
    )
    normalized_replay = _normalize_replay_rows(
        {"plan3_replay_metric_schema_version": 9, "condition_steps": replay_rows}
    )
    normalized_hybrid = _normalize_hybrid_rows(
        {"plan3_hybrid_metric_schema_version": 10, "condition_steps": replay_rows}
    )

    assert normalized_control == normalized_replay
    assert normalized_replay == normalized_hybrid

    normalized_lfu_hybrid = _normalize_hybrid_rows(
        {"plan3_hybrid_metric_schema_version": 11, "condition_steps": replay_rows}
    )
    assert normalized_lfu_hybrid == normalized_hybrid
    normalized_deployment_hybrid = _normalize_hybrid_rows(
        {"plan3_hybrid_metric_schema_version": 12, "condition_steps": replay_rows}
    )
    assert normalized_deployment_hybrid == normalized_hybrid


def test_mean_interval_uses_small_sample_student_t_radius() -> None:
    result = _mean_interval([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["standard_error"] == pytest.approx(1 / 2**0.5)
    assert result["ci95_high"] - result["mean"] == pytest.approx(
        2.776 / 2**0.5
    )


def test_phase7_interval_uses_t_radius_at_ten_replicas() -> None:
    result = _phase7_interval([float(value) for value in range(10)])

    assert result["count"] == 10
    assert result["mean"] == 4.5
    assert result["ci95_high"] - result["mean"] == pytest.approx(
        2.262 * result["standard_error"]
    )


def test_analysis_content_hash_changes_with_the_analysis_definition() -> None:
    original = {"comparisons": [["hybrid", "replay"]]}
    revised = {"comparisons": [["hybrid", "replay"], ["hybrid", "unbounded"]]}

    assert _json_sha256(original) != _json_sha256(revised)
    assert _json_sha256(original) == _json_sha256(original)
