import json
from pathlib import Path

import numpy as np
import pytest

from mnist_experiment.run_hybrid import _validate_config
from src.initialization import replica_design_hash
from src.plan4_challenge import (
    CONDITIONS,
    _challenge_config,
    _controller_summary,
)
from src.plan4_floor import PREDICTIVE_FIELDS, _paired_predictive_summary


REPO_ROOT = Path(__file__).parents[2]
SOURCE_CONFIG = REPO_ROOT / "mnist_experiment" / "configs" / "plan4_phase3_fixed_smoke.json"


def _configs():
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source["estimator"]["low_rank"] = 8
    source["estimator"]["low_rank_grid"] = [0, 8]
    return [
        _challenge_config(
            source,
            source_archive="cache/source.pt",
            schedule="logistic-k64",
            condition=condition,
            device="cpu",
        )
        for condition in CONDITIONS
    ]


def test_challenge_conditions_are_paired_and_oracle_free() -> None:
    configs = _configs()

    assert {config.schema_version for config in configs} == {18}
    assert {config.artifact_schema_version for config in configs} == {10}
    assert {config.metric_schema_version for config in configs} == {14}
    assert {replica_design_hash(config) for config in configs} == {
        replica_design_hash(configs[0])
    }
    assert {config.controller.oracle_mode for config in configs} == {"none"}
    assert [config.controller.risk_metric for config in configs] == [
        "euclidean",
        "euclidean",
        "fisher",
    ]
    assert configs[-1].controller.trend_half_life_p == 0.05
    assert all(config.replay.capacity == 0 for config in configs)
    for config in configs:
        _validate_config(config)


def test_lower_floor_conditions_use_schema19_and_remain_paired() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source["estimator"]["low_rank"] = 8
    source["estimator"]["low_rank_grid"] = [0, 8]
    configs = [
        _challenge_config(
            source,
            source_archive="cache/source.pt",
            schedule="logistic-k128",
            condition=condition,
            device="cpu",
        )
        for condition in (
            "fixed-pi0025",
            "adaptive-fisher-pimin0025-h005",
        )
    ]

    assert {config.schema_version for config in configs} == {19}
    assert {config.controller.pi_min for config in configs} == {0.025}
    assert {replica_design_hash(config) for config in configs} == {
        replica_design_hash(configs[0])
    }
    for config in configs:
        _validate_config(config)


def test_realized_actuation_gate_uses_only_controller_risk_fields() -> None:
    rows = []
    for step in range(12):
        pi = 0.10 + 0.20 * step / 11
        signal = 10.0 + step
        rows.append(
            {
                "step": step,
                "p": step / 100,
                "controller_decision": {
                    "applied_pi": pi,
                    "raw_pi": pi,
                    "plugin_pi": pi,
                    "signal_energy": signal,
                    "old_covariance_risk": 0.1,
                    "new_covariance_risk": 1.0,
                    "cold_start_active": False,
                    "zero_information_fallback": False,
                },
                "classification": {"accuracy": 0.0},
            }
        )

    summary = _controller_summary(rows, np.ones(12, dtype=bool))

    assert summary["passes_actuation_gate"] is True
    assert summary["event_signal_transition_count"] == 12
    assert summary["event_applied_pi_range"] == pytest.approx(0.20)
    assert summary["event_signal_action_correlation"] > 0.99
    assert "classification" not in summary


def test_floor_predictive_comparison_uses_post_transition_event_outcomes() -> None:
    left = []
    right = []
    for step in range(3):
        left_metrics = {field: float(step + 1) for field in PREDICTIVE_FIELDS}
        right_metrics = {field: float(step) for field in PREDICTIVE_FIELDS}
        left.append({"classification": left_metrics})
        right.append({"classification": right_metrics})
    left[0]["classification"]["nine_precision"] = None
    right[0]["classification"]["nine_precision"] = None

    summary = _paired_predictive_summary(
        left,
        right,
        np.asarray([True, False]),
    )

    assert summary["environment_accuracy"]["full_mean_difference"] == pytest.approx(1.0)
    assert summary["environment_accuracy"]["event_outcome_count"] == 1
    assert summary["environment_accuracy"]["event_mean_difference"] == pytest.approx(1.0)
    assert summary["nine_precision"]["common_step_count"] == 2
