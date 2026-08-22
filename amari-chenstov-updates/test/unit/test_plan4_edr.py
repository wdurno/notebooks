import json
from pathlib import Path

import numpy as np
import pytest

from mnist_experiment.run_hybrid import _validate_config
from src.initialization import replica_design_hash
from src.plan4_challenge import _challenge_config
from src.plan4_edr import (
    _edr_config,
    _edr_controller_summary,
    filter_risk_coefficients,
)
from src.plan4_edr_discovery import (
    _discovery_config,
    _discovery_controller_summary,
)


REPO_ROOT = Path(__file__).parents[2]
SOURCE_CONFIG = (
    REPO_ROOT / "mnist_experiment" / "configs" / "plan4_phase3_fixed_smoke.json"
)


def _raw_floor_config():
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source["estimator"]["low_rank"] = 8
    source["estimator"]["low_rank_grid"] = [0, 8]
    return _challenge_config(
        source,
        source_archive="cache/source.pt",
        schedule="logistic-k64",
        condition="adaptive-fisher-pimin0025-h005",
        device="cpu",
    )


def test_edr_schema_is_explicit_and_preserves_replica_pairing() -> None:
    source = _raw_floor_config()
    config = _edr_config(source.to_mapping(), schedule="logistic-k64")

    assert config.schema_version == 20
    assert config.artifact_schema_version == 11
    assert config.metric_schema_version == 15
    assert config.controller.policy == "discounted_risk"
    assert config.controller.fixed_pi == 0.025
    assert config.controller.pi_min == 0.01
    assert config.controller.action_half_life_steps == 4.0
    assert replica_design_hash(config) == replica_design_hash(source)
    assert config.to_mapping()["controller"]["action_half_life_steps"] == 4.0
    _validate_config(config)


def test_edr_discovery_changes_only_the_named_cold_start_treatment() -> None:
    source = _edr_config(_raw_floor_config().to_mapping(), schedule="linear")
    discovery = _discovery_config(source.to_mapping())

    assert discovery.controller.fixed_pi == 0.05
    assert discovery.controller.pi_min == 0.01
    assert discovery.controller.action_half_life_steps == 4.0
    assert discovery.experiment == "mnist_plan4-phase6-edr-discovery_linear"
    assert replica_design_hash(discovery) == replica_design_hash(source)
    _validate_config(discovery)


def test_edr_filter_clips_only_the_final_action() -> None:
    rows = []
    for step, old_risk in enumerate((0.0, 0.01, 0.02, 0.04)):
        rows.append(
            {
                "step": step,
                "p": 0.01 * step,
                "controller_decision": {
                    "signal_energy": old_risk,
                    "old_covariance_risk": 0.0,
                    "new_covariance_risk": 1.0,
                    "applied_pi": 0.025,
                    "cold_start_active": step == 0,
                },
            }
        )
    event = np.asarray([False, True, True, True])

    lower = filter_risk_coefficients(
        rows,
        event,
        half_life_steps=4.0,
        pi_min=0.01,
    )
    upper = filter_risk_coefficients(
        rows,
        event,
        half_life_steps=4.0,
        pi_min=0.03,
    )

    assert [row["old_risk_moment"] for row in lower["trajectory"]] == pytest.approx(
        [row["old_risk_moment"] for row in upper["trajectory"]]
    )
    assert [row["new_risk_moment"] for row in lower["trajectory"]] == pytest.approx(
        [row["new_risk_moment"] for row in upper["trajectory"]]
    )
    assert lower["trajectory"][-1]["unclipped_pi"] < 0.03
    assert upper["trajectory"][-1]["applied_pi"] == pytest.approx(0.03)


def test_edr_controller_summary_reports_event_and_floor_diagnostics() -> None:
    rows = []
    for step, applied in enumerate((0.025, 0.025, 0.04)):
        rows.append(
            {
                "step": step,
                "p": 0.01 * step,
                "controller_decision": {
                    "applied_pi": applied,
                    "plugin_pi": 0.05 + step * 0.01,
                    "edr_unclipped_pi": 0.02 + step * 0.01,
                    "edr_instantaneous_old_risk": 0.1 + step,
                    "edr_instantaneous_new_risk": 1.0,
                    "edr_old_risk_moment": 0.2 + step,
                    "edr_new_risk_moment": 0.5,
                    "cold_start_active": step == 0,
                    "lower_bound_active": step == 1,
                    "edr_zero_denominator_fallback": False,
                },
            }
        )

    summary = _edr_controller_summary(
        rows, np.asarray([False, True, True], dtype=bool)
    )

    assert summary["event_applied_pi_mean"] == pytest.approx(0.0325)
    assert summary["post_cold_lower_bound_fraction"] == pytest.approx(0.5)
    assert summary["cold_start_transition_count"] == 1
    assert summary["zero_denominator_fallback_count"] == 0
    assert len(summary["trajectory"]) == 3


def test_discovery_summary_requires_sustained_unclipped_tail() -> None:
    rows = []
    for step in range(12):
        applied = 0.05 if step < 2 else 0.04 - 0.001 * step
        rows.append(
            {
                "step": step,
                "p": 0.01 * step,
                "controller_decision": {
                    "applied_pi": applied,
                    "plugin_pi": applied,
                    "edr_unclipped_pi": applied,
                    "edr_instantaneous_old_risk": 0.1,
                    "edr_instantaneous_new_risk": 1.0,
                    "edr_old_risk_moment": 0.1,
                    "edr_new_risk_moment": 1.0,
                    "cold_start_active": step < 2,
                    "lower_bound_active": False,
                    "edr_zero_denominator_fallback": False,
                },
            }
        )

    summary = _discovery_controller_summary(
        rows, np.ones(len(rows), dtype=bool)
    )

    assert summary["last_ten_all_below_005"] is True
    assert summary["last_ten_floor_fraction"] == 0.0
    assert summary["mechanical_discovery"] is True
