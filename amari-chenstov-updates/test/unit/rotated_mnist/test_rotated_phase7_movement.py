import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase7_movement_artifacts import (
    MOVEMENT_REQUIRED_ARTIFACTS,
    MovementPremiumAuditStore,
    load_completed_movement_premium_audit,
)
from mnist_experiment.rotated_mnist.phase7_movement_audit import (
    marginal_pi,
    movement_premium,
    summarize_error_breakdowns,
    summarize_movement_rows,
    transition_key,
)
from mnist_experiment.rotated_mnist.phase7_movement_config import (
    MovementPremiumAuditConfig,
    load_movement_premium_audit_config,
)


REPO_ROOT = Path(__file__).parents[3]
CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase7_movement_premium_audit.json"
)


def _attribution_row(**overrides):
    row = {
        "design": "single_lap",
        "schedule": "linear",
        "leg_id": 0,
        "eligible": True,
        "cold_start": False,
        "fallback": False,
        "clipped": False,
        "reversal_window": False,
        "reversal_step": None,
        "oracle_precision_pass": True,
        "applied_pi": 0.21,
        "online_instantaneous_pi": 0.20,
        "tracked_q_covariance_pi": 0.08,
        "population_marginal_pi": 0.10,
        "population_signal_online_scale_pi": 0.101,
        "online_signal_population_scale_pi": 0.50,
        "online_signal": 10.0,
        "population_signal": 0.01,
        "signal_log_ratio": 6.9,
        "online_scale": 100.0,
        "population_new_scale": 10.0,
        "scale_log_ratio": 2.3,
        "discounted_movement_premium": 0.2,
        "population_movement_premium": 0.004,
    }
    row.update(overrides)
    return row


def test_movement_config_round_trips_frozen_artifact_contract() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = load_movement_premium_audit_config(CONFIG)

    assert config.to_mapping() == raw
    assert config == MovementPremiumAuditConfig.from_mapping(raw)
    assert config.deployed_batch_size == 4
    assert config.reversal_window_steps == 8


def test_movement_config_rejects_authoritative_source_drift() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["oracle_run_id"] = "different"

    with pytest.raises(RotatedConfigError, match="path and run ID differ"):
        MovementPremiumAuditConfig.from_mapping(raw)


def test_population_marginal_rule_and_zero_movement_limit() -> None:
    q = 0.02
    batch_size = 4
    covariance_only = marginal_pi(0.0, q, 7.0, 7.0, batch_size)
    complete = marginal_pi(0.5, q, 7.0, 7.0, batch_size)

    assert covariance_only == pytest.approx(batch_size * q / (1 + batch_size * q))
    assert complete > covariance_only
    assert movement_premium(0.5, 7.0, batch_size) == pytest.approx(2.0 / 7.0)


def test_transition_key_preserves_exact_directional_identity() -> None:
    forward = transition_key("linear", 0.0, 0.75)
    reverse = transition_key("linear", 0.75, 0.0)

    assert forward != reverse
    assert forward == transition_key("linear", 0.0, 0.75)


def test_counterfactual_attribution_identifies_inflated_numerator() -> None:
    summaries, classification = summarize_movement_rows(
        [_attribution_row()], attribution_tolerance=0.01
    )

    assert classification["primary"] == "numerator_dominated"
    assert summaries[0]["online_instantaneous_pi_mae"] == pytest.approx(0.10)
    assert summaries[0]["population_signal_online_scale_pi_mae"] == pytest.approx(
        0.001
    )
    assert summaries[0]["online_signal_population_scale_pi_mae"] == pytest.approx(
        0.40
    )


def test_error_breakdowns_separate_legs_and_reversal_windows() -> None:
    rows = [
        _attribution_row(),
        _attribution_row(
            leg_id=1,
            reversal_window=True,
            reversal_step=40,
            applied_pi=0.18,
            online_instantaneous_pi=0.17,
        ),
    ]
    breakdowns = summarize_error_breakdowns(rows, reversal_window_steps=8)

    assert len(breakdowns["legs"]) == 2
    assert len(breakdowns["reversal_windows"]) == 1
    assert breakdowns["reversal_windows"][0]["reversal_step"] == 40


def test_movement_audit_artifact_is_strict_and_immutable(tmp_path: Path) -> None:
    config = load_movement_premium_audit_config(CONFIG)
    store = MovementPremiumAuditStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    session.write_json("source_contract.json", {})
    session.write_json("movement_rows.json", [])
    session.write_json(
        "audit_summary.json",
        {"config_hash": config.config_hash, "row_count": 0},
    )
    path = session.complete(required=MOVEMENT_REQUIRED_ARTIFACTS)

    loaded = load_completed_movement_premium_audit(path)
    assert loaded.config == config
    assert loaded.movement_rows == ()
    with pytest.raises(Exception, match="already completed"):
        store.begin(config, REPO_ROOT)
