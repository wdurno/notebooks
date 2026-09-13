import math
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError
from mnist_experiment.rotated_mnist.pace_control.artifacts import (
    Plan10RunStore,
    validate_completed,
)
from mnist_experiment.rotated_mnist.pace_control.config import (
    load_feasibility_config,
)
from mnist_experiment.rotated_mnist.pace_control.finite_risk import (
    _correlation,
    _spearman,
)
from mnist_experiment.rotated_mnist.pace_control.finite_risk_config import (
    load_finite_risk_config,
)
from mnist_experiment.rotated_mnist.pace_control.reference_config import (
    load_reference_repair_config,
)
from mnist_experiment.rotated_mnist.pace_control.theory import (
    centered_pace,
    fixed_batch_risk,
    fixed_pi_q_equilibrium,
    fixed_pi_q_at_step,
    fixed_pi_q_update,
    movement_energy_target,
    optimal_fixed_batch_pi,
    pace_roots,
    stationary_movement_target,
    validate_pace_action_target,
)


REPO_ROOT = Path(__file__).parents[3]
FEASIBILITY_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "pace_control"
    / "configs"
    / "phase1_feasibility.json"
)
FINITE_RISK_CONFIG = FEASIBILITY_CONFIG.with_name("phase1b_finite_risk.json")
REFERENCE_REPAIR_CONFIG = FEASIBILITY_CONFIG.with_name(
    "phase1c_reference_repair.json"
)


def test_inverted_target_recovers_unique_risk_minimizer() -> None:
    target = movement_energy_target(
        0.05,
        q=1.0 / 30_000,
        old_covariance_shape=15.0,
        new_covariance_shape=16.0,
        batch_size=4,
    )
    optimum = optimal_fixed_batch_pi(
        movement_energy=target,
        q=1.0 / 30_000,
        old_covariance_shape=15.0,
        new_covariance_shape=16.0,
        batch_size=4,
    )
    assert optimum == pytest.approx(0.05)
    center = fixed_batch_risk(
        0.05,
        movement_energy=target,
        q=1.0 / 30_000,
        old_covariance_shape=15.0,
        new_covariance_shape=16.0,
        batch_size=4,
    )
    for action in (0.025, 0.049, 0.051, 0.10):
        assert fixed_batch_risk(
            action,
            movement_energy=target,
            q=1.0 / 30_000,
            old_covariance_shape=15.0,
            new_covariance_shape=16.0,
            batch_size=4,
        ) > center


def test_q_transient_converges_to_fixed_point_and_stationary_target() -> None:
    action = 0.05
    batch_size = 4
    equilibrium = fixed_pi_q_equilibrium(action, batch_size)
    q = 1.0 / 30_000
    for step in range(1_000):
        q = fixed_pi_q_update(q, action, batch_size)
        assert q == pytest.approx(
            fixed_pi_q_at_step(1.0 / 30_000, action, batch_size, step + 1)
        )
    assert q == pytest.approx(equilibrium)
    target = movement_energy_target(
        action,
        q=q,
        old_covariance_shape=15.0,
        new_covariance_shape=15.0,
        batch_size=batch_size,
    )
    assert target == pytest.approx(
        stationary_movement_target(
            action, covariance_shape=15.0, batch_size=batch_size
        )
    )


def test_negative_target_is_infeasible_and_centered_pace_requires_nonnegative() -> None:
    target = movement_energy_target(
        0.01,
        q=0.2,
        old_covariance_shape=10.0,
        new_covariance_shape=10.0,
        batch_size=4,
    )
    assert target < 0.0
    with pytest.raises(ValueError, match="movement_target"):
        centered_pace(target, 2.0)


def test_noncentered_pace_roots_satisfy_quadratic() -> None:
    result = pace_roots(
        5.0,
        direction_energy=4.0,
        direction_trend_inner=2.0,
        trend_energy=2.0,
    )
    assert len(result.roots) == 1
    root = result.roots[0]
    assert 4.0 * root * root - 4.0 * root + 2.0 == pytest.approx(5.0)
    assert result.minimizing_pace == pytest.approx(0.5)


def test_noncentered_pace_reports_no_feasible_root() -> None:
    result = pace_roots(
        0.5,
        direction_energy=1.0,
        direction_trend_inner=0.0,
        trend_energy=1.0,
    )
    assert result.roots == ()
    assert result.minimum_energy == pytest.approx(1.0)


def test_centered_pace_recovers_target_energy() -> None:
    pace = centered_pace(0.125, 0.5)
    assert pace == pytest.approx(0.5)
    assert pace * pace * 0.5 == pytest.approx(0.125)


def test_pace_cannot_rescale_a_realized_learner_update() -> None:
    validate_pace_action_target("next_distribution")
    with pytest.raises(ValueError, match="next encountered distribution"):
        validate_pace_action_target("learner_update")


@pytest.mark.parametrize("bad", [0.0, 1.0, math.inf, math.nan])
def test_fixed_action_must_be_interior(bad: float) -> None:
    with pytest.raises(ValueError, match="fixed_pi"):
        fixed_pi_q_equilibrium(bad, 4)


def test_feasibility_config_keeps_source_path_portable() -> None:
    config = load_feasibility_config(FEASIBILITY_CONFIG)
    assert not Path(config.oracle_run_path).is_absolute()
    assert config.artifact_schema_version == 2
    assert config.fixed_pi == pytest.approx(0.05)


def test_plan10_artifact_completion_is_valid_and_immutable(tmp_path: Path) -> None:
    config = load_feasibility_config(FEASIBILITY_CONFIG)
    store = Plan10RunStore(tmp_path)
    session = store.begin(
        run_id="unit-run",
        run_kind="unit",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=REPO_ROOT,
        experiment_config=config,
    )
    session.write_json("payload.json", {"finite": 1.0})
    completed = session.complete(("config.json", "payload.json"))
    manifest = validate_completed(
        completed, required=("config.json", "payload.json")
    )
    assert set(manifest["artifact_sha256"]) == {"config.json", "payload.json"}
    with pytest.raises(RotatedArtifactError, match="already completed"):
        store.begin(
            run_id="unit-run",
            run_kind="unit",
            config=config.to_mapping(),
            config_hash=config.config_hash,
            repo_root=REPO_ROOT,
            experiment_config=config,
        )


def test_finite_risk_config_freezes_heldout_panel_and_portable_sources() -> None:
    config = load_finite_risk_config(FINITE_RISK_CONFIG)
    assert config.heldout_offset == 2_000
    assert config.heldout_size == 8_000
    assert not Path(config.oracle_run_path).is_absolute()
    assert not Path(config.derivative_run_path).is_absolute()


def test_finite_risk_rank_correlation_handles_ties() -> None:
    assert _correlation([0.0, 1.0, 2.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert _spearman([0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.0, 9.0]) == pytest.approx(1.0)


def test_reference_repair_config_freezes_disjoint_panel_sizes() -> None:
    config = load_reference_repair_config(REFERENCE_REPAIR_CONFIG)
    assert config.candidate_start_count == 3
    assert config.refinement_epochs == 12
    assert config.selection_size + config.heldout_size == 10_000
    assert not Path(config.oracle_run_path).is_absolute()
