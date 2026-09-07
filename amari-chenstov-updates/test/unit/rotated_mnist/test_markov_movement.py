import json
import statistics
from pathlib import Path

import pytest
import torch

from src.representations import DenseFisher, LowRankDiagonalFisher

from mnist_experiment.rotated_mnist.markov_movement.artifacts import (
    Plan9RunStore,
    validate_completed,
)
from mnist_experiment.rotated_mnist.markov_movement.command_center import (
    Bundle,
    WorkUnit,
    execute_bundle,
    load_bundle,
)
from mnist_experiment.rotated_mnist.markov_movement.config import (
    load_attribution_config,
    load_extension_config,
    load_path_screen_config,
    load_retrospective_config,
)
from mnist_experiment.rotated_mnist.markov_movement.gain import (
    combine_fishers,
    ema_gain,
    stationary_effective_observations,
    structured_gain_action,
)
from mnist_experiment.rotated_mnist.markov_movement.gain_config import (
    load_gain_calibration_config,
    load_gain_contract_config,
    load_gain_estimator_config,
)
from mnist_experiment.rotated_mnist.markov_movement.estimators import (
    anchor_cancelled_observations,
    exponential_window_weights,
    innovation_noise_risk,
    marginal_action,
    weighted_cross_moment,
)
from mnist_experiment.rotated_mnist.markov_movement.attribution import (
    _selected_steps,
)
from mnist_experiment.rotated_mnist.markov_movement.path_screen import _schedule
from mnist_experiment.rotated_mnist.config import RuntimeConfig


REPO_ROOT = Path(__file__).parents[3]
PLAN9_CONFIG_ROOT = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "markov_movement"
    / "configs"
)


def test_anchor_cancellation_removes_nonzero_affine_anchor_error() -> None:
    dtype = torch.float64
    pi = torch.tensor([0.2, 0.4, 0.3], dtype=dtype)
    drift = torch.tensor([[1.0, -0.5], [1.2, -0.4], [1.3, -0.2]], dtype=dtype)
    innovations = torch.tensor(
        [[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2], [-0.4, -0.3]], dtype=dtype
    )
    anchor = torch.tensor([2.0, -3.0], dtype=dtype)
    y = []
    errors = [anchor]
    for step in range(3):
        y.append(drift[step] - errors[-1] + innovations[step + 1])
        errors.append(
            (1.0 - pi[step]) * (errors[-1] - drift[step])
            + pi[step] * innovations[step + 1]
        )
    normalized = torch.stack(y)
    displacements = pi[:, None] * normalized

    observed_y, z = anchor_cancelled_observations(displacements, pi)

    assert torch.allclose(observed_y, normalized, atol=1e-15, rtol=0.0)
    expected = torch.stack(
        [drift[t] + innovations[t + 1] - innovations[t] for t in range(1, 3)]
    )
    assert torch.allclose(z, expected, atol=1e-12, rtol=0.0)


def test_weighted_innovation_risk_retains_negative_adjacent_covariance() -> None:
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    risks = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    expected = (
        weights[0] ** 2 * (risks[0] + risks[1])
        + weights[1] ** 2 * (risks[1] + risks[2])
        + weights[2] ** 2 * (risks[2] + risks[3])
        - 2.0 * weights[0] * weights[1] * risks[1]
        - 2.0 * weights[1] * weights[2] * risks[2]
    )
    assert innovation_noise_risk(weights, risks) == pytest.approx(float(expected))


def test_innovation_risk_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(9123)
    weights = exponential_window_weights(
        torch.arange(5, dtype=torch.float64), half_life=2.0
    )
    variances = torch.tensor([0.4, 1.1, 0.7, 1.4, 0.9, 0.5], dtype=torch.float64)
    draws = torch.randn(200_000, 6, generator=generator, dtype=torch.float64)
    innovations = draws * variances.sqrt()
    differences = innovations[:, 1:] - innovations[:, :-1]
    empirical = float((differences @ weights).var(unbiased=True))
    expected = innovation_noise_risk(weights, variances)
    assert empirical == pytest.approx(expected, rel=0.015)


def test_marginal_action_uses_unequal_covariance_shapes() -> None:
    value = marginal_action(0.2, 0.01, 3.0, 5.0, 4)
    assert value == pytest.approx(0.23 / (0.23 + 1.25 + 1e-12))


def test_weighted_cross_moment_uses_each_admissible_pair_once() -> None:
    vectors = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    weights = torch.full((4,), 0.25, dtype=torch.float64)
    fisher = DenseFisher(torch.ones((1, 1), dtype=torch.float64))

    estimate, diagnostics = weighted_cross_moment(
        vectors, weights, fisher, lag_exclusion=1
    )

    assert estimate == pytest.approx((3.0 + 4.0 + 8.0) / 3.0)
    assert diagnostics["pair_count"] == 3
    assert diagnostics["effective_pair_count"] == pytest.approx(3.0)


def test_cross_moment_removes_independent_self_noise_in_expectation() -> None:
    generator = torch.Generator().manual_seed(9018)
    signal = torch.tensor([0.4, -0.2], dtype=torch.float64)
    fisher = DenseFisher(torch.eye(2, dtype=torch.float64))
    estimates = []
    for _ in range(4_000):
        vectors = signal + torch.randn(
            5, 2, generator=generator, dtype=torch.float64
        )
        value, _ = weighted_cross_moment(
            vectors,
            torch.full((5,), 0.2, dtype=torch.float64),
            fisher,
            lag_exclusion=0,
        )
        estimates.append(value)
    assert statistics.fmean(estimates) == pytest.approx(float(signal @ signal), abs=0.025)


def test_plan9_retrospective_configs_and_bundle_are_portable() -> None:
    first = load_retrospective_config(PLAN9_CONFIG_ROOT / "e9_1_primary.json")
    second = load_retrospective_config(PLAN9_CONFIG_ROOT / "e9_2_primary_v2.json")
    bundle = load_bundle(PLAN9_CONFIG_ROOT / "retrospective_bundle.json")

    assert first.study == "e9_1"
    assert second.study == "e9_2"
    assert all(
        not Path(value).is_absolute()
        for value in (
            first.oracle_run_path,
            first.single_lap_run_path,
            first.double_lap_run_path,
        )
    )
    assert bundle.units[1].dependencies == (bundle.units[0].unit_id,)
    assert all(not Path(unit.completed_marker).is_absolute() for unit in bundle.units)


def test_plan9_extension_configs_freeze_pair_weights_and_lags() -> None:
    opportunity = load_extension_config(PLAN9_CONFIG_ROOT / "e9_7_opportunity.json")
    cross = load_extension_config(PLAN9_CONFIG_ROOT / "e9_8_cross_moment.json")

    assert opportunity.study == "e9_7"
    assert cross.study == "e9_8"
    assert cross.primary_lag_exclusion == 1
    assert cross.primary_cross_half_lives_degrees == {
        "double_lap": 3.75,
        "single_lap": 15.0,
    }


def test_plan9_conditional_configs_and_path_schedule_are_frozen() -> None:
    attribution = load_attribution_config(
        PLAN9_CONFIG_ROOT / "e9_9_affinity_attribution.json"
    )
    screen = load_path_screen_config(PLAN9_CONFIG_ROOT / "e9_10_path_screen.json")
    schedule = _schedule(5)

    assert attribution.optimizer_step_budgets == (50, 100)
    assert len(_selected_steps(80, 12)) == 12
    assert len(schedule) == 25
    assert schedule[0]["angle_degrees"] == 0.0
    assert schedule[-1]["next_angle_degrees"] == 30.0
    assert {row["direction_regime"] for row in schedule} == {0, 1, 2}
    assert screen.transitions_per_arrow == (20, 10, 5, 2)

    bundle = load_bundle(PLAN9_CONFIG_ROOT / "extension_bundle.json")
    assert [unit.unit_id for unit in bundle.units] == [
        "e9.7-movement-opportunity",
        "e9.8-lagged-cross-moment",
        "e9.9-affinity-attribution-v2",
        "e9.10-opportunity-path-screen",
    ]


def test_plan9_gain_configs_and_bundle_freeze_causal_chain() -> None:
    contract = load_gain_contract_config(PLAN9_CONFIG_ROOT / "e9_12_gain_contract.json")
    estimator = load_gain_estimator_config(PLAN9_CONFIG_ROOT / "e9_13_gain_estimator.json")
    calibration = load_gain_calibration_config(PLAN9_CONFIG_ROOT / "e9_14_gain_calibration.json")
    bundle = load_bundle(PLAN9_CONFIG_ROOT / "gain_bundle.json")

    assert contract.scalar_half_lives_steps == (4.0, 8.0, 16.0)
    assert estimator.e9_12_run_id == contract.run_id
    assert calibration.e9_13_run_id == estimator.run_id
    assert [unit.dependencies for unit in bundle.units] == [
        (),
        ("e9.12-predictable-gain-contract",),
        ("e9.13-dual-timescale-gain",),
    ]
    assert all(not Path(unit.completed_marker).is_absolute() for unit in bundle.units)


def test_fast_fisher_effective_support_matches_frozen_values() -> None:
    supports = [stationary_effective_observations(value, 4) for value in (4.0, 8.0, 16.0)]

    assert supports == pytest.approx([46.282, 92.390, 184.694], rel=1e-3)
    assert ema_gain(8.0) == pytest.approx(1.0 - 2.0 ** (-1.0 / 8.0))


def test_structured_gain_combines_factors_and_reduces_to_scalar_case() -> None:
    dtype = torch.float64
    old = LowRankDiagonalFisher(
        torch.tensor([[1.0], [0.0], [0.5]], dtype=dtype),
        torch.tensor([0.2, 0.3, 0.4], dtype=dtype),
    )
    new = LowRankDiagonalFisher(
        torch.tensor([[0.0], [0.7], [0.2]], dtype=dtype),
        torch.tensor([0.5, 0.4, 0.1], dtype=dtype),
    )
    combined = combine_fishers(old, new, 0.25)
    expected = 0.75 * old.to_dense() + 0.25 * new.to_dense()
    assert torch.allclose(combined.to_dense(), expected, atol=1e-15, rtol=0.0)

    identity = LowRankDiagonalFisher(
        torch.zeros(3, 0, dtype=dtype), torch.ones(3, dtype=dtype)
    )
    vector = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
    action, diagnostics = structured_gain_action(
        identity,
        identity,
        vector,
        pi=0.2,
        relative_damping=1e-12,
    )
    assert torch.allclose(action, 0.2 * vector, atol=7e-12, rtol=0.0)
    assert diagnostics["normalized_residual"] < 1e-12


def test_fast_fisher_prediction_uses_state_before_current_batch() -> None:
    dtype = torch.float64
    initial = LowRankDiagonalFisher(
        torch.zeros(2, 0, dtype=dtype), torch.ones(2, dtype=dtype)
    )
    current = LowRankDiagonalFisher(
        torch.zeros(2, 0, dtype=dtype), torch.tensor([9.0, 0.1], dtype=dtype)
    )
    displacement = torch.tensor([1.0, 1.0], dtype=dtype)
    before, _ = structured_gain_action(
        initial, initial, displacement, pi=0.1, relative_damping=1e-12
    )
    after, _ = structured_gain_action(
        initial, current, displacement, pi=0.1, relative_damping=1e-12
    )

    assert torch.allclose(before, 0.1 * displacement, atol=2e-12, rtol=0.0)
    assert not torch.allclose(before, after)


def test_plan9_store_hashes_artifacts_and_rejects_mutation(tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        device="cpu", dtype="float64", num_workers=0, deterministic_algorithms=True
    )
    config = {"study": "test"}
    session = Plan9RunStore(tmp_path).begin(
        run_id="test-run",
        run_kind="test",
        config=config,
        config_hash="abc",
        repo_root=REPO_ROOT,
        runtime_config=type("Config", (), {"runtime": runtime})(),
    )
    session.write_json("result.json", {"ok": True})
    path = session.complete(("result.json",))

    validate_completed(path, required=("result.json",))
    with pytest.raises(Exception, match="already completed"):
        Plan9RunStore(tmp_path).begin(
            run_id="test-run",
            run_kind="test",
            config=config,
            config_hash="abc",
            repo_root=REPO_ROOT,
            runtime_config=type("Config", (), {"runtime": runtime})(),
        )


def test_bundle_resume_skips_completed_unit_and_stops_between_units(tmp_path: Path) -> None:
    markers = [tmp_path / name / "COMPLETED" for name in ("a", "b", "c")]
    units = tuple(
        WorkUnit(
            unit_id=name,
            command=(name,),
            dependencies=(() if index == 0 else (chr(ord("a") + index - 1),)),
            completed_marker=str(marker.relative_to(tmp_path)),
        )
        for index, (name, marker) in enumerate(zip(("a", "b", "c"), markers, strict=True))
    )
    bundle = Bundle("test-bundle", units)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle.mapping()), encoding="utf-8")
    calls = []

    def runner(command) -> None:
        name = command[0]
        calls.append(name)
        marker = markers[ord(name) - ord("a")]
        marker.parent.mkdir(parents=True)
        marker.touch()
        if name == "a":
            (tmp_path / "STOP_REQUESTED.json").write_text("{}", encoding="utf-8")

    first = execute_bundle(
        bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        resume=False,
        max_wall_minutes=None,
        runner=runner,
    )
    assert first["completed"] == ["a"]
    (tmp_path / "STOP_REQUESTED.json").unlink()
    second = execute_bundle(
        bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        resume=True,
        max_wall_minutes=None,
        runner=runner,
    )
    assert calls == ["a", "b", "c"]
    assert second["skipped"] == ["a"]


def test_bundle_does_not_promote_an_interrupted_unit(tmp_path: Path) -> None:
    marker = tmp_path / "result" / "COMPLETED"
    unit = WorkUnit("unit", ("unit",), (), str(marker.relative_to(tmp_path)))
    bundle = Bundle("interrupt", (unit,))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle.mapping()), encoding="utf-8")

    with pytest.raises(KeyboardInterrupt):
        execute_bundle(
            bundle,
            bundle_path=bundle_path,
            repo_root=tmp_path,
            resume=False,
            max_wall_minutes=None,
            runner=lambda command: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    assert not marker.exists()
