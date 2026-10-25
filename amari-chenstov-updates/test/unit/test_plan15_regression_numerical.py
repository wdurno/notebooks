"""Bounded CPU audits of regression FD calibration, not predictive selection."""

import copy
import json
import math
import statistics
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory_regression import numerical
from mnist_experiment.fisher_memory_regression.curvature import fd_values, gaussian_nll, jvp_values
from mnist_experiment.fisher_memory_regression.learner import StopRequested
from mnist_experiment.fisher_memory_regression.model import build_model, flat_prediction, memory_indices
from src.seeding import derive_component_seed


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def example():
    model, layout = build_model(17, dtype=torch.float32, factor_seed=51)
    theta = layout.flatten_module(model, detach=True)
    generator = torch.Generator().manual_seed(233)
    for spec in layout.specs:
        if spec.name.endswith("lora_B"):
            theta[spec.start:spec.stop] = .1 * torch.randn(spec.numel, generator=generator)
    ltm, _ = memory_indices(layout)
    images = torch.randn(3, 1, 28, 28, generator=generator)
    snapshot = dict(theta=theta, new_x=images[:1], new_y=torch.tensor([42.]),
                    replay_x=images[1:], replay_y=torch.tensor([-5., 9.]),
                    ages=torch.tensor([.25, 1.]), fisher=torch.eye(len(ltm)),
                    anchor=theta[ltm].clone(), seed=671, n=5, step=18,
                    recipe={"method": "replay_ewc", "capacity": 16, "replay_count": 2},
                    layout=layout.metadata())
    return SimpleNamespace(fd_step=.01, numerical_tolerance=.05, probes=2), model, layout, snapshot


def clone_snapshot(snapshot, **updates):
    return {**copy.deepcopy(snapshot), **updates}


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_float64_calibration_matches_independent_values_and_gradient_ratio(example):
    cfg, model, layout, snapshot = example
    report = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert report["selected_step"] == cfg.fd_step
    assert report["audit_dtype"] == "float64"
    assert report["production_dtype"] == "float32"
    assert report["production_dtypes"] == ["float32"]
    assert report["production_verified"]
    assert report["viable_snapshots"] == report["valid_gradient_ratios"] == 1
    assert len(report["numerical_rows"]) == 3
    assert all(row["passed"] for row in report["numerical_rows"])
    assert report["lambda0"] > 0 and math.isfinite(report["lambda0"])
    json.dumps(report, allow_nan=False)

    theta = snapshot["theta"].double().requires_grad_()
    replay_x = snapshot["replay_x"].double()
    ltm, stm = memory_indices(layout)
    seed = derive_component_seed(snapshot["seed"], numerical.AUDIT_PROBE_COMPONENT)
    gen = torch.Generator().manual_seed(seed)
    probes = (2 * torch.randint(2, (cfg.probes, len(stm)), generator=gen) - 1).double()
    exact = jvp_values(model, layout, theta, replay_x, probes)
    exact_gradient, = torch.autograd.grad((snapshot["ages"].double() * exact.mean(0) / len(stm)).mean(), theta)
    approximate = fd_values(model, layout, theta, replay_x, probes, cfg.fd_step)
    fd_gradient, = torch.autograd.grad((snapshot["ages"].double() * approximate.mean(0) / len(stm)).mean(), theta)
    row = next(row for row in report["numerical_rows"] if row["fd_step"] == cfg.fd_step)
    expected_error = float((approximate.detach() - exact.detach()).norm() / exact.detach().norm())
    expected_gradient_error = float((fd_gradient - exact_gradient).norm() / exact_gradient.norm())
    assert row["value_relative_error"] == pytest.approx(expected_error, rel=1e-12)
    assert row["gradient_relative_error"] == pytest.approx(expected_gradient_error, rel=1e-12)
    assert row["fd_ltm_gradient_norm"] == pytest.approx(float(fd_gradient[ltm].norm()))
    assert row["fd_stm_gradient_norm"] == pytest.approx(float(fd_gradient[stm].norm()))
    assert row["fd_ltm_gradient_norm"] > 0 and row["fd_stm_gradient_norm"] > 0
    theta32 = snapshot["theta"].clone().requires_grad_()
    native = fd_values(model, layout, theta32, snapshot["replay_x"], probes.float(), cfg.fd_step)
    native_grad, = torch.autograd.grad((snapshot["ages"].double() * native.mean(0) / len(stm)).mean(), theta32)
    assert row["production_value_relative_error"] == pytest.approx(
        float((native.detach() - exact.detach()).norm() / exact.detach().norm()), rel=1e-12,
    )
    assert row["production_gradient_relative_error"] == pytest.approx(
        float((native_grad.double() - exact_gradient).norm() / exact_gradient.norm()), rel=1e-12,
    )
    assert row["production_fd_ltm_gradient_norm"] == pytest.approx(float(native_grad[ltm].double().norm()))
    assert row["production_fd_stm_gradient_norm"] == pytest.approx(float(native_grad[stm].double().norm()))
    assert row["float64_passed"] and row["production_passed"]
    ordinary = .5 * (
        gaussian_nll(flat_prediction(model, layout, theta, snapshot["new_x"].double()), snapshot["new_y"]).mean()
        + gaussian_nll(flat_prediction(model, layout, theta, replay_x), snapshot["replay_y"]).mean()
    )
    data_gradient, = torch.autograd.grad(ordinary, theta)
    assert report["lambda0"] == pytest.approx(float(data_gradient.norm() / fd_gradient.norm()), rel=1e-12)
    assert report["gradient_ratios"][0]["probe_seed"] == seed
    assert seed != derive_component_seed(snapshot["seed"], "regression_training_curvature_probes")


def test_common_probes_are_deterministic_and_leave_snapshots_model_and_rng_unchanged(example, monkeypatch):
    cfg, model, layout, snapshot = example
    seen = []
    original_fd, original_jvp = numerical.fd_values, numerical.jvp_values

    def fd(*args):
        seen.append(("fd", args[4].detach().clone()))
        assert args[2].dtype == args[3].dtype == args[4].dtype
        assert args[2].dtype in (torch.float32, torch.float64)
        return original_fd(*args)

    def jvp(*args):
        seen.append(("jvp", args[4].detach().clone()))
        assert args[2].dtype == args[3].dtype == torch.float64
        return original_jvp(*args)

    monkeypatch.setattr(numerical, "fd_values", fd)
    monkeypatch.setattr(numerical, "jvp_values", jvp)
    before, weights, rng = copy.deepcopy(snapshot), copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    first = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    second = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert first == second
    assert len(seen) == 14
    assert sum(probes.dtype == torch.float32 for _, probes in seen) == 6
    assert all(torch.equal(seen[0][1], probes.double()) for _, probes in seen)
    assert torch.equal(rng, torch.get_rng_state())
    assert_same(snapshot, before)
    assert_same(model.state_dict(), weights)
    changed = numerical.calibrate_fd(cfg, [clone_snapshot(snapshot, seed=snapshot["seed"] + 1)], model, layout)
    assert changed["gradient_ratios"][0]["probe_seed"] != first["gradient_ratios"][0]["probe_seed"]
    assert not torch.equal(seen[0][1], seen[-1][1])


def test_selection_prefers_original_then_smaller_equidistant_passing_step(example, monkeypatch):
    cfg, model, layout, snapshot = example
    original = numerical.fd_values

    def wrong_original(*args):
        result = original(*args)
        return result * (1.5 if args[-1] == cfg.fd_step else 1)

    monkeypatch.setattr(numerical, "fd_values", wrong_original)
    report = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert report["selected_step"] == cfg.fd_step / 2
    assert [row["passed"] for row in report["numerical_rows"]] == [True, False, True]
    assert report["gradient_ratios"][0]["fd_step"] == cfg.fd_step / 2


def test_step_must_pass_every_snapshot_not_just_a_mean_error(example, monkeypatch):
    cfg, model, layout, snapshot = example
    other = clone_snapshot(snapshot, n=10, step=30)
    other["replay_x"].add_(10)
    original = numerical.fd_values

    def wrong_only_small_on_second(*args):
        result = original(*args)
        is_second = float(args[3].mean()) > 5
        wrong = args[-1] == cfg.fd_step or (is_second and args[-1] == cfg.fd_step / 2)
        return result * (1.5 if wrong else 1)

    monkeypatch.setattr(numerical, "fd_values", wrong_only_small_on_second)
    report = numerical.calibrate_fd(cfg, [snapshot, other], model, layout)
    assert report["selected_step"] == 2 * cfg.fd_step
    assert report["viable_snapshots"] == 2
    assert report["lambda0"] == statistics.median(row["ratio"] for row in report["gradient_ratios"])


def test_gradient_agreement_is_required_even_when_forward_values_are_exact(example, monkeypatch):
    cfg, model, layout, snapshot = example
    original = numerical.fd_values

    def broken_gradient(*args):
        values = original(*args)
        return 2 * values - values.detach()

    monkeypatch.setattr(numerical, "fd_values", broken_gradient)
    with pytest.raises(numerical.CalibrationError, match="value AND theta-gradient") as error:
        numerical.calibrate_fd(cfg, [snapshot], model, layout)
    report = error.value.diagnostics
    assert report["selected_step"] is None
    assert all(row["value_relative_error"] < cfg.numerical_tolerance for row in report["numerical_rows"])
    assert all(row["gradient_relative_error"] > .9 for row in report["numerical_rows"])
    json.dumps(report, allow_nan=False)


def test_empty_unaged_and_nonfinite_snapshots_are_explicitly_skipped(example):
    cfg, model, layout, snapshot = example
    empty = clone_snapshot(snapshot, replay_x=snapshot["replay_x"][:0],
                           replay_y=snapshot["replay_y"][:0], ages=snapshot["ages"][:0])
    unaged = clone_snapshot(snapshot, ages=torch.zeros_like(snapshot["ages"]))
    nonfinite = clone_snapshot(snapshot)
    nonfinite["theta"][0] = float("nan")
    report = numerical.calibrate_fd(cfg, [empty, unaged, nonfinite, snapshot], model, layout)
    assert report["viable_snapshots"] == report["valid_gradient_ratios"] == 1
    assert [row["reason"] for row in report["gradient_ratios"]] == [
        "empty_replay", "no_aged_replay", "nonfinite_snapshot", None,
    ]
    assert len(report["numerical_rows"]) == 12
    for row in report["numerical_rows"][:9]:
        assert row["float64_status"] == row["production_status"] == "skipped"
        assert row["production_reason"] == row["reason"]
        assert not row["production_passed"]
        assert row["production_value_relative_error"] is None
        assert row["production_gradient_relative_error"] is None
    json.dumps(report, allow_nan=False)
    with pytest.raises(numerical.CalibrationError, match="no viable") as error:
        numerical.calibrate_fd(cfg, [empty, unaged, nonfinite], model, layout)
    assert error.value.diagnostics["lambda0"] is None
    json.dumps(error.value.diagnostics, allow_nan=False)


def test_zero_penalty_gradient_is_not_replaced_by_epsilon(example, monkeypatch):
    cfg, model, layout, snapshot = example

    def flat_curvature(model, layout, theta, x, probes):
        return theta.sum() * 0 + theta.new_ones((len(probes), len(x)))

    monkeypatch.setattr(numerical, "jvp_values", flat_curvature)
    with pytest.raises(numerical.CalibrationError, match="no viable") as error:
        numerical.calibrate_fd(cfg, [snapshot], model, layout)
    report = error.value.diagnostics
    assert report["lambda0"] is None
    assert report["gradient_ratios"][0]["reason"] == "zero_jvp_penalty_gradient"


def test_zero_data_gradient_is_rejected_after_numerical_checks(example):
    cfg, model, layout, snapshot = example
    snapshot = clone_snapshot(snapshot)
    # An exactly zero head predicts zero while its random A retains a nonzero
    # STM Jacobian and a differentiable curvature penalty through the trunk.
    for spec in layout.specs:
        if spec.name in {"classifier.weight", "classifier.bias", "classifier.lora_B"}:
            snapshot["theta"][spec.start:spec.stop] = 0
    snapshot["new_y"].zero_()
    snapshot["replay_y"].zero_()
    with pytest.raises(numerical.CalibrationError, match="no finite positive gradient ratio") as error:
        numerical.calibrate_fd(cfg, [snapshot], model, layout)
    report = error.value.diagnostics
    assert report["selected_step"] == cfg.fd_step and report["lambda0"] is None
    assert report["gradient_ratios"][0]["reason"] == "zero_data_gradient"
    report = numerical.calibrate_fd(cfg, [snapshot, example[-1]], model, layout)
    assert report["viable_snapshots"] == 2 and report["valid_gradient_ratios"] == 1
    assert report["lambda0"] == report["gradient_ratios"][1]["ratio"]


def test_no_valid_gradient_ratios_raises_with_numerical_rows(example, monkeypatch):
    cfg, model, layout, snapshot = example

    def stationary(prediction, target):
        return prediction * 0

    monkeypatch.setattr(numerical, "gaussian_nll", stationary)
    with pytest.raises(numerical.CalibrationError, match="no finite positive gradient ratio") as error:
        numerical.calibrate_fd(cfg, [snapshot], model, layout)
    report = error.value.diagnostics
    assert report["selected_step"] == cfg.fd_step and report["lambda0"] is None
    assert all(row["passed"] for row in report["numerical_rows"])
    assert report["gradient_ratios"][0]["reason"] == "zero_data_gradient"


def test_nonfinite_fd_failure_is_json_safe_and_does_not_poison_other_steps(example, monkeypatch):
    cfg, model, layout, snapshot = example
    original = numerical.fd_values

    def nonfinite_original(*args):
        result = original(*args)
        return result * (float("nan") if args[-1] == cfg.fd_step else 1)

    monkeypatch.setattr(numerical, "fd_values", nonfinite_original)
    report = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert report["selected_step"] == cfg.fd_step / 2
    row = report["numerical_rows"][1]
    assert row["status"] == "failed" and row["value_relative_error"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["values", "gradient", "gradient_direction", "nonfinite"])
def test_production_only_defects_fail_even_with_passing_float64_checks(example, monkeypatch, fault):
    cfg, model, layout, snapshot = example
    original = numerical.fd_values

    def broken_native(*args):
        values = original(*args)
        if args[2].dtype != torch.float32:
            return values
        if fault == "values":
            return 1.5 * values
        if fault == "gradient":
            return 2 * values - values.detach()
        if fault == "gradient_direction":
            return 2 * values.detach() - values
        return values * float("nan")

    monkeypatch.setattr(numerical, "fd_values", broken_native)
    with pytest.raises(numerical.CalibrationError, match="production dtype") as error:
        numerical.calibrate_fd(cfg, [snapshot], model, layout)
    report = error.value.diagnostics
    assert report["selected_step"] is None and not report["production_verified"]
    for row in report["numerical_rows"]:
        assert row["float64_passed"] and row["float64_status"] == "passed"
        assert not row["production_passed"] and row["production_status"] == "failed"
        assert not row["passed"] and row["reason"].startswith("production_")
        if fault in {"gradient", "gradient_direction"}:
            assert row["production_value_relative_error"] < cfg.numerical_tolerance
            assert row["production_gradient_relative_error"] > .9
        if fault == "gradient_direction":
            assert row["production_fd_gradient_norm"] == pytest.approx(row["jvp_gradient_norm"], rel=.01)
        if fault == "nonfinite":
            assert row["production_value_relative_error"] is None
            assert row["production_gradient_relative_error"] is None
    json.dumps(report, allow_nan=False)


def test_production_gate_can_choose_an_alternative_step(example, monkeypatch):
    cfg, model, layout, snapshot = example
    original = numerical.fd_values

    def bad_native_original(*args):
        values = original(*args)
        return values * (1.5 if args[2].dtype == torch.float32 and args[-1] == cfg.fd_step else 1)

    monkeypatch.setattr(numerical, "fd_values", bad_native_original)
    report = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert report["selected_step"] == cfg.fd_step / 2
    assert report["production_verified"]
    assert all(row["float64_passed"] for row in report["numerical_rows"])
    assert [row["production_passed"] for row in report["numerical_rows"]] == [True, False, True]


@pytest.mark.parametrize("dtype", ["float64", torch.float64])
def test_configured_production_dtype_overrides_snapshot_dtype(example, dtype):
    cfg, model, layout, snapshot = example
    cfg.dtype = dtype
    report = numerical.calibrate_fd(cfg, [snapshot], model, layout)
    assert report["production_dtype"] == "float64"
    assert report["production_verified"]
    for row in report["numerical_rows"]:
        assert row["production_value_relative_error"] == row["value_relative_error"]
        assert row["production_gradient_relative_error"] == row["gradient_relative_error"]


def test_missing_dtype_uses_each_snapshot_dtype(example):
    cfg, model, layout, snapshot = example
    other = clone_snapshot(snapshot, theta=snapshot["theta"].double())
    report = numerical.calibrate_fd(cfg, [snapshot, other], model, layout)
    assert report["production_dtype"] == "mixed"
    assert report["production_dtypes"] == ["float32", "float64"]
    assert report["production_verified"]
    assert {row["production_dtype"] for row in report["gradient_ratios"]} == {"float32", "float64"}


@pytest.mark.parametrize("dtype", ["float16", torch.bfloat16, "unknown"])
def test_unsupported_production_precision_is_rejected(example, dtype):
    cfg, model, layout, snapshot = example
    cfg.dtype = dtype
    with pytest.raises(ValueError, match="production FD dtype"):
        numerical.calibrate_fd(cfg, [snapshot], model, layout)


@pytest.mark.parametrize("when", [1, 4, 9, 13])
def test_stop_callback_aborts_and_preserves_snapshot(example, when):
    cfg, model, layout, snapshot = example
    before = copy.deepcopy(snapshot)
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        return calls == when

    with pytest.raises(StopRequested):
        numerical.calibrate_fd(cfg, [snapshot], model, layout, check_stop=check)
    assert calls == when
    assert_same(before, snapshot)


def test_raising_stop_callback_propagates_without_wrapping(example):
    cfg, model, layout, snapshot = example

    class BudgetStop(RuntimeError):
        pass

    def check():
        raise BudgetStop("paused")

    with pytest.raises(BudgetStop, match="paused"):
        numerical.calibrate_fd(cfg, [snapshot], model, layout, check_stop=check)


def test_invalid_layout_or_age_contract_is_not_silently_skipped(example):
    cfg, model, layout, snapshot = example
    malformed = clone_snapshot(snapshot, ages=torch.tensor([-1., 1.]))
    with pytest.raises(ValueError, match="squared arrival weights"):
        numerical.calibrate_fd(cfg, [malformed], model, layout)
    malformed = clone_snapshot(snapshot)
    malformed["layout"]["total_numel"] += 1
    with pytest.raises(ValueError, match="metadata"):
        numerical.calibrate_fd(cfg, [malformed], model, layout)
