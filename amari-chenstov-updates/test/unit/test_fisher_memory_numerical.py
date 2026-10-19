"""Phase 1 calibration checks on tiny deterministic CPU checkpoints."""

import copy
import json

import pytest
import torch

from mnist_experiment.fisher_memory import numerical
from mnist_experiment.fisher_memory.learner import FitSettings, StopRequested
from mnist_experiment.fisher_memory.model import build_model, memory_indices
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested as RuntimeStop


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs():
    model, layout = build_model(13, dtype=torch.float64)
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(".lora_B"):
                value.copy_(.1 * torch.randn(value.shape, dtype=torch.float64, generator=generator))
    theta = layout.flatten_module(model, detach=True)
    x = torch.randn(4, 1, 28, 28, dtype=torch.float64, generator=generator)
    ltm, _ = memory_indices(layout)
    settings = FitSettings(num_probes=2, fd_step=.001, max_iter=2, max_eval=5,
                           direction_max_norm=.05, chunk_size=2)
    return (model, layout, theta, x[:2], torch.tensor([1, 9]), x[2:], torch.tensor([3, 7]),
            torch.tensor([.5, 1.], dtype=torch.float64),
            torch.eye(ltm.numel(), dtype=torch.float64) * .01, theta[ltm].clone(), settings, 19)


@pytest.fixture(scope="module")
def completed_audit():
    args = _inputs()
    state = {name: tensor.clone() for name, tensor in args[0].state_dict().items()}
    theta = args[2].clone()
    rng = torch.random.get_rng_state().clone()
    report = numerical.audit(*args, checkpoint_label="pre_eviction")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(theta, args[2])
    for name, value in args[0].state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    return report


def test_full_audit_json_schema_mathematical_checks_and_actual_shadow(completed_audit):
    report = completed_audit
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["status"] == "passed"
    assert report["valid_methods"] == ["jvp", "finite_difference", "directional"]
    assert report["has_mature_replay"] and report["checkpoint_label"] == "pre_eviction"
    assert report["direction"]["source"] == "ordinary_optimizer_shadow"
    assert report["direction"]["supported"]
    assert report["direction"]["shadow"]["total_closure_calls"] <= 5
    assert report["direction"]["norm"] <= .05 + 1e-12
    assert not report["direction"]["fallback_for_math_check_only"]
    for name in numerical.METHODS:
        check = report["methods"][name]["gradient_check"]
        assert check["passed"] and check["nondegenerate_blocks"] == 2
    assert report["methods"]["distillation"]["valid"]
    assert report["methods"]["distillation"]["stm_auxiliary_gradient_norm"] == 0


def test_probe_ladder_is_diagnostic_not_an_implementation_validity_test(completed_audit):
    report = completed_audit
    assert [row["probes"] for row in report["probe_ladder"]] == [1, 2, 4, 8]
    assert report["independent_trace"]["probes"] == 16
    assert not report["independent_trace"]["used_for_mathematical_validity"]
    assert report["seeds"]["training_curvature_probes"] != report["seeds"]["independent_diagnostic_probes"]
    assert report["probe_ladder"][0]["estimated_relative_standard_error"] is None
    # A finite probe bank is not required to achieve the 2% deterministic target.
    assert any(not row["trace_sampling_error"]["passed"] for row in report["probe_ladder"])
    assert report["methods"]["jvp"]["valid"]
    candidates = report["methods"]["finite_difference"]["candidates"]
    assert [row["h"] for row in candidates] == [.0005, .001, .002]
    assert report["selected_fd_step"] == .001
    assert all(row["full_perturbation_norm"] > row["h"] for row in candidates)


def test_strength_brackets_match_total_gradient_ratios_and_report_both_blocks(completed_audit):
    report = completed_audit
    assert report["strength_ratios"] == [.03, .1, .3]
    for name in (*numerical.METHODS, "distillation"):
        calibration = report["calibration"][name]
        assert calibration["calibrated"]
        denominator = calibration["data_gradient_norms"]["total"]
        numerator = calibration["auxiliary_gradient_norms"]["total"]
        for ratio, strength, row in zip(report["strength_ratios"], report["strengths"][name],
                                         calibration["bracket"], strict=True):
            assert strength * numerator / denominator == pytest.approx(ratio)
            assert row["coefficient"] == strength
            assert set(row["block_ratios"]) == {"ltm", "stm"}
    assert all(row["block_ratios"]["stm"] == 0.
               for row in report["calibration"]["distillation"]["bracket"])


def test_zero_age_or_zero_shadow_is_uncalibrated_not_a_mathematical_failure():
    args = list(_inputs())
    args[7] = torch.zeros_like(args[7])
    _, stm = memory_indices(args[1])
    report = numerical.audit(*args, direction=torch.zeros(stm.numel(), dtype=torch.float64))
    assert report["status"] == "passed" and not report["has_mature_replay"]
    assert report["direction"]["fallback_for_math_check_only"]
    assert not report["direction"]["supported"]
    assert report["calibration"]["directional"]["reason"] == "unsupported_actual_direction"
    for name in (*numerical.METHODS, "distillation"):
        assert not report["calibration"][name]["calibrated"]
        assert report["strengths"][name] == [None, None, None]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("recover", [True, False])
def test_directional_local_check_shrinks_a_bounded_stencil_without_changing_training_radius(monkeypatch, recover):
    original = numerical.step_kl_values
    def distorted(model, layout, theta, inputs, delta, **kwargs):
        value = original(model, layout, theta, inputs, delta, **kwargs)
        norm = float(delta.norm())
        local = norm <= .00101
        too_large = norm > .000251 if recover else True
        return value * 1.1 if local and too_large else value
    monkeypatch.setattr(numerical, "step_kl_values", distorted)
    args = _inputs()
    _, stm = memory_indices(args[1])
    direction = torch.ones(stm.numel(), dtype=torch.float64)
    direction *= .02 / direction.norm()
    report = numerical.audit(*args, direction=direction)
    assert report["direction"]["norm"] == pytest.approx(.02)
    candidates = report["direction"]["local_check_candidates"]
    assert [row["radius"] for row in candidates] == ([.001, .0005, .00025] if recover else [.001, .0005, .00025, .0001])
    assert report["methods"]["directional"]["valid"] is recover
    assert all(not row["agreement"]["passed"] for row in candidates[:-1])


def test_unavailable_jvp_does_not_block_kl_or_distillation_audits(monkeypatch):
    def unsupported(*args, **kwargs):
        raise NotImplementedError("deliberately unsupported forward-mode backend")

    monkeypatch.setattr(numerical, "jvp_values", unsupported)
    report = numerical.audit(*_inputs())
    assert report["status"] == "partial"
    assert report["valid_methods"] == ["finite_difference", "directional"]
    assert "NotImplementedError" in report["methods"]["jvp"]["error"]
    assert report["strengths"]["jvp"] == [None, None, None]
    assert report["methods"]["distillation"]["valid"]


def test_cancellation_is_not_swallowed_as_method_failure():
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls >= 5

    with pytest.raises(StopRequested, match="audit stopped"):
        numerical.audit(*_inputs(), check_stop=stop)
    assert calls == 5


@pytest.mark.parametrize("exception", [RuntimeStop, ResourceLimit])
def test_runtime_controls_are_not_swallowed_by_numerical_handlers(monkeypatch, exception):
    def halt(*args, **kwargs):
        raise exception("intentional runtime control")
    monkeypatch.setattr(numerical, "jvp_values", halt)
    with pytest.raises(exception, match="intentional runtime control"):
        numerical.audit(*_inputs())


def test_aggregate_uses_nondegenerate_medians_and_requires_common_contract(completed_audit):
    first = copy.deepcopy(completed_audit)
    second = copy.deepcopy(completed_audit)
    third = copy.deepcopy(completed_audit)
    for name in (*numerical.METHODS, "distillation"):
        second["calibration"][name]["unit_ratio_coefficient"] *= 2
        third["calibration"][name].update(calibrated=False, unit_ratio_coefficient=None)
    result = numerical.aggregate_audits([first, second, third])
    json.dumps(result, allow_nan=False)
    assert result["checkpoint_count"] == 3 and result["has_mature_replay"]
    for name in (*numerical.METHODS, "distillation"):
        assert result["methods"][name]["calibrated_checkpoints"] == 2
        for actual, original in zip(result["strengths"][name], first["strengths"][name], strict=True):
            assert actual == pytest.approx(1.5 * original)
    second["calibration_signature"]["fd_step"] *= 2
    with pytest.raises(ValueError, match="common numerical contract"):
        numerical.aggregate_audits([first, second])


def test_audit_rejects_empty_replay_and_invalid_age_weights():
    args = list(_inputs())
    args[7] = torch.tensor([-1., .5])
    with pytest.raises(ValueError, match="ages"):
        numerical.audit(*args)
    args[5], args[6], args[7] = args[5][:0], args[6][:0], args[7][:0]
    with pytest.raises(ValueError, match="nonempty replay"):
        numerical.audit(*args)
