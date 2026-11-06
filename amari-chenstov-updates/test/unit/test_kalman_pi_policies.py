from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.diagonal_lfu.curvature import observation, logits
from mnist_experiment.kalman_pi.observations import observed_scores, combine_moments
from mnist_experiment.kalman_pi.policies import Controller, PolicySpec, concentration, covariance_action
from src.parameters import ParameterLayout


def config(**changes):
    values = dict(initialization_size=2048, observer_floor_relative=1e-6,
                  observer_floor_absolute=1e-12, pi_min=1e-4, pi_max=.2,
                  cold_start=2, noise_half_life=32.)
    return SimpleNamespace(**{**values, **changes})


def observer(kind="kalman", value=.001, coordinates=3, **changes):
    controller = Controller(PolicySpec(kind, value), config(**changes), .025)
    diagonal = torch.arange(1, coordinates + 1, dtype=torch.float64)
    controller.initialize(torch.zeros(coordinates, dtype=torch.float64), diagonal, diagonal)
    return controller, diagonal


def test_exact_weight_recursion_and_fixed_stationary_limit():
    c, m, q = .025, 4, 1 / 2048
    for _ in range(5000):
        q = concentration(q, c, m)
    assert q == pytest.approx(c / (m * (2 - c)))
    assert covariance_action(q, m) == pytest.approx(c / 2)
    q = 1 / 2048
    weights = torch.full((2048,), q, dtype=torch.float64)
    for pi in (.01, .05, .001):
        weights = torch.cat(((1 - pi) * weights, torch.full((m,), pi / m, dtype=torch.float64)))
        q = concentration(q, pi, m)
        assert float(weights.square().sum()) == pytest.approx(q)


def test_observed_label_scores_are_negative_loss_gradients_and_not_label_integrated():
    torch.manual_seed(19)
    model = torch.nn.Linear(2, 2).double()
    layout = ParameterLayout.from_module(model)
    theta = layout.flatten_module(model, detach=True).requires_grad_(True)
    x = torch.randn(3, 2, dtype=torch.float64)
    y = torch.tensor([1, 0, 1])
    score = observed_scores(model, layout, theta, x, y)
    loss = torch.nn.functional.cross_entropy(logits(model, layout, theta, x), y)
    gradient = torch.autograd.grad(loss, theta)[0]
    torch.testing.assert_close(score.mean(0), -gradient)
    integrated = sum(logits(model, layout, theta, x).softmax(-1)[:, k, None] *
                     observed_scores(model, layout, theta, x, torch.full_like(y, k)) for k in range(2))
    torch.testing.assert_close(integrated, torch.zeros_like(integrated), atol=1e-14, rtol=0)
    assert observation(model, layout, theta, x).sum() > 0


def test_kalman_update_matches_independent_scalar_matrix_equations():
    controller, diagonal = observer()
    scores = torch.tensor([[1., 2., 3.], [-1., 0., 1.], [0., 1., -2.], [2., -1., 0.]], dtype=torch.float64)
    before = deepcopy(controller.state)
    noise = controller.noise() / 4
    transition = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    process = controller.process_covariance()
    means, covariances = [], []
    for i in range(3):
        predicted = transition @ before["mean"][i]
        prior = transition @ before["covariance"][i] @ transition.T + process[i]
        gain = prior[:, 0] / (prior[0, 0] + noise[i])
        residual = scores[:, i].mean() / diagonal[i] - predicted[0]
        means.append(predicted + gain * residual)
        covariances.append(prior - gain[:, None] @ prior[0, :][None, :])
    controller.assimilate(scores, 0)
    torch.testing.assert_close(controller.state["mean"], torch.stack(means))
    torch.testing.assert_close(controller.state["covariance"], torch.stack(covariances), atol=1e-15, rtol=1e-12)
    assert torch.linalg.eigvalsh(controller.state["covariance"]).min() >= -1e-15


def test_holt_recursion_and_observation_clock_continue_on_holds():
    controller, diagonal = observer("ema", 8.)
    alpha = 1 - 2**(-1 / 8)
    scores = torch.tensor([[1., 1., 1.], [3., 3., 3.]], dtype=torch.float64)
    for step in range(5):
        old = controller.state["mean"].clone()
        measured = scores.mean(0) / diagonal
        level = (1 - alpha) * (old[:, 0] + old[:, 1]) + alpha * measured
        velocity = (1 - alpha) * old[:, 1] + alpha * (level - old[:, 0])
        controller.assimilate(scores, step)
        torch.testing.assert_close(controller.state["mean"], torch.stack((level, velocity), -1))
    assert controller.state["assimilations"] == 5


def test_movement_second_moment_includes_velocity_uncertainty_and_process_zz():
    controller, diagonal = observer()
    controller.state["mean"][:, 1] = 2.
    action = controller.choose(1 / 2048, diagonal, 4, 3)
    mean_energy = float((diagonal * 4).sum())
    uncertainty = float((diagonal * (controller.state["covariance"][:, 1, 1] +
                                    controller.process_covariance()[:, 0, 0])).sum())
    assert action["motion_energy"] == mean_energy
    assert action["uncertainty_energy"] == pytest.approx(uncertainty)
    assert action["rho"] == pytest.approx(4 * (mean_energy + uncertainty) / action["noise_scale"])


def test_reanchoring_preserves_physical_location_velocity_covariance_and_evidence_count():
    controller, diagonal = observer()
    controller.state["mean"][:] = .2
    before = deepcopy(controller.state)
    scores = torch.tensor([[1., 2., 3.], [2., 1., 0.]], dtype=torch.float64)
    shift = torch.full((3,), .1, dtype=torch.float64)
    new_anchor = before["anchor"] + shift
    new_scores = scores - diagonal * shift
    audit = controller.reanchor(new_anchor, diagonal, scores, new_scores, diagonal)
    torch.testing.assert_close(before["anchor"] + before["mean"][:, 0],
                               controller.state["anchor"] + controller.state["mean"][:, 0])
    torch.testing.assert_close(before["mean"][:, 1], controller.state["mean"][:, 1])
    torch.testing.assert_close(before["covariance"], controller.state["covariance"])
    assert controller.state["assimilations"] == 0
    assert audit["mapping_error_energy"] < 1e-28


def test_reanchoring_resets_raw_noise_and_keeps_initial_process_scale_frozen():
    controller, diagonal = observer()
    old = deepcopy(controller.state)
    scores = torch.tensor([[1., 2., 3.], [2., 1., 0.]], dtype=torch.float64)
    changed = 2 * scores
    controller.reanchor(old["anchor"] + .1, 2 * diagonal, scores, changed, diagonal)
    torch.testing.assert_close(controller.state["raw_variance"], changed.var(0, unbiased=True))
    torch.testing.assert_close(controller.state["k0"], old["k0"])


def test_choose_has_no_current_batch_schedule_or_future_input_and_is_read_only():
    assert list(inspect.signature(Controller.choose).parameters) == ["self", "q", "risk_diagonal", "batch_size", "step"]
    controller, diagonal = observer()
    before = deepcopy(controller.state)
    a = controller.choose(.001, diagonal, 4, 3)
    assert a == controller.choose(.001, diagonal, 4, 3)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, controller.state[key])
        else:
            assert value == controller.state[key]


def test_duplicate_assimilation_rejected_and_cold_start_precedes_clipping():
    controller, diagonal = observer()
    assert controller.choose(.1, diagonal, 4, 0)["applied_pi"] == .025
    scores = torch.ones((4, 3), dtype=torch.float64)
    controller.assimilate(scores, 0)
    with pytest.raises(ValueError, match="duplicated or skipped"):
        controller.assimilate(scores, 0)


def test_within_batch_variance_not_uncentered_temporal_score_variation():
    controller, _ = observer()
    controller.state["raw_variance"].zero_()
    for step, constant in enumerate((1., 100., -100.)):
        controller.assimilate(torch.full((4, 3), constant, dtype=torch.float64), step)
    assert torch.count_nonzero(controller.state["raw_variance"]) == 0


def test_single_sample_noise_uses_prior_and_marks_fallback():
    controller, _ = observer()
    variance = controller.state["raw_variance"].clone()
    values = controller.assimilate(torch.ones((1, 3), dtype=torch.float64), 0)
    assert values["measurement_variance_fallback"]
    torch.testing.assert_close(controller.state["raw_variance"], variance)


def test_unsupported_movement_falls_back_without_nonfinite_reporting_numbers():
    controller, diagonal = observer()
    controller.state["mean"][:, 1] = float("inf")
    action = controller.choose(.001, diagonal, 4, 3)
    assert action["fallback"] == "unsupported_movement_or_noise"
    assert action["applied_pi"] == pytest.approx(covariance_action(.001, 4))
    assert action["motion_energy"] is None and action["rho"] is None


def test_huge_finite_movement_saturates_upper_action_not_nan_lower_action():
    controller, diagonal = observer(coordinates=1)
    controller.state["mean"][:, 1] = 1e154
    action = controller.choose(.001, diagonal, 4, 3)
    assert action["applied_pi"] == .2 and action["raw_pi"] == 1.
    assert action["fallback"] == ""
    assert action["rho"] is None
    assert covariance_action(1e308, 4) == 1.


def test_single_sample_refresh_preserves_calibrated_normalized_noise_not_raw_variance():
    controller, diagonal = observer()
    before = controller.noise().clone()
    scores = torch.ones((1, 3), dtype=torch.float64)
    audit = controller.reanchor(controller.state["anchor"] + .1, 2 * diagonal, scores, scores, diagonal)
    torch.testing.assert_close(controller.noise(), before)
    assert audit["measurement_variance_fallback"]


def test_observer_floor_changes_only_mapping_not_fisher():
    controller, _ = observer()
    diagonal = torch.tensor([0., 1e-30, 1.], dtype=torch.float64)
    controller.initialize(torch.zeros(3), diagonal, torch.ones(3))
    torch.testing.assert_close(controller.state["anchor_diagonal"], diagonal)
    assert controller.state["floor_fraction"] == pytest.approx(2 / 3)
    assert (controller.state["mapping"] > 0).all()


def test_welford_calibration_matches_one_pass():
    torch.manual_seed(11)
    values = torch.randn(31, 5, dtype=torch.float64)
    count, mean, m2 = 0, torch.zeros(5), torch.zeros(5)
    for chunk in values.split(4):
        count, mean, m2 = combine_moments(count, mean, m2, chunk)
    assert count == 31
    torch.testing.assert_close(mean, values.mean(0))
    torch.testing.assert_close(m2 / (count - 1), values.var(0, unbiased=True))
