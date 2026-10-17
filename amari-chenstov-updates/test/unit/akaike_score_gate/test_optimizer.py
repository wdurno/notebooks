from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mnist_experiment.akaike_score_gate.budget import WorkBudget
from mnist_experiment.akaike_score_gate.optimizer import fit_restricted
from src.parameters import ParameterLayout
from src.representations import DenseFisher, DiagonalFisher, LowRankDiagonalFisher


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem(dtype=torch.float64):
    model = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=dtype))
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor([[1., 0.], [0.5, 1.], [-1., 0.5]], dtype=dtype)
    targets = torch.tensor([0, 1, 1])
    weights = torch.tensor([1., 2., 7.], dtype=torch.float64)
    anchor = torch.tensor([-0.2, 0.8, 0.1, -0.4], dtype=dtype)
    factor = torch.tensor([[1., 0.3, 0.2, 0.], [0., 1., 0.4, 0.1],
                           [0., 0., 1., 0.5], [0., 0., 0., 1.]], dtype=torch.float64)
    precision = factor @ factor.T
    return model, layout, inputs, targets, weights, anchor, precision


def direct_gradient(model, layout, inputs, targets, weights, anchor, precision):
    data_loss = (nn.functional.cross_entropy(model(inputs), targets, reduction="none").double()
                 @ (weights.double() / weights.sum()))
    gradient = torch.cat([part.reshape(-1) for part in torch.autograd.grad(data_loss, tuple(model.parameters()))]).double()
    displacement = layout.flatten_module(model, detach=True).double() - anchor.double()
    product = precision @ displacement
    return gradient, product, float(data_loss.detach() + 0.5 * displacement @ product)


def test_weighted_gradient_mean_units_and_fixed_old_anchor_cross_penalty() -> None:
    args = problem()
    model, layout, inputs, targets, weights, anchor, precision = args
    indices = torch.tensor([3, 0])
    before = layout.flatten_module(model, detach=True)
    data, penalty, objective = direct_gradient(*args)
    budget = WorkBudget(1, {"fit/depth2/buffer3": 1}, elapsed_arrivals=1)
    result = fit_restricted(*args, indices, budget=budget, charge_key="fit/depth2/buffer3")
    assert result["status"] == "budget_denied"
    assert result["function_evaluations"] == 1
    assert result["retries"] == 0
    torch.testing.assert_close(result["data_gradient"], data[indices])
    torch.testing.assert_close(result["penalty_gradient"], penalty[indices])
    torch.testing.assert_close(result["retained_gradient"], (data + penalty)[indices])
    assert result["objective_before"] == pytest.approx(objective)
    assert result["objective_after"] == pytest.approx(objective)
    assert not torch.allclose(precision[indices][:, indices] @ (before - anchor)[indices], penalty[indices])
    assert torch.equal(layout.flatten_module(model, detach=True), before)
    assert result["restricted_residual_norm"] == pytest.approx(float(torch.linalg.vector_norm((data + penalty)[indices])))


def test_restricted_fit_converges_freezes_other_coordinates_and_keeps_anchor() -> None:
    args = problem()
    model, layout, _, _, _, anchor, _ = args
    anchor_before = anchor.clone()
    before = layout.flatten_module(model, detach=True)
    indices = torch.tensor([3, 0])
    result = fit_restricted(*args, indices, tolerance_grad=1e-10,
                            tolerance_change=1e-15, residual_tolerance=1e-7)
    assert result["converged"]
    assert result["status"] == "converged"
    assert result["objective_after"] < result["objective_before"]
    assert torch.equal(result["after"][[1, 2]], before[[1, 2]])
    assert torch.equal(anchor, anchor_before)
    assert not torch.equal(result["after"][indices], before[indices])
    data, penalty, objective = direct_gradient(*args)
    torch.testing.assert_close(result["retained_gradient"], (data + penalty)[indices], atol=1e-13, rtol=1e-12)
    assert result["objective_after"] == pytest.approx(objective)
    assert result["restricted_residual_max"] <= 1e-7
    assert torch.equal(layout.flatten_module(model, detach=True), result["after"])


class ConstantLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.tensor([0.8, 1.3], dtype=torch.float64))

    def forward(self, inputs):
        return inputs.new_zeros((len(inputs), 2))


def test_quadratic_only_fit_has_exact_cross_block_solution() -> None:
    model = ConstantLogits()
    layout = ParameterLayout.from_module(model)
    anchor = torch.tensor([0.1, -0.4], dtype=torch.float64)
    precision = torch.tensor([[2., 0.6], [0.6, 1.]], dtype=torch.float64)
    before = layout.flatten_module(model, detach=True)
    result = fit_restricted(model, layout, torch.ones((2, 1), dtype=torch.float64),
                            torch.tensor([0, 1]), torch.tensor([0.2, 0.8]),
                            anchor, precision, torch.tensor([0]))
    expected = anchor[0] - precision[0, 1] * (before[1] - anchor[1]) / precision[0, 0]
    assert result["converged"]
    assert result["after"][0] == pytest.approx(float(expected), abs=1e-10)
    assert result["after"][1] == before[1]
    assert result["data_gradient"].count_nonzero() == 0


def test_weight_rescaling_does_not_change_fit() -> None:
    first = problem()
    second = list(problem())
    second[4] = second[4] * 5
    kwargs = {"tolerance_grad": 1e-9, "tolerance_change": 1e-14, "residual_tolerance": 1e-7}
    one = fit_restricted(*first, torch.arange(4), **kwargs)
    two = fit_restricted(*second, torch.arange(4), **kwargs)
    torch.testing.assert_close(one["after"], two["after"], atol=1e-12, rtol=1e-12)
    assert one["function_evaluations"] == two["function_evaluations"]


def test_denied_first_evaluation_has_no_hidden_forward_or_backward() -> None:
    args = problem()
    model, layout = args[:2]
    calls = []
    handle = model.register_forward_hook(lambda *unused: calls.append(1))
    before = layout.flatten_module(model, detach=True)
    budget = WorkBudget(1, {"fit": 1})
    result = fit_restricted(*args, torch.arange(4), budget=budget, charge_key="fit")
    handle.remove()
    assert result["budget_denied"]
    assert result["function_evaluations"] == 0
    assert result["retained_gradient"] is None
    assert result["restricted_residual_norm"] is None
    assert result["objective_before"] is None
    assert calls == []
    assert budget.spent == 0
    assert torch.equal(before, layout.flatten_module(model, detach=True))


def test_budget_stop_preserves_last_completed_step_not_wolfe_trial() -> None:
    reference_args = problem()
    reference_budget = WorkBudget(2, {"fit": 1}, elapsed_arrivals=1)
    reference = fit_restricted(*reference_args, torch.arange(4), budget=reference_budget, charge_key="fit")
    assert reference["accepted_steps"] == 1
    args = problem()
    model = args[0]
    calls = []
    hook = model.register_forward_hook(lambda *unused: calls.append(1))
    budget = WorkBudget(3, {"fit": 1}, elapsed_arrivals=1)
    assert budget.reserve("end", "fit")
    result = fit_restricted(*args, torch.arange(4), budget=budget, charge_key="fit")
    hook.remove()
    assert result["budget_denied"]
    assert result["accepted_steps"] == 1
    assert result["charged_work"] == result["function_evaluations"] == len(calls) == 2
    torch.testing.assert_close(result["after"], reference["after"], atol=0, rtol=0)
    data, penalty, _ = direct_gradient(*args)
    torch.testing.assert_close(result["retained_gradient"], data + penalty)
    assert budget.spend_reserved("end")
    assert budget.spent == 3


def test_denial_during_line_search_does_not_accept_finite_rejected_trial() -> None:
    model = ConstantLogits()
    layout = ParameterLayout.from_module(model)
    with torch.no_grad():
        model.vector[0] = 0.1
    before = layout.flatten_module(model, detach=True)
    budget = WorkBudget(2, {"fit": 1}, elapsed_arrivals=1)
    result = fit_restricted(model, layout, torch.ones((1, 1), dtype=torch.float64),
                            torch.tensor([0]), torch.ones(1), torch.zeros(2),
                            torch.tensor([1000., 1.]), torch.tensor([0]),
                            budget=budget, charge_key="fit")
    assert result["budget_denied"]
    assert result["function_evaluations"] == 2
    assert result["accepted_steps"] == 0
    assert result["objective_after"] == result["objective_before"]
    assert torch.equal(result["after"], before)
    assert torch.equal(layout.flatten_module(model, detach=True), before)
    torch.testing.assert_close(result["retained_gradient"], torch.tensor([100.], dtype=torch.float64))


def test_unresolved_retry_is_exactly_twice_the_ceiling_and_charged() -> None:
    args = problem()
    budget = WorkBudget(None, {"fit": 0.25})
    result = fit_restricted(*args, torch.arange(4), max_iter=1, max_eval=2,
                            tolerance_grad=0, tolerance_change=0, residual_tolerance=0,
                            budget=budget, charge_key="fit")
    assert result["status"] == "restricted_fit_unresolved"
    assert not result["budget_denied"]
    assert result["retries"] == 1
    assert [row["max_iter"] for row in result["attempts"]] == [1, 2]
    assert [row["max_eval"] for row in result["attempts"]] == [2, 4]
    assert all(row["function_evaluations"] <= row["max_eval"] for row in result["attempts"])
    assert result["function_evaluations"] <= 6
    assert result["charged_work"] == budget.spent == 0.25 * result["function_evaluations"]


def test_stopping_tolerance_is_not_a_stationarity_certificate() -> None:
    result = fit_restricted(*problem(), torch.arange(4), tolerance_grad=1e6, residual_tolerance=1e-10)
    assert result["status"] == "restricted_fit_unresolved"
    assert result["retries"] == 1
    assert result["function_evaluations"] == 1
    assert result["restricted_residual_max"] > 1e-10


def test_new_local_fit_resets_history_and_resumed_budget_does_not_refill() -> None:
    first = problem()
    budget = WorkBudget(5, {"fit": 1}, elapsed_arrivals=1)
    fit_restricted(*first, torch.arange(4), max_iter=1, max_eval=2,
                   budget=budget, charge_key="fit")
    second = list(first)
    second[0] = copy.deepcopy(first[0])
    resumed = WorkBudget.from_state_dict(budget.state_dict())
    budget.grant_arrival()
    resumed.grant_arrival()
    result1 = fit_restricted(*first, torch.arange(4), budget=budget, charge_key="fit")
    result2 = fit_restricted(*second, torch.arange(4), budget=resumed, charge_key="fit")
    assert budget.state_dict() == resumed.state_dict()
    assert result1["function_evaluations"] == result2["function_evaluations"]
    torch.testing.assert_close(result1["after"], result2["after"], rtol=0, atol=0)


class SplitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Linear(2, 3, dtype=torch.float64)
        self.output = nn.Linear(3, 2, dtype=torch.float64)
        self.register_buffer("offset", torch.ones(3, dtype=torch.float64))
        self.trace = []

    def forward(self, inputs):
        self.trace.append((self.prefix.weight.requires_grad, self.prefix.bias.requires_grad,
                           self.output.weight.requires_grad, self.output.bias.requires_grad))
        hidden = self.prefix(inputs) + self.offset
        return self.output(hidden.tanh())


def test_frozen_prefix_has_no_autograd_path_and_model_gradients_are_untouched() -> None:
    model = SplitClassifier()
    layout = ParameterLayout.from_module(model)
    before = layout.flatten_module(model, detach=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradient_before = [p.grad.clone() for p in model.parameters()]
    inputs = torch.ones((3, 2), dtype=torch.float64, requires_grad=True)
    result = fit_restricted(model, layout, inputs, torch.tensor([0, 1, 0]), torch.ones(3),
                            before, torch.eye(before.numel(), dtype=torch.float64),
                            torch.arange(9, before.numel()))
    assert model.trace
    assert all(row == (False, False, True, True) for row in model.trace)
    assert len(model.trace) == result["function_evaluations"]
    assert torch.equal(result["after"][:9], before[:9])
    assert inputs.grad is None
    assert torch.equal(model.offset, torch.ones_like(model.offset))
    for parameter, gradient in zip(model.parameters(), gradient_before):
        assert parameter.requires_grad
        assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize("kind", ["dense", "diagonal", "low_rank"])
def test_precision_representations_and_float64_accumulation(kind: str) -> None:
    args = list(problem(dtype=torch.float32))
    if kind == "dense":
        precision = DenseFisher(args[-1])
    elif kind == "diagonal":
        precision = DiagonalFisher(args[-1].diagonal())
    else:
        precision = LowRankDiagonalFisher(torch.ones((4, 1), dtype=torch.float64) * 0.1,
                                         torch.ones(4, dtype=torch.float64))
    args[-1] = precision
    result = fit_restricted(*args, torch.arange(4))
    assert result["data_gradient"].dtype == torch.float64
    assert result["retained_gradient"].dtype == torch.float64
    assert result["after"].dtype == torch.float32
    assert result["objective_after"] < result["objective_before"]


class NonfiniteTrial(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, inputs):
        value = self.value.expand(len(inputs))
        if bool(value.detach().abs().max() > 0.01):
            value = value * float("inf")
        return torch.stack((value, torch.zeros_like(value)), dim=1)


def test_nonfinite_trial_restores_finite_accepted_state() -> None:
    model = NonfiniteTrial()
    layout = ParameterLayout.from_module(model)
    before = layout.flatten_module(model, detach=True)
    budget = WorkBudget(None, {"fit": 1})
    result = fit_restricted(model, layout, torch.ones((1, 1), dtype=torch.float64),
                            torch.tensor([0]), torch.ones(1), before, torch.ones(1),
                            torch.tensor([0]), budget=budget, charge_key="fit")
    assert result["status"] == "restricted_fit_unresolved"
    assert not result["budget_denied"]
    assert result["initial_gradient"] is not None
    assert result["function_evaluations"] == 3
    assert result["retries"] == 1
    assert torch.equal(layout.flatten_module(model, detach=True), before)
    assert torch.isfinite(result["after"]).all()
    assert result["charged_work"] == budget.spent == result["function_evaluations"]


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_plan14_model_accepts_all_six_suffixes_with_charged_restricted_gradients(depth: int) -> None:
    from mnist_experiment.akaike_score_gate.model import active_indices, build_model

    model, layout = build_model(123, dtype=torch.float64)
    indices, _ = active_indices(layout, depth)
    before = layout.flatten_module(model, detach=True)
    anchor = before - 0.1
    inputs = torch.linspace(-1, 1, 2 * 28 * 28, dtype=torch.float64).reshape(2, 1, 28, 28)
    targets, weights = torch.tensor([1, 9]), torch.tensor([1., 3.], dtype=torch.float64)
    precision = torch.eye(before.numel(), dtype=torch.float64) * 0.2
    precision += torch.ones_like(precision) * 0.001
    budget = WorkBudget(1, {f"fit/depth{depth}": 1}, elapsed_arrivals=1)
    result = fit_restricted(model, layout, inputs, targets, weights, anchor, precision,
                            indices, budget=budget, charge_key=f"fit/depth{depth}")
    data, penalty, _ = direct_gradient(model, layout, inputs, targets, weights, anchor, precision)
    assert result["function_evaluations"] == result["forward_evaluations"] == result["gradient_evaluations"] == 1
    torch.testing.assert_close(result["data_gradient"], data[indices], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result["retained_gradient"], (data + penalty)[indices], atol=1e-12, rtol=1e-12)
    assert torch.equal(result["after"], before)


def test_materially_asymmetric_precision_is_rejected() -> None:
    args = list(problem())
    args[-1][0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        fit_restricted(*args, torch.arange(4))


@pytest.mark.parametrize("indices", [torch.tensor([], dtype=torch.long), torch.tensor([0, 0]),
                                     torch.tensor([-1]), torch.tensor([4]), torch.tensor([0.])])
def test_invalid_selection_fails(indices: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="indices"):
        fit_restricted(*problem(), indices)


@pytest.mark.parametrize("weights", [torch.zeros(3), torch.tensor([1., -1., 2.]),
                                     torch.tensor([1., float("nan"), 2.]), torch.ones(2)])
def test_invalid_weights_fail(weights: torch.Tensor) -> None:
    args = list(problem())
    args[4] = weights
    with pytest.raises(ValueError):
        fit_restricted(*args, torch.arange(4))


def test_budgeted_fit_requires_known_explicit_charge_key() -> None:
    budget = WorkBudget(None, {"fit": 1})
    with pytest.raises(ValueError, match="charge_key"):
        fit_restricted(*problem(), torch.arange(4), budget=budget)
    with pytest.raises(KeyError):
        fit_restricted(*problem(), torch.arange(4), budget=budget, charge_key="unknown")
    assert budget.spent == 0
