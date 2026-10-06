import math

import torch

from mnist_experiment.continual_subgd.conditions import (
    adaptive_floor_conditions,
    default_probe_conditions,
    geometry_rate_conditions,
)
from mnist_experiment.continual_subgd.environments.digit9_mixture import (
    evaluate_mixture,
    mixture_data_config,
    nine_ovr_metrics,
)


class _IndexedLogits(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(inputs.shape[0], 10, device=inputs.device)
        predicted = inputs[:, 0, 0, 0].to(torch.long)
        logits.scatter_(1, predicted[:, None], 2.0)
        return logits


def test_mixture_evaluation_interpolates_conditional_risks() -> None:
    model = _IndexedLogits()
    inputs = torch.tensor([0.0, 2.0, 9.0, 1.0]).reshape(4, 1, 1, 1)
    targets = torch.tensor([0, 1, 9, 9])
    left = evaluate_mixture(model, inputs, targets, 0.0)
    middle = evaluate_mixture(model, inputs, targets, 0.25)
    right = evaluate_mixture(model, inputs, targets, 1.0)
    assert left["current_nll"] == left["p0_nll"]
    assert right["current_nll"] == right["p1_nll"]
    assert math.isclose(
        middle["current_nll"],
        0.75 * left["p0_nll"] + 0.25 * right["p1_nll"],
    )
    assert math.isclose(
        middle["current_accuracy"],
        0.75 * left["p0_accuracy"] + 0.25 * right["p1_accuracy"],
    )
    assert math.isclose(middle["nine_ovr_recall"], 0.5)
    assert math.isclose(middle["nine_ovr_specificity"], 1.0)
    assert math.isclose(middle["nine_ovr_accuracy"], 0.875)
    assert math.isclose(middle["nine_ovr_balanced_accuracy"], 0.75)


def test_nine_ovr_accuracy_reweights_conditional_binary_accuracy() -> None:
    predictions = torch.tensor([0, 9, 9, 1, 9, 8])
    targets = torch.tensor([0, 1, 9, 9, 9, 2])
    result = nine_ovr_metrics(predictions, targets, p=0.25)
    assert math.isclose(result["nine_ovr_recall"], 2 / 3)
    assert math.isclose(result["nine_ovr_specificity"], 2 / 3)
    assert math.isclose(result["nine_ovr_accuracy"], 2 / 3)
    assert math.isclose(result["nine_ovr_balanced_accuracy"], 2 / 3)
    assert math.isclose(result["nine_ovr_false_positive_rate"], 1 / 3)


def test_mixture_contract_and_axial_controller_candidates() -> None:
    production = mixture_data_config(False)
    assert production.num_p_steps == 100
    assert production.samples_per_step == 8
    assert production.non_nine_sampling == "empirical"
    controller = default_probe_conditions()[0].controller
    assert controller is not None
    floors = adaptive_floor_conditions(controller)
    assert [condition.epsilon for condition in floors] == [0.0, 0.01, 0.05, 0.10]
    geometry = geometry_rate_conditions(controller)
    assert all(condition.controller.alpha_scale == controller.alpha_scale for condition in geometry)
    assert len({condition.controller.beta_scale for condition in geometry}) == 3
