import math

import pytest
import torch
from torch import nn

from mnist_experiment.akaike_score_gate import evaluation


class LookupModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float64))
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        return self.logits[inputs[:, 0, 0, 0].long()]


def panel(logits, targets):
    inputs = torch.zeros(len(targets), 1, 28, 28, dtype=torch.float64)
    inputs[:, 0, 0, 0] = torch.arange(len(targets))
    return LookupModel(logits), inputs, torch.tensor(targets)


def test_mixture_conditional_weighting_argmax_and_standardized_precision():
    logits = torch.zeros(5, 10, dtype=torch.float64)
    logits[0, 0] = 3
    logits[1, 9] = 3
    logits[2, 0] = 3
    logits[3, 9] = 3
    logits[4, 0] = 3
    model, x, y = panel(logits, [0, 0, 1, 9, 9])
    row = evaluation.evaluate(model, x, y, family="mixture", position=0.25, batch_size=2)
    losses = nn.functional.cross_entropy(logits, y, reduction="none")
    assert row["current_nll"] == pytest.approx(0.75 * losses[:3].mean() + 0.25 * losses[3:].mean())
    assert row["current_accuracy"] == pytest.approx(0.75 / 3 + 0.25 / 2)
    assert row["retention_nll"] == pytest.approx(losses[:3].mean())
    assert row["fixed_nll"] == pytest.approx(0.9 * losses[:3].mean() + 0.1 * losses[3:].mean())
    assert (row["nine_tp"], row["nine_fn"], row["nine_fp"], row["nine_tn"]) == (1, 1, 1, 2)
    assert row["nine_precision"] == pytest.approx(0.05 / (0.05 + 0.9 / 3))
    assert row["nine_precision_defined"] is True
    assert row["nine_recall"] == 0.5
    assert row["nine_false_positive_rate"] == pytest.approx(1 / 3)
    assert max(model.batch_sizes) <= 2
    assert model.training is True
    assert all(not isinstance(v, torch.Tensor) for v in row.values())


def test_no_predicted_nines_is_undefined_not_zero():
    logits = torch.zeros(4, 10)
    logits[:, 0] = 2
    model, x, y = panel(logits, [0, 1, 9, 9])
    row = evaluation.evaluate(model, x, y, family="mixture", position=0)
    assert row["nine_precision"] is None
    assert row["nine_precision_defined"] is False
    assert row["nine_recall"] == row["nine_false_positive_rate"] == 0
    assert row["nine_fn"] == row["nine_tn"] == 2
    assert row["current_nll"] == row["retention_nll"]


def test_defined_zero_precision_is_not_undefined():
    logits = torch.zeros(2, 10)
    logits[0, 9] = 2
    logits[1, 0] = 2
    model, x, y = panel(logits, [0, 9])
    row = evaluation.evaluate(model, x, y, family="mixture", position=0.1)
    assert row["nine_precision"] == 0
    assert row["nine_precision_defined"] is True


def test_pr_curve_ties_and_standardized_average_precision():
    model, x, y = panel(torch.zeros(4, 10), [0, 1, 9, 9])
    row = evaluation.evaluate(model, x, y, family="mixture", position=0.1, return_pr_curve=True)
    curve = row["nine_pr_curve"]
    assert row["nine_average_precision"] == pytest.approx(0.1)
    assert len(curve["thresholds"]) == 2
    assert curve["precision_defined"].tolist() == [False, True]
    assert math.isnan(curve["precision"][0])
    assert curve["precision"][1] == pytest.approx(0.1)
    assert curve["recall"].tolist() == [0, 1]
    assert all(value.device.type == "cpu" for value in curve.values())


def test_perfect_pr_ranking_has_unit_average_precision():
    logits = torch.zeros(4, 10)
    logits[:2, 0] = 3
    logits[2:, 9] = 3
    model, x, y = panel(logits, [0, 0, 9, 9])
    row = evaluation.evaluate(model, x, y, family="mixture", position=0.1)
    assert row["nine_average_precision"] == pytest.approx(1)
    assert row["nine_precision"] == 1


def test_rotation_panels_use_requested_upright_and_fixed_angles_in_batches(monkeypatch):
    class AngleModel(nn.Module):
        def forward(self, batch):
            logits = torch.zeros(len(batch), 10, dtype=batch.dtype)
            logits[:, 0] = batch[:, 0, 0, 0]
            return logits

    calls = []

    def rotate(batch, angle, config):
        calls.append((len(batch), angle))
        return batch + angle / 30

    monkeypatch.setattr(evaluation, "rotate_mnist_batch", rotate)
    model = AngleModel()
    model.eval()
    x, y = torch.zeros(5, 1, 28, 28), torch.zeros(5, dtype=torch.long)
    row = evaluation.evaluate(model, x, y, family="rotation", position=15, batch_size=2)
    assert row["fixed_nll"] < row["current_nll"] < row["retention_nll"]
    assert row["retention_nll"] == pytest.approx(math.log(10))
    assert {angle for _, angle in calls} == {15, 30}
    assert max(size for size, _ in calls) <= 2
    assert not model.training
    calls.clear()
    evaluation.evaluate(model, x, y, family="rotation", position=30, batch_size=2)
    assert len(calls) == 3


def test_real_rotation_is_batch_invariant():
    class PixelModel(nn.Module):
        def forward(self, batch):
            logits = torch.zeros(len(batch), 10)
            logits[:, 0] = batch[:, :, :14, :14].mean((1, 2, 3))
            return logits

    x = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(4))
    y = torch.tensor([0, 1, 9, 0])
    a = evaluation.evaluate(PixelModel(), x, y, family="rotation", position=12, batch_size=1)
    b = evaluation.evaluate(PixelModel(), x, y, family="rotation", position=12, batch_size=4)
    for key in ("current_nll", "retention_nll", "fixed_nll"):
        assert a[key] == pytest.approx(b[key], abs=1e-7)


def test_source_quality_and_invalid_evaluation_restore_model_mode():
    model, x, y = panel(torch.zeros(2, 10), [0, 1])
    assert evaluation.source_quality(model, x, y)["nll"] == pytest.approx(math.log(10))
    with pytest.raises(ValueError, match="both digit"):
        evaluation.evaluate(model, x, y, family="mixture", position=0)
    with pytest.raises(ValueError, match="position"):
        evaluation.evaluate(model, x, y, family="mixture", position=-0.1)
    with pytest.raises(ValueError, match="batch_size"):
        evaluation.evaluate(model, x, y, family="rotation", position=0, batch_size=0)
    with torch.no_grad():
        model.logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite logits"):
        evaluation.evaluate(model, x, y, family="rotation", position=0)
    assert model.training
