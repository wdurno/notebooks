import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.initialization import evaluate_classifier
from src.classification_metrics import nine_environment_metrics


def _dataset() -> TensorDataset:
    logits = torch.full((10, 10), -1.0)
    targets = torch.arange(10)
    logits[torch.arange(10), targets] = 2.0
    logits[9, 9] = -2.0
    logits[9, 0] = 3.0
    return TensorDataset(logits, targets)


def test_evaluate_classifier_calibration_is_opt_in_and_matches_logits() -> None:
    model = nn.Identity()
    dataset = _dataset()
    plain = evaluate_classifier(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    calibrated = evaluate_classifier(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        calibration_bins=5,
    )

    assert "brier" not in plain
    logits, targets = dataset.tensors
    probabilities = torch.softmax(logits, dim=1).double()
    one_hot = nn.functional.one_hot(targets, num_classes=10).double()
    brier = (probabilities - one_hot).square().sum(dim=1)
    assert calibrated["brier"] == pytest.approx(float(brier.mean()))
    assert calibrated["non_nine_brier"] == pytest.approx(
        float(brier[:9].mean())
    )
    assert calibrated["nine_brier"] == pytest.approx(float(brier[9]))
    assert calibrated["expected_calibration_error"] > 0.0
    assert calibrated["calibration_bin_count"] == 5


def test_evaluate_classifier_records_prevalence_adjusted_nine_metrics() -> None:
    logits = torch.full((10, 10), -2.0)
    targets = torch.arange(10)
    logits[torch.arange(10), targets] = 2.0
    logits[0, 0] = -2.0
    logits[0, 9] = 3.0
    logits[1, 1] = -2.0
    logits[1, 2] = 3.0
    metrics = evaluate_classifier(
        nn.Identity(),
        TensorDataset(logits, targets),
        batch_size=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
        nine_prevalence=0.5,
    )

    assert metrics["nine_true_positive_count"] == 1
    assert metrics["nine_false_positive_count"] == 1
    assert metrics["nine_true_negative_count"] == 8
    assert metrics["nine_false_negative_count"] == 0
    assert metrics["nine_recall"] == 1.0
    assert metrics["nine_precision"] == pytest.approx(0.9)
    assert metrics["nine_ovr_accuracy"] == pytest.approx(17 / 18)
    assert metrics["environment_accuracy"] == pytest.approx(8 / 9)


def test_nine_precision_is_missing_when_no_positive_prediction_is_possible() -> None:
    metrics = nine_environment_metrics(
        prevalence=0.0,
        recall=0.0,
        false_positive_rate=0.0,
        non_nine_accuracy=1.0,
    )

    assert metrics["nine_precision"] is None


@pytest.mark.parametrize("bins", [True, 0, 1, 2.5])
def test_evaluate_classifier_rejects_invalid_calibration_bins(bins) -> None:
    with pytest.raises(ValueError, match="calibration_bins"):
        evaluate_classifier(
            nn.Identity(),
            _dataset(),
            batch_size=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            calibration_bins=bins,
        )


@pytest.mark.parametrize("prevalence", [True, -0.1, 1.1, float("nan")])
def test_evaluate_classifier_rejects_invalid_nine_prevalence(prevalence) -> None:
    with pytest.raises(ValueError, match="nine_prevalence"):
        evaluate_classifier(
            nn.Identity(),
            _dataset(),
            batch_size=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            nine_prevalence=prevalence,
        )
