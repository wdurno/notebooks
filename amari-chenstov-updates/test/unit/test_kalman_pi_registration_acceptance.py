"""Bounded, already-seen-data closeout diagnostics are not training."""

import math

import pytest
import torch

from mnist_experiment.kalman_pi.registration.acceptance import closeout, noise_comparison
from mnist_experiment.kalman_pi.registration.config import Config
from mnist_experiment.diagonal_lfu.report import _atomic, _json_bytes


def test_noise_audit_reports_nested_correlated_reference_not_truth():
    result = noise_comparison(torch.tensor([[0., 0.], [2., 4.]]),
        torch.tensor([4., 4.]), torch.tensor([1., 2.]), Config())
    assert result["nested_count"] == 2
    assert result["raw_trace_ratio"] == pytest.approx(1.25)
    assert result["fisher_weighted_noise_ratio"] == pytest.approx(1.)
    assert result["diagonal_relative_l2"] == pytest.approx(math.sqrt(20 / 32))
    assert result["additional_training_labels"] == 0
    assert result["reference_is_truth"] is False


@pytest.mark.parametrize("invalid", ["shape", "negative_variance", "negative_fisher", "nan"])
def test_invalid_noise_audit_inputs_fail(invalid):
    scores, variance, diagonal = torch.ones(2, 2), torch.ones(2), torch.ones(2)
    if invalid == "shape":
        scores = scores[:1]
    elif invalid == "negative_variance":
        variance[0] = -1
    elif invalid == "negative_fisher":
        diagonal[0] = -1
    else:
        scores[0, 0] = float("nan")
    with pytest.raises(ValueError):
        noise_comparison(scores, variance, diagonal, Config())


def test_zero_reference_noise_is_undefined_not_an_infinite_ratio():
    result = noise_comparison(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2), Config())
    assert result["raw_trace_ratio"] is None
    assert result["fisher_weighted_noise_ratio"] is None
    assert result["diagonal_relative_l2"] is None


def test_closeout_refuses_incomplete_study_before_any_allocation_write(tmp_path):
    _atomic(tmp_path / "identity.json", _json_bytes({"identity": {"config": Config().mapping()}}))
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="completed study"):
        closeout(tmp_path, tmp_path / "missing-dataset")
    assert set(tmp_path.iterdir()) == before
