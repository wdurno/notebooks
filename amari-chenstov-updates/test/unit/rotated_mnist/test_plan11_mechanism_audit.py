"""Fast checks for Plan 11's artifact-only mechanism summaries."""

from __future__ import annotations

import numpy as np
import pytest

from mnist_experiment.rotated_mnist.plan11.mechanism_audit import (
    _lag_correlations,
    _sample_summary,
    normalized_auc,
)


def test_normalized_auc_uses_exposure_spacing() -> None:
    assert normalized_auc([0, 1, 3], [0.0, 2.0, 2.0]) == pytest.approx(5 / 3)
    with pytest.raises(ValueError, match="exposure domain"):
        normalized_auc([0, 1, 1], [0.0, 1.0, 2.0])


def test_sample_summary_keeps_mean_estimand_and_sign_count_distinct() -> None:
    summary = _sample_summary([-1.0, -1.0, 5.0])
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["median"] == pytest.approx(-1.0)
    assert summary["positive_count"] == 1


def test_lag_correlations_align_action_with_following_evaluation() -> None:
    action = np.arange(120, dtype=float)
    outcome = np.arange(121, dtype=float)
    rows = _lag_correlations(action, outcome)
    assert len(rows) == 21
    assert rows[0] == {"lag_updates": 0, "correlation": pytest.approx(1.0)}
    assert rows[-1]["correlation"] == pytest.approx(1.0)
