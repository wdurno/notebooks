import pytest
import torch

from src.coupled_trajectory import (
    COUPLED_DENSE_METHODS,
    DenseFisherTracker,
)
from src.fisher import LFUBatchEstimate
from src.fixed_trajectory import OnlineStepStatistics, replay_dense_conditions


def _statistics() -> list[OnlineStepStatistics]:
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    return [
        OnlineStepStatistics(
            LFUBatchEstimate(torch.eye(2, dtype=torch.float64), zeros, zeros),
            score_gradient_count=2,
            hvp_count=0,
            elapsed_seconds=0.0,
        ),
        OnlineStepStatistics(
            LFUBatchEstimate(
                torch.diag(torch.tensor([1.2, 0.8], dtype=torch.float64)),
                torch.diag(torch.tensor([0.1, -0.02], dtype=torch.float64)),
                torch.diag(torch.tensor([0.03, 0.01], dtype=torch.float64)),
            ),
            score_gradient_count=2,
            hvp_count=2,
            elapsed_seconds=0.0,
        ),
        OnlineStepStatistics(
            LFUBatchEstimate(
                torch.diag(torch.tensor([1.4, 0.7], dtype=torch.float64)),
                torch.diag(torch.tensor([0.08, -0.01], dtype=torch.float64)),
                torch.diag(torch.tensor([0.02, 0.02], dtype=torch.float64)),
            ),
            score_gradient_count=2,
            hvp_count=2,
            elapsed_seconds=0.0,
        ),
    ]


def test_coupled_trackers_match_fixed_replay_on_a_shared_path() -> None:
    initial = torch.eye(2, dtype=torch.float64)
    references = [
        initial,
        torch.diag(torch.tensor([1.25, 0.85], dtype=torch.float64)),
        torch.diag(torch.tensor([1.5, 0.75], dtype=torch.float64)),
    ]
    directions = [
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.1, -0.05], dtype=torch.float64),
        torch.tensor([0.08, -0.04], dtype=torch.float64),
    ]
    statistics = _statistics()
    replay = replay_dense_conditions(
        initial,
        references,
        statistics,
        directions,
        (0.0, 0.5, 1.0),
        ema_gain=0.25,
        fresh_fisher_cadence=2,
        ridge_half_life_steps=2.0,
        ridge_amplitude_epsilon=1e-6,
        ridge_coherence_threshold=0.75,
    )

    for method in COUPLED_DENSE_METHODS:
        tracker = DenseFisherTracker(
            method,
            initial,
            ema_gain=0.25,
            fresh_fisher_cadence=2,
            ridge_half_life_steps=2.0,
            ridge_amplitude_epsilon=1e-6,
            ridge_coherence_threshold=0.75,
        )
        updates = [
            tracker.update(step, statistic, direction, reference)
            for step, (statistic, direction, reference) in enumerate(
                zip(statistics, directions, references, strict=True)
            )
        ]
        for update, expected in zip(
            updates,
            replay[method].estimates,
            strict=True,
        ):
            torch.testing.assert_close(update.estimate, expected)


def test_coupled_tracker_requires_sequential_realized_directions() -> None:
    tracker = DenseFisherTracker(
        "full_lfu",
        torch.eye(2, dtype=torch.float64),
        ema_gain=0.25,
        fresh_fisher_cadence=2,
    )

    with pytest.raises(ValueError, match="expected coupled step 0"):
        tracker.update(
            1,
            _statistics()[0],
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )
