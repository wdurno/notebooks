from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mnist_experiment.run_hybrid import (
    _controller_state_from_mapping,
    _controller_state_mapping,
)
from src.config import OptimizerConfig
from src.controller import ControllerState
from src.directional_ridge import DirectionalRidgeLFUState
from src.ewc import build_optimizer, take_ewc_proposal
from src.fisher import LFUBatchEstimate
from src.hybrid import (
    HybridArchiveState,
    blend_archive_fisher,
    load_initial_archive_source,
    stage_hybrid_replay_transition,
    update_archive_fisher_lfu,
)
from src.parameters import ParameterLayout
from src.replay import FifoReplayBuffer, ReplayEvent, stream_events
from src.representations import LowRankDiagonalFisher


def _events(step: int, count: int = 2) -> tuple[ReplayEvent, ...]:
    return stream_events(
        step,
        tuple(range(step * count, (step + 1) * count)),
        tuple(index % 2 for index in range(count)),
        samples_per_step=count,
    )


def test_hybrid_transition_conserves_events_and_archives_only_evictions() -> None:
    replay = FifoReplayBuffer(capacity=3)
    replay.insert(_events(0))

    transition = stage_hybrid_replay_transition(replay, _events(1), ())

    assert [event.event_id for event in replay.events] == [0, 1]
    assert [event.event_id for event in transition.evicted] == [0]
    assert [event.event_id for event in transition.replay.events] == [1, 2, 3]
    assert transition.archived_event_ids == frozenset({0})

    next_transition = stage_hybrid_replay_transition(
        transition.replay,
        _events(2),
        transition.archived_event_ids,
    )
    assert next_transition.archived_event_ids == frozenset({0, 1, 2})
    assert not next_transition.archived_event_ids.intersection(
        event.event_id for event in next_transition.replay.events
    )


def test_unbounded_hybrid_transition_leaves_archive_unchanged() -> None:
    replay = FifoReplayBuffer(capacity=None)
    replay.insert(_events(0))
    archived = frozenset({99})

    transition = stage_hybrid_replay_transition(replay, _events(1), archived)

    assert transition.evicted == ()
    assert transition.archived_event_ids == archived
    assert [event.event_id for event in transition.replay.events] == [0, 1, 2, 3]


def test_hybrid_transition_rejects_active_archive_overlap() -> None:
    replay = FifoReplayBuffer(capacity=2)
    replay.insert(_events(0))

    with pytest.raises(ValueError, match="overlap"):
        stage_hybrid_replay_transition(replay, _events(1), {0})


def test_archive_fisher_blend_preserves_psd_structure_and_accounting() -> None:
    previous = LowRankDiagonalFisher(
        factor=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
        residual_diagonal=torch.tensor([0.2, 0.3], dtype=torch.float64),
    )
    fresh = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float64)

    update = blend_archive_fisher(
        previous,
        fresh,
        blend_gain=0.25,
        rank=2,
        lanczos_seed=7,
    )

    expected = 0.75 * previous.to_dense() + 0.25 * fresh
    torch.testing.assert_close(update.representation.to_dense(), expected)
    assert update.candidate_trace == pytest.approx(float(torch.trace(expected)))
    assert update.representation.rank == 2


def test_archive_full_lfu_projects_and_records_ridge_diagnostics() -> None:
    previous = LowRankDiagonalFisher(
        factor=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        residual_diagonal=torch.tensor([0.1, 0.1], dtype=torch.float64),
    )
    estimate = LFUBatchEstimate(
        fisher=torch.eye(2, dtype=torch.float64),
        amari_chentsov=torch.tensor(
            [[-20.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        ),
        residual=torch.zeros((2, 2), dtype=torch.float64),
    )
    ridge = DirectionalRidgeLFUState(
        half_life_steps=8.0,
        amplitude_epsilon=1e-6,
        coherence_threshold=0.75,
    )

    update = update_archive_fisher_lfu(
        previous,
        estimate,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        ridge,
        correction_method="full_lfu",
        blend_gain=0.05,
        rank=2,
        lanczos_seed=9,
    )

    assert update.projection.minimum_eigenvalue < 0
    assert update.projection.relative_projection_distance > 0
    assert torch.linalg.eigvalsh(update.representation.to_dense()).min() >= -1e-12
    assert update.ridge.cold_started is True
    assert update.correction_method == "full_lfu"


def test_directional_ridge_state_round_trip_preserves_next_update() -> None:
    ridge = DirectionalRidgeLFUState(
        half_life_steps=4.0,
        amplitude_epsilon=1e-6,
        coherence_threshold=0.75,
    )
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    ac = torch.eye(2, dtype=torch.float64)
    residual = 2 * ac
    ridge.update(direction, ac, residual)
    restored = DirectionalRidgeLFUState.from_state_mapping(
        ridge.state_mapping(),
        device="cpu",
        dtype=torch.float64,
    )

    expected = ridge.update(2 * direction, 3 * ac, 4 * residual)
    actual = restored.update(2 * direction, 3 * ac, 4 * residual)

    torch.testing.assert_close(actual.full, expected.full)
    assert restored.tensor_bytes() == ridge.tensor_bytes()
    assert actual.metrics_mapping() == expected.metrics_mapping()


def test_archive_state_round_trip_separates_canonical_and_measured_bytes() -> None:
    state = HybridArchiveState(
        anchor=torch.tensor([1.0, -1.0], dtype=torch.float32),
        fisher=LowRankDiagonalFisher(
            factor=torch.ones((2, 1), dtype=torch.float64),
            residual_diagonal=torch.ones(2, dtype=torch.float64),
        ),
        initial_anchor_observations=100,
        initial_fisher_score_observations=200,
        archived_online_events=3,
        consolidation_steps=2,
    )

    restored = HybridArchiveState.from_mapping(
        state.to_mapping(),
        device="cpu",
        anchor_dtype=torch.float32,
        fisher_dtype=torch.float64,
    )

    assert restored.to_mapping().keys() == state.to_mapping().keys()
    assert restored.canonical_persistent_bytes() == 2 * (1 + 2) * 4
    assert restored.measured_tensor_bytes() == 2 * 4 + (2 + 2) * 8


def test_initial_archive_source_validates_anchor_rank_and_fixed_pi(
    tmp_path: Path,
) -> None:
    anchor = torch.tensor([0.2, -0.1], dtype=torch.float32)
    representation = LowRankDiagonalFisher(
        factor=torch.ones((2, 1), dtype=torch.float64),
        residual_diagonal=torch.ones(2, dtype=torch.float64),
    )
    artifact = tmp_path / "checkpoints.pt"
    torch.save(
        {
            "schema_version": 4,
            "conditions": {
                "low_rank_diagonal_r1": {
                    "0": {
                        "parameter": anchor,
                        "representation": representation.artifact_mapping(),
                        "controller_decision": {"applied_pi": 0.05},
                    }
                }
            },
        },
        artifact,
    )

    source = load_initial_archive_source(
        artifact,
        anchor,
        expected_rank=1,
        initial_anchor_observations=100,
        initial_fisher_score_observations=200,
        device="cpu",
        anchor_dtype=torch.float32,
        fisher_dtype=torch.float64,
    )

    assert source.state.rank == 1
    assert source.state.archived_online_events == 0
    assert len(source.artifact_sha256) == 64


def test_capacity_zero_matches_separate_archive_consolidation_parameters() -> None:
    learner = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    archive_model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    archive_model.load_state_dict(learner.state_dict())
    learner_layout = ParameterLayout.from_module(learner)
    archive_layout = ParameterLayout.from_module(archive_model)
    anchor = archive_layout.flatten_module(archive_model, detach=True)
    fisher = torch.eye(anchor.numel(), dtype=torch.float64)
    inputs = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    config = OptimizerConfig(
        name="sgd",
        learning_rate=0.05,
        inner_steps=3,
        ewc_strength=1.0,
    )

    take_ewc_proposal(
        learner,
        learner_layout,
        inputs,
        targets,
        fisher,
        config,
        build_optimizer(learner, config),
        adaptation_weight=0.05,
        penalty_anchor=anchor,
    )
    take_ewc_proposal(
        archive_model,
        archive_layout,
        inputs,
        targets,
        fisher,
        config,
        build_optimizer(archive_model, config),
        adaptation_weight=0.05,
    )

    torch.testing.assert_close(
        learner_layout.flatten_module(learner, detach=True),
        archive_layout.flatten_module(archive_model, detach=True),
    )


def test_deployment_controller_state_round_trip_preserves_vector_state() -> None:
    state = ControllerState(
        trend=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        q=0.02,
        residual_moment=0.4,
        scale_moment=0.5,
        environment_distance=0.2,
        previous_pi=0.05,
        accepted_steps=7,
    )

    restored = _controller_state_from_mapping(
        _controller_state_mapping(state),
        dtype=torch.float64,
    )

    torch.testing.assert_close(restored.trend, state.trend)
    assert restored.scalar_mapping(1e-12) == state.scalar_mapping(1e-12)
