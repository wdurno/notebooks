"""Rank-deficient Fisher compression must not manufacture information."""

import json

import pytest
import torch

from mnist_experiment.fisher_memory_regression.learner import (
    compress_fisher, consolidate, _fisher_matrix, StopRequested,
)


def low_rank_case(dimension, epsilon):
    generator = torch.Generator().manual_seed(23)
    u = torch.randn(dimension, dtype=torch.float64, generator=generator)
    u /= u.norm()
    return u[:, None] @ u[None, :] + epsilon * torch.eye(dimension, dtype=torch.float64)


@pytest.mark.parametrize("dimension", [16, 32, 287])
@pytest.mark.parametrize("epsilon", [0., 1e-12, 1e-8])
def test_nearly_rank_one_compression_preserves_trace_diagonal_and_psd(dimension, epsilon):
    source = low_rank_case(dimension, epsilon)
    original = source.clone()
    artifact, diagnostics = compress_fisher(source, rank=8, seed=7)
    represented = _fisher_matrix(artifact)
    torch.testing.assert_close(represented.diagonal(), source.diagonal(), rtol=1e-11, atol=1e-14)
    assert float(represented.trace()) == pytest.approx(float(source.trace()), rel=1e-12, abs=1e-14)
    assert float(torch.linalg.eigvalsh(represented).min()) >= -1e-12
    assert diagnostics["compression"]["realized_rank"] <= 8
    assert diagnostics["compression"]["algorithm"] == "exact_symmetric_eigh"
    assert torch.equal(source, original)
    json.dumps(diagnostics, allow_nan=False)


def test_repeated_retirement_updates_obey_convex_trace_bounds():
    source = low_rank_case(32, 1e-10)
    archive, _ = compress_fisher(source, rank=8, seed=3)
    anchor = torch.zeros(32, dtype=torch.float64)
    ids = []
    initial_trace = float(source.trace())
    for step in range(100):
        fresh = source * (1 + step % 3)
        old_trace = float(_fisher_matrix(archive).trace())
        update = consolidate(archive, anchor, anchor, fresh,
            evicted_ids=[step], consolidated_ids=ids, pi=.025, arrival_size=4, rank=8, seed=step)
        archive, ids = update["fisher_L"], update["consolidated_ids"]
        actual = float(_fisher_matrix(archive).trace())
        expected = (1 - update["eta"]) * old_trace + update["eta"] * float(fresh.trace())
        assert actual == pytest.approx(expected, rel=1e-12)
        assert initial_trace * (1 - 1e-12) <= actual <= 3 * initial_trace * (1 + 1e-12)
    assert ids == list(range(100))


@pytest.mark.parametrize("rank", [0, 32, 64])
def test_zero_full_and_overspecified_rank_boundaries(rank):
    source = low_rank_case(32, .1)
    artifact, _ = compress_fisher(source, rank=rank, seed=0)
    represented = _fisher_matrix(artifact)
    if rank == 0:
        torch.testing.assert_close(represented, torch.diag(source.diagonal()))
    else:
        torch.testing.assert_close(represented, source, rtol=1e-11, atol=1e-13)
    zero, _ = compress_fisher(torch.zeros_like(source), rank=rank)
    assert not _fisher_matrix(zero).count_nonzero()


def test_compression_is_deterministic_rng_independent_and_stop_aware():
    source = low_rank_case(16, 1e-10)
    state = torch.get_rng_state().clone()
    first, _ = compress_fisher(source, seed=1)
    second, _ = compress_fisher(source, seed=99)
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(_fisher_matrix(first), _fisher_matrix(second), rtol=0, atol=0)
    with pytest.raises(StopRequested):
        compress_fisher(source, check_stop=lambda: True)
    with pytest.raises(ValueError, match="indefinite"):
        compress_fisher(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64))
