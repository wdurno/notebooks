"""Tiny CPU audits of the Fix5 represented Fisher, never an experiment run."""

import json

import pytest
import torch

from numerical_experiment.picar_head_awac.archive import Archive


def _fixture(dtype=torch.float64, rank=3):
    generator = torch.Generator().manual_seed(1919)
    factor = torch.randn(11, rank, dtype=dtype, generator=generator) * 0.2
    diagonal = factor.square().sum(1) + torch.linspace(0.1, 0.5, 11, dtype=dtype)
    return Archive(diagonal, factor, rank), generator


def _dense(archive):
    return archive.factor @ archive.factor.mT + torch.diag(archive.residual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rank", [0, 1, 3, 10])
def test_updates_preserve_full_diagonal_trace_and_psd(dtype, rank):
    archive, generator = _fixture(dtype, rank)
    for _ in range(4):
        fresh = torch.randn(11, 7, dtype=dtype, generator=generator) * 0.3
        expected_diagonal = 0.8 * archive.diagonal + 0.2 * fresh.square().sum(1)
        diagnostics = archive.update(fresh, 0.8, 0.2)
        dense = _dense(archive)
        torch.testing.assert_close(archive.diagonal, expected_diagonal, rtol=0, atol=0)
        torch.testing.assert_close(dense.diag(), expected_diagonal)
        torch.testing.assert_close(dense.trace(), expected_diagonal.sum())
        torch.testing.assert_close(archive.residual, archive.diagonal - archive.factor.square().sum(1))
        assert archive.factor.shape[1] <= rank
        assert bool((archive.residual >= 0).all())
        assert float(torch.linalg.eigvalsh(dense).min()) >= -1e-6
        assert diagnostics["diagonal_max_abs_error"] <= 1e-6
        assert diagnostics["trace"] == pytest.approx(float(expected_diagonal.sum()))
        json.dumps(diagnostics, allow_nan=False)


def test_one_chunk_matches_direct_thin_svd_and_local_compression_loss():
    archive, generator = _fixture()
    old = _dense(archive)
    fresh = torch.randn(11, 6, dtype=torch.float64, generator=generator)
    augmented = torch.cat((0.7**0.5 * archive.factor, 0.3**0.5 * fresh), 1)
    left, singular, _ = torch.linalg.svd(augmented, full_matrices=False)
    expected_factor = left[:, :3] * singular[:3]
    diagnostics = archive.update(fresh, 0.7, 0.3)
    torch.testing.assert_close(archive.factor @ archive.factor.mT, expected_factor @ expected_factor.mT)
    torch.testing.assert_close(archive.diagonal, (0.7 * old + 0.3 * fresh @ fresh.mT).diag())
    assert diagnostics["compression_tail_trace"] == pytest.approx(float(singular[3:].square().sum()))
    assert diagnostics["largest_discarded_squared_singular_value"] == pytest.approx(float(singular[3].square()))
    expected_eigenvalues = torch.linalg.eigvalsh(archive.factor.mT @ archive.factor).flip(0)
    torch.testing.assert_close(torch.tensor(diagnostics["factor_eigenvalues"], dtype=torch.float64), expected_eigenvalues)


def test_no_truncation_recovers_the_full_uncompressed_operator():
    initial, generator = _fixture(rank=2)
    archive = Archive(initial.diagonal, initial.factor, rank=10)
    fresh = torch.randn(11, 3, dtype=torch.float64, generator=generator)
    expected = 0.6 * _dense(archive) + 0.4 * fresh @ fresh.mT
    diagnostics = archive.update(fresh, 0.6, 0.4)
    torch.testing.assert_close(_dense(archive), expected, rtol=1e-12, atol=1e-12)
    assert diagnostics["compression_tail_trace"] == 0


def test_rank_zero_fast_path_never_decomposes(monkeypatch):
    archive = Archive(torch.tensor([1.0, 2.0, 3.0]))
    fresh = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("rank zero called SVD"))
    expected = 0.6 * archive.diagonal + 0.4 * fresh.square().sum(1)
    archive.update(fresh, 0.6, 0.4)
    assert archive.factor.shape == (3, 0)
    torch.testing.assert_close(archive.residual, expected, rtol=0, atol=0)
    vector = torch.tensor([0.5, 2.0, -1.0])
    torch.testing.assert_close(archive.matvec(vector), expected * vector)
    torch.testing.assert_close(archive.quadratic(vector), (expected * vector.square()).sum())


def test_fresh_column_memory_is_bounded_and_alpha_is_applied_once(monkeypatch):
    archive, generator = _fixture()
    streamed = Archive.from_state_dict(archive.state_dict(), dtype=torch.float64)
    fresh = torch.randn(11, 79, dtype=torch.float64, generator=generator)
    expected = 0.25 * archive.diagonal + 0.75 * fresh.square().sum(1)
    widths = []
    svd = torch.linalg.svd

    def observe(value, **kwargs):
        widths.append(value.shape[1])
        return svd(value, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", observe)
    diagnostics = archive.update(fresh, 0.25, 0.75)
    assert max(widths) <= archive.rank + archive.FRESH_CHUNK_COLUMNS
    assert diagnostics["chunk_sizes"] == [32, 32, 15]
    torch.testing.assert_close(archive.diagonal, expected, rtol=0, atol=0)
    for index, chunk in enumerate(fresh.split(archive.FRESH_CHUNK_COLUMNS, dim=1)):
        streamed.update(chunk, 0.25 if index == 0 else 1.0, 0.75)
    torch.testing.assert_close(_dense(archive), _dense(streamed), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["zero", "dependent", "empty"])
def test_zero_dependent_and_empty_factors(kind):
    archive = Archive(torch.zeros(5, dtype=torch.float64), rank=3)
    column = torch.arange(5, dtype=torch.float64).unsqueeze(1)
    fresh = {"zero": torch.zeros(5, 4, dtype=torch.float64),
             "dependent": column @ torch.tensor([[1.0, -2.0, 0.0, 1.0]], dtype=torch.float64),
             "empty": torch.empty(5, 0, dtype=torch.float64)}[kind]
    diagnostics = archive.update(fresh, 0.0, 1.0)
    torch.testing.assert_close(_dense(archive), fresh @ fresh.mT, atol=1e-12, rtol=1e-12)
    assert diagnostics["realized_rank"] == (1 if kind == "dependent" else 0)
    if kind != "dependent":
        assert diagnostics["correlated_trace_fraction"] is None
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("weight", [0.0, 0.2, 1.0, 4.0])
def test_products_quadratics_and_coordinate_damped_solve_match_dense(rank, weight):
    archive, generator = _fixture(rank=rank)
    dense = _dense(archive)
    rhs = torch.randn(11, 3, dtype=torch.float64, generator=generator)
    damping = torch.linspace(0.03, 0.5, 11, dtype=torch.float64)
    torch.testing.assert_close(archive.matvec(rhs), dense @ rhs)
    torch.testing.assert_close(archive.quadratic(rhs[:, 0]), rhs[:, 0] @ dense @ rhs[:, 0])
    system = torch.diag(damping) + weight * dense
    torch.testing.assert_close(archive.solve(rhs, damping, weight), torch.linalg.solve(system, rhs))
    torch.testing.assert_close(archive.solve(rhs[:, 0], damping, weight), torch.linalg.solve(system, rhs[:, 0]))


def test_cached_solve_reuses_setup_and_invalidates_for_all_inputs(monkeypatch):
    archive, generator = _fixture()
    damping = torch.ones(11, dtype=torch.float64)
    rhs = torch.randn(11, dtype=torch.float64, generator=generator)
    cholesky = torch.linalg.cholesky
    calls = []

    def observe(matrix):
        calls.append(matrix.shape)
        return cholesky(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", observe)
    archive.solve(rhs, damping)
    archive.solve(2 * rhs, damping.clone())
    assert calls == [(3, 3)]
    damping[0] = 0.5
    archive.solve(rhs, damping)
    archive.solve(rhs, damping, 0.5)
    archive.update(torch.zeros(11, 1, dtype=torch.float64), 0.9, 0.1)
    result = archive.solve(rhs, damping, 0.5)
    assert len(calls) == 4
    torch.testing.assert_close(result, torch.linalg.solve(torch.diag(damping) + 0.5 * _dense(archive), rhs))


def test_quadratic_and_solve_remain_differentiable_in_the_vector():
    archive, _ = _fixture()
    vector = torch.linspace(-1, 1, 11, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(archive.quadratic(vector), vector)
    torch.testing.assert_close(gradient, 2 * archive.matvec(vector.detach()))
    solution = archive.solve(vector, torch.ones_like(vector))
    gradient, = torch.autograd.grad(solution.sum(), vector)
    torch.testing.assert_close(gradient, archive.solve(torch.ones_like(vector), torch.ones_like(vector)))


def test_state_round_trip_is_exact_independent_and_continues_identically(tmp_path):
    archive, generator = _fixture()
    fresh = torch.randn(11, 6, dtype=torch.float64, generator=generator)
    archive.update(fresh, 0.8, 0.2)
    state = archive.state_dict()
    assert state["diagonal"].device.type == state["factor"].device.type == "cpu"
    path = tmp_path / "archive.pt"
    torch.save(state, path)
    restored = Archive.from_state_dict(torch.load(path, weights_only=True), dtype=torch.float64)
    torch.testing.assert_close(restored.factor, archive.factor, rtol=0, atol=0)
    torch.testing.assert_close(restored.diagonal, archive.diagonal, rtol=0, atol=0)
    assert restored.diagnostics() == archive.diagnostics()
    assert Archive.from_state_dict(state).diagonal.dtype == torch.float32
    restored.update(fresh, 0.75, 0.25)
    archive.update(fresh, 0.75, 0.25)
    torch.testing.assert_close(_dense(restored), _dense(archive), rtol=0, atol=0)
    state["diagonal"].zero_()
    state["factor"].zero_()
    assert bool((archive.diagonal > 0).all())
    assert bool((restored.diagonal > 0).all())


def test_roundoff_is_audited_and_corrected_without_changing_diagonal():
    diagonal = torch.ones(3)
    factor = torch.full((3, 1), 1 + torch.finfo(torch.float32).eps)
    archive = Archive(diagonal, factor, rank=1)
    diagnostics = archive.diagnostics()
    assert diagnostics["minimum_residual_before_correction"] < 0
    assert diagnostics["negative_residual_mass_before_correction"] > 0
    assert diagnostics["roundoff_corrected_rows"] == 3
    assert bool((archive.residual >= 0).all())
    torch.testing.assert_close(_dense(archive).diag(), diagonal, rtol=0, atol=0)
    with pytest.raises(ValueError, match="material negative residual"):
        Archive(diagonal, factor * 2, rank=1)


@pytest.mark.parametrize("alpha,beta", [(-1, 1), (1, -1), (float("nan"), 1), (1, float("inf"))])
def test_bad_weights_leave_state_unchanged(alpha, beta):
    archive, _ = _fixture()
    before = archive.state_dict()
    with pytest.raises(ValueError):
        archive.update(torch.ones(11, 1, dtype=torch.float64), alpha, beta)
    torch.testing.assert_close(archive.diagonal, before["diagonal"], rtol=0, atol=0)
    torch.testing.assert_close(archive.factor, before["factor"], rtol=0, atol=0)


def test_invalid_shapes_values_and_ranks_are_rejected():
    archive, _ = _fixture()
    for fresh in (torch.ones(11), torch.ones(10, 2), torch.ones(11, 2),
                  torch.full((11, 2), float("nan"), dtype=torch.float64)):
        with pytest.raises(ValueError):
            archive.update(fresh, 0.5, 0.5)
    rhs = torch.ones(11, dtype=torch.float64)
    for damping in (torch.zeros_like(rhs), -rhs, rhs.unsqueeze(1), rhs * float("inf")):
        with pytest.raises(ValueError):
            archive.solve(rhs, damping)
    for rank in (-1, 0.5, True):
        with pytest.raises(ValueError):
            Archive(rhs, rank=rank)
    with pytest.raises(ValueError, match="width"):
        Archive(rhs, torch.zeros(11, 2, dtype=torch.float64), rank=1)
