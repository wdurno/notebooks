"""Tiny CPU audits for the standalone candidate, not production admission."""

import json

import pytest
import torch

from mnist_experiment.plan15_grid_compression_probe import approximate_fisher, StopRequested
from src.representations import representation_from_artifact


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def psd_case(dimension=48, width=5, epsilon=.01):
    generator = torch.Generator().manual_seed(73)
    basis, _ = torch.linalg.qr(torch.randn(dimension, width, generator=generator,
                                         dtype=torch.float64))
    values = torch.linspace(2., .5, width, dtype=torch.float64)
    return (basis * values) @ basis.mT + epsilon * torch.eye(dimension, dtype=torch.float64)


def exact_artifact(matrix, rank):
    values, vectors = torch.linalg.eigh(matrix)
    selected = torch.where(values > 0)[0][-rank:]
    factor = vectors[:, selected] * values[selected].sqrt()
    return dict(kind="low_rank_diagonal", factor=factor,
                residual_diagonal=(matrix.diagonal() - factor.square().sum(1)).clamp_min(0))


def assert_invariants(source, artifact, diagnostics):
    represented = representation_from_artifact(artifact)
    tolerance = diagnostics["compression"]["diagonal_roundoff_tolerance"]
    low_rank = represented.factor @ represented.factor.mT
    assert float(torch.linalg.eigvalsh(source - low_rank).min()) >= -max(1e-13, tolerance)
    assert float(torch.linalg.eigvalsh(represented.to_dense()).min()) >= -max(1e-13, tolerance)
    torch.testing.assert_close(represented.diagonal_vector(), source.diagonal(), rtol=0,
                               atol=max(1e-14, tolerance))
    assert float(represented.diagonal_vector().sum()) == pytest.approx(float(source.trace()),
                                                                    rel=1e-12, abs=1e-13)
    assert represented.factor.device.type == "cpu"
    assert represented.factor.dtype == torch.float64
    assert not represented.factor.requires_grad
    assert diagnostics["pre_compression_min_eigenvalue"] is None
    assert diagnostics["accuracy_status"].startswith("pending_actual_snapshot")
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("rank", [0, 1, 8, 64])
def test_zero(rank):
    source = torch.zeros(16, 16, dtype=torch.float64)
    artifact, diagnostics = approximate_fisher(source, rank=rank)
    assert_invariants(source, artifact, diagnostics)
    assert artifact["factor"].shape[1] == 0
    assert diagnostics["compression"]["largest_omitted_core_eigenvalue"] is None
    assert diagnostics["compression"]["cutoff_relative_gap"] is None
    assert not representation_from_artifact(artifact).to_dense().count_nonzero()


@pytest.mark.parametrize("epsilon", [0., 1e-12, 1e-8])
@pytest.mark.parametrize("warm", [False, True])
def test_rank_one_and_near_rank_one_do_not_manufacture_modes(epsilon, warm):
    source = psd_case(width=1, epsilon=epsilon)
    start = exact_artifact(source, 1)["factor"] if warm else None
    artifact, diagnostics = approximate_fisher(source, warm_start=start)
    assert_invariants(source, artifact, diagnostics)
    represented = representation_from_artifact(artifact).to_dense()
    torch.testing.assert_close(represented, source, rtol=1e-7, atol=2e-8)
    if epsilon <= 1e-12:
        assert diagnostics["compression"]["realized_rank"] == 1


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_psd_ordering_and_diagonal_trace_with_incomplete_subspace(steps):
    source = psd_case(width=12)
    artifact, diagnostics = approximate_fisher(source, rank=4, oversample=3, steps=steps)
    assert_invariants(source, artifact, diagnostics)
    assert diagnostics["compression"]["realized_rank"] <= 4
    assert diagnostics["compression"]["retained_directions_orthogonality_error_fro"] < 1e-10


def test_small_well_separated_case_passes_exact_same_rank_matrix_quadratic_and_solve_audits():
    source = psd_case()
    artifact, diagnostics = approximate_fisher(source, rank=5, oversample=8)
    assert_invariants(source, artifact, diagnostics)
    actual = representation_from_artifact(artifact)
    reference = representation_from_artifact(exact_artifact(source, 5))
    dense, expected = actual.to_dense(), reference.to_dense()
    assert float((dense - expected).norm() / expected.norm()) < .01
    generator = torch.Generator().manual_seed(337)
    probes = torch.randn(len(source), 4, generator=generator, dtype=torch.float64)
    quadratic = (probes * actual.matvec(probes)).sum(0)
    exact_quadratic = (probes * reference.matvec(probes)).sum(0)
    assert float((quadratic - exact_quadratic).norm() / exact_quadratic.norm()) < .01
    solution = actual.damped_solve(probes, .1)
    exact_solution = reference.damped_solve(probes, .1)
    assert float((solution - exact_solution).norm() / exact_solution.norm()) < .01
    assert diagnostics["compression"]["ritz_residual_relative_fro"] < 1e-8


@pytest.mark.parametrize("split_cluster", [False, True])
def test_clustered_and_repeated_spectrum(split_cluster):
    generator = torch.Generator().manual_seed(37)
    q, _ = torch.linalg.qr(torch.randn(48, 48, generator=generator, dtype=torch.float64))
    values = torch.tensor([.01] * 36 + [1.] * 4 + [2.] * 8, dtype=torch.float64)
    source = (q * values) @ q.mT
    rank = 4 if split_cluster else 8
    artifact, diagnostics = approximate_fisher(source, rank=rank)
    assert_invariants(source, artifact, diagnostics)
    if not split_cluster:
        expected = representation_from_artifact(exact_artifact(source, rank)).to_dense()
        actual = representation_from_artifact(artifact).to_dense()
        assert float((actual - expected).norm() / expected.norm()) < .01


def test_isotropic_cutoff_exposes_tie_despite_zero_ritz_residual():
    source = torch.eye(64, dtype=torch.float64)
    artifact, diagnostics = approximate_fisher(source, rank=8, oversample=16)
    assert_invariants(source, artifact, diagnostics)
    compression = diagnostics["compression"]
    assert compression["largest_omitted_core_eigenvalue"] == pytest.approx(1., abs=1e-12)
    assert compression["cutoff_relative_gap"] == pytest.approx(0., abs=1e-12)
    assert compression["ritz_residual_relative_fro"] < 1e-12
    actual = representation_from_artifact(artifact).to_dense()
    expected = representation_from_artifact(exact_artifact(source, 8)).to_dense()
    assert float((actual - expected).norm() / expected.norm()) > .01


@pytest.mark.parametrize("epsilon", [0., 1e-12])
def test_near_rank_one_without_omitted_core_modes_reports_no_cutoff_gap(epsilon):
    source = psd_case(width=1, epsilon=epsilon)
    artifact, diagnostics = approximate_fisher(source)
    assert_invariants(source, artifact, diagnostics)
    compression = diagnostics["compression"]
    assert compression["realized_rank"] == compression["core_rank"] == 1
    assert compression["largest_omitted_core_eigenvalue"] is None
    assert compression["cutoff_relative_gap"] is None


def test_cutoff_gap_uses_smallest_retained_and_largest_omitted_gram_modes():
    source = torch.diag(torch.arange(1., 17., dtype=torch.float64))
    _, diagnostics = approximate_fisher(source, rank=4, oversample=12, steps=0)
    compression = diagnostics["compression"]
    assert compression["largest_omitted_core_eigenvalue"] == pytest.approx(12.)
    assert compression["cutoff_relative_gap"] == pytest.approx(1 / 13)


def test_repeated_updates_preserve_convex_trace_with_warm_artifact():
    source = psd_case()
    archive, _ = approximate_fisher(source)
    for step in range(30):
        previous = representation_from_artifact(archive).to_dense()
        combined = .975 * previous + .025 * source * (1 + step % 3)
        archive, diagnostics = approximate_fisher(combined, seed=step, warm_start=archive)
        assert_invariants(combined, archive, diagnostics)


def test_seed_local_deterministic_and_warm_start_unmodified():
    source = psd_case()
    original = source.clone()
    warm = torch.zeros(48, 8, dtype=torch.float64)
    before = torch.get_rng_state().clone()
    first, one = approximate_fisher(source, seed=7, warm_start=warm)
    second, _ = approximate_fisher(source, seed=7, warm_start=warm)
    torch.testing.assert_close(first["factor"], second["factor"], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(source, original)
    assert not warm.count_nonzero()
    assert one["compression"]["component_seeds"]


def test_no_dense_spectral_diagnostic(monkeypatch):
    source = psd_case(dimension=64)
    original = torch.linalg.eigh
    widths = []

    def small_eigh(block):
        widths.append(len(block))
        assert len(block) <= 12
        return original(block)

    def forbidden(*args, **kwargs):
        raise AssertionError("full-spectrum diagnostic forbidden")

    monkeypatch.setattr(torch.linalg, "eigh", small_eigh)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    approximate_fisher(source, rank=4, oversample=8)
    assert len(widths) == 2


@pytest.mark.parametrize("rank", [16, 32])
def test_symmetrizes_float64_and_rank_zero_is_diagonal_only(rank):
    source = torch.eye(16, dtype=torch.float32, requires_grad=True)
    noise = torch.zeros_like(source)
    noise[0, 1] = .2
    source = source + noise
    artifact, diagnostics = approximate_fisher(source, rank=rank, oversample=0)
    symmetric = .5 * source.detach().double() + .5 * source.detach().double().mT
    torch.testing.assert_close(representation_from_artifact(artifact).to_dense(), symmetric)
    assert_invariants(symmetric, artifact, diagnostics)
    artifact, _ = approximate_fisher(symmetric, rank=0)
    torch.testing.assert_close(representation_from_artifact(artifact).to_dense(),
                               torch.diag(symmetric.diagonal()))


@pytest.mark.parametrize("options", [{"rank": -1}, {"rank": True}, {"steps": -1},
    {"steps": 1.5}, {"oversample": -1}, {"seed": True}, {"seed": -1},
    {"warm_start": torch.zeros(2, 8)}, {"warm_start": {"kind": "dense"}}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        approximate_fisher(torch.eye(16), **options)


@pytest.mark.parametrize("source", [torch.ones(2, 3), torch.empty(0, 0),
    torch.eye(2, dtype=torch.int64), torch.tensor([[float("nan")]]),
    torch.tensor([[float("inf")]]), torch.diag(torch.tensor([-1., 1.])),
    torch.tensor([[1., 2.], [2., 1.]])])
def test_invalid_or_detectably_indefinite_inputs(source):
    with pytest.raises(ValueError):
        approximate_fisher(source)


def test_material_negative_residual_rejected(monkeypatch):
    original = torch.linalg.eigh
    calls = 0

    def broken_core(block):
        nonlocal calls
        calls += 1
        values, vectors = original(block)
        return (values / 2 if calls == 1 else values), vectors

    monkeypatch.setattr(torch.linalg, "eigh", broken_core)
    with pytest.raises(ValueError, match="exceeds the source diagonal"):
        approximate_fisher(torch.eye(16, dtype=torch.float64), rank=16)


@pytest.mark.parametrize("stop_at", [1, 2, 5, 10, 13, 16, 17, 18, 19])
def test_stop_during_power_reconstruction_and_output(stop_at):
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == stop_at

    with pytest.raises(StopRequested):
        approximate_fisher(psd_case(), check_stop=stop)
