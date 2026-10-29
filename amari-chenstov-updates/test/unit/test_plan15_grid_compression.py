"""Focused CPU-only numerical and immutable-profile tests for Phase 11."""

import json
import multiprocessing as mp
import subprocess
import sys

import pytest
import torch

from mnist_experiment.fisher_memory_grid_registration import compression as c
from mnist_experiment.fisher_memory_regression.learner import compress_fisher as regression
from src.representations import representation_from_artifact


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def source_matrix(dimension=16, epsilon=1e-12):
    generator = torch.Generator().manual_seed(23)
    vector = torch.randn(dimension, generator=generator, dtype=torch.float64)
    vector /= vector.norm()
    return vector[:, None] @ vector[None, :] + epsilon * torch.eye(dimension,
                                                               dtype=torch.float64)


def dense(artifact):
    return representation_from_artifact(artifact).to_dense()


@pytest.mark.parametrize("epsilon", [0., 1e-12, 1e-8, .1])
@pytest.mark.parametrize("rank", [0, 1, 8, 16, 32])
def test_rank_boundaries_near_rank_one_diagonal_trace_psd(epsilon, rank):
    source = source_matrix(epsilon=epsilon)
    original = source.clone()
    artifact, diagnostics = c.compress_fisher(source, rank=rank)
    represented = dense(artifact)
    torch.testing.assert_close(represented.diagonal(), source.diagonal(), rtol=1e-11, atol=1e-14)
    assert float(represented.trace()) == pytest.approx(float(source.trace()), rel=1e-12)
    assert float(torch.linalg.eigvalsh(represented).min()) >= -1e-12
    assert artifact["kind"] == "low_rank_diagonal"
    assert artifact["factor"].dtype == torch.float64
    assert artifact["factor"].device.type == "cpu"
    assert diagnostics["compression"]["realized_rank"] <= min(rank, len(source))
    assert diagnostics["compression"]["algorithm"] == "exact_symmetric_eigh"
    assert diagnostics["compression"]["backend"] == "exact_cpu"
    assert diagnostics["compression"]["elapsed_seconds"] >= 0
    assert diagnostics["compression"]["timings_seconds"]["eigh"] >= 0
    assert torch.equal(source, original)
    if rank == 0:
        torch.testing.assert_close(represented, torch.diag(source.diagonal()))
    elif rank >= len(source) or epsilon == 0:
        torch.testing.assert_close(represented, source, rtol=1e-11, atol=1e-13)
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("rank", [0, 1, 8, 32])
def test_zero_source(rank):
    artifact, diagnostics = c.compress_fisher(torch.zeros(16, 16), rank=rank)
    assert not dense(artifact).count_nonzero()
    assert diagnostics["compression"]["realized_rank"] == 0
    assert diagnostics["source_trace"] == diagnostics["represented_trace"] == 0


def test_exact_cpu_matches_regression_and_is_rng_independent():
    source = source_matrix()
    before = torch.get_rng_state().clone()
    first, _ = c.compress_fisher(source, seed=1)
    second, _ = c.compress_fisher(source, seed=99)
    expected, _ = regression(source, seed=5)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(dense(first), dense(second), rtol=0, atol=0)
    torch.testing.assert_close(dense(first), dense(expected), rtol=0, atol=0)


@pytest.mark.parametrize("split_cluster", [False, True])
def test_clustered_and_repeated_eigenvalues_compare_matrices(split_cluster):
    generator = torch.Generator().manual_seed(37)
    basis, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator, dtype=torch.float64))
    values = torch.tensor([.01] * 4 + [1.] * 4 + [2.] * 8, dtype=torch.float64)
    if not split_cluster:
        values[8:] += torch.linspace(0, 1e-9, 8)
    source = (basis * values) @ basis.mT
    artifact, _ = c.compress_fisher(source, rank=4 if split_cluster else 8)
    expected, _ = regression(source, rank=4 if split_cluster else 8)
    torch.testing.assert_close(dense(artifact), dense(expected), rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(dense(artifact).diagonal(), source.diagonal())


def test_repeated_convex_updates_do_not_manufacture_trace():
    source = source_matrix()
    archive, _ = c.compress_fisher(source)
    for step in range(40):
        old = dense(archive)
        fresh = source * (1 + step % 3)
        archive, diagnostics = c.compress_fisher(.975 * old + .025 * fresh, seed=step)
        expected = .975 * float(old.trace()) + .025 * float(fresh.trace())
        assert float(dense(archive).trace()) == pytest.approx(expected, rel=1e-12)
        assert diagnostics["represented_trace"] == pytest.approx(expected, rel=1e-12)


def test_float64_symmetrization_does_not_mutate_input_or_retain_autograd():
    source = torch.eye(16, dtype=torch.float32, requires_grad=True)
    perturbation = torch.zeros_like(source)
    perturbation[0, 1] = .1
    source = source + perturbation
    original = source.detach().clone()
    artifact, diagnostics = c.compress_fisher(source, rank=16)
    torch.testing.assert_close(dense(artifact), (original.double() + original.double().mT) / 2)
    assert diagnostics["compression"]["source_symmetry_error_fro"] > 0
    assert not artifact["factor"].requires_grad
    assert torch.equal(source, original)


@pytest.mark.parametrize("source", [torch.tensor([[1., 2.], [2., 1.]]),
    torch.diag(torch.tensor([-1., 1.])), torch.ones(2, 3), torch.empty(0, 0),
    torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
    torch.eye(2, dtype=torch.int64)])
def test_invalid_and_materially_indefinite_inputs_rejected(source):
    with pytest.raises(ValueError):
        c.compress_fisher(source)


@pytest.mark.parametrize("kwargs", [{"rank": -1}, {"rank": True}, {"rank": 1.5},
    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"backend": "lanczos"}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        c.compress_fisher(torch.eye(2), **kwargs)


def test_roundoff_negatives_are_recorded_before_clipping():
    source = torch.diag(torch.tensor([-1e-15, 1.], dtype=torch.float64))
    artifact, diagnostics = c.compress_fisher(source)
    assert diagnostics["pre_compression_min_eigenvalue"] == -1e-15
    assert diagnostics["pre_compression_negative_mass"] == 1e-15
    assert diagnostics["compression"]["residual_diagonal_minimum_before_clipping"] < 0
    assert dense(artifact)[0, 0] == 0


def test_material_negative_residual_is_not_silently_clipped(monkeypatch):
    original = torch.linalg.eigh

    def broken_eigh(matrix):
        values, vectors = original(matrix)
        return values * 2, vectors

    monkeypatch.setattr(torch.linalg, "eigh", broken_eigh)
    with pytest.raises(ValueError, match="exceeds the source diagonal"):
        c.compress_fisher(torch.eye(16, dtype=torch.float64), rank=16)


@pytest.mark.parametrize("stop_at", [1, 2, 3, 4, 5])
def test_stop_checked_across_numerical_stages(stop_at):
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == stop_at

    with pytest.raises(c.StopRequested):
        c.compress_fisher(source_matrix(), check_stop=stop)


def test_exact_cuda_unavailable_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA"):
        c.compress_fisher(torch.eye(2), backend="exact_cuda")


def test_tiny_cpu_resource_assessment_is_json_and_validates_exact_only():
    before = torch.get_rng_state().clone()
    result = c.assess_resources(dimension=16, rank=8, device="cpu", max_seconds=60,
                                attempt_timeout_seconds=25)
    assert result["selected_backend"] == "exact_cpu"
    assert result["whole_update_profile"]["status"] == "pending_parent_profile"
    assert result["backends"]["exact_cpu"]["validated"]
    assert len(result["attempts"]) == 2
    for attempt in result["attempts"]:
        assert attempt["status"] == "ok"
        assert len(set(attempt["seeds"].values())) == len(attempt["seeds"])
        assert sum(not row["warmup"] for row in attempt["rows"]) == 2
        assert attempt["peak_host_rss_bytes"] > 0
        assert attempt["dense_matrix_bytes"] == 16 * 16 * 8
        assert all(row["audit"]["validated"] for row in attempt["rows"])
    assert torch.equal(before, torch.get_rng_state())
    json.dumps(result, allow_nan=False)


def fake_attempt(config, case, backend, seconds, check_stop):
    return dict(case=case, backend=backend, validated=True, status="ok",
                rows=[dict(warmup=False, transaction_seconds=2 if backend == "exact_cpu" else 1)])


def test_fastest_validated_backend_selected_without_running_cuda(monkeypatch):
    monkeypatch.setattr(c, "_run_profile_attempt", fake_attempt)
    result = c.assess_resources(dimension=16)
    assert result["selected_backend"] == "exact_cuda"

    def invalid_cuda(*args):
        row = fake_attempt(*args)
        row["validated"] = row["backend"] == "exact_cpu"
        return row

    monkeypatch.setattr(c, "_run_profile_attempt", invalid_cuda)
    assert c.assess_resources(dimension=16)["selected_backend"] == "exact_cpu"


def test_profile_stop_and_short_attempt_timeout():
    children = {child.pid for child in mp.active_children()}
    with pytest.raises(c.StopRequested):
        c.assess_resources(dimension=16, device="cpu", check_stop=lambda: True)
    result = c.assess_resources(dimension=16, device="cpu", max_seconds=1,
                                attempt_timeout_seconds=.001)
    assert result["selected_backend"] is None
    assert all(row["status"] == "timeout" for row in result["attempts"])
    assert {child.pid for child in mp.active_children()} == children


def test_profile_stop_propagates_and_reaps_active_worker():
    config = c._profile_config(16, 8, "cpu", 0, 2, 2, 60, 25)
    children = {child.pid for child in mp.active_children()}

    def stop():
        raise RuntimeError("parent RuntimeBudget stop")

    with pytest.raises(RuntimeError, match="RuntimeBudget stop"):
        c._run_profile_attempt(config, c.CASES[0], "exact_cpu", 25, stop)
    assert {child.pid for child in mp.active_children()} == children


def test_cli_module_entry_point_cpu_only(tmp_path):
    output = tmp_path / "cli-profile"
    completed = subprocess.run([sys.executable, "-m", c.__name__, "--output", str(output),
        "--dimension", "16", "--device", "cpu", "--max-seconds", "60",
        "--attempt-timeout-seconds", "25"], check=True, capture_output=True, text=True,
        timeout=65)
    result = json.loads(completed.stdout)
    assert result["selected_backend"] == "exact_cpu"
    assert c.load_assessment(output) == result


def test_cli_seals_and_only_reuses_hash_validated_completed_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c, "_run_profile_attempt", fake_attempt)
    output = tmp_path / "profile"
    args = ["--output", str(output), "--dimension", "16", "--device", "cpu"]
    assert c.main(args) == 0
    assessment = c.load_assessment(output)
    assert assessment["selected_backend"] == "exact_cpu"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        c.main(args)
    assert c.main(args + ["--resume"]) == 0
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="hash mismatch"):
        c.main(args + ["--resume", "--rank", "4"])
    payload = json.loads((output / "assessment.json").read_text())
    payload["selected_backend"] = "exact_cuda"
    (output / "assessment.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        c.load_assessment(output)
    capsys.readouterr()
