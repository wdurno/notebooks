"""Exact Phase 11 Fisher compression and a bounded, standalone resource audit.

The archive is LL^T + diag(d), with the source diagonal restored, not a new
Fisher estimand. No ridge, randomized eigensolver, or model training is used.
Run with ``python -m ...compression --output cache/<unique-profile>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

from mnist_experiment.fisher_memory.learner import StopRequested, _integer, _stop
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed


BACKENDS = ("exact_cpu", "exact_cuda")
CASES = ("near_rank_one", "broad_diagonal_plus_factors")
PROFILE_SCHEMA_VERSION = 1
TOLERANCE_POLICY = "100*n*float64_eps*spectral_radius"
AUDIT_RELATIVE_TOLERANCE = 1e-8


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compress_fisher(matrix: Tensor, rank: int = 8, seed: int = 0,
                    check_stop: Callable | None = None,
                    backend: str = "exact_cpu") -> tuple[dict, dict]:
    """Retain exact leading PSD modes, returning the regression CPU schema.

    Both backends symmetrize in float64. Material negative eigenvalues or
    residual diagonals are errors; only documented float64 roundoff is clipped.
    The deterministic algorithm records but does not consume ``seed``. Timings
    include synchronization, input conversion/transfers and CPU serialization.
    An individual eigh is not interruptible; the assessment isolates it in a
    killable process, while callers check cancellation between numerical stages.
    """
    _integer(rank, "rank", 0)
    derive_component_seed(seed, "compression_seed_validation")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported compression backend: {backend}")
    _stop(check_stop)
    if (not isinstance(matrix, Tensor) or not matrix.is_floating_point()
            or matrix.layout != torch.strided or matrix.ndim != 2
            or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]):
        raise ValueError("Fisher must be a nonempty square floating tensor")
    if backend == "exact_cuda" and not torch.cuda.is_available():
        raise RuntimeError("exact_cuda requires an available CUDA device")
    device = (matrix.device if matrix.device.type == "cuda" else torch.device("cuda")) \
        if backend == "exact_cuda" else torch.device("cpu")
    started = time.perf_counter()
    timings = {}
    _sync(matrix.device)
    stage = time.perf_counter()
    source = matrix.detach().to(device=device, dtype=torch.float64)
    _sync(device)
    timings["input_conversion_transfer"] = time.perf_counter() - stage
    stage = time.perf_counter()
    if not torch.isfinite(source).all():
        raise ValueError("Fisher must be finite")
    asymmetry = float((source - source.mT).norm())
    source = .5 * source + .5 * source.mT
    _sync(device)
    timings["validation_symmetrization"] = time.perf_counter() - stage
    _stop(check_stop)
    stage = time.perf_counter()
    eigenvalues, eigenvectors = torch.linalg.eigh(source)
    _sync(device)
    timings["eigh"] = time.perf_counter() - stage
    _stop(check_stop)
    stage = time.perf_counter()
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(eigenvectors).all():
        raise ValueError("nonfinite exact eigendecomposition")
    scale = max(float(eigenvalues.abs().max()), torch.finfo(torch.float64).tiny)
    tolerance = 100 * len(source) * torch.finfo(torch.float64).eps * scale
    minimum = float(eigenvalues.min())
    if minimum < -tolerance:
        raise ValueError("cannot archive a materially indefinite Fisher")
    count = min(rank, len(source))
    selected = torch.arange(len(source) - count, len(source), device=device)
    selected = selected[eigenvalues[selected] > 0]
    factor = eigenvectors[:, selected] * eigenvalues[selected].sqrt()
    residual = source.diagonal() - factor.square().sum(1)
    residual_minimum = float(residual.min())
    if residual_minimum < -tolerance:
        raise ValueError("spectral factor materially exceeds the source diagonal")
    representation = LowRankDiagonalFisher(factor, residual.clamp_min(0))
    diagonal = representation.diagonal_vector()
    if not torch.allclose(diagonal, source.diagonal(), rtol=0, atol=tolerance):
        raise ValueError("Fisher compression did not preserve the diagonal")
    diagnostics = {
        "pre_compression_min_eigenvalue": minimum,
        "pre_compression_negative_mass": float((-eigenvalues[eigenvalues < 0]).sum()),
        "source_trace": float(source.trace()), "represented_trace": float(diagonal.sum()),
        "compression": dict(algorithm="exact_symmetric_eigh", backend=backend,
            requested_rank=rank, realized_rank=representation.rank, seed=seed,
            randomized=False, tolerance_policy=TOLERANCE_POLICY,
            roundoff_tolerance=tolerance, source_symmetry_error_fro=asymmetry,
            retained_eigenvalues=eigenvalues[selected].flip(0).tolist(),
            residual_diagonal_minimum_before_clipping=residual_minimum,
            represented_diagonal_relative_error=float((diagonal - source.diagonal()).norm()
                / source.diagonal().norm().clamp_min(torch.finfo(torch.float64).tiny))),
    }
    _sync(device)
    timings["classification_factor_reconstruction"] = time.perf_counter() - stage
    _stop(check_stop)
    stage = time.perf_counter()
    artifact = representation.artifact_mapping()
    _sync(device)
    timings["artifact_cpu_transfer"] = time.perf_counter() - stage
    _stop(check_stop)
    diagnostics["compression"].update(timings_seconds=timings,
        elapsed_seconds=time.perf_counter() - started)
    # Reject overflow in scalar diagnostics rather than emitting invalid JSON.
    json.dumps(diagnostics, allow_nan=False)
    return artifact, diagnostics


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__), root / "src" / "representations.py", root / "src" / "seeding.py")
    return _hash({str(i): hashlib.sha256(path.read_bytes()).hexdigest()
                  for i, path in enumerate(paths)})


def _profile_config(dimension, rank, device, seed, repeats, threads,
                    max_seconds, attempt_timeout_seconds) -> dict:
    _integer(dimension, "dimension")
    _integer(rank, "rank", 0)
    _integer(repeats, "repeats", 2)
    _integer(threads, "threads")
    derive_component_seed(seed, "profile_seed_validation")
    if device not in ("cpu", "cuda"):
        raise ValueError("profile device must be cpu or cuda")
    if (not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800
            or not math.isfinite(attempt_timeout_seconds)
            or not 0 < attempt_timeout_seconds <= 1800):
        raise ValueError("profile and attempt time limits must lie in (0, 1800]")
    return dict(schema_version=PROFILE_SCHEMA_VERSION, dimension=dimension, rank=rank,
                device=device, seed=seed, repeats=repeats, threads=threads,
                max_seconds=float(max_seconds),
                attempt_timeout_seconds=float(attempt_timeout_seconds),
                source_hash=_source_hash(), tolerance_policy=TOLERANCE_POLICY,
                audit_relative_tolerance=AUDIT_RELATIVE_TOLERANCE)


def _case_seeds(seed: int, case: str) -> dict:
    return {name: derive_component_seed(seed, f"grid_compression_profile:{case}:{name}")
            for name in ("factors", "diagonal", "probes", "solve_rhs", "compression")}


def _profile_source(dimension: int, case: str, seeds: dict) -> Tensor:
    """Numerical PSD stress inputs, not auxiliary-score or model observations."""
    if case not in CASES:
        raise ValueError("unknown PSD stress case")
    generator = torch.Generator().manual_seed(seeds["factors"])
    width = 1 if case == "near_rank_one" else min(32, dimension)
    factors = torch.randn(dimension, width, generator=generator, dtype=torch.float64)
    factors /= factors.norm(dim=0)
    if case == "near_rank_one":
        diagonal = torch.full((dimension,), 1e-12, dtype=torch.float64)
    else:
        generator = torch.Generator().manual_seed(seeds["diagonal"])
        diagonal = .01 + .1 * torch.rand(dimension, generator=generator, dtype=torch.float64)
        factors *= torch.linspace(1., .2, width, dtype=torch.float64)
    return factors @ factors.mT + torch.diag(diagonal)


def _audit(artifact: dict, reference: dict, source: Tensor, seeds: dict) -> dict:
    actual = representation_from_artifact(artifact)
    expected = representation_from_artifact(reference)
    stage = time.perf_counter()
    dense = actual.to_dense()
    reconstruction_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    reference_dense = expected.to_dense()
    tiny = torch.finfo(torch.float64).tiny

    def relative(a, b):
        return float((a - b).norm() / b.norm().clamp_min(tiny))

    generator = torch.Generator().manual_seed(seeds["probes"])
    probes = torch.randn(len(source), 8, generator=generator, dtype=torch.float64)
    q_actual = (probes * actual.matvec(probes)).sum(0)
    q_reference = (probes * expected.matvec(probes)).sum(0)
    generator = torch.Generator().manual_seed(seeds["solve_rhs"])
    rhs = torch.randn(len(source), 2, generator=generator, dtype=torch.float64)
    damping = max(float(source.diagonal().mean()) * .1, 1e-6)
    solution = actual.damped_solve(rhs, damping)
    reference_solution = expected.damped_solve(rhs, damping)
    errors = dict(matrix_relative_error=relative(dense, reference_dense),
        diagonal_relative_error=relative(actual.diagonal_vector(), source.diagonal()),
        trace_relative_error=abs(float(actual.diagonal_vector().sum() - source.trace()))
            / max(abs(float(source.trace())), tiny),
        quadratic_form_relative_error=relative(q_actual, q_reference),
        accepted_solve_relative_error=relative(solution, reference_solution))
    return dict(**errors, validated=all(math.isfinite(v) and v <= AUDIT_RELATIVE_TOLERANCE
                                      for v in errors.values()),
                damping=damping, dense_reconstruction_seconds=reconstruction_seconds,
                audit_seconds=time.perf_counter() - stage,
                source_approximation_relative_error=relative(dense, source))


def _profile_worker(connection, config: dict, case: str, backend: str) -> None:
    started = time.perf_counter()
    try:
        torch.set_num_threads(config["threads"])
        seeds = _case_seeds(config["seed"], case)
        stage = time.perf_counter()
        source = _profile_source(config["dimension"], case, seeds)
        construction_seconds = time.perf_counter() - stage
        reference, reference_diagnostics = compress_fisher(source, config["rank"],
                                                          seeds["compression"])
        device = torch.device(config["device"])
        stage = time.perf_counter()
        stream_source = source.to(device)
        _sync(device)
        source_transfer_seconds = time.perf_counter() - stage
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for repeat in range(-1, config["repeats"]):
            artifact, diagnostics = compress_fisher(stream_source, config["rank"],
                                                    seeds["compression"], backend=backend)
            audit = _audit(artifact, reference, source, seeds)
            timings = diagnostics["compression"]
            rows.append(dict(repeat=repeat, warmup=repeat == -1, diagnostics=diagnostics,
                audit=audit, transaction_seconds=timings["elapsed_seconds"]
                    + audit["dense_reconstruction_seconds"],
                longest_noninterruptible_kernel_seconds=max(timings["timings_seconds"].values()),
                artifact_bytes=sum(v.numel() * v.element_size() for v in artifact.values()
                                   if isinstance(v, Tensor))))
        result = dict(status="ok", case=case, backend=backend, seeds=seeds, rows=rows,
            validated=all(row["audit"]["validated"] for row in rows),
            source_construction_seconds=construction_seconds,
            source_transfer_seconds=source_transfer_seconds,
            reference_seconds=reference_diagnostics["compression"]["elapsed_seconds"],
            dense_matrix_bytes=source.numel() * source.element_size(),
            temporary_disk_bytes=0,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)
                if device.type == "cuda" else 0,
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda"
                else platform.processor() or "CPU")
        result["longest_noninterruptible_stage_seconds"] = max(
            construction_seconds, source_transfer_seconds,
            reference_diagnostics["compression"]["elapsed_seconds"],
            *(max(row["longest_noninterruptible_kernel_seconds"],
                  row["audit"]["dense_reconstruction_seconds"], row["audit"]["audit_seconds"])
              for row in rows))
        json.dumps(result, allow_nan=False)
    except Exception as error:
        result = dict(status="failed", case=case, backend=backend, validated=False,
                      error=f"{type(error).__name__}: {error}")
    result["elapsed_seconds"] = time.perf_counter() - started
    try:
        connection.send(result)
    finally:
        connection.close()


def _run_profile_attempt(config: dict, case: str, backend: str, seconds: float,
                         check_stop: Callable | None) -> dict:
    """Spawn, never fork an initialized CUDA context; bound the entire attempt."""
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_profile_worker, args=(send, config, case, backend))
    started = time.perf_counter()
    try:
        process.start()
        send.close()
        while time.perf_counter() - started < seconds:
            _stop(check_stop)
            if receive.poll(.05):
                try:
                    return receive.recv()
                except EOFError:
                    break
            if not process.is_alive():
                break
        return dict(status="timeout" if process.is_alive() else "failed",
                    case=case, backend=backend, validated=False, exitcode=process.exitcode,
                    elapsed_seconds=time.perf_counter() - started)
    finally:
        send.close()
        receive.close()
        if process.pid is not None:
            process.join(timeout=.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def assess_resources(dimension: int = 3268, rank: int = 8, device: str = "cuda",
                     check_stop: Callable | None = None, *, seed: int = 0,
                     repeats: int = 2, threads: int = 2, max_seconds: float = 600,
                     attempt_timeout_seconds: float = 120) -> dict:
    """Return JSON-only evidence and the fastest validated exact backend.

    No persistent files, assets, global RNG changes, or integration decisions
    are made here. The parent must seal this assessment and charge its elapsed
    time to the shared RuntimeBudget. Synthetic PSD probes are not a forecast
    of training speed; the parent supplies the later whole-update profile.
    """
    _stop(check_stop)
    config = _profile_config(dimension, rank, device, seed, repeats, threads,
                             max_seconds, attempt_timeout_seconds)
    started, start_time = time.perf_counter(), _utc()
    attempts = []
    backends = BACKENDS if device == "cuda" else ("exact_cpu",)
    for backend in backends:
        for case in CASES:
            _stop(check_stop)
            remaining = max_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                attempts.append(dict(status="budget_exhausted", backend=backend,
                                     case=case, validated=False))
                continue
            attempts.append(_run_profile_attempt(config, case, backend,
                min(attempt_timeout_seconds, remaining), check_stop))
    summaries = {}
    for backend in backends:
        group = [row for row in attempts if row["backend"] == backend]
        valid = len(group) == len(CASES) and all(row["validated"] for row in group)
        samples = [sample["transaction_seconds"] for row in group
                   for sample in row.get("rows", []) if not sample["warmup"]]
        summaries[backend] = dict(validated=valid,
            median_transaction_seconds=statistics.median(samples) if valid and samples else None)
    eligible = [name for name, row in summaries.items() if row["validated"]]
    selected = min(eligible, key=lambda name: summaries[name]["median_transaction_seconds"]) \
        if eligible else None
    _stop(check_stop)
    result = dict(schema_version=PROFILE_SCHEMA_VERSION, artifact_schema_version=1,
        metric_schema_version=1, experiment="plan15_phase11_compression_resource_assessment",
        replica_id=f"numerical-profile-{seed}", config=config, config_hash=_hash(config),
        started_at=start_time, completed_at=_utc(), elapsed_seconds=time.perf_counter() - started,
        status="complete" if selected else "no_validated_backend", selected_backend=selected,
        selection_rule="lowest median transfer-inclusive compression plus dense reconstruction",
        backends=summaries, attempts=attempts,
        environment=dict(python=platform.python_version(), pytorch=torch.__version__,
                         cuda=torch.version.cuda, device=device, dtype="float64", threads=threads),
        whole_update_profile=dict(status="pending_parent_profile",
            required_before_screening=True, stages=["rendering", "gradients_solves", "fd",
                "jacobian", "fisher_accumulation", "compression", "evaluation", "reporting"]),
        limitation="Numerical PSD stress cases only; no actual-model or whole-study speed claim.")
    json.dumps(result, allow_nan=False)
    return result


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_assessment(output: str | Path, *, expected_config_hash: str | None = None) -> dict:
    """Load sealed JSON evidence only after validating its immutable checksums."""
    output = Path(output)
    marker = json.loads((output / "COMPLETED").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if marker.get("manifest_hash") != _hash(manifest):
        raise ValueError("assessment manifest hash mismatch")
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("incompatible assessment schema")
    if _hash(manifest["config"]) != manifest["config_hash"]:
        raise ValueError("assessment config hash mismatch")
    if expected_config_hash is not None and manifest["config_hash"] != expected_config_hash:
        raise ValueError("assessment config hash mismatch")
    result = json.loads((output / "assessment.json").read_text())
    if (_hash(result) != manifest["assessment_hash"]
            or result.get("config_hash") != manifest["config_hash"]):
        raise ValueError("assessment artifact hash mismatch")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=3268)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--attempt-timeout-seconds", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = _profile_config(args.dimension, args.rank, args.device, args.seed, args.repeats,
                             args.threads, args.max_seconds, args.attempt_timeout_seconds)
    config_hash = _hash(config)
    output = args.output
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing existing diagnostic directory: {output}")
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("config_hash") != config_hash or manifest.get("config") != config:
            raise ValueError("resume config/source hash mismatch")
        if (output / "COMPLETED").exists():
            result = load_assessment(output, expected_config_hash=config_hash)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result["selected_backend"] else 1
        raise ValueError("incomplete assessment cannot be reused; choose a unique output directory")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    manifest = dict(schema_version=PROFILE_SCHEMA_VERSION, config=config,
        config_hash=config_hash, status="running", started_at=_utc(),
        git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")))
    _write_json(output / "manifest.json", manifest)
    result = assess_resources(args.dimension, args.rank, args.device, seed=args.seed,
        repeats=args.repeats, threads=args.threads, max_seconds=args.max_seconds,
        attempt_timeout_seconds=args.attempt_timeout_seconds)
    _write_json(output / "assessment.json", result)
    manifest.update(status="complete", completed_at=_utc(), assessment_hash=_hash(result))
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "COMPLETED", dict(manifest_hash=_hash(manifest)))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["selected_backend"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
