"""Assessment-only block-power/Nystrom candidate, outside the frozen engine.

For PSD M, Y=MQ and W=Q.T@Y give Y W^dagger Y.T <= M. Truncating its
factor preserves this ordering, unlike inserting uncorrected Ritz factors.
The residual diagonal restores the source diagonal and trace, without ridge.
This is not a production backend or an accuracy/speed claim. Admission needs
actual-snapshot comparisons to exact same-rank compression, including matrix
error <=1%, quadratic forms, accepted solves and complete transaction timing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from mnist_experiment.fisher_memory.learner import StopRequested, _integer, _stop
from src.representations import LowRankDiagonalFisher
from src.seeding import derive_component_seed


PROBE_VERSION = 1
ROUNDOFF_MULTIPLIER = 100
SKETCH_SEED_COMPONENT = "plan15_grid_compression_probe:initial_subspace"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite {name}")


def _warm_factor(warm_start, dimension: int, device: torch.device) -> Tensor:
    if warm_start is None:
        return torch.empty(dimension, 0, dtype=torch.float64, device=device)
    if isinstance(warm_start, Mapping):
        if warm_start.get("kind") != "low_rank_diagonal":
            raise ValueError("warm_start artifact must be low_rank_diagonal")
        warm_start = warm_start.get("factor")
    if (not isinstance(warm_start, Tensor) or not warm_start.is_floating_point()
            or warm_start.layout != torch.strided or warm_start.ndim != 2
            or warm_start.shape[0] != dimension):
        raise ValueError("warm_start must be a matching (dimension, modes) factor")
    factor = warm_start.detach().to(device=device, dtype=torch.float64)
    _finite(factor, "warm_start")
    return factor


def approximate_fisher(matrix: Tensor, rank: int = 8, seed: int = 0,
                       warm_start=None, steps: int = 4, oversample: int = 16,
                       check_stop: Callable | None = None) -> tuple[dict, dict]:
    """Return a CPU low-rank-plus-diagonal artifact and JSON diagnostics.

    The float64 computation stays on the input device; CPU serialization and
    transfers are timed. ``warm_start`` is an old archive factor or its artifact,
    not an eigensolver state. At most ``rank`` warm columns join an independently
    named Gaussian oversampling stream. No global RNG state is changed.

    PSD of the source is a caller precondition, not certified here. Only the
    sketch-width core and Gram matrices undergo eigh. Their relative cutoff is
    100*n*float64_eps times their respective spectral radii. Negative residual
    diagonals are rejected beyond 100*n*eps*||M||_infinity; roundoff is recorded
    before clipping. The Loewner guarantee is conditional on PSD input and
    exact arithmetic; retained-factor orthogonality and Ritz residuals expose
    numerical error without another dense spectral decomposition.
    Cutoff diagnostics concern the computed Gram spectrum only: no omitted
    positive mode does not certify a gap in the full source spectrum.
    """
    _integer(rank, "rank", 0)
    _integer(steps, "steps", 0)
    _integer(oversample, "oversample", 0)
    sketch_seed = derive_component_seed(seed, SKETCH_SEED_COMPONENT)
    _stop(check_stop)
    if (not isinstance(matrix, Tensor) or not matrix.is_floating_point()
            or matrix.layout != torch.strided or matrix.ndim != 2
            or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]):
        raise ValueError("Fisher must be a nonempty square floating tensor")
    started = time.perf_counter()
    device = matrix.device
    _sync(device)
    source = matrix.detach().to(dtype=torch.float64)
    _finite(source, "Fisher")
    symmetry_error = float((source - source.mT).norm())
    source = .5 * source + .5 * source.mT
    _finite(source, "symmetrized Fisher")
    dimension = len(source)
    eps = torch.finfo(torch.float64).eps
    tiny = torch.finfo(torch.float64).tiny
    relative_cutoff = ROUNDOFF_MULTIPLIER * dimension * eps
    row_absolute_sums = source.abs().sum(1)
    _finite(row_absolute_sums, "Fisher scale bound")
    scale_bound = max(float(row_absolute_sums.max()), tiny)
    tolerance = relative_cutoff * scale_bound
    if float(source.diagonal().min()) < -tolerance:
        raise ValueError("material negative source diagonal; source is not PSD")
    warm = _warm_factor(warm_start, dimension, device)
    width = min(dimension, rank + oversample) if rank else 0
    warm_count = min(rank, width, warm.shape[1])
    timings = {}
    _sync(device)
    timings["validation_symmetrization"] = time.perf_counter() - started

    def orthonormalize(block):
        _stop(check_stop)
        _finite(block, "power iterate")
        q, _ = torch.linalg.qr(block, mode="reduced")
        _finite(q, "orthonormal basis")
        _stop(check_stop)
        return q

    def positive_spectrum(block, label):
        _stop(check_stop)
        _finite(block, label)
        values, vectors = torch.linalg.eigh(.5 * block + .5 * block.mT)
        _stop(check_stop)
        _finite(values, f"{label} eigenvalues")
        _finite(vectors, f"{label} eigenvectors")
        cutoff = relative_cutoff * max(float(values.abs().max()), tiny)
        if float(values.min()) < -cutoff:
            raise ValueError(f"materially indefinite {label}; source PSD precondition failed")
        return values, vectors, values > cutoff, cutoff

    factor = source.new_empty((dimension, 0))
    retained = source.new_empty(0)
    core_minimum = core_negative_mass = core_cutoff = gram_cutoff = None
    largest_omitted = cutoff_relative_gap = None
    core_rank = 0
    stage = time.perf_counter()
    _stop(check_stop)
    if width:
        generator = torch.Generator(device="cpu").manual_seed(sketch_seed)
        random = torch.randn(dimension, width - warm_count, generator=generator,
                             dtype=torch.float64).to(device)
        warm = warm[:, :warm_count]
        warm = warm / warm.norm(dim=0).clamp_min(tiny)
        q = orthonormalize(torch.cat((warm, random), dim=1))
        for _ in range(steps):
            q = orthonormalize(source @ q)
        _sync(device)
        timings["subspace_power_qr"] = time.perf_counter() - stage
        stage = time.perf_counter()
        y = source @ q
        values, vectors, positive, core_cutoff = positive_spectrum(q.mT @ y, "Nystrom core")
        core_minimum = float(values.min())
        core_negative_mass = float((-values[values < 0]).sum())
        core_rank = int(positive.sum())
        if core_rank:
            # Pseudoinverse whitening is PSD-aware; do not use raw Ritz factors.
            b = (y @ vectors[:, positive]) / values[positive].sqrt()
            gram_values, gram_vectors, gram_positive, gram_cutoff = positive_spectrum(
                b.mT @ b, "Nystrom factor Gram")
            selected = torch.where(gram_positive)[0][-min(rank, int(gram_positive.sum())):]
            omitted = gram_values[:len(gram_values) - len(selected)]
            omitted = omitted[omitted > 0]
            if omitted.numel():
                largest_omitted = float(omitted[-1])
            if selected.numel():
                factor = b @ gram_vectors[:, selected]
                retained = gram_values[selected]
                if largest_omitted is not None:
                    minimum_retained = float(retained.min())
                    cutoff_relative_gap = (minimum_retained - largest_omitted) / max(
                        abs(minimum_retained), abs(largest_omitted), tiny)
    else:
        timings["subspace_power_qr"] = 0.
    _sync(device)
    timings["nystrom_reconstruction"] = time.perf_counter() - stage
    _stop(check_stop)
    stage = time.perf_counter()
    _finite(factor, "retained factor")
    residual = source.diagonal() - factor.square().sum(1)
    _finite(residual, "residual diagonal")
    residual_minimum = float(residual.min())
    if residual_minimum < -tolerance:
        raise ValueError("Nystrom factor materially exceeds the source diagonal")
    representation = LowRankDiagonalFisher(factor, residual.clamp_min(0))
    diagonal = representation.diagonal_vector()
    if not torch.allclose(diagonal, source.diagonal(), rtol=0, atol=tolerance):
        raise ValueError("compression did not preserve the source diagonal")
    if factor.shape[1]:
        directions = factor / retained.sqrt()
        orthogonality = float((directions.mT @ directions - torch.eye(
            len(retained), dtype=torch.float64, device=device)).norm())
        action = source @ directions
        ritz = action - directions * retained
        ritz_norms = ritz.norm(dim=0).tolist()
        ritz_relative = float(ritz.norm() / action.norm().clamp_min(tiny))
    else:
        orthogonality, ritz_norms, ritz_relative = 0., [], 0.
    source_trace, represented_trace = float(source.trace()), float(diagonal.sum())
    diagnostics = {
        "schema_version": PROBE_VERSION,
        "pre_compression_min_eigenvalue": None,
        "pre_compression_negative_mass": None,
        "source_psd_status": "caller_precondition_not_globally_audited",
        "accuracy_status": "pending_actual_snapshot_exact_same_rank_audits",
        "source_trace": source_trace, "represented_trace": represented_trace,
        "compression": dict(algorithm="block_power_psd_nystrom", version=PROBE_VERSION,
            backend=f"nystrom_{device.type}", randomized=True, seed=seed,
            component_seeds={SKETCH_SEED_COMPONENT: sketch_seed}, requested_rank=rank,
            realized_rank=representation.rank, sketch_width=width, steps=steps,
            oversample=oversample, warm_start_columns=warm_count, core_rank=core_rank,
            core_min_eigenvalue=core_minimum, core_negative_mass=core_negative_mass,
            core_pseudoinverse_cutoff=core_cutoff, gram_rank_cutoff=gram_cutoff,
            relative_spectral_cutoff=relative_cutoff,
            tolerance_policy="100*n*float64_eps; spectral cores / infinity-norm source bound",
            diagonal_roundoff_tolerance=tolerance, source_symmetry_error_fro=symmetry_error,
            retained_eigenvalues=retained.flip(0).tolist(),
            largest_omitted_core_eigenvalue=largest_omitted,
            cutoff_relative_gap=cutoff_relative_gap,
            residual_diagonal_minimum_before_clipping=residual_minimum,
            clipped_residual_diagonal_mass=float((-residual[residual < 0]).sum()),
            represented_diagonal_relative_error=float((diagonal - source.diagonal()).norm()
                / source.diagonal().norm().clamp_min(tiny)),
            trace_absolute_error=abs(represented_trace - source_trace),
            retained_directions_orthogonality_error_fro=orthogonality,
            ritz_residual_norms=ritz_norms, ritz_residual_relative_fro=ritz_relative),
    }
    _sync(device)
    timings["residual_and_ritz_diagnostics"] = time.perf_counter() - stage
    _stop(check_stop)
    stage = time.perf_counter()
    artifact = representation.artifact_mapping()
    _sync(device)
    timings["artifact_cpu_transfer"] = time.perf_counter() - stage
    _stop(check_stop)
    diagnostics["compression"].update(timings_seconds=timings,
        elapsed_seconds=time.perf_counter() - started)
    json.dumps(diagnostics, allow_nan=False)
    return artifact, diagnostics
