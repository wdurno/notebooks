"""Validated deterministic wrapper around the copied legacy Lanczos routine."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .lanczos import l_lanczos
from .representations import LowRankDiagonalFisher

LEGACY_LANCZOS_SHA256 = (
    "643bb7562ad7ad2d087229414d7adf114f575b342f462be619f90be920691634"
)


def legacy_lanczos_source_hash() -> str:
    path = Path(__file__).with_name("lanczos.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class LanczosDiagnostics:
    requested_rank: int
    realized_rank: int
    retained_eigenvalues: tuple[float, ...]
    residual_diagonal_minimum: float
    residual_diagonal_maximum: float
    residual_diagonal_mean: float
    residual_diagonal_zero_count: int
    represented_diagonal_relative_error: float
    elapsed_seconds: float
    seed: int
    legacy_source_hash: str

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LanczosApproximation:
    representation: LowRankDiagonalFisher
    diagnostics: LanczosDiagnostics


def approximate_low_rank_diagonal(
    operator: Callable[[Tensor], Tensor],
    diagonal: Tensor,
    *,
    rank: int,
    seed: int,
) -> LanczosApproximation:
    """Approximate a symmetric operator by ``AA^T + diag(d)``.

    The copied routine supplies the Krylov recurrence. This wrapper owns input
    validation, deterministic initialization, residual-diagonal clipping,
    numerical-breakdown detection, and stable artifact diagnostics.
    """

    source_hash = legacy_lanczos_source_hash()
    if source_hash != LEGACY_LANCZOS_SHA256:
        raise RuntimeError(
            "src/lanczos.py changed; review and update the wrapper contract"
        )
    if (
        diagonal.ndim != 1
        or not diagonal.is_floating_point()
        or not torch.isfinite(diagonal).all()
        or diagonal.dtype not in {torch.float32, torch.float64}
    ):
        raise ValueError(
            "operator diagonal must be a finite float32 or float64 vector"
        )
    parameter_count = diagonal.numel()
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank <= parameter_count
    ):
        raise ValueError("rank must be an integer in [0, parameter_count]")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")

    probe = torch.zeros(
        parameter_count,
        1,
        device=diagonal.device,
        dtype=diagonal.dtype,
    )
    probe[0] = 1
    output = operator(probe)
    if (
        output.shape != probe.shape
        or output.device != diagonal.device
        or output.dtype != diagonal.dtype
        or not torch.isfinite(output).all()
    ):
        raise ValueError(
            "operator must preserve shape, dtype, device, and finiteness"
        )

    started = time.perf_counter()
    if rank == 0:
        factor = diagonal.new_zeros((parameter_count, 0))
        residual = diagonal.clamp_min(0)
        retained_eigenvalues: tuple[float, ...] = ()
    else:
        devices = (
            [diagonal.device.index or 0]
            if diagonal.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if diagonal.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            default_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(diagonal.dtype)
                try:
                    factor, residual_column = l_lanczos(
                        get_grad_generator=None,
                        r=rank,
                        p=parameter_count,
                        device=diagonal.device,
                        mfi_alternate=operator,
                        diag_alternate=lambda: diagonal.reshape(-1, 1).clone(),
                        disable_tqdm=True,
                        calc_diag=True,
                    )
                finally:
                    torch.set_default_dtype(default_dtype)
            except (RuntimeError, ZeroDivisionError) as exc:
                raise RuntimeError(
                    f"legacy Lanczos failed at requested rank {rank}: {exc}"
                ) from exc
        factor = factor.to(device=diagonal.device, dtype=diagonal.dtype)
        residual = residual_column.reshape(-1).to(
            device=diagonal.device,
            dtype=diagonal.dtype,
        )
        if not torch.isfinite(factor).all() or not torch.isfinite(residual).all():
            raise RuntimeError(
                f"legacy Lanczos numerical breakdown at requested rank {rank}"
            )
        gram = (factor.mT @ factor)
        gram = (gram + gram.mT) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        largest = max(float(eigenvalues.max()), 0.0)
        tolerance = (
            torch.finfo(diagonal.dtype).eps
            * max(parameter_count, rank)
            * max(largest, 1.0)
        )
        retained = eigenvalues > tolerance
        factor = factor @ eigenvectors[:, retained]
        retained_eigenvalues = tuple(
            float(value)
            for value in eigenvalues[retained].flip(0)
        )
        residual = residual.clamp_min(0)

    representation = LowRankDiagonalFisher(factor, residual)
    represented_diagonal = representation.diagonal_vector()
    denominator = torch.linalg.vector_norm(diagonal).clamp_min(
        torch.finfo(diagonal.dtype).eps
    )
    diagonal_error = (
        torch.linalg.vector_norm(represented_diagonal - diagonal) / denominator
    )
    diagnostics = LanczosDiagnostics(
        requested_rank=rank,
        realized_rank=representation.rank,
        retained_eigenvalues=retained_eigenvalues,
        residual_diagonal_minimum=float(residual.min()),
        residual_diagonal_maximum=float(residual.max()),
        residual_diagonal_mean=float(residual.mean()),
        residual_diagonal_zero_count=int((residual == 0).sum()),
        represented_diagonal_relative_error=float(diagonal_error),
        elapsed_seconds=time.perf_counter() - started,
        seed=seed,
        legacy_source_hash=source_hash,
    )
    if not math.isfinite(diagnostics.elapsed_seconds):
        raise RuntimeError("Lanczos elapsed time is nonfinite")
    return LanczosApproximation(representation, diagnostics)
