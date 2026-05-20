from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _grad_to_vector(parameters: list[nn.Parameter]) -> Tensor:
    pieces: list[Tensor] = []
    for param in parameters:
        if param.grad is None:
            pieces.append(torch.zeros_like(param).flatten())
        else:
            pieces.append(param.grad.detach().flatten())
    return torch.cat(pieces)


@dataclass
class SpectralStats:
    eigenvalues: Tensor
    trace: float
    condition_number: float
    effective_rank: float

    @classmethod
    def from_eigenvalues(cls, eigenvalues: Tensor, *, eps: float = 1e-12) -> "SpectralStats":
        values = eigenvalues.detach().float().cpu().clamp_min(0)
        trace = float(values.sum().item())
        positive = values[values > eps]
        if positive.numel() == 0:
            return cls(values, 0.0, float("inf"), 0.0)
        condition = float((positive.max() / positive.min()).item())
        p = positive / positive.sum()
        entropy_rank = float(torch.exp(-(p * torch.log(p + eps)).sum()).item())
        return cls(values, trace, condition, entropy_rank)


class OnlineDiagonalEWC:
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        rho: float = 0.01,
        lambda_: float = 10.0,
    ) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        self.rho = rho
        self.lambda_ = lambda_
        self.fisher = {name: torch.zeros_like(param.detach()) for name, param in named_parameters}
        self.reference = {name: param.detach().clone() for name, param in named_parameters}

    def penalty(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        penalties: list[Tensor] = []
        for name, param in named_parameters:
            penalties.append((self.fisher[name] * (param - self.reference[name]).square()).sum())
        if not penalties:
            raise ValueError("EWC has no parameters")
        return 0.5 * self.lambda_ * torch.stack(penalties).sum()

    @torch.no_grad()
    def update_from_squared_grads(self, squared_grads: dict[str, Tensor]) -> None:
        for name, value in squared_grads.items():
            self.fisher[name].mul_(1.0 - self.rho).add_(value, alpha=self.rho)

    @torch.no_grad()
    def refresh_reference(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> None:
        for name, param in named_parameters:
            self.reference[name] = param.detach().clone()

    def flat_fisher(self) -> Tensor:
        return torch.cat([value.detach().flatten().cpu() for value in self.fisher.values()])


def estimate_diagonal_fisher(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
    *,
    max_samples: int | None = None,
) -> dict[str, Tensor]:
    model.eval()
    n = x.shape[0] if max_samples is None else min(max_samples, x.shape[0])
    accum = {name: torch.zeros_like(param.detach()) for name, param in named_parameters}

    for i in range(n):
        model.zero_grad(set_to_none=True)
        logits = model(x[i : i + 1])
        loss = F.cross_entropy(logits, y[i : i + 1])
        loss.backward()
        for name, param in named_parameters:
            if param.grad is not None:
                accum[name].add_(param.grad.detach().square())

    for name in accum:
        accum[name].div_(float(n))
    model.zero_grad(set_to_none=True)
    return accum


def empirical_fisher_eigenvalues(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    parameters: list[nn.Parameter],
    *,
    max_samples: int | None = None,
    chunk_size: int = 256,
) -> Tensor:
    """Return nonzero empirical Fisher eigenvalues using a score Gram matrix."""

    model.eval()
    n = x.shape[0] if max_samples is None else min(max_samples, x.shape[0])
    scores: list[Tensor] = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        for i in range(start, stop):
            model.zero_grad(set_to_none=True)
            logits = model(x[i : i + 1])
            loss = F.cross_entropy(logits, y[i : i + 1])
            loss.backward()
            scores.append(_grad_to_vector(parameters).cpu())

    score_matrix = torch.stack(scores)
    gram = score_matrix @ score_matrix.T
    gram.div_(float(n))
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).sort(descending=True).values
    model.zero_grad(set_to_none=True)
    return eigenvalues
