"""Canonical smooth CNN used by the MNIST LFU experiment."""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn

from .parameters import ParameterLayout

MNIST_MODEL_SCHEMA_VERSION = 1
CANONICAL_PARAMETER_COUNT = 512


class CanonicalMnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Linear(24, 10)
        parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        if parameter_count != CANONICAL_PARAMETER_COUNT:
            raise RuntimeError(
                f"canonical model must have {CANONICAL_PARAMETER_COUNT} "
                f"trainable parameters, got {parameter_count}"
            )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, start_dim=1))


def resolve_dtype(name: str) -> torch.dtype:
    try:
        dtype = {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc
    return dtype


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {name}")
    return torch.device(name)


def configure_torch_runtime(
    *,
    deterministic_algorithms: bool,
    warn_only: bool = False,
) -> None:
    """Configure the process before model construction or CUDA execution."""

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=warn_only,
    )
    torch.backends.cudnn.deterministic = deterministic_algorithms
    torch.backends.cudnn.benchmark = not deterministic_algorithms


def build_canonical_model(
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[CanonicalMnistCNN, ParameterLayout]:
    """Build from a CPU-seeded state, then move it to the execution device."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CanonicalMnistCNN()
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def mnist_nll(logits: Tensor, targets: Tensor) -> Tensor:
    """Scalar negative log likelihood used by per-sample autodiff calls."""

    return nn.functional.cross_entropy(logits, targets, reduction="sum")


def mnist_losses(logits: Tensor, targets: Tensor) -> Tensor:
    """Unreduced negative log likelihood used for importance ratios."""

    return nn.functional.cross_entropy(logits, targets, reduction="none")
