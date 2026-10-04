"""Identifiable sum-to-zero chart for the canonical MNIST classifier."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.mnist_model import CanonicalMnistCNN
from src.parameters import ParameterLayout


CLASS_COUNT = 10
CLASSIFIER_FEATURES = 24
GAUGE_DIMENSION = CLASSIFIER_FEATURES + 1
GAUGE_FIXED_PARAMETER_COUNT = 487


def helmert_contrast(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return a deterministic orthonormal basis for the zero-sum class space."""

    result = torch.zeros(CLASS_COUNT, CLASS_COUNT - 1, device=device, dtype=dtype)
    for column in range(CLASS_COUNT - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        result[: column + 1, column] = 1.0 / scale
        result[column + 1, column] = -(column + 1) / scale
    return result


class SumToZeroLinear(nn.Module):
    """Linear classifier parameterized by nine orthonormal class contrasts."""

    def __init__(self, in_features: int = CLASSIFIER_FEATURES) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.weight = nn.Parameter(torch.empty(CLASS_COUNT - 1, self.in_features))
        self.bias = nn.Parameter(torch.empty(CLASS_COUNT - 1))
        self.register_buffer("contrast", helmert_contrast(dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def full_parameters(self) -> tuple[Tensor, Tensor]:
        contrast = self.contrast.to(device=self.weight.device, dtype=self.weight.dtype)
        return contrast @ self.weight, contrast @ self.bias

    def forward(self, inputs: Tensor) -> Tensor:
        weight, bias = self.full_parameters()
        return nn.functional.linear(inputs, weight, bias)


class GaugeFixedMnistCNN(nn.Module):
    """Canonical smooth CNN with the exact common-logit gauge removed."""

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
        self.classifier = SumToZeroLinear(CLASSIFIER_FEATURES)
        parameter_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        if parameter_count != GAUGE_FIXED_PARAMETER_COUNT:
            raise RuntimeError(
                "gauge-fixed model must have "
                f"{GAUGE_FIXED_PARAMETER_COUNT} trainable parameters, got {parameter_count}"
            )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, start_dim=1))


def build_gauge_fixed_model(
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GaugeFixedMnistCNN, ParameterLayout]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GaugeFixedMnistCNN()
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def load_canonical_state(
    target: GaugeFixedMnistCNN,
    source: CanonicalMnistCNN | Mapping[str, Tensor],
) -> None:
    """Load a raw canonical state after projecting away its common-logit gauge."""

    state = source.state_dict() if isinstance(source, CanonicalMnistCNN) else source
    target_state = target.state_dict()
    with torch.no_grad():
        for name in target_state:
            if name.startswith("features."):
                target_state[name].copy_(state[name].to(target_state[name]))
        contrast = target.classifier.contrast.to(
            device=target.classifier.weight.device,
            dtype=target.classifier.weight.dtype,
        )
        raw_weight = state["classifier.weight"].to(target.classifier.weight)
        raw_bias = state["classifier.bias"].to(target.classifier.bias)
        target.classifier.weight.copy_(contrast.mT @ raw_weight)
        target.classifier.bias.copy_(contrast.mT @ raw_bias)


def load_gauge_state(
    target: CanonicalMnistCNN,
    source: GaugeFixedMnistCNN,
) -> None:
    """Load the canonical zero-sum representative of a gauge-fixed state."""

    target.features.load_state_dict(source.features.state_dict())
    weight, bias = source.classifier.full_parameters()
    with torch.no_grad():
        target.classifier.weight.copy_(weight.to(target.classifier.weight))
        target.classifier.bias.copy_(bias.to(target.classifier.bias))


def chart_embedding(
    raw_layout: ParameterLayout,
    chart_layout: ParameterLayout,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return the orthonormal Jacobian mapping chart perturbations to raw ones."""

    if raw_layout.total_numel != 512 or chart_layout.total_numel != 487:
        raise ValueError("unexpected raw or chart parameter dimension")
    embedding = torch.zeros(
        raw_layout.total_numel,
        chart_layout.total_numel,
        device=device,
        dtype=dtype,
    )
    raw_specs = {spec.name: spec for spec in raw_layout.specs}
    chart_specs = {spec.name: spec for spec in chart_layout.specs}
    for name, chart_spec in chart_specs.items():
        raw_spec = raw_specs[name]
        if name.startswith("features."):
            if raw_spec.numel != chart_spec.numel:
                raise ValueError(f"feature shape differs for {name}")
            embedding[
                raw_spec.start : raw_spec.stop,
                chart_spec.start : chart_spec.stop,
            ] = torch.eye(chart_spec.numel, device=device, dtype=dtype)
        elif name == "classifier.weight":
            contrast = helmert_contrast(device=device, dtype=dtype)
            block = torch.kron(
                contrast,
                torch.eye(CLASSIFIER_FEATURES, device=device, dtype=dtype),
            )
            embedding[
                raw_spec.start : raw_spec.stop,
                chart_spec.start : chart_spec.stop,
            ] = block
        elif name == "classifier.bias":
            embedding[
                raw_spec.start : raw_spec.stop,
                chart_spec.start : chart_spec.stop,
            ] = helmert_contrast(device=device, dtype=dtype)
        else:
            raise ValueError(f"unexpected chart parameter: {name}")
    return embedding


def exact_gauge_basis(
    raw_layout: ParameterLayout,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return 25 orthonormal common-classifier gauge directions."""

    if raw_layout.total_numel != 512:
        raise ValueError("exact gauge is defined for the canonical 512-vector")
    specs = {spec.name: spec for spec in raw_layout.specs}
    weight = specs["classifier.weight"]
    bias = specs["classifier.bias"]
    result = torch.zeros(512, GAUGE_DIMENSION, device=device, dtype=dtype)
    scale = 1 / math.sqrt(CLASS_COUNT)
    for feature in range(CLASSIFIER_FEATURES):
        for class_index in range(CLASS_COUNT):
            result[weight.start + class_index * CLASSIFIER_FEATURES + feature, feature] = scale
    result[bias.start : bias.stop, -1] = scale
    return result


def split_raw_vector(
    vector: Tensor,
    embedding: Tensor,
    gauge_basis: Tensor,
) -> tuple[Tensor, Tensor]:
    if vector.ndim != 1 or vector.shape[0] != embedding.shape[0]:
        raise ValueError("raw vector has incompatible shape")
    if gauge_basis.shape[0] != vector.shape[0]:
        raise ValueError("gauge basis has incompatible shape")
    return embedding.mT @ vector, gauge_basis.mT @ vector


def raw_fisher_to_chart(matrix: Tensor, embedding: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.shape != (embedding.shape[0], embedding.shape[0]):
        raise ValueError("raw Fisher and chart embedding are incompatible")
    transformed = embedding.mT @ ((matrix + matrix.mT) / 2) @ embedding
    return (transformed + transformed.mT) / 2
