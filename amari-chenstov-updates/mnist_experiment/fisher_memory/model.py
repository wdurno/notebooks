"""Gauge-fixed MNIST CNN with trainable LTM and three LoRA-shaped STM maps."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from mnist_experiment.rotated_mnist.plan12.gauge import (
    GaugeFixedMnistCNN,
    SumToZeroLinear,
)
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
LTM_PARAMETER_COUNT = 487
STM_PARAMETERS_PER_RANK = 88
STM_PARAMETER_COUNT = 176
TOTAL_PARAMETER_COUNT = 663
AFFINE_NAMES = ("features.0", "features.3", "classifier")
FACTOR_SEED_COMPONENT = "fisher_memory_factors"


class _LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.rank = rank
        self.lora_alpha = float(rank)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.weight[0].numel()))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_channels, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        residual = (self.lora_alpha / self.rank) * (self.lora_B @ self.lora_A)
        weight = self.weight + stm_scale * residual.reshape_as(self.weight)
        return nn.functional.conv2d(
            inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _LoRAHead(nn.Module):
    def __init__(self, base: SumToZeroLinear, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.in_features = base.in_features
        self.rank = rank
        self.lora_alpha = float(rank)
        self.register_buffer("contrast", base.contrast)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        residual = (self.lora_alpha / self.rank) * (self.lora_B @ self.lora_A)
        contrast = self.contrast.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = contrast @ (self.weight + stm_scale * residual)
        return nn.functional.linear(inputs, weight, contrast @ self.bias)


class FisherMemoryCNN(GaugeFixedMnistCNN):
    """Canonical 487-coordinate base plus rank-r adapters; no STM biases.

    The base is initialized completely before drawing any adapter parameters.
    A is fan-in-scaled uniform and B is zero, so the initial function equals
    the base model but ordinary loss gradients can immediately update B.
    The factor chart and alpha/r = 1 remain fixed throughout training.
    """

    def __init__(self, rank: int, *, factor_seed: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        super().__init__()
        self.rank = rank
        self.factor_seed = factor_seed
        generator = torch.Generator(device="cpu").manual_seed(factor_seed)
        self.features[0] = _LoRAConv2d(self.features[0], rank, generator)
        self.features[3] = _LoRAConv2d(self.features[3], rank, generator)
        # Fixed 28x28 inputs give a 14x14 map: these four nonoverlapping
        # averages equal adaptive 2x2 pooling, with deterministic CUDA backward.
        self.features[5] = nn.AvgPool2d(kernel_size=7, stride=7)
        self.classifier = _LoRAHead(self.classifier, rank, generator)
        ltm, stm = memory_indices(ParameterLayout.from_module(self))
        if ltm.numel() != LTM_PARAMETER_COUNT or stm.numel() != STM_PARAMETERS_PER_RANK * rank:
            raise RuntimeError("unexpected LTM/STM parameter counts")
        if rank == 2 and ltm.numel() + stm.numel() != TOTAL_PARAMETER_COUNT:
            raise RuntimeError("rank-two memory model must have 663 parameters")

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("the fixed MNIST pooling chart requires (N, 1, 28, 28) inputs")
        hidden = self.features[0](inputs, stm_scale=stm_scale)
        hidden = self.features[2](self.features[1](hidden))
        hidden = self.features[3](hidden, stm_scale=stm_scale)
        hidden = self.features[5](self.features[4](hidden))
        return self.classifier(hidden.flatten(1), stm_scale=stm_scale)


def build_model(
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    rank: int = 2,
    factor_seed: int | None = None,
) -> tuple[FisherMemoryCNN, ParameterLayout]:
    """Initialize on CPU without advancing global RNG; then move to device.

    The LTM seed matches ``build_gauge_fixed_model(seed)``. The independently
    named adapter seed is recorded on ``model.factor_seed``. An explicit
    ``factor_seed`` lets an experiment supply its own recorded seed mapping.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("the memory model supports float32 and float64")
    default_factor_seed = derive_component_seed(seed, FACTOR_SEED_COMPONENT)
    if factor_seed is None:
        factor_seed = default_factor_seed
    if (isinstance(factor_seed, bool) or not isinstance(factor_seed, int)
            or not 0 <= factor_seed < 2**63):
        raise ValueError("factor_seed must be an integer in [0, 2**63)")
    # manual_seed also reseeds initialized CUDA generators, even for CPU draws.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = FisherMemoryCNN(rank, factor_seed=factor_seed)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Return the disjoint LTM/STM partition as CPU int64 global indices."""
    ltm, stm = [], []
    expected = {
        f"{prefix}.{suffix}"
        for prefix in AFFINE_NAMES
        for suffix in ("weight", "bias", "lora_A", "lora_B")
    }
    if set(layout.names) != expected:
        raise ValueError("layout is not the Fisher memory parameter chart")
    for spec in layout.specs:
        destination = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        destination.extend(range(spec.start, spec.stop))
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)


def flat_logits(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    *,
    stm_scale: float = 1.0,
) -> Tensor:
    """Differentiable functional logits without copying theta into the module.

    ``theta`` and ``x`` must share dtype/device. Functional double-precision
    evaluation of a float32 module is allowed; the fixed Helmert buffer is
    converted by the head, and the recorded parameter ordering is unchanged.
    """
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")
    return functional_call(
        model,
        (layout.unflatten_named(theta), dict(model.named_buffers())),
        (x,),
        {"stm_scale": stm_scale},
        strict=True,
    )


def effective_residuals(
    model: FisherMemoryCNN, layout: ParameterLayout, theta: Tensor
) -> dict[str, Tensor]:
    """Return the three differentiable output-by-input STM residual matrices."""
    parameters = layout.unflatten_named(theta)
    result = {}
    for prefix in AFFINE_NAMES:
        module = model.get_submodule(prefix)
        result[prefix] = (module.lora_alpha / module.rank) * (
            parameters[f"{prefix}.lora_B"] @ parameters[f"{prefix}.lora_A"]
        )
    return result
