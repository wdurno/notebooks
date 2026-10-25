"""Smooth scalar-angle CNN with 287 LTM and 135 LoRA-shaped STM coordinates."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
LTM_PARAMETER_COUNT = 287
STM_PARAMETER_COUNT = 135
TOTAL_PARAMETER_COUNT = 422
ANGLE_SCALE_DEGREES = 45.0
AFFINE_NAMES = ("features.0", "features.3", "classifier")
FACTOR_SEED_COMPONENT = "regression_memory_factors"


class _LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, generator: torch.Generator) -> None:
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.stride, self.padding = base.stride, base.padding
        self.dilation, self.groups = base.dilation, base.groups
        self.rank, self.lora_alpha = 2, 2.0
        self.lora_A = nn.Parameter(base.weight.new_empty(2, base.weight[0].numel()))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_channels, 2))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        residual = (self.lora_B @ self.lora_A).reshape_as(self.weight)
        return nn.functional.conv2d(
            inputs, self.weight + stm_scale * residual, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class _LoRAHead(nn.Module):
    def __init__(self, base: nn.Linear, generator: torch.Generator) -> None:
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.in_features, self.out_features = base.in_features, 1
        self.rank, self.lora_alpha = 1, 1.0
        self.lora_A = nn.Parameter(base.weight.new_empty(1, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        weight = self.weight + stm_scale * (self.lora_B @ self.lora_A)
        return nn.functional.linear(inputs, weight, self.bias)


class RegressionMemoryCNN(nn.Module):
    """Predict an unbounded scalar mean in units of 45 degrees, without a gauge.

    Initialize the complete base before the independently seeded factors.
    Random A and zero B make the initial function independent of the factor
    seed while leaving nonzero gradients available to B on the first update.
    """

    def __init__(self, *, factor_seed: int) -> None:
        super().__init__()
        self.factor_seed = factor_seed
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(4, 6, kernel_size=3, padding=1), nn.SiLU(),
            nn.AvgPool2d(kernel_size=7, stride=7),
        )
        self.classifier = nn.Linear(24, 1)
        generator = torch.Generator(device="cpu").manual_seed(factor_seed)
        self.features[0] = _LoRAConv2d(self.features[0], generator)
        self.features[3] = _LoRAConv2d(self.features[3], generator)
        self.classifier = _LoRAHead(self.classifier, generator)
        layout = ParameterLayout.from_module(self)
        ltm, stm = memory_indices(layout)
        if (ltm.numel(), stm.numel(), layout.total_numel) != (
            LTM_PARAMETER_COUNT, STM_PARAMETER_COUNT, TOTAL_PARAMETER_COUNT
        ):
            raise RuntimeError("regression memory model must have 287 LTM + 135 STM parameters")

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("the regression model requires (N, 1, 28, 28) inputs")
        hidden = self.features[0](inputs, stm_scale=stm_scale)
        hidden = self.features[2](self.features[1](hidden))
        hidden = self.features[3](hidden, stm_scale=stm_scale)
        hidden = self.features[5](self.features[4](hidden))
        return self.classifier(hidden.flatten(1), stm_scale=stm_scale).squeeze(-1)


def build_model(
    seed: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    factor_seed: int | None = None,
) -> tuple[RegressionMemoryCNN, ParameterLayout]:
    """Initialize on CPU without advancing global random generators."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("the regression model supports float32 and float64")
    if factor_seed is None:
        factor_seed = derive_component_seed(seed, FACTOR_SEED_COMPONENT)
    for name, value in (("seed", seed), ("factor_seed", factor_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            raise ValueError(f"{name} must be an integer in [0, 2**63)")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = RegressionMemoryCNN(factor_seed=factor_seed)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Return disjoint CPU int64 coordinates; reject categorical parameter charts."""
    shapes = {
        "features.0": ((4, 1, 3, 3), (4,), (2, 9), (4, 2)),
        "features.3": ((6, 4, 3, 3), (6,), (2, 36), (6, 2)),
        "classifier": ((1, 24), (1,), (1, 24), (1, 1)),
    }
    expected = {
        f"{prefix}.{suffix}": shape
        for prefix, entries in shapes.items()
        for suffix, shape in zip(("weight", "bias", "lora_A", "lora_B"), entries)
    }
    if {spec.name: spec.shape for spec in layout.specs} != expected:
        raise ValueError("layout is not the scalar regression memory parameter chart")
    ltm, stm = [], []
    for spec in layout.specs:
        target = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        target.extend(range(spec.start, spec.stop))
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)


def flat_prediction(model: RegressionMemoryCNN, layout: ParameterLayout,
                    theta: Tensor, x: Tensor, *, stm_scale: float = 1.0) -> Tensor:
    """Differentiable normalized mean (N,), without copying into module state."""
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")
    return functional_call(
        model, (layout.unflatten_named(theta), dict(model.named_buffers())),
        (x,), {"stm_scale": stm_scale}, strict=True,
    )
