"""Smooth paired-image CNN with physical-pixel means and LoRA memory blocks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
GAUSSIAN_SCALE_PIXELS = 2.0
AFFINE_NAMES = ("features.0", "features.3", "classifier")
FACTOR_SEED_COMPONENT = "registration_memory_factors"


def _architecture(output_dim: int, channels) -> tuple[int, int]:
    if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim not in (1, 6):
        raise ValueError("registration output_dim must be 1 or 6")
    if (not isinstance(channels, (tuple, list)) or len(channels) != 2
            or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in channels)):
        raise ValueError("channels must contain two positive integers")
    return tuple(channels)


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
        weight = self.weight + stm_scale * (self.lora_alpha / self.rank) * residual
        return nn.functional.conv2d(inputs, weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


class _LoRAHead(nn.Module):
    def __init__(self, base: nn.Linear, generator: torch.Generator) -> None:
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank = 1 if base.out_features == 1 else 2
        self.lora_alpha = float(self.rank)
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        residual = (self.lora_alpha / self.rank) * (self.lora_B @ self.lora_A)
        return nn.functional.linear(inputs, self.weight + stm_scale * residual, self.bias)


class RegistrationMemoryCNN(nn.Module):
    """Predict unbounded landmark displacements in pixels, including for d=1.

    Initialize the complete base before independently seeded random-A/zero-B
    factors. A factor seed therefore cannot change the initial predictions.
    """

    def __init__(self, *, output_dim: int = 6, channels=(4, 6), factor_seed: int) -> None:
        super().__init__()
        self.channels = _architecture(output_dim, channels)
        self.output_dim, self.factor_seed = output_dim, factor_seed
        c1, c2 = self.channels
        self.features = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.SiLU(),
            nn.AvgPool2d(kernel_size=7, stride=7),
        )
        self.classifier = nn.Linear(4 * c2, output_dim)
        generator = torch.Generator(device="cpu").manual_seed(factor_seed)
        self.features[0] = _LoRAConv2d(self.features[0], generator)
        self.features[3] = _LoRAConv2d(self.features[3], generator)
        self.classifier = _LoRAHead(self.classifier, generator)
        layout = ParameterLayout.from_module(self)
        ltm, stm = memory_indices(layout)
        head_rank = self.classifier.rank
        expected_ltm = 19 * c1 + (9 * c1 + 1) * c2 + (4 * c2 + 1) * output_dim
        expected_stm = 36 + 20 * c1 + 2 * c2 + head_rank * (4 * c2 + output_dim)
        if (len(ltm), len(stm), layout.total_numel) != (
                expected_ltm, expected_stm, expected_ltm + expected_stm):
            raise RuntimeError("registration memory parameter counts do not match the architecture")

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 28, 28):
            raise ValueError("the registration model requires (N, 2, 28, 28) inputs")
        hidden = self.features[0](inputs, stm_scale=stm_scale)
        hidden = self.features[2](self.features[1](hidden))
        hidden = self.features[3](hidden, stm_scale=stm_scale)
        hidden = self.features[5](self.features[4](hidden))
        return self.classifier(hidden.flatten(1), stm_scale=stm_scale)


def build_model(
    seed: int, *, output_dim: int = 6, channels=(4, 6),
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    factor_seed: int | None = None,
) -> tuple[RegistrationMemoryCNN, ParameterLayout]:
    """Initialize on CPU without advancing global random generators."""
    channels = _architecture(output_dim, channels)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("the registration model supports float32 and float64")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if factor_seed is None:
        factor_seed = derive_component_seed(seed, FACTOR_SEED_COMPONENT)
    if isinstance(factor_seed, bool) or not isinstance(factor_seed, int) or not 0 <= factor_seed < 2**63:
        raise ValueError("factor_seed must be an integer in [0, 2**63)")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = RegistrationMemoryCNN(output_dim=output_dim, channels=channels,
                                      factor_seed=factor_seed)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Validate the registration chart at either width; return CPU int64 blocks."""
    supplied = {spec.name: spec.shape for spec in layout.specs}
    first, second, head = (supplied.get(f"{prefix}.weight", ()) for prefix in AFFINE_NAMES)
    if len(first) != 4 or len(second) != 4 or len(head) != 2:
        raise ValueError("layout is not a registration memory parameter chart")
    output_dim, channels = head[0], (first[0], second[0])
    c1, c2 = _architecture(output_dim, channels)
    rank = 1 if output_dim == 1 else 2
    shapes = {
        "features.0": ((c1, 2, 3, 3), (c1,), (2, 18), (c1, 2)),
        "features.3": ((c2, c1, 3, 3), (c2,), (2, 9 * c1), (c2, 2)),
        "classifier": ((output_dim, 4 * c2), (output_dim,),
                       (rank, 4 * c2), (output_dim, rank)),
    }
    expected = {f"{prefix}.{suffix}": shape for prefix, entries in shapes.items()
                for suffix, shape in zip(("weight", "bias", "lora_A", "lora_B"), entries)}
    if supplied != expected or layout.names != tuple(expected):
        raise ValueError("layout is not a registration memory parameter chart")
    ltm, stm, offset = [], [], 0
    for spec in layout.specs:
        if spec.start != offset or spec.stop != offset + math.prod(spec.shape):
            raise ValueError("registration parameter offsets must be contiguous and match shapes")
        target = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        target.extend(range(spec.start, spec.stop))
        offset = spec.stop
    if offset != layout.total_numel:
        raise ValueError("registration parameter count does not match the layout")
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)


def flat_prediction(model: RegistrationMemoryCNN, layout: ParameterLayout,
                    theta: Tensor, x: Tensor, *, stm_scale: float = 1.0) -> Tensor:
    """Differentiable physical-pixel means (N, d), without mutating module state."""
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")
    return functional_call(model, (layout.unflatten_named(theta), dict(model.named_buffers())),
                           (x,), {"stm_scale": stm_scale}, strict=True)
