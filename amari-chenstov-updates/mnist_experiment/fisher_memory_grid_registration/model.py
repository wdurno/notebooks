"""Full-mosaic smooth CNN with a dense 54-output head and rank-two STM."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from mnist_experiment.fisher_memory_registration.model import (
    GAUSSIAN_SCALE_PIXELS, _LoRAConv2d, _LoRAHead,
)
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
OUTPUT_DIM = 54
AFFINE_NAMES = ("features.0", "features.3", "classifier")
FACTOR_SEED_COMPONENT = "grid_registration_memory_factors"
PARAMETER_COUNTS = {(4, 6): (3268, 344), (4, 8): (4314, 384)}


def _architecture(output_dim: int, channels) -> tuple[int, int]:
    if type(output_dim) is not int or output_dim != OUTPUT_DIM:
        raise ValueError("grid registration output_dim must be 54")
    if (not isinstance(channels, (tuple, list)) or len(channels) != 2
            or any(type(c) is not int for c in channels)
            or tuple(channels) not in PARAMETER_COUNTS):
        raise ValueError("grid registration channels must be (4, 6) or (4, 8)")
    return tuple(channels)


class GridRegistrationMemoryCNN(nn.Module):
    """Retain channel/row/column features; never pool away cell position.

    All base weights are trainable LTM. Independently seeded random-A/zero-B
    adapters in both convolutions and the unrestricted dense head form STM.
    """

    def __init__(self, *, output_dim: int = OUTPUT_DIM, channels=(4, 6),
                 factor_seed: int) -> None:
        super().__init__()
        self.channels = _architecture(output_dim, channels)
        self.output_dim, self.factor_seed = output_dim, factor_seed
        c1, c2 = self.channels
        self.features = nn.Sequential(
            nn.Conv2d(2, c1, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.SiLU(), nn.AvgPool2d(14),
        )
        self.classifier = nn.Linear(9 * c2, output_dim)
        generator = torch.Generator(device="cpu").manual_seed(factor_seed)
        self.features[0] = _LoRAConv2d(self.features[0], generator)
        self.features[3] = _LoRAConv2d(self.features[3], generator)
        self.classifier = _LoRAHead(self.classifier, generator)
        layout = ParameterLayout.from_module(self)
        ltm, stm = memory_indices(layout)
        if (ltm.numel(), stm.numel()) != PARAMETER_COUNTS[self.channels]:
            raise RuntimeError("grid registration memory parameter counts do not match")

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 84, 84):
            raise ValueError("the grid registration model requires (N, 2, 84, 84) inputs")
        hidden = self.features[0](inputs, stm_scale=stm_scale)
        hidden = self.features[2](self.features[1](hidden))
        hidden = self.features[3](hidden, stm_scale=stm_scale)
        hidden = self.features[5](self.features[4](hidden))
        return self.classifier(hidden.flatten(1), stm_scale=stm_scale)


# Keep the Phase 10 class name available to shared type annotations/callers.
RegistrationMemoryCNN = GridRegistrationMemoryCNN


def build_model(seed: int, *, output_dim: int = OUTPUT_DIM, channels=(4, 6),
                device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                factor_seed: int | None = None) -> tuple[GridRegistrationMemoryCNN, ParameterLayout]:
    """Initialize on CPU without advancing global random generators."""
    channels = _architecture(output_dim, channels)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("the grid registration model supports float32 and float64")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if factor_seed is None:
        factor_seed = derive_component_seed(seed, FACTOR_SEED_COMPONENT)
    if type(factor_seed) is not int or not 0 <= factor_seed < 2**63:
        raise ValueError("factor_seed must be an integer in [0, 2**63)")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = GridRegistrationMemoryCNN(output_dim=output_dim, channels=channels,
                                          factor_seed=factor_seed)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Validate the versioned grid chart and return CPU int64 LTM/STM indices."""
    supplied = {spec.name: spec.shape for spec in layout.specs}
    first, second, head = (supplied.get(f"{name}.weight", ()) for name in AFFINE_NAMES)
    if len(first) != 4 or len(second) != 4 or len(head) != 2:
        raise ValueError("layout is not a grid registration parameter chart")
    c1, c2 = _architecture(head[0], (first[0], second[0]))
    shapes = {
        "features.0": ((c1, 2, 3, 3), (c1,), (2, 18), (c1, 2)),
        "features.3": ((c2, c1, 3, 3), (c2,), (2, 9 * c1), (c2, 2)),
        "classifier": ((54, 9 * c2), (54,), (2, 9 * c2), (54, 2)),
    }
    expected = {f"{prefix}.{suffix}": shape for prefix, entries in shapes.items()
                for suffix, shape in zip(("weight", "bias", "lora_A", "lora_B"), entries)}
    if supplied != expected or layout.names != tuple(expected):
        raise ValueError("layout is not a grid registration parameter chart")
    ltm, stm, offset = [], [], 0
    for spec in layout.specs:
        if spec.start != offset or spec.stop != offset + math.prod(spec.shape):
            raise ValueError("grid parameter offsets must be contiguous and match shapes")
        target = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        target.extend(range(spec.start, spec.stop))
        offset = spec.stop
    if offset != layout.total_numel:
        raise ValueError("grid parameter count does not match the layout")
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)


def flat_prediction(model: GridRegistrationMemoryCNN, layout: ParameterLayout,
                    theta: Tensor, x: Tensor, *, stm_scale: float = 1.0) -> Tensor:
    """Differentiable physical-pixel means (N,54), without state mutation."""
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")
    return functional_call(model, (layout.unflatten_named(theta), dict(model.named_buffers())),
                           (x,), {"stm_scale": stm_scale}, strict=True)
