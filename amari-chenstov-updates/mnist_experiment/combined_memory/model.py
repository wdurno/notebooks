"""Plan 15 head-rank-one chart on Plan 18's cell-aligned 120px mosaic.

LTM retains its original order (3,268 coordinates); the random-A/zero-B head
adds 108 STM coordinates with LoRA scaling gamma = alpha/rank = 1.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from mnist_experiment.fisher_memory_stm_capacity import model as capacity_model
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
INPUT_SHAPE = (2, 120, 120)
OUTPUT_DIM = 54
LTM_COUNT, STM_COUNT = 3268, 108
PARAMETER_COUNT = LTM_COUNT + STM_COUNT
GAUSSIAN_SCALE_PIXELS = 2.
FACTOR_SEED_COMPONENT = capacity_model.FACTOR_SEED_COMPONENT


class CombinedMemoryCNN(capacity_model.GridRegistrationMemoryCNN):
    """All base weights train; only the dense head has a rank-one adapter."""

    def __init__(self, *, factor_seed: int) -> None:
        super().__init__(factor_seed=factor_seed, stm_config="head_r1")
        self.features[5] = nn.AvgPool2d(20)

    def forward(self, inputs: Tensor, stm_scale: float = 1.) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SHAPE:
            raise ValueError("combined-memory inputs must have shape (N,2,120,120)")
        hidden = self.features(inputs).flatten(1)
        if hasattr(self.classifier, "lora_A"):
            return self.classifier(hidden, stm_scale=stm_scale)
        return self.classifier(hidden)


def build_model(seed: int, device: str | torch.device = "cpu",
                dtype: torch.dtype = torch.float32, factor_seed: int | None = None
                ) -> tuple[CombinedMemoryCNN, ParameterLayout]:
    """Preserve the inherited initialization without advancing global RNGs."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("combined memory supports float32 and float64")
    if factor_seed is None:
        factor_seed = derive_component_seed(seed, FACTOR_SEED_COMPONENT)
    if type(factor_seed) is not int or not 0 <= factor_seed < 2**63:
        raise ValueError("factor_seed must be an integer in [0, 2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CombinedMemoryCNN(factor_seed=factor_seed)
    model = model.to(device=device, dtype=dtype)
    layout = ParameterLayout.from_module(model)
    ltm, stm = memory_indices(layout)
    if (ltm.numel(), stm.numel(), layout.total_numel) != (LTM_COUNT, STM_COUNT, PARAMETER_COUNT):
        raise RuntimeError("combined-memory parameter counts do not match Plan 18")
    return model, layout


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Validate the fixed unmerged or merged chart; return ordered CPU indices."""
    shapes = {
        "features.0.weight": (4, 2, 3, 3), "features.0.bias": (4,),
        "features.3.weight": (6, 4, 3, 3), "features.3.bias": (6,),
        "classifier.weight": (54, 54), "classifier.bias": (54,),
    }
    if any(spec.name.endswith((".lora_A", ".lora_B")) for spec in layout.specs):
        shapes.update({"classifier.lora_A": (1, 54), "classifier.lora_B": (54, 1)})
    if [(s.name, s.shape) for s in layout.specs] != list(shapes.items()):
        raise ValueError("layout is not the combined-memory parameter chart/order")
    ltm, stm, offset = [], [], 0
    for spec in layout.specs:
        if spec.start != offset or spec.stop != offset + math.prod(spec.shape):
            raise ValueError("combined-memory offsets must be contiguous and match shapes")
        target = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        target.extend(range(spec.start, spec.stop))
        offset = spec.stop
    if offset != layout.total_numel:
        raise ValueError("combined-memory parameter count does not match layout")
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)


def flat_prediction(model: nn.Module, layout: ParameterLayout, theta: Tensor,
                    x: Tensor, stm_scale: float = 1.) -> Tensor:
    """Functional physical-pixel means (N,54); replace every LTM/STM value."""
    if theta.dtype != x.dtype or theta.device != x.device:
        raise ValueError("inputs and theta must share dtype and device")
    return functional_call(model, (layout.unflatten_named(theta), dict(model.named_buffers())),
                           (x,), {"stm_scale": stm_scale}, strict=True)


def gaussian_nll(pred: Tensor, y: Tensor) -> Tensor:
    """Per-grid N(f,4 I_54) NLL; its gradient g is the negative score."""
    if pred.ndim != 2 or pred.shape[-1] != OUTPUT_DIM or y.shape != pred.shape:
        raise ValueError("predictions and targets must have matching shape (N,54)")
    if pred.dtype != y.dtype or pred.device != y.device or not pred.is_floating_point():
        raise ValueError("predictions and targets must share floating dtype and device")
    return (pred - y).square().sum(-1) / 8 + 27 * math.log(8 * math.pi)


def merge_parameters(model: CombinedMemoryCNN, layout: ParameterLayout, theta: Tensor
                     ) -> tuple[CombinedMemoryCNN, ParameterLayout, Tensor]:
    """Create a separate observer chart, never modifying the training model.

    The merged flat order is exactly the unmerged LTM order. ``ltm_mapping``
    explicitly records both sets of names/offsets for archive metric use.
    """
    ltm, stm = memory_indices(layout)
    if stm.numel() != STM_COUNT:
        raise ValueError("merging requires the unmerged head-rank-one chart")
    if theta.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(theta).all()):
        raise ValueError("merge requires finite floating parameters")
    values = layout.unflatten_named(theta.detach())
    merged = copy.deepcopy(model).to(device=theta.device, dtype=theta.dtype)
    # Meta construction avoids a random initialization that would perturb RNGs.
    head = nn.Linear(54, 54, device="meta", dtype=theta.dtype)
    gamma = model.classifier.lora_alpha / model.classifier.rank
    head.weight = nn.Parameter((values["classifier.weight"] + gamma *
                                (values["classifier.lora_B"] @ values["classifier.lora_A"])).clone())
    head.bias = nn.Parameter(values["classifier.bias"].clone())
    merged.classifier = head
    merged.stm_config = "merged"
    merged_layout = ParameterLayout.from_module(merged)
    flat = torch.cat([values[s.name].reshape(-1) for s in layout.specs
                      if not s.name.endswith((".lora_A", ".lora_B"))]).clone()
    for spec in merged_layout.specs:
        if spec.name == "classifier.weight":
            flat[spec.start:spec.stop] = head.weight.detach().reshape(-1)
    merged_layout.copy_vector_to_module(merged, flat)
    source = {spec.name: spec for spec in layout.specs}
    merged.ltm_mapping = [dict(name=spec.name, shape=list(spec.shape),
                               source_start=source[spec.name].start, source_stop=source[spec.name].stop,
                               merged_start=spec.start, merged_stop=spec.stop)
                          for spec in merged_layout.specs]
    if merged_layout.total_numel != ltm.numel():
        raise RuntimeError("merged coordinates must match the LTM order")
    return merged, merged_layout, flat.detach()
