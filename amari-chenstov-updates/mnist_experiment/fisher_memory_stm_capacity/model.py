"""Phase 12 variable STM charts on the frozen Phase 11 LTM architecture."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor, nn

from mnist_experiment.fisher_memory_grid_registration import model as grid_model
from mnist_experiment.fisher_memory_grid_registration.model import (
    AFFINE_NAMES, FACTOR_SEED_COMPONENT, GAUSSIAN_SCALE_PIXELS, OUTPUT_DIM,
    flat_prediction,
)
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
CONFIGURATIONS = ("head_r1", "all_r1", "all_r2", "all_r4")
RANKS = dict(zip(CONFIGURATIONS, ((0, 0, 1), (1, 1, 1), (2, 2, 2), (4, 4, 4))))
COUNTS = dict(zip(CONFIGURATIONS, ((3268, 108), (3268, 172), (3268, 344), (3268, 688))))
CHART = ContextVar("stm_capacity_chart", default="all_r2")
EXTRA_FACTOR_COMPONENT = "stm_capacity_extra_A"


def _configuration(name: str) -> str:
    if not isinstance(name, str) or name not in CONFIGURATIONS:
        raise ValueError(f"unsupported STM configuration: {name!r}")
    return name


@contextmanager
def use_chart(name: str):
    """Select construction only; an existing layout always identifies its chart."""
    token = CHART.set(_configuration(name))
    try:
        yield
    finally:
        CHART.reset(token)


def _architecture(output_dim, channels):
    channels = grid_model._architecture(output_dim, channels)
    if channels != (4, 6):
        raise ValueError("STM capacity uses the frozen (4, 6) channel architecture")
    return channels


def _adapt(module, rank: int, prefix: str, factor_seed: int, seeds: dict):
    if not rank:
        base = nn.Conv2d(module.weight.shape[1], module.weight.shape[0], 3,
                         stride=module.stride, padding=module.padding,
                         dilation=module.dilation, groups=module.groups,
                         bias=module.bias is not None,
                         device=module.weight.device, dtype=module.weight.dtype)
        base.weight, base.bias = module.weight, module.bias
        return base, []

    # Draw the entire legacy rank-two chart first so placement never shifts A.
    rows = [row.clone() for row in module.lora_A[:min(rank, 2)]]
    mapping = [dict(row=i, source="legacy_rank_two", source_row=i,
                    seed_component=FACTOR_SEED_COMPONENT) for i in range(len(rows))]
    for row in range(2, rank):
        component = f"{EXTRA_FACTOR_COMPONENT}:{prefix}:row{row}"
        seed = derive_component_seed(factor_seed, component)
        seeds[component] = seed
        values = module.lora_A.new_empty(1, module.lora_A.shape[1])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.kaiming_uniform_(values, a=math.sqrt(5), generator=generator)
        rows.append(values[0])
        mapping.append(dict(row=row, source="independent_extra_row", seed_component=component))
    module.lora_A = nn.Parameter(torch.stack(rows))
    module.lora_B = nn.Parameter(module.lora_B.new_zeros(module.lora_B.shape[0], rank))
    module.rank, module.lora_alpha = rank, float(rank)
    return module, mapping


class GridRegistrationMemoryCNN(grid_model.GridRegistrationMemoryCNN):
    """All base weights/biases stay trainable; only present factors form STM."""

    def __init__(self, *, output_dim: int = OUTPUT_DIM, channels=(4, 6),
                 factor_seed: int, stm_config: str | None = None) -> None:
        channels = _architecture(output_dim, channels)
        name = _configuration(CHART.get() if stm_config is None else stm_config)
        super().__init__(output_dim=output_dim, channels=channels, factor_seed=factor_seed)
        self.stm_config, self.ranks = name, RANKS[name]
        self.factor_seeds = {FACTOR_SEED_COMPONENT: factor_seed}
        self.factor_mapping = dict(schema_version=1, legacy_order=list(AFFINE_NAMES), layers={})
        for prefix, rank in zip(AFFINE_NAMES, self.ranks):
            module, rows = _adapt(self.get_submodule(prefix), rank, prefix,
                                  factor_seed, self.factor_seeds)
            if prefix == "classifier":
                self.classifier = module
            else:
                self.features[int(prefix.rsplit(".", 1)[1])] = module
            self.factor_mapping["layers"][prefix] = dict(rank=rank, rows=rows)
        ltm, stm = memory_indices(ParameterLayout.from_module(self))
        if (ltm.numel(), stm.numel()) != COUNTS[name]:
            raise RuntimeError("STM capacity memory parameter counts do not match")

    def forward(self, inputs: Tensor, stm_scale: float = 1.0) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 84, 84):
            raise ValueError("the grid registration model requires (N, 2, 84, 84) inputs")
        hidden = inputs
        for index, rank in ((0, self.ranks[0]), (3, self.ranks[1])):
            affine = self.features[index]
            hidden = affine(hidden, stm_scale=stm_scale) if rank else affine(hidden)
            hidden = self.features[index + 2](self.features[index + 1](hidden))
        return self.classifier(hidden.flatten(1), stm_scale=stm_scale)


RegistrationMemoryCNN = GridRegistrationMemoryCNN


def build_model(seed: int, *, output_dim: int = OUTPUT_DIM, channels=(4, 6),
                device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                factor_seed: int | None = None, stm_config: str | None = None
                ) -> tuple[GridRegistrationMemoryCNN, ParameterLayout]:
    """Legacy-compatible CPU initialization without advancing global RNGs.

    ``all_r2`` is bitwise identical to Phase 11, including its factor draw.
    ``factor_mapping`` and ``factor_seeds`` on the model record coupled A rows.
    """
    channels = _architecture(output_dim, channels)
    name = _configuration(CHART.get() if stm_config is None else stm_config)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("the STM capacity model supports float32 and float64")
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
                                          factor_seed=factor_seed, stm_config=name)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def _shapes(name):
    expected = {}
    layers = (((4, 2, 3, 3), (4,), 18, 4),
              ((6, 4, 3, 3), (6,), 36, 6),
              ((54, 54), (54,), 54, 54))
    for prefix, rank, (weight, bias, width, height) in zip(AFFINE_NAMES, RANKS[name], layers):
        expected[f"{prefix}.weight"] = weight
        expected[f"{prefix}.bias"] = bias
        if rank:
            expected[f"{prefix}.lora_A"] = (rank, width)
            expected[f"{prefix}.lora_B"] = (height, rank)
    return expected


def memory_indices(layout: ParameterLayout) -> tuple[Tensor, Tensor]:
    """Infer exactly one supported ordered chart; return CPU int64 LTM/STM."""
    supplied = [(spec.name, spec.shape) for spec in layout.specs]
    matches = [name for name in CONFIGURATIONS if supplied == list(_shapes(name).items())]
    if len(matches) != 1:
        raise ValueError("layout is not a supported STM capacity parameter chart/order")
    ltm, stm, offset = [], [], 0
    for spec in layout.specs:
        if spec.start != offset or spec.stop != offset + math.prod(spec.shape):
            raise ValueError("STM capacity parameter offsets must be contiguous and match shapes")
        target = stm if spec.name.endswith((".lora_A", ".lora_B")) else ltm
        target.extend(range(spec.start, spec.stop))
        offset = spec.stop
    if offset != layout.total_numel:
        raise ValueError("STM capacity parameter count does not match the layout")
    return torch.tensor(ltm, dtype=torch.long), torch.tensor(stm, dtype=torch.long)
