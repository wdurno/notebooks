"""Paired Plan 17 assets; calibration reuses already-seen initialization labels."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor

from mnist_experiment.diagonal_lfu import data as frozen
from .config import Config


DATA_SCHEMA_VERSION = frozen.DATA_SCHEMA_VERSION
PARTITION_PROTOCOL = frozen.PARTITION_PROTOCOL
TRAIN_ROLES = frozen.TRAIN_ROLES
EVAL_ROLES = frozen.EVAL_ROLES
load_data = frozen.load_data
make_schedule = frozen.make_schedule


def make_assets(cfg: Config, replica: int, cohort: str, data: dict[str, Tensor]) -> dict[str, Any]:
    """Resolve length variation before sampling, commonly for every paired arm.

    Plan 17 protocol-prefixed seeds give independent streams and study-wide
    partitions while preserving Plan 16's global roles and dev/fresh test split.
    Calibration is a subset of init_x/y, not validation/reference/evaluation.
    """
    cfg.validate()
    if cohort not in ("development", "main", "fresh", "stationary"):
        raise ValueError("cohort must be development, main, fresh, or stationary")
    schedule_seed = cfg.seed_for(cohort, replica, "schedule_variation")
    schedule_generator = torch.Generator(device="cpu").manual_seed(schedule_seed)
    choice = int(torch.randint(len(cfg.schedule_factors), (), generator=schedule_generator))
    factor = 1. if cohort == "development" else float(cfg.schedule_factors[choice])
    resolved = dataclasses.replace(cfg, hold_steps=math.ceil(cfg.hold_steps * factor),
                                   ramp_steps=math.ceil(cfg.ramp_steps * factor))
    assets = frozen.make_assets(resolved, replica, cohort, data)
    assets["non9_weights"] = assets["non9_probabilities"]
    calibration_seed = cfg.seed_for(cohort, replica, "noise_calibration")
    generator = torch.Generator(device="cpu").manual_seed(calibration_seed)
    positions = torch.randperm(len(assets["init_y"]), generator=generator)[:cfg.calibration_size]
    assets["calibration_indices"] = positions
    for suffix in ("ids", "x", "y"):
        assets[f"calibration_{suffix}"] = assets[f"init_{suffix}"][positions].clone()
    assets["seeds"].update(schedule_variation=schedule_seed, noise_calibration=calibration_seed)
    schedule = dict(factor=factor, seed=schedule_seed, choices=list(cfg.schedule_factors),
                    base_hold_steps=cfg.hold_steps, base_ramp_steps=cfg.ramp_steps,
                    hold_steps=resolved.hold_steps, ramp_steps=resolved.ramp_steps,
                    num_p_steps=resolved.num_p_steps, rounding="ceil", resolved_before_learning=True)
    assets["schedule_factor"] = factor
    assets["metadata"].update(
        schedule_factor=factor, schedule_variation=schedule,
        resolved_hold_steps=resolved.hold_steps, resolved_ramp_steps=resolved.ramp_steps,
        resolved_num_p_steps=resolved.num_p_steps,
        calibration=dict(source="initialization-training-subset", size=len(positions),
                         seed=calibration_seed, additional_labels=0,
                         initialization_indices=positions.tolist(),
                         ids=assets["calibration_ids"].tolist()),
    )
    return assets
