"""Balanced, paired Phase 5 horizons with globally isolated data roles."""

from __future__ import annotations

import dataclasses
import math

import torch

from mnist_experiment.diagonal_lfu.data import make_assets as make_base_assets


def make_assets(cfg, replica, cohort, data):
    cfg.validate()
    if cohort not in ("development", "fresh"):
        raise ValueError("Phase 5 cohorts are development or fresh")
    factor = cfg.schedule_factors[replica % len(cfg.schedule_factors)]
    resolved = dataclasses.replace(cfg, hold_steps=math.ceil(cfg.hold_steps * factor),
                                   ramp_steps=math.ceil(cfg.ramp_steps * factor))
    assets = make_base_assets(resolved, replica, cohort, data)
    assets["non9_weights"] = assets["non9_probabilities"]
    seed = cfg.seed_for(cohort, replica, "noise_calibration")
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(len(assets["init_y"]), generator=generator)[:cfg.calibration_size]
    for suffix in ("ids", "x", "y"):
        assets["calibration_" + suffix] = assets["init_" + suffix][positions].clone()
    assets["calibration_indices"] = positions
    assets["schedule_factor"] = factor
    assets["seeds"]["noise_calibration"] = seed
    assets["metadata"].update(
        schedule_factor=factor,
        schedule_variation={"factor": factor, "choices": list(cfg.schedule_factors),
            "assignment": "replica-index-modulo-balanced-factors-before-learning",
            "resolved_before_learning": True, "base_hold_steps": cfg.hold_steps,
            "base_ramp_steps": cfg.ramp_steps, "hold_steps": resolved.hold_steps,
            "ramp_steps": resolved.ramp_steps, "num_p_steps": resolved.num_p_steps},
        calibration={"source": "initialization-training-subset", "size": len(positions),
            "seed": seed, "additional_labels": 0, "initialization_indices": positions.tolist(),
            "ids": assets["calibration_ids"].tolist()})
    return assets
