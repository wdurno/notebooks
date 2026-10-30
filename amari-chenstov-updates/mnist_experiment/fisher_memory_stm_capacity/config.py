"""Versioned Phase 12 settings; historical recipes remain fixed anchors."""

from dataclasses import dataclass, replace
import json
import math

from mnist_experiment.plan15_grid_subspace import Config as SubspaceConfig
from mnist_experiment.fisher_memory_grid_registration import PROTOCOL as GRID_PROTOCOL
from mnist_experiment.fisher_memory_regression.config import Recipe, content_hash, ROOT
from src.seeding import derive_component_seed

from . import PROTOCOL

CHARTS = ("head_r1", "all_r1", "all_r2", "all_r4")
NEW_CHARTS = tuple(c for c in CHARTS if c != "all_r2")
FD, REPLAY = "fd_replay_ewc", "replay_ewc"
REFERENCE = "all_r2--" + REPLAY
INCUMBENT = "all_r2--" + FD


@dataclass(frozen=True)
class Config(SubspaceConfig):
    protocol: str = PROTOCOL
    seed: int = 2026102912
    init_epochs: int = 45
    blur_max: float = 2.4
    confirmation_replicas: int = 8
    refinement_settings: int = 4
    ridge_grid: tuple = (1., math.sqrt(10), 10.)
    charts: tuple = CHARTS

    def validate(self):
        SubspaceConfig.validate(replace(self, protocol=GRID_PROTOCOL))
        if self.protocol != PROTOCOL or self.charts != CHARTS:
            raise ValueError("incompatible Phase 12 charts/protocol")
        if type(self.refinement_settings) is not int or not 0 <= self.refinement_settings <= 4:
            raise ValueError("at most four optional Phase 12 proposals")
        if not self.smoke and (self.init_epochs != 45 or self.blur_max != 2.4
                or self.confirmation_replicas not in (8, 10)
                or self.ridge_grid != (1., math.sqrt(10), 10.)
                or (self.fd_step,self.probes,self.max_iter,self.max_eval) != (.01,2,100,150)):
            raise ValueError("fixed Phase 11 task and Phase 12 budgets required")

    def seed_for(self, family, replica, component):
        stage = family.split("--", 1)[0]
        return derive_component_seed(self.seed, f"{PROTOCOL}:{stage}:{replica}:{component}")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        for key in ("charts", "channels", "output_dims", "init_epoch_stages", "capacities",
                    "replay_counts", "ridge_grid", "strength_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        cfg = cls(**values)
        cfg.validate()
        return cfg


def frozen_recipe(cfg, method):
    return Recipe.from_config(cfg, method, ridge=1. if method == FD else math.sqrt(10),
        strength=1.976099143570005 if method == FD else 0.)


def arm(chart, method):
    if chart not in CHARTS or method not in (FD, REPLAY):
        raise ValueError("invalid STM chart/method")
    return chart + "--" + method

