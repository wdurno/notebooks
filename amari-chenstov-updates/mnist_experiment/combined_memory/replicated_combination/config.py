"""Frozen task settings, six policies and independently named random seeds."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace

from mnist_experiment.fisher_memory_stm_capacity.config import Config, Recipe
from src.seeding import derive_component_seed
from mnist_experiment.combined_memory.reproduction.config import SEED as REPRODUCTION_SEED
from . import PROTOCOL

REPO = Path(__file__).resolve().parents[3]
FAMILY = REPO / "cache/mnist_experiment/plan18"
HISTORY = FAMILY / "stm_reproduction_v1"
ROOT = FAMILY / "replicated_combination_v1"
DATA = REPO / "cache/mnist_experiment/datasets"
SEED = 18111009
POLICIES = ("R", "A", "B", "C", "D", "E")
CALIBRATED = ("A", "C", "D", "E")
STRENGTH = 3.34056320604949
PROCESS_SCALES = (1e-5, 1e-4, 1e-3)
BOOTSTRAP_SEED = derive_component_seed(SEED, f"{PROTOCOL}:paired-bootstrap")


@dataclass(frozen=True)
class Policy:
    name: str
    ridge: float = 1.
    process_scale: float = 1e-4

    def __post_init__(self):
        if self.name not in POLICIES or not math.isfinite(self.ridge) or not 0 <= self.ridge <= 1000:
            raise ValueError("invalid Phase 9 policy/ridge")
        if self.name in ("R", "B") and self.ridge != 1:
            raise ValueError("representation references have fixed ridge one")
        if self.process_scale not in PROCESS_SCALES:
            raise ValueError("undeclared Kalman process scale")

    @property
    def adaptive(self):
        return self.name in ("D", "E")

    @property
    def transport(self):
        return self.name in ("C", "E")

    @property
    def representation(self):
        return "low_rank_diagonal" if self.name == "R" else "diagonal"

    def recipe(self, cfg):
        return Recipe.from_config(cfg, method="replay_ewc" if self.name == "A" else "fd_replay_ewc",
                                  ridge=self.ridge, strength=0. if self.name == "A" else STRENGTH)

    def mapping(self):
        return dict(policy=self.name, ridge=self.ridge, process_scale=self.process_scale,
                    representation=self.representation, transport=self.transport, adaptive=self.adaptive)


def configuration(history, namespace="fresh", device="cuda", smoke=False):
    if namespace not in ("fresh", "development", "acceptance"):
        raise ValueError("unknown seed namespace")
    seed = (REPRODUCTION_SEED if namespace == "acceptance" else
            derive_component_seed(SEED, f"{PROTOCOL}:{namespace}"))
    if smoke:
        cfg = Config.smoke_config(device)
        return replace(cfg, seed=seed, novel_count=3, init_epochs=1, init_epoch_stages=(1,),
                       max_iter=4, max_eval=6, reference_capacity=2, capacities=(2,),
                       reference_replay=1, replay_counts=(1,))
    return replace(Config.from_mapping(history), seed=seed, device=device)


def observer_config(cfg):
    return SimpleNamespace(observer_floor_absolute=1e-12, observer_floor_relative=1e-6,
                           initialization_size=cfg.init_size, cold_start=1 if cfg.smoke else 16,
                           noise_half_life=32., pi_min=1e-4, pi_max=.2)
