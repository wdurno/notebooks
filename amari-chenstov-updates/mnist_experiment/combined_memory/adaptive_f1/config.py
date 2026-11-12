"""Two policies; preserve the exact Phase 9 data and initialization seeds."""

from dataclasses import dataclass
import math

from mnist_experiment.combined_memory.replicated_combination.config import (
    REPO, FAMILY, DATA, HISTORY, ROOT as PREVIOUS, STRENGTH, configuration,
    observer_config,
)
from mnist_experiment.fisher_memory_stm_capacity.config import Recipe
from src.seeding import derive_component_seed
from . import PROTOCOL

ROOT = FAMILY / "adaptive_f1_v1"
POLICIES = ("R", "K")
PROCESS_SCALE = 1e-5
BOOTSTRAP_SEED = derive_component_seed(18111109, PROTOCOL + ":paired-bootstrap")


@dataclass(frozen=True)
class Policy:
    name: str
    ridge: float = 1.
    process_scale: float = PROCESS_SCALE

    def __post_init__(self):
        if (self.name not in POLICIES or not math.isfinite(self.ridge)
                or not 0 <= self.ridge <= 1000 or self.process_scale != PROCESS_SCALE
                or (self.name == "R" and self.ridge != 1.)):
            raise ValueError("invalid fixed R/K policy")

    @property
    def adaptive(self):
        return self.name == "K"

    @property
    def representation(self):
        return "low_rank_diagonal"

    def recipe(self, cfg):
        return Recipe.from_config(cfg, method="fd_replay_ewc", ridge=self.ridge,
                                  strength=STRENGTH)

    def mapping(self):
        return dict(policy=self.name, ridge=self.ridge, process_scale=self.process_scale,
                    representation=self.representation, transport=False,
                    adaptive=self.adaptive, strength=STRENGTH)
