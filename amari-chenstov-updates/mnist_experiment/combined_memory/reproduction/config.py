"""Frozen policies and separately named reproduction/confirmation seeds."""

from dataclasses import replace
import math
from pathlib import Path

from mnist_experiment.fisher_memory_stm_capacity.config import Config, Recipe
from src.seeding import derive_component_seed

REPO = Path(__file__).resolve().parents[3]
FAMILY = REPO / "cache/mnist_experiment/plan18"
HISTORY = REPO / "cache/mnist_experiment/plan15/stm_capacity_records_v2"
ROOT = FAMILY / "stm_reproduction_v1"
DATA = REPO / "cache/mnist_experiment/datasets"
SEED = 18110908
BOOTSTRAP_SEED = derive_component_seed(SEED, "plan18-stm-reproduction-v1:paired_bootstrap")
INTERIM_SEED = derive_component_seed(SEED, "plan18-stm-reproduction-v1:interim_bootstrap")
POLICIES = {
    "R2": ("all_r2", dict(method="replay_ewc", ridge=math.sqrt(10), strength=0.)),
    "R1": ("head_r1", dict(method="replay_ewc", ridge=math.sqrt(10), strength=0.)),
    "F1": ("head_r1", dict(method="fd_replay_ewc", ridge=1., strength=3.34056320604949)),
}
TOLERANCES = dict(rtol=1e-5, atol=1e-6, prediction_atol=1e-4,
                  risk_atol=1e-4, operator_rtol=1e-5)


def recipe(cfg, policy):
    return Recipe.from_config(cfg, **POLICIES[policy][1])


def configuration(history, *, fresh=False, smoke=False, device="cuda"):
    if smoke:
        cfg = Config.smoke_config(device)
        return replace(cfg, seed=SEED if fresh else 2026102912, novel_count=3,
                       reference_capacity=2, capacities=(2,), reference_replay=1,
                       replay_counts=(1,), max_iter=4, max_eval=6, init_epochs=1,
                       init_epoch_stages=(1,), bootstrap_samples=100)
    return replace(Config.from_mapping(history), seed=SEED if fresh else history["seed"],
                   device=device)
