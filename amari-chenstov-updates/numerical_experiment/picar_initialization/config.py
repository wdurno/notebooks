"""Initialization-repair protocol, independent of historical budget ceilings."""

from dataclasses import asdict, replace
from pathlib import Path

from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_phase3.config import (
    MODEL_SEED, NORMALIZATION, Recipe, configuration as old_configuration,
    panels, simulation, sources as historical_sources,
)

VERSION = "plan19-phase3-fix1-v1"
MAX_HOURS = 10.
PRIOR_SECONDS = 38590.03602203
DEVELOPMENT_SEEDS = (195101, 195102)
COLLECTION_SEED = 195301
SPLIT_SEED = 195401
FITTING_SEED = 195501
DATA_LADDER = (256, 512, 1024)
CLOSEOUT_SECONDS = 600.
DEFAULT_RECIPE = Recipe("common", temperature=2.8375266913896,
    action_variance=16., exploration_std=4., learning_rate=.001,
    optimizer_steps=1, ridge=.1, pi=.025)


def configuration(recipe=DEFAULT_RECIPE):
    return replace(old_configuration(recipe), version=VERSION, seed=FITTING_SEED)


def protocol():
    return {"version": VERSION, "model_seed": MODEL_SEED,
        "normalization": NORMALIZATION, "data_ladder": list(DATA_LADDER),
        "validation": "whole_episode_hash_modulo_five; fixed first-rung validation panel",
        "split_seed": SPLIT_SEED, "collection_seed": COLLECTION_SEED,
        "data_rung_selection": "minimum fixed-validation joint NLL; earliest tie",
        "data_growth": "attempt at least two rungs if admitted; stop after ready second rung with <1% validation improvement",
        "fitting_seed": FITTING_SEED, "development_seeds": list(DEVELOPMENT_SEEDS),
        "initial_stm": "random_A_zero_B_held_constant",
        "initial_ewc": "equal_weight_model_scores_at_final_anchor; no_pi",
        "initial_replay": "empty; all_training_ids_archived_once",
        "recipe": asdict(DEFAULT_RECIPE), "max_hours": MAX_HOURS,
        "prior_study_seconds": PRIOR_SECONDS, "next_phase_authorized": False}


def sources():
    project = Path(__file__).resolve().parents[2]
    return {**historical_sources(), **{
        str(p.relative_to(project)): sha256_file(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))
        if p.name != "report.py"}}
