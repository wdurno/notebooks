"""Frozen protocol and resource allocation for the solver repair."""

from dataclasses import asdict, replace
from pathlib import Path

from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_initialization.config import (
    DEFAULT_RECIPE, MODEL_SEED, NORMALIZATION, configuration as initial_configuration,
    panels, simulation, sources as initial_sources,
)

VERSION = "plan19-phase3-fix2-v3"
SUPERSEDED_FIX2_SECONDS = 1500.
MAX_HOURS = (9.*3600-SUPERSEDED_FIX2_SECONDS)/3600
PRIOR_SECONDS = 17.911502521 * 3600+SUPERSEDED_FIX2_SECONDS
CLOSEOUT_SECONDS = 300.
ACCEPTANCE_SECONDS = 300.
SCREEN_SECONDS = 1800.
DEVELOPMENT_SEEDS = (196101, 196102)
SCREEN_SEED = 196201
SOLVERS = ("proximal_1", "lagged_1", "proximal_3")
FD_STRENGTH = 1742.42640280057
BASE_RECIPE = replace(DEFAULT_RECIPE, learning_rate=.003)
DEFAULT_SOURCE = Path("cache/numerical_experiment/picar/phase3/initialization-fix1-v1")
DEFAULT_OUTPUT = Path("cache/numerical_experiment/picar/phase3/solver-fix2-v3")
SOURCE_PILOT = "pilot-68cb4a663a6c"


def configuration(recipe=BASE_RECIPE):
    return replace(initial_configuration(recipe), version=VERSION)


def solver_steps(name):
    if name not in SOLVERS:
        raise ValueError("unknown solver recipe")
    return 3 if name == "proximal_3" else 1


def sources():
    project = Path(__file__).resolve().parents[2]
    return {**initial_sources(), **{str(p.relative_to(project)): sha256_file(p)
        for p in sorted(Path(__file__).parent.glob("*.py")) if p.name != "report.py"}}


def protocol():
    return {"version": VERSION, "solvers": list(SOLVERS), "initialization": "sealed_fix1_primary_four_draw",
        "recurrent_center": "advance_once_per_accepted_event", "fisher_admission": "retirement_only",
        "metric": "bias_corrected_live_gradient_second_moment", "beta2": .999,
        "optimizer_epsilon": 1e-8, "momentum": False, "ridge": "implicit_all_banks",
        "base_recipe": asdict(BASE_RECIPE), "fd_strength": FD_STRENGTH,
        "screen_seed": SCREEN_SEED, "screen_events_per_segment": 64,
        "screen_segments": ["early", "late"], "screen_initialization": "same_pre_stream_asset_for_each_segment",
        "screen_seconds": SCREEN_SECONDS, "development_seeds": list(DEVELOPMENT_SEEDS),
        "max_hours": MAX_HOURS, "prior_seconds": PRIOR_SECONDS,
        "superseded_fix2_seconds": SUPERSEDED_FIX2_SECONDS,
        "correction": "retain_full_frozen_validation_context; v1_diagnostic_excluded_from_selection",
        "supervisor_correction": "classify_nonfinite_model_output_as_candidate_failure; one_common_alpha_reduction_if_needed",
        "phase4_authorized": False, "physical_motion": False}
