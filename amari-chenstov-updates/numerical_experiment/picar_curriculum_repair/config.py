"""One versioned scientific contract, independent of historical run ceilings."""

from dataclasses import asdict, replace
from pathlib import Path

from numerical_experiment.picar.runtime import sha256_file, source_fingerprint
from numerical_experiment.picar_reward_repair.config import (
    BASE_RECIPE, MODEL_SEED, NORMALIZATION, configuration as reward_configuration,
    DEFAULT_REFERENCES,
)
from numerical_experiment.picar_phase3.config import simulation as old_simulation
from . import VERSION

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("cache/numerical_experiment/picar/phase3/reward-fix3-v1")
DEFAULT_OUTPUT = Path("cache/numerical_experiment/picar/phase3/curriculum-fix4-v1")
MAX_HOURS = 28.
CLOSEOUT_SECONDS = 600.
PRELAUNCH_ACCEPTANCE_SECONDS = 300.
BOUNDARIES = (0, 128, 768, 1024, 1664, 2048)
CHECKPOINTS = (0, 448, 768, 1024, 1664, 2048)
METHODS = ("replay_ewc", "stm_fd")
ENVIRONMENTS = ("A", "B_initial", "B_final")
MAIN_SEEDS = (198101, 198102)
PILOT_SEED = 198201
CALIBRATION_SEED = 198301
REFERENCE_SEED = 198401
EVALUATION_SEEDS = tuple(range(198501, 198513))


def configuration(*, strength=0., ridge=.1):
    base = reward_configuration()
    return replace(base, version=VERSION, transitions=BOUNDARIES[-1],
        learning=replace(base.learning, fd_strength=strength, ridge=ridge))


def simulation(*, diagnostic=False):
    return replace(old_simulation(BASE_RECIPE), total_training_intervals=2048,
        curriculum_total_intervals=2048, staged_curriculum=True)


def fixed_environment(name):
    if name not in ENVIRONMENTS:
        raise ValueError("unknown fixed evaluation environment")
    return replace(simulation(), curriculum_exposure_override=0. if name == "A" else 768.,
        domain_override="B" if name == "B_final" else "A")


def panels(*, final=False):
    return [{"difficulty": name, "seeds": list(EVALUATION_SEEDS if final else EVALUATION_SEEDS[:8]),
        "config": fixed_environment(name)} for name in ENVIRONMENTS]


def stage_at(n):
    names = ("A hold", "Blocking ramp", "B initial hold", "Steering/texture ramp", "B final hold")
    for i, (start, end) in enumerate(zip(BOUNDARIES, BOUNDARIES[1:])):
        if n < end:
            return {"name": names[i], "start": start, "end": end, "within": max(0, n-start)}
    return {"name": "B final complete", "start": 1664, "end": 2048, "within": n-1664}


def scientific_sources():
    roots = ("picar_curriculum_repair", "picar_reward_repair", "picar_solver_repair",
             "picar_initialization", "picar_phase3", "picar_phase2", "picar_phase2_rescue")
    result = source_fingerprint()
    result["src/seeding.py"] = sha256_file(PROJECT/"src/seeding.py")
    for name in ("picar_ball_width_probe.py", "picar_reference_reward_probe.py"):
        key = "numerical_experiment/"+name
        result[key] = sha256_file(PROJECT/key)
    for root in roots:
        for p in sorted((PROJECT/"numerical_experiment"/root).glob("*.py")):
            if "report" not in p.name:
                result[str(p.relative_to(PROJECT))] = sha256_file(p)
    return {name: digest for name, digest in result.items() if "report" not in Path(name).name}


def protocol():
    return {"version": VERSION, "max_hours": MAX_HOURS,
        "prelaunch_acceptance_charge_seconds": PRELAUNCH_ACCEPTANCE_SECONDS,
        "boundaries": list(BOUNDARIES),
        "checkpoints": list(CHECKPOINTS), "main_seeds": list(MAIN_SEEDS), "methods": list(METHODS),
        "pilot_seed": PILOT_SEED, "calibration_seed": CALIBRATION_SEED,
        "reference_seed": REFERENCE_SEED, "evaluation_seeds": list(EVALUATION_SEEDS),
        "model_seed": MODEL_SEED, "normalization": NORMALIZATION,
        "reward": "six_example_linear", "learning": asdict(configuration().learning),
        "initialization": "reuse_sealed_fix3_linear_equal_weight_four_draw_fisher",
        "reference_targets": "initial_student_fixed_exploration_complete_episode_returns",
        "reference_target_histories": 32, "reference_min_episodes": 8, "reference_interval_cap": 256,
        "fixed_reference_weighting": "equal_record_mean; episode_counts_reported",
        "current_policy_actions": "deterministic_action_equals_mean; action_nll_is_normalizer_not_imitation_accuracy",
        "next_phase_authorized": False, "physical_motion": False}
