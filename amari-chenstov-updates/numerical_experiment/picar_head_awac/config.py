"""Fix 5 settings; replay and target clocks count acquisitions explicitly."""

from dataclasses import replace
from pathlib import Path
from numerical_experiment.picar_reward_repair.config import configuration, simulation, BASE_RECIPE
from src.robotics.drift import smoothstep5

DEFAULT_OUTPUT = Path("cache/numerical_experiment/picar/phase3/head-awac-fix5-v3")
DEFAULT_SOURCE = Path("cache/numerical_experiment/picar/phase3/reward-fix3-v1")
NOTEBOOK = Path("numerical_experiment/picar/plan19_results.ipynb")
METHODS = ("mc_replay", "td_replay", "td_fd")
RANKS = (0, 3, 5, 10)
INITIAL_SEED = 199001
MAIN_SEEDS = (199101, 199102)


def simulator(exposure=0., total=1024, fixed=None):
    progress = 0. if fixed == "initial_A" else 1. if fixed == "moderate_A" else smoothstep5(
        (exposure / total - .125) / .75)
    sigma = 1. * (.5 ** progress)
    return replace(simulation(BASE_RECIPE), domain_override="A", staged_curriculum=False,
        curriculum_total_intervals=max(320, total), total_training_intervals=max(320, total),
        barrier_sigma_easy=sigma, barrier_sigma_hard=sigma, curriculum_exposure_override=0.)


def source_hashes():
    from numerical_experiment.picar.runtime import sha256_file, source_fingerprint
    project = Path(__file__).resolve().parents[2]
    result = source_fingerprint()
    for root in ("picar_head_awac", "picar_reward_repair", "picar_phase2_rescue"):
        for path in sorted((project/"numerical_experiment"/root).glob("*.py")):
            if path.name != "report.py":
                result[str(path.relative_to(project))] = sha256_file(path)
    for name in ("src/seeding.py","src/representations.py",
                 "numerical_experiment/picar_solver_repair/numerics.py"):
        result[name]=sha256_file(project/name)
    return result
