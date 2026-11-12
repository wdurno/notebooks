"""Resolved Phase 2 development recipe; no models or hardware are imported."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.phase1_fix2 import scientific_sources
from numerical_experiment.picar.runtime import sha256_file
from src.robotics.simulator import SimulatorConfig
from src.robotics.actions import Calibration


VERSION = "plan19-phase2-v1"
EASY_SEEDS = tuple(range(192501, 192521))
HARD_SEEDS = EASY_SEEDS[:10]
PILOT_SEEDS = (192101, 192102)


@dataclass(frozen=True)
class Recipe:
    name: str
    transitions: int = 640
    ridge: float = .1
    learning_rate: float = .001
    exploration_std: float = 4.
    teacher_floor: float = .001
    bootstrap_transitions: int = 600

    def validate(self):
        if self.transitions < 320 or self.bootstrap_transitions > 1200:
            raise ValueError("preserve a long curriculum ramp and bounded bootstrap")
        if self.ridge < 0 or self.learning_rate <= 0 or self.exploration_std <= 0:
            raise ValueError("invalid common learner recipe")
        if not 0 < self.teacher_floor < 1/8:
            raise ValueError("invalid pre-execution teacher probability floor")


BASE = Recipe("baseline")
RESCUES = (Recipe("no_ridge", ridge=0.),
           Recipe("softer_bootstrap", ridge=0., teacher_floor=.01))


def learner_config(recipe=BASE):
    recipe.validate()
    base = PicarConfig()
    return replace(base, version=VERSION, mode="production", task_version="ball-contact-v1",
        image_width=160, image_height=120, transitions=recipe.transitions,
        bootstrap_transitions=recipe.bootstrap_transitions,
        model=replace(base.model, kind="qwen", hidden_size=64, context_length=4,
                      return_bins=161, top_layers=1, ltm_rank=4, stm_rank=1),
        learning=replace(base.learning, replay_batch=1, fd_strength=0.,
            reward_version="dense-visual-centering-v1", return_min=-6., return_max=12.,
            ridge=recipe.ridge, learning_rate=recipe.learning_rate,
            exploration_std=recipe.exploration_std, teacher_floor=recipe.teacher_floor),
        runtime=replace(base.runtime, device="cuda", threads=2))


def simulator_config(recipe=BASE, *, bootstrap=False):
    return SimulatorConfig(width=160, height=120, realtime=True,
        control_interval_s=2. if bootstrap else 1., task_version="ball-contact-v1",
        layout_version="curriculum-v1", total_training_intervals=recipe.transitions,
        curriculum_total_intervals=recipe.transitions,
        curriculum_exposure_override=0. if bootstrap else None,
        domain_override="A" if bootstrap else None)


def evaluation_panels(recipe=BASE):
    base = simulator_config(recipe)
    return [{"difficulty": name, "seeds": seeds,
             "config": replace(base, curriculum_exposure_override=exposure, domain_override=domain)}
            for name, seeds, exposure, domain in
            (("easy", EASY_SEEDS, 0., "A"), ("hard", HARD_SEEDS, float(recipe.transitions), "B"))]


def source_hashes():
    own = Path(__file__).parent
    project = own.parents[1]
    return {**scientific_sources(),
            **{str(p.relative_to(project)): sha256_file(p) for p in sorted(own.rglob("*.py"))},
            "numerical_experiment/picar_evaluation_video.py": sha256_file(project / "numerical_experiment/picar_evaluation_video.py")}


def floor_controls(floor):
    """Pre-execution floor-and-renormalize effects for hard demonstration commands."""
    if not 0 < floor < 1/8:
        raise ValueError("invalid teacher floor")
    rows = []
    for action in range(8):
        q = tuple((1. if i == action else floor)/(1+7*floor) for i in range(8))
        rows.append({"action": action, "probabilities": q,
                     "controls": asdict(Calibration().map_distribution(q))})
    return rows


def execution_recipe():
    return {"schema_version": 1, "version": VERSION, "phase": 2,
        "baseline": asdict(BASE), "pilot_seeds": PILOT_SEEDS,
        "easy_panel_seeds": EASY_SEEDS, "hard_panel_seeds": HARD_SEEDS,
        "maximum_rescue_attempts": 2, "rescue_recipes": [asdict(x) for x in RESCUES],
        "rescue_trigger": "Neither baseline pilot improves hard success by ten percentage points",
        "rescue_2_initialization": "New teacher collection and one-time KL at floor .01; never resmooth old actions",
        "initialization_seed": 191101, "max_active_hours": 5., "overall_study_hours": 20.,
        "pre_execution_validation_reserve_seconds": 300.,
        "teacher_floor_control_audit": {str(f): floor_controls(f) for f in (.001, .01)},
        "teacher_readiness": "20 easy-A layouts, separate clear/blocked counts; frozen teacher",
        "initial_asset_reuse": "Only completed compatible bootstrap with disjoint Fisher/replay ledger",
        "readiness_guides": {"teacher_easy": .7, "student_easy": .6, "student_teacher_gap": .15},
        "guides_are_not_significance_tests": True, "fd_comparison": False,
        "stop_after": "Phase 2 readiness and fresh-run forecast check-in"}
