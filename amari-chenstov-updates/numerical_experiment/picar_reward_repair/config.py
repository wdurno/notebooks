"""Fixed scientific settings; historical configurations remain unchanged."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from numerical_experiment.picar.config import LearningConfig
from numerical_experiment.picar_initialization.config import (
    DEFAULT_RECIPE, MODEL_SEED, NORMALIZATION, configuration as initial_configuration,
    panels, simulation,
)
from . import VERSION

CONDITIONS = ("width6_linear", "width6_signed_square")
DEVELOPMENT_SEEDS = (197101, 197102)
FITTING_SEED = 197501
MAX_HOURS = 12.
CLOSEOUT_SECONDS = 600.
ACCEPTANCE_SECONDS = 300.
DEFAULT_SOURCE = Path("cache/numerical_experiment/picar/phase3/initialization-fix1-v1")
DEFAULT_OUTPUT = Path("cache/numerical_experiment/picar/phase3/reward-fix3-v1")
DEFAULT_REFERENCES = Path("cache/numerical_experiment/picar/prompt-probes/refined-width-evaluation-v1-20261116")
BASE_RECIPE = replace(DEFAULT_RECIPE, optimizer_steps=3, learning_rate=.001, fd_strength=0.)


def power(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown reward condition")
    return CONDITIONS.index(condition) + 1


def reward_version(condition):
    return "six-example-width-" + str(power(condition)) + "-v1"


@dataclass(frozen=True)
class RewardLearningConfig(LearningConfig):
    def validate(self):
        if self.reward_version not in {reward_version(c) for c in CONDITIONS}:
            raise ValueError("reward repair requires a frozen six-example reward version")
        # The historical checker owns all unchanged optimizer/replay settings.
        original = LearningConfig(**{**asdict(self), "reward_version": "dense-visual-centering-v1"})
        original.validate()


def configuration(condition=CONDITIONS[0], *, offline_fit=False):
    base = initial_configuration(BASE_RECIPE)
    learning = RewardLearningConfig(**{**asdict(base.learning), "reward_version": reward_version(condition),
        "optimizer_steps": 1 if offline_fit else 3})
    return replace(base, version=VERSION, seed=FITTING_SEED, learning=learning)


def sources():
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar_solver_repair.config import sources as historical_sources

    project = Path(__file__).resolve().parents[2]
    result = historical_sources()
    result.update({str(p.relative_to(project)): sha256_file(p)
                   for p in Path(__file__).parent.glob("*.py") if p.name != "report.py"})
    for name in ("picar_ball_width_probe.py", "picar_reference_reward_probe.py"):
        p = project / "numerical_experiment" / name
        result[str(p.relative_to(project))] = sha256_file(p)
    return result


def protocol():
    return {"version": VERSION, "conditions": list(CONDITIONS), "max_hours": MAX_HOURS,
        "development_seeds": list(DEVELOPMENT_SEEDS), "model_seed": MODEL_SEED,
        "fit_seed": FITTING_SEED, "initial_train": 515, "initial_validation": 65,
        "initial_stm": "neutral_random_A_zero_B", "initial_fisher": "four_draw_equal_history_weight_no_pi",
        "initial_replay": "empty", "return_coding": {CONDITIONS[0]: 1., CONDITIONS[1]: 10.},
        "return_support": [-6., 12.], "return_bins": 161,
        "temperature_coded": BASE_RECIPE.temperature, "action_variance": 16.,
        "evaluator_order": list("ABCFED"), "base_recipe": asdict(BASE_RECIPE),
        "live_solver": "proximal_3", "fd_strength": 0., "normalization": NORMALIZATION,
        "next_phase_authorized": False, "physical_motion": False}
