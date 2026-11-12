"""Resolved development settings, distinct from every frozen earlier campaign."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_phase2.config import learner_config, simulator_config
from numerical_experiment.picar_phase2_rescue.campaign import scientific_sources as previous_sources


VERSION = "plan19-phase3-development-v1"
MODEL_SEED = 191101
BOOTSTRAP_SEED = 193001
DEVELOPMENT_SEEDS = (193101, 193102)
EASY_SEEDS = tuple(range(193501, 193521))
HARD_SEEDS = EASY_SEEDS[:10]
CONFIRMATION_SEEDS = tuple(range(194101, 194105))
NORMALIZATION = "feature_and_projection_rms"
MAX_HOURS = 6.
PRIOR_STUDY_SECONDS = 20719.62480906


@dataclass(frozen=True)
class Recipe:
    name: str
    temperature: float = 5.
    action_variance: float = 1.
    exploration_std: float = 4.
    learning_rate: float = .001
    optimizer_steps: int = 1
    ridge: float = .1
    pi: float = .025
    fd_strength: float = 0.
    fd_step: float = .01
    transitions: int = 640

    def validate(self):
        if not self.name or type(self.optimizer_steps) is not int or not 1 <= self.optimizer_steps <= 4:
            raise ValueError("named recipes require one to four optimizer steps")
        if type(self.transitions) is not int or self.transitions < 640:
            raise ValueError("do not shorten the already admitted 640-interval ramp")
        for key in ("temperature", "action_variance", "exploration_std", "learning_rate", "fd_step"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.pi not in {.01, .025, .05}:
            raise ValueError("use the bounded common adaptation-weight bracket")
        if not all(math.isfinite(v) and v >= 0 for v in (self.ridge, self.fd_strength)):
            raise ValueError("regularization must be finite and nonnegative")


def configuration(recipe, *, bootstrap=False):
    recipe.validate()
    base = learner_config()
    return replace(base, version=VERSION, seed=BOOTSTRAP_SEED,
        transitions=recipe.transitions, learning=replace(base.learning,
            temperature=recipe.temperature, action_variance=recipe.action_variance,
            exploration_std=4. if bootstrap else recipe.exploration_std,
            learning_rate=.001 if bootstrap else recipe.learning_rate,
            optimizer_steps=1 if bootstrap else recipe.optimizer_steps,
            ridge=.1 if bootstrap else recipe.ridge, pi=.025 if bootstrap else recipe.pi,
            fd_strength=0. if bootstrap else recipe.fd_strength,
            fd_step=recipe.fd_step))


def simulation(recipe, *, bootstrap=False):
    base = simulator_config(bootstrap=bootstrap)
    return replace(base, total_training_intervals=recipe.transitions,
                   curriculum_total_intervals=recipe.transitions)


def panels(recipe):
    base = simulation(recipe)
    return [{"difficulty": label, "seeds": seeds,
             "config": replace(base, curriculum_exposure_override=exposure, domain_override=domain)}
            for label, seeds, exposure, domain in
            (("easy", EASY_SEEDS, 0., "A"), ("hard", HARD_SEEDS, float(recipe.transitions), "B"))]


def sources():
    project = Path(__file__).resolve().parents[2]
    return {**previous_sources(), **{str(p.relative_to(project)): sha256_file(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}}


def likelihood_key(recipe):
    """Only a common likelihood and fitting design may share an initial Fisher."""
    config = configuration(recipe, bootstrap=True).to_dict()
    return canonical_json({"normalization": NORMALIZATION, "model_seed": MODEL_SEED,
        "model": config["model"], "temperature": recipe.temperature,
        "action_variance": recipe.action_variance, "return_bins": config["model"]["return_bins"],
        "bootstrap_fit": {"learning_rate": .001, "ridge": .1, "pi": .025, "steps": 1}})


def recipe_dict(recipe):
    recipe.validate()
    return asdict(recipe)
