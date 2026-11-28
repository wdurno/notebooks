"""Frozen protocol and source identities for the coached/uncoached comparison."""

from dataclasses import replace
from pathlib import Path

from src.seeding import derive_component_seed
from numerical_experiment.picar_head_awac.config import source_hashes
from numerical_experiment.picar_head_awac.learner import Recipe
from numerical_experiment.picar.runtime import sha256_file

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path("cache/numerical_experiment/picar/phase3/coaching-fix6-v4")
SOURCE = Path("cache/numerical_experiment/picar/phase3/head-awac-fix5-v3")
NOTEBOOK = Path("numerical_experiment/picar/plan19_results.ipynb")
CONDITIONS = ("U", "G")
TOTAL = 1024
COLLECTION = 768
MAX_HOURS = 28.


def seed(component):
    return derive_component_seed(196001, "plan19-fix6:" + component)


def recipe():
    return Recipe(capacity=256, batch_size=64, rounds=4, steps=3,
                  learning_rate=.001, temperature=1., action_variance=16.,
                  critic_variance=16., exploration_std=2., target_rate=.05,
                  rank=64, ridge=.1, pi=.025, fd_strength=12.142951965332031,
                  fd_step=.01, integration_draws=4)


def fingerprints():
    result = source_hashes()
    for package in ("picar_coaching", "picar_head_rank"):
        for path in sorted((PROJECT / "numerical_experiment" / package).glob("*.py")):
            if path.name not in {"report.py", "video.py"}:
                result[str(path.relative_to(PROJECT))] = sha256_file(path)
    return result


def selected_recipe(source=SOURCE):
    import json
    paths = list((Path(source) / "units").glob("selection-*/summary.json"))
    if len(paths) != 1:
        raise ValueError("Fix 5 selection is missing or ambiguous")
    selected = json.loads(paths[0].read_text())["selected"]
    actual = Recipe(**selected["recipe"])
    actual = replace(actual, rank=64, fd_strength=selected["fd_strength"], ridge=selected["fd_ridge"])
    if actual != recipe():
        raise ValueError("sealed Fix 5 recipe differs from Fix 6's specified recipe")
    return actual
