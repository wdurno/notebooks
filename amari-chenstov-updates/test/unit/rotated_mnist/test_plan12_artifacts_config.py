from __future__ import annotations

from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.artifacts import (
    RotatedArtifactError,
    RotatedCompletedRunError,
    RotatedIncompleteRunError,
)
from mnist_experiment.rotated_mnist.plan12.anchors import anchor_steps
from mnist_experiment.rotated_mnist.plan12.artifacts import UnitStore, freeze_json
from mnist_experiment.rotated_mnist.plan12.config import Plan12Study
from mnist_experiment.rotated_mnist.plan12.trajectory import phase2_conditions


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = REPO_ROOT / "mnist_experiment/rotated_mnist/plan12/configs/smoke.json"


def test_plan12_config_and_condition_seeds_are_deterministic() -> None:
    first = Plan12Study.from_path(SMOKE_CONFIG)
    second = Plan12Study.from_path(SMOKE_CONFIG)
    assert first.config_hash == second.config_hash
    assert first.seed("component", 4) == second.seed("component", 4)
    assert first.seed("component", 4) != first.seed("component", 5)
    assert [item.mapping() for item in phase2_conditions(0.01, 0.1)] == [
        item.mapping() for item in phase2_conditions(0.01, 0.1)
    ]
    assert anchor_steps(120, 4) == (0, 20, 60, 100)
    assert anchor_steps(6, 1) == (3,)


def test_unit_store_completion_collision_and_resume(tmp_path: Path) -> None:
    study = Plan12Study.from_path(SMOKE_CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)
    unit = store.unit("test", "fixture", 1)
    required = ("value.json",)
    session = store.begin(unit, required, resume=False)
    assert session is not None
    session.write_json("value.json", {"answer": 42})
    completed = store.finish(session, required)
    assert store.completed(unit, required) == completed
    assert store.begin(unit, required, resume=True) is None
    with pytest.raises(RotatedCompletedRunError):
        store.begin(unit, required, resume=False)

    incomplete_unit = store.unit("test", "fixture", 2)
    incomplete = store.begin(incomplete_unit, required, resume=False)
    assert incomplete is not None
    with pytest.raises(RotatedIncompleteRunError):
        store.begin(incomplete_unit, required, resume=False)
    resumed = store.begin(incomplete_unit, required, resume=True)
    assert resumed is not None
    assert resumed.working_path == incomplete.working_path


def test_unit_store_detects_artifact_mutation_and_freezes_ledgers(tmp_path: Path) -> None:
    study = Plan12Study.from_path(SMOKE_CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)
    unit = store.unit("test", "fixture", 3)
    required = ("value.json",)
    session = store.begin(unit, required, resume=False)
    assert session is not None
    session.write_json("value.json", {"answer": 42})
    completed = store.finish(session, required)
    (completed / "value.json").write_text('{"answer": 41}\n', encoding="utf-8")
    with pytest.raises(RotatedArtifactError):
        store.completed(unit, required)

    ledger = tmp_path / "ledger.json"
    freeze_json(ledger, {"frozen": True}, resume=False)
    freeze_json(ledger, {"frozen": True}, resume=True)
    with pytest.raises(RotatedArtifactError):
        freeze_json(ledger, {"frozen": False}, resume=True)
