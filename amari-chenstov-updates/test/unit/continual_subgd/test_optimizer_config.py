from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mnist_experiment.continual_subgd.config import Plan13Study
from mnist_experiment.continual_subgd.artifacts import UnitStore, freeze_json
from mnist_experiment.continual_subgd.optimizer import fixed_budget_update
from mnist_experiment.rotated_mnist.artifacts import (
    RotatedArtifactError,
    RotatedCompletedRunError,
    RotatedIncompleteRunError,
)
from src.parameters import ParameterLayout
from src.representations import DenseFisher


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = REPO_ROOT / "mnist_experiment/continual_subgd/configs/smoke.json"


def test_plan13_smoke_config_is_deterministic() -> None:
    first = Plan13Study.from_path(SMOKE_CONFIG)
    second = Plan13Study.from_path(SMOKE_CONFIG)
    assert first.config_hash == second.config_hash
    assert first.seed("component", 1) == second.seed("component", 1)
    assert first.seed("component", 1) != first.seed("component", 2)
    assert first.burn_in_candidates == (1,)
    assert first.rank_candidates == (1,)


def test_fixed_budget_optimizer_decreases_objective_and_obeys_projection() -> None:
    torch.manual_seed(3)
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.5]], dtype=torch.float64)
    targets = torch.tensor([0, 1, 1])
    fisher = DenseFisher(torch.eye(layout.total_numel, dtype=torch.float64))
    before = layout.flatten_module(model, detach=True)
    mask = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    result = fixed_budget_update(
        model,
        layout,
        inputs,
        targets,
        fisher,
        lambda gradient: mask * gradient,
        strength=1.0,
        kappa=0.1,
        inner_steps=4,
        learning_rate=0.2,
        max_backtracks=8,
    )
    after = layout.flatten_module(model, detach=True)
    assert result.objective_after <= result.objective_before
    assert result.accepted_steps > 0
    assert after[0] != before[0]
    assert torch.equal(after[1:], before[1:])


def test_plan13_store_is_immutable_and_resumable(tmp_path: Path) -> None:
    study = Plan13Study.from_path(SMOKE_CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)
    required = ("value.json",)
    unit = store.unit("test", "fixture", 1)
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
    assert store.begin(incomplete_unit, required, resume=True) is not None

    ledger = tmp_path / "ledger.json"
    freeze_json(ledger, {"frozen": True}, resume=False)
    freeze_json(ledger, {"frozen": True}, resume=True)
    with pytest.raises(RotatedArtifactError):
        freeze_json(ledger, {"frozen": False}, resume=True)
