"""Fast checks for Plan 11 action and immutable resume semantics."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.controller import ControllerState, DiscountedMovementState
from src.representations import DiagonalFisher

from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError, RotatedCompletedRunError
from mnist_experiment.rotated_mnist.plan11.artifacts import UnitStore
from mnist_experiment.rotated_mnist.plan11.config import Policy, Study
from mnist_experiment.rotated_mnist.plan11.policy import blend_action, decide
from mnist_experiment.rotated_mnist.plan11 import analysis


REPO = Path(__file__).parents[3]
SMOKE = REPO / "mnist_experiment/rotated_mnist/plan11/configs/smoke.json"


@pytest.mark.parametrize("anchor", (0.01, 0.025, 0.05))
def test_blend_endpoints_and_cold_start(anchor: float) -> None:
    assert blend_action(anchor, 0.0, 0.8, cold=False)[1] == anchor
    assert blend_action(anchor, 1.0, 0.8, cold=False)[1] == 0.8
    assert blend_action(anchor, 0.05, 0.8, cold=True)[1] == anchor
    assert anchor < blend_action(anchor, 0.05, 0.8, cold=False)[1] < 0.8


def test_post_cold_decision_uses_current_q_and_movement() -> None:
    controller = dataclasses.replace(
        ControllerState.initialize(2, 100),
        accepted_steps=8,
        trend=torch.ones(2, dtype=torch.float64),
        residual_moment=1.0,
        scale_moment=1.0,
    )
    fisher = DiagonalFisher(torch.ones(2, dtype=torch.float64))
    movement = DiscountedMovementState()
    fixed, _, recommendation = decide(controller, movement, fisher, Policy(0.01, 0.0), batch_size=4)
    blend, _, second = decide(controller, movement, fisher, Policy(0.01, 0.05), batch_size=4)
    unshrunk, _, third = decide(controller, movement, fisher, Policy(0.01, 1.0), batch_size=4)
    assert recommendation == second == third
    assert fixed.applied_pi == 0.01
    assert fixed.applied_pi < blend.applied_pi < unshrunk.applied_pi
    assert blend.applied_pi == pytest.approx(0.01 + 0.05 * (recommendation - 0.01))


def test_interrupted_unit_reuses_config_and_preserves_completed(tmp_path: Path) -> None:
    study = Study.from_path(SMOKE)
    store = UnitStore(tmp_path, study, REPO)
    unit = store.unit("smoke", 1, "linear", Policy(0.01, 0.0).mapping())
    required = ("metrics.json", "summary.json")
    first = store.begin(unit, required, resume=False)
    assert first is not None
    first.write_json("metrics.json", {"partial": True})
    resumed = store.begin(unit, required, resume=True)
    assert resumed is not None
    resumed.write_json("metrics.json", {"complete": True})
    resumed.write_json("summary.json", {"value": 1})
    completed = store.finish(resumed, required)
    assert store.begin(unit, required, resume=True) is None
    assert (completed / "metrics.json").read_text() == '{\n  "complete": true\n}\n'
    with pytest.raises(RotatedCompletedRunError):
        store.begin(unit, required, resume=False)
    (completed / "summary.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(RotatedArtifactError, match="corrupt artifacts"):
        store.begin(unit, required, resume=True)


def test_linear_shortlist_prefers_smallest_near_best_gain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = Study.from_path(SMOKE)
    store = SimpleNamespace(root=tmp_path, study=study, sources={"test": "frozen"})

    def mean(_store: object, schedule: str, policy: Policy, field: str) -> float:
        assert schedule == "linear"
        if field == "environment_accuracy_auc":
            return 0.8
        if field == "unsupported_scale_fallback_fraction":
            return 0.0
        assert field == "environment_nll_auc"
        if policy.gain == 0:
            return 1.0 + policy.anchor
        return 0.9 + policy.anchor + abs(policy.gain - 0.05) * 0.1

    monkeypatch.setattr(analysis, "_mean", mean)
    selected = analysis.linear_selection(store)
    assert selected["finalists"] == [
        {"anchor": 0.01, "gain": 0.025},
        {"anchor": 0.025, "gain": 0.025},
    ]
    assert len(selected["grid"]) == 18
    assert len(analysis.sigmoid_policies(selected)) == 7


def test_precision_gate_uses_paired_replica_variance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = Study.from_path(SMOKE)
    store = SimpleNamespace(root=tmp_path, study=study)
    selection = {
        "finalists": [
            {"anchor": 0.01, "gain": 0.05},
            {"anchor": 0.025, "gain": 0.05},
        ],
        "fixed_linear_nll": {"0.01": 0.95, "0.025": 0.96, "0.05": 0.97},
    }
    monkeypatch.setattr(analysis, "linear_selection", lambda _store: selection)

    def mean(_store: object, schedule: str, policy: Policy, field: str) -> float:
        if field == "environment_accuracy_auc":
            return 0.8
        assert field == "environment_nll_auc"
        if policy.gain == 0:
            return 0.95 + policy.anchor
        return 0.90 + policy.anchor

    def summary(_store: object, index: int, schedule: str, policy: Policy) -> dict[str, float]:
        return {"environment_nll_auc": mean(_store, schedule, policy, "environment_nll_auc") + 0.001 * index * policy.gain}

    monkeypatch.setattr(analysis, "_mean", mean)
    monkeypatch.setattr(analysis, "_summary", summary)
    result = analysis.development_decision(store, 58.0)
    assert result["status"] == "candidate_ready_for_cost_review"
    assert result["selected_policy"] == {"anchor": 0.01, "gain": 0.05}
    assert result["fixed_comparator"] == 0.01
    assert result["n_required"] == 32
    assert result["confirmation_authorized"] is False
