"""Focused .025 follow-up ledger and artifact-only inference gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError
from mnist_experiment.rotated_mnist.plan11 import followup, followup_analysis
from mnist_experiment.rotated_mnist.plan11.config import Study
from mnist_experiment.rotated_mnist.plan11.run import ASSET_REQUIRED, TRAJECTORY_REQUIRED


REPO = Path(__file__).parents[3]
CONFIG = REPO / followup.DEFAULT_CONFIG


def _complete(store: followup.UnitStore, unit: dict, required: tuple[str, ...], nll: float = 1.0) -> None:
    session = store.begin(unit, required, resume=False)
    assert session is not None
    for name in required:
        if name == "summary.json" and unit["schedule"] is None:
            value = {"stream_identity_paired": True}
        elif name == "summary.json":
            value = {
                "environment_nll_auc": nll,
                "environment_accuracy_auc": 0.7 if unit["policy"]["gain"] == 0 else 0.75,
                "schedule_kind": unit["schedule"],
                "total_wall_time_seconds": 1.0,
                "unsupported_scale_fallback_fraction": 0.0,
                **unit["policy"],
            }
        elif name == "metrics.json":
            value = [
                {
                    "step": step,
                    "schedule_kind": unit["schedule"],
                    "observations_before_evaluation": 4 * step,
                    "angle_degrees": float(step),
                    "current_nll": nll,
                    "current_environment_accuracy": 0.7 if unit["policy"]["gain"] == 0 else 0.75,
                }
                for step in range(121)
            ]
        elif name == "checks.json":
            value = {
                "all_decisions_predictable": True,
                "all_finite": True,
                "every_action_matches_blend": True,
                "shared_initial_model": True,
                "maximum_q_recursion_error": 0.0,
            }
        else:
            value = {}
        session.write_json(name, value)
    store.finish(session, required)


def test_followup_freezes_only_own_anchor_and_rejects_changed_contract(tmp_path: Path) -> None:
    store = followup.make_store(tmp_path, Study.from_path(CONFIG), REPO)
    units = followup.freeze(store, resume=False)
    assert len(units) == 352 * 2 * 2
    assert {unit["schedule"] for unit in units} == {"linear", "sigmoid"}
    assert {unit["policy"]["anchor"] for unit in units} == {0.025}
    assert {unit["policy"]["gain"] for unit in units} == {0.0, 0.025}
    assert units == followup.freeze(store, resume=True)
    with pytest.raises(RotatedArtifactError, match="pass --resume"):
        followup.freeze(store, resume=False)
    (tmp_path / "contract.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RotatedArtifactError, match="differs"):
        followup.freeze(store, resume=True)


def test_progress_never_tests_an_incomplete_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(followup, "REPLICAS", 2)
    monkeypatch.setattr(followup_analysis, "REPLICAS", 2)
    store = followup.make_store(tmp_path, Study.from_path(CONFIG), REPO)
    units = followup.freeze(store, resume=False)
    for index in (1, 2):
        _complete(store, store.unit(followup.PHASE, index, None, None), ASSET_REQUIRED)
    for unit in units:
        if unit["schedule"] == "linear" or unit["replica_index"] == 1:
            nll = 1.0 if unit["policy"]["gain"] == 0 else 0.9 - 0.05 * unit["replica_index"]
            _complete(store, unit, TRAJECTORY_REQUIRED, nll)

    progress = followup_analysis.load_progress(REPO, output_root=tmp_path)
    assert progress["schedules"]["linear"]["completed_pairs"] == 2
    assert progress["schedules"]["linear"]["inference"] is not None
    assert progress["schedules"]["linear"]["mean_trajectory"]["fixed_nll_auc"][-1] == pytest.approx(1.0)
    assert progress["schedules"]["linear"]["mean_trajectory"]["blend_accuracy_auc"][-1] == pytest.approx(0.75)
    assert progress["schedules"]["sigmoid"]["completed_pairs"] == 1
    assert progress["schedules"]["sigmoid"]["inference"] is None
    assert progress["schedules"]["sigmoid"]["mean_trajectory"] is not None
    assert progress["schedules"]["sigmoid"]["incomplete_replica_indices"] == [2]

    for unit in units:
        if unit["schedule"] == "sigmoid" and unit["replica_index"] == 2:
            _complete(store, unit, TRAJECTORY_REQUIRED, 1.0 if unit["policy"]["gain"] == 0 else 0.8)
    complete = followup_analysis.load_progress(REPO, output_root=tmp_path)
    assert complete["schedules"]["sigmoid"]["inference"] is not None

    unit = units[0]
    (store.paths(unit)[0] / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RotatedArtifactError, match="corrupt artifacts"):
        followup_analysis.load_progress(REPO, output_root=tmp_path)
