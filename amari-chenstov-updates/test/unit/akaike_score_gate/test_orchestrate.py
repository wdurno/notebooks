import json

import pytest

from mnist_experiment.akaike_score_gate.artifacts import UnitStore
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.orchestrate import Runner, clean_fixed_policy


def runner(root):
    study = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")
    value = Runner(UnitStore(root, study), root / "report.ipynb")
    value.refresh = lambda **kwargs: None
    return value


def decision_with_parent(root):
    value = runner(root)
    unit = value.store.unit("phase3", "test_parent", 0)
    session = value.store.begin(unit, ("summary.json",), resume=True)
    session.write_json("summary.json", {"nll": 1.})
    parent = value.store.finish(session, ("summary.json",))
    decision = value.decision("phase3", "calibration", {"selected": {"c": 1}}, [parent])
    return decision, parent


def test_resume_requires_sealed_decision_and_parents(tmp_path):
    decision, parent = decision_with_parent(tmp_path)
    assert runner(tmp_path).load_decision("phase3", "calibration") == decision
    (parent / "summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt"):
        runner(tmp_path).load_decision("phase3", "calibration")


def test_resume_rejects_changed_decision_pointer(tmp_path):
    decision, _ = decision_with_parent(tmp_path)
    decision["selected"]["c"] = 4
    (tmp_path / "decisions/calibration.json").write_text(json.dumps(decision))
    with pytest.raises(RuntimeError, match="sealed unit"):
        runner(tmp_path).load_decision("phase3", "calibration")


def test_missing_decision_is_pending(tmp_path):
    assert runner(tmp_path).load_decision("phase3", "calibration") is None


def test_fixed_control_discards_only_irrelevant_gate_settings():
    policy = {"gate": None, "c": 4, "cadence": 4, "half_life": 16, "depth": 6}
    assert clean_fixed_policy(policy) == {**policy, "c": 1., "cadence": 1}
    assert policy["c"] == 4
