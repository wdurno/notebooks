from dataclasses import replace
import json

import pytest

from numerical_experiment.picar.config import LearningConfig, PicarConfig
from numerical_experiment.picar import phase1_fix2


def test_dense_recipe_rejects_old_bounds_and_implicit_scale_changes():
    config = LearningConfig(reward_version="dense-visual-v1", return_min=-6, return_max=12)
    config.validate()
    with pytest.raises(ValueError, match="bounds"):
        replace(config, return_max=11).validate()
    with pytest.raises(ValueError, match="new version"):
        replace(config, visual_reward_budget=4).validate()


def test_retired_feature_requires_protocol_and_primary_keeps_it_off():
    config = replace(LearningConfig(), retired_archive_replay=True,
                     retired_replay_protocol="supplemental-rehearsal-v1")
    config.validate()
    with pytest.raises(ValueError, match="primary"):
        replace(PicarConfig(), mode="production", model=replace(PicarConfig().model, kind="qwen"),
                learning=config).validate()


def test_software_stage_resume_skips_sealed_units_without_gpu(tmp_path, monkeypatch):
    from numerical_experiment.picar import fix2_data
    monkeypatch.setattr(phase1_fix2, "PROJECT", tmp_path)
    monkeypatch.setattr(phase1_fix2, "refresh", lambda: None)
    monkeypatch.setattr(phase1_fix2, "scientific_sources", lambda: {"test": "frozen"})
    calls = []
    def software(store):
        calls.append("software")
        return {"passed": True, "test_result": "bounded fixture passed"}
    monkeypatch.setattr(phase1_fix2, "_software", software)
    monkeypatch.setattr(fix2_data, "make_panels", lambda store, split: {
        "split": split, "panels": [], "count": 0})
    root = tmp_path / "fix2"
    assert phase1_fix2.run(root, stage="software")["status"] == "software_complete"
    completions = {str(p): p.read_bytes() for p in (root / "units").rglob("COMPLETED")}
    assert len(completions) == 4
    assert phase1_fix2.run(root, stage="software", resume=True)["status"] == "software_complete"
    assert calls == ["software"]
    assert completions == {str(p): p.read_bytes() for p in (root / "units").rglob("COMPLETED")}
    assert not (root / "orchestration" / "COMPLETED").exists()
    progress = json.loads((root / "orchestration" / "progress.json").read_text())
    assert progress["counts"]["step"] == 4


def test_timing_replay_masks_retired_prefix_and_never_bridges_a_gap():
    import torch
    from numerical_experiment.picar.fix2_pipeline import eligible_history
    data = {"images": torch.zeros(1, 4, 3, 4, 4), "executed_actions": torch.zeros(1, 4, 4),
        "elapsed": torch.tensor([[[0.], [1.], [1.], [1.]]]), "decision_time": torch.tensor([[0., 1., 2., 3.]]),
        "start_time": torch.tensor([0.]), "contexts": [[]], "episode_deadline": torch.tensor([20.])}
    clipped = eligible_history(data, {0., 2., 3.})
    assert clipped["decision_time"].tolist() == [[2., 3.]]
    assert clipped["start_time"].tolist() == [1.]
    assert eligible_history(data, {0., 3.})["images"].shape[1] == 1
    assert eligible_history(data, {0., 1., 2., 3.})["images"].shape[1] == 4
