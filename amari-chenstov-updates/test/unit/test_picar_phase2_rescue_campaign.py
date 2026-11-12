import json
from pathlib import Path

import pytest

from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_phase2_rescue.campaign import StageProgressStore, baseline_units, seal_baselines


def source_fixture(root):
    config = {"schema_version": 1, "purpose": "test-only-baseline-sealing"}
    for seed in (192101, 192102):
        path = root / "units" / f"pilot-{seed}"
        with RunStore(path, config, checkpoint_policy="journal-v1") as store:
            store.checkpoint({"stage": "done"})
            store.complete({"settings": {"name": "baseline"}, "seed": seed, "unit_path": str(path)})
    with RunStore(root / "orchestration", config, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"charged_seconds": 1234., "units": 8})


def test_requires_two_complete_baselines_without_selecting_outcomes(tmp_path):
    with pytest.raises(RuntimeError, match="both original baseline"):
        seal_baselines(tmp_path)
    assert not (tmp_path / "STOP").exists()


def test_seals_incomplete_parent_and_preserves_all_completed_children(tmp_path):
    source_fixture(tmp_path)
    markers = {str(p): sha256_file(p) for p in tmp_path.glob("units/*/COMPLETED")}
    result = seal_baselines(tmp_path)
    assert result["status"] == "superseded_after_baselines"
    assert result["total_active_seconds"] == 1234
    assert result["original_rescues_executed"] == 0
    assert len(result["baseline_paths"]) == 2
    assert all(sha256_file(Path(p)) == value for p, value in markers.items())
    assert (tmp_path / "orchestration/COMPLETED").exists()
    marker = sha256_file(tmp_path / "orchestration/COMPLETED")
    assert seal_baselines(tmp_path) == result
    assert sha256_file(tmp_path / "orchestration/COMPLETED") == marker


def test_refuses_third_rescue_after_a_previous_one_already_started(tmp_path):
    source_fixture(tmp_path)
    path = tmp_path / "units/no-ridge"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"settings": {"name": "no_ridge"}}))
    with pytest.raises(RuntimeError, match="original rescue already started"):
        seal_baselines(tmp_path)
    assert not (tmp_path / "orchestration/COMPLETED").exists()


@pytest.mark.parametrize("stage,key", [("distill", "kl_step"), ("initial_nll", "nll_step")])
def test_initialization_progress_uses_optimizer_steps_not_old_acquisitions(stage, key):
    from types import SimpleNamespace
    calls = []
    bridge = StageProgressStore(SimpleNamespace(checkpoint=lambda state, **kw: calls.append(kw)))
    bridge.checkpoint({"stage": stage, key: 10, "eligible": list(range(247))}, progress={"phase": 2})
    assert calls[0]["progress"] == {"phase": 2, "counts": {"step": 10, "target": 247}}
