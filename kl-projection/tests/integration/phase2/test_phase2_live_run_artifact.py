import os
from pathlib import Path

import pytest

from picar_kl.actions import validate_action_distribution
from picar_kl.data.phase1 import load_phase1_run, read_run_metadata


pytestmark = [pytest.mark.integration, pytest.mark.robot]


def test_manual_phase2_live_run_artifact_has_expected_robot_smoke_shape():
    run_dir_text = os.environ.get("PICAR_PHASE2_RUN_DIR", "").strip()
    if not run_dir_text:
        pytest.skip("Set PICAR_PHASE2_RUN_DIR to a completed phase2-live run directory.")

    run_dir = Path(run_dir_text)
    metadata = read_run_metadata(run_dir)
    records = load_phase1_run(run_dir)

    assert metadata["phase"] == "phase2-live"
    assert metadata["status"] in {"completed", "interrupted"}
    assert int(metadata["steps_completed"]) == len(records)
    assert records

    sources = [record.action.source for record in records if record.action is not None]
    assert "vlm-bootstrap" in sources
    assert "phase2-lstm" in sources

    for record in records:
        assert record.action is not None
        validate_action_distribution(record.action.distribution)
        assert record.action.metadata.get("fit_id") == metadata.get("fit_id")
        assert record.metadata.get("phase") == "phase2-live"

    latency_names = {event.name for record in records for event in record.latency_events}
    assert {"capture_image", "visual_encoding", "reward_scoring", "apply_action"}.issubset(latency_names)
    assert "lstm_action" in latency_names

    if "vlm-operator-override" in sources:
        assert "vlm_operator_override" in latency_names
