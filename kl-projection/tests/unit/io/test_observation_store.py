import json

import numpy as np

from picar_kl.records import ActionRecord, Phase1ObservationRecord
from picar_kl.io.observation_store import Phase1ObservationStore


def test_phase1_observation_store_writes_jsonl_and_image(tmp_path):
    store = Phase1ObservationStore(tmp_path / "run-a")
    store.write_run_metadata({"uuid": "run-a", "phase": "phase1"})
    action = ActionRecord.from_action_name("look-forward")
    record = Phase1ObservationRecord(
        run_uuid="run-a",
        run_dir=store.run_dir,
        timestamp="2026-01-01T00:00:00+00:00",
        source="step",
        step_index=1,
        image_path=None,
        messages=[],
        user_texts=[],
        action=action,
    )

    stored = store.append(record, image_rgb=np.ones((3, 4, 3), dtype=np.uint8))

    assert stored.image_path is not None
    assert stored.image_file is not None
    assert stored.image_file.exists()
    assert json.loads(store.run_meta_path.read_text(encoding="utf-8"))["uuid"] == "run-a"
    lines = store.observations_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["image_path"] == "images/step_000001.npz"
    assert np.load(stored.image_file)["image"].shape == (3, 4, 3)
