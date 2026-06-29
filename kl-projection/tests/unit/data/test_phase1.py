import json

import numpy as np

from picar_kl.actions import action_name_to_distribution
from picar_kl.data.phase1 import iter_phase1_run_dirs, load_phase1_run, load_record_image


def test_iter_phase1_run_dirs_discovers_nested_runs(tmp_path):
    run_a = tmp_path / "legacy_robot_demo" / "run-a"
    run_b = tmp_path / "legacy_robot_demo" / "phase1" / "run-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "observations.jsonl").write_text("", encoding="utf-8")
    (run_b / "observations.jsonl").write_text("", encoding="utf-8")

    assert set(iter_phase1_run_dirs(tmp_path)) == {run_a, run_b}


def test_load_phase1_run_converts_legacy_actions_and_images(tmp_path):
    run_dir = tmp_path / "run-a"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True)
    image = np.full((2, 3, 3), 17, dtype=np.uint8)
    np.savez_compressed(images_dir / "step_000001.npz", image=image)
    (run_dir / "run_meta.json").write_text(json.dumps({"uuid": "run-a"}), encoding="utf-8")
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source": "reset",
            "step_index": 0,
            "image_path": "images/step_000000.npz",
            "messages": [],
            "user_texts": [],
            "action": None,
            "reward": None,
            "done": False,
        },
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "source": "step",
            "step_index": 1,
            "image_path": "images/step_000001.npz",
            "messages": [],
            "user_texts": [],
            "action": {
                "agentic_action_name": "drive-forward",
                "executed_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
                "generated_text": "Driving forward.",
            },
            "reward": 1.0,
            "done": False,
        },
    ]
    (run_dir / "observations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    records = load_phase1_run(run_dir)

    assert records[0].action is None
    assert records[1].run_uuid == "run-a"
    assert records[1].action is not None
    assert records[1].action.distribution == action_name_to_distribution("drive-forward")
    assert np.array_equal(load_record_image(records[1]), image)
