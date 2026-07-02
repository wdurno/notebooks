import json

import numpy as np

from picar_kl.data.phase1 import load_phase1_run, load_record_image
from picar_kl.phase1.run import FixedActionVLMController, Phase1RunConfig, run_phase1


class FakeImage:
    def __init__(self, value):
        self.image_rgb = np.full((4, 5, 3), value, dtype=np.uint8)


class FakeRobot:
    def __init__(self):
        self.applied_vectors = []
        self.captures = 0

    def capture_image(self):
        self.captures += 1
        return FakeImage(self.captures)

    def apply_vector(self, action_vector):
        self.applied_vectors.append(dict(action_vector))
        return {"status": "ok", "index": len(self.applied_vectors)}


class FakeSpeechSource:
    def drain_texts(self):
        return ["please find the red ball"]


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


def test_run_phase1_writes_records_and_applies_vector(tmp_path):
    robot = FakeRobot()
    speaker = FakeSpeaker()

    summary = run_phase1(
        Phase1RunConfig(
            data_root=tmp_path,
            run_uuid="run-a",
            max_steps=2,
            task_prompt="find the red ball",
        ),
        robot=robot,
        vlm=FixedActionVLMController("drive-forward", generated_text="Driving forward."),
        speech_source=FakeSpeechSource(),
        speaker=speaker,
    )

    assert summary.steps_completed == 2
    assert summary.status == "completed"
    assert len(robot.applied_vectors) == 2
    assert robot.applied_vectors[0] == {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0}
    assert speaker.spoken == ["Driving forward.", "Driving forward."]

    metadata = json.loads((summary.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["uuid"] == "run-a"
    assert metadata["status"] == "completed"
    assert metadata["steps_completed"] == 2
    assert "ended_at" in metadata
    records = load_phase1_run(summary.run_dir)
    assert len(records) == 2
    assert records[0].action is not None
    assert records[0].action.action_name == "drive-forward"
    assert records[0].action.metadata["action_receipt"] == {"status": "ok", "index": 1}
    assert [event.name for event in records[0].latency_events] == [
        "capture_image",
        "reward_scoring",
        "context_render",
        "vlm_decision",
        "apply_action",
        "speaker",
    ]
    assert records[0].reward == 0.0
    assert records[0].metadata["reward_result"]["clipped_reward"] == 0.0
    assert records[0].metadata["context"]["history_window"] == 180
    assert load_record_image(records[0]).shape == (4, 5, 3)


class InterruptingRobot(FakeRobot):
    def capture_image(self):
        if self.captures >= 1:
            raise KeyboardInterrupt
        return super().capture_image()


def test_run_phase1_marks_interrupted_runs(tmp_path):
    robot = InterruptingRobot()

    summary = run_phase1(
        Phase1RunConfig(
            data_root=tmp_path,
            run_uuid="run-interrupted",
            max_steps=None,
            task_prompt="find the red ball",
        ),
        robot=robot,
        vlm=FixedActionVLMController("look-forward", generated_text="Looking forward."),
    )

    assert summary.status == "interrupted"
    assert summary.steps_completed == 1
    metadata = json.loads((summary.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "interrupted"
    assert metadata["steps_completed"] == 1
    assert "ended_at" in metadata
    assert len(load_phase1_run(summary.run_dir)) == 1
