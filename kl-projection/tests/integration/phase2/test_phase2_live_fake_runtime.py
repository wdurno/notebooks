from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picar_kl.actions import action_count, validate_action_distribution
from picar_kl.data.phase1 import load_phase1_run, read_run_metadata
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.live import Phase2LiveRunConfig, run_phase2_live
from picar_kl.phase2.runtime import LoadedPhase2Policy, Phase2FitArtifact
from picar_kl.reward import RewardResult
from picar_kl.vlm.control import VLMDecision


torch = pytest.importorskip("torch")

pytestmark = pytest.mark.integration


class FakeRobot:
    def __init__(self):
        self.applied = []

    def capture_image(self):
        return np.full((12, 16, 3), len(self.applied), dtype=np.uint8)

    def apply_vector(self, action_vector):
        self.applied.append(dict(action_vector))
        return {"ok": True, "applied_index": len(self.applied) - 1}


class FakeVisualEncoder:
    encoder_id = "fake-phase2-integration-encoder"

    def __init__(self):
        self.calls = 0

    def encode_image(self, image_rgb):
        self.calls += 1
        value = float(np.asarray(image_rgb).mean() + self.calls)
        return VisualTokenEncoding(
            np.full((3, 4), value, dtype=np.float32),
            metadata={"encoder_id": self.encoder_id, "call": self.calls},
        )


class FakeVLM:
    def __init__(self):
        self.calls = 0
        self.message_lengths = []

    def decide(self, *, image_rgb, messages):
        del image_rgb
        self.calls += 1
        self.message_lengths.append(len(messages))
        action = "drive-backward" if self.calls == 3 else "look-forward"
        return VLMDecision.from_action_name(
            action,
            generated_text=f"fake response {self.calls}",
            raw_response='{"action": "' + action + '"}',
            metadata={"call": self.calls},
        )


class FakeSpeaker:
    def __init__(self):
        self.texts = []

    def speak(self, text):
        self.texts.append(str(text))


class StepSpeechSource:
    def __init__(self, by_step):
        self.by_step = {int(key): list(value) for key, value in by_step.items()}
        self.step = 0

    def drain_texts(self):
        texts = self.by_step.get(self.step, [])
        self.step += 1
        return list(texts)


class FakeRewardScorer:
    def score(self, image_rgb):
        del image_rgb
        return RewardResult(
            prompt_id="fake-reward",
            raw_text="fake reward",
            reward=0.5,
            clipped_reward=0.5,
            metadata={"source": "fake"},
        )


class FakePolicyModel:
    def __call__(
        self,
        *,
        prefix_visual_tokens,
        prefix_actions,
        prefix_step_mask,
        target_visual_tokens,
        target_previous_actions,
        target_step_mask,
    ):
        del prefix_visual_tokens, prefix_actions, prefix_step_mask, target_previous_actions
        batch_size, target_steps = target_step_mask.shape
        probabilities = torch.zeros((batch_size, target_steps, action_count()), dtype=torch.float32)
        probabilities[..., 2] = 0.7
        probabilities[..., 4] = 0.299999
        probabilities = probabilities.to(target_visual_tokens.device)
        return SimpleNamespace(policy=SimpleNamespace(probabilities=probabilities))


def test_phase2_live_fake_runtime_exercises_bootstrap_lstm_override_and_artifacts(tmp_path):
    loaded_policy = _fake_loaded_policy(tmp_path)
    robot = FakeRobot()
    encoder = FakeVisualEncoder()
    vlm = FakeVLM()
    speaker = FakeSpeaker()

    summary = run_phase2_live(
        Phase2LiveRunConfig(
            data_root=tmp_path / "phase2-live",
            run_uuid="fake-phase2-live",
            max_steps=5,
            device="cpu",
        ),
        robot=robot,
        vlm=vlm,
        visual_encoder=encoder,
        loaded_policy=loaded_policy,
        speech_source=StepSpeechSource({2: ["please back up"]}),
        speaker=speaker,
        reward_scorer=FakeRewardScorer(),
    )

    assert summary.status == "completed"
    assert summary.steps_completed == 5
    assert len(robot.applied) == 5
    assert encoder.calls == 5
    assert vlm.calls == 4
    assert speaker.texts

    metadata = read_run_metadata(summary.run_dir)
    assert metadata["phase"] == "phase2-live"
    assert metadata["status"] == "completed"
    assert metadata["fit_id"] == "fake-integration-fit"
    assert metadata["context_steps"] == 2
    assert metadata["prediction_steps"] == 2

    records = load_phase1_run(summary.run_dir)
    assert [record.action.source for record in records] == [
        "vlm-bootstrap",
        "vlm-bootstrap",
        "vlm-operator-override",
        "phase2-lstm",
        "phase2-lstm",
    ]
    assert records[2].user_texts == ["please back up"]
    assert records[2].metadata["mode"] == "operator-override"

    for record in records:
        assert record.image_file is not None
        assert record.image_file.exists()
        assert record.action is not None
        validate_action_distribution(record.action.distribution)
        assert record.action.metadata["fit_id"] == "fake-integration-fit"
        assert record.action.metadata["visual_encoding"]["shape"] == [3, 4]

    lstm_records = [record for record in records if record.action.source == "phase2-lstm"]
    assert lstm_records
    assert all(record.action.metadata["distribution_normalization"]["normalized"] for record in lstm_records)
    assert all(record.action.action_name == "drive-forward" for record in lstm_records)

    latency_names = {event.name for record in records for event in record.latency_events}
    assert {
        "capture_image",
        "visual_encoding",
        "reward_scoring",
        "context_render",
        "vlm_bootstrap_decision",
        "vlm_operator_override",
        "vlm_head_refresh",
        "lstm_action",
        "apply_action",
        "speaker",
    }.issubset(latency_names)


def _fake_loaded_policy(tmp_path: Path) -> LoadedPhase2Policy:
    artifact_dir = tmp_path / "models" / "fake-integration-fit"
    artifact_dir.mkdir(parents=True)
    checkpoint_path = artifact_dir / "policy.pt"
    checkpoint_path.write_bytes(b"fake policy checkpoint")
    artifact = Phase2FitArtifact(
        fit_id="fake-integration-fit",
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        summary={
            "dataset": {
                "context_steps": 2,
                "prediction_steps": 2,
            }
        },
    )
    return LoadedPhase2Policy(
        artifact=artifact,
        model=FakePolicyModel(),
        device="cpu",
        model_config={},
        training_config={},
    )
