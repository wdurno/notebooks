import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from picar_kl.actions import action_name_to_distribution
from picar_kl.data.phase1 import load_phase1_run
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.live import Phase2LiveRunConfig, run_phase2_live
from picar_kl.phase2.runtime import load_phase2_policy
from picar_kl.phase2.train import FIT_MODE_WINDOW_SAMPLING, Phase2TrainingConfig, run_phase2_training
from picar_kl.records import Phase1ObservationRecord
from picar_kl.vlm.control import VLMDecision


class FakeRobot:
    def __init__(self):
        self.applied = []

    def capture_image(self):
        value = len(self.applied)
        return np.full((4, 5, 3), value, dtype=np.uint8)

    def apply_vector(self, action_vector):
        self.applied.append(dict(action_vector))
        return {"ok": True, "index": len(self.applied) - 1}


class FakeEncoder:
    encoder_id = "fake-live-encoder"

    def __init__(self):
        self.calls = 0

    def encode_image(self, image_rgb):
        del image_rgb
        self.calls += 1
        return VisualTokenEncoding(
            np.full((3, 4), self.calls, dtype=np.float32),
            metadata={"fake_call": self.calls},
        )


class FakeVLM:
    def __init__(self):
        self.calls = 0

    def decide(self, *, image_rgb, messages):
        del image_rgb, messages
        self.calls += 1
        action = "look-left" if self.calls % 2 else "drive-forward"
        return VLMDecision.from_action_name(
            action,
            generated_text=f"fake speech {self.calls}",
            raw_response=f'{{"action": "{action}"}}',
            metadata={"fake_call": self.calls},
        )


class FakeSpeaker:
    def __init__(self):
        self.texts = []

    def speak(self, text):
        self.texts.append(str(text))


def _row(step_index, action_name):
    return {
        "timestamp": f"2026-01-01T00:00:0{step_index}+00:00",
        "source": "step",
        "step_index": step_index,
        "image_path": f"images/step_{step_index:06d}.npz",
        "messages": [],
        "user_texts": [],
        "action": {
            "distribution": list(action_name_to_distribution(action_name)),
            "action_name": action_name,
            "executed_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
            "generated_text": "",
            "source": "vlm",
            "metadata": {},
        },
        "reward": 1.0,
        "done": False,
        "metadata": {"context": {}, "reward_result": {}},
        "latency_events": [{"name": "vlm_decision", "started_at": 0.0, "ended_at": 1.0, "duration_seconds": 1.0}],
    }


def _write_run(tmp_path, rows):
    run_dir = tmp_path / "phase1" / "run-a"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True)
    for row in rows:
        np.savez_compressed(
            images_dir / f"step_{row['step_index']:06d}.npz",
            image=np.full((4, 5, 3), row["step_index"], dtype=np.uint8),
        )
    (run_dir / "run_meta.json").write_text(json.dumps({"uuid": "run-a"}), encoding="utf-8")
    (run_dir / "observations.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return run_dir


def _cache_records(tmp_path, records):
    cache = VisualEncodingCache(
        EncodingCacheConfig(
            cache_root=tmp_path / "cache",
            model_name="qwen-test",
            manifest_path=tmp_path / "vlm_models.json",
            encoder_id="encoder-test",
            config_hash="unit",
        )
    )
    for idx, record in enumerate(records):
        cache.store(
            record,
            image_shape=(4, 5, 3),
            encoding=VisualTokenEncoding(np.full((3, 4), idx + 1, dtype=np.float32)),
        )
    return cache


def _train_tiny_policy(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _row(0, "drive-forward"),
            _row(1, "look-left"),
            _row(2, "drive-backward"),
            _row(3, "look-right"),
        ],
    )
    records = load_phase1_run(run_dir)
    cache = _cache_records(tmp_path, records)
    result = run_phase2_training(
        Phase2TrainingConfig(
            data_roots=(tmp_path / "phase1",),
            cache_root=cache.cache_root,
            cache_model_name=cache.config.model_name,
            cache_manifest_path=cache.config.manifest_path,
            cache_encoder_id=cache.config.encoder_id,
            cache_config_hash=cache.config.config_hash,
            output_root=tmp_path / "runs",
            checkpoint_root=tmp_path / "models",
            run_name="live-fit",
            fit_mode=FIT_MODE_WINDOW_SAMPLING,
            validation_fraction=0.0,
            context_steps=2,
            prediction_steps=2,
            batch_size=1,
            epochs=1,
            model_dim=6,
            conditioning_dim=4,
            conditioning_hidden_dim=5,
            token_type_dim=2,
            lstm_hidden_dim=7,
            learning_rate=1e-2,
            seed=7,
            device="cpu",
        )
    )
    return load_phase2_policy(checkpoint_root=tmp_path / "models", fit_id=result.run_id, device="cpu")


def _records(run_dir: Path) -> list[Phase1ObservationRecord]:
    return load_phase1_run(run_dir)


def test_phase2_live_bootstraps_then_uses_lstm_and_latency_breakouts(tmp_path):
    loaded = _train_tiny_policy(tmp_path)
    robot = FakeRobot()
    vlm = FakeVLM()
    encoder = FakeEncoder()
    speaker = FakeSpeaker()

    summary = run_phase2_live(
        Phase2LiveRunConfig(
            data_root=tmp_path / "phase2-live",
            run_uuid="live-run",
            max_steps=5,
            device="cpu",
        ),
        robot=robot,
        vlm=vlm,
        visual_encoder=encoder,
        loaded_policy=loaded,
        speaker=speaker,
    )

    assert summary.status == "completed"
    assert summary.steps_completed == 5
    assert summary.fit_id == "live-fit"
    assert len(robot.applied) == 5
    assert encoder.calls == 5
    assert vlm.calls == 4
    assert speaker.texts

    records = _records(summary.run_dir)
    assert [record.action.source for record in records[:2]] == ["vlm-bootstrap", "vlm-bootstrap"]
    assert [record.action.source for record in records[2:]] == ["phase2-lstm", "phase2-lstm", "phase2-lstm"]
    assert [record.metadata["mode"] for record in records] == ["bootstrap", "bootstrap", "lstm", "lstm", "lstm"]
    assert records[2].action.metadata["cycle_index"] == 0
    assert records[3].action.metadata["cycle_index"] == 0
    assert records[4].action.metadata["cycle_index"] == 1
    latency_names = {event.name for record in records for event in record.latency_events}
    assert "visual_encoding" in latency_names
    assert "lstm_action" in latency_names
    assert "vlm_head_refresh" in latency_names


def test_phase2_live_rejects_k_that_differs_from_prediction_steps(tmp_path):
    loaded = _train_tiny_policy(tmp_path)

    with pytest.raises(ValueError, match="K == prediction_steps"):
        run_phase2_live(
            Phase2LiveRunConfig(
                data_root=tmp_path / "phase2-live",
                max_steps=1,
                k=3,
                device="cpu",
            ),
            robot=FakeRobot(),
            vlm=FakeVLM(),
            visual_encoder=FakeEncoder(),
            loaded_policy=loaded,
        )
