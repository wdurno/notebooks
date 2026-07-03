import json
import os
import time

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from picar_kl.actions import action_name_to_distribution, validate_action_distribution
from picar_kl.data.phase1 import load_phase1_run
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.runtime import (
    latest_phase2_fit_artifact,
    load_phase2_policy,
    Phase2ReplayConfig,
    replay_phase2_policy,
    select_phase2_fit_artifact,
)
from picar_kl.phase2.train import FIT_MODE_WINDOW_SAMPLING, Phase2TrainingConfig, run_phase2_training


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


def _train_tiny_fit(tmp_path, *, run_name="fit-a"):
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
            run_name=run_name,
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
    return result, cache


def test_select_phase2_fit_artifact_by_latest_fit_id_and_checkpoint_path(tmp_path):
    root = tmp_path / "models"
    older = root / "older"
    newer = root / "newer"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "policy.pt").write_bytes(b"old")
    (newer / "policy.pt").write_bytes(b"new")
    now = time.time()
    os.utime(older / "policy.pt", (now - 10, now - 10))
    os.utime(newer / "policy.pt", (now, now))
    (newer / "summary.json").write_text(json.dumps({"dataset": {"context_steps": 1, "prediction_steps": 1}}), encoding="utf-8")

    latest = latest_phase2_fit_artifact(root)
    by_id = select_phase2_fit_artifact(checkpoint_root=root, fit_id="newer")
    by_checkpoint = select_phase2_fit_artifact(checkpoint_path=older / "policy.pt")

    assert latest.fit_id == "newer"
    assert by_id.fit_id == "newer"
    assert by_id.context_steps == 1
    assert by_id.prediction_steps == 1
    assert by_checkpoint.fit_id == "older"


def test_load_phase2_policy_reconstructs_model_from_checkpoint(tmp_path):
    result, _cache = _train_tiny_fit(tmp_path)

    loaded = load_phase2_policy(checkpoint_root=tmp_path / "models", fit_id=result.run_id, device="cpu")

    assert loaded.artifact.fit_id == result.run_id
    assert loaded.device == "cpu"
    assert loaded.model_config["visual_dim"] == 4
    assert loaded.model.training is False


def test_replay_phase2_policy_emits_valid_action_records(tmp_path):
    result, cache = _train_tiny_fit(tmp_path)

    replay = replay_phase2_policy(
        config=Phase2ReplayConfig(
            data_roots=(tmp_path / "phase1",),
            checkpoint_root=tmp_path / "models",
            fit_id=result.run_id,
            cache_root=cache.cache_root,
            device="cpu",
            max_windows=1,
        )
    )

    assert replay.fit_id == result.run_id
    assert replay.device == "cpu"
    assert replay.context_steps == 2
    assert replay.prediction_steps == 2
    assert replay.window_count == 1
    assert replay.replayed_step_count == 2
    payload = replay.to_dict()
    assert len(payload["steps"]) == 2
    for step in replay.steps:
        validate_action_distribution(step.action.distribution, tolerance=1e-5)
        assert step.action.source == "phase2-lstm-replay"
        assert set(step.action.executed_vector) == {"pan", "tilt", "turn", "drive"}
        assert step.latency_seconds >= 0.0
        assert step.conditioning_norm >= 0.0
