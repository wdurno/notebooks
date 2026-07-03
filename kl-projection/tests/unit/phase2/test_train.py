import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from picar_kl.actions import action_name_to_distribution
from picar_kl.data.phase1 import load_phase1_run
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.dataset import collate_phase2_windows, load_phase2_windows
from picar_kl.phase2.train import (
    FIT_MODE_FULL_SEQUENCE,
    FIT_MODE_WINDOW_SAMPLING,
    Phase2TrainingConfig,
    phase2_window_batch_to_tensors,
    resolve_torch_device,
    run_phase2_training,
)


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


def _training_config(tmp_path, cache, **overrides):
    values = dict(
        data_roots=(tmp_path / "phase1",),
        cache_root=cache.cache_root,
        cache_model_name=cache.config.model_name,
        cache_manifest_path=cache.config.manifest_path,
        cache_encoder_id=cache.config.encoder_id,
        cache_config_hash="unit",
        output_root=tmp_path / "runs",
        checkpoint_root=tmp_path / "models",
        context_steps=1,
        prediction_steps=1,
        window_stride=1,
        batch_size=2,
        epochs=2,
        model_dim=6,
        conditioning_dim=4,
        conditioning_hidden_dim=5,
        token_type_dim=2,
        lstm_hidden_dim=7,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
    )
    values.update(overrides)
    return Phase2TrainingConfig(**values)


def _metric_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase2_window_batch_to_tensors_splits_prefix_and_target(tmp_path):
    run_dir = _write_run(tmp_path, [_row(0, "drive-forward"), _row(1, "look-left")])
    records = load_phase1_run(run_dir)
    cache = _cache_records(tmp_path, records)
    windows = load_phase2_windows(
        [tmp_path / "phase1"],
        cache=cache,
        context_steps=1,
        prediction_steps=1,
    )
    batch = collate_phase2_windows(windows)

    tensors = phase2_window_batch_to_tensors(batch, torch=torch)

    assert tensors["prefix_visual_tokens"].dtype == torch.float32
    assert tensors["target_visual_tokens"].dtype == torch.float32
    assert tensors["prefix_visual_tokens"].shape == (1, 1, 3, 4)
    assert tensors["target_visual_tokens"].shape == (1, 1, 3, 4)
    assert tensors["prefix_actions"].shape == (1, 1, 8)


def test_window_sampling_fit_writes_artifacts_metrics_and_checkpoint(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _row(0, "drive-forward"),
            _row(1, "look-left"),
            _row(2, "drive-backward"),
        ],
    )
    records = load_phase1_run(run_dir)
    cache = _cache_records(tmp_path, records)

    result = run_phase2_training(
        _training_config(
            tmp_path,
            cache,
            run_name="unit-run",
            fit_mode=FIT_MODE_WINDOW_SAMPLING,
            validation_fraction=0.5,
        )
    )

    assert result.window_count == 2
    assert result.train_window_count == 1
    assert result.validation_window_count == 1
    assert result.valid_steps == 2
    assert len(result.losses) == 2
    assert result.final_loss > 0.0
    assert result.summary_path.exists()
    assert result.summary_path.stat().st_size < 1_000_000
    assert result.artifact_dir == tmp_path / "models" / "unit-run"
    assert result.metrics_path == result.artifact_dir / "metrics.jsonl"
    assert (result.artifact_dir / "config.json").exists()
    assert (result.artifact_dir / "summary.json").exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["conditioning_mode"] == "trainable-prefix-head"
    assert summary["fit_mode"] == FIT_MODE_WINDOW_SAMPLING
    assert summary["dataset"]["visual_token_shape"] == [3, 4]
    assert summary["dataset"]["selected_window_count"] == 2
    assert summary["dataset"]["loaded_window_count"] == 2
    assert summary["dataset"]["candidate_window_count"] == 2
    assert summary["dataset"]["skipped_window_count"] == 0
    assert summary["dataset"]["skipped_record_count"] == 0
    assert summary["dataset"]["skipped_missing_cache_count"] == 0
    assert summary["dataset"]["skipped_no_action_count"] == 0
    assert summary["dataset"]["train_window_count"] == 1
    assert summary["dataset"]["validation_window_count"] == 1
    assert summary["training"]["final_loss"] == result.final_loss
    assert summary["training"]["device_requested"] == "cpu"
    assert summary["training"]["device"] == "cpu"
    assert result.device == "cpu"
    assert summary["artifacts"]["artifact_dir"] == str(result.artifact_dir)
    assert summary["artifacts"]["metrics_path"] == str(result.metrics_path)
    assert summary["breakout"]["fit_mode"] == FIT_MODE_WINDOW_SAMPLING
    assert summary["breakout"]["split_name"] == "train"
    assert summary["breakout"]["split_strategy"] == "random_window"
    assert summary["breakout"]["sampling_policy"] == "shuffle_without_replacement"
    assert summary["breakout"]["random_seed"] == 7

    rows = _metric_rows(result.metrics_path)
    assert {row["row_type"] for row in rows} == {"train_batch", "eval_epoch"}
    assert {row["breakout"]["split_name"] for row in rows} == {"train", "validation"}
    assert all(row["breakout"]["fit_mode"] == FIT_MODE_WINDOW_SAMPLING for row in rows)
    assert all(row["breakout"]["context_steps"] == 1 for row in rows)
    for row in rows:
        assert "top1_accuracy" in row["metrics"]
        assert "target_action_probability" in row["metrics"]
        assert "entropy" in row["metrics"]
        assert len(row["metrics"]["confusion_matrix"]) == 8


def test_run_phase2_training_can_skip_checkpoint_but_keeps_artifacts(tmp_path):
    run_dir = _write_run(tmp_path, [_row(0, "drive-forward"), _row(1, "look-left")])
    records = load_phase1_run(run_dir)
    cache = _cache_records(tmp_path, records)

    result = run_phase2_training(
        _training_config(
            tmp_path,
            cache,
            run_name="unit-no-checkpoint",
            batch_size=1,
            epochs=1,
            save_checkpoint=False,
            validation_fraction=0.0,
        )
    )

    assert result.checkpoint_path is None
    assert result.artifact_dir.exists()
    assert result.metrics_path.exists()
    assert (result.artifact_dir / "config.json").exists()
    assert not (result.artifact_dir / "policy.pt").exists()


def test_full_sequence_fit_uses_runtime_breakout(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _row(0, "drive-forward"),
            _row(1, "look-left"),
            _row(2, "drive-backward"),
        ],
    )
    records = load_phase1_run(run_dir)
    cache = _cache_records(tmp_path, records)

    result = run_phase2_training(
        _training_config(
            tmp_path,
            cache,
            run_name="unit-full-sequence",
            fit_mode=FIT_MODE_FULL_SEQUENCE,
            validation_fraction=0.5,
            save_checkpoint=False,
        )
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["fit_mode"] == FIT_MODE_FULL_SEQUENCE
    assert result.train_window_count == result.window_count == 2
    assert result.validation_window_count == 0

    rows = _metric_rows(result.metrics_path)
    eval_rows = [row for row in rows if row["row_type"] == "eval_epoch"]
    assert eval_rows
    assert {row["breakout"]["split_name"] for row in eval_rows} == {"runtime"}
    assert all(row["breakout"]["split_strategy"] == "full_sequence" for row in rows)
    assert all(row["breakout"]["sampling_policy"] == "sequential" for row in rows)
    assert all(row["breakout"]["random_seed"] is None for row in rows)


def test_resolve_torch_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = resolve_torch_device("auto", torch=torch)

    assert str(device) == "cuda"


def test_resolve_torch_device_cpu_is_explicit():
    device = resolve_torch_device("cpu", torch=torch)

    assert str(device) == "cpu"
