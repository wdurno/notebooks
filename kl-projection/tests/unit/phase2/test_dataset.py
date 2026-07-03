import json

import numpy as np
import pytest

from picar_kl.actions import action_name_to_distribution
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.dataset import (
    Phase2DatasetError,
    collate_phase2_sequences,
    collate_phase2_windows,
    load_phase2_sequences,
    load_phase2_windows,
    load_phase2_windows_with_stats,
)


def _write_image(run_dir, step_index):
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"step_{step_index:06d}.npz"
    np.savez_compressed(path, image=np.full((4, 5, 3), step_index, dtype=np.uint8))
    return path.relative_to(run_dir)


def _new_row(step_index, action_name, distribution=None):
    distribution = distribution or action_name_to_distribution(action_name)
    return {
        "timestamp": f"2026-01-01T00:00:0{step_index}+00:00",
        "source": "step",
        "step_index": step_index,
        "image_path": f"images/step_{step_index:06d}.npz",
        "messages": [],
        "user_texts": [],
        "action": {
            "distribution": list(distribution),
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


def _legacy_row(step_index, action_name):
    return {
        "timestamp": f"2026-01-01T00:00:0{step_index}+00:00",
        "source": "step",
        "step_index": step_index,
        "image_path": f"images/step_{step_index:06d}.npz",
        "messages": [],
        "user_texts": [],
        "action": {
            "agentic_action_name": action_name,
            "executed_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
            "generated_text": "",
        },
        "reward": None,
        "done": False,
        "metadata": {"reward_prompt_id": "reward_prompt_1"},
    }


def _write_run(tmp_path, rows, *, run_uuid="run-a"):
    run_dir = tmp_path / run_uuid
    run_dir.mkdir(parents=True)
    for row in rows:
        _write_image(run_dir, row["step_index"])
    (run_dir / "run_meta.json").write_text(json.dumps({"uuid": run_uuid}), encoding="utf-8")
    (run_dir / "observations.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return run_dir


def _cache_for(tmp_path):
    return VisualEncodingCache(
        EncodingCacheConfig(
            cache_root=tmp_path / "cache",
            model_name="qwen-test",
            manifest_path=tmp_path / "vlm_models.json",
            encoder_id="encoder-test",
            config_hash="unit",
        )
    )


def _populate_cache(cache, records, *, shape=(3, 4)):
    for idx, record in enumerate(records):
        tokens = np.full(shape, idx + 1, dtype=np.float32)
        cache.store(
            record,
            image_shape=(4, 5, 3),
            encoding=VisualTokenEncoding(tokens=tokens, metadata={"unit": True}),
        )


def test_phase2_sequences_reconstruct_old_and_new_targets(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _legacy_row(0, "drive-forward"),
            _new_row(1, "look-left"),
            _new_row(2, "drive-backward"),
        ],
    )
    from picar_kl.data.phase1 import load_phase1_run

    records = load_phase1_run(run_dir)
    cache = _cache_for(tmp_path)
    _populate_cache(cache, records)

    sequences = load_phase2_sequences([tmp_path], cache=cache, sequence_length=2)
    batch = collate_phase2_sequences(sequences)

    assert len(sequences) == 2
    assert batch.visual_tokens.shape == (2, 2, 3, 4)
    assert batch.target_distributions[0, 0].tolist() == list(action_name_to_distribution("drive-forward"))
    assert batch.target_distributions[0, 1].tolist() == list(action_name_to_distribution("look-left"))
    assert batch.previous_actions[0, 0].tolist() == list(action_name_to_distribution("look-forward"))
    assert batch.previous_actions[0, 1].tolist() == list(action_name_to_distribution("drive-forward"))
    assert batch.readout_mask.tolist() == [[True, True], [True, False]]
    assert batch.step_mask.tolist() == [[True, True], [True, False]]
    assert batch.metadata[0][0]["missing_context_fields"] == ["context", "reward_result", "latency_events"]


def test_phase2_sequences_require_cached_encodings(tmp_path):
    _write_run(tmp_path, [_new_row(0, "drive-forward")])
    cache = _cache_for(tmp_path)

    with pytest.raises(Phase2DatasetError, match="missing cached visual encoding"):
        load_phase2_sequences([tmp_path], cache=cache, sequence_length=2)


def test_collate_rejects_incompatible_visual_token_shapes(tmp_path):
    run_dir = _write_run(tmp_path, [_new_row(0, "drive-forward"), _new_row(1, "look-left")])
    from picar_kl.data.phase1 import load_phase1_run

    records = load_phase1_run(run_dir)
    cache = _cache_for(tmp_path)
    cache.store(records[0], image_shape=(4, 5, 3), encoding=VisualTokenEncoding(np.zeros((3, 4), dtype=np.float32)))
    cache.store(records[1], image_shape=(4, 5, 3), encoding=VisualTokenEncoding(np.zeros((4, 4), dtype=np.float32)))

    sequences = load_phase2_sequences([tmp_path], cache=cache, sequence_length=1)

    with pytest.raises(Phase2DatasetError, match="incompatible visual token shape"):
        collate_phase2_sequences(sequences)


def test_phase2_windows_split_prefix_and_target_without_leakage(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _new_row(0, "drive-forward"),
            _new_row(1, "look-left"),
            _new_row(2, "drive-backward"),
            _new_row(3, "look-right"),
        ],
    )
    from picar_kl.data.phase1 import load_phase1_run

    records = load_phase1_run(run_dir)
    cache = _cache_for(tmp_path)
    _populate_cache(cache, records)

    windows = load_phase2_windows(
        [tmp_path],
        cache=cache,
        context_steps=2,
        prediction_steps=2,
    )
    batch = collate_phase2_windows(windows)

    assert len(windows) == 1
    assert batch.prefix.visual_tokens.shape == (1, 2, 3, 4)
    assert batch.target.visual_tokens.shape == (1, 2, 3, 4)
    assert batch.prefix.target_distributions[0, 0].tolist() == list(action_name_to_distribution("drive-forward"))
    assert batch.prefix.target_distributions[0, 1].tolist() == list(action_name_to_distribution("look-left"))
    assert batch.target.target_distributions[0, 0].tolist() == list(action_name_to_distribution("drive-backward"))
    assert batch.target.target_distributions[0, 1].tolist() == list(action_name_to_distribution("look-right"))
    assert batch.target.previous_actions[0, 0].tolist() == list(action_name_to_distribution("look-left"))


def test_phase2_windows_with_stats_counts_skipped_cache_gaps(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _new_row(0, "drive-forward"),
            _new_row(1, "look-left"),
            _new_row(2, "drive-backward"),
            _new_row(3, "look-right"),
        ],
    )
    from picar_kl.data.phase1 import load_phase1_run

    records = load_phase1_run(run_dir)
    cache = _cache_for(tmp_path)
    cache.store(records[0], image_shape=(4, 5, 3), encoding=VisualTokenEncoding(np.ones((3, 4), dtype=np.float32)))
    cache.store(records[2], image_shape=(4, 5, 3), encoding=VisualTokenEncoding(np.ones((3, 4), dtype=np.float32) * 3))
    cache.store(records[3], image_shape=(4, 5, 3), encoding=VisualTokenEncoding(np.ones((3, 4), dtype=np.float32) * 4))

    result = load_phase2_windows_with_stats(
        [tmp_path],
        cache=cache,
        context_steps=1,
        prediction_steps=1,
        stride=1,
        allow_missing_cached_encodings=True,
    )

    assert result.candidate_window_count == 3
    assert len(result.windows) == 1
    assert result.skipped_window_count == 2
    assert result.skipped_record_count == 1
    assert result.skipped_missing_cache_count == 1
    assert result.skipped_no_action_count == 0
    assert result.windows[0].prefix.steps[0].metadata["step_index"] == 2
    assert result.windows[0].target.steps[0].metadata["step_index"] == 3
