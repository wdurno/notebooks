"""The reference refinement keeps target answers out of Qwen's context."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from numerical_experiment.picar_ball_width_probe import PROMPT
from numerical_experiment.picar_reference_reward_probe import prepare_references
from numerical_experiment.picar_width_benchmark import error_fields
from numerical_experiment.picar_width_refinement import (
    FRESH_SCENES, METHODS, check_disjoint, select_references, summarize,
)


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "prompt"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"input_ids": torch.zeros((1, 32), dtype=torch.int64)}


def reference_data():
    original = [{"frame_ref": name, "reference": name, "width_percent": width, "reward": min(10, width / 8)}
                for name, width in zip("ABCD", (0, 4.6875, 10.3125, 87.8125))]
    development = [{"frame_ref": "middle", "panel_id": "C18", "truth": {"width_percent": 41.5625},
                    "scene": {"barrier": "none"}},
                   {"frame_ref": "partial", "panel_id": "C17", "truth": {"width_percent": 10.9375},
                    "scene": {"barrier": "partial_right"}}]
    return original, development


def test_reference_order_fills_gap_preserving_original_relative_order():
    original, development = reference_data()
    baseline, refined = select_references(original, development)
    assert [r["reference"] for r in baseline] == list("ABCD")
    assert [r["reference"] for r in refined] == list("ABCFED")
    assert [r["reward"] for r in refined][3:5] == pytest.approx([1.3671875, 5.1953125])
    assert "source_group" not in original[0]


def test_new_panel_has_fixed_coverage_and_distinct_geometry():
    from numerical_experiment.picar_width_scenes import SCENES
    assert len(FRESH_SCENES) == 20
    assert sum(not s.visible for s in FRESH_SCENES) == 3
    assert sum(s.barrier.startswith("partial") for s in FRESH_SCENES) == 2
    assert all(a != b for a, b in zip(FRESH_SCENES, SCENES))


def test_target_disjointness_covers_development_as_well_as_examples():
    original, development = reference_data()
    _, refs = select_references(original, development)
    check_disjoint([{"frame_ref": "fresh"}], refs, development)
    for targets in ([{"frame_ref": "fresh"}] * 2, [{"frame_ref": "middle"}], [{"frame_ref": "A"}]):
        with pytest.raises(ValueError):
            check_disjoint(targets, refs, development)
    with pytest.raises(ValueError, match="development"):
        check_disjoint([{"frame_ref": "unused-development"}], refs,
                       development + [{"frame_ref": "unused-development"}])


def test_six_examples_keep_same_prompt_and_target_is_unanswered():
    original, development = reference_data()
    _, refs = select_references(original, development)
    refs = [{**r, "rgb": np.full((3, 4, 3), i, dtype=np.uint8)} for i, r in enumerate(refs)]
    processor = Processor()
    encoder = SimpleNamespace(processor=processor, max_tokens=2048)
    prepare_references(encoder, refs, np.full((3, 4, 3), 99, dtype=np.uint8), few_shot=True)
    assert processor.messages[0]["content"][0]["text"] == PROMPT
    assert [m["role"] for m in processor.messages] == ["system"] + ["user", "assistant"] * 6 + ["user"]
    for index, ref in enumerate(refs):
        answer = json.loads(processor.messages[2 + 2 * index]["content"][0]["text"])
        assert answer == {"width_percent": round(ref["width_percent"], 4), "reward": round(ref["reward"], 4)}
        assert processor.inputs["images"][index].mean().item() == pytest.approx(index / 255)
    assert len(processor.inputs["images"]) == 7
    assert processor.inputs["images"][-1].mean().item() == pytest.approx(99 / 255)
    assert processor.messages[-1]["content"] == [
        {"type": "image"}, {"type": "text", "text": "Score the image and return JSON."}]


def sample_rows():
    return [{"panel_id": "T01", "method": method, "true_visible": True, "partial_occlusion": True,
             "true_width_percent": 40., "width_percent": width, "reward": reward,
             "seconds": 1., "input_tokens": 20, "target_pixels_sha256": "same",
             **error_fields(width, reward, 40.)}
            for method, width, reward in zip(METHODS, (80., 40.), (10., 5.))]


def test_paired_metrics_keep_native_reward_and_coverage_counts():
    metrics = summarize([{"panel_id": "T01"}], sample_rows())
    assert metrics[METHODS[0]]["reward_mae"] == 5.
    assert metrics[METHODS[1]]["reward_mae"] == 0.
    assert metrics[METHODS[1]]["middle_cases"] == 1
    assert metrics[METHODS[1]]["partial_cases"] == 1
    assert metrics["paired_rewards"] == {"valid_pairs": 1, "six_better": 1, "six_worse": 0, "ties": 0}


def test_incomplete_mismatched_or_invalid_outputs_not_silently_scored():
    rows = sample_rows()
    with pytest.raises(ValueError, match="incomplete"):
        summarize([{"panel_id": "T01"}], rows[:1])
    rows[1]["target_pixels_sha256"] = "changed"
    with pytest.raises(ValueError, match="pixel mismatch"):
        summarize([{"panel_id": "T01"}], rows)
    rows[1].update(target_pixels_sha256="same", parse_error="bad JSON")
    metrics = summarize([{"panel_id": "T01"}], rows)
    assert metrics[METHODS[1]]["invalid"] == 1
    assert metrics[METHODS[1]]["reward_mae"] is None
    assert metrics["paired_rewards"]["valid_pairs"] == 0
