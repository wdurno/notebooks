"""Reference/test separation and image-label binding for the frozen prompt probe."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar_reference_reward_probe import (
    TEST_STATES, parse_reference, pixel_width, prepare_references, select_panels,
)


def panels():
    return [{"seed": seed, "state": state, "frame_ref": f"{seed}-{state}"}
            for seed in range(211, 215) for state in TEST_STATES]


def test_reference_and_test_images_are_disjoint_by_content_and_seed():
    refs, targets = select_panels(panels())
    assert len(refs) == 4 and len(targets) == 18
    assert {p["seed"] for p in refs} == {211}
    assert {p["seed"] for p in targets} == {212, 213, 214}
    bad = panels()
    bad[6]["frame_ref"] = bad[0]["frame_ref"]
    with pytest.raises(ValueError, match="contents overlap"):
        select_panels(bad)


@pytest.mark.parametrize("raw", [
    '{"reference": "E"}', '{"reference": 1}', '{"reference": ["A"]}',
    '{"reference": "A", "reward": 0}', '{"reference": "A", "reference": "B"}',
    'Reference B', '{}',
])
def test_invalid_choice_is_not_fabricated_into_a_reward(raw):
    with pytest.raises(ValueError):
        parse_reference(raw)


def test_choice_is_a_reference_identity_not_an_image_position():
    for identity in "ABCD":
        assert parse_reference('{"reference": "%s"}' % identity) == identity


def test_red_width_check_and_anchor_scale():
    rgb = np.zeros((10, 100, 3), dtype=np.uint8)
    assert pixel_width(rgb) == 0.
    rgb[2:7, 40:50, 0] = 200
    assert pixel_width(rgb) == 10.


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "prompt"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"input_ids": torch.zeros((1, 32), dtype=torch.int64)}


@pytest.mark.parametrize("reverse", [False, True])
def test_four_labeled_examples_then_target_without_target_metadata(reverse):
    processor = Processor()
    encoder = SimpleNamespace(processor=processor, max_tokens=2048)
    refs = [{"reference": identity, "rgb": np.full((3, 4, 3), i, dtype=np.uint8), "reward": i}
            for i, identity in enumerate("ABCD")]
    prepare_references(encoder, refs, np.full((3, 4, 3), 100, dtype=np.uint8), reverse=reverse)
    texts = [x["text"] for x in processor.messages[1]["content"] if x["type"] == "text"]
    expected = "DCBA" if reverse else "ABCD"
    assert [text.split()[1] for text in texts[:4]] == list(expected)
    assert texts[-1].startswith("TARGET image:")
    assert len(processor.inputs["images"]) == 5
    for image, identity in zip(processor.inputs["images"][:4], expected):
        assert image.mean().item() == pytest.approx("ABCD".index(identity) / 255)
    assert processor.inputs["images"][-1].mean().item() == pytest.approx(100 / 255)
    assert processor.inputs["do_rescale"] is False


def test_no_silent_prompt_truncation():
    encoder = SimpleNamespace(processor=Processor(), max_tokens=16)
    refs = [{"reference": identity, "rgb": np.zeros((3, 4, 3), dtype=np.uint8), "reward": i}
            for i, identity in enumerate("ABCD")]
    with pytest.raises(ValueError, match="no silent truncation"):
        prepare_references(encoder, refs, np.zeros((3, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("reverse", [False, True])
def test_few_shot_examples_have_assistant_answers_but_target_has_none(reverse):
    import json

    processor = Processor()
    encoder = SimpleNamespace(processor=processor, max_tokens=2048)
    refs = [{"reference": identity, "rgb": np.full((3, 4, 3), i, dtype=np.uint8),
             "width_percent": i * 8, "reward": i} for i, identity in enumerate("ABCD")]
    prepare_references(encoder, refs, np.full((3, 4, 3), 100, dtype=np.uint8),
                       reverse=reverse, few_shot=True)
    messages = processor.messages
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 4 + ["user"]
    expected = list(reversed(refs)) if reverse else refs
    for index, item in enumerate(expected):
        answer = json.loads(messages[2 + 2 * index]["content"][0]["text"])
        assert answer == {"width_percent": item["width_percent"], "reward": item["reward"]}
        assert processor.inputs["images"][index].mean().item() == pytest.approx(item["reward"] / 255)
    assert len(processor.inputs["images"]) == 5
    assert messages[-1]["content"] == [{"type": "image"}, {"type": "text", "text": "Score the image and return JSON."}]
