"""Reference permutations keep each image attached to its known answer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar_reference_order_probe import ORDERS, ordered_references
from numerical_experiment.picar_reference_reward_probe import SCORE_ONLY_PROMPT, prepare_references


def references():
    return [{"reference": identity, "width_percent": i * 8., "reward": float(i),
             "rgb": np.full((3, 4, 3), i, dtype=np.uint8)} for i, identity in enumerate("ABCD")]


def test_orders_are_balanced_at_each_reference_position():
    assert len(set(ORDERS)) == 4
    for position in range(4):
        assert {order[position] for order in ORDERS} == set("ABCD")


@pytest.mark.parametrize("order", ["AAAA", "ABC", "ABCDE", "ABCE"])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError):
        ordered_references(references(), order)


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "prompt"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"input_ids": torch.zeros((1, 32), dtype=torch.int64)}


@pytest.mark.parametrize("order", ORDERS)
def test_actual_five_image_sequence_and_answers_follow_permutation(order):
    processor = Processor()
    encoder = SimpleNamespace(processor=processor, max_tokens=2048)
    refs = references()
    ordered = ordered_references(refs, order)
    prepare_references(encoder, ordered, np.full((3, 4, 3), 100, dtype=np.uint8), few_shot=True)
    for index, identity in enumerate(order):
        original = next(item for item in refs if item["reference"] == identity)
        answer = json.loads(processor.messages[2 + index * 2]["content"][0]["text"])
        assert answer["width_percent"] == original["width_percent"]
        assert answer["reward"] == original["reward"]
        assert processor.inputs["images"][index].mean().item() == pytest.approx(original["reward"] / 255)
    assert len(processor.inputs["images"]) == 5
    assert processor.inputs["images"][-1].mean().item() == pytest.approx(100 / 255)
    assert processor.messages[-1]["role"] == "user"
    assert "TARGET" not in json.dumps(processor.messages[:-1])


@pytest.mark.parametrize("order", ORDERS)
def test_score_only_context_contains_no_width_or_arithmetic_and_keeps_images(order):
    processor = Processor()
    encoder = SimpleNamespace(processor=processor, max_tokens=2048)
    refs = references()
    ordered = ordered_references(refs, order)
    prepare_references(encoder, ordered, np.full((3, 4, 3), 100, dtype=np.uint8),
                       few_shot=True, score_only=True)
    context = json.dumps(processor.messages)
    assert "width" not in context.lower()
    assert "min(" not in context
    assert "/ 8" not in context
    assert processor.messages[0]["content"][0]["text"] == SCORE_ONLY_PROMPT
    for index, item in enumerate(ordered):
        answer = json.loads(processor.messages[2 + index * 2]["content"][0]["text"])
        assert answer == {"reward": item["reward"]}
        assert processor.inputs["images"][index].mean().item() == pytest.approx(item["reward"] / 255)
    assert len(processor.inputs["images"]) == 5
    assert processor.inputs["images"][-1].mean().item() == pytest.approx(100 / 255)


def test_score_only_cannot_silently_use_reference_choice_format():
    encoder = SimpleNamespace(processor=Processor(), max_tokens=2048)
    with pytest.raises(ValueError, match="worked examples"):
        prepare_references(encoder, references(), np.zeros((3, 4, 3), dtype=np.uint8), score_only=True)
