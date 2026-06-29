import numpy as np

from picar_kl.actions import action_name_to_distribution
from picar_kl.vlm.control import build_phase1_messages
from picar_kl.vlm.qwen import QwenPhase1Config, QwenPhase1Controller


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeRow:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return FakeScalar(sum(self.values))


class FakeMask:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == 0
        return FakeRow(self.values)


class FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        self.messages = messages
        return "prompt"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text == ["prompt"]
        assert len(images) == 1
        assert padding is True
        assert return_tensors == "pt"
        return {"attention_mask": FakeMask([1, 1, 1])}

    def batch_decode(self, sequences, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert sequences == [["{" , '"action"', ":", '"look-up"', ",", '"say"', ":", '"Looking up."', "}"]]
        return ['{"action": "look-up", "say": "Looking up."}']


class FakeModel:
    def parameters(self):
        return iter(())

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["do_sample"] is False
        return [["p0", "p1", "p2", "{", '"action"', ":", '"look-up"', ",", '"say"', ":", '"Looking up."', "}"]]


def test_qwen_controller_parses_generated_action_response():
    processor = FakeProcessor()
    controller = QwenPhase1Controller(
        model=FakeModel(),
        processor=processor,
        config=QwenPhase1Config(),
    )
    messages = build_phase1_messages(
        task_prompt="find the red ball",
        step_index=0,
        user_texts=[],
    )

    decision = controller.decide(image_rgb=np.zeros((2, 2, 3), dtype=np.uint8), messages=messages)

    assert decision.action_distribution == action_name_to_distribution("look-up")
    assert decision.generated_text == "Looking up."
    assert decision.metadata["controller"] == "qwen2.5-vl"
    assert any(item.get("type") == "image" for item in processor.messages[-1]["content"])
