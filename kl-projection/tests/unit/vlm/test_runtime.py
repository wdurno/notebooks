from contextlib import contextmanager

from picar_kl.vlm.runtime import QwenRuntime, count_input_ids


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        if tokenize:
            return [[1, 2, 3, 4]]
        return "prompt"


class FakeModel:
    training = False

    def __init__(self):
        self.adapter_disabled = False
        self.seen_disabled = []

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.seen_disabled.append(True)
            self.adapter_disabled = False


def test_runtime_counts_prompt_tokens():
    runtime = QwenRuntime(model=FakeModel(), processor=FakeProcessor())

    assert runtime.count_prompt_tokens([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]) == 4


def test_base_inference_context_disables_adapter_when_available():
    model = FakeModel()
    runtime = QwenRuntime(model=model, processor=FakeProcessor())

    with runtime.base_inference_context():
        assert model.adapter_disabled is True

    assert model.adapter_disabled is False
    assert model.seen_disabled == [True]


def test_count_input_ids_handles_plain_lists():
    assert count_input_ids([[1, 2], [3, 4]]) == 2
