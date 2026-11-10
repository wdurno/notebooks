import json

import pytest
import torch

from numerical_experiment.picar import models, role_repair as rr
from test_picar_models import miniature_qwen


def test_native_precision_reference_matches_split_multimodal_batch(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen.bfloat16(), None, top_layers=1, ltm_rank=2)
    inputs = {"input_ids": torch.tensor([[0, 1, 37, 35, 38, 3], [1, 37, 35, 38, 4, 5]]),
              "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
              "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
              "pixel_values": torch.linspace(0, 1, 8 * 12).reshape(8, 12)}
    with torch.no_grad():
        split = encoder.encode_processed(inputs, role="teacher")
        expected = rr.native_hidden(encoder, inputs)
    torch.testing.assert_close(split, expected, atol=2e-6, rtol=2e-6)
    for layer in miniature_qwen.model.language_model.layers:
        assert not layer._forward_pre_hooks


@pytest.mark.parametrize("text", ['not JSON', '{}', '{"action":"teleport"}',
    '{"action":"stop","extra":1}', '[]', '{"action":null}'])
def test_invalid_generated_actions_are_not_silently_neutral(text):
    with pytest.raises(ValueError):
        rr.parse_action(text)


def test_stop_is_explicit_neutral_action_and_json_parser_is_strict():
    assert rr.parse_action('{"action":"stop"}') == 7
    assert "stop drive" in rr.GROUNDING
    assert "radians" in rr.GROUNDING and "metres/second" in rr.GROUNDING


def test_multiframe_preparation_uses_causal_rgb_and_text_only():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return json.dumps(messages)
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"input_ids": torch.ones(1, 12, dtype=torch.long),
                    "attention_mask": torch.ones(1, 12, dtype=torch.long)}
    class Encoder:
        context_length = 8
        max_context_chars = 4096
        max_tokens = 100
        processor = Processor()
    data = {"images": torch.linspace(0, 1, 4 * 3 * 4 * 4).reshape(1, 4, 3, 4, 4),
        "executed_actions": torch.zeros(1, 4, 4), "elapsed": torch.full((1, 4, 1), .5),
        "contexts": [[{"text": "approach and stop", "issued_at": 0.},
                      {"text": "future secret", "issued_at": 3.}]],
        "privileged_truth": "do not leak", "panel_id": "stopped_success"}
    encoder = Encoder()
    rr.prepare_role(encoder, data, "rubric", frames=3, max_pixels=76800)
    assert len(encoder.processor.kwargs["images"]) == 3
    torch.testing.assert_close(encoder.processor.kwargs["images"][0], data["images"][0, 1])
    message = encoder.processor.kwargs["text"][0]
    assert "1.000 seconds" in message and "2.000 seconds" in message
    assert "do not leak" not in message and "stopped_success" not in message
    assert "future secret" not in message
    assert encoder.processor.kwargs["max_pixels"] == 76800


def test_fp32_readout_reports_true_unrestricted_answer_mass(miniature_qwen):
    class Tokenizer:
        def encode(self, label, **kwargs):
            return [int(label)]
    class Processor:
        tokenizer = Tokenizer()
    encoder = models.QwenContextEncoder(miniature_qwen, Processor(), top_layers=1, ltm_rank=2)
    features = torch.arange(32).reshape(2, 16).float() / 32
    probs, diagnostics = rr.vocabulary_scores(encoder, features, ("3", "4"))
    head = encoder.qwen.lm_head
    full = torch.nn.functional.linear(features, head.weight.float()).softmax(-1)
    torch.testing.assert_close(torch.tensor(diagnostics["allowed_answer_mass"]), full[:, [3, 4]].sum(-1))
    torch.testing.assert_close(probs, full[:, [3, 4]] / full[:, [3, 4]].sum(-1, keepdim=True))


def test_native_generation_is_bounded_and_restores_head_and_role_state(miniature_qwen):
    from types import SimpleNamespace
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0),
                                batch_decode=lambda outputs, **kwargs: [str(row.tolist()) for row in outputs])
    encoder = models.QwenContextEncoder(miniature_qwen, processor, ltm_rank=2)
    inputs = {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    original = miniature_qwen.lm_head
    outputs = rr.native_text(encoder, inputs, max_new_tokens=2)
    assert len(outputs) == 1 and miniature_qwen.lm_head is original
    assert all(not layer._forward_pre_hooks for layer in miniature_qwen.model.language_model.layers)


def test_live_instruction_layout_preserves_legacy_and_never_reactivates_completed_commands():
    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            self.message = conversation[0]["content"][-1]["text"]
            return self.message
        def __call__(self, **kwargs):
            return {"input_ids": torch.ones(1, 3, dtype=torch.long)}
    class Encoder:
        processor = Processor()
        context_length = 8
        max_context_chars = 4096
        max_tokens = 20
    data = {"images": torch.full((1, 1, 3, 4, 4), .3), "executed_actions": torch.zeros(1, 1, 4),
        "elapsed": torch.ones(1, 1, 1), "contexts": [[
            {"text": "turn right", "issued_at": 0., "completed_at": .5},
            {"text": "stop immediately", "issued_at": .5}]]}
    encoder = Encoder()
    rr.prepare_role(encoder, data, "output schema")
    assert "CURRENT USER INSTRUCTIONS" not in encoder.processor.message
    rr.prepare_role(encoder, data, "output schema", active_last=True)
    active = encoder.processor.message.split("CURRENT USER INSTRUCTIONS:")[1]
    assert "stop immediately" in active and "turn right" not in active
    assert "Navigate past" in rr.GROUNDING and "Navigate past" not in rr.GROUNDING_V2
