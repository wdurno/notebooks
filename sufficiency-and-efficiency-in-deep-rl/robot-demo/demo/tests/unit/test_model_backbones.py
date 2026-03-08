from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.backbones import QwenLoRABackbone, _ensure_image_placeholder, _resolve_model_hidden_size
from model.config import ModelConfig
from model.schemas import ModelObservation


class FakeProcessor:
    def __init__(self):
        self.template_calls = []
        self.encode_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        roles = ",".join(message["role"] for message in messages)
        return f"roles={roles};generation={add_generation_prompt}"

    def __call__(self, text, images, padding=False, return_tensors="pt"):
        self.encode_calls.append(
            {
                "text": text,
                "images": images,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )
        batch_size = len(text)
        seq_len = 6 if any("False" in item for item in text) else 4
        input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64).reshape(batch_size, seq_len)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        del skip_special_tokens
        return ['{"action": "drive-forward", "say": "moving"}' for _ in sequences]


class FakeGenerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Linear(1, 1)
        self.generate_calls = []

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, labels=None):
        del attention_mask, output_hidden_states, return_dict
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones(batch_size, seq_len, self.config.hidden_size)
        logits = torch.ones(batch_size, seq_len, 8, dtype=torch.float32)
        loss = torch.tensor(1.25) if labels is not None else None
        return SimpleNamespace(hidden_states=[hidden], logits=logits, loss=loss)

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return_dict_in_generate = bool(kwargs.get("return_dict_in_generate", False))
        output_scores = bool(kwargs.get("output_scores", False))
        batch_size, seq_len = input_ids.shape
        continuation = torch.full((batch_size, 2), 7, dtype=torch.int64)
        sequences = torch.cat([input_ids, continuation], dim=1)
        if not return_dict_in_generate:
            return sequences
        scores = []
        if output_scores:
            for _ in range(continuation.shape[1]):
                scores.append(torch.ones((batch_size, 8), dtype=torch.float32))
        return SimpleNamespace(sequences=sequences, scores=scores)


def test_ensure_image_placeholder_injects_into_last_user_message():
    original = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    updated = _ensure_image_placeholder(original)

    assert original[0]["content"][0] == {"type": "text", "text": "hello"}
    assert updated[0]["content"][0] == {"type": "image"}
    assert updated[0]["content"][1] == {"type": "text", "text": "hello"}


def test_qwen_backbone_encode_uses_chat_template_messages():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4),
        model=model,
        processor=processor,
    )
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[{"role": "user", "content": [{"type": "text", "text": "find the ball"}]}],
        t=0.0,
        step_index=3,
    )

    output = backbone.encode([observation], target_texts=['{"action":"drive-forward"}'], compute_vlm_loss=True)

    assert processor.template_calls
    first_messages = processor.template_calls[0]["messages"]
    assert first_messages[0]["role"] == "system"
    assert first_messages[1]["role"] == "user"
    assert first_messages[1]["content"][0] == {"type": "image"}
    assert output.agentic_action_names == ["drive-forward"]
    assert output.generated_texts == ["moving"]
    assert float(output.vlm_loss.item()) > 0.0
    assert model.generate_calls
    generation_kwargs = model.generate_calls[-1]
    assert generation_kwargs["do_sample"] is True
    assert generation_kwargs["temperature"] == 0.3
    assert generation_kwargs["top_p"] == 0.95


def test_qwen_backbone_default_keeps_single_image_for_latest_message():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4),
        model=model,
        processor=processor,
    )
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "second question"}]},
        ],
        t=0.0,
        step_index=3,
    )

    backbone.encode([observation], target_texts=[None], compute_vlm_loss=False)

    first_encode = processor.encode_calls[0]
    assert isinstance(first_encode["images"], list)
    assert len(first_encode["images"]) == 1
    first_messages = processor.template_calls[0]["messages"]
    user_messages = [message for message in first_messages if message["role"] == "user"]
    assert user_messages[0]["content"][0] != {"type": "image"}
    assert user_messages[-1]["content"][0] == {"type": "image"}


def test_qwen_backbone_all_images_adds_image_for_each_user_message():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, all_images=True),
        model=model,
        processor=processor,
    )
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "second question"}]},
        ],
        t=0.0,
        step_index=4,
    )

    backbone.encode([observation], target_texts=[None], compute_vlm_loss=False)

    first_encode = processor.encode_calls[0]
    assert isinstance(first_encode["images"], list)
    assert len(first_encode["images"]) == 2
    first_messages = processor.template_calls[0]["messages"]
    user_messages = [message for message in first_messages if message["role"] == "user"]
    assert user_messages[0]["content"][0] == {"type": "image"}
    assert user_messages[1]["content"][0] == {"type": "image"}


def test_qwen_backbone_encode_can_disable_sampling():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, deterministic_coding=True),
        model=model,
        processor=processor,
    )
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[{"role": "user", "content": [{"type": "text", "text": "drive"}]}],
        t=0.0,
        step_index=0,
    )

    output = backbone.encode([observation], target_texts=[None], compute_vlm_loss=False)

    assert output.agentic_action_names == ["drive-forward"]
    assert model.generate_calls
    generation_kwargs = model.generate_calls[-1]
    assert generation_kwargs["do_sample"] is False
    assert "temperature" not in generation_kwargs
    assert "top_p" not in generation_kwargs
    assert "top_k" not in generation_kwargs


def test_resolve_model_hidden_size_uses_nested_text_config():
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=3584),
        )
    )

    hidden_size = _resolve_model_hidden_size(model)

    assert hidden_size == 3584
