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
        if tokenize:
            token_count = 0
            for message in messages:
                token_count += 2
                for item in message.get("content", []):
                    if not isinstance(item, dict):
                        token_count += len(str(item).split())
                        continue
                    item_type = item.get("type")
                    if item_type == "image":
                        token_count += 1
                        continue
                    if item_type == "text":
                        token_count += max(1, len(str(item.get("text", "")).split()))
            if add_generation_prompt:
                token_count += 1
            return list(range(max(1, token_count)))
        roles = ",".join(message["role"] for message in messages)
        text_blocks = []
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    text_blocks.append(str(item.get("text", "")))
        return f"roles={roles};generation={add_generation_prompt};text={' '.join(text_blocks)}"

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


class FakeBatchEncoding:
    def __init__(self, payload):
        self.data = dict(payload)

    def get(self, key, default=None):
        return self.data.get(key, default)


class FakeBatchEncodingProcessor(FakeProcessor):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if tokenize:
            return FakeBatchEncoding({"input_ids": [[10, 11, 12, 13]]})
        return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


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


def test_qwen_backbone_latest_image_only_keeps_single_image_for_latest_message():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, all_images=False),
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


def test_qwen_backbone_default_adds_image_for_each_user_message():
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


def test_qwen_backbone_token_window_preserves_persistent_goal_system_message():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, prompt_token_window=32),
        model=model,
        processor=processor,
    )
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Persistent goals:\n1) Find the red ball."}],
            },
            {"role": "user", "content": [{"type": "text", "text": " ".join(["older"] * 40)}]},
            {"role": "assistant", "content": [{"type": "text", "text": " ".join(["response"] * 40)}]},
        ],
        t=0.0,
        step_index=5,
    )

    backbone.encode([observation], target_texts=[None], compute_vlm_loss=False)

    final_prompt_call = next(
        call
        for call in processor.template_calls
        if call["tokenize"] is False and call["add_generation_prompt"] is True
    )
    first_messages = final_prompt_call["messages"]
    assert first_messages[0]["role"] == "system"
    assert "PiCar-V robot" in first_messages[0]["content"][0]["text"]
    goal_messages = [
        message
        for message in first_messages
        if message.get("role") == "system"
        and any(
            isinstance(item, dict)
            and item.get("type") == "text"
            and str(item.get("text", "")).startswith("Persistent goals:")
            for item in message.get("content", [])
        )
    ]
    assert len(goal_messages) == 1


def test_qwen_backbone_token_window_clips_single_long_latest_message():
    processor = FakeProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, prompt_token_window=96),
        model=model,
        processor=processor,
    )
    long_text = " ".join(["drive-forward"] * 300)
    observation = ModelObservation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Persistent goals:\n1) Find the red ball."}],
            },
            {"role": "user", "content": [{"type": "text", "text": long_text}]},
        ],
        t=0.0,
        step_index=6,
    )

    backbone.encode([observation], target_texts=[None], compute_vlm_loss=False)

    final_prompt_call = next(
        call
        for call in processor.template_calls
        if call["tokenize"] is False and call["add_generation_prompt"] is True
    )
    first_messages = final_prompt_call["messages"]
    assert first_messages[-1]["role"] == "user"
    user_text = next(
        item["text"]
        for item in first_messages[-1]["content"]
        if isinstance(item, dict) and item.get("type") == "text"
    )
    assert user_text.endswith("[truncated]")
    assert backbone._prompt_token_count(first_messages) <= 96


def test_qwen_backbone_prompt_token_count_supports_batch_encoding_like_output():
    processor = FakeBatchEncodingProcessor()
    model = FakeGenerationModel()
    backbone = QwenLoRABackbone(
        config=ModelConfig(hidden_size=4, prompt_token_window=512),
        model=model,
        processor=processor,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]

    token_count = backbone._prompt_token_count(messages)

    assert token_count == 4


def test_resolve_model_hidden_size_uses_nested_text_config():
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=3584),
        )
    )

    hidden_size = _resolve_model_hidden_size(model)

    assert hidden_size == 3584
