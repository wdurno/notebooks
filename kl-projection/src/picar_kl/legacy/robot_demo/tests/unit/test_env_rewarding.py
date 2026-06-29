from pathlib import Path
import sys
import types

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.config import RewardConfig
from env.rewarding import FrozenVLMRewardScorer, RewardPromptRegistry, parse_reward_text


def test_parse_reward_text_prefers_json_and_clips():
    reward, clipped = parse_reward_text('{"reward": 11.5}', min_reward=0.0, max_reward=10.0)

    assert reward == 11.5
    assert clipped == 10.0


def test_parse_reward_text_falls_back_to_numeric_text():
    reward, clipped = parse_reward_text("reward = 4.25", min_reward=0.0, max_reward=10.0)

    assert reward == 4.25
    assert clipped == 4.25


def test_default_reward_registry_exposes_prompt_one():
    registry = RewardPromptRegistry.default()

    assert registry.list_prompt_ids() == ["reward_prompt_1"]
    assert "red ball" in registry.get("reward_prompt_1").prompt_text


class _FakeProcessor:
    def __init__(self):
        self.template_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del messages, tokenize, add_generation_prompt
        self.template_calls += 1
        return "reward prompt"

    def __call__(self, text, images, return_tensors="pt"):
        del text, images, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }

    def batch_decode(self, token_batches, skip_special_tokens=True):
        del token_batches, skip_special_tokens
        return ['{"reward": 7}']


class _FakeSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.disable_calls = 0
        self.generate_kwargs = []
        self.eval_calls = 0
        self.train_calls = 0
        self.training = True

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.train_calls += 1
        self.training = bool(mode)
        return self

    def disable_adapter(self):
        self.disable_calls += 1
        model = self

        class _Ctx:
            def __enter__(self_inner):
                return model

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    def generate(self, **kwargs):
        self.generate_kwargs.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        continuation = torch.tensor([[9]], dtype=torch.int64, device=input_ids.device)
        return torch.cat([input_ids, continuation], dim=1)


def test_reward_scorer_uses_shared_model_and_disables_adapter(monkeypatch):
    fake_pil = types.ModuleType("PIL")

    class _Image:
        @staticmethod
        def fromarray(value):
            return value

    fake_pil.Image = _Image
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    model = _FakeSharedModel()
    processor = _FakeProcessor()
    scorer = FrozenVLMRewardScorer(
        RewardConfig(prompt_id="reward_prompt_1"),
        shared_model=model,
        shared_processor=processor,
    )

    result = scorer.score(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.prompt_id == "reward_prompt_1"
    assert result.reward == pytest.approx(7.0)
    assert model.disable_calls == 1
    assert model.eval_calls == 1
    assert model.train_calls == 1
    assert model.training is True
    assert model.generate_kwargs
    assert model.generate_kwargs[0]["do_sample"] is False


def test_reward_scorer_requires_complete_shared_pair():
    scorer = FrozenVLMRewardScorer(
        RewardConfig(prompt_id="reward_prompt_1"),
        shared_model=_FakeSharedModel(),
        shared_processor=None,
    )
    with pytest.raises(ValueError, match="shared_model"):
        scorer._load_model_and_processor()
