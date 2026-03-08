from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import torch

try:
    from src.model.config import ModelConfig
    from src.model.model_store import ModelStore
    from src.model.processor_loader import load_qwen_2_5_vl_processor
except ModuleNotFoundError:
    from model.config import ModelConfig
    from model.model_store import ModelStore
    from model.processor_loader import load_qwen_2_5_vl_processor

from .config import RewardConfig
from .schemas import RewardPromptSpec, RewardResult

LOGGER = logging.getLogger(__name__)


REWARD_PROMPT_1 = """
You are a reinforcement learning assistant in charge of deciding rewards.
You are receiving images from a mobile, robotic camera.
Primary visual task: finding a red ball.
If you see a red ball in the image, you are to return a reward in [1, 10], otherwise zero.
If the ball is far away, return a 1.
If the ball is close enough to fill the screen while still being fully visible, return a 10.
If the ball is too close, entirely filling the screen, return a 1.
For intermediary distances, return interpolated values in (1, 10).
So, the robot only gets the highest score when the red ball is the right distance from the camera.
The environment applies command-following shaping separately (+2 on completion, -2 when ignored).
Return JSON with exactly one key: `reward`.
Example output when no red ball is visible:
{"reward": 0}
Example output when the red ball is centered and at a good distance:
{"reward": 9}
""".strip()


@dataclass(frozen=True)
class RewardPromptRegistry:
    prompts: dict[str, RewardPromptSpec]

    @classmethod
    def default(cls) -> "RewardPromptRegistry":
        spec = RewardPromptSpec(
            prompt_id="reward_prompt_1",
            prompt_text=REWARD_PROMPT_1,
            task_text="find the red ball",
            min_reward=0.0,
            max_reward=10.0,
        )
        return cls(prompts={spec.prompt_id: spec})

    def get(self, prompt_id: str) -> RewardPromptSpec:
        try:
            return self.prompts[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown reward prompt: {prompt_id}") from exc

    def list_prompt_ids(self) -> list[str]:
        return sorted(self.prompts)


def parse_reward_text(text: str, *, min_reward: float = 0.0, max_reward: float = 10.0) -> tuple[float, float]:
    candidate = (text or "").strip()
    if not candidate:
        raise ValueError("Reward output was empty")
    value = _extract_reward_value(candidate)
    clipped = min(max(value, min_reward), max_reward)
    return value, clipped


class FrozenVLMRewardScorer:
    """Score images with the untuned base VLM and a selected reward prompt."""

    def __init__(
        self,
        config: RewardConfig,
        *,
        model_config: ModelConfig | None = None,
        registry: RewardPromptRegistry | None = None,
        shared_model: Any | None = None,
        shared_processor: Any | None = None,
        disable_shared_adapter: bool = True,
    ):
        self.config = config
        self.registry = registry or RewardPromptRegistry.default()
        self.model_config = model_config or ModelConfig(allow_downloads=config.allow_downloads)
        self._shared_model = shared_model
        self._shared_processor = shared_processor
        self._disable_shared_adapter = bool(disable_shared_adapter)
        self._model = None
        self._processor = None

    def score(self, image_rgb: Any) -> RewardResult:
        prompt_spec = self.registry.get(self.config.prompt_id)
        model, processor = self._load_model_and_processor()
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("[shared-state] reward before_score %s", _adapter_state_summary(model))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt_spec.prompt_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Score the image and return JSON."},
                ],
            },
        ]
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to score images with the reward VLM") from exc

        pil_image = Image.fromarray(image_rgb)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_inputs = processor(text=[prompt_text], images=[pil_image], return_tensors="pt")
        model_device = next(model.parameters()).device
        model_inputs = {name: tensor.to(model_device) for name, tensor in processor_inputs.items()}
        with torch.inference_mode():
            with self._shared_model_inference_context(model):
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.generation_max_new_tokens,
                    do_sample=False,
                )
        prompt_length = int(model_inputs["attention_mask"][0].sum().item())
        completion_ids = generated_ids[0, prompt_length:]
        raw_text = processor.batch_decode([completion_ids], skip_special_tokens=True)[0].strip()
        reward, clipped_reward = parse_reward_text(
            raw_text,
            min_reward=prompt_spec.min_reward,
            max_reward=prompt_spec.max_reward,
        )
        return RewardResult(
            prompt_id=prompt_spec.prompt_id,
            raw_text=raw_text,
            reward=reward,
            clipped_reward=clipped_reward,
            metadata={
                "prompt_text": prompt_spec.prompt_text,
                "task_text": prompt_spec.task_text,
            },
        )

    def _load_model_and_processor(self):
        if self._shared_model is not None or self._shared_processor is not None:
            if self._shared_model is None or self._shared_processor is None:
                raise ValueError("Both `shared_model` and `shared_processor` must be provided together.")
            return self._shared_model, self._shared_processor
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("transformers is required for reward scoring") from exc

        model_path = ModelStore(self.model_config).ensure_base_model()
        self._processor = load_qwen_2_5_vl_processor(model_path)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        return self._model, self._processor

    @contextmanager
    def _shared_model_inference_context(self, model: Any):
        """Temporarily enforce frozen deterministic inference for shared-model reward scoring."""

        with ExitStack() as stack:
            was_training = bool(getattr(model, "training", False))
            if was_training:
                model.eval()
                stack.callback(model.train, True)
            if self._disable_shared_adapter:
                disable_adapter = getattr(model, "disable_adapter", None)
                if callable(disable_adapter):
                    stack.enter_context(disable_adapter())
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("[shared-state] reward context_enter %s", _adapter_state_summary(model))
            yield
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("[shared-state] reward context_exit_pre_restore %s", _adapter_state_summary(model))
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("[shared-state] reward context_exit_post_restore %s", _adapter_state_summary(model))


def _adapter_state_summary(model: Any) -> dict[str, Any]:
    active_adapters = None
    active_adapters_attr = getattr(model, "active_adapters", None)
    if callable(active_adapters_attr):
        try:
            active_adapters = active_adapters_attr()
        except TypeError:
            active_adapters = str(active_adapters_attr)
    elif active_adapters_attr is not None:
        active_adapters = active_adapters_attr

    adapter_layers = 0
    disabled_adapter_layers = 0
    for module in model.modules():
        if hasattr(module, "_disable_adapters"):
            adapter_layers += 1
            if bool(getattr(module, "_disable_adapters", False)):
                disabled_adapter_layers += 1

    config = getattr(model, "config", None)
    return {
        "model_id": hex(id(model)),
        "type": type(model).__name__,
        "training": bool(getattr(model, "training", False)),
        "use_cache": getattr(config, "use_cache", None),
        "is_gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "active_adapter": getattr(model, "active_adapter", None),
        "active_adapters": active_adapters,
        "adapter_layers": adapter_layers,
        "disabled_adapter_layers": disabled_adapter_layers,
    }


def _extract_reward_value(text: str) -> float:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "reward" in payload:
        return float(payload["reward"])

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "reward" in payload:
            return float(payload["reward"])

    numeric_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if numeric_match:
        return float(numeric_match.group(0))
    raise ValueError(f"Could not parse a numeric reward from: {text!r}")
