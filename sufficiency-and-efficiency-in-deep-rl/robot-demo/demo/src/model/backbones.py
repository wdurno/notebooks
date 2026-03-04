from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn as nn

from .action_space import ACTION_NAMES, normalized_action_name
from .config import ModelConfig
from .model_store import ModelStore
from .processor_loader import load_qwen_2_5_vl_processor
from .schemas import ModelObservation


CONTROL_SYSTEM_PROMPT = (
    "You are controlling a PiCar-V robot. "
    "Choose exactly one action from: "
    "drive-left, drive-right, drive-forward, drive-backward, "
    "look-left, look-right, look-up, look-forward. "
    "Return JSON with keys `action` and `say`."
)
SAY_ONLY_SYSTEM_PROMPT = (
    "You are assisting a PiCar-V robot operator. "
    "Do not emit tool calls, action names, or control commands. "
    "Reply with concise plain text only."
)


@dataclass(frozen=True)
class BackboneBatchOutput:
    """Normalized output contract shared by the fake and real backbones."""

    pooled_hidden_state: torch.Tensor
    agentic_action_names: list[str]
    generated_texts: list[str]
    vlm_loss: Optional[torch.Tensor] = None
    debug: list[dict[str, Any]] = field(default_factory=list)


class SupportsBackbone(Protocol):
    hidden_size: int

    def encode(
        self,
        observations: list[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
        allow_agentic_actions: bool = True,
    ) -> BackboneBatchOutput:
        ...


class FakeBackbone(nn.Module):
    """Small deterministic backbone used in unit tests.

    This avoids downloading real Qwen weights in fast tests while still
    exercising the model contract:

    - observation in
    - pooled hidden state out
    - agentic action choice out
    - optional VLM loss out
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        agentic_action_names: list[str] | None = None,
        generated_texts: list[str] | None = None,
        vlm_loss: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.adapter.weight)
        self.agentic_action_names = agentic_action_names or ["look-forward"]
        self.generated_texts = generated_texts or [""]
        self.vlm_loss_value = vlm_loss

    def encode(
        self,
        observations: list[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
        allow_agentic_actions: bool = True,
    ) -> BackboneBatchOutput:
        # Produce deterministic hidden states from simple image statistics so
        # unit tests can control outputs without a heavyweight model.
        hidden_rows = []
        action_names = []
        texts = []
        debug = []
        for idx, observation in enumerate(observations):
            image_tensor = _image_to_tensor(observation.image_rgb).float()
            feature_seed = float(image_tensor.mean().item()) if image_tensor.numel() else 0.0
            base = torch.full(
                (self.hidden_size,),
                feature_seed,
                dtype=torch.float32,
                device=self.adapter.weight.device,
            )
            hidden_rows.append(self.adapter(base))
            if allow_agentic_actions:
                action_names.append(self.agentic_action_names[min(idx, len(self.agentic_action_names) - 1)])
            else:
                action_names.append("look-forward")
            texts.append(self.generated_texts[min(idx, len(self.generated_texts) - 1)])
            debug.append({"step_index": observation.step_index})
        pooled = torch.stack(hidden_rows, dim=0)
        vlm_loss = None
        if compute_vlm_loss:
            vlm_loss = pooled.sum() * 0.0 + float(self.vlm_loss_value)
        return BackboneBatchOutput(
            pooled_hidden_state=pooled,
            agentic_action_names=action_names,
            generated_texts=texts,
            vlm_loss=vlm_loss,
            debug=debug,
        )


class QwenLoRABackbone(nn.Module):
    """Frozen Qwen backbone with LoRA adapters and lightweight inference helpers.

    The base model remains frozen. PEFT inserts LoRA adapters into the selected
    transformer modules, and the outer `PiCarActionModel` adds the separate
    8-way value head.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        model: nn.Module,
        processor: Any,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.hidden_size = int(getattr(model.config, "hidden_size"))

    @classmethod
    def from_config(cls, config: ModelConfig) -> "QwenLoRABackbone":
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers and peft are required to load the Qwen LoRA backbone"
            ) from exc

        # Resolve the base model under `demo/model/` and download it on demand
        # if policy allows.
        model_path = ModelStore(config).ensure_base_model()
        processor = load_qwen_2_5_vl_processor(model_path)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        # The underlying Qwen weights are frozen; only LoRA adapters and the
        # outer value head should receive gradients.
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        return cls(config=config, model=model, processor=processor)

    def encode(
        self,
        observations: list[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
        allow_agentic_actions: bool = True,
    ) -> BackboneBatchOutput:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to process RGB images for the VLM") from exc

        images = []
        prompts = []
        prepared_messages = []
        for observation in observations:
            images.append(Image.fromarray(observation.image_rgb))
            messages = self._build_chat_messages(observation, allow_agentic_actions=allow_agentic_actions)
            prepared_messages.append(messages)
            prompts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # The processor handles multimodal packing so the rest of the code can
        # work with plain RGB frames and message objects.
        processor_inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        model_inputs = {name: tensor.to(model_device) for name, tensor in processor_inputs.items()}
        # Request hidden states so the outer model can attach its own value head
        # to the fused representation instead of modifying the frozen backbone.
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # First-pass pooling strategy: use the final token representation after
        # multimodal fusion. This gives a fixed-width vector of size
        # `model.config.hidden_size`.
        pooled_hidden_state = outputs.hidden_states[-1][:, -1, :]

        # Generation is used only for the agentic branch / optional speech text.
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=64)
        prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for idx, prompt_length in enumerate(prompt_lengths):
            new_token_ids = generated_ids[idx, int(prompt_length) :]
            decoded.append(
                self.processor.batch_decode(
                    [new_token_ids],
                    skip_special_tokens=True,
                )[0]
            )
        if allow_agentic_actions:
            parsed = [_parse_action_and_text(text) for text in decoded]
        else:
            parsed = [("look-forward", text.strip()) for text in decoded]
        action_names = [item[0] for item in parsed]
        generated_texts = [item[1] for item in parsed]
        debug = [{"raw_generation": text} for text in decoded]
        vlm_loss = None
        if compute_vlm_loss and target_texts and any(text is not None for text in target_texts):
            vlm_loss = self._compute_supervised_vlm_loss(
                observations=observations,
                images=images,
                messages_list=prepared_messages,
                target_texts=target_texts,
                model_device=model_device,
            )
        return BackboneBatchOutput(
            pooled_hidden_state=pooled_hidden_state,
            agentic_action_names=action_names,
            generated_texts=generated_texts,
            vlm_loss=vlm_loss,
            debug=debug,
        )

    def _build_chat_messages(self, observation: ModelObservation, *, allow_agentic_actions: bool) -> list[dict[str, Any]]:
        system_prompt = CONTROL_SYSTEM_PROMPT if allow_agentic_actions else SAY_ONLY_SYSTEM_PROMPT
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        messages.extend(_copy_messages(observation.messages))
        return _ensure_image_placeholder(messages)

    def _compute_supervised_vlm_loss(
        self,
        *,
        observations: list[ModelObservation],
        images: list[Any],
        messages_list: list[list[dict[str, Any]]],
        target_texts: list[Optional[str]],
        model_device: torch.device,
    ) -> torch.Tensor:
        """Compute a prompt-masked causal loss aligned to the model inputs.

        For each sample, we tokenize:

        - the prompt alone
        - the prompt followed by the supervised target text

        The label tensor is initialized from the full input ids, then the prompt
        prefix is masked with `-100` so loss is incurred only on the supervised
        completion tokens. This matches the expected shape for decoder-style HF
        models and avoids the prompt/label length mismatch from the initial
        implementation.
        """

        per_sample_losses = []
        for observation, messages, image, target_text in zip(observations, messages_list, images, target_texts):
            if target_text is None or not target_text.strip():
                continue
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_messages = _copy_messages(messages)
            full_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text.strip()}],
                }
            )
            full_prompt = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
            )
            full_inputs = self.processor(
                text=[full_prompt],
                images=[image],
                return_tensors="pt",
            )
            full_inputs = {name: tensor.to(model_device) for name, tensor in full_inputs.items()}
            prompt_length = int(prompt_inputs["attention_mask"][0].sum().item())
            labels = full_inputs["input_ids"].clone()
            labels[:, :prompt_length] = -100
            outputs = self.model(
                **full_inputs,
                labels=labels,
                return_dict=True,
            )
            per_sample_losses.append(outputs.loss)

        if not per_sample_losses:
            parameter = next(self.model.parameters())
            return parameter.sum() * 0.0
        return torch.stack(per_sample_losses).mean()


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(messages))


def _ensure_image_placeholder(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = _copy_messages(messages)
    if copied and _message_has_image(copied[-1]):
        return copied
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            content = _normalize_content(copied[index].get("content"))
            copied[index]["content"] = [{"type": "image"}] + content
            return copied
    copied.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe the current scene."}],
        }
    )
    return copied


def _message_has_image(message: dict[str, Any]) -> bool:
    for item in _normalize_content(message.get("content")):
        if isinstance(item, dict) and item.get("type") == "image":
            return True
    return False


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    normalized = []
    for item in list(content):
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"type": "text", "text": str(item)})
    return normalized


def _parse_action_and_text(text: str) -> tuple[str, str]:
    # Prefer strict JSON output, but retain a forgiving fallback so early
    # prompting failures do not crash the whole interaction loop.
    candidate = (text or "").strip()
    if not candidate:
        return "look-forward", ""
    try:
        payload = json.loads(candidate)
        action_name = normalized_action_name(str(payload.get("action", "")))
        spoken_text = str(payload.get("say", "")).strip()
        return action_name, spoken_text
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                action_name = normalized_action_name(str(payload.get("action", "")))
                spoken_text = str(payload.get("say", "")).strip()
                return action_name, spoken_text
            except json.JSONDecodeError:
                pass
        action_name = _extract_action_name(candidate)
        return action_name, candidate


def _image_to_tensor(image_rgb: Any) -> torch.Tensor:
    # Unit tests often feed lists or tensors directly; normalize them here.
    if isinstance(image_rgb, torch.Tensor):
        return image_rgb.detach().clone()
    return torch.as_tensor(image_rgb)


def _extract_action_name(text: str, *, default_action: str = "look-forward") -> str:
    """Extract the most plausible action from loose model output.

    The parser prefers the last exact action mention instead of the first
    substring match. This avoids bias toward `drive-left` when a generation
    echoes the prompt, which lists all actions in fixed order.
    """

    candidate = (text or "").strip().lower().replace("_", "-")
    if not candidate:
        return default_action
    matches = []
    for action_name in ACTION_NAMES:
        pattern = rf"(?<![a-z0-9]){re.escape(action_name)}(?![a-z0-9])"
        matches.extend((match.start(), action_name) for match in re.finditer(pattern, candidate))
    if not matches:
        return default_action
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]
