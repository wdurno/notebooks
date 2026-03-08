from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import ACTION_NAMES, normalized_action_name
from .config import ModelConfig
from .model_store import ModelStore
from .processor_loader import load_qwen_2_5_vl_processor
from .schemas import ModelObservation

LOGGER = logging.getLogger(__name__)


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
    generated_logp_sums: Optional[torch.Tensor] = None
    target_logp_sums: Optional[torch.Tensor] = None
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
            debug.append({"step_index": observation.step_index, "generated_logp_sum": 0.0})
        pooled = torch.stack(hidden_rows, dim=0)
        zero_vector = pooled.sum(dim=1) * 0.0
        vlm_loss = None
        if compute_vlm_loss:
            vlm_loss = pooled.sum() * 0.0 + float(self.vlm_loss_value)
        return BackboneBatchOutput(
            pooled_hidden_state=pooled,
            agentic_action_names=action_names,
            generated_texts=texts,
            vlm_loss=vlm_loss,
            generated_logp_sums=zero_vector,
            target_logp_sums=zero_vector,
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
        self.hidden_size = _resolve_model_hidden_size(model)
        self._diagnostic_logged_once = False

    @classmethod
    def from_config(cls, config: ModelConfig) -> "QwenLoRABackbone":
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers, peft, and bitsandbytes are required to load the Qwen QLoRA backbone"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA 4-bit loading requires CUDA-capable hardware.")

        bnb_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )

        # Resolve the base model under `demo/model/` and download it on demand
        # if policy allows.
        model_path = ModelStore(config).ensure_base_model()
        processor = load_qwen_2_5_vl_processor(model_path)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=bnb_compute_dtype,
            device_map="auto",
            quantization_config=quantization_config,
        )
        base_model = prepare_model_for_kbit_training(base_model)
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

        per_sample_images = []
        prompts = []
        prepared_messages = []
        for observation in observations:
            latest_image = Image.fromarray(observation.image_rgb)
            messages = self._build_chat_messages(observation, allow_agentic_actions=allow_agentic_actions)
            per_sample_images.append(_build_images_for_messages(messages, latest_image))
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
            images=_prepare_batch_images_for_processor(per_sample_images),
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        model_inputs = {name: tensor.to(model_device) for name, tensor in processor_inputs.items()}
        if LOGGER.isEnabledFor(logging.DEBUG):
            first_step = int(getattr(observations[0], "step_index", -1)) if observations else -1
            LOGGER.debug(
                "[shared-state] policy before_forward step=%d %s",
                first_step,
                _adapter_state_summary(self.model),
            )
        # Request hidden states so the outer model can attach its own value head
        # to the fused representation instead of modifying the frozen backbone.
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # First-pass pooling strategy: use the final token representation after
        # multimodal fusion. This gives a fixed-width vector of size
        # equal to the language hidden state size.
        pooled_hidden_state = outputs.hidden_states[-1][:, -1, :]

        # Use stochastic decoding so collected text actions are sampled from the
        # behavior policy instead of deterministic argmax decoding.
        do_sample = not bool(getattr(self.config, "deterministic_coding", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.generation_max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(self.config.generation_temperature)
            generation_kwargs["top_p"] = float(self.config.generation_top_p)
            top_k = int(self.config.generation_top_k)
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
        if LOGGER.isEnabledFor(logging.DEBUG):
            first_step = int(getattr(observations[0], "step_index", -1)) if observations else -1
            LOGGER.debug(
                "[shared-state] policy before_generate step=%d %s",
                first_step,
                _adapter_state_summary(self.model),
            )
        generation_output = self.model.generate(
            **model_inputs,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = generation_output.sequences
        prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
        self._log_generation_diagnostics_once(
            observations=observations,
            prompts=prompts,
            model_inputs=model_inputs,
            generated_ids=generated_ids,
            prompt_lengths=prompt_lengths,
            generation_kwargs=generation_kwargs,
        )
        generated_logp_sums = _compute_generate_log_prob_sums(
            sequences=generated_ids,
            prompt_lengths=prompt_lengths,
            scores=list(generation_output.scores),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
        )
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
            parsed = [_parse_action_and_text_with_status(text) for text in decoded]
        else:
            parsed = [("look-forward", text.strip(), True) for text in decoded]
        action_names = [item[0] for item in parsed]
        generated_texts = [item[1] for item in parsed]
        json_valid_flags = [bool(item[2]) for item in parsed]
        debug = []
        for idx, text in enumerate(decoded):
            debug.append(
                {
                    "raw_generation": text,
                    "generated_logp_sum": float(generated_logp_sums[idx].detach().cpu().item()),
                    "json_expected": bool(allow_agentic_actions),
                    "json_valid": bool(json_valid_flags[idx]) if allow_agentic_actions else True,
                }
            )
        vlm_loss = None
        target_logp_sums = pooled_hidden_state.sum(dim=1) * 0.0
        if target_texts and any(text is not None and text.strip() for text in target_texts):
            vlm_loss_value, target_logp_sums = self._compute_supervised_token_stats(
                observations=observations,
                per_sample_images=per_sample_images,
                messages_list=prepared_messages,
                target_texts=target_texts,
                model_device=model_device,
            )
            if compute_vlm_loss:
                vlm_loss = vlm_loss_value
        elif compute_vlm_loss:
            parameter = next(self.model.parameters())
            vlm_loss = parameter.sum() * 0.0
        return BackboneBatchOutput(
            pooled_hidden_state=pooled_hidden_state,
            agentic_action_names=action_names,
            generated_texts=generated_texts,
            vlm_loss=vlm_loss,
            generated_logp_sums=generated_logp_sums,
            target_logp_sums=target_logp_sums,
            debug=debug,
        )

    def _log_generation_diagnostics_once(
        self,
        *,
        observations: list[ModelObservation],
        prompts: list[str],
        model_inputs: dict[str, torch.Tensor],
        generated_ids: torch.Tensor,
        prompt_lengths: list[float],
        generation_kwargs: dict[str, Any],
    ) -> None:
        """Emit one detailed generation diagnostic record for prompt/debug triage."""

        if self._diagnostic_logged_once:
            return None
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return None
        if not observations:
            return None

        first_idx = 0
        step_index = int(getattr(observations[first_idx], "step_index", -1))
        prompt = prompts[first_idx] if first_idx < len(prompts) else ""
        prompt_preview = prompt if len(prompt) <= 800 else f"{prompt[:800]}...[truncated]"
        prompt_length = int(prompt_lengths[first_idx]) if first_idx < len(prompt_lengths) else 0
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        input_shape = tuple(input_ids.shape) if input_ids is not None else None
        attention_shape = tuple(attention_mask.shape) if attention_mask is not None else None
        sequence = generated_ids[first_idx]
        completion_ids = sequence[prompt_length:]
        first_new_token_ids = completion_ids[:16].detach().to("cpu").tolist()
        LOGGER.debug(
            "[diag] step=%d do_sample=%s generation_kwargs=%s input_shape=%s attention_shape=%s prompt_tokens=%d first_new_token_ids=%s prompt_preview=%r",
            step_index,
            bool(generation_kwargs.get("do_sample", False)),
            {key: generation_kwargs[key] for key in sorted(generation_kwargs)},
            input_shape,
            attention_shape,
            prompt_length,
            first_new_token_ids,
            prompt_preview,
        )
        self._diagnostic_logged_once = True
        return None

    def _build_chat_messages(self, observation: ModelObservation, *, allow_agentic_actions: bool) -> list[dict[str, Any]]:
        system_prompt = CONTROL_SYSTEM_PROMPT if allow_agentic_actions else SAY_ONLY_SYSTEM_PROMPT
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        messages.extend(_copy_messages(observation.messages))
        messages = _ensure_image_placeholder(messages)
        if bool(getattr(self.config, "all_images", False)):
            messages = _ensure_image_placeholders_for_all_user_messages(messages)
        return messages

    def _compute_supervised_token_stats(
        self,
        *,
        observations: list[ModelObservation],
        per_sample_images: list[list[Any]],
        messages_list: list[list[dict[str, Any]]],
        target_texts: list[Optional[str]],
        model_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked causal loss and target-sequence log-prob sums.

        For each sample, we tokenize:

        - the prompt alone
        - the prompt followed by the supervised target text

        The label tensor is initialized from the full input ids, then the prompt
        prefix is masked with `-100` so loss is incurred only on the supervised
        completion tokens. This matches the expected shape for decoder-style HF
        models and avoids the prompt/label length mismatch from the initial
        implementation.
        """

        per_sample_losses: list[torch.Tensor] = []
        per_sample_logp_sums: list[torch.Tensor] = []
        parameter = next(self.model.parameters())
        zero = parameter.sum() * 0.0
        for observation, messages, sample_images, target_text in zip(
            observations, messages_list, per_sample_images, target_texts
        ):
            if target_text is None or not target_text.strip():
                per_sample_logp_sums.append(zero)
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
                images=_prepare_single_images_for_processor(sample_images),
                return_tensors="pt",
            )
            full_inputs = self.processor(
                text=[full_prompt],
                images=_prepare_single_images_for_processor(sample_images),
                return_tensors="pt",
            )
            full_inputs = {name: tensor.to(model_device) for name, tensor in full_inputs.items()}
            prompt_length = int(prompt_inputs["attention_mask"][0].sum().item())
            labels = full_inputs["input_ids"].clone()
            labels[:, :prompt_length] = -100
            outputs = self.model(
                **full_inputs,
                return_dict=True,
            )
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            token_mask = shift_labels.ne(-100)
            if not bool(token_mask.any().item()):
                per_sample_logp_sums.append(zero)
                continue
            safe_labels = shift_labels.masked_fill(~token_mask, 0)
            token_log_probs = F.log_softmax(shift_logits, dim=-1)
            selected_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            valid_log_probs = selected_log_probs[token_mask]
            per_sample_logp_sums.append(valid_log_probs.sum())
            per_sample_losses.append(-valid_log_probs.mean())

        if not per_sample_losses:
            vlm_loss = zero
        else:
            vlm_loss = torch.stack(per_sample_losses).mean()
        if not per_sample_logp_sums:
            per_sample_logp_sums = [zero for _ in observations]
        logp_sums = torch.stack([value.reshape(()) for value in per_sample_logp_sums], dim=0)
        return vlm_loss, logp_sums


def _compute_generate_log_prob_sums(
    *,
    sequences: torch.Tensor,
    prompt_lengths: list[int],
    scores: list[torch.Tensor],
    eos_token_id: int | list[int] | None,
) -> torch.Tensor:
    batch_size = int(sequences.shape[0])
    if not scores:
        return torch.zeros(batch_size, dtype=torch.float32, device=sequences.device)
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(int(eos_token_id))
    elif isinstance(eos_token_id, list):
        eos_ids.update(int(token_id) for token_id in eos_token_id)
    finished = [False] * batch_size
    logp_sums = torch.zeros(batch_size, dtype=torch.float32, device=sequences.device)
    for step_idx, step_scores in enumerate(scores):
        step_log_probs = F.log_softmax(step_scores.float(), dim=-1)
        for batch_idx in range(batch_size):
            if finished[batch_idx]:
                continue
            token_position = int(prompt_lengths[batch_idx]) + step_idx
            if token_position >= int(sequences.shape[1]):
                finished[batch_idx] = True
                continue
            token_id = int(sequences[batch_idx, token_position].detach().cpu().item())
            logp_sums[batch_idx] += step_log_probs[batch_idx, token_id]
            if token_id in eos_ids:
                finished[batch_idx] = True
    return logp_sums


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(messages))


def _resolve_model_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Backbone model must expose a `config` object")

    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
        getattr(getattr(config, "text_config", None), "d_model", None),
        getattr(getattr(config, "language_config", None), "d_model", None),
    ]
    for value in candidates:
        if value is None:
            continue
        hidden_size = int(value)
        if hidden_size > 0:
            return hidden_size

    raise ValueError(
        "Unable to resolve language hidden size from model config. "
        "Expected one of: config.hidden_size, config.text_config.hidden_size, "
        "config.language_config.hidden_size."
    )


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


def _ensure_image_placeholders_for_all_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = _copy_messages(messages)
    for message in copied:
        if message.get("role") != "user":
            continue
        content = _normalize_content(message.get("content"))
        if any(isinstance(item, dict) and item.get("type") == "image" for item in content):
            message["content"] = content
            continue
        message["content"] = [{"type": "image"}] + content
    return copied


def _build_images_for_messages(messages: list[dict[str, Any]], latest_image: Any) -> list[Any]:
    image_count = 0
    for message in messages:
        for item in _normalize_content(message.get("content")):
            if isinstance(item, dict) and item.get("type") == "image":
                image_count += 1
    return [latest_image] * max(1, image_count)


def _prepare_batch_images_for_processor(per_sample_images: list[list[Any]]) -> list[Any] | list[list[Any]]:
    if len(per_sample_images) == 1:
        return per_sample_images[0]
    if all(len(sample_images) == 1 for sample_images in per_sample_images):
        return [sample_images[0] for sample_images in per_sample_images]
    return per_sample_images


def _prepare_single_images_for_processor(sample_images: list[Any]) -> list[Any]:
    return list(sample_images)


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
    action_name, spoken_text, _ = _parse_action_and_text_with_status(text)
    return action_name, spoken_text


def _parse_action_and_text_with_status(text: str) -> tuple[str, str, bool]:
    # Prefer strict JSON output, but retain a forgiving fallback so early
    # prompting failures do not crash the whole interaction loop.
    candidate = (text or "").strip()
    if not candidate:
        return "look-forward", "", False
    try:
        payload = json.loads(candidate)
        action_name = normalized_action_name(str(payload.get("action", "")))
        spoken_text = str(payload.get("say", "")).strip()
        return action_name, spoken_text, True
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                action_name = normalized_action_name(str(payload.get("action", "")))
                spoken_text = str(payload.get("say", "")).strip()
                return action_name, spoken_text, True
            except json.JSONDecodeError:
                pass
        action_name = _extract_action_name(candidate)
        # Suppress malformed fallback text for speech output. The environment
        # can still inspect the debug metadata and apply penalties.
        return action_name, "", False


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
