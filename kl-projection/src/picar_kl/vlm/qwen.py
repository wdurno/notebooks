"""Qwen2.5-VL phase 1 controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from picar_kl.models.store import ModelStore
from picar_kl.vlm.control import VLMDecision, parse_action_response


class QwenControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenPhase1Config:
    model_name: str = "qwen2.5-vl-3b"
    model_root: Path = Path("artifacts/models")
    manifest_path: Path = Path("artifacts/manifests/tracked/models/vlm_models.json")
    allow_downloads: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


class QwenPhase1Controller:
    """Generate sparse phase 1 actions with Qwen2.5-VL."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        config: QwenPhase1Config | None = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config or QwenPhase1Config()

    @classmethod
    def from_config(cls, config: QwenPhase1Config | None = None) -> "QwenPhase1Controller":
        config = config or QwenPhase1Config()
        store = ModelStore(
            model_root=config.model_root,
            manifest_path=config.manifest_path,
            allow_downloads=config.allow_downloads,
        )
        model_path = store.ensure_vlm_model(config.model_name)
        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise QwenControllerError(
                "torch and transformers are required for Qwen2.5-VL phase 1 control"
            ) from exc

        dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        processor = _load_processor(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=config.device_map,
        )
        model.eval()
        return cls(model=model, processor=processor, config=config)

    def decide(self, *, image_rgb: Any, messages: list[dict[str, Any]]) -> VLMDecision:
        image = Image.fromarray(image_rgb) if not isinstance(image_rgb, Image.Image) else image_rgb
        qwen_messages = _attach_latest_image(messages, image)
        prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = _move_inputs_to_model(inputs, self.model)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": bool(self.config.do_sample),
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = float(self.config.temperature)
            generation_kwargs["top_p"] = float(self.config.top_p)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = _prompt_length(inputs)
        completion_ids = outputs[0][prompt_length:]
        text = self.processor.batch_decode([completion_ids], skip_special_tokens=True)[0]
        decision = parse_action_response(text)
        return VLMDecision.from_distribution(
            decision.action_distribution,
            generated_text=decision.generated_text,
            raw_response=text,
            metadata={"controller": "qwen2.5-vl", **decision.metadata},
        )


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise QwenControllerError(f"Unknown torch dtype: {dtype_name}") from exc


def _load_processor(model_path: Path) -> Any:
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_path)
    except Exception:
        try:
            from picar_kl.vlm.processor_loader import load_qwen_2_5_vl_processor

            return load_qwen_2_5_vl_processor(model_path)
        except Exception as exc:
            raise QwenControllerError("Unable to load Qwen2.5-VL processor") from exc


def _attach_latest_image(messages: list[dict[str, Any]], image: Image.Image) -> list[dict[str, Any]]:
    del image
    copied = []
    latest_user_index = None
    for idx, message in enumerate(messages):
        cloned = {"role": message.get("role"), "content": list(message.get("content") or [])}
        copied.append(cloned)
        if cloned["role"] == "user":
            latest_user_index = idx
    if latest_user_index is None:
        copied.append({"role": "user", "content": [{"type": "image"}]})
        return copied
    content = list(copied[latest_user_index].get("content") or [])
    if not any(isinstance(item, dict) and item.get("type") == "image" for item in content):
        content.insert(0, {"type": "image"})
    copied[latest_user_index]["content"] = content
    return copied


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    try:
        parameter = next(model.parameters())
        device = parameter.device
    except Exception:
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()}


def _prompt_length(inputs: Any) -> int:
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attention_mask is None:
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])
    return int(attention_mask[0].sum().item())
