"""Shared Qwen2.5-VL runtime."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from picar_kl.models.store import ModelStore


class QwenRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenRuntimeConfig:
    model_name: str = "qwen2.5-vl-3b"
    model_root: Path = Path("artifacts/models")
    manifest_path: Path = Path("artifacts/manifests/tracked/models/vlm_models.json")
    allow_downloads: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"


class QwenRuntime:
    """Own one Qwen model+processor pair for shared VLM work."""

    def __init__(self, *, model: Any, processor: Any, config: QwenRuntimeConfig | None = None):
        self.model = model
        self.processor = processor
        self.config = config or QwenRuntimeConfig()

    @classmethod
    def from_config(cls, config: QwenRuntimeConfig | None = None) -> "QwenRuntime":
        config = config or QwenRuntimeConfig()
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
            raise QwenRuntimeError("torch and transformers are required for Qwen2.5-VL") from exc

        dtype = resolve_torch_dtype(torch, config.torch_dtype)
        processor = load_processor(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=config.device_map,
        )
        model.eval()
        return cls(model=model, processor=processor, config=config)

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int | None:
        try:
            template_tokens = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            return None
        return count_input_ids(template_tokens)

    def generate_completion(
        self,
        *,
        image_rgb: Any,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None = None,
        top_p: float | None = None,
        use_base_model: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        image = Image.fromarray(image_rgb) if not isinstance(image_rgb, Image.Image) else image_rgb
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = move_inputs_to_model(inputs, self.model)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            if temperature is not None:
                generation_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                generation_kwargs["top_p"] = float(top_p)
        context = self.base_inference_context() if use_base_model else self.inference_context()
        with context:
            outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = prompt_length_from_inputs(inputs)
        completion_ids = outputs[0][prompt_length:]
        text = self.processor.batch_decode([completion_ids], skip_special_tokens=True)[0].strip()
        return text, {"prompt_tokens": prompt_length}

    @contextmanager
    def inference_context(self):
        with ExitStack() as stack:
            stack.enter_context(_torch_inference_mode())
            was_training = bool(getattr(self.model, "training", False))
            if was_training and hasattr(self.model, "eval"):
                self.model.eval()
                stack.callback(self.model.train, True)
            yield

    @contextmanager
    def base_inference_context(self):
        with self.inference_context():
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if callable(disable_adapter):
                with disable_adapter():
                    yield
            else:
                yield


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise QwenRuntimeError(f"Unknown torch dtype: {dtype_name}") from exc


def load_processor(model_path: Path) -> Any:
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_path)
    except Exception:
        try:
            from picar_kl.vlm.processor_loader import load_qwen_2_5_vl_processor

            return load_qwen_2_5_vl_processor(model_path)
        except Exception as exc:
            raise QwenRuntimeError("Unable to load Qwen2.5-VL processor") from exc


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    try:
        parameter = next(model.parameters())
        device = parameter.device
    except Exception:
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()}


def prompt_length_from_inputs(inputs: Any) -> int:
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attention_mask is None:
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])
    return int(attention_mask[0].sum().item())


def count_input_ids(payload: Any) -> int | None:
    if hasattr(payload, "numel"):
        return int(payload.numel())
    if isinstance(payload, (list, tuple)):
        if not payload:
            return 0
        first = payload[0]
        if hasattr(first, "numel"):
            return int(first.numel())
        if isinstance(first, (list, tuple)):
            return int(len(first))
        return int(len(payload))
    getter = getattr(payload, "get", None)
    if callable(getter):
        return count_input_ids(getter("input_ids"))
    data = getattr(payload, "data", None)
    if isinstance(data, dict):
        return count_input_ids(data.get("input_ids"))
    return None


@contextmanager
def _torch_inference_mode():
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.inference_mode():
        yield
