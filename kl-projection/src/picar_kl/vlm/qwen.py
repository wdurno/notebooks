"""Qwen2.5-VL phase 1 controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picar_kl.vlm.control import VLMDecision, parse_action_response
from picar_kl.vlm.runtime import QwenRuntime, QwenRuntimeConfig


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
        runtime: QwenRuntime | None = None,
    ):
        self.config = config or QwenPhase1Config()
        self.runtime = runtime or QwenRuntime(
            model=model,
            processor=processor,
            config=_runtime_config_from_phase1(self.config),
        )
        self.model = self.runtime.model
        self.processor = self.runtime.processor

    @classmethod
    def from_config(cls, config: QwenPhase1Config | None = None) -> "QwenPhase1Controller":
        config = config or QwenPhase1Config()
        runtime = QwenRuntime.from_config(_runtime_config_from_phase1(config))
        return cls(model=runtime.model, processor=runtime.processor, config=config, runtime=runtime)

    def decide(self, *, image_rgb: Any, messages: list[dict[str, Any]]) -> VLMDecision:
        qwen_messages = _attach_latest_image(messages)
        text, runtime_metadata = self.runtime.generate_completion(
            image_rgb=image_rgb,
            messages=qwen_messages,
            max_new_tokens=int(self.config.max_new_tokens),
            do_sample=bool(self.config.do_sample),
            temperature=float(self.config.temperature),
            top_p=float(self.config.top_p),
        )
        decision = parse_action_response(text)
        return VLMDecision.from_distribution(
            decision.action_distribution,
            generated_text=decision.generated_text,
            raw_response=text,
            metadata={"controller": "qwen2.5-vl", **runtime_metadata, **decision.metadata},
        )


def _runtime_config_from_phase1(config: QwenPhase1Config) -> QwenRuntimeConfig:
    return QwenRuntimeConfig(
        model_name=config.model_name,
        model_root=config.model_root,
        manifest_path=config.manifest_path,
        allow_downloads=config.allow_downloads,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
    )


def _attach_latest_image(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
