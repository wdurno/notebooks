"""Visual-token encoding for phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image

from picar_kl.vlm.runtime import QwenRuntime, move_inputs_to_model


@dataclass(frozen=True)
class VisualTokenEncoding:
    """One image represented as a sequence of visual-token vectors."""

    tokens: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = np.asarray(self.tokens)
        if tokens.ndim != 2:
            raise ValueError(f"visual token encoding must be rank-2 [N, D], got shape {tokens.shape}")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError(f"visual token encoding must be non-empty, got shape {tokens.shape}")
        if not np.issubdtype(tokens.dtype, np.floating):
            raise ValueError(f"visual token encoding must be floating point, got {tokens.dtype}")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.tokens.shape)

    @property
    def dtype(self) -> str:
        return str(self.tokens.dtype)


class VisualTokenEncoder(Protocol):
    """Production encoders produce real visual-token sequences."""

    encoder_id: str

    def encode_image(self, image_rgb: Any) -> VisualTokenEncoding:
        ...


class QwenVisualTokenEncoder:
    """Extract Qwen projected visual-token embeddings for one image.

    The tensor source is `Qwen2_5_VLModel.get_image_features(...).pooler_output`.
    In Transformers 5.2 this name is slightly misleading: it is split into one
    `[num_visual_tokens, hidden_dim]` tensor per image, not mean-pooled to one
    vector. These are the embeddings Qwen scatters into the language sequence.
    """

    encoder_id = "qwen2.5-vl-get-image-features-pooler-output-v1"

    def __init__(self, runtime: QwenRuntime, *, output_dtype: str = "float16"):
        self.runtime = runtime
        self.output_dtype = str(output_dtype)

    def encode_image(self, image_rgb: Any) -> VisualTokenEncoding:
        model = self.runtime.model
        processor = self.runtime.processor
        pil_image = Image.fromarray(image_rgb) if not isinstance(image_rgb, Image.Image) else image_rgb
        processor_inputs = _processor_image_inputs(processor, pil_image)
        model_inputs = move_inputs_to_model(processor_inputs, model)
        pixel_values = model_inputs.get("pixel_values")
        image_grid_thw = model_inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("Qwen processor did not return `pixel_values` and `image_grid_thw`")

        qwen_model = getattr(model, "model", model)
        get_image_features = getattr(qwen_model, "get_image_features", None)
        if not callable(get_image_features):
            raise RuntimeError("Qwen model does not expose `get_image_features`")

        with self.runtime.inference_context():
            outputs = get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
        pooler_output = getattr(outputs, "pooler_output", None)
        if not isinstance(pooler_output, (tuple, list)) or not pooler_output:
            raise RuntimeError("Qwen image features did not return per-image visual token tensors")
        tokens = pooler_output[0].detach().float().cpu().numpy()
        if self.output_dtype:
            tokens = tokens.astype(self.output_dtype)
        return VisualTokenEncoding(
            tokens=tokens,
            metadata={
                "encoder_id": self.encoder_id,
                "source": "Qwen2_5_VLModel.get_image_features.pooler_output",
                "image_grid_thw": _to_python_list(image_grid_thw),
                "image_size": list(pil_image.size),
            },
        )


def _processor_image_inputs(processor: Any, image: Image.Image) -> Any:
    try:
        return processor(images=[image], return_tensors="pt")
    except TypeError:
        return processor(text=[""], images=[image], return_tensors="pt")


def _to_python_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
