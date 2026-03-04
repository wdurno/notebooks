from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Qwen2_5_VLImageOnlyProcessor:
    """Small image-only processor wrapper for Qwen 2.5-VL.

    This bypasses the upstream processor construction path, which currently
    fails in the installed `transformers` build due to an unnecessary
    video-processor dependency. The wrapper implements only the methods used by
    this repo:

    - `apply_chat_template`
    - `__call__`
    - `batch_decode`
    """

    image_processor: Any
    tokenizer: Any

    def __post_init__(self):
        self.image_token = "<|image_pad|>" if not hasattr(self.tokenizer, "image_token") else self.tokenizer.image_token
        self.image_token_id = (
            self.tokenizer.image_token_id
            if getattr(self.tokenizer, "image_token_id", None)
            else self.tokenizer.convert_tokens_to_ids(self.image_token)
        )

    def apply_chat_template(self, messages, *, tokenize: bool = False, add_generation_prompt: bool = True):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, *, text, images, return_tensors: str = "pt", padding: bool = False):
        image_features = self.image_processor(
            images=images,
            return_tensors=return_tensors,
        )
        image_grid_thw = image_features["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_image_tokens = image_grid_thw[index].prod().item() // merge_length
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * int(num_image_tokens), 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_features = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
        )
        payload = dict(text_features)
        payload.update(dict(image_features))
        return payload

    def batch_decode(self, sequences, *, skip_special_tokens: bool = True):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)


def load_qwen_2_5_vl_processor(model_path: str | Path):
    """Build an image-only Qwen 2.5-VL processor from independent components."""

    try:
        from transformers import AutoTokenizer, Qwen2VLImageProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the Qwen 2.5-VL processor") from exc

    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return Qwen2_5_VLImageOnlyProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
