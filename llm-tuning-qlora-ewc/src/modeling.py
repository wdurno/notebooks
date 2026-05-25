from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    name: str
    cache_dir: str = "data/models"
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    attn_implementation: str | None = "flash_attention_2"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class LoraConfigValues:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05


def load_tokenizer(model_name: str, cache_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _attn_implementation(model_config: ModelConfig) -> str | None:
    if model_config.attn_implementation != "flash_attention_2":
        return model_config.attn_implementation
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


def _device_map() -> dict[str, int] | None:
    if torch.cuda.is_available():
        return {"": 0}
    return None


def load_qlora_model(model_config: ModelConfig, lora_config: LoraConfigValues):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    kwargs: dict[str, Any] = {
        "cache_dir": model_config.cache_dir,
        "device_map": _device_map(),
        "trust_remote_code": model_config.trust_remote_code,
        "quantization_config": quantization_config,
    }
    attn_implementation = _attn_implementation(model_config)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_config.name, **kwargs)
    except Exception:
        if "attn_implementation" in kwargs:
            kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(model_config.name, **kwargs)
        else:
            raise

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, peft_config)


def load_base_model(model_config: ModelConfig):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    kwargs: dict[str, Any] = {
        "cache_dir": model_config.cache_dir,
        "device_map": _device_map(),
        "trust_remote_code": model_config.trust_remote_code,
        "quantization_config": quantization_config,
    }
    attn_implementation = _attn_implementation(model_config)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(model_config.name, **kwargs)
    except Exception:
        if "attn_implementation" in kwargs:
            kwargs.pop("attn_implementation")
            return AutoModelForCausalLM.from_pretrained(model_config.name, **kwargs)
        raise


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Instruction:\n{prompt}\n\nAnswer:\n"


def format_training_text(tokenizer, prompt: str, target: str) -> tuple[str, str]:
    prompt_text = format_prompt(tokenizer, prompt)
    return prompt_text, f"{prompt_text}{target}"


def get_primary_device(model) -> torch.device:
    return next(model.parameters()).device
