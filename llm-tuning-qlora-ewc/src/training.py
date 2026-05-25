from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.io import JsonlExample
from src.metrics import exact_match, extract_constrained_answer, format_validity, token_f1
from src.modeling import format_prompt, format_training_text, get_primary_device


@dataclass(frozen=True)
class TrainingConfig:
    max_seq_length: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    epochs: int = 2


class SupervisedDataset(Dataset):
    def __init__(self, examples: list[JsonlExample], tokenizer, max_seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        prompt_text, full_text = format_training_text(self.tokenizer, example.prompt, example.target)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(tokenizer) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in items)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = tokenizer.pad_token_id
        for item in items:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    return collate


def train_supervised(
    model,
    tokenizer,
    examples: list[JsonlExample],
    config: TrainingConfig,
    ewc_penalty: Callable[[], torch.Tensor] | None = None,
    ewc_lambda: float = 0.0,
) -> dict[str, float]:
    model.train()
    device = get_primary_device(model)
    dataset = SupervisedDataset(examples, tokenizer, config.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_batch(tokenizer),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    start = time.time()
    total_loss = 0.0
    steps = 0

    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(config.epochs):
        for batch_idx, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if ewc_penalty is not None and ewc_lambda > 0:
                loss = loss + ewc_lambda * ewc_penalty()
            (loss / config.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach().cpu())
            steps += 1
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - start
    return {
        "train_loss": total_loss / max(steps, 1),
        "train_wall_seconds": elapsed,
        "train_steps": float(steps),
    }


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    model.eval()
    device = get_primary_device(model)
    prompt_text = format_prompt(tokenizer, prompt)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_examples(model, tokenizer, examples: Iterable[JsonlExample], max_new_tokens: int = 32) -> dict[str, float]:
    rows = evaluate_examples_with_predictions(model, tokenizer, examples, max_new_tokens=max_new_tokens)
    if not rows:
        return {"exact_match": 0.0, "token_f1": 0.0, "format_validity": 0.0}
    return {
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
        "format_validity": sum(row["format_validity"] for row in rows) / len(rows),
    }


def evaluate_examples_with_predictions(
    model,
    tokenizer,
    examples: Iterable[JsonlExample],
    max_new_tokens: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        prediction = generate_answer(model, tokenizer, example.prompt, max_new_tokens=max_new_tokens)
        validator = example.metadata.get("validator") if example.metadata else None
        parsed_prediction = extract_constrained_answer(prediction, validator)
        rows.append(
            {
                "id": example.id,
                "prompt": example.prompt,
                "target": example.target,
                "metadata": example.metadata,
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "exact_match": exact_match(parsed_prediction, example.target),
                "token_f1": token_f1(parsed_prediction, example.target),
                "format_validity": format_validity(parsed_prediction, validator),
            }
        )
    return rows
