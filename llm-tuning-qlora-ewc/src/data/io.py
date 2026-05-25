from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class JsonlExample:
    id: str
    task_family: str
    task_id: str
    split: str
    prompt: str
    target: str
    answer_key: str
    metadata: dict


def read_jsonl(path: str | Path) -> list[JsonlExample]:
    examples: list[JsonlExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(JsonlExample(**json.loads(line)))
    return examples


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def filter_split(examples: Iterable[JsonlExample], split: str) -> list[JsonlExample]:
    return [example for example in examples if example.split == split]


def take_n(examples: list[JsonlExample], n: int | None) -> list[JsonlExample]:
    if n is None or n < 0:
        return examples
    return examples[:n]
