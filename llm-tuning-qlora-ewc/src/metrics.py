from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def normalize_text(text: str, *, lowercase: bool = True) -> str:
    text = " ".join(text.strip().split())
    return text.lower() if lowercase else text


def exact_match(prediction: str, target: str, *, lowercase: bool = True) -> float:
    return float(normalize_text(prediction, lowercase=lowercase) == normalize_text(target, lowercase=lowercase))


def token_f1(prediction: str, target: str, *, lowercase: bool = True) -> float:
    pred_tokens = normalize_text(prediction, lowercase=lowercase).split()
    target_tokens = normalize_text(target, lowercase=lowercase).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum((pred_counts & target_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def format_validity(prediction: str, validator: Mapping[str, object] | None) -> float:
    if not validator:
        return 1.0
    kind = validator.get("type")
    if kind == "choice":
        return float(prediction.strip() in set(validator.get("choices", [])))
    if kind == "regex":
        pattern = str(validator["pattern"])
        return float(re.fullmatch(pattern, prediction.strip()) is not None)
    if kind == "json":
        try:
            json.loads(prediction)
        except json.JSONDecodeError:
            return 0.0
        return 1.0
    raise ValueError(f"Unknown validator type: {kind}")


def extract_constrained_answer(prediction: str, validator: Mapping[str, object] | None) -> str:
    prediction = prediction.strip()
    if not validator:
        return prediction
    kind = validator.get("type")
    if kind == "choice":
        choices = [str(choice) for choice in validator.get("choices", [])]
        for choice in choices:
            if prediction == choice or prediction.startswith(choice):
                return choice
        return prediction
    if kind == "regex":
        pattern = str(validator["pattern"])
        match = re.search(pattern, prediction)
        return match.group(0) if match else prediction
    return prediction


@dataclass(frozen=True)
class RetentionScores:
    score_before: float
    score_after: float
    epsilon: float = 1e-12

    @property
    def forgetting_delta(self) -> float:
        return self.score_before - self.score_after

    @property
    def retention_ratio(self) -> float:
        return self.score_after / max(self.score_before, self.epsilon)


def average_retention(retention_ratios: Sequence[float]) -> float:
    if not retention_ratios:
        return 1.0
    return sum(retention_ratios) / len(retention_ratios)


def worst_task_retention(retention_ratios: Iterable[float]) -> float:
    ratios = list(retention_ratios)
    if not ratios:
        return 1.0
    return min(ratios)
