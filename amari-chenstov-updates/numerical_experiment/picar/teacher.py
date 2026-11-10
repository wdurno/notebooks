"""Explicitly separated frozen Qwen roles and one-time recorded-data bootstrap.

The test fake is labeled and opt-in. Real role scoring uses the shared frozen
Qwen vocabulary head and role-specific frozen banks, never learner readouts.
Distillation consumes distributions recorded *before execution*, not new
teacher calls during online learning. Joint-NLL initialization is a separate
stage owned by the numerical interface/runner, not mixed into this KL fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from numerical_experiment.picar.models import (
    PiCarStudent, QwenContextEncoder, batch_digest, render_context, validate_batch,
)


@dataclass(frozen=True)
class TeacherDistribution:
    original_probabilities: Tensor
    executed_probabilities: Tensor
    reference_logits: Tensor
    probability_floor: float
    target_kind: str


def prepare_teacher_distribution(probabilities: Tensor, *, probability_floor: float,
                                 target_kind: str = "soft") -> TeacherDistribution:
    """Call before controls are executed; preserve both distribution versions."""
    from src.continual.picar_math import teacher_finite_logits

    if target_kind not in ("soft", "hard_engineered"):
        raise ValueError("target_kind must be soft or hard_engineered")
    original = probabilities.detach().clone()
    executed, logits = teacher_finite_logits(original, probability_floor)
    return TeacherDistribution(original, executed.detach().clone(), logits.detach().clone(),
                               probability_floor, target_kind)


class FrozenQwenRole:
    """Restricted next-token verbalizer, not calibrated teacher uncertainty.

    Labels must be distinct single tokens in the actual checkpoint tokenizer.
    The fixed extraction rule is the softmax over those vocabulary logits.
    Role prompts and token IDs are provenance and must be frozen at collection.
    This does not claim the uncalibrated evaluator is a validated task reward.
    """

    def __init__(self, encoder: QwenContextEncoder, *, role: str, prompt: str,
                 labels: Sequence[str]):
        if not isinstance(encoder, QwenContextEncoder):
            raise TypeError("frozen Qwen roles require a real QwenContextEncoder")
        if role not in ("teacher", "evaluator"):
            raise ValueError("frozen role must be teacher or evaluator")
        if encoder.processor is None or not prompt.strip() or len(labels) < 2:
            raise ValueError("frozen roles require a processor, fixed prompt and labels")
        encoded = [encoder.processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(row) != 1 for row in encoded) or len({row[0] for row in encoded}) != len(encoded):
            raise ValueError("verbalizer labels must map to distinct single tokenizer tokens")
        if any(p.requires_grad for p in encoder.qwen.lm_head.parameters()):
            raise ValueError("frozen role vocabulary readout must be frozen")
        self.encoder, self.role, self.prompt = encoder, role, prompt
        self.token_ids = tuple(row[0] for row in encoded)
        self.provenance = {"kind": "real_qwen_frozen_role", "role": role,
            "extraction": "restricted_next_token_softmax_v1", "prompt": prompt,
            "labels": list(labels), "token_ids": list(self.token_ids),
            "frame_extraction": "last_rgb_full_recorded_context_v2",
            "calibrated": False}

    @torch.no_grad()
    def probabilities(self, batch: Mapping[str, Any]) -> Tensor:
        # There is no inherited active adapter or shared generation/KV cache.
        with self.encoder._lock:
            inputs, (b, t) = self.encoder.prepare(batch, suffix="\n" + self.prompt, last_only=True)
            hidden = self.encoder.encode_processed(inputs, role=self.role)
            mask = inputs["attention_mask"].to(hidden.device)
            last = (torch.arange(mask.shape[1], device=hidden.device)[None] * mask).max(-1).values
            features = hidden[torch.arange(b * t, device=hidden.device), last]
            # Avoid allocating [B,T,S,vocabulary] just to score fixed labels.
            head = self.encoder.qwen.lm_head
            ids = torch.tensor(self.token_ids, device=head.weight.device)
            logits = F.linear(features.to(head.weight.dtype), head.weight[ids],
                              head.bias[ids] if head.bias is not None else None)
            return logits.float().softmax(-1).reshape(b, t, -1)[:, -1].detach()


class FrozenQwenTeacher(FrozenQwenRole):
    def __init__(self, encoder: QwenContextEncoder, *, primitive_names: Sequence[str]):
        if len(primitive_names) != 8 or len(set(primitive_names)) != 8:
            raise ValueError("teacher requires eight distinct declared primitive names")
        prompt = ("Select the next action for the visible task. Completed instructions are history, "
                  "not new commands. Reply with exactly one digit.\n" +
                  "\n".join(f"{i}: {name}" for i, name in enumerate(primitive_names)))
        super().__init__(encoder, role="teacher", prompt=prompt, labels=tuple(map(str, range(8))))

    def distribution(self, batch, *, probability_floor: float) -> TeacherDistribution:
        return prepare_teacher_distribution(self.probabilities(batch),
                                             probability_floor=probability_floor)


class FrozenQwenEvaluator(FrozenQwenRole):
    def __init__(self, encoder: QwenContextEncoder):
        super().__init__(encoder, role="evaluator", labels=("0", "1"),
            prompt="Assess task completion using only the visible image and recorded history. "
                   "Reply 1 if the task is complete, or 0 if it is not complete.")

    def evaluate(self, batch) -> Tensor:
        """Bounded soft visible-completion score; calibration remains required."""
        return self.probabilities(batch)[:, 1]


class DeterministicFakeTeacher:
    """TEST/SMOKE ONLY. No production path constructs or falls back to this."""

    def __init__(self, *, test_only: bool = False):
        if test_only is not True:
            raise ValueError("fake teacher requires explicit test_only=True")
        self.provenance = {"kind": "deterministic_test_fake", "real_qwen": False,
                           "schema": 1, "calibrated": False}

    def probabilities(self, batch) -> Tensor:
        b, t = validate_batch(batch, max_history=1024)
        rows = []
        for i in range(b):
            payload = render_context(batch, i, t - 1)
            image = batch["images"][i, :t].detach().cpu().contiguous()
            digest = hashlib.sha256(payload.encode() + image.view(torch.uint8).numpy().tobytes()).digest()
            rows.append(torch.tensor(list(digest[:8]), dtype=torch.float64).div(64).softmax(-1))
        return torch.stack(rows)

    def distribution(self, batch, *, probability_floor: float) -> TeacherDistribution:
        return prepare_teacher_distribution(self.probabilities(batch), probability_floor=probability_floor)


def _validate_demonstration(record: Mapping[str, Any], student: PiCarStudent):
    if record.get("domain") != "A":
        raise ValueError("one-time distillation accepts A-domain bootstrap records only")
    if record.get("target_kind") not in ("soft", "hard_engineered"):
        raise ValueError("record must label soft versus engineered hard demonstrations")
    b, _ = validate_batch(record["batch"], student.encoder.context_length)
    probabilities = record["executed_probabilities"]
    if probabilities.shape != (b, 8) or not probabilities.is_floating_point():
        raise ValueError("recorded executed_probabilities must be floating [B,8]")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0).any()):
        raise ValueError("recorded executed distribution must be finite and strictly positive")
    if not torch.allclose(probabilities.sum(-1), torch.ones(b).to(probabilities), atol=1e-6, rtol=1e-6):
        raise ValueError("recorded executed distribution must be normalized")


def distill_once(student: PiCarStudent, demonstrations: Sequence[Mapping[str, Any]], *,
                 steps: int, learning_rate: float = 1e-3) -> dict[str, Any]:
    """Bounded KL projection on recorded executed distributions, once per model.

    Each record has batch, executed_probabilities[B,8], domain='A', and
    target_kind='soft' or 'hard_engineered'. No online teacher query occurs.
    Adam has no implicit weight decay. The receipt persists in state_dict;
    the subsequent joint-NLL stage/initial Fisher is deliberately separate.
    """
    if student.distillation_receipt is not None:
        raise ValueError("one-time distillation has already completed")
    if type(steps) is not int or steps < 1 or not demonstrations:
        raise ValueError("distillation requires recorded demonstrations and positive steps")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for record in demonstrations:
        _validate_demonstration(record, student)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=0.)
    was_training = student.training
    losses = []
    student.train()
    try:
        for step in range(steps):
            record = demonstrations[step % len(demonstrations)]
            output = student(record["batch"])["action_mean"]
            # The eighth reference logit is fixed zero, not a learned gauge.
            logits = torch.cat((output, output.new_zeros(output.shape[0], 1)), -1)
            target = record["executed_probabilities"].detach().to(logits)
            loss = F.kl_div(logits.log_softmax(-1), target, reduction="batchmean")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite bootstrap distillation KL")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        student.train(was_training)
    receipt = {"stage": "one_time_recorded_teacher_kl", "steps": steps,
        "learning_rate": learning_rate, "batches": len(demonstrations),
        "target_kinds": sorted({r["target_kind"] for r in demonstrations}),
        "initial_kl": losses[0], "last_preupdate_kl": losses[-1],
        "history_digests": [batch_digest(r["batch"]) for r in demonstrations]}
    student.distillation_receipt = receipt
    student.mark_published()
    return receipt
