"""KL-projection losses and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for KL-projection training") from exc
    return torch, F


torch, F = _require_torch()


@dataclass(frozen=True)
class KLProjectionStepResult:
    loss: float
    valid_steps: int


def masked_action_kl_loss(
    *,
    logits: "torch.Tensor",
    target_distributions: "torch.Tensor",
    mask: "torch.Tensor",
    epsilon: float = 1e-8,
) -> "torch.Tensor":
    """Mean KL(target || policy) over valid readout positions."""

    if logits.shape != target_distributions.shape:
        raise ValueError("logits and target_distributions must have matching shapes")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, steps, action_dim]")
    if mask.shape != logits.shape[:2]:
        raise ValueError("mask must have shape [batch, steps]")
    valid_mask = mask.to(device=logits.device, dtype=torch.bool)
    valid_count = valid_mask.sum()
    if int(valid_count.item()) < 1:
        raise ValueError("at least one valid readout position is required")
    targets = target_distributions.to(device=logits.device, dtype=logits.dtype).clamp_min(float(epsilon))
    targets = targets / targets.sum(dim=-1, keepdim=True)
    per_action = F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="none")
    per_step = per_action.sum(dim=-1)
    return per_step[valid_mask].mean()


def train_kl_projection_step(
    *,
    model: Any,
    optimizer: Any,
    visual_tokens: "torch.Tensor",
    previous_actions: "torch.Tensor",
    target_distributions: "torch.Tensor",
    conditioning: "torch.Tensor | None" = None,
    step_mask: "torch.Tensor | None" = None,
) -> KLProjectionStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        visual_tokens=visual_tokens,
        previous_actions=previous_actions,
        conditioning=conditioning,
        step_mask=step_mask,
    )
    mask = output.readout_mask if step_mask is None else step_mask
    loss = masked_action_kl_loss(logits=output.logits, target_distributions=target_distributions, mask=mask)
    loss.backward()
    optimizer.step()
    return KLProjectionStepResult(loss=float(loss.detach().cpu().item()), valid_steps=int(mask.to(dtype=torch.bool).sum().item()))
