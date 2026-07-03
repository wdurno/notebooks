"""Combined phase 2 KL-projection model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from picar_kl.actions import action_count
from picar_kl.models.lstm_policy import LSTMPolicyConfig, LSTMPolicyOutput, TokenStreamLSTMPolicy
from picar_kl.models.vlm_head import PrefixConditioningHead, PrefixConditioningHeadConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for the phase 2 KL model") from exc
    return torch, nn


@dataclass(frozen=True)
class Phase2KLModelConfig:
    visual_dim: int
    model_dim: int = 128
    conditioning_dim: int = 32
    conditioning_hidden_dim: int = 128
    token_type_dim: int = 8
    action_dim: int = action_count()
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    dropout: float = 0.0


class Phase2KLModelOutput(NamedTuple):
    conditioning: "torch.Tensor"
    policy: LSTMPolicyOutput

    @property
    def logits(self) -> "torch.Tensor":
        return self.policy.logits

    @property
    def readout_mask(self) -> "torch.Tensor":
        return self.policy.readout_mask


torch, nn = _require_torch()


class Phase2KLModel(nn.Module):
    """Jointly train prefix Dh conditioning and target-step LSTM policy."""

    def __init__(self, config: Phase2KLModelConfig):
        super().__init__()
        self.config = config
        self.conditioning_head = PrefixConditioningHead(
            PrefixConditioningHeadConfig(
                visual_dim=config.visual_dim,
                action_dim=config.action_dim,
                conditioning_dim=config.conditioning_dim,
                hidden_dim=config.conditioning_hidden_dim,
            )
        )
        self.policy = TokenStreamLSTMPolicy(
            LSTMPolicyConfig(
                visual_dim=config.visual_dim,
                model_dim=config.model_dim,
                conditioning_dim=config.conditioning_dim,
                token_type_dim=config.token_type_dim,
                action_dim=config.action_dim,
                lstm_hidden_dim=config.lstm_hidden_dim,
                lstm_layers=config.lstm_layers,
                dropout=config.dropout,
            )
        )

    def forward(
        self,
        *,
        prefix_visual_tokens: "torch.Tensor",
        prefix_actions: "torch.Tensor",
        target_visual_tokens: "torch.Tensor",
        target_previous_actions: "torch.Tensor",
        prefix_step_mask: "torch.Tensor | None" = None,
        target_step_mask: "torch.Tensor | None" = None,
    ) -> Phase2KLModelOutput:
        conditioning = self.conditioning_head(
            visual_tokens=prefix_visual_tokens,
            action_distributions=prefix_actions,
            step_mask=prefix_step_mask,
        )
        target_steps = target_visual_tokens.shape[1]
        target_conditioning = conditioning.unsqueeze(1).expand(-1, target_steps, -1)
        policy_output = self.policy(
            visual_tokens=target_visual_tokens,
            previous_actions=target_previous_actions,
            conditioning=target_conditioning,
            step_mask=target_step_mask,
        )
        return Phase2KLModelOutput(conditioning=conditioning, policy=policy_output)
