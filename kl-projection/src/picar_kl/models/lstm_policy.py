"""Phase 2 token-stream LSTM policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from picar_kl.actions import action_count


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for the phase 2 LSTM policy") from exc
    return torch, nn, F


@dataclass(frozen=True)
class LSTMPolicyConfig:
    visual_dim: int
    model_dim: int = 128
    conditioning_dim: int = 32
    token_type_dim: int = 8
    action_dim: int = action_count()
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    dropout: float = 0.0


class LSTMPolicyOutput(NamedTuple):
    logits: "torch.Tensor"
    log_probs: "torch.Tensor"
    probabilities: "torch.Tensor"
    readout_mask: "torch.Tensor"
    token_mask: "torch.Tensor"


def validate_lstm_policy_config(config: LSTMPolicyConfig) -> LSTMPolicyConfig:
    for name in ("visual_dim", "model_dim", "conditioning_dim", "token_type_dim", "action_dim", "lstm_hidden_dim"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    if int(config.lstm_layers) < 1:
        raise ValueError("lstm_layers must be >= 1")
    if float(config.dropout) < 0.0 or float(config.dropout) >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return config


torch, nn, F = _require_torch()


class TokenStreamLSTMPolicy(nn.Module):
    """Consume visual tokens plus conditioning and emit one action per step."""

    def __init__(self, config: LSTMPolicyConfig):
        super().__init__()
        self.config = validate_lstm_policy_config(config)
        self.visual_projection = nn.Linear(self.config.visual_dim, self.config.model_dim)
        self.action_readout_token = nn.Parameter(torch.zeros(self.config.model_dim))
        self.token_type_embedding = nn.Embedding(2, self.config.token_type_dim)
        stream_dim = self.config.model_dim + self.config.conditioning_dim + self.config.action_dim + self.config.token_type_dim
        lstm_dropout = self.config.dropout if self.config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=stream_dim,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.config.lstm_hidden_dim, self.config.action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_readout_token, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        visual_tokens: "torch.Tensor",
        previous_actions: "torch.Tensor",
        conditioning: "torch.Tensor | None" = None,
        step_mask: "torch.Tensor | None" = None,
    ) -> LSTMPolicyOutput:
        self._validate_inputs(
            visual_tokens=visual_tokens,
            previous_actions=previous_actions,
            conditioning=conditioning,
            step_mask=step_mask,
        )
        batch_size, step_count, visual_token_count, _ = visual_tokens.shape
        if conditioning is None:
            conditioning = visual_tokens.new_zeros((batch_size, step_count, self.config.conditioning_dim))
        if step_mask is None:
            step_mask = torch.ones((batch_size, step_count), dtype=torch.bool, device=visual_tokens.device)
        else:
            step_mask = step_mask.to(device=visual_tokens.device, dtype=torch.bool)

        projected_visual = self.visual_projection(visual_tokens)
        readout = self.action_readout_token.to(dtype=projected_visual.dtype, device=projected_visual.device)
        readout = readout.view(1, 1, 1, self.config.model_dim).expand(batch_size, step_count, 1, -1)
        model_tokens = torch.cat([projected_visual, readout], dim=2)

        tokens_per_step = visual_token_count + 1
        conditioning_tokens = conditioning.unsqueeze(2).expand(-1, -1, tokens_per_step, -1)
        action_tokens = previous_actions.unsqueeze(2).expand(-1, -1, tokens_per_step, -1)
        token_type_ids = torch.zeros((batch_size, step_count, tokens_per_step), dtype=torch.long, device=visual_tokens.device)
        token_type_ids[:, :, -1] = 1
        token_types = self.token_type_embedding(token_type_ids).to(dtype=model_tokens.dtype)

        stream = torch.cat([model_tokens, conditioning_tokens, action_tokens, token_types], dim=-1)
        stream = stream.reshape(batch_size, step_count * tokens_per_step, -1)
        token_mask = step_mask.unsqueeze(-1).expand(-1, -1, tokens_per_step).reshape(batch_size, step_count * tokens_per_step)
        stream = stream * token_mask.unsqueeze(-1).to(dtype=stream.dtype)

        hidden, _ = self.lstm(stream)
        hidden_by_step = hidden.reshape(batch_size, step_count, tokens_per_step, self.config.lstm_hidden_dim)
        readout_hidden = hidden_by_step[:, :, -1, :]
        logits = self.action_head(readout_hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return LSTMPolicyOutput(
            logits=logits,
            log_probs=log_probs,
            probabilities=log_probs.exp(),
            readout_mask=step_mask,
            token_mask=token_mask,
        )

    def _validate_inputs(
        self,
        *,
        visual_tokens: "torch.Tensor",
        previous_actions: "torch.Tensor",
        conditioning: "torch.Tensor | None",
        step_mask: "torch.Tensor | None",
    ) -> None:
        if visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must have shape [batch, steps, visual_tokens, visual_dim]")
        if visual_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(f"expected visual_dim {self.config.visual_dim}, got {visual_tokens.shape[-1]}")
        if previous_actions.shape != (visual_tokens.shape[0], visual_tokens.shape[1], self.config.action_dim):
            raise ValueError("previous_actions must have shape [batch, steps, action_dim]")
        if conditioning is not None and conditioning.shape != (
            visual_tokens.shape[0],
            visual_tokens.shape[1],
            self.config.conditioning_dim,
        ):
            raise ValueError("conditioning must have shape [batch, steps, conditioning_dim]")
        if step_mask is not None and step_mask.shape != visual_tokens.shape[:2]:
            raise ValueError("step_mask must have shape [batch, steps]")
