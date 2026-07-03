"""Small trainable heads for phase 2 VLM conditioning vectors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMConditioningHeadConfig:
    input_dim: int
    conditioning_dim: int
    hidden_dim: int | None = None


@dataclass(frozen=True)
class PrefixConditioningHeadConfig:
    visual_dim: int
    action_dim: int
    conditioning_dim: int
    hidden_dim: int = 128


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for phase 2 VLM conditioning heads") from exc
    return torch, nn


def validate_vlm_conditioning_head_config(config: VLMConditioningHeadConfig) -> VLMConditioningHeadConfig:
    if int(config.input_dim) < 1:
        raise ValueError("input_dim must be >= 1")
    if int(config.conditioning_dim) < 1:
        raise ValueError("conditioning_dim must be >= 1")
    if config.hidden_dim is not None and int(config.hidden_dim) < 1:
        raise ValueError("hidden_dim must be >= 1 when provided")
    return config


def validate_prefix_conditioning_head_config(config: PrefixConditioningHeadConfig) -> PrefixConditioningHeadConfig:
    for name in ("visual_dim", "action_dim", "conditioning_dim", "hidden_dim"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    return config


torch, nn = _require_torch()


class VLMConditioningHead(nn.Module):
    """Map VLM-derived source features to a `Dh` conditioning vector."""

    def __init__(self, config: VLMConditioningHeadConfig):
        super().__init__()
        self.config = validate_vlm_conditioning_head_config(config)
        if self.config.hidden_dim is None:
            self.net = nn.Linear(self.config.input_dim, self.config.conditioning_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.conditioning_dim),
            )

    def forward(self, source_features: "torch.Tensor") -> "torch.Tensor":
        if source_features.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected source feature width {self.config.input_dim}, got {source_features.shape[-1]}")
        return self.net(source_features)


class PrefixConditioningHead(nn.Module):
    """Create one trainable `Dh` vector from a prefix of observations and actions."""

    def __init__(self, config: PrefixConditioningHeadConfig):
        super().__init__()
        self.config = validate_prefix_conditioning_head_config(config)
        self.visual_projection = nn.Linear(self.config.visual_dim, self.config.hidden_dim)
        self.action_projection = nn.Linear(self.config.action_dim, self.config.hidden_dim)
        self.step_encoder = nn.GRU(
            input_size=self.config.hidden_dim * 2,
            hidden_size=self.config.hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(self.config.hidden_dim, self.config.conditioning_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        *,
        visual_tokens: "torch.Tensor",
        action_distributions: "torch.Tensor",
        step_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        if visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must have shape [batch, steps, visual_tokens, visual_dim]")
        if visual_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(f"expected visual_dim {self.config.visual_dim}, got {visual_tokens.shape[-1]}")
        if action_distributions.shape != (visual_tokens.shape[0], visual_tokens.shape[1], self.config.action_dim):
            raise ValueError("action_distributions must have shape [batch, steps, action_dim]")
        if step_mask is None:
            step_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
        if step_mask.shape != visual_tokens.shape[:2]:
            raise ValueError("step_mask must have shape [batch, steps]")
        step_mask = step_mask.to(device=visual_tokens.device, dtype=torch.bool)

        visual_summary = visual_tokens.mean(dim=2)
        visual_features = self.activation(self.visual_projection(visual_summary))
        action_features = self.activation(self.action_projection(action_distributions))
        step_features = torch.cat([visual_features, action_features], dim=-1)
        step_features = step_features * step_mask.unsqueeze(-1).to(dtype=step_features.dtype)
        encoded, _ = self.step_encoder(step_features)
        lengths = step_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.config.hidden_dim)
        final_state = encoded.gather(dim=1, index=gather_index).squeeze(1)
        return self.output(final_state)
