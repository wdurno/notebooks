from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import (
    action_name_to_one_hot,
    action_vector_to_tensor,
    clamp_action_tensor,
    mix_action_tensors,
    one_hot_to_action_vector,
    tensor_to_action_vector,
)
from .backbones import QwenLoRABackbone
from .config import ModelConfig
from .schemas import ModelActionOutput, ModelObservation, TransitionBatch


@dataclass(frozen=True)
class ForwardBatchOutput:
    """Intermediate batch output used by `forward()` and `loss()`."""

    actions: list[ModelActionOutput]
    vlm_loss: torch.Tensor
    hidden_state: torch.Tensor
    agentic_action_tensor: torch.Tensor
    actor_action_tensor: torch.Tensor
    executed_action_tensor: torch.Tensor
    critic_value: torch.Tensor
    value_estimate: torch.Tensor
    generated_logp_sums: torch.Tensor
    target_logp_sums: torch.Tensor
    t_tensor: torch.Tensor
    debug: list[dict[str, object]]


class ContinuousActorHead(nn.Module):
    """Map hidden states into bounded PiCar control vectors."""

    def __init__(self, hidden_size: int, *, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 4, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raw = self.proj(hidden_state)
        pan = torch.tanh(raw[:, 0:1])
        tilt = torch.sigmoid(raw[:, 1:2])
        turn = torch.tanh(raw[:, 2:3])
        drive = torch.tanh(raw[:, 3:4])
        return torch.cat([pan, tilt, turn, drive], dim=1)


class ContinuousQCritic(nn.Module):
    """Single-critic network for continuous-action Q estimation."""

    def __init__(self, hidden_size: int, *, critic_hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size + 4, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([hidden_state, action_tensor], dim=1)
        return self.network(critic_input).squeeze(1)


class PiCarActionModelBase:
    """Shared PiCar architecture, loss, and target-network behavior."""

    config: ModelConfig
    backbone: nn.Module
    actor_head: ContinuousActorHead
    critic: ContinuousQCritic
    value_head: nn.Linear
    target_actor_head: ContinuousActorHead
    target_critic: ContinuousQCritic

    def _init_picar_modules(
        self,
        *,
        config: ModelConfig | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.backbone = backbone if backbone is not None else QwenLoRABackbone.from_config(self.config)
        hidden_size = int(getattr(self.backbone, "hidden_size", self.config.hidden_size or 0))
        if hidden_size <= 0:
            raise ValueError("Backbone must expose a positive `hidden_size`")

        self.actor_head = ContinuousActorHead(hidden_size, bias=self.config.value_head_bias)
        critic_hidden_size = int(self.config.critic_hidden_size or hidden_size)
        self.critic = ContinuousQCritic(hidden_size, critic_hidden_size=critic_hidden_size)
        self.value_head = nn.Linear(hidden_size, 1, bias=self.config.value_head_bias)
        self.target_actor_head = copy.deepcopy(self.actor_head)
        self.target_critic = copy.deepcopy(self.critic)
        for parameter in self.target_actor_head.parameters():
            parameter.requires_grad = False
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        self.to(self.device)
        self.sync_target_networks()
        trainable_params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.SGD(trainable_params, lr=self.config.learning_rate)
        return None

    def sync_target_networks(self) -> None:
        """Hard-sync target networks from the currently loaded live modules."""

        self.target_actor_head.load_state_dict(self.actor_head.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        return None

    def post_snapshot_load(self) -> None:
        """Restore derived runtime state after snapshot weights are loaded."""

        self.sync_target_networks()
        return None

    def forward(
        self,
        observations: ModelObservation | Sequence[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
    ) -> ModelActionOutput | ForwardBatchOutput:
        single = isinstance(observations, ModelObservation)
        observation_list = [observations] if single else list(observations)
        batch = self._forward_batch(
            observation_list,
            target_texts=target_texts,
            compute_vlm_loss=compute_vlm_loss,
        )
        if single:
            return batch.actions[0]
        return batch

    def loss(self, transitions: TransitionBatch) -> torch.Tensor:
        """Compute the full objective over VLM, control RL, token PG, and value loss."""

        current = self._forward_batch(
            transitions.observations,
            target_texts=transitions.target_text,
            compute_vlm_loss=True,
        )
        reward_tensor = transitions.reward.to(self.device)
        done_tensor = transitions.done.to(self.device)
        logp_beta_sum = transitions.logp_beta_sum.to(self.device)
        executed_action_tensor = transitions.executed_action_vector.to(self.device)
        q_current = self.critic(current.hidden_state, executed_action_tensor)
        value_current = self.value_head(current.hidden_state).squeeze(1)

        with torch.no_grad():
            next_batch = self._forward_batch(
                transitions.next_observations,
                compute_vlm_loss=False,
            )
            next_actor_action = self.target_actor_head(next_batch.hidden_state)
            next_executed_action = mix_action_tensors(
                next_batch.agentic_action_tensor.detach(),
                next_actor_action,
                next_batch.t_tensor,
            )
            next_q = self.target_critic(next_batch.hidden_state, next_executed_action)
            value_next = self.value_head(next_batch.hidden_state).squeeze(1)
            td_target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q
            value_target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * value_next

        critic_loss = F.smooth_l1_loss(q_current, td_target, reduction="mean")
        value_loss = F.smooth_l1_loss(value_current, value_target, reduction="mean")
        advantage = (value_target - value_current).detach()

        agentic_detached = current.agentic_action_tensor.detach()
        actor_executed_action = mix_action_tensors(
            agentic_detached,
            current.actor_action_tensor,
            current.t_tensor,
        )
        actor_value = self._critic_value_for_actor(current.hidden_state, actor_executed_action)
        anchor_penalty = (current.actor_action_tensor - agentic_detached).pow(2).mean(dim=1)
        anchor_weight = self.config.actor_anchor_weight * (1.0 - current.t_tensor)
        actor_loss = -actor_value.mean() + (anchor_weight * anchor_penalty).mean()

        rl_loss = actor_loss + critic_loss
        has_target_text = torch.tensor(
            [bool(text and text.strip()) for text in transitions.target_text],
            dtype=torch.bool,
            device=self.device,
        )
        valid_pg = has_target_text & torch.isfinite(logp_beta_sum) & torch.isfinite(current.target_logp_sums)
        if bool(valid_pg.any().item()):
            rho = torch.exp(current.target_logp_sums.detach() - logp_beta_sum)
            rho = torch.where(torch.isfinite(rho), rho, torch.zeros_like(rho))
            token_pg_terms = -(rho * advantage * current.target_logp_sums)
            token_pg_loss = token_pg_terms[valid_pg].mean()
        else:
            token_pg_loss = current.hidden_state.sum() * 0.0
        total_loss = (
            self.config.alpha * current.vlm_loss
            + self.config.beta * rl_loss
            + self.config.token_pg_weight * token_pg_loss
            + self.config.value_loss_weight * value_loss
        )
        return total_loss

    def set_inference_mode(self) -> None:
        """Switch model to rollout/inference mode."""

        self.eval()
        backbone_model = getattr(self.backbone, "model", None)
        if backbone_model is not None:
            backbone_model.eval()
            disable_checkpointing = getattr(backbone_model, "gradient_checkpointing_disable", None)
            if callable(disable_checkpointing):
                try:
                    disable_checkpointing()
                except Exception:
                    pass
        return None

    def set_optimization_mode(self) -> None:
        """Switch model to optimization mode for fitting/memorization."""

        self.train()
        backbone_model = getattr(self.backbone, "model", None)
        if backbone_model is not None:
            backbone_model.train()
            enable_checkpointing = getattr(backbone_model, "gradient_checkpointing_enable", None)
            if callable(enable_checkpointing):
                try:
                    enable_checkpointing()
                except Exception:
                    pass
        return None

    def _forward_batch(
        self,
        observations: list[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
    ) -> ForwardBatchOutput:
        allow_agentic_actions = any(observation.t < 1.0 for observation in observations)
        backbone_output = self.backbone.encode(
            observations,
            target_texts=target_texts,
            compute_vlm_loss=compute_vlm_loss,
            allow_agentic_actions=allow_agentic_actions,
        )
        head_dtype = self.actor_head.proj.weight.dtype
        hidden_state = backbone_output.pooled_hidden_state.to(self.device, dtype=head_dtype)
        actor_action_tensor = clamp_action_tensor(self.actor_head(hidden_state))
        t_tensor = torch.tensor(
            [float(observation.t) for observation in observations],
            dtype=hidden_state.dtype,
            device=hidden_state.device,
        )

        agentic_vectors = []
        agentic_tensors = []
        actions = []
        for idx, observation in enumerate(observations):
            agentic_name = backbone_output.agentic_action_names[idx]
            agentic_one_hot = action_name_to_one_hot(agentic_name, device=hidden_state.device)
            agentic_vector = one_hot_to_action_vector(agentic_one_hot)
            agentic_tensor = action_vector_to_tensor(agentic_vector, device=hidden_state.device, dtype=hidden_state.dtype)
            agentic_vectors.append(agentic_vector)
            agentic_tensors.append(agentic_tensor)

        agentic_action_tensor = torch.stack(agentic_tensors, dim=0)
        executed_action_tensor = mix_action_tensors(agentic_action_tensor, actor_action_tensor, t_tensor)
        critic_value = self.critic(hidden_state, executed_action_tensor)
        value_estimate = self.value_head(hidden_state).squeeze(1)
        generated_logp_sums = backbone_output.generated_logp_sums
        if generated_logp_sums is None:
            generated_logp_sums = hidden_state.sum(dim=1) * 0.0
        else:
            generated_logp_sums = generated_logp_sums.to(hidden_state.device, dtype=torch.float32)
        target_logp_sums = backbone_output.target_logp_sums
        if target_logp_sums is None:
            target_logp_sums = hidden_state.sum(dim=1) * 0.0
        else:
            target_logp_sums = target_logp_sums.to(hidden_state.device, dtype=torch.float32)

        for idx, observation in enumerate(observations):
            del observation
            actions.append(
                ModelActionOutput(
                    agentic_action_name=backbone_output.agentic_action_names[idx],
                    agentic_action_one_hot=action_name_to_one_hot(
                        backbone_output.agentic_action_names[idx],
                        device=torch.device("cpu"),
                    ),
                    agentic_action_vector=agentic_vectors[idx],
                    actor_action_vector=tensor_to_action_vector(actor_action_tensor[idx]),
                    executed_action_vector=tensor_to_action_vector(executed_action_tensor[idx]),
                    critic_value=float(critic_value[idx].detach().cpu().item()),
                    generated_text=backbone_output.generated_texts[idx],
                    logp_beta_sum=float(generated_logp_sums[idx].detach().cpu().item()),
                    debug=backbone_output.debug[idx] if idx < len(backbone_output.debug) else {},
                )
            )

        vlm_loss = backbone_output.vlm_loss
        if vlm_loss is None:
            vlm_loss = hidden_state.sum() * 0.0
        else:
            vlm_loss = vlm_loss.to(self.device)
        return ForwardBatchOutput(
            actions=actions,
            vlm_loss=vlm_loss,
            hidden_state=hidden_state,
            agentic_action_tensor=agentic_action_tensor,
            actor_action_tensor=actor_action_tensor,
            executed_action_tensor=executed_action_tensor,
            critic_value=critic_value,
            value_estimate=value_estimate,
            generated_logp_sums=generated_logp_sums,
            target_logp_sums=target_logp_sums,
            t_tensor=t_tensor,
            debug=backbone_output.debug,
        )

    def _critic_value_for_actor(self, hidden_state: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        critic_requires_grad = [parameter.requires_grad for parameter in self.critic.parameters()]
        try:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            return self.critic(hidden_state, action_tensor)
        finally:
            for parameter, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                parameter.requires_grad_(requires_grad)

    def _soft_update_targets(self) -> None:
        tau = float(self.config.target_update_tau)
        if tau <= 0.0:
            return None
        self._soft_update_module(self.target_actor_head, self.actor_head, tau)
        self._soft_update_module(self.target_critic, self.critic, tau)
        return None

    @staticmethod
    def _soft_update_module(target_module: nn.Module, source_module: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target_module.parameters(), source_module.parameters()):
                target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)
        return None
