from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import (
    ACTION_NAMES,
    action_name_to_one_hot,
    one_hot_to_action_vector,
    mix_action_vectors,
)
from .backbones import BackboneBatchOutput, FakeBackbone, QwenLoRABackbone
from .config import ModelConfig
from .replay_buffer import TransitionReplayBuffer
from .schemas import ModelActionOutput, ModelObservation, TransitionBatch


@dataclass(frozen=True)
class ForwardBatchOutput:
    """Intermediate batch output used by `forward()` and `loss()`."""

    actions: list[ModelActionOutput]
    value_logits: torch.Tensor
    vlm_loss: torch.Tensor
    hidden_state: torch.Tensor
    debug: list[dict[str, object]]


from core.ssr_agent import SSRAgent


class PiCarActionModel(SSRAgent):
    """SSR-compatible PiCar model with agentic and value-driven action branches.

    The model has two decision paths over the same 8-action discrete space:

    1. the agentic branch, produced by the VLM/tool-calling path
    2. the value branch, produced by an 8-way value head

    Each branch first selects a one-hot action. That one-hot is then mapped into
    the 4-key PiCar control dictionary expected by `apply_vector`:

    - `pan`
    - `tilt`
    - `turn`
    - `drive`

    The execution vector is blended inside the model, not in the environment:

    `mixed = (1 - t) * agentic_vector + t * value_vector`

    This keeps the environment simple: it only passes observations in and
    executes the returned mixed vector.
    """

    def __init__(
        self,
        replay_buffer: TransitionReplayBuffer,
        *,
        config: ModelConfig | None = None,
        backbone: nn.Module | None = None,
        ssr_rank: int = 2,
        gpu_saver: bool = True,
        dt_mean_N: int = 10,
    ):
        """Construct the PiCar action model.

        The trainable parameters are intentionally small:

        - LoRA parameters inside the backbone
        - the new 8-way value head

        The base VLM remains frozen. A plain SGD optimizer is used because the
        surrounding `SSRAgent` code already supplies the natural-gradient-style
        geometry; adding momentum here would be redundant and more memory-hungry.
        """

        self.config = config or ModelConfig()
        super().__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank, gpu_saver=gpu_saver, dt_mean_N=dt_mean_N)
        self.backbone = backbone if backbone is not None else QwenLoRABackbone.from_config(self.config)
        hidden_size = int(getattr(self.backbone, "hidden_size", self.config.hidden_size or 0))
        if hidden_size <= 0:
            raise ValueError("Backbone must expose a positive `hidden_size`")
        self.value_head = nn.Linear(hidden_size, len(ACTION_NAMES), bias=self.config.value_head_bias)
        self.to(self.device)
        trainable_params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.SGD(trainable_params, lr=self.config.learning_rate)

    def forward(
        self,
        observations: ModelObservation | Sequence[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
    ) -> ModelActionOutput | ForwardBatchOutput:
        """Run inference for one observation or a batch of observations.

        For a single observation, this returns the final `ModelActionOutput`
        consumed by the environment. For a batch, it returns a richer internal
        structure used by training.
        """

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
        """Compute the mean-scaled combined VLM + RL objective.

        The total loss is:

        `L = alpha * L_vlm + beta * L_rl`

        where:

        - `L_vlm` is the backbone-provided supervised language/tool-call loss
        - `L_rl` is the mean Huber TD loss over the selected action values

        For the RL term:

        - `Q(s)` is the 8-way value head output
        - `Q(s, a)` is gathered at the executed action index
        - the bootstrap target is
          `r + gamma * (1 - done) * max_a' Q(s', a')`

        The next-state value is detached in the first pass, matching the
        planning decision to keep the bootstrap target simple.
        """

        current = self._forward_batch(
            transitions.observations,
            target_texts=transitions.target_text,
            compute_vlm_loss=True,
        )
        # Select Q(s, a) from the 8-way value head for the action that was
        # actually executed and stored in the replay buffer.
        q_selected = current.value_logits.gather(1, transitions.action_index.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state = self._forward_batch(transitions.next_observations, compute_vlm_loss=False)
            # Standard one-step bootstrap target:
            # target = r + gamma * (1 - done) * max_a' Q(s', a')
            next_q = next_state.value_logits.max(dim=1).values
            td_target = transitions.reward + self.config.gamma * (1.0 - transitions.done) * next_q.detach()
        # Huber loss is more stable than plain squared error for TD residuals.
        rl_loss = F.smooth_l1_loss(q_selected, td_target, reduction="mean")
        total_loss = self.config.alpha * current.vlm_loss + self.config.beta * rl_loss
        return total_loss

    def _forward_batch(
        self,
        observations: list[ModelObservation],
        *,
        target_texts: list[Optional[str]] | None = None,
        compute_vlm_loss: bool = False,
    ) -> ForwardBatchOutput:
        """Internal batch forward pass shared by inference and training.

        The backbone produces:

        - a pooled hidden representation for each observation
        - an agentic action choice
        - optional generated text
        - optional VLM supervision loss

        The value head then maps each pooled hidden state into 8 logits, one per
        discrete PiCar action. Both the agentic branch and the value branch are
        converted into 4-key control dictionaries and blended by `t`.
        """

        backbone_output = self.backbone.encode(
            observations,
            target_texts=target_texts,
            compute_vlm_loss=compute_vlm_loss,
        )
        hidden_state = backbone_output.pooled_hidden_state.to(self.device)
        # The value head parameterizes Q(s, a) for the 8 discrete actions.
        value_logits = self.value_head(hidden_state)
        value_action_indices = torch.argmax(value_logits, dim=1)
        actions = []
        for idx, observation in enumerate(observations):
            agentic_name = backbone_output.agentic_action_names[idx]
            agentic_one_hot = action_name_to_one_hot(agentic_name, device=value_logits.device)
            value_one_hot = F.one_hot(value_action_indices[idx], num_classes=len(ACTION_NAMES)).to(dtype=value_logits.dtype)
            # Discrete actions are converted into the vector-valued PiCar control
            # space before interpolation.
            agentic_vector = one_hot_to_action_vector(agentic_one_hot)
            value_vector = one_hot_to_action_vector(value_one_hot)
            mixed_vector = mix_action_vectors(agentic_vector, value_vector, observation.t)
            actions.append(
                ModelActionOutput(
                    agentic_action_name=agentic_name,
                    agentic_action_one_hot=agentic_one_hot.detach().cpu(),
                    value_logits=value_logits[idx].detach().cpu(),
                    value_action_index=int(value_action_indices[idx].item()),
                    value_action_one_hot=value_one_hot.detach().cpu(),
                    agentic_action_vector=agentic_vector,
                    value_action_vector=value_vector,
                    mixed_action_vector=mixed_vector,
                    generated_text=backbone_output.generated_texts[idx],
                    debug=backbone_output.debug[idx] if idx < len(backbone_output.debug) else {},
                )
            )
        vlm_loss = backbone_output.vlm_loss
        if vlm_loss is None:
            # Keep the loss tensor connected to the graph shape even when the
            # backbone does not provide a supervised language term.
            vlm_loss = value_logits.sum() * 0.0
        else:
            vlm_loss = vlm_loss.to(self.device)
        return ForwardBatchOutput(
            actions=actions,
            value_logits=value_logits,
            vlm_loss=vlm_loss,
            hidden_state=hidden_state,
            debug=backbone_output.debug,
        )
