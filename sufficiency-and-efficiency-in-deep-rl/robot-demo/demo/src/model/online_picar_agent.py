from __future__ import annotations

import torch
import torch.nn as nn

from .action_space import action_vector_to_tensor
from .config import ModelConfig
from .picar_core import PiCarActionModelBase
from .replay_buffer import TransitionReplayBuffer
from .schemas import Transition, TransitionBatch

try:
    from src.online_core.online_ssr_agent import OnlineSSRAgent
except ModuleNotFoundError:
    from online_core.online_ssr_agent import OnlineSSRAgent


class OnlinePiCarActionModel(PiCarActionModelBase, OnlineSSRAgent):
    """Online PiCar model that reuses PiCarActionModel architecture and loss."""

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
        OnlineSSRAgent.__init__(
            self,
            replay_buffer=replay_buffer,
            ssr_rank=ssr_rank,
            gpu_saver=gpu_saver,
            dt_mean_N=dt_mean_N,
        )
        self._init_picar_modules(config=config, backbone=backbone)

    def transition_loss(self, transition: Transition) -> torch.Tensor:
        """Build the scalar online task loss for a single fresh transition."""

        batch = TransitionBatch(
            observations=[transition.observation],
            executed_action_vector=torch.stack(
                [
                    action_vector_to_tensor(
                        transition.executed_action_vector,
                        device=self.device,
                    )
                ],
                dim=0,
            ),
            reward=torch.tensor([transition.reward], dtype=torch.float32, device=self.device),
            next_observations=[transition.next_observation],
            done=torch.tensor([transition.done], dtype=torch.float32, device=self.device),
            target_text=[transition.target_text],
            logp_beta_sum=torch.tensor(
                [
                    float(transition.logp_beta_sum)
                    if transition.logp_beta_sum is not None
                    else float("nan")
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            target_action_name=[transition.target_action_name],
            agentic_action_vector=[transition.agentic_action_vector],
            actor_action_vector=[transition.actor_action_vector],
            metadata=[transition.metadata],
        )
        return self.loss(batch)

    def _after_optimizer_step(self):
        self._soft_update_targets()
        return None
