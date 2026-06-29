from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig
from .picar_core import PiCarActionModelBase
from .replay_buffer import TransitionReplayBuffer

try:
    from src.core.ssr_agent import SSRAgent
except ModuleNotFoundError:
    from core.ssr_agent import SSRAgent


class PiCarActionModel(PiCarActionModelBase, SSRAgent):
    """SSR-compatible PiCar model with agentic VLM and continuous actor-critic control."""

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
        SSRAgent.__init__(
            self,
            replay_buffer=replay_buffer,
            ssr_rank=ssr_rank,
            gpu_saver=gpu_saver,
            dt_mean_N=dt_mean_N,
        )
        self._init_picar_modules(config=config, backbone=backbone)

    def fit(self, batch_size, iters=1, pi_min=0.1, pi_max=0.9, grad_clip=None):
        result = super().fit(batch_size=batch_size, iters=iters, pi_min=pi_min, pi_max=pi_max, grad_clip=grad_clip)
        self._soft_update_targets()
        return result
