from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch

from .action_space import action_vector_to_tensor
from .schemas import Transition, TransitionBatch


class TransitionReplayBuffer:
    """Replay buffer for multimodal transitions consumed by `PiCarActionModel`."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._storage: list[Transition] = []
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def add(self, transition: Transition) -> None:
        if self.n >= self.capacity:
            self._storage.pop(0)
            self.n -= 1
        self._storage.append(transition)
        self.n += 1

    def sample(
        self,
        batch_size: int = 32,
        idx_list: list[int] | torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> TransitionBatch:
        if self.n == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        if idx_list is None:
            indices = [random.randint(0, self.n - 1) for _ in range(batch_size)]
        elif isinstance(idx_list, torch.Tensor):
            indices = [int(idx) for idx in idx_list.tolist()]
        else:
            indices = [int(idx) for idx in idx_list]

        transitions = [self._storage[idx] for idx in indices]
        target_device = device or torch.device("cpu")
        return TransitionBatch(
            observations=[transition.observation for transition in transitions],
            executed_action_vector=torch.stack(
                [
                    action_vector_to_tensor(
                        transition.executed_action_vector,
                        device=target_device,
                    )
                    for transition in transitions
                ],
                dim=0,
            ),
            reward=torch.tensor(
                [transition.reward for transition in transitions],
                dtype=torch.float32,
                device=target_device,
            ),
            next_observations=[transition.next_observation for transition in transitions],
            done=torch.tensor(
                [transition.done for transition in transitions],
                dtype=torch.float32,
                device=target_device,
            ),
            target_text=[transition.target_text for transition in transitions],
            target_action_name=[transition.target_action_name for transition in transitions],
            agentic_action_vector=[transition.agentic_action_vector for transition in transitions],
            actor_action_vector=[transition.actor_action_vector for transition in transitions],
            metadata=[transition.metadata for transition in transitions],
        )

    def clear(self, n: int | None = None) -> None:
        if n is None or n >= self.n:
            self._storage = []
            self.n = 0
            return
        self._storage = self._storage[n:]
        self.n = len(self._storage)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"capacity": self.capacity, "storage": self._storage}, handle)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        self.capacity = int(data["capacity"])
        self._storage.extend(data["storage"])
        if len(self._storage) > self.capacity:
            self._storage = self._storage[-self.capacity :]
        self.n = len(self._storage)
