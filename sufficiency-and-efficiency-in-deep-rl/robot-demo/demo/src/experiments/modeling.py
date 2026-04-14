from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schemas import ExperimentPhase, ExperimentRunConfig

try:
    from src.env.schemas import TrainingSummary
    from src.model import ModelConfig, OnlinePiCarActionModel, PiCarActionModel, Transition, TransitionReplayBuffer
except ModuleNotFoundError:
    from env.schemas import TrainingSummary
    from model import ModelConfig, OnlinePiCarActionModel, PiCarActionModel, Transition, TransitionReplayBuffer


ResolvedUpdateMode = Literal["batch", "online"]


def resolve_update_mode(*, phase: ExperimentPhase, update_mode: str) -> ResolvedUpdateMode:
    if update_mode == "auto":
        if phase == "init":
            return "batch"
        return "online"
    return update_mode


def build_experiment_model(
    *,
    config: ExperimentRunConfig,
    replay_buffer: TransitionReplayBuffer,
    model_config: ModelConfig,
    backbone=None,
):
    resolved_update_mode = resolve_update_mode(phase=config.phase, update_mode=config.update_mode)
    if resolved_update_mode == "online":
        return OnlinePiCarActionModel(replay_buffer=replay_buffer, config=model_config, backbone=backbone)
    return PiCarActionModel(replay_buffer=replay_buffer, config=model_config, backbone=backbone)


@dataclass(frozen=True)
class BatchTrainingAdapter:
    pi_override: float | None = None

    def train(
        self,
        *,
        model,
        transition: Transition,
        training,
        replay_size: int,
        effective_fit_iters: int,
        step_index: int,
    ) -> TrainingSummary:
        del transition, replay_size, step_index
        pi = 0.0
        loss = 0.0
        for _ in range(max(1, int(training.epochs))):
            fit_kwargs = {
                "batch_size": training.batch_size,
                "iters": effective_fit_iters,
            }
            if self.pi_override is not None:
                fit_kwargs["pi_min"] = float(self.pi_override)
                fit_kwargs["pi_max"] = float(self.pi_override)
            pi, loss = model.fit(**fit_kwargs)
        memorized = None
        if training.memorize_every_steps > 0 and (step_index + 1) % training.memorize_every_steps == 0:
            if training.memorize_n < 0:
                memorize_count = replay_size
            else:
                memorize_count = min(training.memorize_n, replay_size)
            if memorize_count > 0:
                model.memorize(
                    n=memorize_count,
                    random_idx=training.memorize_random_idx,
                    disable_tqdm=True,
                )
                replay_buffer = model.replay_buffer
                if hasattr(replay_buffer, "clear"):
                    replay_buffer.clear(memorize_count)
                memorized = memorize_count
        return TrainingSummary(
            triggered=True,
            replay_size=replay_size,
            pi=float(pi),
            loss=float(loss),
            memorized=memorized,
        )


@dataclass(frozen=True)
class OnlineTrainingAdapter:
    pi_override: float | None = None

    def train(
        self,
        *,
        model,
        transition: Transition,
        training,
        replay_size: int,
        effective_fit_iters: int,
        step_index: int,
    ) -> TrainingSummary:
        del replay_size, effective_fit_iters
        pi = 0.0
        loss = 0.0
        memorize_now = training.memorize_every_steps > 0 and (step_index + 1) % training.memorize_every_steps == 0
        for epoch_idx in range(max(1, int(training.epochs))):
            scalar_loss = model.transition_loss(transition)
            pi, loss = model.fit(
                loss=scalar_loss,
                pi=self.pi_override,
                memorize=bool(memorize_now and epoch_idx == max(1, int(training.epochs)) - 1),
            )
        return TrainingSummary(
            triggered=True,
            replay_size=len(model.replay_buffer),
            pi=float(pi),
            loss=float(loss),
            memorized=(1 if memorize_now else None),
        )


def build_training_adapter(*, config: ExperimentRunConfig):
    resolved_update_mode = resolve_update_mode(phase=config.phase, update_mode=config.update_mode)
    if resolved_update_mode == "online":
        return OnlineTrainingAdapter(pi_override=config.pi)
    return BatchTrainingAdapter(pi_override=config.pi)
