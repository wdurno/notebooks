from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
import time
from typing import Any

from model import ModelObservation, Transition

from .config import EnvConfig
from .persistence import append_jsonl, create_experiment_paths, save_frame_array, save_model_artifacts, save_replay_snapshot, write_metadata
from .schemas import ExperimentPaths, RewardResult, TrainingSummary


def build_operator_message(
    *,
    user_texts: list[str],
    reward_result: RewardResult,
    step_index: int,
    last_reward: float,
) -> dict[str, Any]:
    """Build the user-side Qwen message for one control iteration.

    The environment feeds operator speech and a compact reward/status summary
    back into the model as plain text so the model can condition on both the
    latest instruction and the recent scalar feedback signal.
    """

    text_blocks = []
    if user_texts:
        joined = "\n".join(user_texts)
        text_blocks.append(f"Operator speech:\n{joined}")
    else:
        text_blocks.append("Operator speech:\n<none>")
    text_blocks.append(
        "Environment status:\n"
        f"step_index={step_index}\n"
        f"last_reward={last_reward:.3f}\n"
        f"current_reward={reward_result.clipped_reward:.3f}\n"
        f"reward_prompt_id={reward_result.prompt_id}"
    )
    return {
        "role": "user",
        "content": [{"type": "text", "text": "\n\n".join(text_blocks)}],
    }


def trim_history(history: list[dict[str, Any]], history_window: int) -> list[dict[str, Any]]:
    """Keep only the most recent messages allowed by the configured window."""

    if history_window <= 0:
        return []
    return list(history[-history_window:])


class PiCarGymEnv:
    """Gym-like PiCar environment that owns the orchestration loop.

    This class sits between the trainable policy model and the real robot. It
    coordinates:

    - speech intake
    - frozen reward scoring
    - message/history assembly
    - action execution
    - replay-buffer insertion
    - periodic training
    - experiment persistence
    """

    def __init__(
        self,
        *,
        model: Any,
        reward_scorer: Any,
        picar_client: Any,
        speech_stream: Any | None = None,
        speaker: Any | None = None,
        config: EnvConfig | None = None,
    ):
        """Construct the environment with externally supplied runtime services."""

        self.model = model
        self.reward_scorer = reward_scorer
        self.picar_client = picar_client
        self.speech_stream = speech_stream
        self.speaker = speaker
        self.config = config or EnvConfig()
        self.history: list[dict[str, Any]] = []
        self.step_index = 0
        self.last_reward = 0.0
        self.paths: ExperimentPaths | None = None
        self._run_metadata: dict[str, Any] = {}
        self._last_vector_command_at: float | None = None
        self._started = False
        self._closed = False

    def reset(self) -> ModelObservation:
        """Start a fresh run directory and return the first observation.

        Reset captures an initial frame, scores it with the frozen reward VLM,
        drains any pending speech events, and builds the first model
        observation. It also starts the continuous speech worker if one is
        configured.
        """

        self._closed = False
        self.history = []
        self.step_index = 0
        self.last_reward = 0.0
        self.paths = create_experiment_paths(self.config.data_dir, self.config.experiment_name_prefix)
        self._run_metadata = {}
        self._last_vector_command_at = None
        if self.speech_stream is not None:
            self.speech_stream.start()
        frame = self.picar_client.reset()
        reward_result = self.reward_scorer.score(frame)
        user_texts = [event.text for event in self._drain_speech_events()]
        user_message = build_operator_message(
            user_texts=user_texts,
            reward_result=reward_result,
            step_index=self.step_index,
            last_reward=self.last_reward,
        )
        messages = trim_history([user_message], self.config.history_window)
        observation = ModelObservation(
            image_rgb=frame,
            messages=messages,
            t=self._interpolation_t(self.step_index),
            last_reward=self.last_reward,
            done=False,
            step_index=self.step_index,
            metadata={"reward_prompt_id": reward_result.prompt_id},
        )
        self.history = messages
        self._started = True
        self._write_run_metadata(initial_reward=reward_result.clipped_reward)
        return observation

    def step(self) -> tuple[ModelObservation, float, bool, dict[str, Any]]:
        """Execute one environment iteration and return a Gym-like step tuple."""

        if not self._started:
            self.reset()

        # Speech is collected continuously in the background; each step simply
        # drains whatever complete utterances arrived since the previous one.
        user_texts = [event.text for event in self._drain_speech_events()]
        current_frame = self.picar_client.capture_image()
        reward_result = self.reward_scorer.score(current_frame)
        user_message = build_operator_message(
            user_texts=user_texts,
            reward_result=reward_result,
            step_index=self.step_index,
            last_reward=self.last_reward,
        )
        current_messages = trim_history(self.history + [user_message], self.config.history_window)
        observation = ModelObservation(
            image_rgb=current_frame,
            messages=current_messages,
            t=self._interpolation_t(self.step_index),
            last_reward=self.last_reward,
            done=False,
            step_index=self.step_index,
            metadata={"reward_prompt_id": reward_result.prompt_id},
        )

        # The model owns action interpolation and returns the final mixed PiCar
        # control vector ready to send to the robot API.
        action = self.model.forward(observation)
        if action.generated_text and self.speaker is not None:
            self.speaker.speak(action.generated_text)
        self._respect_command_rate_limit()
        action_receipt = self.picar_client.apply_vector(action.executed_action_vector)
        self._last_vector_command_at = time.monotonic()

        # Persist the assistant reply in the same structured format expected by
        # the Qwen chat template so future steps can reuse the dialogue history.
        execution_mode = "mixed" if observation.t < 1.0 else "actor_only"
        assistant_payload = (
            json.dumps(
                {"action": action.agentic_action_name, "say": action.generated_text},
                ensure_ascii=True,
            )
            if observation.t < 1.0
            else action.generated_text
        )
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_payload}],
        }
        self.history = trim_history(current_messages + [assistant_message], self.config.history_window)

        next_frame = self.picar_client.capture_image()
        next_observation = ModelObservation(
            image_rgb=next_frame,
            messages=self.history,
            t=self._interpolation_t(self.step_index + 1),
            last_reward=reward_result.clipped_reward,
            done=False,
            step_index=self.step_index + 1,
            metadata={"reward_prompt_id": reward_result.prompt_id},
        )

        # Replay stores the actual executed control vector so the critic trains
        # on the same action that was sent to the robot.
        target_text = assistant_payload if assistant_payload else None
        transition = Transition(
            observation=observation,
            executed_action_vector=action.executed_action_vector,
            reward=reward_result.clipped_reward,
            next_observation=next_observation,
            done=False,
            target_text=target_text,
            target_action_name=action.agentic_action_name,
            agentic_action_vector=action.agentic_action_vector,
            actor_action_vector=action.actor_action_vector,
            metadata={
                "reward_prompt_id": reward_result.prompt_id,
                "generated_text": action.generated_text,
                "agentic_action_name": action.agentic_action_name,
                "execution_mode": execution_mode,
                "action_receipt": action_receipt,
            },
        )
        self.model.replay_buffer.add(transition)
        training_summary = self._maybe_train()

        frame_path = None
        if self.paths is not None and self.config.write_frame_blobs:
            frame_path = self.paths.blobs_dir / f"step-{self.step_index:06d}.npy"
            save_frame_array(frame_path, current_frame)

        if self.paths is not None:
            append_jsonl(
                self.paths.logs_dir / "events.jsonl",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step_index": self.step_index,
                    "reward": reward_result.clipped_reward,
                    "reward_raw_text": reward_result.raw_text,
                    "user_texts": user_texts,
                    "action": {
                        "agentic_action_name": action.agentic_action_name,
                        "agentic_action_vector": action.agentic_action_vector,
                        "actor_action_vector": action.actor_action_vector,
                        "executed_action_vector": action.executed_action_vector,
                        "critic_value": action.critic_value,
                        "generated_text": action.generated_text,
                        "execution_mode": execution_mode,
                    },
                    "training": asdict(training_summary),
                    "frame_path": str(frame_path) if frame_path is not None else None,
                },
            )

        self.last_reward = reward_result.clipped_reward
        self.step_index += 1
        info = {
            "reward_result": reward_result,
            "action": action,
            "user_texts": user_texts,
            "training": training_summary,
            "action_receipt": action_receipt,
        }
        return next_observation, reward_result.clipped_reward, False, info

    def run_forever(self) -> None:
        """Run the control loop until the operator interrupts with `Ctrl-C`."""

        self.reset()
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        return None

    def close(self) -> None:
        """Stop background services, park the camera, and flush run artifacts."""

        if self._closed:
            return None
        if self.speech_stream is not None:
            self.speech_stream.stop()
        try:
            # Best-effort robot cleanup; persistence should still run even if
            # the robot connection is already gone.
            self.picar_client.look_forward()
        except Exception:
            pass
        if self.paths is not None:
            self._run_metadata.update(
                {
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "steps_completed": self.step_index,
                    "last_reward": self.last_reward,
                }
            )
            write_metadata(self.paths, self._run_metadata)
            save_replay_snapshot(self.model.replay_buffer, self.paths.blobs_dir / "replay_buffer.pkl")
            save_model_artifacts(self.model, self.paths.artifacts_dir, self.config.checkpoint_basename)
        self._closed = True
        return None

    def _drain_speech_events(self) -> list[Any]:
        """Return any queued speech events without blocking the control loop."""

        if self.speech_stream is None:
            return []
        return self.speech_stream.drain()

    def _respect_command_rate_limit(self) -> None:
        """Keep `apply_vector` calls below the Raspberry Pi stability threshold."""

        min_interval = float(self.config.picar.min_command_interval_seconds)
        if min_interval <= 0.0 or self._last_vector_command_at is None:
            return None
        elapsed = time.monotonic() - self._last_vector_command_at
        remaining = min_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        return None

    def _interpolation_t(self, step_index: int) -> float:
        """Map the current step index onto the configured action-mixing ramp."""

        schedule = self.config.training
        if schedule.t_ramp_steps <= 0:
            return schedule.t_end
        alpha = min(max(step_index, 0), schedule.t_ramp_steps) / float(schedule.t_ramp_steps)
        return float(schedule.t_start + alpha * (schedule.t_end - schedule.t_start))

    def _maybe_train(self) -> TrainingSummary:
        """Run periodic policy updates once replay has enough data."""

        replay_size = len(self.model.replay_buffer)
        training = self.config.training
        if replay_size < training.min_replay_size or training.train_every_steps <= 0:
            return TrainingSummary(triggered=False, replay_size=replay_size)
        if (self.step_index + 1) % training.train_every_steps != 0:
            return TrainingSummary(triggered=False, replay_size=replay_size)

        pi, loss = self.model.fit(batch_size=training.batch_size, iters=training.fit_iters)
        memorized = None
        if training.memorize_every_steps > 0 and (self.step_index + 1) % training.memorize_every_steps == 0:
            # SSR memorization is less frequent than SGD-style fitting because
            # it is materially more expensive and does not need to happen every
            # environment step.
            memorized = min(training.memorize_n, replay_size)
            self.model.memorize(n=memorized, random_idx=training.memorize_random_idx, disable_tqdm=True)
        return TrainingSummary(
            triggered=True,
            replay_size=replay_size,
            pi=float(pi),
            loss=float(loss),
            memorized=memorized,
        )

    def _write_run_metadata(self, *, initial_reward: float) -> None:
        """Write the initial run metadata file as soon as a run starts."""

        if self.paths is None:
            return None
        self._run_metadata = {
            "experiment_name": self.paths.run_dir.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "reward_prompt_id": self.config.reward.prompt_id,
            "host": self.config.picar.host,
            "history_window": self.config.history_window,
            "initial_reward": initial_reward,
        }
        write_metadata(self.paths, self._run_metadata)
        return None
