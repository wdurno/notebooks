from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Any

import torch

try:
    from src.model.action_space import ACTION_NAMES
    from src.model import ModelObservation, Transition
except ModuleNotFoundError:
    from model.action_space import ACTION_NAMES
    from model import ModelObservation, Transition

from .config import EnvConfig
from .persistence import append_jsonl, create_experiment_paths, save_frame_array, save_model_artifacts, save_replay_snapshot, write_metadata
from .schemas import ExperimentPaths, RewardResult, TrainingSummary

LOGGER = logging.getLogger(__name__)

MALFORMED_JSON_REWARD_PENALTY = -1.0
COMMAND_FOLLOW_COMPLETION_REWARD = 2.0
COMMAND_IGNORED_PENALTY = -2.0


@dataclass
class PendingUserCommand:
    action_name: str
    remaining_steps: int
    source_text: str


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


def build_goal_system_message(*, reward_result: RewardResult) -> dict[str, Any]:
    task_text = str(reward_result.metadata.get("task_text") or "").strip()
    if not task_text:
        task_text = f"task linked to reward prompt `{reward_result.prompt_id}`"
    text = (
        "Persistent goals:\n"
        f"1) Primary task: {task_text}.\n"
        "2) Always follow operator commands faithfully; if a command requests repeated "
        "actions, continue until completion.\n"
        "3) If the operator asks a question, provide a direct spoken answer in `say`."
    )
    return {
        "role": "system",
        "content": [{"type": "text", "text": text}],
    }


def _is_goal_system_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = str(item.get("text") or "")
        if text.startswith("Persistent goals:"):
            return True
    return False


def _normalize_command_text(text: str) -> str:
    return str(text or "").strip().lower().replace("_", "-")


def _extract_action_command(text: str) -> str | None:
    normalized = _normalize_command_text(text)
    if not normalized:
        return None
    pattern_map = {
        "drive-left": (r"\bdrive\s+left\b", r"\bgo\s+left\b", r"\bturn\s+left\b"),
        "drive-right": (r"\bdrive\s+right\b", r"\bgo\s+right\b", r"\bturn\s+right\b"),
        "drive-forward": (r"\bdrive\s+forward\b", r"\bdrive\s+forwards\b", r"\bgo\s+forward\b"),
        "drive-backward": (r"\bdrive\s+backward\b", r"\bdrive\s+backwards\b", r"\bgo\s+backward\b", r"\bgo\s+back\b"),
        "look-left": (r"\blook\s+left\b",),
        "look-right": (r"\blook\s+right\b",),
        "look-up": (r"\blook\s+up\b",),
        "look-forward": (r"\blook\s+forward\b", r"\blook\s+ahead\b"),
    }
    for action_name in ACTION_NAMES:
        for pattern in pattern_map.get(action_name, ()):
            if re.search(pattern, normalized):
                return action_name
    return None


def _extract_command_repetitions(text: str) -> int:
    normalized = _normalize_command_text(text)
    numeric_match = re.search(r"\b(\d+)\b", normalized)
    if numeric_match:
        return max(1, int(numeric_match.group(1)))
    word_to_int = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, value in word_to_int.items():
        if re.search(rf"\b{word}\b", normalized):
            return value
    return 1


def extract_pending_user_command(user_texts: list[str]) -> PendingUserCommand | None:
    for text in reversed(user_texts):
        action_name = _extract_action_command(text)
        if action_name is None:
            continue
        repetitions = _extract_command_repetitions(text)
        return PendingUserCommand(
            action_name=action_name,
            remaining_steps=repetitions,
            source_text=str(text),
        )
    return None


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
        self._pending_user_command: PendingUserCommand | None = None
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
        if self.config.enable_persistence:
            self.paths = create_experiment_paths(self.config.data_dir, self.config.experiment_name_prefix)
        else:
            self.paths = None
        self._run_metadata = {}
        self._last_vector_command_at = None
        self._pending_user_command = None
        if self.speech_stream is not None:
            self.speech_stream.start()
        self._set_model_inference_mode()
        frame = self.picar_client.reset()
        reward_result = self.reward_scorer.score(frame)
        user_texts = [event.text for event in self._drain_speech_events()]
        if user_texts:
            LOGGER.info("[stt] step=%d drained_texts=%r", self.step_index, user_texts)
        user_message = build_operator_message(
            user_texts=user_texts,
            reward_result=reward_result,
            step_index=self.step_index,
            last_reward=self.last_reward,
        )
        messages = self._with_goal_message([user_message], reward_result=reward_result)
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
        self._set_model_inference_mode()
        user_texts = [event.text for event in self._drain_speech_events()]
        if user_texts:
            LOGGER.info("[stt] step=%d drained_texts=%r", self.step_index, user_texts)
            command = extract_pending_user_command(user_texts)
            if command is not None:
                self._pending_user_command = command
                LOGGER.info(
                    "[command] step=%d pending action=%s remaining_steps=%d source=%r",
                    self.step_index,
                    command.action_name,
                    command.remaining_steps,
                    command.source_text,
                )
        current_frame = self.picar_client.capture_image()
        reward_result = self.reward_scorer.score(current_frame)
        user_message = build_operator_message(
            user_texts=user_texts,
            reward_result=reward_result,
            step_index=self.step_index,
            last_reward=self.last_reward,
        )
        if (
            self.history
            and self.history[-1].get("role") == "user"
            and self.history[-1].get("content") == user_message.get("content")
        ):
            current_messages = self._with_goal_message(self.history, reward_result=reward_result)
        else:
            current_messages = self._with_goal_message(self.history + [user_message], reward_result=reward_result)
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
        with torch.inference_mode():
            action = self.model.forward(observation)
        action_debug = action.debug if isinstance(action.debug, dict) else {}
        json_expected = bool(action_debug.get("json_expected", False))
        json_valid = bool(action_debug.get("json_valid", True))
        malformed_json = json_expected and not json_valid
        malformed_adjustment = MALFORMED_JSON_REWARD_PENALTY if malformed_json else 0.0
        command_adjustment, command_status = self._score_command_following(action.agentic_action_name)
        reward_adjustment = float(malformed_adjustment + command_adjustment)
        step_reward = float(reward_result.clipped_reward + reward_adjustment)
        LOGGER.info(
            "[model] step=%d mode=%s action=%s json_valid=%s say=%r",
            self.step_index,
            "mixed" if observation.t < 1.0 else "actor_only",
            action.agentic_action_name,
            json_valid,
            action.generated_text,
        )
        LOGGER.debug(
            "[model-debug] step=%d raw_generation=%r",
            self.step_index,
            action_debug.get("raw_generation"),
        )
        LOGGER.info(
            "[reward] step=%d base=%.3f malformed_adj=%.3f command_adj=%.3f final=%.3f command_status=%s",
            self.step_index,
            float(reward_result.clipped_reward),
            float(malformed_adjustment),
            float(command_adjustment),
            float(step_reward),
            command_status,
        )
        LOGGER.debug("[reward-debug] step=%d raw_text=%r", self.step_index, reward_result.raw_text)
        if action.generated_text and self.speaker is not None and not malformed_json:
            LOGGER.info("[tts] step=%d speaking text=%r", self.step_index, action.generated_text)
            self.speaker.speak(action.generated_text)
        elif malformed_json:
            LOGGER.debug("[tts] step=%d speech_suppressed reason=malformed_json", self.step_index)
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
        self.history = self._with_goal_message(current_messages + [assistant_message], reward_result=reward_result)

        next_frame = self.picar_client.capture_image()
        next_observation = ModelObservation(
            image_rgb=next_frame,
            messages=self.history,
            t=self._interpolation_t(self.step_index + 1),
            last_reward=step_reward,
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
            reward=step_reward,
            next_observation=next_observation,
            done=False,
            target_text=target_text,
            logp_beta_sum=action.logp_beta_sum,
            target_action_name=action.agentic_action_name,
            agentic_action_vector=action.agentic_action_vector,
            actor_action_vector=action.actor_action_vector,
            metadata={
                "reward_prompt_id": reward_result.prompt_id,
                "generated_text": action.generated_text,
                "agentic_action_name": action.agentic_action_name,
                "execution_mode": execution_mode,
                "malformed_json": malformed_json,
                "command_status": command_status,
                "command_adjustment": command_adjustment,
                "malformed_json_adjustment": malformed_adjustment,
                "reward_adjustment": reward_adjustment,
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
                    "reward": step_reward,
                    "reward_base": float(reward_result.clipped_reward),
                    "reward_malformed_adjustment": float(malformed_adjustment),
                    "reward_command_adjustment": float(command_adjustment),
                    "reward_adjustment": reward_adjustment,
                    "command_status": command_status,
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
                        "malformed_json": malformed_json,
                    },
                    "training": asdict(training_summary),
                    "frame_path": str(frame_path) if frame_path is not None else None,
                },
            )

        self.last_reward = step_reward
        self.step_index += 1
        info = {
            "reward_result": reward_result,
            "reward_malformed_adjustment": float(malformed_adjustment),
            "reward_command_adjustment": float(command_adjustment),
            "reward_adjustment": reward_adjustment,
            "command_status": command_status,
            "malformed_json": malformed_json,
            "action": action,
            "user_texts": user_texts,
            "training": training_summary,
            "action_receipt": action_receipt,
        }
        return next_observation, step_reward, False, info

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

    def _with_goal_message(
        self,
        history: list[dict[str, Any]],
        *,
        reward_result: RewardResult,
    ) -> list[dict[str, Any]]:
        goal_message = build_goal_system_message(reward_result=reward_result)
        non_goal_messages = [message for message in history if not _is_goal_system_message(message)]
        if self.config.history_window <= 1:
            return [goal_message]
        tail_messages = trim_history(non_goal_messages, self.config.history_window - 1)
        return [goal_message] + tail_messages

    def _score_command_following(self, action_name: str) -> tuple[float, str]:
        pending = self._pending_user_command
        if pending is None:
            return 0.0, "none"
        if action_name == pending.action_name:
            pending.remaining_steps -= 1
            if pending.remaining_steps <= 0:
                self._pending_user_command = None
                LOGGER.info(
                    "[reward-command] step=%d status=completed action=%s delta=%.1f source=%r",
                    self.step_index,
                    action_name,
                    COMMAND_FOLLOW_COMPLETION_REWARD,
                    pending.source_text,
                )
                return COMMAND_FOLLOW_COMPLETION_REWARD, "completed"
            return 0.0, f"in_progress({pending.remaining_steps})"
        LOGGER.info(
            "[reward-command] step=%d status=ignored expected=%s got=%s delta=%.1f source=%r",
            self.step_index,
            pending.action_name,
            action_name,
            COMMAND_IGNORED_PENALTY,
            pending.source_text,
        )
        return COMMAND_IGNORED_PENALTY, f"ignored(expected={pending.action_name})"

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
        if schedule.fixed_t is not None:
            fixed_t = float(schedule.fixed_t)
            if fixed_t < 0.0 or fixed_t > 1.0:
                raise ValueError(f"training.fixed_t must be in [0, 1], got {fixed_t}")
            return fixed_t
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

        self._set_model_optimization_mode()
        try:
            pi, loss = self.model.fit(batch_size=training.batch_size, iters=training.fit_iters)
            memorized = None
            if training.memorize_every_steps > 0 and (self.step_index + 1) % training.memorize_every_steps == 0:
                # SSR memorization is less frequent than SGD-style fitting because
                # it is materially more expensive and does not need to happen every
                # environment step.
                if training.memorize_n < 0:
                    memorize_count = replay_size
                else:
                    memorize_count = min(training.memorize_n, replay_size)

                if memorize_count > 0:
                    self.model.memorize(
                        n=memorize_count,
                        random_idx=training.memorize_random_idx,
                        disable_tqdm=True,
                    )
                    # Once observations are committed into SSR sufficient statistics,
                    # they can be dropped from replay to keep memory bounded.
                    replay_buffer = self.model.replay_buffer
                    if hasattr(replay_buffer, "clear"):
                        replay_buffer.clear(memorize_count)
                    memorized = memorize_count
        finally:
            self._set_model_inference_mode()
        return TrainingSummary(
            triggered=True,
            replay_size=replay_size,
            pi=float(pi),
            loss=float(loss),
            memorized=memorized,
        )

    def _set_model_inference_mode(self) -> None:
        set_inference_mode = getattr(self.model, "set_inference_mode", None)
        if callable(set_inference_mode):
            set_inference_mode()
            return None
        if hasattr(self.model, "eval"):
            self.model.eval()
        return None

    def _set_model_optimization_mode(self) -> None:
        set_optimization_mode = getattr(self.model, "set_optimization_mode", None)
        if callable(set_optimization_mode):
            set_optimization_mode()
            return None
        if hasattr(self.model, "train"):
            self.model.train()
        return None

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
            "t_mode": "fixed" if self.config.training.fixed_t is not None else "ramp",
            "fixed_t": self.config.training.fixed_t,
            "t_start": self.config.training.t_start,
            "t_end": self.config.training.t_end,
            "t_ramp_steps": self.config.training.t_ramp_steps,
            "initial_reward": initial_reward,
        }
        write_metadata(self.paths, self._run_metadata)
        return None
