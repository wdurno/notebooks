"""Phase 1 VLM-only robot data generation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from picar_kl.actions import action_distribution_to_vector, distribution_to_action_name
from picar_kl.context import ContextConfig, EpisodeContext
from picar_kl.io.observation_store import Phase1ObservationStore
from picar_kl.latency import LatencyEvent, LatencyTimer
from picar_kl.records import ActionRecord, Phase1ObservationRecord
from picar_kl.reward import ConstantRewardScorer, RewardResult
from picar_kl.vlm.control import VLMDecision


class Phase1Robot(Protocol):
    def capture_image(self) -> Any:
        ...

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        ...


class Phase1VLMController(Protocol):
    def decide(self, *, image_rgb: Any, messages: list[dict[str, Any]]) -> VLMDecision:
        ...


class SpeechSource(Protocol):
    def drain_texts(self) -> list[str]:
        ...


class Speaker(Protocol):
    def speak(self, text: str) -> None:
        ...


class RewardScorer(Protocol):
    def score(self, image_rgb: Any) -> RewardResult:
        ...


@dataclass(frozen=True)
class Phase1RunConfig:
    data_root: Path
    task_prompt: str = "find the red ball"
    run_uuid: str | None = None
    max_steps: int | None = None
    speak_generated_text: bool = True
    context: ContextConfig = field(default_factory=ContextConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase1RunSummary:
    run_uuid: str
    run_dir: Path
    steps_completed: int
    status: str = "completed"


class NoSpeechSource:
    def drain_texts(self) -> list[str]:
        return []


class NoSpeaker:
    def speak(self, text: str) -> None:
        del text
        return None


class FixedActionVLMController:
    """Small controller for smoke tests before real Qwen integration."""

    def __init__(self, action_name: str = "look-forward", *, generated_text: str = ""):
        self.action_name = action_name
        self.generated_text = generated_text

    def decide(self, *, image_rgb: Any, messages: list[dict[str, Any]]) -> VLMDecision:
        del image_rgb, messages
        return VLMDecision.from_action_name(
            self.action_name,
            generated_text=self.generated_text,
            metadata={"controller": "fixed-action"},
        )


def run_phase1(
    config: Phase1RunConfig,
    *,
    robot: Phase1Robot,
    vlm: Phase1VLMController,
    speech_source: SpeechSource | None = None,
    speaker: Speaker | None = None,
    reward_scorer: RewardScorer | None = None,
    prompt_tokenizer: Any | None = None,
) -> Phase1RunSummary:
    run_uuid = config.run_uuid or str(uuid4())
    run_dir = Path(config.data_root) / run_uuid
    store = Phase1ObservationStore(run_dir)
    speech_source = speech_source or NoSpeechSource()
    speaker = speaker or NoSpeaker()
    reward_scorer = reward_scorer or ConstantRewardScorer(task_text=config.task_prompt)
    context = EpisodeContext(config=config.context)
    started_at = datetime.now(timezone.utc).isoformat()
    base_metadata = {
        "uuid": run_uuid,
        "phase": "phase1",
        "task_prompt": config.task_prompt,
        "started_at": started_at,
        "created_at": started_at,
        "status": "running",
        "steps_completed": 0,
        "history_window": config.context.history_window,
        "prompt_token_window": config.context.prompt_token_window,
        "metadata": dict(config.metadata),
    }
    store.write_run_metadata(base_metadata)

    steps_completed = 0
    status = "completed"
    step_iter = range(int(config.max_steps)) if config.max_steps is not None else count()
    try:
        for step_index in step_iter:
            latency_events: list[LatencyEvent] = []
            user_texts = speech_source.drain_texts()
            with LatencyTimer("capture_image", metadata={"step_index": step_index}) as timer:
                image = robot.capture_image()
            latency_events.append(_event(timer))
            image_rgb = getattr(image, "image_rgb", image)

            with LatencyTimer("reward_scoring", metadata={"step_index": step_index}) as timer:
                reward_result = reward_scorer.score(image_rgb)
            latency_events.append(_event(timer))

            with LatencyTimer("context_render", metadata={"step_index": step_index}) as timer:
                context_render = context.add_observation(
                    user_texts=user_texts,
                    reward_result=reward_result,
                    step_index=step_index,
                    tokenizer=prompt_tokenizer,
                )
            latency_events.append(_event(timer))
            messages = context_render.messages

            with LatencyTimer("vlm_decision", metadata={"step_index": step_index}) as timer:
                decision = vlm.decide(image_rgb=image_rgb, messages=messages)
            latency_events.append(_event(timer))

            action_vector = action_distribution_to_vector(decision.action_distribution)
            with LatencyTimer("apply_action", metadata={"step_index": step_index}) as timer:
                action_receipt = robot.apply_vector(action_vector)
            latency_events.append(_event(timer))

            if decision.generated_text and config.speak_generated_text:
                with LatencyTimer("speaker", metadata={"step_index": step_index}) as timer:
                    speaker.speak(decision.generated_text)
                latency_events.append(_event(timer))

            action_name = distribution_to_action_name(decision.action_distribution)
            action = ActionRecord.from_distribution(
                decision.action_distribution,
                generated_text=decision.generated_text,
                executed_vector=action_vector,
                source="vlm",
                metadata={
                    "raw_response": decision.raw_response,
                    "decision_metadata": dict(decision.metadata),
                    "action_receipt": action_receipt,
                },
            )
            context.add_assistant_action(
                action_name=action_name,
                generated_text=decision.generated_text,
                reward_result=reward_result,
            )
            record = Phase1ObservationRecord(
                run_uuid=run_uuid,
                run_dir=run_dir,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="step",
                step_index=step_index,
                image_path=None,
                messages=messages,
                user_texts=user_texts,
                action=action,
                reward=float(reward_result.clipped_reward),
                done=False,
                metadata={
                    "task_prompt": config.task_prompt,
                    "action_name": action_name,
                    "reward_result": reward_result.to_dict(),
                    "context": {
                        "history_window": config.context.history_window,
                        **context_render.metadata,
                    },
                },
                latency_events=tuple(latency_events),
            )
            store.append(record, image_rgb=image_rgb)
            steps_completed = step_index + 1
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        final_metadata = dict(base_metadata)
        final_metadata.update(
            {
                "status": status,
                "steps_completed": steps_completed,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        store.write_run_metadata(final_metadata)

    return Phase1RunSummary(
        run_uuid=run_uuid,
        run_dir=run_dir,
        steps_completed=steps_completed,
        status=status,
    )


def _event(timer: LatencyTimer) -> LatencyEvent:
    if timer.event is None:
        raise RuntimeError("Latency timer did not produce an event")
    return timer.event
