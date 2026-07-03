"""Live Phase 2 robot loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from picar_kl.actions import action_distribution_to_vector, distribution_to_action_name
from picar_kl.context import ContextConfig, EpisodeContext
from picar_kl.io.observation_store import Phase1ObservationStore
from picar_kl.latency import LatencyEvent, LatencyTimer
from picar_kl.models.visual import VisualTokenEncoding, VisualTokenEncoder
from picar_kl.phase1.run import NoSpeaker, NoSpeechSource, RewardScorer, Speaker, SpeechSource
from picar_kl.phase2.runtime import LoadedPhase2Policy, load_phase2_policy
from picar_kl.records import ActionRecord, Phase1ObservationRecord
from picar_kl.reward import ConstantRewardScorer, RewardResult
from picar_kl.vlm.control import VLMDecision


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for phase 2 live runtime") from exc
    return torch


class Phase2Robot(Protocol):
    def capture_image(self) -> Any:
        ...

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        ...


class Phase2VLMController(Protocol):
    def decide(self, *, image_rgb: Any, messages: list[dict[str, Any]]) -> VLMDecision:
        ...


@dataclass(frozen=True)
class Phase2LiveRunConfig:
    data_root: Path
    task_prompt: str = "find the red ball"
    checkpoint_root: Path = Path("artifacts/models/phase2")
    fit_id: str | None = None
    checkpoint_path: Path | None = None
    device: str = "auto"
    run_uuid: str | None = None
    max_steps: int | None = None
    k: int | None = None
    speak_generated_text: bool = True
    context: ContextConfig = field(default_factory=ContextConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase2LiveRunSummary:
    run_uuid: str
    run_dir: Path
    steps_completed: int
    status: str = "completed"
    fit_id: str = ""


@dataclass(frozen=True)
class _HistoryStep:
    visual_tokens: np.ndarray
    action_distribution: tuple[float, ...]
    action_name: str


def run_phase2_live(
    config: Phase2LiveRunConfig,
    *,
    robot: Phase2Robot,
    vlm: Phase2VLMController,
    visual_encoder: VisualTokenEncoder,
    loaded_policy: LoadedPhase2Policy | None = None,
    speech_source: SpeechSource | None = None,
    speaker: Speaker | None = None,
    reward_scorer: RewardScorer | None = None,
    prompt_tokenizer: Any | None = None,
) -> Phase2LiveRunSummary:
    loaded_policy = loaded_policy or load_phase2_policy(
        checkpoint_root=config.checkpoint_root,
        fit_id=config.fit_id,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )
    prediction_steps = loaded_policy.artifact.prediction_steps
    context_steps = loaded_policy.artifact.context_steps
    k = prediction_steps if config.k is None else int(config.k)
    if k != prediction_steps:
        raise ValueError(f"Phase 2 live runtime requires K == prediction_steps; got K={k}, prediction_steps={prediction_steps}")

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
        "phase": "phase2-live",
        "task_prompt": config.task_prompt,
        "started_at": started_at,
        "created_at": started_at,
        "status": "running",
        "steps_completed": 0,
        "fit_id": loaded_policy.artifact.fit_id,
        "checkpoint_path": str(loaded_policy.artifact.checkpoint_path),
        "context_steps": context_steps,
        "prediction_steps": prediction_steps,
        "k": k,
        "device": loaded_policy.device,
        "history_window": config.context.history_window,
        "prompt_token_window": config.context.prompt_token_window,
        "metadata": dict(config.metadata),
    }
    store.write_run_metadata(base_metadata)

    history: deque[_HistoryStep] = deque(maxlen=context_steps)
    cycle_targets: list[np.ndarray] = []
    cycle_previous_actions: list[tuple[float, ...]] = []
    cycle_index = -1
    cycle_step_index = 0
    current_vlm_decision: VLMDecision | None = None
    last_generated_text = ""
    previous_action_distribution: tuple[float, ...] | None = None
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

            with LatencyTimer("visual_encoding", metadata={"step_index": step_index}) as timer:
                encoding = visual_encoder.encode_image(image_rgb)
            latency_events.append(_event(timer))

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

            mode = "bootstrap" if len(history) < context_steps else "lstm"
            if mode == "bootstrap":
                with LatencyTimer("vlm_bootstrap_decision", metadata={"step_index": step_index}) as timer:
                    current_vlm_decision = vlm.decide(image_rgb=image_rgb, messages=messages)
                latency_events.append(_event(timer))
                action_distribution = current_vlm_decision.action_distribution
                action_source = "vlm-bootstrap"
                cycle_step_index = 0
            else:
                if cycle_step_index == 0:
                    cycle_index += 1
                    cycle_targets = []
                    cycle_previous_actions = []
                    with LatencyTimer("vlm_head_refresh", metadata={"step_index": step_index, "cycle_index": cycle_index}) as timer:
                        current_vlm_decision = vlm.decide(image_rgb=image_rgb, messages=messages)
                    latency_events.append(_event(timer))

                if previous_action_distribution is None:
                    raise RuntimeError("Phase 2 LSTM step has no previous action distribution")
                cycle_targets.append(np.asarray(encoding.tokens))
                cycle_previous_actions.append(previous_action_distribution)
                with LatencyTimer(
                    "lstm_action",
                    metadata={"step_index": step_index, "cycle_index": cycle_index, "cycle_step_index": cycle_step_index},
                ) as timer:
                    action_distribution = _predict_cycle_action(
                        loaded_policy=loaded_policy,
                        prefix=list(history),
                        target_visual_tokens=cycle_targets,
                        target_previous_actions=cycle_previous_actions,
                    )
                latency_events.append(_event(timer))
                action_source = "phase2-lstm"

            action_name = distribution_to_action_name(action_distribution)
            action_vector = action_distribution_to_vector(action_distribution)
            with LatencyTimer("apply_action", metadata={"step_index": step_index, "source": action_source}) as timer:
                action_receipt = robot.apply_vector(action_vector)
            latency_events.append(_event(timer))

            generated_text = current_vlm_decision.generated_text if current_vlm_decision is not None else ""
            if generated_text and generated_text != last_generated_text and config.speak_generated_text:
                with LatencyTimer("speaker", metadata={"step_index": step_index}) as timer:
                    speaker.speak(generated_text)
                latency_events.append(_event(timer))
                last_generated_text = generated_text

            action = ActionRecord.from_distribution(
                action_distribution,
                generated_text=generated_text,
                executed_vector=action_vector,
                source=action_source,
                metadata={
                    "fit_id": loaded_policy.artifact.fit_id,
                    "cycle_index": cycle_index,
                    "cycle_step_index": cycle_step_index,
                    "k": k,
                    "prediction_steps": prediction_steps,
                    "context_steps": context_steps,
                    "vlm_refresh": None
                    if current_vlm_decision is None
                    else {
                        "raw_response": current_vlm_decision.raw_response,
                        "decision_metadata": dict(current_vlm_decision.metadata),
                        "distribution": list(current_vlm_decision.action_distribution),
                    },
                    "visual_encoding": {
                        "shape": list(encoding.shape),
                        "dtype": encoding.dtype,
                        "metadata": dict(encoding.metadata),
                    },
                    "action_receipt": action_receipt,
                },
            )
            context.add_assistant_action(
                action_name=action_name,
                generated_text=generated_text,
                reward_result=reward_result,
            )
            record = Phase1ObservationRecord(
                run_uuid=run_uuid,
                run_dir=run_dir,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="phase2-step",
                step_index=step_index,
                image_path=None,
                messages=messages,
                user_texts=user_texts,
                action=action,
                reward=float(reward_result.clipped_reward),
                done=False,
                metadata={
                    "phase": "phase2-live",
                    "task_prompt": config.task_prompt,
                    "action_name": action_name,
                    "mode": mode,
                    "reward_result": reward_result.to_dict(),
                    "context": {
                        "history_window": config.context.history_window,
                        **context_render.metadata,
                    },
                },
                latency_events=tuple(latency_events),
            )
            store.append(record, image_rgb=image_rgb)

            history.append(
                _HistoryStep(
                    visual_tokens=np.asarray(encoding.tokens),
                    action_distribution=tuple(float(value) for value in action_distribution),
                    action_name=action_name,
                )
            )
            previous_action_distribution = tuple(float(value) for value in action_distribution)
            if mode == "lstm":
                cycle_step_index = (cycle_step_index + 1) % k
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

    return Phase2LiveRunSummary(
        run_uuid=run_uuid,
        run_dir=run_dir,
        steps_completed=steps_completed,
        status=status,
        fit_id=loaded_policy.artifact.fit_id,
    )


def _predict_cycle_action(
    *,
    loaded_policy: LoadedPhase2Policy,
    prefix: list[_HistoryStep],
    target_visual_tokens: list[np.ndarray],
    target_previous_actions: list[tuple[float, ...]],
) -> tuple[float, ...]:
    torch = _require_torch()
    device = torch.device(loaded_policy.device)
    prefix_visual = torch.as_tensor(
        np.asarray([[step.visual_tokens for step in prefix]]),
        dtype=torch.float32,
        device=device,
    )
    prefix_actions = torch.as_tensor(
        np.asarray([[step.action_distribution for step in prefix]], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    prefix_mask = torch.ones((1, len(prefix)), dtype=torch.bool, device=device)
    target_visual = torch.as_tensor(
        np.asarray([target_visual_tokens]),
        dtype=torch.float32,
        device=device,
    )
    previous_actions = torch.as_tensor(
        np.asarray([target_previous_actions], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    target_mask = torch.ones((1, len(target_visual_tokens)), dtype=torch.bool, device=device)
    with torch.no_grad():
        output = loaded_policy.model(
            prefix_visual_tokens=prefix_visual,
            prefix_actions=prefix_actions,
            prefix_step_mask=prefix_mask,
            target_visual_tokens=target_visual,
            target_previous_actions=previous_actions,
            target_step_mask=target_mask,
        )
    distribution = output.policy.probabilities[0, -1].detach().cpu().tolist()
    return tuple(float(value) for value in distribution)


def _event(timer: LatencyTimer) -> LatencyEvent:
    if timer.event is None:
        raise RuntimeError("Latency timer did not produce an event")
    return timer.event
