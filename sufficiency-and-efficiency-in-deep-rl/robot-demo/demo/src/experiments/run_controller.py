from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from env import (
    EnvConfig,
    FrozenVLMRewardScorer,
    PiCarGymEnv,
    RewardConfig,
    RewardPromptRegistry,
    RewardPromptSpec,
    SpeechStreamConfig,
    TrainingConfig,
)
from env.picar_bridge import PiCarControlClient
from env.speech_stream import ContinuousSpeechStream
from model import ModelConfig, PiCarActionModel, TransitionReplayBuffer
from speech.audio_io import AudioIO
from speech.config import SpeechConfig
from speech.stt import FasterWhisperSTT
from speech.tts import PiperTTS

from .observation_store import ObservationStore
from .schemas import ExperimentRunConfig, ExperimentRunSummary
from .snapshot_store import SnapshotStore, resolve_snapshot_path


class _Speaker:
    def __init__(self, *, tts: PiperTTS, audio: AudioIO):
        self.tts = tts
        self.audio = audio

    def speak(self, text: str) -> None:
        synthesis = self.tts.synthesize_to_tempfile(text)
        self.audio.play_wav(synthesis.audio_path)
        return None


def default_demo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_training_config(config: ExperimentRunConfig) -> tuple[TrainingConfig, bool, bool]:
    _validate_config(config)
    training_enabled = config.phase in ("tune", "retask")
    if config.phase == "init":
        fixed_t = 0.0 if config.fixed_t is None else float(config.fixed_t)
        training = TrainingConfig(
            train_every_steps=0,
            min_replay_size=max(1, int(config.min_replay_size)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=max(1, int(config.fit_iters)),
            memorize_every_steps=0,
            memorize_n=max(1, int(config.memorize_n)),
            memorize_random_idx=bool(config.memorize_random_idx),
            fixed_t=fixed_t,
            t_start=0.0,
            t_end=0.0,
            t_ramp_steps=1,
        )
        return training, training_enabled, False

    if config.fixed_t is not None:
        fixed_t = float(config.fixed_t)
        training = TrainingConfig(
            train_every_steps=max(1, int(config.train_every_steps)),
            min_replay_size=max(1, int(config.min_replay_size)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=max(1, int(config.fit_iters)),
            memorize_every_steps=max(1, int(config.memorize_every_steps)),
            memorize_n=int(config.memorize_n),
            memorize_random_idx=bool(config.memorize_random_idx),
            fixed_t=fixed_t,
            t_start=fixed_t,
            t_end=fixed_t,
            t_ramp_steps=1,
        )
        return training, training_enabled, False

    if config.phase == "retask":
        training = TrainingConfig(
            train_every_steps=max(1, int(config.train_every_steps)),
            min_replay_size=max(1, int(config.min_replay_size)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=max(1, int(config.fit_iters)),
            memorize_every_steps=max(1, int(config.memorize_every_steps)),
            memorize_n=int(config.memorize_n),
            memorize_random_idx=bool(config.memorize_random_idx),
            fixed_t=1.0,
            t_start=1.0,
            t_end=1.0,
            t_ramp_steps=1,
        )
        return training, training_enabled, False

    ramp_steps = max(1, int(math.ceil(1.0 / float(config.t_step))))
    training = TrainingConfig(
        train_every_steps=max(1, int(config.train_every_steps)),
        min_replay_size=max(1, int(config.min_replay_size)),
        batch_size=max(1, int(config.batch_size)),
        fit_iters=max(1, int(config.fit_iters)),
        memorize_every_steps=max(1, int(config.memorize_every_steps)),
        memorize_n=int(config.memorize_n),
        memorize_random_idx=bool(config.memorize_random_idx),
        fixed_t=None,
        t_start=0.0,
        t_end=1.0,
        t_ramp_steps=ramp_steps,
    )
    return training, training_enabled, True


def run_experiment(config: ExperimentRunConfig) -> ExperimentRunSummary:
    _validate_config(config)
    run_uuid = config.run_uuid or str(uuid4())
    demo_root = default_demo_root()
    data_root = (config.data_root or (demo_root / "data")).resolve()
    model_root = (config.model_root or (demo_root / "model")).resolve()
    data_run_dir = data_root / run_uuid
    model_run_dir = model_root / run_uuid
    data_run_dir.mkdir(parents=True, exist_ok=False)
    model_run_dir.mkdir(parents=True, exist_ok=False)

    observation_store = ObservationStore(data_run_dir)
    snapshot_store = SnapshotStore(model_run_dir, max_keep=config.snapshot_keep)
    training_config, training_enabled, t_is_traversing = build_training_config(config)

    started_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "uuid": run_uuid,
        "phase": config.phase,
        "started_at": started_at,
        "load_snapshot": str(config.load_snapshot) if config.load_snapshot is not None else None,
        "reward_prompt": config.reward_prompt,
        "training": asdict(training_config),
        "snapshot_keep": int(config.snapshot_keep),
    }
    observation_store.write_run_metadata(run_metadata)
    snapshot_store.write_run_metadata(run_metadata)

    print(f"[experiment] uuid={run_uuid}", flush=True)
    print(f"[experiment] data_dir={data_run_dir}", flush=True)
    print(f"[experiment] model_dir={model_run_dir}", flush=True)

    replay_buffer = TransitionReplayBuffer(capacity=10_000)
    model_config = ModelConfig(model_dir=model_root)
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=model_config,
    )
    if config.load_snapshot is not None:
        snapshot_path = resolve_snapshot_path(config.load_snapshot)
        load_result = snapshot_store.load_into_model(snapshot_path=snapshot_path, model=model)
        print(
            f"[snapshot] loaded={load_result.path} "
            f"trainable_keys={len(load_result.loaded_trainable_keys)} ssr={load_result.has_ssr_state}",
            flush=True,
        )

    reward_registry, reward_prompt_id = _build_reward_registry(config.reward_prompt)
    reward_scorer = FrozenVLMRewardScorer(
        RewardConfig(prompt_id=reward_prompt_id),
        model_config=ModelConfig(model_dir=model_root),
        registry=reward_registry,
    )

    speech_config = SpeechConfig(model_dir=model_root)
    speech_stream = ContinuousSpeechStream(SpeechStreamConfig(), transcriber=FasterWhisperSTT(speech_config))
    speaker = _Speaker(tts=PiperTTS(speech_config), audio=AudioIO(speech_config.audio))
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=PiCarControlClient(EnvConfig().picar),
        speech_stream=speech_stream,
        speaker=speaker,
        config=EnvConfig(
            data_dir=data_run_dir / ".env_internal",
            enable_persistence=False,
            reward=RewardConfig(prompt_id=reward_prompt_id),
            speech=SpeechStreamConfig(),
            training=training_config,
        ),
    )

    steps_completed = 0
    try:
        initial_observation = env.reset()
        observation_store.append_observation(
            observation=initial_observation,
            reward=None,
            user_texts=[],
            action=None,
            training=None,
            action_receipt=None,
            source="reset",
        )
        if not training_enabled:
            snapshot_path = snapshot_store.save_snapshot(
                model=model,
                replay_buffer=model.replay_buffer,
                step_index=int(initial_observation.step_index),
                t=float(initial_observation.t),
                reason="initial",
            )
            print(f"[snapshot] saved={snapshot_path.name}", flush=True)
        if t_is_traversing:
            _log_t_progress(step_index=int(initial_observation.step_index), t=float(initial_observation.t))

        while True:
            next_observation, reward, done, info = env.step()
            steps_completed = int(next_observation.step_index)
            observation_store.append_observation(
                observation=next_observation,
                reward=float(reward),
                user_texts=[str(item) for item in info.get("user_texts", [])],
                action=info.get("action"),
                training=info.get("training"),
                action_receipt=info.get("action_receipt"),
                source="step",
            )
            if t_is_traversing and steps_completed % int(config.t_log_every) == 0:
                _log_t_progress(step_index=steps_completed, t=float(next_observation.t))

            training_summary = info.get("training")
            memorized = getattr(training_summary, "memorized", None)
            if training_enabled and memorized is not None and memorized > 0:
                snapshot_path = snapshot_store.save_snapshot(
                    model=model,
                    replay_buffer=model.replay_buffer,
                    step_index=steps_completed,
                    t=float(next_observation.t),
                    reason="memorize",
                    memorize_count=int(memorized),
                )
                print(f"[snapshot] saved={snapshot_path.name}", flush=True)

            if done:
                break
    except KeyboardInterrupt:
        print("[experiment] Ctrl-C received. Stopping run.", flush=True)
    finally:
        env.close()
        ended_at = datetime.now(timezone.utc).isoformat()
        final_metadata = {
            **run_metadata,
            "ended_at": ended_at,
            "steps_completed": int(steps_completed),
        }
        observation_store.write_run_metadata(final_metadata)
        snapshot_store.write_run_metadata(final_metadata)
    return ExperimentRunSummary(
        run_uuid=run_uuid,
        phase=config.phase,
        steps_completed=int(steps_completed),
        data_run_dir=data_run_dir,
        model_run_dir=model_run_dir,
    )


def _build_reward_registry(prompt_text: str) -> tuple[RewardPromptRegistry, str]:
    default_prompt = "find the red ball"
    if prompt_text.strip().lower() == default_prompt:
        spec = RewardPromptSpec(
            prompt_id="reward_prompt_1",
            prompt_text=(
                "You are a reinforcement learning reward scorer for a PiCar-V robot.\n"
                "Return JSON with exactly one key: `reward`.\n"
                "Task: find the red ball.\n"
                "If the task is not visible, return 0.\n"
                "If the task is strongly satisfied, return a value near 10.\n"
                "Use values in [0, 10]."
            ),
            min_reward=0.0,
            max_reward=10.0,
        )
    else:
        spec = RewardPromptSpec(
            prompt_id="reward_prompt_custom",
            prompt_text=(
                "You are a reinforcement learning reward scorer for a PiCar-V robot.\n"
                "Return JSON with exactly one key: `reward`.\n"
                f"Task: {prompt_text.strip()}.\n"
                "If the task is not visible or not being satisfied, return 0.\n"
                "If the task is strongly satisfied, return a value near 10.\n"
                "Use values in [0, 10]."
            ),
            min_reward=0.0,
            max_reward=10.0,
        )
    return RewardPromptRegistry(prompts={spec.prompt_id: spec}), spec.prompt_id


def _log_t_progress(*, step_index: int, t: float) -> None:
    t_clamped = min(max(float(t), 0.0), 1.0)
    width = 24
    filled = int(round(width * t_clamped))
    bar = "#" * filled + "-" * (width - filled)
    print(f"[t] step={step_index:06d} t={t_clamped:.3f} [{bar}]", flush=True)


def _validate_config(config: ExperimentRunConfig) -> None:
    if config.fixed_t is not None and not (0.0 <= float(config.fixed_t) <= 1.0):
        raise ValueError(f"--fixed-t must be in [0, 1], got {config.fixed_t}")
    if float(config.t_step) <= 0.0:
        raise ValueError(f"--t-step must be positive, got {config.t_step}")
    if int(config.t_log_every) < 1:
        raise ValueError(f"--t-log-every must be >= 1, got {config.t_log_every}")
    if int(config.snapshot_keep) < 1:
        raise ValueError(f"--snapshot-keep must be >= 1, got {config.snapshot_keep}")
    return None
