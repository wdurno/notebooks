from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from src.env import (
        EnvConfig,
        FrozenVLMRewardScorer,
        PiCarControlConfig,
        PiCarGymEnv,
        RewardConfig,
        RewardPromptRegistry,
        RewardPromptSpec,
        SpeechStreamConfig,
        TrainingConfig,
    )
    from src.env.picar_bridge import PiCarControlClient
    from src.env.speech_stream import ContinuousSpeechStream
    from src.model import ModelConfig, PiCarActionModel, TransitionReplayBuffer
    from src.speech.audio_io import AudioIO
    from src.speech.config import SpeechConfig
    from src.speech.stt import FasterWhisperSTT
    from src.speech.tts import PiperTTS
except ModuleNotFoundError:
    from env import (
        EnvConfig,
        FrozenVLMRewardScorer,
        PiCarControlConfig,
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
from .snapshot_store import SnapshotStore, resolve_latest_snapshot_from_model_root, resolve_snapshot_path

LOGGER = logging.getLogger(__name__)


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
            epochs=max(1, int(config.epochs)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=(None if config.fit_iters is None else max(1, int(config.fit_iters))),
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
            epochs=max(1, int(config.epochs)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=(None if config.fit_iters is None else max(1, int(config.fit_iters))),
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
            epochs=max(1, int(config.epochs)),
            batch_size=max(1, int(config.batch_size)),
            fit_iters=(None if config.fit_iters is None else max(1, int(config.fit_iters))),
            memorize_every_steps=max(1, int(config.memorize_every_steps)),
            memorize_n=int(config.memorize_n),
            memorize_random_idx=bool(config.memorize_random_idx),
            fixed_t=1.0,
            t_start=1.0,
            t_end=1.0,
            t_ramp_steps=1,
        )
        return training, training_enabled, False

    start_t = float(config.init_t)
    ramp_steps = max(1, int(math.ceil((1.0 - start_t) / float(config.t_step))))
    training = TrainingConfig(
        train_every_steps=max(1, int(config.train_every_steps)),
        min_replay_size=max(1, int(config.min_replay_size)),
        epochs=max(1, int(config.epochs)),
        batch_size=max(1, int(config.batch_size)),
        fit_iters=(None if config.fit_iters is None else max(1, int(config.fit_iters))),
        memorize_every_steps=max(1, int(config.memorize_every_steps)),
        memorize_n=int(config.memorize_n),
        memorize_random_idx=bool(config.memorize_random_idx),
        fixed_t=None,
        t_start=start_t,
        t_end=1.0,
        t_ramp_steps=ramp_steps,
    )
    return training, training_enabled, True


def run_experiment(config: ExperimentRunConfig) -> ExperimentRunSummary:
    _validate_config(config)
    run_uuid = config.run_uuid or str(uuid4())
    demo_root = default_demo_root()
    data_root = (config.data_root or (demo_root / "data")).resolve()
    model_assets_root = (demo_root / "model").resolve()
    snapshot_root = (config.model_root or model_assets_root).resolve()
    load_snapshot_path = _resolve_load_snapshot_path(config=config, snapshot_root=snapshot_root)
    data_run_dir = data_root / run_uuid
    model_run_dir = snapshot_root / run_uuid
    data_run_dir.mkdir(parents=True, exist_ok=False)
    model_run_dir.mkdir(parents=True, exist_ok=False)

    observation_store = ObservationStore(data_run_dir)
    snapshot_store = SnapshotStore(model_run_dir, max_keep=config.snapshot_keep)
    training_config, training_enabled, t_is_traversing = build_training_config(config)
    picar_host = _resolve_picar_host(config)

    started_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "uuid": run_uuid,
        "phase": config.phase,
        "started_at": started_at,
        "load_snapshot": str(load_snapshot_path) if load_snapshot_path is not None else None,
        "load_latest_from_model_root": bool(config.load_latest_from_model_root),
        "reward_prompt": config.reward_prompt,
        "deterministic_coding": bool(config.deterministic_coding),
        "history_window": int(config.history_window),
        "all_images": bool(config.all_images),
        "prompt_token_window": int(config.prompt_token_window),
        "learning_rate": float(config.learning_rate),
        "init_t": float(config.init_t),
        "training": asdict(training_config),
        "snapshot_keep": int(config.snapshot_keep),
        "picar_host": picar_host,
    }
    observation_store.write_run_metadata(run_metadata)
    snapshot_store.write_run_metadata(run_metadata)

    print(f"[experiment] uuid={run_uuid}", flush=True)
    print(f"[experiment] data_dir={data_run_dir}", flush=True)
    print(f"[experiment] model_dir={model_run_dir}", flush=True)
    print(f"[experiment] picar_host={picar_host}", flush=True)

    replay_buffer = TransitionReplayBuffer(capacity=10_000)
    model_config = ModelConfig(
        model_dir=model_assets_root,
        deterministic_coding=bool(config.deterministic_coding),
        all_images=bool(config.all_images),
        prompt_token_window=(
            int(config.prompt_token_window) if int(config.prompt_token_window) > 0 else None
        ),
        learning_rate=float(config.learning_rate),
    )
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=model_config,
    )
    if load_snapshot_path is not None:
        load_result = snapshot_store.load_into_model(snapshot_path=load_snapshot_path, model=model)
        print(
            f"[snapshot] loaded={load_result.path} "
            f"trainable_keys={len(load_result.loaded_trainable_keys)} ssr={load_result.has_ssr_state}",
            flush=True,
        )

    shared_reward_model = getattr(getattr(model, "backbone", None), "model", None)
    shared_reward_processor = getattr(getattr(model, "backbone", None), "processor", None)
    reward_registry, reward_prompt_id = _build_reward_registry(config.reward_prompt)
    reward_scorer = FrozenVLMRewardScorer(
        RewardConfig(prompt_id=reward_prompt_id),
        model_config=ModelConfig(model_dir=model_assets_root),
        registry=reward_registry,
        shared_model=shared_reward_model,
        shared_processor=shared_reward_processor,
        disable_shared_adapter=True,
    )

    speech_config = SpeechConfig(model_dir=model_assets_root)
    speech_stream = ContinuousSpeechStream(SpeechStreamConfig(), transcriber=FasterWhisperSTT(speech_config))
    speaker = _Speaker(tts=PiperTTS(speech_config), audio=AudioIO(speech_config.audio))
    base_picar_config = EnvConfig().picar
    picar_config = replace(base_picar_config, host=picar_host)
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=PiCarControlClient(picar_config),
        speech_stream=speech_stream,
        speaker=speaker,
        config=EnvConfig(
            data_dir=data_run_dir / ".env_internal",
            enable_persistence=False,
            history_window=int(config.history_window),
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
            fit_triggered = bool(getattr(training_summary, "triggered", False))
            memorized = getattr(training_summary, "memorized", None)
            memorized_count = int(memorized) if memorized is not None and memorized > 0 else None
            memorized_triggered = memorized_count is not None
            if training_enabled and (fit_triggered or memorized_triggered):
                if fit_triggered and memorized_triggered:
                    snapshot_reason = "fit+memorize"
                elif fit_triggered:
                    snapshot_reason = "fit"
                else:
                    snapshot_reason = "memorize"
                snapshot_path = snapshot_store.save_snapshot(
                    model=model,
                    replay_buffer=model.replay_buffer,
                    step_index=steps_completed,
                    t=float(next_observation.t),
                    reason=snapshot_reason,
                    memorize_count=memorized_count,
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
                "Primary visual task: find the red ball.\n"
                "A separate environment-side shaping term handles command-following "
                "(+2 when the robot follows a user command to completion, -2 when ignored).\n"
                "If the task is not visible, return 0.\n"
                "If the task is strongly satisfied, return a value near 10.\n"
                "Use values in [0, 10]."
            ),
            task_text=default_prompt,
            min_reward=0.0,
            max_reward=10.0,
        )
    else:
        custom_task = prompt_text.strip()
        spec = RewardPromptSpec(
            prompt_id="reward_prompt_custom",
            prompt_text=(
                "You are a reinforcement learning reward scorer for a PiCar-V robot.\n"
                "Return JSON with exactly one key: `reward`.\n"
                f"Primary visual task: {custom_task}.\n"
                "A separate environment-side shaping term handles command-following "
                "(+2 when the robot follows a user command to completion, -2 when ignored).\n"
                "If the task is not visible or not being satisfied, return 0.\n"
                "If the task is strongly satisfied, return a value near 10.\n"
                "Use values in [0, 10]."
            ),
            task_text=custom_task,
            min_reward=0.0,
            max_reward=10.0,
        )
    return RewardPromptRegistry(prompts={spec.prompt_id: spec}), spec.prompt_id


def _log_t_progress(*, step_index: int, t: float) -> None:
    t_clamped = min(max(float(t), 0.0), 1.0)
    width = 24
    filled = int(round(width * t_clamped))
    bar = "#" * filled + "-" * (width - filled)
    message = f"[t] step={step_index:06d} t={t_clamped:.3f} [{bar}]"
    LOGGER.info(message)
    if not LOGGER.isEnabledFor(logging.INFO):
        print(message, flush=True)


def _validate_config(config: ExperimentRunConfig) -> None:
    if int(config.epochs) < 1:
        raise ValueError(f"--epochs must be >= 1, got {config.epochs}")
    if config.fit_iters is not None and int(config.fit_iters) < 1:
        raise ValueError(f"--fit-iters must be >= 1, got {config.fit_iters}")
    if not (0.0 <= float(config.init_t) <= 1.0):
        raise ValueError(f"--init-t must be in [0, 1], got {config.init_t}")
    if config.fixed_t is not None and not (0.0 <= float(config.fixed_t) <= 1.0):
        raise ValueError(f"--fixed-t must be in [0, 1], got {config.fixed_t}")
    if float(config.t_step) <= 0.0:
        raise ValueError(f"--t-step must be positive, got {config.t_step}")
    if int(config.t_log_every) < 1:
        raise ValueError(f"--t-log-every must be >= 1, got {config.t_log_every}")
    if int(config.snapshot_keep) < 1:
        raise ValueError(f"--snapshot-keep must be >= 1, got {config.snapshot_keep}")
    if int(config.history_window) < 1:
        raise ValueError(f"--history-window must be >= 1, got {config.history_window}")
    if int(config.prompt_token_window) < 0:
        raise ValueError(
            f"--prompt-token-window must be >= 0 (0 disables token truncation), got {config.prompt_token_window}"
        )
    if float(config.learning_rate) <= 0.0:
        raise ValueError(f"--learning-rate must be > 0, got {config.learning_rate}")
    if config.load_snapshot is not None and bool(config.load_latest_from_model_root):
        raise ValueError("--load-snapshot and --load-latest-from-model-root are mutually exclusive")
    return None


def _resolve_picar_host(config: ExperimentRunConfig) -> str:
    if config.picar_host is not None and config.picar_host.strip():
        return config.picar_host.strip()
    env_host = os.environ.get("PICAR_V_HOST")
    if env_host is not None and env_host.strip():
        return env_host.strip()
    return PiCarControlConfig().host


def _resolve_load_snapshot_path(*, config: ExperimentRunConfig, snapshot_root: Path) -> Path | None:
    if config.load_snapshot is not None:
        return resolve_snapshot_path(config.load_snapshot)
    if bool(config.load_latest_from_model_root):
        return resolve_latest_snapshot_from_model_root(snapshot_root)
    return None
