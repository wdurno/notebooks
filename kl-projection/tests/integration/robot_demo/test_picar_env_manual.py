from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "picar_kl" / "legacy" / "robot_demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.config import EnvConfig, PiCarControlConfig, RewardConfig, SpeechStreamConfig, TrainingConfig
from env.picar_bridge import PiCarControlClient
from env.picar_env import PiCarGymEnv
from env.rewarding import FrozenVLMRewardScorer
from env.speech_stream import ContinuousSpeechStream
from model.backbones import FakeBackbone
from model.config import ModelConfig
from model.picar_agent import PiCarActionModel
from model.replay_buffer import TransitionReplayBuffer
from speech.config import AudioIOConfig, SpeechConfig, STTConfig, TTSConfig
from speech.errors import MissingDependencyError, SpeechError
from speech.service import SpeechService


START_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 0.25


@pytest.mark.integration
def test_manual_picar_env_smoke(tmp_path):
    host = os.environ.get("PICAR_V_HOST")
    if not host:
        pytest.skip("Set PICAR_V_HOST=<host:port> to run the manual PiCar env smoke test.")

    step_count = int(os.environ.get("PICAR_TEST_STEPS", "2"))
    use_real_action_model = os.environ.get("PICAR_USE_REAL_ACTION_MODEL", "0") == "1"
    model_assets_root = PROJECT_ROOT / "artifacts" / "models"

    speech_config = SpeechConfig(
        model_dir=model_assets_root,
        stt=STTConfig(vad_filter=True),
        tts=TTSConfig(),
        audio=AudioIOConfig(),
    )
    env_config = EnvConfig(
        data_dir=tmp_path,
        picar=PiCarControlConfig(host=host, min_command_interval_seconds=0.5),
        reward=RewardConfig(prompt_id="reward_prompt_1"),
        speech=SpeechStreamConfig(
            sample_rate=speech_config.audio.sample_rate,
            channels=speech_config.audio.channels,
            dtype=speech_config.audio.dtype,
            blocksize=speech_config.audio.blocksize,
        ),
        training=TrainingConfig(
            train_every_steps=1,
            min_replay_size=1,
            batch_size=1,
            fit_iters=1,
            memorize_every_steps=1,
            memorize_n=1,
            memorize_random_idx=False,
            t_start=0.2,
            t_end=0.2,
            t_ramp_steps=1,
        ),
    )

    try:
        speech_service = SpeechService(speech_config)
        detected_devices = speech_service.audio.list_devices()
        speech_stream = ContinuousSpeechStream(env_config.speech, transcriber=speech_service.stt)
        reward_scorer = FrozenVLMRewardScorer(
            env_config.reward,
            model_config=ModelConfig(model_dir=model_assets_root),
        )
        picar_client = PiCarControlClient(env_config.picar)
        model = _build_action_model(use_real_action_model=use_real_action_model, model_dir=model_assets_root)
    except MissingDependencyError as exc:
        pytest.skip(str(exc))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    instructions = [
        "Manual PiCar environment smoke test.",
        f"Detected audio devices: {detected_devices}",
        "1. Set up the PiCar and confirm the API server is reachable.",
        "2. Put the PiCar on blocks or pick it up so the wheels can spin safely.",
        "3. Put a red ball in the camera view.",
        '4. Say "start" when ready.',
    ]

    for line in instructions:
        print(line, flush=True)

    spoken_prompt = (
        "Manual PiCar environment smoke test. "
        "Set up the PiCar. Put it on blocks or pick it up. "
        'Put a red ball in view. Say "start" when ready.'
    )

    try:
        speech_service.speak(spoken_prompt)
    except (MissingDependencyError, SpeechError) as exc:
        pytest.fail(f"Failed to speak the integration instructions: {exc}")

    try:
        speech_stream.start()
        start_event = _wait_for_start(speech_stream)
        print(f'Start phrase recognized: "{start_event.text}"', flush=True)

        env = PiCarGymEnv(
            model=model,
            reward_scorer=reward_scorer,
            picar_client=picar_client,
            speech_stream=speech_stream,
            speaker=speech_service,
            config=env_config,
        )

        initial_observation = env.reset()
        print(f"Initial step index: {initial_observation.step_index}", flush=True)

        last_info = None
        for step_idx in range(step_count):
            observation, reward, done, info = env.step()
            last_info = info
            print(
                f"Step {step_idx + 1}/{step_count}: reward={reward:.3f}, "
                f'action={info["action"].agentic_action_name}, done={done}',
                flush=True,
            )
            print(f'Reward raw output: {info["reward_result"].raw_text}', flush=True)
            assert done is False
            assert observation.step_index == step_idx + 1

        env.close()
    except KeyboardInterrupt:
        pytest.fail("Manual test interrupted before completion.")
    except SpeechError as exc:
        pytest.fail(f"Speech subsystem failed during the manual env test: {exc}")
    finally:
        speech_stream.stop()

    run_dirs = sorted(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()
    assert (run_dir / "blobs" / "replay_buffer.pkl").exists()
    assert (run_dir / "artifacts" / "picar_policy.state.pt").exists()
    assert last_info is not None
    assert last_info["action_receipt"]["status"] == "ok"
    assert last_info["training"].triggered is True


def _wait_for_start(speech_stream: ContinuousSpeechStream):
    deadline = time.time() + START_TIMEOUT_SECONDS
    while time.time() < deadline:
        for event in speech_stream.drain():
            print(f'Heard: "{event.text}"', flush=True)
            if "start" in event.text.lower():
                return event
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail('Timed out waiting for the operator to say "start".')


def _build_action_model(*, use_real_action_model: bool, model_dir: Path) -> PiCarActionModel:
    replay_buffer = TransitionReplayBuffer(capacity=32)
    if use_real_action_model:
        return PiCarActionModel(replay_buffer=replay_buffer, config=ModelConfig(model_dir=model_dir, default_t=0.2))

    backbone = FakeBackbone(
        hidden_size=4,
        agentic_action_names=["drive-forward"],
        generated_texts=["Searching for the red ball."],
    )
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(model_dir=model_dir, hidden_size=4, learning_rate=0.01, default_t=0.2),
        backbone=backbone,
    )
    with torch.no_grad():
        model.actor_head.proj.weight.zero_()
        model.actor_head.proj.bias.zero_()
        model.actor_head.proj.bias[1] = -20.0
    return model
