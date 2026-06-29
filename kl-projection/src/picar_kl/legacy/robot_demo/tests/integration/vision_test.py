from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.config import EnvConfig, PiCarControlConfig, TrainingConfig
from env.picar_bridge import PiCarControlClient
from env.picar_env import PiCarGymEnv
from env.schemas import QueuedSpeechEvent, RewardResult
from model.config import ModelConfig
from model.picar_agent import PiCarActionModel
from model.replay_buffer import TransitionReplayBuffer
from speech.audio_io import AudioIO
from speech.config import AudioIOConfig, SpeechConfig, TTSConfig
from speech.errors import MissingDependencyError, SpeechError
from speech.tts import PiperTTS


VISION_PROMPT_TEXT = (
    "Vision diagnostic mode. Keep the robot stationary. "
    "Do not use any drive-* actions. "
    "Return strict JSON with keys action and say. "
    'Set action to "look-forward". '
    "In say, provide one short sentence describing one thing you can currently see. "
    'Example: {"action":"look-forward","say":"I see a red ball near the left side."}'
)


class ConstantRewardScorer:
    """Fast reward stub for vision/manual integration runs."""

    def score(self, image_rgb):
        del image_rgb
        return RewardResult(
            prompt_id="vision_test",
            raw_text='{"reward": 0}',
            reward=0.0,
            clipped_reward=0.0,
        )


class PromptEveryStepStream:
    """Speech-stream adapter that injects a fixed operator prompt each step."""

    def __init__(self, text: str):
        self.text = text
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def drain(self):
        if not self.started:
            return []
        return [QueuedSpeechEvent(text=self.text, received_at=time.time())]


class TTSSpeaker:
    """Minimal speaker facade used by `PiCarGymEnv` for generated robot text."""

    def __init__(self, config: SpeechConfig):
        self.audio = AudioIO(config.audio)
        self.tts = PiperTTS(config)

    def speak(self, text: str):
        synthesis = self.tts.synthesize_to_tempfile(text)
        self.audio.play_wav(synthesis.audio_path)
        return synthesis


class DesktopFrameStreamer:
    """Display RGB frames in a desktop OpenCV window."""

    def __init__(self, window_name: str):
        self.window_name = window_name
        self.cv2 = self._load_cv2()
        self._created = False

    def show(self, image_rgb):
        frame_bgr = self.cv2.cvtColor(image_rgb, self.cv2.COLOR_RGB2BGR)
        if not self._created:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self._created = True
        self.cv2.imshow(self.window_name, frame_bgr)
        self.cv2.waitKey(1)

    def close(self):
        if self._created:
            self.cv2.destroyWindow(self.window_name)
            self.cv2.waitKey(1)

    @staticmethod
    def _load_cv2():
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise MissingDependencyError(
                "Desktop display not available (`DISPLAY`/`WAYLAND_DISPLAY` not set) for frame streaming."
            )
        try:
            import cv2
        except ImportError as exc:
            raise MissingDependencyError("opencv-python is required for vision frame streaming.") from exc
        return cv2


@pytest.mark.integration
def test_manual_vision_stream(tmp_path, vision_seconds: float):
    host = os.environ.get("PICAR_V_HOST")
    if not host:
        pytest.skip("Set PICAR_V_HOST=<host:port> to run the manual vision integration test.")

    duration_label = "infinite" if vision_seconds < 0 else f"{vision_seconds:.1f} seconds"
    print("Manual vision test starting.", flush=True)
    print(f"Configured run duration: {duration_label}", flush=True)
    print("Press Ctrl-C in this terminal to end the test early.", flush=True)
    print("A desktop window will stream the robot camera frames.", flush=True)

    env_config = EnvConfig(
        data_dir=tmp_path,
        picar=PiCarControlConfig(host=host, min_command_interval_seconds=0.5),
        training=TrainingConfig(
            train_every_steps=0,
            memorize_every_steps=0,
            min_replay_size=10_000_000,
            batch_size=1,
            fit_iters=1,
            memorize_n=1,
            memorize_random_idx=False,
            t_start=0.0,
            t_end=0.0,
            t_ramp_steps=1,
        ),
    )

    speech_stream = PromptEveryStepStream(VISION_PROMPT_TEXT)
    try:
        speaker = TTSSpeaker(
            SpeechConfig(
                tts=TTSConfig(),
                audio=AudioIOConfig(),
            )
        )
        frame_streamer = DesktopFrameStreamer("PiCar Vision Test")
        model = PiCarActionModel(
            replay_buffer=TransitionReplayBuffer(capacity=256),
            config=ModelConfig(default_t=0.0),
        )
    except (MissingDependencyError, RuntimeError) as exc:
        pytest.skip(str(exc))

    env = PiCarGymEnv(
        model=model,
        reward_scorer=ConstantRewardScorer(),
        picar_client=PiCarControlClient(env_config.picar),
        speech_stream=speech_stream,
        speaker=speaker,
        config=env_config,
    )

    step_count = 0
    interrupted = False
    started_at = time.monotonic()
    try:
        observation = env.reset()
        frame_streamer.show(observation.image_rgb)
        while True:
            if vision_seconds >= 0 and (time.monotonic() - started_at) >= vision_seconds:
                break
            observation, reward, done, info = env.step()
            step_count += 1
            frame_streamer.show(observation.image_rgb)
            generated_text = info["action"].generated_text
            print(
                f"step={step_count} reward={reward:.3f} said={generated_text!r}",
                flush=True,
            )
            if done:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Ctrl-C detected. Ending vision test early.", flush=True)
    except SpeechError as exc:
        pytest.fail(f"TTS failure during vision test: {exc}")
    finally:
        env.close()
        frame_streamer.close()

    print(
        f"Vision test complete. steps={step_count}, interrupted={interrupted}",
        flush=True,
    )
