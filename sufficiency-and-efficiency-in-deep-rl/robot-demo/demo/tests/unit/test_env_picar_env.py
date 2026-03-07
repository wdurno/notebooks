from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn as nn
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.config import EnvConfig, TrainingConfig
from env.picar_env import PiCarGymEnv
from env.schemas import QueuedSpeechEvent, RewardResult
from model.schemas import ModelActionOutput


class FakeReplayBuffer:
    def __init__(self):
        self.items = []
        self.saved_path = None
        self.clear_calls = []

    def __len__(self):
        return len(self.items)

    def add(self, transition):
        self.items.append(transition)

    def clear(self, n=None):
        self.clear_calls.append(n)
        if n is None or n >= len(self.items):
            self.items = []
            return
        self.items = self.items[n:]

    def save(self, path):
        self.saved_path = Path(path)
        Path(path).write_bytes(b"replay")


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_head = nn.Linear(1, 1)
        self.critic = nn.Linear(1, 1)
        self.replay_buffer = FakeReplayBuffer()
        self.fit_calls = []
        self.memorize_calls = []

    def forward(self, observation):
        del observation
        return ModelActionOutput(
            agentic_action_name="drive-forward",
            agentic_action_one_hot=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
            agentic_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            actor_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            critic_value=1.5,
            generated_text="moving forward",
        )

    def fit(self, batch_size, iters=1):
        self.fit_calls.append((batch_size, iters))
        return 0.5, 1.25

    def memorize(self, n=None, random_idx=False, disable_tqdm=False):
        self.memorize_calls.append((n, random_idx, disable_tqdm))

    def save(self, path):
        Path(path + ".state.pt").write_bytes(b"state")
        Path(path + ".ssr.pt").write_bytes(b"ssr")


class FakeMalformedJsonModel(FakeModel):
    def forward(self, observation):
        del observation
        return ModelActionOutput(
            agentic_action_name="drive-forward",
            agentic_action_one_hot=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
            agentic_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            actor_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            critic_value=1.5,
            generated_text="garbled fallback text",
            debug={"json_expected": True, "json_valid": False},
        )


class FakeRewardScorer:
    def __init__(self):
        self.calls = 0

    def score(self, image_rgb):
        self.calls += 1
        return RewardResult(
            prompt_id="reward_prompt_1",
            raw_text='{"reward": 3}',
            reward=3.0,
            clipped_reward=3.0,
            metadata={"mean_pixel": float(np.asarray(image_rgb).mean())},
        )


class FakePiCarClient:
    def __init__(self):
        self.capture_calls = 0
        self.applied_vectors = []
        self.looked_forward = 0

    def reset(self):
        return self.capture_image()

    def capture_image(self):
        image = np.full((2, 2, 3), self.capture_calls, dtype=np.uint8)
        self.capture_calls += 1
        return image

    def apply_vector(self, vector):
        self.applied_vectors.append(dict(vector))
        return {"status": "ok"}

    def look_forward(self):
        self.looked_forward += 1


class FakeSpeechStream:
    def __init__(self):
        self.started = False
        self.stopped = False
        self._drained = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def drain(self):
        if self._drained:
            return []
        self._drained = True
        return [QueuedSpeechEvent(text="find the red ball", received_at=0.0)]


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


def test_picar_env_step_records_transition_and_training(tmp_path):
    model = FakeModel()
    reward_scorer = FakeRewardScorer()
    client = FakePiCarClient()
    speech_stream = FakeSpeechStream()
    speaker = FakeSpeaker()
    config = EnvConfig(
        data_dir=tmp_path,
        training=TrainingConfig(
            train_every_steps=1,
            min_replay_size=1,
            batch_size=4,
            fit_iters=2,
            memorize_every_steps=1,
            memorize_n=1,
            memorize_random_idx=False,
            t_ramp_steps=10,
        ),
    )
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=client,
        speech_stream=speech_stream,
        speaker=speaker,
        config=config,
    )

    initial_observation = env.reset()
    next_observation, reward, done, info = env.step()
    env.close()

    assert initial_observation.step_index == 0
    assert next_observation.step_index == 1
    assert reward == 3.0
    assert done is False
    assert len(model.replay_buffer) == 0
    assert model.fit_calls == [(4, 2)]
    assert model.memorize_calls == [(1, False, True)]
    assert model.replay_buffer.clear_calls == [1]
    assert speaker.spoken == ["moving forward"]
    assert client.applied_vectors == [{"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0}]
    assert info["training"].triggered is True
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()
    assert (run_dir / "blobs" / "replay_buffer.pkl").exists()
    assert (run_dir / "artifacts" / "picar_policy.state.pt").exists()


def test_picar_env_memorize_minus_one_clears_all_replay(tmp_path):
    model = FakeModel()
    reward_scorer = FakeRewardScorer()
    client = FakePiCarClient()
    config = EnvConfig(
        data_dir=tmp_path,
        training=TrainingConfig(
            train_every_steps=1,
            min_replay_size=1,
            batch_size=1,
            fit_iters=1,
            memorize_every_steps=1,
            memorize_n=-1,
            memorize_random_idx=False,
            t_ramp_steps=10,
        ),
    )
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=client,
        config=config,
    )

    model.replay_buffer.add("prefill")
    env.reset()
    _, _, _, info = env.step()

    assert model.memorize_calls == [(2, False, True)]
    assert model.replay_buffer.clear_calls == [2]
    assert len(model.replay_buffer) == 0
    assert info["training"].memorized == 2


def test_picar_env_penalizes_malformed_json_and_suppresses_speech(tmp_path):
    model = FakeMalformedJsonModel()
    reward_scorer = FakeRewardScorer()
    client = FakePiCarClient()
    speaker = FakeSpeaker()
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=client,
        speaker=speaker,
        config=EnvConfig(data_dir=tmp_path),
    )

    env.reset()
    next_observation, reward, _, info = env.step()

    assert reward == pytest.approx(2.0)
    assert next_observation.last_reward == pytest.approx(2.0)
    assert speaker.spoken == []
    assert info["malformed_json"] is True
    assert info["reward_adjustment"] == pytest.approx(-1.0)
    assert model.replay_buffer.items[-1].reward == pytest.approx(2.0)
    assert model.replay_buffer.items[-1].metadata["malformed_json"] is True


def test_picar_env_rate_limits_vector_commands(tmp_path, monkeypatch):
    model = FakeModel()
    reward_scorer = FakeRewardScorer()
    client = FakePiCarClient()
    env = PiCarGymEnv(
        model=model,
        reward_scorer=reward_scorer,
        picar_client=client,
        config=EnvConfig(data_dir=tmp_path),
    )

    sleeps = []
    clock = {"now": 10.0}

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(duration):
        sleeps.append(duration)
        clock["now"] += duration

    monkeypatch.setattr("env.picar_env.time.monotonic", fake_monotonic)
    monkeypatch.setattr("env.picar_env.time.sleep", fake_sleep)

    env._last_vector_command_at = 9.8
    env._respect_command_rate_limit()

    assert sleeps == [pytest.approx(0.3)]


def test_interpolation_t_uses_fixed_value_when_configured(tmp_path):
    env = PiCarGymEnv(
        model=FakeModel(),
        reward_scorer=FakeRewardScorer(),
        picar_client=FakePiCarClient(),
        config=EnvConfig(
            data_dir=tmp_path,
            training=TrainingConfig(
                fixed_t=1.0,
                t_start=0.0,
                t_end=0.0,
                t_ramp_steps=10,
            ),
        ),
    )

    assert env._interpolation_t(0) == pytest.approx(1.0)
    assert env._interpolation_t(100) == pytest.approx(1.0)


def test_interpolation_t_uses_ramp_when_fixed_t_is_none(tmp_path):
    env = PiCarGymEnv(
        model=FakeModel(),
        reward_scorer=FakeRewardScorer(),
        picar_client=FakePiCarClient(),
        config=EnvConfig(
            data_dir=tmp_path,
            training=TrainingConfig(
                fixed_t=None,
                t_start=0.2,
                t_end=0.8,
                t_ramp_steps=10,
            ),
        ),
    )

    assert env._interpolation_t(0) == pytest.approx(0.2)
    assert env._interpolation_t(5) == pytest.approx(0.5)
    assert env._interpolation_t(10) == pytest.approx(0.8)
    assert env._interpolation_t(999) == pytest.approx(0.8)


def test_interpolation_t_rejects_invalid_fixed_t(tmp_path):
    env = PiCarGymEnv(
        model=FakeModel(),
        reward_scorer=FakeRewardScorer(),
        picar_client=FakePiCarClient(),
        config=EnvConfig(
            data_dir=tmp_path,
            training=TrainingConfig(fixed_t=1.25),
        ),
    )

    with pytest.raises(ValueError, match="training.fixed_t must be in \\[0, 1\\]"):
        env._interpolation_t(0)
