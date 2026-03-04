from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.replay_buffer import TransitionReplayBuffer
from model.schemas import ModelObservation, Transition


def _observation(step_index: int) -> ModelObservation:
    return ModelObservation(
        image_rgb=[[[step_index, step_index, step_index]]],
        messages=[{"role": "user", "content": [{"type": "text", "text": f"step {step_index}"}]}],
        t=0.2,
        step_index=step_index,
    )


def test_transition_replay_buffer_samples_structured_batches():
    buffer = TransitionReplayBuffer(capacity=8)
    buffer.add(
        Transition(
            observation=_observation(0),
            executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
            reward=1.0,
            next_observation=_observation(1),
            done=False,
            target_text="drive-forward",
            target_action_name="drive-forward",
        )
    )

    batch = buffer.sample(batch_size=1)

    assert len(batch.observations) == 1
    assert batch.executed_action_vector.shape == (1, 4)
    assert float(batch.executed_action_vector[0, 3].item()) == 1.0
    assert float(batch.reward[0].item()) == 1.0
    assert batch.target_action_name == ["drive-forward"]


def test_transition_replay_buffer_enforces_capacity():
    buffer = TransitionReplayBuffer(capacity=2)
    for step in range(3):
        buffer.add(
            Transition(
                observation=_observation(step),
                executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": float(step % 2), "drive": float(step)},
                reward=float(step),
                next_observation=_observation(step + 1),
                done=False,
            )
        )

    assert len(buffer) == 2
    batch = buffer.sample(idx_list=[0, 1])
    assert batch.observations[0].step_index == 1
    assert batch.observations[1].step_index == 2
