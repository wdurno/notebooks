from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.backbones import FakeBackbone
from model.config import ModelConfig
from model.picar_agent import PiCarActionModel
from model.replay_buffer import TransitionReplayBuffer
from model.schemas import ModelObservation, Transition


def _observation(step_index: int, t: float = 0.25) -> ModelObservation:
    image = torch.full((2, 2, 3), float(step_index), dtype=torch.float32)
    return ModelObservation(
        image_rgb=image,
        messages=[{"role": "user", "content": [{"type": "text", "text": f"step {step_index}"}]}],
        t=t,
        step_index=step_index,
    )


def test_forward_returns_mixed_vector_owned_by_model():
    replay_buffer = TransitionReplayBuffer(capacity=4)
    backbone = FakeBackbone(
        hidden_size=4,
        agentic_action_names=["drive-left"],
        generated_texts=["turning left"],
    )
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=backbone,
    )
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()
        model.value_head.bias[2] = 5.0

    action = model.forward(_observation(0, t=0.25))

    assert action.agentic_action_name == "drive-left"
    assert action.value_action_index == 2
    assert action.mixed_action_vector == {
        "pan": 0.0,
        "tilt": 0.0,
        "turn": -0.75,
        "drive": 0.25,
    }
    assert action.generated_text == "turning left"


def test_loss_combines_vlm_and_td_terms():
    replay_buffer = TransitionReplayBuffer(capacity=8)
    backbone = FakeBackbone(
        hidden_size=4,
        agentic_action_names=["drive-left", "drive-left"],
        generated_texts=["", ""],
        vlm_loss=2.0,
    )
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=backbone,
    )
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()
        model.value_head.bias[0] = 1.0
        model.value_head.bias[2] = 2.0

    transition = Transition(
        observation=_observation(0),
        action_index=2,
        reward=1.5,
        next_observation=_observation(1),
        done=False,
        target_text="drive-forward",
        target_action_name="drive-forward",
    )
    replay_buffer.add(transition)
    batch = replay_buffer.sample(batch_size=1)

    loss = model.loss(batch)

    assert loss.ndim == 0
    assert float(loss.item()) > 1.0


def test_optimizer_only_tracks_trainable_parameters():
    replay_buffer = TransitionReplayBuffer(capacity=4)
    backbone = FakeBackbone(hidden_size=4)
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(hidden_size=4),
        backbone=backbone,
    )

    optimizer_params = {id(parameter) for group in model.optimizer.param_groups for parameter in group["params"]}
    trainable_params = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    assert optimizer_params == trainable_params
