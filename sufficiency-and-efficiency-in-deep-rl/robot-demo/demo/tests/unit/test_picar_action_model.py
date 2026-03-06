from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.backbones import FakeBackbone
from model.backbones import BackboneBatchOutput
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


class BFloat16Backbone:
    def __init__(self, hidden_size: int = 4):
        self.hidden_size = hidden_size

    def encode(self, observations, *, target_texts=None, compute_vlm_loss=False, allow_agentic_actions=True):
        del target_texts, compute_vlm_loss, allow_agentic_actions
        batch_size = len(observations)
        hidden = torch.zeros((batch_size, self.hidden_size), dtype=torch.bfloat16)
        return BackboneBatchOutput(
            pooled_hidden_state=hidden,
            agentic_action_names=["look-forward"] * batch_size,
            generated_texts=[""] * batch_size,
            vlm_loss=None,
            debug=[{} for _ in range(batch_size)],
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
        model.actor_head.proj.weight.zero_()
        model.actor_head.proj.bias.zero_()
        model.actor_head.proj.bias[1] = -20.0
        for module in (model.critic, model.target_critic):
            for parameter in module.parameters():
                parameter.zero_()

    action = model.forward(_observation(0, t=0.25))

    assert action.agentic_action_name == "drive-left"
    assert action.actor_action_vector["pan"] == 0.0
    assert action.actor_action_vector["tilt"] == pytest.approx(0.0, abs=1e-6)
    assert action.actor_action_vector["turn"] == 0.0
    assert action.actor_action_vector["drive"] == 0.0
    assert action.executed_action_vector["pan"] == 0.0
    assert action.executed_action_vector["tilt"] == pytest.approx(0.0, abs=1e-6)
    assert action.executed_action_vector["turn"] == -0.75
    assert action.executed_action_vector["drive"] == 0.0
    assert action.critic_value == 0.0
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
        model.actor_head.proj.weight.zero_()
        model.actor_head.proj.bias.zero_()
        model.actor_head.proj.bias[1] = -20.0
        for module in (model.critic, model.target_critic):
            for parameter in module.parameters():
                parameter.zero_()

    transition = Transition(
        observation=_observation(0),
        executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
        reward=1.5,
        next_observation=_observation(1),
        done=False,
        target_text="drive-forward",
        target_action_name="drive-forward",
        agentic_action_vector={"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 0.0},
        actor_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    )
    replay_buffer.add(transition)
    batch = replay_buffer.sample(batch_size=1)

    loss = model.loss(batch)

    assert loss.ndim == 0
    assert float(loss.item()) > 0.0


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


def test_get_grad_vec_uses_trainable_parameters_only():
    replay_buffer = TransitionReplayBuffer(capacity=8)
    backbone = FakeBackbone(
        hidden_size=4,
        agentic_action_names=["drive-left"],
        generated_texts=[""],
        vlm_loss=1.0,
    )
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=backbone,
    )

    transition = Transition(
        observation=_observation(0),
        executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.5},
        reward=0.5,
        next_observation=_observation(1),
        done=False,
        target_text="look-forward",
        target_action_name="look-forward",
        agentic_action_vector={"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 0.0},
        actor_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    )
    replay_buffer.add(transition)

    grad_vec = model.get_grad_vec(0)
    param_vec = model.get_param()

    assert grad_vec.shape == param_vec.shape
    assert grad_vec.device == model.device


def test_forward_handles_bfloat16_backbone_hidden_state():
    replay_buffer = TransitionReplayBuffer(capacity=4)
    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=BFloat16Backbone(hidden_size=4),
    )

    action = model.forward(_observation(0, t=0.0))

    assert action.agentic_action_name == "look-forward"
