from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.modeling import OnlineTrainingAdapter, build_experiment_model, resolve_update_mode
from experiments.schemas import ExperimentRunConfig
from experiments.snapshot_store import SnapshotStore
from model.backbones import FakeBackbone
from model.config import ModelConfig
from model.online_picar_agent import OnlinePiCarActionModel
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


def _transition() -> Transition:
    return Transition(
        observation=_observation(0),
        executed_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.5},
        reward=0.5,
        next_observation=_observation(1),
        done=False,
        target_text="look-forward",
        logp_beta_sum=0.0,
        target_action_name="look-forward",
        agentic_action_vector={"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 0.0},
        actor_action_vector={"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    )


def test_resolve_update_mode_auto_uses_batch_for_init_and_online_for_tune():
    assert resolve_update_mode(phase="init", update_mode="auto") == "batch"
    assert resolve_update_mode(phase="tune", update_mode="auto") == "online"
    assert resolve_update_mode(phase="retask", update_mode="batch") == "batch"


def test_build_experiment_model_uses_resolved_update_mode():
    replay_buffer = TransitionReplayBuffer(capacity=4)
    model_config = ModelConfig(hidden_size=4, learning_rate=0.05)

    batch_model = build_experiment_model(
        config=ExperimentRunConfig(phase="init"),
        replay_buffer=replay_buffer,
        model_config=model_config,
        backbone=FakeBackbone(hidden_size=4),
    )
    online_model = build_experiment_model(
        config=ExperimentRunConfig(phase="tune"),
        replay_buffer=replay_buffer,
        model_config=model_config,
        backbone=FakeBackbone(hidden_size=4),
    )

    assert isinstance(batch_model, PiCarActionModel)
    assert isinstance(online_model, OnlinePiCarActionModel)


def test_snapshot_load_into_online_model_hard_syncs_targets(tmp_path):
    transition = _transition()
    source_buffer = TransitionReplayBuffer(capacity=4)
    source_buffer.add(transition)
    source_model = PiCarActionModel(
        replay_buffer=source_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=FakeBackbone(hidden_size=4, agentic_action_names=["look-forward"], generated_texts=[""]),
    )
    source_model.memorize(n=1, disable_tqdm=True)
    with torch.no_grad():
        source_model.actor_head.proj.bias.fill_(0.25)
        for parameter in source_model.critic.parameters():
            parameter.fill_(0.1)
        source_model.sync_target_networks()

    store = SnapshotStore(tmp_path / "source_run")
    snapshot_path = store.save_snapshot(
        model=source_model,
        replay_buffer=source_buffer,
        step_index=1,
        t=0.25,
        reason="unit-test",
    )

    target_buffer = TransitionReplayBuffer(capacity=4)
    target_model = OnlinePiCarActionModel(
        replay_buffer=target_buffer,
        config=ModelConfig(hidden_size=4, learning_rate=0.05),
        backbone=FakeBackbone(hidden_size=4, agentic_action_names=["look-forward"], generated_texts=[""]),
    )
    with torch.no_grad():
        target_model.target_actor_head.proj.bias.fill_(9.0)
        for parameter in target_model.target_critic.parameters():
            parameter.fill_(9.0)

    load_result = store.load_into_model(snapshot_path=snapshot_path, model=target_model)

    assert load_result.loaded_trainable_keys
    for target_parameter, source_parameter in zip(
        target_model.target_actor_head.parameters(),
        target_model.actor_head.parameters(),
    ):
        assert torch.allclose(target_parameter, source_parameter)
    for target_parameter, source_parameter in zip(
        target_model.target_critic.parameters(),
        target_model.critic.parameters(),
    ):
        assert torch.allclose(target_parameter, source_parameter)


def test_online_training_adapter_passes_pi_override_to_model_fit():
    class FakeOnlineModel:
        def __init__(self):
            self.replay_buffer = [object()]
            self.transition_losses = []
            self.fit_calls = []

        def transition_loss(self, transition):
            self.transition_losses.append(transition)
            return torch.tensor(1.0)

        def fit(self, *, loss, pi=None, memorize=True, grad_clip=None):
            del grad_clip
            self.fit_calls.append({"loss": float(loss), "pi": pi, "memorize": memorize})
            return float(pi), float(loss)

    adapter = OnlineTrainingAdapter(pi_override=0.125)
    model = FakeOnlineModel()
    transition = _transition()
    training = type(
        "Training",
        (),
        {"epochs": 2, "memorize_every_steps": 1},
    )()

    summary = adapter.train(
        model=model,
        transition=transition,
        training=training,
        replay_size=1,
        effective_fit_iters=1,
        step_index=0,
    )

    assert len(model.transition_losses) == 2
    assert [call["pi"] for call in model.fit_calls] == [0.125, 0.125]
    assert [call["memorize"] for call in model.fit_calls] == [False, True]
    assert summary.pi == 0.125
