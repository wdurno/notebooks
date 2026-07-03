import pytest


torch = pytest.importorskip("torch")

from picar_kl.models.phase2 import Phase2KLModel, Phase2KLModelConfig
from picar_kl.training.kl_projection import masked_action_kl_loss


def _inputs(batch_size=2, context_steps=2, prediction_steps=3):
    prefix_visual = torch.randn(batch_size, context_steps, 4, 5)
    prefix_actions = torch.zeros(batch_size, context_steps, 8)
    prefix_actions[..., 2] = 1.0
    target_visual = torch.randn(batch_size, prediction_steps, 4, 5)
    target_previous = torch.zeros(batch_size, prediction_steps, 8)
    target_previous[..., -1] = 1.0
    targets = torch.zeros(batch_size, prediction_steps, 8)
    targets[..., 4] = 1.0
    prefix_mask = torch.ones(batch_size, context_steps, dtype=torch.bool)
    target_mask = torch.ones(batch_size, prediction_steps, dtype=torch.bool)
    return prefix_visual, prefix_actions, target_visual, target_previous, targets, prefix_mask, target_mask


def test_phase2_kl_model_computes_conditioning_inside_forward():
    torch.manual_seed(11)
    model = Phase2KLModel(
        Phase2KLModelConfig(
            visual_dim=5,
            model_dim=7,
            conditioning_dim=3,
            conditioning_hidden_dim=6,
            token_type_dim=2,
            lstm_hidden_dim=9,
        )
    )
    prefix_visual, prefix_actions, target_visual, target_previous, _, prefix_mask, target_mask = _inputs()

    output = model(
        prefix_visual_tokens=prefix_visual,
        prefix_actions=prefix_actions,
        prefix_step_mask=prefix_mask,
        target_visual_tokens=target_visual,
        target_previous_actions=target_previous,
        target_step_mask=target_mask,
    )

    assert output.conditioning.shape == (2, 3)
    assert output.logits.shape == (2, 3, 8)
    assert output.readout_mask.tolist() == [[True, True, True], [True, True, True]]


def test_phase2_kl_model_prefix_changes_logits_and_receives_gradients():
    torch.manual_seed(12)
    model = Phase2KLModel(
        Phase2KLModelConfig(
            visual_dim=5,
            model_dim=7,
            conditioning_dim=3,
            conditioning_hidden_dim=6,
            token_type_dim=2,
            lstm_hidden_dim=9,
        )
    )
    prefix_visual, prefix_actions, target_visual, target_previous, targets, prefix_mask, target_mask = _inputs(batch_size=1)
    changed_prefix = prefix_visual + 3.0

    first = model(
        prefix_visual_tokens=prefix_visual,
        prefix_actions=prefix_actions,
        prefix_step_mask=prefix_mask,
        target_visual_tokens=target_visual,
        target_previous_actions=target_previous,
        target_step_mask=target_mask,
    )
    second = model(
        prefix_visual_tokens=changed_prefix,
        prefix_actions=prefix_actions,
        prefix_step_mask=prefix_mask,
        target_visual_tokens=target_visual,
        target_previous_actions=target_previous,
        target_step_mask=target_mask,
    )
    assert not torch.allclose(first.conditioning, second.conditioning)
    assert not torch.allclose(first.logits, second.logits)

    loss = masked_action_kl_loss(logits=first.logits, target_distributions=targets, mask=target_mask)
    loss.backward()

    assert model.conditioning_head.visual_projection.weight.grad is not None
    assert model.conditioning_head.step_encoder.weight_ih_l0.grad is not None
    assert model.policy.lstm.weight_ih_l0.grad is not None
    assert model.policy.action_head.weight.grad is not None
