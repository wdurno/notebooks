import pytest


torch = pytest.importorskip("torch")

from picar_kl.models.lstm_policy import LSTMPolicyConfig, TokenStreamLSTMPolicy
from picar_kl.models.vlm_head import VLMConditioningHead, VLMConditioningHeadConfig
from picar_kl.training.kl_projection import masked_action_kl_loss


def _policy(**kwargs):
    config = LSTMPolicyConfig(
        visual_dim=5,
        model_dim=7,
        conditioning_dim=3,
        token_type_dim=2,
        action_dim=8,
        lstm_hidden_dim=11,
        **kwargs,
    )
    return TokenStreamLSTMPolicy(config)


def _batch(batch_size=2, steps=3, visual_tokens=4):
    visual = torch.randn(batch_size, steps, visual_tokens, 5)
    previous = torch.zeros(batch_size, steps, 8)
    previous[..., -1] = 1.0
    targets = torch.zeros(batch_size, steps, 8)
    targets[..., 2] = 1.0
    step_mask = torch.ones(batch_size, steps, dtype=torch.bool)
    return visual, previous, targets, step_mask


def test_lstm_policy_forward_shapes_with_zero_conditioning():
    torch.manual_seed(1)
    model = _policy()
    visual, previous, _, step_mask = _batch()
    step_mask[1, -1] = False

    output = model(visual_tokens=visual, previous_actions=previous, step_mask=step_mask)

    assert output.logits.shape == (2, 3, 8)
    assert output.log_probs.shape == (2, 3, 8)
    assert output.probabilities.shape == (2, 3, 8)
    assert output.readout_mask.tolist() == [[True, True, True], [True, True, False]]
    assert output.token_mask.shape == (2, 15)
    assert output.token_mask[1, -5:].tolist() == [False, False, False, False, False]
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_nonzero_conditioning_changes_logits():
    torch.manual_seed(2)
    model = _policy()
    visual, previous, _, step_mask = _batch(batch_size=1, steps=2)
    zeros = torch.zeros(1, 2, 3)
    nonzero = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 0.25, -0.25]]])

    zero_output = model(
        visual_tokens=visual,
        previous_actions=previous,
        conditioning=zeros,
        step_mask=step_mask,
    )
    nonzero_output = model(
        visual_tokens=visual,
        previous_actions=previous,
        conditioning=nonzero,
        step_mask=step_mask,
    )

    assert not torch.allclose(zero_output.logits, nonzero_output.logits)


def test_conditioning_head_receives_gradients_through_policy():
    torch.manual_seed(3)
    policy = _policy()
    head = VLMConditioningHead(VLMConditioningHeadConfig(input_dim=6, hidden_dim=5, conditioning_dim=3))
    visual, previous, targets, step_mask = _batch(batch_size=2, steps=2)
    source_features = torch.randn(2, 2, 6)

    conditioning = head(source_features)
    output = policy(
        visual_tokens=visual,
        previous_actions=previous,
        conditioning=conditioning,
        step_mask=step_mask,
    )
    loss = masked_action_kl_loss(logits=output.logits, target_distributions=targets, mask=step_mask)
    loss.backward()

    assert policy.visual_projection.weight.grad is not None
    assert policy.action_readout_token.grad is not None
    assert policy.lstm.weight_ih_l0.grad is not None
    assert policy.action_head.weight.grad is not None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_lstm_policy_rejects_bad_shapes():
    model = _policy()
    visual, previous, _, step_mask = _batch()

    with pytest.raises(ValueError, match="visual_dim"):
        model(visual_tokens=torch.randn(2, 3, 4, 6), previous_actions=previous, step_mask=step_mask)

    with pytest.raises(ValueError, match="previous_actions"):
        model(visual_tokens=visual, previous_actions=torch.zeros(2, 3, 7), step_mask=step_mask)

    with pytest.raises(ValueError, match="conditioning"):
        model(
            visual_tokens=visual,
            previous_actions=previous,
            conditioning=torch.zeros(2, 3, 4),
            step_mask=step_mask,
        )
