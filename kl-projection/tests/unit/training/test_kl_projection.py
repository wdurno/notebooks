import pytest


torch = pytest.importorskip("torch")

from picar_kl.models.lstm_policy import LSTMPolicyConfig, TokenStreamLSTMPolicy
from picar_kl.models.vlm_head import VLMConditioningHead, VLMConditioningHeadConfig
from picar_kl.training.kl_projection import masked_action_kl_loss, train_kl_projection_step


def test_masked_action_kl_loss_prefers_matching_distribution():
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, False]])
    matching_logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])
    wrong_logits = torch.tensor([[[-8.0, 8.0], [-8.0, 8.0]]])

    matching_loss = masked_action_kl_loss(logits=matching_logits, target_distributions=targets, mask=mask)
    wrong_loss = masked_action_kl_loss(logits=wrong_logits, target_distributions=targets, mask=mask)

    assert matching_loss < wrong_loss
    assert matching_loss < 1e-3


def test_masked_action_kl_loss_rejects_empty_mask():
    logits = torch.zeros(1, 2, 8)
    targets = torch.zeros(1, 2, 8)
    targets[..., 0] = 1.0

    with pytest.raises(ValueError, match="at least one valid"):
        masked_action_kl_loss(logits=logits, target_distributions=targets, mask=torch.zeros(1, 2, dtype=torch.bool))


def test_train_kl_projection_step_updates_policy_and_conditioning_head():
    torch.manual_seed(4)
    policy = TokenStreamLSTMPolicy(
        LSTMPolicyConfig(
            visual_dim=5,
            model_dim=7,
            conditioning_dim=3,
            token_type_dim=2,
            action_dim=8,
            lstm_hidden_dim=11,
        )
    )
    head = VLMConditioningHead(VLMConditioningHeadConfig(input_dim=6, hidden_dim=5, conditioning_dim=3))
    optimizer = torch.optim.Adam([*policy.parameters(), *head.parameters()], lr=1e-2)
    visual = torch.randn(2, 3, 4, 5)
    previous = torch.zeros(2, 3, 8)
    previous[..., -1] = 1.0
    targets = torch.zeros(2, 3, 8)
    targets[..., 2] = 1.0
    step_mask = torch.tensor([[True, True, True], [True, False, False]])
    source_features = torch.randn(2, 3, 6)
    before_policy = policy.visual_projection.weight.detach().clone()
    before_head = next(head.parameters()).detach().clone()

    result = train_kl_projection_step(
        model=policy,
        optimizer=optimizer,
        visual_tokens=visual,
        previous_actions=previous,
        target_distributions=targets,
        conditioning=head(source_features),
        step_mask=step_mask,
    )

    assert result.valid_steps == 4
    assert result.loss > 0.0
    assert not torch.allclose(policy.visual_projection.weight, before_policy)
    assert not torch.allclose(next(head.parameters()), before_head)
