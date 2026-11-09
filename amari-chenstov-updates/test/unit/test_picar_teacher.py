"""Test-only teacher separation and real recorded-data distillation."""

import copy

import pytest
import torch

from numerical_experiment.picar import models, teacher


def bootstrap():
    model = models.build_tiny_model(8, width=8, depth=2, hidden_size=8, return_bins=3,
                                    dtype=torch.float64)
    batch = {"images": torch.ones(1, 1, 3, 4, 4, dtype=torch.float64) * .3,
             "executed_actions": torch.zeros(1, 1, 4, dtype=torch.float64),
             "elapsed": torch.ones(1, 1, 1, dtype=torch.float64),
             "contexts": [(models.LanguageEvent("stop", 0.),)]}
    p = torch.tensor([[.7, .1, .05, .03, .03, .03, .03, .03]], dtype=torch.float64)
    record = {"batch": batch, "executed_probabilities": p,
              "domain": "A", "target_kind": "soft"}
    return model, record


def test_deterministic_fake_requires_test_opt_in_and_validates_visible_inputs():
    with pytest.raises(ValueError, match="test_only"):
        teacher.DeterministicFakeTeacher()
    _, record = bootstrap()
    fake = teacher.DeterministicFakeTeacher(test_only=True)
    data = record["batch"]
    before = torch.get_rng_state().clone()
    p = fake.probabilities(data)
    torch.testing.assert_close(p, fake.probabilities(copy.deepcopy(data)), atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())
    assert p.shape == (1, 8) and bool((p > 0).all())
    torch.testing.assert_close(p.sum(-1), torch.ones(1, dtype=torch.float64))
    assert fake.provenance["real_qwen"] is False
    changed = copy.deepcopy(data)
    changed["contexts"] = [(models.LanguageEvent("turn left", 0.),)]
    assert not torch.equal(p, fake.probabilities(changed))
    changed["images"].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        fake.probabilities(changed)


def test_teacher_floor_is_the_recorded_executed_distribution():
    p = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
    result = teacher.prepare_teacher_distribution(p, probability_floor=.001,
                                                  target_kind="hard_engineered")
    assert result.target_kind == "hard_engineered"
    assert not result.executed_probabilities.requires_grad
    recovered = torch.cat((result.reference_logits, torch.zeros(1, 1)), -1).softmax(-1)
    torch.testing.assert_close(recovered, result.executed_probabilities)
    torch.testing.assert_close(result.original_probabilities, p)
    p.detach().zero_()
    assert result.original_probabilities[0, 0] == 1
    assert bool((result.executed_probabilities > 0).all())


def test_one_time_distillation_trains_both_banks_and_receipt_survives_checkpoint():
    model, record = bootstrap()
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    rng = torch.get_rng_state().clone()
    receipt = teacher.distill_once(model, [record], steps=20, learning_rate=.02)
    assert receipt["last_preupdate_kl"] < receipt["initial_kl"]
    assert receipt["stage"] == "one_time_recorded_teacher_kl"
    assert model.encoder.publication == 1
    assert torch.equal(rng, torch.get_rng_state())
    for bank, names in model.parameter_banks().items():
        assert any(not torch.equal(before[name], dict(model.named_parameters())[name]) for name in names), bank
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="already completed"):
        teacher.distill_once(model, [record], steps=1)
    restored, _ = bootstrap()
    restored.load_state_dict(model.state_dict())
    assert restored.distillation_receipt == receipt
    with pytest.raises(ValueError, match="already completed"):
        teacher.distill_once(restored, [record], steps=1)


@pytest.mark.parametrize("change,match", [({"domain": "B"}, "A-domain"),
    ({"target_kind": "unlabeled"}, "label"),
    ({"executed_probabilities": torch.zeros(1, 8)}, "strictly positive"),
    ({"executed_probabilities": torch.ones(1, 8)}, "normalized")])
def test_bootstrap_validation_before_optimizer_changes(change, match):
    model, record = bootstrap()
    before = {n: p.clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match=match):
        teacher.distill_once(model, [{**record, **change}], steps=1)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert model.distillation_receipt is None


def test_real_qwen_roles_never_accept_tiny_encoder_as_production():
    model, _ = bootstrap()
    with pytest.raises(TypeError, match="real Qwen"):
        teacher.FrozenQwenEvaluator(model.encoder)
