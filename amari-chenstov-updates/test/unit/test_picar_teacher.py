"""Test-only teacher separation and real recorded-data distillation."""

import copy
import hashlib
import json

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


@pytest.fixture
def multimodal_role_encoder():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2_5_VLConfig(text_config={
        "vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}, vision_config={
        "depth": 1, "hidden_size": 16, "intermediate_size": 24, "num_heads": 2,
        "patch_size": 2, "temporal_patch_size": 1, "spatial_merge_size": 2,
        "window_size": 8, "out_hidden_size": 16, "fullatt_block_indexes": [0]},
        image_token_id=35, video_token_id=36, vision_start_token_id=37, vision_end_token_id=38)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(9)
        qwen = transformers.Qwen2_5_VLForConditionalGeneration(config).float().eval()

    class Processor:
        def __init__(self):
            self.tokenizer = self
            self.calls = []

        def encode(self, label, **kwargs):
            return [3 + int(label)]

        def apply_chat_template(self, conversation, **kwargs):
            return conversation[0]["content"][1]["text"]

        def __call__(self, *, text, images, **kwargs):
            self.calls.append((text, [image.clone() for image in images]))
            tokens = [[1, 37, 35, 38, *[3 + v % 25 for v in hashlib.sha256(row.encode()).digest()[:2]]]
                      for row in text]
            patches = [image.unfold(1, 2, 2).unfold(2, 2, 2).permute(1, 2, 0, 3, 4).reshape(4, 12)
                       for image in images]
            return {"input_ids": torch.tensor(tokens), "attention_mask": torch.ones(len(tokens), 6, dtype=torch.long),
                    "image_grid_thw": torch.tensor([[1, 2, 2]] * len(tokens)), "pixel_values": torch.cat(patches)}

    return models.QwenContextEncoder(qwen, Processor(), ltm_rank=2)


@pytest.mark.parametrize("role", ["teacher", "evaluator"])
@pytest.mark.parametrize("enable_cache", [False, True])
def test_real_roles_last_only_matches_full_multimodal_batch_final_row(multimodal_role_encoder, role, enable_cache):
    encoder = multimodal_role_encoder
    encoder.enable_prefix_cache = enable_cache
    scorer = (teacher.FrozenQwenTeacher(encoder, primitive_names=tuple(f"primitive_{i}" for i in range(8)))
              if role == "teacher" else teacher.FrozenQwenEvaluator(encoder))
    data = {"images": torch.linspace(0, 1, 2 * 3 * 3 * 4 * 4).reshape(2, 3, 3, 4, 4),
            "executed_actions": torch.arange(24).reshape(2, 3, 4).float() / 32,
            "elapsed": torch.full((2, 3, 1), .5),
            "contexts": [(models.LanguageEvent("turn right", 0., 1.), models.LanguageEvent("stop", 1.))] * 2}
    suffix = "\n" + scorer.prompt
    full, shape = encoder.prepare(data, suffix=suffix)
    assert shape == (2, 3)
    full_messages, full_images = encoder.processor.calls[-1]
    with torch.no_grad():
        hidden = encoder.encode_processed(full, role=role)
        head = encoder.qwen.lm_head
        ids = torch.tensor(scorer.token_ids)
        logits = torch.nn.functional.linear(hidden[:, -1], head.weight[ids],
            head.bias[ids] if head.bias is not None else None)
        expected = logits.float().softmax(-1).reshape(2, 3, -1)[:, -1]
    actual = scorer.probabilities(data)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert not actual.requires_grad
    last_messages, last_images = encoder.processor.calls[-1]
    assert last_messages == [full_messages[2], full_messages[5]]
    for actual_image, expected_image in zip(last_images, (full_images[2], full_images[5])):
        torch.testing.assert_close(actual_image, expected_image, rtol=0, atol=0)
    context = json.loads(last_messages[0][:-len(suffix)])
    assert len(context["executed_actions"]) == len(context["elapsed"]) == 3
    assert context["instructions"][0]["status"] == "completed"
    assert context["instructions"][1]["text"] == "stop"
    assert scorer.provenance["extraction"] == "restricted_next_token_softmax_v1"
    assert scorer.provenance["frame_extraction"] == "last_rgb_full_recorded_context_v2"
    assert encoder.prepare(data, last_only=True)[1] == (2, 1)
    with pytest.raises(TypeError, match="boolean"):
        encoder.prepare(data, last_only=1)


def test_role_last_only_keeps_past_actions_but_not_discarded_rgb(multimodal_role_encoder):
    encoder = multimodal_role_encoder
    data = {"images": torch.full((1, 2, 3, 4, 4), .25), "executed_actions": torch.zeros(1, 2, 4),
            "elapsed": torch.full((1, 2, 1), .5), "contexts": [(models.LanguageEvent("stop", 0.),)]}
    before, _ = encoder.prepare(data, last_only=True)
    message = encoder.processor.calls[-1][0][0]
    data["images"][:, 0].fill_(.75)
    changed_image, _ = encoder.prepare(data, last_only=True)
    for name, value in before.items():
        torch.testing.assert_close(changed_image[name], value, rtol=0, atol=0)
    data["executed_actions"][:, 0, 2] = .5
    encoder.prepare(data, last_only=True)
    assert encoder.processor.calls[-1][0][0] != message
