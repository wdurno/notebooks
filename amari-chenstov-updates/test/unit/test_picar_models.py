"""CPU-only model contracts; random miniature Qwen needs no model download."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from numerical_experiment.picar import models
from src.continual import picar_math as numerics


def batch(steps=2, *, text="turn right", completed_at=None, count=1):
    return {"images": torch.linspace(0, 1, count * steps * 3 * 4 * 4,
                                     dtype=torch.float64).reshape(count, steps, 3, 4, 4),
            "executed_actions": torch.zeros(count, steps, 4, dtype=torch.float64),
            "elapsed": torch.ones(count, steps, 1, dtype=torch.float64),
            "contexts": [(models.LanguageEvent(text, 0., completed_at),)] * count}


def small_model(seed=5, **kwargs):
    return models.build_tiny_model(seed, width=8, depth=2, hidden_size=5,
                                   return_bins=3, ltm_rank=2, dtype=torch.float64, **kwargs)


def nonzero_factors(module):
    generator = torch.Generator().manual_seed(81)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, models.LowRankBank) and child.B.requires_grad:
                child.B.copy_(.2 * torch.randn(child.B.shape, generator=generator, dtype=child.B.dtype))


def test_import_does_not_import_optional_stack_or_load_models():
    code = ("import sys; import numerical_experiment.picar.models; "
            "import numerical_experiment.picar.teacher; "
            "assert not any(x in sys.modules for x in ('transformers','peft','bitsandbytes','huggingface_hub'))")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_deterministic_model_inventory_and_named_seed_independence():
    before = torch.get_rng_state().clone()
    model = small_model()
    assert torch.equal(before, torch.get_rng_state())
    other = small_model()
    for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    banks = model.parameter_banks()
    inventory = model.trainable_inventory()
    assert banks["ltm"] and banks["stm"]
    assert set(banks["ltm"]).isdisjoint(banks["stm"])
    assert set(banks["ltm"]) | set(banks["stm"]) == {p["name"] for p in inventory}
    assert len(set(model.encoder.provenance["seeds"].values())) == 4
    selected = [(name, p) for name, p in model.encoder.named_parameters() if p.requires_grad]
    assert len(selected) == 7 * 4
    assert all("layers.1." in name and ".banks." in name for name, _ in selected)
    assert sum(p.numel() for _, p in selected) == 17 * 8 * (2 + 1)
    for child in model.modules():
        if isinstance(child, models.LowRankBank):
            assert child.B.count_nonzero() == 0
            assert child.A.count_nonzero() == child.A.numel()
            assert not child.scale.requires_grad
    output = model(batch())
    assert set(output) == {"return_logits", "action_mean"}
    assert output["return_logits"].shape == (1, 3)
    assert output["action_mean"].shape == (1, 7)


@pytest.mark.parametrize("role", models.ROLES)
def test_parallel_effective_matrices_and_explicit_role(role):
    torch.manual_seed(12)
    adapter = models.ParallelAdapterLinear(nn.Linear(4, 3), ltm_rank=2, stm_rank=1,
                                           ltm_scale=.7, stm_scale=1.3).double()
    with torch.no_grad():
        for bank in adapter.banks.values():
            bank.B.fill_(.3)
    x = torch.randn(2, 4, dtype=torch.float64)
    with models._role_scope(role):
        actual = adapter(x)
    expected = F.linear(x, adapter.effective_weight(role), adapter.base.bias)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert not adapter.base.weight.requires_grad
    assert adapter.banks.teacher.B.requires_grad is False
    with pytest.raises(RuntimeError, match="explicit model role"):
        adapter(x)


def test_lstm_and_all_projection_readout_effective_weights():
    model = small_model()
    nonzero_factors(model)
    generator = torch.Generator().manual_seed(23)
    dense = nn.LSTMCell(5, 5).double()
    with torch.no_grad():
        dense.weight_ih.copy_(model.recurrent.input_gates.effective_weight())
        dense.weight_hh.copy_(model.recurrent.hidden_gates.effective_weight())
        dense.bias_ih.copy_(model.recurrent.input_gates.ltm.bias)
        dense.bias_hh.copy_(model.recurrent.hidden_gates.ltm.bias)
    x = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    state = tuple(torch.randn(2, 5, generator=generator, dtype=torch.float64) for _ in range(2))
    for actual, expected in zip(model.recurrent(x, state), dense(x, state)):
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    for child in model.modules():
        if isinstance(child, models.STMLinear):
            x = torch.randn(2, child.ltm.in_features, generator=generator, dtype=torch.float64)
            torch.testing.assert_close(child(x), F.linear(x, child.effective_weight(), child.ltm.bias))


def test_language_image_and_executed_action_reach_both_readouts():
    model = small_model()
    reference = batch()
    variants = [batch(text="turn left"), batch(completed_at=1.), batch()]
    variants[-1]["images"] = 1 - variants[-1]["images"]
    changed_action = batch()
    changed_action["executed_actions"][:, 0, 2] = .8
    variants.append(changed_action)
    before = model(reference)
    for variant in variants:
        features = model.encoder(variant)
        assert not torch.allclose(features, model.encoder(reference), atol=1e-10, rtol=0)
        for key, value in model(variant).items():
            assert not torch.allclose(value, before[key], atol=1e-10, rtol=0)


def test_context_completion_truncation_and_no_future_leakage():
    data = batch(completed_at=2.)
    first = json.loads(models.render_context(data, 0, 0))
    second = json.loads(models.render_context(data, 0, 1))
    assert first["instructions"][0]["status"] == "active"
    assert "completed_at" not in first["instructions"][0]
    assert second["instructions"][0]["status"] == "completed"
    future = copy.deepcopy(data)
    future["contexts"][0] += (models.LanguageEvent("secret future instruction", 100.),)
    assert models.render_context(data, 0, 0) == models.render_context(future, 0, 0)
    assert models.batch_digest(data) == models.batch_digest(future)
    future["contexts"] = [(models.LanguageEvent("old " * 100, 0., .1),
                            models.LanguageEvent("keep active", .2))]
    bounded = json.loads(models.render_context(future, 0, 0, max_chars=300))
    assert [e["text"] for e in bounded["instructions"]] == ["keep active"]
    with pytest.raises(ValueError, match="unfinished"):
        models.render_context(data, 0, 0, max_chars=10)
    with pytest.raises(ValueError, match="completion"):
        models.LanguageEvent("bad", 2., 1.)


def test_batch_companions_do_not_change_prediction():
    model = small_model()
    single = batch(count=1)
    combined = {k: torch.cat((v, v), 0) if isinstance(v, torch.Tensor)
                else [v[0], (models.LanguageEvent("long command " * 20, 0.),)]
                for k, v in single.items()}
    for name, value in model(single).items():
        torch.testing.assert_close(model(combined)[name][:1], value, rtol=1e-12, atol=1e-12)


def test_frozen_prefix_boundary_and_role_isolation():
    model = small_model()
    data = batch()
    data["images"].requires_grad_()
    calls = []
    hooks = [layer.register_forward_hook(lambda module, args, out: calls.append(
        (torch.is_grad_enabled(), args[0].requires_grad, out.requires_grad)))
        for layer in model.encoder.layers]
    features = model.encoder(data)
    assert calls[0] == (False, False, False)
    assert calls[1][0] and features.requires_grad
    for hook in hooks:
        hook.remove()
    features.square().sum().backward()
    assert data["images"].grad is None
    assert any(p.grad is not None and bool(p.grad.abs().sum())
               for p in model.encoder.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.encoder.layers[0].parameters())
    frozen = {role: model.encoder(data, role=role) for role in ("teacher", "evaluator")}
    for value in frozen.values():
        assert not value.requires_grad
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(.2)
    for role, value in frozen.items():
        torch.testing.assert_close(model.encoder(data, role=role), value, atol=0, rtol=0)


def test_inference_caches_and_recurrent_publication_policy():
    model = small_model().eval()
    short, full = batch(steps=1), batch(steps=2)
    full["images"][:, :1] = short["images"]
    _, state, _ = model.predict(short)
    incremental, _, _ = model.predict(full, state)
    reconstructed, _, _ = model.predict(full)
    for key, value in incremental.items():
        torch.testing.assert_close(value, reconstructed[key], atol=1e-14, rtol=1e-14)
    with torch.no_grad():
        _, first = model.encoder.inference_features(full, role="teacher")
        _, second = model.encoder.inference_features(full, role="teacher")
        assert not first["cache_hit"] and second["cache_hit"]
        model.mark_published()
        _, third = model.encoder.inference_features(full, role="teacher")
        _, learner = model.encoder.inference_features(full, role="learner")
        assert third["cache_hit"] and not learner["cache_hit"]
        _, changed = model.encoder.inference_features(batch(text="stop"), role="teacher")
        assert not changed["cache_hit"]
    with pytest.raises(models.StaleStateError):
        model.predict(full, state)
    with pytest.raises(models.StaleStateError):
        small_model().eval().predict(full, state)
    with pytest.raises(RuntimeError, match="inference-only"):
        model.encoder.inference_features(full, role="learner")
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        model.predict(full)


def test_functional_likelihood_fd_keeps_both_banks_and_buffers_unchanged():
    model = small_model()
    nonzero_factors(model)
    data = batch(steps=1)
    params = {name: p.detach().clone().requires_grad_()
              for name, p in model.named_parameters() if p.requires_grad}
    banks = model.parameter_banks()
    before = {k: v.clone() for k, v in model.named_buffers()}
    predict = lambda p: numerics.functional_predict(model, p, (data,))
    design = numerics.LikelihoodDesign(torch.tensor([-1., 0., 1.], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), 1., torch.eye(7, dtype=torch.float64))
    labels, actions = torch.tensor([2]), torch.linspace(-.2, .3, 7, dtype=torch.float64)[None]
    output = predict(params)
    expected = model(data)
    for key in output:
        torch.testing.assert_close(output[key], expected[key], rtol=0, atol=0)
    loss = numerics.joint_nll(output, labels, actions, design).mean()
    # Same normalized objective evaluated by Torch distributions through real heads.
    p = output["return_logits"].softmax(-1)
    gaussian = torch.distributions.MultivariateNormal(output["action_mean"],
        covariance_matrix=torch.eye(7, dtype=torch.float64)[None] / design.weights[:, 2, None, None])
    reference = -p[:, 2].log().mean() - gaussian.log_prob(actions).mean()
    torch.testing.assert_close(loss, reference)
    for actual, wanted in zip(torch.autograd.grad(loss, tuple(params.values()), retain_graph=True),
                              torch.autograd.grad(reference, tuple(params.values()))):
        torch.testing.assert_close(actual, wanted, rtol=1e-11, atol=1e-11)
    probes = numerics.rademacher_probes(params, banks["stm"], 1,
                                       generator=torch.Generator().manual_seed(98))
    def objective(values):
        return numerics.stm_fd_penalty(predict, values, design, banks["stm"], probes,
                                       torch.ones(1, dtype=torch.float64), .02)
    penalty = objective(params)
    gradients = dict(zip(params, torch.autograd.grad(penalty, tuple(params.values()))))
    for bank in ("ltm", "stm"):
        name = max(banks[bank], key=lambda n: float(gradients[n].abs().max()))
        assert gradients[name].abs().max() > 1e-10
        index = int(gradients[name].abs().flatten().argmax())
        direction = torch.zeros_like(params[name]).flatten()
        direction[index] = 1
        direction = direction.reshape_as(params[name])
        eps = 1e-5
        plus, minus = dict(params), dict(params)
        plus[name], minus[name] = params[name] + eps * direction, params[name] - eps * direction
        finite = (objective(plus) - objective(minus)) / (2 * eps)
        torch.testing.assert_close(gradients[name].flatten()[index], finite, atol=3e-8, rtol=3e-4)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert model.encoder.publication == 0 and model.encoder._cache == {}


@pytest.fixture
def miniature_qwen():
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
        return transformers.Qwen2_5_VLForConditionalGeneration(config).float().eval()


def test_real_qwen_split_matches_original_with_padding_images_and_no_cache(miniature_qwen):
    qwen = miniature_qwen
    inputs = {"input_ids": torch.tensor([[0, 1, 37, 35, 38, 3], [1, 37, 35, 38, 4, 5]]),
              "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
              "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
              "pixel_values": torch.linspace(0, 1, 8 * 12).reshape(8, 12)}
    with torch.no_grad():
        original = qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    encoder = models.QwenContextEncoder(qwen, None, top_layers=1, ltm_rank=2)
    for role in models.ROLES:
        actual = encoder.encode_processed(inputs, role=role)
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-6)
    original_delta = qwen.model.rope_deltas.clone()
    seen = []
    hook = qwen.model.language_model.layers[0].register_forward_hook(
        lambda module, args, out: seen.append((torch.is_grad_enabled(), out[0].requires_grad)))
    hidden = encoder.encode_processed(inputs, role="learner")
    hidden.square().sum().backward()
    hook.remove()
    assert seen == [(False, False)]
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in encoder.parameters() if p.requires_grad)
    assert all(p.grad is None for p in qwen.model.visual.parameters())
    assert all(p.grad is None for p in qwen.model.language_model.layers[0].parameters())
    torch.testing.assert_close(qwen.model.rope_deltas, original_delta, atol=0, rtol=0)
    with pytest.raises(ValueError, match="KV"):
        encoder.encode_processed({**inputs, "past_key_values": None}, role="learner")


def test_loader_defaults_download_pinned_snapshot_only_at_entrypoint(miniature_qwen, monkeypatch, tmp_path):
    import huggingface_hub
    import transformers
    from types import SimpleNamespace

    snapshot = tmp_path / models.QWEN_REVISION
    miniature_qwen.save_pretrained(snapshot)
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    def processor_load(source, **kwargs):
        assert source == str(snapshot)
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        return SimpleNamespace()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor_load)
    model = models.load_qwen_model(quantized=False, device="cpu", compute_dtype=torch.float32,
                                   hidden_size=5, return_bins=3)
    assert calls == [{"repo_id": models.QWEN_MODEL_ID, "revision": models.QWEN_REVISION,
                      "local_files_only": False, "cache_dir": models.QWEN_CACHE_DIR}]
    provenance = model.encoder.provenance
    assert provenance["revision_verified"] and provenance["checkpoint_manifest"]["files"]
    assert provenance["cache_dir"] == models.QWEN_CACHE_DIR
    json.dumps(provenance, allow_nan=False)
    calls.clear()
    model = models.load_qwen_model(snapshot, quantized=False, device="cpu", compute_dtype=torch.float32)
    assert calls == []
    assert model.encoder.provenance["revision_verified"] is False


def test_real_frozen_qwen_role_scorer_has_no_learner_dependency(miniature_qwen, monkeypatch):
    from types import SimpleNamespace
    from numerical_experiment.picar.teacher import FrozenQwenTeacher, FrozenQwenEvaluator

    class TestTokenizer:
        def encode(self, label, **kwargs):
            return [3 + int(label)]
    encoder = models.QwenContextEncoder(miniature_qwen,
                                        SimpleNamespace(tokenizer=TestTokenizer()), ltm_rank=2)
    inputs = {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    monkeypatch.setattr(encoder, "prepare", lambda data, **kwargs: (inputs, (1, 1)))
    teacher = FrozenQwenTeacher(encoder, primitive_names=tuple(f"primitive_{i}" for i in range(8)))
    evaluator = FrozenQwenEvaluator(encoder)
    before = teacher.probabilities(batch(1))
    score = evaluator.evaluate(batch(1))
    assert before.shape == (1, 8) and score.shape == (1,)
    assert not before.requires_grad and not score.requires_grad
    nonzero_factors(encoder)
    torch.testing.assert_close(before, teacher.probabilities(batch(1)), atol=0, rtol=0)
    torch.testing.assert_close(score, evaluator.evaluate(batch(1)), atol=0, rtol=0)
    assert teacher.encoder is evaluator.encoder
    assert teacher.provenance["calibrated"] is False


def test_default_cache_is_project_local_independent_of_working_directory(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    assert Path(models.QWEN_CACHE_DIR) == Path(models.__file__).resolve().parents[2] / "cache/numerical_experiment/picar/models"
    with pytest.raises(FileNotFoundError, match="explicit model directory"):
        models.load_qwen_model(tmp_path / "missing-checkpoint")


@pytest.mark.parametrize("enable_prefix_cache", [False, True])
def test_bf16_frozen_qwen_prefix_fp32_top_masks_rotary_and_fd(miniature_qwen, monkeypatch, enable_prefix_cache):
    qwen = miniature_qwen.bfloat16()
    encoder = models.QwenContextEncoder(qwen, None, top_layers=1, ltm_rank=2,
                                        enable_prefix_cache=enable_prefix_cache)
    text = qwen.model.language_model
    assert text.embed_tokens.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in text.layers[0].parameters())
    assert all(p.dtype == torch.float32 for p in text.layers[1].parameters())
    assert text.norm.weight.dtype == torch.float32
    assert all(p.dtype == torch.bfloat16 for p in qwen.model.visual.parameters())
    inputs = {"input_ids": torch.tensor([[1, 5, 6, 7]]),
              "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    seen = []
    def inspect_top(module, args, kwargs):
        seen.append((args[0].dtype, args[0].requires_grad,
                     kwargs["position_embeddings"][0].dtype,
                     kwargs["attention_mask"].dtype, torch.is_autocast_enabled("cpu")))
    hook = text.layers[1].register_forward_pre_hook(inspect_top, with_kwargs=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder.encode_processed(inputs, role="learner")
    hook.remove()
    assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
    assert seen == [(torch.float32, False, torch.float32, torch.float32, False)]
    monkeypatch.setattr(encoder, "prepare", lambda data, **kwargs: (inputs, (1, 1)))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(24)
        model = models.PiCarStudent(encoder, hidden_size=5, return_bins=3, factor_seed=7)
    nonzero_factors(model)
    data = batch(1)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    banks = model.parameter_banks()
    fixed = numerics.LikelihoodDesign(torch.tensor([-1., 0., 1.]), torch.zeros(1),
                                      1., torch.eye(7))
    probes = numerics.rademacher_probes(params, banks["stm"], 1,
                                       generator=torch.Generator().manual_seed(98))
    penalties = []
    for epsilon in (.005, .01, .02):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            predict = lambda p: numerics.functional_predict(model, p, (data,))
            prediction = predict(params)
            assert all(v.dtype == torch.float32 for v in prediction.values())
            penalty = numerics.stm_fd_penalty(predict, params, fixed, banks["stm"], probes,
                                             torch.ones(1), epsilon)
        gradients = dict(zip(params, torch.autograd.grad(penalty, tuple(params.values()))))
        assert all(any(gradients[n].abs().sum() > 0 for n in banks[bank]) for bank in banks)
        penalties.append(float(penalty.detach()))
    assert max(penalties) / min(penalties) < 1.03
    assert all(p.grad is None for p in text.layers[0].parameters())
    assert encoder.compute_policy["prefix_dtype"] == "torch.bfloat16"
    assert encoder.compute_policy["top_dtype"] == "torch.float32"
    if enable_prefix_cache:
        assert encoder.prefix_cache_stats["hits"] > 0


def test_real_bnb_top_configuration_keeps_packed_weight_bytes(miniature_qwen):
    bnb = pytest.importorskip("bitsandbytes")
    qwen = miniature_qwen.bfloat16()
    attention = qwen.model.language_model.layers[-1].self_attn
    original = attention.q_proj
    packed = bnb.nn.Linear4bit(original.in_features, original.out_features,
        bias=original.bias is not None, compute_dtype=torch.bfloat16,
        quant_type="nf4", compress_statistics=False)
    with torch.no_grad():
        packed.weight.copy_(original.weight)
        packed.bias.copy_(original.bias)
    packed = packed.to("cpu")
    bits, pointer = packed.weight.detach().clone(), packed.weight.data_ptr()
    assert packed.weight.dtype == torch.uint8
    attention.q_proj = packed
    models.QwenContextEncoder(qwen, None, top_layers=1, ltm_rank=2)
    assert packed.compute_dtype == torch.float32 and packed.compute_type_is_set
    assert packed.bias.dtype == torch.float32 and not packed.bias.requires_grad
    assert packed.weight.data_ptr() == pointer
    torch.testing.assert_close(packed.weight, bits, rtol=0, atol=0)


def test_checkpoint_manifest_verifies_bytes_and_hf_snapshot_symlinks(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
    weight = tmp_path / "weight_blob"
    weight.write_bytes(b"test checkpoint bytes")
    (tmp_path / "model.safetensors").symlink_to(weight)
    first = models.checkpoint_manifest(tmp_path)
    assert first["files"]["model.safetensors"]["sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest()
    weight.write_bytes(b"changed")
    assert first["sha256"] != models.checkpoint_manifest(tmp_path)["sha256"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../unsafe"}}))
    with pytest.raises(ValueError, match="unsafe"):
        models.checkpoint_manifest(tmp_path)


def test_loader_pin_checks_before_any_model_access():
    with pytest.raises(ValueError, match="40-hex"):
        models.load_qwen_model(revision="main")
    with pytest.raises(ValueError, match="pinned"):
        models.load_qwen_model(revision=None)
    with pytest.raises(ValueError, match="top_layers"):
        small_model(top_layers=3)
    with pytest.raises(ValueError, match="history exceeds"):
        small_model(context_length=1)(batch())


def processed_images():
    return {"input_ids": torch.tensor([[0, 1, 37, 35, 38, 3], [1, 37, 35, 38, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "pixel_values": torch.linspace(0, 1, 8 * 12).reshape(8, 12)}


def test_prefix_cache_is_opt_in_cpu_only_and_never_caches_adapted_suffix(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen, None, ltm_rank=2)
    nonzero_factors(encoder)
    inputs = processed_images()
    reference = encoder.encode_processed(inputs, role="learner")
    assert encoder.prefix_cache_stats["entries"] == 0
    assert not encoder.enable_prefix_cache and not encoder.enable_cache
    encoder.enable_prefix_cache = True
    calls = []
    layers = miniature_qwen.model.language_model.layers
    hooks = [layer.register_forward_hook(lambda module, args, out, i=i: calls.append(i))
             for i, layer in enumerate(layers)]
    try:
        first = encoder.encode_processed(inputs, role="learner")
        second = encoder.encode_processed(inputs, role="learner")
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == [0, 1, 1]
    assert encoder.prefix_cache_stats["hits"] == encoder.prefix_cache_stats["misses"] == 1
    for output in (first, second):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert output.requires_grad
    cache_x, cache_positions = next(iter(encoder._prefix_cache.values()))
    assert cache_x.device.type == cache_positions.device.type == "cpu"
    assert cache_x.dtype == torch.float32 and cache_positions.dtype == torch.int64
    assert not cache_x.requires_grad and cache_x.grad_fn is None
    assert encoder.prefix_cache_stats["bytes"] == sum(x.numel() * x.element_size() for x in (cache_x, cache_positions))
    assert encoder.prefix_cache_stats["max_entries"] == 2
    assert encoder.prefix_cache_stats["max_bytes"] == 256 * 1024**2
    assert encoder.provenance["prefix_memoization"]["suffix_and_fd_endpoints"] == "never_cached"
    with torch.no_grad():
        layers[-1].self_attn.q_proj.banks.ltm.B.add_(.2)
    changed = encoder.encode_processed(inputs, role="learner")
    assert not torch.equal(changed, second)
    assert encoder.prefix_cache_stats["hits"] == 2
    encoder.enable_cache = False
    assert encoder.prefix_cache_stats["entries"] == 0
    with pytest.raises(TypeError, match="boolean"):
        encoder.enable_prefix_cache = 1


def test_prefix_cache_fd_and_replay_nll_match_values_and_both_bank_gradients(miniature_qwen, monkeypatch):
    encoder = models.QwenContextEncoder(miniature_qwen, None, ltm_rank=2)
    inputs = processed_images()
    replay, fresh = batch(1, count=2), batch(1, count=2)
    replay["elapsed"].fill_(.5)
    fresh["elapsed"].fill_(.75)

    def prepare(data, **kwargs):
        prepared = {k: v.clone() for k, v in inputs.items()}
        # Model the prompt-token consequence of binary-exact distinct clocks.
        if data["elapsed"][0, 0, 0] == .75:
            prepared["input_ids"][:, -1] = 7
        return prepared, (2, 1)

    monkeypatch.setattr(encoder, "prepare", prepare)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(24)
        model = models.PiCarStudent(encoder, hidden_size=5, return_bins=3, factor_seed=7)
    nonzero_factors(model)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    banks = model.parameter_banks()
    fixed = numerics.LikelihoodDesign(torch.tensor([-1., 0., 1.]), torch.zeros(2), 1., torch.eye(7))
    probes = numerics.rademacher_probes(params, banks["stm"], 1, generator=torch.Generator().manual_seed(98))
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    results = []
    prefix_calls = []
    hook = miniature_qwen.model.language_model.layers[0].register_forward_hook(
        lambda *args: prefix_calls.append(1))
    try:
        for enabled in (False, True):
            encoder.enable_prefix_cache = enabled
            encoder.clear_caches()
            prefix_calls.clear()
            predict = lambda p: numerics.functional_predict(model, p, (replay,))
            output = predict(params)
            nll = numerics.joint_nll(output, torch.tensor([1, 2]), torch.zeros(2, 7), fixed).mean()
            fd = numerics.stm_fd_penalty(predict, params, fixed, banks["stm"], probes, torch.ones(2), .01)
            nll_grad = torch.autograd.grad(nll, tuple(params.values()))
            fd_grad = torch.autograd.grad(fd, tuple(params.values()))
            numerics.functional_predict(model, params, (fresh,))
            predict(params)
            results.append((output, nll, fd, nll_grad, fd_grad))
            assert len(prefix_calls) == (2 if enabled else 5)
    finally:
        hook.remove()
    for key in results[0][0]:
        torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
    for left, right in zip(results[0][1:3], results[1][1:3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(results[0][3:], results[1][3:]):
        for actual, expected in zip(left, right):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for gradients in results[1][3:]:
        named = dict(zip(params, gradients))
        for bank in banks.values():
            assert any(named[name].abs().sum() > 1e-12 for name in bank)
    assert encoder.prefix_cache_stats["misses"] == 2 and encoder.prefix_cache_stats["hits"] == 3
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "token_type_ids"])
def test_prefix_cache_keys_every_prepared_tensor(miniature_qwen, field):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    encoder.encode_processed(inputs, role="learner")
    changed = {k: v.clone() for k, v in inputs.items()}
    if field == "input_ids":
        changed[field][0, -1] = 7
    elif field == "attention_mask":
        changed[field][0, 0] = 1
    elif field == "pixel_values":
        changed[field][0, 0] += .125
    elif field == "image_grid_thw":
        changed[field] = changed[field].to(torch.int32)
    else:
        changed[field][0, 0] = 1
    encoder.encode_processed(changed, role="learner")
    assert encoder.prefix_cache_stats["misses"] == 2 and encoder.prefix_cache_stats["hits"] == 0


@pytest.mark.parametrize("entries,byte_limit,expected_entries", [(2, 100_000, 2), (2, 1500, 1), (2, 1, 0)])
def test_prefix_cache_respects_entry_byte_limits_and_lru(miniature_qwen, entries, byte_limit, expected_entries):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True,
        prefix_cache_max_entries=entries, prefix_cache_max_bytes=byte_limit)
    inputs = processed_images()
    for token in (3, 4, 5):
        inputs["input_ids"][0, -1] = token
        encoder.encode_processed(inputs, role="learner")
        assert encoder.prefix_cache_stats["bytes"] <= byte_limit
    assert encoder.prefix_cache_stats["entries"] == expected_entries
    if expected_entries:
        hits = encoder.prefix_cache_stats["hits"]
        inputs["input_ids"][0, -1] = 5
        encoder.encode_processed(inputs, role="learner")
        assert encoder.prefix_cache_stats["hits"] == hits + 1
        inputs["input_ids"][0, -1] = 3
        misses = encoder.prefix_cache_stats["misses"]
        encoder.encode_processed(inputs, role="learner")
        assert encoder.prefix_cache_stats["misses"] == misses + 1


def test_prefix_cache_clear_publication_reload_dtype_and_provenance(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    encoder.encode_processed(inputs, role="learner")
    encoder.mark_published()
    encoder.encode_processed(inputs, role="learner")
    assert encoder.prefix_cache_stats["hits"] == 1
    encoder.clear_caches()
    assert encoder.prefix_cache_stats["entries"] == encoder.prefix_cache_stats["bytes"] == 0
    assert encoder.prefix_cache_stats["hits"] == 1
    encoder.encode_processed(inputs, role="learner")
    encoder.provenance["checkpoint"] = "changed"
    encoder.encode_processed(inputs, role="learner")
    assert encoder.prefix_cache_stats["misses"] == 3
    encoder.load_state_dict(encoder.state_dict())
    assert encoder.prefix_cache_stats["entries"] == 0
    encoder.encode_processed(inputs, role="learner")
    encoder.to(dtype=torch.float64)
    assert encoder.prefix_cache_stats["entries"] == 0


@pytest.mark.parametrize("requires_grad", [False, True])
def test_frozen_prefix_functional_parameter_overrides_never_hit_stale_cache(miniature_qwen, monkeypatch, requires_grad):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    monkeypatch.setattr(encoder, "prepare", lambda data, **kwargs: (inputs, (2, 1)))
    data = batch(1, count=2)
    baseline = encoder(data)
    name = "qwen.model.language_model.embed_tokens.weight"
    original = dict(encoder.named_parameters())[name]
    override = (original.detach().clone() * 2).requires_grad_(requires_grad)
    expected = None
    for enabled in (False, True):
        encoder.enable_prefix_cache = enabled
        encoder(data)
        output = torch.func.functional_call(encoder, {name: override}, (data,))
        if not enabled:
            expected = output
        else:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert encoder.prefix_cache_stats["hits"] == 0
            if requires_grad:
                assert encoder.prefix_cache_stats["bypasses"] == 1
    assert not torch.equal(output, baseline)


def test_prefix_cache_detects_frozen_inplace_mutation_and_buffer_override(miniature_qwen, monkeypatch):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    monkeypatch.setattr(encoder, "prepare", lambda data, **kwargs: (inputs, (2, 1)))
    data = batch(1, count=2)
    encoder(data)
    with torch.no_grad():
        miniature_qwen.model.language_model.layers[0].self_attn.q_proj.weight.mul_(2)
    encoder(data)
    assert encoder.prefix_cache_stats["misses"] == 2
    key = "qwen.model.language_model.rotary_emb.inv_freq"
    replacement = dict(encoder.named_buffers())[key].clone() * .7
    cached = torch.func.functional_call(encoder, {key: replacement}, (data,))
    assert encoder.prefix_cache_stats["misses"] == 3
    encoder.enable_prefix_cache = False
    uncached = torch.func.functional_call(encoder, {key: replacement}, (data,))
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_cached_prefix_does_not_alias_decoder_inputs_and_bypasses_stochastic_training(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    expected = encoder.encode_processed(inputs, role="learner")
    stored = [v.clone() for v in next(iter(encoder._prefix_cache.values()))]

    def overwrite(module, args):
        args[0].zero_()

    hook = miniature_qwen.model.language_model.layers[-1].register_forward_pre_hook(overwrite)
    encoder.encode_processed(inputs, role="learner")
    hook.remove()
    for actual, reference in zip(next(iter(encoder._prefix_cache.values())), stored):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(encoder.encode_processed(inputs, role="learner"), expected, rtol=0, atol=0)
    encoder.train()
    encoder.encode_processed(inputs, role="learner")
    assert encoder.prefix_cache_stats["bypasses"] == 1 and encoder.prefix_cache_stats["entries"] == 0


def test_prefix_cache_preserves_frozen_role_bank_isolation(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    adapter = miniature_qwen.model.language_model.layers[-1].self_attn.q_proj
    with torch.no_grad():
        adapter.banks.teacher.B.fill_(.2)
        adapter.banks.evaluator.B.fill_(-.1)
    inputs = processed_images()
    references = {role: encoder.encode_processed(inputs, role=role) for role in ("teacher", "evaluator")}
    nonzero_factors(encoder)
    for role, expected in references.items():
        actual = encoder.encode_processed(inputs, role=role)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
    assert not torch.equal(references["teacher"], references["evaluator"])
    assert encoder.prefix_cache_stats["misses"] == 1 and encoder.prefix_cache_stats["hits"] == 3


def test_prefix_populated_in_inference_mode_remains_usable_for_learner_gradients(miniature_qwen):
    encoder = models.QwenContextEncoder(miniature_qwen, None, enable_prefix_cache=True)
    inputs = processed_images()
    with torch.inference_mode():
        encoder.encode_processed(inputs, role="teacher")
    output = encoder.encode_processed(inputs, role="learner")
    output.square().sum().backward()
    assert encoder.prefix_cache_stats["hits"] == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters() if p.requires_grad)


@pytest.mark.parametrize("limits", [{"prefix_cache_max_entries": 0}, {"prefix_cache_max_bytes": 0},
                                  {"prefix_cache_max_entries": True}])
def test_prefix_cache_limits_require_positive_integers(miniature_qwen, limits):
    with pytest.raises(ValueError, match="positive integer"):
        models.QwenContextEncoder(miniature_qwen, None, **limits)
