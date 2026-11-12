"""Tiny CPU-only normalization and sealed-demonstration restoration checks."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.func import functional_call

from numerical_experiment.picar.config import PicarConfig, canonical_json, config_from_dict
from numerical_experiment.picar.fix2_data import TASK, history_batch
from numerical_experiment.picar.models import PiCarStudent, batch_digest, build_tiny_model
from numerical_experiment.picar.storage import IntegrityError, RunStore
from numerical_experiment.picar.teacher import prepare_teacher_distribution
from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations
from numerical_experiment.picar_phase2_rescue.model import MODES, NormalizedStudent
from src.robotics.actions import Calibration
from src.robotics.simulator import SimulatorConfig
from src.seeding import derive_seed_map


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny(**kwargs):
    return build_tiny_model(31, width=4, depth=1, hidden_size=kwargs.pop("hidden_size", 4),
                           context_length=2, return_bins=7, dtype=torch.float64, **kwargs).eval()


def batch():
    return {"images": torch.linspace(0, 1, 96, dtype=torch.float64).reshape(1, 2, 3, 4, 4),
        "executed_actions": torch.zeros(1, 2, 4, dtype=torch.float64),
        "elapsed": torch.tensor([[[0.], [.1]]], dtype=torch.float64),
        "contexts": [[{"text": "Turn left.", "issued_at": 0.}]],
        "start_time": torch.zeros(1, dtype=torch.float64)}


@pytest.mark.parametrize("mode", MODES)
def test_same_inventory_modules_rng_and_frozen_encoder_roles(mode):
    base = tiny()
    data = batch()
    roles = {r: base.encoder(data, role=r).detach().clone() for r in ("learner", "teacher", "evaluator")}
    parameters = dict(base.named_parameters())
    inventory, banks = base.trainable_inventory(), base.parameter_banks()
    rng = torch.get_rng_state().clone()
    model = NormalizedStudent(base, mode)
    assert model.trainable_inventory() == inventory and model.parameter_banks() == banks
    assert dict(model.named_parameters()).keys() == parameters.keys()
    assert all(p is parameters[n] for n, p in model.named_parameters())
    assert set(model.state_dict()) == set(base.state_dict())
    for name, child in base.named_children():
        assert getattr(model, name) is child
    assert torch.equal(rng, torch.get_rng_state())
    assert model.mode == mode and model.eps == model.provenance["eps"] == 1e-6
    assert model.provenance["parameter_free"]
    for role, original in roles.items():
        torch.testing.assert_close(model.encoder(data, role=role), original, rtol=0, atol=0)
    with torch.no_grad():
        model.projection.ltm.weight.add_(.1)
    for role in ("teacher", "evaluator"):
        torch.testing.assert_close(model.encoder(data, role=role), roles[role], rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_forward_predict_streaming_and_functional_gradients_agree(mode):
    model, data = NormalizedStudent(tiny(), mode), batch()
    model.train()
    trained = model(data)
    model.eval()
    inferred, _, _ = model.predict(data)
    short = {k: v[:, :1] if k in {"images", "executed_actions", "elapsed"} else v
             for k, v in data.items()}
    _, state, _ = model.predict(short)
    streamed, _, _ = model.predict(data, state)
    params = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters() if p.requires_grad}
    functional = functional_call(model, params, (data,))
    for name, value in trained.items():
        for other in (inferred, streamed, functional):
            torch.testing.assert_close(other[name], value, rtol=1e-12, atol=1e-12)
    loss = sum(x.square().sum() for x in functional.values())
    gradients = torch.autograd.grad(loss, tuple(params.values()))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    assert any(n.startswith("encoder.") and bool(g.abs().sum()) for n, g in zip(params, gradients))
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_rms_denominator_gradients_match_finite_differences_and_zero_is_finite(mode):
    model, data = NormalizedStudent(tiny(), mode), batch()
    features = torch.linspace(.1, .8, 8, dtype=torch.float64).reshape(1, 2, 4).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: model._run(data, x)[0]["action_mean"],
                                    (features,), eps=1e-6, atol=1e-7, rtol=1e-4)
    zero = torch.zeros_like(features, requires_grad=True)
    output = model._run(data, zero)[0]
    gradient, = torch.autograd.grad(sum(x.square().sum() for x in output.values()), (zero,))
    assert bool(torch.isfinite(gradient).all())
    assert all(bool(torch.isfinite(v).all()) for v in output.values())


def test_feature_rms_mitigates_large_input_saturation_without_scaling_controls():
    base = tiny()
    with torch.no_grad():
        base.projection.ltm.weight.zero_()
        base.projection.ltm.weight[:, :4].fill_(.5)
        base.projection.ltm.bias.zero_()
    model, data = NormalizedStudent(base), batch()
    features = torch.full((1, 2, 4), 126., dtype=torch.float64)
    seen = []
    hook = model.recurrent.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    PiCarStudent._run(model, data, features)
    before = torch.stack(seen)
    seen.clear()
    projection_inputs = []
    projection_hook = model.projection.register_forward_pre_hook(
        lambda module, args: projection_inputs.append(args[0].detach()))
    actual = model._run(data, features)[0]
    after = torch.stack(seen)
    projection_hook.remove()
    hook.remove()
    assert bool((before.abs() > .99).all())
    assert not bool((after.abs() > .99).any())
    assert float((1 - after.square()).median()) > .05
    torch.testing.assert_close(projection_inputs[0][..., :4].square().mean(-1), torch.ones(1, 2, dtype=torch.float64))
    torch.testing.assert_close(projection_inputs[0][..., 4:],
                               torch.cat((data["executed_actions"], data["elapsed"]), -1))
    reference = PiCarStudent._run(model, data, features / (features.square().mean(-1, keepdim=True) + model.eps).sqrt())[0]
    for name in actual:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-12, atol=1e-12)


def test_projection_rms_bounds_saturated_channels_even_with_large_projection_bias():
    base = tiny(hidden_size=64)
    with torch.no_grad():
        base.projection.ltm.weight.zero_()
        base.projection.ltm.bias.fill_(1e4)
        base.projection.ltm.bias[0] = 1e8
    model, data = NormalizedStudent(base, "feature_and_projection_rms"), batch()
    seen = []
    hook = model.recurrent.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    output = model._run(data, torch.ones(1, 2, 4, dtype=torch.float64))[0]
    hook.remove()
    values = torch.stack(seen)
    threshold = math.atanh(.99)
    assert float((values.abs() > .99).double().mean()) <= 1 / threshold**2
    assert int((values[0].abs() > .99).sum()) == 1
    assert all(bool(torch.isfinite(v).all()) for v in output.values())


def test_model_rejects_trained_or_invalid_variants_and_versions_extra_state():
    base = tiny()
    with pytest.raises(ValueError, match="mode"):
        NormalizedStudent(base, "unknown")
    for eps in (0., -1., float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="epsilon"):
            NormalizedStudent(base, eps=eps)
    base.distillation_receipt = {"steps": 1}
    with pytest.raises(ValueError, match="fresh"):
        NormalizedStudent(base)
    base.distillation_receipt = None
    base.mark_published()
    with pytest.raises(ValueError, match="fresh"):
        NormalizedStudent(base)
    model = NormalizedStudent(tiny())
    other = NormalizedStudent(tiny())
    other.load_state_dict(model.state_dict())
    assert other.get_extra_state() == model.get_extra_state()
    with pytest.raises(ValueError, match="identity"):
        NormalizedStudent(tiny(), MODES[1]).set_extra_state(model.get_extra_state())


@pytest.fixture
def source(tmp_path):
    base = PicarConfig()
    config = replace(base, seed=31, image_width=8, image_height=8, task_version="ball-contact-v1",
        model=replace(base.model, hidden_size=4, context_length=2, return_bins=161),
        learning=replace(base.learning, fd_strength=0., return_min=-6., return_max=12.,
                         reward_version="dense-visual-centering-v1"))
    resolved = {"config": config.to_dict(), "component_seeds": config.seeds(0), "model_initialization_seed": 191101}
    episodes, digests = [], []
    calibration = Calibration()
    with RunStore(tmp_path / "source", resolved, checkpoint_policy="journal-v1") as store:
        for i in range(2):
            rgb = [np.full((8, 8, 3), 20 + 70*i + j*15, dtype=np.uint8) for j in range(3)]
            refs = [store.frames.put(x) for x in rgb]
            intervals = []
            for j in range(3):
                row = {"index": j, "arrival": 3*i + j, "terminated": j == 2, "truncated": False,
                    "frame_ref": refs[j], "probabilities": None, "logits": None, "teacher_diagnostic": None,
                    "controls": asdict(calibration.neutral())}
                if j:
                    action = 2 if j == 1 else (0 if i == 0 else 7)
                    target = prepare_teacher_distribution(torch.nn.functional.one_hot(torch.tensor([action]), 8).float(),
                        probability_floor=.001, target_kind="hard_engineered")
                    controls = calibration.map_distribution(target.executed_probabilities[0])
                    if j == 2:
                        controls = calibration.neutral(controls)
                    history = history_batch([{"rgb": rgb[k], "time": .1*k, "elapsed": .1 if k else 0.,
                        "action": tuple(asdict(calibration.neutral()).values())} for k in range(j)],
                        contexts=[{"text": TASK, "issued_at": 0.}], deadline=20., width=8, height=8,
                        statuses=[{"text": "Approaching.", "available_at": .05, "kind": "robot_status"}])
                    digest = batch_digest(history)
                    digests.append(digest)
                    row.update(probabilities=target.executed_probabilities[0].tolist(),
                        logits=target.reference_logits[0].tolist(), baseline=0., controls=asdict(controls),
                        teacher_diagnostic={"actions": [action], "raw_text": ["synthetic recorded decision"]},
                        context_refs=refs[:j], context_actions=history["executed_actions"][0].tolist(),
                        context_elapsed=history["elapsed"][0, :, 0].tolist(),
                        context_decision_times=history["decision_time"][0].tolist(),
                        context_start_time=float(history["start_time"][0]), episode_deadline=20.,
                        contexts=history["contexts"][0], context_status_events=history["status_events"][0],
                        history_digest=digest)
                intervals.append(row)
            edges = torch.linspace(-6., 12., 162)
            episodes.append({"episode_id": f"bootstrap:{i}", "seed": derive_seed_map(config.seeds(0)["layouts"],
                (f"episode_{i}",))[f"episode_{i}"], "complete": True, "label_complete": True,
                "truncated": False, "evaluation": False, "intervals": intervals,
                "reset_exposure": 0., "layout": {"curriculum": {"curriculum_exposure_intervals": 0.}},
                "returns": [-6., float(edges[80]), 12.], "stage_rewards": [.1, .2, 10.],
                "visual_scores": [{"original": True}], "end_reason": "success", "success": True})
        summary = {"schema_version": 1, "kind": "phase1_fix3_bootstrap", "seed": config.seed,
            "acquired_count": 6, "distinct_bootstrap_samples": 4, "episodes": episodes,
            "distill_receipt": {"batches": 4, "steps": 4, "history_digests": digests,
                "teacher_provenance": {"kind": "synthetic-test-only"}}}
        store.checkpoint({"learner": {"must_not_load": "old Fisher and optimizer"}})
        store.complete(summary)
        return store.root


def reseal_summary(source, mutate):
    """Adversarial synthetic fixtures only: alter content under a valid seal."""
    summary_path, manifest_path = source / "summary.json", source / "manifest.json"
    summary = json.loads(summary_path.read_text())
    mutate(summary)
    encoded = (canonical_json(summary) + "\n").encode()
    summary_path.write_bytes(encoded)
    manifest = json.loads(manifest_path.read_text())
    manifest["final_files"]["summary.json"] = {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
    encoded_manifest = (canonical_json(manifest) + "\n").encode()
    manifest_path.write_bytes(encoded_manifest)
    (source / "COMPLETED").write_text(canonical_json({"manifest_sha256": hashlib.sha256(encoded_manifest).hexdigest()}))


def test_restore_exact_original_histories_targets_bins_seeds_and_no_old_state(source, monkeypatch):
    summary = json.loads((source / "summary.json").read_text())
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("must never unpickle old learner assets"))
    restored = restore_demonstrations(source)
    assert restored["acquired_count"] == 6 and len(restored["records"]) == 4
    assert restored["episodes"] == summary["episodes"]
    assert [r["arrival"] for r in restored["records"]] == [1, 2, 4, 5]
    for record in restored["records"]:
        original = summary["episodes"][int(record["episode_id"].split(":")[1])]
        row = original["intervals"][record["source_interval_index"]]
        assert batch_digest(record["history"]) == row["history_digest"]
        assert record["return"] == original["returns"][record["source_interval_index"]]
        assert record["bin"] == (80 if record["source_interval_index"] == 1 else 160)
        assert not record["learned"] and record["source_seed"] == original["seed"]
        torch.testing.assert_close(record["logits"], torch.tensor(row["logits"]), rtol=0, atol=0)
        torch.testing.assert_close(record["executed_probabilities"][0], torch.tensor(row["probabilities"]), rtol=0, atol=0)
    again = restore_demonstrations(source)
    assert again["provenance"] == restored["provenance"]
    assert not restored["provenance"]["old_fisher_reused"]
    assert not restored["provenance"]["old_anchors_reused"]
    assert not restored["provenance"]["rewards_recomputed"]
    assert torch.equal(rng, torch.get_rng_state())
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("defect", ["episode_seed", "source_seed", "history_hash", "teacher_action",
    "probabilities", "logits", "controls", "terminal_controls", "context_control", "arrival", "returns", "receipt",
    "missing_teacher_action", "fallback_control", "terminal_alignment", "easy_exposure"])
def test_rejects_semantically_corrupt_sealed_demonstrations(source, defect):
    def mutate(summary):
        episode = summary["episodes"][0]
        row = episode["intervals"][1]
        if defect == "episode_seed":
            episode["seed"] += 1
        elif defect == "source_seed":
            summary["seed"] += 1
        elif defect == "history_hash":
            row["contexts"][0]["text"] = "Altered instruction."
        elif defect == "teacher_action":
            row["teacher_diagnostic"]["actions"][0] = 3
        elif defect == "missing_teacher_action":
            row["teacher_diagnostic"] = None
        elif defect == "fallback_control":
            episode["intervals"][0]["controls"]["drive_mps"] = .1
        elif defect == "terminal_alignment":
            row["terminated"] = True
        elif defect == "easy_exposure":
            episode["reset_exposure"] = 100.
        elif defect in {"probabilities", "logits"}:
            row[defect][0] += .01
        elif defect == "controls":
            row["controls"]["drive_mps"] = 0.
        elif defect == "terminal_controls":
            episode["intervals"][-1]["controls"]["drive_mps"] = .1
        elif defect == "context_control":
            row["context_actions"][0][-1] = .5
        elif defect == "arrival":
            row["arrival"] = 0
        elif defect == "returns":
            episode["returns"][1] = 13.
        else:
            summary["distill_receipt"]["history_digests"].reverse()
    reseal_summary(source, mutate)
    with pytest.raises(ValueError):
        restore_demonstrations(source)


@pytest.mark.parametrize("defect", ["missing_frame", "corrupt_frame", "missing_marker", "summary_checksum"])
def test_rejects_missing_or_corrupt_immutable_artifacts(source, defect):
    if defect in {"missing_frame", "corrupt_frame"}:
        summary = json.loads((source / "summary.json").read_text())
        ref = summary["episodes"][0]["intervals"][1]["context_refs"][0]
        path = source / "frames" / (ref[7:] + ".npy")
        if defect == "missing_frame":
            path.unlink()
        else:
            path.write_bytes(b"corrupt frame")
    elif defect == "missing_marker":
        (source / "COMPLETED").unlink()
    else:
        with (source / "summary.json").open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(IntegrityError):
        restore_demonstrations(source)


@pytest.mark.parametrize("mode", MODES)
def test_restored_fresh_initialization_and_mid_kl_resume_match(source, tmp_path, monkeypatch, mode):
    from numerical_experiment.picar import fix3_student
    from numerical_experiment.picar_phase2.student import ReadinessRun
    from test_picar_fix3_student import equal_learner

    # There is intentionally no score/distribution method: any renewed query
    # fails, and this stage never constructs a simulator worker.
    monkeypatch.setattr(fix3_student, "CenteringTeacher", lambda encoder, variant:
                        SimpleNamespace(provenance={"kind": "recorded-data-only-test"}))
    monkeypatch.setattr(fix3_student, "CenteringReward", lambda encoder:
                        SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None))
    original = config_from_dict(json.loads((source / "config.json").read_text())["config"])
    config = replace(original, learning=replace(original.learning, replay_capacity=2))
    sim = SimulatorConfig(width=8, height=8, task_version="ball-contact-v1")
    store_config = {"test": "new-normalized-initialization", "mode": mode}

    def make(store, stopped=lambda: None):
        base = build_tiny_model(31, width=4, depth=1, hidden_size=4, context_length=2, return_bins=161).eval()
        run = ReadinessRun(NormalizedStudent(base, mode), config, sim, store, stopped,
                          mode="bootstrap", seed=config.seed, max_transitions=6)
        if run.s["stage"] == "collect":
            assert run.learner.version == run.learner.retired_count == 0
            assert all(not bool(x.any()) for x in run.learner.diagonal.values())
            data = restore_demonstrations(source)
            run.s.update(eligible=data["records"], acquired_count=data["acquired_count"],
                         episodes=data["episodes"], stage="distill")
            run.save()
        return run

    with RunStore(tmp_path / "reference", store_config, checkpoint_policy="journal-v1") as store:
        expected = make(store)
        result = expected.run()
        expected_state = store.load_checkpoint(trusted=True)
        assert result["distill_receipt"]["steps"] == expected.s["nll_step"] == 4
        assert result["acquired_count"] == 6 and result["retired_count"] == 2
        assert result["eligible_count"] == 2
    with RunStore(tmp_path / "interrupted", store_config, checkpoint_policy="journal-v1") as store:
        def stop():
            saved = store.load_checkpoint(trusted=True)
            if saved["stage"] == "distill" and saved["kl_step"] == 2:
                raise InterruptedError("mid-KL")
        with pytest.raises(InterruptedError, match="mid-KL"):
            make(store, stop).run()
    with RunStore(tmp_path / "interrupted", store_config, resume=True, checkpoint_policy="journal-v1") as store:
        resumed = make(store)
        resumed.run()
        equal_learner(expected_state["learner"], store.load_checkpoint(trusted=True)["learner"])
        assert resumed.s["distill_receipt"] == expected_state["distill_receipt"]
