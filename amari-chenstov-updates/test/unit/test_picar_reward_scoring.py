"""Frozen RGB-only context, reference recovery and durable CPU fake scoring."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from numerical_experiment.picar.storage import FrameStore, RunStore
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_reward_repair import scoring as rs


class Processor:
    chat_template = "frozen template v1"

    def __init__(self):
        self.calls = []
        self.tokens = 32
        self.image_count = 7
        self.setting = "unchanged"
        self.tokenizer = SimpleNamespace(special_tokens_map={"pad_token": "<pad>"},
            get_vocab=lambda: {"<pad>": 0, "a": 1}, padding_side="left", truncation_side="right")

    def to_dict(self):
        return {"setting": self.setting}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = copy.deepcopy(messages)
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return json.dumps(messages)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.inputs = kwargs
        return {"input_ids": torch.arange(self.tokens)[None],
                "attention_mask": torch.ones((1, self.tokens), dtype=torch.int64),
                "image_grid_thw": torch.ones((self.image_count, 3), dtype=torch.int64),
                "pixel_values": torch.stack([rgb.flatten() for rgb in kwargs["images"]])}


def encoder():
    return SimpleNamespace(processor=Processor(), max_tokens=2048, _lock=threading.RLock(),
        provenance={"kind": "fake_native", "declared_revision": "pinned-revision", "checkpoint_manifest": {"weights": "abc"},
                    "seeds": {"model": 1}, "trainable_inventory": {"ltm": 4}})


def references():
    return tuple({"reference": identity, "width_percent": width, "reward": reward,
                  "rgb": np.full((4, 5, 3), i + 1, dtype=np.uint8)}
                 for i, (identity, width, reward) in enumerate(zip(rs.ORDER, rs.WIDTHS, rs.REWARDS)))


def target(value=100):
    return np.full((4, 5, 3), value, dtype=np.uint8)


def generator(monkeypatch, outputs):
    calls = []
    iterator = iter(outputs)

    def native(enc, inputs, **kwargs):
        assert kwargs == {"role": "evaluator", "max_new_tokens": 96}
        calls.append({k: v.clone() for k, v in inputs.items()})
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return [value]

    monkeypatch.setattr(rs, "native_text", native)
    return calls


VALID = '{"width_percent": 16, "reward": 7.25}'  # Deliberate arithmetic disagreement.


def test_exact_ordered_reference_answers_rgb_and_target_only_context(tmp_path, monkeypatch):
    enc, refs = encoder(), references()
    calls = generator(monkeypatch, [VALID])
    scorer = rs.WidthScorer(enc, refs, tmp_path)
    history = {"images": torch.stack([torch.full((3, 4, 5), .9),
                                      torch.full((3, 4, 5), 100 / 255)])[None],
               "instructions": "SECRET task", "geometry": "SECRET pose", "scores": [9.9],
               "actions": "SECRET actions", "elapsed": "SECRET timestamps", "masks": "SECRET mask"}
    out = scorer.score(history, pan_rad=object())
    assert out["score"] == out["reward"] == 7.25
    assert out["width_percent"] == 16
    assert out["raw_text"] == VALID
    assert out["diagnostic"]["reward_arithmetic_disagreement"] == 5.25
    messages = enc.processor.messages
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 6 + ["user"]
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": rs.PROMPT}]}
    assert "SECRET" not in json.dumps(messages)
    for i, ref in enumerate(refs):
        assert messages[2 + 2 * i]["content"][0]["text"] == json.dumps({
            "width_percent": round(ref["width_percent"], 4), "reward": round(ref["reward"], 4)})
        np.testing.assert_array_equal((enc.processor.inputs["images"][i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), ref["rgb"])
    assert messages[-1] == {"role": "user", "content": [{"type": "image"},
        {"type": "text", "text": "Score the image and return JSON."}]}
    assert len(enc.processor.inputs["images"]) == 7
    assert enc.processor.inputs["max_pixels"] == 76800
    assert enc.processor.inputs["do_rescale"] is False
    np.testing.assert_array_equal((enc.processor.inputs["images"][-1].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), target())
    # Pan, history, train/test role and condition never participate in scoring.
    assert scorer.score_rgb(target())["score"] == 7.25
    history["images"][0, 0] = float("nan")
    history["instructions"] = "different task"
    assert scorer.score(history, pan_rad=999.)["diagnostic"]["cache_hit"]
    assert len(calls) == 1
    assert scorer.provenance["input_whitelist"] == ["RGB"]


def test_disk_cache_and_state_resume_use_identity_not_model_seed_or_trainable_banks(tmp_path, monkeypatch):
    calls = generator(monkeypatch, [VALID, '{"width_percent": 8, "reward": 1}'])
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    first = scorer.score_rgb(target())
    state = scorer.state_dict()
    success = next(tmp_path.rglob("success.json"))
    seal = (success.read_bytes(), success.stat().st_mtime_ns)
    other_encoder = encoder()
    other_encoder.provenance["seeds"] = {"model": 9876}
    other_encoder.provenance["trainable_inventory"] = {"ltm": 100}
    resumed = rs.WidthScorer(other_encoder, references(), tmp_path)
    resumed.load_state_dict(state)
    assert resumed.identity == scorer.identity
    assert resumed.score_rgb(target())["diagnostic"]["cache_hit"]
    assert resumed.score_rgb(target())["raw_text"] == first["raw_text"]
    assert (success.read_bytes(), success.stat().st_mtime_ns) == seal
    resumed.score_rgb(target(101))
    assert len(calls) == 2
    state["provenance"]["model"]["declared_revision"] = "mutated"
    assert scorer.provenance["model"]["declared_revision"] == "pinned-revision"
    with pytest.raises(ValueError, match="identity"):
        resumed.load_state_dict(state)


@pytest.mark.parametrize("change", ["model", "processor", "template", "reference_pixels", "tokenizer"])
def test_any_evaluator_input_identity_change_invalidates_resume(tmp_path, change):
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    enc, refs = encoder(), references()
    if change == "model":
        enc.provenance["checkpoint_manifest"]["weights"] = "different"
    elif change == "processor":
        enc.processor.setting = "different"
    elif change == "template":
        enc.processor.chat_template = "different"
    elif change == "tokenizer":
        enc.processor.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer": "different"}')
    else:
        refs[0]["rgb"][0, 0, 0] = 35
    other = rs.WidthScorer(enc, refs, tmp_path)
    assert scorer.identity != other.identity
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(scorer.state_dict())


@pytest.mark.parametrize("raw", ["not JSON", "{}", '{"width_percent": 3, "reward": true}',
    '{"width_percent": 3, "reward": -1}', '{"width_percent": 101, "reward": 1}',
    '{"width_percent": 3, "reward": 11}', '{"width_percent": NaN, "reward": 0}',
    '{"width_percent": 3, "reward": 1, "reward": 0}', '{"width_percent": 3, "reward": "0"}',
    '{"width_percent": 3, "reward": 0, "pose": 12}'])
def test_two_invalid_attempts_are_durable_pending_not_zero_and_not_retried_on_resume(tmp_path, monkeypatch, raw):
    calls = generator(monkeypatch, [raw, raw])
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    with pytest.raises(rs.PendingScoreError) as error:
        scorer.score_rgb(target())
    diagnostic = error.value.diagnostic
    assert diagnostic["pending"]
    assert "score" not in diagnostic
    assert [a["raw_text"] for a in diagnostic["attempts"]] == [raw, raw]
    assert len({a["context_sha256"] for a in diagnostic["attempts"]}) == 1
    assert len(list(tmp_path.rglob("attempt-*.json"))) == 2
    assert not list(tmp_path.rglob("success.json"))
    for key in calls[0]:
        torch.testing.assert_close(calls[0][key], calls[1][key])
    resumed = rs.WidthScorer(encoder(), references(), tmp_path)
    resumed.load_state_dict(scorer.state_dict())
    with pytest.raises(rs.PendingScoreError):
        resumed.score_rgb(target())
    assert len(calls) == 2


def test_invalid_first_attempt_then_valid_zero_is_a_real_success(tmp_path, monkeypatch):
    calls = generator(monkeypatch, ["invalid", '{"width_percent": 0, "reward": 0}'])
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    out = scorer.score_rgb(target())
    assert out["score"] == 0
    assert out["attempts"][0]["raw_text"] == "invalid"
    assert out["attempts"][1]["attempt"] == 2
    assert len(calls) == 2
    assert rs.WidthScorer(encoder(), references(), tmp_path).score_rgb(target())["diagnostic"]["cache_hit"]


def test_generation_errors_never_become_zero(tmp_path, monkeypatch):
    generator(monkeypatch, [RuntimeError("failed generation"), RuntimeError("failed generation")])
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    with pytest.raises(rs.PendingScoreError):
        scorer.score_rgb(target())
    assert len(scorer.last_diagnostic["attempts"]) == 2
    assert all("generation_error" in a for a in scorer.last_diagnostic["attempts"])


@pytest.mark.parametrize("interrupt_after", ["invalid", "valid"])
def test_crash_after_durable_attempt_neither_loses_raw_text_nor_repeats_success(tmp_path, monkeypatch, interrupt_after):
    calls = generator(monkeypatch, ["bad" if interrupt_after == "invalid" else VALID, VALID])
    publish = rs._publish

    def interrupted(path, value):
        publish(path, value)
        if path.name == "attempt-1.json":
            raise KeyboardInterrupt("simulated crash after flush")

    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    monkeypatch.setattr(rs, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        scorer.score_rgb(target())
    monkeypatch.setattr(rs, "_publish", publish)
    resumed = rs.WidthScorer(encoder(), references(), tmp_path)
    assert resumed.score_rgb(target())["score"] == 7.25
    assert len(calls) == (2 if interrupt_after == "invalid" else 1)


def test_concurrent_cache_clients_generate_the_same_input_only_once(tmp_path, monkeypatch):
    calls = generator(monkeypatch, [VALID])
    clients = [rs.WidthScorer(encoder(), references(), tmp_path) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda scorer: scorer.score_rgb(target()), clients))
    assert all(out["score"] == 7.25 for out in outputs)
    assert sorted(out["diagnostic"]["cache_hit"] for out in outputs) == [False, True]
    assert len(calls) == 1


def test_retry_after_resume_rejects_changed_processed_context(tmp_path, monkeypatch):
    generator(monkeypatch, ["invalid"])
    publish = rs._publish

    def interrupted(path, value):
        publish(path, value)
        if path.name == "attempt-1.json":
            raise KeyboardInterrupt

    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    monkeypatch.setattr(rs, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        scorer.score_rgb(target())
    monkeypatch.setattr(rs, "_publish", publish)
    enc = encoder()
    enc.processor.tokens = 33  # Simulate a processor bug not declared in its config.
    resumed = rs.WidthScorer(enc, references(), tmp_path)
    with pytest.raises(ValueError, match="processed context changed"):
        resumed.score_rgb(target())
    assert len(list(tmp_path.rglob("attempt-*.json"))) == 1


@pytest.mark.parametrize("case", ["overflow", "missing_image", "reference_order", "reference_label", "encoder_limit"])
def test_context_and_reference_validation_never_truncates(tmp_path, monkeypatch, case):
    enc, refs = encoder(), references()
    if case == "overflow":
        enc.processor.tokens = 2049
    elif case == "missing_image":
        enc.processor.image_count = 6
    elif case == "reference_order":
        refs = tuple(reversed(refs))
    elif case == "reference_label":
        refs[0]["reward"] = .1
    else:
        enc.max_tokens = 4096
    calls = generator(monkeypatch, [])
    with pytest.raises(ValueError):
        rs.WidthScorer(enc, refs, tmp_path).score_rgb(target())
    assert not calls


@pytest.mark.parametrize("rgb", [np.zeros((4, 5, 3)), np.zeros((4, 5), dtype=np.uint8),
                                 np.zeros((0, 5, 3), dtype=np.uint8)])
def test_rgb_api_rejects_non_rgb_or_non_byte_data(tmp_path, rgb):
    with pytest.raises(ValueError, match="uint8 RGB"):
        rs.WidthScorer(encoder(), references(), tmp_path).score_rgb(rgb)


def test_cached_entry_target_binding_is_checked(tmp_path, monkeypatch):
    generator(monkeypatch, [VALID])
    scorer = rs.WidthScorer(encoder(), references(), tmp_path)
    scorer.score_rgb(target())
    path = next(tmp_path.rglob("success.json"))
    original = json.loads(path.read_text())
    # Fault injection through the same atomic writer's immutable contract.
    with pytest.raises(ValueError, match="immutable"):
        rs._publish(path, {**original, "score": 0})
    assert json.loads(path.read_text()) == original


def _complete(root, config, summary, images=()):
    with RunStore(root, {"schema_version": 1, **config}) as store:
        refs = [store.frames.put(image) for image in images]
        store.checkpoint({"fixture": True})
        store.complete({"schema_version": 1, **summary})
    return refs


@pytest.fixture
def source_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "cache" / "probes"
    raw = root / "raw"
    refs = references()
    original_items = [refs[i] for i in (0, 1, 2, 5)]
    development_items = [refs[i] for i in (3, 4)]
    frame_refs = _complete(raw, {}, {}, [r["rgb"] for r in original_items])
    original_refs = [{k: r[k] for k in ("reference", "width_percent", "reward")} | {"frame_ref": f}
                     for r, f in zip(original_items, frame_refs)]
    original = root / "original"
    _complete(original, {"source": str(raw), "source_completion_sha256": sha256_file(raw / "COMPLETED"),
                        "format": "few-shot", "references": original_refs}, {"format": "few-shot"})
    development = root / "development"
    with RunStore(development, {"schema_version": 1}) as store:
        panels = [{"panel_id": panel, "frame_ref": store.frames.put(r["rgb"]),
                   "truth": {"width_percent": r["width_percent"]}}
                  for panel, r in zip(("C17", "C18"), development_items)]
        store.checkpoint({"fixture": True})
        store.complete({"schema_version": 1, "panels": panels})
    dev_refs = [{k: r[k] for k in ("reference", "width_percent", "reward")} | {"frame_ref": p["frame_ref"]}
                for r, p in zip(development_items, panels)]
    bank = [next(r for r in original_refs + dev_refs if r["reference"] == identity) for identity in rs.ORDER]
    bank = [{**r, "source_group": "development" if r["reference"] in "FE" else "original"} for r in bank]
    fresh = root / "fresh"
    _complete(fresh, {}, {})
    config = {"kind": "width_refinement_evaluation_v1", "prompt": rs.PROMPT, "role": "evaluator",
              "methods": {"six_examples": list(rs.ORDER)}, "references": bank,
              "max_new_tokens": 96, "max_tokens": 2048, "max_pixels_per_image": 76800,
              "source_run": str(original), "source_completion_sha256": sha256_file(original / "COMPLETED"),
              "development_run": str(development), "development_completion_sha256": sha256_file(development / "COMPLETED"),
              "panels_run": str(fresh), "panels_completion_sha256": sha256_file(fresh / "COMPLETED"),
              "reference_frames_root": str(raw)}
    source = root / rs.REFERENCE_RUN
    return SimpleNamespace(source=source, bundle=tmp_path / "cache" / "bundle", config=config,
                           summary={"kind": "width_refinement_evaluation_v1"}, originals=original_refs, refs=refs)


def test_load_references_recovers_original_and_development_rgb_and_immutable_metadata(source_bundle):
    s = source_bundle
    _complete(s.source, s.config, s.summary)
    bank = rs.load_references(s.source, s.bundle)
    assert isinstance(bank, tuple)
    assert tuple(r["reference"] for r in bank) == rs.ORDER
    for actual, expected in zip(bank, s.refs):
        np.testing.assert_array_equal(actual["rgb"], expected["rgb"])
        assert actual["width_percent"] == expected["width_percent"]
        assert actual["reward"] == expected["reward"]
        assert actual["metadata"]["parents"]["evaluation"]["completion_sha256"] == sha256_file(s.source / "COMPLETED")
        with pytest.raises(TypeError):
            actual["reward"] = 0
        with pytest.raises(TypeError):
            actual["metadata"]["parents"]["evaluation"]["path"] = "elsewhere"
        with pytest.raises(ValueError):
            actual["rgb"].setflags(write=True)
        np.testing.assert_array_equal(FrameStore(s.bundle / "frames", readonly=True).get(actual["frame_ref"]), actual["rgb"])
    assert len(list((s.bundle / "frames").glob("*.npy"))) == 6
    manifest = (s.bundle / "references.json").read_bytes()
    assert rs.load_references(s.source, s.bundle)[0]["frame_ref"] == bank[0]["frame_ref"]
    assert (s.bundle / "references.json").read_bytes() == manifest


@pytest.mark.parametrize("fault", ["order", "labels", "parent_seal", "original_identity", "frame_source", "prompt", "cap", "outside_cache"])
def test_reference_recovery_rejects_changed_frozen_contract(source_bundle, fault):
    s = source_bundle
    if fault == "order":
        s.config["references"] = list(reversed(s.config["references"]))
    elif fault == "labels":
        s.config["references"][0]["reward"] = .5
    elif fault == "parent_seal":
        s.config["source_completion_sha256"] = "0" * 64
    elif fault == "original_identity":
        s.config["references"][0]["frame_ref"] = s.config["references"][1]["frame_ref"]
    elif fault == "frame_source":
        s.config["reference_frames_root"] = s.config["development_run"]
    elif fault == "prompt":
        s.config["prompt"] = "another prompt"
    elif fault == "cap":
        s.config["max_pixels_per_image"] = 19200
    else:
        s.bundle = s.bundle.parents[1] / "outside-cache"
    _complete(s.source, s.config, s.summary)
    with pytest.raises(ValueError):
        rs.load_references(s.source, s.bundle)


def test_reference_recovery_requires_completed_sealed_source(source_bundle):
    s = source_bundle
    s.source.mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete"):
        rs.load_references(s.source, s.bundle)
