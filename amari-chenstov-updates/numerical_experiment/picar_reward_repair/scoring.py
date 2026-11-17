"""Frozen six-example evaluator and durable, role-independent RGB score cache.

load_references returns a tuple in ABCFED order. Each item is a read-only
mapping containing reference, frame_ref, width_percent, reward, rgb, metadata.
RGB is an immutable HWC uint8 array; metadata recursively records source seals
and the original reference record. Only RGB and rounded answers enter Qwen.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import MappingProxyType

import numpy as np
import torch

from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.role_repair import native_text
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import FrameStore
from numerical_experiment.picar_ball_width_probe import PROMPT, parse
from numerical_experiment.picar_reference_reward_probe import prepare_references


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUN = "refined-width-evaluation-v1-20261116"
ORDER = tuple("ABCFED")
WIDTHS = (0., 4.6875, 10.3125, 10.9375, 41.5625, 87.8125)
REWARDS = (0., .5859375, 1.2890625, 1.3671875, 5.1953125, 10.)
CACHE_SCHEMA = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _rgb(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
            or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape) < 1):
        raise ValueError("expected nonempty HWC uint8 RGB")
    # A bytes backing store prevents callers from re-enabling write access.
    return np.frombuffer(value.tobytes(), dtype=np.uint8).reshape(value.shape)


def _rgb_identity(rgb):
    return {"shape": list(rgb.shape), "dtype": "uint8",
            "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}


def _publish(path, value):
    """Fsync then atomically publish without replacing an existing entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (_json(value) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError("immutable scoring artifact collision: " + str(path))
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _read(path):
    if path.is_symlink():
        raise ValueError("scoring artifacts cannot be symlinks")
    return json.loads(path.read_text())


def _check_bank(references):
    if tuple(r["reference"] for r in references) != ORDER:
        raise ValueError("frozen reference order must be ABCFED")
    for r, width, reward in zip(references, WIDTHS, REWARDS):
        if (type(r["width_percent"]) not in (int, float) or type(r["reward"]) not in (int, float)
                or r["width_percent"] != width or r["reward"] != reward):
            raise ValueError("frozen reference labels changed")


def load_references(source_run, bundle_root):
    """Verify sealed parents, copy six blobs into bundle_root/frames, return tuple.

    bundle_root must be inside this checkout's ignored cache/. The immutable
    references.json records source identities; reopening verifies rather than
    overwrites it. Source evaluation/diagnostic targets are never bundled.
    """
    source, bundle = Path(source_run).resolve(), Path(bundle_root).resolve()
    if source.name != REFERENCE_RUN:
        raise ValueError("expected completed " + REFERENCE_RUN)
    if not bundle.is_relative_to(PROJECT_ROOT / "cache"):
        raise ValueError("reference bundle must be in the project-local ignored cache")
    unit = read_completed_unit(source)
    config = unit["config"]
    if (config.get("kind") != "width_refinement_evaluation_v1"
            or unit["summary"].get("kind") != "width_refinement_evaluation_v1"
            or config.get("prompt") != PROMPT or config.get("role") != "evaluator"
            or config.get("methods", {}).get("six_examples") != list(ORDER)
            or any(config.get(k) != v for k, v in (("max_tokens", 2048),
                ("max_new_tokens", 96), ("max_pixels_per_image", 76800)))):
        raise ValueError("incompatible frozen refinement evaluator")
    references = config["references"]
    _check_bank(references)
    parents, seals = {}, {"evaluation": {"path": str(source), "completion_sha256": sha256_file(source / "COMPLETED")}}
    for name, key in (("original", "source"), ("development", "development"), ("panels", "panels")):
        root = Path(config[key + "_run"]).resolve()
        parent = read_completed_unit(root)
        digest = sha256_file(root / "COMPLETED")
        if digest != config[key + "_completion_sha256"]:
            raise ValueError("reference parent completion hash mismatch: " + name)
        parents[name] = parent
        seals[name] = {"path": str(root), "completion_sha256": digest}
    original = parents["original"]["config"]
    frame_root = Path(config["reference_frames_root"]).resolve()
    if original.get("format") != "few-shot" or Path(original["source"]).resolve() != frame_root:
        raise ValueError("original reference frame source changed")
    read_completed_unit(frame_root)
    digest = sha256_file(frame_root / "COMPLETED")
    if digest != original["source_completion_sha256"]:
        raise ValueError("original frame source completion hash mismatch")
    seals["original_frames"] = {"path": str(frame_root), "completion_sha256": digest}
    stores = {"original": FrameStore(frame_root / "frames", readonly=True),
              "development": FrameStore(Path(seals["development"]["path"]) / "frames", readonly=True)}
    frame_store = None
    result, metadata = [], []
    for ref in references:
        group = "development" if ref["reference"] in "FE" else "original"
        if ref.get("source_group") != group:
            raise ValueError("reference source group changed")
        if group == "original":
            candidates = [r for r in original["references"] if r["reference"] == ref["reference"]]
            if len(candidates) != 1 or any(candidates[0][k] != ref[k] for k in ("frame_ref", "width_percent", "reward")):
                raise ValueError("original reference identity changed")
        else:
            panel_id = "C17" if ref["reference"] == "F" else "C18"
            candidates = [r for r in parents["development"]["summary"]["panels"] if r["panel_id"] == panel_id]
            if (len(candidates) != 1 or candidates[0]["frame_ref"] != ref["frame_ref"]
                    or candidates[0]["truth"]["width_percent"] != ref["width_percent"]):
                raise ValueError("development reference identity changed")
        rgb = _rgb(stores[group].get(ref["frame_ref"]))
        frame_store = frame_store or FrameStore(bundle / "frames")
        if frame_store.put(rgb) != ref["frame_ref"]:
            raise ValueError("bundled reference content changed")
        item = {k: ref[k] for k in ("reference", "frame_ref", "width_percent", "reward")}
        item["metadata"] = {"parents": seals, "source_record": ref, "rgb_identity": _rgb_identity(rgb)}
        metadata.append(item)
        result.append(_freeze({**item, "rgb": rgb}))
    if len({r["frame_ref"] for r in result}) != 6:
        raise ValueError("six distinct reference images required")
    _publish(bundle / "references.json", {"schema_version": CACHE_SCHEMA, "references": metadata})
    return tuple(result)


def _processor_identity(processor):
    result = {"class": type(processor).__module__ + "." + type(processor).__qualname__}
    for key, value in (("processor", processor), ("image_processor", getattr(processor, "image_processor", None))):
        if hasattr(value, "to_dict"):
            result[key] = value.to_dict()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        result["tokenizer"] = {"class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            "special_tokens": tokenizer.special_tokens_map,
            "vocabulary_sha256": _hash(tokenizer.get_vocab()),
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            result["tokenizer"]["backend_sha256"] = _hash(json.loads(backend.to_str()))
    result["chat_template"] = getattr(processor, "chat_template", getattr(tokenizer, "chat_template", None))
    return result


class PendingScoreError(RuntimeError):
    """Two identical attempts are durably recorded, but no valid label exists."""

    def __init__(self, diagnostic):
        self.diagnostic = copy.deepcopy(diagnostic)
        super().__init__("pending frozen width score: " + diagnostic["cache_key"])


class WidthScorer:
    """score_rgb returns score, width_percent, raw_text, attempts and diagnostic.

    score is Qwen's returned reward, never recomputed from width. Successful
    entries and individual attempts are immutable JSON in cache_root/identity/.
    At most two attempts per exact input, including across process restarts.
    state_dict stores identity and last diagnostic only; scores live on disk.
    """

    def __init__(self, encoder, references, cache_root):
        _check_bank(references)
        if encoder.max_tokens != 2048:
            raise ValueError("frozen evaluator requires max_tokens=2048")
        self.encoder = encoder
        self.references = tuple(_freeze({k: r[k] for k in ("reference", "width_percent", "reward")}
            | {"rgb": _rgb(r["rgb"])}) for r in references)
        bank = [{"reference": r["reference"], "rgb": _rgb_identity(r["rgb"]),
                 "answer": {"width_percent": round(r["width_percent"], 4), "reward": round(r["reward"], 4)}}
                for r in self.references]
        if len({_hash(r["rgb"]) for r in bank}) != 6:
            raise ValueError("six distinct reference images required")
        # Adapter state, fitting seeds and local cache paths do not change the
        # native frozen evaluator. Checkpoint/revision and numeric policy do.
        provenance = encoder.provenance
        model = {k: v for k, v in provenance.items() if k not in {
            "seeds", "selected_layers", "adapter_targets", "factor_scales", "initialization",
            "trainable_inventory", "prefix_memoization", "kv_cache", "resolved_path", "cache_dir", "local_files_only"}}
        if not model:
            raise ValueError("frozen model provenance is required")
        self.provenance = json.loads(_json({"schema_version": CACHE_SCHEMA, "kind": "width6_frozen_v1",
            "model": model, "prompt": PROMPT, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "references": bank, "processor": _processor_identity(encoder.processor),
            "role": "evaluator", "max_new_tokens": 96, "max_tokens": 2048,
            "max_pixels_per_image": 76800, "few_shot": True, "do_sample": False,
            "input_whitelist": ["RGB"], "attempt_limit": 2}))
        self.identity = _hash(self.provenance)
        self.root = Path(cache_root).resolve() / self.identity
        _publish(self.root / "identity.json", self.provenance)
        self._lock = threading.RLock()
        self.last_diagnostic = {}

    def state_dict(self):
        return copy.deepcopy({"schema_version": CACHE_SCHEMA, "identity": self.identity,
            "provenance": self.provenance, "last_diagnostic": self.last_diagnostic})

    def load_state_dict(self, value):
        if (value.get("schema_version") != CACHE_SCHEMA or value.get("identity") != self.identity
                or value.get("provenance") != self.provenance):
            raise ValueError("frozen evaluator identity changed")
        self.last_diagnostic = copy.deepcopy(value.get("last_diagnostic", {}))

    def _validate_attempt(self, attempt, key, target, number):
        if (attempt.get("schema_version") != CACHE_SCHEMA or attempt.get("identity") != self.identity
                or attempt.get("cache_key") != key or attempt.get("target") != target
                or attempt.get("attempt") != number):
            raise ValueError("cached attempt identity mismatch")

    def _result(self, entry, hit, started):
        self.last_diagnostic = {**copy.deepcopy(entry), "cache_hit": hit, "seconds": time.monotonic() - started}
        return {"score": entry["score"], "reward": entry["score"], "width_percent": entry["width_percent"],
                "raw_text": entry["raw_text"], "attempts": copy.deepcopy(entry["attempts"]),
                "diagnostic": copy.deepcopy(self.last_diagnostic)}

    @torch.no_grad()
    def score_rgb(self, rgb):
        rgb = _rgb(rgb)
        target = _rgb_identity(rgb)
        key = _hash({"identity": self.identity, "target": target})
        started = time.monotonic()
        directory = self.root / "inputs" / key
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock, (directory / "LOCK").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            attempts = []
            for number in (1, 2):
                path = directory / f"attempt-{number}.json"
                if path.exists():
                    attempt = _read(path)
                    self._validate_attempt(attempt, key, target, number)
                    attempts.append(attempt)
                elif number == 1 and (directory / "attempt-2.json").exists():
                    raise ValueError("cached attempts have a gap")
            common = {"schema_version": CACHE_SCHEMA, "identity": self.identity, "cache_key": key, "target": target}
            success = directory / "success.json"
            if success.exists():
                entry = _read(success)
                if any(entry.get(k) != v for k, v in common.items()) or entry.get("attempts") != attempts:
                    raise ValueError("cached success identity mismatch")
                parsed = parse(entry["raw_text"])
                if entry["score"] != parsed["reward"] or entry["width_percent"] != parsed["width_percent"]:
                    raise ValueError("cached success disagrees with raw output")
                return self._result(entry, True, started)

            def finish(attempt, hit):
                parsed = parse(attempt["raw_text"])
                rule = min(10., parsed["width_percent"] / 8.)
                entry = {**common, "score": parsed["reward"], "width_percent": parsed["width_percent"],
                    "raw_text": attempt["raw_text"], "attempts": attempts,
                    "rule_reward_from_reported_width": rule, "reward_arithmetic_disagreement": parsed["reward"] - rule}
                _publish(success, entry)
                return self._result(entry, hit, started)

            for attempt in attempts:
                if "parse_error" not in attempt and "generation_error" not in attempt:
                    return finish(attempt, True)  # Recovery after attempt flush, before success publication.
            if len(attempts) < 2:
                with getattr(self.encoder, "_lock", nullcontext()):
                    inputs = prepare_references(self.encoder, self.references, rgb, few_shot=True)
                    if inputs["input_ids"].shape[1] > 2048 or inputs["image_grid_thw"].shape[0] != 7:
                        raise ValueError("expected six references and one target without truncation")
                    binding = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                        "sha256": hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                        for k, v in inputs.items()}
                    context_hash = _hash(binding)
                    if any(a["context_sha256"] != context_hash for a in attempts):
                        raise ValueError("retry processed context changed")
                    for number in range(len(attempts) + 1, 3):
                        tick = time.monotonic()
                        attempt = {**common, "attempt": number, "context_sha256": context_hash}
                        try:
                            raw = native_text(self.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
                            attempt["raw_text"] = raw
                            parse(raw)
                        except ValueError as exc:
                            attempt["parse_error"] = str(exc)
                        except Exception as exc:
                            attempt["generation_error"] = type(exc).__name__ + ": " + str(exc)
                        attempt["seconds"] = time.monotonic() - tick
                        _publish(directory / f"attempt-{number}.json", attempt)
                        attempts.append(attempt)
                        if "parse_error" not in attempt and "generation_error" not in attempt:
                            return finish(attempt, False)
            self.last_diagnostic = {**common, "pending": True, "attempts": copy.deepcopy(attempts),
                                    "seconds": time.monotonic() - started}
            raise PendingScoreError(self.last_diagnostic)

    def score(self, history, pan_rad):
        """Legacy [1,T,3,H,W] float-[0,1] history; only its last RGB is read.

        pan_rad and every other history field are deliberately ignored.
        uint8 image tensors are also accepted without normalization.
        """
        images = history["images"]
        if (not isinstance(images, torch.Tensor) or images.ndim != 5
                or images.shape[0] != 1 or images.shape[1] < 1 or images.shape[2] != 3):
            raise ValueError("score requires a single [1,T,3,H,W] history")
        last = images[0, -1].detach().cpu()
        if last.dtype != torch.uint8:
            if not last.is_floating_point() or not bool(torch.isfinite(last).all()) or bool(((last < 0) | (last > 1)).any()):
                raise ValueError("history RGB must be uint8 or finite float in [0,1]")
            last = (last.to(torch.float64) * 255).round().to(torch.uint8)
        return self.score_rgb(last.permute(1, 2, 0).contiguous().numpy())
