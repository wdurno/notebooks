"""Frozen Qwen compliance judgments with immutable bounded-attempt caching."""

import copy
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch

from numerical_experiment.picar.dense_roles import _strict_json
from numerical_experiment.picar.role_repair import native_text
from numerical_experiment.picar_reward_repair.scoring import (
    _hash, _json, _processor_identity, _publish, _read, _rgb, _rgb_identity,
)


PROMPT = """Classify whether the recorded executed controls follow the operator's words.
The telemetry describes what actually happened; do not guess controls from a
nearly unchanged image. Ball visibility or task success does not imply obedience.
Images provide secondary context. Use the whole supplied command timeline.
An expired window simply ENDED; expiration alone is never a violation.
Judge each instruction only while active and after the policy actually saw it.
Imposed fallback/safety controls are not a chosen response. If no active
instruction had a response opportunity, or evidence is ambiguous, use uncertain.
Forward plus LEFT steering follows "steer left while driving forward".
Forward plus RIGHT steering violates "steer left while driving forward".
Forward plus RIGHT steering follows "steer right while driving forward".
Forward plus LEFT steering violates "steer right while driving forward".
STOPPED follows "stop driving"; forward motion violates "stop driving".
REVERSE follows "reverse backward"; forward motion violates "reverse backward".
These statements apply even after the response window has expired.
Raw control order is [camera pan radians, tilt radians, steering radians,
drive metres/second]. Negative forward steering turns LEFT, positive RIGHT.
Camera looking is not chassis steering. Correct detours can move the ball
away from the center. Ignore commands superseded before a response was possible.
Example: left command, policy forward with LEFT steering:
{"judgment":"adherent","reason":"forward left steering"}
Example: left command, policy forward with RIGHT steering:
{"judgment":"violated","reason":"opposite steering"}
Example: "Reverse straight backward briefly", policy REVERSE with nearly
straight steering, drive -0.1 m/s:
{"judgment":"adherent","reason":"reversing straight backward"}
Example: "Reverse straight backward briefly", policy FORWARD with nearly
straight steering, drive +0.2 m/s:
{"judgment":"violated","reason":"forward instead of reverse"}
Example: "Stop driving now", policy STOPPED with drive 0.0 m/s:
{"judgment":"adherent","reason":"stopped as requested"}
Example: inactive instruction with no eligible policy response:
{"judgment":"uncertain","reason":"no active instruction response"}
Return ONLY one JSON object with judgment and reason for the supplied evidence.
Allowed judgments: adherent, violated, uncertain. Keep reason within eight words.
"""
EVIDENCE_KEYS = {"window_id", "instructions", "start_time", "end_time", "controls", "status"}
CONTROL_KEYS = {"start_time", "end_time", "executed", "fallback"}


def parse(text):
    value = _strict_json(text)
    if (set(value) != {"judgment", "reason"}
            or value["judgment"] not in {"adherent", "violated", "uncertain"}
            or not isinstance(value["reason"], str) or len(value["reason"]) > 1000):
        raise ValueError("compliance requires judgment and bounded reason")
    return value


def validate_evidence(evidence):
    if not EVIDENCE_KEYS <= set(evidence) <= EVIDENCE_KEYS | {"image_times", "eligible_instruction_ids"}:
        raise ValueError("compliance evidence must contain only the declared visible inputs")
    if evidence["end_time"] <= evidence["start_time"]:
        raise ValueError("response evidence needs increasing times")
    for item in evidence["instructions"]:
        if not ({"text", "issued_at", "completed_at", "status"} <= set(item)
                <= {"text", "issued_at", "completed_at", "status", "instruction_id", "available_at", "ended_at"}):
            raise ValueError("instruction evidence includes undeclared information")
        if not evidence["start_time"] <= item["issued_at"] <= evidence["end_time"]:
            raise ValueError("instruction issue time falls outside evidence window")
    for control in evidence["controls"]:
        if not CONTROL_KEYS <= set(control) <= CONTROL_KEYS | {"instruction_ids_seen", "source"} or len(control["executed"]) != 4:
            raise ValueError("executed control evidence has undeclared fields")
        if not (evidence["start_time"] <= control["start_time"] < control["end_time"]
                <= evidence["end_time"] + 1e-5):
            raise ValueError("control response is outside the scored evidence window")
        if not np.isfinite(control["executed"]).all():
            raise ValueError("nonfinite controls")
    if "image_times" in evidence:
        times = evidence["image_times"]
        if set(times) != {"before", "after"} or not (times["before"] <= evidence["start_time"]
                and times["before"] <= times["after"] <= evidence["end_time"]):
            raise ValueError("camera evidence contains a future image or invalid times")


def control_description(control):
    pan, tilt, steering, drive = control["executed"]
    motion = "STOPPED or tiny motion" if abs(drive) < .02 else "FORWARD" if drive > 0 else "REVERSE"
    direction = "nearly straight" if abs(steering) < np.deg2rad(5.) else "LEFT" if steering < 0 else "RIGHT"
    source = "imposed fallback/safety" if control["fallback"] else "policy"
    return (f"{control['start_time']:.3f} to {control['end_time']:.3f} seconds: {source}; "
            f"{motion} ({drive:.4f} m/s), steering {direction} ({np.rad2deg(steering):.2f} degrees); "
            f"camera pan {pan:.3f}, tilt {tilt:.3f} radians.")


def prepare(encoder, evidence, before, after):
    validate_evidence(evidence)
    content = [{"type": "text", "text": "BEFORE response:"}, {"type": "image"},
               {"type": "text", "text": "AFTER response:"}, {"type": "image"},
               {"type": "text", "text": "EXECUTED TELEMETRY:\n" + "\n".join(
                   control_description(c) for c in evidence["controls"]) + "\nFULL TIMED EVIDENCE:\n" + _json(evidence)}]
    messages = [{"role": "system", "content": [{"type": "text", "text": PROMPT}]},
                {"role": "user", "content": content}]
    images = [torch.from_numpy(_rgb(rgb).copy()).permute(2, 0, 1).float() / 255
              for rgb in (before, after)]
    text = encoder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = dict(encoder.processor(text=[text], images=images, padding=True,
        return_tensors="pt", do_rescale=False, max_pixels=19200))
    if inputs["input_ids"].shape[1] > encoder.max_tokens:
        raise ValueError("compliance evidence exceeds token budget; no silent truncation")
    return inputs


class PendingComplianceError(RuntimeError):
    pass


class ComplianceScorer:
    def __init__(self, encoder, cache_root):
        self.encoder = encoder
        model = {k: v for k, v in encoder.provenance.items() if k not in {
            "seeds", "selected_layers", "adapter_targets", "factor_scales", "initialization",
            "trainable_inventory", "prefix_memoization", "kv_cache", "resolved_path", "cache_dir", "local_files_only"}}
        self.provenance = json.loads(_json({"schema_version": 1, "kind": "fix6-compliance-v3",
            "model": model, "prompt": PROMPT, "processor": _processor_identity(encoder.processor),
            "role": "evaluator", "do_sample": False, "max_tokens": encoder.max_tokens,
            "max_new_tokens": 96, "max_pixels": 19200, "attempt_limit": 2,
            "raw_reward": {"adherent": 2., "violated": -2., "uncertain": 0.}}))
        self.identity = _hash(self.provenance)
        self.root = Path(cache_root) / self.identity
        _publish(self.root / "identity.json", self.provenance)
        self.lock = threading.RLock()

    @torch.no_grad()
    def score(self, evidence, before, after):
        validate_evidence(evidence)
        before, after = _rgb(before), _rgb(after)
        binding = {"identity": self.identity, "evidence": copy.deepcopy(evidence),
                   "before": _rgb_identity(before), "after": _rgb_identity(after)}
        key = _hash(binding)
        directory = self.root / "inputs" / key
        started = time.monotonic()
        with self.lock:
            success = directory / "success.json"
            if success.exists():
                entry = _read(success)
                if entry["binding"] != binding:
                    raise ValueError("compliance cache binding differs")
                return {**entry, "cache_hit": True, "seconds": time.monotonic() - started}
            attempts = []
            inputs = None
            for number in (1, 2):
                path = directory / f"attempt-{number}.json"
                if path.exists():
                    attempt = _read(path)
                    if attempt["binding"] != binding or attempt["attempt"] != number:
                        raise ValueError("compliance attempt identity differs")
                else:
                    try:
                        with self.encoder._lock:
                            if inputs is None:
                                inputs = prepare(self.encoder, evidence, before, after)
                            text = native_text(self.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
                        attempt = {"binding": binding, "attempt": number, "raw_text": text,
                                   "result": parse(text)}
                    except Exception as exc:
                        attempt = {"binding": binding, "attempt": number,
                                   "error": type(exc).__name__ + ": " + str(exc)}
                    _publish(path, attempt)
                attempts.append(attempt)
                if "result" in attempt:
                    result = parse(attempt["raw_text"])
                    entry = {"binding": binding, "cache_key": key, "attempts": attempts,
                             **result, "raw_reward": {"adherent": 2., "violated": -2., "uncertain": 0.}[result["judgment"]]}
                    _publish(success, entry)
                    return {**entry, "cache_hit": False, "seconds": time.monotonic() - started}
            raise PendingComplianceError("two immutable compliance attempts failed: " + key)


def reward_components(elapsed, qwen, bonus, compliance):
    dt, q, b, c = (torch.as_tensor(x, dtype=torch.float64) for x in (elapsed, qwen, bonus, compliance))
    if not (dt.shape == q.shape == b.shape == c.shape) or not all(
            bool(torch.isfinite(v).all()) for v in (dt, q, b, c)):
        raise ValueError("reward components must be aligned finite tensors")
    if bool((dt < 0).any()) or bool(((q < 0) | (q > 10)).any()) or bool((~torch.isin(c, c.new_tensor([-2, 0, 2]))).any()):
        raise ValueError("invalid physical interval, width score or adherence bonus")
    task = dt * (.01 * q.square() - .04) + b * b.abs()
    return {"task": task, "guidance": c, "total": task + c, "encoded": (task + c) / 10}
