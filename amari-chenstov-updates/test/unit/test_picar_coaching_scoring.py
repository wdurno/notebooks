"""Immutable compliance attempts and complete-evidence cache identity."""

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from numerical_experiment.picar_coaching import scoring


def evidence():
    return {"window_id": "window", "start_time": 1., "end_time": 3., "status": "expired",
        "instructions": [{"text": "Turn left while moving forward.", "issued_at": 1.,
                           "completed_at": None, "status": "active"}],
        "controls": [{"start_time": 1.5, "end_time": 2.5, "executed": [0., 0., -.4, .2], "fallback": False}]}


def encoder():
    return SimpleNamespace(provenance={"checkpoint": "test-only"}, processor=SimpleNamespace(),
                           max_tokens=2048, _lock=threading.RLock())


def test_full_input_cache_and_restart_cannot_change_adherence_or_spend_twice(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scoring, "prepare", lambda *args: {"test_only": True})
    monkeypatch.setattr(scoring, "native_text", lambda *a, **k: calls.append(1) or
                        ['{"judgment":"adherent","reason":"negative forward steering"}'])
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    evaluator = scoring.ComplianceScorer(encoder(), tmp_path)
    first = evaluator.score(evidence(), rgb, rgb)
    assert first["raw_reward"] == 2 and len(calls) == 1
    restored = scoring.ComplianceScorer(encoder(), tmp_path)
    assert restored.score(evidence(), rgb, rgb)["cache_hit"] and len(calls) == 1
    changed = evidence()
    changed["instructions"][0]["text"] = "Turn right while moving forward."
    assert restored.score(changed, rgb, rgb)["cache_key"] != first["cache_key"] and len(calls) == 2
    changed = evidence()
    changed["controls"][0]["executed"][2] *= -1
    assert restored.score(changed, rgb, rgb)["cache_key"] != first["cache_key"] and len(calls) == 3


def test_invalid_generation_is_not_neutral_and_attempt_limit_survives_restart(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scoring, "prepare", lambda *args: {"test_only": True})
    monkeypatch.setattr(scoring, "native_text", lambda *a, **k: calls.append(1) or ["invalid"])
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(2):
        with pytest.raises(scoring.PendingComplianceError):
            scoring.ComplianceScorer(encoder(), tmp_path).score(evidence(), rgb, rgb)
    assert len(calls) == 2
    assert not list(tmp_path.rglob("success.json"))


def test_controls_can_cover_window_after_last_camera_without_using_future_images():
    value = evidence()
    value["image_times"] = {"before": 1., "after": 2.}
    value["controls"][0]["end_time"] = 3.
    scoring.validate_evidence(value)
    assert "FORWARD" in scoring.control_description(value["controls"][0])
    assert "LEFT" in scoring.control_description(value["controls"][0])
    value["image_times"]["after"] = 3.1
    with pytest.raises(ValueError, match="future image"):
        scoring.validate_evidence(value)
