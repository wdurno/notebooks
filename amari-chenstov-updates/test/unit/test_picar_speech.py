from dataclasses import replace
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import wave

import pytest

from demo.picar.speech import (
    AudioUtterance, FasterWhisperTranscriber, InstructionEvent, InstructionLedger,
    SpeechBridge, STTConfig,
)


def test_typed_transcribed_equivalence_and_no_early_transcript_visibility():
    typed, speech = InstructionLedger(), InstructionLedger()
    typed.typed("typed", "Turn left", at=5.)
    speech.transcript("audio", "Turn left", utterance_start=1., utterance_end=2., available_at=5.)
    assert typed.context(5.) == speech.context(5.) == [
        {"text": "Turn left", "issued_at": 5., "completed_at": None}]
    assert speech.context(4.999) == []
    assert speech.get("audio").utterance_start == 1.
    assert speech.get("audio").source == "speech"
    assert set(speech.context(5.)[0]) == {"text", "issued_at", "completed_at"}


def test_corrections_interruption_completion_and_as_of_reconstruction():
    ledger = InstructionLedger(clock_domain="episode_seconds")
    ledger.typed("go", "Go to the red ball", at=1.)
    ledger.transcript("correct", "No, the blue ball", utterance_start=2., utterance_end=3.,
                      available_at=6., kind="correction", target_id="go")
    ledger.typed("pause", "Wait for me", at=7., kind="interrupted", target_id="correct")
    ledger.typed("resume", "Go to the blue ball", at=8.)
    ledger.typed("done", "Task complete", at=10., kind="completed", target_id="resume")
    ledger.typed("reply", "Acknowledged", at=11., kind="status")
    assert ledger.context(5.) == [{"text": "Go to the red ball", "issued_at": 1., "completed_at": None}]
    assert ledger.context(6.) == [
        {"text": "Go to the red ball", "issued_at": 1., "completed_at": 6.},
        {"text": "No, the blue ball", "issued_at": 6., "completed_at": None}]
    assert ledger.context(7.)[1]["completed_at"] == 7.
    assert ledger.context(9.)[-1]["completed_at"] is None
    assert ledger.context(10.)[-1]["completed_at"] == 10.
    assert ledger.status_events(5.) == []
    assert ledger.status_events(10.) == [
        {"text": "No, the blue ball", "available_at": 6., "kind": "correction"},
        {"text": "Wait for me", "available_at": 7., "kind": "interrupted"},
        {"text": "Task complete", "available_at": 10., "kind": "completed"}]
    restored = InstructionLedger.from_snapshot(json.loads(json.dumps(ledger.snapshot())))
    assert restored.clock_domain == "episode_seconds"
    for at in (0., 1., 5., 6., 7., 8., 10., 11.):
        assert restored.context(at) == ledger.context(at)
        assert restored.status_events(at) == ledger.status_events(at)
    assert not restored.record(ledger.get("correct"))
    assert restored.snapshot() == ledger.snapshot()


def test_duplicate_conflict_validation_and_out_of_order_delivery():
    ledger = InstructionLedger()
    assert ledger.typed("later", "later", at=10.)
    assert ledger.typed("earlier", "earlier", at=2.)
    assert [e["issued_at"] for e in ledger.context(10.)] == [2., 10.]
    assert not ledger.typed("earlier", "earlier", at=2.)
    with pytest.raises(ValueError, match="conflicting"):
        ledger.typed("earlier", "different", at=2.)
    with pytest.raises(ValueError, match="target"):
        ledger.typed("bad", "correct", at=1., kind="correction", target_id="later")
    with pytest.raises(ValueError, match="target"):
        ledger.typed("bad", "complete", at=11., kind="completed", target_id="missing")
    with pytest.raises(ValueError, match="availability"):
        InstructionEvent("bad", "late text", 1., 4., 3., "speech")
    with pytest.raises(ValueError):
        ledger.context(float("nan"))
    snapshot = ledger.snapshot()
    snapshot["schema_version"] = 100
    with pytest.raises(ValueError, match="incompatible"):
        InstructionLedger.from_snapshot(snapshot)
    snapshot = ledger.snapshot()
    snapshot["events"].append(snapshot["events"][0])
    with pytest.raises(ValueError, match="duplicate"):
        InstructionLedger.from_snapshot(snapshot)


def test_model_status_export_matches_batch_schema_without_mutating_causal_history():
    import torch
    from numerical_experiment.picar.models import LanguageEvent, validate_batch

    ledger = InstructionLedger()
    ledger.typed("go", "Go to the ball", at=1.)
    ledger.transcript("fix", "Use the blue ball", utterance_start=2., utterance_end=3.,
                      available_at=5., kind="correction", target_id="go")
    ledger.typed("pause", "Wait", at=6., kind="interrupted", target_id="fix")
    ledger.typed("done", "Finished", at=7., kind="completed", target_id="fix")
    ledger.typed("reply", "Acknowledged", at=8., kind="status")
    snapshot = ledger.snapshot()
    assert ledger.model_status_events(4.999) == []
    assert ledger.model_status_events(5.) == [
        {"text": "Use the blue ball", "available_at": 5., "kind": "robot_status"}]
    model_events = ledger.model_status_events(8.)
    assert model_events == [{**event, "kind": "robot_status"} for event in ledger.status_events(8.)]
    assert [event["kind"] for event in ledger.status_events(8.)] == [
        "correction", "interrupted", "completed", "status"]
    assert ledger.snapshot() == snapshot
    batch = {
        "images": torch.zeros(1, 1, 3, 2, 2),
        "executed_actions": torch.zeros(1, 1, 4),
        "elapsed": torch.ones(1, 1, 1),
        "contexts": [[LanguageEvent(**event) for event in ledger.context(8.)]],
        "decision_time": torch.tensor([[8.]]),
        "episode_deadline": torch.tensor([20.]),
        "status_events": [model_events],
    }
    assert validate_batch(batch, max_history=1) == (1, 1)
    restored = InstructionLedger.from_snapshot(snapshot)
    assert restored.model_status_events(8.) == model_events


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "recorded.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(320))
    return path


def collect(bridge, ledger, *, count=1):
    deadline = time.monotonic() + 2
    outcomes = []
    while len(outcomes) < count and time.monotonic() < deadline:
        outcomes.extend(bridge.drain(ledger))
        if len(outcomes) < count:
            time.sleep(.005)
    assert len(outcomes) == count
    return outcomes


def test_background_worker_delays_availability_deduplicates_and_reports_cost(recording):
    entered, release = threading.Event(), threading.Event()
    now = [3.]
    calls = []
    class Transcriber:
        def transcribe_file(self, path):
            calls.append(path)
            entered.set()
            assert release.wait(2)
            return SimpleNamespace(text="  Turn left  ")
    ledger = InstructionLedger()
    bridge = SpeechBridge(Transcriber(), clock=lambda: now[0], max_pending=1)
    utterance = AudioUtterance("audio", recording, 1., 2.)
    with pytest.raises(RuntimeError, match="started"):
        bridge.submit(utterance)
    bridge.start()
    try:
        assert bridge.submit(utterance)
        assert entered.wait(1)
        assert bridge.drain(ledger) == [] and ledger.context(3.) == []
        assert not bridge.submit(utterance)
        with pytest.raises(ValueError, match="conflicting"):
            bridge.submit(replace(utterance, utterance_start=0.))
        with pytest.raises(queue.Full):
            bridge.submit(replace(utterance, event_id="overflow"))
        now[0] = 7.
        release.set()
        outcome, = collect(bridge, ledger)
        assert outcome.error is None and outcome.inference_s >= 0
        assert (outcome.queued_at, outcome.started_at, outcome.finished_at) == (3., 3., 7.)
        assert ledger.context(6.999) == []
        assert ledger.context(7.)[0]["issued_at"] == 7.
        restored = InstructionLedger.from_snapshot(ledger.snapshot())
        assert not bridge.submit(utterance, ledger=restored)
        with pytest.raises(ValueError, match="conflicting resumed"):
            bridge.submit(replace(utterance, utterance_end=2.5), ledger=restored)
        assert calls == [recording]
    finally:
        release.set()
        assert bridge.close(timeout=2)


def test_transcribed_correction_uses_same_ledger_as_typed_input(recording):
    ledger = InstructionLedger()
    ledger.typed("original", "Go forward", at=1.)
    class Transcriber:
        def transcribe_file(self, path):
            return "Actually go backward"
    with SpeechBridge(Transcriber(), clock=lambda: 5.) as bridge:
        bridge.submit(AudioUtterance("correction", recording, 2., 3., "correction", "original"))
        collect(bridge, ledger)
    assert ledger.context(4.) == [{"text": "Go forward", "issued_at": 1., "completed_at": None}]
    assert ledger.context(5.)[0]["completed_at"] == 5.
    assert ledger.context(5.)[1]["text"] == "Actually go backward"
    assert ledger.status_events(5.)[0]["kind"] == "correction"


def test_transcriber_failure_empty_audio_and_oversized_file_are_visible(recording):
    class Transcriber:
        def transcribe_file(self, path):
            raise RuntimeError("injected STT failure")
    ledger = InstructionLedger()
    with SpeechBridge(Transcriber(), clock=lambda: 4.) as bridge:
        bridge.submit(AudioUtterance("failed", recording, 1., 2.))
        outcome, = collect(bridge, ledger)
        assert "injected STT failure" in outcome.error
        assert outcome.event is None and not ledger.contains("failed")
        # Failure is explicit and retry is the owner's decision, not automatic.
        bridge.submit(AudioUtterance("retry", recording, 1., 2.))
        assert collect(bridge, ledger)[0].error
    with SpeechBridge(Transcriber(), clock=lambda: 4., max_audio_bytes=1) as bridge:
        bridge.submit(AudioUtterance("big", recording, 1., 2.))
        assert "max_audio_bytes" in collect(bridge, ledger)[0].error
    recording.write_bytes(b"")
    with SpeechBridge(Transcriber(), clock=lambda: 4.) as bridge:
        bridge.submit(AudioUtterance("empty", recording, 1., 2.))
        assert "empty" in collect(bridge, ledger)[0].error


def test_bounded_shutdown_reports_unfinished_work_without_late_injection(recording):
    entered, release = threading.Event(), threading.Event()
    class Transcriber:
        def transcribe_file(self, path):
            entered.set()
            assert release.wait(2)
            return "late result"
    ledger = InstructionLedger()
    bridge = SpeechBridge(Transcriber(), clock=lambda: 4.).start()
    try:
        bridge.submit(AudioUtterance("inflight", recording, 1., 2.))
        assert entered.wait(1)
        assert not bridge.close(timeout=0.)
        assert bridge.abandoned_ids == ("inflight",)
        assert ledger.context(100.) == []
        with pytest.raises(RuntimeError, match="closed"):
            bridge.drain(ledger)
        with pytest.raises(RuntimeError, match="closed"):
            bridge.start()
    finally:
        release.set()
        assert bridge.close(timeout=2)


def test_missing_correction_target_does_not_silently_lose_worker_result(recording):
    class Transcriber:
        def transcribe_file(self, path):
            return "Correction"
    ledger = InstructionLedger()
    bridge = SpeechBridge(Transcriber(), clock=lambda: 5.).start()
    try:
        bridge.submit(AudioUtterance("correct", recording, 2., 3., "correction", "target"))
        deadline = time.monotonic() + 2
        while bridge._results.empty() and time.monotonic() < deadline:
            time.sleep(.005)
        with pytest.raises(ValueError, match="target"):
            bridge.drain(ledger)
        ledger.typed("target", "Original", at=1.)
        outcome, = collect(bridge, ledger)
        assert outcome.event.event_id == "correct"
        assert ledger.context(5.)[0]["completed_at"] == 5.
    finally:
        assert bridge.close(timeout=2)


def model_files(path):
    path.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (path / name).write_bytes(b"injected model placeholder")


def test_real_wrapper_is_lazy_pinned_project_cached_and_exhausts_segment_iterator(tmp_path):
    config = STTConfig(allow_downloads=False)
    assert config.cache_dir == Path(__file__).resolve().parents[2] / "cache/picar/speech/hub"
    path = tmp_path / config.revision
    model_files(path)
    calls = []
    def snapshot(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(path)
    def loader(source, **kwargs):
        calls.append(("model", source, kwargs))
        class Model:
            def transcribe(self, audio, **options):
                calls.append(("transcribe", audio, options))
                def segments():
                    calls.append(("inference",))
                    yield SimpleNamespace(text=" Go ")
                    yield SimpleNamespace(text=" left ")
                return segments(), SimpleNamespace()
        return Model()
    transcriber = FasterWhisperTranscriber(config, model_loader=loader, snapshot_loader=snapshot)
    assert calls == []
    assert transcriber.transcribe_file(Path("injected.wav")) == "Go left"
    assert calls[0] == ("snapshot", {"repo_id": config.repo_id, "revision": config.revision,
                                    "cache_dir": str(config.cache_dir), "local_files_only": True})
    assert calls[1][2]["local_files_only"] is True
    assert calls[1][2]["device"] == "cpu"
    assert calls[-1] == ("inference",)
    transcriber.transcribe_file(Path("another.wav"))
    assert sum(c[0] == "model" for c in calls) == 1


def test_model_override_never_downloads_and_bad_assets_fail_clearly(tmp_path):
    path = tmp_path / "reuse"
    model_files(path)
    def no_download(**kwargs):
        raise AssertionError("unexpected download")
    transcriber = FasterWhisperTranscriber(
        STTConfig(model_path=path), snapshot_loader=no_download,
        model_loader=lambda *a, **kw: SimpleNamespace(transcribe=lambda *a, **kw: ([], None)))
    assert transcriber.transcribe_file(Path("injected.wav")) == ""
    (path / "model.bin").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        FasterWhisperTranscriber(STTConfig(model_path=path)).transcribe_file(Path("injected.wav"))
    with pytest.raises(ValueError, match="exact"):
        STTConfig(revision="main")
    with pytest.raises(RuntimeError, match="pinned"):
        FasterWhisperTranscriber(snapshot_loader=lambda **kw: path).transcribe_file(Path("injected.wav"))
