"""Server-side causal instruction ledger and optional queued recorded-audio STT.

No microphone, model import, worker, download, or robot access occurs on import
or construction. Clock values must share the coordinator's absolute clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import queue
import re
import threading
import time
from typing import Callable


def _time(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("event times must be finite nonnegative seconds")


def _identity(value: str) -> None:
    if not isinstance(value, str) or not 0 < len(value) <= 160:
        raise ValueError("event IDs must be nonempty strings of at most 160 characters")


@dataclass(frozen=True)
class InstructionEvent:
    event_id: str
    text: str
    utterance_start: float
    utterance_end: float
    available_at: float
    source: str = "typed"
    kind: str = "instruction"
    target_id: str | None = None

    def __post_init__(self):
        _identity(self.event_id)
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 4096:
            raise ValueError("event text must contain 1--4096 characters")
        for value in (self.utterance_start, self.utterance_end, self.available_at):
            _time(value)
        if not self.utterance_start <= self.utterance_end <= self.available_at:
            raise ValueError("utterance must end before transcript availability")
        if self.source not in ("typed", "speech"):
            raise ValueError("source must be typed or speech")
        if self.kind not in ("instruction", "correction", "interrupted", "completed", "status"):
            raise ValueError("unknown instruction event kind")
        targeted = self.kind in ("correction", "interrupted", "completed")
        if targeted != (self.target_id is not None):
            raise ValueError("correction/interrupted/completed require a target ID; others do not")
        if targeted:
            _identity(self.target_id)
            if self.target_id == self.event_id:
                raise ValueError("an event cannot target itself")


class InstructionLedger:
    """Append-only event identities; the owner checkpoints this with its trajectory.

    No model dependency. context() exports exactly LanguageEvent's three fields.
    It never backdates transcript availability to the original utterance time.
    """

    SCHEMA_VERSION = 1

    def __init__(self, *, clock_domain: str = "coordinator_monotonic_seconds"):
        _identity(clock_domain)
        self.clock_domain = clock_domain
        self._events: dict[str, InstructionEvent] = {}

    def contains(self, event_id: str) -> bool:
        return event_id in self._events

    def get(self, event_id: str) -> InstructionEvent | None:
        return self._events.get(event_id)

    def record(self, event: InstructionEvent) -> bool:
        if not isinstance(event, InstructionEvent):
            raise TypeError("expected InstructionEvent")
        previous = self._events.get(event.event_id)
        if previous is not None:
            if previous != event:
                raise ValueError("conflicting delivery for existing event ID")
            return False
        if event.target_id is not None:
            target = self._events.get(event.target_id)
            if (target is None or target.kind not in ("instruction", "correction")
                    or target.available_at > event.available_at):
                raise ValueError("target must be an already-recorded causally available instruction")
        self._events[event.event_id] = event
        return True

    def typed(self, event_id: str, text: str, *, at: float,
              kind: str = "instruction", target_id: str | None = None) -> bool:
        return self.record(InstructionEvent(event_id, text, at, at, at, "typed", kind, target_id))

    def transcript(self, event_id: str, text: str, *, utterance_start: float,
                   utterance_end: float, available_at: float, kind: str = "instruction",
                   target_id: str | None = None) -> bool:
        return self.record(InstructionEvent(event_id, text, utterance_start, utterance_end,
                                            available_at, "speech", kind, target_id))

    def events(self, as_of: float) -> tuple[InstructionEvent, ...]:
        _time(as_of)
        return tuple(sorted((e for e in self._events.values() if e.available_at <= as_of),
                            key=lambda e: (e.available_at, e.event_id)))

    def context(self, as_of: float) -> list[dict]:
        events = self.events(as_of)
        closed = {}
        for event in events:
            if event.target_id is not None:
                closed[event.target_id] = min(closed.get(event.target_id, event.available_at),
                                              event.available_at)
        return [{"text": e.text, "issued_at": e.available_at,
                 "completed_at": closed.get(e.event_id)}
                for e in events if e.kind in ("instruction", "correction")]

    def status_events(self, as_of: float) -> list[dict]:
        return [{"text": e.text, "available_at": e.available_at, "kind": e.kind}
                for e in self.events(as_of) if e.kind != "instruction"]

    def model_status_events(self, as_of: float) -> list[dict]:
        """Model-batch status schema; original kinds remain in the ledger/export."""
        return [{**event, "kind": "robot_status"} for event in self.status_events(as_of)]

    def snapshot(self) -> dict:
        return {"schema_version": self.SCHEMA_VERSION, "clock_domain": self.clock_domain,
                "events": [asdict(e) for e in self._events.values()]}

    @classmethod
    def from_snapshot(cls, value: dict) -> InstructionLedger:
        if (set(value) != {"schema_version", "clock_domain", "events"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != cls.SCHEMA_VERSION
                or not isinstance(value["events"], list)):
            raise ValueError("incompatible instruction-ledger snapshot")
        ledger = cls(clock_domain=value["clock_domain"])
        for entry in value["events"]:
            if not ledger.record(InstructionEvent(**entry)):
                raise ValueError("duplicate event ID in snapshot")
        return ledger


@dataclass(frozen=True)
class AudioUtterance:
    """A finalized recorded file, not a microphone handle or mutable sample buffer."""

    event_id: str
    path: Path
    utterance_start: float
    utterance_end: float
    kind: str = "instruction"
    target_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        # Reuse the same timing, identity and correction validation as typed input.
        InstructionEvent(self.event_id, "pending", self.utterance_start, self.utterance_end,
                         self.utterance_end, "speech", self.kind, self.target_id)


@dataclass(frozen=True)
class TranscriptionOutcome:
    event_id: str
    event: InstructionEvent | None
    error: str | None
    queued_at: float
    started_at: float
    finished_at: float
    inference_s: float


class SpeechBridge:
    """One explicit server worker, bounded pending work, no waiting in submit/drain.

    transcriber.transcribe_file(path) may return text or an object with .text.
    Queue overload raises queue.Full (never silently drops operator input).
    Errors become outcomes, not invisible worker deaths or automatic retries.
    """

    def __init__(self, transcriber, *, clock: Callable[[], float] = time.monotonic,
                 max_pending: int = 8, max_audio_bytes: int = 32 * 1024 * 1024):
        if type(max_pending) is not int or not 1 <= max_pending <= 1024:
            raise ValueError("max_pending must be an integer in [1, 1024]")
        if type(max_audio_bytes) is not int or max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be positive")
        self.transcriber, self.clock = transcriber, clock
        self.max_pending, self.max_audio_bytes = max_pending, max_audio_bytes
        self._jobs = queue.Queue(maxsize=max_pending)
        self._results = queue.Queue(maxsize=max_pending)
        self._pending: dict[str, AudioUtterance] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.abandoned_ids: tuple[str, ...] = ()

    def start(self) -> SpeechBridge:
        with self._lock:
            if self._closed:
                raise RuntimeError("speech bridge closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="picar-server-stt", daemon=True)
                self._thread.start()
        return self

    def submit(self, utterance: AudioUtterance, *, ledger: InstructionLedger | None = None) -> bool:
        if not isinstance(utterance, AudioUtterance):
            raise TypeError("expected AudioUtterance")
        with self._lock:
            if self._closed or self._thread is None:
                raise RuntimeError("speech bridge must be explicitly started and open")
            if ledger is not None and ledger.contains(utterance.event_id):
                original = ledger.get(utterance.event_id)
                if (original.source != "speech" or original.utterance_start != utterance.utterance_start
                        or original.utterance_end != utterance.utterance_end
                        or original.kind != utterance.kind or original.target_id != utterance.target_id):
                    raise ValueError("conflicting resumed audio delivery for event ID")
                return False  # A restored transcript must not be transcribed/injected again.
            previous = self._pending.get(utterance.event_id)
            if previous is not None:
                if previous != utterance:
                    raise ValueError("conflicting audio delivery for event ID")
                return False
            if len(self._pending) >= self.max_pending:
                raise queue.Full("speech backlog full; retain/retry audio outside the control loop")
            queued = self.clock()
            _time(queued)
            if queued < utterance.utterance_end:
                raise ValueError("cannot submit an utterance before it has ended")
            self._jobs.put_nowait((utterance, queued))
            self._pending[utterance.event_id] = utterance
            return True

    def _run(self):
        while not self._stop.is_set():
            try:
                utterance, queued = self._jobs.get(timeout=.02)
            except queue.Empty:
                continue
            started, timer = self.clock(), time.perf_counter()
            event, error = None, None
            try:
                if not utterance.path.is_file() or not 0 < utterance.path.stat().st_size <= self.max_audio_bytes:
                    raise ValueError("recorded audio missing, empty or exceeds max_audio_bytes")
                result = self.transcriber.transcribe_file(utterance.path)
                text = result if isinstance(result, str) else result.text
                finished = self.clock()
                event = InstructionEvent(utterance.event_id, text.strip(), utterance.utterance_start,
                                         utterance.utterance_end, finished, "speech",
                                         utterance.kind, utterance.target_id)
            except Exception as exc:
                finished = self.clock()
                error = f"{type(exc).__name__}: {exc}"[:1024]
            outcome = TranscriptionOutcome(utterance.event_id, event, error, queued, started,
                                           finished, time.perf_counter() - timer)
            if not self._stop.is_set():
                # Pending includes completed-but-undrained items, so this cannot overflow.
                self._results.put_nowait(outcome)
            self._jobs.task_done()

    def drain(self, ledger: InstructionLedger) -> list[TranscriptionOutcome]:
        if self._closed:
            raise RuntimeError("speech bridge closed; resubmit abandoned IDs after restoring ledger")
        outcomes = []
        # Bound this call even while the producer is running.
        for _ in range(self.max_pending):
            try:
                outcome = self._results.get_nowait()
            except queue.Empty:
                break
            try:
                if outcome.event is not None:
                    ledger.record(outcome.event)
            except Exception:
                # Do not lose a correction if its target has not been restored yet.
                self._results.put_nowait(outcome)
                raise
            with self._lock:
                self._pending.pop(outcome.event_id, None)
            outcomes.append(outcome)
        return outcomes

    def close(self, *, timeout: float = .2) -> bool:
        """Cancel pending work; return False if a native inference call still runs.

        No forced thread termination or claim of persisted work. Persist audio IDs
        and ledger together at a drained boundary, then resubmit unfinished IDs.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("close timeout must be finite and nonnegative")
        with self._lock:
            self._closed = True
            self._stop.set()
            self.abandoned_ids = tuple(self._pending)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def __enter__(self) -> SpeechBridge:
        return self.start()

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class STTConfig:
    # Exact revision observed in the prior design's local Hub refs, not a floating alias.
    repo_id: str = "Zoont/faster-whisper-large-v3-turbo-int8-ct2"
    revision: str = "846f74797293e5f605d509a6ab5395a1f2f2874e"
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] /
                           "cache" / "picar" / "speech" / "hub")
    model_path: Path | None = None
    allow_downloads: bool = True
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 5
    cpu_threads: int = 2

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("speech model revision must be an exact 40-hex commit")
        if any(type(x) is not int or x <= 0 for x in (self.beam_size, self.cpu_threads)):
            raise ValueError("beam_size and cpu_threads must be positive integers")


class FasterWhisperTranscriber:
    """Lazy real STT for finalized audio files; use only in the server worker.

    Dependencies/model factories are injectable for no-network acceptance tests.
    Reuse paths are optional; a fresh clone resolves the pinned model to cache/.
    """

    def __init__(self, config: STTConfig | None = None, *, model_loader=None, snapshot_loader=None):
        self.config = config or STTConfig()
        self._model_loader, self._snapshot_loader = model_loader, snapshot_loader
        self._model = None

    def _load(self):
        if self._model is None:
            config = self.config
            if config.model_path is not None:
                path = Path(config.model_path)
            else:
                loader = self._snapshot_loader
                if loader is None:
                    from huggingface_hub import snapshot_download
                    loader = snapshot_download
                path = Path(loader(repo_id=config.repo_id, revision=config.revision,
                                   cache_dir=str(config.cache_dir),
                                   local_files_only=not config.allow_downloads))
                if path.name != config.revision:
                    raise RuntimeError("speech checkpoint does not match pinned revision")
            if not all((path / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
                raise FileNotFoundError("incomplete speech model; expected model.bin/config.json/tokenizer.json")
            model_loader = self._model_loader
            if model_loader is None:
                from faster_whisper import WhisperModel
                model_loader = WhisperModel
            self._model = model_loader(str(path), device=config.device, compute_type=config.compute_type,
                                       cpu_threads=config.cpu_threads, num_workers=1, local_files_only=True)
        return self._model

    def transcribe_file(self, path: Path) -> str:
        segments, _ = self._load().transcribe(str(path), beam_size=self.config.beam_size,
                                             language=self.config.language, vad_filter=False)
        # The iterator performs inference lazily. Exhaust it before availability is stamped.
        return " ".join(segment.text.strip() for segment in segments).strip()
