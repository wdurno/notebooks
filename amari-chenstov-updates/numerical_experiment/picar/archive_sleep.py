"""Stopped-only, default-off supplemental rehearsal of already retired data.

This deliberately retains EWC: replaying its source observations is additional
likelihood exposure, not a disjoint-data likelihood decomposition. No method in
this module admits evidence, retires a record, or estimates Fisher information.

Integration: construct an ArchiveSleepSession with capture_owner/restore_owner
callbacks for the *whole* runner checkpoint (learner, ledger, clocks, queues,
etc.). Call restore() on resume, then step(index=..., permission=...). The
session adds only its small state under state_key. The owner must carry that
key through subsequent checkpoints, e.g. state[state_key] = session.state_dict().
Model access and the stopped interval must remain exclusive until step returns.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import torch

from .config import canonical_json
from .learner import LearningBatch
from .runtime import ArchiveUnavailable, BudgetExceeded
from .schemas import TransitionRecord
from .storage import (IntegrityError, _StorageGuard, _fsync_directory,
                      _safe_child, _write_fsynced)


PROTOCOL = "supplemental-rehearsal-v1"
SCHEMA_VERSION = 1


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return canonical_json(value).encode()


@dataclass(frozen=True)
class LikelihoodIdentity:
    """Frozen versions include task/reward rules, model revision, and labels.

    `baseline_version` identifies the original baseline construction, not the
    current predictor. Each record retains its original scalar baseline.
    """

    task_version: str
    reward_version: str
    model_version: str
    label_version: str
    baseline_version: str
    return_edges: tuple[float, ...]
    temperature: float
    action_variance: float

    def __post_init__(self):
        for name in ("task_version", "reward_version", "model_version",
                     "label_version", "baseline_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"missing likelihood identity: {name}")
        object.__setattr__(self, "return_edges", tuple(self.return_edges))
        if (len(self.return_edges) < 3 or not all(math.isfinite(x) for x in self.return_edges)
                or any(a >= b for a, b in zip(self.return_edges, self.return_edges[1:]))):
            raise ValueError("invalid frozen return bins")
        if any(not math.isfinite(x) or x <= 0 for x in (self.temperature, self.action_variance)):
            raise ValueError("invalid frozen likelihood scales")

    @classmethod
    def from_learner(cls, learner, *, task_version: str, model_version: str,
                     label_version: str, baseline_version: str):
        return cls(task_version, learner.config.reward_version, model_version,
                   label_version, baseline_version, tuple(learner.edges.cpu().tolist()),
                   learner.config.temperature, learner.config.action_variance)

    def check_learner(self, learner):
        if (tuple(learner.edges.cpu().tolist()) != self.return_edges
                or learner.config.reward_version != self.reward_version
                or learner.config.temperature != self.temperature
                or learner.config.action_variance != self.action_variance):
            raise IntegrityError("learner has an incompatible frozen likelihood design")


@dataclass(frozen=True)
class SleepPermission:
    interval_id: str
    stopped: bool

    def require(self):
        if self.stopped is not True or not isinstance(self.interval_id, str) or not self.interval_id:
            raise PermissionError("archive access requires an explicit stopped sleep interval")


@dataclass
class ReadBudget:
    max_bytes: int
    max_frames: int
    bytes: int = 0
    frames: int = 0

    def charge(self, size: int, *, frame: bool = False):
        if self.bytes + size > self.max_bytes or self.frames + int(frame) > self.max_frames:
            raise BudgetExceeded("sleep context read budget exhausted")
        self.bytes += size
        self.frames += int(frame)


def _check_record(record, identity, max_context_frames):
    count = len(record.context_refs)
    if not record.ready or record.truncated or not 0 <= record.return_bin < len(identity.return_edges) - 1:
        raise IntegrityError("archive requires a fully labeled compatible transition")
    if (not 0 < count <= max_context_frames
            or not count == len(record.action_history) == len(record.context_arrivals) == len(record.context_elapsed)
            or record.context_arrivals[-1] != record.arrival_index
            or record.context_refs[-1] != record.observation_ref
            or any(a >= b for a, b in zip(record.context_arrivals, record.context_arrivals[1:]))
            or not record.instruction_events):
        raise IntegrityError("archive requires reconstructible original context")
    if any(len(row) != 4 or not all(math.isfinite(x) for x in row) for row in record.action_history):
        raise IntegrityError("invalid archived executed-action history")


class RetiredArchive:
    """Project-local JSON archive plus byte-bounded decoded-frame LRU.

    An explicitly supplied root may be on external storage. Missing roots are
    never created unless create=True; that flag creates only the final leaf.
    Frame loaders return the original content-addressed .npy bytes, never a
    synthetic image. Every access rechecks bytes, including cache hits, so a
    vanished or corrupt source cannot be masked by a warm decoded cache.
    """

    def __init__(self, root: str | Path, identity: LikelihoodIdentity, *, create=False,
                 cache_bytes=8_000_000, max_entry_bytes=262_144,
                 max_frame_bytes=32_000_000, max_context_frames=64,
                 quota_bytes=1_000_000_000, reserve_bytes=0):
        self.root = Path(root).absolute()
        if create:
            if not self.root.parent.is_dir() or self.root.parent.is_symlink():
                raise ArchiveUnavailable(f"archive parent unavailable: {self.root.parent}")
            if any((p / "COMPLETED").exists() for p in (self.root, *self.root.parents)):
                raise RuntimeError("cannot create archive under a completed artifact")
            self.root.mkdir(exist_ok=True)
        for value in (cache_bytes, max_entry_bytes, max_frame_bytes, max_context_frames):
            if type(value) is not int or value < 0:
                raise ValueError("archive bounds must be nonnegative integers")
        if min(max_entry_bytes, max_frame_bytes, max_context_frames) < 1:
            raise ValueError("archive read bounds must be positive")
        self.identity = identity
        self.guard = _StorageGuard(self.root, quota_bytes=quota_bytes, reserve_bytes=reserve_bytes)
        self.cache_limit, self.max_entry_bytes = cache_bytes, max_entry_bytes
        self.max_frame_bytes, self.max_context_frames = max_frame_bytes, max_context_frames
        self._cache = OrderedDict()
        self.cache_bytes = 0
        self.metrics = dict(entry_read_bytes=0, frame_read_bytes=0, context_bytes=0,
                            read_seconds=0., decode_seconds=0., cache_hits=0,
                            frames=0, peak_cache_bytes=0)

    def _path(self, transition_id):
        if not isinstance(transition_id, str) or not transition_id:
            raise IntegrityError("invalid retired transition ID")
        return _safe_child(self.root, _digest(transition_id.encode()) + ".json")

    def put(self, record: TransitionRecord, *, replay, context: Mapping[str, Any] | None = None) -> str:
        """Archive only ledger-retired IDs; publication never overwrites an entry.

        Optional context preserves episode_deadline, decision_time and
        status_events from the original model input, if that schema used them.
        This is storage preparation, not permission to read retired histories.
        """
        self.guard.check()
        if any((p / "COMPLETED").exists() for p in (self.root, *self.root.parents)):
            raise RuntimeError("completed archives are immutable")
        if record.transition_id not in replay.retired_ids:
            raise PermissionError("only ledger-retired IDs may enter the archive")
        _check_record(record, self.identity, self.max_context_frames)
        context = dict(context or {})
        if set(context) - {"episode_deadline", "decision_time", "status_events"}:
            raise IntegrityError("unsupported frozen context schema")
        payload = dict(schema_version=SCHEMA_VERSION, identity=asdict(self.identity),
                       record=record.to_dict(), context=context)
        seal = _digest(_json(payload))
        data = _json(dict(sha256=seal, payload=payload))
        if len(data) > self.max_entry_bytes:
            raise BudgetExceeded("archive entry exceeds bounded JSON size")
        path = self._path(record.transition_id)
        if path.exists():
            if path.read_bytes() != data:
                raise IntegrityError("immutable archive entry conflict")
            return seal
        self.guard.check(len(data))
        temporary = self.root / f".{uuid.uuid4().hex}.tmp"
        try:
            _write_fsynced(temporary, data)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise IntegrityError("immutable archive entry conflict")
            else:
                self.guard.used_bytes += len(data)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        return seal

    def get(self, transition_id: str, *, replay, permission: SleepPermission,
            budget: ReadBudget, expected_sha256: str | None = None):
        permission.require()
        if transition_id not in replay.retired_ids:
            raise PermissionError("requested archive ID is not in the retirement ledger")
        self.guard.check()
        path = self._path(transition_id)
        start = time.monotonic()
        try:
            size = path.stat().st_size
            if size > self.max_entry_bytes:
                raise BudgetExceeded("archive entry exceeds bounded JSON size")
            budget.charge(size)
            with path.open("rb") as stream:
                data = stream.read(self.max_entry_bytes + 1)
        except OSError as exc:
            raise ArchiveUnavailable(f"retired archive entry unavailable: {transition_id}") from exc
        finally:
            self.metrics["read_seconds"] += time.monotonic() - start
        self.metrics["entry_read_bytes"] += len(data)
        self.metrics["context_bytes"] += len(data)
        start = time.monotonic()
        try:
            entry = json.loads(data)
            payload = entry["payload"]
            seal = _digest(_json(payload))
            if (seal != entry["sha256"] or expected_sha256 not in (None, seal)
                    or payload["schema_version"] != SCHEMA_VERSION
                    or _json(payload["identity"]) != _json(asdict(self.identity))):
                raise IntegrityError("archive checksum/schema/likelihood identity mismatch")
            record = TransitionRecord.from_dict(payload["record"])
            if record.transition_id != transition_id:
                raise IntegrityError("archive retirement ID mismatch")
            _check_record(record, self.identity, self.max_context_frames)
            if set(payload["context"]) - {"episode_deadline", "decision_time", "status_events"}:
                raise IntegrityError("unsupported frozen context schema")
            return record, payload["context"], seal
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError(f"invalid archived transition: {transition_id}: {exc}") from exc
        finally:
            self.metrics["decode_seconds"] += time.monotonic() - start

    def _frame(self, reference, *, frames, loader, permission, budget):
        permission.require()
        self.guard.check()
        if not isinstance(reference, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", reference):
            raise IntegrityError("invalid archived frame reference")
        start = time.monotonic()
        try:
            if loader is None:
                if frames is None:
                    raise ArchiveUnavailable("sleep requires store.frames or an explicit frame loader")
                path = _safe_child(frames.root, reference[7:] + ".npy")
                size = path.stat().st_size
                if size > self.max_frame_bytes:
                    raise BudgetExceeded("archive frame exceeds bounded blob size")
                # FrameStore.verify also reads the complete blob for hashing.
                budget.charge(2 * size, frame=True)
                self.metrics["frame_read_bytes"] += size
                frames.verify(reference)
                with path.open("rb") as stream:
                    data = stream.read(self.max_frame_bytes + 1)
            else:
                if budget.frames >= budget.max_frames or budget.bytes >= budget.max_bytes:
                    raise BudgetExceeded("sleep context read budget exhausted")
                data = loader(reference)
            if not isinstance(data, bytes):
                raise IntegrityError("frame loader must return original .npy bytes")
            if len(data) > self.max_frame_bytes:
                raise BudgetExceeded("archive frame exceeds bounded blob size")
            if loader is not None:
                budget.charge(len(data), frame=True)
            self.metrics["frame_read_bytes"] += len(data)
            self.metrics["frames"] += 1
            if reference != "sha256:" + _digest(data):
                raise IntegrityError("archive frame checksum mismatch")
        except OSError as exc:
            raise ArchiveUnavailable(f"archive frame unavailable: {reference}") from exc
        finally:
            self.metrics["read_seconds"] += time.monotonic() - start
        if reference in self._cache:
            self.metrics["cache_hits"] += 1
            self._cache.move_to_end(reference)
            return self._cache[reference]
        start = time.monotonic()
        try:
            stream = io.BytesIO(data)
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                raise IntegrityError("unsupported archived frame encoding version")
            shape, _, dtype = reader(stream)
            if (dtype != np.dtype("uint8") or len(shape) != 3 or shape[-1] != 3
                    or min(shape) < 1 or math.prod(shape) > self.max_frame_bytes):
                raise IntegrityError("archive frame has incompatible RGB schema")
            array = np.load(io.BytesIO(data), allow_pickle=False)
        except (ValueError, EOFError) as exc:
            raise IntegrityError("invalid archive frame encoding") from exc
        finally:
            self.metrics["decode_seconds"] += time.monotonic() - start
        if array.nbytes <= self.cache_limit:
            while self.cache_bytes + array.nbytes > self.cache_limit:
                _, evicted = self._cache.popitem(last=False)
                self.cache_bytes -= evicted.nbytes
            self._cache[reference] = array
            self.cache_bytes += array.nbytes
            self.metrics["peak_cache_bytes"] = max(self.metrics["peak_cache_bytes"], self.cache_bytes)
        return array

    def history(self, record, *, replay, permission, budget, frames=None, loader=None,
                archived=False, context=None, dtype=torch.float32, device="cpu"):
        permission.require()
        _check_record(record, self.identity, self.max_context_frames)
        if record.transition_id not in (replay.retired_ids if archived else replay.eligible_ids):
            raise PermissionError("sleep source is no longer eligible")
        first = 0
        if not archived:
            available = {r.arrival_index for r in (*replay.eligible(), *replay.pending())}
            first = len(record.context_refs) - 1
            while first > 0 and record.context_arrivals[first - 1] in available:
                first -= 1
        refs = record.context_refs[first:]
        images = [torch.from_numpy(self._frame(ref, frames=frames, loader=loader,
                  permission=permission, budget=budget).copy()).permute(2, 0, 1) for ref in refs]
        if any(image.shape != images[0].shape for image in images):
            raise IntegrityError("archived history frame dimensions changed")
        result = dict(images=torch.stack(images).to(device=device, dtype=dtype)[None] / 255,
                      executed_actions=torch.tensor(record.action_history[first:], device=device, dtype=dtype)[None],
                      elapsed=torch.tensor(record.context_elapsed[first:], device=device, dtype=dtype)[None, :, None],
                      contexts=[list(record.instruction_events)],
                      start_time=torch.tensor([record.context_start_s + sum(record.context_elapsed[:first])],
                                              device=device, dtype=dtype))
        context = context or {}
        if "episode_deadline" in context:
            result["episode_deadline"] = torch.tensor([context["episode_deadline"]], device=device, dtype=dtype)
        if "decision_time" in context:
            if len(context["decision_time"]) != len(record.context_refs):
                raise IntegrityError("frozen decision times do not match original history")
            result["decision_time"] = torch.tensor([context["decision_time"][first:]], device=device, dtype=dtype)
        if "status_events" in context:
            result["status_events"] = [context["status_events"]]
        return result


class ArchiveSleepSession:
    """One monotone, restartable stream of fixed-budget sleep optimizer events.

    capture_owner() returns the full current runner state; restore_owner(state)
    restores it after *either* pre-publication or post-publication exceptions.
    No full state is retained here. Only one plan and the latest receipt are
    carried; the journal is the durable history. Retrying older indices is a
    no-op, and retrying the latest returns its original receipt.
    The archived quota is floor(sleep_batch * fraction); neither source's
    shortage is backfilled. A step uses exactly sleep_steps optimizer steps.
    """

    def __init__(self, *, learner, replay, store, identity: LikelihoodIdentity,
                 archive: RetiredArchive | Callable[[], RetiredArchive] | None,
                 seed: int, capture_owner: Callable[[], dict], restore_owner: Callable[[dict], None],
                 session_id="retired-sleep", state_key="archive_sleep", frame_loader=None,
                 eligible_context: Callable[[TransitionRecord], Mapping] | None = None,
                 max_read_bytes=64_000_000, max_frames=256):
        if not session_id or not state_key or type(seed) is not int:
            raise ValueError("sleep requires a session ID, state key, and named integer seed")
        if type(max_read_bytes) is not int or type(max_frames) is not int or min(max_read_bytes, max_frames) < 1:
            raise ValueError("sleep read limits must be positive integers")
        self.learner, self.replay, self.store = learner, replay, store
        self.identity, self.archive = identity, archive
        self.seed, self.session_id, self.state_key = seed, session_id, state_key
        self.capture_owner, self.restore_owner = capture_owner, restore_owner
        self.frame_loader, self.eligible_context = frame_loader, eligible_context
        self.max_read_bytes, self.max_frames = max_read_bytes, max_frames
        self.rng = random.Random(seed)
        self.next_index, self.pending, self.last_receipt, self.pause_reason = 0, None, None, None
        self.io_totals = {}

    def _contract(self):
        cfg = self.learner.config
        return dict(protocol=cfg.retired_replay_protocol, enabled=cfg.retired_archive_replay,
                    fraction=cfg.retired_replay_fraction, batch_size=cfg.sleep_batch,
                    steps=cfg.sleep_steps, learning_config=asdict(cfg), identity=asdict(self.identity), seed=self.seed,
                    session_id=self.session_id, max_read_bytes=self.max_read_bytes, max_frames=self.max_frames)

    def state_dict(self):
        # Small JSON structures are copied, never the learner or runner state.
        return dict(schema_version=SCHEMA_VERSION, contract=json.loads(_json(self._contract())),
                    rng=self.rng.getstate(), next_index=self.next_index,
                    pending=json.loads(_json(self.pending)), last_receipt=json.loads(_json(self.last_receipt)),
                    pause_reason=self.pause_reason, io_totals=dict(self.io_totals))

    def load_state_dict(self, state):
        if (state["schema_version"] != SCHEMA_VERSION
                or _json(state["contract"]) != _json(self._contract())):
            raise IntegrityError("incompatible sleep session checkpoint")
        self.rng.setstate(state["rng"])
        self.next_index = state["next_index"]
        self.pending, self.last_receipt = state["pending"], state["last_receipt"]
        self.pause_reason, self.io_totals = state["pause_reason"], dict(state["io_totals"])

    def restore(self):
        state = self.store.load_checkpoint(trusted=True)
        if state is None:
            return False
        self.restore_owner(state)
        if self.state_key in state:
            self.load_state_dict(state[self.state_key])
        else:
            self.rng.seed(self.seed)
            self.next_index, self.pending, self.last_receipt, self.pause_reason = 0, None, None, None
            self.io_totals = {}
        return True

    def _publish(self, suffix):
        before = self.state_dict()
        state = dict(self.capture_owner())
        state[self.state_key] = before
        progress = {**self.store.progress(), self.state_key: dict(
            protocol=PROTOCOL, next_index=self.next_index, pending=self.pending,
            receipt=self.last_receipt, pause_reason=self.pause_reason, io_totals=self.io_totals)}
        tx = "sleep-" + _digest(self.session_id.encode())[:16] + "-" + suffix
        try:
            return self.store.checkpoint(state, progress=progress, transaction_id=tx)
        except BaseException:
            if not self.restore():
                # Only an initial plan can fail without a prior owner checkpoint;
                # fitting is impossible until that plan has been published.
                self.rng.seed(self.seed)
                self.next_index, self.pending, self.last_receipt = 0, None, None
            raise

    def _open_archive(self):
        archive = self.archive() if callable(self.archive) else self.archive
        if archive is None:
            raise ArchiveUnavailable("retired replay requested without an archive")
        if archive.identity != self.identity:
            raise IntegrityError("incompatible archive likelihood identity")
        return archive

    def step(self, *, index: int, permission: SleepPermission, acquired_count: int):
        cfg = self.learner.config
        if not cfg.retired_archive_replay:
            return {"status": "disabled", "protocol": "off"}
        cfg.validate()
        if cfg.retired_replay_protocol != PROTOCOL:
            raise ValueError("unsupported retired replay protocol")
        permission.require()
        self.identity.check_learner(self.learner)
        if self.store.checkpoint_policy != "journal-v1":
            raise ValueError("archive sleep requires RunStore journal-v1")
        if type(index) is not int or index < 0 or type(acquired_count) is not int or acquired_count < 0:
            raise ValueError("sleep indices/counts must be nonnegative integers")
        if index < self.next_index:
            return (json.loads(_json(self.last_receipt)) if index == self.next_index - 1 else
                    dict(status="already_committed", index=index))
        if index != self.next_index:
            raise ValueError("sleep indices must be contiguous")
        if self.pending is None:
            archived_target = math.floor(cfg.sleep_batch * cfg.retired_replay_fraction)
            eligible_target = cfg.sleep_batch - archived_target
            eligible = self.rng.sample(self.replay.eligible_ids, min(eligible_target, len(self.replay)))
            retired = self.rng.sample(self.replay.retired_ids, min(archived_target, len(self.replay.retired_ids)))
            self.pending = dict(index=index, interval_id=permission.interval_id,
                                eligible_ids=eligible, archived_ids=retired, seals={},
                                read_attempts=0, io={},
                                acquired_count=acquired_count, learner_version=self.learner.version,
                                update_count=self.learner.update_count,
                                eligible_target=eligible_target, archived_target=archived_target)
            self._publish(f"{index}-plan")
        plan = self.pending
        if (plan["interval_id"] != permission.interval_id or plan["acquired_count"] != acquired_count
                or plan["learner_version"] != self.learner.version
                or plan["update_count"] != self.learner.update_count):
            raise IntegrityError("sleep plan no longer matches its stopped owner state")
        archive, io_start = None, {}
        try:
            archive = self._open_archive()
            io_start = dict(archive.metrics)
            budget = ReadBudget(self.max_read_bytes, self.max_frames)
            eligible = {r.transition_id: r for r in self.replay.eligible()}
            records, histories, masks = [], [], []
            for archived, ids in ((False, plan["eligible_ids"]), (True, plan["archived_ids"])):
                for key in ids:
                    if archived:
                        record, context, seal = archive.get(key, replay=self.replay, permission=permission,
                            budget=budget, expected_sha256=plan["seals"].get(key))
                        plan["seals"][key] = seal
                    else:
                        if key not in eligible:
                            raise IntegrityError("planned eligible replay ID disappeared")
                        record = eligible[key]
                        context = {} if self.eligible_context is None else self.eligible_context(record)
                    histories.append(archive.history(record, replay=self.replay, permission=permission,
                        budget=budget, frames=self.store.frames, loader=self.frame_loader,
                        archived=archived, context=context, device=self.learner.device, dtype=self.learner.dtype))
                    records.append(record)
                    masks.append(archived)
            if not records:
                raise ArchiveUnavailable("no eligible or retired records for requested sleep batch")
            if any(r.arrival_index >= acquired_count for r in records):
                raise IntegrityError("sleep records exceed the owner's acquired-observation count")
            # Persist read work and original entry seals before any optimizer
            # mutation, also when a resumed event must reread its contexts.
            delta = self._charge_io(archive, io_start)
            io_start = dict(archive.metrics)
            for key, value in delta.items():
                plan["io"][key] = plan["io"].get(key, 0) + value
            attempt = plan["read_attempts"]
            plan["read_attempts"] += 1
            self._publish(f"{index}-sealed" + (f"-{attempt}" if attempt else ""))
            batch = LearningBatch(histories, torch.tensor([r.baseline for r in records]),
                torch.tensor([r.return_bin for r in records]),
                torch.tensor([r.action_logits for r in records], dtype=self.learner.dtype),
                torch.tensor([r.arrival_index for r in records]), torch.tensor(masks, dtype=torch.bool))
            metrics = self.learner.fit(new=None, replay=batch, acquired_count=acquired_count, steps=cfg.sleep_steps)
        except (ArchiveUnavailable, IntegrityError, BudgetExceeded, OSError, ValueError) as exc:
            self._charge_io(archive, io_start)
            self.pause_reason = f"{type(exc).__name__}: {exc}"
            self._publish(f"{index}-pause-{uuid.uuid4().hex}")
            raise
        self._charge_io(archive, io_start)
        self.last_receipt = dict(status="committed", protocol=PROTOCOL, index=index,
            interval_id=permission.interval_id, eligible_ids=plan["eligible_ids"], archived_ids=plan["archived_ids"],
            eligible_count=len(plan["eligible_ids"]), archived_count=len(plan["archived_ids"]),
            requested_batch_size=cfg.sleep_batch, realized_batch_size=len(records),
            eligible_target=plan["eligible_target"], archived_target=plan["archived_target"],
            acquired_count=acquired_count, new_data_count=0, drift_exposure=0, ewc_admissions=0,
            optimizer_steps=cfg.sleep_steps, learner_version=self.learner.version,
            update_count=self.learner.update_count, metrics=metrics, io=plan["io"])
        self.next_index += 1
        self.pending, self.pause_reason = None, None
        self._publish(f"{index}-commit")
        return json.loads(_json(self.last_receipt))

    def _charge_io(self, archive, start):
        delta = {} if archive is None else {key: value - start.get(key, value)
                                          for key, value in archive.metrics.items()
                                          if key != "peak_cache_bytes"}
        for key, value in delta.items():
            self.io_totals[key] = self.io_totals.get(key, 0) + value
        if archive is not None:
            self.io_totals["peak_cache_bytes"] = max(self.io_totals.get("peak_cache_bytes", 0),
                                                    archive.metrics["peak_cache_bytes"])
        return delta
