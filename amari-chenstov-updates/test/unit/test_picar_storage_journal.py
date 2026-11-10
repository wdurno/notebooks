"""CPU-only journal transactions, fault boundaries and immutable evidence."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from numerical_experiment.picar import report, storage
from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import BudgetExceeded
from numerical_experiment.picar.schemas import TransitionRecord, metric_row
from numerical_experiment.picar.storage import FIFOReplay, IntegrityError, RunCollisionError, RunStore


CONFIG = {"schema_version": 1, "experiment": "plan19", "seed": 19, "mode": "smoke"}
POLICY = "journal-v1"


def journal(root, **kwargs):
    return RunStore(root, CONFIG, checkpoint_policy=POLICY, **kwargs)


def record(index, reference, *, ready=True):
    return TransitionRecord(
        transition_id=f"t{index}", episode_id="episode", arrival_index=index,
        observation_ref=reference, next_observation_ref=reference, context_refs=(reference,),
        instruction="stop", action_logits=(0.,) * 7, action_probabilities=(.125,) * 8,
        requested_control={"drive_mps": 0.}, executed_control={"drive_mps": 0.},
        baseline=0., model_version=0, elapsed_s=.5, reward=-.02, discount=.99,
        terminated=False, truncated=False, end_reason=None,
        return_value=.1 if ready else None, return_bin=2 if ready else None,
    )


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def used_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def directory(root, commit):
    return root / "transactions" / commit


def read_json(path):
    return json.loads(path.read_bytes())


def seed_run(run):
    reference = run.frames.put(np.zeros((2, 3, 3), dtype=np.uint8))
    first = run.checkpoint({"step": 0}, records=[record(0, reference)],
                           metrics=[metric_row(step=0)], transaction_id="event-0")
    second = run.checkpoint({"step": 1}, records=[record(1, reference)],
                            metrics=[metric_row(step=1)], transaction_id="event-1")
    return reference, first, second


def test_policy_identity_is_explicit_and_legacy_layout_stays_unchanged(tmp_path):
    legacy_root, journal_root = tmp_path / "legacy", tmp_path / "journal"
    with RunStore(legacy_root, CONFIG) as run:
        legacy_hash = run.config_hash
        assert run.config == CONFIG and "checkpoint_policy" not in run.manifest
        for step in range(4):
            run.checkpoint({"step": step}, metrics=[metric_row(step=step)], transaction_id=str(step))
        assert len(list(legacy_root.rglob("state.pt"))) == 4
        current = read_json(run._current["directory"] / "manifest.json")
        assert len(current["transactions"]) == 4
        assert len(read_json(run._current["directory"] / "metrics.json")) == 4
        assert "state.pt" in current["files"] and "journal_schema_version" not in current
        with pytest.raises(ValueError, match="milestone"):
            run.checkpoint({}, milestone=True)
        run.complete({"smoke_only": True})
        baseline = snapshot(legacy_root)
    with journal(journal_root) as run:
        assert run.config_hash != legacy_hash
        assert run.config["checkpoint_policy"] == storage.JOURNAL_CHECKPOINT_POLICY
        assert run.manifest["checkpoint_policy"] == storage.JOURNAL_CHECKPOINT_POLICY
        assert run.config_hash == hashlib.sha256(canonical_json(run.config).encode()).hexdigest()
        resolved, resolved_hash = dict(run.config), run.config_hash
        run.checkpoint({})
    with RunStore(journal_root, resolved, resolved_hash, checkpoint_policy=POLICY, resume=True):
        pass
    with pytest.raises(RunCollisionError, match="configuration"):
        RunStore(journal_root, CONFIG, resume=True)
    with pytest.raises(RunCollisionError, match="configuration"):
        journal(legacy_root, resume=True)
    with pytest.raises(ValueError, match="checkpoint policy"):
        RunStore(tmp_path / "bad-version", {**CONFIG, "checkpoint_policy": {"version": 99}},
                 checkpoint_policy=POLICY)
    with pytest.raises(ValueError, match="checkpoint_policy"):
        RunStore(tmp_path / "unknown", CONFIG, checkpoint_policy="journal")
    with pytest.raises(ValueError, match="hash"):
        RunStore(tmp_path / "bad-hash", CONFIG, legacy_hash, checkpoint_policy=POLICY)
    with RunStore(legacy_root, CONFIG, resume=True) as run:
        assert run.completed and run.verify_completed()["status"] == "completed"
    assert snapshot(legacy_root) == baseline


def test_append_only_events_bounded_snapshots_milestones_and_old_retries(tmp_path, monkeypatch):
    root = tmp_path / "run"
    commits, evidence, events = [], {}, []
    # Fail if the compact writer even serializes an accumulated record/metric list.
    serialize = storage._json_bytes
    def compact_only(value):
        if isinstance(value, list):
            assert len(value) <= 1
        return serialize(value)
    with journal(root) as run:
        reference = run.frames.put(np.zeros((2, 3, 3), dtype=np.uint8))
        monkeypatch.setattr(storage, "_json_bytes", compact_only)
        for step in range(16):
            events.append(dict(state={"step": step}, records=[record(step, reference)],
                metrics=[metric_row(step=step)], progress={"phase": 1, "counts": {"step": step}},
                transaction_id=f"event-{step}", milestone=step in (0, 5)))
            commit = run.checkpoint(**events[-1])
            commits.append(commit)
            for name in ("manifest.json", "records.json", "metrics.json", "progress.json"):
                path = directory(root, commit) / name
                evidence[path] = (path.read_bytes(), path.stat().st_mtime_ns)
            manifest = read_json(directory(root, commit) / "manifest.json")
            assert len(manifest["transactions"]) == 1
            assert manifest["sequence"] == step + 1
            assert len(read_json(directory(root, commit) / "records.json")) == 1
            assert len(read_json(directory(root, commit) / "metrics.json")) == 1
            assert set(manifest["files"]) == {"records.json", "metrics.json", "progress.json"}
            assert run._local_guard.used_bytes == used_bytes(root)
        assert {p.parent.name for p in root.rglob("state.pt")} == {commits[i] for i in (0, 5, 14, 15)}
        assert run.progress()["counts"]["step"] == 15
        monkeypatch.setattr(storage, "_json_bytes", serialize)
        baseline = snapshot(root)
        assert run.checkpoint(**events[2]) == commits[2]  # Its full state is already pruned.
        assert run.checkpoint(**events[0]) == commits[0]
        assert snapshot(root) == baseline
        with pytest.raises(ValueError, match="conflict"):
            run.checkpoint(**{**events[0], "milestone": False})
    with journal(root, resume=True) as run:
        assert [r.transition_id for r in run.records()] == [f"t{i}" for i in range(16)]
        assert [m["step"] for m in run.metrics()] == list(range(16))
        assert run.checkpoint(**events[2]) == commits[2]
        assert run.load_checkpoint(trusted=True) == {"step": 15}
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in evidence.items())


@pytest.mark.parametrize("boundary", ["state.pt", "records.json", "metrics.json", "progress.json",
    "manifest.json", "transaction-fsync", "rename", "parent-fsync", "CURRENT.json"])
def test_prepublication_crashes_preserve_old_state_evidence_and_orphan_accounting(tmp_path, monkeypatch, boundary):
    root = tmp_path / "run"
    with journal(root) as run:
        reference, first, second = seed_run(run)
        before = (root / "CURRENT.json").read_bytes()
        committed_evidence = snapshot(directory(root, first))
        with monkeypatch.context() as faults:
            write, sync, rename = storage._write_fsynced, storage._fsync_directory, storage.os.rename
            def fail_write(path, data):
                if path.name == boundary or boundary == "CURRENT.json" and path.name.startswith(".CURRENT.json."):
                    raise OSError("injected write failure")
                return write(path, data)
            def fail_sync(path):
                if (boundary == "transaction-fsync" and path.name.startswith(".tmp-")
                        or boundary == "parent-fsync" and path == root / "transactions"):
                    raise OSError("injected fsync failure")
                return sync(path)
            def fail_rename(source, destination):
                if boundary == "rename":
                    raise OSError("injected rename failure")
                return rename(source, destination)
            faults.setattr(storage, "_write_fsynced", fail_write)
            faults.setattr(storage, "_fsync_directory", fail_sync)
            faults.setattr(storage.os, "rename", fail_rename)
            with pytest.raises(OSError, match="injected"):
                run.checkpoint({"step": 2}, records=[record(2, reference)], metrics=[metric_row(step=2)],
                               transaction_id="event-2")
        assert (root / "CURRENT.json").read_bytes() == before
        assert snapshot(directory(root, first)) == committed_evidence
        assert run.load_checkpoint(trusted=True) == {"step": 1}
        assert len(run.records()) == len(run.metrics()) == 2
        assert run._local_guard.used_bytes == used_bytes(root)
        orphans = {p: snapshot(p) for p in (root / "transactions").iterdir()
                   if p.name not in (first, second)}
        assert orphans
    with journal(root, resume=True) as run:
        assert run._local_guard.used_bytes == used_bytes(root)
        commit = run.checkpoint({"step": 2}, records=[record(2, reference)], metrics=[metric_row(step=2)],
                                transaction_id="event-2")
        assert run.load_checkpoint(trusted=True) == {"step": 2}
        assert len(run.records()) == len(run.metrics()) == 3
        assert not (directory(root, first) / "state.pt").exists()
        assert (directory(root, second) / "state.pt").is_file()
        assert (directory(root, commit) / "state.pt").is_file()
        assert run._local_guard.used_bytes == used_bytes(root)
        assert all(snapshot(p) == value for p, value in orphans.items())


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("boundary", ["before-current-fsync", "after-current-fsync"])
def test_current_replacement_failure_retries_exactly_once_and_barriers_before_prune(tmp_path, monkeypatch, resume, boundary):
    root = tmp_path / "run"
    run = journal(root)
    try:
        reference, first, _ = seed_run(run)
        before = (root / "CURRENT.json").read_bytes()
        args = dict(state={"step": 2}, records=[record(2, reference)], metrics=[metric_row(step=2)],
                    transaction_id="event-2")
        with monkeypatch.context() as faults:
            sync = storage._fsync_directory
            def fail_sync(path):
                if path == root and (root / "CURRENT.json").read_bytes() != before:
                    if boundary == "after-current-fsync":
                        sync(path)
                    raise OSError("pointer replacement interrupted")
                return sync(path)
            faults.setattr(storage, "_fsync_directory", fail_sync)
            with pytest.raises(OSError, match="interrupted"):
                run.checkpoint(**args)
        committed = read_json(root / "CURRENT.json")["commit_id"]
        assert run._current["pointer"]["commit_id"] == committed
        assert (directory(root, first) / "state.pt").exists()
        assert run._local_guard.used_bytes == used_bytes(root)
        operations = []
        with monkeypatch.context() as checks:
            sync, unlink = storage._fsync_directory, Path.unlink
            def track_sync(path):
                result = sync(path)
                if path == root:
                    operations.append("durable-current")
                return result
            def track_unlink(path, *args, **kwargs):
                if path.name == "state.pt":
                    operations.append("prune")
                return unlink(path, *args, **kwargs)
            checks.setattr(storage, "_fsync_directory", track_sync)
            checks.setattr(Path, "unlink", track_unlink)
            if resume:
                run.close()
                run = journal(root, resume=True)
            assert run.checkpoint(**args) == committed
        assert operations.index("durable-current") < operations.index("prune")
        assert not (directory(root, first) / "state.pt").exists()
        assert len(run.records()) == len(run.metrics()) == 3
        assert run.load_checkpoint(trusted=True) == {"step": 2}
        assert len(list((root / "transactions").iterdir())) == 3
        assert run._local_guard.used_bytes == used_bytes(root)
    finally:
        run.close()


@pytest.mark.parametrize("boundary", ["before-unlink", "after-unlink", "before-prune-fsync", "after-prune-fsync"])
@pytest.mark.parametrize("resume", [False, True])
def test_prune_crashes_keep_current_previous_all_events_and_exactly_once_retries(tmp_path, monkeypatch, boundary, resume):
    root = tmp_path / "run"
    run = journal(root)
    try:
        reference, first, second = seed_run(run)
        oldest = directory(root, first)
        evidence = {name: (oldest / name).read_bytes() for name in
                    ("records.json", "metrics.json", "progress.json", "manifest.json")}
        args = dict(state={"step": 2}, records=[record(2, reference)], metrics=[metric_row(step=2)],
                    transaction_id="event-2")
        with monkeypatch.context() as faults:
            sync, unlink = storage._fsync_directory, Path.unlink
            def fail_unlink(path, *args, **kwargs):
                if path == oldest / "state.pt" and boundary == "before-unlink":
                    raise OSError("injected prune failure")
                result = unlink(path, *args, **kwargs)
                if path == oldest / "state.pt" and boundary == "after-unlink":
                    raise OSError("injected prune failure")
                return result
            def fail_sync(path):
                if path == oldest and boundary == "before-prune-fsync":
                    raise OSError("injected prune failure")
                result = sync(path)
                if path == oldest and boundary == "after-prune-fsync":
                    raise OSError("injected prune failure")
                return result
            faults.setattr(Path, "unlink", fail_unlink)
            faults.setattr(storage, "_fsync_directory", fail_sync)
            with pytest.raises(OSError, match="prune failure"):
                run.checkpoint(**args)
        commit = read_json(root / "CURRENT.json")["commit_id"]
        assert (directory(root, commit) / "state.pt").exists()
        assert (directory(root, second) / "state.pt").exists()
        assert {name: (oldest / name).read_bytes() for name in evidence} == evidence
        assert run._local_guard.used_bytes == used_bytes(root)
        if resume:
            run.close()
            run = journal(root, resume=True)
        assert run.checkpoint(**args) == commit
        assert not (oldest / "state.pt").exists()
        assert len(run.records()) == len(run.metrics()) == 3
        assert run.load_checkpoint(trusted=True) == {"step": 2}
        assert run._local_guard.used_bytes == used_bytes(root)
        assert len(list((root / "transactions").iterdir())) == 3
    finally:
        run.close()


def test_pending_labels_and_retirement_remain_exactly_once_after_snapshot_pruning(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with journal(root) as run:
        reference = run.frames.put(np.zeros((2, 3, 3), dtype=np.uint8))
        replay = FIFOReplay(1)
        pending = record(0, reference, ready=False)
        replay.add_pending(pending)
        observe = dict(state={"replay": replay.state_dict(), "retired_count": 0}, records=[pending],
                       metrics=[metric_row(step=0)], transaction_id="observe")
        run.checkpoint(**observe)
        with pytest.raises(ValueError, match="pending"):
            run.complete()
        ready = pending.labeled(.1, 2)
        replay.finalize(ready)
        label = dict(state={"replay": replay.state_dict(), "retired_count": 0}, records=[ready],
                     metrics=[metric_row(step=1)], transaction_id="label")
        run.checkpoint(**label)
        replay.admit(record(1, reference))
        assert replay.retire(["t0"], transaction_id="retire-t0")
        retire = dict(state={"replay": replay.state_dict(), "retired_count": 1},
                      records=[record(1, reference)], metrics=[metric_row(step=2, retired_transitions=1)],
                      transaction_id="retire-t0")
        with monkeypatch.context() as faults:
            atomic = storage._atomic_bytes
            def fail_pointer(path, data):
                if path.name == "CURRENT.json":
                    raise OSError("retirement publication failed")
                return atomic(path, data)
            faults.setattr(storage, "_atomic_bytes", fail_pointer)
            with pytest.raises(OSError):
                run.checkpoint(**retire)
        assert run.load_checkpoint(trusted=True)["retired_count"] == 0
        retire_commit = run.checkpoint(**retire)
        for step in (3, 4):
            run.checkpoint(retire["state"], transaction_id=f"tick-{step}")
        assert not (directory(root, retire_commit) / "state.pt").exists()
    with journal(root, resume=True) as run:
        baseline = snapshot(root)
        assert run.checkpoint(**retire) == retire_commit
        run.checkpoint(**observe)
        run.checkpoint(**label)
        assert snapshot(root) == baseline
        assert len(run.records()) == 2 and len(run.metrics()) == 3
        restored = run.load_checkpoint(trusted=True)
        assert restored["retired_count"] == 1
        replay = FIFOReplay(1)
        replay.load_state_dict(restored["replay"])
        assert replay.eligible_ids == ("t1",) and replay.retired_ids == ("t0",)
        assert not replay.retire(["t0"], transaction_id="retire-t0")
        with pytest.raises(ValueError, match="conflict"):
            run.checkpoint(**{**retire, "state": {"retired_count": 2}})
        with pytest.raises(ValueError, match="once"):
            run.checkpoint({}, records=[replace(ready, return_value=.2)])
        run.complete({"smoke_only": True})


def test_quota_includes_renamed_orphans_and_never_prunes_to_admit_a_commit(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with journal(root) as run:
        reference, first, second = seed_run(run)
        old_pointer = (root / "CURRENT.json").read_bytes()
        with monkeypatch.context() as faults:
            sync = storage._fsync_directory
            def fail_after_rename(path):
                if path == root / "transactions":
                    raise OSError("renamed orphan")
                return sync(path)
            faults.setattr(storage, "_fsync_directory", fail_after_rename)
            with pytest.raises(OSError, match="orphan"):
                run.checkpoint({"blob": torch.zeros(4096)}, transaction_id="orphan")
        total = used_bytes(root)
        assert run._local_guard.used_bytes == total
        run._local_guard.quota_bytes = total + 1000
        before = snapshot(root)
        with pytest.raises(BudgetExceeded):
            run.checkpoint({"blob": torch.zeros(4096)}, transaction_id="orphan")
        assert snapshot(root) == before
        assert (root / "CURRENT.json").read_bytes() == old_pointer
        assert (directory(root, first) / "state.pt").exists()
        assert (directory(root, second) / "state.pt").exists()
    with pytest.raises(BudgetExceeded):
        journal(root, resume=True, local_quota_bytes=total - 1)
    with journal(root, resume=True) as run:
        assert run._local_guard.used_bytes == total


@pytest.mark.parametrize("boundary", ["state.pt", "manifest.json", "CURRENT.json"])
def test_partial_writes_and_repeated_orphans_are_counted_before_retry(tmp_path, monkeypatch, boundary):
    root = tmp_path / "run"
    with journal(root) as run:
        reference, first, second = seed_run(run)
        before = (root / "CURRENT.json").read_bytes()
        args = dict(state={"step": 2}, records=[record(2, reference)], metrics=[metric_row(step=2)],
                    transaction_id="event-2")
        with monkeypatch.context() as faults:
            write = storage._write_fsynced
            def partial_write(path, data):
                if path.name == boundary or boundary == "CURRENT.json" and path.name.startswith(".CURRENT.json."):
                    write(path, data[:len(data) // 2])
                    raise OSError("partial write")
                return write(path, data)
            faults.setattr(storage, "_write_fsynced", partial_write)
            for _ in range(2):
                with pytest.raises(OSError, match="partial write"):
                    run.checkpoint(**args)
                assert (root / "CURRENT.json").read_bytes() == before
                assert run._local_guard.used_bytes == used_bytes(root)
        orphaned = {p: snapshot(p) for p in (root / "transactions").iterdir()
                    if p.name not in (first, second)}
        assert len(orphaned) == 2
    with journal(root, resume=True) as run:
        run.checkpoint(**args)
        assert len(run.metrics()) == len(run.records()) == 3
        assert run._local_guard.used_bytes == used_bytes(root)
        assert all(snapshot(p) == value for p, value in orphaned.items())


def test_failed_prune_barrier_cannot_delete_old_recovery_state(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with journal(root) as run:
        _, first, _ = seed_run(run)
        sync = storage._fsync_directory
        barriers = 0
        with monkeypatch.context() as faults:
            def fail_prune_barrier(path):
                nonlocal barriers
                if path == root:
                    barriers += 1
                    if barriers == 2:
                        raise OSError("prune barrier failed")
                return sync(path)
            faults.setattr(storage, "_fsync_directory", fail_prune_barrier)
            with pytest.raises(OSError, match="prune barrier"):
                run.checkpoint({"step": 2}, transaction_id="event-2")
        assert (directory(root, first) / "state.pt").exists()
        committed = read_json(root / "CURRENT.json")["commit_id"]
        assert run.checkpoint({"step": 2}, transaction_id="event-2") == committed
        assert not (directory(root, first) / "state.pt").exists()
        assert run.load_checkpoint(trusted=True) == {"step": 2}


@pytest.mark.parametrize("target", ["current-state", "previous-state", "milestone-state", "old-records",
                                      "old-metrics", "old-progress", "old-manifest"])
def test_recovery_verifies_retained_snapshots_and_entire_evidence_chain_before_unpickling(tmp_path, monkeypatch, target):
    root = tmp_path / "run"
    with journal(root) as run:
        commits = [run.checkpoint({"step": step}, metrics=[metric_row(step=step)], milestone=step == 0)
                   for step in range(5)]
    index, name = {"current-state": (4, "state.pt"), "previous-state": (3, "state.pt"),
        "milestone-state": (0, "state.pt"), "old-records": (1, "records.json"),
        "old-metrics": (1, "metrics.json"), "old-progress": (1, "progress.json"),
        "old-manifest": (1, "manifest.json")}[target]
    (directory(root, commits[index]) / name).write_bytes(b"corrupt")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("unpickled before verification"))
    with pytest.raises(IntegrityError, match="checksum"):
        journal(root, resume=True)


def test_schema1_live_report_reads_current_not_stale_mirror_without_torch_load(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with journal(root) as run:
        for step in range(4):
            run.checkpoint({"step": step}, progress={"schema_version": 1, "phase": 1,
                           "status": "working", "counts": {"step": step}}, transaction_id=str(step))
        (root / "progress.json").write_text("not valid JSON")
        pointer = read_json(root / "CURRENT.json")
        assert pointer["storage_schema_version"] == 1
        baseline = snapshot(root)
        monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("report unpickled state"))
        result = report.load_report(root)
        assert result["live"][0]["counts"] == {"step": 3}
        assert not result["complete_units"]
        assert baseline == snapshot(root)


@pytest.mark.parametrize("boundary", ["metrics.csv", "summary.json", "manifest.json", "COMPLETED", "after-COMPLETED"])
def test_completion_crashes_resume_with_all_seals_and_completed_files_are_immutable(tmp_path, monkeypatch, boundary):
    root = tmp_path / "run"
    with journal(root) as run:
        for step in range(4):
            run.checkpoint({"step": step}, metrics=[metric_row(step=step, smoke_only=True)],
                           transaction_id=str(step), milestone=step == 0)
        with monkeypatch.context() as faults:
            atomic = storage._atomic_bytes
            def fail_completion(path, data):
                if path.name == boundary:
                    raise OSError("completion interrupted")
                result = atomic(path, data)
                if boundary == "after-COMPLETED" and path.name == "COMPLETED":
                    raise OSError("completion interrupted after marker")
                return result
            faults.setattr(storage, "_atomic_bytes", fail_completion)
            with pytest.raises(OSError, match="completion interrupted"):
                run.complete({"schema_version": 1, "smoke_only": True})
        if (root / "COMPLETED").exists():
            sealed = snapshot(root)
            with pytest.raises(RuntimeError, match="immutable"):
                run.checkpoint({})
    with journal(root, resume=True) as run:
        if not run.completed:
            run.complete({"schema_version": 1, "smoke_only": True})
            sealed = snapshot(root)
        assert run.verify_completed()["status"] == "completed"
        assert len(run.metrics()) == 4
        with pytest.raises(RuntimeError, match="immutable"):
            run.complete()
        with pytest.raises(RuntimeError, match="immutable"):
            run.frames.put(np.ones((2, 3, 3), dtype=np.uint8))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("verification unpickled state"))
    with journal(root, resume=True) as run:
        run.verify_completed()
        with pytest.raises(RuntimeError, match="immutable"):
            run._prune_journal()
        with pytest.raises(RunCollisionError, match="completed artifact"):
            journal(root / "child")
    completed = report.read_completed_unit(root)
    assert len(completed["metrics"]) == 4 and completed["smoke_only"]
    assert sealed == snapshot(root)
