from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from numerical_experiment.picar import storage
from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import ArchiveUnavailable, BudgetExceeded
from numerical_experiment.picar.schemas import TransitionRecord, metric_row
from numerical_experiment.picar.storage import (
    FIFOReplay, FrameStore, IntegrityError, RunCollisionError, RunStore,
)

CONFIG = {"experiment": "plan19", "seed": 19, "mode": "smoke"}


def record(index, reference="sha256:" + "0" * 64, *, ready=True):
    return TransitionRecord(
        transition_id=f"t{index}", episode_id="e0", arrival_index=index,
        observation_ref=reference, next_observation_ref=reference, context_refs=(reference,),
        instruction="approach and stop", action_logits=(0.,) * 7, action_probabilities=(.125,) * 8,
        requested_control={"drive_mps": .1}, executed_control={"drive_mps": .1},
        baseline=0., model_version=0, elapsed_s=.5, reward=-.02, discount=.99,
        terminated=False, truncated=False, end_reason=None,
        return_value=-.1 if ready else None, return_bin=2 if ready else None,
    )


def state(value=1):
    generator = torch.Generator().manual_seed(19)
    return {"learner": {"parameters": torch.tensor([value, 2.]), "optimizer": {"step": value}},
            "rng": {"fisher": generator.get_state()},
            "replay": FIFOReplay(2).state_dict(), "simulator": {"ticks": 12},
            "pending": {"labels": ["pending1"], "actions": [5]}, "clock": .5}


def snapshots(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_atomic_checkpoint_full_state_and_json_only_readers(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with RunStore(root, CONFIG) as run:
        reference = run.frames.put(np.zeros((4, 5, 3), dtype=np.uint8))
        first = record(0, reference)
        payload = state()
        commit = run.checkpoint(payload, records=[first], metrics=[metric_row(step=0, new_transitions=1)],
                                progress={"stage": "smoke", "step": 1}, transaction_id="action-0")
        assert run.checkpoint(payload, records=[first], metrics=[metric_row(step=0, new_transitions=1)],
                              progress={"stage": "smoke", "step": 1}, transaction_id="action-0") == commit
        assert len(run.records()) == 1 and len(run.metrics()) == 1
        with pytest.raises(ValueError, match="conflict"):
            run.checkpoint(state(9), transaction_id="action-0")
        with pytest.raises(ValueError, match="trusted"):
            run.load_checkpoint()
    with RunStore(root, CONFIG, resume=True) as run:
        loaded = run.load_checkpoint(trusted=True)
        torch.testing.assert_close(loaded["learner"]["parameters"], payload["learner"]["parameters"])
        assert torch.equal(loaded["rng"]["fisher"], payload["rng"]["fisher"])
        assert loaded["pending"] == payload["pending"] and loaded["simulator"] == payload["simulator"]
        assert run.progress() == {"stage": "smoke", "step": 1}
        monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("report unpickled checkpoint"))
        assert run.records() == (first,)
        assert run.metrics()[0]["new_transitions"] == 1


def test_frame_store_hashes_deduplicates_and_rejects_corruption_pickle_and_paths(tmp_path):
    store = FrameStore(tmp_path / "frames")
    rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    reference = store.put(rgb)
    before = snapshots(store.root)
    assert store.put(rgb.copy()) == reference
    assert before == snapshots(store.root)
    np.testing.assert_array_equal(store.get(reference), rgb)
    with pytest.raises(ValueError):
        store.put(np.array([object()], dtype=object))
    with pytest.raises(IntegrityError):
        store.get("../../private.npy")
    path = next(store.root.glob("*.npy"))
    path.write_bytes(b"corrupt")
    with pytest.raises(IntegrityError, match="corrupt"):
        store.get(reference)
    with pytest.raises(IntegrityError):
        store.put(rgb)


def test_run_hash_configuration_collision_and_single_writer(tmp_path):
    root = tmp_path / "run"
    expected = hashlib.sha256(canonical_json(CONFIG).encode()).hexdigest()
    with RunStore(root, CONFIG, expected) as run:
        with pytest.raises(RunCollisionError):
            RunStore(root, CONFIG)
        with pytest.raises(RunCollisionError, match="writer"):
            RunStore(root, CONFIG, resume=True)
    with pytest.raises(RunCollisionError, match="configuration"):
        RunStore(root, {**CONFIG, "seed": 20}, resume=True)
    with pytest.raises(ValueError, match="hash"):
        RunStore(tmp_path / "invalid", CONFIG, "wrong")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("boundary", ["state.pt", "records.json", "manifest.json", "CURRENT.json"])
def test_interrupted_writes_leave_previous_transaction_committed(tmp_path, monkeypatch, boundary):
    root = tmp_path / "run"
    with RunStore(root, CONFIG) as run:
        ref = run.frames.put(np.zeros((4, 5, 3), dtype=np.uint8))
        run.checkpoint(state(1), records=[record(0, ref)], metrics=[metric_row(step=0)], transaction_id="one")
        previous = (root / "CURRENT.json").read_bytes()
        write = storage._write_fsynced
        def interrupted(path, data):
            # CURRENT uses a hidden temporary basename; fail before replacement.
            if path.name == boundary or boundary == "CURRENT.json" and path.name.startswith(".CURRENT.json."):
                raise OSError("injected power loss before publication")
            return write(path, data)
        monkeypatch.setattr(storage, "_write_fsynced", interrupted)
        with pytest.raises(OSError, match="power loss"):
            run.checkpoint(state(2), records=[record(1, ref)], metrics=[metric_row(step=1)], transaction_id="two")
        assert (root / "CURRENT.json").read_bytes() == previous
        monkeypatch.setattr(storage, "_write_fsynced", write)
    with RunStore(root, CONFIG, resume=True) as run:
        assert [r.transition_id for r in run.records()] == ["t0"]
        assert len(run.metrics()) == 1
        assert run.load_checkpoint(trusted=True)["learner"]["optimizer"]["step"] == 1
        run.checkpoint(state(2), records=[record(1, ref)], metrics=[metric_row(step=1)], transaction_id="two")
        assert [r.transition_id for r in run.records()] == ["t0", "t1"]
        assert len(run.metrics()) == 2


def test_failure_after_pointer_move_reconciles_transaction_identity(tmp_path, monkeypatch):
    with RunStore(tmp_path / "run", CONFIG) as run:
        original = storage._atomic_bytes
        def fail_after(path, data):
            original(path, data)
            if path.name == "CURRENT.json":
                raise OSError("injected failure after pointer replacement")
        monkeypatch.setattr(storage, "_atomic_bytes", fail_after)
        payload = state()
        with pytest.raises(OSError):
            run.checkpoint(payload, metrics=[metric_row(step=1)], transaction_id="one")
        monkeypatch.setattr(storage, "_atomic_bytes", original)
        run.checkpoint(payload, metrics=[metric_row(step=1)], transaction_id="one")
        assert len(run.metrics()) == 1


def test_pending_records_finalize_only_once_and_cannot_complete_early(tmp_path):
    with RunStore(tmp_path / "run", CONFIG) as run:
        ref = run.frames.put(np.zeros((3, 4, 3), dtype=np.uint8))
        pending = record(0, ref, ready=False)
        run.checkpoint(state(), records=[pending], transaction_id="observe")
        with pytest.raises(ValueError, match="pending"):
            run.complete()
        ready = pending.labeled(.2, 3)
        run.checkpoint(state(), records=[ready], transaction_id="label")
        with pytest.raises(ValueError, match="committed"):
            run.checkpoint(state(), records=[ready])
        with pytest.raises(ValueError, match="once"):
            run.checkpoint(state(), records=[replace(ready, return_value=.3)])
        assert run.records() == (ready,)


def test_completed_is_immutable_verified_and_json_csv_readable_without_torch_load(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with RunStore(root, CONFIG) as run:
        ref = run.frames.put(np.zeros((3, 4, 3), dtype=np.uint8))
        run.checkpoint(state(), records=[record(0, ref)], metrics=[metric_row(step=1, phase="smoke")])
        run.complete({"status": "smoke_only"})
        baseline = snapshots(root)
        with pytest.raises(RuntimeError, match="immutable"):
            run.checkpoint(state())
        with pytest.raises(RuntimeError, match="immutable"):
            run.frames.put(np.ones((3, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("completion verification unpickled state"))
    with RunStore(root, CONFIG, resume=True) as run:
        assert run.completed and run.verify_completed()["status"] == "completed"
        assert len(run.metrics()) == 1
        assert (root / "metrics.csv").read_text().startswith("schema_version,phase,condition")
    assert baseline == snapshots(root)
    (root / "metrics.csv").write_text("corrupt")
    with pytest.raises(IntegrityError, match="checksum"):
        RunStore(root, CONFIG, resume=True)


def test_corrupt_current_checkpoint_rejected_before_unpickling(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with RunStore(root, CONFIG) as run:
        run.checkpoint(state())
        path = run._current["directory"] / "state.pt"
    path.write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("corrupt checkpoint was unpickled"))
    with pytest.raises(IntegrityError, match="checksum"):
        RunStore(root, CONFIG, resume=True)


def test_external_archive_identity_quota_and_disappearance_never_spill_local(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    with pytest.raises(ArchiveUnavailable, match="local filesystem"):
        RunStore(tmp_path / "refused", CONFIG, archive_root=archive)
    with RunStore(tmp_path / "run", CONFIG, archive_root=archive, allow_same_filesystem_archive=True) as run:
        ref = run.frames.put(np.zeros((3, 4, 3), dtype=np.uint8))
        run.checkpoint(state(), records=[record(0, ref)])
        archive.rename(tmp_path / "disconnected")
        archive.mkdir()  # An empty mountpoint must not count as the same archive.
        with pytest.raises(ArchiveUnavailable, match="identity"):
            run.frames.put(np.ones((3, 4, 3), dtype=np.uint8))
        with pytest.raises(ArchiveUnavailable):
            run.checkpoint(state(2))
        assert not (archive / "frames").exists()
        assert not (run.root / "frames").exists()
    with pytest.raises(ArchiveUnavailable, match="identity"):
        RunStore(tmp_path / "run", CONFIG, resume=True, allow_same_filesystem_archive=True)
    with RunStore(tmp_path / "tiny", CONFIG, local_quota_bytes=10_000) as run:
        with pytest.raises(BudgetExceeded):
            run.frames.put(np.zeros((100, 100, 3), dtype=np.uint8))


def test_external_archive_missing_and_no_directory_autocreation(tmp_path):
    missing = tmp_path / "absent-archive"
    with pytest.raises(ArchiveUnavailable):
        RunStore(tmp_path / "run", CONFIG, archive_root=missing)
    assert not missing.exists()


def test_replay_initial_admission_pending_fifo_and_exactly_once_retirement():
    replay = FIFOReplay(2)
    assert replay.admit(record(1)) == ()
    replay.add_pending(record(0, ready=False))
    with pytest.raises(ValueError, match="duplicate"):
        replay.admit(record(0))
    with pytest.raises(ValueError, match="pending"):
        replay.admit(record(3, ready=False))
    replay.finalize(record(0))
    assert replay.eligible_ids == ("t0", "t1")
    assert replay.admit(record(2)) == ("t0",)
    assert replay.retire(["t0"], transaction_id="retire-0")
    assert not replay.retire(["t0"], transaction_id="retire-0")
    assert replay.retired_ids == ("t0",) and replay.eligible_ids == ("t1", "t2")
    with pytest.raises(ValueError, match="conflict"):
        replay.retire(["t1"], transaction_id="retire-0")
    with pytest.raises(ValueError, match="prefix"):
        replay.retire(["t0"], transaction_id="new-retire-id")
    with pytest.raises(ValueError, match="duplicate"):
        replay.admit(record(0))
    with pytest.raises(ValueError, match="arrival"):
        replay.admit(replace(record(9), arrival_index=1))
    state_dict = replay.state_dict()
    restored = FIFOReplay(2)
    restored.load_state_dict(json.loads(json.dumps(state_dict)))
    assert restored.state_dict() == replay.state_dict()
    assert restored.sample(2, rng=np.random.default_rng(3)) == replay.sample(2, rng=np.random.default_rng(3))
    assert restored.sample(2, rng=random.Random(3)) == replay.sample(2, rng=random.Random(3))


def test_replay_finalization_identity_and_corrupt_ledgers_rejected():
    replay = FIFOReplay(2)
    replay.add_pending(record(0, ready=False))
    with pytest.raises(ValueError, match="experience"):
        replay.finalize(replace(record(0), reward=1.))
    replay.finalize(record(0))
    replay.retire(["t0"], transaction_id="evict")
    value = replay.state_dict()
    value["transactions"]["another"] = ["t0"]
    with pytest.raises(IntegrityError, match="ledger"):
        FIFOReplay(2).load_state_dict(value)


def test_replay_retirement_and_learner_state_commit_or_rollback_together(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with RunStore(root, CONFIG) as run:
        ref = run.frames.put(np.zeros((3, 4, 3), dtype=np.uint8))
        replay = FIFOReplay(1)
        replay.admit(record(0, ref))
        run.checkpoint({"replay": replay.state_dict(), "learner": {"retired_count": 0}}, records=[record(0, ref)])
        replay.admit(record(1, ref))
        replay.retire(["t0"], transaction_id="retire-t0")
        atomic = storage._atomic_bytes
        def fail(path, data):
            if path.name == "CURRENT.json":
                raise OSError("publication interrupted")
            return atomic(path, data)
        monkeypatch.setattr(storage, "_atomic_bytes", fail)
        with pytest.raises(OSError):
            run.checkpoint({"replay": replay.state_dict(), "learner": {"retired_count": 1}}, records=[record(1, ref)])
    monkeypatch.setattr(storage, "_atomic_bytes", atomic)
    with RunStore(root, CONFIG, resume=True) as run:
        restored = run.load_checkpoint(trusted=True)
        assert restored["learner"]["retired_count"] == 0
        replay = FIFOReplay(1)
        replay.load_state_dict(restored["replay"])
        assert replay.eligible_ids == ("t0",) and replay.retired_ids == ()
