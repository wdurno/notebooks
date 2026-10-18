"""Fast Plan 15 artifact/runtime tests; no data download or model training."""

import errno
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory import artifacts, control
from mnist_experiment.fisher_memory.artifacts import ArtifactError, ImmutableError, Store, iter_units
from mnist_experiment.fisher_memory.control import ResourceLimit, RuntimeControl, StopRequested


IDENTITY = {"source_hashes": {"learner.py": "frozen"}, "config_hash": "configuration-1"}


def store_at(root, **kwargs):
    options = {"free_reserve_bytes": 0, "finalization_reserve_bytes": 4096}
    options.update(kwargs)
    return Store(root, IDENTITY, **options)


def payload(step):
    return {"parameters": torch.arange(step + 1),
            "metric_rows": [{"step": i, "nll": 1 / (i + 1)} for i in range(step + 1)],
            "step": step, "rng": torch.get_rng_state()}


def test_checkpoint_and_metrics_are_one_resumable_transaction(tmp_path):
    root = tmp_path / "study"
    with store_at(root) as store:
        assert store.begin("replica-1", {"seed": 42}) == root / "units" / "replica-1"
        store.save_checkpoint("replica-1", payload(1))
        store.save_checkpoint("replica-1", payload(2))
        assert store.load_checkpoint("replica-1", previous=True)["step"] == 1
    with store_at(root, resume=True) as store:
        store.begin("replica-1", {"seed": 42})
        saved = store.load_checkpoint("replica-1")
        assert saved["step"] == 2
        assert len(saved["metric_rows"]) == 3
        assert torch.equal(saved["parameters"], torch.arange(3))
        store.save_checkpoint("replica-1", payload(3))
        assert len(list(store.unit_path("replica-1").glob("checkpoint-*.pt"))) == 2
        store.write_csv("replica-1", "metrics.csv", saved["metric_rows"])
        store.write_json("replica-1", "summary.json", {"n": 3})
        store.complete("replica-1", {"accepted": True})
        assert store.is_complete("replica-1")
        assert store.begin("replica-1", {"seed": 42}) is None
        assert list(store.iter_units(completed_only=True))[0]["completion"]["metadata"]["accepted"]


def test_root_identity_and_unit_identity_collisions(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {"seed": 1})
        with pytest.raises(ImmutableError):
            store.begin("one", {"seed": 2})
        with pytest.raises(ImmutableError):
            store.begin("one", {"seed": 1})
    with pytest.raises(ImmutableError):
        store_at(tmp_path)
    for identity in ({**IDENTITY, "config_hash": "new"},
                     {**IDENTITY, "source_hashes": {"learner.py": "changed"}}):
        with pytest.raises(ImmutableError, match="identity changed"):
            Store(tmp_path, identity, resume=True)
    with store_at(tmp_path, resume=True):
        pass


@pytest.mark.parametrize("operation", ["checkpoint", "json", "csv", "complete"])
def test_completed_unit_is_immutable(tmp_path, operation):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        store.complete("one")
        before = {p.name: p.read_bytes() for p in store.unit_path("one").iterdir()}
        with pytest.raises(ImmutableError):
            if operation == "checkpoint":
                store.save_checkpoint("one", payload(1))
            elif operation == "json":
                store.write_json("one", "summary.json", {})
            elif operation == "csv":
                store.write_csv("one", "metrics.csv", [{"x": 1}])
            else:
                store.complete("one")
        assert before == {p.name: p.read_bytes() for p in store.unit_path("one").iterdir()}


def test_corruption_invalidates_cached_and_persisted_validation(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.write_json("one", "metrics.json", {"gain": 1})
        store.complete("one")
        assert store.is_complete("one")
        file = store.unit_path("one") / "metrics.json"
        original_stat = file.stat()
        file.write_bytes(file.read_bytes().replace(b"1", b"9"))
        os.utime(file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with pytest.raises(ArtifactError, match="checksum"):
            store.is_complete("one")
    with pytest.raises(ArtifactError, match="checksum"):
        list(iter_units(tmp_path))


@pytest.mark.parametrize("corruption", ["missing", "extra", "marker"])
def test_completed_file_set_and_marker_are_sealed(tmp_path, corruption):
    with store_at(tmp_path) as store:
        unit = store.begin("one", {})
        store.write_json("one", "metrics.json", {})
        store.complete("one")
        if corruption == "missing":
            (unit / "metrics.json").unlink()
        elif corruption == "extra":
            (unit / "extra.json").write_text("{}")
        else:
            marker = json.loads((unit / "COMPLETED").read_text())
            marker["metadata"] = {"unrecorded": "change"}
            (unit / "COMPLETED").write_text(json.dumps(marker))
        with pytest.raises(ArtifactError):
            store.is_complete("one")


def test_checkpoint_corruption_is_not_silently_replaced_with_previous(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        latest = store.save_checkpoint("one", payload(1))
        latest.write_bytes(b"damaged")
        with pytest.raises(ArtifactError):
            store.load_checkpoint("one")
        assert store.load_checkpoint("one", previous=True)["step"] == 0
        with pytest.raises(ArtifactError):
            store.complete("one")


def test_interrupted_pointer_publication_keeps_previous_transaction(tmp_path, monkeypatch):
    with store_at(tmp_path) as store:
        unit = store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        pointer_before = (unit / "latest.json").read_bytes()
        replace = artifacts.os.replace

        def interrupted(source, destination):
            if Path(destination).name == "latest.json":
                raise RuntimeError("interrupted before pointer commit")
            return replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(artifacts.os, "replace", interrupted)
            with pytest.raises(RuntimeError, match="interrupted"):
                store.save_checkpoint("one", payload(1))
        assert (unit / "latest.json").read_bytes() == pointer_before
        assert store.load_checkpoint("one")["metric_rows"] == payload(0)["metric_rows"]
        assert len(list(unit.glob("checkpoint-*.pt"))) == 1
        assert not list(unit.glob(".tmp-*"))


def test_disk_full_during_checkpoint_preserves_latest(tmp_path, monkeypatch):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        original = artifacts._BoundedWriter.write

        def full(writer, data):
            if writer.position > 100:
                writer.error = OSError(errno.ENOSPC, "mock disk full")
                raise writer.error
            return original(writer, data)

        with monkeypatch.context() as patch:
            patch.setattr(artifacts._BoundedWriter, "write", full)
            with pytest.raises(ResourceLimit, match="checkpoint write"):
                store.save_checkpoint("one", payload(1))
        assert store.load_checkpoint("one")["step"] == 0
        assert store.storage_status()["reserved_bytes"] == 0


def test_quota_includes_existing_temporary_and_finalization_bytes(tmp_path):
    with store_at(tmp_path, max_bytes=100_000, external_bytes=2000) as store:
        store.write_json(None, "status.json", {"text": "x" * 10_000})
        old = (tmp_path / "status.json").read_bytes()
        used = store.storage_status()["used_bytes"]
        store.max_bytes = used + 2000 + 4096 + 5000
        with pytest.raises(ResourceLimit, match="byte allocation"):
            store.write_json(None, "status.json", {"text": "y" * 10_000})
        assert (tmp_path / "status.json").read_bytes() == old
        store.max_bytes = 100_000
        store.write_json(None, "large_log.json", {"line": "z" * 20_000})
        assert store.storage_status()["used_bytes"] >= 30_000
        store.reconcile()
        assert store.storage_status()["used_bytes"] == sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())


def test_tiny_quota_and_free_disk_admission(tmp_path, monkeypatch):
    with pytest.raises(ResourceLimit):
        store_at(tmp_path / "tiny", max_bytes=50)
    with store_at(tmp_path / "normal") as store:
        with monkeypatch.context() as patch:
            patch.setattr(artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
            with pytest.raises(ResourceLimit, match="filesystem"):
                store.write_json(None, "data.json", {"x": 1})
        assert not (store.root / "data.json").exists()


def test_external_report_reservation_and_stale_cleanup(tmp_path):
    with store_at(tmp_path) as store:
        with store.reserve(20_000):
            assert store.storage_status()["reserved_bytes"] == 20_000
            (tmp_path / "report.txt").write_text("x" * 10_000)
        assert store.storage_status()["used_bytes"] >= 10_000
        assert store.storage_status()["reserved_bytes"] == 0
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
    (tmp_path / ".tmp-stale").write_bytes(b"unfinished report")
    (tmp_path / "units/one/.tmp-stale").write_bytes(b"unfinished checkpoint")
    (tmp_path / "units/one/checkpoint-orphan.pt").write_bytes(b"unpublished checkpoint")
    ledger = json.loads((tmp_path / "byte_ledger.json").read_text())
    ledger["reserved_bytes"] = 999_999_999
    ledger["reservations"] = {"dead-process": 999_999_999}
    (tmp_path / "byte_ledger.json").write_text(json.dumps(ledger))
    with store_at(tmp_path, resume=True) as store:
        assert not list(tmp_path.rglob(".tmp-*"))
        assert not list(tmp_path.rglob("checkpoint-orphan.pt"))
        assert store.load_checkpoint("one")["step"] == 0
        assert store.storage_status()["reserved_bytes"] == 0


def test_exclusive_lock_allows_read_only_reporting(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.complete("one")
        with pytest.raises(ArtifactError, match="another writer"):
            store_at(tmp_path, resume=True)
        assert list(iter_units(tmp_path))[0]["status"] == "completed"
        assert control.status(tmp_path)["writer_active"]
    assert not control.status(tmp_path)["writer_active"]


def test_stop_is_persistent_and_final_checkpoint_is_allowed(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        control.request_stop(tmp_path)
        control.request_stop(tmp_path)
        with pytest.raises(StopRequested):
            store.check()
        store.save_checkpoint("one", payload(2), finalizing=True)
    with pytest.raises(StopRequested):
        store_at(tmp_path, resume=True)
    assert (tmp_path / "STOP").exists()
    (tmp_path / "STOP").unlink()  # Explicit operator action, never an implicit resume action.
    with store_at(tmp_path, resume=True) as store:
        assert store.load_checkpoint("one")["step"] == 2


def test_signal_and_runtime_guards(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(control.time, "monotonic", lambda: now[0])
    runtime = RuntimeControl(tmp_path, active_seconds=10, max_active_seconds=15, max_wall_seconds=20)
    now[0] = 104
    runtime.check()
    assert runtime.active_seconds == 14
    now[0] = 105
    with pytest.raises(ResourceLimit, match="cumulative"):
        runtime.check()
    wall = RuntimeControl(tmp_path, max_wall_seconds=2)
    now[0] = 107
    with pytest.raises(ResourceLimit, match="wall-clock"):
        wall.check()
    runtime = RuntimeControl(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    with runtime.signal_handlers():
        os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(StopRequested):
            runtime.check()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_clear_stop_requires_resume_matching_identity_and_exclusive_lock(tmp_path):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        control.request_stop(tmp_path)
        with pytest.raises(ArtifactError, match="another writer"):
            store_at(tmp_path, resume=True, clear_stop=True)
        assert (tmp_path / "STOP").exists()
    with pytest.raises(ValueError, match="resume=True"):
        store_at(tmp_path, clear_stop=True)
    assert (tmp_path / "STOP").exists()
    with pytest.raises(ImmutableError, match="identity changed"):
        Store(tmp_path, {"source": "different"}, resume=True, clear_stop=True)
    assert (tmp_path / "STOP").exists()
    with store_at(tmp_path, resume=True, clear_stop=True) as store:
        assert not (tmp_path / "STOP").exists()
        assert store.load_checkpoint("one")["step"] == 0
        store.check()


def test_clear_stop_does_not_claim_an_uninitialized_root(tmp_path):
    (tmp_path / "STOP").touch()
    with pytest.raises(ArtifactError, match="existing matching"):
        store_at(tmp_path, resume=True, clear_stop=True)
    assert (tmp_path / "STOP").exists()


def test_runtime_counter_survives_resume_without_idle_time(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(control.time, "monotonic", lambda: now[0])
    with store_at(tmp_path) as store:
        store.begin("one", {})
        now[0] = 120
        store.save_checkpoint("one", payload(0))
    now[0] = 1000
    with store_at(tmp_path, resume=True, max_active_seconds=30) as store:
        assert store.control.active_seconds == 20
        now[0] = 1009
        store.check()
        now[0] = 1010
        with pytest.raises(ResourceLimit, match="cumulative"):
            store.check()


def test_resume_uses_validation_index_without_rehashing_prior_payloads(tmp_path, monkeypatch):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        store.save_checkpoint("one", payload(0))
        store.complete("one")
    with store_at(tmp_path, resume=True) as store:
        def unexpected_hash(*args, **kwargs):
            raise AssertionError("unchanged completed data should not be reread")
        monkeypatch.setattr(artifacts, "_file_hash", unexpected_hash)
        assert store.is_complete("one")
        assert store.begin("one", {}) is None
        with pytest.raises(AssertionError, match="reread"):
            store.is_complete("one", full=True)


def test_no_filesystem_rescan_per_checkpoint(tmp_path, monkeypatch):
    with store_at(tmp_path) as store:
        store.begin("one", {})
        monkeypatch.setattr(store, "reconcile", lambda **kwargs: pytest.fail("unexpected full rescan"))
        for step in range(3):
            store.save_checkpoint("one", payload(step))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", "..", ".hidden"])
def test_invalid_paths_are_rejected(tmp_path, name):
    with store_at(tmp_path) as store:
        with pytest.raises(ValueError):
            store.begin(name, {})
        with pytest.raises(ValueError):
            store.write_json(None, name, {})


def test_system_files_and_symlinks_are_rejected(tmp_path):
    with store_at(tmp_path) as store:
        with pytest.raises(ValueError):
            store.write_json(None, "identity.json", {})
        (tmp_path / "units" / "linked").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ArtifactError):
            store.begin("linked", {})
        (tmp_path / "units" / "linked").unlink()


def test_control_cli(tmp_path, capsys):
    with store_at(tmp_path):
        assert control.main(["status", "--root", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out)["writer_active"]
        assert control.main(["stop", "--root", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out)["stop_requested"]


@pytest.mark.parametrize("before_commit", [False, True])
def test_force_kill_preserves_checkpoint_transactions(tmp_path, before_commit):
    script = '''
import os, signal, sys
from pathlib import Path
from mnist_experiment.fisher_memory.artifacts import Store
import mnist_experiment.fisher_memory.artifacts as artifacts
s = Store(Path(sys.argv[1]), {"test": "kill"}, free_reserve_bytes=0)
s.begin("one", {})
s.save_checkpoint("one", {"step": 0, "rows": [0]})
replace = artifacts.os.replace
def killed(source, destination):
    if Path(destination).name == "latest.json" and sys.argv[2] == "before":
        os.kill(os.getpid(), signal.SIGKILL)
    replace(source, destination)
    if Path(destination).name == "latest.json":
        os.kill(os.getpid(), signal.SIGKILL)
artifacts.os.replace = killed
s.save_checkpoint("one", {"step": 1, "rows": [0, 1]})
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path),
                             "before" if before_commit else "after"], timeout=30,
                            capture_output=True, text=True)
    assert result.returncode == -signal.SIGKILL, result.stderr
    with Store(tmp_path, {"test": "kill"}, free_reserve_bytes=0, resume=True) as store:
        expected = {"step": 0, "rows": [0]} if before_commit else {"step": 1, "rows": [0, 1]}
        assert store.load_checkpoint("one") == expected
        assert store.load_checkpoint("one", previous=True) == (None if before_commit else {"step": 0, "rows": [0]})
        assert store.storage_status()["reserved_bytes"] == 0


def test_hashing_and_serialization_check_stop_between_chunks(tmp_path):
    file = tmp_path / "large.bin"
    file.write_bytes(b"x" * (3 * artifacts.CHUNK_BYTES))
    checks = [0]

    def stop_after_two_chunks():
        checks[0] += 1
        if checks[0] >= 3:
            raise StopRequested("interrupt bounded operation")

    with pytest.raises(StopRequested):
        artifacts._file_hash(file, stop_after_two_chunks)
    assert checks[0] == 3
    checks[0] = 0
    with (tmp_path / "partial.bin").open("wb") as handle:
        writer = artifacts._BoundedWriter(handle, file.stat().st_size, stop_after_two_chunks)
        with pytest.raises(StopRequested):
            writer.write(file.read_bytes())
    assert (tmp_path / "partial.bin").stat().st_size < file.stat().st_size


def test_stop_close_does_not_start_a_new_directory_audit(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    store.begin("one", {})
    store._last_reconcile -= 100
    monkeypatch.setattr(store, "reconcile", lambda **kwargs: pytest.fail("uninterruptible close audit"))
    control.request_stop(tmp_path)
    store.save_checkpoint("one", payload(0), finalizing=True)
    store.close()
