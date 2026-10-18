"""Data-only archive safety using tiny local trees, never the real experiment."""

import io
import json
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from mnist_experiment import archive_plan14 as backup


class LocalGuard:
    uuid = "test-drive"

    def check(self, *, force=False):
        pass


@pytest.fixture
def sample(tmp_path):
    if shutil.which("zstd") is None:
        pytest.skip("zstd CLI required for archive round trips")
    repo = tmp_path / "repo"
    root = repo / backup.DATA
    unit = root / "default_float64" / "phase5" / "unit"
    unit.mkdir(parents=True)
    (unit / "metrics.json").write_text('{"mean":0.5}\n')
    (unit / "weights.pt").write_bytes(bytes(range(256)) * 20)
    (unit / "COMPLETED").write_bytes(b"sealed\n")
    (root / ".run.lock").write_bytes(b"old PID\n")
    (root / "pilot" / ".incomplete" / "empty").mkdir(parents=True)
    (repo / "source.py").write_bytes(b"do not move\n")
    other = repo / "cache" / "other_experiment"
    other.mkdir()
    (other / "data.pt").write_bytes(b"do not move either\n")
    destination = tmp_path / "external" / "archive"
    destination.mkdir(parents=True)
    return SimpleNamespace(repo=repo, root=root, destination=destination, guard=LocalGuard())


def create(sample, **kwargs):
    return backup.archive(sample.repo, sample.destination, sample.guard, shard_bytes=7000, **kwargs)


def files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_round_trip_and_data_only_prune(sample):
    before = files(sample.repo)
    metadata = {path.relative_to(sample.root): (path.stat().st_mode & 0o777, path.stat().st_mtime_ns)
                for path in sample.root.rglob("*")}
    manifest = create(sample)
    assert len(manifest["shards"]) > 1
    assert files(sample.repo) == before
    assert backup.load_manifest(sample.destination) == manifest
    backup.verify(sample.destination, manifest, sample.guard)
    backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert not sample.root.exists()
    assert (sample.repo / "source.py").read_bytes() == b"do not move\n"
    assert (sample.repo / "cache/other_experiment/data.pt").exists()
    backup.restore(sample.repo, sample.destination, manifest, sample.guard)
    assert files(sample.repo) == before
    for relative, expected in metadata.items():
        current = (sample.root / relative).stat()
        assert (current.st_mode & 0o777, current.st_mtime_ns) == expected
    assert (sample.root / "pilot/.incomplete/empty").is_dir()


def test_archive_resume_keeps_verified_shards(sample, monkeypatch):
    original = backup.build_shard

    def fail_second(*args, **kwargs):
        if args[3] == 1:
            raise InterruptedError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(backup, "build_shard", fail_second)
    with pytest.raises(InterruptedError):
        create(sample)
    first, receipt = backup.shard_paths(sample.destination, 0)
    accepted = (first.stat().st_mtime_ns, receipt.read_bytes())
    monkeypatch.setattr(backup, "build_shard", original)
    manifest = create(sample, resume=True)
    assert (first.stat().st_mtime_ns, receipt.read_bytes()) == accepted
    backup.verify(sample.destination, manifest, sample.guard)


def test_uncommitted_shard_is_retried(sample, monkeypatch):
    original = backup.read_shard

    def fail(*args, **kwargs):
        raise InterruptedError("before receipt")

    monkeypatch.setattr(backup, "read_shard", fail)
    with pytest.raises(InterruptedError):
        create(sample)
    archive, receipt = backup.shard_paths(sample.destination, 0)
    assert archive.exists() and not receipt.exists()
    archive.write_bytes(b"incomplete junk")
    monkeypatch.setattr(backup, "read_shard", original)
    create(sample, resume=True)
    assert receipt.exists()


def test_corrupt_archive_blocks_all_pruning(sample):
    manifest = create(sample)
    before = files(sample.root)
    archive, _ = backup.shard_paths(sample.destination, len(manifest["shards"]) - 1)
    archive.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt compressed"):
        backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert files(sample.root) == before


def test_mutation_before_prune_blocks_deletion(sample):
    manifest = create(sample)
    changed = sample.root / "default_float64/phase5/unit/metrics.json"
    changed.write_bytes(b"changed")
    before = files(sample.root)
    with pytest.raises(RuntimeError, match="source changed"):
        backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert files(sample.root) == before


def test_content_mutation_with_preserved_metadata_is_not_deleted(sample):
    manifest = create(sample)
    changed = sample.root / "default_float64/phase5/unit/metrics.json"
    previous = changed.stat()
    changed.write_bytes(b"x" * previous.st_size)
    backup.os.utime(changed, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(RuntimeError, match="content changed"):
        backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert changed.exists()


def test_extra_file_blocks_prune(sample):
    manifest = create(sample)
    (sample.root / "new-data").write_bytes(b"new work")
    before = files(sample.root)
    with pytest.raises(RuntimeError, match="gained or lost"):
        backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert files(sample.root) == before


def test_prune_resumes_after_some_files_are_removed(sample):
    manifest = create(sample)
    (sample.root / "default_float64/phase5/unit/metrics.json").unlink()
    backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert not sample.root.exists()
    backup.restore(sample.repo, sample.destination, manifest, sample.guard)
    assert (sample.root / "default_float64/phase5/unit/metrics.json").is_file()


def test_restored_data_can_be_offloaded_again_without_recompression(sample):
    manifest = create(sample)
    backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    backup.restore(sample.repo, sample.destination, manifest, sample.guard)
    backup.prune(sample.repo, sample.destination, manifest, sample.guard)
    assert not sample.root.exists()


def test_resume_refuses_source_changes(sample):
    manifest = create(sample)
    (sample.destination / "COMPLETED").unlink()
    (sample.root / "new-file").touch()
    with pytest.raises(RuntimeError, match="gained or lost"):
        create(sample, resume=True)
    assert manifest["entries"]


def test_restore_refuses_existing_tree(sample):
    manifest = create(sample)
    before = files(sample.root)
    with pytest.raises(RuntimeError, match="overwrite"):
        backup.restore(sample.repo, sample.destination, manifest, sample.guard)
    assert files(sample.root) == before


def test_restore_can_resume_without_publishing_partial_tree(sample, monkeypatch):
    before = files(sample.root)
    manifest = create(sample)
    shutil.rmtree(sample.root)
    original = backup.read_shard

    def fail_second(*args, **kwargs):
        if args[2] == 1:
            raise InterruptedError("restore halted")
        return original(*args, **kwargs)

    monkeypatch.setattr(backup, "read_shard", fail_second)
    with pytest.raises(InterruptedError):
        backup.restore(sample.repo, sample.destination, manifest, sample.guard)
    assert not sample.root.exists()
    monkeypatch.setattr(backup, "read_shard", original)
    backup.restore(sample.repo, sample.destination, manifest, sample.guard, resume=True)
    assert files(sample.root) == before


def test_symlink_source_is_rejected(sample):
    (sample.root / "link").symlink_to(sample.repo / "source.py")
    with pytest.raises(RuntimeError, match="symlinks"):
        create(sample)


def test_restore_symlink_staging_is_rejected(sample):
    manifest = create(sample)
    shutil.rmtree(sample.root)
    staging = sample.repo / "cache" / f".plan14-restore-{manifest['archive_id']}"
    staging.symlink_to(sample.repo, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        backup.restore(sample.repo, sample.destination, manifest, sample.guard, resume=True)


@pytest.mark.parametrize("name", ["/etc/passwd", "../outside", "cache/other/data", "cache/mnist_experiment/akaike_score_gate/../evil"])
def test_paths_are_confined_to_plan14(name):
    with pytest.raises(RuntimeError, match="unsafe"):
        backup.safe_name(name)


def test_metadata_corruption_is_detected(sample):
    create(sample)
    path = sample.destination / "manifest.json"
    value = backup.read_json(path)
    value["payload"]["entries"][0]["mode"] ^= 1
    backup.write_json(path, value)
    with pytest.raises(RuntimeError, match="checksum"):
        backup.load_manifest(sample.destination)


def test_receipt_corruption_is_detected_on_resume(sample):
    create(sample)
    (sample.destination / "COMPLETED").unlink()
    _, path = backup.shard_paths(sample.destination, 0)
    value = backup.read_json(path)
    value["payload"]["digests"][0] = "broken"
    backup.write_json(path, value)
    with pytest.raises(RuntimeError, match="checksum"):
        create(sample, resume=True)


def test_archive_completion_seal_is_checked(sample):
    manifest = create(sample)
    _, path = backup.shard_paths(sample.destination, 0)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="receipt checksum"):
        backup.verify(sample.destination, manifest, sample.guard)


def test_archive_requires_resume_and_empty_destination(sample):
    create(sample)
    with pytest.raises(RuntimeError, match="use --resume"):
        create(sample)
    other = sample.destination.parent / "nonempty"
    other.mkdir()
    (other / "unrelated").touch()
    with pytest.raises(RuntimeError, match="must be empty"):
        backup.archive(sample.repo, other, sample.guard)


def test_initial_manifest_partial_can_resume(sample):
    (sample.destination / "manifest.json.partial").write_bytes(b"interrupted")
    create(sample, resume=True)
    assert (sample.destination / "COMPLETED").exists()


def test_lock_refuses_concurrent_writer(sample):
    path = sample.root / ".run.lock"
    before = path.read_bytes()
    with backup.exclusive_lock(path, create=False):
        with pytest.raises(RuntimeError, match="another process"):
            with backup.source_locks(sample.repo):
                pytest.fail("must not acquire a writer's lock")
    assert path.read_bytes() == before


def test_cli_requires_explicit_destination_and_deletion_consent():
    with pytest.raises(SystemExit):
        backup.main(["archive"])
    with pytest.raises(SystemExit):
        backup.main(["prune", "--destination", "backup"])
    with pytest.raises(SystemExit):
        backup.main(["archive", "--destination", "backup", "--allow-relocated-archive"])


def test_status_is_lightweight(sample, monkeypatch, capsys):
    create(sample)
    monkeypatch.setattr(backup, "read_shard", lambda *a, **kw: pytest.fail("status must not decompress"))
    capsys.readouterr()
    backup.status(sample.destination)
    result = json.loads(capsys.readouterr().out)
    assert result["completed"] and result["committed_shards"] == result["total_shards"]


@pytest.fixture
def fake_mount(tmp_path, monkeypatch):
    mount = tmp_path / "mount"
    mount.mkdir()
    destination = mount / "archive"
    info = {"target": str(mount), "uuid": "drive-A", "fstype": "ntfs3", "options": "rw"}
    monkeypatch.setattr(backup, "mount_info", lambda path: dict(info))
    return destination, info


def test_readonly_mount_blocks_writes_but_allows_verification(fake_mount):
    destination, info = fake_mount
    info["options"] = "ro,nosuid"
    with pytest.raises(RuntimeError, match="read-only"):
        backup.MountGuard(destination, write=True)
    backup.MountGuard(destination, write=False)


def test_mount_identity_change_is_detected(fake_mount):
    destination, info = fake_mount
    guard = backup.MountGuard(destination)
    info["uuid"] = "drive-B"
    with pytest.raises(RuntimeError, match="no longer mounted"):
        guard.check(force=True)


def test_wrong_drive_or_same_source_filesystem_is_rejected(fake_mount, tmp_path):
    destination, _ = fake_mount
    with pytest.raises(RuntimeError, match="wrong destination"):
        backup.MountGuard(destination, expected_uuid="other")
    with pytest.raises(RuntimeError, match="same filesystem"):
        backup.MountGuard(destination, source=tmp_path)


def test_root_mount_fallback_is_rejected(fake_mount):
    destination, info = fake_mount
    info["target"] = "/"
    with pytest.raises(RuntimeError, match="mounted drive"):
        backup.MountGuard(destination)


def test_tar_traversal_is_rejected_even_with_matching_compressed_checksum(sample):
    manifest = create(sample)
    archive, _ = backup.shard_paths(sample.destination, 0)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    result = subprocess.run(["zstd", "-q", "-c"], input=stream.getvalue(), capture_output=True, check=True)
    archive.write_bytes(result.stdout)
    receipt = backup.read_checked_json(backup.shard_paths(sample.destination, 0)[1])
    receipt["compressed_sha256"] = backup.digest_file(archive)
    with pytest.raises(RuntimeError, match="member differs"):
        backup.read_shard(sample.destination, manifest, 0, receipt, sample.guard)


def test_zstd_footer_is_checked_after_tar_eof(sample):
    manifest = create(sample)
    archive, receipt_path = backup.shard_paths(sample.destination, 0)
    raw = subprocess.run(["zstd", "-q", "-d", "-c", str(archive)], capture_output=True, check=True).stdout
    # Make the frame larger than the decoder's output buffer, so tar EOF is
    # reached before the decoder encounters the corrupted compression footer.
    result = subprocess.run(["zstd", "-q", "--check", "-c"],
                            input=raw + b"\0" * (2 * backup.BLOCK), capture_output=True, check=True)
    content = bytearray(result.stdout)
    content[-1] ^= 1
    archive.write_bytes(content)
    receipt = backup.read_checked_json(receipt_path)
    receipt["compressed_sha256"] = backup.digest_file(archive)
    with pytest.raises(RuntimeError, match="zstd failed"):
        backup.read_shard(sample.destination, manifest, 0, receipt, sample.guard)


def test_drive_loss_before_pruning_leaves_primary_intact(sample):
    manifest = create(sample)
    before = files(sample.root)

    class LostDrive(LocalGuard):
        def check(self, *, force=False):
            raise RuntimeError("drive disappeared")

    with pytest.raises(RuntimeError, match="disappeared"):
        backup.prune(sample.repo, sample.destination, manifest, LostDrive())
    assert files(sample.root) == before


def test_no_space_failure_does_not_delete_source(sample, monkeypatch):
    before = files(sample.root)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="capacity"):
        create(sample)
    assert files(sample.root) == before
