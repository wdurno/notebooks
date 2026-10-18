"""Resumable, verified offloading of Plan 14 data; no scientific-code imports.

Only cache/mnist_experiment/akaike_score_gate is archived or removed. Source,
notebooks, shared datasets, and other experiments stay in the checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


DATA = Path("cache/mnist_experiment/akaike_score_gate")
REPO = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
BLOCK = 1024 ** 2
SCHEMA = 1


def log(message):
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(f"{now} {message}", flush=True)


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_checked_json(path, value):
    write_json(path, {"payload": value, "sha256": value_digest(value)})


def read_checked_json(path):
    wrapped = read_json(path)
    if value_digest(wrapped["payload"]) != wrapped["sha256"]:
        raise RuntimeError(f"metadata checksum mismatch: {path.name}")
    return wrapped["payload"]


def safe_name(name):
    if not isinstance(name, str):
        raise RuntimeError("archive paths must be strings")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or not path.is_relative_to(DATA.as_posix())):
        raise RuntimeError(f"unsafe or out-of-scope archive path: {name!r}")
    return Path(name)


def mount_info(path):
    while not path.exists():
        path = path.parent
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,UUID,FSTYPE,OPTIONS"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["filesystems"][0]


class MountGuard:
    """Pin the mounted filesystem, not a directory that might become local."""

    def __init__(self, destination, *, write=False, expected_uuid=None, source=None):
        self.destination = destination
        self.write = write
        info = mount_info(destination)
        self.mount = Path(info["target"])
        self.uuid = info.get("uuid")
        if self.mount == Path("/") or not self.uuid or destination == self.mount:
            raise RuntimeError("destination must be an archive subdirectory on a mounted drive with a UUID")
        if not destination.is_relative_to(self.mount):
            raise RuntimeError("destination is outside the detected mount")
        self.device = self.mount.stat().st_dev
        if source is not None and source.exists() and source.stat().st_dev == self.device:
            raise RuntimeError("offload destination is on the same filesystem as the source")
        if expected_uuid is not None and self.uuid != expected_uuid:
            raise RuntimeError("wrong destination drive UUID")
        self.last_check = 0.0
        self.check(force=True)

    def check(self, *, force=False):
        if self.mount.stat().st_dev != self.device:
            raise RuntimeError("backup drive disappeared or was replaced")
        if force or time.monotonic() - self.last_check > 5:
            info = mount_info(self.destination)
            if info.get("uuid") != self.uuid or Path(info["target"]) != self.mount:
                raise RuntimeError("backup drive is no longer mounted at the expected location")
            if self.write and "ro" in info["options"].split(","):
                raise RuntimeError("backup drive is read-only; fix its mount before archiving")
            self.last_check = time.monotonic()


@contextlib.contextmanager
def exclusive_lock(path, *, create=True):
    with path.open("a+b" if create else "rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process holds {path.name}") from error
        yield


@contextlib.contextmanager
def source_locks(repo):
    with contextlib.ExitStack() as stack:
        for path in sorted((repo / DATA).rglob(".run.lock")):
            stack.enter_context(exclusive_lock(path, create=False))
        yield


def file_record(repo, path):
    value = path.lstat()
    if stat.S_ISDIR(value.st_mode):
        kind = "dir"
    elif stat.S_ISREG(value.st_mode):
        kind = "file"
    else:
        raise RuntimeError(f"symlinks and special files are not supported: {path}")
    return {"path": path.relative_to(repo).as_posix(), "kind": kind,
            "size": value.st_size if kind == "file" else 0,
            "mode": stat.S_IMODE(value.st_mode), "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino, "device": value.st_dev}


def inventory(repo):
    root = repo / DATA
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Plan 14 data root is missing, not a directory, or is a symlink")
    result = [file_record(repo, root)]
    for directory, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        for name in directories + files:
            result.append(file_record(repo, Path(directory) / name))
    return sorted(result, key=lambda entry: entry["path"])


def check_source(repo, entry, *, identity=True):
    path = repo / safe_name(entry["path"])
    current = file_record(repo, path)
    keys = ("kind", "size", "mode", "mtime_ns")
    if identity:
        keys += ("inode", "device")
    if any(current[key] != entry[key] for key in keys):
        raise RuntimeError(f"source changed since inventory: {entry['path']}")


def check_tree(repo, manifest, *, missing_ok=False, identity=True):
    root = repo / DATA
    if root.is_symlink():
        raise RuntimeError("Plan 14 data root is a symlink")
    if missing_ok and not root.exists():
        return
    current = {entry["path"]: entry for entry in inventory(repo)}
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    if current.keys() - expected.keys() or (not missing_ok and expected.keys() - current.keys()):
        raise RuntimeError("source tree gained or lost entries; refusing to continue")
    for name, entry in current.items():
        if entry["kind"] == "file":
            check_source(repo, expected[name], identity=identity)


def make_manifest(repo, guard, shard_bytes, threads):
    entries = inventory(repo)
    shards, start, size = [], 0, 0
    for index, entry in enumerate(entries):
        cost = entry["size"] + 2048
        if size and size + cost > shard_bytes:
            shards.append([start, index])
            start, size = index, 0
        size += cost
    shards.append([start, len(entries)])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True)
    return {"schema": SCHEMA, "archive_id": uuid.uuid4().hex,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "data_root": DATA.as_posix(), "mount_uuid": guard.uuid,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "compression": {"codec": "zstd", "level": 1, "threads": threads},
            "entries": entries, "shards": shards}


def load_manifest(destination):
    manifest = read_checked_json(destination / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("data_root") != DATA.as_posix():
        raise RuntimeError("incompatible archive manifest")
    names = set()
    for entry in manifest["entries"]:
        safe_name(entry["path"])
        if entry["path"] in names or entry["kind"] not in ("dir", "file") or entry["size"] < 0:
            raise RuntimeError("invalid or duplicated inventory entry")
        names.add(entry["path"])
    end = 0
    for start, stop in manifest["shards"]:
        if start != end or not start < stop <= len(manifest["entries"]):
            raise RuntimeError("invalid shard inventory ranges")
        end = stop
    if end != len(manifest["entries"]) or not end:
        raise RuntimeError("incomplete shard inventory")
    return manifest


def shard_paths(destination, index):
    base = destination / "shards" / f"{index:05d}"
    return base.with_suffix(".tar.zst"), base.with_suffix(".json")


def shard_entries(manifest, index):
    start, stop = manifest["shards"][index]
    return manifest["entries"][start:stop]


def read_receipt(destination, manifest, index):
    archive, path = shard_paths(destination, index)
    receipt = read_checked_json(path)
    if (receipt.get("archive_id") != manifest["archive_id"] or receipt.get("index") != index
            or len(receipt.get("digests", [])) != len(shard_entries(manifest, index))):
        raise RuntimeError("invalid shard receipt")
    if digest_file(archive) != receipt["compressed_sha256"]:
        raise RuntimeError(f"corrupt compressed shard: {archive.name}")
    return receipt


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.handle.read(size)
        self.digest.update(block)
        return block


@contextlib.contextmanager
def compressor(command, **kwargs):
    process = subprocess.Popen(command, **kwargs)
    try:
        yield process
        if process.wait() != 0:
            raise RuntimeError("zstd failed; no shard was committed")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


def build_shard(repo, destination, manifest, index, guard):
    archive, _ = shard_paths(destination, index)
    temporary = archive.with_name(archive.name + ".partial")
    entries = shard_entries(manifest, index)
    needed = sum(entry["size"] + 4096 for entry in entries) + 256 * BLOCK
    if shutil.disk_usage(destination).free < needed:
        raise RuntimeError("insufficient backup space for the next shard and reserve")
    digests = []
    with temporary.open("wb") as output:
        command = ["zstd", "-q", "-1", f"-T{manifest['compression']['threads']}", "--check", "-c"]
        with compressor(command, stdin=subprocess.PIPE, stdout=output) as process:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for entry in entries:
                    guard.check()
                    check_source(repo, entry)
                    info = tarfile.TarInfo(entry["path"])
                    info.mode = entry["mode"]
                    info.mtime = entry["mtime_ns"] / 1e9
                    info.type = tarfile.DIRTYPE if entry["kind"] == "dir" else tarfile.REGTYPE
                    info.size = entry["size"]
                    if entry["kind"] == "file":
                        with (repo / entry["path"]).open("rb") as source:
                            reader = HashingReader(source)
                            tar.addfile(info, reader)
                        check_source(repo, entry)
                        digests.append(reader.digest.hexdigest())
                    else:
                        tar.addfile(info)
                        digests.append(None)
            process.stdin.close()
        output.flush()
        os.fsync(output.fileno())
    guard.check(force=True)
    os.replace(temporary, archive)
    sync_directory(archive.parent)
    return {"archive_id": manifest["archive_id"], "index": index, "digests": digests,
            "compressed_sha256": digest_file(archive)}


def safe_parents(root, relative):
    current = root
    if current.is_symlink():
        raise RuntimeError("restore staging directory is a symlink")
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise RuntimeError("symlink in restore destination")
        current.mkdir(exist_ok=True)


def read_shard(destination, manifest, index, receipt, guard, *, staging=None):
    archive, _ = shard_paths(destination, index)
    entries = shard_entries(manifest, index)
    command = ["zstd", "-q", "-d", "-c", str(archive)]
    with compressor(command, stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for entry, expected_digest in zip(entries, receipt["digests"], strict=True):
                guard.check()
                member = tar.next()
                if (member is None or member.name.rstrip("/") != entry["path"]
                        or member.size != entry["size"] or member.mode != entry["mode"]
                        or (entry["kind"] == "dir" and not member.isdir())
                        or (entry["kind"] == "file" and not member.isreg())):
                    raise RuntimeError("archive member differs from inventory")
                relative = safe_name(member.name.rstrip("/"))
                target = staging / relative if staging is not None else None
                if target is not None:
                    safe_parents(staging, relative)
                if entry["kind"] == "dir":
                    if expected_digest is not None:
                        raise RuntimeError("invalid directory digest")
                    if target is not None:
                        if target.is_symlink():
                            raise RuntimeError("symlink in restore staging")
                        target.mkdir(exist_ok=True)
                    continue
                temporary = None
                with contextlib.ExitStack() as stack:
                    output = None
                    if target is not None:
                        descriptor, name = tempfile.mkstemp(dir=staging, prefix=".restore-", suffix=".partial")
                        temporary = Path(name)
                        output = stack.enter_context(os.fdopen(descriptor, "wb"))
                    source = stack.enter_context(tar.extractfile(member))
                    digest = hashlib.sha256()
                    for block in iter(lambda: source.read(BLOCK), b""):
                        guard.check()
                        digest.update(block)
                        if output is not None:
                            output.write(block)
                    if digest.hexdigest() != expected_digest:
                        raise RuntimeError(f"uncompressed content checksum mismatch: {entry['path']}")
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                if target is not None:
                    if target.is_symlink():
                        raise RuntimeError("symlink in restore staging")
                    if target.exists():
                        if digest_file(target) != expected_digest:
                            raise RuntimeError("restore collision; existing content differs")
                        temporary.unlink()
                    else:
                        os.link(temporary, target)
                        temporary.unlink()
                    os.chmod(target, entry["mode"])
                    os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
                    with target.open("rb") as handle:
                        os.fsync(handle.fileno())
                    sync_directory(target.parent)
            if tar.next() is not None:
                raise RuntimeError("unexpected extra member in archive")
        # Consume the frame footer too: tar EOF alone does not validate zstd's checksum.
        for block in iter(lambda: process.stdout.read(BLOCK), b""):
            guard.check()
    guard.check(force=True)


def check_seal(destination, manifest):
    seal = read_json(destination / "COMPLETED")
    if (seal["manifest_sha256"] != digest_file(destination / "manifest.json")
            or len(seal["receipts"]) != len(manifest["shards"])):
        raise RuntimeError("archive completion seal mismatch")
    for index, expected in enumerate(seal["receipts"]):
        _, receipt = shard_paths(destination, index)
        if digest_file(receipt) != expected:
            raise RuntimeError("archive receipt checksum mismatch")


def verify(destination, manifest, guard):
    check_seal(destination, manifest)
    for index in range(len(manifest["shards"])):
        guard.check(force=True)
        receipt = read_receipt(destination, manifest, index)
        read_shard(destination, manifest, index, receipt, guard)
        log(f"Verified shard {index + 1}/{len(manifest['shards'])}")


def archive(repo, destination, guard, *, resume=False, shard_bytes=4 * GIB, threads=2):
    if (destination / "manifest.json").exists():
        if not resume:
            raise RuntimeError("archive already exists; use --resume")
        manifest = load_manifest(destination)
        if manifest["mount_uuid"] != guard.uuid:
            raise RuntimeError("wrong destination drive")
    else:
        initial_partial = destination / "manifest.json.partial"
        if resume and initial_partial.is_file():
            initial_partial.unlink()
        if any(path.name != ".archive.lock" for path in destination.iterdir()):
            raise RuntimeError("new archive directory must be empty")
        manifest = make_manifest(repo, guard, shard_bytes, threads)
        total = sum(entry["size"] for entry in manifest["entries"])
        if shutil.disk_usage(destination).free < total * 1.03 + GIB:
            raise RuntimeError("backup needs conservative uncompressed-size capacity plus reserve")
        write_checked_json(destination / "manifest.json", manifest)
    if (destination / "COMPLETED").exists():
        check_seal(destination, manifest)
        log("Archive is already sealed; use verify, prune, or restore as needed")
        return manifest
    check_tree(repo, manifest)
    (destination / "shards").mkdir(exist_ok=True)
    for index in range(len(manifest["shards"])):
        guard.check(force=True)
        archive_path, receipt_path = shard_paths(destination, index)
        if receipt_path.exists():
            read_receipt(destination, manifest, index)
            log(f"Reused verified shard {index + 1}/{len(manifest['shards'])}")
            continue
        # An archive without a committed receipt is retryable, not accepted evidence.
        if archive_path.exists():
            archive_path.unlink()
        log(f"Compressing shard {index + 1}/{len(manifest['shards'])}")
        receipt = build_shard(repo, destination, manifest, index, guard)
        read_shard(destination, manifest, index, receipt, guard)
        write_checked_json(receipt_path, receipt)
        log(f"Committed verified shard {index + 1}/{len(manifest['shards'])}")
    check_tree(repo, manifest)
    guard.check(force=True)
    write_json(destination / "COMPLETED", {
        "manifest_sha256": digest_file(destination / "manifest.json"),
        "receipts": [digest_file(shard_paths(destination, index)[1])
                     for index in range(len(manifest["shards"]))],
    })
    log("Archive sealed. Primary data have not been removed.")
    return manifest


def prune(repo, destination, manifest, guard):
    # A fresh, full readback precedes the first deletion, including on resume.
    verify(destination, manifest, guard)
    check_tree(repo, manifest, missing_ok=True, identity=False)
    for index in range(len(manifest["shards"])):
        receipt = read_checked_json(shard_paths(destination, index)[1])
        for entry, digest in zip(shard_entries(manifest, index), receipt["digests"], strict=True):
            if entry["kind"] != "file":
                continue
            guard.check()
            path = repo / safe_name(entry["path"])
            if not path.exists():
                continue
            check_source(repo, entry, identity=False)
            before = file_record(repo, path)
            if digest_file(path) != digest:
                raise RuntimeError(f"source content changed; not deleting {entry['path']}")
            guard.check()
            check_source(repo, before)
            path.unlink()
        log(f"Primary removal progress {index + 1}/{len(manifest['shards'])} shards")
    directories = [entry for entry in manifest["entries"] if entry["kind"] == "dir"]
    for entry in sorted(directories, key=lambda entry: len(Path(entry["path"]).parts), reverse=True):
        guard.check()
        path = repo / safe_name(entry["path"])
        if path.exists():
            path.rmdir()
    sync_directory((repo / DATA).parent)
    guard.check(force=True)
    write_json(destination / "PRIMARY_REMOVED", {"archive_id": manifest["archive_id"]})
    log("Plan 14 data removed from primary. Code, notebooks, datasets, and other experiments remain.")


def restore(repo, destination, manifest, guard, *, resume=False):
    check_seal(destination, manifest)
    final = repo / DATA
    staging = repo / "cache" / f".plan14-restore-{manifest['archive_id']}"
    if final.exists() or final.is_symlink():
        raise RuntimeError("restore refuses to overwrite an existing Plan 14 data tree")
    if staging.exists() and not resume:
        raise RuntimeError("restore staging exists; use --resume")
    if staging.is_symlink():
        raise RuntimeError("restore staging directory is a symlink")
    staging.mkdir(parents=True, exist_ok=True)
    for partial in staging.glob(".restore-*.partial"):
        partial.unlink()
    total = sum(entry["size"] for entry in manifest["entries"])
    retained = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    if shutil.disk_usage(staging).free + retained < total + 2 * GIB:
        raise RuntimeError("insufficient primary free space for uncompressed restore plus reserve")
    for index in range(len(manifest["shards"])):
        guard.check(force=True)
        receipt = read_receipt(destination, manifest, index)
        read_shard(destination, manifest, index, receipt, guard, staging=staging)
        log(f"Restored and verified shard {index + 1}/{len(manifest['shards'])}")
    for entry in sorted(manifest["entries"], key=lambda entry: len(Path(entry["path"]).parts), reverse=True):
        if entry["kind"] == "dir":
            path = staging / entry["path"]
            os.chmod(path, entry["mode"])
            os.utime(path, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            sync_directory(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() or final.is_symlink():
        raise RuntimeError("restore target appeared during extraction; not publishing")
    guard.check(force=True)
    os.rename(staging / DATA, final)
    sync_directory(final.parent)
    shutil.rmtree(staging)
    log("Original relative paths restored; immutable scientific artifacts were not regenerated.")


def status(destination, *, repo=REPO):
    manifest = load_manifest(destination)
    count = sum(shard_paths(destination, index)[1].exists() for index in range(len(manifest["shards"])))
    print(json.dumps({"archive_id": manifest["archive_id"],
                      "source_GiB": sum(entry["size"] for entry in manifest["entries"]) / GIB,
                      "files": sum(entry["kind"] == "file" for entry in manifest["entries"]),
                      "committed_shards": count, "total_shards": len(manifest["shards"]),
                      "completed": (destination / "COMPLETED").exists(),
                      "primary_removal_recorded": (destination / "PRIMARY_REMOVED").exists(),
                      "primary_data_present": (repo / DATA).exists()}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "verify", "prune", "restore", "status"))
    parser.add_argument("--destination", type=Path, required=True,
                        help="dedicated archive directory on the mounted external drive (no default)")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune-source", action="store_true",
                        help="archive only: also perform a fresh verification and remove primary data")
    parser.add_argument("--yes-delete-source", action="store_true", help="required for the prune action")
    parser.add_argument("--allow-relocated-archive", action="store_true",
                        help="verify/restore/status only: read a verified archive copied to a different drive")
    parser.add_argument("--shard-gib", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.shard_gib <= 0 or args.threads < 1:
        parser.error("shard size and thread count must be positive")
    if args.prune_source and args.action != "archive":
        parser.error("--prune-source is only valid with archive")
    if args.action == "prune" and not args.yes_delete_source:
        parser.error("prune requires --yes-delete-source")
    if args.allow_relocated_archive and args.action not in ("verify", "restore", "status"):
        parser.error("--allow-relocated-archive cannot authorize deletion or archive writes")
    if shutil.which("zstd") is None and args.action != "status":
        parser.error("zstd CLI is required")
    repo, destination = args.repo_root.resolve(), args.destination.expanduser().resolve()
    if destination == repo or destination.is_relative_to(repo):
        parser.error("destination must be outside the checkout")
    writable = args.action in ("archive", "prune")
    guard = MountGuard(destination, write=writable, source=repo / DATA if writable else None)
    if args.action == "archive":
        guard.check(force=True)
        destination.mkdir(parents=True, exist_ok=True)
    manifest = None if args.action == "archive" else load_manifest(destination)
    if manifest is not None and manifest["mount_uuid"] != guard.uuid and not args.allow_relocated_archive:
        raise RuntimeError("archive belongs to a different destination drive UUID")
    if args.action == "status":
        status(destination, repo=repo)
        return 0
    def interrupted(signum, frame):
        raise InterruptedError("stop requested; rerun the same command with --resume")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    log(f"Starting {args.action}; PID {os.getpid()}; no GPU use")
    with exclusive_lock(destination / ".archive.lock", create=writable):
        with source_locks(repo) if writable else contextlib.nullcontext():
            if args.action == "archive":
                manifest = archive(repo, destination, guard, resume=args.resume,
                                   shard_bytes=int(args.shard_gib * GIB), threads=args.threads)
                if args.prune_source:
                    prune(repo, destination, manifest, guard)
            elif args.action == "verify":
                verify(destination, manifest, guard)
            elif args.action == "prune":
                prune(repo, destination, manifest, guard)
            elif args.action == "restore":
                restore(repo, destination, manifest, guard, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        log(f"STOPPED: {error}")
        raise SystemExit(1)
