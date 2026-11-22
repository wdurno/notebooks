"""Resume frozen Fix 3 against its original theory snapshot, not newer notes.

This operational wrapper does not change any numerical source or run identity.
It redirects only the campaign's two theory-provenance reads to an exact,
hash-verified historical notebook and records the current documentation version.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from numerical_experiment.picar.storage import _atomic_bytes
from numerical_experiment.picar_reward_repair import campaign


VERSION = "plan19-fix3-theory-pinned-resume-v1"
THEORY = Path("mathematical_overview.ipynb")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(project, *args):
    return subprocess.check_output(["git", *args], cwd=project, stderr=subprocess.PIPE)


def recover_theory(project, expected):
    data = (project / THEORY).read_bytes()
    if digest(data) == expected:
        return data, "working_notebook_matches_frozen_hash"
    commits = git(project, "log", "-n", "128", "--format=%H", "--", str(THEORY)).decode().splitlines()
    for commit in commits:
        data = git(project, "show", commit + ":./" + str(THEORY))
        if digest(data) == expected:
            return data, commit
    raise ValueError("cannot recover the exact frozen theory notebook; no resume is allowed")


@dataclass(frozen=True)
class TheoryPinnedProject(os.PathLike):
    project: Path
    theory: Path

    def __fspath__(self):
        return os.fspath(self.project)

    def __truediv__(self, part):
        return self.theory if Path(part) == THEORY else self.project / part


@contextmanager
def pinned_project(snapshot):
    original = campaign.PROJECT
    campaign.PROJECT = TheoryPinnedProject(Path(original), snapshot)
    try:
        yield
    finally:
        campaign.PROJECT = original


def publish(path, payload):
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    _atomic_bytes(path, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n")


def prepare(root):
    root = Path(root).resolve()
    saved = json.loads((root / "orchestration/config.json").read_text())
    project = Path(campaign.PROJECT)
    if campaign.sources() != saved["source_sha256"]:
        raise ValueError("frozen scientific sources changed; this wrapper cannot bypass that guard")
    current_hash = digest((project / THEORY).read_bytes())
    metadata = root / "resume_provenance"
    metadata.mkdir(exist_ok=True)
    snapshot = metadata / ("theory-" + saved["theory_sha256"] + ".ipynb")
    if snapshot.is_symlink():
        raise ValueError("theory snapshot must not be a symlink")
    if snapshot.exists():
        if digest(snapshot.read_bytes()) != saved["theory_sha256"]:
            raise ValueError("frozen theory snapshot is corrupt")
        origin = "previously_verified_snapshot"
    else:
        data, origin = recover_theory(project, saved["theory_sha256"])
        _atomic_bytes(snapshot, data)
    model_path = saved["model_source"]
    if model_path.startswith("/") and not Path(model_path).is_dir():
        raise ValueError("recorded model-data override is unavailable")
    model_path = model_path if Path(model_path).is_dir() else None
    with pinned_project(snapshot):
        resolved = campaign.run_identity(root, saved["source_campaign"], saved["reference_campaign"],
                                         model_path, saved["max_hours"])
    expected = {k: v for k, v in saved.items() if k != "checkpoint_policy"}
    if resolved != expected:
        raise ValueError("resume identity differs beyond the pinned theory document")
    receipt = {"schema_version": 1, "version": VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(), "campaign": str(root),
        "theory_snapshot": str(snapshot), "theory_origin": origin,
        "frozen_theory_sha256": saved["theory_sha256"], "current_theory_sha256": current_hash,
        "current_git_commit": git(project, "rev-parse", "HEAD").decode().strip(),
        "current_git_status": git(project, "status", "--porcelain").decode(),
        "wrapper_sha256": digest(Path(__file__).read_bytes()),
        "resolved_identity_matches": True, "scientific_sources_unchanged": True,
        "numerical_settings_changed": False}
    path = metadata / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex + ".json")
    publish(path, receipt)
    return saved, snapshot, path


def run(root, *, clear_stop=False, offline=False):
    root = Path(root).resolve()
    started = time.monotonic()
    saved, snapshot, receipt = prepare(root)
    if clear_stop:
        (root / "STOP").unlink(missing_ok=True)
    model_path = saved["model_source"]
    model_path = model_path if Path(model_path).is_dir() else None
    status, failure = "completed", None
    try:
        with pinned_project(snapshot):
            return campaign.run(root, saved["source_campaign"], saved["reference_campaign"],
                model_path=model_path, offline=offline, resume=True, max_hours=saved["max_hours"])
    except BaseException as exc:
        status, failure = "interrupted_or_failed", type(exc).__name__
        raise
    finally:
        publish(receipt.with_suffix(".outcome.json"), {"schema_version": 1, "version": VERSION,
            "receipt": str(receipt), "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": status, "exception_type": failure,
            "invocation_wall_seconds": time.monotonic() - started,
            "snapshot_sha256": digest(snapshot.read_bytes())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, clear_stop=args.clear_stop, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
