"""Report-only resume driver for a frozen Plan 14 Phase 5 study.

The original executor still schedules, validates, and runs every scientific
unit. Only duplicate forced notebook rebuilds are coalesced. Completed
provenance directories describe finished invocations, not finished cohorts.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import uuid4

from mnist_experiment.akaike_score_gate import orchestrate
from mnist_experiment.akaike_score_gate.artifacts import file_hash, source_hashes
from mnist_experiment.akaike_score_gate.config import REPO_ROOT, Study, canonical_hash
from mnist_experiment.rotated_mnist.artifacts import (
    RotatedRunSession, _read_json, _utc_now, _write_json, runtime_metadata,
)


DRIVER_VERSION = 1
_FROZEN_RUNNER = orchestrate.Runner
_VOLATILE_PROGRESS = frozenset({
    "active_invocation_seconds", "updated_at", "started_at", "completed_at",
})


def _stable_progress(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_progress(item) for key, item in value.items()
                if key not in _VOLATILE_PROGRESS}
    if isinstance(value, list):
        return [_stable_progress(item) for item in value]
    return value


def report_identity(progress: dict, units_created: int) -> str:
    return canonical_hash({"units_created": units_created,
                           "progress": _stable_progress(progress)})


class CoalescedReportRunner(_FROZEN_RUNNER):
    """Retain the frozen refresh timer, progress writes, and report validator."""

    terminal_phase = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_report_identity = None
        self.report_renders = 0
        self.coalesced_refreshes = 0

    def refresh(self, *, force: bool = False):
        identity = report_identity(self.progress, self.store.limit.units)
        status = self.progress.get("status")
        terminal = status in {"blocked", "interrupted"} or (
            status == "completed"
            and self.progress.get("phase") == f"phase{self.terminal_phase}"
        )
        coalesce = (force and not terminal and self.notebook.is_file()
                    and identity == self._last_report_identity)
        before = self.last_refresh
        # The parent always writes progress and retains its 900-second heartbeat.
        super().refresh(force=force and not coalesce)
        if coalesce:
            self.coalesced_refreshes += 1
        if (force and not coalesce) or self.last_refresh != before:
            self._last_report_identity = identity
            self.report_renders += 1


def _inventory(root: Path) -> dict:
    paths = sorted(str(path.parent.relative_to(root))
                   for path in root.glob("phase*/**/COMPLETED")
                   if ".incomplete" not in path.parts)
    return {"completed_units": len(paths), "completed_unit_ids_sha256": canonical_hash(paths)}


def _begin_provenance(runner: CoalescedReportRunner, payload: dict) -> RotatedRunSession:
    run_id = f"{time.time_ns()}-{os.getpid()}-{uuid4().hex[:12]}"
    parent = runner.store.root / "execution" / "report_resume"
    working, final = parent / ".incomplete" / run_id, parent / run_id
    working.mkdir(parents=True, exist_ok=False)
    config = {**payload, "schema_version": 1, "driver_version": DRIVER_VERSION,
              "study_hash": runner.study.config_hash,
              "scientific_source_hashes": runner.store.sources}
    _write_json(working / "config.json", config)
    _write_json(working / "manifest.json", {
        "manifest_schema_version": 1, "artifact_schema_version": 1,
        "metric_schema_version": 1, "run_id": run_id,
        "experiment": "plan14-report-only-resume-invocation",
        "config_hash": canonical_hash(config), "status": "incomplete",
        "started_at": _utc_now(), "completed_at": None, "pid": os.getpid(),
        "runtime": runtime_metadata(REPO_ROOT, SimpleNamespace(runtime=SimpleNamespace(
            device=runner.study.device, dtype=runner.study.dtype))),
    })
    session = RotatedRunSession(run_id, working, final)
    source_path = Path(__file__).resolve()
    session.write_json("driver_source.json", {
        "path": str(source_path), "sha256": file_hash(source_path),
        "text": source_path.read_text(encoding="utf-8"),
    })
    session.write_json("initial_inventory.json", _inventory(runner.store.root))
    return session


def _finish_provenance(runner: CoalescedReportRunner, session: RotatedRunSession,
                       failure: BaseException | None) -> Path:
    unchanged = source_hashes() == runner.store.sources
    session.write_json("summary.json", {
        "schema_version": 1, "driver_version": DRIVER_VERSION,
        "invocation_status": "failed" if failure is not None else "finished",
        "study_status": runner.progress.get("status"),
        "study_complete": (runner.progress.get("phase") == "phase6"
                           and runner.progress.get("status") == "completed"),
        "progress": runner.progress, "final_inventory": _inventory(runner.store.root),
        "elapsed_invocation_seconds": time.monotonic() - runner.store.limit.started,
        "report_renders": runner.report_renders,
        "coalesced_forced_refreshes": runner.coalesced_refreshes,
        "scientific_sources_unchanged": unchanged,
        "error_type": type(failure).__name__ if failure is not None else None,
        "error": str(failure) if failure is not None else None,
    })
    required = ("config.json", "driver_source.json", "initial_inventory.json", "summary.json")
    session.write_json("integrity.json", {name: file_hash(session.path / name) for name in required})
    result = session.complete((*required, "integrity.json"))
    if not unchanged:
        raise RuntimeError("Frozen scientific sources changed during the resume invocation")
    return result


@contextlib.contextmanager
def coalesced_reporting(*, terminal_phase: int = 6,
                        provenance: dict | None = None) -> Iterator[list[CoalescedReportRunner]]:
    """Install only the reporting subclass; restore the original on every exit."""
    if terminal_phase not in range(-1, 7):
        raise ValueError("invalid terminal phase")
    if orchestrate.Runner is not _FROZEN_RUNNER:
        raise RuntimeError("another Runner override is already installed")
    runners = []
    sessions = []

    class InvocationRunner(CoalescedReportRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.terminal_phase = terminal_phase
            runners.append(self)
            if provenance is not None:
                sessions.append((self, _begin_provenance(self, provenance)))

    orchestrate.Runner = InvocationRunner
    failure = None
    try:
        yield runners
    except BaseException as error:
        failure = error
        raise
    finally:
        orchestrate.Runner = _FROZEN_RUNNER
        for runner, session in sessions:
            try:
                _finish_provenance(runner, session, failure)
            except BaseException as error:
                if failure is None:
                    raise
                warnings.warn(f"Resume provenance finalization also failed: {error}", RuntimeWarning)


def validate_resume(args) -> dict:
    """Restrict this driver to a pre-existing, already-frozen final study."""
    if not args.resume:
        raise ValueError("The report-only driver requires --resume")
    root = args.output_root.resolve()
    study = Study.from_path(args.config)
    resolved = _read_json(root / "study.json")
    if resolved != {**study.mapping(), "study_hash": study.config_hash}:
        raise RuntimeError("Resolved study differs from the frozen study")
    ledger = _read_json(root / "phase1_ledger.json")
    if ledger.get("source_hashes") != source_hashes():
        raise RuntimeError("Frozen scientific source hashes changed")
    for path in (root / "final_ledger.json", root / "decisions" / "phase4_selection.json"):
        if _read_json(path).get("study_hash") != study.config_hash:
            raise RuntimeError(f"Missing or incompatible frozen final-study decision: {path}")
    return {"driver_module": "mnist_experiment.akaike_score_gate_resume",
            "driver_source_sha256": file_hash(Path(__file__).resolve()),
            "resolved_config": study.mapping(), "notebook": str(args.notebook.resolve()),
            "through_phase": args.through_phase,
            "scientific_settings_changed": False,
            "reporting_only": True, "heartbeat_seconds": 900}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = orchestrate.parse_arguments(arguments)
    provenance = {**validate_resume(args),
                  "command": [sys.executable, "-m", "mnist_experiment.akaike_score_gate_resume", *arguments]}
    with coalesced_reporting(terminal_phase=args.through_phase, provenance=provenance):
        orchestrate.main(arguments)


if __name__ == "__main__":
    main()
