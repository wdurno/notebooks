"""Run bounded checks with persistent Phase 7 compute accounting."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import uuid

from mnist_experiment.diagonal_lfu.report import _atomic, _json_bytes
from mnist_experiment.fisher_memory.control import utc_now

from .runtime import ACCEPTANCE_LEDGER, FAMILY, historical, phase_seconds, read_json


def audit(root):
    """Verify a completed study and append a new, uniquely named audit record."""
    import json
    import nbformat
    import numpy as np
    from PIL import Image
    from mnist_experiment.fisher_memory.artifacts import iter_units
    from .report import FIGURES, MARKER, _validated, _verify_generation
    from .runtime import REPO, file_digest, scientific_sources

    original = Path(root)
    if original.is_symlink():
        raise ValueError("audit root cannot be a symlink")
    root = original.resolve()
    if not root.is_relative_to(FAMILY.resolve()) or not (root / "study_completed.json").is_file():
        raise ValueError("audit requires a completed Phase 7 family root")
    with (root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = read_json(root / "identity.json")
        if identity["identity"]["sources"] != scientific_sources():
            raise ValueError("frozen scientific source manifest changed")
        snapshot, result, _ = _validated(root)
        if snapshot["status"] != "completed" or result["counts"]["incomplete"]:
            raise ValueError("final report contains incomplete evidence")
        units = list(iter_units(root, full=True))
        if any(u["status"] != "completed" for u in units):
            raise ValueError("incomplete scientific unit remains")
        selection = read_json(root / "units/decision-selection/decision.json")
        if snapshot["selection"] != selection or read_json(root / "selection.json") != selection:
            raise ValueError("frozen selection mismatch")
        if read_json(root / "study_completed.json").get("selection") != selection:
            raise ValueError("completion marker differs from frozen selection")
        trajectories = {t["unit_id"]: t for t in snapshot["trajectories"]}
        fresh = 0
        for record in units:
            if record["metadata"].get("kind") != "trajectory":
                continue
            uid = record["unit_id"]
            expected = {"unit_id": uid, **record["metadata"], "completed": True,
                "metrics": read_json(root / "units" / uid / "metrics.json"),
                "summary": read_json(root / "units" / uid / "summary.json")}
            if trajectories.pop(uid, None) != expected:
                raise ValueError("final snapshot differs from sealed trajectory")
            fresh += expected["stage"] == "fresh"
        target = 4 * (2 * selection["main_replicas"] + selection["stationary_replicas"])
        if trajectories or fresh != target or result["awaiting_full_block_units"]:
            raise ValueError("frozen fresh cohort is not fully paired and complete")
        manifest = read_json(root / "report/latest.json")
        directory = Path(manifest["report_dir"])
        _verify_generation(directory, manifest)
        notebook = Path(manifest["notebook"])
        if notebook.read_bytes() != (directory / "report.ipynb").read_bytes():
            raise ValueError("canonical notebook differs from its sealed generation")
        book = nbformat.read(notebook, as_version=4)
        nbformat.validate(book)
        history = snapshot["historical_report_dir"]
        inherited = 0
        if history is not None:
            parent = Path(history)
            if file_digest(parent / "report.ipynb") != snapshot["historical_notebook_sha256"]:
                raise ValueError("historical notebook changed")
            old = nbformat.read(parent / "report.ipynb", as_version=4)
            by_id = {cell.id: cell for cell in book.cells}
            if len(by_id) != len(book.cells) or any(by_id.get(cell.id) != cell for cell in old.cells):
                raise ValueError("historical notebook cells or outputs changed")
            if any(book.metadata.get(k) != v for k, v in old.metadata.items()):
                raise ValueError("historical notebook metadata changed")
            inherited = len(old.cells)
        if book.metadata[MARKER]["study_id"] != snapshot["study_id"]:
            raise ValueError("notebook ownership changed")
        pictures = {}
        for name, _ in FIGURES:
            path = directory / (name + ".png")
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
            if min(pixels.shape[:2]) < 64 or float(pixels.var()) <= 1:
                raise ValueError("blank or invalid final figure")
            pictures[path.name] = {"sha256": file_digest(path), "shape": list(pixels.shape)}
        payload = {"schema": 1, "completed_at": utc_now(), "root": str(root),
            "unit_count": len(units), "fresh_paths": fresh, "paired_blocks": result["counts"]["paired_blocks"],
            "source_count": len(identity["identity"]["sources"]), "historical_cells_preserved": inherited,
            "notebook_sha256": file_digest(notebook), "report_dir": str(directory),
            "nonblank_figures": pictures, "recomputed_analysis_matches": True}
        target = root / ("closeout_validation-" + uuid.uuid4().hex[:12] + ".json")
        if target.exists():
            raise ValueError("audit record collision")
        _atomic(target, _json_bytes(payload))
        _atomic(root / "closeout_latest.json", _json_bytes({"record": target.name, "sha256": file_digest(target)}))
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--overhead-seconds", type=float)
    parser.add_argument("--description", default="Unrooted acceptance and preflight overhead")
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.audit_root is not None:
        if command or args.overhead_seconds is not None:
            parser.error("audit-root cannot be combined with a command or overhead")
        command = [sys.executable, "-c",
            "from mnist_experiment.kalman_pi.discrepancy.acceptance import audit; import sys; audit(sys.argv[1])",
            str(args.audit_root)]
    if args.overhead_seconds is not None:
        if command or not 0 < args.overhead_seconds <= 3600:
            parser.error("overhead requires no command and seconds in (0,3600]")
    elif not command or not 0 < args.timeout <= 3600:
        parser.error("supply a command and timeout in (0,3600]")
    FAMILY.mkdir(parents=True, exist_ok=True)
    key = "phase7-check-" + uuid.uuid4().hex[:12]
    with (FAMILY / ".allocation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserve = args.overhead_seconds if args.overhead_seconds is not None else args.timeout
        if phase_seconds(FAMILY / "unrooted-checks") + reserve > 72000:
            raise RuntimeError("Phase 7 cannot reserve this validation time")
        ledger = read_json(ACCEPTANCE_LEDGER) if ACCEPTANCE_LEDGER.exists() else {"schema": 1, "entries": []}
        ledger["entries"].append({"id": key, "seconds": reserve,
            "status": "conservative-overhead" if args.overhead_seconds is not None else "running",
            "description": args.description, "started_at": utc_now(), "command": command})
        _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
        if args.overhead_seconds is not None:
            return 0
        # Keep the family lock through settlement: a live worker must never
        # publish a provisional test reservation as settled historical compute.
        started = time.monotonic()
        code = 1
        try:
            code = subprocess.run(command, timeout=args.timeout, check=False).returncode
            return code
        finally:
            elapsed = time.monotonic() - started
            ledger = read_json(ACCEPTANCE_LEDGER)
            entry = next(row for row in ledger["entries"] if row["id"] == key)
            entry.update(seconds=elapsed, status="completed" if code == 0 else "failed",
                         exit_code=code, finished_at=utc_now())
            _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))


if __name__ == "__main__":
    sys.exit(main())
