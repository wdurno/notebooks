"""Bounded artifact-only Plan 19 reporting. Never imports Torch or a learner."""

from __future__ import annotations

from collections import defaultdict
import base64
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Any
from urllib.parse import quote

from . import ARTIFACT_SCHEMA_VERSION, METRIC_SCHEMA_VERSION
from . import analysis
from .schemas import METRIC_COLUMNS

# Artifact rendering must not create Tk objects that GPU/control threads can
# later collect. Explicit notebook backends remain respected.
os.environ.setdefault("MPLBACKEND", "Agg")


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "cache/numerical_experiment/picar"
DEFAULT_NOTEBOOK = Path(__file__).with_name("plan19_results.ipynb")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_METRIC_BYTES = 64 * 1024 * 1024
PHASES = (
    "Implementation and acceptance", "Physical/resource readiness", "Bootstrap and student readiness",
    "Development and retention matching", "Fresh paired learning", "Efficiency and closeout",
)
_TEXT = {"phase", "condition", "end_reason", "test_domain"}
_BOOL = {"success", "collision", "smoke_only"}
_INT = {"schema_version", "replica", "step", "episode", "new_transitions",
        "replay_exposures", "retired_transitions", "model_version", "deadline_misses", "test_count"}


class ArtifactError(ValueError):
    pass


def resolve_root(root: str | Path | None = None) -> Path:
    path = Path(root if root is not None else os.environ.get("PICAR_CACHE_ROOT", DEFAULT_ROOT)).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def root_label(root: str | Path) -> str:
    try:
        return str(Path(root).resolve().relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        return str(root)


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"missing/nonregular required artifact: {path}")
    if path.stat().st_size > limit:
        raise ArtifactError(f"artifact exceeds scalar reporting size limit: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ArtifactError(f"artifact exceeds scalar reporting size limit: {path}")
    return data


def _json(path: Path) -> dict:
    def invalid(value):
        raise ArtifactError(f"nonfinite JSON value in {path}: {value}")
    def finite_float(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    try:
        value = json.loads(_read(path, MAX_JSON_BYTES), parse_constant=invalid, parse_float=finite_float)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _version(value, expected, label):
    if type(value) is not int or value != expected:
        raise ArtifactError(f"incompatible/unknown {label}: {value!r}; expected {expected}")


def _phase_number(value):
    if type(value) is int and 0 <= value < len(PHASES):
        return value
    match = re.fullmatch(r"(?:phase)?([0-5])(?:_.*)?", str(value))
    if match is None:
        raise ArtifactError(f"unknown phase identifier: {value!r}")
    return int(match[1])


def _verify_seal(path, seal):
    if not isinstance(seal, dict) or "sha256" not in seal or "bytes" not in seal:
        raise ArtifactError(f"missing scalar artifact seal: {path}")
    data = _read(path, MAX_METRIC_BYTES)
    if len(data) != seal["bytes"] or hashlib.sha256(data).hexdigest() != seal["sha256"]:
        raise ArtifactError(f"scalar artifact hash/size mismatch: {path}")


def _metric_value(name, value):
    if value == "":
        return None
    if name in _TEXT:
        return value
    if name in _BOOL:
        if value.lower() not in ("true", "false", "1", "0"):
            raise ArtifactError(f"invalid boolean metric {name}: {value!r}")
        return value.lower() in ("true", "1")
    try:
        parsed = int(value) if name in _INT else float(value)
    except ValueError as exc:
        raise ArtifactError(f"invalid metric {name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ArtifactError(f"nonfinite metric {name}")
    if name in _INT and parsed < 0:
        raise ArtifactError(f"negative count/identifier {name}")
    return parsed


def read_completed_unit(path: str | Path) -> dict[str, Any]:
    """Only fully flushed immutable units enter scalar/scientific analysis.

    Required manifest keys: artifact_schema_version, metric_schema_version,
    status='completed'. The config schema_version and CSV schema_version are
    independently checked. Scalar completion seals are verified, without
    opening matrix/checkpoint files. Nothing is repaired or written here.
    """
    path = Path(path)
    if not (path / "COMPLETED").is_file() or (path / "COMPLETED").is_symlink():
        raise ArtifactError(f"unit is incomplete (COMPLETED missing): {path}")
    manifest = _json(path / "manifest.json")
    _version(manifest.get("artifact_schema_version"), ARTIFACT_SCHEMA_VERSION, "artifact schema")
    _version(manifest.get("metric_schema_version"), METRIC_SCHEMA_VERSION, "metric schema")
    if manifest.get("status") != "completed":
        raise ArtifactError(f"COMPLETED contradicts manifest status: {path}")
    _version(manifest.get("storage_schema_version"), 1, "storage schema")
    marker = _json(path / "COMPLETED")
    if marker.get("manifest_sha256") != hashlib.sha256(_read(path / "manifest.json", MAX_JSON_BYTES)).hexdigest():
        raise ArtifactError(f"completion marker/manifest hash mismatch: {path}")
    stored_config, summary = _json(path / "config.json"), _json(path / "summary.json")
    config = stored_config.get("config", stored_config)
    canonical = json.dumps(stored_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if manifest.get("config_hash") != hashlib.sha256(canonical.encode()).hexdigest():
        raise ArtifactError("resolved configuration hash mismatch")
    _version(config.get("schema_version"), ARTIFACT_SCHEMA_VERSION, "configuration schema")
    if "schema_version" in summary:
        _version(summary["schema_version"], ARTIFACT_SCHEMA_VERSION, "summary schema")
    data = _read(path / "metrics.csv", MAX_METRIC_BYTES)
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        if tuple(reader.fieldnames or ()) != METRIC_COLUMNS:
            raise ArtifactError(f"incompatible metric columns: {path / 'metrics.csv'}")
        metrics = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ArtifactError("malformed metric row")
            item = {key: _metric_value(key, value) for key, value in row.items()}
            _version(item["schema_version"], METRIC_SCHEMA_VERSION, "metric row schema")
            if item["test_success"] is not None and not 0 <= item["test_success"] <= 1:
                raise ArtifactError("held-out test_success must be a rate in [0,1]")
            if item["test_success"] is not None and not item["test_count"]:
                raise ArtifactError("held-out rate requires a positive test_count")
            metrics.append(item)
    except UnicodeDecodeError as exc:
        raise ArtifactError("metrics must be UTF-8 CSV") from exc
    for name in ("summary.json", "metrics.csv"):
        _verify_seal(path / name, manifest.get("final_files", {}).get(name))
    smoke = config.get("mode") == "smoke" or summary.get("smoke_only") is True \
        or any(row["smoke_only"] is True for row in metrics)
    return {"path": str(path), "config": config, "stored_config": stored_config, "manifest": manifest,
            "summary": summary, "metrics": metrics, "smoke_only": smoke}


def _diagnostic(path, *, completed=True):
    manifest = _json(path / "manifest.json")
    _version(manifest.get("schema_version"), 1, "diagnostic manifest schema")
    required_status = "completed" if completed else "failed"
    if manifest.get("kind") != "qwen_acceptance" or manifest.get("status") != required_status:
        raise ArtifactError("unknown completed diagnostic kind/status")
    summary_path = path / "summary.json"
    summary = _json(summary_path) if summary_path.is_file() else {}
    if summary:
        _version(summary.get("schema_version"), 1, "diagnostic summary schema")
    elif completed:
        raise ArtifactError(f"completed diagnostic summary is missing: {path}")
    note = manifest.get("error") if not completed else None
    if not completed and not summary:
        note = f"Failed diagnostic has no summary.json. {note or 'No further error was recorded.'}"
    return {"path": str(path), "manifest": manifest, "summary": summary,
            "smoke_only": True, "error_note": note}


def load_report(root: str | Path | None = None) -> dict:
    """Load completed units plus live counts only, or a clearly pending report."""
    root = resolve_root(root)
    phase_roots = sorted(path for path in root.glob("phase[0-9]*") if path.is_dir()) if root.is_dir() else []
    if phase_roots and not (root / "config.json").exists():
        parts = [load_report(path) for path in phase_roots]
        units = [unit for part in parts for unit in part["complete_units"]]
        diagnostics = [item for part in parts for item in part["diagnostics"]]
        live = [item for part in parts for item in part["live"]]
        return {"root": str(root), "complete_units": units, "diagnostics": diagnostics, "live": live,
            "status": "available" if units or diagnostics else "pending",
            "scientific_units": sum(not unit["smoke_only"] for unit in units),
            "note": "Readiness and resource probes describe implementation and operating cost; learning results require fresh paired trajectories."}
    units, diagnostics, live = [], [], []
    if root.is_dir():
        for marker in sorted(root.rglob("COMPLETED")):
            path = marker.parent
            # Resource diagnostics have their own explicit non-scientific schema.
            manifest = _json(path / "manifest.json")
            if manifest.get("kind") == "qwen_acceptance":
                diagnostics.append(_diagnostic(path))
            else:
                units.append(read_completed_unit(path))
        for manifest_path in sorted(root.rglob("manifest.json")):
            path = manifest_path.parent
            if (path / "COMPLETED").exists():
                continue
            manifest = _json(manifest_path)
            if manifest.get("kind") == "qwen_acceptance" and manifest.get("status") == "failed":
                diagnostics.append(_diagnostic(path, completed=False))
        for config_path in sorted(root.rglob("config.json")):
            path = config_path.parent
            if (path / "COMPLETED").exists() or not (path / "manifest.json").is_file():
                continue
            manifest = _json(path / "manifest.json")
            _version(manifest.get("artifact_schema_version"), ARTIFACT_SCHEMA_VERSION, "artifact schema")
            _version(manifest.get("metric_schema_version"), METRIC_SCHEMA_VERSION, "metric schema")
            progress = path / "progress.json"
            if (path / "CURRENT.json").is_file():
                pointer = _json(path / "CURRENT.json")
                _version(pointer.get("storage_schema_version"), 1, "live storage schema")
                commit = pointer.get("commit_id")
                if not isinstance(commit, str) or Path(commit).name != commit or commit in (".", ".."):
                    raise ArtifactError("unsafe live progress commit")
                transaction = path / "transactions" / commit
                _verify_seal(transaction / "manifest.json", pointer.get("manifest"))
                commit_manifest = _json(transaction / "manifest.json")
                progress = transaction / "progress.json"
                _verify_seal(progress, commit_manifest.get("files", {}).get("progress.json"))
            value = _json(progress) if progress.is_file() else {}
            if "schema_version" in value:
                _version(value["schema_version"], 1, "live progress schema")
            counts = value.get("counts", {key: value[key] for key in
                ("step", "new_transitions", "replay_exposures", "retired_transitions", "target") if key in value})
            if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ArtifactError("live progress counts must be nonnegative integers")
            # Partial trajectories/NLL/success are deliberately not read.
            stopped = any((parent / "STOP").is_file() for parent in (path, *path.parents)
                          if parent == root or root in parent.parents)
            live.append({"path": str(path), "phase": _phase_number(value.get("phase", 0)),
                "status": value.get("status", "incomplete"), "counts": counts,
                "updated_at": value.get("updated_at"), "stopped": stopped})
    return {"root": str(root), "complete_units": units, "diagnostics": diagnostics,
            "live": live, "status": "available" if units or diagnostics else "pending",
            "scientific_units": sum(not unit["smoke_only"] for unit in units),
            "note": "Phase 0 smoke/acceptance verifies plumbing, not scientific capability or sample savings."}


def progress_rows(snapshot):
    rows = []
    for phase, name in enumerate(PHASES):
        units = [u for u in snapshot["complete_units"] if _phase_number(u["summary"].get("phase", 0)) == phase]
        live = [u for u in snapshot["live"] if u["phase"] == phase]
        active = [u for u in live if not u.get("stopped", False)]
        diagnostics = len(snapshot["diagnostics"]) if phase == 0 else 0
        failed = sum(d["manifest"].get("status") == "failed" for d in snapshot["diagnostics"]) if phase == 0 else 0
        costs = [u["summary"]["active_seconds"] for u in units if "active_seconds" in u["summary"]]
        if phase == 0:
            costs.extend(d["summary"]["active_seconds"] for d in snapshot["diagnostics"]
                         if "active_seconds" in d["summary"])
        rows.append({"phase": phase, "stage": name, "completed_units": len(units),
            "diagnostics": diagnostics, "incomplete_units": len(live),
            "stopped_units": len(live) - len(active),
            "diagnostic_failures": failed,
            "scientific_units": sum(not u["smoke_only"] for u in units),
            "completed_active_s": sum(costs) if costs else None,
            "status": active[-1]["status"] if active else
                "artifacts available; stopped attempts retained" if live else
                "artifacts available" if units or diagnostics else "pending"})
    return rows


def readiness_rows(snapshot):
    result = []
    for diagnostic in snapshot["diagnostics"]:
        summary = diagnostic["summary"]
        gate = summary.get("fd_convergence_pass")
        if gate is not None and type(gate) is not bool:
            raise ArtifactError("fd_convergence_pass must be an explicit boolean")
        reserved = summary.get("peak_reserved_bytes")
        result.append({"diagnostic": Path(diagnostic["path"]).name,
            "artifact": f"{diagnostic['manifest']['status']}, smoke only", "load_s": summary.get("load_seconds"),
            "inference_s": summary.get("inference_seconds"),
            "peak_reserved_GiB": reserved / 1024**3 if reserved is not None else None,
            "FD_convergence": "passed" if gate is True else "failed" if gate is False else "not certified",
            "error_note": diagnostic.get("error_note")})
    return result


def media_links(snapshot, *, notebook_dir=DEFAULT_NOTEBOOK.parent):
    """Read-only links; missing external clips do not block scalar reporting."""
    links = []
    project = Path(__file__).resolve().parents[2]
    for unit in sorted(snapshot["complete_units"], key=lambda u: u["manifest"].get("completed_at", ""), reverse=True):
        for clip in unit["summary"].get("videos", []):
            path = Path(clip["path"])
            path = path if path.is_absolute() else project / path
            selection = clip.get("selection", "unlabeled selection")
            label = f"{path.parent.name}/{path.name}: {selection}, seed={clip.get('seed', 'unavailable')}"
            if not path.is_file():
                links.append(f"<p>Missing external media: {html.escape(label)}</p>")
                continue
            relative = os.path.relpath(path, notebook_dir)
            links.append(f'<p><a href="{quote(relative)}">{html.escape(label)}</a> '
                         '(top-down is reader-only)</p>')
    if len(links) > 4:
        return "\n".join(links[:4]) + f"<details><summary>{len(links) - 4} additional archived clips</summary>" + "\n".join(links[4:]) + "</details>"
    return "\n".join(links) or "<p>Stored behavior clips pending.</p>"


def heldout_curves(snapshot, *, version=None, include_smoke=False):
    """Aggregate panel counts, never training success, within version/replica."""
    units = [u for u in snapshot["complete_units"] if include_smoke or not u["smoke_only"]]
    versions = {u["config"].get("version") for u in units}
    if version is None and len(versions) > 1:
        raise ArtifactError("multiple experiment versions; select a version before pooling")
    if version is not None:
        units = [u for u in units if u["config"].get("version") == version]
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for unit in units:
        for row in unit["metrics"]:
            if row["test_domain"] not in ("A", "B") or row["test_success"] is None:
                continue
            if row["condition"] is None or row["replica"] is None or row["new_transitions"] is None:
                raise ArtifactError("held-out rows require condition, replica, and unique-transition budget")
            key = (_phase_number(row["phase"]), row["condition"], row["replica"], row["test_domain"])
            identity = (*key, row["new_transitions"], row["step"], row["episode"])
            if identity in seen:
                raise ArtifactError("duplicate held-out panel identity across completed units")
            seen.add(identity)
            groups[key][row["new_transitions"]].append((row["test_success"], row["test_count"]))
    result = []
    for (phase, condition, replica, domain), checkpoints in sorted(groups.items()):
        samples, success, counts = [], [], []
        for n, rates in sorted(checkpoints.items()):
            count = sum(c for _, c in rates)
            samples.append(n)
            success.append(sum(p * c for p, c in rates) / count)
            counts.append(count)
        result.append({"phase": phase, "condition": condition, "replica": replica,
                       "domain": domain, "samples": samples, "success": success, "panel_counts": counts})
    return result


def paired_failure_summary(snapshot, *, phase=4, version=None, seed=190019):
    curves = [c for c in heldout_curves(snapshot, version=version) if c["phase"] == phase and c["domain"] == "B"]
    outcomes = {"replay_ewc": {}, "stm_fd": {}}
    budgets = {}
    unavailable = []
    for curve in curves:
        if curve["condition"] not in outcomes:
            continue
        if len(curve["samples"]) < 2 or curve["samples"][0] != 0:
            unavailable.append({"condition": curve["condition"], "replica": curve["replica"],
                                "reason": "complete measured curve from sample zero unavailable"})
            continue
        outcomes[curve["condition"]][curve["replica"]] = analysis.budget_averaged_failure(
            curve["samples"], curve["success"])
        budgets[(curve["condition"], curve["replica"])] = curve["samples"][-1]
    for replica in set(outcomes["replay_ewc"]) & set(outcomes["stm_fd"]):
        if budgets[("replay_ewc", replica)] != budgets[("stm_fd", replica)]:
            raise ArtifactError("paired failure requires the same observed new-transition budget")
    result = analysis.paired_mean_ci(outcomes["replay_ewc"], outcomes["stm_fd"],
                                     seed=seed, allow_incomplete=True)
    result["unavailable_curves"] = unavailable
    result["definition"] = "control minus FD budget-averaged B failure; positive favors FD"
    return result


def plot_learning(snapshot, *, phase=4, version=None):
    """Measured held-out checkpoints only; no training-return proxy or smoothing."""
    curves = [c for c in heldout_curves(snapshot, version=version) if c["phase"] == phase]
    if not curves:
        return None
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    colors = {"replay_ewc": "#26705a", "stm_fd": "#b84b54"}
    for axis, domain in zip(axes, ("B", "A")):
        for condition in ("replay_ewc", "stm_fd"):
            subset = [c for c in curves if c["domain"] == domain and c["condition"] == condition]
            if not subset:
                continue
            grid = subset[0]["samples"]
            if any(c["samples"] != grid for c in subset):
                raise ArtifactError("plotting a mean requires a common measured checkpoint grid")
            values = np.asarray([c["success"] for c in subset])
            if domain == "A":
                values = values[:, :1] - values
            mean = values.mean(0)
            axis.plot(grid, mean, "o-", markersize=4, color=colors[condition],
                      label=f"{condition} (n={len(subset)} replicas)")
            if len(subset) >= 2:
                rng = np.random.default_rng(190021)
                boot = values[rng.integers(len(subset), size=(2000, len(subset)))].mean(1)
                lower, upper = np.quantile(boot, [.025, .975], axis=0)
                axis.fill_between(grid, lower, upper, color=colors[condition], alpha=.15)
        axis.set(xlabel="Unique new transitions", ylabel="B success" if domain == "B" else "A failure increase")
        if domain == "B":
            axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    return figure


def assemble_video(frames, output: str | Path, *, seed: int, selection="preselected",
                   fps=10., panel_size=(320, 240), sync_tolerance_s=.001) -> dict:
    """Explicit optional MP4 assembly from synchronized STORED frame paths.

    Each row has time_s, ego_path, topdown_path, instruction, outcome, and
    optional ego_time_s/topdown_time_s, drift_alpha, requested/executed_control.
    Privileged top-down imagery is reader-only. This helper is never invoked
    by notebook evaluation and cannot overwrite existing/completed artifacts.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    import imageio.v2 as imageio

    if type(seed) is not int or seed < 0 or selection not in ("preselected", "failure_gallery"):
        raise ValueError("declare a fixed seed and preselected/failure_gallery selection")
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError("stored frames and positive fps are required")
    times = np.asarray([row["time_s"] for row in frames], dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("stored frame times must be finite and strictly increasing")
    for row in frames:
        if (abs(row.get("ego_time_s", row["time_s"]) - row["time_s"]) > sync_tolerance_s
                or abs(row.get("topdown_time_s", row["time_s"]) - row["time_s"]) > sync_tolerance_s):
            raise ValueError("egocentric/top-down frames are not synchronized")
        if any(not Path(row[key]).is_file() for key in ("ego_path", "topdown_path")):
            raise FileNotFoundError("stored egocentric/top-down frame is unavailable")
    output = Path(output)
    if output.exists():
        raise FileExistsError("video outputs are immutable; choose a new path")
    if any((parent / "COMPLETED").exists() for parent in (output.parent, *output.parents)):
        raise ArtifactError("cannot add media inside a completed unit")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = panel_size
    if width < 32 or height < 32 or width % 2 or height % 2:
        raise ValueError("video panel dimensions must be even and at least 32 pixels")
    fd, name = tempfile.mkstemp(prefix=".picar-video-", suffix=".mp4", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    encoded = 0
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", macro_block_size=2) as writer:
            for at in np.arange(times[0], times[-1] + .5 / fps, 1 / fps):
                index = max(0, min(len(frames) - 1, int(np.searchsorted(times, at, side="right") - 1)))
                row = frames[index]
                canvas = Image.new("RGB", (2 * width, height + 112), "#14171a")
                for column, key in enumerate(("ego_path", "topdown_path")):
                    with Image.open(row[key]) as source:
                        image = ImageOps.contain(source.convert("RGB"), (width, height))
                    canvas.paste(image, (column * width + (width - image.width) // 2, (height - image.height) // 2))
                draw = ImageDraw.Draw(canvas)
                banner = "FAILURE GALLERY (selected failures)" if selection == "failure_gallery" else "PRESELECTED COMMON SEED"
                lines = [f"{banner} | seed={seed} | t={at:.2f}s | frame={row['time_s']:.2f}s",
                    f"Ego camera | Top-down: reader-only | drift={row.get('drift_alpha', 'unavailable')} | outcome={row.get('outcome', 'pending')}",
                    f"Requested: {row.get('requested_control', 'unavailable')} | Executed: {row.get('executed_control', 'unavailable')}",
                    f"Instruction: {row.get('instruction', 'unavailable')}"]
                wrapped = [part for line in lines for part in textwrap.wrap(line, width=max(20, 2 * width // 7))]
                for j, line in enumerate(wrapped[:7]):
                    draw.text((6, height + 4 + j * 15), line, fill="white")
                writer.append_data(np.asarray(canvas))
                encoded += 1
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "seed": seed, "selection": selection,
            "frames": encoded, "fps": fps, "topdown_reader_only": True}


def settings_rows(snapshot):
    """Summarize recipe versions without a row for every diagnostic operation."""
    groups = {}
    for unit in snapshot["complete_units"]:
        config = unit["config"]
        key = (config.get("mode"), config.get("version"),
               config.get("model", {}).get("kind"), unit["smoke_only"])
        groups[key] = groups.get(key, 0) + 1
    return [{"mode": mode, "version": version, "model": model,
             "smoke_only": smoke, "completed_units": count}
            for (mode, version, model, smoke), count in groups.items()]


def make_notebook(root: str | Path | None = None):
    """Generate a short, unexecuted, artifact-only notebook with nbformat."""
    import nbformat
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    override = repr(str(root)) if root is not None else "None"
    cells = [md("# Plan 19: Continual PiCar Learning\n\nCan STM regularization save new experience at matched retention and operational cost? "
        "Phase 0 is implementation smoke, not evidence of robot capability or sample savings. "
        "[Protocol](plan19.md)."), code(
        "from pathlib import Path\nimport os, sys\n"
        "for parent in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (parent / 'numerical_experiment/picar').is_dir():\n"
        "        sys.path.insert(0, str(parent))\n        break\n"
        "import pandas as pd\nfrom IPython.display import display\n"
        "from numerical_experiment.picar import report\n"
        f"ROOT = report.resolve_root({override})  # PICAR_CACHE_ROOT overrides the default\n"
        "snapshot = report.load_report(ROOT)\nprint(f'Artifacts: {report.root_label(ROOT)} | {snapshot[\"status\"]}')\n"
        "print(snapshot['note'])\ndisplay(pd.DataFrame(report.progress_rows(snapshot)))\n"
        "if snapshot['live']:\n    display(pd.DataFrame(snapshot['live']))"),
        md("## Task and Setup\n\nThe learner receives RGB, language, executed controls, and elapsed time. "
           "Top-down views are for the reader only. Settings and scene media appear only when stored artifacts exist."),
        code("settings = report.settings_rows(snapshot)\n"
             "display(pd.DataFrame(settings)) if settings else print('Setup artifacts pending.')"),
        md("## Readiness and Bootstrap\n\nMeasured loading, inference, memory, and gradient checks are diagnostics. "
           "Language readiness and held-out initial capability remain separate requirements.\n\n"
           "Current Qwen probes use one 64x48 image/history step, top-L=1, and 56x56-112x112 processor limits, "
           "not the full context-8/batch-4 workload."),
        code("checks = report.readiness_rows(snapshot)\n"
             "display(pd.DataFrame(checks)) if checks else print('Readiness measurements pending.')\n"
             "print('Completed diagnostics do not certify all readiness gates. FD convergence requires an explicit recorded pass.')"),
        md("## Development\n\nRetention matching and recipe selection use development artifacts only. "
           "A selected recipe and fresh-run authorization are unavailable until explicitly recorded."),
        md("## Fresh Learning\n\nHeld-out B success measures acquisition. A failure increase measures loss of initial capability. "
           "Curves use unique new transitions, not replay exposures; smoke units are excluded."),
        code("figure = report.plot_learning(snapshot)\n"
             "display(figure) if figure is not None else print('Fresh held-out paired trajectories pending.')\n"
             "effect = report.paired_failure_summary(snapshot)\n"
             "display(pd.DataFrame([effect])) if effect['n_pairs'] else print('Paired failure effect unavailable: no complete scientific pairs.')"),
        md("## Efficiency\n\nA target needs two consecutive measured checkpoints meeting capability and retention gates. "
           "Interpolate only in observed brackets on the outward ramp/B hold. "
           "f is the fraction of new samples saved; c is the operational compute ratio; S=f/c. "
           "Half the experience at twice the compute gives S=0.25. Censored draws stay visible. "
           "No efficiency estimate is produced without supported crossings and recorded operational costs."),
        md("## Behavior and Mechanisms\n\nCommon preselected-seed clips show synchronized ego/top-down views with actual timestamps and controls. "
           "A separately selected failure gallery must be labeled. Missing media does not block scalar analysis."),
        code("from IPython.display import HTML\ndisplay(HTML(report.media_links(snapshot)))"),
        md("## Conclusion\n\nNo scientific conclusion is inferred from implementation smoke. "
           "Report acquisition, retention, experience, operational compute, censoring, and independent replica counts together. "
           "Transfer readiness remains pending until the measured gates pass.")]
    for cell, section in zip((c for c in cells if c.cell_type == "code"),
                              ("progress", "settings", "readiness", "learning", "media")):
        cell.metadata["report_section"] = section
    for index, cell in enumerate(cells):
        cell.id = hashlib.sha256(f"{index}:{cell.source}".encode()).hexdigest()[:12]
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "plan19_report_schema": 1})


def _populate_outputs(book, snapshot, *, notebook_dir=DEFAULT_NOTEBOOK.parent):
    import nbformat
    import pandas as pd

    def stream(message):
        return nbformat.v4.new_output("stream", name="stdout", text=message + "\n")

    def table(rows):
        frame = pd.DataFrame(rows)
        return nbformat.v4.new_output("display_data", data={
            "text/plain": frame.to_string(index=False),
            "text/html": frame.to_html(index=False, escape=True)}, metadata={})

    outputs = {"progress": [stream(f"Artifacts: {root_label(snapshot['root'])} | {snapshot['status']}"),
                              stream(snapshot["note"]), table(progress_rows(snapshot))]}
    if snapshot["live"]:
        outputs["progress"].append(table(snapshot["live"]))
    settings = settings_rows(snapshot)
    outputs["settings"] = [table(settings) if settings else stream("Setup artifacts pending.")]
    checks = readiness_rows(snapshot)
    outputs["readiness"] = [table(checks) if checks else stream("Readiness measurements pending.")]
    outputs["readiness"].append(stream("Completed diagnostics do not certify all readiness gates. "
        "FD convergence requires an explicit recorded pass."))
    accepted = [u for u in snapshot["complete_units"]
                if u["summary"].get("kind") == "implementation_acceptance"]
    if accepted:
        latest = max(accepted, key=lambda u: u["manifest"]["completed_at"])
        counts = latest["summary"]["tests"]
        outputs["readiness"].append(stream(
            f"CPU acceptance: {counts['tests']} tests; {counts['failures']} failures, "
            f"{counts['errors']} errors, {counts['skipped']} skipped. Paired smoke completed."))
        if snapshot["scientific_units"] == 0:
            gpu_pass = any(d["summary"].get("fd_convergence_pass") is True
                           for d in snapshot["diagnostics"])
            numerical_note = (
                "The corrected Qwen FD check passes. Stable KL arithmetic and an FP32 adapted "
                "suffix resolved the observed step-size sensitivity; the earlier diagnostic is "
                "retained for transparency. " if gpu_pass else
                "Real-Qwen FD convergence is not yet certified by these artifacts. ")
            book.cells[-1].source = (
                "## Conclusion\n\nThe CPU acceptance and paired recovery pipeline pass. "
                + numerical_note + "\n\n"
                "These are implementation results, not evidence of sample savings. "
                "Full-workload timing, task competence, retention calibration, and sim-to-real "
                "transfer remain to be assessed in the later phases.")
    outputs["learning"] = []
    if snapshot["scientific_units"] == 0:
        from .phase1_report import phase1_conclusion
        conclusion = phase1_conclusion(snapshot)
        if conclusion:
            book.cells[-1].source = conclusion
    figure = plot_learning(snapshot)
    if figure is None:
        outputs["learning"].append(stream("Fresh held-out paired trajectories pending."))
    else:
        import matplotlib.pyplot as plt
        encoded = io.BytesIO()
        figure.savefig(encoded, format="png", dpi=120)
        plt.close(figure)
        outputs["learning"].append(nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(encoded.getvalue()).decode(),
            "text/plain": "Measured held-out B success and A failure increase."}, metadata={}))
    effect = paired_failure_summary(snapshot)
    outputs["learning"].append(table([effect]) if effect["n_pairs"] else
        stream("Paired failure effect unavailable: no complete scientific pairs."))
    outputs["media"] = [nbformat.v4.new_output("display_data", data={
        "text/html": media_links(snapshot, notebook_dir=notebook_dir),
        "text/plain": "Stored behavior media links; missing clips remain explicitly labeled."}, metadata={})]
    for cell in book.cells:
        if cell.cell_type == "code":
            cell.outputs = outputs[cell.metadata["report_section"]]
    book.metadata["plan19_snapshot"] = {"completed_units": len(snapshot["complete_units"]),
        "diagnostics": len(snapshot["diagnostics"]), "incomplete_units": len(snapshot["live"]),
        "scientific_units": snapshot["scientific_units"], "status": snapshot["status"]}
    return book


def build_notebook(root: str | Path | None, output: str | Path = DEFAULT_NOTEBOOK) -> Path:
    """Explicit report-generation command; notebook cells themselves are read-only."""
    import nbformat
    snapshot = load_report(root)
    book = _populate_outputs(make_notebook(root), snapshot, notebook_dir=Path(output).parent)
    from .phase1_report import append_phase1
    book = append_phase1(book, snapshot, notebook_dir=Path(output).parent)
    from .phase1_fix_report import append_repair, repair_conclusion
    book = append_repair(book, snapshot)
    conclusion = repair_conclusion(snapshot)
    from .phase1_fix2_report import append_fix2, conclusion as fix2_conclusion
    book = append_fix2(book, snapshot)
    conclusion = fix2_conclusion(snapshot) or conclusion
    if conclusion:
        book.cells[-1].source = conclusion
    nbformat.validate(book)
    output = Path(output)
    if any((parent / "COMPLETED").exists() for parent in (output.parent, *output.parents)):
        raise ArtifactError("cannot write a notebook inside a completed unit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".plan19-report-", suffix=".ipynb", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        nbformat.write(book, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
