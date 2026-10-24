"""Concise, artifact-only Plan 15 notebook.

Wire contract (v1): units/<id>/metadata.json contains phase, environment,
condition, replica and config. metrics.json is a list of scalar rows (or an
envelope with schema_version=1 and rows). Every row declares schema_version=1,
step, exposure, split and its available metrics. Identity fields may also
appear on rows and must agree with metadata. COMPLETED seals a unit.

An incomplete unit is visible ONLY via atomically published progress.json:
{"schema_version": 1, "sealed": true, "rows": [...], "metadata": {...}}.
Missing/absent partial snapshots are pending, not repaired. Corrupt completed
or explicitly sealed data raise ArtifactError. Only JSON metrics are read.

Historical rows need eviction_exposure to enter post-eviction summaries.
Distinct strengths use metadata treatment_id (or distinct condition strings).
Protocol/source revisions and phases never pool. row_adapter may translate a
runner's explicit schema before strict validation; unknown schemas never guess.

refresh(..., publish=callable) passes (Path, bytes) to a storage-reserved atomic
publisher. The callable owns reservation, replacement and accounting. Without
a publisher, refresh refuses publication; no unbudgeted write is permitted.
The CLI opens the existing Store. build_notebook reads/renders and publishes
nothing. The CLI uses the same publication path as the runner.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np

from .metrics import cohort_retention_auc, normalized_auc, paired_summary

REPORT_SCHEMA_VERSION = 1
MAX_JSON_BYTES = 32_000_000
MAX_NOTEBOOK_BYTES = 24_000_000
ENVIRONMENTS = ("rotation_linear", "rotation_sigmoid", "digit9_mixture")
ENVIRONMENT_LABELS = {"rotation_linear": "Linear rotation", "rotation_sigmoid": "Sigmoid rotation",
                      "digit9_mixture": "Digit-9 mixture"}
POPULATIONS = {"current": "New/current holdout", "historical": "Historical holdout",
               "evicted_train": "Evicted training cohort"}
SPLITS = {*POPULATIONS, "initial", "prequential", "dependence", "curvature", "calibration", "pr",
          "cohort_pre_eviction", "historical_pre_eviction"}
RowAdapter = Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]
Publisher = Callable[[Path, bytes], Any]


class ArtifactError(ValueError):
    """An advertised sealed artifact cannot support the requested report."""


def _read_json(path: Path, *, record: Mapping[str, Any] | None = None) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON value {value}")

    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ArtifactError(f"{path.name}: exceeds bounded JSON read size")
        if record is not None and (len(content) != record.get("bytes")
                                   or hashlib.sha256(content).hexdigest() != record.get("sha256")):
            raise ArtifactError(f"{path.name}: sealed JSON checksum mismatch")
        return json.loads(content, parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactError(f"{path.parent.name}/{path.name}: {error}") from error


def _phase_key(value: Any) -> tuple[int, str]:
    label = str(value)
    match = re.search(r"\d+", label)
    return (int(match.group()) if match else -1, label)


def order_notebook_chronologically(notebook: Any) -> Any:
    """Keep the original study first and later phase blocks in execution order."""
    groups = defaultdict(list)
    phase = (0, "")
    for cell in notebook.cells:
        tagged = next((match for tag in cell.metadata.get("tags", [])
                       if (match := re.fullmatch(r"plan15_phase(\d+)([a-z]*)", tag, re.I))), None)
        heading = (re.match(r"^## Phase\s+(\d+)([a-z]*)\b", cell.source, re.I)
                   if cell.cell_type == "markdown" else None)
        if tagged:
            key = (int(tagged[1]), tagged[2].upper())
        else:
            if heading:
                phase = (int(heading[1]), heading[2].upper())
            key = phase
        groups[key].append(cell)
    notebook.cells = [cell for key in sorted(groups) for cell in groups[key]]
    notebook.metadata["plan15_section_order"] = "oldest_to_newest"
    return notebook


def _version(value: Mapping[str, Any], where: str, *, required: bool = True) -> None:
    version = value.get("schema_version", 1 if not required else None)
    if isinstance(version, bool) or version != REPORT_SCHEMA_VERSION:
        raise ArtifactError(f"{where}: unsupported or missing schema_version {version!r}")


def _unit(unit_dir: Path, *, complete: bool, row_adapter: RowAdapter | None,
          manifest_metadata: dict[str, Any] | None = None,
          records: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if complete:
        metadata = manifest_metadata if manifest_metadata is not None else _read_json(unit_dir / "metadata.json")
        payload = _read_json(unit_dir / "metrics.json", record=None if records is None else records["metrics.json"])
    else:
        path = unit_dir / "progress.json"
        if not path.exists():
            return None
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("sealed") is not True:
            return None
        _version(payload, f"{unit_dir.name}/progress.json")
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = manifest_metadata if manifest_metadata is not None else _read_json(unit_dir / "metadata.json")
    if not isinstance(metadata, dict):
        raise ArtifactError(f"{unit_dir.name}: metadata must be an object")
    _version(metadata, f"{unit_dir.name}/metadata.json", required=False)
    for field in ("phase", "environment", "condition", "replica", "config"):
        if field not in metadata:
            raise ArtifactError(f"{unit_dir.name}: metadata missing {field}")
    if metadata["environment"] not in ENVIRONMENTS:
        raise ArtifactError(f"{unit_dir.name}: unknown environment")
    if not isinstance(metadata["config"], dict) or not isinstance(metadata["condition"], str):
        raise ArtifactError(f"{unit_dir.name}: invalid config or condition")
    if isinstance(metadata["replica"], bool) or not isinstance(metadata["replica"], int):
        raise ArtifactError(f"{unit_dir.name}: replica must be an integer")
    if isinstance(payload, dict):
        _version(payload, f"{unit_dir.name}/metrics envelope")
        payload = payload.get("rows")
    if not isinstance(payload, list):
        raise ArtifactError(f"{unit_dir.name}: metrics must be a row list")
    rows = []
    seen = set()
    for original in payload:
        if not isinstance(original, dict):
            raise ArtifactError(f"{unit_dir.name}: metric rows must be objects")
        row = row_adapter(dict(original), metadata) if row_adapter else dict(original)
        _version(row, f"{unit_dir.name}/row")
        for field in ("phase", "environment", "condition", "replica"):
            if field in row and row[field] != metadata[field]:
                raise ArtifactError(f"{unit_dir.name}: row/metadata {field} mismatch")
            row[field] = metadata[field]
        for field in ("step", "exposure"):
            if isinstance(row.get(field), bool) or not isinstance(row.get(field), (int, float)):
                raise ArtifactError(f"{unit_dir.name}: row missing numeric {field}")
            if not np.isfinite(row[field]) or row[field] < 0:
                raise ArtifactError(f"{unit_dir.name}: invalid {field}")
        if row.get("split") not in SPLITS:
            raise ArtifactError(f"{unit_dir.name}: unsupported metric split {row.get('split')!r}")
        for metric in ("nll", "accuracy", "nine_recall", "nine_fpr", "nine_precision",
                       "reliance_nll", "predictive_kl", "ltm_only_nll", "vulnerability_nll"):
            value = row.get(metric)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not np.isfinite(value)):
                raise ArtifactError(f"{unit_dir.name}: invalid scalar {metric}")
            if metric in ("accuracy", "nine_recall", "nine_fpr", "nine_precision") and value is not None:
                if not -1e-12 <= value <= 1 + 1e-12:
                    raise ArtifactError(f"{unit_dir.name}: {metric} must lie in [0, 1]")
                row[metric] = float(np.clip(value, 0, 1))
        key = tuple(str(row.get(field)) for field in
                    ("split", "population", "exposure", "cohort_id", "tau", "horizon", "threshold", "probe_id"))
        if key in seen:
            raise ArtifactError(f"{unit_dir.name}: duplicate metric identity {key}")
        seen.add(key)
        rows.append(row)
    if complete and not rows:
        raise ArtifactError(f"{unit_dir.name}: completed unit contains no metric rows")
    return {"id": unit_dir.name, "metadata": metadata, "rows": rows, "complete": complete}


def _sealed_json_records(unit_dir: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify the Store seal and consumed JSON without reading checkpoints."""
    def digest(value: Any) -> str:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        return hashlib.sha256(raw).hexdigest()

    marker = _read_json(unit_dir / "COMPLETED")
    if not isinstance(marker, dict) or marker.get("schema") != 1:
        raise ArtifactError(f"{unit_dir.name}: invalid completion seal")
    if marker.get("seal_hash") != digest({k: v for k, v in marker.items() if k != "seal_hash"}):
        raise ArtifactError(f"{unit_dir.name}: completion seal checksum mismatch")
    if (marker.get("unit_id") != unit_dir.name or marker.get("identity_hash") != manifest.get("identity_hash")
            or manifest.get("metadata_hash") != digest(manifest["metadata"])):
        raise ArtifactError(f"{unit_dir.name}: manifest/completion identity mismatch")
    records = marker.get("files")
    if not isinstance(records, dict) or not {"manifest.json", "metrics.json"}.issubset(records):
        raise ArtifactError(f"{unit_dir.name}: completion missing required JSON artifacts")
    observed = set()
    for path in unit_dir.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"{unit_dir.name}: symlinks are not scientific artifacts")
        if path.is_file() and path.name != "COMPLETED":
            observed.add(path.relative_to(unit_dir).as_posix())
    if observed != set(records):
        raise ArtifactError(f"{unit_dir.name}: completed artifact file set changed")
    _read_json(unit_dir / "manifest.json", record=records["manifest.json"])
    return records


def collect_artifacts(root: Path | str, *, row_adapter: RowAdapter | None = None) -> dict[str, Any]:
    """Read sealed metrics only; pending units are counted, never repaired."""
    root = Path(root)
    status = _read_json(root / "status.json") if (root / "status.json").exists() else {}
    if not isinstance(status, dict):
        raise ArtifactError("status.json must be an object")
    _version(status, "status.json", required=False)
    units, pending = [], 0
    for unit_dir in sorted((root / "units").glob("*")):
        if not unit_dir.is_dir():
            continue
        manifest_metadata = None
        records = None
        complete = (unit_dir / "COMPLETED").exists()
        if (unit_dir / "manifest.json").exists():
            manifest = _read_json(unit_dir / "manifest.json")
            if not isinstance(manifest, dict) or not isinstance(manifest.get("metadata"), dict):
                raise ArtifactError(f"{unit_dir.name}: invalid manifest metadata")
            manifest_metadata = manifest["metadata"]
            if manifest_metadata.get("kind", "trajectory") != "trajectory":
                continue
            if complete and manifest.get("schema") == 1:
                records = _sealed_json_records(unit_dir, manifest)
        unit = _unit(unit_dir, complete=complete, row_adapter=row_adapter,
                     manifest_metadata=manifest_metadata, records=records)
        if unit is None:
            pending += 1
        else:
            units.append(unit)
    units.sort(key=lambda unit: (_phase_key(unit["metadata"]["phase"]), unit["id"]))
    return {"status": status, "units": units, "pending": pending}


def _text(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "/").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join("| " + " | ".join(map(_text, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def _variant(unit: Mapping[str, Any]) -> str:
    meta = unit["metadata"]
    if "treatment_id" in meta:
        return str(meta["treatment_id"])
    strength = meta["config"].get("strength")
    return meta["condition"] + (f" ({strength:g})" if strength else "")


def _revision(unit: Mapping[str, Any]) -> str:
    meta = unit["metadata"]
    protocol = meta.get("comparison_id", meta.get("protocol_revision", meta.get("scientific_revision", meta["config"].get("protocol", "v1"))))
    return f"{protocol}/{meta.get('family', 'unspecified')}/{meta.get('source_hash', 'unspecified')}"


def _series_name(unit: Mapping[str, Any]) -> str:
    meta = unit["metadata"]
    name = {"replay_ewc": "Replay + EWC", "jvp": "JVP", "finite_difference": "Finite-difference KL",
            "directional": "Directional KL", "distillation": "Distillation",
            "replay_compute_matched": "Compute-matched replay"}.get(meta["condition"], meta["condition"].replace("_", " "))
    strength = meta["config"].get("strength", 0)
    return name + (f" (lambda={strength:g})" if strength else "")


def _display_labels(units: list[dict[str, Any]]) -> dict[str, str]:
    labels = {_variant(unit): _series_name(unit) for unit in units}
    counts = defaultdict(int)
    for name in labels.values():
        counts[name] += 1
    return {key: name + (f" [{key[-6:]}]" if counts[name] > 1 else "") for key, name in labels.items()}


def _progress_cell(data: Mapping[str, Any]) -> Any:
    status, units = data["status"], data["units"]
    progress = status.get("progress", {})
    if not isinstance(progress, dict):
        progress = {}
    completed = sum(unit["complete"] for unit in units)
    partial = sum(not unit["complete"] for unit in units)
    planned = completed if status.get("status") == "acceptance_complete" else progress.get("planned", status.get("planned_units", "pending"))
    rows = [["Execution status", status.get("status", "pending")],
            ["Phase / protocol", f"{status.get('phase', 'not started')} / {status.get('protocol_revision', 'pending')}"],
            ["Completed / planned units", f"{completed} / {planned}"],
            ["Sealed partial / pending units", f"{partial} / {data['pending']}"],
            ["Active compute (hours)", status.get("active_seconds", 0) / 3600],
            ["Report / validation / restart (seconds)",
             " / ".join(_text(status.get(k)) for k in ("report_seconds", "validation_seconds", "restart_seconds"))],
            ["Projected remaining (hours)", None if status.get("remaining_seconds") is None else status["remaining_seconds"] / 3600],
            ["Storage used / reserved / cap (GB)",
             " / ".join(_text(None if status.get(k) is None else status[k] / 1e9)
                        for k in ("used_bytes", "reserved_bytes", "max_bytes"))],
            ["Additional shared / sibling-run charge (GB)", status.get("external_bytes", 0) / 1e9],
            ["Forecast storage (GB)", None if status.get("forecast_bytes") is None else status["forecast_bytes"] / 1e9],
            ["Heartbeat", status.get("heartbeat", status.get("updated_at"))]]
    phases = defaultdict(lambda: [0, 0])
    for unit in units:
        key = (str(unit["metadata"]["phase"]), unit["metadata"]["environment"])
        phases[key][0 if unit["complete"] else 1] += 1
    phase_rows = [[phase, ENVIRONMENT_LABELS[env], *counts] for (phase, env), counts in phases.items()]
    smoke = any(u["metadata"].get("smoke") for u in units) or "smoke" in str(status.get("note", "")).lower()
    banner = ("**SMOKE ACCEPTANCE ONLY.** Tiny acceptance trajectories validate software and recovery, "
              "not learning effectiveness. No scientific conclusion is supported.\n\n") if smoke else ""
    source = ("# Plan 15: Fisher-Regularized Memory\n\n" + banner +
              "Continual learning with scarce observations and compute: acquire new information "
              "while retaining predictions after replay eviction.\n\n## Execution progress\n\n"
              + _table(["Measure", "Status"], rows))
    if phase_rows:
        source += "\n\n" + _table(["Phase", "Environment", "Complete", "Sealed partial"], phase_rows)
    if status.get("failure"):
        source += "\n\n**Execution issue:** " + _text(status["failure"])
    elif status.get("note") and not smoke:
        source += "\n\n" + _text(status["note"])
    source += ("\n\n**Artifact-only report.** Partial trajectories are descriptive, not completed "
               "confirmation. No training, inference, checkpoint loading, or artifact repair occurs here.")
    if not units:
        source += "\n\n**No sealed results yet.** Empty or pending phases are not negative findings."
    return nbformat.v4.new_markdown_cell(source)


def _changes_cell(status: Mapping[str, Any]) -> Any | None:
    changes = status.get("changes", [])
    if not changes:
        return None
    if not isinstance(changes, list) or not all(isinstance(c, dict) for c in changes):
        raise ArtifactError("status changes must be a list of objects")
    rows = [[c.get("date"), c.get("revision"), c.get("problem", c.get("summary")),
             c.get("change"), c.get("affected_results", c.get("validity")),
             c.get("validation"), c.get("next", c.get("record"))] for c in changes[-6:]]
    return nbformat.v4.new_markdown_cell(
        "## Significant changes\n\n" + _table(
            ["Date", "Revision", "Problem", "Correction", "Affected evidence", "Validation", "Next / record"], rows)
        + f"\n\nShowing the latest {len(rows)} of {len(changes)} changes; the complete record remains in `status.json`."
    )


def _plot_cell(figure: Any, caption: str) -> Any:
    from matplotlib import pyplot as plt

    buffer = io.BytesIO()
    figure.canvas.draw()
    extras = [*figure.legends, *[axis.xaxis.label for axis in figure.axes],
              *[axis.yaxis.label for axis in figure.axes]]
    if figure._suptitle is not None:
        extras.append(figure._suptitle)
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight", bbox_extra_artists=extras)
    plt.close(figure)
    return nbformat.v4.new_code_cell(
        "# Rendered from sealed scalar metrics only; this cell performs no computation.",
        outputs=[nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text/plain": caption,
        })], execution_count=None,
    )


def _mean_curve(units: list[dict[str, Any]], split: str, metric: str, *, xfield: str = "exposure",
                tau: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_replica = defaultdict(lambda: defaultdict(list))
    for unit in units:
        replica = unit["metadata"]["replica"]
        for row in unit["rows"]:
            if row["split"] != split or row.get(metric) is None or row.get(xfield) is None:
                continue
            if tau is not None and row.get("tau") != tau:
                continue
            if split in ("historical", "evicted_train"):
                if row.get("eviction_exposure") is None or row["exposure"] < row["eviction_exposure"]:
                    continue
            by_replica[replica][row[xfield]].append(float(row[metric]))
    points = defaultdict(list)
    for values in by_replica.values():
        for x, y in values.items():
            points[x].append(float(np.mean(y)))
    x = np.array(sorted(points))
    mean = np.array([np.mean(points[p]) for p in x])
    sem = np.array([np.std(points[p], ddof=1) / np.sqrt(len(points[p])) if len(points[p]) > 1 else np.nan for p in x])
    counts = np.array([len(points[p]) for p in x])
    return x, mean, sem, counts


def _draw(axis: Any, units: list[dict[str, Any]], split: str, metric: str, *, xfield: str = "exposure",
          percent: bool = False, tau: float | None = None) -> None:
    groups = defaultdict(list)
    labels = _display_labels(units)
    for unit in units:
        groups[_variant(unit)].append(unit)
    factor = 100 if percent else 1
    for group in groups.values():
        x, mean, sem, _ = _mean_curve(group, split, metric, xfield=xfield, tau=tau)
        if not len(x):
            continue
        line, = axis.plot(x, factor * mean, label=labels[_variant(group[0])], linewidth=1.7)
        axis.fill_between(x, factor * (mean - sem), factor * (mean + sem), color=line.get_color(), alpha=.15)
    axis.grid(alpha=.18)
    if not axis.lines:
        axis.text(.5, .5, "Not yet observed", ha="center", va="center", transform=axis.transAxes)
    if percent:
        axis.set_ylim(0, 100)


def _legend(figure: Any, axes: Any) -> None:
    entries = {}
    for axis in np.asarray(axes).ravel():
        handles, labels = axis.get_legend_handles_labels()
        entries.update(zip(labels, handles))
    if entries:
        figure.legend(entries.values(), entries.keys(), loc="outside lower center", ncol=min(3, len(entries)),
                      fontsize=8, frameon=False)


def _draw_frozen_nll(axis: Any, units: list[dict[str, Any]]) -> None:
    controls = [unit for unit in units if unit["metadata"]["condition"] == "replay_ewc"]
    x, mean, sem, _ = _mean_curve(controls, "current", "frozen_nll")
    if len(x):
        axis.plot(x, mean, color=".3", linestyle="--", linewidth=1.5, label="Frozen initial model")
        axis.fill_between(x, mean - sem, mean + sem, color=".3", alpha=.1)


def _predictive_figure(units: list[dict[str, Any]], environment: str, phase: str) -> Any:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col", sharey="row", layout="constrained")
    for col, (split, title) in enumerate(POPULATIONS.items()):
        xfield = "exposure" if split == "current" else "cohort_age"
        _draw(axes[0, col], units, split, "nll", xfield=xfield)
        _draw(axes[1, col], units, split, "accuracy", xfield=xfield, percent=True)
        axes[0, col].set_title(title, fontsize=10)
        axes[1, col].set_xlabel("Unique digits integrated" if split == "current" else "Digits since cohort eviction")
    _draw_frozen_nll(axes[0, 0], units)
    axes[0, 0].set_ylabel("NLL (lower is better)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    smoke = " (acceptance only)" if any(u["metadata"].get("smoke") for u in units) else ""
    figure.suptitle(f"{ENVIRONMENT_LABELS[environment]} / Phase {phase.removeprefix('phase')}{smoke}")
    _legend(figure, axes)
    return _plot_cell(figure, "Means and +/-1 replica SEM; equal available-cohort means within replica. Dashed NLL: the unadapted initial model on the current holdout. A gain over replay/EWC can still fall short of this frozen reference. Right censoring changes contributing counts with age.")


def _selected_groups(units: list[dict[str, Any]], status: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """One latest phase/revision per environment; never pool inspected pilots."""
    chosen, notes = [], []
    for environment in ENVIRONMENTS:
        candidates = [u for u in units if u["metadata"]["environment"] == environment]
        if not candidates:
            continue
        latest_phase = max((u["metadata"]["phase"] for u in candidates), key=_phase_key)
        candidates = [u for u in candidates if u["metadata"]["phase"] == latest_phase]
        revisions = sorted({_revision(u) for u in candidates})
        if len(revisions) > 1:
            notes.append(f"{ENVIRONMENT_LABELS[environment]} has multiple revisions; showing {revisions[-1]} only.")
            candidates = [u for u in candidates if _revision(u) == revisions[-1]]
        identities = set()
        for unit in candidates:
            key = (_variant(unit), unit["metadata"]["replica"])
            if key in identities:
                raise ArtifactError("duplicate phase/environment/treatment/replica; provide distinct treatment_id or comparison_id")
            identities.add(key)
        screening = _phase_key(latest_phase)[0] == 1 or any("screen" in u["metadata"].get("family", "") for u in candidates)
        if screening:
            groups = defaultdict(dict)
            for unit in candidates:
                groups[unit["metadata"]["condition"]][_variant(unit)] = unit
            keep = set()
            policy = (status or {}).get("common_policy", {}).get(environment)
            for condition, variants in groups.items():
                preferred = [key for key, unit in variants.items()
                             if condition == "replay_ewc" and policy is not None and unit["metadata"]["config"] == policy]
                strengths = sorted({float(unit["metadata"]["config"].get("strength", 0)) for unit in variants.values()})
                middle = strengths[(len(strengths) - 1) // 2]
                eligible = sorted(key for key, unit in variants.items() if float(unit["metadata"]["config"].get("strength", 0)) == middle)
                keep.add(sorted(preferred)[0] if preferred else eligible[0])
            if len(keep) < len({_variant(u) for u in candidates}):
                notes.append(f"{ENVIRONMENT_LABELS[environment]}: one display representative per method; middle tested strength "
                             "(lower middle for an even grid), NOT the best outcome. Published common baseline is used when available; "
                             "otherwise equal-strength ties use the stable treatment ID. Full grids remain in unit artifacts.")
            candidates = [u for u in candidates if _variant(u) in keep]
        chosen.extend(candidates)
    return chosen, notes


def _auc_for_unit(unit: Mapping[str, Any], split: str) -> tuple[float | None, Any]:
    rows = [r for r in unit["rows"] if r["split"] == split and r.get("nll") is not None]
    if split == "current":
        rows.sort(key=lambda r: r["exposure"])
        return normalized_auc([r["exposure"] for r in rows], [r["nll"] for r in rows]), tuple(r["exposure"] for r in rows)
    groups = defaultdict(list)
    for row in rows:
        if row.get("cohort_id") is not None:
            groups[str(row["cohort_id"])].append(row)
    current = [r["exposure"] for r in unit["rows"] if r["split"] == "current"]
    capacity = unit["metadata"].get("replay_capacity", unit["metadata"]["config"].get("replay_capacity"))
    if capacity is None:
        capacities = {r["capacity"] for r in rows if r.get("capacity") is not None}
        capacity = next(iter(capacities)) if len(capacities) == 1 else None
    if not current or not capacity or not groups:
        return None, None
    values, windows = [], []
    for cohort, cohort_rows in sorted(groups.items()):
        cohort_rows.sort(key=lambda r: r["exposure"])
        evictions = {r["eviction_exposure"] for r in cohort_rows if r.get("eviction_exposure") is not None}
        if len(evictions) != 1:
            return None, None
        summary = cohort_retention_auc(
            [r["exposure"] for r in cohort_rows], [r["nll"] for r in cohort_rows],
            eviction_exposure=next(iter(evictions)),
            end_exposure=unit["metadata"].get("end_exposure", max(current)), capacity=capacity)
        if not summary["complete_for_declared_horizon"]:
            return None, None
        values.append(summary["auc"])
        windows.append((cohort, summary["eviction_exposure"], summary["declared_horizon"],
                        tuple(r["exposure"] for r in cohort_rows)))
    return float(np.mean(values)), tuple(windows)


def _conclusion(selected: list[dict[str, Any]], status: Mapping[str, Any]) -> Any:
    if any(u["metadata"].get("smoke") for u in selected) or "smoke" in str(status.get("note", "")).lower():
        return nbformat.v4.new_markdown_cell(
            "## Conclusion\n\n**Acceptance evidence only; no scientific result.** "
            "These tiny runs exercise metrics, memory lifecycle, restart and reporting. "
            "Neither their predictive scores nor apparent treatment differences establish acquisition, retention, "
            "consolidation or statistical efficiency. Production calibration and independent paired cohorts remain necessary."
        )
    rows, readout, cost_matches = [], [], []
    labels = _display_labels(selected)
    release = status.get("confirmation_complete") is True
    for environment in ENVIRONMENTS:
        units = [u for u in selected if u["metadata"]["environment"] == environment and u["complete"]]
        controls = {u["metadata"]["replica"]: u for u in units if u["metadata"]["condition"] == "replay_ewc"}
        treatments = defaultdict(dict)
        for unit in units:
            if unit["metadata"]["condition"] != "replay_ewc":
                treatments[_variant(unit)][unit["metadata"]["replica"]] = unit
        for condition, replicas in treatments.items():
            endpoints = {}
            for split in ("current", "historical"):
                a, b = {}, {}
                for replica in controls.keys() & replicas.keys():
                    ca, wa = _auc_for_unit(controls[replica], split)
                    cb, wb = _auc_for_unit(replicas[replica], split)
                    if wa is not None and wa == wb and ca is not None and cb is not None:
                        a[replica], b[replica] = ca, cb
                summary = paired_summary(a, b)
                endpoints[split] = summary
                interval = summary["ci"] if release else None
                interval_text = (f"[{interval[0]:.4g}, {interval[1]:.4g}]" if interval else "pending fixed-cohort completion")
                rows.append([ENVIRONMENT_LABELS[environment], labels[condition], POPULATIONS[split], summary["n_pairs"],
                             summary["mean_gain"], summary["median_gain"], summary["favorable_replicas"], interval_text])
            if not release or next(iter(replicas.values()))["metadata"]["condition"] not in (
                    "jvp", "finite_difference", "directional"):
                continue
            supported = all(value["ci"] is not None and value["ci"][0] > 0 for value in endpoints.values())
            result = ("nominal 95% paired intervals favor the treatment on both NLL endpoints"
                      if supported else "the paired intervals do not establish improvement on both NLL endpoints")
            details = []
            frozen, adapted = [], []
            for replica in controls.keys() & replicas.keys():
                current = sorted((r for r in controls[replica]["rows"] if r["split"] == "current"),
                                 key=lambda r: r["exposure"])
                if len(current) < 2 or any(r.get("frozen_nll") is None for r in current):
                    continue
                value, _ = _auc_for_unit(replicas[replica], "current")
                if value is not None:
                    frozen.append(normalized_auc([r["exposure"] for r in current], [r["frozen_nll"] for r in current]))
                    adapted.append(value)
            if frozen:
                details.append(f"mean current NLL AUC {_text(float(np.mean(adapted)))} versus "
                               f"{_text(float(np.mean(frozen)))} for the frozen initial model (descriptive)")
            def online_seconds(units):
                values = [max(times) for unit in units if
                          (times := [r["cumulative_active_seconds"] for r in unit["rows"]
                                     if r.get("cumulative_active_seconds") is not None])]
                return float(np.mean(values)) if values else None
            treated_time, control_time = online_seconds(replicas.values()), online_seconds(controls.values())
            if treated_time and control_time:
                details.append(f"online time {treated_time / control_time:.2f}x replay/EWC")
            matched_time = online_seconds(u for u in units if u["metadata"]["condition"] == "replay_compute_matched")
            if treated_time and matched_time:
                cost_matches.append(matched_time / treated_time)
            readout.append(f"- **{ENVIRONMENT_LABELS[environment]}:** {result}."
                           + (" " + "; ".join(details) + "." if details else ""))
    source = "## Conclusion\n\n"
    if readout:
        source += "\n".join(readout) + "\n\n"
    if rows:
        source += _table(["Environment", "Treatment", "Endpoint", "Pairs", "Mean gain", "Median", "Favorable", "95% paired interval"], rows)
        source += "\n\nGain is control minus treatment NLL AUC; positive favors treatment. Only matched complete trajectories and follow-up grids enter."
    else:
        source += "No complete paired acquisition/retention comparison is available yet."
    source += ("\n\nNo best balance is declared automatically. Acquisition, evicted-data prediction, "
               "functional dependence and actual compute cost must be considered together. "
               "Small-replica intervals are fragile; a nonsignificant retention difference is not noninferiority. "
               "Development and confirmation are never pooled.")
    conclusion = status.get("conclusion")
    if readout:
        source += ("\n\n**Mechanism remains unresolved.** These predictive effects do not identify STM-to-LTM transfer; "
                   "ordinary regularization or capacity suppression remain alternatives. Negative STM-removal harm "
                   "means muting STM helps, not negative knowledge. Pre-eviction snapshots are after rehearsal.")
        if cost_matches:
            source += (f" The calibrated larger-ceiling replay control used {min(cost_matches):.0%}--"
                       f"{max(cost_matches):.0%} of the selected method's actual online time across environments; "
                       "it is an allowance control, not a realized equal-time comparison.")
        if isinstance(conclusion, str) and "under-solving" in conclusion:
            source += (" Persistent under-solving warnings limit interpretation to the configured bounded solver, "
                       "not exact objective minimization.")
    elif isinstance(conclusion, str):
        source += "\n\n" + conclusion
    return nbformat.v4.new_markdown_cell(source)


def _extension_cell(extension_root: Path, primary_root: Path, selected: list[dict[str, Any]]) -> Any:
    from .extension_report import collect

    data = collect(extension_root)
    ledger, status = data["ledger"], data["status"]
    if Path(ledger["source_root"]).resolve() != primary_root.resolve():
        raise ArtifactError("extension ledger references a different Plan 15 source root")
    complete = len(data["observed"]) == len(ledger["jobs"])
    if status.get("status") == "completed" and not complete:
        raise ArtifactError("extension reports completion without all planned trajectories")
    fd = {}
    for unit in selected:
        meta = unit["metadata"]
        if (unit["complete"] and meta.get("family") == "confirmation"
                and meta.get("phase") == "phase3" and meta.get("condition") == "finite_difference"):
            fd[(meta["environment"], meta["replica"])] = unit

    rows, comparisons = [], []
    for environment in ledger["scope"]["environments"]:
        for method in ledger["scope"]["methods"]:
            jobs = [(job, data["observed"].get(unit_id)) for unit_id, job in data["expected"].items()
                    if job["environment"] == environment and job["method"] == method]
            observed = [(job, summary) for job, summary in jobs if summary is not None]
            summaries = {}
            for endpoint in ("current_auc", "historical_auc"):
                control = {job["replica"]: job["control"].get(endpoint) for job, _ in observed}
                treatment = {job["replica"]: summary.get(endpoint) for job, summary in observed}
                summaries[endpoint] = paired_summary(control, treatment)

            def gain_text(endpoint):
                result = summaries[endpoint]
                if not result["n_pairs"]:
                    return "unavailable"
                if not complete or result["n_pairs"] != ledger["scope"]["replicas"]:
                    return f"{result['mean_gain']:+.4f} (descriptive)"
                low, high = result["ci"]
                return f"{result['mean_gain']:+.4f} [{low:+.4f}, {high:+.4f}]"

            current, historical = summaries["current_auc"], summaries["historical_auc"]
            favorable = (f"{current['favorable_replicas']}/{current['n_pairs']}; "
                         f"{historical['favorable_replicas']}/{historical['n_pairs']}")
            seconds = float(np.mean([summary["active_seconds"] for _, summary in observed])) if observed else None
            rows.append([ENVIRONMENT_LABELS[environment], "JVP" if method == "jvp" else "Directional KL",
                         f"{len(observed)}/{len(jobs)}", gain_text("current_auc"),
                         gain_text("historical_auc"), favorable, f"{seconds:.0f}" if seconds else "-"])
            if not complete or len(observed) != len(jobs):
                continue
            fd_rows = {}
            for endpoint, split in (("current_auc", "current"), ("historical_auc", "historical")):
                reference = {}
                for job, _ in observed:
                    replica = job["replica"]
                    if (environment, replica) in fd:
                        reference[replica], _ = _auc_for_unit(fd[(environment, replica)], split)
                treatment = {job["replica"]: summary.get(endpoint) for job, summary in observed}
                fd_rows[endpoint] = paired_summary(reference, treatment)
            if all(result["n_pairs"] == len(jobs) for result in fd_rows.values()):
                fd_seconds = float(np.mean([
                    max(row["cumulative_active_seconds"] for row in fd[(environment, job["replica"])]["rows"]
                        if row.get("cumulative_active_seconds") is not None)
                    for job, _ in observed]))
                comparisons.append((environment, method, fd_rows, seconds / fd_seconds))
    source = ("## Phase 5: Alternative STM penalties\n\n"
              f"**Progress:** {len(data['observed'])}/{len(ledger['jobs'])} sealed trajectories; "
              f"status **{status.get('status', 'unknown')}**. "
              "Gain is replay/EWC minus method NLL AUC; positive is better. "
              "The two favorable endpoints are current and post-eviction holdout prediction, respectively.\n\n"
              + _table(["Environment", "Method", "Pairs", "Current gain [95%]", "Historical gain [95%]",
                        "Favorable c; h", "Online s"], rows))
    if comparisons:
        directional_rotations = [entry for entry in comparisons
                                 if entry[1] == "directional" and entry[0] in ("rotation_linear", "rotation_sigmoid")]
        if len(directional_rotations) == 2 and all(
                abs(entry[2][endpoint]["mean_gain"]) < .02 and entry[3] < 1
                for entry in directional_rotations for endpoint in ("current_auc", "historical_auc")):
            source += ("\n\n**Practical readout:** directional KL has rotation means close to finite-difference KL "
                       "with lower measured online time. This is a promising compute tradeoff, not a proof "
                       "of equivalent efficacy. JVP is slower and its sigmoid gain is substantially smaller.")
        source += "\n\n**Compared with selected finite-difference KL:** "
        fragments = []
        for environment, method, endpoints, ratio in comparisons:
            if environment == "digit9_mixture":
                continue
            c = endpoints["current_auc"]["mean_gain"]
            h = endpoints["historical_auc"]["mean_gain"]
            fragments.append(f"{ENVIRONMENT_LABELS[environment]} {('JVP' if method == 'jvp' else 'directional KL')} "
                             f"has FD-minus-method gains {c:+.4f} current and {h:+.4f} historical "
                             f"at {ratio:.2f}x FD's online time")
        source += "; ".join(fragments) + ". Positive FD-minus-method gains favor the alternative."
    source += ("\n\nThese are nominal paired-bootstrap intervals from a post-selection extension, "
               "not additional prespecified confirmation claims or evidence of STM-to-LTM transfer. "
               "Rotation optimizers remained frequently under-solved; the digit-9 mixture showed no clear "
               "positive lift. The separate extension notebook contains the per-replica plot.")
    return nbformat.v4.new_markdown_cell(source)


def _pr_records(unit: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Accept nested curves and the runner's explicit scalar threshold rows."""
    records = [dict(row) for row in unit["rows"] if row["split"] == split and row.get("pr_curve")]
    groups = defaultdict(list)
    for row in unit["rows"]:
        if row["split"] == "pr" and row.get("population", "current") == split:
            groups[(row["exposure"], row.get("cohort_id"))].append(row)
    for points in groups.values():
        points.sort(key=lambda r: -float("inf") if r.get("threshold") is None else -r["threshold"])
        if any(r.get("nine_recall") is None or r.get("nine_precision") is None for r in points):
            continue
        row = dict(points[0])
        row["pr_curve"] = {"recall": [r["nine_recall"] for r in points],
                           "precision": [r["nine_precision"] for r in points],
                           "average_precision": row.get("average_precision"),
                           "p_ref": row.get("p_ref", .1)}
        records.append(row)
    return records


def _efficiency_cell(selected: list[dict[str, Any]]) -> Any | None:
    groups = defaultdict(list)
    labels = _display_labels(selected)
    for unit in selected:
        current = [r for r in unit["rows"] if r["split"] == "current"]
        times = [r["cumulative_active_seconds"] for r in unit["rows"] if r.get("cumulative_active_seconds") is not None]
        if not current or not times:
            continue
        latest = max(current, key=lambda r: r["exposure"])
        prequential = [r for r in unit["rows"] if r["split"] == "prequential" and r.get("nll") is not None]
        initial = [r for r in unit["rows"] if r["split"] == "initial" and r.get("nll") is not None]
        mass = sum(r.get("n", 1) for r in prequential)
        incoming = sum(r["nll"] * r.get("n", 1) for r in prequential) / mass if mass else None
        groups[(unit["metadata"]["environment"], _variant(unit))].append(
            [latest["exposure"], max(times), incoming,
             max(initial, key=lambda r: r["exposure"])["nll"] if initial else None])
    if not groups:
        return None
    rows = []
    for (environment, variant), values in groups.items():
        means = []
        for column in zip(*values):
            observed = [value for value in column if value is not None]
            means.append(float(np.mean(observed)) if observed else None)
        rows.append([ENVIRONMENT_LABELS[environment], labels[variant], len(values), *means])
    return nbformat.v4.new_markdown_cell(
        "### Earlier study: sample and compute audit\n\n"
        "These are the original rotation and digit-9 conditions. Prequential NLL averages "
        "prediction loss on incoming observations before their updates; lower is better. "
        "It does not estimate how many samples a method saves.\n\n"
        "<details>\n<summary>Original study measurements</summary>\n\n" + _table(
            ["Environment", "Treatment", "Replicas", "Digits integrated", "Online seconds", "Prequential NLL", "Initial-task NLL"], rows)
        + "\n\nLatest observed endpoints; partial runs remain descriptive. Incoming NLL is weighted by arrival size "
        "within replica, then replicas are weighted equally. Initialization cost is separate from online compute."
        "\n\n</details>"
    )


def _phase5c_efficiency_plot(units: list[dict], result: dict) -> Any:
    """Visualize the frozen acquisition estimand and observed cost separately."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"replay_ewc": "#a43b68", "finite_difference": "#147d73"}
    labels = {"replay_ewc": "Replay/EWC", "finite_difference": "FD KL"}
    mixed = [unit for unit in units if unit["completed"] and unit["metadata"]["stream"] == "mixed"]
    target = result["event_target"]
    figure, grid = plt.subplots(2, 2, figsize=(11.4, 6.8))
    learning, acquisition, effect_axis, compute = grid.flat
    curves = {}
    for condition in colors:
        cohort = [unit for unit in mixed if unit["metadata"]["condition"] == condition]
        curves[condition] = np.mean([[row["rotated_nll"] for row in unit["rows"]]
                                     for unit in cohort], axis=0)
        learning.plot(range(target + 1), curves[condition], color=colors[condition],
                      linewidth=2, label=labels[condition])
        # The registered score excludes the pre-training (n=0) checkpoint.
        running = np.cumsum(curves[condition][1:]) / np.arange(1, target + 1)
        acquisition.plot(range(1, target + 1), running, color=colors[condition], linewidth=2,
                         label=f"{labels[condition]}: " + rf"$A^R({target})={running[-1]:.3f}$")
    learning.fill_between(range(1, target + 1), curves["finite_difference"][1:],
                          curves["replay_ewc"][1:], color=colors["finite_difference"], alpha=.10)
    learning.set(title="Held-out NLL after batch updates", xlabel="New rotated images (batch checkpoints)",
                 ylabel=r"Test NLL $R_m^R(n)$ (lower is better)", xlim=(0, target))
    acquisition.set(title="Sample-efficiency score (lower is better)", xlabel="New rotated images observed",
                    ylabel=r"$A_m^R(n)$: mean NLL through image $n$", xlim=(1, target))
    for axis in (learning, acquisition):
        axis.axhline(curves["replay_ewc"][0], color="#666666", linestyle="--", linewidth=1,
                     label="Frozen initial model")
        axis.legend(frameon=False, fontsize=8)

    effect = result["primary"]
    gains = [row["gain"] for row in effect["paired_values"]]
    low, high = effect["ci"]
    effect_axis.axvline(0, color="#666666", linewidth=1, linestyle="--")
    effect_axis.scatter(gains, np.linspace(-.16, .16, len(gains)), s=23,
                        color=colors["finite_difference"], alpha=.55)
    effect_axis.hlines(.64, low, high, color=colors["finite_difference"], linewidth=3)
    effect_axis.scatter([effect["mean_gain"]], [.64], s=65, color=colors["finite_difference"], zorder=3)
    effect_axis.text(.5, .97, f"{effect['mean_gain']:+.3f}  [{low:+.3f}, {high:+.3f}]",
                     transform=effect_axis.transAxes, ha="center", va="top", fontsize=10)
    span = max(.1, *[abs(value) for value in [*gains, low, high]])
    effect_axis.set(title=rf"Paired gain in $A^R({target})$ (positive favors FD)",
                    xlabel=rf"$A^R_{{\mathrm{{Replay}}}}({target})-A^R_{{\mathrm{{FD}}}}({target})$",
                    xlim=(min(0, min(gains), low) - .09 * span, max(0, max(gains), high) + .12 * span),
                    ylim=(-.35, 1.15), yticks=[0, .64],
                    yticklabels=[f"{result['n_pairs']} replicas", "Mean / 95% CI"])

    costs = {row["condition"]: row["active_seconds"] / 60 for row in result["arm_summaries"]
             if row["stream"] == "mixed"}
    for position, condition in enumerate(colors):
        compute.barh(position, costs[condition], color=colors[condition], height=.48)
        compute.text(costs[condition] + .12, position, f"{costs[condition]:.2f}", va="center", fontsize=10)
    ratio = costs["finite_difference"] / costs["replay_ewc"]
    compute.set(title=f"Compute cost: FD uses {ratio:.2f}x time",
                xlabel="Mean online minutes per trajectory", yticks=[0, 1],
                yticklabels=[labels[key] for key in colors], xlim=(0, max(costs.values()) * 1.23))
    compute.invert_yaxis()
    for axis in grid.flat:
        axis.grid(axis="y" if axis in (learning, acquisition) else "x", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(w_pad=2, h_pad=2)
    return _plot_cell(figure, "Phase 5C: fixed rotated test-panel NLL after batch updates, running acquisition score, paired score gain, and compute cost. Counts crossed in one batch share its evaluation.")


def _phase5c_evaluation_cell(result: dict, initial_nll: float) -> Any:
    target, replicas = result["event_target"], result["n_pairs"]
    fd = next(row for row in result["arm_summaries"]
              if row["stream"] == "mixed" and row["condition"] == "finite_difference")
    return nbformat.v4.new_markdown_cell(
        "### Evaluation protocol and comparable NLLs\n\n"
        "Each replica uses one **fixed, class-stratified panel from the official MNIST test split**. "
        "For new-domain evaluation, every panel image is rotated by 30 degrees. The frozen reference "
        "and all four adaptive arms use **identical test images, labels, rotations, and class weights** "
        "within that replica at every checkpoint. These test examples never enter pretraining, online "
        "updates, replay, Fisher estimation, or curvature penalties. Phase 5C test IDs are disjoint "
        "from the Phase 5B development panels used to select the coefficient; that coefficient was "
        "sealed before confirmation began. Panels may overlap between replicas.\n\n"
        "**Timing:** finish optimization on an incoming batch of up to four training images plus "
        "the resident replay examples, then measure loss on the separate test panel. If a batch "
        "contains two new rotated images, both event counts receive that same post-update test loss. "
        "The test loss is not loss on the incoming batch or replay buffer.\n\n"
        "<details>\n<summary>NLL equations, weighting, and frozen-reference construction</summary>\n\n"
        "Let $H_{r,c}$ contain replica $r$'s fixed test-image IDs of digit $c$, "
        "$T_{30^\\circ}$ be the fixed rotation, and $q_c$ the digit's frequency in the full MNIST "
        "training split. Class weights stay fixed; they do not track the stream's rotated fraction. "
        "The common predictive loss is\n\n"
        "$$\n"
        r"\ell_r^R(\theta)=-\sum_{c=0}^{9}\frac{q_c}{|H_{r,c}|}"
        r"\sum_{i\in H_{r,c}}\log p_\theta(c\mid T_{30^\circ}x_i)."
        "\n$$\n\n"
        "This is mean per-image NLL in nats. It **excludes EWC, Ridge, and KL penalties**. "
        "Let $b_r(k)$ be the batch containing the $k$th rotated training image, "
        "$\\theta_{r,m,b}$ the parameters after optimizing that whole batch, and "
        "$\\theta_{r,0}$ the shared pretrained parameters before any online batches. Then\n\n"
        "$$\n"
        r"\begin{aligned}"
        r"F_r^R&=\ell_r^R(\theta_{r,0}),\\"
        r"E_{r,m}^R&=\ell_r^R(\theta_{r,m,b_r(N)}),\\"
        r"A_{r,m}^R(N)&=\frac{1}{N}\sum_{k=1}^{N}\ell_r^R(\theta_{r,m,b_r(k)})."
        r"\end{aligned}"
        "\n$$\n\n"
        f"Here $N={target}$ and $m$ identifies the adaptive method. The three reported numbers "
        f"are equal-weight means over {replicas} replicas: "
        f"$\\overline{{F^R}}={initial_nll:.4f}$, "
        f"$\\overline{{A_{{\\mathrm{{FD}}}}^R}}={fd['rotated_auc']:.4f}$, and "
        f"$\\overline{{E_{{\\mathrm{{FD}}}}^R}}={fd['final_rotated_nll']:.4f}$. "
        "The frozen reference is measured at checkpoint zero on this same rotated test panel, "
        "not on its unrotated pretraining data. Holding that deterministic model fixed gives "
        "$R_{r,\\mathrm{frozen}}^R(k)=F_r^R$ at every count, so its acquisition score and final "
        "NLL both equal $F_r^R$; it is not a separately fitted control. The adaptive endpoint "
        "and acquisition score use the same predictive loss but different time aggregation. "
        "Repeated evaluations are not independent observations; uncertainty is summarized across replicas."
        "\n\n</details>"
    )


def build_phase5c_presentation(base_notebook: Path | str, root: Path | str) -> Any:
    """Reformat completed confirmation without changing frozen analysis or sources."""
    from .drip_confirmation_report import TAG, analyze, collect

    data = collect(root)
    result = analyze(data["units"], data["ledger"])
    if not result["complete"] or not result["inference_ready"]:
        raise ArtifactError("visual confirmation summary requires the complete production cohort")
    sealed = _read_json(Path(root) / "units" / result["result_id"] / "result.json")
    if result != sealed:
        raise ArtifactError("confirmation presentation disagrees with sealed analysis")
    notebook = nbformat.read(base_notebook, as_version=4)
    old = [cell for cell in notebook.cells if TAG in cell.metadata.get("tags", [])]
    if not old:
        raise ArtifactError("existing Phase 5C notebook section required for presentation refresh")
    accuracy = next((cell for cell in old if cell.cell_type == "code" and any(
        "Phase 5C accuracy:" in output.get("data", {}).get("text/plain", "")
        for output in cell.get("outputs", []))), None)
    if accuracy is None:
        raise ArtifactError("completed Phase 5C accuracy figure missing")
    details = [cell.source for cell in old if cell.cell_type == "markdown" and
               cell.source.startswith(("### Paired effects", "### Prediction and compute audit"))]
    if not details:
        details = [cell.source for cell in old if cell.cell_type == "markdown"
                   and cell.source.startswith("<details>\n<summary>Secondary comparisons")]
    target, effect = result["event_target"], result["primary"]
    arms = {row["condition"]: row for row in result["arm_summaries"] if row["stream"] == "mixed"}
    ratio = arms["finite_difference"]["active_seconds"] / arms["replay_ewc"]["active_seconds"]
    relative_reduction = 100 * effect["mean_gain"] / arms["replay_ewc"]["rotated_auc"]
    initial_nll = float(np.mean([unit["rows"][0]["rotated_nll"] for unit in data["units"]
                                if unit["completed"] and unit["metadata"]["stream"] == "mixed"
                                and unit["metadata"]["condition"] == "replay_ewc"]))
    progress = (
        "## Phase 5C: Sample and compute efficiency\n\n"
        f"**Complete: {result['completed']}/{result['expected']} trajectories; {result['n_pairs']} paired replicas.** "
        f"Each mixed-stream trajectory receives {target} unique rotated images, about ten per digit, "
        "with a 32-image replay buffer.\n\n"
        "**Statistical efficiency here means prediction quality while new labeled images are scarce.** "
        "The prespecified acquisition score is the normalized area under the sample-indexed learning curve:\n\n"
        "$$\nA_m^R(n)=\\frac{1}{n}\\sum_{k=1}^{n}R_m^R(k),"
        f"\\qquad n=1,\\ldots,{target}.\n$$\n\n"
        "Here $m$ identifies the method, the superscript $R$ means the rotated domain, and "
        "$R_m^R(k)$ is NLL on the fixed rotated test panel after optimizing the entire training "
        "batch containing the $k$th new rotated image. "
        f"Thus $A_m^R({target})$ averages NLL after images 1 through {target}; lower is better. "
        "The initial checkpoint ($k=0$) is excluded. Good prediction earlier in the scarce-data period "
        "counts, even if final predictions coincide.\n\n"
        f"**Same {target} new images, lower acquisition loss:** mean $A_m^R({target})$ is "
        f"**{arms['finite_difference']['rotated_auc']:.3f} for FD KL** versus "
        f"**{arms['replay_ewc']['rotated_auc']:.3f} for replay/EWC** "
        f"({relative_reduction:.1f}% lower; a descriptive ratio of means)."
    )
    reading = (
        "**Reading the figure:** top left, held-out test NLL at batch-aligned sample counts; top right, running "
        "$A_m^R(n)$, whose endpoint is the registered score. These curves show replica means, not "
        "confidence bands. Bottom left, each small dot is one replica's "
        f"$D_r=A^R_{{r,\\mathrm{{replay}}}}({target})-A^R_{{r,\\mathrm{{FD}}}}({target})$; "
        "the large dot and line give its mean and paired-bootstrap 95% interval. "
        "Bottom right, online time for the same observations.\n\n"
        f"**Efficiency result:** mean gain **{effect['mean_gain']:.3f}** "
        f"(95% CI **[{effect['ci'][0]:.3f}, {effect['ci'][1]:.3f}]**), favorable in "
        f"**{effect['favorable_replicas']}/{result['n_pairs']} replicas** "
        f"(prespecified one-sided sign-flip $p={effect['p_value']:.6g}$), "
        f"at **{ratio:.2f}x online time**. This supports better use of a fixed scarce-data budget "
        "relative to replay/EWC with the same buffer; "
        "it does not estimate a percentage of samples saved or establish equal-compute superiority. "
        f"Time covers the whole mixed stream (about {arms['replay_ewc']['exposure']:.0f} total images), "
        "including fitting, eviction processing, and evaluation; initialization is excluded.\n\n"
        f"**Starting-point context:** the frozen initial model has mean rotated-domain NLL {initial_nll:.3f} "
        f"(dashed lines). FD's final NLL is {arms['finite_difference']['final_rotated_nll']:.3f}. "
        "Compare those with the acquisition scores above: an advantage over replay is not necessarily "
        "net improvement over initialization throughout learning.\n\n"
        f"The accuracy curves below give a second view: final rotated-domain accuracy averaged "
        f"**{100 * arms['finite_difference']['final_rotated_accuracy']:.1f}% with FD** versus "
        f"**{100 * arms['replay_ewc']['final_rotated_accuracy']:.1f}% with replay/EWC** (descriptive)."
    )
    plot = _phase5c_efficiency_plot(data["units"], result)
    cells = [nbformat.v4.new_markdown_cell(progress), _phase5c_evaluation_cell(result, initial_nll), plot,
             nbformat.v4.new_markdown_cell(reading), accuracy]
    if len(details) == 1 and details[0].startswith("<details>"):
        detail_text = details[0]
    else:
        detail_text = ("<details>\n<summary>Secondary comparisons and execution diagnostics</summary>\n\n"
                       + "\n\n".join(details) + "\n\n</details>")
    if "**Execution health:**" not in detail_text:
        health = result["health"]
        diagnostics = (
            f"**Execution health:** {health['solver_failures']} solver failures; "
            f"{health['nonfinite_trajectories']} nonfinite trajectories; "
            f"{health['under_solved_updates']} under-solved updates under the common optimizer ceiling. "
            f"The selected FD coefficient is {result['selected_strength']:.2f}."
        )
        detail_text = detail_text.removesuffix("</details>").rstrip() + "\n\n" + diagnostics + "\n\n</details>"
    cells.append(nbformat.v4.new_markdown_cell(detail_text))
    for cell in cells:
        cell.metadata["tags"] = [TAG]
    notebook.cells = [cell for cell in notebook.cells if TAG not in cell.metadata.get("tags", [])]
    notebook.cells.extend(cells)
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("### Sample and compute efficiency\n"):
            _, body = cell.source.split("\n", 1)
            cell.source = (
                "### Earlier study: sample and compute audit\n\n"
                "These measurements concern the original rotation and digit-9 study. "
                "The new rare-domain confirmation and its sample-efficiency figure are in Phase 5C at the end. "
                "Prequential NLL averages loss on incoming observations before their updates; "
                "lower is better. It is not a sample-savings estimate.\n\n"
                "<details>\n<summary>Original study measurements</summary>\n"
                + body + "\n\n</details>")
        if cell.cell_type == "markdown" and cell.source.startswith("### Earlier study: sample and compute audit"):
            cell.source = cell.source.replace("in Phase 5C above.", "in Phase 5C at the end.")
    notebook.metadata["plan15_phase5c_presentation"] = dict(
        schema_version=1, artifact_only=True, analysis_unit=result["result_id"],
        refreshed_at=datetime.now(timezone.utc).isoformat())
    order_notebook_chronologically(notebook)
    nbformat.validate(notebook)
    return notebook


def refresh_phase5c(root: Path, notebook: Path, *, publish: Publisher) -> Path:
    rendered = build_phase5c_presentation(notebook, root)
    content = nbformat.writes(rendered).encode("utf-8")
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("notebook exceeds bounded publication size")
    target = root / "reports" / "fisher_memory_results.ipynb"
    publish(target, content)
    overview = next(cell for cell in rendered.cells if cell.cell_type == "code"
                    and "plan15_phase5c" in cell.metadata.get("tags", []))
    publish(root / "reports" / "phase5c_efficiency.png",
            base64.b64decode(overview.outputs[0].data["image/png"]))
    return target


def build_notebook(root: Path | str, *, extension_root: Path | str | None = None,
                   row_adapter: RowAdapter | None = None,
                   check_stop: Callable[[], None] | None = None) -> Any:
    """Build <=8 figures from sealed scalar JSON; does not publish files."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    check = check_stop or (lambda: None)
    check()
    def figure_cell(figure, caption):
        check()
        result = _plot_cell(figure, caption)
        check()
        return result
    data = collect_artifacts(root, row_adapter=row_adapter)
    check()
    units, status = data["units"], data["status"]
    cells = [_progress_cell(data)]
    changes = _changes_cell(status)
    if changes is not None:
        cells.append(changes)
    cells.append(nbformat.v4.new_markdown_cell(
        "## Question and reader's glossary\n\n"
        "Does reducing old-data sensitivity to **STM** (the LoRA residual) improve new-data learning "
        "without harming evicted-data prediction? **LTM** is the trainable base protected approximately by EWC. "
        "**Eviction** removes an observation from replay and archives its information once.\n\n"
        "**Shared model size:** every condition trains the same 663 coordinates: 487 in LTM and "
        "176 in the rank-two LoRA STM. Curvature probes and the detached distillation teacher "
        "add no trainable parameters.\n\n"
        "**Reliance** is the NLL increase when STM is attenuated; **vulnerability** is the NLL increase "
        "when later STM replaces earlier STM while earlier LTM is fixed. Positive means harm. "
        "**Curvature** measures local predictive sensitivity, not a memory share. "
        "**Fixed prevalence** $p_{ref}=0.1$ standardizes digit-9 precision during evaluation only.\n\n"
        "**Pre-eviction** means just before the cohort's first member leaves replay, after rehearsal; it is not a fresh-arrival measurement.\n\n"
        "**Under-solving** means an accepted update ended with loss-gradient infinity norm above $10^{-3}$; "
        "this warning is distinct from a failed line search or nonfinite result.\n\n"
        "New/current holdout, evicted training examples, and independent historical holdouts are different populations. "
        "Prediction panels show means and +/-1 replica SEM, not significance intervals. "
        "Cohorts are averaged within replica; missing future follow-up is never filled in. "
        "Retention and dependence panels use digits since the last cohort member's eviction. "
        "Available-cohort and replica counts can vary with age because of right censoring; these are not fixed-population curves."
    ))
    selected, notes = _selected_groups(units, status)
    pilots, pilot_notes = _selected_groups([u for u in units if _phase_key(u["metadata"]["phase"])[0] == 2 and u["complete"]], status)
    notes = list(dict.fromkeys(notes + pilot_notes))
    calibration = []
    for entry in status.get("calibration", []):
        if not isinstance(entry, dict):
            raise ArtifactError("status calibration must contain summary objects")
        calibration.append([entry.get("phase"), entry.get("environment"), entry.get("method"),
                            entry.get("relative_error"), entry.get("gradient_error"),
                            entry.get("seconds"), entry.get("decision")])
    for unit in units:
        for row in unit["rows"]:
            if row["split"] == "calibration":
                calibration.append([unit["metadata"]["phase"], ENVIRONMENT_LABELS[unit["metadata"]["environment"]],
                                    _variant(unit), row.get("relative_error"), row.get("gradient_error"),
                                    row.get("seconds"), row.get("decision")])
    source = "## Calibration and pilot\n\n"
    source += (_table(["Phase", "Environment", "Method", "Value error", "Gradient error", "Seconds", "Decision"], calibration[-12:])
               if calibration else "Numerical validation and pilot decisions are pending in the sealed metric stream.")
    if len(calibration) > 12:
        source += "\n\nLatest 12 calibration rows shown; complete diagnostics remain in unit JSON."
    if notes:
        source += "\n\n" + " ".join(notes)
    cells.append(nbformat.v4.new_markdown_cell(source))
    if pilots:
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for axis, environment in zip(axes, ENVIRONMENTS):
            for unit in pilots:
                if unit["metadata"]["environment"] != environment:
                    continue
                new, _ = _auc_for_unit(unit, "current")
                old, _ = _auc_for_unit(unit, "historical")
                if new is not None and old is not None:
                    axis.scatter(new, old, s=20, label=_display_labels(pilots)[_variant(unit)], alpha=.7)
            axis.set(title=ENVIRONMENT_LABELS[environment], xlabel="New NLL AUC", ylabel="Historical NLL AUC")
            axis.grid(alpha=.18)
        figure.suptitle("Development acquisition-retention tradeoff (lower / lower is better)")
        _legend(figure, axes)
        cells.append(figure_cell(figure, "Individual development replicas, not pooled confirmation."))
    cells.append(nbformat.v4.new_markdown_cell(
        "## Rotation results\n\nLinear and sigmoid are separate conditions. "
        "Each panel uses the latest available phase shown in its title; earlier development is not pooled into confirmation."
    ))
    for environment in ENVIRONMENTS[:2]:
        subset = [u for u in selected if u["metadata"]["environment"] == environment]
        if subset:
            check()
            cells.append(_predictive_figure(subset, environment, str(subset[0]["metadata"]["phase"])))
            check()
        else:
            cells.append(nbformat.v4.new_markdown_cell(f"**{ENVIRONMENT_LABELS[environment]}:** no sealed predictions yet."))
    mixtures = [u for u in selected if u["metadata"]["environment"] == "digit9_mixture"]
    expected = sorted({float(u["metadata"]["expected_nines"]) for u in mixtures
                       if u["metadata"].get("expected_nines") is not None})
    cells.append(nbformat.v4.new_markdown_cell(
        "## Digit-9 mixture results\n\n"
        + (f"Expected online 9s per trajectory: **{', '.join(f'{n:g}' for n in expected)}**. " if expected else "Expected counts pending. ")
        + "Replay passes are not new observations. Pretraining cost is separate. "
        "Current and historical standardized PR need not differ systematically: mixture drift changes prevalence, not class-conditional images. "
        "Missing positive cohorts have undefined recall, not zero recall."
    ))
    if mixtures:
        mixture_labels = _display_labels(mixtures)
        check()
        cells.append(_predictive_figure(mixtures, "digit9_mixture", str(mixtures[0]["metadata"]["phase"])))
        check()
        figure, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
        for col, (split, title) in enumerate(POPULATIONS.items()):
            groups = defaultdict(list)
            for unit in mixtures:
                groups[_variant(unit)].append(unit)
            for group in groups.values():
                # Aggregate matched operating points; these are not threshold-swept PR curves.
                x, recall, _, _ = _mean_curve(group, split, "nine_recall")
                xp, precision, _, _ = _mean_curve(group, split, "nine_precision")
                common = np.intersect1d(x, xp)
                if len(common):
                    axes[0, col].plot(100 * recall[np.searchsorted(x, common)],
                                      100 * precision[np.searchsorted(xp, common)],
                                      marker=".", markersize=3, label=mixture_labels[_variant(group[0])])
            axes[0, col].set(title=title, xlabel="Argmax recall (%)", ylabel="Standardized precision (%)", xlim=(-2, 102), ylim=(-2, 102))
            if not axes[0, col].lines:
                axes[0, col].text(.5, .5, "No defined positive-class operating point", ha="center", fontsize=8,
                                  transform=axes[0, col].transAxes)
            elif all(np.all(line.get_xdata() == 0) and np.all(line.get_ydata() == 0)
                     for line in axes[0, col].lines):
                axes[0, col].text(.5, .5, "All observed argmax points: 0% recall / 0% precision", ha="center", fontsize=8,
                                  transform=axes[0, col].transAxes)
            _draw(axes[1, col], mixtures, split, "nine_recall", percent=True)
            axes[1, col].set(xlabel="Unique digits integrated", ylabel="Recall (%)")
        _legend(figure, axes)
        cells.append(figure_cell(figure, "Argmax precision-recall progressions; these are not threshold-swept curves. No-positive strata remain missing."))
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for col, split in enumerate(("current", "historical")):
            groups = defaultdict(list)
            for unit in mixtures:
                candidates = _pr_records(unit, split)
                if split == "historical":
                    candidates = [r for r in candidates if r.get("eviction_exposure") is not None
                                  and r["exposure"] >= r["eviction_exposure"]]
                if not candidates:
                    continue
                endpoint = max(r["exposure"] for r in candidates)
                curves = []
                for row in candidates:
                    if row["exposure"] != endpoint:
                        continue
                    curve = row["pr_curve"]
                    if curve.get("average_precision") is None:
                        continue
                    if not np.isclose(curve.get("p_ref", .1), .1):
                        raise ArtifactError("main PR report requires fixed evaluation prevalence 0.1")
                    recall = np.asarray(curve.get("recall"), dtype=float)
                    precision = np.asarray(curve.get("precision"), dtype=float)
                    if recall.ndim != 1 or recall.shape != precision.shape or len(recall) < 2:
                        raise ArtifactError("pr_curve requires aligned recall and precision vectors")
                    if not np.all(np.isfinite(recall)) or not np.all(np.isfinite(precision)) or np.any(np.diff(recall) < 0):
                        raise ArtifactError("pr_curve must be finite with nondecreasing recall")
                    # Right endpoint step integration matches non-interpolated AP.
                    grid = np.linspace(.01, 1, 100)
                    index = np.minimum(np.searchsorted(recall, grid, side="left"), len(recall) - 1)
                    curves.append(precision[index])
                if curves:
                    groups[_variant(unit)].append(np.mean(curves, axis=0))
            for label, curves in groups.items():
                values = np.asarray(curves)
                mean = values.mean(axis=0)
                line, = axes[col].plot(np.linspace(1, 100, 100), 100 * mean, label=mixture_labels[label])
                if len(values) > 1:
                    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                    axes[col].fill_between(np.linspace(1, 100, 100), 100 * (mean - sem),
                                           100 * (mean + sem), color=line.get_color(), alpha=.15)
            if not groups:
                axes[col].text(.5, .5, "Threshold-swept PR pending", ha="center", transform=axes[col].transAxes)
            axes[col].set(title=POPULATIONS[split], xlabel="Threshold-swept recall (%)",
                          ylabel="Standardized precision (%)", xlim=(0, 100), ylim=(0, 100))
        # Exposure counts are properties of the paired stream, not treatment outcomes.
        streams = {}
        for unit in mixtures:
            streams.setdefault(unit["metadata"]["replica"], unit)
        for unit in streams.values():
            for metric, style, label in (("expected_nines", "--", "Expected 9s"),
                                         ("realized_nines", "-", "Observed 9s")):
                count_rows = [r for r in unit["rows"] if r["split"] == "current" and r.get(metric) is not None]
                if count_rows:
                    axes[2].plot([r["exposure"] for r in count_rows], [r[metric] for r in count_rows],
                                 style, alpha=.5, label=label)
        if not axes[2].lines:
            axes[2].text(.5, .5, "Realized exposure counts pending", ha="center", transform=axes[2].transAxes)
        axes[2].set(title="Positive exposure", xlabel="Unique digits integrated", ylabel="Number of 9s")
        for axis in axes:
            axis.grid(alpha=.18)
        figure.suptitle("Digit-9 endpoint threshold sweeps and statistical efficiency")
        _legend(figure, axes)
        cells.append(figure_cell(figure, "Latest available threshold sweeps; cohort means within replica, then replica means and SEM. Partial endpoints remain descriptive."))
    cells.append(nbformat.v4.new_markdown_cell(
        "## Functional memory dependence\n\n"
        "Reliance and vulnerability are functional interventions, not percentages of stored knowledge. "
        "Low vulnerability without useful acquisition may indicate lost plasticity. Raw Fisher traces are subordinate diagnostics."
        " Low STM reliance after rehearsal is compatible with transfer, but does not distinguish it from capacity suppression. Fresh-arrival STM use is not measured by the pre-eviction snapshot."
    ))
    if any(r["split"] in ("dependence", "curvature") for u in selected for r in u["rows"]):
        figure, axes = plt.subplots(3, 3, figsize=(12, 8), layout="constrained")
        for col, environment in enumerate(ENVIRONMENTS):
            subset = [u for u in selected if u["metadata"]["environment"] == environment]
            axes[0, col].set_title(ENVIRONMENT_LABELS[environment], fontsize=10)
            for row, metric in enumerate(("reliance_nll", "ltm_only_nll", "vulnerability_nll")):
                _draw(axes[row, col], subset, "dependence", metric, xfield="cohort_age", tau=0 if row < 2 else None)
                label = ("STM reliance (NLL)", "LTM-only NLL", "Update vulnerability (NLL)")[row]
                axes[row, col].set(xlabel="Digits since cohort eviction", ylabel=label)
                axes[row, col].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        _legend(figure, axes)
        cells.append(figure_cell(figure, "STM-muted reliance and LTM-only loss; future-update vulnerability. Unavailable ages remain missing."))
    trace_groups = defaultdict(list)
    for unit in selected:
        rows = [r for r in unit["rows"] if r["split"] == "curvature"]
        if rows:
            latest = max(rows, key=lambda r: r["exposure"])
            trace_groups[(unit["metadata"]["environment"], _variant(unit))].append(latest)
    trace_rows = []
    display_labels = _display_labels(selected)
    for (environment, variant), values in trace_groups.items():
        means = []
        for field in ("trace_stm", "trace_ltm", "factor_norm", "effective_update_norm"):
            observed = [v[field] for v in values if v.get(field) is not None]
            means.append(float(np.mean(observed)) if observed else None)
        trace_rows.append([ENVIRONMENT_LABELS[environment], display_labels[variant], *means])
    if trace_rows:
        cells.append(nbformat.v4.new_markdown_cell(
            "### Secondary curvature diagnostics\n\n" + _table(
                ["Environment", "Treatment", "STM trace", "LTM trace", "Factor norm", "Effective update norm"],
                trace_rows[-8:]) + "\n\nReplica means at latest checkpoints; full rows remain in unit JSON. "
            "Trace decreases alone do not establish consolidation."
        ))
    efficiency = _efficiency_cell(selected)
    if efficiency is not None:
        cells.append(efficiency)
    cells.append(_conclusion(selected, status))
    if extension_root is not None:
        cells.append(_extension_cell(Path(extension_root), Path(root), selected))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "plan15": {"schema_version": REPORT_SCHEMA_VERSION, "artifact_only": True,
                   "validation_scope": "Consumed sealed JSON and file inventory; checkpoint integrity is audited by runner",
                   "refreshed_at": datetime.now(timezone.utc).isoformat(),
                   "complete_units": sum(u["complete"] for u in units),
                   "sealed_partial_units": sum(not u["complete"] for u in units)},
    })
    check()
    order_notebook_chronologically(notebook)
    nbformat.validate(notebook)
    return notebook


def refresh(
    root: Path | str,
    notebook: Path | str | None = None,
    *,
    publish: Publisher | None = None,
    extension_root: Path | str | None = None,
    row_adapter: RowAdapter | None = None,
) -> Path:
    """Build once, then publish through a reserved atomic storage operation."""
    root = Path(root)
    target = Path(notebook) if notebook is not None else root / "reports" / "fisher_memory_results.ipynb"
    check = getattr(publish, "check_stop", lambda: None)
    content = nbformat.writes(build_notebook(
        root, extension_root=extension_root, row_adapter=row_adapter, check_stop=check)).encode("utf-8")
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("notebook exceeds bounded publication size")
    if publish is None:
        raise RuntimeError("report publication requires a storage-reserved publish(Path, bytes) hook")
    publish(target, content)
    return target


def reserved_publisher(store: Any, *, finalizing: bool = False) -> Publisher:
    """Publish under the tracked reports directory using an already-held Store.

    External notebook exports require the caller's separately accounted hook.
    Existing destination bytes and new temporary bytes coexist in the ledger.
    """
    check = (lambda: None) if finalizing else getattr(store, "check", lambda: None)
    def publish(target: Path, content: bytes) -> None:
        check()
        root = Path(store.root).resolve()
        target = Path(target).absolute()
        reports = root / "reports"
        if not target.resolve().is_relative_to(reports):
            raise ValueError("reserved report publication must remain under study-root/reports")
        if len(content) > MAX_NOTEBOOK_BYTES:
            raise ArtifactError("notebook exceeds bounded publication size")
        with store.reserve(len(content), finalizing=finalizing):
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".tmp-report-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    for offset in range(0, len(content), 1024 * 1024):
                        check()
                        handle.write(content[offset:offset + 1024 * 1024])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(temporary).unlink(missing_ok=True)
    publish.check_stop = check
    return publish


def main(argv: list[str] | None = None) -> None:
    from .artifacts import Store
    from .config import ROOT
    from .run import budget_lock, file_bytes

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plan15-matplotlib"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--extension-root", type=Path,
                        help="completed or in-progress Plan 15 optional-method extension")
    parser.add_argument("--phase5c", action="store_true",
                        help="refresh the completed Phase 5C visual summary in an existing notebook")
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    arguments = parser.parse_args(argv)
    if arguments.phase5c and (arguments.notebook is None or arguments.extension_root is not None):
        parser.error("--phase5c requires --notebook and does not take --extension-root")
    root, budget = arguments.root.resolve(), arguments.budget_root.resolve()
    if root == budget or not root.is_relative_to(budget):
        parser.error("run root must be a child of the shared Plan 15 budget root")
    if arguments.extension_root is not None:
        extension_root = arguments.extension_root.resolve()
        if extension_root == root or not extension_root.is_relative_to(budget):
            parser.error("extension root must be a distinct child of the shared Plan 15 budget root")

    identity_file = root / "identity.json"
    if not identity_file.exists():
        parser.error("root must be an initialized study; the report never creates scientific artifacts")
    with budget_lock(budget):
        identity = _read_json(identity_file)["identity"]
        ledger = _read_json(root / "byte_ledger.json")
        provenance_file = root / "provenance.json"
        manifest = _read_json(provenance_file).get("dataset_manifest", {}) if provenance_file.exists() else {}
        dependencies = max(file_bytes(arguments.data_root), sum(entry["bytes"] for entry in manifest.values()))
        orphan_exports = sum(path.stat().st_size for path in (ROOT / "mnist_experiment").glob(".plan15-report-*"))
        # Another study may have grown since this root's last ledger update.
        external = max(ledger.get("external_bytes", 0), dependencies + file_bytes(budget, exclude=root)
                       + MAX_NOTEBOOK_BYTES + orphan_exports)
        with Store(root, identity, resume=True, max_bytes=ledger.get("max_bytes", 500_000_000_000),
                   external_bytes=external,
                   finalization_reserve_bytes=ledger.get("finalization_reserve_bytes", 1_048_576)) as store:
            if arguments.phase5c:
                report = refresh_phase5c(root, arguments.notebook, publish=reserved_publisher(store))
            else:
                report = refresh(root, publish=reserved_publisher(store),
                                 extension_root=arguments.extension_root)
            if arguments.notebook is not None:
                target = arguments.notebook.absolute()
                if target.is_symlink():
                    raise ValueError("notebook export must not be a symlink")
                content = report.read_bytes()
                with store.reserve(len(content), finalizing=True):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix=".plan15-report-", dir=target.parent)
                    try:
                        with os.fdopen(fd, "wb") as handle:
                            handle.write(content)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, target)
                    finally:
                        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
