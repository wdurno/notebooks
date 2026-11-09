"""Bounded Phase 5 scalar reporting, composed with sealed Phase 0--4 history.

No learner imports, checkpoint reads, or scientific-artifact writes. The old
analysis retains its legacy field names internally; all new presentation uses
the outbound-grid clock and the configured endpoint.
"""

from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import nbformat
import numpy as np

from .. import analysis
from .. import report as original

# Pin the code actually imported by a long-running interpreter. A later file
# edit must not be presented as the implementation that produced this report.
_SOURCE_BYTES = {
    "report_source.py": Path(__file__).read_bytes(),
    "original_report_source.py": Path(original.__file__).read_bytes(),
    "analysis_source.py": Path(analysis.__file__).read_bytes(),
}

PROTOCOL = "plan18-broad-drift-v1"
REPORT_SCHEMA = original.REPORT_SCHEMA
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "cache/mnist_experiment/plan18/broad_drift_v1"
HISTORY_ROOT = DEFAULT_ROOT.with_name("production_v1")
DEFAULT_NOTEBOOK = original.DEFAULT_NOTEBOOK
MAX_JSON_BYTES = original.MAX_JSON_BYTES
MAX_GENERATION_BYTES = original.MAX_GENERATION_BYTES
ARMS, COLORS = analysis.ARMS, original.COLORS
setting = analysis.setting


def _snapshot(root, cfg, check):
    seals, consumed = {}, 0

    def read(path, required=False, seal=None):
        nonlocal consumed
        check()
        if not path.exists() and not required:
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"report input must be a regular file: {path.name}")
        if consumed + path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        with path.open("rb") as handle:
            data = handle.read(MAX_JSON_BYTES - consumed + 1)
        consumed += len(data)
        if consumed > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        digest = original._hash(data)
        if seal is not None and (seal.get("sha256") != digest or seal.get("bytes") != len(data)):
            raise ValueError(f"completed scalar seal mismatch: {path.name}")
        seals[path.relative_to(root).as_posix()] = digest
        value = json.loads(data)
        original._bytes(value)  # Reject nonfinite JSON, including exponent overflow.
        return value

    identity = read(root / "identity.json")
    scientific = (identity or {}).get("identity", {})
    if identity is not None:
        if (identity.get("schema") != 1 or identity.get("identity_hash") != original._hash(original._bytes(scientific))
                or scientific.get("protocol") != PROTOCOL
                or scientific.get("metric_schema_version", 1) != 1
                or scientific.get("artifact_schema_version", 1) != 1):
            raise ValueError("incompatible Phase 5 identity/protocol/schema")
    stored = read(root / "config.json") or scientific.get("config", {})
    if cfg is None:
        cfg = stored or dict(protocol=PROTOCOL, analysis_novel_count=1216)
    resolved = original._cfg_mapping(cfg)
    if resolved.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 5 configuration protocol")
    if stored and original._bytes(resolved) != original._bytes(stored):
        raise ValueError("report configuration differs from sealed configuration")
    N = setting(cfg, "analysis_novel_count", 1216)
    if type(N) is not int or N < 1 or (not setting(cfg, "smoke", False) and N != 1216):
        raise ValueError("Phase 5 requires the 1216 outbound-grid production budget")
    status = read(root / "status.json") or {}
    if status.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 5 status protocol")
    records = []
    units = root / "units"
    if units.is_symlink():
        raise ValueError("scientific units must not be symlinks")
    for path in sorted(units.glob("*")):
        check()
        if identity is None:
            raise ValueError("scientific units exist without a valid Store identity")
        if path.is_symlink() or not path.is_dir():
            raise ValueError("invalid scientific unit directory")
        manifest = read(path / "manifest.json", required=True)
        metadata = manifest.get("metadata", {})
        if (manifest.get("schema") != 1 or manifest.get("unit_id") != path.name
                or manifest.get("identity_hash") != identity["identity_hash"]
                or manifest.get("metadata_hash") != original._hash(original._bytes(metadata))):
            raise ValueError("scientific unit identity mismatch")
        if metadata.get("kind") != "trajectory":
            continue
        if metadata.get("protocol", PROTOCOL) != PROTOCOL:
            raise ValueError("incompatible trajectory protocol")
        record = dict(unit_id=path.name, metadata=metadata, status="incomplete")
        marker = read(path / "COMPLETED")
        if marker is not None:
            if (marker.get("schema") != 1 or marker.get("unit_id") != path.name
                    or marker.get("identity_hash") != identity["identity_hash"]
                    or marker.get("seal_hash") != original._hash(original._bytes(
                        {k: v for k, v in marker.items() if k != "seal_hash"}))):
                raise ValueError("invalid scientific completion seal")
            files = marker.get("files", {})
            for name in ("manifest.json", "summary.json", "metrics.json", "diagnostics.json"):
                if name not in files:
                    raise ValueError(f"completed scalar export missing from seal: {name}")
                value = read(path / name, required=True, seal=files[name])
                if name != "manifest.json":
                    record[name.removesuffix(".json")] = value
            if (not isinstance(record["summary"], dict) or not isinstance(record["metrics"], list)
                    or not isinstance(record["diagnostics"], list)):
                raise ValueError("incompatible completed trajectory scalar schema")
            m, hold, ramp = (setting(cfg, name, default) for name, default in
                             (("samples_per_step", 4), ("hold_steps", 160), ("ramp_steps", 304)))
            for row in record["metrics"]:
                step = row.get("accepted_step", -1)
                expected = 0 if metadata["cohort"] == "stationary" else m * max(0, min(step - hold, ramp))
                if (row.get("novel_count") != expected or row.get("outbound_count", expected) != expected
                        or row.get("incoming_grids") != m * step):
                    raise ValueError("incompatible outbound acquisition clock")
            record["status"] = "completed"
            record["completed_at"] = marker.get("completed_at", "")
        records.append(record)
    readiness = read(root / "readiness.json") or {}
    selection = read(root / "selection.json") or status.get("selection")
    numerical = read(root / "numerics.json") or readiness.get("numerical_policy") or status.get("numerical_policy") or {}
    notes = read(root / "execution_notes.json") or []
    inputs = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, files=seals, config=resolved,
        implementation={"broad_report": original._hash(_SOURCE_BYTES["report_source.py"]),
            "original_report": original._hash(_SOURCE_BYTES["original_report_source.py"]),
            "original_analysis": original._hash(_SOURCE_BYTES["analysis_source.py"])})
    return dict(records=records, cfg=cfg, status=status, readiness=readiness, selection=selection,
                numerical_policy=numerical, execution_notes=notes, inputs=inputs)


def _verified_generation(path, digest, check):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("invalid immutable report directory")
    size = 0
    for file in path.rglob("*"):
        check()
        if file.is_symlink():
            raise ValueError("symlink in immutable report generation")
        if file.is_file():
            size += file.stat().st_size
            if size > MAX_GENERATION_BYTES:
                raise ValueError("bounded report generation allocation exceeded")
    return original._verify_generation(path, digest, check)


def _history(root, smoke, check):
    latest = root / "report/latest.json"
    if not latest.exists() and smoke:
        return nbformat.v4.new_notebook(), None
    if latest.is_symlink() or not latest.is_file() or latest.stat().st_size > 4096:
        raise ValueError("sealed original production report history is required")
    pointer_bytes = latest.read_bytes()
    pointer = json.loads(pointer_bytes)
    name = pointer.get("generation", "")
    if not name or Path(name).name != name or name in (".", "..") or pointer.get("schema") != REPORT_SCHEMA:
        raise ValueError("invalid historical report generation pointer")
    path = latest.parent / name
    metadata = _verified_generation(path, pointer["source_digest"], check)
    if metadata.get("protocol") != original.PROTOCOL or metadata.get("status") != "completed":
        raise ValueError("history must be the completed original production report")
    book = nbformat.read(path / "results.ipynb", as_version=4)
    nbformat.validate(book)
    return book, dict(generation=name, source_digest=pointer["source_digest"],
                      pointer_sha256=original._hash(pointer_bytes),
                      notebook_sha256=original._hash((path / "results.ipynb").read_bytes()))


def _summarize(snapshot, check=lambda: None):
    result = analysis.summarize(snapshot["records"], snapshot["cfg"], snapshot["selection"], check=check)
    original._add_incumbent_contrast(snapshot, result)
    N = setting(snapshot["cfg"], "analysis_novel_count", 1216)

    def aliases(value):
        if isinstance(value, dict):
            for child in list(value.values()):
                aliases(child)
            for old, new in (("H100", "H_endpoint"), ("equal100_time_ratio", "equal_budget_time_ratio")):
                if old in value:
                    value[new] = value[old]
            if "H100" in value:
                value[f"H{N}"] = value["H100"]
        elif isinstance(value, list):
            for child in value:
                aliases(child)

    aliases(result)
    result.update(protocol=PROTOCOL, analysis_outbound_count=N,
                  acquisition_clock="unique_outbound_grids; stationary has no acquisition")
    result["units"].update(novel="unique_outbound_grids", acquisition="unique_outbound_grids")
    # The protocol compares with an allocated endpoint. Show when the reference
    # itself already reaches that quality earlier on a nonmonotone curve.
    result["reference_clock_diagnostics"] = {}
    complete = original._paired_records(snapshot,result)
    for arm in ("A","B"):
        method = result["methods"].get(arm)
        if method is None:
            continue
        curves = method["curves"]
        risk = curves["novel_mse"]["mean"]
        if not risk or risk[-1] is None:
            continue
        initial = np.mean([r["frozen_novel_mse"] for r in complete if r["recipe"]["arm"] == arm])
        own = analysis.crossing(risk,risk[-1],frozen_initial=float(initial))
        own.update(allocated_n=N,
            earlier_than_one_batch=bool(own["n"] is not None and own["n"] < N-setting(snapshot["cfg"],"samples_per_step",4)),
            interpretation="Diagnostic only; primary S retains the allocated endpoint denominator.")
        result["reference_clock_diagnostics"][arm] = own
    if "B-A" in result["comparisons"]:
        result["report_additions"] = {"B-A": "Planned Phase 5 contrast using the original paired bootstrap draws."}
    return result


def _shade(axis, cfg):
    m, hold, ramp = (setting(cfg, k, d) for k, d in
                     (("samples_per_step", 4), ("hold_steps", 160), ("ramp_steps", 304)))
    boundaries = np.array([0, hold, hold + ramp, hold + 2 * ramp, 2 * (hold + ramp)]) * m
    for start, end in zip(boundaries[:-1:2], boundaries[1::2]):
        axis.axvspan(start, end, color="#777777", alpha=.06)
    for boundary in boundaries[1:-1]:
        axis.axvline(boundary, color="#777777", linestyle=":", linewidth=.7)


def _process(snapshot, result):
    cfg = snapshot["cfg"]
    figure, axes = original._figure(2, 2)
    m, hold, ramp = (setting(cfg, k, d) for k, d in
                     (("samples_per_step", 4), ("hold_steps", 160), ("ramp_steps", 304)))
    steps = np.arange(2 * (hold + ramp) + 1)
    h = np.zeros_like(steps, dtype=float)
    out = (steps > hold) & (steps <= hold + ramp)
    back = (steps > hold + ramp) & (steps <= hold + 2 * ramp)
    h[out] = (1 - np.cos(np.pi * (steps[out] - hold) / ramp)) / 2
    h[back] = (1 + np.cos(np.pi * (steps[back] - hold - ramp) / ramp)) / 2
    axes[0].plot(steps * m, 45 * h, color="#21866c")
    axes[0].set(xlabel="All incoming grids", ylabel="Signed-cell mean magnitude (degrees)",
                title="Quiet / outbound / return / quiet; all grids drift")
    axes[1].plot(steps * m, m * np.clip(steps - hold, 0, ramp), color="#414950")
    axes[1].axhline(result["analysis_outbound_count"], color="#b84f65", linestyle="--")
    axes[1].set(xlabel="All incoming grids", ylabel="Unique outbound grids", title="Acquisition stops before return")
    for axis in axes[:2]:
        _shade(axis, cfg)
    replicas = snapshot["readiness"].get("detectability", {}).get("replicas", [])
    for replica_index, replica in enumerate(replicas):
        states = sorted(replica.get("states", []), key=lambda s: s["outbound_batch"])
        for law, color in (("rare", "#b07924"), ("broad", "#21866c")):
            rows = [s for s in states if s.get(law)]
            if not rows:
                continue
            x = [s["outbound_batch"] * m for s in rows]
            values = [s[law] for s in rows]
            label = law
            for field, style in (("motion_energy", ":"), ("split_motion_energy", "-"), ("batch_noise", "--")):
                axes[2].plot(x, [v[field] for v in values], color=color, linestyle=style,
                             linewidth=1, alpha=.8,
                             label=label if field == "split_motion_energy" and replica_index == 0 else None)
            ci = np.asarray([v["motion_ci"] for v in values], dtype=float)
            if ci.shape != (len(values), 2) or not np.isfinite(ci).all() or (ci[:, 0] > ci[:, 1]).any():
                raise ValueError("invalid paired motion uncertainty interval")
            axes[2].fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=.09)
            for field, style in (("motion_to_batch_noise", ":"), ("split_motion_to_batch_noise", "-")):
                if all(v.get(field) is not None for v in values):
                    axes[3].plot(x, [v[field] for v in values], color=color, linestyle=style,
                                 label=f"{label} {'split' if field.startswith('split') else 'raw'}" if replica_index == 0 else None)
    if replicas:
        for style, name in (("-", "split energy"), (":", "raw squared mean"), ("--", "actual batch V/m")):
            axes[2].plot([], [], color="#666666", linestyle=style, label=name)
    for axis, title, ylabel in zip(axes[2:], ("Paired scores: raw, split and live noise", "Detectability against actual batch noise"),
                                  ("Fixed-metric score energy (linear)", "Motion / (V/m), signed split estimate")):
        if not axis.lines:
            original._pending(axis, title)
        axis.axhline(0, color="#777777", linewidth=.6)
        axis.set(title=title, xlabel="Outbound grids at diagnostic state", ylabel=ylabel)
        original._legend(axis)
    return figure


def _development(snapshot, result):
    figure = original._development(snapshot, result)
    axes = figure.axes
    N = result["analysis_outbound_count"]
    for axis, label in zip(axes, ("Havg", f"H{N}")):
        axis.set(xlabel=f"{label} old deterioration (px; lower better)",
                 ylabel=f"Rotated A({N}) (px squared; lower better)")
    return figure


def _prediction(snapshot, result):
    figure, axes = original._figure(2, 3)
    N = result["analysis_outbound_count"]
    records = original._paired_records(snapshot, result)
    for arm, method in result["methods"].items():
        curves = method["curves"]
        for axis, field in zip(axes[:2], ("novel_mse", "old_mse")):
            item = curves[field]
            axis.plot(curves["n"], np.sqrt(np.asarray(item["mean"], dtype=float)), color=COLORS[arm], label=arm)
            if item["ci"] is not None:
                lo, hi = np.sqrt(np.asarray(item["ci"], dtype=float))
                axis.fill_between(curves["n"], lo, hi, color=COLORS[arm], alpha=.1)
            if field == "old_mse":
                axis.scatter([N] * len(item["per_replica"]),
                             np.sqrt(np.asarray(item["per_replica"], dtype=float)[:, -1]),
                             s=12, alpha=.5, color=COLORS[arm])
        for field, style in (("novel_accuracy", "-"), ("old_accuracy", "--")):
            axes[3].plot(curves["n"], curves[field]["mean"], color=COLORS[arm], linestyle=style,
                         label=f"{arm} {'rotated' if field == 'novel_accuracy' else 'old'}")
        group = [r for r in records if r["recipe"]["arm"] == arm]
        for row in group:
            regular = [r for r in row["metrics"] if r["kind"] in ("initial", "regular")]
            clock = [r["incoming_grids"] for r in regular]
            axes[2].plot(clock, np.sqrt([r["current_mse"] for r in regular]), color=COLORS[arm],
                         alpha=.5, label=arm if row is group[0] else None)
            axes[5].plot(clock, np.sqrt([r["old_mse"] for r in regular]) - math.sqrt(row["frozen_old_mse"]),
                         color=COLORS[arm], alpha=.5, label=arm if row is group[0] else None)
        if group and arm == "A":
            for axis, field in zip(axes[:2], ("frozen_novel_mse", "frozen_old_mse")):
                axis.axhline(math.sqrt(np.mean([r[field] for r in group])), color="#222222",
                             linestyle="--", label="Frozen initialization")
            for row in group:
                regular = [r for r in row["metrics"] if r["kind"] in ("initial", "regular")]
                if all(r.get("frozen_current_mse") is not None for r in regular):
                    axes[2].plot([r["incoming_grids"] for r in regular],
                                 np.sqrt([r["frozen_current_mse"] for r in regular]),
                                 color="#222222", linestyle="--", alpha=.4, label="Frozen" if row is group[0] else None)
    for name, item in result["efficiency"].items():
        p = item["point"]
        if p.get("S") is not None:
            axes[4].scatter(p["matched_time_ratio"], p["saved_fraction"], color=COLORS[p["arm"]],
                            marker="s" if name == "E-B" else "o", label=name)
    specs = [("Fixed rotated panel", "Outbound grids", "RMS (px; lower better)"),
             (f"Fixed old panel; endpoint H{N} scatter", "Outbound grids", "RMS (px; lower better)"),
             ("Current law: individual replicas", "All incoming grids", "RMS (px; lower better)"),
             ("One-pixel accuracy: rotated / old", "Outbound grids", "Fraction (higher better)"),
             ("Common A target; E-B secondary", "Matched compute multiplier (lower better)", "Saved fraction (higher better)"),
             ("Whole-path old retention (separate clock)", "All incoming grids", "Deterioration (px; lower better)")]
    for axis, (title, xlabel, ylabel) in zip(axes, specs):
        if not axis.lines and not axis.collections:
            if axis is axes[4] and result["methods"]:
                axis.text(.5,.5,"No supported sample-equivalence point",ha="center",va="center",
                          transform=axis.transAxes,fontsize=9,wrap=True)
            else:
                original._pending(axis, title)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        original._legend(axis)
    for index in (2, 5):
        _shade(axes[index], snapshot["cfg"])
    return figure


def _health_records(snapshot,result):
    records = original._paired_records(snapshot,result)
    if records:
        return records
    latest = {}
    for record in snapshot["records"]:
        if record.get("status") != "completed" or record["metadata"]["cohort"] != "development":
            continue
        arm = record["metadata"]["recipe"]["arm"]
        key = lambda row:(row.get("completed_at",""),row["unit_id"])
        if arm not in latest or key(record)>key(latest[arm]):
            latest[arm] = record
    return [original._flat(latest[a]) for a in ARMS if a in latest]


def _diagnostics(snapshot, result):
    figure, axes = original._figure(2, 3)
    records = _health_records(snapshot, result)
    m = setting(snapshot["cfg"], "samples_per_step", 4)
    for arm in ARMS:
        rows = [d for r in records if r["recipe"]["arm"] == arm for d in r["diagnostics"]]

        def series(axis, name, label, style="-", transform=lambda d: None):
            by_clock = {}
            for d in rows:
                value = d.get(name) if name else transform(d)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    by_clock.setdefault(d.get("incoming_grids", d.get("batch", 0) * m), []).append(value)
            x = sorted(by_clock)
            y = [float(np.mean(by_clock[t])) for t in x]
            if x:
                axis.plot(x, y, color=COLORS[arm], linestyle=style, linewidth=1, label=f"{arm} {label}")
            return x, y

        if arm == "A":
            series(axes[0], "applied_pi", "fixed .025")
        if arm in ("D","E"):
            for field, style in (("applied_pi", "-"), ("raw_pi", "--")):
                series(axes[0], field, field.replace("_", " "), style)
            for axis, gain in zip(axes[1:3], ("level", "velocity")):
                prefix = f"kalman_{gain}_gain_"
                x, _ = series(axis, prefix + "mean", gain)
                by_clock = {}
                for row in rows:
                    if prefix+"p10" in row and prefix+"p90" in row:
                        by_clock.setdefault(row["incoming_grids"],[]).append(
                            [row[prefix+"p10"],row[prefix+"p90"]])
                if x and all(t in by_clock for t in x):
                    bounds = np.asarray([np.mean(by_clock[t],axis=0) for t in x])
                    axis.fill_between(x,bounds[:,0],bounds[:,1],color=COLORS[arm],alpha=.12)
            series(axes[3], "noise_scale", "one-grid V", ":")
            series(axes[3], None, "batch V/m", "--", lambda d: d["noise_scale"] / m if d.get("noise_scale") is not None else None)
            series(axes[3], None, "motion + uncertainty", "-", lambda d:
                   d["motion_energy"] + d["uncertainty_energy"]
                   if d.get("motion_energy") is not None and d.get("uncertainty_energy") is not None else None)
        if arm in ("C","E"):
            for field, style in (("correction_norm", "-"), ("projection_norm", "--")):
                series(axes[4], field, field.replace("_", " "), style)
    seen = set()
    for row in original._paired_records(snapshot, result, "stationary"):
        regular = [r for r in row["metrics"] if r["kind"] in ("initial", "regular")]
        arm = row["recipe"]["arm"]
        axes[5].plot([r["incoming_grids"] for r in regular], np.sqrt([r["current_mse"] for r in regular]),
                     color=COLORS[arm], alpha=.6, label=arm if arm not in seen else None)
        seen.add(arm)
    for axis, title, ylabel in zip(axes,
            (r"Raw and applied $\pi$", "Kalman level gain: mean / p10 / p90", "Kalman velocity gain: mean / p10 / p90",
             "Movement and distinct noise units", "LFU correction / projection", "Stationary: descriptive only, no S"),
            (r"$\pi$", "Level gain", "Velocity gain", "Archive-weighted energy", "Diagonal vector norm", "Current RMS (px)")):
        if not axis.lines:
            original._pending(axis, title)
        axis.set(title=title, xlabel="All incoming grids", ylabel=ylabel)
        _shade(axis, snapshot["cfg"])
        original._legend(axis)
    return figure


def _progress(snapshot, result):
    status = snapshot["status"]
    counts = Counter(r["metadata"]["cohort"] for r in snapshot["records"] if r["status"] == "completed")
    blocks = "; ".join(f"{c} {r['n_pairs']}/{r['planned_pairs']} paired blocks" for c, r in result["cohorts"].items())
    forecast = status.get("forecast_remaining_seconds")
    if forecast is None and isinstance(status.get("forecast"), dict):
        forecast = status["forecast"].get("remaining_seconds")
    fmt = original._fmt
    current = status.get("current") or {}
    allocation = status.get("total_paths")
    if allocation is None:
        allocation = 5 * sum(c["planned_pairs"] for c in result["cohorts"].values())
    allocation = f"{allocation} {'frozen' if snapshot['selection'] else 'provisional; allocation pending freeze'}"
    initialization = f"; initialization epoch {current['epoch']}" if current.get("epoch") is not None else ""
    state = {"running":"Running", "completed":"Complete", "paused":"Paused",
             "resource_limited":"At resource limit"}.get(status.get("status"),"Preparing")
    complete = status.get("status") == "completed"
    active = ("none (execution complete)" if complete else
        f"{current.get('unit', 'none')} / {current.get('batch', 'pending')}/"
        f"{current.get('total', 'pending')}{initialization}")
    note = ("All planned paired blocks and the final artifact audit are complete. "
        "Unavailable statistical estimates are labelled 'not estimable'; no further computation is queued. "
        if complete else "Partial blocks are progress only and excluded from inference. "
        "Diagnostics/selection may still be pending; no outcome is implied. ")
    return ("# Plan 18 Progress\n\n"
        f"**Phase 5: {status.get('phase', '5A implementation; execution pending')}** | {state}.  \n"
        f"Completed paths: development {counts['development']}, fresh {counts['fresh']}, stationary {counts['stationary']}; "
        f"planned fresh/stationary paths: {allocation}. "
        "Development bound: 34 paths + at most 4 refinement paths. "
        f"{blocks}.  \n"
        f"Cumulative charged active hours: {fmt(None if status.get('active_seconds') is None else status['active_seconds'] / 3600)} "
        f"of 20 (shared with Phase 0--4); remaining forecast: {fmt(None if forecast is None else forecast / 3600)} active h. "
        f"Current unit/batch: {active}.\n\n"
        + note + "Phase 0--4 history is preserved below; the chronological Phase 5 section follows it.")


def _effects(result):
    N = result["analysis_outbound_count"]
    rows = [["E-C-D+B" if name == "interaction" else name,
             *[original._effect(value[field]) for field in ("novel_auc", "Havg", "H_endpoint", "current_whole_rms")]]
            for name, value in result["comparisons"].items()]
    return ("### Phase 5 Paired Effects\n\n" + original._table(
        ("Contrast", "Rotated risk (px squared)", "Havg (px)", f"H{N} (px)", "Current RMS (px)"),
        rows or [["Pending complete fresh blocks", *([None] * 4)]])
        + "\nLower is better; negative contrasts favor the first policy. B-A is planned here. "
        f"Rotated risk is fixed-rotated holdout MSE averaged over outbound budgets 1--{N}; "
        "it contains no regularizers. "
        "Points and nominal paired 95% intervals use whole replicas and the original sign-stratified draws. "
        "Havg takes the root after averaging old risk within each replica, relative to frozen initialization; "
        f"H{N} is its endpoint counterpart. Current RMS takes the root of full-path, arrival-averaged "
        "current-law MSE within each replica, then compares paired replicas. "
        "E-C-D+B is a calibrated-policy outcome interaction, not a "
        "mechanistic test at identical hyperparameters. " + original._small_cohort_note(result))


def _efficiency(result):
    N = result["analysis_outbound_count"]
    rows = []
    missing = "not estimable" if result.get("status") == "completed" else None
    for name, item in result["efficiency"].items():
        if name == "A-A":
            continue
        p, ci = item["point"], item.get("ci") or {}
        effect = lambda key: (original._effect(dict(mean=p[key], ci=ci.get(key)))
                              if p.get(key) is not None else missing)
        qualification = ("weak target; " if p.get("weak_target") else "") + p["status"].replace("_", " ")
        qualification += "; retention supported" if item["retention_supported"] else "; retention unestablished"
        rows.append([name + (" secondary" if name == "E-B" else " common A"),
            p.get("n") if p.get("n") is not None else missing,
            effect("saved_fraction"), effect("matched_time_ratio"), effect("S"),
            p.get("equal_budget_time_ratio"), item["bootstrap_failure_frequency"],
            original._effect(item["full_budget_old_gap"]),
            item["matched_old_gap_upper_one_sided_95"]
                if item["matched_old_gap_upper_one_sided_95"] is not None else missing, qualification])
    reference_note = " ".join(f"{arm}: allocated {item['allocated_n']}, own sustained crossing "
        + (f"{original._fmt(item['n'])} grids." if item["n"] is not None else f"{missing or 'pending'}.")
        for arm,item in result.get("reference_clock_diagnostics",{}).items())
    return ("### Phase 5 Sample Equivalence\n\n" + original._table(
        ("Target", "Outbound n", "Saved fraction", "Matched compute x", "S", f"Equal-{N} compute x",
         "Unsupported draws", "Full old gap (px)", "Matched old upper (px)", "Qualification"),
        rows or [["Pending", *([None] * 8), "pending"]])
        + f"\nS = saved outbound-grid fraction / matched-quality compute multiplier at the reference's {N}-grid endpoint. "
        "A saves zero by definition; E-B has a separate target. Persistent last crossings are used without "
        "extrapolation. Already-achieved targets are rejected; reference improvement below 5% is weak. "
        "'Not estimable' means the target was not reached or the bootstrap could not support the estimate; "
        "it does not mean unfinished computation. Any unsupported bootstrap crossing withholds inverse intervals "
        "and the matched-budget retention bound. Retention requires both full- and "
        "matched-budget one-sided 95% old-gap bounds below .10 px. Learning compute includes the initial quiet "
        "interval and all method work; initialization, development, evaluation and study overhead are separate. "
        "Stationary paths have no acquisition clock or S. Old/new protocol savings are never pooled."
        + "\n\n**Reference-clock check:** " + (reference_note or "Pending complete paired blocks.")
        + " A reference can meet its own endpoint target earlier on these nonmonotone curves. "
          "The table retains the planned saving against its allocated budget; it does not establish an "
          "additional method advantage when the candidate crosses at the same time as that earlier reference crossing.")


def _conclusion(snapshot, result):
    selection = snapshot.get("selection") or {}
    unmatched = [arm for arm, violation in selection.get("matching", {}).items()
                 if arm != "B" and violation > 0]
    failed = result["status"] == "completed" and bool(unmatched)
    qualification = ""
    if failed:
        recipes = result["selection"]
        ridges = ", ".join(f"{arm}={recipes[arm]['ridge']:g}" for arm in ARMS)
        qualification = (
            "**Calibration failed: this phase does not provide a valid five-condition comparison "
            "at comparable retention.** The STM reference B was refined to satisfy an absolute "
            "retention ceiling, while the other arms were reselected from their existing candidate sets. "
            f"The final ridge ratios were {ridges}. Development matching failed for "
            f"{', '.join(unmatched)}, but confirmation proceeded.\n\n"
            "B's stronger ridge substantially traded new learning for old-data retention. "
            "Contrasts involving B and the factorial interaction therefore confound the intended "
            "method contributions with different retention-acquisition operating points. "
            "They cannot establish that STM failed or that adding LFU/adaptive pi improves it "
            "at comparable retention. Other contrasts require their own retention checks. "
            "The trajectories remain valid observations of their recorded configurations; "
            "all numerical results and earlier report generations are preserved.\n\n"
            "[Plan 18 Phase 6](plan18.md#phase-6-retention-matched-recomparison) proposes the repair: "
            "anchor calibration to Replay/EWC, require all five development retention means to "
            "fit within the same tolerance, and complete calibration before fresh confirmation. "
            "This is a proposed follow-up, not an executed repair.\n\n"
            "Descriptive outcomes of the selected configurations: ")
    if not result["comparisons"]:
        text = "Pending complete fresh paired blocks: incremental predictive benefit over B and A and its retention support are unresolved."
    else:
        text = " ".join(f"E versus {arm}: rotated-risk difference "
            f"{original._effect(result['comparisons'][f'E-{arm}']['novel_auc'])} px squared; "
            f"retention {'supported' if result['efficiency'][f'E-{arm}']['retention_supported'] else 'not established'}."
            for arm in ("B", "A"))
    return ("### Phase 5 Conclusion\n\n" + qualification + text + " "
        "The paired score diagnostic assesses broader observer-coordinate motion, not optimum velocity. "
        "Raw squared-mean energy can be positive from Monte Carlo bias; zero/negative split energy means unresolved signal. "
        "Stronger resolved motion without predictive benefit leaves action/model mismatch open; improvement alone would "
        "not prove dilution as the mechanism. Fixed-anchor mapping discrepancy remains an alternative. "
        "Neither diagnostic gains nor a larger pi prove adaptive optimality or generally useful LFU composition. "
        + ("Calibration failure limits method attribution; it is not evidence of a general STM failure."
           if failed else "Null results remain evidence about scope; incomplete allocation is not closeout."))


def _settings(snapshot, result):
    reference = [r for r in result["development"] if r["recipe"] == result["selection"].get("B")
                 and r.get("Havg") is not None]
    rows = []
    for arm in ARMS:
        recipe = result["selection"].get(arm,{})
        candidates = [r for r in result["development"] if r["recipe"] == recipe
                      and r.get("Havg") is not None]
        matched = "pending"
        if len(reference) == len(candidates) == 2:
            gap = max(abs(np.mean([r[k] for r in candidates])-np.mean([r[k] for r in reference]))
                      for k in ("Havg","H100"))
            matched = "reference" if arm == "B" else "matched" if gap <= setting(
                snapshot["cfg"],"retention_tolerance",.1) else "unmatched"
        rows.append([arm+" "+analysis.LABELS[arm],"fixed .025" if arm in "ABC" else "velocity Kalman",
            recipe.get("ridge"),recipe.get("strength"),
            recipe.get("process_scale") if arm in "DE" else "unused",matched])
    return ("### Phase 5 Settings\n\n"+original._table(
        ("Method","pi policy","Ridge c","FD strength","Process-noise multiplier","Development match"),rows)
        +"\nEvery method trains 3,268 LTM and 108 STM coordinates with the same FIFO32/replay2. "
         r"Ridge is $\kappa=c\,\mathrm{mean}(d)$. The process-noise multiplier applies only to D/E; "
         r"it is distinct from the measured Kalman gain and from $\pi$."
         + "\n\n" + _calibration_health(snapshot,result))


def _calibration_health(snapshot,result):
    groups = {}
    for row in result["development"]:
        if row["recipe"]["arm"] == "B" and all(row.get(k) is not None for k in ("Havg","H100","novel_auc")):
            groups.setdefault(json.dumps(row["recipe"],sort_keys=True),[]).append(row)
    paired = [rows for rows in groups.values() if len(rows) == 2 and {r["replica"] for r in rows} == {0,1}]
    if not paired:
        return "Paired STM retention calibration is pending; the absolute deterioration preference is .50 px."
    ceiling = setting(snapshot["cfg"],"retention_ceiling",.5)
    rank = lambda rows: (max(0.,*[np.mean([r[k] for r in rows])-ceiling for k in ("Havg","H100")]),
                         np.mean([r["novel_auc"] for r in rows]))
    selected = result["selection"].get("B")
    rows = next((r for r in paired if r[0]["recipe"] == selected),None) if selected else None
    rows = min(paired,key=rank) if rows is None else rows
    avg,end = (float(np.mean([r[k] for r in rows])) for k in ("Havg","H100"))
    label = "Selected" if selected else "Provisional paired"
    return (f"**Absolute retention check:** {label} STM reference has Havg={avg:.3f} and "
        f"H{result['analysis_outbound_count']}={end:.3f} px; the preference is {ceiling:.2f} px for both. "
        + ("The preference is unmet. Matching other methods to this reference does not establish low forgetting. "
           "The phase continues with its best available bounded recipes." if max(avg,end)>ceiling else
           "Both development means satisfy the preference; fresh retention still needs separate assessment."))


def _readiness(snapshot):
    text = original._readiness_text(snapshot)
    audits = snapshot.get("numerical_policy",{}).get("audit",[])
    ratios = [a["anchor_mapping"]["energy_to_noise"] for a in audits if a.get("anchor_mapping")]
    if ratios:
        text += (f"\n\n**Observer qualification:** fixed-anchor mapping residual energy was "
            f"{min(ratios):.1f}--{max(ratios):.1f} times the estimated sampling-variance energy "
            "of that residual mean at the audited states. This resolves mapping discrepancy beyond "
            "Monte Carlo noise; it does not directly measure the live filter's batch noise. "
            "Stronger score motion therefore does not guarantee accurate parameter velocity or an optimal pi.")
    states = [s for replica in snapshot["readiness"].get("detectability",{}).get("replicas",[])
              for s in replica.get("states",[])]
    paired = [s["broad"]["split_motion_to_batch_noise"]/s["rare"]["split_motion_to_batch_noise"]
              for s in states if s.get("rare",{}).get("split_motion_to_batch_noise",0)>0
              and s.get("broad",{}).get("split_motion_to_batch_noise",0)>0]
    if paired:
        text += (f"\n\nPaired frozen-anchor diagnostics found {min(paired):.1f}--{max(paired):.1f} times "
            "larger motion/batch-noise ratios under broad drift across the checked states/replicas. "
            "This supports the dilution hypothesis in these score coordinates; it does not establish a predictive benefit.")
    return text


def _notebook(snapshot, result, history, figures):
    N = result["analysis_outbound_count"]
    cells = [nbformat.v4.new_markdown_cell(
        "## Phase 5: Broader Continuous Drift\n\n"
        "The earlier rare branch may have diluted population motion even when conditional rotated prediction was difficult. "
        "This amendment moves every incoming grid slowly, keeping the five methods and ramp speed unchanged. "
        f"**Efficiency budget: {N:,} unique outbound grids**, nine digit pairs per grid; not rare-component draws. "
        "Quiet -> outbound -> return -> quiet; return arrivals never overwrite acquisition states. "
        "This changes both process and budget, so old/new savings do not isolate process breadth. [Design](plan18.md).\n\n"
        "**Evaluation:** three separate holdout panels use nominal rotations, fixed rotated severities "
        r"$h\sim U[.5,1]$, and the current stream law. Their source images are disjoint from training, "
        "and development/confirmation use separate test-source halves. Holdouts never supply training, "
        "Fisher, or filter updates. All reported prediction losses exclude regularizers.")]
    captions = [
        ("Process And Paired Score Detectability", "Solid split-half energy and dotted raw squared-mean energy share a linear axis, "
         "including negative unresolved estimates. Shading is paired diagnostic uncertainty, not live-batch precision; dashed "
         "noise is actual V/m. The schedule is offline only and unavailable to the controller."),
        ("Development Retention Versus Acquisition", "All attempted recipes remain tuning evidence; outlines mark selected recipes. "
         "Old retention and rotated acquisition are separate axes, lower is better. No confirmation tuning is implied."),
        ("Fresh Prediction And Sample Equivalence", "Old/rotated fixed panels use outbound counts, including near-nominal early grids. "
         "n=0 is after initial quiet training, separate from frozen initialization. Bands are pointwise nominal paired 95%; "
         "old endpoint dots show replica scatter. Current-law and final old retention use all arrivals, with regime boundaries shaded. "
         "One-pixel accuracy is higher-is-better; old is dashed. Crossings/cost points require observed support."),
        ("Pi, Gain And Motion Diagnostics", r"$\pi$ is the applied new-data weight, not a Kalman gain or process-noise multiplier. "
         "Gain p10/p90 are coordinate quantiles, not confidence intervals. One-grid noise_scale is V; batch noise is V/m; "
         "predicted movement adds motion_energy + uncertainty_energy, weighted by each path's current archive diagonal. "
         "LFU health is measured before projection. "
         "Stationary trajectories are descriptive, never sample-equivalence evidence.")]
    for index, ((title, caption), (_, png)) in enumerate(zip(captions, figures)):
        if index == 3 and not original._paired_records(snapshot,result):
            choices = "; ".join(f"{r['recipe']['arm']} r{r['replica']} c={r['recipe']['ridge']:g}"
                +(f", eta={r['recipe']['process_scale']:g}" if r['recipe']['arm'] in "DE" else "")
                for r in _health_records(snapshot,result))
            caption = ("**Development health view:** latest complete candidate per arm "
                f"({choices or 'pending'}). These recipe diagnostics do not enter fresh inference.\n\n"+caption)
        name = f"phase5-{index + 1}.png"
        cell = nbformat.v4.new_markdown_cell(f"### {title}\n\n![{title}](attachment:{name})\n\n{caption}")
        cell["attachments"] = {name: {"image/png": base64.b64encode(png).decode("ascii")}}
        cells.append(cell)
        if index == 0:
            cells.append(nbformat.v4.new_markdown_cell(_readiness(snapshot)))
            cells.append(nbformat.v4.new_markdown_cell(_settings(snapshot,result)))
        if index == 2:
            cells.extend(nbformat.v4.new_markdown_cell(text) for text in (_effects(result), _efficiency(result)))
    cells.append(nbformat.v4.new_markdown_cell(_conclusion(snapshot, result)))
    history.cells.insert(0, nbformat.v4.new_markdown_cell(_progress(snapshot, result)))
    history.cells.extend(cells)
    history.metadata["plan18_phase5_report"] = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL,
        source_digest=snapshot["source_digest"], history=snapshot["inputs"]["history"])
    nbformat.validate(history)
    return nbformat.writes(history).encode()


def build_report(root=DEFAULT_ROOT, cfg=None, notebook_path=None, check=lambda: None, force=False, *, history_root=None):
    """Publish one immutable, atomic scalar-report generation; never run models.

    Production requires sealed original history. Smoke always publishes only
    root/smoke_results.ipynb, and may stand alone when no history is available.
    """
    root = Path(root)
    report = root / "report"
    with original._lease(report, check):
        snapshot = _snapshot(root, cfg, check)
        smoke = setting(snapshot["cfg"], "smoke", False)
        notebook = root / "smoke_results.ipynb" if smoke else Path(notebook_path or DEFAULT_NOTEBOOK)
        history, history_seal = _history(Path(history_root or HISTORY_ROOT), smoke, check)
        snapshot["inputs"]["history"] = history_seal
        snapshot["artifact_hrefs"] = {name: os.path.relpath(root / name, notebook.parent)
            for name in ("readiness.json", "numerics.json", "execution_notes.json")
            if name in snapshot["inputs"]["files"]}
        snapshot["inputs"]["artifact_hrefs"] = snapshot["artifact_hrefs"]
        digest = original._hash(original._bytes(snapshot["inputs"]))
        snapshot["source_digest"] = digest
        generation = digest[:24] + ("-" + uuid.uuid4().hex[:10] if force else "")
        latest = report / "latest.json"
        if latest.exists() and not force:
            pointer = json.loads(latest.read_bytes())
            if pointer.get("source_digest") == digest:
                generation = pointer.get("generation", "")
                if not generation or Path(generation).name != generation or generation in (".", ".."):
                    raise ValueError("invalid report generation pointer")
        destination = report / generation
        if destination.exists():
            metadata = _verified_generation(destination, digest, check)
            if metadata.get("protocol") != PROTOCOL:
                raise ValueError("incompatible Phase 5 report protocol")
            data = (destination / "results.ipynb").read_bytes()
            changed = False
        else:
            result = _summarize(snapshot, check)
            temporary = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=report))
            try:
                files, figures = dict(_SOURCE_BYTES), []
                for name, render in (("process", _process), ("development", _development),
                                     ("prediction", _prediction), ("diagnostics", _diagnostics)):
                    check()
                    figure = render(snapshot, result)
                    try:
                        for axis in figure.axes:
                            axis.title.set_fontsize(9)
                            axis.xaxis.label.set_fontsize(8)
                            axis.yaxis.label.set_fontsize(8)
                        output = io.BytesIO()
                        figure.savefig(output, format="png", dpi=100, metadata={"Software": "Plan18 Phase5 artifact report"})
                        png = output.getvalue()
                    finally:
                        figure.clear()
                    figures.append((name, png))
                    files[f"figures/{name}.png"] = png
                data = _notebook(snapshot, result, history, figures)
                metadata = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, generation=generation,
                    source_digest=digest, created_at=datetime.now(timezone.utc).isoformat(), status=result["status"],
                    completed_pairs={c: r["n_pairs"] for c, r in result["cohorts"].items()},
                    notebook="results.ipynb", figure_count=4)
                files.update({"results.ipynb": data, "analysis.json": original._bytes(result),
                              "inputs.json": original._bytes(snapshot["inputs"]), "generation.json": original._bytes(metadata)})
                if sum(map(len, files.values())) > MAX_GENERATION_BYTES:
                    raise ValueError("bounded report generation allocation exceeded")
                for name, content in files.items():
                    check()
                    original._atomic(temporary / name, content)
                original._atomic(temporary / "COMPLETED", original._bytes(dict(schema=REPORT_SCHEMA,
                    source_digest=digest, files={name: original._hash(content) for name, content in files.items()})))
                check()
                os.rename(temporary, destination)
                original._sync_directory(report)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            changed = True
        check()
        if not notebook.exists() or notebook.read_bytes() != data:
            original._atomic(notebook, data)
        pointer_bytes = original._bytes(dict(schema=REPORT_SCHEMA, generation=generation, source_digest=digest))
        if not latest.exists() or latest.read_bytes() != pointer_bytes:
            original._atomic(latest, pointer_bytes)
        return {**metadata, "changed": changed}
