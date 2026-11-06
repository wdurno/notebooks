"""Artifact-only Phase 6 appendix, preserving the sealed Phase 5E notebook.

Only bounded JSON, inherited report artifacts and reporting sources are read.
No model, execution, data, configuration, torch, or experiment-lock imports.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
import numpy as np

from mnist_experiment.diagonal_lfu import report as report_helpers
from mnist_experiment.diagonal_lfu.report import (
    MAX_JSON_BYTES, _atomic, _figure, _finish, _hash, _json_bytes,
    _load, _number, _table, _text,
)
from mnist_experiment.kalman_pi.report import _finite_json
from mnist_experiment.kalman_pi.retention.report import _publish, _read_book

from . import analysis

KIND = "plan17-registration-snapshot"
REPORT_KIND = "plan17-registration-report"
MARKER = "plan17_registration_report"
PREFIX = "plan17-registration-"
FIGURES = (("12_registration_context", "Task And Readiness"),
           ("13_registration_development", "Development Trade-Offs And Observer Health"),
           ("14_registration_fresh", "Fresh Predictive Quality"),
           ("15_registration_effects", "Applied Weights, Motion And Paired Effects"))
NAMES = {"fixed": "Fixed .025", "covariance": "Covariance", "ema": "Anchor EMA", "kalman": "Anchor Kalman",
         "kalman_matched": "Kalman calibrated ridge", "fixed_matched": "Fixed calibrated ridge"}
COLORS = {"fixed": "#3976a4", "covariance": "#817736", "ema": "#278574", "kalman": "#b34c65",
          "kalman_matched": "#9251a5", "fixed_matched": "#666666"}
SCHEDULES = ("main", "stationary")


def _effect(value, *, signed=False, digits=4):
    if not value or value.get("mean") is None:
        return "pending"
    result = f"{value['mean']:+.{digits}f}" if signed else _number(value["mean"], digits)
    if value.get("ci95") is not None:
        result += " [" + ", ".join(_number(v, digits) for v in value["ci95"]) + "]"
    return result + (" (unreliable)" if not value.get("interval_reliable") else "")


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id=PREFIX + name)


def _validated(root):
    snapshot, data = _load(root / "report_snapshot.json")
    _finite_json(snapshot)
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1 or snapshot.get("kind") != KIND:
        raise ValueError("requires schema 1 kind plan17-registration-snapshot")
    for name in ("study_id", "status", "phase"):
        if not isinstance(snapshot.get(name), str) or not snapshot[name]:
            raise ValueError(f"snapshot.{name} must be a nonempty string")
    for name in ("config", "progress", "readiness", "diagnostics"):
        if not isinstance(snapshot.get(name), dict):
            raise ValueError(f"snapshot.{name} must be an object")
    if type(snapshot["config"].get("smoke", False)) is not bool:
        raise ValueError("config.smoke must be boolean")
    if snapshot["progress"].get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
        raise ValueError("progress/snapshot study_id mismatch")
    selected = snapshot.get("selection")
    analysis._selection(selected)
    seed = (selected or {}).get("bootstrap_seed", snapshot["config"].get("bootstrap_seed", analysis.DEFAULT_SEED))
    result = analysis.summarize(snapshot.get("trajectories"), selected,
                                bootstrap_draws=snapshot["config"].get("bootstrap_draws", 2000),
                                seed=seed, config=snapshot["config"])
    _finite_json(result)
    if snapshot.get("analysis") and snapshot["analysis"] != result:
        raise ValueError("snapshot.analysis differs from validated scalar analysis")
    return snapshot, result, data


def _checked_files(directory, manifest, required):
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("report generation missing required artifact hashes")
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("report artifact path escapes generation")
        path = directory / name
        if path.resolve().parent != directory or path.stat().st_size > MAX_JSON_BYTES or _hash(path.read_bytes()) != digest:
            raise ValueError("corrupt report artifact: " + name)


def _historical(snapshot):
    source = snapshot.get("historical_report_dir")
    if snapshot["config"].get("smoke", False):
        if source is not None or snapshot.get("historical_notebook_sha256") is not None:
            raise ValueError("smoke must not inherit production history")
        return None, None, {}
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("historical_report_dir must pin an absolute sealed Phase 5E generation")
    directory = Path(source).resolve()
    manifest, manifest_bytes = _load(directory / "manifest.json")
    if (directory / "COMPLETED").read_text().strip() != _hash(manifest_bytes):
        raise ValueError("corrupt historical completion seal")
    required = {"report.ipynb", "report_snapshot.json", "analysis.json", "10_zero_ridge_effects.png", "11_zero_ridge_accuracy.png"}
    _checked_files(directory, manifest, required)
    book, data = _read_book(directory / "report.ipynb")
    sealed, sealed_bytes = _load(directory / "report_snapshot.json")
    meta = book.metadata.get("plan17_zero_ridge_report", {})
    if (manifest.get("schema") != 1 or manifest.get("kind") != "plan17-zero-ridge-report"
            or manifest.get("report_dir") != str(directory) or manifest.get("smoke") is not False
            or sealed.get("schema") != 1 or sealed.get("kind") != "plan17-zero-ridge-snapshot"
            or sealed.get("status") not in ("complete", "completed")
            or meta.get("schema") != 1 or meta.get("smoke") is not False
            or meta.get("status") != "Completed supplement" or meta.get("generation") != directory.name
            or meta.get("study_id") != manifest.get("study_id") or sealed.get("study_id") != manifest.get("study_id")
            or meta.get("snapshot_sha256") != _hash(sealed_bytes) or manifest.get("snapshot_sha256") != _hash(sealed_bytes)
            or snapshot.get("historical_notebook_sha256") != _hash(data) or MARKER in book.metadata):
        raise ValueError("history must be the pinned sealed completed non-smoke Phase 5E report")
    required_cells = {"plan17-conclusion", "plan17-retention-conclusion", "plan17-zero-ridge-conclusion"}
    if not required_cells <= {c.id for c in book.cells}:
        raise ValueError("history lacks inherited Phase 0--5E conclusions")
    for cell in book.cells:
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png is not None:
                try:
                    valid = base64.b64decode(png, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
                except (ValueError, TypeError) as error:
                    raise ValueError("corrupt inherited image output") from error
                if not valid:
                    raise ValueError("corrupt inherited image output")
    paths = [directory / "manifest.json", directory / "COMPLETED", *(directory / n for n in manifest["files_sha256"])]
    return book, directory, {str(p): _hash(p.read_bytes()) for p in paths}


def _complete_counts(result):
    return {schedule: sum(g["n_blocks"] for g in result["groups"] if g["schedule"] == schedule) for schedule in SCHEDULES}


def _status(snapshot, result):
    selected = snapshot.get("selection")
    if not selected:
        return "Pending selection / Phase 6A acceptance"
    counts = _complete_counts(result)
    completed = snapshot["status"] in ("completed", "complete") and not result["counts"]["incomplete"]
    completed = completed and all(counts[s] == selected[s + "_replicas"] for s in SCHEDULES)
    return "Completed Phase 6" if completed else "Partial / incomplete Phase 6"


def _progress(snapshot, result):
    p, selected = snapshot["progress"], snapshot.get("selection") or {}
    current = p.get("current") if isinstance(p.get("current"), dict) else {}
    counts = _complete_counts(result)
    phase_hours = p.get("phase6_active_hours", p.get("active_hours"))
    if phase_hours is None and p.get("active_seconds") is not None:
        phase_hours = p["active_seconds"] / 3600
    historical = p.get("historical_active_hours", 14.318)
    total = p.get("total_family_active_hours", historical + phase_hours if phase_hours is not None else None)
    storage = p.get("storage") or {}
    used = storage.get("used_bytes")
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    limit = storage.get("limit_bytes", storage.get("max_bytes"))
    free = storage.get("free_bytes", storage.get("filesystem_free_bytes"))
    rows = [("Phase / revision / state", f"{snapshot['phase']} / {snapshot['config'].get('protocol', snapshot['config'].get('revision', 'pending'))} / {snapshot['status']}"),
            ("Paired blocks: changing; stationary (complete/planned)", "; ".join(f"{counts[s]}/{selected.get(s + '_replicas', 'pending')}" for s in SCHEDULES)),
            ("Active unit / accepted checkpoint", f"{p.get('active_unit') or current.get('active_unit') or 'none'} / {p.get('last_accepted_checkpoint') or current.get('last_accepted_checkpoint') or p.get('last_checkpoint') or 'pending'}"),
            ("Heartbeat / accepted step", f"{p.get('heartbeat', 'pending')} / {current.get('step', p.get('step', 'pending'))}"),
            ("Phase 6 active / separate allowance (h)", f"{_number(phase_hours, 3)} / {_number(p.get('phase6_allocation_hours', p.get('allocation_hours', 20)), 3)}"),
            ("Prior phases preserved / total family active (h)", f"{_number(historical, 3)} / {_number(total, 3)}"),
            ("Remaining forecast (h)", _number(p['forecast_seconds'] / 3600) if p.get('forecast_seconds') is not None else "pending"),
            ("Storage used / ceiling / free (GB)", " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, limit, free)))]
    return _markdown("progress", "## Current Phase 6 Progress\n\n" + _table(("Item", "Recorded state"), rows) +
                     "\n\n" + _text(_status(snapshot, result)) + ". Snapshot freshness is not a liveness check. "
                     f"{result['counts']['complete']} completed trajectories; {result['counts']['incomplete']} incomplete; "
                     f"{len(result['awaiting_full_block_units'])} completed arms await full blocks. "
                     "The 20-hour Phase 6 allowance is additional to, not shared with, the preserved 14.318 historical hours. "
                     "Incomplete trajectories are provisional live previews: they may roll back or temporarily disappear on resume; "
                     "only sealed full paths enter inference. "
                     "Inherited progress panels below are immutable historical snapshots. No experiment is started by this report.")


def _pending(ax, message):
    ax.text(.5, .5, message, ha="center", va="center", wrap=True, transform=ax.transAxes, fontsize=9)


def _context(snapshot, result, destination):
    readiness = snapshot["readiness"]
    with _figure() as (fig, axes):
        for ax, key in zip(axes[0], ("reference", "query")):
            examples = readiness.get("examples") or []
            example = examples[0] if examples else {}
            pixels = example.get(key)
            if pixels is not None:
                pixels = np.asarray(pixels, dtype=float)
                if pixels.shape != (28, 28) or not np.isfinite(pixels).all():
                    raise ValueError("readiness images must be finite 28x28 arrays")
                ax.imshow(pixels, cmap="gray", origin="upper", vmin=0, vmax=1)
                points = example.get(key + "_landmarks")
                if points is not None:
                    points = np.asarray(points, dtype=float)
                    if points.shape != (3, 2) or not np.isfinite(points).all():
                        raise ValueError("readiness overlays require three finite pixel landmarks")
                    ax.scatter(points[:, 0] + 13.5, points[:, 1] + 13.5, c="#dd5f5f", marker="+", s=65)
            else:
                _pending(ax, "Pending recorded " + key + "/landmark example")
            ax.set_title(key.title() + " (three virtual landmarks)")
            ax.set_axis_off()
        law = readiness.get("angle_law") or {}
        angle = law.get("angles")
        if angle is not None:
            for kind, style in (("proposed", ":"), ("accepted", "-")):
                for label, density in law.get(kind, {}).items():
                    axes[1, 0].plot(angle, density, style, label=f"{kind}: {label}")
        else:
            _pending(axes[1, 0], "Pending proposed/accepted angle densities\nand geometric-acceptance diagnostics")
        axes[1, 0].set(title="Angle law", xlabel="Degrees", ylabel="Density")
        schedule = readiness.get("schedule") or []
        if schedule:
            axes[1, 1].plot([r["step"] for r in schedule], [r["h"] for r in schedule], color="#3976a4")
        else:
            _pending(axes[1, 1], "Pending frozen-model readiness and schedule")
        probes = _probe_summary(readiness.get("probes") or [])
        if probes:
            text = "\n".join(f"{angle:g} deg, landmark MSE (px^2): old {_number(p['old'])}; shifted {_number(p['shifted'])}; fit {_number(p['fitted'])}" for angle, p in probes.items())
            axes[1, 1].text(.02, .98, text, va="top", transform=axes[1, 1].transAxes, fontsize=7)
        axes[1, 1].set(title="Preassigned context / frozen readiness", xlabel="Incoming batches", ylabel="h (not a controller input)")
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _probe_summary(probes):
    grouped = defaultdict(lambda: defaultdict(list))
    for p in probes:
        angle = p.get("angle_max", p.get("angle"))
        if angle is None:
            continue
        angle = analysis._number(angle, "probe angle", 0, 50)
        for key, flat, nested in (("old", "old_risk", "frozen_old"), ("shifted", "shifted_risk", "frozen_shifted"),
                                  ("fitted", "capability_shifted_risk", "fitted_shifted")):
            value = p.get(flat, (p.get(nested) or {}).get("risk"))
            if value is not None:
                grouped[angle][key].append(analysis._number(value, "probe risk", 0))
    return {angle: {key: float(np.mean(group[key])) if group[key] else None for key in ("old", "shifted", "fitted")}
            for angle, group in sorted(grouped.items())}


def _development(snapshot, result, destination):
    paths = [p for p in result["path_summaries"].values() if p["stage"] == "development"]
    selected_specs = (snapshot.get("selection") or {}).get("arms", [])
    with _figure(rows=1, cols=2) as (fig, axes):
        for arm in NAMES:
            selected = [p for p in paths if p["arm"] == arm]
            for p in selected:
                axes[0, 0].scatter(p["old_rms"], p["shifted_rms"], c=COLORS[arm], label=NAMES[arm], s=22)
                axes[0, 0].annotate(f"r={p['ridge']:g}", (p["old_rms"], p["shifted_rms"]), fontsize=6)
                if any(a["condition"] == p["condition"] and a["ridge"] == p["ridge"] for a in selected_specs):
                    axes[0, 0].scatter(p["old_rms"], p["shifted_rms"], marker="*", s=100, edgecolor="#111111", facecolor="none", label="Selected package")
        axes[0, 0].set(title="All development candidates", xlabel="Whole-path nominal budget RMS (pixels)", ylabel="Acquisition-window shifted RMS (pixels)")
        for t in snapshot["trajectories"]:
            if t["stage"] != "development" or t["arm"] not in ("ema", "kalman"):
                continue
            rows = [r for r in t["metrics"] if r.get("innovation_ratio") is not None]
            if rows:
                label = t["condition"] + (" (provisional)" if not t["completed"] else "")
                axes[0, 1].plot([r["arrivals"] for r in rows], [r["innovation_ratio"] for r in rows], alpha=.65, label=label, color=COLORS[t["arm"]])
        if axes[0, 1].has_data():
            axes[0, 1].axhline(1, color="#777777", linestyle=":")
        axes[0, 1].set(title="Observer health, not a drift oracle", xlabel="Incoming pairs", ylabel="Predictive innovation ratio")
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _eligible(snapshot, result, schedule):
    by_id = {t["unit_id"]: t for t in snapshot["trajectories"]}
    excluded = set(result["awaiting_full_block_units"])
    return [by_id[u] for u in result["complete_units"] if by_id[u]["stage"] == "fresh" and by_id[u]["schedule"] == schedule and u not in excluded]


def _curves(ax, trajectories, metric, *, rms=False, percent=False):
    cohorts = defaultdict(list)
    for t in trajectories:
        grid = tuple((r["step"], r["arrivals"]) for r in t["metrics"])
        cohorts[(t["cohort"], t["arm"], t["schedule_factor"], grid)].append(t)
    for (_, arm, factor, _), paths in sorted(cohorts.items()):
        values = [[r.get(metric) for r in t["metrics"]] for t in paths]
        keep = [all(row[i] is not None for row in values) for i in range(len(values[0]))]
        if not any(keep):
            continue
        y = np.asarray([[v for v, valid in zip(row, keep) if valid] for row in values], dtype=float)
        if rms:
            y = np.sqrt(y)
        if percent:
            y *= 100
        x = [r["arrivals"] for r, valid in zip(paths[0]["metrics"], keep) if valid]
        mean = y.mean(axis=0)
        label = f"{NAMES[arm]}, L={factor:g}, n={len(paths)}"
        ax.plot(x, mean, color=COLORS[arm], linestyle="-" if factor == 1 else "--", alpha=.85, label=label)
        if len(paths) > 1:
            half = 1.96 * y.std(axis=0, ddof=1) / np.sqrt(len(paths))
            ax.fill_between(x, mean - half, mean + half, alpha=.08, color=COLORS[arm])
        if metric.endswith("_risk"):
            ax.plot([x[0], x[-1]], [mean[0]] * 2, color=COLORS[arm], linestyle=":", alpha=.5)


def _regimes(ax, trajectories, snapshot):
    for i, t in enumerate(trajectories):
        if t["arm"] != "fixed":
            continue
        boundaries = analysis._boundaries(*analysis._horizon(t, snapshot.get("selection"), snapshot["config"]))
        clock = {r["step"]: r["arrivals"] for r in t["metrics"]}
        for start, end in ((boundaries[1], boundaries[2]), (boundaries[3], boundaries[4])):
            ax.axvspan(clock[start], clock[end], color="#999999", alpha=.035)
        if i > 4:
            break


def _fresh(snapshot, result, destination):
    with _figure(rows=2, cols=4) as (fig, axes):
        for row, schedule in enumerate(SCHEDULES):
            paths = _eligible(snapshot, result, schedule)
            for ax, metric in zip(axes[row], ("old_risk", "shifted_risk", "old_accuracy", "shifted_accuracy")):
                _curves(ax, paths, metric, rms=metric.endswith("risk"), percent=metric.endswith("accuracy"))
                _regimes(ax, paths, snapshot)
                outcome = {"old_risk": "nominal RMS", "shifted_risk": "shifted RMS",
                           "old_accuracy": "nominal accuracy", "shifted_accuracy": "shifted accuracy"}[metric]
                title = ("Changing" if schedule == "main" else "Stationary") + ": " + outcome
                ax.set(title=title, xlabel="Incoming pairs", ylabel="RMS pixels" if metric.endswith("risk") else "Within 1 pixel (%)")
                ax.set_ylim(bottom=0, top=100 if metric.endswith("accuracy") else None)
        # Identical outcome axes across schedules; sign/length cohorts never interpolate.
        for col in range(4):
            upper = max(ax.get_ylim()[1] for ax in axes[:, col])
            for ax in axes[:, col]:
                ax.set_ylim(0, upper)
        _finish(axes)
        for ax in axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=7)
        fig.savefig(destination, dpi=135)


def _effects(snapshot, result, destination):
    with _figure(rows=2, cols=3) as (fig, axes):
        for row, schedule in enumerate(SCHEDULES):
            paths = _eligible(snapshot, result, schedule)
            _curves(axes[row, 0], paths, "applied_pi")
            actions = [r["applied_pi"] for t in paths for r in t["metrics"] if r.get("applied_pi") is not None]
            upper = min(.2, max(.025, max(actions, default=.025)) * 1.15)
            title = "Changing" if schedule == "main" else "Stationary"
            axes[row, 0].set(title=title + ": adaptation weight", xlabel="Incoming pairs", ylabel="pi", ylim=(0, upper))
            for metric, style in (("motion_energy", "-"), ("uncertainty_energy", "--")):
                for t in paths:
                    if t["arm"] not in ("kalman", "kalman_matched"):
                        continue
                    rows = [r for r in t["metrics"] if r.get(metric) is not None]
                    if rows:
                        label = "Estimated motion" if metric == "motion_energy" else "Filter uncertainty"
                        axes[row, 1].plot([r["arrivals"] for r in rows], [r[metric] for r in rows], style, color=COLORS[t["arm"]], alpha=.5, label=label)
            axes[row, 1].set(title=title + ": distinct motion terms", xlabel="Incoming pairs", ylabel="Energy")
            comparisons = [(g, c) for g in result["groups"] if g["schedule"] == schedule for c in g["comparisons"]]
            labels = []
            for i, (group, contrast) in enumerate(comparisons):
                labels.append(NAMES[contrast["arm"]] + " vs " + NAMES[contrast["control"]])
                for metric, offset, marker in (("old_risk_gain", -.1, "o"), ("shifted_risk_gain", .1, "s")):
                    value = contrast["effects"][metric]
                    if value["mean"] is None:
                        continue
                    if value["ci95"] is not None:
                        axes[row, 2].plot(value["ci95"], [i + offset] * 2, color=COLORS[contrast["arm"]])
                    label = "Old-domain risk gain" if metric == "old_risk_gain" else "Shifted-domain risk gain"
                    axes[row, 2].plot(value["mean"], i + offset, marker, color=COLORS[contrast["arm"]], label=label)
            if comparisons:
                axes[row, 2].axvline(0, color="#777777", linewidth=.8)
                axes[row, 2].set_yticks(range(len(labels)), labels, fontsize=6)
            axes[row, 2].set(title=title + ": paired risk gains", xlabel="Control minus policy (squared pixels)")
        _finish(axes)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes[:, 0]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if handles:
            fig.legend(handles, [label.split(", n=")[0] for label in labels], loc="outside lower center", ncols=4, fontsize=7)
        fig.savefig(destination, dpi=135)


def _settings(snapshot):
    s, config = snapshot.get("selection") or {}, snapshot["config"]
    arms = "; ".join(f"{a['arm']}: {a['condition']}, r={a['ridge']:g}" for a in s.get("arms", [])) or "pending frozen selection"
    rows = [("Task / likelihood", "One reference/query digit; three virtual landmarks; Gaussian covariance 4 I6"),
            ("Policies / common ridge", arms),
            ("Resolved base hold / ramp (batches)", f"{s.get('hold_steps', config.get('hold_steps', 'pending'))} / {s.get('ramp_steps', config.get('ramp_steps', 'pending'))}"),
            ("Retention calibration", "Development old budget-RMS band +/-0.10 pixels; no fresh matching"),
            ("Bootstrap", f"{(s or config).get('bootstrap_seed', config.get('bootstrap_seed', analysis.DEFAULT_SEED))}; {min(max(config.get('bootstrap_draws', 2000), 0), analysis.MAX_BOOTSTRAP_DRAWS)} shared whole-replica draws")]
    return _markdown("settings", "### Frozen Settings\n\n" + _table(("Setting", "Recorded value"), rows))


def _conclusions(snapshot, result):
    rows = []
    for group in result["groups"]:
        for c in group["comparisons"]:
            effects = c["effects"]
            rows.append((group["schedule"] + ": " + NAMES[c["arm"]] + " - " + NAMES[c["control"]], c["n_pairs"],
                         _effect(effects["old_rms_gap"], signed=True), _effect(effects["shifted_rms_gap"], signed=True),
                         _effect(effects["learning_seconds_ratio"], digits=3), _effect(effects["online_wall_seconds_ratio"], digits=3)))
    if not rows:
        rows = [("Pending complete frozen blocks", 0, "pending", "pending", "pending", "pending")]
    header = ("Contrast", "Blocks", "Old RMS gap [95% CI], px", "Shifted RMS gap [95% CI], px", "Learning ratio [95% CI]", "Online wall ratio [95% CI]")
    return _markdown("conclusion", "### Phase 6 Interpretation\n\n" + _table(header, rows) +
                     "\n\n" + _reading(snapshot, result) +
                     "\n\nNegative RMS gaps favor the first arm; positive risk gains in Figure 15 favor it. "
                     "Old RMS covers the whole path; shifted RMS covers rise plus high hold, including the same virtual window in stationary controls. "
                     "Roots are taken within replicas. Stationary shifted error measures untrained-domain generalization, not acquisition from shifted arrivals. Retention status: " + _text(_retention_status(snapshot.get("selection"))) + ". "
                     "Calibration within +/-0.10 pixels is not an equivalence conclusion. "
                     "Missing costs/intervals are pending, never zero. Learning includes solver, curvature, controller and render work; online wall also includes evaluation and I/O. "
                     "Whole-path shifted risk, every regime, forgetting, endpoint recovery, post-return retention and unfavorable candidates remain in the linked analysis. "
                     "Ordinary bootstrap cohorts include between-sign/length variation when a stratum has fewer than two replicas. "
                     "No significance stopping, checkpoint pseudo-replication, actual-samples-saved claim, or equivalence claim from an interval crossing zero. "
                     "Without timing controls, any benefit is a policy-package result, not proof of accurate latent-parameter tracking or temporal-alignment value.")


def _retention_status(selected):
    if not selected:
        return "pending development calibration"
    ridge = selected.get("calibrated_ridge")
    if ridge is None:
        ridge = selected.get("common_ridge")
    candidate = next((c for c in selected.get("retention_candidates", []) if c.get("ridge") == ridge), None)
    if candidate is None:
        return "development matching gap not recorded in this snapshot"
    gap = candidate["old_gap"]
    status = "within development band" if abs(gap) <= .10 else "outside development band / unmatched"
    if gap < -.10 and candidate["acquisition_gap"] < 0 and ridge == selected.get("common_ridge"):
        status = "both development errors lower; matching not forced"
    return f"{status}; selected r={ridge:g}, old RMS gap {gap:+.4f} pixels; no fresh matching"


def _reading(snapshot, result):
    if snapshot["config"].get("smoke", False):
        return "**Smoke acceptance only, not production evidence.**"
    primary = [c for g in result["groups"] if g["schedule"] == "main" for c in g["comparisons"] if c["arm"] == "kalman" and c["control"] == "fixed" and c["n_pairs"]]
    if not primary:
        return "Pending: no complete fresh changing block; no efficacy conclusion is available."
    readings = []
    for c in primary:
        old, shifted = (c["effects"][m] for m in ("old_rms_gap", "shifted_rms_gap"))
        if shifted["ci95"] is not None and shifted["ci95"][0] > 0:
            if old["ci95"] is not None and old["ci95"][1] < 0:
                readings.append("The nominal intervals support an acquisition/retention trade-off: Kalman retains old performance better but has higher shifted error.")
            elif old["ci95"] is not None and old["ci95"][0] > 0:
                readings.append("Both nominal intervals favor fixed .025 over Kalman at common ridge.")
            else:
                readings.append("The changing-schedule nominal interval favors fixed .025 on shifted error; an old-domain improvement for Kalman is not established. This context does not establish a joint applied benefit from adaptive weights.")
        elif old["mean"] * shifted["mean"] < 0:
            readings.append("The common-ridge point estimates show an acquisition/retention trade-off.")
        elif all(e["ci95"] is not None and e["ci95"][1] < 0 for e in (old, shifted)):
            readings.append("Both nominal intervals favor Kalman at common ridge; this is exploratory policy-package evidence.")
        else:
            readings.append("A joint common-ridge improvement remains uncertain; all directions and intervals are shown.")
    return " ".join(readings)


def _section(snapshot, result, destination, notebook):
    cells = [_markdown("section", "## Phase 6: Adaptation Weights For Single-Digit Registration\n\n" +
                       "Phase 5E found a common-zero-ridge retention benefit but no established acquisition benefit; removing ridge also improved the fixed policy. "
                       "Its motion term was largely uncertainty, with little intended rise/fall action pattern. "
                       "Single-digit reference/query registration instead changes the rotation distribution of ordinary incoming observations. "
                       "The hypothesis is that observable, recoverable continuous drift makes causal adaptation weights useful relative to fixed pi=.025. "
                       "The task law remains fixed and has one changing rotational degree of freedom, not six independent changes.\n\n" +
                       ("**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.**" if snapshot["config"].get("smoke", False) else "**" + _status(snapshot, result) + ".**") +
                       " Readiness and efficacy are distinct; no training, inference, download or artifact repair occurs here."), _settings(snapshot)]
    captions = ("Recorded examples and accepted angle law only; readiness probes, including unsuccessful probes, are not fresh efficacy evidence.",
                "All completed development candidates remain visible. Selection uses normalized risks, not an efficacy gate; zero denominators are explicitly flagged.",
                "Complete frozen blocks only. Rotation directions are averaged within each length and exact checkpoint grid; different grids are never interpolated. Solid/dashed lines distinguish length factors 1/1.25. Dotted lines mark group-mean frozen initial errors; shaded bands mark assigned ramps, not actual stationary changes. Pointwise bands are descriptive mean +/-1.96 SE.",
                "Weights are the actions used for batches ending at each checkpoint; initialization has no action. Mean-motion and uncertainty are separate. Circle/square effects denote old/shifted risk; nominal paired 95% intervals resample whole replicas, shared across endpoints and contrasts.")
    for (name, title), caption in zip(FIGURES, captions):
        cells.append(_markdown(name + "-heading", "### " + title + "\n\n" + caption))
        relative = os.path.relpath(destination / (name + ".png"), notebook.parent)
        cells.append(nbformat.v4.new_code_cell("from IPython.display import Image, display\n" + f"display(Image(filename={relative!r}))", id=PREFIX + name, execution_count=1, outputs=[]))
    cells.append(_conclusions(snapshot, result))
    cells.append(_markdown("artifacts", "[Sealed Phase 6 snapshot](" + os.path.relpath(destination / "report_snapshot.json", notebook.parent) +
                           ") | [Complete scalar analysis and regime diagnostics](" + os.path.relpath(destination / "analysis.json", notebook.parent) +
                           "). Four PNGs are embedded. Every inherited cell, output and metadata field is preserved; historical report generations are never regenerated."))
    return cells


def _sources():
    files = {Path(__file__).resolve(), Path(analysis.__file__).resolve(), Path(report_helpers.__file__).resolve()}
    # These artifact helpers depend on the unchanged historical scalar modules.
    package = Path(__file__).resolve().parents[1]
    files.update(package / name for name in ("report.py", "analysis.py", "retention/report.py", "retention/analysis.py"))
    files.add(package.parent / "diagonal_lfu/analysis.py")
    return {str(p): _hash(p.read_bytes()) for p in sorted(files)}


def _generation(identity):
    return "generation-" + _hash(_json_bytes(identity))[:24]


def _verify_generation(directory, expected=None):
    directory = Path(directory).resolve()
    manifest, data = _load(directory / "manifest.json")
    if manifest.get("schema") != 1 or manifest.get("kind") != REPORT_KIND or expected is not None and manifest != expected:
        raise ValueError("registration report generation identity collision")
    if (directory / "COMPLETED").read_text().strip() != _hash(data):
        raise ValueError("corrupt registration report completion seal")
    _checked_files(directory, manifest, {"report.ipynb", "report_snapshot.json", "analysis.json", *(n + ".png" for n, _ in FIGURES)})
    return manifest


def _previous(root, snapshot, history):
    if not (root / "report/latest.json").exists():
        return None
    manifest, _ = _load(root / "report/latest.json")
    directory = Path(manifest.get("report_dir", "")).resolve()
    if directory.parent != root / "report":
        raise ValueError("latest generation escapes registration root")
    _verify_generation(directory, manifest)
    old, _ = _load(directory / "report_snapshot.json")
    if manifest.get("root") != str(root) or old["study_id"] != snapshot["study_id"] or old["config"] != snapshot["config"]:
        raise ValueError("previous report ownership/configuration mismatch")
    if manifest.get("historical_sha256") != history:
        raise ValueError("pinned historical report changed")
    if old.get("selection") and old["selection"] != snapshot.get("selection"):
        raise ValueError("frozen selection changed or disappeared")
    new = {t["unit_id"]: t for t in snapshot["trajectories"]}
    for t in old["trajectories"]:
        other = new.get(t["unit_id"])
        if t["completed"]:
            if t != other:
                raise ValueError("snapshot regresses or rewrites completed evidence")
        elif other is not None:
            # Live previews need not be checkpoint-durable. Sealed paths remain exact.
            identity = (set(t) | set(other)) - {"metrics", "summary", "completed"}
            if any(t.get(k) != other.get(k) for k in identity):
                raise ValueError("snapshot rewrites provisional trajectory identity")
    for name in ("phase6_active_hours", "active_hours", "total_family_active_hours"):
        before, after = old["progress"].get(name), snapshot["progress"].get(name)
        if before is not None and (after is None or after < before):
            raise ValueError("snapshot regresses recorded compute")
    if old["progress"].get("historical_active_hours", 14.318) != snapshot["progress"].get("historical_active_hours", 14.318):
        raise ValueError("snapshot changes preserved historical compute")
    return manifest


def _target_bytes(notebook, root, snapshot, historical, previous):
    if not notebook.exists():
        return None
    book, data = _read_book(notebook)
    marker = book.metadata.get(MARKER)
    if marker and (marker.get("root") != str(root) or marker.get("study_id") != snapshot["study_id"]):
        raise ValueError("target notebook belongs to another registration owner")
    candidates = []
    if historical is not None:
        candidates.append((historical / "report.ipynb").read_bytes())
    if previous is not None:
        candidates.append((Path(previous["report_dir"]) / "report.ipynb").read_bytes())
    if data not in candidates:
        raise ValueError("refusing to replace newer, modified or unrelated notebook content")
    return data


def build_report(root, notebook=None):
    """Publish an atomic, source-hashed immutable generation, without learning."""
    root = Path(root).resolve()
    snapshot, result, snapshot_bytes = _validated(root)
    smoke = snapshot["config"].get("smoke", False)
    notebook = Path(notebook).resolve() if notebook is not None else root / "registration_smoke.ipynb" if smoke else Path(__file__).resolve().parents[2] / "plan17_results.ipynb"
    if notebook.suffix != ".ipynb" or notebook.is_relative_to(root / "report"):
        raise ValueError("output must be an .ipynb outside immutable generations")
    if smoke and not notebook.is_relative_to(root):
        raise ValueError("smoke notebook must stay inside its own root")
    original, historical, history = _historical(snapshot)
    if historical is not None and (root.is_relative_to(historical.parent.parent) or notebook.is_relative_to(historical.parent.parent)):
        raise ValueError("registration must not write inside its historical root")
    previous = _previous(root, snapshot, history)
    expected_target = _target_bytes(notebook, root, snapshot, historical, previous)
    expected_latest = (root / "report/latest.json").read_bytes() if previous is not None else None
    identity = {"root": str(root), "study_id": snapshot["study_id"], "smoke": smoke, "notebook": str(notebook),
                "snapshot_sha256": _hash(snapshot_bytes), "sources_sha256": _sources(), "historical_sha256": history}
    destination = root / "report" / _generation(identity)
    staging = None
    try:
        if destination.exists():
            manifest = _verify_generation(destination)
            if any(manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("report generation source/identity collision")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".building-", dir=destination.parent))
            for (name, _), renderer in zip(FIGURES, (_context, _development, _fresh, _effects)):
                renderer(snapshot, result, staging / (name + ".png"))
            section = _section(snapshot, result, destination, notebook)
            for cell in section:
                if cell.cell_type == "code":
                    png = (staging / (cell.id.removeprefix(PREFIX) + ".png")).read_bytes()
                    cell.outputs = [nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode(), "text/plain": cell.id})]
            book = nbformat.v4.new_notebook(cells=[_markdown("title", "# Phase 6 Smoke Acceptance Only")]) if smoke else deepcopy(original)
            book.cells.insert(1, _progress(snapshot, result))
            book.cells.extend(section)
            book.metadata[MARKER] = {"schema": 1, "root": str(root), "study_id": snapshot["study_id"], "smoke": smoke,
                                     "generation": destination.name, "snapshot_sha256": identity["snapshot_sha256"],
                                     "status": _status(snapshot, result), "historical_report_dir": str(historical) if historical else None}
            nbformat.validate(book)
            _atomic(staging / "report.ipynb", nbformat.writes(book).encode())
            _atomic(staging / "report_snapshot.json", snapshot_bytes)
            _atomic(staging / "analysis.json", _json_bytes(result))
            manifest = {"schema": 1, "kind": REPORT_KIND, **identity, "report_dir": str(destination),
                        "manifest": str(root / "report/latest.json"), "status": _status(snapshot, result), "counts": result["counts"],
                        "files_sha256": {p.name: _hash(p.read_bytes()) for p in sorted(staging.iterdir())}}
            _atomic(staging / "manifest.json", _json_bytes(manifest))
            _atomic(staging / "COMPLETED", (_hash(_json_bytes(manifest)) + "\n").encode())
            _fsync_directory(staging)
            try:
                os.rename(staging, destination)
                _fsync_directory(destination.parent)
            except OSError:
                if not destination.is_dir():
                    raise
                _verify_generation(destination, manifest)
        _verify_generation(destination, manifest)
        # No stale renderer can replace a changed snapshot, target or latest pointer.
        latest = root / "report/latest.json"
        live_latest = latest.read_bytes() if latest.exists() else None
        if ((root / "report_snapshot.json").read_bytes() != snapshot_bytes or _sources() != identity["sources_sha256"]
                or _historical(snapshot)[2] != history or live_latest != expected_latest
                or _target_bytes(notebook, root, snapshot, historical, previous) != expected_target):
            raise ValueError("snapshot, sources, history or publication target changed during rendering")
        _publish(notebook, latest, (destination / "report.ipynb").read_bytes(), manifest)
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish artifact-only Phase 6 registration appendix")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
