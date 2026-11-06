"""Artifact-only Phase 7 appendix preserving every sealed Phase 6 cell.

Read only the compact report_snapshot.json, pinned historical report, and scalar
reporting sources. Completed paths are immutable; incomplete paths are explicitly
provisional previews which may roll back or disappear after process recovery.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
import numpy as np

from mnist_experiment.diagonal_lfu import report as report_helpers
from mnist_experiment.diagonal_lfu.report import (
    _atomic, _figure, _finish, _hash, _json_bytes, _load, _number, _table, _text,
)
from mnist_experiment.kalman_pi.report import _finite_json
from mnist_experiment.kalman_pi.registration import analysis as historical_analysis
from mnist_experiment.kalman_pi.registration import report as historical_report
from mnist_experiment.kalman_pi.retention.report import _publish, _read_book

from . import analysis

KIND = "plan17-discrepancy-snapshot"
REPORT_KIND = "plan17-discrepancy-report"
MARKER = "plan17_discrepancy_report"
PREFIX = "plan17-discrepancy-"
FIGURES = (("16_discrepancy_context", "Task And Readiness"),
           ("17_discrepancy_development", "Development And Observer Health"),
           ("18_discrepancy_fresh", "Fresh Tracking And Endpoint Quality"),
           ("19_discrepancy_effects", "Applied Weights, Discrepancy And Paired Effects"))
NAMES = {"fixed": "Fixed .025", "covariance": "Covariance", "velocity": "Velocity Kalman",
         "discrepancy": "Discrepancy Kalman"}
COLORS = {"fixed": "#3976a4", "covariance": "#817736", "velocity": "#278574", "discrepancy": "#b34c65"}
CONTEXTS = (("main", 25.), ("main", 45.), ("stationary", 0.))


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id=PREFIX + name)


def validate_snapshot(snapshot):
    """Validate scalar snapshot identity/content, returning recomputed analysis."""
    _finite_json(snapshot)
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1 or snapshot.get("kind") != KIND:
        raise ValueError("requires schema 1 kind plan17-discrepancy-snapshot")
    for name in ("study_id", "status", "phase"):
        if not isinstance(snapshot.get(name), str) or not snapshot[name]:
            raise ValueError(f"snapshot.{name} must be a nonempty string")
    for name in ("config", "progress", "readiness", "diagnostics"):
        if not isinstance(snapshot.get(name), dict):
            raise ValueError(f"snapshot.{name} must be an object")
    config = snapshot["config"]
    if type(config.get("smoke", False)) is not bool:
        raise ValueError("config.smoke must be boolean")
    if snapshot["progress"].get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
        raise ValueError("progress/snapshot ownership mismatch")
    selection = snapshot.get("selection")
    seed = (selection or {}).get("bootstrap_seed", config.get("bootstrap_seed", analysis.DEFAULT_SEED))
    result = analysis.summarize(snapshot.get("trajectories"), selection,
                                bootstrap_draws=config.get("bootstrap_draws", 2000), seed=seed, config=config)
    _finite_json(result)
    if snapshot.get("analysis") and snapshot["analysis"] != result:
        raise ValueError("snapshot.analysis differs from validated scalar analysis")
    return result


def _validated(root):
    snapshot, encoded = _load(root / "report_snapshot.json")
    return snapshot, validate_snapshot(snapshot), encoded


def _historical(snapshot):
    source = snapshot.get("historical_report_dir")
    if snapshot["config"].get("smoke", False):
        if source is not None or snapshot.get("historical_notebook_sha256") is not None:
            raise ValueError("smoke cannot inherit production history")
        return None, None, {}
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("historical_report_dir must pin an absolute sealed Phase 6 generation")
    directory = Path(source).resolve()
    manifest = historical_report._verify_generation(directory)
    book, encoded = _read_book(directory / "report.ipynb")
    sealed, sealed_bytes = _load(directory / "report_snapshot.json")
    meta = book.metadata.get("plan17_registration_report", {})
    if (manifest.get("report_dir") != str(directory) or manifest.get("smoke") is not False
            or sealed.get("kind") != "plan17-registration-snapshot" or sealed.get("schema") != 1
            or sealed.get("status") not in ("complete", "completed")
            or meta.get("schema") != 1 or meta.get("smoke") is not False
            or meta.get("status") != "Completed Phase 6" or meta.get("generation") != directory.name
            or meta.get("study_id") != manifest.get("study_id") or sealed.get("study_id") != manifest.get("study_id")
            or meta.get("snapshot_sha256") != _hash(sealed_bytes) or manifest.get("snapshot_sha256") != _hash(sealed_bytes)
            or snapshot.get("historical_notebook_sha256") != _hash(encoded) or MARKER in book.metadata):
        raise ValueError("history must be the pinned completed non-smoke Phase 6 report")
    required = {"plan17-conclusion", "plan17-retention-conclusion", "plan17-zero-ridge-conclusion", "plan17-registration-conclusion"}
    if not required <= {cell.id for cell in book.cells}:
        raise ValueError("history lacks inherited conclusions through Phase 6")
    for cell in book.cells:
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png is not None:
                try:
                    valid = base64.b64decode(png, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
                except (ValueError, TypeError) as error:
                    raise ValueError("corrupt inherited PNG") from error
                if not valid:
                    raise ValueError("corrupt inherited PNG")
    paths = [directory / "manifest.json", directory / "COMPLETED", *(directory / n for n in manifest["files_sha256"])]
    return book, directory, {str(p): _hash(p.read_bytes()) for p in paths}


def _counts(result):
    return {(schedule, angle): sum(g["n_blocks"] for g in result["groups"]
                                 if g["schedule"] == schedule and g["angle_max"] == angle)
            for schedule, angle in CONTEXTS}


def _status(snapshot, result):
    selected = snapshot.get("selection")
    if not selected:
        return "Pending calibration / Phase 7A--7C"
    counts = _counts(result)
    complete = snapshot["status"] in ("complete", "completed") and not result["counts"]["incomplete"]
    complete = complete and all(counts[(s, a)] == selected[s + "_replicas"] for s, a in CONTEXTS)
    return "Completed Phase 7" if complete else "Partial / incomplete Phase 7"


def _progress(snapshot, result):
    p, selection = snapshot["progress"], snapshot.get("selection") or {}
    current = p.get("current") if isinstance(p.get("current"), dict) else {}
    counts = _counts(result)
    hours = p.get("phase7_active_hours", p.get("active_hours"))
    if hours is None and p.get("active_seconds") is not None:
        hours = p["active_seconds"] / 3600
    storage = p.get("storage") or {}
    used = storage.get("used_bytes")
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    limit = storage.get("limit_bytes", storage.get("max_bytes"))
    free = storage.get("free_bytes", storage.get("filesystem_free_bytes"))
    state = f"{snapshot['phase']} / {snapshot['config'].get('protocol', snapshot['config'].get('revision', 'pending'))} / {snapshot['status']}"
    blocks = "; ".join(f"{('changing ' + str(int(a)) + ' deg') if s == 'main' else 'stationary'}: {counts[(s,a)]}/{selection.get(s + '_replicas', 'pending')}" for s, a in CONTEXTS)
    finished = _status(snapshot, result) == "Completed Phase 7"
    active = "none (all paths complete)" if finished else p.get('active_unit') or current.get('active_unit') or 'none'
    step = current.get('step', p.get('step'))
    accepted = ("all trajectories sealed" if finished else
                p.get('last_accepted_checkpoint') or current.get('last_accepted_checkpoint') or p.get('last_checkpoint')
                or (f"batch {step}" if step is not None else "not yet accepted"))
    text = ("## Current Phase 7 Progress\n\n"
            f"**{_text(state)}**  \nPaired blocks: {_text(blocks)}.  \n"
            f"Active unit: {_text(active)}; accepted checkpoint: {_text(accepted)}.  \n"
            f"Heartbeat at publication: {_text(p.get('heartbeat'))}.  \n"
            f"Phase 7 active hours: {_number(hours, 3)} / {_number(p.get('phase7_allocation_hours', p.get('allocation_hours', 20)), 3)}; "
            f"remaining forecast: {_number(p['forecast_seconds'] / 3600) if p.get('forecast_seconds') is not None else 'pending'} h.  \n"
            f"Preserved prior compute: {_number(p.get('historical_active_hours', 18.619), 3)} h. "
            "Storage used / ceiling / free: " + " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, limit, free)) + " GB.\n\n"
            f"{result['counts']['complete']} complete trajectories; {result['counts']['incomplete']} provisional; "
            f"{len(result['awaiting_full_block_units'])} completed arms await full blocks. "
            "Incomplete previews may roll back or disappear on resume and never enter paired inference. "
            "Heartbeat freshness is not a process-liveness check. Inherited progress panels are historical snapshots.")
    return _markdown("progress", text)


def _pending(ax, text="Pending sealed artifacts"):
    ax.text(.5, .5, text, ha="center", va="center", wrap=True, transform=ax.transAxes, fontsize=9)


def _context(snapshot, result, destination):
    ready, config = snapshot["readiness"], snapshot["config"]
    examples = ready.get("contexts") or ready.get("examples") or []
    with _figure(rows=2, cols=3) as (fig, axes):
        for ax, key, angle in zip(axes[0], ("reference", "query", "query"), (25, 25, 45)):
            example = next((e for e in examples if e.get("angle_max", e.get("angle")) == angle), examples[0] if examples else {})
            pixels = example.get(key)
            if pixels is not None:
                pixels = np.asarray(pixels, dtype=float)
                if pixels.shape != (40, 40) or not np.isfinite(pixels).all():
                    raise ValueError("padded readiness images must be finite 40x40 arrays")
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
                points = example.get(key + "_landmarks")
                if points is not None:
                    points = np.asarray(points, dtype=float)
                    if points.shape != (3, 2) or not np.isfinite(points).all():
                        raise ValueError("overlays require three finite physical-pixel landmarks")
                    ax.scatter(points[:, 0] + 19.5, points[:, 1] + 19.5, c="#dd5f5f", marker="+", s=45)
            else:
                _pending(ax, f"Pending {angle}-degree {key}")
            ax.set_title("Padded reference" if key == "reference" else f"Query: {angle}-degree excursion")
            ax.set_axis_off()
        law = ready.get("angle_law") or {}
        angle = law.get("angles")
        if angle is not None:
            for kind, style in (("proposed", "--"), ("accepted", "-"), ("density", "-")):
                densities = law.get(kind) or {}
                for label, density in densities.items():
                    axes[1, 0].plot(angle, density, style, label=f"{kind}: {label}")
        for example in examples:
            all_diagnostics = example.get("angle_diagnostics") or {}
            amplitude = example.get("angle_max")
            diagnostics = (all_diagnostics.get("shifted")
                           or all_diagnostics.get(f"shifted{amplitude:g}" if amplitude is not None else "shifted")
                           or all_diagnostics.get("diagnostic_shifted") or {})
            edges = diagnostics.get("angle_bin_edges")
            density = diagnostics.get("accepted_angle_density")
            if edges is not None and density is not None:
                x = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
                axes[1, 0].plot(x, density, label=f"Sampled endpoint {amplitude:g} deg")
            if amplitude is not None:
                mean = example.get("sign", 1) * amplitude
                limit, std = config.get("angle_limit", 70.), config.get("angle_std", 6.)
                x = np.linspace(-limit, limit, 400)
                normalizer = .5 * (math.erf((limit - mean) / (std * math.sqrt(2))) - math.erf((-limit - mean) / (std * math.sqrt(2))))
                pdf = np.exp(-.5 * ((x - mean) / std)**2) / (std * math.sqrt(2 * math.pi) * normalizer)
                axes[1, 0].plot(x, pdf, "--", label=f"Truncated law {amplitude:g} deg")
        if not axes[1, 0].has_data():
            _pending(axes[1, 0], "Pending recorded truncated densities / histograms")
        axes[1, 0].set(title="Actual angle laws", xlabel="Rotation (degrees)", ylabel="Density")
        h, r, m = config.get("hold_steps", 352), config.get("ramp_steps", 304), config.get("samples_per_step", 4)
        for amplitude, color in ((25, COLORS["velocity"]), (45, COLORS["discrepancy"])):
            for factor, style in ((1., "-"), (1.25, "--")):
                hold, ramp = math.ceil(h * factor), math.ceil(r * factor)
                outward = .5 - .5 * np.cos(np.linspace(0, np.pi, ramp))
                y = np.r_[np.zeros(hold), outward, outward[::-1], np.zeros(hold)] * amplitude
                axes[1, 1].plot(np.arange(1, len(y) + 1) * m, y, style, color=color, label=f"{amplitude} deg, L={factor:g}")
        for example in examples:
            schedule = example.get("schedule") or []
            if schedule:
                axes[1, 1].plot([row["arrivals"] for row in schedule], [abs(row["mu"]) for row in schedule],
                    ":", color="#555555", alpha=.5, label=f"Recorded {example.get('angle_max'):g} deg")
        axes[1, 1].set(title="Assigned location (sign mirrored across replicas)", xlabel="Incoming pairs", ylabel="Location magnitude (degrees)")
        probes = ready.get("probes") or []
        for p in probes:
            amplitude = p.get("angle_max", p.get("angle"))
            if amplitude is None:
                continue
            for name, flat, nested, offset, marker in (("Frozen", "shifted_risk", "frozen_shifted", -.5, "o"),
                                                       ("Capability fit", "capability_shifted_risk", "fitted_shifted", .5, "x")):
                value = p.get(flat, (p.get(nested) or {}).get("risk"))
                if value is not None:
                    axes[1, 2].scatter(amplitude + offset, math.sqrt(analysis._number(value, name, 0)), marker=marker, label=name)
        if not axes[1, 2].has_data():
            _pending(axes[1, 2], "Pending frozen and capability readiness")
        axes[1, 2].set(title="Development readiness", xlabel="Excursion (degrees)", ylabel="Endpoint RMS (pixels)")
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _development(snapshot, result, destination):
    selected = (snapshot.get("selection") or {}).get("arms", [])
    with _figure(rows=1, cols=3) as (fig, axes):
        for col, angle in enumerate((25., 45.)):
            for p in result["path_summaries"].values():
                if p["stage"] != "development" or p["schedule"] != "main" or p["angle_max"] != angle:
                    continue
                arm = p["arm"]
                axes[0, col].scatter(p["old_rms"], p["current_rms"], c=COLORS[arm], label=NAMES[arm], s=24)
                if any(a["condition"] == p["condition"] and a["ridge"] == p["ridge"] for a in selected):
                    axes[0, col].scatter(p["old_rms"], p["current_rms"], marker="*", s=95, edgecolor="#111111", facecolor="none", label="Selected package")
            if not axes[0, col].has_data():
                _pending(axes[0, col], "Pending completed development paths")
            axes[0, col].set(title=f"{angle:g} degrees: every candidate", xlabel="Nominal budget RMS (pixels)", ylabel="Current-law budget RMS (pixels)")
        for t in snapshot["trajectories"]:
            if t["stage"] != "development" or t["arm"] not in ("velocity", "discrepancy"):
                continue
            rows = [r for r in t["metrics"] if r.get("innovation_ratio") is not None]
            if rows:
                axes[0, 2].plot([r["arrivals"] for r in rows], [r["innovation_ratio"] for r in rows], color=COLORS[t["arm"]], alpha=.4, label=NAMES[t["arm"]])
        axes[0, 2].axhline(1, color="#777777", linestyle=":")
        axes[0, 2].set(title="Observer innovations (diagnostic)", xlabel="Incoming pairs", ylabel="Predictive innovation ratio")
        samples_per_step = snapshot["config"].get("samples_per_step", 4)
        for probe in snapshot["diagnostics"].get("anchor_audit", []):
            key, ratio = probe.get("key"), probe.get("ratio")
            if isinstance(key, list) and len(key) == 2 and ratio is not None:
                axes[0, 2].scatter(key[1] * samples_per_step, ratio, marker="x", color="#555555",
                    s=22, label="Anchor residual / sampling noise")
        if snapshot["diagnostics"].get("anchor_audit"):
            axes[0, 2].set_yscale("symlog", linthresh=1.)
            axes[0, 2].set_ylabel("Innovation / anchor-residual ratios")
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _eligible(snapshot, result, schedule, angle):
    ids = set(result["complete_units"]) - set(result["awaiting_full_block_units"])
    return [t for t in snapshot["trajectories"] if t["unit_id"] in ids and t["stage"] == "fresh"
            and t["schedule"] == schedule and t["angle_max"] == angle]


def _curves(ax, trajectories, metric, *, rms=False, percent=False, frozen=False):
    groups = defaultdict(list)
    for t in trajectories:
        if frozen and t["arm"] != "fixed":
            continue
        grid = tuple((r["step"], r["arrivals"]) for r in t["metrics"])
        groups[(t["cohort"], t["arm"], t["schedule_factor"], grid)].append(t)
    for (_, arm, factor, _), paths in sorted(groups.items()):
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
        label = ("Frozen initial" if frozen else NAMES[arm]) + f", L={factor:g}, n={len(paths)}"
        style = ":" if frozen else "-" if factor == 1 else "--"
        color = "#555555" if frozen else COLORS[arm]
        ax.plot(x, mean, style, color=color, label=label)
        if len(y) > 1 and not frozen:
            error = 1.96 * y.std(axis=0, ddof=1) / math.sqrt(len(y))
            ax.fill_between(x, mean - error, mean + error, color=color, alpha=.08)


def _regimes(ax, paths, snapshot):
    if paths and all(t["schedule"] == "stationary" for t in paths):
        return
    seen = set()
    for t in paths:
        h, r = analysis._horizon(t, snapshot.get("selection"), snapshot["config"])
        boundaries = analysis._boundaries(h, r)
        mapping = {row["step"]: row["arrivals"] for row in t["metrics"]}
        key = tuple(mapping[b] for b in boundaries)
        if key in seen:
            continue
        seen.add(key)
        for left, right in zip(key[1:3], key[2:4]):
            ax.axvspan(left, right, color="#777777", alpha=.035)
        for boundary in key[1:-1]:
            ax.axvline(boundary, color="#777777", linestyle=":" if t["schedule_factor"] == 1 else "--", linewidth=.6, alpha=.6)


def _fresh(snapshot, result, destination):
    with _figure(rows=3, cols=4) as (fig, axes):
        for row, (schedule, angle) in enumerate(CONTEXTS):
            paths = _eligible(snapshot, result, schedule, angle)
            name = f"{angle:g} degrees" if schedule == "main" else "Stationary"
            for col, metric in enumerate(("current_risk", "current_accuracy", "old_risk", "shifted_risk")):
                ax = axes[row, col]
                _curves(ax, paths, metric, rms=metric.endswith("risk"), percent=metric.endswith("accuracy"))
                _curves(ax, paths, "frozen_" + metric, rms=metric.endswith("risk"), percent=metric.endswith("accuracy"), frozen=True)
                _regimes(ax, paths, snapshot)
                if not ax.has_data():
                    _pending(ax, "Pending complete fresh blocks")
                target = {"current_risk": "current-law RMS", "current_accuracy": "current-law accuracy",
                          "old_risk": "nominal RMS", "shifted_risk": "endpoint RMS"}[metric]
                if schedule == "stationary" and metric == "shifted_risk":
                    target = f"{snapshot['config'].get('amplitudes', [25, 45])[0]:g}-degree endpoint RMS"
                title = name + (":\n" if schedule == "stationary" and metric == "shifted_risk" else ": ") + target
                ax.set(title=title, xlabel="Incoming pairs", ylabel="Within 1 pixel (%)" if metric.endswith("accuracy") else "RMS (pixels)")
                ax.set_ylim(bottom=0, top=100 if metric.endswith("accuracy") else None)
        for col in range(4):
            upper = max(ax.get_ylim()[1] for ax in axes[:, col])
            for ax in axes[:, col]:
                ax.set_ylim(0, upper)
        _finish(axes)
        handles, labels = [], []
        for ax in axes.flat:
            h, labels_ = ax.get_legend_handles_labels()
            for handle, label in zip(h, labels_):
                short = label.split(", n=")[0]
                if short not in labels:
                    handles.append(handle)
                    labels.append(short)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=7)
        fig.savefig(destination, dpi=135)


def _effects(snapshot, result, destination):
    with _figure(rows=3, cols=3) as (fig, axes):
        for row, (schedule, angle) in enumerate(CONTEXTS):
            paths = _eligible(snapshot, result, schedule, angle)
            label = f"{angle:g} degrees" if schedule == "main" else "Stationary"
            _curves(axes[row, 0], paths, "applied_pi")
            _regimes(axes[row, 0], paths, snapshot)
            axes[row, 0].set(title=label + ": adaptation weight", xlabel="Incoming pairs", ylabel="pi", ylim=(0, .2))
            for arm, metric, style, title in (("velocity", "motion_energy", "-", "Velocity mean"),
                                            ("velocity", "uncertainty_energy", "--", "Velocity uncertainty"),
                                            ("discrepancy", "discrepancy_energy", "-", "Discrepancy mean"),
                                            ("discrepancy", "level_uncertainty_energy", "--", "Level uncertainty")):
                for t in paths:
                    if t["arm"] != arm:
                        continue
                    rows = [r for r in t["metrics"] if r.get(metric) is not None]
                    if rows:
                        axes[row, 1].plot([r["arrivals"] for r in rows], [r[metric] for r in rows], style, color=COLORS[arm], alpha=.45, label=title)
            _regimes(axes[row, 1], paths, snapshot)
            energy_ax = axes[row, 1]
            energy_ax.set_yscale("symlog", linthresh=1e-3)
            energy_ax.set_ylim(bottom=0)
            upper = energy_ax.get_ylim()[1]
            exponents = range(-3, math.ceil(math.log10(max(upper, 1e-3))) + 1, 2)
            energy_ax.set_yticks([0., *(10. ** exponent for exponent in exponents if 10. ** exponent <= upper)])
            energy_ax.set(title=label + ": separate signal terms", xlabel="Incoming pairs", ylabel="Energy")
            comparisons = [c for g in result["groups"] if g["schedule"] == schedule and g["angle_max"] == angle for c in g["comparisons"]]
            for i, comparison in enumerate(comparisons):
                effect = comparison["effects"]["current_rms_gap"]
                if effect["mean"] is None:
                    continue
                if effect["ci95"] is not None:
                    axes[row, 2].plot(effect["ci95"], [i, i], color=COLORS[comparison["arm"]])
                axes[row, 2].plot(effect["mean"], i, "o", color=COLORS[comparison["arm"]])
            if comparisons:
                axes[row, 2].set_yticks(range(len(comparisons)), [NAMES[c["arm"]] + " vs " + NAMES[c["control"]] for c in comparisons], fontsize=6)
            else:
                _pending(axes[row, 2], "Pending paired effects")
            axes[row, 2].axvline(0, color="#777777", linewidth=.7)
            axes[row, 2].set(title=label + ": current-law paired gaps", xlabel="Policy minus control RMS (pixels)\nNegative favors first policy")
        _finish(axes)
        for col in (0, 1):
            handles, labels = [], []
            for ax in axes[:, col]:
                h, names = ax.get_legend_handles_labels()
                for handle, name in zip(h, names):
                    label = name.split(", n=")[0]
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            if handles:
                fig.legend(handles, labels, loc="outside lower left" if col == 0 else "outside upper right", ncols=2, fontsize=7)
        fig.savefig(destination, dpi=135)


def _settings(snapshot):
    selection, config = snapshot.get("selection") or {}, snapshot["config"]
    arms = "; ".join(f"{NAMES[a['arm']]}: {a['condition']}, r={a['ridge']:g}" for a in selection.get("arms", [])) or "pending calibration"
    rows = [("Task / likelihood", "40x40 pair; three virtual landmarks; 448 parameters; Gaussian covariance 4 I6"),
            ("Policies / shared ridge", arms),
            ("Base hold / ramp; pairs per batch", f"{selection.get('hold_steps', config.get('hold_steps', 352))} / {selection.get('ramp_steps', config.get('ramp_steps', 304))}; {config.get('samples_per_step', 4)}"),
            ("Amplitudes / stationary cohort", "25 and 45 degrees, paired by replica; one stationary cohort, not duplicated by endpoint"),
            ("Current-law evaluation", "Independent official-test holdout after each saved batch; changing law, fixed sources/uniforms"),
            ("Calibration / inference", f"Development current-law squared error; common ridge, no retention matching; {config.get('bootstrap_draws', 2000)} whole-replica bootstrap draws")]
    return _markdown("settings", "### Settings\n\n" + _table(("Setting", "Recorded value"), rows))


def _reading(snapshot, result):
    if snapshot["config"].get("smoke", False):
        return "**Smoke acceptance only; not production evidence.**"
    statements = []
    for group in result["groups"]:
        if group["schedule"] != "main" or not group["n_blocks"]:
            continue
        findings = []
        for arm in ("velocity", "discrepancy"):
            contrast = next(c for c in group["comparisons"] if c["arm"] == arm and c["control"] == "fixed")
            effects = contrast["effects"]
            ci = effects["current_rms_gap"]["ci95"]
            if ci is not None and ci[1] < 0:
                finding = NAMES[arm] + " improves whole-path tracking"
                ramp_intervals = [effects[k + "_current_rms_gap"]["ci95"] for k in ("rise", "return")]
                if all(interval is not None and interval[1] < 0 for interval in ramp_intervals):
                    finding += " and both ramps"
            elif ci is not None and ci[0] > 0:
                finding = NAMES[arm] + " worsens whole-path tracking"
            else:
                finding = NAMES[arm] + " has an unresolved tracking difference"
            findings.append(finding)
        statements.append(f"At {group['angle_max']:g} degrees, " + "; ".join(findings) + ".")
    return " ".join(statements) or "Pending complete fresh blocks; no efficacy conclusion is available."


def _point(effect, *, signed=False, digits=4):
    value = effect.get("mean")
    return (format(value, f"{'+' if signed else ''}.{digits}f")
            if value is not None else "pending")


def _conclusion(snapshot, result):
    rows = []
    for group in result["groups"]:
        context = f"{group['angle_max']:g} deg" if group["schedule"] == "main" else "Stationary"
        for c in group["comparisons"]:
            if c["control"] != "fixed":
                continue
            effect = c["effects"]
            endpoint = _point(effect["shifted_rms_gap"], signed=True)
            if group["schedule"] == "stationary" and "shifted25_rms_gap" in effect and "shifted45_rms_gap" in effect:
                endpoint = "; ".join(f"{angle}: " + _point(effect[f"shifted{angle}_rms_gap"], signed=True)
                                     for angle in (25, 45))
            rows.append((context + ": " + NAMES[c["arm"]], c["n_pairs"],
                         historical_report._effect(effect["current_rms_gap"], signed=True),
                         _point(effect["old_rms_gap"], signed=True),
                         endpoint,
                         _point(effect["learning_seconds_ratio"], digits=3),
                         _point(effect["online_wall_seconds_ratio"], digits=3)))
    if not rows:
        rows = [("Pending complete fresh blocks", 0, "pending", "pending", "pending", "pending", "pending")]
    headers = ("Policy vs fixed .025", "Blocks", "Current RMS gap [95% CI], px", "Nominal gap, px", "Endpoint gap, px", "Learning ratio", "Wall ratio")
    diagnostic = ("\n\nThe fixed-anchor consistency checks are poor, and discrepancy adapts aggressively even on stationary data. "
        "This supports concern about its score-to-parameter proxy, not a unique causal explanation or a refutation of the conditional risk identity."
        if snapshot.get("diagnostics", {}).get("anchor_audit") else "")
    return _markdown("conclusion", "### Phase 7 Interpretation\n\n" + _reading(snapshot, result) + "\n\n" + _table(headers, rows) +
        "\n\nAll rows compare against fixed .025 at the shared calibrated ridge. Negative RMS gaps favor the policy; ratios below one are cheaper. "
        "Secondary gaps and costs are point estimates; their full intervals and all 15 contrasts remain in the linked scalar results. "
        "Stationary endpoint entries identify the 25/45-degree panels.\n\n"
        "**Current-law budget RMS measures typical landmark error while following the changing stream; lower is better.** "
        "Integrate squared independent holdout errors over incoming-pair counts, then root within each replica. "
        "Nominal/endpoint gaps reveal trade-offs: retention was not matched. Stationary endpoint error is untrained-domain generalization." + diagnostic +
        "\n\nIntervals are nominal 95% whole-replica bootstrap intervals, conditional on development selection and exploratory multiple contrasts. "
        "Paired amplitudes share draws; small cohorts and descriptive checkpoint bands limit inference. Nonsignificance is not equivalence. "
        "Costs are equal-data, not compute-matched. These results do not establish LAN, global optimality, or superiority to every constant pi. "
        "No samples-saved or S statistic is inferred from these nonstationary curves.")


def _section(snapshot, result, destination, notebook):
    cells = [_markdown("section", "## Phase 7: Tracking Discrepancy During Sustained Drift\n\n"
        "Phase 6 reacted to distribution changes without establishing a joint predictive benefit. Its ramps occupied only about 20% of arrivals, "
        "and population velocity omitted accumulated learner error. This phase lengthens uninterrupted drift and estimates the remaining target-to-learner discrepancy. "
        "The goal is understanding efficient continual learning with limited observations and compute. "
        "A padded renderer permits both 25- and 45-degree excursions; current-distribution holdouts reveal tracking between endpoints.\n\n"
        "**Current-law budget RMS measures typical landmark error while following the changing stream; lower is better.** "
        "Every evaluation is an independent post-batch holdout, never training-batch loss or prequential prediction. "
        "The discrepancy numerator includes predicted-level uncertainty and no extra old-variance q term. Local quadratic and fixed-anchor assumptions remain approximations.\n\n"
        "**" + _status(snapshot, result) + ".** " + ("Smoke acceptance is not production evidence. " if snapshot["config"].get("smoke", False) else "") +
        "This notebook reads artifacts only."), _settings(snapshot)]
    captions = ("Recorded padded examples and angle laws; schedules use the same incoming-pair clock as every trajectory. Readiness is development evidence.",
                "All completed development candidates remain visible. Current-law squared error selects settings; neither innovation scores nor anchor checks establish latent-parameter truth.",
                "Complete four-policy blocks only. Solid/dashed curves denote length factors 1/1.25; no unmatched grids are interpolated. Frozen initial-model curves are dotted and evaluated on each current-law panel. Vertical lines and shaded ramps show changing-schedule timing; stationary controls have no drift shading. Bands are descriptive mean +/-1.96 SE.",
                "Actions omit the action-free initial checkpoint. Fisher-weighted signal energies use a symmetric-log scale (linear within 0.001 of zero) so velocity, discrepancy and uncertainty remain separately visible. Effect bars are nominal whole-replica paired 95% intervals; negative gaps favor the first policy.")
    for (name, title), caption in zip(FIGURES, captions):
        cells.append(_markdown(name + "-heading", "### " + title + "\n\n" + caption))
        relative = os.path.relpath(destination / (name + ".png"), notebook.parent)
        cells.append(nbformat.v4.new_code_cell("from IPython.display import Image, display\n" + f"display(Image(filename={relative!r}))",
            id=PREFIX + name, execution_count=1, outputs=[]))
    cells.append(_conclusion(snapshot, result))
    if not snapshot["config"].get("smoke", False) and _status(snapshot, result) == "Completed Phase 7":
        cells.append(_markdown("application", "### Phase 7 Conclusion: Adaptation When Tuning Trials Are Unavailable\n\n"
            "**Velocity Kalman is a promising way to choose adaptation weights from one continuing data stream when repeated tuning trials are unavailable.** "
            "Once configured, it uses past score observations to estimate movement and noise and chooses each weight before the next batch. "
            "Deployment needs neither competing learner runs nor holdout feedback. This fits continual learning with limited data and compute, "
            "where finding a good fixed weight through repeated training may be impractical.\n\n"
            "Phase 7 provides encouraging evidence: modest current-distribution tracking gains over fixed .025, including both ramp directions, "
            "with slightly lower measured online wall time. Automatically choosing a useful average weight could itself be valuable; "
            "the benefit need not come entirely from precisely timed changes.\n\n"
            "The observer settings were selected on development trajectories, so this is a calibrated recipe whose transfer to unfamiliar streams remains to be tested. "
            "The present comparisons do not separate average-weight selection from adaptive timing, and fixed endpoint prediction worsened. "
            "The practical opportunity is to reduce dependence on unavailable tuning trials while continuing to measure both tracking and retention."))
    cells.append(_markdown("artifacts", "[Sealed Phase 7 snapshot](" + os.path.relpath(destination / "report_snapshot.json", notebook.parent) +
        ") | [All scalar outcomes, regimes and paired amplitude effects](" + os.path.relpath(destination / "analysis.json", notebook.parent) +
        "). The four figures are embedded; every inherited Phase 6 cell, output and metadata field is preserved."))
    return cells


def _sources():
    files = {Path(__file__).resolve(), Path(analysis.__file__).resolve(), Path(historical_analysis.__file__).resolve(),
             Path(historical_report.__file__).resolve(), Path(report_helpers.__file__).resolve()}
    package = Path(__file__).resolve().parents[1]
    files.update(package / p for p in ("report.py", "analysis.py", "retention/report.py", "retention/analysis.py"))
    files.add(package.parent / "diagonal_lfu/analysis.py")
    return {str(p): _hash(p.read_bytes()) for p in sorted(files)}


def _generation(identity):
    return "generation-" + _hash(_json_bytes(identity))[:24]


def _verify_generation(directory, expected=None):
    directory = Path(directory).resolve()
    manifest, encoded = _load(directory / "manifest.json")
    if (manifest.get("schema") != 1 or manifest.get("kind") != REPORT_KIND
            or expected is not None and manifest != expected or manifest.get("report_dir") != str(directory)):
        raise ValueError("discrepancy report generation identity collision")
    if (directory / "COMPLETED").read_text().strip() != _hash(encoded):
        raise ValueError("corrupt discrepancy report completion seal")
    historical_report._checked_files(directory, manifest,
        {"report.ipynb", "report_snapshot.json", "analysis.json", *(name + ".png" for name, _ in FIGURES)})
    return manifest


def _previous(root, snapshot, history):
    pointer = root / "report/latest.json"
    if not pointer.exists():
        return None
    manifest, _ = _load(pointer)
    directory = Path(manifest.get("report_dir", "")).resolve()
    if directory.parent != root / "report":
        raise ValueError("latest generation escapes discrepancy root")
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
            identity = (set(t) | set(other)) - {"metrics", "summary", "completed"}
            if any(t.get(k) != other.get(k) for k in identity):
                raise ValueError("snapshot rewrites provisional trajectory identity")
    for name in ("phase7_active_hours", "active_hours", "active_seconds", "total_family_active_hours"):
        before, after = old["progress"].get(name), snapshot["progress"].get(name)
        if before is not None and (after is None or after < before):
            raise ValueError("snapshot regresses recorded compute")
    if old["progress"].get("historical_active_hours", 18.619) != snapshot["progress"].get("historical_active_hours", 18.619):
        raise ValueError("snapshot changes preserved historical compute")
    return manifest


def _target_bytes(notebook, root, snapshot, historical, previous):
    if not notebook.exists():
        return None
    book, encoded = _read_book(notebook)
    marker = book.metadata.get(MARKER)
    if marker and (marker.get("root") != str(root) or marker.get("study_id") != snapshot["study_id"]):
        raise ValueError("target notebook belongs to another discrepancy owner")
    candidates = []
    if historical is not None:
        candidates.append((historical / "report.ipynb").read_bytes())
    if previous is not None:
        candidates.append((Path(previous["report_dir"]) / "report.ipynb").read_bytes())
    if encoded not in candidates:
        raise ValueError("refusing to replace newer, modified or unrelated notebook content")
    return encoded


def build_report(root, notebook=None):
    """Atomically publish a source-hashed generation without executing models."""
    root = Path(root).resolve()
    snapshot, result, encoded = _validated(root)
    smoke = snapshot["config"].get("smoke", False)
    notebook = Path(notebook).resolve() if notebook is not None else root / "discrepancy_smoke.ipynb" if smoke else Path(__file__).resolve().parents[2] / "plan17_results.ipynb"
    if notebook.suffix != ".ipynb" or notebook.is_relative_to(root / "report"):
        raise ValueError("output must be an .ipynb outside immutable generations")
    if smoke and not notebook.is_relative_to(root):
        raise ValueError("smoke notebook must stay inside its own root")
    original, historical, history = _historical(snapshot)
    if historical is not None and (root.is_relative_to(historical.parent.parent) or notebook.is_relative_to(historical.parent.parent)):
        raise ValueError("discrepancy report cannot write inside its historical root")
    previous = _previous(root, snapshot, history)
    expected_target = _target_bytes(notebook, root, snapshot, historical, previous)
    latest = root / "report/latest.json"
    expected_latest = latest.read_bytes() if previous else None
    identity = {"root": str(root), "study_id": snapshot["study_id"], "smoke": smoke, "notebook": str(notebook),
                "snapshot_sha256": _hash(encoded), "sources_sha256": _sources(), "historical_sha256": history}
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
            book = nbformat.v4.new_notebook(cells=[_markdown("title", "# Phase 7 Smoke Acceptance Only")]) if smoke else deepcopy(original)
            book.cells.insert(1, _progress(snapshot, result))
            book.cells.extend(section)
            book.metadata[MARKER] = {"schema": 1, "root": str(root), "study_id": snapshot["study_id"], "smoke": smoke,
                "generation": destination.name, "snapshot_sha256": identity["snapshot_sha256"], "status": _status(snapshot, result),
                "historical_report_dir": str(historical) if historical else None}
            nbformat.validate(book)
            _atomic(staging / "report.ipynb", nbformat.writes(book).encode())
            _atomic(staging / "report_snapshot.json", encoded)
            _atomic(staging / "analysis.json", _json_bytes(result))
            manifest = {"schema": 1, "kind": REPORT_KIND, **identity, "report_dir": str(destination),
                "manifest": str(latest), "status": _status(snapshot, result), "counts": result["counts"],
                "files_sha256": {p.name: _hash(p.read_bytes()) for p in sorted(staging.iterdir())}}
            _atomic(staging / "manifest.json", _json_bytes(manifest))
            _atomic(staging / "COMPLETED", (_hash(_json_bytes(manifest)) + "\n").encode())
            historical_report._fsync_directory(staging)
            try:
                os.rename(staging, destination)
                historical_report._fsync_directory(destination.parent)
            except OSError:
                if not destination.is_dir():
                    raise
                _verify_generation(destination, manifest)
        _verify_generation(destination, manifest)
        if ((root / "report_snapshot.json").read_bytes() != encoded or _sources() != identity["sources_sha256"]
                or _historical(snapshot)[2] != history or (latest.read_bytes() if latest.exists() else None) != expected_latest
                or _target_bytes(notebook, root, snapshot, historical, previous) != expected_target):
            raise ValueError("snapshot, sources, history or publication target changed during rendering")
        _publish(notebook, latest, (destination / "report.ipynb").read_bytes(), manifest)
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish artifact-only Plan 17 Phase 7 tracking report")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
