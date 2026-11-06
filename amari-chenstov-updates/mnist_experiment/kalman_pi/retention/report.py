"""Bounded Phase 5 publication from sealed scalar artifacts only.

Production extends an immutable original report notebook, never its renderer.
No execution, data, checkpoint, torch, or inference modules are imported.
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

from mnist_experiment.diagonal_lfu.report import (
    MAX_JSON_BYTES, _atomic, _change_brief, _curve, _effect, _figure,
    _finish, _hash, _json_bytes, _load, _number, _table, _text,
)
from mnist_experiment.kalman_pi.report import (
    FIGURES as HISTORICAL_FIGURES, _color, _finite_json, _name,
)

from . import analysis

KIND = "plan17-retention-snapshot"
ANALYSIS_KIND = "plan17-retention-analysis"
MARKER = "plan17_retention_report"
CONDITIONS = ("fixed", "covariance", "ema_h16", "kalman_eta0.0001")
SCHEDULES = ("main", "stationary")
FIGURES = (
    ("07_retention_development", "Development: Acquisition And Retention"),
    ("08_retention_fresh", "Fresh Accuracy By Schedule And Length"),
    ("09_retention_effects", "Paired Whole-Path Effects"),
)


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id="plan17-retention-" + name)


def _schedule(name):
    return "Changing" if name == "main" else "Stationary"


def _stat(row, name):
    value = row.get(name)
    return value if isinstance(value, dict) else {
        "mean": value, "ci95": row.get(name + "_ci95", row.get(name.removesuffix("_pp") + "_ci95")),
    }


def _selection(snapshot):
    selected = snapshot["selection"]
    conditions = selected.get("conditions", list(CONDITIONS))
    if (not isinstance(conditions, list) or len(conditions) != len(CONDITIONS)
            or set(conditions) != set(CONDITIONS)):
        raise ValueError("selection.conditions must contain the four Phase 5 policies")
    for key in ("ridges", "development_selection", "development_replicas"):
        if not isinstance(selected.get(key, {}), dict):
            raise ValueError(f"selection.{key} must be an object")
    for schedule, ridges in selected.get("ridges", {}).items():
        if schedule not in SCHEDULES or not isinstance(ridges, dict):
            raise ValueError("selection.ridges requires separate main/stationary maps")
        for condition, ridge in ridges.items():
            if (condition not in CONDITIONS or isinstance(ridge, bool)
                    or not isinstance(ridge, (int, float)) or ridge < 0
                    or (condition == "fixed" and ridge != .1)):
                raise ValueError("invalid selected ridge or changed fixed reference")
    for name in ("main_replicas", "stationary_replicas"):
        if name in selected and (type(selected[name]) is not int or selected[name] < 0):
            raise ValueError(f"selection.{name} must be a nonnegative integer")


def _validated(root):
    snapshot, data = _load(root / "report_snapshot.json")
    _finite_json(snapshot)
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1 or snapshot.get("kind") != KIND:
        raise ValueError("report_snapshot.json requires schema 1 kind plan17-retention-snapshot")
    for name in ("study_id", "status", "phase"):
        if not isinstance(snapshot.get(name), str) or not snapshot[name]:
            raise ValueError(f"snapshot.{name} must be a nonempty string")
    for name in ("config", "progress", "selection", "analysis"):
        if not isinstance(snapshot.get(name), dict):
            raise ValueError(f"snapshot.{name} must be an object")
    for name in ("changes", "trajectories"):
        if not isinstance(snapshot.get(name), list):
            raise ValueError(f"snapshot.{name} must be a list")
    if snapshot["progress"].get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
        raise ValueError("progress and snapshot study_id mismatch")
    if "smoke" in snapshot["config"] and type(snapshot["config"]["smoke"]) is not bool:
        raise ValueError("config.smoke must be boolean")
    _selection(snapshot)
    # Validation belongs before both cached-analysis acceptance and plotting.
    result = analysis.summarize(
        snapshot["trajectories"], snapshot["selection"],
        bootstrap_draws=snapshot["config"].get("bootstrap_draws", 2000),
        seed=snapshot["selection"].get("bootstrap_seed", snapshot["config"].get("bootstrap_seed", analysis.DEFAULT_SEED)),
    )
    if type(result.get("schema")) is not int or result["schema"] != 1 or result.get("kind") != ANALYSIS_KIND:
        raise ValueError("incompatible Phase 5 analysis schema/kind")
    _finite_json(result)
    if snapshot["analysis"] and snapshot["analysis"] != result:
        raise ValueError("snapshot.analysis is corrupt or incompatible with trajectories/selection")
    return snapshot, result, data


def _read_book(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"report notebook exceeds bounded artifact size: {path}")
    data = path.read_bytes()
    try:
        book = nbformat.reads(data.decode(), as_version=4)
        nbformat.validate(book)
    except (ValueError, UnicodeDecodeError, nbformat.ValidationError) as exc:
        raise ValueError(f"invalid report notebook: {path}") from exc
    if len({cell.id for cell in book.cells}) != len(book.cells):
        raise ValueError("duplicate report cell identities")
    return book, data


def _historical(snapshot):
    source = snapshot.get("historical_report_dir")
    if not source and snapshot["config"].get("smoke"):
        return None, None, {}
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("historical_report_dir must be an absolute immutable report generation directory")
    directory = Path(source).resolve()
    book, data = _read_book(directory / "report.ipynb")
    if snapshot.get("historical_notebook_sha256", _hash(data)) != _hash(data):
        raise ValueError("pinned historical notebook hash mismatch")
    meta = book.metadata.get("plan17_report", {})
    if meta.get("schema") != 1 or meta.get("smoke") is not False or MARKER in book.metadata:
        raise ValueError("historical report must be an original production generation, not smoke or combined")
    manifest, manifest_bytes = _load(directory / "manifest.json")
    if (manifest.get("schema") != 1 or manifest.get("snapshot_sha256") != meta.get("snapshot_sha256")
            or Path(manifest.get("report_dir", "")).resolve() != directory):
        raise ValueError("historical report manifest identity mismatch")
    cells = {cell.id: cell for cell in book.cells}
    placeholder = cells.get("plan17-retention-followup")
    if (placeholder is None or placeholder.cell_type != "markdown"
            or "not started" not in placeholder.source.lower()):
        raise ValueError("historical report requires the pending plan17-retention-followup placeholder")
    if ("plan17-conclusion" not in cells or "plan17-progress" not in cells
            or book.cells.index(cells["plan17-conclusion"]) >= book.cells.index(placeholder)):
        raise ValueError("historical report is missing its original conclusion/progress")
    hashes = {str(directory / "report.ipynb"): _hash(data),
              str(directory / "manifest.json"): _hash(manifest_bytes)}
    for name, _ in HISTORICAL_FIGURES:
        cell = cells.get("plan17-" + name)
        outputs = cell.get("outputs", []) if cell else []
        if cell is None or cell.cell_type != "code" or len(outputs) != 1:
            raise ValueError(f"historical report missing figure output: {name}")
        try:
            png = base64.b64decode(outputs[0]["data"]["image/png"], validate=True)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid historical figure output: {name}") from exc
        path = directory / (name + ".png")
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or path.read_bytes() != png:
            raise ValueError(f"historical figure artifact/output mismatch: {name}")
        hashes[str(path)] = _hash(png)
    return book, directory, hashes


def _progress(snapshot, result):
    p, selection = snapshot["progress"], snapshot["selection"]
    rows = [("Phase / status", snapshot["phase"] + " / " + snapshot["status"])]
    for schedule in SCHEDULES:
        development = [t for t in snapshot["trajectories"] if t["schedule"] == schedule and t["stage"] == "development"]
        blocks = sum(len(g["eligible_replicas"]) for g in result["groups"] if g["schedule"] == schedule)
        awaiting = sum(len(g["awaiting_full_block_units"]) for g in result["groups"] if g["schedule"] == schedule)
        rows.append((_schedule(schedule) + " development / fresh",
                     f"{sum(t['completed'] for t in development)} complete development paths; "
                     f"{selection.get('development_replicas', {}).get(schedule, 'pending')} planned development replicas; "
                     f"{blocks} / {selection.get(schedule + '_replicas', 'pending')} fresh blocks; {awaiting} arms awaiting full block"))
    complete_development = sum(t["completed"] and t["stage"] == "development" for t in snapshot["trajectories"])
    complete_fresh = sum(t["completed"] and t["stage"] == "fresh" for t in snapshot["trajectories"])
    rows.append(("Trajectories (complete / planned)",
                 f"Development {complete_development} / {p.get('planned_development', 'pending')}; "
                 f"fresh {complete_fresh} / {p.get('planned_fresh', selection.get('fresh_trajectory_target', 'pending'))}; "
                 f"all {result['counts']['complete']} / {p.get('planned_trajectories', 'pending')}"))
    current = p.get("current") or {}
    checkpoint = p.get("last_checkpoint") or p.get("last_accepted_checkpoint") or current.get("last_checkpoint") or current.get("last_accepted_checkpoint")
    storage = p.get("storage") or {}
    used = storage.get("used_bytes")
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    cap = storage.get("limit_bytes", storage.get("max_bytes"))
    forecast = p.get("forecast_seconds", selection.get("forecast_fresh_seconds"))
    rows.extend([
        ("Complete / incomplete paths", f"{result['counts']['complete']} / {result['counts']['incomplete']}"),
        ("Active unit / checkpoint", f"{p.get('active_unit') or current.get('active_unit') or 'none'} / {checkpoint or 'pending'}"),
        ("Accepted step / total", f"{current.get('step', p.get('step', 'pending'))} / {current.get('total_steps', p.get('total_steps', 'pending'))}"),
        ("Heartbeat (snapshot freshness)", p.get("heartbeat", "pending")),
        ("Cumulative active / allocated hours", f"{_number(p.get('active_hours', p.get('active_seconds', 0) / 3600))} / {_number(p.get('allocation_hours', p.get('allocation_seconds', 72000) / 3600))}"),
        ("Remaining forecast hours", _number(forecast / 3600) if forecast is not None else "calibrating"),
        ("Storage used / ceiling (GB)", " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, cap))),
        ("Filesystem free (GB)", _number(storage.get("free_bytes", storage.get("filesystem_free_bytes")) / 1e9) if storage.get("free_bytes", storage.get("filesystem_free_bytes")) is not None else "pending"),
    ])
    return _markdown("progress", "## Phase 5 Progress\n\n" + _table(("Item", "Recorded state"), rows) +
                     "\n\nSealed-artifact counts, not a liveness check. The original progress and summary below describe Phases 0--4; this panel supersedes their pending follow-up notice.")


def _development(snapshot, result, destination):
    with _figure(rows=1, cols=2) as (fig, axes):
        labels = []
        for ax, schedule in zip(axes.flat, SCHEDULES):
            ax.axvspan(-.5, .5, color="#dce7df", alpha=.65, label="Development band +/-0.5 pp")
            ax.axvline(0, color="#999999", linewidth=.8)
            ax.axhline(0, color="#999999", linewidth=.8)
            ax.plot([0], [0], "D", color=_color("fixed"), label="Fixed pi=.025; r=.1")
            for condition, candidates in result["development"].get(schedule, {}).items():
                rows = sorted((r for r in candidates if r.get("n_pairs", 0) and _stat(r, "old_gap_pp")["mean"] is not None
                               and _stat(r, "nine_gap_pp")["mean"] is not None), key=lambda r: r["ridge"])
                color = _color(condition)
                ax.plot([_stat(r, "old_gap_pp")["mean"] for r in rows], [_stat(r, "nine_gap_pp")["mean"] for r in rows],
                        color=color, alpha=.5, linewidth=.8)
                selected = snapshot["selection"].get("ridges", {}).get(schedule, {}).get(condition)
                for i, row in enumerate(rows):
                    x, y = _stat(row, "old_gap_pp"), _stat(row, "nine_gap_pp")
                    if x.get("ci95") is not None:
                        ax.plot(x["ci95"], [y["mean"]] * 2, color=color)
                    if y.get("ci95") is not None:
                        ax.plot([x["mean"]] * 2, y["ci95"], color=color)
                    chosen = condition != "fixed" and row["ridge"] == selected
                    marker = "D" if condition == "fixed" else "*" if chosen else "o"
                    ax.scatter(x["mean"], y["mean"], color=color, marker=marker,
                               s=95 if chosen else 22, label=_name(condition) if i == 0 and condition != "fixed" else None)
                    if condition != "fixed" and (chosen or row["ridge"] == 0):
                        labels.append((ax, x["mean"], y["mean"], f"r={row['ridge']:.4g}", color))
            ax.set(title=_schedule(schedule), xlabel="0--8 whole-path gap versus fixed (pp)", ylabel="9 recall whole-path gap versus fixed (pp)")
            if not any(r.get("n_pairs", 0) for rows in result["development"].get(schedule, {}).values() for r in rows):
                ax.text(.5, .9, "Pending complete development pairs", ha="center", transform=ax.transAxes)
        _finish(axes)
        # Move labels, never measured points, when ridge candidates coincide.
        from matplotlib.text import Text
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        occupied = defaultdict(list)
        for ax in axes.flat:
            if ax.get_legend() is not None:
                occupied[ax].append(ax.get_legend().get_window_extent(renderer))
        for ax, x, y, label, color in labels:
            annotation = ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=6,
                                     color=color, arrowprops={"arrowstyle": "-", "color": color, "linewidth": .4})
            for offset in ((dx, dy) for distance in range(6, 150, 12) for dy in (distance, -distance) for dx in (8, -72, 80, -144)):
                annotation.set_position(offset)
                annotation.update_positions(renderer)
                box = Text.get_window_extent(annotation, renderer).expanded(1.05, 1.15)
                if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                        and not any(box.overlaps(other) for other in occupied[ax])):
                    break
            occupied[ax].append(box)
        fig.savefig(destination, dpi=135)


def _fresh_arms(snapshot, result):
    by_id = {t["unit_id"]: t for t in snapshot["trajectories"]}
    seen = set()
    for group in result["groups"]:
        for comparison in group["comparisons"]:
            if comparison["control"] != "fixed" or not comparison["n_pairs"]:
                continue
            for key, condition in (("control_units", "fixed"), ("treatment_units", comparison["condition"])):
                ids = tuple(comparison[key])
                identity = (group["schedule"], condition, ids)
                if identity not in seen:
                    seen.add(identity)
                    yield group["schedule"], condition, [by_id[unit] for unit in ids]


def _fresh_accuracy(snapshot, result, destination):
    with _figure() as (fig, axes):
        fig.set_size_inches(11.6, 7.6)
        references = set()
        exposures = defaultdict(dict)
        for schedule, condition, arm in _fresh_arms(snapshot, result):
            cohorts = defaultdict(list)
            for t in arm:
                cohorts[(t.get("schedule_factor", 1), tuple(r["arrivals"] for r in t["metrics"]))].append(t)
            for (factor, grid), cohort in sorted(cohorts.items()):
                exposure = []
                for key in ("expected_nines", "observed_nines"):
                    values = [t["metrics"][-1].get(key) for t in cohort]
                    exposure.append(_number(float(np.mean(values)), 1) if all(v is not None for v in values) else "pending")
                note = f"; L={factor:g}, N={grid[-1]:g}"
                if condition == "fixed":
                    exposures[schedule][(factor, grid)] = f"L={factor:g}, N={grid[-1]:g}: online 9 exp/obs=" + "/".join(exposure)
                style = "-" if factor == 1 else "--"
                for col, metric in enumerate(("nine_recall", "old_accuracy")):
                    ax = axes[SCHEDULES.index(schedule), col]
                    _curve(ax, cohort, metric, _name(condition) + note, color=_color(condition), linestyle=style)
                    ref = (schedule, factor, grid, metric)
                    if condition == "fixed" and ref not in references:
                        references.add(ref)
                        initial = float(np.mean([t["metrics"][0][metric] for t in cohort])) * 100
                        ax.plot([0, grid[-1]], [initial, initial], ":", color="#555555", linewidth=.8,
                                label=f"Initial reference; L={factor:g}, N={grid[-1]:g}")
        for row, schedule in enumerate(SCHEDULES):
            xmax = max((line.get_xdata()[-1] for ax in axes[row] for line in ax.lines), default=1)
            for col, metric in enumerate(("9 recall", "0--8 macroaccuracy")):
                axes[row, col].set(title=_schedule(schedule) + ": " + metric,
                                   xlabel="Online digits observed", ylabel="Accuracy (%)", ylim=(0, 100), xlim=(0, xmax))
                axes[row, col].set_title(_schedule(schedule) + ": " + metric, pad=30)
                axes[row, col].text(0, 1.02, "\n".join(exposures[schedule].values()), fontsize=6,
                                        va="bottom", transform=axes[row, col].transAxes)
        _finish(axes)
        for left, right in axes:
            if left.get_legend() is not None:
                left.get_legend().remove()
            handles, labels = right.get_legend_handles_labels()
            if handles:
                right.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, .5), fontsize=6)
        fig.savefig(destination, dpi=135)


def _comparisons(result, *, primary=False):
    for group in result["groups"]:
        for comparison in group["comparisons"]:
            if comparison["n_pairs"] and (not primary or comparison["control"] == "fixed"):
                yield group, comparison


def _paired_effects(snapshot, result, destination):
    with _figure(rows=1, cols=2) as (fig, axes):
        pairs = list(_comparisons(result, primary=True))
        for ax, metric in zip(axes.flat, ("nine_recall", "old_accuracy")):
            if metric == "old_accuracy":
                ax.axvspan(-.5, .5, color="#dce7df", alpha=.65, label="Development band (not equivalence)")
            ax.axvline(0, color="#888888", linewidth=.8)
            for i, (group, comparison) in enumerate(pairs):
                value = comparison["observation_mean_pp"][metric]
                if value["mean"] is None:
                    continue
                color = _color(comparison["condition"])
                if value.get("ci95") is not None:
                    ax.plot(value["ci95"], [i, i], color=color)
                ax.plot([value["mean"]], [i], "o", color=color)
            ax.set_yticks(range(len(pairs)), [f"{_schedule(g['schedule'])}: {_name(c['condition'])} (n={c['n_pairs']})" for g, c in pairs], fontsize=7)
            ax.set(title="9 acquisition" if metric == "nine_recall" else "0--8 retention", xlabel="Whole-path paired difference versus fixed (pp)")
            if not pairs:
                ax.text(.5, .8, "Pending complete fresh four-arm blocks", ha="center", transform=ax.transAxes)
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _settings(snapshot):
    selection = snapshot["selection"]
    rows = []
    for schedule in SCHEDULES:
        for condition in CONDITIONS:
            record = selection.get("development_selection", {}).get(schedule, {}).get(condition, {}) or {}
            ridge = selection.get("ridges", {}).get(schedule, {}).get(condition)
            nonestimable = condition in selection.get("development_selection", {}).get(schedule, {}) and not record
            status = "reference" if condition == "fixed" else "non-estimable" if nonestimable else "pending" if ridge is None else "matched" if record.get("matched") else "unmatched"
            rows.append((_schedule(schedule), _name(condition), _number(ridge, 4),
                         _effect(_stat(record, "old_gap_pp"), signed=True), status))
    return _markdown("settings", "### Frozen Settings\n\n" + _table(("Schedule", "Policy", "Ridge", "Development old gap (pp)", "Status"), rows) +
                     f"\n\nFrozen at: {_text(selection.get('frozen_at'))}. Independently calibrated policy packages; no causal isolation of the observer. "
                     "Selection uses development whole-path old macroaccuracy within +/-0.5 pp, then greatest 9 recall; unmatched candidates remain trade-offs. "
                     "The band is an illustrative calibration resolution, not an application safety limit.")


def _results(result):
    rows = []
    for group, c in _comparisons(result, primary=True):
        costs = c.get("costs", {})
        time = costs.get("learning_seconds", costs.get("wall_seconds", {}))
        rows.append((_schedule(group["schedule"]), _name(c["condition"]), c["n_pairs"],
                     _effect(c["observation_mean_pp"]["nine_recall"], signed=True),
                     _effect(c["observation_mean_pp"]["old_accuracy"], signed=True), _effect(time.get("ratio"))))
    return _markdown("results", "### Fresh Paired Results\n\n" +
                     (_table(("Schedule", "Policy - fixed", "Pairs", "9 gap (pp), 95% CI", "Old gap (pp), 95% CI", "Measured time ratio, 95% CI"), rows)
                      if rows else "Pending: no complete fresh four-arm block. Completed unmatched arms and incomplete prefixes are not paired evidence.") +
                     "\n\nWhole-replica paired 95% bootstrap intervals use shared block draws across outcomes/contrasts, conditional on development-selected settings. "
                     "A missing interval is not a zero-width interval. No checkpoint resampling. Schedule cohorts remain separate.")


def _interpretation(snapshot, result):
    readings, reversals = defaultdict(list), defaultdict(list)
    for group, c in _comparisons(result, primary=True):
        nine, old = (c["observation_mean_pp"][m] for m in ("nine_recall", "old_accuracy"))
        selected = snapshot["selection"].get("development_selection", {}).get(group["schedule"], {}).get(c["condition"], {}) or {}
        if not selected.get("matched"):
            reading = "development target unmatched; acquisition remains a trade-off"
        elif abs(old["mean"]) > .5:
            acquisition = "lower" if nine["mean"] < 0 else "higher" if nine["mean"] > 0 else "unchanged"
            retention = "higher" if old["mean"] > 0 else "lower"
            reading = (f"fresh retention misses the development band; {acquisition} acquisition accompanies "
                       f"{retention} retention, not a matched-retention comparison")
        elif nine.get("ci95") is None or old.get("ci95") is None:
            reading = "uncertain result; paired uncertainty is not estimable yet"
        elif nine["ci95"][0] > 0:
            reading = ("acquisition gain at a plausibly comparable retention point; interval width still qualifies that comparison"
                       if -.5 <= old["ci95"][0] <= old["ci95"][1] <= .5 else
                       "acquisition gain, but comparable retention remains uncertain: the old-accuracy interval extends outside the development band")
        elif nine["ci95"][1] < 0:
            reading = "lower acquisition at a plausibly comparable retention point"
        else:
            reading = "uncertain acquisition difference at a plausibly comparable retention point"
        readings[(group["schedule"], reading)].append(_name(c["condition"]))
        for segment, values in c.get("segments", {}).items():
            for metric, estimate in values.get("observation_mean_pp", {}).items():
                whole = c["observation_mean_pp"].get(metric, {}).get("mean")
                if whole is not None and estimate.get("mean") is not None and whole * estimate["mean"] < 0:
                    reversals[(group["schedule"], segment, metric)].append(_name(c["condition"]))
    lines = [f"{_schedule(schedule)} ({', '.join(names)}): {reading}."
             for (schedule, reading), names in readings.items()]
    if reversals:
        lines.append("Contrary regime effects: " + "; ".join(
            f"{segment} {metric} ({_schedule(schedule)}; {', '.join(names)}) reverses the whole-path point-estimate sign"
            for (schedule, segment, metric), names in reversals.items()) + ". Regime intervals are in the linked analysis.")
    if not lines:
        lines.append("Pending fresh confirmation; no acquisition-at-comparable-retention conclusion is available.")
    costs = defaultdict(float)
    for t in snapshot["trajectories"]:
        if t["completed"] and t["stage"] == "development":
            for name, value in t["summary"].get("costs", {}).items():
                if value is not None:
                    costs[name] += value
    measured = (f"{costs['wall_seconds'] / 3600:.2f} h total path wall time, including "
                f"{costs.get('checkpoint_seconds', 0.) / 3600:.2f} h checkpoint I/O"
                if costs.get("wall_seconds") else "pending")
    return _markdown("conclusion", "### Phase 5 Interpretation\n\n" + "\n\n".join(lines) +
                     "\n\nA retention interval crossing zero is not evidence of equivalence; the +/-0.5 pp band guided development only. "
                     "Whole-path means can hide regime/class trade-offs. Intervals are exploratory under multiple comparisons, intermediate inspection and repeated MNIST use. "
                     "No confirmation retuning or combined accuracy score is used.\n\n" +
                     "Recorded development trajectory costs (separate from online confirmation costs above): " + measured +
                     ". Initialization and shared calibration are separate; their per-trajectory attribution must not be summed as additional compute. "
                     "The family ledger counts shared units once. "
                     "Repeated checkpoint I/O is a tiny-model timing limitation. Equal-exposure accuracy does not establish actual samples saved.")


def _section(snapshot, result, destination, notebook):
    smoke = snapshot["config"].get("smoke", False)
    pending = not snapshot["trajectories"] and snapshot["status"] in ("drafted", "planned", "pending")
    status = "**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.**" if smoke else "**Drafted, not started.**" if pending else "**" + snapshot["status"].title() + " artifact snapshot.**"
    cells = [_markdown("followup", "## Phase 5: Acquisition At Comparable Retention\n\n" + status +
                       "\n\nScarce-data continual learning needs new learning without buying it through greater forgetting. "
                       "The original equal-weight acquisition/retention score did not retention-match methods; common ridge did not imply common retention. "
                       "The results above remain the original common-ridge operating points, not invalidated or already explained by this follow-up.\n\n"
                       "**Question:** do independently calibrated adaptive policies improve digit-9 recall at comparable whole-path 0--8 macroaccuracy to fixed pi=.025, ridge=.1, and at what measured cost?\n\n"
                       "Match the whole-path development old-accuracy gap within +/-0.5 pp separately by schedule, then freeze ridge settings before fresh paired confirmation. "
                       "Report both metrics with 95% intervals; non-significance is not equivalence.")]
    cells.append(_settings(snapshot))
    captions = (
        "Every measured finite development candidate is shown, with paired 95% intervals when estimable; stars mark selected ridges. Only selected and zero-ridge points are annotated; all ridge values and replica counts remain in the linked analysis. Lines guide the eye, not interpolation or monotonicity claims. Numerical failures are not finite candidates.",
        "Only eligible complete four-arm fresh blocks enter these curves. Length/evaluation-grid cohorts are separate, with actual digit counts and expected/observed online 9 exposure in labels. Initial references are context, not trained again. Bands are descriptive pointwise mean +/-1.96 SE, not paired effect intervals; single replicas have no band. Initialization 9s are excluded from online exposure.",
        "Positive effects favor the calibrated package. Retention shading repeats the development matching band, not an equivalence region. Both outcomes use full-path trapezoidal means on the incoming-digit clock, including the shared initial checkpoint; whole replicas have equal weight regardless of horizon.",
    )
    for (name, title), caption in zip(FIGURES, captions):
        cells.append(_markdown(name + "-heading", "### " + title + "\n\n" + caption))
        path = destination / (name + ".png")
        cells.append(nbformat.v4.new_code_cell(
            "from IPython.display import Image, display\n" + f"display(Image(filename={os.path.relpath(path, notebook.parent)!r}))",
            id="plan17-retention-" + name, execution_count=len(cells), outputs=[]))
        if name == "09_retention_effects":
            cells.append(_results(result))
    if smoke:
        cells.append(_markdown("conclusion", "### Smoke Acceptance Only\n\nNo production efficacy or runtime conclusion is supported by these tiny artifacts."))
    else:
        cells.append(_interpretation(snapshot, result))
    cells.append(_markdown("artifacts", "[Sealed Phase 5 snapshot](" + os.path.relpath(destination / "report_snapshot.json", notebook.parent) +
                           ") | [Full regime/class diagnostics, costs and secondary calibrated-package contrasts](" + os.path.relpath(destination / "analysis.json", notebook.parent) +
                           "). PNG outputs are embedded; opening the notebook never trains, downloads, infers, or repairs artifacts. " +
                           " ".join(_change_brief(c) for c in snapshot["changes"][-3:])))
    return cells


def _source_hashes():
    package = Path(__file__).resolve().parents[1]
    files = (Path(__file__).resolve(), Path(analysis.__file__).resolve(), package / "report.py", package / "analysis.py",
             package.parent / "diagonal_lfu" / "report.py", package.parent / "diagonal_lfu" / "analysis.py")
    return {str(path): _hash(path.read_bytes()) for path in files}


def _verify_generation(directory, expected=None):
    manifest, data = _load(directory / "manifest.json")
    if expected is not None and manifest != expected:
        raise ValueError("report generation identity collision")
    if (directory / "COMPLETED").read_text().strip() != _hash(data):
        raise ValueError("corrupt report generation completion seal")
    for name, digest in manifest.get("files_sha256", {}).items():
        if Path(name).name != name or _hash((directory / name).read_bytes()) != digest:
            raise ValueError("corrupt report generation artifact: " + name)
    if not {"report.ipynb", "report_snapshot.json", "analysis.json", *(name + ".png" for name, _ in FIGURES)} <= manifest.get("files_sha256", {}).keys():
        raise ValueError("report generation is missing required artifact hashes")
    return manifest


def _previous_report(root, snapshot, historical_hashes):
    latest = root / "report" / "latest.json"
    if not latest.exists():
        return
    manifest, _ = _load(latest)
    directory = Path(manifest.get("report_dir", "")).resolve()
    if directory.parent != root / "report":
        raise ValueError("latest report generation escapes this study")
    _verify_generation(directory, expected=manifest)
    old, _ = _load(directory / "report_snapshot.json")
    if old["study_id"] != snapshot["study_id"] or old["config"].get("smoke", False) != snapshot["config"].get("smoke", False):
        raise ValueError("previous report belongs to another study or production/smoke cohort")
    if manifest["historical_sha256"] != historical_hashes:
        raise ValueError("pinned historical report changed since previous publication")
    if old["selection"].get("frozen_at"):
        for key in ("conditions", "ridges", "development_replicas", "main_replicas", "stationary_replicas",
                    "frozen_at", "development_selection", "bootstrap_seed"):
            if old["selection"].get(key) != snapshot["selection"].get(key):
                raise ValueError("frozen selection changed since previous publication: " + key)


def _publish(notebook, latest, notebook_bytes, manifest):
    previous = {path: path.read_bytes() if path.exists() else None for path in (notebook, latest)}
    changed = []
    try:
        for path, data in ((latest, _json_bytes(manifest)), (notebook, notebook_bytes)):
            _atomic(path, data)
            changed.append(path)
    except BaseException:
        for path in reversed(changed):
            if previous[path] is None:
                path.unlink()
            else:
                _atomic(path, previous[path])
        raise


def build_report(root: Path, notebook: Path | None = None) -> dict:
    """Validate, extend original cells, seal one generation, publish atomically."""
    root = Path(root).resolve()
    snapshot, result, snapshot_bytes = _validated(root)
    smoke = snapshot["config"].get("smoke", False)
    canonical = Path(__file__).resolve().parents[2] / "plan17_results.ipynb"
    notebook = Path(notebook).resolve() if notebook is not None else root / "retention_smoke.ipynb" if smoke else canonical
    if notebook.suffix != ".ipynb":
        raise ValueError("notebook output must have .ipynb suffix")
    original, historical_dir, historical_hashes = _historical(snapshot)
    if (notebook.is_relative_to(root / "report") or historical_dir is not None and notebook.is_relative_to(historical_dir)
            or smoke and notebook == canonical):
        raise ValueError("cannot overwrite immutable report artifacts or production notebook with smoke")
    if notebook.exists():
        previous, _ = _read_book(notebook)
        if "plan17_zero_ridge_report" in previous.metadata:
            raise ValueError("refusing to remove Phase 5E; use a separate --notebook destination")
        marker = previous.metadata.get(MARKER)
        if marker and (marker.get("study_id") != snapshot["study_id"] or marker.get("smoke") != smoke):
            raise ValueError("refusing to replace another study or production/smoke report")
        legacy = previous.metadata.get("plan17_report")
        if smoke and (not marker or legacy and not legacy.get("smoke", False)):
            raise ValueError("refusing to overwrite a production or unrelated notebook with smoke")
        if not smoke and legacy and legacy.get("smoke"):
            raise ValueError("refusing to overwrite smoke notebook with production")
    _previous_report(root, snapshot, historical_hashes)
    sources = _source_hashes()
    identity = {"snapshot_sha256": _hash(snapshot_bytes), "sources_sha256": sources,
                "historical_sha256": historical_hashes, "notebook": str(notebook)}
    generation = "generation-" + _hash(_json_bytes(identity))[:24]
    report_dir = root / "report"
    destination = report_dir / generation
    if destination.exists():
        manifest = _verify_generation(destination)
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("report generation identity collision")
        _publish(notebook, report_dir / "latest.json", (destination / "report.ipynb").read_bytes(), manifest)
        return manifest
    report_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=report_dir))
    try:
        for (name, _), renderer in zip(FIGURES, (_development, _fresh_accuracy, _paired_effects)):
            renderer(snapshot, result, staging / (name + ".png"))
        section = _section(snapshot, result, destination, notebook)
        for cell in section:
            if cell.cell_type == "code":
                name = cell.id.removeprefix("plan17-retention-")
                cell.outputs = [nbformat.v4.new_output("display_data", data={
                    "image/png": base64.b64encode((staging / (name + ".png")).read_bytes()).decode(), "text/plain": name})]
        if smoke:
            book = nbformat.v4.new_notebook(cells=[_markdown("title", "# Plan 17 Phase 5: Smoke Acceptance Only\n\n**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.**"), _progress(snapshot, result)] + section)
        else:
            book = deepcopy(original)
            index = next(i for i, c in enumerate(book.cells) if c.id == "plan17-retention-followup")
            book.cells[index:index + 1] = section
            book.cells.insert(1, _progress(snapshot, result))
        book.metadata[MARKER] = {"schema": 1, "study_id": snapshot["study_id"], "smoke": smoke,
                                 "phase": snapshot["phase"], "status": snapshot["status"], "generation": generation,
                                 "snapshot_sha256": identity["snapshot_sha256"], "historical_report_dir": str(historical_dir) if historical_dir else None}
        nbformat.validate(book)
        notebook_bytes = nbformat.writes(book).encode()
        _atomic(staging / "report.ipynb", notebook_bytes)
        _atomic(staging / "report_snapshot.json", snapshot_bytes)
        _atomic(staging / "analysis.json", _json_bytes(result))
        manifest = {"schema": 1, "kind": "plan17-retention-report", "study_id": snapshot["study_id"],
                    "smoke": smoke, "report_dir": str(destination), "manifest": str(report_dir / "latest.json"),
                    **identity, "counts": result["counts"], "historical_report_dir": str(historical_dir) if historical_dir else None,
                    "files_sha256": {path.name: _hash(path.read_bytes()) for path in sorted(staging.iterdir())}}
        _atomic(staging / "manifest.json", _json_bytes(manifest))
        _atomic(staging / "COMPLETED", (_hash(_json_bytes(manifest)) + "\n").encode())
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.is_dir():
                raise
            _verify_generation(destination, expected=manifest)
        _verify_generation(destination, expected=manifest)
        _publish(notebook, report_dir / "latest.json", notebook_bytes, manifest)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish bounded Phase 5 from scalar artifacts only")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
