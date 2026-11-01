"""Artifact-only Plan 16 sidecar, independent of execution/data modules.

build_report(root, notebook=None) reads only progress.json and the atomically
sealed schema-1 report_snapshot.json. No checkpoint/model/dataset is loaded.
An empty snapshot.analysis triggers bounded summarize(); a populated analysis
must be the matching output of analysis.summarize (including its input hash).
PNG files, analysis.json and a portable executed notebook are published under
root/report/<generation>/. latest.json is the atomic reporting index. The
canonical notebook (or explicit notebook path) is atomically replaced last.

Return: {schema, notebook, report_dir, manifest, snapshot_sha256, counts}.
Costs/exposures follow analysis.py; frozen context comes from constant metric
fields frozen_nine_recall/frozen_old_accuracy, or summary.frozen_initial.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap

import nbformat
import numpy as np

from .analysis import ACCURACIES, condition_name, summarize, validate_trajectories

MAX_JSON_BYTES = 64 * 1024 * 1024
NAMES = {"ema": "EMA", "paired_fd": "Paired FD", "analytic_lfu": "Analytic LFU"}
COLORS = {"ema": "#3475a7", "paired_fd": "#be4b61", "analytic_lfu": "#278574"}
FIGURES = (
    ("01_development", "Development: accuracy paths and measured cost"),
    ("02_fresh_main", "Fresh main: matched acquisition and retention trajectories"),
    ("03_fresh_stationary", "Fresh stationary: matched acquisition and retention trajectories"),
    ("04_precision_exposure", "Precision / recall and cumulative digit-9 exposure"),
    ("05_paired_cost", "Paired accuracy differences and measured compute"),
    ("06_health", "Development numerical health before clipping and solver diagnostics"),
)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _load(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"compact reporting artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    data = path.read_bytes()
    def reject(value):
        raise ValueError(f"nonfinite JSON number {value} in {path}")
    try:
        value = json.loads(data, parse_constant=reject)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid reporting JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"reporting artifact must be an object: {path}")
    return value, data


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _text(value, limit=240):
    if value is None:
        return "pending"
    if isinstance(value, (list, dict)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    value = str(value)
    if len(value) > limit:
        value = value[:limit - 3] + "..."
    return html.escape(value, quote=False).replace("|", "/").replace("\n", " ")


def _number(value, digits=2):
    return "pending" if value is None else f"{value:.{digits}f}"


def _recorded(value, *, missing="pending"):
    return missing if value is None else str(value)


def _effect(value, *, signed=False):
    if not value or value.get("mean") is None:
        return "pending"
    result = f"{value['mean']:+.2f}" if signed else _number(value["mean"])
    if value.get("ci95") is not None:
        result += " [" + ", ".join(_number(v) for v in value["ci95"]) + "]"
    if not value.get("interval_reliable"):
        result += " (unreliable)"
    return result


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id="plan16-" + name)


def _table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
        "| " + " | ".join(_text(value) for value in row) + " |" for row in rows)


def _progress(snapshot, progress, analysis):
    p = {**snapshot["progress"], **progress}
    storage = p.get("storage") or {}
    used, cap = storage.get("used_bytes"), storage.get("limit_bytes", storage.get("max_bytes"))
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    headroom = cap - used if used is not None and cap is not None else None
    active_hours = p.get("active_hours", p.get("active_seconds", 0) / 3600)
    allocation_hours = p.get("allocation_hours", p.get("allocation_seconds", 72000) / 3600)
    free = storage.get("free_bytes", storage.get("filesystem_free_bytes"))
    fresh_blocks = {}
    for t in snapshot["trajectories"]:
        if t["completed"] and _is_fresh(t):
            key = (t["schedule"], t["c"], t["ridge"], t["replica"])
            fresh_blocks.setdefault(key, set()).add(condition_name(t["condition"]))
    fd_pairs = sum({"ema", "paired_fd"} <= arms for arms in fresh_blocks.values())
    analytic_pairs = sum({"ema", "analytic_lfu"} <= arms for arms in fresh_blocks.values())
    retained = set(snapshot["selection"].get("conditions") or NAMES)
    blocks = sum(retained <= arms for arms in fresh_blocks.values())
    selection = snapshot["selection"]
    cohort_counts = [selection.get(k) for k in ("main_replicas", "stationary_replicas")]
    planned_blocks = sum(cohort_counts) if all(n is not None for n in cohort_counts) else None
    current = p.get("current")
    current_info = current if isinstance(current, dict) else {}
    step = current_info.get("step", p.get("step"))
    total_steps = current_info.get("total_steps", p.get("total_steps"))
    accepted = f"{_recorded(step)}/{_recorded(total_steps, missing='calibrating')}"
    active_unit = p.get("active_unit") or current_info.get("active_unit") or (current if isinstance(current, str) else None)
    checkpoint = (p.get("last_checkpoint") or p.get("last_accepted_checkpoint") or
                  current_info.get("last_checkpoint") or current_info.get("last_accepted_checkpoint"))
    if checkpoint is None and step is not None:
        checkpoint = "step " + accepted
    rows = [
        ("Phase / status", f"{snapshot['phase']} / {p.get('status', snapshot['status'])}"),
        ("Retained-method paired blocks completed / planned", f"{_recorded(p.get('completed_blocks', p.get('paired_blocks', blocks)))} / {_recorded(p.get('planned_blocks', planned_blocks), missing='calibrating')}"),
        ("Complete trajectories (operational / sealed)", f"{p.get('completed_trajectories', analysis['counts']['complete'])} / {analysis['counts']['complete']}"),
        ("All trajectories planned", _recorded(p.get("planned_trajectories"), missing="calibrating")),
        ("Fresh trajectories completed / planned", f"{_recorded(p.get('completed_fresh'))} / {_recorded(p.get('planned_fresh'), missing='calibrating')}"),
        ("Complete fresh FD / analytic pairs", f"{fd_pairs} / {analytic_pairs}"),
        ("Sealed incomplete trajectories", analysis["counts"]["incomplete"]),
        ("Active unit / checkpoint", f"{_recorded(active_unit, missing='none (completed)' if p.get('status') == 'completed' else 'pending')} / {checkpoint or 'pending'}"),
        ("Accepted incoming-batch step / total_steps", accepted),
        ("Cumulative active / allocated hours", f"{_number(active_hours)} / {_number(allocation_hours)}"),
        ("Allocation hours remaining", _number(p.get("remaining_hours"))),
        ("Remaining forecast hours", _number(p.get("forecast_seconds") / 3600) if p.get("forecast_seconds") is not None else "calibrating"),
        ("Heartbeat / revision", f"{_recorded(p.get('heartbeat'))} / {_recorded(p.get('revision'))}"),
        ("Storage used / ceiling / headroom (GB)", " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, cap, headroom))),
        ("Filesystem free (GB)", _number(free / 1e9) if free is not None else "pending"),
    ]
    return _markdown("progress", "## Progress\n\n" + _table(("Item", "Recorded state"), rows) +
        "\n\nProgress is operational telemetry, not a process-liveness check. Scientific plots use only the sealed snapshot; "
        "no live metric files are scanned. Counts describe sealed rows, not unsealed work.")


def _change_brief(change):
    if isinstance(change, dict):
        source = next((change[k] for k in ("summary", "title", "change", "description", "reason")
                       if isinstance(change.get(k), str) and change[k].strip()), "Scientific revision recorded")
    else:
        source = str(change)
    sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(source.split()), maxsplit=1)[0]
    sentence = sentence.split(";", 1)[0]
    prefix, using, detail = sentence.partition(" using ")
    if using and "=" in detail:
        sentence = prefix
    sentence = textwrap.shorten(sentence, width=120, placeholder="...")
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def _settings(snapshot, result, *, snapshot_link="report_snapshot.json"):
    rows = []
    for group in result["groups"]:
        rows.append(("Setting", group["stage"], group["schedule"], f"c={group['c']:g}; ridge={group['ridge']:g}",
                     f"{len(group['complete_units'])} complete; {len(group['incomplete_units'])} incomplete"))
    if not rows:
        rows.append(("Setting", "pending", "pending", "pending", "No sealed trajectories"))
    selection = snapshot["selection"]
    rows.extend([
        ("Selected setting", "development only", "not an efficacy gate",
         f"c={selection.get('common_c', 'pending')}; ridge={selection.get('common_ridge', 'pending')}",
         f"Frozen: {selection.get('frozen_at', 'pending')}"),
        ("Fresh allocation", "frozen cohort", f"main={selection.get('main_replicas', 'pending')}; stationary={selection.get('stationary_replicas', 'pending')}",
         ", ".join(NAMES.get(c, c) for c in selection.get("conditions", [])) or "pending",
         f"{selection.get('fresh_trajectory_target', 'pending')} trajectories"),
        ("Cost forecast", "before fresh outcomes", "whole fresh allocation",
         (_number(selection["forecast_fresh_seconds"] / 3600) + " h") if selection.get("forecast_fresh_seconds") is not None else "pending",
         f"Contingency: {selection.get('contingency', 'pending')}"),
    ])
    for change in snapshot["changes"][-6:]:
        details = f"[Full event and validation]({snapshot_link})"
        if isinstance(change, dict):
            rows.append(("Change", change.get("phase", change.get("stage", change.get("date", "recorded"))),
                         change.get("revision", change.get("time", "")), _change_brief(change), details))
        else:
            rows.append(("Change", "recorded", "", _change_brief(change), details))
    config = snapshot["config"]
    concise = {k: config[k] for k in ("model", "dtype", "fisher_dtype", "curvature_dtype", "samples_per_arrival", "samples_per_step", "optimizer") if k in config}
    return _markdown("settings", "## Glossary And Settings\n\n"
        "**Diagonal EWC** constrains the learner with diagonal Fisher curvature plus a separately recorded ridge. "
        "**LFU** tracks curvature along the learner's already-realized displacement. **Paired FD** differences "
        "model-label-integrated diagonals at two parameters using identical inputs; it is not Plan 15's FD KL. "
        "**EMA** is the common learning control. All treatments track the same likelihood Fisher.\n\n"
        "The learner has 487 trainable coordinates, with diagonal EWC and isotropic ridge only: "
        "no STM, replay training, or FD-KL penalty. The fixed action $c$ weights fresh Fisher observations "
        "and the learning objective; ridge is $\\kappa_t=r\\,\\mathrm{mean}(\\widehat d_t)$.\n\n"
        "**9 acquisition accuracy (recall)** conditions on true 9s. **0--8 retention accuracy** is the mean of "
        "the nine within-class accuracies, not overall mixture accuracy. **9 precision** conditions on predicted "
        "9s and is standardized to evaluation prevalence 0.1; FPR uses the configured non-9 frequencies. "
        "Undefined precision remains a gap. Frozen initialization is inference-only context.\n\n"
        f"All plotted accuracies use an independent stratified holdout ({config.get('eval_per_class', 'recorded count')} "
        "images per class), never training batches or Fisher-reference inputs. Development and fresh holdouts "
        "are separate. Predictive NLL excludes every regularizer.\n\n"
        "A **step** is one incoming batch. The saved `arrivals` field counts **online digits observed** "
        "(`step * samples_per_step`), excluding initialization. Cost per online digit uses that observation "
        "count, not the number of batches.\n\n" +
        _table(("Kind", "Stage", "Schedule / revision", "Common setting / change", "Evidence / validation"), rows) +
        f"\n\nRecorded core settings: {_text(concise or config)}. Full resolved configuration, selection and all "
        f"{len(snapshot['changes'])} changes remain in the linked compact snapshot; only brief opening summaries of the latest six events appear here. "
        "Development choices are not fresh evidence; different stages, schedules and settings are never pooled.")


def _is_fresh(t):
    return t["stage"] == "fresh" or t["stage"].startswith("fresh_")


def _label(group):
    return f"{group['stage']} / {group['schedule']} / c={group['c']:g}, r={group['ridge']:g}"


def _legend_label(condition, group, groups, *, development=False):
    parts = [NAMES.get(condition, condition)]
    settings = {(g["stage"], g["schedule"], g["c"], g["ridge"]) for g in groups}
    if development or len(settings) > 1:
        parts.append(f"c={group['c']:g}")
    if development or len({g["schedule"] for g in groups}) > 1:
        parts.append(group["schedule"])
    if len({g["ridge"] for g in groups}) > 1:
        parts.append(f"r={group['ridge']:g}")
    if len({g["stage"] for g in groups}) > 1:
        parts.append(group["stage"])
    return ", ".join(parts)


def _matched(snapshot, result, *, stationary=None):
    by_id = {t["unit_id"]: t for t in snapshot["trajectories"]}
    for group in result["groups"]:
        if not _is_fresh(group) or (stationary is not None and (group["schedule"] == "stationary") != stationary):
            continue
        for contrast in group["comparisons"]:
            if contrast["n_pairs"]:
                yield group, contrast, [by_id[u] for u in contrast["control_units"]], [by_id[u] for u in contrast["treatment_units"]]


@contextmanager
def _figure(rows=2, cols=2):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7,
                               "axes.spines.top": False, "axes.spines.right": False}):
        fig = Figure(figsize=(11.6, 6.3), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, cols, squeeze=False)
        try:
            yield fig, axes
        finally:
            fig.clear()


def _finish(axes):
    for ax in axes.flat:
        if not ax.has_data() and not ax.texts:
            ax.text(.5, .5, "Pending complete matched evidence", ha="center", va="center", transform=ax.transAxes)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="best")
        ax.grid(alpha=.18)


def _curve(ax, trajectories, metric, label, *, color, linestyle="-", percent=True):
    # Whole trajectories contribute once at each grid point; no interpolation
    # bridges missing precision and no incomplete arm changes the cohort.
    if not trajectories:
        return
    grids = {tuple(r["arrivals"] for r in t["metrics"]) for t in trajectories}
    if len(grids) != 1:
        raise ValueError("replicas in one curve must share the recorded evaluation grid")
    x = np.asarray(next(iter(grids)), dtype=float)
    y = np.asarray([[np.nan if r.get(metric) is None else r[metric] for r in t["metrics"]] for t in trajectories], dtype=float)
    y *= 100 if percent else 1
    valid = np.all(np.isfinite(y), axis=0)
    mean = np.full(len(x), np.nan)
    mean[valid] = y[:, valid].mean(axis=0)
    ax.plot(x, mean, color=color, linestyle=linestyle, label=f"{label} (n={len(y)})")
    if len(y) >= 2:
        half = np.full(len(x), np.nan)
        half[valid] = 1.96 * y[:, valid].std(axis=0, ddof=1) / np.sqrt(len(y))
        lo, hi = mean - half, mean + half
        if percent:
            lo, hi = np.maximum(lo, 0), np.minimum(hi, 100)
        ax.fill_between(x, lo, hi, color=color, alpha=.10)
    ax.set_xlabel("Online digits observed")


def _regimes(ax, trajectory):
    rows = trajectory["metrics"]
    if len(rows) < 2 or not all(r.get("segment") for r in rows[1:]):
        return
    labels = {"low_hold_initial": "Low hold", "low_hold": "Low hold", "rise": "Rise",
              "high_hold": "High hold", "return": "Return", "low_hold_final": "Low hold",
              "stationary_low": "Stationary low"}
    start = rows[0]["arrivals"]
    previous = rows[1].get("segment", "unspecified")
    index = 0
    for i, row in enumerate(rows[1:], start=1):
        segment = row.get("segment", "unspecified")
        if segment != previous:
            end = rows[i - 1]["arrivals"]
            ax.axvspan(start, end, color="#dce7df" if index % 2 == 0 else "#edf0f2", alpha=.45, zorder=0)
            ax.text((start + end) / 2, .02, labels.get(previous, previous.replace("_", " ")), fontsize=6, ha="center", transform=ax.get_xaxis_transform())
            start, previous, index = end, segment, index + 1
    end = rows[-1]["arrivals"]
    ax.axvspan(start, end, color="#dce7df" if index % 2 == 0 else "#edf0f2", alpha=.45, zorder=0)
    ax.text((start + end) / 2, .02, labels.get(previous, previous.replace("_", " ")), fontsize=6, ha="center", transform=ax.get_xaxis_transform())


def _frozen(ax, trajectories, metric):
    values = [t["metrics"][0].get("frozen_" + metric, t["summary"].get("frozen_initial", {}).get(metric)) for t in trajectories]
    if values and all(v is not None for v in values):
        ax.axhline(100 * np.mean(values), color="#626262", linestyle=":", linewidth=1.2,
                   label=f"Frozen initial reference (n={len(values)})")


def _development(snapshot, result, destination):
    with _figure() as (fig, axes):
        groups = [g for g in result["groups"] if not _is_fresh(g) and g["complete_units"]]
        selection = snapshot["selection"]
        selected = selection.get("common_c") is not None and selection.get("common_ridge") is not None
        curve_groups = [g for g in groups if g["schedule"] == "main" and
                        g["c"] == selection["common_c"] and g["ridge"] == selection["common_ridge"]] if selected else groups
        curve_units = {u for g in curve_groups for u in g["complete_units"]}
        trajectories = [t for t in snapshot["trajectories"] if t["unit_id"] in curve_units]
        reference = next((t for t in trajectories if t["schedule"] == "main"), next(iter(trajectories), None))
        if reference is not None:
            for ax in axes[0]:
                _regimes(ax, reference)
            context = "Main schedule regime bands" if reference["schedule"] == "main" else "Stationary low-prevalence schedule"
            if selected:
                context = f"Selected main c={selection['common_c']:g}, ridge={selection['common_ridge']:g}; recorded regime bands"
            elif reference["schedule"] == "main" and any(t["schedule"] == "stationary" for t in trajectories):
                context = "Main regime bands; stationary paths dashed"
            fig.suptitle(context, fontsize=10)
        for group in groups:
            rows = [t for t in snapshot["trajectories"] if t["unit_id"] in group["complete_units"]]
            for condition in sorted({condition_name(t["condition"]) for t in rows}):
                arm = [t for t in rows if condition_name(t["condition"]) == condition]
                label = _legend_label(condition, group, curve_groups, development=True)
                color = COLORS.get(condition, "#777777")
                stationary = group["schedule"] == "stationary"
                if group in curve_groups:
                    for ax, metric in zip(axes[0], ACCURACIES):
                        _curve(ax, arm, metric, label, color=color, linestyle="--" if stationary else "-")
                summaries = [r for r in group["replicas"] if r["condition"] == condition]
                for r in summaries:
                    axes[1, 0].scatter(100 * r["observation_mean"]["old_accuracy"], 100 * r["observation_mean"]["nine_recall"], color=color, s=20, marker="s" if stationary else "o")
                    cost = r["costs"].get("seconds_per_arrival")
                    if cost is not None:
                        axes[1, 1].scatter(cost, 100 * r["observation_mean"]["nine_recall"], color=color, s=20, marker="s" if stationary else "o")
        for ax, name in zip(axes[0], ("9 recall (%)", "0--8 retention accuracy (%)")):
            ax.set(title=name, ylim=(0, 100))
        axes[1, 0].set(title="Development trade-off: individual complete paths", xlabel="Observation-averaged 0--8 accuracy (%)", ylabel="Observation-averaged 9 recall (%)")
        axes[1, 1].set(title="Measured development cost", xlabel="Seconds per online digit", ylabel="Observation-averaged 9 recall (%)")
        _finish(axes)
        legend = axes[0, 0].get_legend()
        if legend is not None:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axes[0, 0].legend(unique.values(), unique.keys(), loc="upper left")
        legend = axes[0, 1].get_legend()
        if legend is not None:
            legend.remove()
        fig.savefig(destination, dpi=120)


def _fresh(snapshot, result, destination, *, stationary):
    with _figure() as (fig, axes):
        regime_seen = set()
        matches = list(_matched(snapshot, result, stationary=stationary))
        groups = [g for g, *_ in matches]
        for group, contrast, controls, targets in matches:
            row = 0 if contrast["condition"] == "paired_fd" else 1
            for col, metric in enumerate(ACCURACIES):
                ax = axes[row, col]
                tag = (row, col)
                if tag not in regime_seen:
                    _regimes(ax, controls[0])
                    regime_seen.add(tag)
                _curve(ax, controls, metric, _legend_label("ema", group, groups), color=COLORS["ema"], linestyle="--")
                _curve(ax, targets, metric, _legend_label(contrast["condition"], group, groups),
                       color=COLORS.get(contrast["condition"], "#777777"))
                _frozen(ax, controls, metric)
        for row in range(2):
            for col, metric in enumerate(("9 recall (%)", "0--8 retention accuracy (%)")):
                axes[row, col].set(title=("Paired FD vs EMA: " if row == 0 else "Analytic LFU vs EMA: ") + metric, ylim=(0, 100))
        _finish(axes)
        fig.savefig(destination, dpi=120)


def _precision_exposure(snapshot, result, destination):
    with _figure() as (fig, axes):
        for col, stationary in enumerate((False, True)):
            seen_exposure, seen_curves = set(), set()
            defined_checkpoints = []
            matches = list(_matched(snapshot, result, stationary=stationary))
            groups = [g for g, *_ in matches]
            control_cohorts = {tuple(t["unit_id"] for t in controls) for _, _, controls, _ in matches}
            for group, contrast, controls, targets in matches:
                for arm, condition in ((controls, "ema"), (targets, contrast["condition"])):
                    key = tuple(t["unit_id"] for t in arm)
                    if key in seen_curves:
                        continue
                    seen_curves.add(key)
                    label = _legend_label(condition, group, groups) + f" (n={len(arm)})"
                    if condition == "ema" and len(control_cohorts) > 1:
                        label += " / " + NAMES.get(contrast["condition"], contrast["condition"]) + " cohort"
                    means = []
                    for metric in ("nine_recall", "nine_precision"):
                        values = np.asarray([[np.nan if row.get(metric) is None else row[metric] for row in t["metrics"]] for t in arm])
                        valid = np.all(np.isfinite(values), axis=0)
                        mean = np.full(values.shape[1], np.nan)
                        mean[valid] = 100 * values[:, valid].mean(axis=0)
                        means.append(mean)
                    defined = int(np.isfinite(means[1]).sum())
                    defined_checkpoints.append(f"{NAMES.get(condition, condition)}: {defined}/{len(means[1])}")
                    if defined:
                        axes[0, col].plot(*means, color=COLORS.get(condition, "#777777"), marker=".", markersize=3, label=label)
                key = tuple(t["replica"] for t in controls), group["stage"], group["c"], group["ridge"]
                if key not in seen_exposure:
                    initial = [t["summary"].get("initial_nines", t["metrics"][0].get("initial_nines")) for t in controls]
                    initial_note = f"; initial mean={np.mean(initial):g}" if all(v is not None for v in initial) else ""
                    _curve(axes[1, col], controls, "observed_nines", f"Observed online 9s{initial_note}; c={group['c']:g}", color="#3475a7", percent=False)
                    _curve(axes[1, col], controls, "expected_nines", f"Expected 9s; c={group['c']:g}", color="#be4b61", linestyle="--", percent=False)
                    seen_exposure.add(key)
            if defined_checkpoints and not axes[0, col].has_data():
                axes[0, col].text(.5, .55, "No complete-cohort precision path.\nAt every checkpoint, at least one replica\nhas no predicted 9s (undefined precision).",
                                  ha="center", va="center", transform=axes[0, col].transAxes)
                axes[0, col].text(.5, .33, "Defined checkpoints: " + "; ".join(defined_checkpoints),
                                  ha="center", fontsize=7, transform=axes[0, col].transAxes)
            title = "Stationary" if stationary else "Main"
            axes[0, col].set(title=title + ": precision / recall path", xlabel="9 recall (%)", ylabel="9 precision @ p_ref=.1 (%)", xlim=(0, 100), ylim=(0, 100))
            axes[1, col].set(title=title + ": shared online exposure", ylabel="Cumulative digit-9 count")
        _finish(axes)
        fig.savefig(destination, dpi=120)


def _paired_cost(snapshot, result, destination):
    with _figure() as (fig, axes):
        labels = []
        for i, (group, contrast, _, _) in enumerate(_matched(snapshot, result)):
            labels.append(f"{group['schedule']} {NAMES.get(contrast['condition'], contrast['condition'])}\nc={group['c']:g}, r={group['ridge']:g}; n={contrast['n_pairs']}")
            for col, metric in enumerate(ACCURACIES):
                ax = axes[0, col]
                for scope, shift, color in (("observation_mean_pp", -.12, "#3475a7"), ("endpoint_pp", .12, "#be4b61")):
                    effect = contrast[scope][metric]
                    if effect["mean"] is not None:
                        ax.plot(effect["mean"], i + shift, "o", color=color, label=scope.removesuffix("_pp").replace("_", " ") if i == 0 else None)
                        if effect["ci95"] is not None:
                            ax.plot(effect["ci95"], [i + shift] * 2, color=color, linewidth=1)
                        if contrast["n_pairs"] < 4:
                            ax.scatter(effect["individual"], [i + shift] * len(effect["individual"]), color=color, s=10, marker="x")
            wall = contrast["costs"].get("wall_seconds")
            if wall is not None and wall["ratio"]["mean"] is not None:
                axes[1, 0].plot(wall["ratio"]["mean"], i, "o", color="#278574")
                if wall["ratio"]["ci95"] is not None:
                    axes[1, 0].plot(wall["ratio"]["ci95"], [i] * 2, color="#278574")
        segment_labels = {"low_hold_initial": "Low hold\n(initial)", "rise": "Rise", "high_hold": "High hold",
                          "return": "Return", "low_hold_final": "Low hold\n(final)", "low_hold": "Low hold"}
        for group, contrast, controls, _ in _matched(snapshot, result, stationary=False):
            segments = list(dict.fromkeys(r["segment"] for r in controls[0]["metrics"][1:]))
            shift = -.12 if contrast["condition"] == "paired_fd" else .12
            for metric, marker, name in (("nine_recall", "o", "9 recall"), ("old_accuracy", "s", "0--8 accuracy")):
                effects = [contrast["segments"][segment]["observation_mean_pp"][metric]["mean"] for segment in segments]
                axes[1, 1].plot(np.arange(len(segments)) + shift, effects, marker=marker, linestyle="none",
                               color=COLORS[contrast["condition"]], markersize=4,
                               label=f"{NAMES[contrast['condition']]}: {name}")
            axes[1, 1].set_xticks(range(len(segments)), [segment_labels.get(s, s) for s in segments], fontsize=7)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
        for ax, title in zip(axes[0], ("9 recall difference", "0--8 accuracy difference")):
            ax.axvline(0, color="#777777", linewidth=.7)
            ax.set(title=title, xlabel="Treatment minus EMA (percentage points)")
        axes[1, 0].axvline(1, color="#777777", linewidth=.7)
        axes[1, 0].set(title="Measured total wall cost", xlabel="Treatment / EMA ratio")
        axes[1, 1].axhline(0, color="#777777", linewidth=.7)
        axes[1, 1].set(title="Main schedule: segment-averaged differences", ylabel="Treatment minus EMA (pp)")
        _finish(axes)
        fig.savefig(destination, dpi=120)


def _health(snapshot, result, destination):
    with _figure() as (fig, axes):
        seen = set()
        selection = snapshot["selection"]
        selected = selection.get("common_c") is not None and selection.get("common_ridge") is not None
        for t in snapshot["trajectories"]:
            if not t["completed"] or _is_fresh(t):
                continue
            if selected and (t["schedule"] != "main" or t["c"] != selection["common_c"] or
                             t["ridge"] != selection["common_ridge"]):
                continue
            condition = condition_name(t["condition"])
            key = (t["stage"], t["schedule"], condition, t["c"], t["ridge"])
            if key in seen:
                continue
            seen.add(key)
            for ax, metric in zip(axes.flat, ("negative_fraction", "negative_mass", "grad_norm", "closure_count")):
                _curve(ax, [t], metric, NAMES.get(condition, condition) + f" {t['schedule']} c={t['c']:g}, r={t['ridge']:g}",
                       color=COLORS.get(condition, "#777777"), percent=metric == "negative_fraction")
        for ax, title in zip(axes.flat, ("Pre-clipping negative fraction (%)", "Pre-clipping negative mass", "Solver gradient norm", "Recorded closure count")):
            ax.set_title(title)
        _finish(axes)
        fig.savefig(destination, dpi=120)


def _paired_table(result):
    rows = []
    fresh = [group for group in result["groups"] if _is_fresh(group)]
    for group in fresh or result["groups"]:
        for contrast in group["comparisons"]:
            observation, endpoint = contrast["observation_mean_pp"], contrast["endpoint_pp"]
            wall = contrast["costs"].get("wall_seconds", {})
            rows.append((_label(group), NAMES.get(contrast["condition"], contrast["condition"]) + " - EMA", contrast["n_pairs"],
                         _effect(observation["nine_recall"]), _effect(observation["old_accuracy"]),
                         _number(endpoint["nine_recall"]["mean"]) + " / " + _number(endpoint["old_accuracy"]["mean"]),
                         _effect(wall.get("ratio"))))
    if not rows:
        rows = [("pending", "both contrasts pending", 0, "pending", "pending", "pending", "pending")]
    return _markdown("paired-table", "### Paired Summary\n\n" + _table(
        ("Cohort / setting", "Contrast", "Complete pairs", "Full-path 9 (pp, CI)", "Full-path 0--8 (pp, CI)", "Endpoint 9 / 0--8 (pp)", "Wall ratio (CI)"), rows) +
        "\n\nNominal 95% percentile intervals resample complete replica pairs, not timepoints. They are unreliable "
        "below four pairs and absent for one; individual differences are plotted. Comparisons and interim looks "
        "are unadjusted. Exact differences, segment and endpoint intervals, costs, and unit IDs remain in "
        "analysis.json. This table shows fresh cohorts when available; development remains in the earlier "
        "figure and full analysis. Equal digit exposure is not equal compute.")


def _conclusion_summary(result):
    groups = [g for g in result["groups"] if _is_fresh(g) and g["schedule"] == "main"]
    comparisons = [(g, c) for g in groups for c in sorted(
        g["comparisons"], key=lambda c: (c["condition"] != "paired_fd", c["condition"])) if c["n_pairs"]]
    if not comparisons:
        return ""
    counts = {c["n_pairs"] for _, c in comparisons}
    rows = []
    for group, contrast in comparisons:
        label = NAMES.get(contrast["condition"], contrast["condition"])
        if len(groups) > 1:
            label += f" (c={group['c']:g}, r={group['ridge']:g})"
        if len(counts) > 1:
            label += f" (n={contrast['n_pairs']})"
        wall = contrast["costs"].get("wall_seconds", {}).get("ratio", {}).get("mean")
        effects = contrast["observation_mean_pp"]
        rows.append((label, _effect(effects["nine_recall"], signed=True),
                     _effect(effects["old_accuracy"], signed=True),
                     "pending" if wall is None else f"{100 * (wall - 1):.1f}%"))
    pairing = (f"{next(iter(counts))} complete paired replicas per method" if len(counts) == 1 else
               "complete paired replicas (pair counts appear beside method names)")
    return (f"Fresh changing-mixture results versus EMA, using {pairing}. Accuracy changes are "
            "observation-averaged over the full trajectory, in percentage points (pp).\n\n" +
            _table(("Method", "9 recall change (pp)", "0--8 retention change (pp)", "Extra wall time"), rows) +
            "\n\nBrackets are exploratory, nominal 95% paired-bootstrap confidence intervals, unadjusted "
            "for multiple comparisons or interim inspection. Extra wall time is the mean paired runtime "
            "ratio minus one, expressed as a percentage; it is not an isolated Fisher-update benchmark.\n\n")


def _conclusion(result):
    observations = []
    unmatched, incomplete = set(), result["counts"]["incomplete"]
    for group in result["groups"]:
        if not _is_fresh(group):
            continue
        statements = []
        for contrast in group["comparisons"]:
            unmatched.update(contrast["unpaired_control_units"] + contrast["unpaired_treatment_units"])
            if not contrast["n_pairs"]:
                continue
            nine = contrast["observation_mean_pp"]["nine_recall"]["mean"]
            old = contrast["observation_mean_pp"]["old_accuracy"]["mean"]
            if nine > 0 and old < 0:
                direction = "higher 9 recall but lower 0--8 retention"
            elif nine < 0 and old > 0:
                direction = "lower 9 recall but higher 0--8 retention"
            elif nine > 0 and old > 0:
                direction = "higher acquisition and retention"
            elif nine < 0 and old < 0:
                direction = "lower acquisition and retention"
            else:
                direction = "an unchanged acquisition or retention outcome"
            interval = contrast["observation_mean_pp"]["nine_recall"]["ci95"]
            uncertain = " Its nominal recall interval includes zero." if interval and interval[0] <= 0 <= interval[1] else ""
            statements.append(f"{NAMES.get(contrast['condition'], contrast['condition'])} ({contrast['n_pairs']} pairs): "
                              f"point estimates show {direction}.{uncertain}")
        if statements:
            cohort = "Changing mixture" if group["schedule"] == "main" else "Stationary low prevalence"
            observations.append(f"**{cohort}:** " + " ".join(statements))
    return _markdown("conclusion", "## Conclusion\n\n" + _conclusion_summary(result) + (
        "\n\n".join(observations[:6]) if observations else "No complete fresh paired evidence yet; development alone cannot establish fresh replication.") +
        "\n\nA recall gain with lower retention is a trade-off, not an improvement in both. "
        "See the paired summary for full-path effects, nominal intervals and measured overhead. "
        "Equal exposure is not equal compute or matched retention; these results alone do not establish sample savings. "
        "The final endpoint follows a return to rare 9s, so it is not a stand-alone acquisition score.\n\n"
        f"{incomplete} sealed incomplete trajectories; {len(unmatched)} unmatched fresh arms. "
        "Unpaired development-screen controls remain descriptive. No efficacy gate is applied.\n\n"
        "Diagonal curvature is chart-dependent and omits off-diagonal information; matched current inputs "
        "remain a local input-sampling approximation. Clipping and finite-tolerance solves can affect behavior. "
        "This small, single-architecture study neither proves diagonal EWC sufficient nor rules out LFUs elsewhere; "
        "accuracy proxies do not establish separate visual domains.")


def build_report(root: Path, notebook: Path | None = None) -> dict:
    """Publish a portable executed notebook and static PNGs from sealed scalars.

    Missing/malformed/incompatible artifacts fail before publication. Existing
    reports survive a failed refresh. Reporting never seals or repairs a unit.
    """
    root = Path(root).resolve()
    notebook = Path(notebook).resolve() if notebook is not None else Path(__file__).resolve().parents[1] / "plan16_results.ipynb"
    if notebook.suffix != ".ipynb":
        raise ValueError("notebook output must have .ipynb suffix")
    progress, progress_bytes = _load(root / "progress.json")
    snapshot, snapshot_bytes = _load(root / "report_snapshot.json")
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1:
        raise ValueError("report_snapshot.json requires compatible schema 1")
    for key in ("study_id", "status", "phase", "config", "progress", "changes", "trajectories", "selection", "analysis"):
        if key not in snapshot:
            raise ValueError(f"report snapshot missing {key}")
    for key in ("config", "progress", "selection"):
        if not isinstance(snapshot[key], dict):
            raise ValueError(f"snapshot.{key} must be an object")
    if not isinstance(snapshot["changes"], list):
        raise ValueError("snapshot.changes must be a list")
    if progress.get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
        raise ValueError("progress and snapshot study_id mismatch")
    validate_trajectories(snapshot["trajectories"])
    expected_hash = _hash(json.dumps(snapshot["trajectories"], sort_keys=True, allow_nan=False).encode())
    result = snapshot["analysis"]
    if result in (None, {}):
        result = summarize(snapshot["trajectories"])
    elif not isinstance(result, dict) or result.get("kind") != "plan16-paired-analysis" or result.get("schema") != 1 or result.get("trajectories_sha256") != expected_hash:
        raise ValueError("snapshot.analysis must be empty or matching analysis.summarize output")
    report = root / "report"
    code_hash = _hash(Path(__file__).read_bytes() + Path(__file__).with_name("analysis.py").read_bytes())
    generation = "generation-" + _hash(snapshot_bytes + progress_bytes + code_hash.encode() + str(notebook).encode())[:24]
    destination = report / generation
    report.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=report))
    try:
        _development(snapshot, result, staging / "01_development.png")
        _fresh(snapshot, result, staging / "02_fresh_main.png", stationary=False)
        _fresh(snapshot, result, staging / "03_fresh_stationary.png", stationary=True)
        _precision_exposure(snapshot, result, staging / "04_precision_exposure.png")
        _paired_cost(snapshot, result, staging / "05_paired_cost.png")
        _health(snapshot, result, staging / "06_health.png")
        status = "Partial / incomplete" if snapshot["status"] not in ("completed", "complete") or result["counts"]["incomplete"] else "Completed snapshot"
        smoke = snapshot["config"].get("smoke") is True
        banner = ("**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.** Tiny acceptance trajectories "
                  "check implementation, checkpoint recovery, and reporting. Completion here is not completion "
                  "of the production study. All curves, intervals, and tables below are acceptance diagnostics only.\n\n") if smoke else ""
        cells = [_markdown("title", "# Plan 16: Diagonal LFU" + (" (Smoke Acceptance Only)" if smoke else "") + "\n\n" + banner +
            "Can matched-input Fisher updates improve rare-digit acquisition while preserving 0--8 recognition, "
            "and at what measured cost?\n\n"
            f"**{status}.** Study `{_text(snapshot['study_id'])}`; sealed scientific snapshot. All evidence is exploratory. "
            "Opening this notebook requires no execution. It never trains, runs inference, downloads data, or repairs artifacts."),
            _progress(snapshot, progress, result), _settings(snapshot, result,
                snapshot_link=os.path.relpath(root / "report_snapshot.json", notebook.parent))]
        introductions = {
            "01_development": "## Development\n\nComplete development paths are descriptive, including explicitly unpaired arms. "
                "After selection, top panels show only the common c/ridge on the main schedule; before selection, "
                "all completed development curves are shown. All candidate settings remain in the lower trade-off/cost "
                "panels and settings table; stationary points are squares. Regime bands follow recorded segments. "
                "Lower points are whole-replica means. Axes count online digits; steps count incoming batches.",
            "02_fresh_main": "## Fresh Main Trajectories\n\nEach row uses its own complete EMA/treatment cohort. "
                "Curves are means across replicas; bands are descriptive pointwise mean +/- 1.96 SE, not paired effect intervals. "
                "No band is drawn for a single pair. Regime shading follows the recorded segments. Frozen initialization is the dotted reference in the legend.",
            "03_fresh_stationary": "## Fresh Stationary Trajectories\n\nStationary controls are not pooled with the main schedule. "
                "The same complete-pair and band conventions apply.",
            "04_precision_exposure": "## Precision, Recall, And Exposure\n\nPrecision at evaluation prevalence 0.1 is undefined "
                "when there are no predicted positives. Gaps are not interpolated. Expected and observed exposure counts are "
                "cumulative online counts, not including initialization (reported in the exposure legend). "
                "A cohort precision point is shown only when every replica has defined precision; an empty path is labeled explicitly. "
                "FPR remains in the sealed snapshot; initialization class counts and unique-ID accounting "
                "are recorded in each `units/assets-<cohort>-rNNN/summary.json` under the study root.",
            "05_paired_cost": "## Paired Differences And Compute\n\n" + result["averaging"] +
                f" Bootstrap draws: {result['bootstrap']['draws']}; persisted seed: {result['bootstrap']['seed']}. "
                "The resampling unit is a complete paired replica trajectory. Crosses show individual differences when fewer than four pairs exist. "
                "The lower-right panel shows main-schedule segment means only; method colors and metric markers identify every series.",
            "06_health": "## Supporting Numerical Health\n\nPre-clipping diagnostics cannot be hidden by PSD projection. "
                "After selection, one completed main path per method at the common setting is shown; "
                "before selection, all development settings are shown. These are supporting diagnostics, "
                "not efficacy gates. Full scalar diagnostics remain in the compact snapshot.",
        }
        for name, title in FIGURES:
            introduction = introductions[name]
            if smoke:
                introduction = introduction.replace("## Development", "## Smoke Development").replace("## Fresh Main", "## Smoke Main").replace("## Fresh Stationary", "## Smoke Stationary")
                title = title.replace("Fresh", "Smoke")
            cells.append(_markdown(name + "-heading", introduction))
            relative = os.path.relpath(destination / (name + ".png"), notebook.parent)
            source = ("from IPython.display import Image, display\n"
                      f"display(Image(filename={relative!r}))")
            png = base64.b64encode((staging / (name + ".png")).read_bytes()).decode()
            cells.append(nbformat.v4.new_code_cell(source, id="plan16-" + name, execution_count=len(cells),
                outputs=[nbformat.v4.new_output("display_data", data={"image/png": png, "text/plain": title})]))
            if name == "05_paired_cost":
                cells.append(_paired_table(result))
        cells.append(_markdown("conclusion", "## Smoke Acceptance Conclusion\n\n"
            "**Smoke acceptance only; not production evidence.** The sealed tiny trajectories exercise the "
            "execution / reporting contract. Their accuracies, intervals, and costs cannot establish production "
            "acquisition, retention, efficacy, or compute overhead. Production results require the separately "
            "identified non-smoke study. No efficacy gate is applied.") if smoke else _conclusion(result))
        snapshot_link = os.path.relpath(root / "report_snapshot.json", notebook.parent)
        analysis_link = os.path.relpath(destination / "analysis.json", notebook.parent)
        cells.append(_markdown("artifacts", f"Compact artifacts: [sealed snapshot]({snapshot_link}), "
            f"[paired analysis and diagnostics]({analysis_link}). PNGs are embedded in executed display outputs "
            "and also stored in the report generation; moving the notebook does not lose its displayed figures."))
        book = nbformat.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "plan16_report": {"schema": 1, "study_id": snapshot["study_id"], "status": status, "smoke": smoke,
                              "snapshot_sha256": _hash(snapshot_bytes), "generation": generation}})
        nbformat.validate(book)
        notebook_bytes = nbformat.writes(book).encode()
        _atomic(staging / "analysis.json", _json_bytes(result))
        _atomic(staging / "report.ipynb", notebook_bytes)
        manifest = {"schema": 1, "notebook": str(notebook), "report_dir": str(destination),
                    "manifest": str(report / "latest.json"), "snapshot_sha256": _hash(snapshot_bytes),
                    "counts": result["counts"]}
        _atomic(staging / "manifest.json", _json_bytes(manifest))
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.is_dir():
                raise
            # A repeated/concurrent identical build already sealed this generation.
            shutil.rmtree(staging)
        _atomic(notebook, notebook_bytes)
        _atomic(report / "latest.json", _json_bytes(manifest))
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish Plan 16 from sealed compact scalar artifacts only")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
