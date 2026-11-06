"""Artifact-only Plan 17 report; no training, data, or torch dependencies.

Read only progress.json and the schema-1 report_snapshot.json. Publish a sealed
generation under root/report, with embedded PNG notebook outputs and an atomic
latest index. Empty snapshots are valid pending reports, not repaired runs.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
import numpy as np

from mnist_experiment.diagonal_lfu.report import (
    _atomic, _change_brief, _curve as _base_curve, _effect, _figure, _finish,
    _hash, _json_bytes, _load, _number, _regimes, _table, _text,
)

from .analysis import (
    ACCURACIES, DEFAULT_SEED, KIND, summarize, trajectories_hash,
    validate_shadows, validate_trajectories,
)

FIGURES = (
    ("01_shadow_calibration", "Shadow Calibration"),
    ("02_development", "Development"),
    ("03_fresh_accuracy", "Fresh Accuracy"),
    ("04_precision_exposure", "Precision And Exposure"),
    ("05_policy", "Policy And Nominal Weights"),
    ("06_paired_cost", "Paired Effects And Compute"),
)
COLORS = {
    "fixed": "#3976a4", "covariance": "#777777",
    "ema_h8": "#278574", "ema_h16": "#98902b", "ema_h32": "#2599a0",
    "kalman_eta0.0001": "#b34c65", "kalman_eta0.001": "#76599b", "kalman_eta0.01": "#b36329",
}


def _name(condition):
    if condition == "fixed":
        return "Fixed"
    if condition == "covariance":
        return "Covariance only"
    if condition.startswith("ema_h"):
        return "EMA H=" + condition.removeprefix("ema_h")
    if condition.startswith("kalman_mean_eta"):
        return "Kalman mean eta=" + condition.removeprefix("kalman_mean_eta")
    if condition.startswith("kalman_eta"):
        return "Kalman eta=" + condition.removeprefix("kalman_eta")
    return condition


def _color(condition):
    if condition in COLORS:
        return COLORS[condition]
    # Candidate IDs also have distinct colors during development screening.
    palette = ("#278574", "#9a6a18", "#3a8a9c") if condition.startswith("ema_") else ("#b34c65", "#76599b", "#b36329")
    return palette[int(_hash(condition.encode())[:8], 16) % len(palette)]


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id="plan17-" + name)


def _fresh_group(group):
    return group["stage"] == "fresh" or group["stage"].startswith("fresh_")


def _group_label(group):
    return f"{group['stage']} / {group['schedule']} / c={group['c']:g}, r={group['ridge']:g}"


def _recorded(value, missing="calibrating"):
    return missing if value is None else str(value)


def _eligibility_note(result):
    required = result.get("required_conditions")
    if required is not None:
        arms = "all four retained arms" if len(required) == 4 else "all required arms"
        return (f"Fresh contrasts require {arms} to complete in the same replica/stage/schedule/c/ridge block. "
                "Completed arms awaiting a full block remain unmatched; development remains pairwise.")
    return ("This snapshot retains legacy independent complete-pair eligibility (no required_conditions metadata). "
            "New fresh analyses require a full four-arm retained-policy block; development remains pairwise.")


def _grid_cohorts(trajectories):
    by_grid = defaultdict(list)
    for t in trajectories:
        by_grid[tuple(r["arrivals"] for r in t["metrics"])].append(t)
    groups = sorted(by_grid.items())
    for i, (grid, arm) in enumerate(groups, start=1):
        note = f"; horizon={grid[-1]:g} digits" if len(groups) > 1 else ""
        if len({g[-1] for g, _ in groups}) < len(groups):
            note += f", grid {i}"
        yield arm, note


def _curve(ax, trajectories, metric, label, **kwargs):
    """Keep length/evaluation-grid cohorts separate; never interpolate gaps."""
    for arm, note in _grid_cohorts(trajectories):
        if metric == "fallback":
            arm = [{**t, "metrics": [{**r, "fallback": None if r.get("fallback") is None else bool(r["fallback"])} for r in t["metrics"]]} for t in arm]
        _base_curve(ax, arm, metric, label + note, **kwargs)


def _progress(snapshot, progress, result):
    p = {**snapshot["progress"], **progress}
    selection = snapshot["selection"]
    required = set(selection.get("conditions") or ())
    blocks = defaultdict(set)
    for t in snapshot["trajectories"]:
        if t["completed"] and _fresh_group(t):
            blocks[(t["stage"], t["schedule"], t["c"], t["ridge"], t["replica"])].add(t["condition"])
    rows = [("Phase / status / revision", f"{snapshot['phase']} / {p.get('status', snapshot['status'])} / {_recorded(p.get('revision'), 'pending')} ")]
    for schedule in ("main", "stationary"):
        count = sum(bool(required) and required <= arms for key, arms in blocks.items() if key[1] == schedule)
        complete = sum(t["completed"] and _fresh_group(t) and t["schedule"] == schedule for t in snapshot["trajectories"])
        planned = selection.get(schedule + "_replicas")
        planned_units = planned * len(required) if planned is not None and required else None
        rows.append((schedule.title() + " fresh blocks / trajectories (complete / planned)",
                     f"{count} / {_recorded(planned)} blocks; {complete} / {_recorded(planned_units)} trajectories"))
    current = p.get("current")
    current = current if isinstance(current, dict) else {}
    step, total = current.get("step", p.get("step")), current.get("total_steps", p.get("total_steps"))
    unit = p.get("active_unit") or current.get("active_unit")
    checkpoint = p.get("last_checkpoint") or p.get("last_accepted_checkpoint") or current.get("last_checkpoint") or current.get("last_accepted_checkpoint")
    storage = p.get("storage") or {}
    used = storage.get("used_bytes")
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    cap = storage.get("limit_bytes", storage.get("max_bytes"))
    free = storage.get("free_bytes", storage.get("filesystem_free_bytes"))
    remaining = p.get("forecast_seconds", selection.get("forecast_fresh_seconds"))
    rows.extend([
        ("Complete / sealed incomplete trajectories", f"{result['counts']['complete']} / {result['counts']['incomplete']}"),
        ("All trajectories (operational complete / planned)", f"{p.get('completed_trajectories', result['counts']['complete'])} / {_recorded(p.get('planned_trajectories'))}"),
        ("Fresh trajectories (operational complete / planned)", f"{_recorded(p.get('completed_fresh'))} / {_recorded(p.get('planned_fresh', selection.get('fresh_trajectory_target')))}"),
        ("Complete / incomplete shadow units", f"{sum(s['completed'] for s in snapshot['shadows'])} / {sum(not s['completed'] for s in snapshot['shadows'])}"),
        ("Active unit / checkpoint", f"{_recorded(unit, 'none (completed)' if snapshot['status'] == 'completed' else 'pending')} / {_recorded(checkpoint, 'pending')}"),
        ("Accepted incoming-batch step / total_steps", f"{_recorded(step, 'pending')}/{_recorded(total)}"),
        ("Heartbeat", _recorded(p.get("heartbeat"), "pending")),
        ("Cumulative active / allocated hours", f"{_number(p.get('active_hours', p.get('active_seconds', 0) / 3600))} / {_number(p.get('allocation_hours', p.get('allocation_seconds', 72000) / 3600))}"),
        ("Remaining forecast hours", _number(remaining / 3600) if remaining is not None else "calibrating"),
        ("Storage used / ceiling / headroom (GB)", " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, cap, cap - used if cap is not None and used is not None else None))),
        ("Filesystem free (GB)", _number(free / 1e9) if free is not None else "pending"),
    ])
    if result.get("required_conditions") is not None:
        rows.append(("Completed fresh arms awaiting a full block", sum(len(g.get("awaiting_full_block_units", [])) for g in result["groups"])))
    if not snapshot["config"].get("smoke"):
        rows.append(("Phase 5 follow-up", "Drafted for review; not started (see final section)"))
    return _markdown("progress", "## Progress\n\n" + _table(("Item", "Recorded state"), rows) +
        "\n\nOperational telemetry is not a liveness check. Scientific plots use only the sealed snapshot. "
        + _eligibility_note(result) + " " +
        "Incomplete prefixes and completed unmatched arms remain explicit, never promoted to fresh evidence.")


def _settings(snapshot, result, *, snapshot_link="report_snapshot.json"):
    selection = snapshot["selection"]
    rows = [("Selected setting", f"c={_recorded(selection.get('common_c'), 'pending')}; ridge={_recorded(selection.get('common_ridge'), 'pending')}",
             "Frozen: " + _recorded(selection.get("frozen_at"), "pending")),
            ("Fresh allocation", f"main={_recorded(selection.get('main_replicas'))}; stationary={_recorded(selection.get('stationary_replicas'))}",
             ", ".join(_name(c) for c in selection.get("conditions", [])) or "pending"),
            ("Measured forecast", (_number(selection["forecast_fresh_seconds"] / 3600) + " h") if selection.get("forecast_fresh_seconds") is not None else "pending",
             _recorded(selection.get("fresh_trajectory_target"), "pending") + " fresh trajectories; optional variants are development only")]
    for group in result["groups"]:
        rows.append(("Setting", _group_label(group), f"{len(group['complete_units'])} complete / {len(group['incomplete_units'])} incomplete"))
    for change in snapshot["changes"][-6:]:
        revision = change.get("revision", change.get("phase", "recorded")) if isinstance(change, dict) else "recorded"
        rows.append(("Change " + str(revision), _change_brief(change), "FULL_EVENT_LINK"))
    table = _table(("Item", "Setting / significant change", "Evidence"), rows).replace("FULL_EVENT_LINK", f"[Full event and validation]({snapshot_link})")
    config = snapshot["config"]
    keys = ("samples_per_step", "samples_per_arrival", "dtype", "fisher_dtype", "curvature_dtype", "eval_per_class", "anchor_refresh_steps")
    concise = "; ".join(f"{k}={config[k]}" for k in keys if k in config) or "pending"
    return _markdown("settings", "## Settings And Glossary\n\n"
        "**Action pi** weights the arriving batch in both diagonal Fisher forgetting and EWC odds; "
        "**prevalence p** controls digit-9 arrivals, not the action. The action is frozen before the current batch; "
        "its observed-label score updates the observer for the next action only. **Anchor** is a common local "
        "score coordinate, not an oracle. The **observation clock** advances on incoming batches, including quiet holds.\n\n"
        "Observer scores differentiate log likelihood using the stream's observed labels; Fisher diagonals "
        "integrate over model labels instead. These are not interchangeable.\n\n"
        "**Movement** is the observer's posterior mean contribution; **uncertainty** is a separate covariance "
        "contribution, not proof of environmental motion. Kalman covariance is not another Fisher estimand. "
        "**Covariance-only** sets movement energy to zero in the action rule; it is not the Kalman filter's covariance. "
        "EMA is a Holt level/trend point plug-in with possible squared-noise bias. Kalman versus EMA tests the "
        "complete estimator package, including uncertainty, not just its gain.\n\n"
        "**9 recall** conditions on true 9s; **old retention** is macro accuracy over digits 0--8. "
        "Precision is standardized to fixed p_ref=.1 and remains undefined with no predicted positives. "
        "Predictive metrics use independent stratified holdouts, separate from training and Fisher inputs; "
        "development and fresh evaluation holdouts are independent. Predictive NLL is pure NLL, excluding every regularizer. "
        "Mixture-weighted all-accuracy is secondary, never a success gate.\n\n"
        "A step is one incoming batch; `arrivals = step * samples_per_step` counts **online digits observed**, "
        "excluding initialization. Exact nominal concentration q gives weight-based effective size 1/q. "
        "The product of (1-pi) is the initialization's nominal remaining weight, not measured knowledge.\n\n" + table +
        f"\n\nRecorded core settings: {_text(concise)}. Full settings, IDs, significant changes and validation: "
        f"[sealed snapshot]({snapshot_link}). No pooling across stages, schedules, c, ridge, or policy IDs.")


def _matched(snapshot, result, *, control="fixed"):
    by_id = {t["unit_id"]: t for t in snapshot["trajectories"]}
    for group in result["groups"]:
        if not _fresh_group(group):
            continue
        for comparison in group["comparisons"]:
            if comparison["control"] == control and comparison["n_pairs"]:
                yield group, comparison, [by_id[u] for u in comparison["control_units"]], [by_id[u] for u in comparison["treatment_units"]]


def _cohorts(snapshot, result):
    """Draw each arm once, with explicit context if its paired cohort differs."""
    matches = list(_matched(snapshot, result))
    seen = set()
    contexts = {(g["stage"], g["c"], g["ridge"]) for g, *_ in matches}
    control_sets = defaultdict(set)
    for g, _, controls, _ in matches:
        control_sets[(g["stage"], g["schedule"], g["c"], g["ridge"])].add(tuple(t["unit_id"] for t in controls))
    for group, comparison, controls, targets in matches:
        for arm, condition in ((controls, "fixed"), (targets, comparison["condition"])):
            key = tuple(t["unit_id"] for t in arm)
            if key in seen:
                continue
            seen.add(key)
            label = _name(condition)
            if len(contexts) > 1:
                label += f"; {group['stage']}, c={group['c']:g}, r={group['ridge']:g}"
            if condition == "fixed" and len(control_sets[(group['stage'], group['schedule'], group['c'], group['ridge'])]) > 1:
                label += " / " + _name(comparison["condition"]) + " cohort"
            for cohort, note in _grid_cohorts(arm):
                yield group, cohort, condition, label + note


def _shadow_calibration(snapshot, result, destination):
    complete = [s for s in snapshot["shadows"] if s["completed"]]
    with _figure() as (fig, axes):
        labels = []
        for i, shadow in enumerate(complete):
            summary, condition = shadow["summary"], shadow["condition"]
            labels.append(f"{shadow['schedule']} {_name(condition)} replica {shadow['replica']}")
            color = _color(condition)
            axes[0, 0].scatter(summary["log_score"], i, color=color, s=18)
            if summary.get("pi_mean") is not None:
                axes[1, 0].scatter(summary["pi_mean"], i, color=color, s=18)
            for ax, schedule in ((axes[0, 1], "main"), (axes[1, 1], "stationary")):
                if shadow["schedule"] == schedule:
                    regimes = summary["regime_log_scores"]
                    ordered = [segment for segment in dict.fromkeys(row.get("segment") for row in shadow["metrics"])
                               if segment in regimes]
                    ordered.extend(segment for segment in regimes if segment not in ordered)
                    ax.plot(ordered, [regimes[segment] for segment in ordered], marker=".", color=color,
                            label=f"{_name(condition)} replica {shadow['replica']}")
        for ax in (axes[0, 0], axes[1, 0]):
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
        axes[0, 0].set(title="Sealed regime-balanced predictive log score", xlabel="Mean of regime means (higher is better)")
        axes[1, 0].set(title="Shadow recommendation, not an applied learning policy", xlabel="Recorded mean pi")
        axes[0, 1].set(title="Main: per-regime predictive log scores", ylabel="Log score (higher is better)")
        axes[1, 1].set(title="Stationary: per-regime predictive log scores", ylabel="Log score (higher is better)")
        for ax in (axes[0, 1], axes[1, 1]):
            ax.tick_params(axis="x", labelsize=6, rotation=15)
        _finish(axes)
        fig.savefig(destination, dpi=120)


def _development(snapshot, result, destination):
    groups = [g for g in result["groups"] if not _fresh_group(g)]
    selection = snapshot["selection"]
    selected = selection.get("common_c") is not None and selection.get("common_ridge") is not None
    curve_groups = [g for g in groups if g["schedule"] == "main" and g["c"] == selection["common_c"] and g["ridge"] == selection["common_ridge"]] if selected else groups
    with _figure() as (fig, axes):
        fig.set_size_inches(12.4, 7.8)
        shaded = False
        for group in groups:
            trajectories = [t for t in snapshot["trajectories"] if t["unit_id"] in group["complete_units"]]
            for condition in group["conditions"]:
                arm = [t for t in trajectories if t["condition"] == condition]
                label = f"{_name(condition)}; {group['schedule']}, c={group['c']:g}"
                if len({g['ridge'] for g in groups}) > 1:
                    label += f", r={group['ridge']:g}"
                if len({g['stage'] for g in groups}) > 1:
                    label += "; " + group['stage']
                if arm and group in curve_groups:
                    if not shaded:
                        for ax in axes[0]:
                            _regimes(ax, arm[0])
                        shaded = True
                    for ax, metric in zip(axes[0], ACCURACIES):
                        _curve(ax, arm, metric, label, color=_color(condition), linestyle="--" if group["schedule"] == "stationary" else "-")
                points = [r for r in group["replicas"] if r["condition"] == condition]
                if points:
                    axes[1, 0].scatter([100 * r["observation_mean"]["old_accuracy"] for r in points],
                                       [100 * r["observation_mean"]["nine_recall"] for r in points], color=_color(condition), s=18,
                                       marker="s" if group["schedule"] == "stationary" else "o", label=label)
                for r in points:
                    cost = r["costs"].get("seconds_per_arrival")
                    if cost is not None:
                        axes[1, 1].scatter(cost, 100 * r["observation_mean"]["nine_recall"], color=_color(condition), s=18)
        for ax, title in zip(axes[0], ("Development: 9 recall (%)", "Development: 0--8 retention (%)")):
            ax.set(title=title, ylim=(0, 100))
        axes[1, 0].set(title="Complete-path acquisition / retention trade-off", xlabel="Observation-averaged 0--8 accuracy (%)", ylabel="Observation-averaged 9 recall (%)")
        axes[1, 1].set(title="Measured complete-path cost", xlabel="Seconds per online digit", ylabel="Observation-averaged 9 recall (%)")
        _finish(axes)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        for ax in axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if unique:
            fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=3, fontsize=6)
        fig.savefig(destination, dpi=120)


def _fresh_accuracy(snapshot, result, destination):
    cohorts = list(_cohorts(snapshot, result))
    horizons = {}
    with _figure() as (fig, axes):
        fig.set_size_inches(15, 8)
        frozen = set()
        for row, schedule in enumerate(("main", "stationary")):
            references = [t for group, arm, _, _ in cohorts if group['schedule'] == schedule for t in arm]
            if references:
                reference = max(references, key=lambda t: t['metrics'][-1]['arrivals'])
                horizons[row] = reference['metrics'][-1]['arrivals']
                for ax in axes[row]:
                    _regimes(ax, reference)
        for group, arm, condition, label in cohorts:
            row = 1 if group["schedule"] == "stationary" else 0
            for col, metric in enumerate(ACCURACIES):
                ax = axes[row, col]
                _curve(ax, arm, metric, label, color=_color(condition), linestyle="--" if condition == "fixed" else "-")
                key = row, col, tuple(t["unit_id"] for t in arm)
                if condition == "fixed" and key not in frozen:
                    values = [t["metrics"][0].get("frozen_" + metric,
                              t["summary"].get("frozen_initial", {}).get(metric)) for t in arm]
                    if values and all(v is not None for v in values):
                        horizon = arm[0]["metrics"][-1]["arrivals"]
                        ax.plot([0, horizon], [100 * np.mean(values)] * 2, color="#626262", linestyle=":",
                                label=f"Frozen initial reference; horizon={horizon:g} (n={len(arm)})")
                    frozen.add(key)
        for row, schedule in enumerate(("Main", "Stationary")):
            for col, endpoint in enumerate(("9 recall", "0--8 retention accuracy")):
                note = f"\nRegime bands: {horizons[row]:g}-digit reference" if row in horizons else ""
                axes[row, col].set(title=f"{schedule}: {endpoint} (%)" + note, ylim=(0, 100))
        _finish(axes)
        for row in range(2):
            handles, labels = axes[row, 1].get_legend_handles_labels()
            for ax in axes[row]:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            unique = dict(zip(labels, handles))
            if unique:
                axes[row, 1].legend(unique.values(), unique.keys(), loc="center left",
                                    bbox_to_anchor=(1.02, .5), fontsize=7)
        fig.savefig(destination, dpi=120)


def _precision_exposure(snapshot, result, destination):
    cohorts = list(_cohorts(snapshot, result))
    horizons = {}
    with _figure() as (fig, axes):
        fig.set_size_inches(14, 10)
        exposed = set()
        undefined = defaultdict(list)
        for col, schedule in enumerate(("main", "stationary")):
            references = [t for group, arm, condition, _ in cohorts if group['schedule'] == schedule and condition == 'fixed' for t in arm]
            if references:
                reference = max(references, key=lambda t: t['metrics'][-1]['arrivals'])
                horizons[col] = reference['metrics'][-1]['arrivals']
                _regimes(axes[1, col], reference)
        for group, arm, condition, label in cohorts:
            col = 1 if group["schedule"] == "stationary" else 0
            ax = axes[0, col]
            values = np.asarray([[np.nan if row.get("nine_precision") is None else row["nine_precision"] for row in t["metrics"]] for t in arm], dtype=float)
            valid = np.all(np.isfinite(values), axis=0)
            undefined[col].append((int(valid.sum()), len(valid)))
            if valid.any():
                _curve(ax, arm, "nine_precision", label, color=_color(condition))
            key = group["stage"], group["schedule"], group["c"], group["ridge"], tuple(t["replica"] for t in arm)
            if condition != "fixed" or key in exposed:
                continue
            exposed.add(key)
            initial = [t["summary"].get("initial_nines", t["metrics"][0].get("initial_nines")) for t in arm]
            note = f"; initial mean={np.mean(initial):g}" if all(v is not None for v in initial) else "; initialization count pending"
            for metric, color, style, caption in (("observed_nines", "#3976a4", "-", "Realized online 9s"), ("expected_nines", "#b34c65", "--", "Expected online 9s")):
                _curve(axes[1, col], arm, metric, caption + note + "; " + label, color=color, linestyle=style, percent=False)
        for col, schedule in enumerate(("Main", "Stationary")):
            if undefined[col]:
                ax = axes[0, col]
                if not ax.has_data():
                    ax.text(.5, .65, "No complete-cohort precision path.\nAt least one replica is undefined at every checkpoint.", ha="center", va="center", transform=ax.transAxes)
                coverage = undefined[col]
                if len(set(coverage)) == 1:
                    count, total = coverage[0]
                    note = f"{count}/{total} defined checkpoints per curve"
                else:
                    rates = [100 * count / total for count, total in coverage]
                    note = f"Defined checkpoints: {min(rates):.0f}--{max(rates):.0f}% across curves"
                ax.text(.02, .02, note, fontsize=7, va="bottom", transform=ax.transAxes)
            axes[0, col].set(title=schedule + ": 9 precision at p_ref=.1 (%)", ylim=(0, 100), xlabel="Online digits observed")
            note = f"\nRegime bands: {horizons[col]:g}-digit reference" if col in horizons else ""
            axes[1, col].set(title=schedule + ": shared cumulative digit-9 exposure" + note, ylabel="Online digit-9 count")
        _finish(axes)
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(unique.values(), unique.keys(), loc="upper center",
                          bbox_to_anchor=(.5, -.17), fontsize=6)
        fig.savefig(destination, dpi=120)


def _policy(snapshot, result, destination):
    cohorts = [(g, arm, condition, label) for g, arm, condition, label in _cohorts(snapshot, result) if g["schedule"] == "main"]
    if not cohorts:
        # Earlier development evidence may be shown, but is never mixed with fresh.
        selection = snapshot["selection"]
        for group in result["groups"]:
            if _fresh_group(group) or group["schedule"] != "main":
                continue
            if selection.get("common_c") is not None and (group["c"] != selection["common_c"] or group["ridge"] != selection.get("common_ridge")):
                continue
            for condition in group["conditions"]:
                arm = [t for t in snapshot["trajectories"] if t["unit_id"] in group["complete_units"] and t["condition"] == condition]
                if arm:
                    cohorts.append((group, arm, condition, _name(condition) + "; " + _group_label(group)))
    with _figure(2, 3) as (fig, axes):
        fig.set_size_inches(13, 7.5)
        fields = (
            (("applied_pi", "-", "applied"), ("raw_pi", "--", "raw")),
            (("q", "-", "q"),),
            (("log_replacement_mass", "-", "log remaining weight"),),
            (("motion_energy", "-", "mean movement"), ("uncertainty_energy", "--", "uncertainty")),
            (("noise_scale", "-", "measurement noise scale"),),
            (("boundary_low", "-", "lower boundary"), ("boundary_high", "--", "upper boundary"), ("fallback", ":", "fallback")),
        )
        for group, arm, condition, label in cohorts:
            for ax, metrics in zip(axes.flat, fields):
                for metric, style, caption in metrics:
                    if any(row.get(metric) is not None for t in arm for row in t["metrics"]):
                        _curve(ax, arm, metric, label, color=_color(condition), linestyle=style, percent=False)
        titles = ("Actions: applied (solid), raw (dashed)", "Exact nominal weight concentration q", "Nominal initialization weight (log)",
                  "Mean movement (solid) / uncertainty (dashed)", "Measurement-noise scale", "Boundaries: low (solid), high (--), fallback (:)")
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
        axes[0, 1].set_yscale("log")
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_ylim(bottom=0)
        axes[1, 2].set_ylim(0, 1)
        _finish(axes)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        for ax in axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if unique:
            fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=2, fontsize=7)
        fig.savefig(destination, dpi=120)


def _fresh_comparisons(result):
    return [(g, c) for g in result["groups"] if _fresh_group(g) for c in g["comparisons"]]


def _paired_cost(snapshot, result, destination):
    comparisons = [(g, c) for g, c in _fresh_comparisons(result) if c["n_pairs"]]
    with _figure(3, 2) as (fig, axes):
        fig.set_size_inches(13, 11.5)
        labels = []
        for i, (group, comparison) in enumerate(comparisons):
            label = f"{group['schedule']}: {_name(comparison['condition'])} - {_name(comparison['control'])}\nc={group['c']:g}, r={group['ridge']:g}; n={comparison['n_pairs']}"
            labels.append(label)
            for ax, metric in zip(axes[0], ACCURACIES):
                value = comparison["observation_mean_pp"][metric]
                ax.plot(value["mean"], i, "o", color="#3976a4", markersize=4)
                if value["ci95"] is not None:
                    ax.plot(value["ci95"], [i] * 2, color="#3976a4", linewidth=1)
                if comparison["n_pairs"] < 4:
                    ax.scatter(value["individual"], [i] * comparison["n_pairs"], marker="x", color="#3976a4", s=10)
            ratio = comparison["costs"].get("wall_seconds", {}).get("ratio", {})
            if ratio.get("mean") is not None:
                axes[2, 0].plot(100 * (ratio["mean"] - 1), i, "o", color="#278574", markersize=4)
                if ratio.get("ci95") is not None:
                    axes[2, 0].plot([100 * (v - 1) for v in ratio["ci95"]], [i] * 2, color="#278574")
        for ax in (axes[0, 0], axes[0, 1], axes[2, 0]):
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
            ax.axvline(0, color="#777777", linewidth=.7)
        for ax, title in zip(axes[0], ("9 recall change", "0--8 retention change")):
            ax.set(title=title, xlabel="Full-path mean difference (pp); positive favors treatment")
        regime_labels = []
        regime_comparisons = [(g, c) for g, c in comparisons if g["schedule"] == "main"
                              and c["control"] == "fixed" and c["condition"].startswith("kalman_")]
        order = ("low_hold_initial", "rise", "high_hold", "return", "low_hold_final")
        for group, comparison in regime_comparisons:
            segments = comparison["segments"]
            ordered = [segment for segment in order if segment in segments]
            ordered.extend(segment for segment in segments if segment not in ordered)
            for segment in ordered:
                i = len(regime_labels)
                label = segment.replace("_", " ")
                if len(regime_comparisons) > 1:
                    label += "; " + _name(comparison["condition"]) + "; " + _group_label(group)
                regime_labels.append(label)
                for ax, metric in zip(axes[1], ACCURACIES):
                    value = segments[segment]["observation_mean_pp"][metric]
                    ax.plot(value["mean"], i, "o", color="#b34c65", markersize=4)
                    if value["ci95"] is not None:
                        ax.plot(value["ci95"], [i] * 2, color="#b34c65", linewidth=1)
        for ax, metric in zip(axes[1], ("9 recall", "0--8 retention")):
            ax.set_yticks(range(len(regime_labels)), regime_labels, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color="#777777", linewidth=.7)
            if not regime_labels:
                ax.text(.5, .5, "Pending complete main Kalman pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=f"Main regimes: Kalman - fixed, {metric}", xlabel="Within-regime mean difference (pp)")
        axes[2, 0].set(title="Measured whole-unit runtime overhead", xlabel="Mean paired wall ratio minus one (%)")
        for group in result["groups"]:
            if not _fresh_group(group):
                continue
            for condition in group["conditions"]:
                rows = [r for r in group["replicas"] if r["condition"] == condition]
                names = ("solver_seconds", "curvature_seconds", "controller_seconds", "evaluation_seconds", "checkpoint_seconds")
                costs = [np.mean([r["costs"][name] for r in rows if name in r["costs"]]) if any(name in r["costs"] for r in rows) else np.nan for name in names]
                if np.isfinite(costs).any():
                    label = _name(condition) + "; " + group['schedule']
                    if len({(g['stage'], g['c'], g['ridge']) for g in result['groups'] if _fresh_group(g)}) > 1:
                        label += "; " + _group_label(group)
                    axes[2, 1].plot([name.removesuffix("_seconds") for name in names], costs, marker=".", color=_color(condition), label=label)
        axes[2, 1].set(title="Measured cost components (complete paths)", ylabel="Mean seconds")
        axes[2, 1].tick_params(axis="x", labelsize=7, rotation=15)
        _finish(axes)
        handles, labels = axes[2, 1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if axes[2, 1].get_legend() is not None:
            axes[2, 1].get_legend().remove()
        if unique:
            fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=4, fontsize=7)
        fig.savefig(destination, dpi=120)


def _paired_table(result):
    rows = []
    for group, comparison in _fresh_comparisons(result):
        effects = comparison["observation_mean_pp"]
        rows.append((_group_label(group), _name(comparison["condition"]) + " - " + _name(comparison["control"]), comparison["n_pairs"],
                     _effect(effects["nine_recall"], signed=True), _effect(effects["old_accuracy"], signed=True)))
    return _markdown("paired-summary", "### Paired Summary\n\n" + _table(
        ("Cohort / setting", "Contrast", "Pairs", "Full-path 9 recall (pp, CI)", "Full-path old retention (pp, CI)"),
        rows or [("pending", "fixed and covariance contrasts pending", 0, "pending", "pending")]) +
        "\n\nFive contrasts per schedule for four retained policies: all adaptive policies versus fixed, "
        "and movement policies also versus covariance only. Individual differences are plotted for small cohorts. "
        + _eligibility_note(result) + " " +
        "Nominal 95% whole-paired-replica percentile intervals are absent for one pair and unreliable below four. "
        "Multiple comparisons and repeated inspection are exploratory and unadjusted. "
        "Exact individual effects, endpoints, regime-specific means and intervals, full policy IDs and cost components remain in analysis.json.")


def _conclusion_summary(result):
    comparisons = [(g, c) for g, c in _fresh_comparisons(result) if c["control"] == "fixed" and c["n_pairs"]]
    if not comparisons:
        return ""
    rows = []
    contexts = {(g["stage"], g["c"], g["ridge"]) for g, _ in comparisons}
    for group, comparison in comparisons:
        label = _name(comparison["condition"]) + f" (n={comparison['n_pairs']})"
        if len(contexts) > 1:
            label += "; " + _group_label(group)
        effects = comparison["observation_mean_pp"]
        ratio = comparison["costs"].get("wall_seconds", {}).get("ratio", {}).get("mean")
        rows.append(("Changing" if group["schedule"] == "main" else "Stationary", label,
                     _effect(effects["nine_recall"], signed=True), _effect(effects["old_accuracy"], signed=True),
                     "pending" if ratio is None else f"{100 * (ratio - 1):.1f}%"))
    return ("Fresh policies versus the tuned fixed control, observation-averaged over each full trajectory. "
            "**Positive means better accuracy; the two schedules are not pooled.**\n\n" +
            _table(("Schedule", "Method", "9 recall change (pp)", "0--8 retention change (pp)", "Extra wall time"), rows) +
            "\n\nBrackets are nominal 95% paired-bootstrap confidence intervals. Extra wall time is the mean "
            "paired runtime ratio minus one, not an isolated observer-update benchmark. Movement-arm wall time "
            "may include measured one-time observer calibration; shared initial learner/Fisher costs are excluded.\n\n")


def _conclusion_reading(result):
    sentences = []
    for group, comparison in _fresh_comparisons(result):
        if comparison["control"] != "fixed" or not comparison["condition"].startswith("kalman_") or not comparison["n_pairs"]:
            continue
        clauses = []
        for metric, label in (("nine_recall", "9 recall"), ("old_accuracy", "0--8 retention")):
            interval = comparison["observation_mean_pp"][metric]["ci95"]
            if interval is None:
                clauses.append(f"{label} has no estimated interval yet")
            elif interval[0] > 0:
                clauses.append(f"higher {label}")
            elif interval[1] < 0:
                clauses.append(f"lower {label}")
            else:
                clauses.append(f"the {label} difference remains unresolved")
        schedule = "Changing schedule" if group["schedule"] == "main" else "Stationary schedule"
        sentences.append(f"{schedule}: {'; '.join(clauses)}.")
    if not sentences:
        return ""
    text = "**Kalman versus fixed, reading the nominal intervals:** " + " ".join(sentences) + "\n\n"
    for group, comparison in _fresh_comparisons(result):
        if (group["schedule"] == "main" and comparison["control"] == "covariance"
                and comparison["condition"].startswith("kalman_") and comparison["n_pairs"]):
            intervals = [comparison["observation_mean_pp"][metric]["ci95"] for metric in ACCURACIES]
            if all(interval is not None and interval[0] > 0 for interval in intervals):
                text += ("Against covariance-only on changing schedules, Kalman improves both accuracies at the nominal "
                         "interval level. This supports the complete Kalman policy package over that control; "
                         "it does not isolate genuine-motion detection from its uncertainty contribution.\n\n")
    return text


def _conclusion(result):
    unmatched = {u for _, c in _fresh_comparisons(result) for name in ("unpaired_control_units", "unpaired_treatment_units") for u in c[name]}
    unmatched.update(u for g in result["groups"] for u in g.get("awaiting_full_block_units", []))
    summary = _conclusion_summary(result)
    return _markdown("conclusion", "## Conclusion: Original Phases 0--4\n\n" + (summary or "No complete fresh main paired evidence yet. Development and shadow calibration do not establish closed-loop predictive value.\n\n") +
        _conclusion_reading(result) +
        "Movement-policy contrasts against covariance-only remain in the paired summary and effect plot. "
        "Higher recall with lower old retention is a trade-off, not improvement in both. An interval crossing zero "
        "does not prove preservation. No all-accuracy success gate or efficacy threshold is applied.\n\n"
        f"{result['counts']['incomplete']} sealed incomplete trajectories; {len(unmatched)} unmatched fresh arms. "
        "Unpaired development controls are descriptive. Nominal intervals are exploratory under multiple comparisons "
        "and repeated inspection. Equal digit exposure does not establish sample savings or equal compute.\n\n"
        "Observer calibration is distinct from predictive value. Uncertainty alone is not environmental movement; "
        "anchor mapping, diagonal curvature, possible EMA squared-noise bias and finite-tolerance solves remain limitations. "
        "The primary study uses a fixed initialization anchor; it does not validate reliable ongoing reanchoring. "
        "Frequent recovery-checkpoint I/O is substantial in this tiny-model benchmark, so wall-time ratios are not deployment FLOP estimates. "
        "Nominal weights do not measure knowledge. These class-specific proxies are not separate visual tasks, "
        "and this single-architecture study does not establish global policy optimality.")


def _retention_followup():
    return _markdown("retention-followup", r"""## Phase 5: Acquisition At Comparable Retention

**Drafted for review; not started.** [Design and compute estimates](plan17.md#phase-5-acquisition-at-comparable-retention).

**Why this follow-up?** Efficient continual learning needs new learning without
buying it through greater forgetting. The original study calibrated observer
noise and ranked average acquisition/retention, but did not match retention
between methods. Its results remain useful operating points; they do not tell
us whether adaptive $\pi$ improves acquisition at comparable old-digit accuracy.
Calibration may or may not explain the observed acquisition deficit.

**Question:** can adaptive policies improve digit-9 recall at comparable
observation-averaged 0--8 accuracy to fixed $\pi=.025$?

Development will vary each adaptive policy's existing ridge strength, keeping
the reference at ridge ratio $.1$ and the observer settings fixed. The matching
target is a retention difference within **$\pm0.5$ percentage points**, measured
over the whole trajectory. Changing and stationary schedules are calibrated
and reported separately. Settings then freeze before fresh paired confirmation.

Both accuracy differences will appear with 95% intervals. A retention interval
crossing zero does not establish equality; its width and remaining gap matter.
Unattainable matches and unsuccessful outcomes will remain visible. No Phase 5
results are available yet; the original results above are preserved.
""")


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite scalar in reporting artifact")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _validated(root):
    progress, progress_bytes = _load(root / "progress.json")
    snapshot, snapshot_bytes = _load(root / "report_snapshot.json")
    _finite_json(snapshot)
    _finite_json(progress)
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1:
        raise ValueError("report_snapshot.json requires compatible schema 1")
    for name in ("study_id", "status", "phase", "config", "progress", "selection", "changes", "trajectories", "shadows", "analysis"):
        if name not in snapshot:
            raise ValueError(f"report snapshot missing {name}")
    for name in ("study_id", "status", "phase"):
        if not isinstance(snapshot[name], str) or not snapshot[name]:
            raise ValueError(f"snapshot.{name} must be a nonempty string")
    for name in ("config", "progress", "selection"):
        if not isinstance(snapshot[name], dict):
            raise ValueError(f"snapshot.{name} must be an object")
    if not isinstance(snapshot["changes"], list):
        raise ValueError("snapshot.changes must be a list")
    for telemetry in (progress, snapshot["progress"]):
        if telemetry.get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
            raise ValueError("progress and snapshot study_id mismatch")
    validate_trajectories(snapshot["trajectories"])
    validate_shadows(snapshot["shadows"])
    result = snapshot["analysis"]
    if result in (None, {}):
        config = snapshot["config"]
        result = summarize(snapshot["trajectories"], bootstrap_draws=config.get("bootstrap_draws", 2000),
                           seed=config.get("bootstrap_seed", config.get("seed", DEFAULT_SEED)),
                           required_conditions=snapshot["selection"].get("conditions"))
    else:
        if not isinstance(result, dict) or type(result.get("schema")) is not int or result["schema"] != 1 or result.get("kind") != KIND or result.get("trajectories_sha256") != trajectories_hash(snapshot["trajectories"]):
            raise ValueError("snapshot.analysis must be empty or matching Plan 17 analysis.summarize output")
        bootstrap = result.get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise ValueError("snapshot.analysis missing bootstrap metadata")
        # A hash alone cannot validate corrupted cached effects or cohort IDs.
        expected = summarize(snapshot["trajectories"], bootstrap_draws=bootstrap.get("requested_draws"), seed=bootstrap.get("seed"),
                             required_conditions=result.get("required_conditions"))
        if "required_conditions" in result:
            selected = snapshot["selection"].get("conditions")
            if (not isinstance(selected, (list, tuple)) or any(not isinstance(c, str) for c in selected)
                    or len(set(selected)) != len(selected) or set(selected) != set(result["required_conditions"] or [])):
                raise ValueError("snapshot.analysis.required_conditions must match snapshot.selection.conditions")
        if result != expected:
            raise ValueError("snapshot.analysis is corrupt or incompatible with sealed trajectories")
    return snapshot, progress, result, snapshot_bytes, progress_bytes


def build_report(root: Path, notebook: Path | None = None) -> dict:
    """Publish atomically, failing before publication on incompatible artifacts."""
    root = Path(root).resolve()
    notebook = Path(notebook).resolve() if notebook is not None else Path(__file__).resolve().parents[1] / "plan17_results.ipynb"
    if notebook.suffix != ".ipynb":
        raise ValueError("notebook output must have .ipynb suffix")
    if notebook.exists():
        existing = nbformat.read(notebook, as_version=4)
        if existing.metadata.get("plan17_retention_report"):
            raise ValueError("Phase 5 owns this combined notebook; use its retention.report publisher or a different --notebook path")
    snapshot, progress, result, snapshot_bytes, progress_bytes = _validated(root)
    package = Path(__file__).parent
    sources = (package / "report.py", package / "analysis.py",
               package.parent / "diagonal_lfu" / "report.py", package.parent / "diagonal_lfu" / "analysis.py")
    code_hash = _hash(b"".join(source.read_bytes() for source in sources))
    generation = "generation-" + _hash(snapshot_bytes + progress_bytes + code_hash.encode() + str(notebook).encode())[:24]
    report_dir = root / "report"
    destination = report_dir / generation
    report_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=report_dir))
    try:
        renderers = (_shadow_calibration, _development, _fresh_accuracy, _precision_exposure, _policy, _paired_cost)
        for (name, _), render in zip(FIGURES, renderers):
            render(snapshot, result, staging / (name + ".png"))
        partial = snapshot["status"] not in ("completed", "complete") or result["counts"]["incomplete"]
        status = "Partial / incomplete" if partial else "Completed snapshot"
        smoke = snapshot["config"].get("smoke") is True
        banner = "**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.** Tiny trajectories validate implementation and reporting only. Their curves, intervals and measured costs cannot support production acquisition, retention or overhead claims.\n\n" if smoke else ""
        cells = [_markdown("title", "# Plan 17: Predictable Adaptive Actions" + (" (Smoke Acceptance Only)" if smoke else "") + "\n\n" + banner +
            "Do past-only fixed-anchor observations help choose the next batch's weight while balancing rare-digit acquisition and old-digit retention?\n\n" +
            f"**{status}.** Study `{_text(snapshot['study_id'])}`. Exploratory sealed-artifact report; opening this notebook never trains, runs inference, downloads data, or repairs missing artifacts."),
            _progress(snapshot, progress, result), _settings(snapshot, result, snapshot_link=os.path.relpath(root / "report_snapshot.json", notebook.parent))]
        captions = (
            "Only completed sealed shadow summaries are shown. Log score is the equally weighted mean of per-segment mean one-step predictive innovation log scores, not an observation-weighted score. Higher is better. Shared-state calibration is not closed-loop learning efficacy; unsealed/incomplete shadow prefixes never enter this panel. Mapping and innovation diagnostics remain in the snapshot.",
            "Development paths are descriptive, including unmatched arms. Top curves use the selected main c/ridge when frozen; all complete candidate settings remain in the trade-off/cost panels. Stationary points are squares. Stages and policy IDs are never pooled.",
            "Main and stationary schedules occupy separate rows. " + _eligibility_note(result) + " Only eligible complete fixed/treatment pairs enter curves; labels distinguish different paired cohorts. Variable horizons/evaluation grids form separate labeled grid cohorts in actual digit units: no visual interpolation or truncation is used. Regime bands follow the longest recorded reference horizon, identified in each panel; shorter paths retain their own timing. Dotted frozen-model references are context only. Bands show descriptive pointwise mean +/- 1.96 SE across replicas within a grid cohort, not paired effect intervals; single replicas have no band. Whole-path and regime statistics still use each replica's original grid.",
            "Precision is at fixed p_ref=.1. A point exists only if every replica in its complete paired grid cohort has defined precision; undefined-cohort gaps are never interpolated. Variable horizons are kept as separate labeled lines. Exposure is expected/realized cumulative online digit-9 count; shared initialization 9s are labeled separately and excluded from the x axis. FPR and unique-ID accounting remain in the snapshot.",
            "Main-schedule complete paired cohorts are shown, or clearly labeled development paths while fresh evidence is pending. Raw/applied actions, exact q, log nominal initialization weight, posterior mean movement versus uncertainty, noise scale and boundary/fallback fractions are algorithm diagnostics. Bands are descriptive pointwise mean +/- 1.96 SE within a length cohort, displayed within physical axis bounds. The effective size 1/q and nominal mass are weight bookkeeping, not independent information or measured knowledge. Initial missing diagnostics remain gaps.",
            result["averaging"] + f" Bootstrap draws: {result['bootstrap']['draws']}; persisted seed: {result['bootstrap']['seed']}. Whole-replica resampling preserves within-path covariance; identical cohorts share bootstrap blocks across endpoints and contrasts. Top effect panels show full-trajectory observation means; middle panels show every predefined main regime for Kalman versus fixed. Regime intervals are exploratory, not independent confirmations. Endpoints remain in analysis.json. Positive accuracy effects favor the treatment. Costs are measured, not equalized: movement-arm wall time may include one-time observer calibration; shared initial learner/Fisher costs are excluded. Component curves show only recorded costs, never inferred zeroes.",
        )
        for (name, title), caption in zip(FIGURES, captions):
            display_title = title.replace("Fresh", "Smoke") if smoke else title
            cells.append(_markdown(name + "-heading", "## " + display_title + "\n\n" + caption))
            relative = os.path.relpath(destination / (name + ".png"), notebook.parent)
            png = base64.b64encode((staging / (name + ".png")).read_bytes()).decode()
            cells.append(nbformat.v4.new_code_cell("from IPython.display import Image, display\n" + f"display(Image(filename={relative!r}))",
                id="plan17-" + name, execution_count=len(cells), outputs=[nbformat.v4.new_output("display_data", data={"image/png": png, "text/plain": display_title})]))
            if name == "06_paired_cost":
                cells.append(_paired_table(result))
        cells.append(_markdown("conclusion", "## Smoke Acceptance Conclusion\n\n**Smoke acceptance only; not production evidence.** These tiny artifacts exercise the execution/reporting contract. No production predictive or runtime claim is supported; no efficacy gate is applied.") if smoke else _conclusion(result))
        cells.append(_markdown("artifacts", f"[Sealed snapshot]({os.path.relpath(root / 'report_snapshot.json', notebook.parent)}) | " +
            f"[Paired analysis and diagnostics]({os.path.relpath(destination / 'analysis.json', notebook.parent)}). " +
            "All PNGs are embedded in executed display outputs; the displayed report remains portable without its source paths."))
        if not smoke:
            cells.append(_retention_followup())
        book = nbformat.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
            "plan17_report": {"schema": 1, "study_id": snapshot["study_id"], "status": status, "smoke": smoke,
                              "snapshot_sha256": _hash(snapshot_bytes), "generation": generation}})
        nbformat.validate(book)
        notebook_bytes = nbformat.writes(book).encode()
        manifest = {"schema": 1, "notebook": str(notebook), "report_dir": str(destination), "manifest": str(report_dir / "latest.json"),
                    "snapshot_sha256": _hash(snapshot_bytes), "counts": result["counts"]}
        _atomic(staging / "analysis.json", _json_bytes(result))
        _atomic(staging / "report.ipynb", notebook_bytes)
        _atomic(staging / "manifest.json", _json_bytes(manifest))
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.is_dir():
                raise
            shutil.rmtree(staging)
        _atomic(notebook, notebook_bytes)
        _atomic(report_dir / "latest.json", _json_bytes(manifest))
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish Plan 17 from sealed scalar artifacts only")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
