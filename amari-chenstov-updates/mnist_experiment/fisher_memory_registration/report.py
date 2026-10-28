"""Concise Phase 10 publication from scalar artifacts only; never run models."""

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-registration-matplotlib")

from mnist_experiment.fisher_memory_regression.report import (
    _figure, _image, _style, COLORS, LABELS, MAX_NOTEBOOK_BYTES,
)
from .analysis import FD, REPLAY, EWC, flatten_record, record_context, validate_records

TAG, PREFIX = "plan15_phase10", "plan15-phase10-"
TASKS = ("vector", "scalar")


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def cell(name, text):
    return nbformat.v4.new_markdown_cell(text, id=PREFIX + name, metadata={TAG: True})


def picture(name, figure, caption):
    result = _image("phase10-" + name, figure, caption)
    result["id"], result["metadata"] = PREFIX + name, {TAG: True}
    return result


def interval(value):
    if value.get("ci") is None:
        return f"{value['mean']:.3g} (descriptive mean only)"
    return f"{value['mean']:.3g} [{value['ci'][0]:.3g}, {value['ci'][1]:.3g}]"


def _error(task):
    return "RMS landmark error (pixels)" if task == "vector" else "Coordinate-one RMSE (pixels)"


def _decisions(snapshot, name):
    return snapshot.get("decisions", {}).get(name, snapshot.get(name)) or {}


def _policies(snapshot, task):
    selection = _decisions(snapshot, "selection").get(task, {})
    return selection.get("policies", selection)


def _analysis(snapshot):
    analyzed = {k: v for k, v in (snapshot.get("analysis") or {}).items() if k in TASKS and v}
    provisional = _provisional(snapshot, analyzed)
    return {task: analyzed.get(task, provisional.get(task)) for task in TASKS
            if task in analyzed or task in provisional}


def _provisional(snapshot, analyzed):
    """Display means for complete three-arm pairs, without estimating inference.

    Seals are verified by the snapshot caller. Incomplete arm groups wait for
    their controls; numerically invalid complete pairs remain visible, flagged.
    Final analysis supersedes this display independently for each task.
    """
    by_task, results = defaultdict(list), {}
    for raw in snapshot.get("confirmation") or []:
        row = flatten_record(raw)
        task = record_context(row)
        if task not in analyzed:
            by_task[task].append(row)
    for task, raw in by_task.items():
        N = (snapshot.get("config") or {}).get("novel_count", raw[0]["target"])
        rows = validate_records(raw, N, "confirmation", context_key=task)
        arms = defaultdict(dict)
        policies = defaultdict(set)
        for row in rows:
            method, replica = row["method"], row["replica"]
            if method in arms[replica]:
                raise ValueError("duplicate confirmation method/replica")
            arms[replica][method] = row
            policies[method].add(row["recipe_key"])
        if any(len(keys) != 1 for keys in policies.values()):
            raise ValueError("confirmation display requires one frozen policy per method")
        replicas = sorted(r for r, group in arms.items() if set(group) == set(LABELS))
        if not replicas:
            continue
        result = dict(provisional=True, is_final=False, context_key=task, output_dim=rows[0]["output_dim"],
            N=N, replicas=replicas, incomplete_replicas=sorted(set(arms) - set(replicas)),
            invalid_replicas=[r for r in replicas if any(not a["valid"] for a in arms[r].values())], methods={})
        for method in LABELS:
            group = [arms[r][method] for r in replicas]
            curves = dict(n=list(range(N + 1)), n_pairs=len(replicas))
            for domain in ("base", "novel"):
                risks = np.asarray([[m[f"{domain}_mse"] for m in r["metrics"]] for r in group])
                curves[domain] = dict(mean_mse=risks.mean(axis=0).tolist(),
                                      mean_rmse=np.sqrt(risks.mean(axis=0)).tolist())
                for metric in ("common_mse", "mae", "accuracy1"):
                    name = f"{domain}_{metric}"
                    present = [name in m for r in group for m in r["metrics"]]
                    if not any(present):
                        continue
                    if not all(present):
                        raise ValueError(f"incomplete optional display curve: {name}")
                    values = np.asarray([[m[name] for m in r["metrics"]] for r in group], dtype=float)
                    if (not np.isfinite(values).all() or (values < 0).any()
                            or (metric == "accuracy1" and (values > 1).any())):
                        raise ValueError(f"invalid optional display curve: {name}")
                    curves[domain][f"mean_{metric}"] = values.mean(axis=0).tolist()
            for name in ("observations", "learning_seconds"):
                curves[name] = np.asarray([[m[name] for m in r["metrics"]] for r in group]).mean(axis=0).tolist()
            result["methods"][method] = dict(curves=curves,
                acquisition=dict(mean=float(np.mean([r["novel_auc"] for r in group]))),
                retention=dict(mean=float(np.mean([np.sqrt(r["base_auc"]) - np.sqrt(r["frozen_base_mse"])
                                                   for r in group]))),
                costs=dict(mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in group]))))
        results[task] = result
    return results


def _progress(snapshot):
    p = snapshot.get("progress") or {}
    status = p.get("status", "pending")
    if status in ("running", "live") and not any(p.get(k) is True for k in
            ("process_verified", "lease_verified", "process_alive", "status_verified")):
        status = "running status unverified (process/lease check required)"
    counts = p.get("pairs", p.get("tasks", {}))
    if isinstance(counts, dict) and counts:
        count = "; ".join(f"{t}: {v.get('completed', 0)}/{v.get('planned', '?')} complete/planned pairs"
                          for t, v in counts.items() if isinstance(v, dict))
    else:
        count = f"{p.get('completed', 0)}/{p.get('planned', '?')} scheduled trajectories"
    if status == "completed":
        count = f"{p.get('completed', 0)} completed trajectories"
    forecast = p.get("forecast_seconds")
    forecast = "0 h" if status == "completed" else "pending" if forecast is None else f"{forecast / 3600:.2f} h"
    remaining = max(0., p.get("allocation_seconds", 72000.)-p.get("active_seconds", 0.)) / 3600
    storage = p.get("storage", {})
    used = storage.get("used_bytes", 0) + storage.get("external_bytes", 0)
    checkpoint = p.get("last_accepted_checkpoint", p.get("last_checkpoint", "pending"))
    return cell("progress", f"**Phase 10:** {status} / {p.get('stage', '10A')} | {count} | "
        f"{p.get('active_seconds', 0) / 3600:.2f}/20 active h | workload extrapolation: {forecast} | "
        f"remaining allocation: {remaining:.2f} h | "
        f"storage {used / 1e9:.2f}/500 GB ({(500_000_000_000-used)/1e9:.2f} GB headroom). "
        f"Last accepted checkpoint: {checkpoint}. Partial results are descriptive; no significance stopping. "
        "The extrapolation applies a conservative observed duration across pending work; early expensive settings can inflate it. "
        "The 20-hour cap and confirmation reserve still govern scheduling.")


def _overview(snapshot):
    init, readiness = _decisions(snapshot, "initialization"), _decisions(snapshot, "readiness")
    preview = snapshot.get("preview") or readiness.get("preview")
    checks = []
    for task in TASKS:
        item = init.get(task, {})
        for row in item.get("checks", []):
            if "mse" in row and "zero_predictor_mse" in row:
                checks.append((task, row))
    schedule = snapshot.get("schedule")
    if not (preview or checks or schedule):
        return None
    figure = _figure(12, 3.4)
    axes = figure.subplots(1, 3)
    if isinstance(preview, dict) and "reference" in preview and "query" in preview:
        ref, query = (np.asarray(preview[k]).squeeze() for k in ("reference", "query"))
        if ref.ndim != 2 or query.shape != ref.shape:
            raise ValueError("preview reference/query must be same-shaped 2D images")
        axes[0].imshow(np.concatenate((ref, query), axis=1), cmap="gray")
        for key, color in (("true_landmarks", "#168477"), ("predicted_landmarks", "#ad503d")):
            if key in preview:
                points = np.asarray(preview[key], dtype=float)
                if points.shape != (3, 2):
                    raise ValueError("preview landmarks must be 3x2 centered pixel coordinates")
                axes[0].scatter(points[:, 0] + (query.shape[1]-1)/2 + ref.shape[1],
                                points[:, 1] + (query.shape[0]-1)/2, c=color, s=18, label=key.replace("_", " "))
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=6)
        axes[0].set(title="Reference / query; forward map")
    else:
        axes[0].text(.5, .5, "Image-pair preview pending", ha="center", va="center")
    axes[0].set_axis_off()
    for task in TASKS:
        rows = [r for t, r in checks if t == task]
        if rows:
            axes[1].plot([r.get("replica", i) for i, r in enumerate(rows)],
                         [1-r["mse"]/r["zero_predictor_mse"] if r["zero_predictor_mse"] else 0 for r in rows],
                         "o", label=task)
    axes[1].axhline(.5, ls="--", c="#777", label="Capability preference")
    axes[1].set(title="Nominal pretraining", xlabel="Development replica", ylabel="Risk reduction vs zero displacement")
    axes[1].legend(fontsize=7)
    if schedule:
        axes[2].plot(schedule["group"], schedule["p"], label="Novel probability p")
        axes[2].plot(schedule["group"], schedule["u"], label="Severity fraction u")
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(.5, .5, "Schedule artifact pending", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set(title="Continuous sampling law", xlabel="Original four-pair group", ylabel="Probability / severity fraction")
    for axis in axes[1:]:
        _style(axis)
    return picture("overview", figure, "One source produces two views, not two observations. "
        "Six outputs encode x/y displacement of three fixed virtual landmarks; overlays are forward maps, "
        "not sampling-grid inverses. Scalar y1 uses exactly the same fully varying input-pair law. "
        "Pretraining and readiness evidence is development-only; model previews are illustrative artifacts.")


def _calibration_note(snapshot):
    initialization = _decisions(snapshot, "initialization")
    sentences, sizes, reductions, recovery = [], [], [], []
    readiness = _decisions(snapshot, "readiness")
    for task, dim in (("vector", 6), ("scalar", 1)):
        item = initialization.get(task, {})
        channels = item.get("channels")
        if channels:
            c1, c2 = channels
            # Count the accepted rank-two architecture without constructing a model.
            ltm = 19*c1 + (9*c1+1)*c2 + (4*c2+1)*dim
            stm = 36 + 20*c1 + 2*c2 + min(2, dim)*(4*c2+dim)
            sizes.append(f"{task} {ltm+stm:,} ({ltm:,} LTM + {stm:,} STM)")
        ratio = item.get("ratios", {}).get(task)
        if ratio is not None:
            reductions.append(f"{task} {100*(1-ratio):.1f}%")
        checks = readiness.get(task, [])
        if isinstance(checks, dict):
            checks = checks.get(str(readiness.get("blur_max")), [])
        if checks and all(type(r.get("recovered")) is bool for r in checks):
            recovery.append(f"{task} {sum(r['recovered'] for r in checks)}/{len(checks)}")
    if sizes:
        sentences.append("All adaptive methods within each task train the same parameter count: "
                         + "; ".join(sizes) + ".")
    if reductions:
        sentences.append("Nominal pretraining reduced validation risk versus zero displacement by "
                         + ", ".join(reductions) + ".")
    if recovery:
        sentences.append("The extra-label diagnostic met its learnability preference on "
                         + ", ".join(recovery) + " development replicas; this is not a guarantee of online gains.")
    geometry = snapshot.get("geometry") or {}
    rank = geometry.get("rank")
    errors = [row["relative_frobenius"] for row in geometry.get("rows", [])
              if row.get("rank") == rank and "relative_frobenius" in row]
    if errors:
        sentences.append(f"Rank-{rank} compression differed from the dense pilot estimates by at most "
                         f"{100*max(errors):.2f}% in relative Frobenius norm, not error to a known population Fisher.")
    return cell("calibration", "**Calibration:** " + " ".join(sentences)) if sentences else None


def _frontier(snapshot):
    development = snapshot.get("development") or {}
    tasks = [t for t in TASKS if development.get(t)]
    if not tasks:
        return None
    figure = _figure(11, 3.4 * len(tasks))
    axes = figure.subplots(len(tasks), 2, squeeze=False)
    cfg = snapshot.get("config") or {}
    delta, factor = cfg.get("retention_margin", .5), cfg.get("envelope_factor", 2.5)
    for index, task in enumerate(tasks):
        rows, policies = development[task], _policies(snapshot, task)
        selected_replay = policies.get(REPLAY, {}).get("selected")
        replay = next((r for r in rows if selected_replay and r["recipe_key"] == selected_replay["recipe_key"]), None)
        if selected_replay and replay is None:
            raise ValueError("selected replay recipe is missing from the grid cohort")
        axis, heat = axes[index]
        for method in LABELS:
            for row in (r for r in rows if r["method"] == method):
                excluded = not row.get("resource_feasible", False) or row["base_harm"] > delta
                if method == FD and replay:
                    excluded |= (row["seconds_per_arrival"] > factor * replay["seconds_per_arrival"]
                                 or row["base_harm"]-replay["base_harm"] > cfg.get("matching_tolerance", .1))
                axis.scatter(row["base_harm"], row["novel_auc"], c=COLORS[method], marker="x" if excluded else "o", s=23)
            axis.plot([], [], "o", color=COLORS[method], label=LABELS[method])
            selected = policies.get(method, {}).get("selected")
            if selected:
                grid_row = next((r for r in rows if r["recipe_key"] == selected["recipe_key"]), None)
                if grid_row is None:
                    raise ValueError("selected recipe is missing from the grid cohort")
                axis.scatter(grid_row["base_harm"], grid_row["novel_auc"], marker="*", s=90,
                             facecolor=COLORS[method], edgecolor="#111", linewidth=.6)
        axis.axvline(delta, ls="--", color="#555")
        axis.set(title=f"{task}: acquisition / retention", xlabel="Nominal deterioration H (pixels)",
                 ylabel="Novel budget-mean risk (square pixels)")
        axis.legend(fontsize=6)
        _style(axis)
        fd = [r for r in rows if r["method"] == FD]
        if fd:
            ridges = sorted({r["recipe"]["ridge"] for r in fd})
            strengths = sorted({r["recipe"]["strength"] for r in fd})
            values = np.full((len(ridges), len(strengths)), np.nan)
            for row in fd:
                i, j = ridges.index(row["recipe"]["ridge"]), strengths.index(row["recipe"]["strength"])
                values[i, j] = row["novel_auc"]
                ratio = row["seconds_per_arrival"] / replay["seconds_per_arrival"] if replay and replay["seconds_per_arrival"] > 0 else None
                label = f"{ratio:.1f}x" if ratio is not None else "pending"
                heat.text(j, i, label, ha="center", va="center", fontsize=7,
                          bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.5))
            image = heat.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", cmap="cividis")
            heat.set(xticks=range(len(strengths)), xticklabels=[f"{s:.2g}" for s in strengths],
                     yticks=range(len(ridges)), yticklabels=[f"{r:.4g}" for r in ridges],
                     xlabel="FD strength", ylabel="Ridge c", title=f"{task}: FD risk / selected-replay cost")
            heat.tick_params(axis="x", labelrotation=45, labelsize=6)
            figure.colorbar(image, ax=heat, label="Risk (square pixels)", shrink=.8)
    return picture("development", figure, "Exploratory initial-grid cohort means, including exclusions (crosses). "
        "Stars locate recipes selected using the expanded development cohort, not their expanded-cohort scores. "
        "FD heatmap labels show measured per-arrival cost relative to the selected replay recipe in the same grid cohort, "
        "not a fixed reference. Final eligibility is in the expanded-cohort policy table. "
        "Cost labels remain pending until replay policy selection; blank cells are untested gaps. "
        "Complete grids, counts, noisy optima and failed checks are in the supplement.")


def _curves(axis, result, domain, task, metric="rmse"):
    for method, item in result["methods"].items():
        curves, key = item["curves"], "mean_" + metric
        if key not in curves[domain]:
            continue
        values = np.asarray(curves[domain][key]) * (100 if metric == "accuracy1" else 1)
        axis.plot(curves["n"], values, color=COLORS[method], label=LABELS[method])
        if metric == "rmse" and "ci_rmse" in curves[domain]:
            lo, hi = curves[domain]["ci_rmse"]
            axis.fill_between(curves["n"], lo, hi, color=COLORS[method], alpha=.12)
    reference = result["methods"][REPLAY]["curves"][domain].get("mean_" + metric)
    if reference is not None:
        axis.axhline(reference[0] * (100 if metric == "accuracy1" else 1), ls="--", c="#555", label="Frozen initialization")
    ylabel = _error(task) if metric == "rmse" else ("Within one pixel (%)" if metric == "accuracy1" else "Mean landmark distance (pixels)")
    axis.set(xlabel="Unique novel training pairs", ylabel=ylabel)
    axis.legend(fontsize=6)
    _style(axis)


def _confirmation(result):
    figure = _figure(11, 3.6)
    count = len(result["replicas"])
    pairs = "pair" if count == 1 else "pairs"
    figure.suptitle(("Provisional descriptive means" if result.get("provisional") else "Analyzed confirmation")
                    + f": {count} complete three-arm replica {pairs}", fontsize=10)
    for axis, domain in zip(figure.subplots(1, 2), ("novel", "base")):
        _curves(axis, result, domain, "vector")
        axis.set_title("Vector: " + ("novel" if domain == "novel" else "nominal") + " held-out pairs")
    uncertainty = ("Provisional descriptive means only: no bootstrap intervals, shading or significance claims. "
                   if result.get("provisional") else
                   "Bands are pointwise nominal 95% whole-paired-replica bootstrap intervals. ")
    return picture("vector", figure, f"{count} complete three-arm vector replica {pairs}. "
        "RMS landmark error is sqrt(mean squared landmark distance), not coordinate RMSE. "
        + uncertainty +
        "Fixed held-out source pairs are disjoint from training and never supply Fisher estimates or gradients.")


def _efficiency(results):
    results = {task: result for task, result in results.items() if not result.get("provisional")}
    if not results:
        return None
    figure = _figure(12, 3.3 * len(results))
    axes = figure.subplots(len(results), 3, squeeze=False)
    for row, (task, result) in zip(axes, results.items()):
        point = result["efficiency"]["point"]
        for method in (FD, REPLAY):
            curves = result["methods"][method]["curves"]
            row[0].plot(curves["n"], np.sqrt(curves["novel"]["mean_mse"]), c=COLORS[method], label=LABELS[method])
            row[1].plot(curves["n"], curves["learning_seconds"], c=COLORS[method], label=LABELS[method])
        row[0].axhline(np.sqrt(point["target_mse"]), ls="--", c="#555", label="Replay endpoint target")
        if point["status"] == "valid":
            row[0].axvline(point["matching_budget"], ls=":", c=COLORS[FD])
            row[2].bar(["FD match", "Replay endpoint"], [point["fd_old_rmse"], point["replay_old_rmse"]],
                       color=[COLORS[FD], COLORS[REPLAY]])
        else:
            row[2].text(.5, .5, "Matched retention unresolved", ha="center", va="center", transform=row[2].transAxes)
        if not point["useful_target"]:
            state = "weak target"
        elif point["status"] == "valid":
            state = "point match; CI unresolved" if result["efficiency"]["bootstrap"]["ci"] is None else "observed match"
        elif point["status"] == "target_not_reached_at_observed_endpoint":
            state = "endpoint target not reached"
        else:
            state = "no evaluable match"
        row[0].set(title=f"{task}: {state}", xlabel="Novel pairs", ylabel="Root mean risk (pixels)")
        row[1].set(title="Measured learning compute", xlabel="Novel pairs", ylabel="Mean learning seconds")
        row[2].set(title="Matched-budget nominal retention", ylabel="Mean replica error (pixels)")
        row[0].legend(fontsize=6)
        for axis in row:
            _style(axis)
    return picture("efficiency", figure, "Quality curves show the square root of mean replica risk, "
        "not the mean replica RMSE used in the predictive panels. The last downward crossing persists through "
        "the observed endpoint; interpolation uses unrooted risk. No fit, smoothing or extrapolation. "
        "Matched time includes both views, all intervening nominal arrivals, replay, FD and retirement. "
        "Scalar-coordinate and full-landmark errors are distinct tasks on separate axes.")


def _applied(results):
    figure = _figure(11, 7)
    axes = figure.subplots(2, 2)
    if "vector" in results:
        for axis, metric in zip(axes[0], ("mae", "accuracy1")):
            _curves(axis, results["vector"], "novel", "vector", metric)
    else:
        for axis in axes[0]:
            axis.text(.5, .5, "Vector applied metrics pending", ha="center", va="center", transform=axis.transAxes)
    if "scalar" in results:
        _curves(axes[1, 0], results["scalar"], "novel", "scalar")
    else:
        axes[1, 0].text(.5, .5, "Scalar companion pending", ha="center", va="center", transform=axes[1, 0].transAxes)
    for task, result in results.items():
        for method in (FD, REPLAY):
            curves = result["methods"][method]["curves"]
            common = curves["novel"].get("mean_common_mse")
            if common is None and task == "scalar":
                common = curves["novel"]["mean_mse"]
            if common is not None:
                axes[1, 1].plot(curves["n"], np.sqrt(common), color=COLORS[method],
                                ls="-" if task == "vector" else "--", label=f"{task}: {LABELS[method]}")
    axes[1, 1].set(xlabel="Novel pairs", ylabel="Common y1 RMSE (pixels)", title="Same-coordinate diagnostic only")
    if axes[1, 1].get_legend_handles_labels()[0]:
        axes[1, 1].legend(fontsize=6)
    for axis in axes.ravel():
        _style(axis)
    counts = "; ".join(f"{task}: {len(result['replicas'])} complete three-arm replica "
                       + ("pair" if len(result["replicas"]) == 1 else "pairs")
                       + (" (provisional descriptive means only)" if result.get("provisional") else " (analyzed)")
                       for task, result in results.items())
    return picture("applied", figure, counts + ". One-pixel success is a landmark-distance threshold, not digit accuracy. "
        "The scalar companion predicts only y1 from the same six-parameter affine distribution. "
        "Only the common-y1 panel compares the same prediction coordinate; the vector head, pretraining and "
        "extra supervision also differ. Output dimension is not a causal explanation.")


def _comparison_table(results):
    columns = {}
    for task, result in results.items():
        if result.get("provisional"):
            continue
        efficiency = result["efficiency"]
        point, boot = efficiency["point"], efficiency["bootstrap"]
        valid = point["status"] == "valid"

        def estimate(key, spec=".3g", scale=1., suffix=""):
            if not valid:
                return "Unresolved"
            value = f"{point[key] * scale:{spec}}{suffix}"
            if boot["ci"] is not None:
                lo, hi = boot["ci"][key]
                value += f" [{lo * scale:{spec}}{suffix}, {hi * scale:{spec}}{suffix}]"
            return value

        fd_time, replay_time = (result["methods"][m]["costs"]["mean_learning_seconds"] for m in (FD, REPLAY))
        matched = ("Supported" if efficiency["matched_budget_retention_supported"] else
                   "Unresolved" if not valid or boot["ci"] is None else "Not supported")
        status = ("Descriptive; CI withheld" if boot["ci"] is None else "Point estimate with 95% CI") if valid else (
            "Unresolved: " + point["status"].replace("_", " "))
        gain = point.get("target_mse_gain_fraction")
        columns[task] = [
            interval(result["G"]),
            f"{point['replay_target_rmse']:.3g}",
            f"{estimate('matching_budget', '.2f')} / {result['N']}",
            estimate("saved_fraction", ".2f", 100., "%"),
            estimate("learning_time_ratio", suffix="x"),
            "**" + estimate("S") + "**" if valid else "Unresolved",
            f"{fd_time/replay_time:.3g}x" if replay_time > 0 else "Unavailable",
            "Supported" if efficiency["full_budget_retention_supported"] else "Not supported",
            matched,
            status,
            f"{boot['invalid']}/{boot['draws']} unresolved draws",
            f"{boot['multiple_crossing_draws']}/{boot['draws']}",
            f"{100*gain:.2f}%" if gain is not None else "Unavailable",
        ]
    if not columns:
        return ""
    labels = [
        "Paired acquisition gain, replay minus FD $A$ (square pixels)",
        "Replay target RMS error (pixels)",
        "Novel pairs: FD match / replay endpoint",
        "Samples saved",
        "Matched-quality compute (FD / replay)",
        "**$S$**",
        "Equal-budget compute (FD / replay)",
        "Full-budget retention",
        "Matched-budget retention",
        "Inverse estimate status",
        "Unresolved bootstrap inversions",
        "Multiple-crossing bootstrap draws",
        "Replay target risk reduction from initialization",
    ]
    text = "| Statistic | " + " | ".join(task.title() for task in columns) + " |\n"
    text += "| --- | " + " | ".join("---" for _ in columns) + " |\n"
    for i, label in enumerate(labels):
        text += f"| {label} | " + " | ".join(values[i] for values in columns.values()) + " |\n"
    return text + ("\n$S$ = fraction of novel pairs saved / matched-quality learning-compute multiplier. "
        "Bracketed intervals are nominal paired 95% CIs. Inverse-derived values without intervals are "
        "descriptive point estimates; **unresolved is not zero**. Matched-quality compute is distinct from equal-budget compute.")


def build_cells(snapshot, diagnostics_href=None):
    results = _analysis(snapshot)
    cfg = snapshot.get("config") or {}
    out = [_progress(snapshot), cell("intro", "## Phase 10: Multivariate MNIST Registration\n\n"
        "Phase 9 found no confirmed FD acquisition advantage on its selected scalar angle task. "
        "This phase tests whether predicting several geometric quantities creates a more useful setting for FD, "
        "while retaining MNIST and continuously changing the sampling distribution.\n\n"
        "Can FD KL earn its compute by learning useful registration from fewer new pairs without forgetting old behavior? "
        "The primary vector task predicts three landmark displacements (six coordinates); the secondary scalar task predicts y1. "
        "Vector risk is squared-coordinate error summed and divided by **three landmarks**, in square pixels. "
        "Its square root is RMS landmark error in pixels. Scalar risk is y1 MSE, not full registration.\n\n"
        f"Retention ceiling: {cfg.get('retention_margin', .5):g} pixels; tolerance: {cfg.get('matching_tolerance', .1):g} pixels. "
        "Replay is selected first for acquisition under its ceiling; FD must meet the same ceiling, one-sided retention "
        f"tolerance and {cfg.get('envelope_factor', 2.5):g} times **selected replay** per-arrival cost. "
        "EWC-only is independently selected and need not be retention-matched. "
        "S = fraction of novel pairs saved / matched-quality learning-compute multiplier. "
        "Development uses held-out summaries for exploratory selection; fresh confirmation is separate.\n\n"
        f"Each model receives {cfg.get('init_size', 30000):,} nominal pretraining pairs before online adaptation. "
        f"Each online trajectory then reaches {cfg.get('novel_count', 100)} unique novel pairs, with every intervening "
        "nominal pair retained. A pair contains two views of one source and supplies six scalar targets for the vector "
        "task or one for the scalar task. Policies within a task share those observations; the savings claim concerns "
        "online adaptation conditional on pretraining and calibration.")]
    for item in (_overview(snapshot), _calibration_note(snapshot), _frontier(snapshot)):
        if item:
            out.append(item)
    selections = [(task, _policies(snapshot, task)) for task in TASKS if _policies(snapshot, task)]
    if selections:
        text = "### Selected Policies\n\n| Task | Method | Ridge | FD strength | Eligibility |\n| --- | --- | ---: | ---: | --- |\n"
        for task, policies in selections:
            for method in LABELS:
                choice = policies.get(method)
                if not choice:
                    continue
                r = choice.get("selected", choice)
                recipe = r.get("recipe", r)
                text += (f"| {task} | {LABELS[method]} | {recipe['ridge']:.3g} | {recipe['strength']:.3g} | "
                         f"{choice.get('reason', 'see sealed decision')} |\n")
        out.append(cell("selection", text))
    if "vector" in results:
        out.append(_confirmation(results["vector"]))
    if results:
        efficiency_figure = _efficiency(results)
        if efficiency_figure:
            out.append(efficiency_figure)
        out.append(_applied(results))
        text = "### Outcomes\n\n| Task / complete pairs | Method | Novel A (square pixels) | Nominal H (pixels) | Learning seconds |\n"
        text += "| --- | --- | ---: | ---: | ---: |\n"
        for task, result in results.items():
            label = f"{task} / {len(result['replicas'])}" + (" provisional" if result.get("provisional") else " analyzed")
            for method, item in result["methods"].items():
                text += (f"| {label} | {LABELS[method]} | {interval(item['acquisition'])} | "
                         f"{interval(item['retention'])} | {item['costs']['mean_learning_seconds']:.3g} |\n")
        text += ("\nA is the mean held-out risk over exact novel budgets 1 through N. H takes square roots within each replica. "
                 "Bracketed intervals are nominal 95% CIs for analyzed tasks only; provisional means have no intervals.")
        out.append(cell("outcomes", text))
        paragraphs = []
        for task, result in results.items():
            if result.get("provisional"):
                count = len(result["replicas"])
                text = (f"**{task.title()} provisional confirmation:** {count} complete three-arm replica "
                        f"{'pair' if count == 1 else 'pairs'}. Curves and outcomes are descriptive means of these "
                        "paired trajectories only. No bootstrap, confidence intervals, significance tests or "
                        "sample-equivalence estimates are calculated; acquisition and retention conclusions await analysis. ")
                if result["incomplete_replicas"]:
                    text += f"{len(result['incomplete_replicas'])} additional replica groups await all three arms. "
                if result["invalid_replicas"]:
                    text += (f"**Numerical warning:** {len(result['invalid_replicas'])} complete pairs include "
                             "invalid trajectories; retained as diagnostics, not silently excluded. ")
                paragraphs.append(text)
                continue
            efficiency = result["efficiency"]
            point, boot = efficiency["point"], efficiency["bootstrap"]
            text = f"**{task.title()}:** "
            text += ("Positive budget-averaged acquisition evidence at the same unique-pair budgets. "
                     if result["G"]["ci"][0] > 0 else "An acquisition advantage is not established. ")
            matched_unresolved = point["status"] != "valid" or boot["ci"] is None
            text += ("**Old-data retention:** supported at full and matched budgets. " if efficiency["retention_qualified"] else
                     "**Old-data retention:** full-budget supported; matched-budget unresolved; score is retention-unqualified. "
                     if efficiency["full_budget_retention_supported"] and matched_unresolved else
                     "**Old-data retention:** not supported at both budgets; score is retention-unqualified. ")
            if efficiency["full_budget_retention_supported"] and point["status"] == "valid" and boot["ci"] is None:
                text += "This reflects withheld inverse uncertainty, not a failed full-budget retention test. "
            if point["status"] == "valid":
                if boot["ci"] is None or not efficiency["retention_qualified"]:
                    text += "Samples saved remain a descriptive point estimate. "
            if not point["useful_target"]:
                gain = point.get("target_mse_gain_fraction")
                text += (f"**Weak target:** replay improves frozen risk by {100*gain:.2f}%, below 5%. " if gain is not None else
                         "**Weak target:** practical improvement over frozen risk is unavailable. ")
                text += "Numerical savings are descriptive, not an established practical-efficiency win. "
            if not result.get("is_final"):
                text += "This is smoke/development evidence, not final confirmation. "
            paragraphs.append(text)
        if any(not r.get("provisional") and r["efficiency"]["bootstrap"]["ci"] is None for r in results.values()):
            paragraphs.append("**Inverse uncertainty:** the prespecified rule withholds ordinary inversion intervals "
                "when any bootstrap inversion is unresolved; no failed inversion was discarded.")
        contrast = snapshot.get("common_coordinate_contrast")
        if contrast:
            confidence = " (nominal paired 95% CI)" if contrast.get("ci") is not None else ""
            bounds = contrast.get("ci")
            direction = ("The common-coordinate gain is larger in the vector context. " if bounds and bounds[0] > 0 else
                         "The common-coordinate gain is larger in the scalar context. " if bounds and bounds[1] < 0 else
                         "A difference in common-coordinate gains is not established. ")
            paragraphs.append("**Secondary common-y1 contrast:** vector minus scalar replay-minus-FD acquisition "
                f"gain = {interval(contrast)} square pixels{confidence}. This compares the same "
                "coordinate, not full registration. " + direction +
                "Head size, pretraining and extra supervision also differ, so output dimension is not isolated.")
        paragraphs.append("**Dimensionality:** these results can identify a useful context, not prove dimension caused "
            "an advantage or demonstrate memory-block information transfer. Common-y1 comparisons are secondary. "
            "Capability, learnability and compression checks describe this context, not guaranteed algorithmic advantage. "
            "Online savings exclude separately "
            "accounted initialization, diagnostic labels, development, evaluation and archival costs. "
            "Different matched budgets include different background exposure and drift histories. Nominal 95% intervals "
            "are not familywise claims; S=0.25 is not a success threshold.")
        comparison = _comparison_table(results)
        out.append(cell("conclusion", "### Interpretation\n\n" + (comparison + "\n\n" if comparison else "")
                        + "\n\n".join(paragraphs)))
    else:
        out.append(cell("pending", "### Fresh Confirmation\n\nWaiting for complete fresh three-arm replica pairs or final analysis. "
            "No model evaluation or artifact repair is performed by this notebook."))
    if diagnostics_href:
        out.append(cell("supplement", f"[Supplement: readiness, full frontiers, exclusions, per-digit/coordinate errors, "
            f"Jacobian spectra/effective ranks, manifests and all failed inversions]({diagnostics_href})."))
    return out


def _geometry_supplement(store):
    """Read small sealed diagnostics only; never load checkpoints or fit spectra."""
    complete, unit_path = getattr(store, "is_complete", None), getattr(store, "unit_path", None)
    if not callable(complete) or not callable(unit_path):
        return {}
    geometry = {}
    for task in TASKS:
        unit = "decision-geometry-" + task
        item = dict(unit_id=unit, status="pending")
        if complete(unit):
            path = Path(unit_path(unit)) / "result.json"
            try:
                if path.stat().st_size > 1_000_000:
                    raise ValueError("geometry JSON exceeds small-artifact size bound")
                result = json.loads(path.read_bytes())
                if not isinstance(result, dict):
                    raise ValueError("geometry result must be an object")
                json.dumps(result, allow_nan=False)
                item.update(status="completed", result=result)
            except (OSError, ValueError) as error:
                item.update(status="unavailable", warning=str(error))
        geometry[task] = item
    return geometry


def publish(snapshot, notebook, store=None):
    """Replace only Phase10-owned cells; append its narrative after all history.

    Snapshot: config, progress, development={task: table_rows}, decisions with
    initialization/readiness/selection/manifest, analysis={task: result}, and
    confirmation=[sealed full_records] for provisional whole-pair means. Optional
    preview and schedule contain precomputed display artifacts, never models.
    store needs root and reserve(nbytes); the supplement is content-addressed.
    Optional is_complete/unit_path methods supply sealed per-task geometry JSON.
    """
    notebook = Path(notebook)
    if notebook.exists() and notebook.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    previous = json.loads(notebook.read_bytes()) if notebook.exists() else dict(nbformat.v4.new_notebook())
    if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
        raise ValueError("cannot publish into a frozen historical notebook")
    href = None
    if store is not None:
        supplement = {**snapshot, "sealed_geometry": _geometry_supplement(store)}
        text = ("<!doctype html><meta charset='utf-8'><title>Phase 10 supplement</title>"
                "<style>body{font:15px sans-serif;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap}</style>"
                "<h1>Phase 10 artifact supplement</h1><p>Exploratory development; fixed held-out panels. "
                "Per-digit/coordinate errors, Jacobian spectra and compression checks, manifests, cost breakdowns "
                "and failed inversions are retained as recorded; missing diagnostics are pending, not passing. "
                "Sealed task-specific geometry contains STM likelihood-Fisher eigenvalues, trace and effective rank; "
                "it supplements the snapshot's archive-rank approximation diagnostics. Eigenvalues are not relabeled "
                "as Jacobian singular values.</p><pre>"
                + html.escape(json.dumps(supplement, indent=2, allow_nan=False)) + "</pre>")
        payload = text.encode()
        path = Path(store.root) / "reports" / ("phase10-supplement-" + hashlib.sha256(payload).hexdigest() + ".html")
        if not path.exists():
            with store.reserve(len(payload)):
                _atomic(path, payload)
        href = os.path.relpath(path, notebook.parent)
    cells = build_cells(snapshot, href)
    history = [c for c in previous["cells"] if not owned(c)]
    book = deepcopy(previous)
    book["cells"] = history[:1] + [cells[0]] + history[1:] + cells[1:]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    payload = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    with store.reserve(len(payload)) if store is not None else nullcontext():
        _atomic(notebook, payload)


def _atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-registration-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
