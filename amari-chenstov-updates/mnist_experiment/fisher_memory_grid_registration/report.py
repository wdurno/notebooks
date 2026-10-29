"""Concise Phase 11 publication from runner-supplied scalar artifacts only."""

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import nbformat
import numpy as np

from mnist_experiment.fisher_memory_registration.report import _atomic, _decisions, interval
from mnist_experiment.fisher_memory_regression.analysis import flatten_record
from mnist_experiment.fisher_memory_regression.report import (
    _figure, _image, _style, _display, COLORS, LABELS, MAX_NOTEBOOK_BYTES,
)
from . import PROTOCOL
from .analysis import validate_records

TAG, PREFIX = "plan15_phase11", "plan15-phase11-"
TASKS = ("grid",)
FD, REPLAY, EWC = LABELS


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def cell(name, text):
    return nbformat.v4.new_markdown_cell(text, id=PREFIX + name, metadata={TAG: True})


def picture(name, figure, caption):
    result = _image("phase11-" + name, figure, caption)
    result["id"], result["metadata"] = PREFIX + name, {TAG: True}
    return result


def _policies(snapshot):
    selection = _decisions(snapshot, "selection").get("grid", {})
    return selection.get("policies", selection)


def _analysis(snapshot):
    analyses = snapshot.get("analysis") or {}
    if set(analyses) - set(TASKS):
        raise ValueError("Phase 11 analysis must be keyed by grid")
    result = analyses.get("grid")
    if not result:
        return _provisional(snapshot)
    if (result.get("schema_version") != 1 or result.get("protocol") != PROTOCOL
            or result.get("context_key") != "grid" or result.get("output_dim") != 54
            or result.get("status") not in ("completed", "smoke_completed")
            or set(result.get("methods", {})) != set(LABELS)
            or len(result.get("replicas", [])) < 2):
        raise ValueError("completed compatible grid/54-output paired analysis required")
    return result


def _provisional(snapshot):
    """Whole three-arm cohort means only; the caller verifies completion seals."""
    raw = snapshot.get("confirmation") or []
    if not raw:
        return None
    N = (snapshot.get("config") or {}).get("novel_count")
    if N is None:
        N = flatten_record(raw[0]).get("target")
    rows = validate_records(raw, N, "confirmation", context_key="grid")
    arms, policies = defaultdict(dict), defaultdict(set)
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
        return None
    result = dict(provisional=True, is_final=False, context_key="grid", output_dim=54, N=N,
        replicas=replicas, incomplete_replicas=sorted(set(arms)-set(replicas)),
        invalid_replicas=[r for r in replicas if any(not a["valid"] for a in arms[r].values())], methods={})
    for method in LABELS:
        group = [arms[r][method] for r in replicas]
        curves = dict(n=list(range(N+1)), n_pairs=len(replicas))
        for domain in ("base", "novel"):
            risks = np.asarray([[m[f"{domain}_mse"] for m in r["metrics"]] for r in group])
            curves[domain] = dict(mean_mse=risks.mean(axis=0).tolist())
            for metric in ("mae", "accuracy1"):
                name = f"{domain}_{metric}"
                present = [name in m for r in group for m in r["metrics"]]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(f"incomplete optional display curve: {name}")
                values = np.asarray([[m[name] for m in r["metrics"]] for r in group], dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or (metric == "accuracy1" and (values > 1).any()):
                    raise ValueError(f"invalid optional display curve: {name}")
                curves[domain][f"mean_{metric}"] = values.mean(axis=0).tolist()
        for name in ("observations", "learning_seconds"):
            curves[name] = np.asarray([[m[name] for m in r["metrics"]] for r in group]).mean(axis=0).tolist()
        result["methods"][method] = dict(curves=curves,
            acquisition=dict(mean=float(np.mean([r["novel_auc"] for r in group]))),
            retention=dict(mean=float(np.mean([np.sqrt(r["base_auc"])-np.sqrt(r["frozen_base_mse"]) for r in group]))),
            costs=dict(mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in group]))))
    return result


def _progress(snapshot, result=None):
    p = snapshot.get("progress") or {}
    status = p.get("status", "pending")
    if status in ("running", "live") and not any(p.get(k) is True for k in
            ("process_verified", "lease_verified", "process_alive", "status_verified")):
        status = "running status unverified (process/lease check required)"
    counts = p.get("pairs", p.get("tasks", {}))
    grid = counts.get("grid", {}) if isinstance(counts, dict) else {}
    count = (f"{grid.get('completed', 0)}/{grid.get('planned', '?')} complete/planned three-arm replica pairs"
             if grid else f"{p.get('completed', 0)}/{p.get('planned', '?')} scheduled trajectories; paired count pending")
    if result:
        planned = grid.get("planned", (snapshot.get("config") or {}).get("confirmation_replicas", "?"))
        count = f"{len(result['replicas'])}/{planned} complete/planned three-arm replica pairs"
    active, allocation = p.get("active_seconds", 0), p.get("allocation_seconds", 72000)
    forecast = "0 h" if status == "completed" else (
        "pending" if p.get("forecast_seconds") is None else f"{p['forecast_seconds']/3600:.2f} h")
    storage = p.get("storage") or {}
    used = storage.get("used_bytes", 0) + storage.get("external_bytes", 0)
    cap = storage.get("limit_bytes", 500_000_000_000)
    return cell("progress", f"**Phase 11:** {status} / {p.get('stage', '11A')} | {count} | "
        f"{active/3600:.2f}/{allocation/3600:g} active h | remaining allocation {max(0, allocation-active)/3600:.2f} h | "
        f"remaining-work forecast {forecast} | storage {used/1e9:.2f}/{cap/1e9:g} GB; "
        f"{(cap-used)/1e9:.2f} GB headroom. Last accepted checkpoint: "
        f"{p.get('last_accepted_checkpoint', p.get('last_checkpoint', 'pending'))}. "
        f"Last recorded update: {p.get('heartbeat', 'not recorded')}. "
        "Partial evidence is descriptive; no significance stopping. Caps and confirmation reserve do not renew on resume.")


def _overview(snapshot):
    preview = snapshot.get("preview") or _decisions(snapshot, "readiness").get("preview")
    schedule = snapshot.get("schedule")
    if not preview and not schedule:
        return None
    figure = _figure(11, 3.5)
    image, law = figure.subplots(1, 2)
    if preview:
        ref, query = (np.asarray(preview[k]).squeeze() for k in ("reference", "query"))
        if ref.shape != (84, 84) or query.shape != ref.shape:
            raise ValueError("grid preview requires two 84x84 mosaics")
        image.imshow(np.concatenate((ref, query), axis=1), cmap="gray")
        for key, color in (("true_landmarks", "#168477"), ("predicted_landmarks", "#ad503d")):
            if key in preview:
                points = np.asarray(preview[key], dtype=float)
                if points.shape not in ((9, 3, 2), (27, 2)) or not np.isfinite(points).all():
                    raise ValueError("grid preview landmarks require 9x3x2 cell-local pixel coordinates")
                offsets = np.array([[(j % 3)*28 + 13.5 + 84, (j // 3)*28 + 13.5] for j in range(9)])
                points = points.reshape(9, 3, 2) + offsets[:, None, :]
                image.scatter(*points.reshape(27, 2).T, s=9, c=color, label=key.replace("_", " "))
        if image.get_legend_handles_labels()[0]:
            image.legend(fontsize=6)
    else:
        image.text(.5, .5, "Grid preview pending", ha="center")
    image.set(title="Reference / query: forward geometry")
    image.set_axis_off()
    if schedule:
        for key, label in (("p", "Novel probability p"), ("u", "Severity fraction u")):
            law.plot(schedule["group"], schedule[key], label=label)
        law.legend(fontsize=7)
    law.set(title="Continuous sampling law", xlabel="Original four-grid group", ylabel="Probability / severity")
    _style(law)
    return picture("overview", figure, "Each reference/query mosaic pair is one grid observation: nine digit pairs, "
        "18 image views, 27 landmarks and 54 coordinates. Overlays add cell offsets only for display; "
        "predictions are forward maps in centered cell-local pixels, not sampling-grid inverses. Previews are illustrative.")


def _settings(snapshot):
    cfg = snapshot.get("config") or {}
    init = _decisions(snapshot, "initialization").get("grid", {})
    replay = _policies(snapshot).get(REPLAY, {}).get("selected", {})
    recipe = replay.get("recipe", replay)
    capacity = recipe.get("capacity", cfg.get("capacities", [32])[0])
    count = recipe.get("replay_count", cfg.get("replay_counts", [2])[0])
    rows = [("Nominal pretraining", f"{cfg.get('init_size', 3000):,} grids / {9*cfg.get('init_size', 3000):,} digit pairs", "Shared across arms"),
            ("Replay capacity / per update", f"{capacity} / {count} grids = {9*capacity} / {9*count} digit pairs", "EWC-only: zero replay; actual bytes in supplement")]
    if init:
        channels = init.get("channels", [4, 6])
        counts = {(4, 6): "3,268 LTM + 344 STM = 3,612", (4, 8): "4,314 LTM + 384 STM = 4,698"}
        rows.append(("Shared capacity", counts.get(tuple(channels), "See initialization artifact"),
                     f"Channels {channels}; epochs {_display(init.get('epochs'))}"))
        ratio = init.get("ratios", {}).get("grid")
        if ratio is not None:
            rows.append(("Capability", f"{100*(1-ratio):.1f}% risk reduction vs zero displacement",
                         "Supported" if init.get("capability_supported") else "Limitation / not established"))
        if init.get("cell_ratios"):
            rows.append(("Worst-cell capability", f"{100*(1-max(init['cell_ratios'])):.1f}% risk reduction", "Preference: at least 30% in every cell"))
    readiness = _decisions(snapshot, "readiness")
    blur = readiness.get("blur_max")
    evidence = readiness.get("grid", {}).get(str(blur), [])
    if evidence:
        ratios = [100*(row["frozen_degradation_ratio"]-1) for row in evidence]
        recovered = sum(bool(row["recovered"]) for row in evidence)
        rows.append(("Drift readiness", f"Blur maximum {blur:g} pixels; frozen RMS increases " +
                     ", ".join(f"{value:.1f}%" for value in ratios),
                     "Preference: at least 10% degradation"))
        rows.append(("Extra-label recovery check", f"{recovered}/{len(evidence)} development replicas met recovery and retention preferences",
                     "Supported on all diagnostic replicas" if recovered == len(evidence) else
                     "Limitation: mixed recoverability; proceeding with the best valid context"))
    for method, choice in _policies(snapshot).items():
        selected = choice.get("selected", choice)
        recipe = selected.get("recipe", selected)
        status = choice.get("reason", "See sealed selection")
        if choice.get("eligible") is False:
            status = "Diagnostic; failed: " + ", ".join(choice.get("failed_conditions", [])) + "; " + status
        rows.append((LABELS[method], f"Ridge {recipe['ridge']:.3g}; FD strength {recipe['strength']:.3g}", status))
    profile = _decisions(snapshot, "resource-assessment") or snapshot.get("resource_assessment") or {}
    profile = profile.get("grid", profile)
    comparison = profile.get("numerical_comparison")
    if comparison:
        rows.append(("Compression assessment", f"Subspace {comparison['candidate_median_seconds']:.3g} s / exact CUDA {comparison['exact_median_seconds']:.3g} s",
                     "Saved-matrix measurements; whole-run speedup measured separately"))
        rows.append(("Accuracy versus exact compression", f"Matrix difference at most {100*comparison['max_matrix_relative_error']:.3g}%; accepted prediction difference {100*comparison['max_prediction_relative_error']:.3g}% of the reference update",
                     "Admission limits: 1% represented-matrix error and 5% accepted prediction change"))
        rows.append(("Numerical policy", "Cold block-power/QR and PSD-aware Nystrom, rank 64",
                     "4, then 8 or 16 power steps; exact fallback for unresolved residuals or tied cutoffs"))
        rows.append(("Numerical telemetry limitation", "Only the latest retirement diagnostics survive checkpoint rotation",
                     "Checks enforced on every retirement; historical fallback totals are unavailable, not zero"))
    elif profile.get("backends"):
        cpu, cuda = (profile["backends"].get(k, {}) for k in ("exact_cpu", "exact_cuda"))
        rows.append(("Profile: compression + reconstruction median",
            f"CPU {_display(cpu.get('median_transaction_seconds'))} s / CUDA {_display(cuda.get('median_transaction_seconds'))} s",
            "; ".join(f"{name}: {'validated' if item.get('validated') else 'not validated / not measured'}"
                      for name, item in (("CPU", cpu), ("CUDA", cuda)))))
        rows.append(("Whole-update profile", profile.get("whole_update_profile", {}).get("status", "pending"),
                     "Numerical-kernel timing is not a full-study forecast"))
    for row in ([] if comparison else profile.get("rows", [])[:8]):
        rows.append(("Profile: " + row["name"],
                     f"CPU {_display(row.get('cpu_seconds'))} s / CUDA {_display(row.get('cuda_seconds'))} s",
                     row.get("status", "Measured transaction; not end-to-end speedup")))
    text = "### Selected Settings\n\n| Setting / method | Value | Eligibility / evidence |\n| --- | --- | --- |\n"
    text += "\n".join("| " + " | ".join(_display(v) for v in row) + " |" for row in rows)
    if profile:
        text += f"\n\nResource assessment: {_display(profile.get('selected_backend', profile.get('backend', profile.get('status'))))}. "
        text += "Timings are supplied measurements, not estimates from GPU utilization; full profile and timeouts are in the supplement."
    text += "\n\nSelection is exploratory: replay first, then FD under the retention ceiling, one-sided tolerance and "
    text += f"{cfg.get('envelope_factor', 2.5):g}x selected-replay time per total grid; EWC-only is selected independently."
    return cell("settings", text)


def _frontier(snapshot):
    rows = (snapshot.get("development") or {}).get("grid", [])
    if not rows:
        return None
    cfg, policies = snapshot.get("config") or {}, _policies(snapshot)
    selected = policies.get(REPLAY, {}).get("selected")
    replay = next((r for r in rows if selected and r["recipe_key"] == selected["recipe_key"]), None)
    if selected and replay is None:
        raise ValueError("selected replay recipe missing from development grid cohort")
    figure = _figure(11, 3.5)
    axis, heat = figure.subplots(1, 2)
    for method in LABELS:
        for row in (r for r in rows if r["method"] == method):
            excluded = not row.get("resource_feasible", False) or row["base_harm"] > cfg.get("retention_margin", .5)
            if method == FD and replay:
                excluded |= (row["seconds_per_arrival"] > cfg.get("envelope_factor", 2.5)*replay["seconds_per_arrival"]
                             or row["base_harm"]-replay["base_harm"] > cfg.get("matching_tolerance", .1))
            axis.scatter(row["base_harm"], row["novel_auc"], color=COLORS[method], marker="x" if excluded else "o")
        axis.plot([], [], "o", color=COLORS[method], label=LABELS[method])
        choice = policies.get(method, {}).get("selected")
        if choice:
            row = next((r for r in rows if r["recipe_key"] == choice["recipe_key"]), None)
            if row is None:
                raise ValueError("selected recipe missing from development grid cohort")
            axis.scatter(row["base_harm"], row["novel_auc"], marker="*", s=90, color=COLORS[method], edgecolor="#111")
    axis.axvline(cfg.get("retention_margin", .5), ls="--", color="#555")
    axis.set(xlabel="Old-domain deterioration H (pixels)", ylabel="Novel A (square pixels)", title="Development acquisition / retention")
    axis.legend(fontsize=6)
    fd = [r for r in rows if r["method"] == FD]
    if fd:
        ridges, strengths = (sorted({r["recipe"][key] for r in fd}) for key in ("ridge", "strength"))
        values = np.full((len(ridges), len(strengths)), np.nan)
        for row in fd:
            i, j = ridges.index(row["recipe"]["ridge"]), strengths.index(row["recipe"]["strength"])
            values[i, j] = row["novel_auc"]
            cost = f"{row['seconds_per_arrival']/replay['seconds_per_arrival']:.1f}x" if replay and replay["seconds_per_arrival"] > 0 else "pending"
            heat.text(j, i, cost, ha="center", va="center", fontsize=7, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        image = heat.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", cmap="cividis")
        heat.set(xticks=range(len(strengths)), xticklabels=[f"{v:.2g}" for v in strengths],
                 yticks=range(len(ridges)), yticklabels=[f"{v:.3g}" for v in ridges])
        figure.colorbar(image, ax=heat, label="Novel A (square pixels)", shrink=.8)
    heat.set(xlabel="FD strength", ylabel="Ridge c", title="FD risk / selected-replay cost")
    for ax in (axis, heat):
        _style(ax)
    return picture("development", figure, "Exploratory grid-cohort means; exclusions are crosses. Stars locate selected "
        "recipes in this same cohort, not expanded-cohort scores. Heatmap labels compare per-grid cost to selected replay "
        "in this cohort, pending until selection; blanks are untested. Full frontiers and exclusions remain in the supplement.")


def _curves(axis, result, domain, metric="mse"):
    for method, item in result["methods"].items():
        curves = item["curves"]
        values = curves[domain].get("mean_" + metric)
        if values is None:
            continue
        transform = np.sqrt if metric == "mse" else lambda v: np.asarray(v)*(100 if metric == "accuracy1" else 1)
        axis.plot(curves["n"], transform(values), c=COLORS[method], label=LABELS[method])
        if metric == "mse" and "ci_mse" in curves[domain]:
            lo, hi = np.sqrt(curves[domain]["ci_mse"])
            axis.fill_between(curves["n"], lo, hi, color=COLORS[method], alpha=.12)
    frozen = result["methods"][REPLAY]["curves"][domain].get("mean_" + metric)
    if frozen is not None:
        value = np.sqrt(frozen[0]) if metric == "mse" else frozen[0]*(100 if metric == "accuracy1" else 1)
        axis.axhline(value, c="#555", ls="--", label="Frozen initialization")
    axis.set(xlabel="Novel grids (9 digit pairs/grid)")
    axis.legend(fontsize=6)
    _style(axis)


def _predictive(result):
    figure = _figure(11, 6)
    axes = figure.subplots(2, 2)
    for column, domain in enumerate(("novel", "base")):
        for row, metric in enumerate(("mse", "mae")):
            ax = axes[row, column]
            _curves(ax, result, domain, metric)
            ax.set(title=("New" if domain == "novel" else "Old") + " held-out grids",
                   ylabel="RMS landmark error (pixels)" if metric == "mse" else "Mean landmark distance (pixels)")
    uncertainty = ("Provisional descriptive means only: no bootstrap intervals, shading or significance claims. "
                   if result.get("provisional") else "Bands are runner-computed pointwise nominal paired 95% intervals. ")
    return picture("predictive", figure, f"{len(result['replicas'])} complete three-arm paired replicas. "
        "Risk is squared-coordinate error summed / 27 landmarks; RMS curves are sqrt(mean replica risk), "
        "not mean replica RMS. " + uncertainty +
        "Both new and old panels are held out; retired-training-cohort errors are separate supplement diagnostics.")


def _efficiency(result):
    point, boot = result["efficiency"]["point"], result["efficiency"]["bootstrap"]
    figure = _figure(12, 3.4)
    quality, cost, retention = figure.subplots(1, 3)
    for method in (FD, REPLAY):
        curves = result["methods"][method]["curves"]
        quality.plot(curves["n"], np.sqrt(curves["novel"]["mean_mse"]), c=COLORS[method], label=LABELS[method])
        cost.plot(curves["n"], curves["learning_seconds"], c=COLORS[method], label=LABELS[method])
    quality.axhline(np.sqrt(point["target_mse"]), ls="--", c="#555", label="Replay endpoint target")
    frozen = result["methods"][REPLAY]["curves"]["novel"]["mean_mse"][0]
    quality.axhline(np.sqrt(frozen), ls=":", c="#777", label="Frozen initialization")
    valid = point["status"] == "valid"
    if valid:
        quality.axvline(point["matching_budget"], ls=":", c=COLORS[FD])
        retention.bar(["FD match", "Replay endpoint"], [point["fd_old_rmse"], point["replay_old_rmse"]], color=[COLORS[FD], COLORS[REPLAY]])
    else:
        retention.text(.5, .5, "Matched retention unresolved", ha="center", transform=retention.transAxes, fontsize=8)
    state = "point match; inverse CI unresolved" if valid and (boot["ci"] is None or boot["invalid"]) else "observed match" if valid else "match unresolved"
    quality.set(title=state, xlabel="Novel grids", ylabel="Root mean risk (pixels)")
    quality.secondary_xaxis("top", functions=(lambda n: 9*n, lambda n: n/9)).set_xlabel("Novel digit pairs")
    quality.legend(fontsize=6)
    cost.set(title="Measured learning compute", xlabel="Novel grids", ylabel="Mean learning seconds")
    retention.set(title="Matched-budget old retention", ylabel="Mean replica RMS (pixels)")
    for ax in (quality, cost, retention):
        _style(ax)
    return picture("efficiency", figure, "Invert mean square-pixel risk at the last downward crossing persisting to N; "
        "plots show its square root. No smoothing, fitting or extrapolation. Matched time includes all nominal arrivals, "
        "preparation, solves, replay, FD, retirement and compression; it differs from equal-budget compute. "
        "Matched retention takes roots within replicas and is descriptive when inverse intervals are withheld.")


def _applied(snapshot, result):
    figure = _figure(11, 3.4)
    success, cells = figure.subplots(1, 2)
    _curves(success, result, "novel", "accuracy1")
    success.set(title="New held-out landmarks", ylabel="Within one pixel (%)")
    per_cell = (snapshot.get("per_cell") or {}).get("grid", {})
    if not per_cell:
        groups = {m: {} for m in LABELS}
        for raw in snapshot.get("confirmation") or []:
            row = flatten_record(raw)
            if row.get("context_key") == "grid" and row.get("replica") in result["replicas"] and row.get("method") in groups:
                endpoint = row.get("metrics", [])[-1:]
                if endpoint and endpoint[0].get("n") == result["N"] and "per_cell_mse" in endpoint[0]:
                    group = groups[row["method"]]
                    if row["replica"] in group:
                        raise ValueError("duplicate per-cell endpoint in analyzed cohort")
                    group[row["replica"]] = endpoint[0]["per_cell_mse"]["novel"]
        if all(set(group) == set(result["replicas"]) for group in groups.values()):
            arrays = {m: np.asarray(list(group.values()), dtype=float) for m, group in groups.items()}
            if any(a.shape != (len(result["replicas"]), 9) or not np.isfinite(a).all() or (a < 0).any() for a in arrays.values()):
                raise ValueError("per_cell grid diagnostic requires nine nonnegative endpoint risks per method")
            per_cell = {m: a.mean(axis=0) for m, a in arrays.items()}
    for method, values in per_cell.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (9,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("per_cell grid diagnostic requires nine nonnegative endpoint risks per method")
        cells.plot(range(1, 10), np.sqrt(values), "o-", c=COLORS[method], label=LABELS[method])
    if per_cell:
        cells.legend(fontsize=6)
    else:
        cells.text(.5, .5, "Per-cell artifact pending", ha="center", transform=cells.transAxes)
    cells.set(title="New holdout: endpoint by cell", xlabel="Cell (row-major)", ylabel="RMS landmark error (pixels)", xticks=range(1, 10))
    _style(cells)
    return picture("applied", figure, "One-pixel success uses Euclidean landmark distance, not digit classification. "
        "Cell risks divide each six-coordinate squared-error sum by three landmarks. Cells are not independent replica "
        "trajectories; extended digit/cell/anchor/coordinate diagnostics and old-domain success remain in the supplement.")


def _outcomes(result):
    text = "### Predictive Outcomes\n\n| Method | Novel A (square pixels) | Old H (pixels) | Learning seconds |\n| --- | ---: | ---: | ---: |\n"
    for method, item in result["methods"].items():
        text += f"| {LABELS[method]} | {interval(item['acquisition'])} | {interval(item['retention'])} | {item['costs']['mean_learning_seconds']:.3g} |\n"
    if result.get("provisional"):
        return cell("outcomes", text + "\nProvisional descriptive means only; no paired intervals yet. "
            "A averages budgets 1..N; H takes roots within replicas. Acquisition/retention inference awaits runner analysis.")
    difference = result["comparisons"][f"{FD}_vs_{REPLAY}"]["retention_difference"]
    text += (f"\nPaired acquisition gain (replay minus FD): {interval(result['G'])} square pixels. "
             f"Paired full-budget H difference (FD minus replay): {interval(difference)} pixels. "
             "A averages risk at budgets 1..N; H roots within each replica before subtracting frozen old error. "
             "Brackets are nominal whole-paired-replica 95% CIs, conditional on frozen policies and fixed panels; not familywise claims.")
    return cell("outcomes", text)


def _comparison_table(result):
    efficiency = result["efficiency"]
    point, boot = efficiency["point"], efficiency["bootstrap"]
    valid = point["status"] == "valid"
    # An unresolved draw withholds ALL inverse intervals, not just its own score.
    ci = boot.get("ci") if valid and boot["invalid"] == 0 else None

    def estimate(key, scale=1, spec=".3g", suffix=""):
        if not valid:
            return "Unresolved"
        text = f"{point[key]*scale:{spec}}{suffix}"
        if ci is not None:
            lo, hi = ci[key]
            text += f" [{lo*scale:{spec}}{suffix}, {hi*scale:{spec}}{suffix}]"
        return text + (" (descriptive; inverse CI withheld)" if ci is None else "")

    fd_time, replay_time = (result["methods"][m]["costs"]["mean_learning_seconds"] for m in (FD, REPLAY))
    matched = ("Unresolved" if not valid or ci is None else
               "Supported" if efficiency["matched_budget_retention_supported"] else "Not supported")
    rows = [("**$S$**", "**" + estimate("S") + "**"),
            ("Samples saved", estimate("saved_fraction", 100, ".2f", "%")),
            ("Novel grids: FD match / replay endpoint", estimate("matching_budget", spec=".2f") + f" / {result['N']}"),
            ("Novel digit pairs: FD match / replay endpoint", estimate("matching_budget", 9, ".2f") + f" / {9*result['N']}"),
            ("Matched-quality compute (FD / replay)", estimate("learning_time_ratio", suffix="x")),
            ("Equal-budget compute (FD / replay)", f"{fd_time/replay_time:.3g}x" if replay_time > 0 else "Unavailable"),
            ("Full-budget retention", "Supported" if efficiency["full_budget_retention_supported"] else "Not supported"),
            ("Matched-budget retention", matched),
            ("Matched old RMS difference, FD minus replay (pixels)", estimate("old_rmse_difference")),
            ("Inverse status", "Point estimate with 95% CI" if ci is not None else "Descriptive; CI withheld" if valid else "Unresolved: " + point["status"].replace("_", " ")),
            ("Unresolved bootstrap inversions", f"{boot['invalid']}/{boot['draws']} unresolved draws"),
            ("Multiple-crossing draws", f"{boot['multiple_crossing_draws']}/{boot['draws']}")]
    return "| Statistic | Grid task |\n| --- | --- |\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def build_cells(snapshot, diagnostics_href=None):
    result = _analysis(snapshot)
    cfg = snapshot.get("config") or {}
    N = cfg.get("novel_count", 100)
    out = [_progress(snapshot, result), cell("intro", "## Phase 11: Joint Nine-Digit Registration\n\n"
        "Can FD KL learn useful new geometry from fewer grids while retaining old behavior at an acceptable compute premium? "
        "One 84x84 reference/query grid pair contains **nine digit pairs**, 18 image views, 27 landmarks and 54 targets. "
        f"The online endpoint is {N} novel grids = {9*N} novel digit pairs, plus every intervening nominal grid. "
        "Independent cell affines share one CNN and unrestricted dense head; this is one grid/54-output task.\n\n"
        r"Gaussian working likelihood: $Y|x\sim N(f_\theta(x),4I_{54})$, "
        r"$L_i=\|f_\theta(x_i)-y_i\|^2/8+27\log(8\pi)$ and $F_i=J_i^TJ_i/4$. "
        "Sum coordinates and average grids for likelihood/Fisher; **predictive risk divides summed squared error by 27 landmarks**, "
        "not 54 coordinates. RMS error is its square root in pixels. A is budget-mean novel risk; H is old-domain deterioration. "
        f"Retention ceiling {cfg.get('retention_margin', .5):g} pixels; one-sided tolerance {cfg.get('matching_tolerance', .1):g} pixels. "
        "**S = fraction saved / matched-quality learning-compute multiplier**; 0.25 is aspirational, not a stopping rule.")]
    for item in (_overview(snapshot), _settings(snapshot), _frontier(snapshot)):
        if item:
            out.append(item)
    if result is None:
        out.append(cell("pending", "### Fresh Confirmation\n\nWaiting for complete fresh three-arm replica cohorts or final analysis. "
            "Partial cohorts have no inference or sample-equivalence claim. This report never trains, bootstraps, "
            "evaluates models, downloads data or repairs incomplete artifacts."))
    else:
        out.append(_predictive(result))
        if not result.get("provisional"):
            out.append(_efficiency(result))
        out.extend([_applied(snapshot, result), _outcomes(result)])
    if result and result.get("provisional"):
        warning = (f"**Numerical warning:** {len(result['invalid_replicas'])} complete cohorts include invalid trajectories; "
                   "retained as diagnostics, not silently excluded. " if result["invalid_replicas"] else "")
        out.append(cell("provisional", f"**Provisional confirmation:** {len(result['replicas'])} complete three-arm replica cohorts. "
            f"{len(result['incomplete_replicas'])} additional replica groups await all three arms. " + warning +
            "No bootstrap, confidence intervals, significance tests or sample-equivalence estimates are calculated. "
            "Curves and outcomes are descriptive means only; conclusions await final paired analysis."))
    elif result:
        efficiency, point = result["efficiency"], result["efficiency"]["point"]
        boot = efficiency["bootstrap"]
        prose = ("Acquisition advantage supported at full budgets. " if result["G"]["ci"][0] > 0 else "An acquisition advantage is not established. ")
        prose += ("Old-data retention supported at full and matched budgets. " if efficiency["retention_qualified"] and point["status"] == "valid" and boot["ci"] is not None and boot["invalid"] == 0 else
                  "Full-budget retention supported; matched-budget unresolved; score is retention-unqualified. "
                  if efficiency["full_budget_retention_supported"] and (point["status"] != "valid" or boot["ci"] is None or boot["invalid"]) else
                  "Retention is not supported at both budgets; score is retention-unqualified. ")
        if boot["ci"] is None or boot["invalid"]:
            prose += "**Inverse uncertainty:** no failed inversion was discarded; ordinary inverse CIs are withheld if any draw is unresolved. "
        if not point["useful_target"]:
            prose += "**Weak target:** replay improves frozen risk by less than 5% (or improvement is unavailable); numerical savings are descriptive. "
        prose += ("A practical sample-efficiency win is supported in this grid task. " if result.get("is_final") and efficiency.get("evidence_sample_savings") and point["useful_target"] and boot["ci"] is not None and boot["invalid"] == 0 else
                  "Savings do not establish a practical efficiency win; unqualified point estimates remain descriptive. ")
        if not result.get("is_final"):
            prose += "**Smoke/development evidence, not final confirmation.** "
        out.append(cell("conclusion", "### Efficiency And Retention\n\n" + _comparison_table(result) + "\n\n" + prose +
            "\n\nBracketed values are nominal paired 95% CIs; **unresolved is not zero**. Matched-quality compute differs from "
            "equal-budget compute. Online savings condition on pretraining; initialization, calibration, evaluation and archival "
            "costs are separate. Different matched budgets include different background/drift histories, not a causal label-deletion test. "
            "A higher S identifies a context, not a causal effect of output dimension or dense mixing, nor proof of STM-to-LTM transfer. "
            "Historical Phase 10 vector S=0.0712 was descriptive with unresolved inverse uncertainty; its six-output task, smaller "
            "capacity and 256/16 digit-pair replay resources differ from this 54-output grid task and 32/2-grid resources."))
    if diagnostics_href:
        out.append(cell("supplement", f"[Machine-readable supplement: resource profile, readiness, full frontiers, manifests, "
            f"cell/digit/anchor errors, spectra, numerical audits and all inverse failures]({diagnostics_href}). "
            "Missing diagnostics are pending, not passing."))
    return out


def publish(snapshot, path, store=None):
    """Append Phase 11 after history, replacing only its owned cells.

    Phase10-shaped snapshot: initialization/readiness/selection/manifest/analysis
    are keyed by ``grid`` (54 outputs); decisions may wrap the first four.
    Development rows and precomputed paired analysis retain Phase10 field names,
    with grid counts and risk normalized by 27. Sealed confirmation rows supply
    descriptive means for complete three-arm cohorts until final runner analysis;
    incomplete groups wait, invalid complete groups are flagged, and no inference
    runs here. Optional preview landmarks are
    (9,3,2) cell-local positions; per_cell[grid][method] holds nine novel endpoint
    risks, or confirmation endpoints supply per_cell_mse[novel] for every analyzed
    arm/replica. resource_assessment (or resource-assessment decision) accepts the
    compressor's backends/selected_backend/whole_update_profile plus optional rows
    {name, cpu_seconds, cuda_seconds, status} (eight displayed). Full snapshot goes
    to immutable content-addressed JSON. Store supplies root/reserve and verifies seals.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    previous = json.loads(path.read_bytes()) if path.exists() else dict(nbformat.v4.new_notebook())
    if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
        raise ValueError("cannot publish into a frozen historical notebook")
    cells = build_cells(snapshot)
    supplement = None
    if store is not None:
        supplement = (json.dumps(snapshot, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
        target = Path(store.root) / "reports" / ("phase11-supplement-" + hashlib.sha256(supplement).hexdigest() + ".json")
        href = os.path.relpath(target, path.parent)
        cells.append(cell("supplement", f"[Machine-readable supplement: full profile, readiness, frontiers, manifests, "
            f"cell/digit errors, spectra, audits and inverse failures]({href}). Missing diagnostics are pending, not passing."))
    history = [c for c in previous["cells"] if not owned(c)]
    book = deepcopy(previous)
    book["cells"] = history[:1] + [cells[0]] + history[1:] + cells[1:]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    payload = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    if supplement is not None and not target.exists():
        with store.reserve(len(supplement)):
            _atomic(target, supplement)
    with store.reserve(len(payload)) if store is not None else nullcontext():
        _atomic(path, payload)
