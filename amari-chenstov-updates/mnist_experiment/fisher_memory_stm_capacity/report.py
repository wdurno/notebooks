"""Bounded Phase 12 publication from compact, precomputed scalar artifacts.

publish accepts progress, configurations (list or name-keyed mapping), a list
of development summaries, arm-keyed selection, analysis or None, and changes.
No inference, bootstrap, model loading or artifact repair occurs here.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path

import nbformat
import numpy as np

from mnist_experiment.fisher_memory_registration.report import _atomic
from mnist_experiment.fisher_memory_regression.report import _figure, _image, _style, MAX_NOTEBOOK_BYTES

from .analysis import PROTOCOL, FD, REPLAY, STM_CONFIGS, ARMS, REFERENCE, INCUMBENT

TAG, PREFIX = "plan15_phase12", "plan15-phase12-"
COLORS = {"head_r1": "#a14f79", "all_r1": "#b06927", "all_r2": "#3273a6", "all_r4": "#268575"}
NAMES = {"head_r1": "Head r1", "all_r1": "All r1", "all_r2": "All r2", "all_r4": "All r4"}


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def cell(name, text):
    return nbformat.v4.new_markdown_cell(text, id=PREFIX+name, metadata={TAG: True})


def picture(name, figure, caption):
    result = _image("phase12-"+name, figure, caption)
    result["id"], result["metadata"] = PREFIX+name, {TAG: True}
    return result


def _text(value):
    return html.escape(str(value), quote=False).replace("|", "/").replace("\n", " ")


def _number(value, digits=3):
    return "pending" if value is None else f"{value:.{digits}g}"


def _interval(effect, digits=3):
    if not effect or effect.get("mean") is None:
        return "Unresolved"
    text = _number(effect["mean"], digits)
    if effect.get("ci") is not None:
        text += " ["+", ".join(_number(v, digits) for v in effect["ci"])+"]"
    return text


def _label(arm):
    chart, method = arm.split("--")
    return NAMES[chart]+(" FD" if method == FD else " Replay")


def _style_arm(arm):
    chart, method = arm.split("--")
    return dict(color=COLORS[chart], linestyle="-" if method == FD else "--", label=_label(arm))


def _analysis(snapshot):
    result = snapshot.get("analysis")
    if result is None:
        return None
    if (not isinstance(result, dict) or result.get("protocol") != PROTOCOL or result.get("schema_version") != 1
            or result.get("status") not in ("completed", "smoke_completed") or result.get("context_key") != "grid"
            or result.get("output_dim") != 54 or result.get("reference_arm") != REFERENCE
            or result.get("incumbent_arm") != INCUMBENT or set(result.get("arms", {})) != set(ARMS)
            or len(result.get("replicas", [])) < 2):
        raise ValueError("completed compatible eight-arm Phase 12 grid analysis required")
    if result.get("partition") != "confirmation":
        raise ValueError("development analysis cannot be displayed as confirmation")
    return result


def _partial(snapshot):
    progress = snapshot.get("progress") or {}
    completed = progress.get("completed_blocks", progress.get("paired_blocks"))
    planned = progress.get("planned_blocks", progress.get("confirmation_blocks_target"))
    return progress.get("partial_confirmation") is True or (
        isinstance(completed, int) and isinstance(planned, int) and completed < planned)


def _progress(snapshot):
    p = snapshot.get("progress") or {}
    status = p.get("status", "pending")
    if status in ("running", "live") and not any(p.get(k) is True for k in
            ("process_verified", "lease_verified", "process_alive", "status_verified")):
        status = "running unverified (process/lease check required)"
    active, allocation = p.get("active_seconds", 0), p.get("allocation_seconds", 72000)
    storage = p.get("storage") or {}
    used = storage.get("used_bytes", 0)+storage.get("external_bytes", 0)
    cap = storage.get("limit_bytes", 500_000_000_000)
    forecast = "pending" if p.get("forecast_seconds") is None else f"{p['forecast_seconds']/3600:.2f} h"
    counts = p.get("completed_by_partition") or {}
    completed_blocks = p.get("completed_blocks", p.get("paired_blocks", 0))
    planned_blocks = p.get("planned_blocks", p.get("confirmation_blocks_target", "?"))
    mandatory = p.get("mandatory_planned", p.get("mandatory_development", "?"))
    optional = p.get("optional_planned", p.get("optional_development", "?"))
    return cell("progress", f"**Phase 12:** {_text(status)} / {_text(p.get('stage', '12A'))} | "
        f"{_text(p.get('completed', 0))}/{_text(p.get('planned', '?'))} trajectories | "
        f"{_text(completed_blocks)}/{_text(planned_blocks)} eight-arm paired blocks | "
        f"{_text(p.get('completed_pairs', 0))}/{_text(p.get('planned_pairs', '?'))} within-size pairs. "
        f"Mandatory development {_text(p.get('mandatory_completed', '?'))}/{_text(mandatory)}; "
        f"optional {_text(p.get('optional_completed', '?'))}/{_text(optional)}. "
        f"Active unit/checkpoint: {_text(p.get('current', p.get('active_unit', 'pending')))}/"
        f"{_text(p.get('last_accepted_checkpoint', p.get('last_checkpoint', 'pending')))}. "
        f"{active/3600:.2f}/{allocation/3600:g} active h; forecast remaining {forecast}; "
        f"storage {used/1e9:.2f}/{cap/1e9:g} GB, {(cap-used)/1e9:.2f} GB headroom. "
        f"Development: {_text(p.get('development_status', counts.get('development', 'pending')))}; confirmation: "
        f"{_text(p.get('confirmation_status', counts.get('confirmation', 'pending')))}. Heartbeat: {_text(p.get('heartbeat', 'not recorded'))}. "
        "Partial evidence is descriptive; no significance stopping or renewed allocation on resume.")


def _settings(snapshot, result):
    configurations = snapshot.get("configurations") or {}
    if isinstance(configurations, list):
        configurations = {r["name"]: r for r in configurations}
    selection = snapshot.get("selection") or {}
    selection = selection.get("policies", selection)
    rows = []
    for chart, defaults in STM_CONFIGS.items():
        setting = {**defaults, **configurations.get(chart, {})}
        if any(setting[k] != defaults[k] for k in ("stm_parameters", "ltm_parameters")):
            raise ValueError("configuration counts disagree with Phase 12 chart")
        for method in (FD, REPLAY):
            arm = f"{chart}--{method}"
            choice = selection.get(arm) or {}
            selected = choice.get("selected", choice) or {}
            recipe = selected.get("recipe") or (result["arms"][arm]["recipe"] if result else {})
            note = "Common reference B; historical frozen policy" if arm == REFERENCE else (
                "Incumbent; historical frozen policy" if arm == INCUMBENT else "Selected" if recipe else "Pending")
            if choice.get("eligible") is False:
                note = "Diagnostic; failed: "+_text(choice.get("failures", choice.get("failed_conditions", [])))
            rows.append(f"| {_label(arm)} | {setting['stm_parameters']:,} / {setting['ltm_parameters']:,} | "
                f"{'Head only' if chart == 'head_r1' else 'Both convolutions + head'} | "
                f"{_number(recipe.get('ridge'))} / {_number(recipe.get('strength'))} | {note} |")
    return cell("settings", "### Settings\n\n| Arm | STM / LTM parameters | Placement | Ridge / FD strength | Policy |\n"
        "| --- | ---: | --- | ---: | --- |\n"+"\n".join(rows)+"\n\n"
        "Both **FD** and **Replay** use a replay buffer, EWC, ridge, and trainable STM; "
        "**FD** also adds the finite-difference KL penalty.\n\n"
        "All base weights and biases remain trainable. Counts are stored parameters, not identifiable degrees of freedom. "
        "The common reference and incumbent are rerun on fresh paired replicas, not pooled with historical confirmation. "
        "Historical rank-two policies had a larger lifetime development search; tuning effort is not identical across sizes. "
        "Task and resources remain Phase 11's (4,6)-channel CNN, 45 pretraining epochs, blur 2.4, FIFO32/replay2, "
        "100 novel grids (900 digit pairs), archive rank 64 and Gaussian covariance 4I_54.")


def _frontier(snapshot):
    rows = snapshot.get("development") or []
    if not rows:
        return None
    figure = _figure(12, 3.6)
    retention, cost = figure.subplots(1, 2)
    seen = set()
    for row in rows:
        chart, method = row.get("stm_config"), row.get("method")
        if chart not in STM_CONFIGS or method not in (FD, REPLAY):
            raise ValueError("development summary needs STM chart and adaptive method")
        arm = f"{chart}--{method}"
        acquisition = row.get("novel_auc")
        harm = row.get("H", row.get("base_harm"))
        if isinstance(harm, dict):
            harm = harm.get("mean")
        seconds = row.get("seconds_per_arrival")
        label = _label(arm) if arm not in seen else None
        seen.add(arm)
        for axis, x in ((retention, harm), (cost, seconds)):
            if x is not None and acquisition is not None:
                axis.scatter(x, acquisition, color=COLORS[chart], marker="o" if method == FD else "x",
                             label=label, s=28, alpha=.8)
    retention.axvline(.5, color="#555", ls=":", label="Retention ceiling (point guide)")
    retention.set(title="Bounded development: quality / retention", xlabel="Old H (pixels)", ylabel="Novel A (square pixels)")
    cost.set(title="Quality / measured online cost", xlabel="Learning seconds / arrival", ylabel="Novel A (square pixels)")
    for axis in (retention, cost):
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=6, ncol=2)
        _style(axis)
    return picture("development", figure, "Development only; lower acquisition risk and cost are favorable. "
        "Points are compact recipe summaries, not independent replicas or confirmation evidence. "
        "The ceiling guide does not replace the paired one-sided selection rules; full tuning records and numerical audits are in the supplement.")


def _predictive(result):
    figure = _figure(12, 7)
    axes = figure.subplots(2, 2)
    for axis, domain, title in ((axes[0, 0], "novel", "New held-out grids"), (axes[0, 1], "base", "Old held-out grids")):
        for arm, item in result["arms"].items():
            curves = item["curves"]
            axis.plot(curves["n"], np.sqrt(curves[domain]["mean_mse"]), **_style_arm(arm))
            lo, hi = np.sqrt(curves[domain]["ci_mse"])
            axis.fill_between(curves["n"], lo, hi, color=COLORS[item["stm_config"]], alpha=.07)
        frozen = result["arms"][REFERENCE]["curves"][domain]["mean_mse"][0]
        axis.axhline(np.sqrt(frozen), color="#555", ls=":", label="Frozen initialization")
        axis.set(title=title, xlabel="Novel grids", ylabel="RMS landmark error (pixels)")
        axis.legend(fontsize=6, ncol=3)
    success, retired = axes[1]
    for arm, item in result["arms"].items():
        curves = item["curves"]
        if "mean_accuracy1" in curves["novel"]:
            success.plot(curves["n"], 100*np.asarray(curves["novel"]["mean_accuracy1"]), **_style_arm(arm))
        audit = item.get("retired_training_cohort")
        if audit:
            retired.plot(audit["n"], np.sqrt(audit["mean_mse"]), **_style_arm(arm))
    for axis in (success, retired):
        if not axis.lines:
            axis.text(.5, .5, "Diagnostic artifact pending", ha="center", transform=axis.transAxes)
        else:
            axis.legend(fontsize=6, ncol=2)
    success.set(title="New holdout: one-pixel outcomes", xlabel="Novel grids", ylabel="Landmarks within one pixel (%)")
    retired.set(title="Evicted training cohort (not held out)", xlabel="Novel grids", ylabel="RMS landmark error (pixels)")
    for axis in axes.flat:
        _style(axis)
    return picture("predictive", figure, "Bands are precomputed pointwise nominal paired 95% intervals. "
        "RMS curves root mean replica risk; retention roots within each replica. "
        "Risk sums 54 squared-coordinate errors / 27 landmarks, not / 54 coordinates. "
        "Evicted-training-cohort diagnostics are separate, descriptive and may have varying replica coverage. "
        "New and old endpoint RMS/one-pixel outcomes appear in the predictive table.")


def _matching(result):
    figure = _figure(12, 3.8)
    quality, cost = figure.subplots(1, 2)
    target = result["arms"][REFERENCE]["curves"]["novel"]["mean_mse"][-1]
    for arm, item in result["arms"].items():
        curves, point = item["curves"], item["efficiency"]["point"]
        quality.plot(curves["n"], np.sqrt(curves["novel"]["mean_mse"]), **_style_arm(arm))
        cost.plot(curves["n"], curves["learning_seconds"], **_style_arm(arm))
        if point["status"] == "valid":
            n = point["matching_budget"]
            quality.scatter(n, np.sqrt(target), color=COLORS[item["stm_config"]], marker="x", s=36)
            cost.scatter(n, point["arm_learning_seconds"], color=COLORS[item["stm_config"]], marker="x", s=36)
    quality.axhline(np.sqrt(target), color="#555", ls=":", label="Common B endpoint target")
    quality.secondary_xaxis("top", functions=(lambda n: 9*n, lambda n: n/9)).set_xlabel("Novel digit pairs")
    quality.set(title="Observed common-target matches", xlabel="Novel grids", ylabel="Root mean novel risk (pixels)")
    cost.set(title="Measured learning compute", xlabel="Novel grids", ylabel="Mean cumulative learning seconds")
    for axis in (quality, cost):
        axis.legend(fontsize=6, ncol=2)
        _style(axis)
    return picture("matching", figure, "The last sustained downward crossing of mean squared risk fixes the match; "
        "markers are point matches, not valid confidence intervals when bootstrap inverses fail. "
        "No smoothing, extrapolation or fitted trend. Every arm uses the same all-r2 Replay/EWC target and "
        "endpoint time denominator. The reference has zero savings by definition. Learning time includes intervening "
        "arrivals, preparation, solves, replay, FD, retirement and compression; matched histories differ in background/drift exposure.")


def _scores(result):
    figure = _figure(12, 3.8)
    score, difference = figure.subplots(1, 2)
    for method in (FD, REPLAY):
        connected = []
        for chart, configuration in STM_CONFIGS.items():
            arm = f"{chart}--{method}"
            efficiency = result["arms"][arm]["efficiency"]
            point, ci = efficiency["point"], efficiency["bootstrap"]["ci"]
            if point.get("S") is None or point["status"] != "valid":
                continue
            x, y = configuration["stm_parameters"], point["S"]
            marker = "D" if chart == "head_r1" else "o" if method == FD else "s"
            score.scatter(x, y, color=COLORS[chart], marker=marker, label=_label(arm), s=35,
                          facecolors="none" if ci is None else COLORS[chart])
            if ci:
                score.vlines(x, *ci["S"], color=COLORS[chart])
            if chart != "head_r1":
                connected.append((x, y))
            if method == FD:
                effect = result["incumbent_differences"].get(arm, {}).get("S_difference", dict(mean=0., ci=[0., 0.]))
                if effect.get("mean") is not None:
                    difference.scatter(x, effect["mean"], color=COLORS[chart], marker=marker, label=NAMES[chart])
                    if effect.get("ci"):
                        difference.vlines(x, *effect["ci"], color=COLORS[chart])
        if len(connected) == 3:
            score.plot(*np.asarray(connected).T, color="#555", ls="-" if method == FD else "--", alpha=.5)
    for axis in (score, difference):
        axis.axhline(0, color="#888", ls=":")
        axis.set(xlabel="Stored STM parameters", xticks=[108, 172, 344, 688])
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=6, ncol=2)
        _style(axis)
    score.set(title="One common-reference efficiency score", ylabel="S = fraction saved / compute ratio")
    difference.set(title="FD minus current rank-two FD", ylabel="Paired difference in S")
    return picture("scores", figure, "Head-only (diamond) changes placement and size and is never connected to the all-location ranks. "
        "Replay-size controls use the same headline reference; their gains are not credited to FD. "
        "Whiskers are nominal paired 95% intervals; hollow points have withheld inverse intervals. "
        "Differences use shared draws, not overlap of separate intervals. The largest observed S does not establish an optimal size.")


def _predictive_table(result):
    lines = ["### Predictive And Retention Outcomes", "",
        "| Arm | Novel A | Gain vs B / favorable replicas | Within-size replay minus FD A | Old H | New / old endpoint RMS | New / old within 1px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, item in result["arms"].items():
        common = result["common_reference_comparisons"][arm]["acquisition_gain"]
        within = result["within_size"][item["stm_config"]]["acquisition_gain"]
        applied = item["applied"]
        rms = " / ".join(_interval(applied[d]["endpoint_rms"]) for d in ("novel", "base"))
        success = " / ".join(_interval(applied[d].get("endpoint_accuracy1")) for d in ("novel", "base"))
        lines.append(f"| {_label(arm)} | {_interval(item['acquisition'])} | {_interval(common)}; "
            f"{common['favorable_replicas']}/{len(result['replicas'])} | "
            f"{_interval(within) if item['method'] == FD else 'Control'} | {_interval(item['retention'])} | {rms} | {success} |")
    lines.append("\nA is mean squared-pixel risk over budgets 1..N. H is sqrt(mean old risk over 1..N) "
        "minus frozen old RMS, computed within replicas. One-pixel entries are landmark success fractions. "
        "Gains are positive when acquisition improves. Brackets are nominal paired 95% CIs, not familywise claims.")
    return cell("outcomes", "\n".join(lines))


def _efficiency_table(result):
    lines = ["### Common-Reference Efficiency", "",
        "| Arm | **S [95% CI]** | Paired S minus incumbent | Saved % | Matched grids / digit pairs | Matched / equal-budget compute | Retention full / matched | Invalid draws |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for arm, item in result["arms"].items():
        efficiency = item["efficiency"]
        point, bootstrap = efficiency["point"], efficiency["bootstrap"]
        ci = bootstrap["ci"] if bootstrap["invalid"] == 0 and point["status"] == "valid" else None

        def estimate(key, scale=1):
            if point["status"] != "valid" or point.get(key) is None:
                return "Unresolved"
            text = _number(scale*point[key])
            if ci is not None:
                text += " ["+", ".join(_number(scale*v) for v in ci[key])+"]"
            return text+(" (CI withheld)" if ci is None else "")

        difference = result["incumbent_differences"].get(arm, {}).get("S_difference")
        difference_text = "Incumbent (0)" if arm == INCUMBENT else _interval(difference)
        if arm != INCUMBENT and difference and difference.get("ci") is None:
            difference_text += " (CI withheld)"
        equal = dict(mean=efficiency["equal_budget_learning_time_ratio"], ci=efficiency.get("equal_budget_learning_time_ratio_ci"))
        full = "Supported" if efficiency["full_budget_retention_supported"] else "Not supported"
        matched = "Unresolved" if ci is None else "Supported" if efficiency["matched_budget_retention_supported"] else "Not supported"
        lines.append(f"| {_label(arm)} | **{estimate('S')}** | {difference_text} | {estimate('saved_fraction', 100)} | "
            f"{estimate('matching_budget')} / {estimate('matching_budget', 9)} | {estimate('learning_time_ratio')} / "
            f"{_interval(equal)} | {full} / {matched} | {bootstrap['invalid']}/{bootstrap['draws']} |")
    lines.append("\nAll eight arms, including size-matched replay, use **all_r2 Replay/EWC (B)** at N. "
        "B has S=0 by definition, not self-inversion. Unresolved is not zero; any invalid draw withholds that arm's "
        "ordinary inverse CIs and every affected paired S difference, without removing full-budget A/H intervals. "
        f"Both full and matched retention require one-sided 95% bounds under the {result['delta']:g}px ceiling "
        f"and {result['epsilon']:g}px tolerance. "
        "Pretraining, calibration, evaluation and restart costs are disclosed separately, not silently folded into online S.")
    return cell("efficiency", "\n".join(lines))


def build_cells(snapshot, diagnostics_href=None):
    result = _analysis(snapshot)
    partial = _partial(snapshot)
    endpoint = result["N"] if result else 100
    out = [_progress(snapshot), cell("intro", "## Phase 12: STM Capacity And Placement\n\n"
        "Does a smaller or larger short-term memory improve new-domain learning per compute while retaining old behavior? "
        "STM is the LoRA-shaped adapter; LTM is the trainable base network. "
        "S is fraction of novel grids saved / matched-quality learning-compute ratio. "
        "The task, replay resources and common rank-two Replay/EWC reference are fixed; eight paired policies vary STM size/placement. "
        "Head-only is a joint size-and-placement intervention, not a pure dimension experiment. "
        f"Reported endpoint: {endpoint} novel grids = {9*endpoint} novel digit pairs.")]
    if result and partial:
        out.append(cell("partial", "**Partial confirmation, descriptive only.** Only complete eight-arm replica blocks "
            "contribute; the frozen planned cohort is not complete. Displayed nominal intervals and retention labels "
            "are provisional repeated-look summaries, not final confirmatory evidence. No confirmed winner, optimal "
            "size or stopping decision follows before all frozen blocks finish. Unfinished blocks are not exclusions."))
    out.append(_settings(snapshot, result))
    frontier = _frontier(snapshot)
    if frontier:
        out.append(frontier)
    if result is None:
        out.append(cell("pending", "### Fresh Confirmation Pending\n\nWaiting for complete fresh eight-arm paired analysis. "
            "Partial cohorts have no inference or sample-equivalence claim. This notebook displays artifacts only: "
            "no training, data downloads, model evaluation, bootstrap or repair of incomplete runs."))
    else:
        out.extend([_predictive(result), _matching(result), _scores(result), _predictive_table(result), _efficiency_table(result)])
        supported = []
        for chart in ("head_r1", "all_r1", "all_r4"):
            arm = f"{chart}--{FD}"
            effect = result["incumbent_differences"][arm]["S_difference"]
            efficiency = result["arms"][arm]["efficiency"]
            if (not partial and result.get("is_final") and effect.get("ci") is not None and effect["ci"][0] > 0
                    and efficiency["retention_qualified"] and efficiency["point"]["useful_target"]):
                supported.append(NAMES[chart])
        conclusion = ("Partial results are descriptive until every frozen confirmation block is complete; no winner is established. "
            if partial else "Retention-qualified improvement in S over the incumbent: "+", ".join(supported)+". " if supported else
            "No retention-qualified improvement in S over the incumbent is established; point estimates remain trade-offs, not an optimum. ")
        weak = [NAMES[c] for c in STM_CONFIGS if not result["arms"][f"{c}--{FD}"]["efficiency"]["point"]["useful_target"]]
        if weak:
            conclusion += "The common reference improves frozen risk by less than 5%; savings are descriptive. "
        if not result.get("is_final"):
            conclusion += "Smoke/development-sized evidence is not final confirmation. "
        out.append(cell("interpretation", "### Interpretation\n\n"+conclusion+
            "Results concern calibrated regularized policies, not an architecture-independent capacity law. "
            "Positive S without valid retention is not a supported win. No STM-to-LTM information-transfer claim follows "
            "from parameter counts, Fisher traces or efficiency. Savings are conditional on pretraining and different "
            "background/drift histories at matched budgets, not a causal deletion of novel labels."))
    changes = snapshot.get("changes") or []
    if changes:
        out.append(cell("changes", "### Corrections\n\n"+"\n".join("- "+_text(c) for c in changes)))
    href = diagnostics_href or snapshot.get("supplement_href")
    if href:
        out.append(cell("supplement", f"[Machine-readable supplement: all development recipes, numerical audits, "
            f"seals, cost ledgers, retention and inverse failures]({href}). Missing diagnostics are pending, not passing."))
    return out


def publish(snapshot, notebook, store=None):
    """Update only Phase 12 cells and one near-top progress cell, atomically.

    A Store supplies root/reserve to account for notebook and content-addressed
    JSON supplement bytes. Without a Store, the caller owns storage accounting.
    Existing cells, metadata and outputs of every other phase are preserved.
    """
    notebook = Path(notebook)
    if notebook.exists() and notebook.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    previous = json.loads(notebook.read_bytes()) if notebook.exists() else dict(nbformat.v4.new_notebook())
    if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
        raise ValueError("cannot publish into a frozen historical notebook")
    supplement, target = None, None
    if store is not None:
        supplement = (json.dumps(snapshot, sort_keys=True, indent=1, allow_nan=False)+"\n").encode()
        target = Path(store.root)/"reports"/("phase12-supplement-"+hashlib.sha256(supplement).hexdigest()+".json")
    cells = build_cells(snapshot, os.path.relpath(target, notebook.parent) if target else None)
    history = [c for c in previous["cells"] if not owned(c)]
    # Preserve a section's existing chronological position, including later user cells.
    position = next((sum(not owned(c) for c in previous["cells"][:i])
                     for i, c in enumerate(previous["cells"]) if owned(c) and c.get("id") != PREFIX+"progress"), len(history))
    section = history[:position]+cells[1:]+history[position:]
    book = deepcopy(previous)
    book["cells"] = section[:1]+[cells[0]]+section[1:]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    payload = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False)+"\n").encode()
    if len(payload) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    if supplement is not None and not target.exists():
        with store.reserve(len(supplement)):
            _atomic(target, supplement)
    with store.reserve(len(payload)) if store is not None else nullcontext():
        _atomic(notebook, payload)
