"""Concise artifact-only Phase 9 notebook publication."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-shift-matplotlib")

from mnist_experiment.fisher_memory_regression.report import (
    _figure, _image, _style, COLORS, LABELS, MAX_NOTEBOOK_BYTES,
)

TAG, PREFIX = "plan15_phase9", "plan15-phase9-"


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def cell(name, text):
    return nbformat.v4.new_markdown_cell(text, id=PREFIX + name, metadata={TAG: True})


def picture(name, figure, caption):
    result = _image("phase9-" + name, figure, caption)
    result["id"], result["metadata"] = PREFIX + name, {TAG: True}
    return result


def interval(value, digits=2):
    return f"{value['mean']:.{digits}f} [{value['ci'][0]:.{digits}f}, {value['ci'][1]:.{digits}f}]"


def paired_records(snapshot):
    arms = {}
    for row in snapshot.get("confirmation", []):
        arms.setdefault(row["replica"], {})[row["method"]] = row
    return [value for value in arms.values() if set(value) == set(LABELS)]


def curve_figure(analysis=None, pairs=None, *, applied=False):
    figure = _figure(10, 4)
    axes = figure.subplots(1, 2)
    for method in LABELS:
        if analysis:
            curves = analysis["methods"][method]["curves"]
            n = curves["n"]
        else:
            rows = [p[method]["metrics"] for p in pairs]
            n = [row["n"] for row in rows[0]]
        for index, axis in enumerate(axes):
            domain = "novel" if applied or index == 0 else "base"
            metric = ("mae" if index == 0 else "accuracy10") if applied else "rmse"
            if analysis:
                values = np.asarray(curves[domain]["mean_" + metric])
            else:
                key = domain + ("_mse" if metric == "rmse" else "_" + metric)
                values = np.asarray([[r[key] for r in row] for row in rows]).mean(0)
                if metric == "rmse":
                    values = np.sqrt(values)
            if metric == "accuracy10":
                values = 100 * values
            axis.plot(n, values, label=LABELS[method], color=COLORS[method])
            if analysis and not applied:
                lo, hi = curves[domain]["ci_rmse"]
                axis.fill_between(n, lo, hi, color=COLORS[method], alpha=.12)
    for index, axis in enumerate(axes):
        domain = "novel" if applied or index == 0 else "base"
        if analysis:
            curves = next(iter(analysis["methods"].values()))["curves"]
            metric = ("mae" if index == 0 else "accuracy10") if applied else "rmse"
            frozen = curves[domain]["mean_" + metric][0] * (100 if metric == "accuracy10" else 1)
        else:
            key = domain + "_mse"
            frozen = np.sqrt(np.mean([p[next(iter(LABELS))]["metrics"][0][key] for p in pairs]))
        axis.axhline(frozen, color="#555", linestyle="--", label="Frozen initial model")
        axis.set(xlabel="Unique novel training images",
            ylabel=("Mean absolute angle error (degrees)" if index == 0 else "Within 10 degrees (%)")
                   if applied else "sqrt(mean held-out MSE) (degrees)",
            title=("New-domain angular error" if index == 0 else "New-domain useful accuracy")
                  if applied else ("New measurement conditions" if index == 0 else "Nominal conditions"))
        axis.legend(fontsize=7)
        _style(axis)
    return figure


def build_cells(snapshot, diagnostics_href=None):
    progress, cfg = snapshot["progress"], snapshot["config"]
    forecast = progress.get("forecast_seconds")
    estimate = "pending" if forecast is None else f"{forecast/3600:.2f} h"
    storage = progress.get("storage", {})
    total_bytes = storage.get("used_bytes", 0) + storage.get("external_bytes", 0)
    count = (f"{progress.get('completed',0)}/{progress['planned']} currently scheduled trajectories"
             if progress.get("planned",0) else
             f"{progress.get('completed',0)} completed calibration trajectories")
    out = [cell("progress",
        f"**Phase 9 progress:** {progress['status']} / {progress['stage']} | "
        f"{count} | "
        f"{progress.get('active_seconds',0)/3600:.2f}/20 active h | scheduled-work estimate: {estimate}. "
        f"Storage: {total_bytes/1e9:.2f}/500 GB. Heartbeat: {progress.get('heartbeat','pending')}. "
        "Check heartbeat freshness after interruptions. Partial results describe progress; inference waits for the frozen cohort."),
        cell("intro", "## Phase 9: Continuous Measurement Changes\n\n"
        "**Motivation:** can a model that already predicts angle learn a coherent measurement change "
        "from fewer new examples, at an acceptable compute and retention cost?\n\n"
        "All ten digits use angles sampled continuously from **-60 to 60 degrees**, including pretraining. "
        "Gamma changes intensity response; Gaussian blur changes spatial resolution. Their strengths change smoothly "
        "while novel observations drip into the background stream. Replay capacity stays fixed at 256.\n\n"
        "Prediction is measured on fixed, independent held-out image/angle panels. The nominal panel uses ordinary "
        "rendering; the new-domain panel covers a fixed range of transformed conditions. Neither panel trains "
        "the model or supplies Fisher/FD inputs. Errors exclude all regularizers.\n\n"
        f"The nominal deterioration ceiling is {cfg['retention_margin_degrees']:g} degree, "
        f"with {cfg['matching_tolerance']:g}-degree comparison tolerance. "
        r"$S=\text{fraction of novel samples saved}/\text{learning-compute multiplier}$. "
        "For example, **50% fewer novel examples for twice the compute gives S = 0.25**. "
        "Both components refer to the same observed prediction quality; comparable nominal retention is checked "
        "at the full and matched budgets. Novel count also changes background exposure and drift history. "
        "This is a marginal online comparison after separately reported pretraining and calibration.")]
    initialization = snapshot.get("initialization")
    readiness = snapshot.get("readiness")
    if initialization:
        checks = initialization["checks"]
        figure = _figure(10, 3.8)
        axes = figure.subplots(1, 2)
        reps = [r["replica"] for r in checks]
        axes[0].plot(reps, [np.sqrt(r["zero_predictor_mse"]) for r in checks], "o--",
                     label="Constant-zero prediction", color="#777")
        axes[0].plot(reps, [np.sqrt(r["mse"]) for r in checks], "o",
                     label="Pretrained model", color="#168477")
        axes[0].set(xlabel="Development replica", ylabel="Nominal validation RMSE (degrees)",
                    xticks=reps, title=f"Initialization: {initialization['epochs']} epochs")
        axes[0].legend(fontsize=8)
        groups = np.arange(1, 321)
        v = np.minimum((groups-1)/199, 1)
        u = .5+.5*(3*v*v-2*v*v*v)
        for context in snapshot["contexts"]:
            if context["gamma_max"] != 1:
                axes[1].plot(groups, context["gamma_max"]**u, label=context["key"]+" gamma")
            if context["blur_max"] != 0:
                axes[1].plot(groups, context["blur_max"]*u, linestyle="--",
                             label=context["key"]+" blur (pixels)")
        axes[1].set(xlabel="Original four-image arrival group",
                    ylabel="Gamma exponent / blur sigma (pixels)", title="Continuous rendering schedule")
        axes[1].legend(fontsize=6)
        for axis in axes:
            _style(axis)
        out.append(picture("initialization", figure,
            "Initialization is shared across methods and contexts. Nominal validation is independent of training. "
            "The schedules are smooth functions sampled at the original arrival groups; splitting groups does not reset them."))
        out.append(cell("capability",
            f"Nominal mean-MSE reduction from zero prediction: **{100*(1-initialization['mse_ratio']):.1f}%**. "
            + ("The capability preference was met." if initialization["capability_supported"] else
               "**Capability remains limited:** the best common valid recipe is retained, with this warning.")
            + (f" Context readiness covered {len(readiness['readiness'])} measured renderings; diagnostics are linked below."
               if readiness else "")))
    if readiness:
        previews = [(c["key"], readiness["readiness"][c["key"]][0].get("preview"))
                    for c in snapshot["contexts"] if c["key"] in readiness["readiness"]]
        previews = [(key, value) for key, value in previews if value]
        if previews:
            figure = _figure(2.3*len(previews), 2.5)
            axes = np.asarray(figure.subplots(1, len(previews), squeeze=False)).ravel()
            for axis, (key, preview) in zip(axes, previews):
                original = np.concatenate(np.asarray(preview["nominal"]), axis=1)
                changed = np.concatenate(np.asarray(preview["transformed"]), axis=1)
                axis.imshow(np.concatenate((original, changed), axis=0), cmap="gray", vmin=0, vmax=255)
                axis.set(title=key)
                axis.set_axis_off()
            out.append(picture("renderings", figure,
                "The same three development-training images and angles in nominal rendering (top) "
                "and changed rendering (bottom). These illustrative images are not evaluation evidence."))
    development = snapshot.get("development", {})
    if any(development.values()):
        contexts = [key for key, rows in development.items() if rows]
        figure = _figure(11, max(3.5, 3.1*((len(contexts)+1)//2)))
        axes = np.asarray(figure.subplots((len(contexts)+1)//2, 2, squeeze=False)).ravel()
        for axis, key in zip(axes, contexts):
            for method in LABELS:
                rows = [r for r in development[key] if r["method"] == method]
                for valid in (True, False):
                    chosen = [r for r in rows if r["resource_feasible"] == valid]
                    axis.scatter([r["base_harm"] for r in chosen], [r["novel_auc"] for r in chosen],
                                 label=LABELS[method]+("" if valid else " (partial / resource limit)"),
                                 color=COLORS[method], marker="o" if valid else "x", s=20)
            delta, eps = cfg["retention_margin_degrees"], cfg["matching_tolerance"]
            axis.axvspan(delta-eps, delta, color="#b8c7b2", alpha=.22)
            axis.axvline(delta, linestyle="--", color="#666", linewidth=1)
            axis.set(title=key, xlabel="Old-domain deterioration H (degrees)",
                     ylabel="New-domain mean MSE (deg$^2$)")
            _style(axis)
        axes[0].legend(fontsize=6)
        for axis in axes[len(contexts):]:
            axis.set_visible(False)
        out.append(picture("development", figure,
            "Development only: each point is a measured policy, with its available replica count in the supplement. "
            "Lower and left is better. Shading is the preferred retention band, not confidence. "
            "Crosses mark incomplete or resource-limited points; incomplete pairs cannot select policies. "
            "All tested contexts remain visible; selected-context confirmation is separate."))
    selected = snapshot.get("selection")
    if selected:
        text = f"### Frozen Choices\n\nSelected context: **{selected['context']['key']}**.\n\n"
        text += "| Method | Replay count | Ridge | FD strength |\n| --- | ---: | ---: | ---: |\n"
        for method, item in selected["policies"].items():
            r = item["selected"]["recipe"]
            text += f"| {LABELS[method]} | {r['replay_count']} | {r['ridge']:.3g} | {r['strength']:.3g} |\n"
        warnings = [w.replace("_", " ") for w in selected["score"].get("warnings", [])]
        if warnings:
            text += "\nDevelopment limitations: " + "; ".join(warnings) + "."
        out.append(cell("selection", text))
    analysis = snapshot.get("analysis")
    pairs = paired_records(snapshot)
    if analysis or pairs:
        out.append(picture("confirmation", curve_figure(analysis, pairs),
            (f"{len(analysis['replicas'])} complete fresh paired replicas; pointwise nominal 95% whole-replica bootstrap bands."
             if analysis else f"Descriptive progress: {len(pairs)} complete fresh paired replicas. No interim significance claim.")
            + " Dashed lines identify frozen initial performance on the same held-out panels. "
            "Bands show marginal variation across replicas; method comparisons use paired intervals."))
    if analysis:
        efficiency = analysis["efficiency"]
        point, bootstrap = efficiency["point"], efficiency["bootstrap"]
        figure = _figure(11, 3.8)
        axes = figure.subplots(1, 3)
        for method in ("fd_replay_ewc", "replay_ewc"):
            curves = analysis["methods"][method]["curves"]
            axes[0].plot(curves["n"], curves["novel"]["mean_rmse"], label=LABELS[method],
                         color=COLORS[method])
            axes[1].plot(curves["n"], curves["learning_seconds"], label=LABELS[method],
                         color=COLORS[method])
        target = np.sqrt(point["target_mse"])
        axes[0].axhline(target, linestyle="--", color="#555", label="Replay at 100")
        if point["status"] == "valid":
            n = point["matching_budget"]
            axes[0].scatter([n], [target], color=COLORS["fd_replay_ewc"], zorder=4)
            axes[0].axvline(n, linestyle=":", color=COLORS["fd_replay_ewc"])
            axes[1].scatter([n,100], [point["fd_learning_seconds"], point["replay_learning_seconds"]],
                            c=[COLORS["fd_replay_ewc"], COLORS["replay_ewc"]])
            axes[2].bar(["FD match", "Replay 100"], [point["fd_old_rmse"], point["replay_old_rmse"]],
                        color=[COLORS["fd_replay_ewc"], COLORS["replay_ewc"]])
        else:
            axes[2].text(.5,.5,"Matched-budget retention\nunresolved",ha="center",va="center",
                         transform=axes[2].transAxes)
            axes[2].set(xticks=[], yticks=[])
        axes[0].set(xlabel="Novel training images", ylabel="New-domain RMSE (degrees)",
                    title="Observed sample equivalence")
        axes[1].set(xlabel="Novel training images", ylabel="Mean learning seconds", title="Actual compute")
        axes[2].set(ylabel="Mean replica old-domain RMSE (degrees)", title="Retention at matched budgets")
        axes[0].legend(fontsize=6)
        for axis in axes:
            _style(axis)
        out.append(picture("efficiency", figure,
            "The target is replay's observed endpoint MSE; the FD crossing uses adjacent MSE segments, "
            "even though the display uses degrees. It must remain below target through the observed endpoint. "
            "Time includes background, replay, FD and retirement. Shared initialization and tuning cost are separate."))
        out.append(picture("applied", curve_figure(analysis, applied=True),
            "Secondary applied metrics on the same fixed new-domain panel. Within-10-degree accuracy is the "
            "fraction of predictions with absolute angular error at most 10 degrees; it is not digit identity accuracy."))
        primary = analysis["G"]
        text = "### Conclusion\n\n"
        if point["status"] == "valid":
            text += (f"**{100*point['saved_fraction']:.1f}% fewer novel examples for "
                     f"{point['learning_time_ratio']:.2f} times the learning compute; S = {point['S']:.3f}.** "
                     f"FD matches replay's 100-image error with {point['matching_budget']:.1f} images. "
                     f"The target RMSE is {target:.2f} degrees.\n\n")
            ci = bootstrap["ci"]
            if ci:
                text += (f"Nominal paired 95% intervals: savings "
                         f"[{100*ci['saved_fraction'][0]:.1f}%, {100*ci['saved_fraction'][1]:.1f}%]; "
                         f"compute multiplier [{ci['learning_time_ratio'][0]:.2f}, {ci['learning_time_ratio'][1]:.2f}]; "
                         f"S [{ci['S'][0]:.3f}, {ci['S'][1]:.3f}].\n\n")
            else:
                text += (f"**Ordinary inversion intervals unavailable:** {bootstrap['invalid']}/{bootstrap['draws']} "
                         "bootstrap draws have unresolved matches. No failed draw was discarded.\n\n")
        else:
            text += f"**Sample equivalence is unresolved:** {point['status'].replace('_',' ')}.\n\n"
            text += (f"{bootstrap['invalid']}/{bootstrap['draws']} bootstrap matches are unresolved; "
                     "ordinary inverse-curve intervals are withheld.\n\n")
        target_gain = point.get("target_mse_gain_fraction")
        if target_gain is not None:
            text += (f"Replay's endpoint MSE improvement over frozen initialization is "
                     f"**{100*target_gain:.2f}%**. ")
            text += ("It passes the prespecified 5% practical-target check.\n\n"
                     if point["useful_target"] else
                     "**It does not pass the prespecified 5% practical-target check:** "
                     "any numerical savings score describes a small prediction change, "
                     "not an established practical-efficiency win.\n\n")
        if efficiency["retention_qualified"]:
            text += "**Comparable retention is supported at both budgets.** "
        elif point["status"] != "valid" and efficiency["full_budget_retention_supported"]:
            text += "**Full-budget retention is supported; matched-budget retention is unresolved.** "
        else:
            text += "**Comparable retention is not established at both budgets; the raw score is retention-unqualified.** "
        text += (f"Budget-averaged new-domain MSE gain, replay minus FD: **{interval(primary)} square degrees**. "
                 f"Bootstrap draws with multiple crossings: {bootstrap['multiple_crossing_draws']}/{bootstrap['draws']}.\n\n")
        if primary["ci"][0] <= 0 <= primary["ci"][1]:
            text += "**An acquisition advantage was not statistically established.** "
        costs = {m: analysis["methods"][m]["costs"]["mean_learning_seconds"]
                 for m in ("fd_replay_ewc", "replay_ewc")}
        text += (f"At the full 100-image budget, FD used "
                 f"**{costs['fd_replay_ewc']/costs['replay_ewc']:.2f} times** replay's learning time "
                 f"({costs['fd_replay_ewc']:.2f} versus {costs['replay_ewc']:.2f} seconds). "
                 "This is not a matched-quality compute ratio.\n\n")
        text += "| Method | New-domain A(100), 95% CI | Old-domain H, 95% CI | Learning seconds |\n"
        text += "| --- | ---: | ---: | ---: |\n"
        for method, item in analysis["methods"].items():
            text += (f"| {LABELS[method]} | {interval(item['acquisition'])} | "
                     f"{interval(item['retention'])} | {item['costs']['mean_learning_seconds']:.2f} |\n")
        text += ("\n$A(100)$ averages new-domain held-out MSE across the 100 sample budgets. "
                 "$H$ is each replica's square root of budget-mean nominal MSE minus its initial RMSE, in degrees.\n\n")
        calibration_cost = analysis.get("calibration_cost")
        if calibration_cost:
            observations = analysis["methods"]["replay_ewc"]["curves"]["observations"][-1]
            text += (f"**Accounting:** each online trajectory uses 100 novel images and {observations:.1f} "
                     f"mean total arrivals. Each initialization uses {cfg['init_size']:,} training images, "
                     f"{cfg['init_validation_size']:,} validation images and {cfg['reference_size']:,} independent "
                     f"Fisher inputs. Readiness fits use {cfg['diagnostic_base_size']} additional nominal and "
                     f"{cfg['diagnostic_novel_size']} transformed training images per development replica, "
                     "reusing their source IDs across contexts. "
                     f"The {calibration_cost['development_trajectories']} development trajectory executions reuse "
                     f"paired streams; they are not distinct label budgets. Total active allocation used: "
                     f"{progress.get('active_seconds',0)/3600:.2f} hours, including acceptance and calibration.\n\n")
        text += ("This result concerns the selected rendering context after calibration on a reused benchmark. "
                 "Different sample budgets include different background exposure and drift history. "
                 "S = 0.25 is an aspiration, not an early-stop threshold. "
                 + analysis["assumptions"])
        out.append(cell("conclusion", text))
    elif not pairs:
        out.append(cell("confirmation-pending", "### Fresh Confirmation\n\nPending calibration and frozen context/policy selection."))
    if snapshot.get("changes"):
        out.append(cell("changes", "**Significant changes:** " + " ".join(snapshot["changes"][-3:])))
    if diagnostics_href:
        out.append(cell("diagnostics", f"[Detailed calibration, numerical and retirement diagnostics]({diagnostics_href})."))
    return out


def heatmap_html(snapshot):
    parts = []
    for key, rows in snapshot.get("development", {}).items():
        rows = [r for r in rows if r["method"] == "fd_replay_ewc" and r.get("complete")]
        if not rows:
            continue
        replay_counts = sorted({r["replay_count"] for r in rows})
        figure = _figure(5*len(replay_counts), 4)
        axes = np.asarray(figure.subplots(1,len(replay_counts),squeeze=False)).ravel()
        low, high = min(r["novel_auc"] for r in rows), max(r["novel_auc"] for r in rows)
        for axis, b in zip(axes,replay_counts):
            points = [r for r in rows if r["replay_count"] == b]
            ridges = sorted({r["recipe"]["ridge"] for r in points})
            strengths = sorted({r["recipe"]["strength"] for r in points})
            values = np.full((len(ridges),len(strengths)),np.nan)
            for row in points:
                i,j = ridges.index(row["recipe"]["ridge"]),strengths.index(row["recipe"]["strength"])
                values[i,j] = row["novel_auc"]
                axis.text(j,i,f"{row['base_harm']:.2f}",ha="center",va="center",fontsize=7,
                          color="#111" if row["novel_auc"] > (low+high)/2 else "white")
            img = axis.imshow(np.ma.masked_invalid(values),origin="lower",aspect="auto",
                              cmap="cividis",vmin=low,vmax=high if high>low else low+1)
            axis.set(xticks=range(len(strengths)),xticklabels=[f"{s:.3g}" for s in strengths],
                     yticks=range(len(ridges)),yticklabels=[f"{c:.3g}" for c in ridges],
                     xlabel="FD strength",ylabel="Ridge c",title=f"{key}: replay count {b}")
            axis.tick_params(axis="x",labelrotation=45,labelsize=7)
        figure.colorbar(img,ax=list(axes),label="New-domain budget-mean MSE (degrees squared)",shrink=.8)
        data=io.BytesIO()
        figure.canvas.draw()
        figure.savefig(data,format="png",dpi=110,bbox_inches="tight")
        figure.clear()
        parts.append("<h2>"+html.escape(key)+"</h2><p>FD policies on complete development pairs. "
                     "Cell color is predictive MSE; labels are nominal deterioration H in degrees. "
                     "Blank cells were not measured. Colors share a scale within this context.</p>"
                     "<img style='max-width:100%' src='data:image/png;base64,"
                     +base64.b64encode(data.getvalue()).decode("ascii")+"'>")
    return "".join(parts)


def publish(snapshot, notebook, store=None):
    notebook = Path(notebook)
    previous = json.loads(notebook.read_bytes()) if notebook.exists() else dict(nbformat.v4.new_notebook())
    if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
        raise ValueError("cannot publish into the frozen Phase 8 notebook")
    href = None
    if store:
        # Detailed values stay outside the main narrative, in a content-addressed supplement.
        text = ("<!doctype html><meta charset='utf-8'><title>Phase 9 diagnostics</title>"
                "<style>body{font:15px sans-serif;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap}</style>"
                "<h1>Phase 9 diagnostics</h1><p>Development selection is exploratory. "
                "Full records and immutable decisions remain under the study root.</p>"
                +heatmap_html(snapshot)+"<details><summary>Recorded calibration and development values</summary><pre>"
                + html.escape(json.dumps({k:v for k,v in snapshot.items() if k not in ("confirmation", "analysis")},
                                         indent=2, allow_nan=False)) + "</pre></details>")
        payload = text.encode()
        name = "phase9-diagnostics-" + hashlib.sha256(payload).hexdigest() + ".html"
        path = store.root / "reports" / name
        if not path.exists():
            with store.reserve(len(payload)):
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic(path, payload)
        href = os.path.relpath(path, notebook.parent)
    new = build_cells(snapshot, href)
    history = [c for c in previous["cells"] if not owned(c)]
    book = deepcopy(previous)
    book["cells"] = history[:1] + [new[0]] + history[1:] + new[1:]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    payload = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False)+"\n").encode()
    if len(payload) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    with store.reserve(len(payload)) if store else nullcontext():
        _atomic(notebook, payload)


def _atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-shift-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
