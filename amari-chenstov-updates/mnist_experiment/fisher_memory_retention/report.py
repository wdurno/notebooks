"""Small static Phase 8.1 report; historical notebook cells remain unchanged."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

from mnist_experiment.fisher_memory_regression.report import (
    _figure, _image, _style, LABELS, COLORS, MAX_NOTEBOOK_BYTES,
)

TAG="plan15_phase81"
PREFIX="plan15-retention-"


def cell(name,text):
    return nbformat.v4.new_markdown_cell(text,id=PREFIX+name,metadata={TAG:True})


def owned(item):
    return TAG in item.get("metadata",{}) or item.get("id","").startswith(PREFIX)


def image(name,figure,caption):
    item=_image(name,figure,caption)
    item["id"]=PREFIX+name
    item["metadata"]={TAG:True}
    return item


def effects_text(effect):
    return f"{effect['mean']:.3g} [{effect['ci'][0]:.3g}, {effect['ci'][1]:.3g}]"


def diagnostic_html(snapshot,records):
    """Descriptive development contrasts, never additional confirmation tests."""
    analysis=snapshot["analysis"]
    count=snapshot["config"]["calibration_replicas"]
    common=[r for r in records if r["replica"]<count]
    controls={(r["replica"],r["recipe"]["ridge"],r["recipe"]["replay_count"]):r
              for r in common if r["method"]=="replay_ewc"}
    groups={}
    for row in common:
        if row["method"]=="fd_replay_ewc":
            groups.setdefault(row["recipe_key"],[]).append(row)
    def table(headers,rows):
        def cells(tag,values):
            return "<tr>"+"".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)+"</tr>"
        return "<table><thead>"+cells("th",headers)+"</thead><tbody>"+"".join(cells("td",r) for r in rows)+"</tbody></table>"
    contrasts=[]
    for _,group in sorted(groups.items()):
        if {r["replica"] for r in group}!=set(range(count)): continue
        pairs=[]
        for fd in group:
            control=controls.get((fd["replica"],fd["recipe"]["ridge"],fd["recipe"]["replay_count"]))
            if control is None: break
            if control["pairing_id"]!=fd["pairing_id"]: raise ValueError("unpaired development diagnostic")
            pairs.append((fd,control))
        if len(pairs)!=count: continue
        recipe=group[0]["recipe"]
        gains=[control["novel_auc"]-fd["novel_auc"] for fd,control in pairs]
        harms=[math.sqrt(fd["base_auc"])-math.sqrt(control["base_auc"]) for fd,control in pairs]
        contrasts.append((recipe["capacity"],recipe["replay_count"],f"{recipe['ridge']:.6g}",
            f"{recipe['strength']:.6g}",f"{np.mean(gains):.3f}",f"{np.mean(harms):.4f}",count))
    text=("<!doctype html><html lang='en'><meta charset='utf-8'><title>Phase 8.1 Diagnostics</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}"
        "table{border-collapse:collapse;font-size:14px}th,td{padding:5px 10px;border-bottom:1px solid #ddd;text-align:right}"
        "th:first-child,td:first-child{text-align:left}summary{cursor:pointer}code{overflow-wrap:anywhere}</style>"
        "<h1>Phase 8.1 Diagnostics</h1><p>Artifact-only descriptive supplement. No training, new policy selection "
        "or additional confirmatory tests. The main notebook contains the fresh confirmation results.</p>"
        "<h2>Same-allocation Development Contrasts</h2><p>FD versus zero-FD replay/EWC at exactly the same "
        "capacity C, replay count b and ridge c, on the common first four paired development replicas. "
        "Positive acquisition gain favors FD; positive H difference means more forgetting with FD. "
        "Only measured complete pairs are shown, without interpolation or significance claims. These are tuning data, "
        "not fresh confirmation or matched-retention comparisons.</p>")
    text+=table(("C","b","c","FD strength","Replay A - FD A","FD H - Replay H","Pairs"),contrasts)
    cost=analysis.get("calibration_cost",{})
    text+="<h2>Calibration Cost</h2>"+table(("Quantity","Value"),[
        ("Development trajectories",cost.get("development_trajectories")),
        ("Novel training presentations across trials",cost.get("novel_training_presentations")),
        ("Validation source images per domain",cost.get("validation_unique_source_images_per_domain")),
        ("Validation image-angle pairs per domain",cost.get("validation_image_angle_pairs_per_domain")),
        ("Repeated validation predictions",cost.get("validation_prediction_presentations"))])
    text+=("<p>Presentations across settings are repeated use of development data, not independent new labels. "
           "The 100-novel-image claim concerns online learning after calibration, excluding shared pretraining.</p>"
           "<h2>Retired Training Cohorts</h2><p>Errors on previously trained examples after retirement. "
           "Counts pool repeated audit measurements with method-specific availability. These are not unique images, "
           "independent samples or the common held-out retention endpoint H.</p>")
    text+=table(("Method","Audit measurements","Mean squared error"),[
        (LABELS[m],v["retirement_audit"]["count"],v["retirement_audit"]["mean_squared_error"])
        for m,v in analysis["methods"].items()])
    text+="<details><summary>All individual development trajectories</summary>"+table(
        ("Method","Replica","C","b","c","FD strength","A(100)","H","Learning seconds"),[
            (LABELS[r["method"]],r["replica"],r["recipe"]["capacity"],r["recipe"]["replay_count"],
             f"{r['recipe']['ridge']:.6g}",f"{r['recipe']['strength']:.6g}",f"{r['novel_auc']:.3f}",
             f"{math.sqrt(r['base_auc'])-math.sqrt(r['frozen_base_mse']):.4f}",f"{r['learning_seconds']:.3f}")
            for r in sorted(records,key=lambda r:(r["method"],r["recipe_key"],r["replica"]))])+"</details>"
    text+=("<h2>Provenance And Raw Data</h2><p>The authoritative sampling seeds, source IDs and angles are in "
        "each sealed assets-*/plan.json. The confirmation manifest's stream_ids/stream_novel seed aliases are not "
        "driving RNG names. In raw interval objects, favorable_replicas counts values above zero; this indicates "
        "favorability only for signed gains, not for raw errors or H.</p><p>Frozen confirmation manifest hash: <code>"
        +html.escape(analysis["manifest_hash"])+"</code>. "
        "<a href='../units/decision-analysis/result.json'>Sealed machine-readable analysis</a>.</p></html>")
    return text


def frontier(rows,cfg):
    figure=_figure(11,4.6)
    axes=figure.subplots(1,2)
    for axis in axes:
        axis.axvspan(cfg.retention_margin_degrees-cfg.matching_tolerance,cfg.retention_margin_degrees,
                    color="#ced9c9",alpha=.55,label="Development matching band")
        axis.axvline(cfg.retention_margin_degrees,color="#555",linestyle="--",linewidth=1)
        for method in LABELS:
            group=[r for r in rows if r["method"]==method]
            for count in sorted({r["count"] for r in group}):
                points=[r for r in group if r["count"]==count]
                for b in sorted({r["replay_count"] for r in points}):
                    subset=[r for r in points if r["replay_count"]==b]
                    for eligible in (True,False):
                        selected=[r for r in subset if r["resource_feasible"]==eligible]
                        if selected:
                            axis.scatter([r["base_harm"] for r in selected],[r["novel_auc"] for r in selected],
                                c=COLORS[method],marker=("o" if b in (0,4,1) else "s") if eligible else "x",
                                s=30,alpha=.7,label=f"{LABELS[method]}, b={b}, r={count}"+("; resource limit" if not eligible else ""))
                best=float("inf"); boundary=[]
                for r in sorted([r for r in points if r["resource_feasible"]],key=lambda r:(r["base_harm"],r["novel_auc"])):
                    if r["novel_auc"]<best:
                        boundary.append(r);best=r["novel_auc"]
                if boundary:
                    axis.plot([r["base_harm"] for r in boundary],[r["novel_auc"] for r in boundary],
                              color=COLORS[method],linewidth=1,alpha=.6)
        axis.set(xlabel="Mean old-domain deterioration H (degrees)",
                 ylabel="Mean new-domain A(100) (degrees squared)")
        _style(axis)
    axes[0].set_title("Development: full measured range")
    axes[1].set_title("Same points near the retention target")
    low=cfg.retention_margin_degrees-.5; high=cfg.retention_margin_degrees+.5
    axes[1].set_xlim(low,high)
    nearby=[r["novel_auc"] for r in rows if low<=r["base_harm"]<=high]
    if nearby:
        padding=max(2.,.1*(max(nearby)-min(nearby)))
        axes[1].set_ylim(min(nearby)-padding,max(nearby)+padding)
    handles,labels=axes[0].get_legend_handles_labels()
    if handles: axes[0].legend(handles,labels,fontsize=6,loc="best")
    return figure


def build_cells(snapshot):
    cfg=snapshot["config"]
    progress=snapshot["progress"]
    seconds=progress.get("active_seconds",0)
    forecast=progress.get("forecast_seconds")
    forecast="pending" if forecast is None else f"{forecast/3600:.2f} h"
    summary=(f"**Phase 8.1 progress:** {progress['status']} / {progress['stage']} | "
        f"{progress.get('completed',0)}/{progress.get('planned',0)} trajectories | "
        f"{seconds/3600:.2f}/20 active h | remaining forecast: {forecast}. "
        f"Heartbeat: {progress.get('heartbeat','pending')}. "
        f"Storage: {(progress.get('storage',{}).get('used_bytes',0)+progress.get('storage',{}).get('external_bytes',0))/1e9:.3g}/500 GB.\n\n"
        "[Initial Phase 8 report](fisher_memory_phase8_initial_results.ipynb). "
        "Partial results are descriptive; final inference requires the frozen complete cohort.")
    cells=[cell("progress",summary),cell("intro",
        "## Phase 8.1: Learning At Comparable Retention Cost\n\n"
        "**Motivation:** initial regression policies learned and forgot by different amounts. "
        "This follow-up independently calibrates each method to ask how much learning is available "
        "at comparable forgetting cost, under limited observations and compute.\n\n"
        r"$R^D(n)$ is predictive MSE on a fixed held-out image/angle panel after exactly $n$ unique novel training images. "
        r"$A^D(100)=100^{-1}\sum_{n=1}^{100}R^D(n)$ averages those sample budgets, excluding initialization. "
        r"$H=\sqrt{A^0(100)}-\sqrt{R^0(0)}$ measures old-domain deterioration within each replica before averaging. "
        "Lower is better for both A and H; predictive errors never include regularizers.\n\n"
        f"The benchmark-informed ceiling is **{cfg['retention_margin_degrees']:.4f} degrees**, "
        f"with a **{cfg['matching_tolerance']:.2f}-degree** matching tolerance. "
        "A shared ceiling is not demonstrated retention equivalence. All arms use the same paired initialization, "
        "stream and held-out panels. Development validation panels guide tuning but never train; separate "
        "confirmation panels never select settings or provide replay, FD or Fisher inputs. "
        "This is a reused MNIST benchmark, not untouched external validation.\n\n"
        "The 100-novel-observation budget describes online learning **after calibration**. "
        "Tuning labels, validation panels, background arrivals and shared pretraining are reported separately. "
        "No online retention controller, confirmation-based retuning or fitted samples-saved claim is used.")]
    rows=snapshot.get("development",[])
    if rows:
        from types import SimpleNamespace
        cells.append(image("frontier",frontier(rows,SimpleNamespace(**cfg)),
            "Development only; lower and left is better. Circles/squares distinguish replay sample counts; "
            "crosses mark resource-infeasible policies. Lines join only measured nondominated settings on equal replica sets; "
            "they do not guarantee attainable interpolated policies. Left: every measured setting; right: the same data "
            "zoomed near the target, with separate axis limits. r is the number of complete replicas."))
    else:
        cells.append(cell("frontier-pending","### Acquisition And Retention\n\nPending complete development pairs."))
    selection=snapshot.get("selection")
    if selection:
        text=("### Frozen Policy Choices\n\nFinal choices use "
            f"{cfg['calibration_replicas']+cfg['refinement_replicas']} development replicas; "
            f"the trade-off map retains the common first {cfg['calibration_replicas']} for comparable coverage.\n\n"
            "| Method | Ridge c | FD strength | C, b | Development H | Status |\n| --- | ---: | ---: | --- | ---: | --- |\n")
        for method,item in selection.items():
            row=item["selected"]; r=row["recipe"]
            text+=f"| {LABELS[method]} | {r['ridge']:.4g} | {r['strength']:.4g} | {r['capacity']}, {r['replay_count']} | {row['base_harm']:.3f} | {item['reason']} |\n"
        cells.append(cell("selection",text))
    analysis=snapshot.get("analysis")
    if analysis:
        figure=_figure(10,4); axes=figure.subplots(1,2)
        for method,item in analysis["methods"].items():
            curves=item["curves"];n=curves["n"]
            for domain,axis in zip(("novel","base"),axes):
                values=curves[domain]
                axis.plot(n,values["mean_rmse"],color=COLORS[method],label=LABELS[method])
                low,high=np.sqrt(values["ci_mse"])
                axis.fill_between(n,low,high,color=COLORS[method],alpha=.12)
                axis.set(xlabel="Unique novel training images",ylabel="sqrt(mean test MSE) (degrees)",
                         title="New-domain prediction" if domain=="novel" else "Old-domain prediction")
                _style(axis)
        for domain,axis in zip(("novel","base"),axes):
            value=next(iter(analysis["methods"].values()))["curves"][domain]["mean_rmse"][0]
            axis.axhline(value,color="#555",linestyle="--",label="Frozen initial model")
            axis.legend(fontsize=7)
        cells.append(image("confirmation",figure,
            f"{len(analysis['replicas'])} complete paired confirmation replicas. Pointwise nominal 95% whole-replica bootstrap bands. "
            f"The 100 novel images accompany {next(iter(analysis['methods'].values()))['curves']['observations'][-1]:.1f} "
            "mean total arriving images, including the background stream but excluding pretraining and calibration. "
            "Held-out images and applied angles are fixed within each replica; evaluation never changes the model."))
        figure=_figure(10,4);axes=figure.subplots(1,2)
        names=[];gains=[];harms=[]
        for item in analysis["comparisons"].values():
            names.append(LABELS[item["first"]]+" vs "+LABELS[item["second"]])
            gains.append(item["acquisition_gain"]);harms.append(item["retention_difference"])
        for axis,items,title in zip(axes,(gains,harms),("Acquisition gain: positive favors first","Retention difference: negative favors first")):
            means=np.array([v["mean"] for v in items]);lo=np.array([v["ci"][0] for v in items]);hi=np.array([v["ci"][1] for v in items])
            axis.errorbar(means,np.arange(len(items)),xerr=np.array([means-lo,hi-means]),fmt="o",color="#333",capsize=3)
            axis.axvline(0,color="#777",linewidth=1)
            axis.set(yticks=np.arange(len(items)),yticklabels=names,title=title,
                xlabel="Budget-averaged MSE gain (degrees squared)" if items is gains else "Mean H difference (degrees)")
            axis.tick_params(axis="y",labelsize=7);_style(axis)
        axes[1].axvspan(-analysis["epsilon"],analysis["epsilon"],color="#ced9c9",alpha=.5)
        cells.append(image("effects",figure,"Nominal paired 95% intervals. Retention equivalence requires the full difference interval "
            "inside the shaded tolerance band; nonsignificance is not equivalence. Secondary comparisons are not familywise-adjusted."))
        figure=_figure(10,4);axes=figure.subplots(1,2)
        for method,item in analysis["methods"].items():
            curves=item["curves"];n=np.array(curves["n"])[1:]
            values=np.array(curves["novel"]["mean_mse"])[1:]
            axes[0].plot(n,np.cumsum(values)/n,color=COLORS[method],label=LABELS[method])
            axes[1].plot(curves["n"],curves["learning_seconds"],color=COLORS[method],label=LABELS[method])
        for axis in axes: axis.set(xlabel="Unique novel training images");axis.legend(fontsize=7);_style(axis)
        axes[0].set(ylabel="Running new-domain A(n) (degrees squared)",title="Budget-averaged sample efficiency")
        axes[1].set(ylabel="Mean learning seconds",title="Measured compute, including retirement")
        cells.append(image("efficiency",figure,"Running A(n) averages held-out new-domain MSE after the first n novel observations. "
            "More novel observations also bring more background observations. Curves do not directly establish actual samples saved."))
        text="### Conclusion\n\n| Method | New-domain A(100) [95% CI] | Old-domain H [95% CI] | Ceiling | Resources | Learning s/arrival |\n| --- | ---: | ---: | --- | --- | ---: |\n"
        for method,item in analysis["methods"].items():
            text+=f"| {LABELS[method]} | {effects_text(item['acquisition'])} | {effects_text(item['retention'])} | "+(
                "supported" if item["retention"]["ceiling_supported"] else "not established")+(
                " | feasible" if item["costs"]["resource_feasible"] else " | infeasible")+f" | {item['costs']['seconds_per_arrival']:.4f} |\n"
        primary=analysis["comparisons"]["fd_replay_ewc_vs_replay_ewc"]
        effect=primary["acquisition_gain"]
        direction="FD" if effect["ci"][0]>0 else "replay/EWC" if effect["ci"][1]<0 else "neither method"
        text+=f"\n**Primary acquisition:** {effects_text(effect)} square degrees; the nominal interval supports {direction}. "
        text+=("**Retention equivalence is supported** for that comparison. " if primary["retention_equivalence_supported"] else
               "**Retention equivalence is not established** for that comparison; do not call this a matched-retention efficiency result. ")
        text+="Ceiling support uses the one-sided 95% upper bound, independently of equivalence. A practical conclusion also requires measured resource feasibility.\n\n"+analysis["assumptions"]
        fd=analysis["methods"]["fd_replay_ewc"];replay=analysis["methods"]["replay_ewc"]
        if effect["ci"][0]>0 and primary["retention_equivalence_supported"]:
            relative=100*effect["mean"]/replay["acquisition"]["mean"]
            text+=(f"\n\n**Budget-averaged sample efficiency:** the selected FD policy has {relative:.2f}% lower "
                f"new-domain error over the first {analysis['N']} novel observations, at retention equivalent "
                f"within {analysis['epsilon']:.2f} degrees. This is an advantage after calibration, not an estimate "
                "of actual samples saved.")
        ratio=fd["costs"]["mean_learning_seconds"]/replay["costs"]["mean_learning_seconds"]
        text+=(f"\n\n**Compute trade-off:** mean learning time is {fd['costs']['mean_learning_seconds']:.2f} s for FD "
            f"versus {replay['costs']['mean_learning_seconds']:.2f} s for replay/EWC ({ratio:.2f} times as much). "
            "This compares separately calibrated policies, including different replay counts and ridge strengths; "
            "it does not isolate the FD penalty or demonstrate an equal-compute advantage.")
        cost=analysis.get("calibration_cost")
        if cost:
            text+=(f"\n\n**Separate calibration cost:** {cost['development_trajectories']} development trajectories, "
                f"{cost['novel_training_presentations']} novel training presentations across those trials, "
                f"{cost['validation_images_per_replica_per_domain']} validation images per replica/domain, "
                f"and {cost['development_active_seconds']/3600:.2f} active trajectory hours (excluding shared pretraining). "
                "Repeated presentations across settings are not independent new observations. Unique validation counts "
                "and retirement-cohort diagnostics remain in the linked analysis artifact.")
        href=snapshot.get("diagnostics_href",snapshot.get("analysis_href"))
        if href:
            text+=f"\n\n[Calibration, same-allocation contrasts and retired-cohort diagnostics]({href})."
        cells.append(cell("conclusion",text))
    else:
        confirmation=snapshot.get("confirmation",[])
        by_replica={}
        for row in confirmation:
            by_replica.setdefault(row["replica"],{})[row["method"]]=row
        paired=[arms for arms in by_replica.values() if set(arms)==set(LABELS)]
        if paired:
            figure=_figure(10,4);axes=figure.subplots(1,2)
            for method in LABELS:
                for domain,axis in zip(("novel","base"),axes):
                    matrix=np.array([[m[f"{domain}_mse"] for m in arms[method]["metrics"]] for arms in paired])
                    axis.plot(range(matrix.shape[1]),np.sqrt(matrix.mean(0)),color=COLORS[method],label=LABELS[method])
                    axis.set(xlabel="Unique novel training images",ylabel="sqrt(mean test MSE) (degrees)",
                        title="New-domain prediction" if domain=="novel" else "Old-domain prediction")
                    _style(axis);axis.legend(fontsize=7)
            cells.append(image("confirmation-partial",figure,
                f"Descriptive progress only: {len(paired)} complete paired replicas so far; no confidence or significance claim. "
                "Every planned replica remains in the final cohort; final inference waits for its completion."))
        else:
            cells.append(cell("confirmation-pending","### Fresh Confirmation\n\nPending the frozen complete cohort. No final inference is published from partial runs."))
    changes=snapshot.get("changes",[])
    if changes:
        cells.append(cell("changes","**Significant changes:** "+" ".join(str(v) for v in changes[-3:])+" Detailed decisions remain in the immutable run artifacts."))
    return cells


def publish(snapshot,notebook,store):
    notebook=Path(notebook)
    previous=json.loads(notebook.read_bytes()) if notebook.exists() else dict(nbformat.v4.new_notebook())
    if previous.get("metadata",{}).get("plan15_phase8_presentation",{}).get("role")=="frozen_archive":
        raise ValueError("cannot update the frozen Phase 8 archive")
    if snapshot.get("analysis"):
        records=[]
        for unit in store.iter_units(completed_only=True):
            store.check()
            meta=unit["metadata"]
            if meta.get("kind")=="regression_trajectory" and meta.get("partition")=="development":
                records.append(json.loads((store.unit_path(unit["unit_id"])/"summary.json").read_bytes()))
        content=diagnostic_html(snapshot,records).encode()
        path=store.root/"reports"/f"phase81-diagnostics-{hashlib.sha256(content).hexdigest()}.html"
        if path.exists():
            if path.read_bytes()!=content: raise ValueError("content-addressed report collision")
        else:
            with store.reserve(len(content)):
                path.parent.mkdir(parents=True,exist_ok=True)
                fd,tmp=tempfile.mkstemp(prefix=".tmp-diagnostics-",dir=path.parent)
                try:
                    with os.fdopen(fd,"wb") as handle:
                        handle.write(content);handle.flush();os.fsync(handle.fileno())
                    store.check();os.replace(tmp,path)
                finally: Path(tmp).unlink(missing_ok=True)
        snapshot={**snapshot,"diagnostics_href":os.path.relpath(path,notebook.parent)}
    followup=[c for c in previous["cells"] if c.get("metadata",{}).get("plan15_phase81e")]
    historical=[c for c in previous["cells"] if not owned(c) and c not in followup]
    new=build_cells(snapshot)
    book=deepcopy(previous);book["cells"]=historical[:1]+[new[0]]+historical[1:]+new[1:]+followup
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    content=(json.dumps(book,indent=1,ensure_ascii=False,allow_nan=False)+"\n").encode()
    if len(content)>MAX_NOTEBOOK_BYTES: raise ValueError("bounded notebook size exceeded")
    with store.reserve(len(content)):
        store.check();notebook.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix=".tmp-retention-",dir=notebook.parent)
        try:
            with os.fdopen(fd,"wb") as handle:
                handle.write(content);handle.flush();os.fsync(handle.fileno())
            store.check();os.replace(tmp,notebook)
        finally: Path(tmp).unlink(missing_ok=True)
