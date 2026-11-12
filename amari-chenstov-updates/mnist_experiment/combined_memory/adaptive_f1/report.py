"""Concise artifact-only Phase 9E notebook; sealed publication generations."""

import base64
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.combined_memory.reproduction.runtime import atomic_json
from mnist_experiment.combined_memory.replicated_combination.report import _atomic
from . import PROTOCOL
from .config import REPO

TAG = "plan18-phase9e"
LABELS = dict(R="F1 fixed pi", K="F1 Kalman pi")
COLORS = dict(R="#3b718e", K="#bd4770")
SOURCE = Path(__file__).read_bytes()


def owned(cell):
    return cell.get("metadata", {}).get(TAG, False) or cell.get("id", "").startswith(TAG+"-")


def md(name, source):
    return nbformat.v4.new_markdown_cell(source, id=TAG+"-"+name, metadata={TAG: True})


def number(x):
    return "unavailable" if x is None else f"{x:.4g}"


def effect(x):
    if x is None:
        return "unavailable"
    return number(x["mean"])+" ["+", ".join(number(v) for v in x["ci"])+"]"


def table(headers, rows):
    return "\n".join("| "+" | ".join(str(v) for v in r)+" |"
                     for r in (headers, ["---"]*len(headers), *rows))


def figures(snapshot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = {}

    def save(name, fig):
        fig.tight_layout()
        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=140)
        plt.close(fig)
        result[name] = out.getvalue()

    development = snapshot.get("development", {})
    rows = development.get("candidates", [])
    ref = next((r for r in rows if r["policy"] == "R"), None)
    if rows and ref:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
        for ax, field, title in zip(axes, ("Havg", "H100"), ("Average retention", "Endpoint retention")):
            ax.axvspan(ref[field]-.01, ref[field]+.01, color=COLORS["R"], alpha=.12, label="R matching band")
            right_edge = max(row[field] for row in rows)
            for row in rows:
                p = row["policy"]
                ax.scatter(row[field], row["novel_auc"], color=COLORS[p], marker="s" if p=="R" else "o")
                at_right = row[field] == right_edge
                ax.annotate(f"{p}, c={row['ridge']:.3g} ({len(row['members'])}/4)",
                    (row[field], row["novel_auc"]), xytext=(-5 if at_right else 5, 4),
                    textcoords="offset points", ha="right" if at_right else "left", fontsize=8)
            ax.set(title=title, xlabel="Old RMS deterioration (pixels)", ylabel="Novel mean squared error / landmark")
            ax.margins(x=.08, y=.15)
            ax.grid(alpha=.15)
        save("development", fig)
    analysis = snapshot.get("analysis") or {}
    curves = analysis.get("curves") or development.get("curves", {})
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, domain, title in zip(axes, ("query", "base"), ("Novel held-out prediction", "Old held-out prediction")):
            for p in ("R", "K"):
                if p not in curves:
                    continue
                curve = curves[p]
                ax.plot(curve["n"], curve[domain]["mean_mse"], color=COLORS[p], label=LABELS[p])
                if "ci_mse" in curve[domain]:
                    lo, hi = curve[domain]["ci_mse"]
                    ax.fill_between(curve["n"], lo, hi, color=COLORS[p], alpha=.12)
            baseline = curves["R"][domain]["mean_mse"][0]
            ax.axhline(baseline, color="#777777", linestyle="--", linewidth=1, label="Frozen initialization")
            ax.set(title=title, xlabel="New-domain training grids integrated", ylabel="Mean squared error / landmark (px²)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        save("prediction", fig)
    diagnostics = snapshot.get("diagnostics", {})
    if diagnostics.get("observations"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
        x = diagnostics["observations"]
        axes[0].plot(x, diagnostics["applied_pi"], color=COLORS["K"], label="Kalman pi")
        axes[0].axhline(.025, linestyle="--", color=COLORS["R"], label="Fixed pi")
        axes[0].set_ylabel("pi")
        for field, label in (("kalman_level_gain_mean", "Level gain"), ("kalman_velocity_gain_mean", "Velocity gain")):
            if field in diagnostics:
                axes[1].plot(x, diagnostics[field], label=label)
        for field, label in (("motion_energy", "Motion energy"), ("noise_scale", "One-grid noise")):
            if field in diagnostics:
                values = np.asarray(diagnostics[field], dtype=float)
                axes[2].plot(x, np.where(values > 0, values, np.nan), label=label)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Fisher-weighted scale (log)")
        axes[1].set_ylabel("Kalman gain")
        for ax in axes:
            ax.set_xlabel("Incoming training grids")
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        save("pi-motion", fig)
    if analysis.get("efficiency"):
        efficiency = analysis["efficiency"]
        point = efficiency["point"]
        if point["status"] == "valid":
            fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
            for ax, field, title, scale in zip(axes, ("saved_fraction", "learning_time_ratio", "S"),
                    ("Samples saved (%)", "Compute ratio at equal quality", "S = saving / compute ratio"), (100, 1, 1)):
                ax.bar([0], [point[field]*scale], color=COLORS["K"], width=.5)
                ci = efficiency["bootstrap"].get("ci")
                if ci:
                    lo, hi = ci[field]
                    ax.vlines(0, lo*scale, hi*scale, color="#333333")
                ax.set_title(title, fontsize=10)
                ax.set_xticks([0], ["Kalman pi vs fixed F1"])
                ax.axhline(0, color="#888888", linewidth=.7)
            save("efficiency", fig)
    return result


def cells(snapshot, generation, images):
    progress = snapshot["progress"]
    progress_cell = md("progress", "### Phase 9E Progress\n\n"+table(
        ["Stage / status", "K development", "Fresh pairs", "Phase 9 compute", "Fresh + closeout forecast"],
        [[progress["stage"]+" / "+progress["status"], progress["development_completed"],
          f"{progress['fresh_blocks_completed']}/8", f"{progress['charged_hours']:.2f}/20 h",
          f"{progress['forecast_hours']:.2f} h"]])+"\n\nThe original Phase 9 A–E calibration remains closed. This R/K extension has its own retention gate. "
          "The forecast covers remaining fresh trajectories and closeout; development and preparation are additional until calibration finishes.")
    revision_note = ("\n\nImplementation note: scheduler revision 2 preserves numerical code and accepted checkpoints, "
        "and corrects the final ridge repair to use existing brackets without repeating complete candidates."
        if snapshot.get("scheduler_revision") else "")
    output = [md("intro", "## Phase 9E: F1 With Adaptive Pi\n\n"
        "Phase 9's representation comparison favored non-diagonal F1 for both acquisition and retention. "
        "This continuation tests whether velocity-Kalman pi adds value while preserving that STM recipe. "
        "LFU is off. Four fixed-F1 development paths are reused; confirmation uses fresh paired paths.\n\n"+
        table(["Condition", "EWC", "pi", "Ridge c", "STM FD strength"],
              [[LABELS[p], "Rank 64 + diagonal", ".025" if p=="R" else "Velocity Kalman",
                number(1. if p=="R" else snapshot.get("ridge", 1.)), "3.34056"] for p in ("R", "K")])+
        "\n\nBoth use FIFO32/replay2, head-rank-one STM, 100 novel grids and the original continuous blur law. "
        "Scores from an incoming group affect only later pi choices. The observer uses a diagonal approximation; EWC retains off-diagonal structure. "
        "The motion plot shows posterior mean-motion energy and one-grid noise; the pi rule also uses velocity uncertainty, the reference concentration and batch size four."+revision_note),
        md("development", "### Development Retention Match\n\n"+
           snapshot.get("calibration_text", "Calibration is pending. Both four-replica mean retention gaps must be within 0.01 pixels.")+
           "\n\nPrediction errors are unregularized held-out MSE, divided by 27 landmarks. "
           "Retention roots each replica's old-panel mean MSE before averaging replicas; negative deterioration is allowed.")]
    analysis = snapshot.get("analysis")
    if analysis:
        d = analysis["contrast"]
        output.append(md("paired-effects", "### Fresh Paired Effects\n\n"+table(
            ["Kalman minus fixed F1", "Mean [nominal 95% CI]"],
            [[label, effect(d[field])] for field, label in (
                ("acquisition_difference", "Novel budget-averaged MSE"),
                ("endpoint_mse_difference", "Novel endpoint MSE"),
                ("retention_difference", "Average old RMS deterioration (px)"),
                ("endpoint_retention_difference", "Endpoint old RMS deterioration (px)"))])+
            f"\n\nLower is better. Novel mean error favors Kalman in {d['favorable_replicas']}/{len(analysis['replicas'])} pairs. "
            "Intervals resample whole paired trajectories, conditional on development-selected settings; "
            "they are nominal and curves have pointwise bands. Holdout images never enter training or the observer."))
    for name, data in images.items():
        relative = str(generation.relative_to(REPO)) if generation.is_relative_to(REPO) else str(generation)
        c = nbformat.v4.new_code_cell(
            f"from mnist_experiment.combined_memory.adaptive_f1.report import display_figures\n"
            f"display_figures({relative!r}, {name!r})", id=TAG+"-"+name+"-plot", metadata={TAG: True})
        c.outputs = [nbformat.v4.new_output("display_data", data={"image/png":base64.b64encode(data).decode()}, metadata={})]
        c.execution_count = None
        output.append(c)
    if analysis:
        e = analysis["efficiency"]
        point, boot = e["point"], e["bootstrap"]
        lines = ["### Sample And Compute Efficiency", "",
            "S is the fraction of samples saved divided by the compute ratio needed to reach fixed F1's final predictive quality.", ""]
        time_ratio = e["equal_budget_learning_time_ratio"]
        time_ci = e["equal_budget_learning_time_ratio_ci"]
        lines.append("Equal-budget online compute ratio: **"+number(time_ratio)+"**"+
            (" ["+", ".join(number(v) for v in time_ci)+"]" if time_ci else "")+" (Kalman / fixed F1).\n")
        if point["status"] == "valid":
            ci = boot.get("ci")
            def inverse_interval(field, scale=1.):
                return "unavailable" if not ci else "["+", ".join(number(v*scale) for v in ci[field])+"]"
            lines.append(table(["Quantity", "Point", "Nominal 95% CI"],
                [["Samples saved", f"{100*point['saved_fraction']:.2f}%", inverse_interval("saved_fraction", 100)],
                 ["Compute ratio at equal quality", number(point["learning_time_ratio"]), inverse_interval("learning_time_ratio")],
                 ["S", number(point["S"]), inverse_interval("S")]]))
        else:
            lines.append("Sample equivalence is unavailable: "+point["status"].replace("_", " ")+".")
        lines.append(f"\nUnsupported inverse draws: {boot['invalid']}/{boot['draws']}. "
            "Ordinary inverse confidence intervals are withheld whenever any draw is unsupported. "
            "This target measures incremental savings beyond F1, not savings against Replay/EWC.")
        output.append(md("efficiency-notes", "\n".join(lines)))
    conclusion = snapshot.get("conclusion", "Implementation/calibration is underway; fresh efficacy is not yet available.")
    if analysis and analysis.get("is_final"):
        d, e = analysis["contrast"], analysis["efficiency"]
        ci = d["acquisition_difference"]["ci"]
        finding = ("Kalman pi improves average novel prediction error" if ci[1] < 0 else
                   "Fixed pi gives lower average novel prediction error" if ci[0] > 0 else
                   "The comparison does not resolve an average novel prediction advantage")
        conclusion = "**"+finding+".** The Kalman-minus-fixed gap is "+effect(d["acquisition_difference"])+" px² per landmark. "
        conclusion += f"Novel error favors Kalman in {d['favorable_replicas']}/8 paired replicas. "
        if d["retention_equivalent"]:
            conclusion += "Both old-data retention intervals lie within the 0.01-pixel equivalence band. "
        elif all(d[f+"_noninferiority_supported"] for f in ("retention", "endpoint_retention")):
            conclusion += "Old-data retention is supported as no worse within 0.01 pixels, although two-sided equivalence is not established. "
        else:
            conclusion += "Fresh retention matching is not established; read the old-data differences alongside acquisition. "
        conclusion += "The equal-budget online compute ratio is "+number(e["equal_budget_learning_time_ratio"])+". "
        if snapshot.get("ridge", 1.) != 1.:
            conclusion += "Ridge was calibrated separately, so this tests the two complete policies rather than adaptive timing alone. "
        conclusion += "These nominal paired intervals concern this frozen F1 recipe and continuously blurred grid task."
    output.append(md("conclusion", "### Conclusion\n\n"+conclusion))
    return progress_cell, output


def publish(snapshot, notebook, store):
    if snapshot.get("protocol") != PROTOCOL:
        raise ValueError("Phase 9E scalar snapshot required")
    snapshot = deepcopy(snapshot)
    identity_path = Path(store.root)/"identity.json"
    if identity_path.exists():
        revision = json.loads(identity_path.read_text())["identity"].get("scheduler_revision")
        if revision:
            snapshot["scheduler_revision"] = revision
    # Reject tensors/nonfinite JSON before touching the public notebook.
    encoded = artifacts._json_bytes(snapshot)
    notebook, root = Path(notebook), Path(store.root)
    notebook.parent.mkdir(parents=True, exist_ok=True)
    with notebook.with_suffix(".phase9e.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = nbformat.read(notebook, as_version=4) if notebook.exists() else nbformat.v4.new_notebook()
        if len(nbformat.writes(previous).encode()) > 64*1024*1024:
            raise ValueError("notebook exceeds bounded report size")
        historical = [deepcopy(c) for c in previous.cells if not owned(c)]
        identity = hashlib.sha256(encoded+SOURCE+artifacts._json_bytes([dict(c) for c in historical])).hexdigest()[:24]
        destination = root / "report" / "generations" / identity
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=destination.parent))
            images = figures(snapshot)
            if hasattr(store, "reserve"):
                store.reserve(3*len(nbformat.writes(previous).encode()) + 2*sum(map(len, images.values())) + len(encoded)+len(SOURCE))
            (staging / "figures").mkdir()
            for name, data in images.items():
                _atomic(staging / "figures" / (name+".png"), data)
            _atomic(staging / "snapshot.json", encoded)
            _atomic(staging / "report_source.py", SOURCE)
            progress, new = cells(snapshot, destination, images)
            built = deepcopy(previous)
            built.cells = historical[:1]+[progress]+historical[1:]+new
            _atomic(staging / "results.ipynb", nbformat.writes(built).encode())
            files = {str(p.relative_to(staging)): artifacts._file_hash(p) for p in staging.rglob("*") if p.is_file()}
            atomic_json(staging / "COMPLETED", dict(protocol=PROTOCOL, generation=identity, files=files))
            os.rename(staging, destination)
            artifacts._sync_directory(destination.parent)
        manifest = json.loads((destination / "COMPLETED").read_text())
        for name, digest in manifest["files"].items():
            if artifacts._file_hash(destination / name) != digest:
                raise artifacts.ArtifactError("report generation checksum mismatch")
        _atomic(notebook, (destination / "results.ipynb").read_bytes())
        atomic_json(root / "report" / "latest.json", dict(generation=identity, path=str(destination)))
        return identity


def display_figures(generation, name):
    from IPython.display import Image, display
    generation = Path(generation)
    if not generation.is_absolute():
        generation = REPO / generation
    manifest = json.loads((generation / "COMPLETED").read_text())
    file = "figures/"+name+".png"
    if file not in manifest["files"] or artifacts._file_hash(generation / file) != manifest["files"][file]:
        raise artifacts.ArtifactError("missing/incompatible report figure")
    display(Image(filename=str(generation / file)))
