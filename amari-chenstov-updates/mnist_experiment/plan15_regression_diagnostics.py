"""Artifact-only Phase 8 supplements, plus a few cached-MNIST renderer examples.

No fitting, model inference, Fisher estimation or new statistical tests occur.
Run after confirmation/analysis; the study ledger charges rendering and I/O.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plan15-regression-matplotlib"))

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.report import reserved_publisher
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_regression.config import ROOT, Config, content_hash
from mnist_experiment.fisher_memory_regression.data import MNISTSource, rotate_batch
from mnist_experiment.fisher_memory_regression.design import aggregate
from mnist_experiment.fisher_memory_regression.report import LABELS, COLORS, publish
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, file_bytes, file_digest, other_compute_seconds, read_json, scientific_sources,
)


def figure(width=10, height=4):
    result = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(result)
    return result


def png(figure):
    buffer = io.BytesIO()
    figure.canvas.draw()
    figure.savefig(buffer, format="png", dpi=125, bbox_inches="tight")
    figure.clear()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def frontier(rows, calibration):
    """Development policy means; each point retains its own replica count."""
    cells = aggregate(rows)
    fig = figure()
    axis = fig.subplots()
    for method in ("replay_ewc", "fd_replay_ewc", "ewc_only"):
        group = [row for row in cells if row["method"] == method and row["valid"]]
        feasible = [row for row in group if row["base_harm"] <= calibration["retention_margin_degrees"]]
        for row in group:
            good = row in feasible
            axis.scatter(row["seconds_per_arrival"], math.sqrt(row["novel_auc"]),
                marker="o" if good else "x", color=COLORS[method], s=45)
        nondominated = []
        best = math.inf
        for row in sorted(feasible, key=lambda item: item["seconds_per_arrival"]):
            if row["novel_auc"] < best:
                nondominated.append(row)
                best = row["novel_auc"]
        if nondominated:
            axis.plot([row["seconds_per_arrival"] for row in nondominated],
                [math.sqrt(row["novel_auc"]) for row in nondominated],
                color=COLORS[method], label=LABELS[method], linewidth=1)
            for row in nondominated:
                axis.annotate(f"({row['capacity']},{row['replay_count']}), r={row['count']}",
                    (row["seconds_per_arrival"], math.sqrt(row["novel_auc"])),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
        elif group:
            axis.scatter([], [], marker="x", color=COLORS[method], label=LABELS[method])
    axis.axvline(calibration["envelope_seconds_per_arrival"], color="#555", linestyle="--", label="Frozen time allowance")
    axis.set(xlabel="Measured learning seconds per incoming observation",
             ylabel="sqrt(mean new-domain A(N)) (degrees)", title="Development risk and compute trade-offs")
    axis.grid(alpha=.15)
    axis.legend(fontsize=8)
    return fig


def per_digit(rows, frozen_rows):
    """Final held-out errors, averaging MSE before square rooting."""
    fig = figure(width=11, height=4)
    axes = fig.subplots(1, 2)
    count = len(rows[0]["metrics"]) - 1
    for domain, axis in zip(("novel", "base"), axes):
        for method in ("fd_replay_ewc", "replay_ewc", "ewc_only", "frozen_regression"):
            group = frozen_rows if method == "frozen_regression" else [row for row in rows if row["method"] == method]
            errors = [[row["metrics"][0 if method == "frozen_regression" else -1]["per_digit_mse"][domain][str(k)]
                       for k in range(10)] for row in group]
            if errors:
                axis.plot(range(10), np.sqrt(np.mean(errors, axis=0)), marker=".",
                    linestyle="--" if method == "frozen_regression" else "-",
                    color=COLORS.get(method, "#555"), label=LABELS.get(method, "Frozen initialization"))
        axis.set(xticks=range(10), xlabel="Digit", ylabel="sqrt(mean predictive MSE) (degrees)",
                 title=f"Novel domain at n={count}" if domain == "novel" else f"Near-upright domain at n={count}")
        axis.grid(alpha=.15)
        axis.legend(fontsize=7)
    return fig


def examples(source, plan, check):
    selected = []
    pool = plan["evaluation_ids"]["base"]
    for digit in range(10):
        selected.append(next(index for index in pool if int(source.test_labels[index]) == digit))
    images = source.images(selected, test=True)
    fig = figure(width=11, height=3.8)
    axes = fig.subplots(3, 10)
    for row, degrees in enumerate((-40., 0., 40.)):
        rendered = rotate_batch(images, torch.full((10,), degrees), check_stop=check)
        for digit, axis in enumerate(axes[row]):
            axis.imshow(rendered[digit, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set(xticks=[], yticks=[])
            if row == 0:
                axis.set_title(str(digit), fontsize=9)
            if digit == 0:
                axis.set_ylabel(f"{degrees:+g} degrees", fontsize=8)
    fig.suptitle("Renderer examples; digit labels and angles are never model inputs", fontsize=11)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root, budget_root = args.root.resolve(), args.budget_root.resolve()
    if root.parent != budget_root:
        raise ValueError("diagnostics root must be a direct child of the shared budget root")
    identity = read_json(root / "identity.json")["identity"]
    if identity["source_hash"] != content_hash(scientific_sources()):
        raise ValueError("scientific source differs from the completed study")
    cfg = Config.read(root / "resolved_config.json")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg.threads)
    with budget_lock(budget_root):
        external = file_bytes(budget_root, exclude=root) + file_bytes(args.data_root) + 48_000_000
        other = other_compute_seconds(budget_root, root)
        with Store(root, identity, resume=True, external_bytes=external, max_bytes=cfg.max_artifact_bytes,
                   free_reserve_bytes=cfg.free_reserve_bytes, finalization_reserve_bytes=32_000_000,
                   max_active_seconds=72000 - other) as store:
            budget = RuntimeBudget(store, other_seconds=other)
            try:
                with store.control.signal_handlers():
                    if not store.is_complete("decision-analysis"):
                        raise ValueError("supplement requires completed confirmation and analysis")
                    snapshot = read_json(root / "report_snapshot.json")
                    analysis = read_json(root / "units/decision-analysis/result.json")
                    unit = "diagnostics-v2"
                    output = root / "reports/phase8-diagnostics-v2.html"
                    if store.is_complete(unit):
                        result = read_json(store.unit_path(unit) / "result.json")
                        if file_digest(output, budget.check) != result["sha256"]:
                            raise ValueError("sealed supplemental report changed")
                    else:
                        store.begin(unit, dict(kind="regression_diagnostics", schema_version=1,
                            source_hash=file_digest(Path(__file__)), analysis_hash=content_hash(analysis)))
                        confirmation = snapshot["confirmation"]["rows"]
                        allocation = snapshot["allocation"]
                        calibration = snapshot["calibration"]
                        assets = root / "units/assets-confirmation-0000/plan.json"
                        plan = read_json(assets)
                        source = MNISTSource(args.data_root)
                        contents = [("Rotation renderer", examples(source, plan, budget.check),
                            "Fixed inspection angles show sign, padding and interpolation. These examples are not a prediction task."),
                            ("Resource frontier", frontier(allocation, calibration),
                             "Exploratory development means. Circles meet the mean one-degree retention guard; crosses fail it. "
                             "Labels identify frontier points by (C,b) and replica count. Lines join nondominated admissible points within a method. "
                             "Unequal replica sets are not paired; this plot does not establish performance at new compute allowances."),
                            ("Per-digit prediction", per_digit(confirmation, [row for row in confirmation if row["method"] == "replay_ewc"]),
                             "Descriptive final held-out errors across the fixed fresh cohort. No per-digit significance tests. "
                             "The ten digit labels define evaluation strata only.")]
                        body = ["<!doctype html><meta charset='utf-8'><title>Phase 8 supplementary diagnostics</title>",
                                "<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 16px}img{max-width:100%}p{line-height:1.5}</style>",
                                "<h1>Phase 8 supplementary diagnostics</h1>"]
                        for title, plot, caption in contents:
                            budget.check()
                            body.append(f"<h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{png(plot)}'><p>{html.escape(caption)}</p>")
                        if store.is_complete("allocation-fit-review-evidence"):
                            image = read_json(root / "units/allocation-fit-review-evidence/figure.json")["png_base64"]
                            reason = read_json(root / "units/allocation-visual-review/fit_review.json")["reason"]
                            body.append("<h2>Development linear-fit diagnostic</h2>"
                                f"<img alt='Development mean and linear MSE fit' src='data:image/png;base64,{image}'>"
                                f"<p>{html.escape(reason)} This review preceded confirmation.</p>")
                        content = "\n".join(body).encode()
                        reserved_publisher(store)(output, content)
                        review_path = root / "units/allocation-visual-review/result.json"
                        review = read_json(review_path)["note"] if review_path.exists() else "No sealed visual review supplied."
                        result = dict(schema_version=1, sha256=file_digest(output),
                                      manifest_hash=analysis["manifest_hash"], report=str(output.relative_to(root)),
                                      allocation_review=review)
                        store.write_json(unit, "result.json", result)
                        store.complete(unit)
                    snapshot["diagnostics"] = dict(result, href=os.path.relpath(output, args.notebook.resolve().parent))
                    snapshot["allocation_review"] = result["allocation_review"]
                    snapshot["progress"]["active_seconds"] = budget.used
                    store.write_json(None, "report_snapshot.json", snapshot)
                    publish(snapshot, args.notebook, store)
            finally:
                budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
