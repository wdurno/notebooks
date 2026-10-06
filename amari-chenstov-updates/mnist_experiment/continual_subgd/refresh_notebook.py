"""Artifact-only progressive notebook for Plan 13."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan13-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from mnist_experiment.rotated_mnist.artifacts import _atomic_write, _read_json

from .artifacts import UnitStore
from .config import Plan13Study


DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
DEFAULT_NOTEBOOK = Path("mnist_experiment/continual_subgd_results.ipynb")
DIGIT9_OVR_SCHEMA_VERSION = "plan13-digit9-ovr-v1"
DIGIT9_OVR_REQUIRED = ("metrics.json", "summary.json", "checks.json")
PHASE_ORDER = {
    "phase0": 0,
    "phase1": 1,
    "phase2": 2,
    "phase3a": 3,
    "phase3a_repair": 4,
    "phase3b_trust": 5,
    "phase3b_geometry": 6,
    "phase3b_floor": 7,
    "phase4": 8,
    "phase5": 9,
    "phase5r": 10,
    "phase5r_smoke": 11,
    "phase6_low_prevalence": 12,
    "phase6a_ten_positive": 13,
    "phase6b_hundred_positive": 14,
}
PHASE_LABELS = {
    "phase0": "Phase 0: Contracts, reproduction, and smoke timing",
    "phase1": "Phase 1: Geometry identification",
    "phase2": "Phase 2: Foundational rotation pilot",
    "phase3a": "Phase 3A: Closed-loop mechanism probe",
    "phase3a_repair": "Phase 3A: Repair probe",
    "phase3b_trust": "Phase 3B: Trust probing",
    "phase3b_geometry": "Phase 3B: Geometry probing",
    "phase3b_floor": "Phase 3B: Floor probing",
    "phase4": "Phase 4: Independent rotation confirmation",
    "phase5": "Phase 5: Original digit-9 transport",
    "phase5r": "Phase 5R: Repaired digit-9 transport",
    "phase6_low_prevalence": "Phase 6: Low-prevalence few-shot study",
    "phase6a_ten_positive": "Phase 6A: Ten-positive scaling study",
    "phase6b_hundred_positive": "Phase 6B: Hundred-positive closeout study",
}


def _image_output(figure: plt.Figure) -> dict[str, Any]:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "output_type": "display_data",
        "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
        "metadata": {},
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(
    source: str,
    outputs: list[dict[str, Any]],
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    metadata = {} if phase is None else {"plan13_phase": phase}
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": metadata,
        "outputs": outputs,
        "source": source,
    }


def _phase_sort_key(phase: str) -> tuple[int, str]:
    return PHASE_ORDER.get(phase, len(PHASE_ORDER)), phase


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No artifacts are available yet._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def collect_progress(store: UnitStore) -> dict[str, Any]:
    progress = []
    trajectories = []
    curve_rows: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    pr_curves: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    analyses = {}
    gates = {}
    ledger_root = store.root / "ledgers"
    for path in sorted(ledger_root.glob("*.json")) if ledger_root.exists() else []:
        ledger = _read_json(path)
        if ledger["study_hash"] != store.study.config_hash:
            continue
        counts = defaultdict(int)
        for item in ledger["items"]:
            required = tuple(item["required"])
            completed = store.completed(item["unit"], required)
            if completed is None:
                _, working = store.paths(item["unit"])
                counts["incomplete" if working.exists() else "pending"] += 1
                continue
            counts["completed"] += 1
            if item["action"] in {
                "trajectory",
                "mixture_trajectory",
                "phase5r_trajectory",
                "phase6_trajectory",
            }:
                summary = _read_json(completed / "summary.json")
                trajectories.append(summary)
                metrics = _read_json(completed / "metrics.json")
                curve_rows[
                    (
                        ledger["phase"],
                        summary["schedule_kind"],
                        summary["condition"]["name"],
                    )
                ].append(metrics)
                if item["action"] == "phase6_trajectory":
                    pr_curves[(ledger["phase"], summary["condition"]["name"])].append(
                        _read_json(completed / "pr_curves.json")
                    )
            elif item["action"].endswith("analysis"):
                analyses[ledger["phase"]] = _read_json(completed / "summary.json")
            elif item["action"] == "phase5r_gate":
                gates[ledger["phase"]] = _read_json(completed / "summary.json")
        progress.append(
            {
                "phase": ledger["phase"],
                "planned": len(ledger["items"]),
                "completed": counts["completed"],
                "pending": counts["pending"],
                "incomplete": counts["incomplete"],
                "status": "complete" if counts["completed"] == len(ledger["items"]) else "partial",
            }
        )
    return {
        "progress": progress,
        "trajectories": trajectories,
        "curve_rows": curve_rows,
        "pr_curves": pr_curves,
        "analyses": analyses,
        "gates": gates,
        "digit9_ovr": _collect_digit9_ovr(store),
    }


def _collect_digit9_ovr(store: UnitStore) -> dict[str, Any] | None:
    root = store.root / "digit9_mixture" / "phase5_posthoc"
    candidates = []
    for path in root.iterdir() if root.exists() else []:
        if not path.is_dir() or not (path / "COMPLETED").is_file():
            continue
        unit = _read_json(path / "config.json")
        if (
            unit.get("study_hash") != store.study.config_hash
            or unit.get("kind") != "digit9_ovr"
            or unit.get("detail", {}).get("schema_version")
            != DIGIT9_OVR_SCHEMA_VERSION
        ):
            continue
        completed = store.completed(unit, DIGIT9_OVR_REQUIRED)
        assert completed is not None
        manifest = _read_json(completed / "manifest.json")
        candidates.append((manifest["completed_at"], completed))
    if not candidates:
        return None
    _, selected = max(candidates)
    return {
        "metrics": _read_json(selected / "metrics.json"),
        "summary": _read_json(selected / "summary.json"),
        "checks": _read_json(selected / "checks.json"),
        "artifact_path": selected.resolve().relative_to(
            store.repo_root.resolve()
        ).as_posix(),
    }


def _mean_curve(replicas: list[list[dict[str, Any]]], field: str) -> tuple[list[int], list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for rows in replicas:
        for row in rows:
            by_step[int(row["post_burn_in_observations"])].append(float(row[field]))
    steps = sorted(by_step)
    return steps, [statistics.fmean(by_step[step]) for step in steps]


def _mean_curve_ci(
    replicas: list[list[dict[str, Any]]],
    field: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for rows in replicas:
        for row in rows:
            by_step[int(row["post_burn_in_observations"])].append(float(row[field]))
    steps = sorted(by_step)
    means = [statistics.fmean(by_step[step]) for step in steps]
    margins = [
        0.0
        if len(by_step[step]) < 2
        else 1.96 * statistics.stdev(by_step[step]) / len(by_step[step]) ** 0.5
        for step in steps
    ]
    return (
        steps,
        means,
        [mean - margin for mean, margin in zip(means, margins)],
        [mean + margin for mean, margin in zip(means, margins)],
    )


def _expected_positive_curve(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    """Align cumulative expected positives with pre-update evaluation rows."""
    ordered = sorted(rows, key=lambda row: int(row["post_burn_in_observations"]))
    observations = [int(row["post_burn_in_observations"]) for row in ordered]
    expected = [0.0]
    for current, start, stop in zip(ordered, observations, observations[1:]):
        integrated = stop - start
        if integrated <= 0:
            raise ValueError("Evaluation observations must increase strictly")
        expected.append(expected[-1] + integrated * float(current["p"]))
    return observations, expected


def _low_prevalence_cells(
    collected: dict[str, Any],
    *,
    phase: str,
    heading: str,
    short_label: str,
    expected_count: float,
    complete_description: str,
) -> list[dict[str, Any]]:
    progress = next(
        (row for row in collected["progress"] if row["phase"] == phase),
        None,
    )
    if progress is None:
        return []
    complete = progress["status"] == "complete"
    cells: list[dict[str, Any]] = [
        _markdown_cell(
            f"## {heading}\n\n"
            + (
                f"**Fixed 64-replica ledger complete.** {complete_description}"
                if complete
                else "**INCOMPLETE LEDGER: descriptive execution-health views only.** "
                "Do not interpret intervals or select a condition while artifacts are "
                f"still arriving ({progress['completed']}/{progress['planned']} units complete)."
            )
            + f" **Design target: {expected_count:g} expected observed digit-9 "
            "examples per trajectory.** The training path is restricted to "
            "$0\\leq p\\leq.1$; precision is "
            "standardized to deployment prevalence $p_{\\mathrm{ref}}=.1$."
        )
    ]
    keys = [key for key in collected["curve_rows"] if key[0] == phase]
    if not keys:
        return cells

    condition_replicas = {
        condition: collected["curve_rows"][(phase, schedule, condition)]
        for _, schedule, condition in sorted(keys)
    }

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for condition, replicas in condition_replicas.items():
        for axis, field in zip(axes, ("precision_ref_0.1", "nine_ovr_recall")):
            x, mean, low, high = _mean_curve_ci(replicas, field)
            line = axis.plot(x, mean, label=condition)[0]
            axis.fill_between(x, low, high, color=line.get_color(), alpha=0.12)
    axes[0].set(title="Fixed-.1 precision", ylabel="Precision")
    axes[1].set(title="Digit-9 recall", ylabel="Recall")
    for axis in axes:
        axis.set(xlabel="Post-burn-in observations", ylim=(0, 1.01))
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle(
        f"{short_label} co-primary trajectories "
        f"($E[N_9]={expected_count:g}$; mean and approximate 95% CI)"
    )
    cells.append(
        _code_cell(
            "# Rendered from immutable metric rows; no model inference occurs here.",
            [_image_output(figure)],
            phase=phase,
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for condition, replicas in condition_replicas.items():
        x, mean, low, high = _mean_curve_ci(replicas, "false_positives_per_1000")
        line = axes[0].plot(x, mean, label=condition)[0]
        axes[0].fill_between(x, low, high, color=line.get_color(), alpha=0.10)
    count_replicas = condition_replicas.get(
        "full_space", next(iter(condition_replicas.values()))
    )
    x, mean, low, high = _mean_curve_ci(count_replicas, "cumulative_observed_nines")
    axes[1].plot(x, mean, color="#222222", label="mean cumulative count")
    axes[1].fill_between(x, low, high, color="#777777", alpha=0.20)
    axes[1].axhline(
        expected_count,
        color="#b44",
        linestyle="--",
        linewidth=1,
        label=f"expected final count ({expected_count:g})",
    )
    axes[0].set(
        title="False positives per 1,000 non-9s",
        xlabel="Post-burn-in observations",
        ylabel="False positives",
    )
    axes[1].set(
        title="Observed digit-9 examples",
        xlabel="Post-burn-in observations",
        ylabel="Cumulative count",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7, frameon=False)
    figure.suptitle(f"{short_label} error burden and realized scarcity")
    cells.append(
        _code_cell(
            "# Counts are shown once because paired conditions share each replica stream.",
            [_image_output(figure)],
            phase=phase,
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for condition, replicas in condition_replicas.items():
        x, current = _mean_curve(replicas, "current_nll")
        _, retention = _mean_curve(replicas, "p0_nll")
        axes[0].plot(x, current, label=condition)
        axes[1].plot(x, retention, label=condition)
    axes[0].set(title="Actual-current mixture NLL", ylabel="NLL")
    axes[1].set(title="$p=0$ retention NLL", ylabel="NLL")
    for axis in axes:
        axis.set(xlabel="Post-burn-in observations")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle(f"{short_label} learning and retention")
    cells.append(
        _code_cell(
            "# Current risk uses the displayed p coordinate; retention stays at p=0.",
            [_image_output(figure)],
            phase=phase,
        )
    )

    adaptive = condition_replicas.get("adaptive_floor_0.1")
    if adaptive:
        derived: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for replica in adaptive:
            for row in replica:
                geometry = row["geometry"]
                step = int(row["post_burn_in_observations"])
                derived[step]["rank"].append(float(geometry["rank"]))
                derived[step]["orthogonal_gain"].append(
                    float(geometry["orthogonal_gain"])
                )
                innovation = geometry.get("innovation")
                if innovation is not None:
                    derived[step]["innovation"].append(float(innovation))
                    derived[step]["shadow_alignment"].append(
                        max(0.0, 1 - float(innovation)) ** 0.5
                    )
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for axis, (field, title) in zip(
            axes.flat,
            (
                ("rank", "Realized basis rank"),
                ("innovation", "Shadow innovation fraction"),
                ("orthogonal_gain", "Orthogonal floor gain"),
                ("shadow_alignment", "Shadow-to-basis alignment"),
            ),
        ):
            x = [step for step in sorted(derived) if derived[step].get(field)]
            y = [statistics.fmean(derived[step][field]) for step in x]
            axis.plot(x, y, color="#3b6f8f")
            axis.set(title=title, xlabel="Post-burn-in observations")
            axis.grid(alpha=0.2)
        figure.suptitle("Adaptive SubGD geometry health")
        cells.append(
            _code_cell(
                "# Alignment is sqrt(1 - innovation), derived from stored diagnostics.",
                [_image_output(figure)],
                phase=phase,
            )
        )

    phase_pr = {
        condition: rows
        for (pr_phase, condition), rows in collected["pr_curves"].items()
        if pr_phase == phase
    }
    if phase_pr:
        checkpoints = (0.01, 0.05, 0.10)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
        for axis, checkpoint in zip(axes, checkpoints):
            for condition, replica_curves in sorted(phase_pr.items()):
                selected = [
                    row
                    for replica in replica_curves
                    for row in replica
                    if abs(float(row["p"]) - checkpoint) < 1e-12
                ]
                if not selected:
                    continue
                recall = [
                    statistics.fmean(float(row["recall"][i]) for row in selected)
                    for i in range(len(selected[0]["recall"]))
                ]
                precision = [
                    statistics.fmean(float(row["precision"][i]) for row in selected)
                    for i in range(len(selected[0]["precision"]))
                ]
                axis.plot(recall, precision, label=condition)
            axis.set(title=f"After p={checkpoint:.2f} batch", xlabel="Recall", xlim=(0, 1.01), ylim=(0, 1.01))
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Precision at $p_{ref}=.1$")
        axes[-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.suptitle(
            f"{short_label} threshold-swept precision-recall curves "
            f"($E[N_9]={expected_count:g}$)"
        )
        cells.append(
            _code_cell(
                "# Curves are pointwise replica means of stored evaluation artifacts.",
                [_image_output(figure)],
                phase=phase,
            )
        )

    analysis = collected["analyses"].get(phase)
    if analysis is not None:
        cells.append(
            _markdown_cell(
                "### Frozen paired comparisons\n\n"
                "Positive gain favors the named treatment. Precision and recall remain "
                "separate co-primary outcomes.\n\n"
                + _table(
                    [
                        "Contrast",
                        "Precision gain [95% CI]",
                        "Recall gain [95% CI]",
                        "Favorable precision",
                        "Favorable recall",
                    ],
                    [
                        [
                            row["name"],
                            (
                                f"{row['precision_ref_0.1']['mean_gain']:.5f} "
                                f"[{row['precision_ref_0.1']['ci95_low']:.5f}, "
                                f"{row['precision_ref_0.1']['ci95_high']:.5f}]"
                            ),
                            (
                                f"{row['nine_ovr_recall']['mean_gain']:.5f} "
                                f"[{row['nine_ovr_recall']['ci95_low']:.5f}, "
                                f"{row['nine_ovr_recall']['ci95_high']:.5f}]"
                            ),
                            (
                                f"{row['precision_ref_0.1']['favorable_replicas']}/"
                                f"{row['precision_ref_0.1']['replicas']}"
                            ),
                            (
                                f"{row['nine_ovr_recall']['favorable_replicas']}/"
                                f"{row['nine_ovr_recall']['replicas']}"
                            ),
                        ]
                        for row in analysis["comparisons"]
                    ],
                )
            )
        )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(14, 4.5),
            constrained_layout=True,
        )
        display_labels = {
            "primary_head_vs_full": "head vs full",
            "absolute_head_vs_no_update": "head vs no update",
            "subgd_adaptive_vs_full": "adaptive vs full",
            "practical_head_vs_adaptive": "head vs adaptive",
            "dimensional_bias_vs_head": "bias vs head",
        }
        labels = [
            display_labels.get(row["name"], row["name"].replace("_", " "))
            for row in analysis["comparisons"]
        ]
        for axis, (field, title) in zip(
            axes,
            (("precision_ref_0.1", "Fixed-.1 precision AUC gain"), ("nine_ovr_recall", "Digit-9 recall AUC gain")),
        ):
            axis.boxplot(
                [row[field]["gains"] for row in analysis["comparisons"]],
                tick_labels=labels,
                vert=False,
                showmeans=True,
            )
            axis.axvline(0, color="#555555", linewidth=1)
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(f"{short_label} paired effect distributions")
        cells.append(
            _code_cell(
                "# Every prespecified contrast is shown; replicas are the independent units.",
                [_image_output(figure)],
                phase=phase,
            )
        )
    return cells


def _phase6_cells(collected: dict[str, Any]) -> list[dict[str, Any]]:
    return _low_prevalence_cells(
        collected,
        phase="phase6_low_prevalence",
        heading="Phase 6: Low-prevalence few-shot study",
        short_label="Phase 6",
        expected_count=4.4,
        complete_description=(
            "The paired estimates below are the prospectively specified Phase 6 analysis."
        ),
    )


def _scaling_phase_cells(
    collected: dict[str, Any],
    *,
    phase: str,
    heading: str,
    short_label: str,
    expected_count: float,
    complete_description: str,
) -> list[dict[str, Any]]:
    cells = _low_prevalence_cells(
        collected,
        phase=phase,
        heading=heading,
        short_label=short_label,
        expected_count=expected_count,
        complete_description=complete_description,
    )
    phase_pr = {
        condition: rows
        for (pr_phase, condition), rows in collected["pr_curves"].items()
        if pr_phase == phase
    }
    keys = [key for key in collected["curve_rows"] if key[0] == phase]
    if not phase_pr or not keys:
        return cells

    practical = (
        "no_update",
        "full_space",
        "head_only",
        "adaptive_floor_0.1",
    )
    figure, axis = plt.subplots(1, 1, figsize=(8, 5))
    table_rows = []
    for condition in practical:
        replica_curves = phase_pr.get(condition)
        if not replica_curves:
            continue
        selected = [
            row
            for replica in replica_curves
            for row in replica
            if abs(float(row["p"]) - 0.10) < 1e-12
        ]
        if not selected:
            continue
        recall = [
            statistics.fmean(float(row["recall"][i]) for row in selected)
            for i in range(len(selected[0]["recall"]))
        ]
        precision = [
            statistics.fmean(float(row["precision"][i]) for row in selected)
            for i in range(len(selected[0]["precision"]))
        ]
        line = axis.plot(recall, precision, label=condition)[0]
        replicas = next(
            collected["curve_rows"][key]
            for key in keys
            if key[2] == condition
        )
        _, endpoint_precision = _mean_curve(replicas, "precision_ref_0.1")
        _, endpoint_recall = _mean_curve(replicas, "nine_ovr_recall")
        axis.scatter(
            endpoint_recall[-1],
            endpoint_precision[-1],
            color=line.get_color(),
            edgecolor="white",
            linewidth=0.8,
            s=45,
            zorder=3,
        )
        table_rows.append(
            [
                condition,
                f"{statistics.fmean(float(row['average_precision']) for row in selected):.4f}",
                f"{endpoint_precision[-1]:.4f}",
                f"{endpoint_recall[-1]:.4f}",
            ]
        )
    axis.set(
        title="After the p=.10 batch; dots show mean argmax operating points",
        xlabel="Digit-9 recall",
        ylabel="Precision at $p_{ref}=.1$",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        f"{short_label} focused endpoint precision-recall tradeoff "
        f"($E[N_9]={expected_count:g}$)"
    )
    cells.append(
        _code_cell(
            "# Focused practical view; the complete five-condition panel remains above.",
            [_image_output(figure)],
            phase=phase,
        )
    )
    cells.append(
        _markdown_cell(
            f"### {short_label} endpoint operating points\n\n"
            f"**Design target: {expected_count:g} expected observed digit-9 "
            "examples per trajectory.** "
            "The threshold-swept average precision and ordinary argmax point use the "
            "same fixed evaluation panel; no threshold is selected here.\n\n"
            + _table(
                ["Condition", "Mean standardized AP", "Argmax precision", "Argmax recall"],
                table_rows,
            )
        )
    )
    return cells


def _phase6a_cells(collected: dict[str, Any]) -> list[dict[str, Any]]:
    return _scaling_phase_cells(
        collected,
        phase="phase6a_ten_positive",
        heading="Phase 6A: Ten-positive low-prevalence scaling study",
        short_label="Phase 6A",
        expected_count=9.9,
        complete_description=(
            "These estimates are post-Phase-6 exploratory development evidence."
        ),
    )


def _phase6b_cells(collected: dict[str, Any]) -> list[dict[str, Any]]:
    phase = "phase6b_hundred_positive"
    cells = _scaling_phase_cells(
        collected,
        phase=phase,
        heading="Phase 6B: Hundred-positive low-prevalence closeout study",
        short_label="Phase 6B",
        expected_count=100.1,
        complete_description=(
            "These estimates are exploratory closeout evidence after Phases 6 and 6A."
        ),
    )
    head_keys = [
        key
        for key in collected["curve_rows"]
        if key[0] == phase and key[2] == "head_only"
    ]
    if not head_keys:
        return cells
    if len(head_keys) != 1:
        raise RuntimeError("Phase 6B head-only efficiency plot requires one schedule")

    replicas = collected["curve_rows"][head_keys[0]]
    x, precision, precision_low, precision_high = _mean_curve_ci(
        replicas, "precision_ref_0.1"
    )
    recall_x, recall, recall_low, recall_high = _mean_curve_ci(
        replicas, "nine_ovr_recall"
    )
    expected_x, expected_nines = _expected_positive_curve(replicas[0])
    if x != recall_x or x != expected_x:
        raise RuntimeError("Phase 6B efficiency curves have incompatible evaluations")
    if abs(expected_nines[-1] - 100.1) > 1e-9:
        raise RuntimeError("Phase 6B expected digit-9 count no longer matches its design")

    precision_color = "#0072B2"
    recall_color = "#D55E00"
    count_color = "#333333"
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    count_axis = axis.twinx()
    precision_line = axis.plot(
        x,
        precision,
        color=precision_color,
        marker="o",
        linewidth=2.2,
        label="Head-only precision",
    )[0]
    axis.fill_between(
        x,
        [max(0.0, value) for value in precision_low],
        [min(1.0, value) for value in precision_high],
        color=precision_color,
        alpha=0.13,
        linewidth=0,
    )
    recall_line = axis.plot(
        x,
        recall,
        color=recall_color,
        marker="o",
        linewidth=2.2,
        label="Head-only recall",
    )[0]
    axis.fill_between(
        x,
        [max(0.0, value) for value in recall_low],
        [min(1.0, value) for value in recall_high],
        color=recall_color,
        alpha=0.13,
        linewidth=0,
    )
    count_line = count_axis.plot(
        x,
        expected_nines,
        color=count_color,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label=r"Expected observed 9s, $\mathbb{E}[N_9]$",
    )[0]
    axis.set(
        title="Head-only performance as evidence accumulates",
        xlabel="Digits integrated after burn-in",
        ylabel="Mean holdout performance",
        xlim=(0, x[-1]),
        ylim=(0, 1.01),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    count_axis.set(
        ylabel="Expected observed digit-9 examples",
        ylim=(0, 105),
    )
    count_axis.tick_params(axis="y", colors=count_color)
    count_axis.yaxis.label.set_color(count_color)
    axis.grid(alpha=0.2)
    lines = [precision_line, recall_line, count_line]
    axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="upper left",
    )
    figure.suptitle(r"Phase 6B statistical efficiency ($\mathbb{E}[N_9]=100.1$)")
    cells.extend(
        [
            _markdown_cell(
                "### Phase 6B head-only statistical efficiency\n\n"
                "Precision and recall are replica means on the fixed-$p_{ref}=.1$ "
                "holdout; shaded regions are pointwise approximate 95% confidence "
                "intervals. The expected-count curve is calculated from the frozen "
                "mixture schedule and all integrated digits, not from realized "
                "digit-9 draws. The two vertical scales are different, so line "
                "crossings have no quantitative interpretation."
            ),
            _code_cell(
                "# Artifact-backed statistical-efficiency figure; no fitting occurs here.",
                [_image_output(figure)],
                phase=phase,
            ),
        ]
    )
    return cells


def _phase5r_metric_cells(collected: dict[str, Any]) -> list[dict[str, Any]]:
    phase = "phase5r"
    keys = [key for key in collected["curve_rows"] if key[0] == phase]
    if not keys:
        return []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for _, schedule, condition in sorted(keys):
        replicas = collected["curve_rows"][(phase, schedule, condition)]
        x, recall = _mean_curve(replicas, "nine_ovr_recall")
        _, balanced = _mean_curve(replicas, "nine_ovr_balanced_accuracy")
        axes[0].plot(x, recall, label=condition)
        axes[1].plot(x, balanced, label=condition)
    axes[0].set(
        title="Digit-9 recall",
        xlabel="Post-burn-in observations",
        ylabel="Recall",
        ylim=(0, 1.01),
    )
    axes[1].set(
        title="Digit-9 balanced OvR accuracy",
        xlabel="Post-burn-in observations",
        ylabel="Balanced accuracy",
        ylim=(0, 1.01),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    figure.suptitle("Phase 5R / repaired digit-9 transport")
    return [
        _code_cell(
            "# Artifact-backed Phase 5R figure; no fitting occurs here.",
            [_image_output(figure)],
            phase=phase,
        )
    ]


def _digit9_ovr_cells(digit9_ovr: dict[str, Any] | None) -> list[dict[str, Any]]:
    if digit9_ovr is None:
        return []

    cells = [
        _markdown_cell(
            "#### Digit-9 one-vs-rest diagnostics\n\n"
            "At each mixture coordinate, digit 9 is the positive class and digits "
            "0 through 8 form the negative class. Current-mixture OvR accuracy is "
            "$p_t\\,\\mathrm{TPR}_{9,t}+(1-p_t)\\,\\mathrm{TNR}_{9,t}$. "
            "Balanced OvR accuracy is "
            "$(\\mathrm{TPR}_{9,t}+\\mathrm{TNR}_{9,t})/2$ and is shown beside it "
            "so changing prevalence cannot hide a one-class classifier. These curves "
            "are reconstructed from immutable Phase 5 parameter trajectories and the "
            "fixed evaluation panel; no fitting occurs in this notebook.\n\n"
            + _table(
                [
                    "Condition",
                    "OvR accuracy AUC",
                    "Balanced OvR AUC",
                    "9 recall AUC",
                    "Non-9 specificity AUC",
                ],
                [
                    [
                        row["condition"],
                        f"{row['mean_nine_ovr_accuracy_auc']:.4f}",
                        f"{row['mean_nine_ovr_balanced_accuracy_auc']:.4f}",
                        f"{row['mean_nine_ovr_recall_auc']:.4f}",
                        f"{row['mean_nine_ovr_specificity_auc']:.4f}",
                    ]
                    for row in digit9_ovr["summary"]["condition_summary"]
                ],
            )
        )
    ]
    grouped_ovr: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in digit9_ovr["metrics"]:
        grouped_ovr[row["condition"]][int(row["replica_index"])].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for condition in sorted(grouped_ovr):
        replicas = list(grouped_ovr[condition].values())
        x, accuracy = _mean_curve(replicas, "nine_ovr_accuracy")
        _, balanced = _mean_curve(replicas, "nine_ovr_balanced_accuracy")
        axes[0].plot(x, accuracy, label=condition)
        axes[1].plot(x, balanced, label=condition)
    axes[0].set(
        title="Current-mixture digit-9 OvR accuracy",
        xlabel="Post-burn-in observations",
        ylabel="OvR accuracy",
        ylim=(0, 1.01),
    )
    axes[1].set(
        title="Digit-9 balanced OvR accuracy",
        xlabel="Post-burn-in observations",
        ylabel="Balanced accuracy",
        ylim=(0, 1.01),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(
        fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    figure.suptitle("Phase 5 / Digit-9 one-vs-rest (16 paired replicas)")
    cells.append(
        _code_cell(
            "# Artifact-backed post-hoc figure; no computation is run by this notebook.",
            [_image_output(figure)],
            phase="phase5",
        )
    )
    return cells


def build_notebook(store: UnitStore) -> dict[str, Any]:
    collected = collect_progress(store)
    notebook: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "plan13": {
                "study_hash": store.study.config_hash,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "artifact_only": True,
            },
        },
    }
    progress_rows = [
        [
            row["phase"],
            row["completed"],
            row["planned"],
            row["pending"],
            row["incomplete"],
            row["status"],
        ]
        for row in collected["progress"]
    ]
    notebook["cells"].append(
        _markdown_cell(
            "# Plan 13: Continual SubGD\n\n"
            "This notebook is an artifact-only live scientific record. Partial phases are "
            "descriptive and do not unlock fixed-size inferential claims.\n\n"
            "## Execution progress\n\n"
            + _table(
                ["Phase", "Complete", "Planned", "Pending", "Incomplete", "Status"],
                progress_rows,
            )
        )
    )

    analyses = collected["analyses"]
    if "phase1" in analyses:
        phase1 = analyses["phase1"]
        selected = phase1["selected"]
        notebook["cells"].append(
            _markdown_cell(
                "## Geometry identification\n\n"
                f"Classification: **{phase1['classification']}**. "
                f"The reconstruction rule selected $K={selected['burn_in_steps']}$ and "
                f"$r={selected['rank']}$.\n\n"
                + _table(
                    ["K", "Rank", "Held-out residual", "Random residual", "Gradient proxy", "Units"],
                    [
                        [
                            row["burn_in_steps"],
                            row["rank"],
                            f"{row['mean_held_out_residual_ratio']:.4f}",
                            f"{row['mean_random_residual_ratio']:.4f}",
                            f"{row['mean_gradient_proxy_residual_ratio']:.4f}",
                            row["units"],
                        ]
                        for row in phase1["aggregate"]
                    ],
                )
            )
        )

    selection_rows = []
    for phase in (
        "phase0",
        "phase2",
        "phase3a",
        "phase3a_repair",
        "phase3b_trust",
        "phase3b_geometry",
        "phase3b_floor",
    ):
        if phase not in analyses:
            continue
        selection = analyses[phase].get("selection", {})
        selected = selection.get("selected_condition")
        if isinstance(selected, dict):
            selected = selected.get("name")
        if selected is None and selection.get("survivor_conditions"):
            selected = ", ".join(
                value["name"] for value in selection["survivor_conditions"]
            )
        if selected is None:
            selected = selection.get("online_condition", "diagnostic only")
        selection_rows.append(
            [
                phase,
                selected,
                selection.get("classification", "development selection"),
                selection.get("fallback_used", False),
            ]
        )
    if selection_rows:
        notebook["cells"].append(
            _markdown_cell(
                "## Development decisions\n\n"
                + _table(
                    ["Stage", "Carried condition(s)", "Classification", "Fallback"],
                    selection_rows,
                )
            )
        )

    if "phase4" in analyses:
        phase4 = analyses["phase4"]
        notebook["cells"].append(
            _markdown_cell(
                "## Independent rotation confirmation\n\n"
                f"Classification: **{phase4['selection']['classification']}**. "
                "Positive gain means the selected adaptive method has lower NLL.\n\n"
                + _table(
                    [
                        "Schedule",
                        "Current gain",
                        "Current 95% CI",
                        "Retention gain",
                        "Retention 95% CI",
                        "Favorable",
                        "Promoted",
                    ],
                    [
                        [
                            row["schedule"],
                            f"{row['adaptive_vs_full_current_nll']['mean_gain']:.5f}",
                            (
                                f"[{row['adaptive_vs_full_current_nll']['ci95_low']:.5f}, "
                                f"{row['adaptive_vs_full_current_nll']['ci95_high']:.5f}]"
                            ),
                            f"{row['adaptive_vs_full_retention_nll']['mean_gain']:.5f}",
                            (
                                f"[{row['adaptive_vs_full_retention_nll']['ci95_low']:.5f}, "
                                f"{row['adaptive_vs_full_retention_nll']['ci95_high']:.5f}]"
                            ),
                            (
                                f"{row['adaptive_vs_full_current_nll']['favorable_replicas']}/"
                                f"{row['adaptive_vs_full_current_nll']['replicas']}"
                            ),
                            row["promotion_gate"],
                        ]
                        for row in phase4["comparisons"]
                    ],
                )
            )
        )

    if "phase5" in analyses:
        phase5 = analyses["phase5"]
        notebook["cells"].append(
            _markdown_cell(
                "## Digit-9-mixture transport\n\n"
                "**Invalid for SubGD efficacy.** The common first-order optimizer failed "
                "to solve the EWC objective before treatment assignment. These immutable "
                "results characterize that optimizer pathology and are not pooled with "
                "rotation confirmation or repaired Phase 5R. Positive gain means lower "
                "NLL than the equally under-optimized full-space learner.\n\n"
                + _table(
                    ["Condition", "Current gain", "Current 95% CI", "Retention gain"],
                    [
                        [
                            row["condition"],
                            f"{row['current_nll_vs_full']['mean_gain']:.5f}",
                            (
                                f"[{row['current_nll_vs_full']['ci95_low']:.5f}, "
                                f"{row['current_nll_vs_full']['ci95_high']:.5f}]"
                            ),
                            f"{row['retention_nll_vs_full']['mean_gain']:.5f}",
                        ]
                        for row in phase5["comparisons"]
                    ],
                )
            )
        )

    gates = collected["gates"]
    if "phase5r" in gates:
        gate = gates["phase5r"]
        notebook["cells"].append(
            _markdown_cell(
                "## Phase 5R viability gate\n\n"
                f"Treatment status: **{'unlocked' if gate['passed'] else 'locked'}**. "
                "This is an engineering validity gate, not a significance test.\n\n"
                + _table(
                    ["Metric", "Observed", "Threshold", "Passed"],
                    [
                        [
                            name,
                            f"{gate['observed'][name]:.4f}",
                            f"{threshold:.4f}",
                            gate["criteria"][name],
                        ]
                        for name, threshold in gate["thresholds"].items()
                    ],
                )
            )
        )

    if "phase5r" in analyses:
        phase5r = analyses["phase5r"]
        notebook["cells"].append(
            _markdown_cell(
                "## Phase 5R repaired transport\n\n"
                "These reused streams are post-diagnostic development evidence, not fresh "
                "confirmation. All conditions use coordinate strong-Wolfe L-BFGS. Positive "
                "gain means improvement over repaired full-space learning.\n\n"
                + _table(
                    [
                        "Condition",
                        "Current NLL gain",
                        "Current accuracy gain",
                        "Balanced OvR gain",
                        "9 recall gain",
                        "Favorable recall",
                    ],
                    [
                        [
                            row["condition"],
                            f"{row['current_nll_vs_full']['mean_gain']:.5f}",
                            f"{row['current_accuracy_vs_full']['mean_gain']:.5f}",
                            f"{row['nine_ovr_balanced_accuracy_vs_full']['mean_gain']:.5f}",
                            f"{row['nine_ovr_recall_vs_full']['mean_gain']:.5f}",
                            (
                                f"{row['nine_ovr_recall_vs_full']['favorable_replicas']}/"
                                f"{row['nine_ovr_recall_vs_full']['replicas']}"
                            ),
                        ]
                        for row in phase5r["comparisons"]
                    ],
                )
            )
        )

    trajectories = collected["trajectories"]
    if trajectories:
        grouped_summary: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in trajectories:
            grouped_summary[(row["phase"], row["schedule_kind"], row["condition"]["name"])].append(row)
        notebook["cells"].append(
            _markdown_cell(
                "## Predictive summaries\n\n"
                + _table(
                    ["Phase", "Schedule", "Condition", "Replicas", "NLL AUC", "Accuracy AUC", "Hours"],
                    [
                        [
                            phase,
                            schedule,
                            condition,
                            len(values),
                            f"{statistics.fmean(item['post_burn_in_current_nll_auc'] for item in values):.4f}",
                            f"{statistics.fmean(item['post_burn_in_current_accuracy_auc'] for item in values):.4f}",
                            f"{sum(item['total_wall_time_seconds'] for item in values) / 3600:.2f}",
                        ]
                        for (phase, schedule, condition), values in sorted(
                            grouped_summary.items(),
                            key=lambda item: (
                                _phase_sort_key(item[0][0]),
                                item[0][1],
                                item[0][2],
                            ),
                        )
                    ],
                )
            )
        )
        for phase in sorted(
            {key[0] for key in collected["curve_rows"]},
            key=_phase_sort_key,
        ):
            if phase in {
                "phase6_low_prevalence",
                "phase6a_ten_positive",
                "phase6b_hundred_positive",
            }:
                continue
            schedules = sorted(
                {key[1] for key in collected["curve_rows"] if key[0] == phase}
            )
            if schedules:
                notebook["cells"].append(
                    _markdown_cell(
                        "### " + PHASE_LABELS.get(phase, phase.replace("_", " ").title())
                    )
                )
            for schedule in schedules:
                keys = [key for key in collected["curve_rows"] if key[0] == phase and key[1] == schedule]
                if not keys:
                    continue
                figure, axes = plt.subplots(1, 2, figsize=(12, 4))
                for _, _, condition in sorted(keys):
                    replicas = collected["curve_rows"][(phase, schedule, condition)]
                    x, nll = _mean_curve(replicas, "current_nll")
                    _, accuracy = _mean_curve(replicas, "current_accuracy")
                    axes[0].plot(x, nll, label=condition)
                    axes[1].plot(x, accuracy, label=condition)
                axes[0].set(title="Current NLL", xlabel="Post-burn-in observations", ylabel="NLL")
                axes[1].set(title="Current accuracy", xlabel="Post-burn-in observations", ylabel="Accuracy")
                for axis in axes:
                    axis.grid(alpha=0.2)
                axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
                figure.suptitle(
                    f"{PHASE_LABELS.get(phase, phase.replace('_', ' ').title())} / "
                    f"{schedule.replace('_', ' ').title()} (partial artifacts allowed)"
                )
                cell = _code_cell(
                    "# Artifact-backed figure; no computation is run by this notebook.",
                    [_image_output(figure)],
                    phase=phase,
                )
                notebook["cells"].append(cell)

            if phase == "phase5":
                notebook["cells"].extend(_digit9_ovr_cells(collected["digit9_ovr"]))
            elif phase == "phase5r":
                notebook["cells"].extend(_phase5r_metric_cells(collected))

    notebook["cells"].extend(_phase6_cells(collected))
    notebook["cells"].extend(_phase6a_cells(collected))
    notebook["cells"].extend(_phase6b_cells(collected))

    phase6b = analyses.get("phase6b_hundred_positive")
    if phase6b is not None:
        comparisons = {row["name"]: row for row in phase6b["comparisons"]}
        primary = comparisons["primary_head_vs_full"]
        absolute = comparisons["absolute_head_vs_no_update"]
        adaptive = comparisons["subgd_adaptive_vs_full"]
        interpretation = (
            "## Interpretation status\n\n"
            "Phase 6B completed its fixed 64-replica exploratory closeout cohort "
            "with 100.1 expected observed digit-9 examples per trajectory. Head only "
            "versus full space had fixed-$.1$ precision AUC gain "
            f"{primary['precision_ref_0.1']['mean_gain']:.5f} and digit-9 recall AUC "
            f"gain {primary['nine_ovr_recall']['mean_gain']:.5f}. Against no update, "
            "the corresponding gains were "
            f"{absolute['precision_ref_0.1']['mean_gain']:.5f} and "
            f"{absolute['nine_ovr_recall']['mean_gain']:.5f}. Adaptive SubGD versus "
            "full space had precision and recall AUC gains "
            f"{adaptive['precision_ref_0.1']['mean_gain']:.5f} and "
            f"{adaptive['nine_ovr_recall']['mean_gain']:.5f}; its current-NLL, "
            "current-accuracy, and retention-NLL AUC gains were "
            f"{adaptive['current_nll']['mean_gain']:.5f}, "
            f"{adaptive['current_accuracy']['mean_gain']:.5f}, and "
            f"{adaptive['p0_nll']['mean_gain']:.5f}. Thus adaptive SubGD matched "
            "full-space precision-recall behavior with statistically resolved but "
            "practically tiny secondary improvements. Phase 6B was designed after "
            "inspecting Phases 6 and 6A and remains development evidence."
        )
    elif any(
        row["phase"] == "phase6b_hundred_positive" for row in collected["progress"]
    ):
        interpretation = (
            "## Interpretation status\n\n"
            "Phase 6B is incomplete. Its rolling curves diagnose execution and "
            "estimator health only; no condition is selected and no inferential claim "
            "is made yet. Its design target is 100.1 expected observed digit-9 "
            "examples per trajectory."
        )
    else:
        phase6a = analyses.get("phase6a_ten_positive")
        if phase6a is not None:
            comparisons = {row["name"]: row for row in phase6a["comparisons"]}
            primary = comparisons["primary_head_vs_full"]
            absolute = comparisons["absolute_head_vs_no_update"]
            interpretation = (
                "## Interpretation status\n\n"
                "Phase 6A completed its fixed 64-replica exploratory cohort. Head only "
                "versus full space had fixed-$.1$ precision AUC gain "
                f"{primary['precision_ref_0.1']['mean_gain']:.5f} and digit-9 recall AUC "
                f"gain {primary['nine_ovr_recall']['mean_gain']:.5f}. Against no update, "
                "the corresponding gains were "
                f"{absolute['precision_ref_0.1']['mean_gain']:.5f} and "
                f"{absolute['nine_ovr_recall']['mean_gain']:.5f}. Phase 6A was designed "
                "after inspecting Phase 6 and remains development evidence."
            )
        elif any(
            row["phase"] == "phase6a_ten_positive" for row in collected["progress"]
        ):
            interpretation = (
                "## Interpretation status\n\n"
                "Phase 6A is incomplete. Its rolling curves diagnose execution and "
                "estimator health only; no condition is selected and no inferential "
                "claim is made yet."
            )
        else:
            phase6 = analyses.get("phase6_low_prevalence")
            if phase6 is not None:
                comparisons = {row["name"]: row for row in phase6["comparisons"]}
                primary = comparisons["primary_head_vs_full"]
                adaptive = comparisons["subgd_adaptive_vs_full"]
                interpretation = (
                    "## Interpretation status\n\n"
                    "Phase 6 completed its prospectively frozen 64-replica low-prevalence "
                    "cohort. For head only versus full space, fixed-$.1$ precision AUC gain "
                    f"was {primary['precision_ref_0.1']['mean_gain']:.5f} (95% CI "
                    f"[{primary['precision_ref_0.1']['ci95_low']:.5f}, "
                    f"{primary['precision_ref_0.1']['ci95_high']:.5f}]) and digit-9 recall "
                    f"AUC gain was {primary['nine_ovr_recall']['mean_gain']:.5f} (95% CI "
                    f"[{primary['nine_ovr_recall']['ci95_low']:.5f}, "
                    f"{primary['nine_ovr_recall']['ci95_high']:.5f}]). For adaptive SubGD "
                    "versus full space, the corresponding gains were "
                    f"{adaptive['precision_ref_0.1']['mean_gain']:.5f} and "
                    f"{adaptive['nine_ovr_recall']['mean_gain']:.5f}. Interpret the two "
                    "co-primary outcomes jointly; an apparent tradeoff is not a scalar win."
                )
            elif any(
                row["phase"] == "phase6_low_prevalence" for row in collected["progress"]
            ):
                interpretation = (
                    "## Interpretation status\n\n"
                    "Phase 6 is incomplete. Its rolling curves diagnose execution and "
                    "estimator health only; no condition is selected and no inferential "
                    "claim is made yet."
                )
            elif "phase5r" in analyses:
                phase5r_rows = {
                    row["condition"]: row
                    for row in analyses["phase5r"]["comparisons"]
                }
                head = phase5r_rows["head_only"]
                static = phase5r_rows["static_subgd"]
                adaptive = phase5r_rows["adaptive_floor_0.1"]
                interpretation = (
                    "## Interpretation status\n\n"
                    "Phase 5R completed all 16 paired replicas after its repaired burn-in "
                    "gate passed. Head-only tuning produced the clearest current-NLL and "
                    "retention tradeoff: its current NLL gain was "
                    f"{head['current_nll_vs_full']['mean_gain']:.5f} with paired 95% CI "
                    f"[{head['current_nll_vs_full']['ci95_low']:.5f}, "
                    f"{head['current_nll_vs_full']['ci95_high']:.5f}], while its digit-9 "
                    "recall gain was "
                    f"{head['nine_ovr_recall_vs_full']['mean_gain']:.5f}. Static SubGD had "
                    f"current NLL gain {static['current_nll_vs_full']['mean_gain']:.5f} "
                    f"(95% CI [{static['current_nll_vs_full']['ci95_low']:.5f}, "
                    f"{static['current_nll_vs_full']['ci95_high']:.5f}]); the selected "
                    "adaptive condition had gain "
                    f"{adaptive['current_nll_vs_full']['mean_gain']:.5f} (95% CI "
                    f"[{adaptive['current_nll_vs_full']['ci95_low']:.5f}, "
                    f"{adaptive['current_nll_vs_full']['ci95_high']:.5f}]). Thus the "
                    "optimizer repair succeeded, but these frozen SubGD geometries did not "
                    "outperform repaired full-space learning on the digit-9-mixture path. "
                    "These reused streams remain post-diagnostic development evidence, not "
                    "fresh confirmation."
                )
            else:
                interpretation = (
                    "## Interpretation status\n\n"
                    "No condition is promoted until its frozen confirmation ledger is "
                    "complete. Intermediate curves are useful for execution health and "
                    "mechanistic study only."
                )
    notebook["cells"].append(_markdown_cell(interpretation))
    return notebook


def refresh(store: UnitStore, notebook_path: Path) -> None:
    value = json.dumps(build_notebook(store), indent=1, sort_keys=False)
    _atomic_write(notebook_path, value.encode("utf-8"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    refresh(store, arguments.notebook)


if __name__ == "__main__":
    main()
