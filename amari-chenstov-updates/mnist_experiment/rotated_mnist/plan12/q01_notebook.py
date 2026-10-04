"""Artifact-only notebook refresh for the Plan 12 MP q=.01 probe."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..artifacts import _atomic_write, _read_json, _write_json
from .artifacts import UnitStore, canonical_hash
from .q01_probe import (
    ANALYSIS_REQUIRED,
    CONTROL_CONDITIONS,
    MP_Q01_CONDITION,
    MP_Q01_RATIO,
    PROBE_REPLICAS,
    ExternalReference,
    ProbeItem,
)
from .refresh_notebook import (
    _code_cell,
    _markdown_cell,
    _png_output,
    _text_output,
    build_notebook,
)


COLORS = {
    "gauge_no_ridge": "#4c566a",
    "mp_q01_isotropic": "#007f73",
    "spectral_selector": "#c14953",
}
LABELS = {
    "gauge_no_ridge": "No added ridge",
    "mp_q01_isotropic": r"MP $q_{.01}$",
    "spectral_selector": r"MP $q_{.99}$",
}


def _trajectory_item_map(items: list[ProbeItem]) -> dict[tuple[int, str], ProbeItem]:
    return {
        (int(item.index), str(item.schedule)): item
        for item in items
        if item.action == "trajectory" and item.index is not None and item.schedule is not None
    }


def _elapsed(path: Path) -> float:
    summary = _read_json(path / "summary.json")
    return float(summary.get("total_wall_time_seconds", 0.0))


def collect_q01_progress(
    store: UnitStore,
    items: list[ProbeItem],
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
) -> dict[str, Any]:
    counts = defaultdict(int)
    elapsed = 0.0
    completed_by_schedule: dict[str, list[int]] = {"linear": [], "sigmoid": []}
    paths = _trajectory_item_map(items)
    analysis = None
    for item in items:
        path = store.completed(item.unit, item.required)
        if path is None:
            _, working = store.paths(item.unit)
            failure = store.root / "failures" / f"{canonical_hash(item.unit)}.json"
            state = "failed" if failure.is_file() else "incomplete" if working.exists() else "pending"
            counts[state] += 1
            continue
        counts["completed"] += 1
        if item.action == "trajectory":
            assert item.index is not None and item.schedule is not None
            completed_by_schedule[item.schedule].append(item.index)
            elapsed += _elapsed(path)
        elif item.action == "analysis":
            analysis = _read_json(path / "summary.json")

    curve_values: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for schedule, indices in completed_by_schedule.items():
        for index in sorted(indices):
            q_item = paths[(index, schedule)]
            q_path = store.completed(q_item.unit, q_item.required)
            assert q_path is not None
            condition_paths = {
                MP_Q01_CONDITION.name: q_path,
                **{
                    name: references[("trajectory", index, schedule, name)].path
                    for name in CONTROL_CONDITIONS
                },
            }
            for condition, path in condition_paths.items():
                for row in _read_json(path / "metrics.json"):
                    curve_values[(schedule, condition, int(row["step"]))].append(row)

    curves = []
    for (schedule, condition, step), values in sorted(curve_values.items()):
        nll = [float(value["current_nll"]) for value in values]
        accuracy = [float(value["current_accuracy"]) for value in values]
        curves.append(
            {
                "schedule": schedule,
                "condition": condition,
                "step": step,
                "observations_before_evaluation": int(values[0]["observations_before_evaluation"]),
                "angle_degrees": float(values[0]["angle_degrees"]),
                "replicas": len(values),
                "current_nll_mean": statistics.fmean(nll),
                "current_nll_se": statistics.stdev(nll) / len(nll) ** 0.5 if len(nll) > 1 else 0.0,
                "current_accuracy_mean": statistics.fmean(accuracy),
                "current_accuracy_se": (
                    statistics.stdev(accuracy) / len(accuracy) ** 0.5 if len(accuracy) > 1 else 0.0
                ),
            }
        )
    planned = len(items)
    return {
        "status": "complete" if counts["completed"] == planned else "partial",
        "planned": planned,
        "completed": counts["completed"],
        "pending": counts["pending"],
        "incomplete": counts["incomplete"],
        "failed": counts["failed"],
        "completion_fraction": counts["completed"] / planned,
        "recorded_compute_hours": elapsed / 3600.0,
        "replicas_planned_per_schedule": PROBE_REPLICAS,
        "replicas_completed": {
            key: len(value) for key, value in completed_by_schedule.items()
        },
        "ridge_ratio": MP_Q01_RATIO,
        "curves": curves,
        "analysis": analysis,
        "interpretation": (
            "This is a post hoc descriptive probe. The 16-replica result measures trajectory "
            "health and does not validate the MP selector or support outcome-driven stopping."
        ),
    }


def _probe_figure(probe: dict[str, Any]) -> plt.Figure | None:
    rows = probe["curves"]
    if not rows:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex="col")
    for column, schedule in enumerate(("linear", "sigmoid")):
        for condition in ("gauge_no_ridge", "mp_q01_isotropic", "spectral_selector"):
            values = sorted(
                (
                    row
                    for row in rows
                    if row["schedule"] == schedule and row["condition"] == condition
                ),
                key=lambda row: row["step"],
            )
            if not values:
                continue
            x = [row["observations_before_evaluation"] for row in values]
            for axis, metric in ((axes[0, column], "current_nll"), (axes[1, column], "current_accuracy")):
                y = [row[f"{metric}_mean"] for row in values]
                se = [row[f"{metric}_se"] for row in values]
                axis.plot(
                    x,
                    y,
                    color=COLORS[condition],
                    label=LABELS[condition],
                    linewidth=2.0 if condition == "mp_q01_isotropic" else 1.55,
                )
                if condition == "mp_q01_isotropic" and max(se, default=0.0) > 0:
                    axis.fill_between(
                        x,
                        [value - 1.96 * error for value, error in zip(y, se)],
                        [value + 1.96 * error for value, error in zip(y, se)],
                        color=COLORS[condition],
                        alpha=0.13,
                        linewidth=0,
                    )
        axes[0, column].set(title=f"{schedule.title()} NLL", ylabel="Mean current NLL")
        axes[1, column].set(
            title=f"{schedule.title()} accuracy",
            xlabel="Observations before evaluation",
            ylabel="Mean accuracy",
        )
        for axis in axes[:, column]:
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    completed = probe["replicas_completed"]
    figure.suptitle(
        f"MP q=.01 probe: {completed['linear']} linear and {completed['sigmoid']} sigmoid replicas"
    )
    figure.tight_layout()
    return figure


def _movement_figure(probe: dict[str, Any]) -> plt.Figure | None:
    analysis = probe.get("analysis")
    if analysis is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, result in zip(axes, analysis["schedule_results"]):
        rows = {
            value["control"]: value["cumulative_squared_displacement_ratio"]
            for value in result["comparisons"]
        }
        names = list(CONTROL_CONDITIONS)
        means = [rows[name]["mean"] for name in names]
        low = [means[i] - rows[name]["ci95_low"] for i, name in enumerate(names)]
        high = [rows[name]["ci95_high"] - means[i] for i, name in enumerate(names)]
        axis.bar(
            range(len(names)),
            means,
            yerr=[low, high],
            color=[COLORS[name] for name in names],
            width=0.62,
            capsize=4,
        )
        axis.axhline(1.0, color="#222222", linewidth=1, linestyle="--")
        axis.set_xticks(range(len(names)), [LABELS[name] for name in names])
        axis.set(title=result["schedule"].title(), ylabel=r"$q_{.01}$ / control displacement")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Cumulative squared parameter displacement")
    figure.tight_layout()
    return figure


def _analysis_brief(probe: dict[str, Any]) -> dict[str, Any] | str:
    analysis = probe.get("analysis")
    if analysis is None:
        return "Final 16-replica artifact analysis pending."
    return {
        "interpretation_contract": analysis["interpretation_contract"],
        "ridge_ratio": analysis["ridge_ratio"],
        "schedule_results": [
            {
                "schedule": result["schedule"],
                "optimizer_health": result["optimizer_health"],
                "fisher_health": result["fisher_health"],
                "comparisons": result["comparisons"],
            }
            for result in analysis["schedule_results"]
        ],
    }


def _append_probe_cells(notebook: dict[str, Any], progress_path: Path, probe: dict[str, Any]) -> None:
    common = (
        "from pathlib import Path\n"
        "import json\n"
        f"PROGRESS = Path({str(progress_path.resolve())!r})\n"
        "report = json.loads(PROGRESS.read_text(encoding='utf-8'))\n"
        "probe = report['phase5_probe']\n"
    )
    outputs = [
        _text_output(
            probe["interpretation"]
            + "\nProgress: "
            + json.dumps(probe["replicas_completed"], sort_keys=True)
        )
    ]
    figure = _probe_figure(probe)
    outputs.append(_text_output("No Phase 5 trajectories complete yet.")) if figure is None else outputs.append(_png_output(figure))
    analysis_outputs = [_text_output(json.dumps(_analysis_brief(probe), indent=2))]
    movement = _movement_figure(probe)
    analysis_outputs.append(_text_output("Movement comparison pending.")) if movement is None else analysis_outputs.append(_png_output(movement))
    cells = [
        _markdown_cell(
            "## Exploratory MP $q_{.01}$ probe\n\n"
            "The archived bootstrap maxima are evaluated with the same `higher` order-statistic "
            "convention as $q_{.99}$. The median normalized checkpoint quantile is\n\n"
            "$$\n"
            "\\kappa_t/s_t=11.818529434984821.\n"
            "$$\n\n"
            "This isotropic condition is post hoc and descriptive. It asks whether weaker spectral "
            "regularization restores adaptation while preserving some retention; it is not a repaired "
            "or calibrated selector. Shading shows pointwise $95\\%$ normal intervals for the new "
            "$q_{.01}$ trajectory mean. Linear and sigmoid remain separate conditions.\n"
        ),
        _code_cell(
            common
            + "\nprint(probe['interpretation'])\n"
            + "print('Completed replicas:', probe['replicas_completed'])\n",
            outputs,
        ),
        _markdown_cell(
            "### Paired trajectory and movement diagnostics\n\n"
            "For NLL effects, positive gain means the $q_{.01}$ condition has lower NLL. For accuracy "
            "effects, positive gain means higher accuracy. Movement ratios below one indicate less "
            "cumulative squared movement than the named control. Intervals remain exploratory.\n"
        ),
        _code_cell(
            common + "\nprint(json.dumps(probe.get('analysis'), indent=2))\n",
            analysis_outputs,
        ),
    ]
    insertion = max(0, len(notebook["cells"]) - 1)
    notebook["cells"][insertion:insertion] = cells


def _write_findings(probe: dict[str, Any], path: Path) -> None:
    if probe.get("analysis") is None:
        return
    analysis = probe["analysis"]
    lines = [
        "# Plan 12 Phase 5 Findings",
        "",
        "**Status:** Complete exploratory probe",
        "",
        f"The immutable sample contains {PROBE_REPLICAS} paired replicas per schedule at "
        f"$\\kappa_t/s_t={MP_Q01_RATIO:.6f}$.",
        "",
        analysis["interpretation_contract"],
        "",
        "## Paired Results",
        "",
    ]
    for result in analysis["schedule_results"]:
        lines.extend([f"### {result['schedule'].title()}", ""])
        for comparison in result["comparisons"]:
            nll = comparison["nll_auc_gain"]
            accuracy = comparison["accuracy_auc_gain"]
            movement = comparison["cumulative_squared_displacement_ratio"]
            lines.extend(
                [
                    f"Against `{comparison['control']}`:",
                    "",
                    f"- NLL-AUC gain: {nll['mean']:.6g} "
                    f"(95% CI [{nll['ci95_low']:.6g}, {nll['ci95_high']:.6g}]; "
                    f"{nll['favorable_count']}/{nll['replicas']} favorable).",
                    f"- Accuracy-AUC gain: {accuracy['mean']:.6g} "
                    f"(95% CI [{accuracy['ci95_low']:.6g}, {accuracy['ci95_high']:.6g}]; "
                    f"{accuracy['favorable_count']}/{accuracy['replicas']} favorable).",
                    f"- Cumulative squared displacement ratio: {movement['mean']:.6g} "
                    f"(95% CI [{movement['ci95_low']:.6g}, {movement['ci95_high']:.6g}]).",
                    "",
                ]
            )
    lines.extend(
        [
            "## Full Artifact Summary",
            "",
            "```json",
            json.dumps(analysis, indent=2),
            "```",
            "",
        ]
    )
    _atomic_write(path, ("\n".join(lines) + "\n").encode("utf-8"))


def refresh_q01_notebook(
    store: UnitStore,
    items: list[ProbeItem],
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
    base_progress_path: Path,
    notebook_path: Path,
) -> dict[str, Any]:
    base = _read_json(base_progress_path)
    probe = collect_q01_progress(store, items, references)
    phase_row = {
        key: probe[key]
        for key in (
            "planned",
            "completed",
            "pending",
            "incomplete",
            "failed",
            "completion_fraction",
            "recorded_compute_hours",
            "status",
        )
    }
    phase_row["phase"] = "phase5"
    base["phase_progress"] = [
        row for row in base["phase_progress"] if row["phase"] != "phase5"
    ] + [phase_row]
    base["phase5_probe"] = probe
    base["refreshed_at"] = datetime.now(timezone.utc).isoformat()
    base["warnings"] = [
        warning for warning in base["warnings"] if not warning.startswith("Phase 5")
    ]
    if probe["status"] != "complete":
        base["warnings"].append(
            "Phase 5 MP q=.01 probe is partial; displayed summaries are execution-health views."
        )
    progress_path = store.root / "reports/progress.json"
    _write_json(progress_path, base)
    notebook = build_notebook(base, progress_path)
    _append_probe_cells(notebook, progress_path, probe)
    notebook["metadata"]["plan12"]["phase5_probe_status"] = probe["status"]
    _atomic_write(notebook_path, (json.dumps(notebook, indent=1) + "\n").encode("utf-8"))
    _write_findings(probe, Path(__file__).parent / "PHASE5_FINDINGS.md")
    return {
        "notebook": str(notebook_path),
        "progress": str(progress_path),
        "phase5": phase_row,
    }
