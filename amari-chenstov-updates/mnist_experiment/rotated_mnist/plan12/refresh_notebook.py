"""Validate Plan 12 JSON artifacts and atomically refresh its live notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
from .config import Plan12Study


DEFAULT_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12")
DEFAULT_NOTEBOOK = Path("mnist_experiment/rotated_mnist/ridge_estimator_health.ipynb")


def _phase3_scientific_audit(analyses: dict[str, Any]) -> dict[str, Any] | None:
    """Apply the preregistered deployment guardrail to the frozen selector."""
    phase1 = analyses.get("phase1_analysis")
    phase3 = analyses.get("phase3_analysis")
    if phase1 is None or phase3 is None:
        return None

    eligible = set(phase1["selection"]["isotropic"]["eligible_improving_conditions"])
    useful_ratios = sorted(
        float(row["ridge_ratio"])
        for row in phase1["aggregate"]
        if row["condition"] in eligible
    )
    selected_ratio = float(phase3["selection"]["selected_scale_ratio"])
    useful_interval = None if not useful_ratios else [useful_ratios[0], useful_ratios[-1]]
    lands_in_useful_region = bool(
        useful_interval
        and useful_interval[0] <= selected_ratio <= useful_interval[1]
    )
    pit_values = [float(value) for value in phase3.get("pseudo_tail_pit_values", [])]

    confirmations = []
    movement_diagnostics = []
    phase4 = analyses.get("phase4_analysis")
    if phase4 is not None:
        for row in phase4.get("paired_against_gauge_no_ridge", []):
            if row["condition"] != "spectral_selector":
                continue
            confirmations.append(
                {
                    "schedule": row["schedule"],
                    "nll_auc_gain": row["nll_auc_gain"],
                    "accuracy_auc_gain": row["accuracy_auc_gain"],
                }
            )
        health_rows = {
            (row["schedule"], row["condition"]): row
            for row in phase4.get("rows", [])
        }
        for schedule in sorted({key[0] for key in health_rows}):
            baseline = health_rows.get((schedule, "gauge_no_ridge"))
            spectral = health_rows.get((schedule, "spectral_selector"))
            if baseline is None or spectral is None:
                continue
            baseline_movement = float(baseline["mean_total_displacement_squared"])
            spectral_movement = float(spectral["mean_total_displacement_squared"])
            movement_diagnostics.append(
                {
                    "schedule": schedule,
                    "gauge_no_ridge_mean_total_displacement_squared": baseline_movement,
                    "spectral_selector_mean_total_displacement_squared": spectral_movement,
                    "spectral_to_no_ridge_displacement_ratio": (
                        spectral_movement / baseline_movement
                    ),
                }
            )
    phase4_condition_names = {
        condition["name"] for condition in (phase4 or {}).get("conditions", [])
    }
    no_update_control_present = "no_update" in phase4_condition_names

    if lands_in_useful_region:
        status = "not_failed_by_phase1_region_check"
        primary_reason = (
            "The selector lies inside the Phase 1 useful ridge interval; its remaining "
            "calibration requirements must still be assessed separately."
        )
    else:
        status = "failed_not_calibrated_or_deployable"
        primary_reason = (
            f"The frozen ratio {selected_ratio:.6g} lies outside the Phase 1 useful "
            f"isotropic interval {useful_interval}; this violates the preregistered "
            "selector contract."
        )

    return {
        "selector_contract_status": status,
        "primary_reason": primary_reason,
        "selected_scale_ratio": selected_ratio,
        "phase1_useful_isotropic_ratio_interval": useful_interval,
        "lands_in_useful_region": lands_in_useful_region,
        "selector_to_useful_upper_multiple": (
            None if not useful_interval else selected_ratio / useful_interval[1]
        ),
        "mean_empirical_to_theoretical_ratio": phase3.get(
            "mean_empirical_to_theoretical_ratio"
        ),
        "pseudo_tail_pit_values": pit_values,
        "pseudo_tail_pit_mean": None if not pit_values else statistics.fmean(pit_values),
        "pseudo_tail_pit_interpretation": (
            "The values are a descriptive dependence-sensitive diagnostic, not eight "
            "independent observations for a nominal coverage test. Their concentration "
            "near zero is nevertheless inconsistent with a reassuring calibration story."
            if pit_values
            else "No pseudo-tail PIT diagnostics were available."
        ),
        "phase4_predictive_confirmation": confirmations,
        "phase4_movement_diagnostic": movement_diagnostics,
        "phase4_no_update_control_present": no_update_control_present,
        "phase4_no_update_limitation": (
            "Phase 4 did not include a literal no-update trajectory. Because the strong "
            "spectral condition nearly eliminates parameter displacement, its fresh gains "
            "do not yet distinguish beneficial regularized updating from preservation of "
            "the initial model."
            if confirmations and not no_update_control_present
            else None
        ),
        "interpretation": (
            "Fresh Phase 4 predictive gains can validate this frozen ridge value as an "
            "empirically useful strong-shrinkage condition. They do not retroactively "
            "validate the MP calibration mechanism that selected it."
            if confirmations
            else "The frozen selector remains a model-failure diagnostic pending fresh confirmation."
        ),
    }


def _elapsed_seconds(path: Path) -> float:
    for name in ("summary.json", "projection.json", "validation.json", "audit.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        value = _read_json(candidate)
        for key in ("total_wall_time_seconds", "wall_time_seconds", "total_seconds"):
            if key in value:
                return float(value[key])
    return 0.0


def collect_progress(store: UnitStore) -> dict[str, Any]:
    phase_rows = []
    trajectories = []
    branches = []
    calibrations = []
    curve_values: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    analyses: dict[str, Any] = {}
    phase0: dict[str, Any] = {}
    ledger_paths = sorted((store.root / "ledgers").glob("phase*.json"))
    for ledger_path in ledger_paths:
        ledger = _read_json(ledger_path)
        if ledger.get("study_hash") != store.study.config_hash:
            raise RuntimeError(f"ledger study hash differs: {ledger_path}")
        if ledger.get("source_hashes") != store.sources:
            raise RuntimeError(f"ledger source hashes differ: {ledger_path}")
        counts = defaultdict(int)
        elapsed = 0.0
        for item in ledger["items"]:
            unit = item["unit"]
            required = tuple(item["required"])
            path = store.completed(unit, required)
            if path is None:
                _, working = store.paths(unit)
                failure = store.root / "failures" / f"{canonical_hash(unit)}.json"
                state = "failed" if failure.is_file() else "incomplete" if working.exists() else "pending"
                counts[state] += 1
                continue
            counts["completed"] += 1
            elapsed += _elapsed_seconds(path)
            action = item["action"]
            parameters = item.get("parameters", {})
            if action == "trajectory":
                row = _read_json(path / "summary.json")
                trajectories.append({**row, **parameters, "ledger_phase": ledger["phase"]})
                for step in _read_json(path / "metrics.json"):
                    key = (
                        ledger["phase"],
                        parameters["schedule"],
                        parameters["condition"]["name"],
                        int(step["step"]),
                    )
                    curve_values[key].append(step)
            elif action == "branch":
                row = _read_json(path / "summary.json")
                branches.append({**row, **parameters})
            elif action == "calibration":
                calibrations.append(_read_json(path / "summary.json"))
            elif action.endswith("analysis"):
                analyses[action] = _read_json(path / "summary.json")
            elif action == "existing_audit":
                phase0["audit"] = _read_json(path / "audit.json")
            elif action == "gauge_validation":
                phase0["validation"] = _read_json(path / "validation.json")
            elif action == "compute_projection":
                phase0["projection"] = _read_json(path / "projection.json")
            elif action == "local_benchmark":
                phase0["local_benchmark"] = _read_json(path / "summary.json")
        planned = len(ledger["items"])
        phase_rows.append(
            {
                "phase": ledger["phase"],
                "planned": planned,
                "completed": counts["completed"],
                "pending": counts["pending"],
                "incomplete": counts["incomplete"],
                "failed": counts["failed"],
                "completion_fraction": counts["completed"] / planned if planned else 1.0,
                "recorded_compute_hours": elapsed / 3600.0,
                "status": "complete" if counts["completed"] == planned else "partial",
            }
        )

    warnings = []
    phase1 = analyses.get("phase1_analysis")
    if phase1:
        for geometry in ("isotropic", "tail"):
            decision = phase1["selection"][geometry]
            if decision["classification"] != "promising_contiguous_region":
                warnings.append(
                    f"Phase 1 {geometry}: {decision['classification']} ({decision['selected_condition']})."
                )
    phase3 = analyses.get("phase3_analysis")
    if phase3 and phase3["selection"]["fallback"]:
        warnings.append(
            f"Phase 3 selector fallback: {phase3['selection']['selection_source']}."
        )
    phase3_audit = _phase3_scientific_audit(analyses)
    if phase3_audit and not phase3_audit["lands_in_useful_region"]:
        warnings.append(
            "Phase 3 scientific contract: the spectral selector is not calibrated or "
            "deployable because its frozen ratio lies outside the Phase 1 useful region."
        )
    if phase3_audit and phase3_audit["phase4_no_update_limitation"]:
        warnings.append(
            "Phase 4 control limitation: the strongly regularized selector nearly stops "
            "movement, but no literal no-update trajectory was included."
        )
    failures = sorted((store.root / "failures").glob("*.json"))
    warnings.extend(f"Execution failure: {_read_json(path)['error']}" for path in failures)
    trajectory_curves = [
        {
            "phase": phase,
            "schedule": schedule,
            "condition": condition,
            "step": step,
            "observations_before_evaluation": values[0]["observations_before_evaluation"],
            "replicas": len(values),
            "current_nll": statistics.fmean(float(value["current_nll"]) for value in values),
            "current_accuracy": statistics.fmean(float(value["current_accuracy"]) for value in values),
            "current_brier": statistics.fmean(float(value["current_brier"]) for value in values),
        }
        for (phase, schedule, condition, step), values in sorted(curve_values.items())
    ]
    return {
        "schema_version": 1,
        "study_hash": store.study.config_hash,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": hashlib.sha256(
            json.dumps(store.sources, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "phase_progress": phase_rows,
        "phase0": phase0,
        "phase1_branch_summaries": branches,
        "trajectory_summaries": trajectories,
        "trajectory_curves": trajectory_curves,
        "calibration_summaries": calibrations,
        "analyses": analyses,
        "phase3_scientific_audit": phase3_audit,
        "warnings": warnings,
        "interpretation": (
            "Partial summaries are descriptive execution-health views. Fixed-size inference "
            "is unlocked only after the corresponding ledger is complete."
        ),
    }


def _png_output(figure: plt.Figure) -> dict[str, Any]:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "output_type": "display_data",
        "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
        "metadata": {},
    }


def _text_output(text: str) -> dict[str, Any]:
    return {"output_type": "stream", "name": "stdout", "text": text.rstrip() + "\n"}


def _optional_figure_output(figure: plt.Figure | None, pending: str) -> list[dict[str, Any]]:
    return [_text_output(pending)] if figure is None else [_png_output(figure)]


def _progress_figure(progress: dict[str, Any]) -> plt.Figure:
    rows = progress["phase_progress"] or [{"phase": "waiting", "completion_fraction": 0.0}]
    figure, axis = plt.subplots(figsize=(9.0, 2.8 + 0.35 * len(rows)))
    labels = [row["phase"].replace("phase", "Phase ") for row in rows]
    values = [row["completion_fraction"] for row in rows]
    colors = ["#2a9d8f" if value == 1 else "#e9c46a" for value in values]
    axis.barh(labels, values, color=colors, height=0.58)
    for y, (value, row) in enumerate(zip(values, rows)):
        axis.text(min(value + 0.015, 0.93), y, f"{row.get('completed', 0)}/{row.get('planned', 0)}", va="center")
    axis.set(xlim=(0, 1.08), xlabel="Validated immutable units / frozen units", title="Plan 12 execution progress")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _phase1_figure(progress: dict[str, Any]) -> plt.Figure | None:
    analysis = progress["analyses"].get("phase1_analysis")
    if not analysis:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    palette = {"isotropic": "#277da1", "tail": "#d1495b"}
    for geometry in ("isotropic", "tail"):
        rows = sorted(
            (row for row in analysis["aggregate"] if row["ridge_geometry"] == geometry and row["ridge_ratio"] > 0),
            key=lambda row: row["ridge_ratio"],
        )
        if not rows:
            continue
        x = [row["ridge_ratio"] for row in rows]
        axes[0, 0].plot(x, [row["fisher_variance"] for row in rows], marker="o", label=geometry, color=palette[geometry])
        axes[0, 1].plot(x, [row["fisher_total_bias_squared"] for row in rows], marker="o", label=geometry, color=palette[geometry])
        axes[1, 0].plot(x, [row["fisher_total_mse"] for row in rows], marker="o", label=geometry, color=palette[geometry])
        axes[1, 1].plot(x, [row["current_nll"] for row in rows], marker="o", label=geometry, color=palette[geometry])
    for axis, title, ylabel in zip(
        axes.flat,
        ("Fixed-anchor batch variance", "Total squared bias", "Estimator health", "Held-out probability quality"),
        ("Reference-Fisher variance", "Reference-Fisher bias", "Reference-Fisher total MSE", "Current NLL"),
    ):
        axis.set_xscale("log")
        axis.set(title=title, xlabel=r"Ridge scale $\kappa / (\mathrm{tr}(F)/p)$", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    return figure


def _phase0_figure(progress: dict[str, Any]) -> plt.Figure | None:
    audit = progress["phase0"].get("audit")
    if not audit or not audit["rows"]:
        return None
    rows = audit["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    schedules = sorted({str(row.get("schedule") or "unknown") for row in rows})
    colors = {name: color for name, color in zip(schedules, ("#277da1", "#e76f51", "#2a9d8f"))}
    for schedule in schedules:
        subset = [row for row in rows if str(row.get("schedule") or "unknown") == schedule]
        axes[0].scatter(
            [row["effective_rank"] for row in subset],
            [row["top8_trace_fraction"] for row in subset],
            label=schedule,
            color=colors[schedule],
            alpha=0.75,
        )
        axes[1].scatter(
            [row["resolved_condition_ratio"] for row in subset if row["resolved_condition_ratio"] is not None],
            [row["gauge_action_frobenius"] for row in subset if row["resolved_condition_ratio"] is not None],
            label=schedule,
            color=colors[schedule],
            alpha=0.75,
        )
    axes[0].set(title="Represented spectral concentration", xlabel="Entropy effective rank", ylabel="Top-eight trace fraction")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(title="Conditioning and spurious gauge curvature", xlabel="Resolved condition ratio", ylabel=r"$\|\widehat F G_0\|_F$")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    return figure


def _phase0_brief(progress: dict[str, Any]) -> dict[str, Any]:
    phase0 = progress["phase0"]
    brief: dict[str, Any] = {}
    if "audit" in phase0:
        brief["historical_audit"] = {
            key: phase0["audit"][key]
            for key in ("candidate_count", "selected_count", "valid_count", "aggregate", "unrecoverable_from_historical_artifacts")
        }
    if "validation" in phase0:
        keep = (
            "raw_parameter_count",
            "chart_parameter_count",
            "removed_dimension",
            "probability_error",
            "direct_transformed_fisher_relative_error",
            "raw_exact_gauge_action_frobenius",
            "raw_compressed_gauge_action_frobenius",
            "raw_compression_relative_error",
            "chart_compression_relative_error",
        )
        brief["gauge_and_dense_validation"] = {key: phase0["validation"][key] for key in keep}
    if "projection" in phase0:
        brief["compute_projection"] = phase0["projection"]
    if "local_benchmark" in phase0:
        brief["local_benchmark"] = {
            "condition_count": len(phase0["local_benchmark"]["conditions"]),
            "mean_seconds": phase0["local_benchmark"]["mean_seconds"],
            "median_seconds": phase0["local_benchmark"]["median_seconds"],
        }
    return brief


def _trajectory_figure(progress: dict[str, Any]) -> plt.Figure | None:
    rows = progress["trajectory_curves"]
    if not rows:
        return None
    selected_phase = next(
        phase for phase in ("phase4", "phase2", "phase0") if any(row["phase"] == phase for row in rows)
    )
    subset = [row for row in rows if row["phase"] == selected_phase]
    conditions = sorted({row["condition"] for row in subset})
    palette = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(conditions)}
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex="col")
    for column, schedule in enumerate(("linear", "sigmoid")):
        for condition in conditions:
            values = sorted(
                (row for row in subset if row["schedule"] == schedule and row["condition"] == condition),
                key=lambda row: row["step"],
            )
            if not values:
                continue
            x = [row["observations_before_evaluation"] for row in values]
            axes[0, column].plot(x, [row["current_nll"] for row in values], label=condition, color=palette[condition])
            axes[1, column].plot(x, [row["current_accuracy"] for row in values], label=condition, color=palette[condition])
        axes[0, column].set(title=f"{schedule.title()} NLL", ylabel="Mean current NLL")
        axes[1, column].set(title=f"{schedule.title()} accuracy", xlabel="Observations before evaluation", ylabel="Mean accuracy")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"Available {selected_phase.title()} trajectories")
    figure.tight_layout()
    return figure


def _pi_figure(progress: dict[str, Any]) -> plt.Figure | None:
    analysis = progress["analyses"].get("phase4_analysis") or progress["analyses"].get("phase2_analysis")
    if not analysis:
        return None
    rows = analysis["rows"]
    labels = [f"{row['schedule'][:3]}\n{row['condition']}" for row in rows]
    colors = ["#277da1" if row["schedule"] == "linear" else "#e76f51" for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(max(10.0, 0.7 * len(rows)), 4.0))
    axes[0].bar(range(len(rows)), [row["shadow_pi_variance"] for row in rows], color=colors)
    axes[1].bar(range(len(rows)), [row["shadow_pi_boundary_fraction"] for row in rows], color=colors)
    for axis, title, ylabel in zip(
        axes,
        (r"Shadow $\pi^\star$ variation", "Unsupported or boundary recommendations"),
        ("Variance pooled over steps and replicas", "Fraction of decisions"),
    ):
        axis.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
        axis.set(title=title, ylabel=ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _calibration_figure(progress: dict[str, Any]) -> plt.Figure | None:
    rows = [row for row in progress["calibration_summaries"] if row["theoretical"] is not None]
    if not rows:
        return None
    x = [row["anchor_index"] for row in rows]
    empirical = [row["theoretical"]["empirical_quantile_99"] for row in rows]
    theoretical = [row["theoretical"]["edge"]["upper_quantile"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.5, 3.8))
    axis.plot(x, empirical, "o-", label="Empirical 99% tail maximum", color="#2a9d8f")
    axis.plot(x, theoretical, "s--", label="Deformed-MP edge", color="#d1495b")
    axis.set(title="Tail-spectrum edge diagnostic", xlabel="Frozen checkpoint", ylabel="Curvature scale")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _code_cell(source: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook(progress: dict[str, Any], progress_path: Path) -> dict[str, Any]:
    phase_lines = [
        "phase  planned  completed  pending  incomplete  failed  compute_h",
        *[
            f"{row['phase']:<6} {row['planned']:>7} {row['completed']:>10} {row['pending']:>8} "
            f"{row['incomplete']:>11} {row['failed']:>7} {row['recorded_compute_hours']:>10.2f}"
            for row in progress["phase_progress"]
        ],
    ]
    warning_text = "No scientific or execution warnings recorded."
    if progress["warnings"]:
        warning_text = "\n".join(f"WARNING: {item}" for item in progress["warnings"])
    phase0_brief = _phase0_brief(progress)
    common = (
        "from pathlib import Path\n"
        "import json\n"
        "import matplotlib.pyplot as plt\n"
        f"PROGRESS = Path({str(progress_path.resolve())!r})\n"
        "report = json.loads(PROGRESS.read_text(encoding='utf-8'))\n"
    )
    progress_source = common + "\nprint(report['interpretation'])\nprint(report['phase_progress'])\n"
    phase1_source = common + "\n# Plot the frozen ridge response after Phase 1 analysis becomes available.\nphase1 = report['analyses'].get('phase1_analysis')\nprint('Phase 1 pending' if phase1 is None else phase1['selection'])\n"
    trajectory_source = common + "\n# Summaries include only validated completed trajectories.\nprint(len(report['trajectory_summaries']), 'trajectory artifacts available')\n"
    calibration_source = (
        common
        + "\nphase3 = report['analyses'].get('phase3_analysis')\n"
        + "print('Phase 3 pending' if phase3 is None else phase3['selection'])\n"
        + "print(json.dumps(report.get('phase3_scientific_audit'), indent=2))\n"
    )
    cells = [
        _markdown_cell(
            "# Ridge regularization and estimator health\n\n"
            "Live Plan 12 lab notebook. Every displayed result is loaded from validated, immutable artifacts. "
            "Partial views diagnose progression and do not unlock fixed-size inference.\n"
        ),
        _markdown_cell(
            "## Reader's glossary\n\n"
            "- **Gauge:** the 25 parameter directions that add the same value to every class logit and therefore "
            "leave softmax probabilities and NLL unchanged. The gauge-fixed chart removes them.\n"
            "- **$s=\\operatorname{tr}(\\widehat F)/487$:** mean Fisher eigenvalue in the identifiable chart. "
            "$\\kappa/s$ expresses ridge strength in dimensionless average-curvature units.\n"
            "- **Top-eight trace fraction:** the fraction of total nonnegative Fisher trace carried by its eight "
            "largest eigenvalues.\n"
            "- **$U_{8,t}$:** the predictable resolved basis derived from the archived Fisher before the current "
            "batch update. The archive is represented as $\\widehat F_t=A_tA_t^T+\\operatorname{diag}(d_t)$, "
            "where $A_t$ is the deterministic Lanczos-derived factor with at most eight columns. A reduced QR "
            "factorization $A_t=U_{8,t}T_t$ orthonormalizes its span; columns whose corresponding "
            "$|T_{t,jj}|$ is below the floating-point rank tolerance are removed. Thus $U_{8,t}$ spans the "
            "archive's rank-eight component, but is not an exact dense eigendecomposition of the full "
            "low-rank-plus-diagonal matrix.\n"
            "- **Fixed-anchor batch variance:** estimator variance across independent four-example branch batches, "
            "holding the pretrained anchor state fixed, then averaged over anchors.\n"
            "- **Isotropic ridge:** $\\widehat F+\\kappa I$. **Tail ridge:** "
            "$\\widehat F+\\kappa(I-U_8U_8^T)$, leaving the leading eight directions unpenalized by the added ridge.\n"
            "- **Spectral selector:** the isotropic ridge value frozen from the median empirical 99th-percentile "
            "unresolved-spectrum maximum across eight checkpoints. It failed the calibration contract.\n"
            "- **Tail-spectrum edge diagnostic:** a comparison of empirical resampled maxima after projecting out "
            "the leading eight directions with a deformed-MP upper edge. It is not prediction calibration.\n"
            "- **Pooled recommendation variance:** variance of all shadow $\\widehat\\pi^\\star$ recommendations "
            "pooled across steps and replicas. It mixes temporal and between-replica variation.\n"
            "- **PIT:** the empirical CDF of resampled pseudo-tail maxima evaluated at a held-out observed maximum. "
            "Calibrated PIT values should be approximately uniform, with mean $.5$ under independence.\n"
        ),
        _markdown_cell("## Execution health\n"),
        _code_cell(
            progress_source,
            [_text_output("\n".join(phase_lines) + "\n\n" + warning_text), _png_output(_progress_figure(progress))],
        ),
        _markdown_cell("## Identifiability and measured cost\n"),
        _code_cell(
            common + "\n# The executed output displays a concise view; the full audit remains in the immutable JSON artifact.\nprint('Phase 0 keys:', sorted(report['phase0']))\n",
            [
                _text_output(json.dumps(phase0_brief, indent=2) if phase0_brief else "Phase 0 evidence pending."),
                *_optional_figure_output(_phase0_figure(progress), "Historical spectral audit pending."),
            ],
        ),
        _markdown_cell("## Fixed-anchor repeated-batch ridge response\n"),
        _code_cell(phase1_source, _optional_figure_output(_phase1_figure(progress), "Phase 1 response analysis pending.")),
        _markdown_cell(
            "## Propagated estimator health\n\n"
            "The propagated conditions differ through $R_t$ in the penalized Fisher "
            "$\\widehat F_t+\\kappa_tR_t$:\n\n"
            "$$\n"
            "\\begin{aligned}\n"
            "\\texttt{gauge\\_no\\_ridge}: \\quad\n"
            "& \\kappa_t=0, \\\\[4pt]\n"
            "\\texttt{isotropic\\_ridge}: \\quad\n"
            "& R_t=I, \\\\[4pt]\n"
            "\\texttt{tail\\_ridge}: \\quad\n"
            "& R_t=I-U_{8,t}U_{8,t}^\\top, \\\\[4pt]\n"
            "\\texttt{spectral\\_selector}: \\quad\n"
            "& R_t=I, \\qquad \\kappa_t=41.66\\,s_t.\n"
            "\\end{aligned}\n"
            "$$\n\n"
            "Here $s_t=\\operatorname{tr}(\\widehat F_t)/487$. The `gauge_no_ridge` "
            "condition still uses the base Fisher EWC penalty; only its added "
            "$\\kappa_tR_t$ term is zero.\n"
        ),
        _code_cell(trajectory_source, _optional_figure_output(_trajectory_figure(progress), "No completed trajectory artifacts yet.")),
        _markdown_cell("## Spectral calibration and secondary pi diagnostics\n"),
        _code_cell(
            calibration_source,
            [
                _text_output(
                    "Phase 3 scientific audit pending."
                    if progress["phase3_scientific_audit"] is None
                    else json.dumps(progress["phase3_scientific_audit"], indent=2)
                ),
                *_optional_figure_output(_calibration_figure(progress), "Spectral calibration pending."),
                *_optional_figure_output(_pi_figure(progress), "Shadow pi diagnostics pending."),
            ],
        ),
        _markdown_cell(
            "## Interpretation guardrails\n\n"
            "Linear and sigmoid schedules are separate experimental conditions. Ridge is assessed through bias, "
            "variance, total estimator error, held-out NLL, and retention before any secondary $\\pi^\\star$ result. "
            "Isolated favorable grid points and failed selectors remain visibly labeled. Fresh predictive success "
            "at a frozen selector value does not retroactively validate its calibration mechanism.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "plan12": {"study_hash": progress["study_hash"], "refreshed_at": progress["refreshed_at"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_findings(progress: dict[str, Any], package_dir: Path) -> None:
    for row in progress["phase_progress"]:
        if row["status"] != "complete":
            continue
        number = row["phase"].removeprefix("phase")
        lines = [
            f"# Plan 12 Phase {number} Findings",
            "",
            f"**Status:** Complete as of {progress['refreshed_at']}",
            "",
            f"Validated {row['completed']} of {row['planned']} frozen units; recorded compute was {row['recorded_compute_hours']:.2f} hours.",
            "",
        ]
        analysis = progress["analyses"].get(f"phase{number}_analysis")
        if analysis:
            lines.extend(["## Frozen Analysis", "", "```json", json.dumps(analysis.get("selection", analysis), indent=2), "```", ""])
        if number == "3" and progress["phase3_scientific_audit"] is not None:
            audit = progress["phase3_scientific_audit"]
            lines.extend(
                [
                    "## Scientific Contract Audit",
                    "",
                    f"**Selector status:** `{audit['selector_contract_status']}`",
                    "",
                    audit["primary_reason"],
                    "",
                    audit["pseudo_tail_pit_interpretation"],
                    "",
                    audit["interpretation"],
                    "",
                ]
            )
        if number == "4" and progress["phase3_scientific_audit"] is not None:
            audit = progress["phase3_scientific_audit"]
            if audit["phase4_predictive_confirmation"]:
                lines.extend(
                    [
                        "## Spectral Selector Interpretation",
                        "",
                        audit["interpretation"],
                        "",
                        audit["phase4_no_update_limitation"],
                        "",
                        "```json",
                        json.dumps(audit["phase4_predictive_confirmation"], indent=2),
                        "```",
                        "",
                    ]
                )
        related = [item for item in progress["warnings"] if f"Phase {number}" in item]
        if related:
            lines.extend(["## Warnings", "", *[f"- {item}" for item in related], ""])
        _atomic_write(package_dir / f"PHASE{number}_FINDINGS.md", ("\n".join(lines) + "\n").encode("utf-8"))


def refresh(store: UnitStore, notebook_path: Path) -> dict[str, Any]:
    progress = collect_progress(store)
    progress_path = store.root / "reports" / "progress.json"
    _write_json(progress_path, progress)
    notebook = build_notebook(progress, progress_path)
    _atomic_write(notebook_path, (json.dumps(notebook, indent=1) + "\n").encode("utf-8"))
    _write_findings(progress, Path(__file__).parent)
    return {"notebook": str(notebook_path), "progress": str(progress_path), "phases": progress["phase_progress"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    study = Plan12Study.from_path(args.config)
    repo_root = Path(__file__).parents[3]
    store = UnitStore(args.output_root, study, repo_root)
    print(json.dumps(refresh(store, args.notebook), indent=2))


if __name__ == "__main__":
    main()
