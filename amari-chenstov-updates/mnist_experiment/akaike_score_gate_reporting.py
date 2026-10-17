"""Artifact-only Plan 14 Phase 3 supplement, outside the frozen science package.

Use build_phase3_supplement(analysis.load_report(root)) for notebook cells, or
python -m mnist_experiment.akaike_score_gate_reporting --output-root ROOT
--notebook NOTEBOOK. The CLI requires a completed Phase 3 and holds the existing
writer lock throughout validation and replacement. Only the destination notebook
and .run.lock bookkeeping are written. Helpers consume verified scalars, never
tensors, and do not calculate covariance matrices or audit-row confidence intervals.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

from mnist_experiment.akaike_score_gate import analysis


SUPPLEMENT_VERSION = 1
SUPPLEMENT_TAG = "plan14-phase3-supplement"
CONDITIONS = ("rotation_linear", "rotation_sigmoid", "mixture_m18", "mixture_m182")
ESTIMATORS = ("model", "model_dense_same_bank", "observed", "repeated_history",
              "mean_observed_over_histories")
LABELS = {"model": "Model", "model_dense_same_bank": "Model, dense same bank",
          "observed": "Observed, within arrivals", "repeated_history": "Repeated history",
          "mean_observed_over_histories": "Mean observed over histories"}
ERROR_DOMAINS = ("gradient_noise_error", "parameter_covariance_error")
ERROR_KEYS = ("frobenius_error", "relative_frobenius_error", "relative_diagonal_error",
              "trace_error", "leading_subspace_overlap")
GAIN_PAIRS = (("forecast_gain", "realized_weighted_training_gain"),
              ("forecast_risk_reduction", "realized_next_gain"))
GLOSSARY = """### Reader Glossary

**Conditional history covariance:** the configured history refits (32 in the
principal study) resample observations and their actual labels with replacement from a finite reference
pool, preserving the saved arrival schedule, weights, frozen archive, starting
state, and EWC anchor/precision. These are not fresh experimental replicas or
final-replica efficacy evidence. Covariance includes **all assigned accepted
states**, including unresolved fits; convergence warnings accompany these errors.
When fits are unresolved, this describes accepted optimizer-state variation,
not certified MLE sampling variance.
The usual sample covariance denominator is R - 1, not the number of resolved fits.

**Model covariance** uses model-label score noise and the recorded effective
sample size. **Observed covariance** is centered within arrivals (singleton
strata are unresolved). **Repeated-history covariance** estimates gradient noise
at the fixed restricted state. Its gradient-error comparison is against itself,
so recorded zero error is not independent validation. Parameter-covariance errors
compare a local curvature-response approximation with the accepted refit states,
not a new Fisher estimand. Dense-same-bank and mean-observed estimates are labeled
separately; they are not additional history or experimental replicas.

**Forecast versus realization:** forecast_gain is compared with weighted training
NLL gain; forecast_risk_reduction is compared with next-condition panel NLL gain.
Next-condition differences include drift and selection, not pure optimism.
Terminal next gains remain missing. Only open/closed gates with both shadow fits
converged enter forecast plots. Unresolved forecasts are excluded even if numeric.
All points and medians are descriptive audit diagnostics, with **no iid or replica
confidence intervals**. Conditions are never pooled; no missing value is zero-filled.
"""


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _status(flags: list[Any]) -> bool | None:
    known = {flag for flag in flags if type(flag) is bool}
    if len(known) > 1:
        raise analysis.ArtifactError("Conflicting convergence flags for one assigned fit")
    return next(iter(known)) if known else None


def _describe(values: list[float | None]) -> dict[str, Any]:
    valid = sorted(value for value in values if value is not None)
    n = len(valid)
    median = ((valid[(n - 1) // 2] + valid[n // 2]) / 2) if n else None
    return {"n": n, "assigned": len(values), "missing": len(values) - n,
            "median": median, "minimum": valid[0] if n else None,
            "maximum": valid[-1] if n else None}


def _audits(report: dict[str, Any]) -> list[dict[str, Any]]:
    units = [unit for unit in report["units"] if unit["config"].get("phase") == "phase3"
             and unit["config"].get("kind") in {"calibration_audit", "shadow_audit"}]
    state_indices = set()
    assigned_states = _integer(report["study"].get("history_variance_states"))
    for unit in units:
        if unit["config"].get("environment") not in CONDITIONS:
            raise analysis.ArtifactError("Phase 3 supplement requires the four assigned conditions")
        if unit["config"]["kind"] == "calibration_audit":
            index = _integer(unit["config"]["index"])
            if index is None or index in state_indices:
                raise analysis.ArtifactError("Invalid or duplicate assigned calibration-state index")
            state_indices.add(index)
            # The frozen orchestrator assigns history states in four-condition order.
            if assigned_states is not None and (index >= assigned_states or
                                                unit["config"]["environment"] != CONDITIONS[index % 4]):
                raise analysis.ArtifactError("Calibration-state assignment disagrees with the frozen condition ledger")
    return sorted(units, key=lambda unit: (CONDITIONS.index(unit["config"]["environment"]),
                                          unit["config"]["kind"], unit["config"]["index"]))


def _case(unit: dict[str, Any]) -> dict[str, Any]:
    case = unit["summary"].get("case", unit["config"].get("detail", {}).get("case", {}))
    if not isinstance(case, dict):
        raise analysis.ArtifactError("Phase 3 audit case must be a scalar metadata object")
    return case


def _rows(unit: dict[str, Any], category: str) -> list[dict[str, Any]]:
    return [row for row in unit.get("audit_metrics") or [] if row.get("category") == category]


def _health(unit: dict[str, Any], study: dict[str, Any]) -> dict[str, Any]:
    case, summary = _case(unit), unit["summary"]
    frontier = case.get("frontier")
    if type(frontier) is not int or not 1 <= frontier <= 6:
        raise analysis.ArtifactError("Phase 3 audit requires a recorded frontier in 1 through 6")
    shadows = _rows(unit, "expansion_shadow")
    roles = {"restricted": [_status([summary.get("restricted_reference_converged"),
                                     *(row.get("restricted_fit_converged") for row in shadows)])]}
    for layers in (1, 2):
        roles[f"expanded_{layers}"] = ([_status([row.get("expanded_fit_converged")
                                                for row in shadows if row.get("expansion_layers") == layers])]
                                         if layers <= 6 - frontier else [])
    repeats = 0
    if unit["config"]["kind"] == "calibration_audit":
        repeats = _integer(summary.get("history_repeats"))
        configured = _integer(study.get("history_variance_repeats"))
        if repeats is not None and configured is not None and repeats != configured:
            raise analysis.ArtifactError("Assigned history-refit denominator disagrees with the frozen study")
        if repeats is None:
            repeats = configured
        if repeats is None or repeats < 2:
            raise analysis.ArtifactError("Missing assigned history-refit denominator")
    history = _rows(unit, "history_fit")
    by_repeat = {}
    for row in history:
        repeat = _integer(row.get("repeat"))
        if repeat is None or repeat >= repeats or repeat in by_repeat:
            raise analysis.ArtifactError("Invalid or duplicate assigned history repeat")
        by_repeat[repeat] = row
    roles["history"] = [_status([by_repeat.get(repeat, {}).get("converged")])
                        for repeat in range(repeats)]
    counts = {role: {"assigned": len(flags), "resolved": flags.count(True),
                     "unresolved": flags.count(False), "missing": flags.count(None)}
              for role, flags in roles.items()}
    return {"condition": unit["config"]["environment"], "unit": unit["path"],
            "index": unit["config"]["index"], "kind": unit["config"]["kind"],
            "step": case.get("step"), "frontier": frontier, "roles": roles, "counts": counts,
            "history_rows": len(history), "covariance_denominator": repeats - 1 if repeats >= 2 else None,
            "restricted_residual_max": _describe([analysis.metric_value(row, "restricted_residual_max")
                                                   for row in history]),
            "summary_unresolved_history_fits": summary.get("unresolved_history_fits")}


def _covariance_rows(unit: dict[str, Any], study: dict[str, Any]) -> list[dict[str, Any]]:
    health = _health(unit, study)
    history = health["counts"]["history"]
    certified = health["roles"]["restricted"] == [True] and history["resolved"] == history["assigned"]
    return [{**row, "_recorded_fits_resolved": certified} for row in _rows(unit, "covariance_calibration")]


def summarize_phase3(report: dict[str, Any]) -> dict[str, Any]:
    """Describe all assignments, without pooling conditions or filtering refits.

    Fit roles count unique fits, not duplicated covariance-estimator rows. Missing
    rows/flags remain unknown. Coverage of unsealed shadow fits cannot be inferred
    from scalar data and is reported as unavailable, not zero assignments.
    """
    units, study = _audits(report), report["study"]
    health = [_health(unit, study) for unit in units]
    warnings = set()
    coverage, fit_counts, forecasts, covariances = [], [], [], []
    assigned_states = _integer(study.get("history_variance_states"))
    configured_repeats = _integer(study.get("history_variance_repeats"))
    for condition in CONDITIONS:
        members = [unit for unit in units if unit["config"]["environment"] == condition]
        states = [unit for unit in members if unit["config"]["kind"] == "calibration_audit"]
        planned = (sum(CONDITIONS[index % 4] == condition for index in range(assigned_states))
                   if assigned_states is not None else None)
        missing_states = max(planned - len(states), 0) if planned is not None else None
        coverage.append({"condition": condition, "calibration_states_assigned": planned,
                         "calibration_states_sealed": len(states), "calibration_states_missing": missing_states,
                         "shadow_units_sealed": len(members) - len(states)})
        if missing_states:
            warnings.add("Missing calibration states: configured history assignments remain missing, not resolved.")
        condition_health = [row for row in health if row["condition"] == condition]
        for role in ("restricted", "expanded_1", "expanded_2", "history"):
            counts = {name: sum(row["counts"][role][name] for row in condition_health)
                      for name in ("assigned", "resolved", "unresolved", "missing")}
            if role == "history" and missing_states and configured_repeats is not None:
                counts["assigned"] += missing_states * configured_repeats
                counts["missing"] += missing_states * configured_repeats
            fit_counts.append({"condition": condition, "role": role, **counts})
        for row in condition_health:
            if any(count["unresolved"] or count["missing"] for count in row["counts"].values()):
                warnings.add("Unresolved or missing fits: covariance retains accepted states; forecasts require resolved shadow fits.")
            declared = _integer(row["summary_unresolved_history_fits"])
            if declared is not None and declared != row["counts"]["history"]["unresolved"]:
                warnings.add("Summary unresolved-history count differs from recorded flags; missing flags are not inferred.")
        # An expected scalar slot exists for each estimator/layer, even if absent.
        for estimator in ("model", "observed", "repeated_history"):
            for layers in (1, 2):
                slots = []
                for unit in members:
                    if layers > 6 - _case(unit)["frontier"]:
                        continue
                    if estimator == "repeated_history" and unit["config"]["kind"] != "calibration_audit":
                        continue
                    matches = [row for row in _rows(unit, "expansion_shadow")
                               if row.get("covariance") == estimator and row.get("expansion_layers") == layers]
                    if len(matches) > 1:
                        raise analysis.ArtifactError("Duplicate expansion-shadow scalar slot")
                    slots.append(matches[0] if matches else None)
                resolved = [row for row in slots if row is not None and _forecast_resolved(row)]
                pairs = {}
                for predicted, realized in GAIN_PAIRS:
                    values = [(analysis.metric_value(row, predicted), analysis.metric_value(row, realized))
                              for row in resolved]
                    paired = [(x, y) for x, y in values if x is not None and y is not None]
                    pairs[predicted] = {"paired": len(paired), "missing_values": len(values) - len(paired),
                                        "forecast": _describe([x for x, _ in paired]),
                                        "realized": _describe([y for _, y in paired])}
                forecasts.append({"condition": condition, "covariance": estimator, "expansion_layers": layers,
                                  "assigned": len(slots), "recorded": sum(row is not None for row in slots),
                                  "resolved": len(resolved), "missing_rows": slots.count(None),
                                  "unresolved": sum(row is not None and not _forecast_resolved(row) for row in slots),
                                  "pairs": pairs})
        for estimator in ESTIMATORS:
            slots = []
            for unit in states:
                matches = [row for row in _covariance_rows(unit, study)
                           if row.get("covariance") == estimator]
                if len(matches) > 1:
                    raise analysis.ArtifactError("Duplicate covariance-calibration scalar slot")
                slots.append(matches[0] if matches else None)
            available = [row for row in slots if row is not None and row.get("status") == "available"]
            flagged = sum(not _covariance_certified(row) for row in available)
            if flagged:
                warnings.add("Available covariance errors include unresolved accepted states or unknown convergence; marked x in plots.")
            for domain in ERROR_DOMAINS:
                errors = [row.get(domain) if isinstance(row.get(domain), dict) else {} for row in available]
                covariances.append({"condition": condition, "covariance": estimator, "domain": domain,
                                    "assigned": planned if planned is not None else len(slots),
                                    "recorded": sum(row is not None for row in slots), "available": len(available),
                                    "unavailable": sum(row is not None and row.get("status") != "available" for row in slots),
                                    "missing_rows": slots.count(None) + (missing_states or 0),
                                    "uncertified_available": flagged,
                                    "all_assigned_states_included": all(row.get("includes_all_assigned_history_fits") is True
                                                                        for row in available) if available else None,
                                    "errors": {key: _describe([analysis.metric_value(error, key) for error in errors])
                                               for key in ERROR_KEYS}})
    phase = next((row for row in report.get("phase_progress", []) if row["phase"] == "phase3"), {})
    incomplete = phase.get("incomplete")
    if incomplete:
        warnings.add("Unsealed Phase 3 units are not read; their shadow-fit assignments are unknown.")
    return {"coverage": coverage, "fit_health": health, "fit_counts": fit_counts,
            "forecasts": forecasts, "covariances": covariances, "incomplete_phase3_units": incomplete,
            "warnings": sorted(warnings)}


def _forecast_resolved(row: dict[str, Any]) -> bool:
    return (row.get("gate_status") in {"open", "closed"}
            and row.get("restricted_fit_converged") is True and row.get("expanded_fit_converged") is True)


def _covariance_certified(row: dict[str, Any]) -> bool:
    return (row.get("restricted_reference_converged") is True
            and row.get("all_history_fits_converged") is True
            and row.get("includes_all_assigned_history_fits") is True
            and row.get("_recorded_fits_resolved") is True)


def phase3_figures(report: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Small CPU figures of recorded scalars; callers close yielded figures."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    units = _audits(report)
    colors = dict(zip(ESTIMATORS, ("#167c80", "#646b75", "#d17718", "#8a4e99", "#3d7340")))
    markers = {"model": "o", "observed": "s", "repeated_history": "^"}
    for predicted, realized in GAIN_PAIRS:
        figure, axes = plt.subplots(4, 2, figsize=(12, 13), squeeze=False)
        try:
            for i, condition in enumerate(CONDITIONS):
                for j, layers in enumerate((1, 2)):
                    axis = axes[i, j]
                    rows = [row for unit in units if unit["config"]["environment"] == condition
                            for row in _rows(unit, "expansion_shadow")
                            if row.get("expansion_layers") == layers and _forecast_resolved(row)]
                    plotted = 0
                    for estimator, marker in markers.items():
                        pairs = [(analysis.metric_value(row, predicted), analysis.metric_value(row, realized))
                                 for row in rows if row.get("covariance") == estimator]
                        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                        if pairs:
                            axis.scatter([x for x, _ in pairs], [y for _, y in pairs],
                                         color=colors[estimator], marker=marker, s=23, alpha=.7)
                            plotted += len(pairs)
                    axis.set_title(f"{condition} | +{layers} layer{'s' if layers == 2 else ''} | paired rows={plotted}", fontsize=10)
                    axis.set_xlabel("Forecast weighted training gain (nat)" if predicted == "forecast_gain"
                                    else "Forecast local risk reduction (nat)", fontsize=9)
                    axis.set_ylabel("Realized weighted training gain (nat)" if realized == "realized_weighted_training_gain"
                                    else "Realized next-condition NLL gain (nat)", fontsize=9)
                    if plotted:
                        # Keep the identity guide from changing the observed data ranges.
                        xlim, ylim = axis.get_xlim(), axis.get_ylim()
                        lower, upper = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
                        if lower <= upper:
                            axis.plot([lower, upper], [lower, upper], color=".6", linestyle=":", linewidth=1)
                        axis.set_xlim(xlim)
                        axis.set_ylim(ylim)
                    else:
                        axis.text(.5, .5, "No resolved paired measurements", ha="center", va="center", transform=axis.transAxes)
                    axis.grid(alpha=.2)
            figure.legend([Line2D([], [], color=colors[name], marker=markers[name], linestyle="") for name in markers],
                          [LABELS[name] for name in markers], loc="upper center", ncol=3, fontsize=9)
            figure.suptitle("Local shadow diagnostics; no audit-row confidence intervals", y=.965, fontsize=12)
            figure.tight_layout(rect=(0, 0, 1, .95))
        except BaseException:
            plt.close(figure)
            raise
        yield predicted, figure
    covariance_rows = [(unit["config"]["environment"], row) for unit in units
                       if unit["config"]["kind"] == "calibration_audit"
                       for row in _covariance_rows(unit, report["study"])]
    figure, axes = plt.subplots(4, 2, figsize=(13, 13), squeeze=False)
    try:
        for i, condition in enumerate(CONDITIONS):
            for j, domain in enumerate(ERROR_DOMAINS):
                axis = axes[i, j]
                plotted = 0
                for k, estimator in enumerate(ESTIMATORS):
                    rows = [row for recorded_condition, row in covariance_rows if recorded_condition == condition
                            and row.get("covariance") == estimator and row.get("status") == "available"]
                    for row in rows:
                        metrics = row.get(domain)
                        value = analysis.metric_value(metrics, "relative_frobenius_error") if isinstance(metrics, dict) else None
                        if value is not None:
                            axis.scatter(value, k, color=colors[estimator], s=30,
                                         marker="o" if _covariance_certified(row) else "x", alpha=.75)
                            plotted += 1
                axis.set_yticks(range(len(ESTIMATORS)), [LABELS[name] for name in ESTIMATORS], fontsize=8)
                axis.set_ylim(-.5, len(ESTIMATORS) - .5)
                axis.set_xlabel("Relative Frobenius error to finite conditional reference", fontsize=9)
                axis.set_title(f"{condition} | {'Gradient noise' if j == 0 else 'Parameter response'} | values={plotted}", fontsize=10)
                if not plotted:
                    axis.text(.5, .5, "No available error measurements", ha="center", transform=axis.transAxes)
                axis.grid(alpha=.2)
        figure.legend([Line2D([], [], color=".3", marker=marker, linestyle="") for marker in ("o", "x")],
                      ["All refits and restricted reference converged", "Unresolved accepted states / unknown flags"],
                      loc="upper center", ncol=2, fontsize=9)
        figure.suptitle("Conditional covariance errors, all accepted states retained; no replica CIs", y=.965, fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .95))
    except BaseException:
        plt.close(figure)
        raise
    yield "conditional-covariance-errors", figure
    summary = summarize_phase3(report)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    try:
        for axis, condition in zip(axes.flat, CONDITIONS):
            health = [row for row in summary["fit_health"] if row["condition"] == condition
                      and row["kind"] == "calibration_audit"]
            left = [0] * len(health)
            for name, color in (("resolved", "#3d7340"), ("unresolved", "#b54b43"), ("missing", "#989da4")):
                values = [row["counts"]["history"][name] for row in health]
                axis.barh(range(len(health)), values, left=left, color=color, label=name.capitalize())
                left = [x + y for x, y in zip(left, values)]
            axis.set_yticks(range(len(health)), [f"State {row['index']} | step {row['step']} | frontier {row['frontier']}" for row in health], fontsize=8)
            axis.set_xlabel("Assigned history refits (counts, not experimental replicas)", fontsize=9)
            axis.set_title(condition, fontsize=10)
            if not health:
                axis.text(.5, .5, "No sealed history-refit states", ha="center", transform=axis.transAxes)
            axis.grid(axis="x", alpha=.2)
        repeats = report["study"].get("history_variance_repeats", "recorded")
        figure.suptitle(f"{repeats}-refit fit health; missing and unresolved fits retained", fontsize=12)
        figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
        figure.tight_layout(rect=(0, .04, 1, .95))
    except BaseException:
        plt.close(figure)
        raise
    yield "history-fit-health", figure


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def escaped(value):
        return _display(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(escaped(value) for value in row) + " |" for row in rows)])


def phase3_markdown(report: dict[str, Any]) -> str:
    summary = summarize_phase3(report)
    text = ["### Phase 3 Scalar Diagnostic Supplement", GLOSSARY,
            "### Assignment Coverage\n\n" + _table(
                ["Condition", "History states assigned", "Sealed", "Missing", "Sealed shadow units"],
                [[row[name] for name in ("condition", "calibration_states_assigned", "calibration_states_sealed",
                                         "calibration_states_missing", "shadow_units_sealed")] for row in summary["coverage"]]),
            "All unique fit roles in sealed audits are counted below, including failed fits. Covariance-estimator rows do not multiply fit counts. "
            "Missing calibration states contribute configured history assignments. Unsealed shadow-fit counts are unavailable. "
            f"Unsealed Phase 3 units: {_display(summary['incomplete_phase3_units'])}.",
            "### Fit Counts\n\n" + _table(["Condition", "Fit role", "Assigned", "Resolved", "Unresolved", "Missing row/flag"],
                [[row[name] for name in ("condition", "role", "assigned", "resolved", "unresolved", "missing")]
                 for row in summary["fit_counts"]]),
            "### History Fit Health\n\n" + _table(
                ["Condition", "State", "Step", "Frontier", "Rows / assigned", "Resolved", "Unresolved", "Missing", "R - 1", "Max recorded residual"],
                [[row["condition"], row["index"], row["step"], row["frontier"],
                  f"{row['history_rows']} / {row['counts']['history']['assigned']}", row["counts"]["history"]["resolved"],
                  row["counts"]["history"]["unresolved"], row["counts"]["history"]["missing"],
                  row["covariance_denominator"], row["restricted_residual_max"]["maximum"]]
                 for row in summary["fit_health"] if row["kind"] == "calibration_audit"])]
    for predicted, realized in GAIN_PAIRS:
        text.append(f"### {predicted} vs {realized}\n\n" + _table(
            ["Condition", "Covariance", "Layers", "Assigned", "Resolved", "Unresolved", "Missing rows", "Paired", "Missing values", "Forecast median", "Realized median"],
            [[row["condition"], LABELS[row["covariance"]], row["expansion_layers"], row["assigned"], row["resolved"],
              row["unresolved"], row["missing_rows"], row["pairs"][predicted]["paired"], row["pairs"][predicted]["missing_values"],
              row["pairs"][predicted]["forecast"]["median"], row["pairs"][predicted]["realized"]["median"]]
             for row in summary["forecasts"]]))
    text.append("### Conditional Covariance Errors\n\nRelative Frobenius medians/ranges are descriptive, not efficacy estimates. "
                "Other recorded scalar matrix-error keys remain available through summarize_phase3.\n\n" + _table(
        ["Condition", "Estimator", "Error domain", "Assigned", "Available", "Unavailable", "Missing rows", "Uncertified available", "Finite / available", "Median", "Range"],
        [[row["condition"], LABELS[row["covariance"]], row["domain"], row["assigned"], row["available"],
          row["unavailable"], row["missing_rows"], row["uncertified_available"],
          f"{row['errors']['relative_frobenius_error']['n']} / {row['available']}",
          row["errors"]["relative_frobenius_error"]["median"],
          f"{_display(row['errors']['relative_frobenius_error']['minimum'])} to {_display(row['errors']['relative_frobenius_error']['maximum'])}"]
         for row in summary["covariances"]]))
    if summary["warnings"]:
        text.append("### Warnings\n\n" + "\n".join("- " + warning for warning in summary["warnings"]))
    return "\n\n".join(text)


def build_phase3_supplement(report: dict[str, Any], *, render: bool = True) -> list[dict[str, Any]]:
    """Return two stable-tagged notebook cells. Requires a verified scalar report.

    The code cell reuses the notebook's existing report variable. render=False
    skips embedded outputs; executing that cell only summarizes/plots artifacts.
    """
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {"tags": [SUPPLEMENT_TAG], "plan14_phase": "phase3",
                "phase3_supplement_version": SUPPLEMENT_VERSION, "phase3_supplement_source_sha256": source_hash}
    summary = summarize_phase3(report)
    markdown = phase3_markdown(report) + f"\n\nSupplement version: {SUPPLEMENT_VERSION}; source SHA256: `{source_hash}`."
    cells = [{"cell_type": "markdown", "id": SUPPLEMENT_TAG + "-summary",
              "metadata": {**metadata, "assignment_coverage": summary["coverage"], "fit_counts": summary["fit_counts"]},
              "source": markdown},
             {"cell_type": "code", "id": SUPPLEMENT_TAG + "-figures", "metadata": dict(metadata),
              "execution_count": None, "outputs": [],
              "source": "from mnist_experiment import akaike_score_gate_reporting as phase3_view\n"
                        "from IPython.display import display\nimport matplotlib.pyplot as plt\n"
                        "for name, figure in phase3_view.phase3_figures(report):\n"
                        "    try:\n        display(figure)\n    finally:\n        plt.close(figure)"}]
    if render:
        import matplotlib.pyplot as plt
        figures = phase3_figures(report)
        try:
            for name, figure in figures:
                try:
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight",
                                   metadata={"Software": "Plan14 Phase3 scalar supplement"})
                    cells[1]["outputs"].append({"output_type": "display_data", "metadata": {},
                                               "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                                                        "text/plain": f"Phase 3 supplement: {name}"}})
                finally:
                    plt.close(figure)
        finally:
            figures.close()
    return cells


def insert_phase3_supplement(notebook: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace only our tagged cells, directly before the Phase 4 heading."""
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise analysis.ArtifactError("Expected an existing nbformat-4 report notebook")
    kept = [cell for cell in notebook["cells"] if SUPPLEMENT_TAG not in cell.get("metadata", {}).get("tags", [])]
    headings = {3: [], 4: []}
    for index, cell in enumerate(kept):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "markdown":
            for phase in headings:
                if re.match(rf"^##\s+Phase {phase}(?:\s|:|$)", source):
                    headings[phase].append(index)
    if any(len(indices) != 1 for indices in headings.values()) or headings[3][0] >= headings[4][0]:
        raise analysis.ArtifactError("Expected unique ordered Phase 3 and Phase 4 headings")
    if any(SUPPLEMENT_TAG not in cell.get("metadata", {}).get("tags", []) for cell in cells):
        raise analysis.ArtifactError("Supplement cells must have the stable ownership tag")
    ids = [cell.get("id") for cell in [*kept, *cells] if cell.get("id") is not None]
    if len(ids) != len(set(ids)):
        raise analysis.ArtifactError("Supplement cell ID collides with an existing cell")
    at = headings[4][0]
    return {**notebook, "cells": kept[:at] + cells + kept[at:]}


def _require_completed_phase3(report: dict[str, Any]) -> None:
    progress = (report.get("progress") or {}).get("phases", {}).get("phase3", {})
    measured = next((row for row in report["phase_progress"] if row["phase"] == "phase3"), {})
    decision = report.get("decisions", {}).get("phase3_calibration", {})
    sealed = [unit for unit in report["units"] if unit["config"].get("phase") == "phase3"
              and unit["config"].get("kind") == "decision"
              and unit["config"].get("policy") == "phase3_calibration"]
    if progress.get("status") != "completed" or measured.get("incomplete") != 0 or len(sealed) != 1:
        raise analysis.ArtifactError("Supplement CLI requires completed Phase 3 with a sealed calibration decision and no incomplete Phase 3 units")
    if sealed[0].get("decision") != decision:
        raise analysis.ArtifactError("Phase 3 sealed calibration decision disagrees with report")
    assigned = decision.get("audit_paths")
    actual = {Path(unit["path"]).resolve() for unit in _audits(report)}
    if (not isinstance(assigned, list) or not assigned or any(not isinstance(path, str) for path in assigned)
            or len(assigned) != len(set(assigned)) or {Path(path).resolve() for path in assigned} != actual):
        raise analysis.ArtifactError("Completed Phase 3 must include every assigned sealed audit path")
    expected = _integer(report["study"].get("history_variance_states"))
    if expected is None or sum(unit["config"]["kind"] == "calibration_audit" for unit in _audits(report)) != expected:
        raise analysis.ArtifactError("Completed Phase 3 is missing assigned history-variance states")


def supplement_notebook(root: Path, notebook_path: Path) -> Path:
    """Lock, validate, render on CPU, then atomically replace only the notebook.

    The original run_lock imports shared artifact dependencies, but no tensor
    loader, numerical kernel, training entry point, or CUDA API is called here.
    A paused later phase may remain incomplete; Phase 3 itself must be complete.
    """
    root, notebook_path = Path(root).resolve(), Path(notebook_path).resolve()
    if not root.is_dir() or not (root / "study.json").is_file():
        raise analysis.ArtifactError("Expected an existing study root; reporting never creates a study")
    if notebook_path.is_relative_to(root) or any((parent / "COMPLETED").is_file() for parent in notebook_path.parents):
        raise analysis.ArtifactError("Supplement destination must be outside study artifacts and completed units")
    from mnist_experiment.akaike_score_gate.artifacts import run_lock

    with run_lock(root):
        report = analysis.load_report(root)
        _require_completed_phase3(report)
        notebook = analysis._object(notebook_path)
        # Check headings before CPU rendering and before touching the destination.
        insert_phase3_supplement(notebook, [])
        import matplotlib
        matplotlib.use("Agg")
        document = insert_phase3_supplement(notebook, build_phase3_supplement(report))
        payload = (json.dumps(document, indent=1, allow_nan=False) + "\n").encode("utf-8")
        descriptor, name = tempfile.mkstemp(prefix=f".{notebook_path.name}.", suffix=".tmp", dir=notebook_path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, notebook_path)
        finally:
            temporary.unlink(missing_ok=True)
    return notebook_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args(argv)
    print(supplement_notebook(args.output_root, args.notebook))


if __name__ == "__main__":
    main()
