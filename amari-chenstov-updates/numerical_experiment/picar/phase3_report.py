"""Read-only Phase 3 notebook cells from validated scalar artifact snapshots.

No training, model/checkpoint imports, downloads, or artifact repairs. Direct
cache reads are restricted to orchestration progress; results are sealed units.
"""

from __future__ import annotations

import base64
from collections import defaultdict
import io
from pathlib import Path

from .phase2_report import (
    PENDING, _compact, _mapping, _note, _number, _order, _pairing, _rate,
    _rows, _scalars, _table, _text, _videos,
)


_PREFIX = "phase3_"
_NOTEBOOK_DIR = Path(__file__).parent
_MOTIVATION = (
    "### Phase 3: Common Learning and FD Calibration\n\n"
    "Calibrate shared learning scales and FD resolution before selecting a retention-qualified comparison. "
    "Easy-A retention and hard-B acquisition use the same fixed 20-A/10-B development panels before and after "
    "each candidate. Repeated layouts are measurements, not independent training replicas; versions and seeds "
    "are not pooled. Selection is not a presumed FD win or a sample-savings result: fresh paired Phase 4 "
    "trajectories are still required. Only sealed units enter results; live progress is operational information."
)
_PARAMETERS = ("fd_strength", "ridge", "lr", "steps", "tau", "action_variance", "exploration_std", "epsilon", "fd_step", "pi0")
_DIAGNOSTICS = (
    "value", "definition", "reason", "count", "n", "mean", "median", "minimum", "maximum", "low", "high",
    "return_min", "return_max", "tau", "temperature", "weight_max", "max_weight", "weight_ess", "effective_count",
    "normalized_weight_max", "saturation_fraction", "drive_std_mps", "steering_std_rad", "exploration_std",
    "action_variance", "gradient_norm", "data_gradient_norm", "ltm_gradient_norm", "stm_gradient_norm",
    "fd_gradient_norm", "fd_to_data_ratio", "ratio",
    "fd_strength", "epsilon", "fd_epsilon", "resolution", "fd_resolution", "resolvable", "selected", "seed",
    "records", "return_mean", "return_std", "return_bins", "robust_spread", "selected_temperature", "spread_multiplier",
    "weight_effective_fraction", "maximum_weight_share", "weight_p90_to_p10", "log_weight_min", "log_weight_max", "cap_hits",
    "numeric_weight_safe", "teacher_logit_variance_max", "hypothetical_draws", "coverage_only_not_reward_selection",
    "selected_exploration_std", "drive_mean_mps", "drive_fraction_above_004_mps", "forward_steering_fraction",
    "left_steering_fraction", "right_steering_fraction", "large_pan_fraction", "dominant_probability_mean",
    "selected_learning_rate", "selected_optimizer_steps", "learning_rate", "optimizer_steps", "data_nll_before",
    "data_nll_after", "data_nll_decrease", "symmetric_model_kl", "finite", "seconds", "update_seconds",
    "fd_convergence_pass", "selected_fd_step", "penalty", "cosine_to_previous", "target_gradient_ratio",
    "probe_seed", "probe_count", "probe_age_weight", "scratch_state_restored", "new_observations",
    "median_update_seconds", "repetitions",
)
_CHECK_ROWS = ("rows", "temperature_rows", "exploration_rows", "optimizer_rows", "fd_strength_rows", "fit_cost_rows")
_SCOPE = ("scope", "pairs", "observations_per_arm", "transitions_per_arm", "fd_cost_multiplier_scenario",
    "control_processing_hours", "fd_processing_hours", "training_processing_hours", "evaluation_hours_at_pilot_panel_size",
    "evaluation_hours", "total_hours", "not_total_study_forecast", "fd_cost_measured")
_CANDIDATE = ("name", "seed", "fd_strength", "ridge", "old_gap_average", "old_gap_endpoint",
              "new_gain_average", "new_gain_endpoint", "checkpoints", "panel_resolution")
_COMPARISON = ("name", "seed", "retention_average_difference", "retention_endpoint_difference",
               "new_endpoint_difference", "retention_matched")


def _kind(unit):
    return unit["summary"]["kind"][len(_PREFIX):]


def _is_phase3(path):
    return any(p.name == "phase3" or p.name.startswith(("phase3-", "phase3_")) for p in (path, *path.parents))


def _campaign(path, root):
    for parent in (path, *path.parents):
        if parent.name.startswith(("development-", "development_")):
            return parent
        if parent.name == "phase3" or parent.name.startswith(("phase3-", "phase3_")):
            relative = path.relative_to(parent)
            return parent / relative.parts[0] if relative.parts and relative.parts[0] not in ("units", "orchestration") else parent
        if parent == root:
            break
    return root


def _campaigns(snapshot):
    root = Path(snapshot.get("root") or _NOTEBOOK_DIR).expanduser().resolve()
    campaigns = {}

    def group(path):
        campaign = _campaign(Path(path).resolve(), root)
        return campaigns.setdefault(campaign, {"units": [], "progress": {}})

    for unit in snapshot.get("complete_units", []):
        if str(_mapping(unit.get("summary")).get("kind", "")).startswith(_PREFIX):
            group(unit["path"])["units"].append(unit)
    for live in snapshot.get("live", []):
        path = Path(live["path"]).resolve()
        if _is_phase3(path):
            group(path)["progress"].update(live)
    if snapshot.get("root") and root.is_dir():
        from .report import _json, _version

        for path in sorted(root.rglob("orchestration/progress.json")):
            if _is_phase3(path) or root.name.startswith(("development-", "development_")):
                progress = _json(path)
                if "schema_version" in progress:
                    _version(progress["schema_version"], 1, "Phase 3 progress schema")
                group(path.parent)["progress"].update(progress)
    for data in campaigns.values():
        data["units"].sort(key=lambda u: (str(_mapping(u.get("manifest")).get("completed_at", "")), u["path"]))
    return campaigns


def _latest(data, kind):
    return next((u["summary"] for u in reversed(data["units"]) if _kind(u) == kind), {})


def _status(path, data):
    if data["progress"].get("stopped") or (path / "STOP").exists():
        return "stopped"
    final = _latest(data, "campaign")
    return final.get("status", "completed") if final else data["progress"].get("status", PENDING)


def _progress_row(path, data):
    progress, final, forecast = data["progress"], _latest(data, "campaign"), _latest(data, "forecast")

    def cost(*keys):
        for source in (final, progress, forecast):
            for key in keys:
                if _number(source.get(key)):
                    return source[key]
        return None

    def count_text(value):
        counts = _mapping(value)
        return (f"{counts['step']}/{counts['target']}"
                if all(type(counts.get(k)) is int and counts[k] >= 0 for k in ("step", "target")) else None)

    child = _mapping(progress.get("active_unit_progress"))
    child_status, child_count = child.get("status"), count_text(child.get("counts"))
    active_progress = ": ".join(v for v in (child_status, child_count) if isinstance(v, str) and v) or None
    completed = final.get("completed_trajectories", progress.get("completed_trajectories"))
    active = progress.get("active_unit")
    return {"Campaign": path.name, "Status": _status(path, data),
        "Completed trajectories": completed if type(completed) is int and completed >= 0 else None,
        "Active unit": Path(active).name if isinstance(active, str) else None, "Active progress": active_progress,
        "Budget remaining (s)": cost("remaining_budget_seconds"),
        "Remaining estimate (s)": cost("remaining_estimate_seconds"),
        "Phase 3 active (s)": cost("phase3_active_seconds", "total_active_seconds", "active_seconds"),
        "Study charge (s)": cost("cumulative_study_seconds", "study_charged_seconds"),
        "Unit progress": count_text(progress.get("counts")), "Sealed units": len(data["units"])}


def progress_rows(snapshot):
    """Optional parent progress-table hook; counts never stand in for results."""
    return [_progress_row(path, data) for path, data in sorted(_campaigns(snapshot).items(), key=_order)]


def _settings(unit):
    summary = unit["summary"]
    sources = [_mapping(summary.get("settings")), summary, _mapping(unit.get("stored_config")).get("settings", {})]
    aliases = {"lr": ("lr", "learning_rate"), "steps": ("steps", "optimizer_steps", "solve_steps"),
               "tau": ("tau", "temperature"), "action_variance": ("action_variance", "actionvariance"),
               "exploration_std": ("exploration_std", "explorationstd"), "epsilon": ("epsilon", "fd_epsilon"),
               "pi0": ("pi0", "pi")}
    result = {}
    for key in ("name", "seed", *_PARAMETERS):
        for source in sources:
            source = _mapping(source)
            found = next((source[k] for k in aliases.get(key, (key,)) if k in source and source[k] is not None), None)
            if type(found) in (str, bool) or _number(found):
                result[key] = found
                break
            if key == "action_variance" and isinstance(found, list):
                result[key] = f"{len(found)} components archived"
                break
    return result


def _identity(unit):
    settings = _settings(unit)
    return f"{_text(settings.get('name'))} / {_text(settings.get('seed'))}"


def _evaluations(unit):
    grouped = defaultdict(list)
    for evaluation in _rows(unit["summary"].get("evaluations")):
        grouped[(_text(evaluation.get("checkpoint")), _text(evaluation.get("difficulty")))].extend(_rows(evaluation.get("rows")))
    return grouped


def _panel(unit, difficulty):
    grouped = _evaluations(unit)
    before, after = (grouped.get((checkpoint, difficulty), []) for checkpoint in ("initial", "final"))
    paired, reason = _pairing(before, after)
    expected = 20 if difficulty == "easy" else 10
    coverage = f"{len(before)}/{expected} to {len(after)}/{expected} rows"
    complete = paired and len(before) == len(after) == expected and all(type(r.get("success")) is bool for r in before + after)
    return before, after, complete, reason + "; " + coverage


def _comparisons(units):
    result = []
    for unit in units:
        easy, hard = _panel(unit, "easy"), _panel(unit, "hard")
        identity = "matched seed IDs; A 20/20, B 10/10" if easy[2] and hard[2] else f"A: {easy[3]}; B: {hard[3]}"
        result.append({"Candidate / seed": _identity(unit), "Stage": _kind(unit),
            "Easy A, before to after": f"{_rate(easy[0])} to {_rate(easy[1])}",
            "Hard B, before to after": f"{_rate(hard[0])} to {_rate(hard[1])}",
            "Panel identity / coverage": identity})
    return result


def _frontier(units):
    import nbformat
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(7, 3.8), layout="constrained")
    axis = figure.subplots()
    colors = ("#347aa3", "#31845e", "#b34c55", "#896a3e", "#777777")
    descriptions, plotted = [], False
    for index, unit in enumerate(units):
        easy, hard = _panel(unit, "easy"), _panel(unit, "hard")
        label = _identity(unit)
        descriptions.append(f"{label}: A {easy[3]}; B {hard[3]}")
        if not easy[2] or not hard[2]:
            continue
        x = [sum(r["success"] for r in rows) / len(rows) for rows in easy[:2]]
        y = [sum(r["success"] for r in rows) / len(rows) for rows in hard[:2]]
        color = colors[index % len(colors)]
        axis.plot(x, y, color=color, linewidth=1, alpha=.65)
        axis.scatter(x[0], y[0], facecolors="none", edgecolors=color, s=40)
        axis.scatter(x[1], y[1], color=color, s=40, label=label)
        plotted = True
    axis.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Easy-A success (retention)",
             ylabel="Hard-B success (acquisition)", title="Verified 20-A/10-B panels: initial open, final filled")
    axis.xaxis.set_major_formatter(PercentFormatter(1.))
    axis.yaxis.set_major_formatter(PercentFormatter(1.))
    axis.grid(alpha=.2)
    if plotted:
        axis.legend(fontsize=7, loc="best")
    else:
        axis.text(.5, .5, "Verified complete panels pending", ha="center", transform=axis.transAxes)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    figure.clear()
    return nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode(),
        "text/plain": "Development retention/acquisition frontier, not a significance test. " + "; ".join(descriptions)})


def _diagnostics(unit):
    summary = unit["summary"]
    result = []
    sources = [(_kind(unit), summary)]
    sources.extend((key, summary[key]) for key in ("bootstrap_statistics", "returns", "weights", "exploration", "optimizer", "fd")
                   if isinstance(summary.get(key), dict))

    def add(label, row, status=None):
        label = next((row[k] for k in ("name", "metric", "diagnostic", "check", "scope", "stage")
                      if isinstance(row.get(k), str)), label)
        recorded = _scalars(row, _DIAGNOSTICS)
        recorded_status = row.get("status")
        if recorded_status is None:
            recorded_status = row.get("passed", row.get("fd_convergence_pass", status))
        result.append({"Check": label, "Recorded": recorded or None,
                       "Status": recorded_status})

    for label, source in sources:
        if _scalars(source, _DIAGNOSTICS):
            add(label, source)
        for key in _CHECK_ROWS:
            for row in _rows(source.get(key)):
                add(label if key == "rows" else label + " / " + key, row)
        for index, group in enumerate(_rows(source.get("fd_resolution_groups")), 1):
            for row in _rows(group.get("rows")):
                add(f"{label} / FD resolution {index}", row, group.get("passed"))
    if not result:
        add(_kind(unit), summary)
    return result


def _reference(value):
    if isinstance(value, dict):
        return _scalars(value, ("name", "seed", *_PARAMETERS, "learning_rate", "optimizer_steps")) or None
    return value if type(value) in (str, bool) or _number(value) else None


def _selection_rows(summary):
    result = [{"Field": key, "Recorded": _reference(summary.get(key, summary.get("retentionstatus") if key == "retention_status" else None))}
              for key in ("selected_control", "selected_fd", "control_learning_observed", "retention_matched",
                          "retention_status", "next_phase_authorized", "reason")]
    result[4:4] = [{"Field": key, "Recorded": value} for key, value in _scalars(summary,
        ("ready_to_propose_confirmation", "development_replica_count", "retention_tolerance", "control_retention_limit")).items()]
    strengths = summary.get("fd_strengths")
    if isinstance(strengths, list) and all(_number(v) for v in strengths):
        result.insert(4, {"Field": "fd_strengths", "Recorded": ", ".join(str(v) for v in strengths[:6])
                         + (f"; {len(strengths) - 6} more archived" if len(strengths) > 6 else "")})
    if summary.get("kind") == "phase3_selection" and "selected_fd" in summary and summary["selected_fd"] is None:
        for row in result:
            if row["Field"] == "selected_fd":
                row["Recorded"] = "Not selected"
            elif row["Field"] == "retention_matched" and row["Recorded"] is None:
                row["Recorded"] = "Not evaluated"
    return result


def _selection_evidence(summary):
    result = []
    candidates = _rows(summary.get("candidate_rows")) or _rows(summary.get("selected_control_rows"))
    for fields, rows in ((_CANDIDATE, candidates), (_COMPARISON, _rows(summary.get("comparisons")))):
        compact = []
        for row in rows:
            values = _scalars(row, fields)
            if _number(values.get("fd_strength")):
                values["fd_strength"] = str(values["fd_strength"])
            compact.append(values)
        if compact:
            result.append(_table(compact))
    if result:
        result.append(_note("Recorded old_gap values are A retention loss; new_gain values are B acquisition. "
                            "Comparison differences are FD minus the same-seed control, not independent replicas."))
    return result


def _forecast_rows(summary):
    fields = _scalars(summary, ("phase3_active_seconds", "prior_study_seconds", "study_charged_seconds",
        "remaining_to_20h_seconds", "remaining20h", "fd_cost_measured", "fd_unmeasured",
        "control_seconds_per_captured_interval", "fd_seconds_per_captured_interval", "fd_overhead_ratio",
        "fd_processing_multiplier", "next_phase_authorized"))
    return [{"Field": key, "Recorded": value} for key, value in fields.items()]


def phase3_outputs(snapshot, *, notebook_dir=None):
    """Render bounded scalar evidence; never reopen a scientific summary."""
    campaigns = sorted(_campaigns(snapshot).items(), key=_order)
    if not campaigns:
        return [_note("Phase 3 pending: no sealed units or orchestration progress.")]
    outputs = []
    for path, data in campaigns:
        units = data["units"]
        outputs.extend([_note(f"Phase 3 campaign: {path.name}; versions are not pooled."), _table([_progress_row(path, data)])])
        candidates = [u for u in units if _kind(u) == "pilot" or (_kind(u) == "probe" and u["summary"].get("evaluations"))]
        if candidates:
            outputs.append(_table(_comparisons(candidates)))
        if path != campaigns[-1][0]:
            continue
        by_kind = defaultdict(list)
        for unit in units:
            by_kind[_kind(unit)].append(unit)
        for reference in by_kind["reference_controls"]:
            outputs.append(_note("Historical ridge screen, already charged: comparison references, not new trajectories. "
                "The retention-directed repair replaces an unexecuted plasticity repair; the original results remain archived."))
            repair = _mapping(_mapping(reference.get("stored_config")).get("gentle_recipe"))
            if repair:
                outputs.append(_note("Retention-repair rationale (2026-11-12): the ridge screen did not combine "
                    "hard-task acquisition with acceptable easy-task retention. The shared repair checks gentler updates "
                    "and a working action variance matched to exploration variance; these are calibration hypotheses, "
                    "not demonstrated causes. The changed likelihood gets fresh one-time KL/NLL initialization and "
                    "Fisher from the same original demonstrations, with no new teacher data or retired-data online replay. "
                    "All versions share the original six-hour Phase 3 ceiling."))
                outputs.append(_table([{"Shared repair": _scalars(repair,
                    ("ridge", "learning_rate", "optimizer_steps", "pi", "temperature", "exploration_std", "action_variance"))}]))
            rows = [_scalars(row,_CANDIDATE) for row in _rows(reference["summary"].get("candidate_rows"))]
            outputs.append(_table(rows) if rows else _note("Historical control references: " + PENDING))
        checks = [row for kind in ("calibration", "probe") for u in by_kind[kind] for row in _diagnostics(u)]
        outputs.append(_table(checks) if checks else _note("Common scales / gradient ratios / FD resolution: " + PENDING))
        bootstrap = [{"Asset": Path(u["path"]).name, **_scalars(u["summary"], ("source", "reused", "acquired_count",
            "distinct_bootstrap_samples", "retired_count", "eligible_count", "initial_evidence_disjoint", "normalization"))}
            for u in by_kind["bootstrap"]]
        outputs.append(_table(bootstrap) if bootstrap else _note("Common bootstrap provenance: " + PENDING))
        if not candidates:
            outputs.append(_note("Candidate before/after panels: " + PENDING))
        settings = [{"Candidate / seed": _identity(u), "Settings": {k: v for k, v in _settings(u).items() if k not in ("name", "seed")} or None,
            "New captured": u["summary"].get("acquired_count"),
            "Timings (s)": _scalars(u["summary"].get("timings"), ("active_seconds", "evaluation_seconds", "fit_seconds",
                "fd_seconds", "fisher_seconds", "scoring_seconds", "sleep_seconds", "checkpoint_seconds")) or None}
            for u in candidates]
        if settings:
            outputs.append(_table(settings))
            outputs.append(_frontier([u for u in candidates if _kind(u) == "pilot"]))
        mids = [{"Candidate / seed": _identity(u), "Checkpoint": checkpoint, "Level": difficulty, "Success": _rate(rows)}
                for u in candidates for (checkpoint, difficulty), rows in _evaluations(u).items() if checkpoint not in ("initial", "final")]
        if mids:
            outputs.append(_table(mids))
        selection = _latest(data, "selection")
        outputs.append(_table(_selection_rows(selection)) if selection else _note("Recipe selection / retention match: " + PENDING))
        outputs.extend(_selection_evidence(selection))
        skipped = _rows(selection.get("budget_skips"))
        if skipped:
            outputs.append(_note("Not run because of the shared development budget; these are omissions, not failed trajectories. "
                "Admission also retains the recorded safety and closeout allowances."))
            outputs.append(_table([{"Candidate": _mapping(row.get("settings")).get("name"),
                "Estimated minutes, before safety margin": row["estimated_seconds"] / 60 if _number(row.get("estimated_seconds")) else None,
                "Remaining minutes, excluding closeout": row["remaining_seconds"] / 60 if _number(row.get("remaining_seconds")) else None,
                "Reason": row.get("reason")} for row in skipped]))
        for unit in by_kind["forecast"]:
            summary = unit["summary"]
            scopes, costs = [_scalars(row, _SCOPE) for row in _rows(summary.get("scope_rows"))], _forecast_rows(summary)
            if summary.get("fd_unmeasured") is True:
                outputs.append(_note("No FD trajectory cost was measured. The panel-time rows below are evaluation-only "
                    "forecasts, not total fresh-study times; a scratch update ratio does not measure whole-trajectory overhead."))
            outputs.append(_table(scopes) if scopes else _note("Fresh-run scope forecast: " + PENDING))
            outputs.append(_table(costs) if costs else _note("Measured operational cost: " + PENDING))
        if not by_kind["forecast"]:
            outputs.append(_note("Fresh-run forecast / measured FD cost: " + PENDING))
        media = [u for u in candidates if _rows(u["summary"].get("videos"))]
        for unit in media:
            outputs.append(_note("Preselected candidate clips: " + _identity(unit)))
            outputs.append(_videos(unit, path, notebook_dir or _NOTEBOOK_DIR))
        if not media:
            outputs.append(_note("Preselected learned-controller clips: " + PENDING))
        software = [{"Unit": _kind(u), "Recorded": _scalars(u["summary"], ("passed", "status", "test_result", "archive_sha256")) or None}
                    for kind in ("software", "provenance") for u in by_kind[kind]]
        if software:
            outputs.append(_table(software))
        outputs.append(_note("Development is not confirmation; panel repetition adds no independent training replicas. "
            "A weighting effective count is not an iid sample count. Missing evidence stays unverified; "
            "sample savings are unavailable until fresh paired Phase 4 trajectories."))
    return outputs


def display_phase3(snapshot, *, notebook_dir=None):
    from IPython.display import display

    for output in phase3_outputs(snapshot, notebook_dir=notebook_dir or Path.cwd()):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_phase3(book, snapshot, notebook_dir=None):
    """Insert/replace two cells after Phase 2, before Development/Fresh learning."""
    import nbformat

    book.cells = [c for c in book.cells if not c.metadata.get("plan19_phase3")]
    if not _campaigns(snapshot):
        return book
    directory = str(Path(notebook_dir).resolve()) if notebook_dir is not None else None
    section = [nbformat.v4.new_markdown_cell(_MOTIVATION), nbformat.v4.new_code_cell(
        "from numerical_experiment.picar.phase3_report import display_phase3\n"
        f"display_phase3(snapshot, notebook_dir={directory!r})", outputs=phase3_outputs(snapshot, notebook_dir=notebook_dir))]
    for i, cell in enumerate(section):
        cell.id, cell.metadata["plan19_phase3"] = f"plan19-phase3-{i}", True
    after = max((i + 1 for i, c in enumerate(book.cells) if c.metadata.get("plan19_phase2")), default=0)
    index = next((i for i in range(after, len(book.cells)) if book.cells[i].cell_type == "markdown"
                  and book.cells[i].source.startswith(("## Development", "## Fresh", "## Conclusion"))), len(book.cells))
    book.cells[index:index] = section
    return book


def conclusion(snapshot):
    """Read the latest recorded gates, without selecting or certifying a winner."""
    campaigns = _campaigns(snapshot)
    if not campaigns:
        return None
    path, data = max(campaigns.items(), key=_order)
    selection = _latest(data, "selection")
    learning, matched = selection.get("control_learning_observed"), selection.get("retention_matched")
    resolution = next((row["Recorded"]["fd_convergence_pass"] for unit in reversed(data["units"])
        if _kind(unit) in ("calibration", "probe") for row in _diagnostics(unit)
        if "fd_convergence_pass" in _mapping(row["Recorded"])), None)
    if learning is False:
        decision = "Common-control learning remains unresolved; this is not a negative FD result."
    elif learning is not True:
        decision = "Common-control learning and FD efficacy remain unverified."
    elif resolution is False:
        decision = "Control learning is recorded, but FD numerical resolution failed; this is not an FD efficacy result."
    elif matched is False:
        decision = "Control learning is recorded, but retention is unmatched; the selected comparison is a qualified trade-off, not a matched win."
    elif matched is not True:
        decision = "Control learning is recorded; retention matching remains unverified."
    elif selection.get("ready_to_propose_confirmation") is False:
        decision = "Control learning and retention matching are recorded, but the recorded replication or design gates still block a confirmation proposal."
    elif all(type(selection.get(k)) in (str, dict) and _reference(selection[k]) for k in ("selected_control", "selected_fd")):
        decision = "Control learning and retention matching are recorded; the selected comparison is ready for the development check-in, not a presumed FD win."
    else:
        decision = "Control learning and retention matching are recorded, but sealed recipe selection remains incomplete."
    if selection.get("fd_convergence_pass") is True and not _rows(selection.get("comparisons")):
        decision += " FD passed the numerical checks, but no paired FD efficacy trajectory was completed."
        if _rows(selection.get("budget_skips")):
            decision += " Its full trajectory was not admitted within the remaining development budget."
    candidates = _rows(selection.get("candidate_rows"))
    outcomes = ""
    if candidates:
        def percentage(value):
            return f"{100 * value:.0f}%" if _number(value) and 0 <= value <= 1 else "unverified"

        outcomes = ("| Candidate / seed | Easy A, start to end | Hard B, start to end | Average A loss (pp) |\n"
                    "|---|---|---|---:|\n")
        for row in candidates:
            label = (_text(row.get("name")) + " / " + _text(row.get("seed"))).replace("|", "\\|").replace("\n", " ")
            loss = row.get("old_gap_average")
            loss_text = f"{100 * loss:.1f}" if _number(loss) else "unverified"
            outcomes += (f"| {label} | {percentage(row.get('easy_initial'))} to {percentage(row.get('easy_final'))} "
                         f"| {percentage(row.get('hard_initial'))} to {percentage(row.get('hard_final'))} | {loss_text} |\n")
        outcomes += ("\nEach row is one development trajectory, not a pooled estimate. Average A loss is initial "
                     "success minus exposure-weighted mean success between the three measured checkpoints "
                     "(trapezoidal interpolation), in percentage points; "
                     "positive means deterioration. Retention during the stream and final retention can differ.\n\n")
    rows = _selection_rows(selection)
    if candidates:
        rows = [row for row in rows if row["Field"] in {
            "selected_control", "selected_fd", "retention_matched", "next_phase_authorized", "reason"}]
    table = "| Selection | Recorded |\n|---|---|\n" + "\n".join(
        "| " + row["Field"] + " | " + _compact(row["Recorded"]).replace("|", "\\|").replace("\n", " ") + " |" for row in rows)
    return (f"## Conclusion\n\nPhase 3 (`{path.name}`): {_status(path, data)}. " + decision + "\n\n" + outcomes + table
            + "\n\nThese are descriptive development panels, not independent training replicas; sample savings are unavailable "
            "until fresh paired Phase 4 trajectories. Review calibration, retention qualifications and measured cost at the "
            "next check-in; no automatic Phase 4 execution follows from artifact completion.")
