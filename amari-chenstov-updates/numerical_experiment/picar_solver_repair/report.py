"""Artifact-only Plan 19 Phase 3 Fix 2 reporting.

Parent integration: build the historical notebook, then call
``augment_notebook(book, root=shared_root, notebook_dir=output.parent)``.
``build_notebook`` performs both operations, defaulting to a separate notebook.

Schema 1 uses the existing sealed unit envelope. ``solver_recorded`` summaries
contain settings, solver_recipe, condition, steps_per_event, completed_events,
fit_seconds, validation.initial/final.old_nll/live_nll, timing and scalar event
rows. Optional per_method/segment summaries use the same fields. Selection is
recorded in solver_selection; solver_pilot retains historical fit_metrics,
timings and evaluations. No checkpoint, tensor, event payload or media is read.
"""

from __future__ import annotations

import html
import copy
import hashlib
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.phase2_report import PENDING, _mapping, _note, _number, _rows, _table, _videos
from numerical_experiment.picar_initialization import report as initialreport


VERSION = "plan19-phase3-fix2-v3"
CAMPAIGN = "solver-fix2-v3"
CAMPAIGN_VERSIONS = {"solver-fix2-v1": "plan19-phase3-fix2-v1",
                     "solver-fix2-v2": "plan19-phase3-fix2-v2", CAMPAIGN: VERSION}
DEFAULT_NOTEBOOK = Path(__file__).with_name("plan19_solver_results.ipynb")
RECIPES = {"proximal_1": 1, "lagged_1": 1, "proximal_3": 3}
_TAG = "plan19_phase3_fix2"
_BASE_CACHE = {}
_KINDS = {"solver_recorded", "solver_comparison", "solver_selection", "solver_pilot", "solver_dataset",
          "solver_campaign", "solver_software", "solver_provenance", "solver_integrity",
          "solver_acceptance", "solver_readiness", "solver_timing", "solver_supervisor"}
_INTRO = (
    "### Phase 3 Fix 2: Curvature-Aware Updates and Solver Cost\n\n"
    "The historical one-step explicit/recentered solver reported zero EWC penalty in 2,134/2,170 "
    "events. This exposed a solver blind spot, not the cause of every retention or navigation failure. "
    "The repair preserves the recurrent centered information summary: a centered penalty can act "
    "through curvature while its initial gradient is zero. No archived-score estimator is added. "
    "Compare two one-step recipes and three proximal steps for quality and measured cost, then use "
    "one selected recipe for four paired development trajectories. Recorded-history forks are not "
    "counterfactual robot rollouts; development is not confirmation. Both live conditions use the same "
    "trainable STM/LTM banks, replay, diagonal EWC and ridge; FD adds the age-weighted STM sensitivity penalty. "
    "Only sealed results enter plots. "
    "Both early and late segments restart from the common pre-stream initializer; the late segment "
    "is not a historical continuation or a restored late-policy checkpoint. "
    "Committed live counters are incomplete operational progress; missing values remain pending. "
    "The separate ceiling is nine additional active compute hours. Full derivations remain in plan19.md; "
    "full traces remain in linked artifacts."
)


def _count(value):
    return type(value) is int and value >= 0


def _require(condition, message):
    if not condition:
        raise shared.ArtifactError("Phase 3 Fix 2 schema-incompatible: " + message)


def _campaign_root(root=None):
    path = shared.resolve_root(root)
    if path.name.startswith("solver-fix2-v"):
        _require(path.name in CAMPAIGN_VERSIONS, "unsupported solver campaign " + path.name)
        return path
    return path / "phase3" / CAMPAIGN


def _path_version(path):
    name = next((part for part in reversed(Path(path).parts) if part.startswith("solver-fix2-v")), None)
    if name is not None:
        _require(name in CAMPAIGN_VERSIONS, "unsupported solver campaign " + name)
    return CAMPAIGN_VERSIONS.get(name)


def _version(value, expected=None):
    if "version" in value:
        _require(value["version"] in CAMPAIGN_VERSIONS.values()
                 and (expected is None or value["version"] == expected),
                 "unexpected solver version; do not pool campaigns")


def _recipe(value, *, optional=False):
    _require((optional and value is None) or (isinstance(value, str) and value in RECIPES),
             "unknown solver recipe " + repr(value))


def _method_rows(summary):
    """Normalize compact summaries without turning trace rows into narrative."""
    value = summary.get("per_method")
    if isinstance(value, dict):
        _require(all(isinstance(row, dict) for row in value.values()), "per_method requires objects")
        rows = [{"solver_recipe": key, **row} for key, row in value.items()]
    elif isinstance(value, list):
        _require(all(isinstance(row, dict) for row in value), "per_method requires objects")
        rows = value
    else:
        _require(value is None, "per_method requires a list or mapping")
        segments = summary.get("segments")
        rows = segments if isinstance(segments, list) and segments and all(
            isinstance(s, dict) and any(k in s for k in ("fit_seconds", "seconds", "validation", "validation_final"))
            for s in segments) else [{}]
    result = []
    for row in rows:
        settings = {**_mapping(summary.get("settings")), **_mapping(row.get("settings"))}
        merged = {**settings, **summary, **row}
        merged["solver_recipe"] = merged.get("solver_recipe", merged.get("solver"))
        merged["steps_per_event"] = merged.get("steps_per_event", merged.get("steps"))
        merged["completed_events"] = merged.get("completed_events", merged.get("updates"))
        merged["fit_seconds"] = merged.get("fit_seconds", merged.get("seconds"))
        merged["segment"] = merged.get("segment", merged.get("name"))
        if "validation_initial" in row or "validation_final" in row:
            merged["validation"] = {c: row.get("validation_" + c, {}) for c in ("initial", "final")}
            merged["completed_events"] = row.get("events", row.get("completed_events"))
            events = [e for e in _rows(summary.get("rows")) if e.get("segment") == row.get("segment")]
            if events and all(_number(e.get("update_seconds")) for e in events):
                merged["fit_seconds"] = sum(e["update_seconds"] for e in events)
            else:
                start = _mapping(row.get("validation_initial")).get("cumulative_fit_seconds")
                end = _mapping(row.get("validation_final")).get("cumulative_fit_seconds")
                merged["fit_seconds"] = end - start if _number(start) and _number(end) else None
        if merged["segment"] is None:
            segments = summary.get("segments")
            if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
                merged["segment"] = ", ".join(segments) or None
            elif isinstance(segments, list) and segments and all(isinstance(s, dict) for s in segments):
                merged["segment"] = ", ".join(str(s.get("segment", s.get("name", s.get("id", "unnamed")))) for s in segments)
        result.append(merged)
    return result


def _validate(unit):
    summary = unit["summary"]
    shared._version(summary.get("schema_version"), 1, "Phase 3 Fix 2 summary schema")
    expected = _path_version(unit["path"])
    _version(_mapping(unit.get("config")), expected)
    _version(summary, expected)
    kind = summary.get("kind")
    _require(kind in _KINDS, "unknown solver kind " + repr(kind))
    if kind in {"solver_recorded", "solver_comparison"}:
        rows = _method_rows(summary)
        _require(bool(rows), "nonempty recorded summaries required")
        for row in rows:
            _recipe(row.get("solver_recipe"))
            _require(row.get("condition") in ("replay_ewc", "stm_fd"), "recorded condition required")
            steps = row.get("steps_per_event")
            _require(type(steps) is int and steps == RECIPES[row["solver_recipe"]], "steps_per_event contradicts recipe")
            _require(_count(row.get("completed_events")), "completed_events must be a nonnegative count")
            _validate_numbers(row)
            if "rows" in row:
                _require(isinstance(row["rows"], list) and all(isinstance(e, dict) for e in row["rows"]), "event rows require objects")
    if kind == "solver_selection":
        _recipe(summary.get("selected_solver"), optional=True)
        _require(isinstance(summary.get("reason"), str) and bool(summary["reason"]), "selection reason required")
        _validate_numbers(summary)
    if kind == "solver_pilot":
        _recipe(summary.get("solver_recipe"))
        _require(isinstance(summary.get("evaluations"), list) and bool(summary["evaluations"]), "pilot evaluation panels required")
        seen = set()
        for panel in summary["evaluations"]:
            _require(isinstance(panel, dict), "pilot panels must be objects")
            checkpoint = panel.get("checkpoint")
            _require(checkpoint in ("initial", "mid", "midpoint", "final"), "unknown pilot checkpoint")
            _require(panel.get("difficulty") in ("easy", "hard"), "unknown pilot difficulty")
            key = ("midpoint" if checkpoint == "mid" else checkpoint, panel["difficulty"])
            _require(key not in seen, "duplicate pilot evaluation panel")
            seen.add(key)
            _require(initialreport._panel(panel)[0] is not None, "pilot outcomes require unique seed IDs and Boolean success")
            _require(_count(panel.get("acquired_count")), "pilot acquired_count required")
    for key in ("timing", "timings"):
        if key in summary:
            _require(isinstance(summary[key], dict), key + " must be an object")
    return unit


def _validate_numbers(row):
    for key in ("fit_seconds", "seconds", "learning_rate", "selected_learning_rate"):
        if row.get(key) is not None:
            _require(_number(row[key]) and row[key] >= 0, key + " must be finite and nonnegative")
    validation = row.get("validation")
    if validation is not None:
        _require(isinstance(validation, dict), "validation must be an object")
        for checkpoint in ("initial", "final"):
            panel = validation.get(checkpoint, {})
            _require(isinstance(panel, dict), "validation checkpoint must be an object")
            for key in ("old_nll", "live_nll"):
                if panel.get(key) is not None:
                    _require(_number(panel[key]), "validation NLL must be finite")


def read_completed_unit(path):
    """Validate the immutable scalar envelope; never inspect referenced binaries."""
    return _validate(shared.read_completed_unit(path))


def _committed_progress(path):
    value = initialreport._committed_progress(path, strict=Path(path).name == "orchestration")
    if not value:
        return {}
    _version(value, _path_version(path))
    _recipe(value.get("selected_solver"), optional=True)
    for key in ("repair_active_seconds", "cumulative_study_seconds", "remaining_budget_seconds"):
        if value.get(key) is not None:
            _require(_number(value[key]) and value[key] >= 0, key + " must be finite and nonnegative")
    for counts in (value.get("counts", {}), _mapping(value.get("active_unit_progress")).get("counts", {})):
        _require(isinstance(counts, dict) and all(_count(v) for v in counts.values()), "progress counts must be nonnegative integers")
    if value.get("completed_trajectories") is not None:
        _require(_count(value["completed_trajectories"]), "completed_trajectories must be a count")
    return value


def load_report(root=None):
    """Load just Fix 2 sealed scalars and committed incomplete progress.

    Missing roots are pending and are never created. A progress.json mirror is
    not a commit and cannot supply counters, costs, selection or outcomes.
    """
    campaign = _campaign_root(root)
    units, live = [], []
    if campaign.exists():
        _require(campaign.is_dir() and not campaign.is_symlink(), "campaign must be a regular directory")
        units = [read_completed_unit(marker.parent) for marker in sorted(campaign.rglob("COMPLETED"))]
        paths = {p.parent for pattern in ("config.json", "CURRENT.json") for p in campaign.rglob(pattern)
                 if "transactions" not in p.relative_to(campaign).parts}
        if (campaign / "orchestration").is_dir():
            paths.add(campaign / "orchestration")
        for path in sorted(paths):
            if (path / "COMPLETED").exists():
                continue
            if (path / "manifest.json").is_file():
                manifest = shared._json(path / "manifest.json")
                shared._version(manifest.get("artifact_schema_version"), 1, "artifact schema")
                shared._version(manifest.get("metric_schema_version"), 1, "metric schema")
            progress = _committed_progress(path)
            live.append({"path": str(path), "progress": progress, "status": "incomplete"})
    units.sort(key=lambda u: (str(u["manifest"].get("completed_at", "")), u["path"]))
    return {"root": str(campaign), "complete_units": units, "live": live,
            "status": "available" if units else "incomplete" if live else "not yet started"}


def _units(snapshot):
    campaign = _campaign_root(snapshot.get("root"))
    return [_validate(u) for u in snapshot.get("complete_units", [])
            if str(_mapping(u.get("summary")).get("kind", "")).startswith("solver_")
            and Path(u["path"]).resolve().is_relative_to(campaign.resolve())]


def _latest(units, kind):
    return next((u["summary"] for u in reversed(units) if u["summary"]["kind"] == kind), {})


def _progress(snapshot):
    campaign = _campaign_root(snapshot.get("root"))
    parents = [u for u in snapshot.get("live", []) if Path(u["path"]).resolve().is_relative_to(campaign.resolve())
               and Path(u["path"]).name == "orchestration"]
    _require(len(parents) <= 1, "multiple solver orchestration parents")
    return _committed_progress(parents[0]["path"]) if parents else {}


def _shown(value):
    return PENDING if value is None else str(value)


def _hours(value):
    return f"{value / 3600:.3f}" if _number(value) else PENDING


def _correction_note(snapshot):
    campaign = _campaign_root(snapshot.get("root"))
    notes = []
    if (campaign.parent / "solver-fix2-v1").is_dir():
        notes.append("v1 diagnostic halt before live admission: RecordedRun.evaluate used _batch(row, validationpool), "
                "trimming recurrent context instead of using the frozen full holdout. v1 validation/selection "
                "is superseded and is not pooled with v2 science or v3 science. Sealed v1 artifacts remain preserved; "
                "all attempts share the same nine-hour Fix 2 ceiling.")
    if campaign.name == CAMPAIGN and (campaign.parent / "solver-fix2-v2").is_dir():
        notes.append("v2 stopped when lagged updates generated a nonfinite model output during retirement. "
                     "v3 classifies this as a failed numerical candidate and gives all methods one common "
                     "step-size reduction. Its unchanged completed proximal measurements are explicitly "
                     "referenced with source seals, not counted as extra replicas. The two attempts carry "
                     "1,500 seconds into the same nine-hour budget, leaving 30,900 seconds for v3.")
    elif campaign.name == "solver-fix2-v2" and notes:
        notes.append("v1's 700-second charge leaves 31,700 seconds within the same nine-hour ceiling.")
    return " ".join(notes) if notes else None


def _prior_panel_seconds(units):
    references = {(u.get("config", {}).get("source_campaign"),
                   u.get("config", {}).get("source_completion")) for u in units}
    values = []
    for source, seal in references:
        if not source or not seal:
            continue
        marker = Path(source) / "orchestration/COMPLETED"
        _require(marker.is_file() and hashlib.sha256(marker.read_bytes()).hexdigest() == seal,
                 "forecast source completion differs")
        for path in sorted((Path(source) / "units").glob("pilot-*/COMPLETED")):
            value = _mapping(shared.read_completed_unit(path.parent)["summary"].get("timings")).get("evaluation_seconds")
            if _number(value) and value >= 0:
                values.append(value)
    return sum(values)/len(values) if values else None


def _pilot_seconds(summary):
    # The callback timer restarts on resume; the learner timer is cumulative.
    cumulative = _mapping(summary.get("timings")).get("active_seconds")
    return cumulative if _number(cumulative) else summary.get("active_seconds")


def _forecast_seconds(units):
    if any(u["summary"]["kind"] == "solver_campaign" for u in units):
        return [0., 0.]
    pilots = [u["summary"] for u in units if u["summary"]["kind"] == "solver_pilot"]
    ready = [u["summary"] for u in units if u["summary"]["kind"] == "solver_readiness"]
    total, panels = 0., None
    for condition in ("replay_ewc", "stm_fd"):
        done = [r for r in pilots if r.get("condition") == condition]
        remaining = max(0, 2-len(done))
        if not remaining:
            continue
        measured = [_pilot_seconds(r) for r in done if _number(_pilot_seconds(r))]
        if measured:
            estimate = sum(measured)/len(measured)
        else:
            samples = [r for r in ready if r.get("condition") == condition
                       and _number(_pilot_seconds(r)) and _number(r.get("acquired_count"))
                       and r["acquired_count"] > 0]
            if not samples:
                return None
            if panels is None:
                panels = _prior_panel_seconds(units)
            if panels is None:
                return None
            estimate = sum(_pilot_seconds(r)*640/r["acquired_count"] for r in samples)/len(samples)+panels
        total += remaining*estimate
    return [.75*total, 1.5*total]


def progress_rows(snapshot):
    units = _units(snapshot)
    progress = {**_progress(snapshot), **_latest(units, "solver_campaign")}
    selection = _latest(units, "solver_selection")
    pilots = [u for u in units if u["summary"]["kind"] == "solver_pilot"]
    completed = progress.get("completed_trajectories", len(pilots) if pilots else None)
    remaining = progress.get("remaining_trajectories")
    if remaining is None and _count(completed):
        remaining = max(0, 4 - completed)
    estimate = progress.get("remaining_time_range_seconds", progress.get("remaining_estimate_seconds"))
    if estimate is None:
        estimate = _forecast_seconds(units)
    forecast = ("--".join(f"{v / 3600:.2f}" for v in estimate) if isinstance(estimate, list)
                and len(estimate) == 2 and all(_number(v) and v >= 0 for v in estimate) else _hours(estimate))
    campaign = _campaign_root(snapshot.get("root"))
    incomplete = any(Path(u["path"]).resolve().is_relative_to(campaign.resolve()) for u in snapshot.get("live", []))
    status = progress.get("status", "incomplete; committed counters pending" if incomplete else "not yet started")
    if incomplete and not _latest(units, "solver_campaign"):
        status = "incomplete: " + status
    methods = sorted({r["solver_recipe"] + "/" + r["condition"] for u in units
                      if u["summary"]["kind"] in ("solver_recorded", "solver_comparison")
                      for r in _method_rows(u["summary"])})
    counts = _mapping(_mapping(progress.get("active_unit_progress")).get("counts", progress.get("counts")))
    counter = " / ".join(_shown(counts.get(k)) for k in ("step", "target")) if counts else PENDING
    if not counts and _latest(units, "solver_campaign"):
        counter = "Not active"
    by_kind = {kind: sum(u["summary"]["kind"] == kind for u in units) for kind in
               ("solver_software", "solver_recorded", "solver_selection", "solver_readiness", "solver_pilot", "solver_campaign")}
    derived_stages = sum((by_kind["solver_software"] >= 1, by_kind["solver_recorded"] >= 6,
                          by_kind["solver_selection"] >= 1, by_kind["solver_readiness"] >= 2,
                          by_kind["solver_pilot"] >= 4, by_kind["solver_campaign"] >= 1))
    done = progress.get("completed_stages", derived_stages if units else None)
    left = progress.get("remaining_stages", 6-derived_stages if units else None)
    charged = progress.get("repair_active_seconds")
    remaining_budget = progress.get("remaining_budget_seconds")
    if remaining_budget is None and _number(charged):
        parents = [u for u in units if u["summary"]["kind"] == "solver_campaign"]
        limit = parents[-1].get("stored_config", {}).get("max_hours") if parents else None
        if _number(limit) and limit >= 0:
            remaining_budget = max(0., limit * 3600 - charged)
    carry = {"solver-fix2-v2": 700., CAMPAIGN: 1500.}.get(campaign.name, 0.)
    if _correction_note(snapshot) and _number(charged):
        charged += carry
    return [{"Stage": status, "Stages completed / remaining":
             _shown(done) + " / " + _shown(left),
             "Recorded methods": str(len(methods)) + " / 6" if methods else PENDING,
             "Pilots complete / remaining (four planned)": _shown(completed) + " / " + _shown(remaining),
             "Active counter / target": counter,
             "Charged repair / study h": _hours(charged) + " / " + _hours(progress.get("cumulative_study_seconds")),
             "Forecast / remaining ceiling h": forecast + " / " + _hours(remaining_budget),
             "Selected solver": selection.get("selected_solver") or PENDING,
             **({"Diagnostic correction": "Full frozen holdout; numerical failure handling; shared 9 h ceiling"}
                if _correction_note(snapshot) else {})}]


def _progress_output(snapshot):
    output = _table([{"Metric": key, "Value": value}
                     for row in progress_rows(snapshot) for key, value in row.items()])
    output.metadata[_TAG + "_progress"] = True
    return output


def _recorded(snapshot):
    return [{**r, "unit_path": u["path"]} for u in _units(snapshot)
            if u["summary"]["kind"] in ("solver_recorded", "solver_comparison")
            for r in _method_rows(u["summary"])]


def _settings_rows(snapshot):
    rows = [{"Recipe": name, "Steps/event": steps,
             "EWC": "lagged explicit" if name == "lagged_1" else "centered proximal",
             "Ridge": "current center; proximal", "Plan alpha / beta2 / epsilon": ".003 / .999 / 1e-8"}
            for name, steps in RECIPES.items()]
    return rows


def _validation(row, checkpoint, domain):
    value = _mapping(_mapping(row.get("validation")).get(checkpoint)).get(domain + "_nll")
    if value is None and domain == "live":
        value = row.get("validation_" + checkpoint + "_nll")
    return value if _number(value) else None


def recorded_rows(snapshot):
    return [{"Solver": r["solver_recipe"], "Condition": r["condition"], "Alpha": r.get("learning_rate"), "Segment": r.get("segment"),
             "Status": r.get("status"), "Steps/event": r["steps_per_event"], "Accepted events": r["completed_events"],
             "Fit seconds": r.get("fit_seconds"),
             "Old NLL before / after": _shown(_validation(r, "initial", "old")) + " / " + _shown(_validation(r, "final", "old")),
             "Live NLL before / after": _shown(_validation(r, "initial", "live")) + " / " + _shown(_validation(r, "final", "live"))}
            for r in _recorded(snapshot)]


def _label(row):
    label = " / ".join(str(row.get(k)) for k in ("solver_recipe", "condition", "segment") if row.get(k) is not None)
    return label + (f" / alpha={row['learning_rate']:g}" if _number(row.get("learning_rate")) else "")


def _short_label(row):
    names = {"proximal_1": "Prox 1", "lagged_1": "Lag 1", "proximal_3": "Prox 3"}
    return names.get(row.get("solver_recipe"), "?") + " / " + ("FD" if row.get("condition") == "stm_fd" else "Replay/EWC")


def _plot_forks(snapshot):
    rows = _fork_summaries(snapshot)
    selected = _latest(_units(snapshot), "solver_selection").get("selected_learning_rate")
    rates = [r.get("learning_rate") for r in rows if _number(r.get("learning_rate"))]
    rate = selected if _number(selected) else min(rates, default=None)
    return [r for r in rows if (rate is None or r.get("learning_rate") == rate) and r.get("status") == "complete"]


def _averaged_points(row, ykey, *, compute=False):
    groups, origins = {}, {}
    for value in _rows(row.get("validation_rows")):
        segment = value.get("segment")
        if not _number(value.get("events")) or not _number(value.get(ykey)):
            continue
        cost = value.get("cumulative_fit_seconds")
        if compute and not _number(cost):
            continue
        origins.setdefault(segment, cost)
        x = cost-origins[segment] if compute else value["events"]
        groups.setdefault(value["events"], []).append((x, value[ykey]))
    return [(sum(x for x, _ in group)/len(group), sum(y for _, y in group)/len(group))
            for _, group in sorted(groups.items())]


def _quality(snapshot):
    from matplotlib.figure import Figure

    rows = _plot_forks(snapshot)
    if not rows:
        return None
    figure = Figure(figsize=(10, 5), layout="constrained")
    axes = figure.subplots(2, 2)
    for row in rows:
        for i, domain in enumerate(("old", "live")):
            before, after = (_validation(row, c, domain) for c in ("initial", "final"))
            for j, xkey in enumerate(("completed_events", "fit_seconds")):
                points = _averaged_points(row, domain+"_nll", compute=j == 1)
                if points:
                    axes[i, j].plot(*zip(*points), "o-", markersize=3, label=_short_label(row))
                    continue
                x = row.get(xkey)
                if _number(x) and after is not None:
                    axis = axes[i, j]
                    if before is not None:
                        axis.plot([0, x], [before, after], "o-", markersize=3, label=_short_label(row))
                    else:
                        axis.scatter([x], [after], label=_short_label(row))
    for i, domain in enumerate(("Old-domain", "Live-domain")):
        for j, xlabel in enumerate(("Accepted events within segment", "Measured fit seconds since segment restart")):
            axes[i, j].set(xlabel=xlabel, ylabel=domain + " validation NLL")
    initialreport._finish(axes.ravel())
    return initialreport._figure(figure, "Means across the two diagnostic segments at the selected/latest common alpha; all scales and failures remain in the table. "
                                "Held-out predictive NLL vs accepted events and measured fit cost. "
                                "Each segment restarts from the common pre-stream initializer, including the late segment; not historical continuation. "
                                "Lower is better on the same pool; different-center regularized losses are not pooled.")


def _diagnostics(snapshot):
    from matplotlib.figure import Figure

    rows = _plot_forks(snapshot)
    if not rows:
        return None
    figure = Figure(figsize=(10, 6), layout="constrained")
    axes = figure.subplots(2, 2).ravel()
    fields = (("first_step_ewc_suppression", "ewc_suppression", "update_suppression", "ewc_update_relative_difference"),
              ("step_cosine", "oscillator", "oscillation", "step_oscillation"),
              ("stiffness_max", "ewc_stiffness_max", "ewc_ratio_max"),
              ("post_centered_scaled_gradient_residual", "post_scaled_gradient_residual"))
    for row in rows:
        events = _rows(row.get("rows")) or [row]
        events = [{**_mapping((_rows(e.get("inner_steps")) or [{}])[-1]), **e} for e in events]
        for axis, keys in zip(axes, fields):
            for key in keys:
                points = [(e.get("event", e.get("event_id", i)), e.get(key)) for i, e in enumerate(events, 1)]
                points = [(x, y) for x, y in points if _number(x) and _number(y)]
                if points:
                    groups = {}
                    for x, y in points:
                        groups.setdefault(x, []).append(y)
                    axis.plot(sorted(groups), [sum(groups[x])/len(groups[x]) for x in sorted(groups)],
                              ".-", markersize=2, label=_short_label(row))
                    break
    axes[0].set_yscale("symlog", linthresh=.1)
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].set_yscale("symlog", linthresh=1.)
    axes[2].axhline(1., color="black", linestyle="--", linewidth=.7)
    axes[3].set_yscale("symlog", linthresh=.1)
    for axis, title in zip(axes, ("Relative EWC influence", "Successive-step cosine", "Maximum delayed stiffness",
                                  "Scaled centered-objective residual")):
        axis.set(xlabel="Recorded event / diagnostic index", title=title)
    initialreport._finish(axes)
    return initialreport._figure(figure, "Counterfactual E=0 proposal differences and stiffness/oscillation where recorded. "
                                "A zero pre-step centered penalty does not imply absent curvature influence. "
                                "Three-step proposal diagnostics are local per iteration, not a no-EWC trajectory. "
                                "Residuals use the centered target objective at preselected diagnostic events: "
                                "post-update gradient norm divided by max(1, last live-gradient norm); lower is more stationary, not proof of convergence.")


def timing_rows(snapshot):
    rows = []
    fields = ("mean_seconds", "median_seconds", "p95_seconds", "mean_ms", "median_ms", "p95_ms", "gpu_seconds", "ms_per_inner_step", "ms_per_event",
              "gradient_evaluations", "fd_evaluations", "optimizer_exposures", "peak_vram_bytes",
              "actor_wait_seconds", "actor_blocked_seconds", "publication_latency_seconds", "deadline_misses",
              "setup_seconds", "warmup_seconds", "checkpoint_seconds", "evaluation_seconds", "reporting_seconds",
              "gpu_mean_seconds", "diagnostic_fit_seconds", "cost_ratio_to_proximal_1")
    summaries = _fork_summaries(snapshot)
    for row in summaries:
        timing = {**_mapping(row.get("timings")), **_mapping(row.get("timing")), **row}
        measured = _mapping(row.get("timing"))
        values = {k: timing.get(k) for k in fields if _number(timing.get(k))}
        if "cost_ratio_to_proximal_1" not in values:
            baselines = [r for r in summaries if r.get("solver_recipe") == "proximal_1"
                         and r.get("condition") == row.get("condition")
                         and r.get("learning_rate") == row.get("learning_rate")]
            baseline = _mapping(baselines[0].get("timing")).get("mean_seconds") if len(baselines) == 1 else None
            current = measured.get("mean_seconds")
            values["cost_ratio_to_proximal_1"] = current / baseline if _number(current) and _number(baseline) and baseline > 0 else None
        rows.append({"Solver / condition": _label(row), "Charged fit seconds (all events)": row.get("fit_seconds"),
                     "Timed fit seconds (diagnostics excluded)": measured.get("fit_seconds"), **values})
    return rows


def _fork_summaries(snapshot):
    return [r for unit in _units(snapshot) if unit["summary"]["kind"] in ("solver_recorded", "solver_comparison")
            for r in _method_rows({**unit["summary"], "segments": None})]


def _selection_rows(selection):
    rows = [{"Selected solver": selection.get("selected_solver"),
             "Selected alpha": selection.get("selected_learning_rate"), "Reason": selection.get("reason")}]
    for candidate in _rows(selection.get("candidates")):
        rows.append({"Candidate": candidate.get("solver_recipe", candidate.get("solver")),
                     **{k: candidate[k] for k in ("learning_rate", "status", "eligible", "reason", "fit_seconds", "stable", "runs", "old_gap", "mean_seconds",
                                                  "cost_ratio_to_proximal_1", "old_nll", "live_nll")
                        if k in candidate and (candidate[k] is None or type(candidate[k]) in (str, bool) or _number(candidate[k]))}})
    return rows


def _pilot_units(snapshot):
    return [u for u in _units(snapshot) if u["summary"]["kind"] == "solver_pilot"]


def _pilot_identity(summary):
    settings = _mapping(summary.get("settings"))
    phase3 = _mapping(summary.get("phase3"))
    return (settings.get("seed", summary.get("seed")),
            settings.get("initial_asset", phase3.get("initial_asset_id", summary.get("initial_asset_id"))),
            summary.get("solver_recipe"), settings.get("name", summary.get("condition")))


def _panels(summary):
    return {(p["difficulty"], "midpoint" if p["checkpoint"] == "mid" else p["checkpoint"]): p
            for p in summary["evaluations"]}


def _paired_pilots(snapshot):
    groups = {}
    for unit in _pilot_units(snapshot):
        seed, asset, recipe, condition = _pilot_identity(unit["summary"])
        if _count(seed) and isinstance(asset, str) and asset:
            groups.setdefault((seed, asset, recipe), {}).setdefault(condition, []).append(unit)
    result = []
    for key, arms in groups.items():
        if all(len(arms.get(c, [])) == 1 for c in ("replay_ewc", "stm_fd")):
            result.append((key, {c: arms[c][0] for c in ("replay_ewc", "stm_fd")}))
    return result


def _capability(snapshot):
    from matplotlib.figure import Figure

    if not _paired_pilots(snapshot):
        return None
    figure = Figure(figsize=(10, 3.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for (seed, _, recipe), arms in _paired_pilots(snapshot):
        for axis, difficulty in zip(axes, ("easy", "hard")):
            panels = {c: _panels(u["summary"]) for c, u in arms.items()}
            checkpoints = [c for c in ("initial", "midpoint", "final") if all((difficulty, c) in p for p in panels.values())]
            if not {"initial", "final"} <= set(checkpoints):
                continue
            identity = [initialreport._panel(panels[c][difficulty, cp])[2]
                        for c in arms for cp in checkpoints]
            if any(ids != identity[0] for ids in identity):
                continue
            for condition in arms:
                points = [panels[condition][difficulty, cp] for cp in checkpoints]
                axis.plot([p["acquired_count"] for p in points], [initialreport._panel(p)[0] for p in points],
                          "o-", markersize=3,
                          label=f"{'FD' if condition == 'stm_fd' else 'Replay/EWC'} / {seed}")
    for axis, title in zip(axes, ("Paired A retention", "Paired B success")):
        axis.set(title=title, xlabel="Unique acquired transitions (initial / mid / final)", ylabel="Held-out success fraction", ylim=(-.03, 1.03))
    if not any(axis.lines for axis in axes):
        return None
    initialreport._finish(axes)
    return initialreport._figure(figure, "Initial/mid/final capability requires common seed, initialization asset, solver and layout IDs. "
                                "Unpaired outcomes remain in the table, not pooled into paired curves.")


def pilot_rows(snapshot):
    rows = []
    for unit in _pilot_units(snapshot):
        summary = unit["summary"]
        seed, _, recipe, condition = _pilot_identity(summary)
        row = {"Seed": seed, "Solver": recipe, "Condition": condition, "Acquired": summary.get("acquired_count")}
        panels = _panels(summary)
        for difficulty, label in (("easy", "A"), ("hard", "B")):
            for cp in ("initial", "midpoint", "final"):
                rate, count, _ = initialreport._panel(panels.get((difficulty, cp), {}))
                row[label + " " + cp] = f"{round(rate * count)}/{count}" if rate is not None else None
        timing = _mapping(summary.get("timings"))
        row["Active / fit seconds"] = _shown(timing.get("active_seconds")) + " / " + _shown(timing.get("fit_seconds"))
        rows.append(row)
    return rows


def _artifact_links(units, notebook_dir):
    import nbformat

    links, labels = [], []
    for unit in units:
        path = Path(unit["path"]) / "summary.json"
        label = unit["summary"]["kind"] + ": " + path.parent.name
        links.append(f'<a href="{quote(os.path.relpath(path, notebook_dir))}">{html.escape(label)}</a>')
        labels.append(label + ": " + str(path))
    return nbformat.v4.new_output("display_data", data={"text/html":
        f"<details><summary>Scalar summaries and full traces ({len(links)} artifacts)</summary><ul>"
        + "".join("<li>"+link+"</li>" for link in links)+"</ul></details>",
        "text/plain": f"{len(links)} scalar summaries and full traces (expand the HTML artifact links)."})


def _brief_comparison(snapshot):
    rows = []
    for r in _fork_summaries(snapshot):
        old0, old1 = (_validation(r, c, "old") for c in ("initial", "final"))
        live0, live1 = (_validation(r, c, "live") for c in ("initial", "final"))
        rows.append({"Method": _short_label(r), "Alpha": r.get("learning_rate"),
            "Status": r.get("status"), "Events": r.get("completed_events"),
            "Old NLL change": f"{old1-old0:+.3f}" if old0 is not None and old1 is not None else PENDING,
            "Live NLL change": f"{live1-live0:+.3f}" if live0 is not None and live1 is not None else PENDING})
    return rows


def _brief_timing(snapshot):
    rows = []
    for r in timing_rows(snapshot):
        def show(key, multiplier=1.):
            value = r.get(key)
            return f"{value*multiplier:.2f}" if _number(value) else PENDING
        rows.append({"Method / alpha": r["Solver / condition"],
            "Mean ms/event": show("mean_seconds", 1000.),
            "P95 ms/event": show("p95_seconds", 1000.),
            "GPU mean ms/event": show("gpu_mean_seconds", 1000.),
            "Cost / Prox 1": show("cost_ratio_to_proximal_1"),
            "Torch peak GiB": show("peak_vram_bytes", 1/1024**3)})
    return rows


def _brief_pilots(snapshot):
    rows = []
    for r, unit in zip(pilot_rows(snapshot), _pilot_units(snapshot)):
        timing = _mapping(unit["summary"].get("timings"))
        cost = " / ".join(f"{timing[k]/60:.1f}" if _number(timing.get(k)) else PENDING
                          for k in ("active_seconds", "fit_seconds"))
        rows.append({"Seed": r["Seed"], "Method": "FD" if r["Condition"] == "stm_fd" else "Replay/EWC",
            "Acquired": r["Acquired"],
            "A initial / mid / final": " / ".join(_shown(r["A " + cp]) for cp in ("initial", "midpoint", "final")),
            "B initial / mid / final": " / ".join(_shown(r["B " + cp]) for cp in ("initial", "midpoint", "final")),
            "Wall / fit min": cost})
    return rows


def operational_rows(snapshot):
    rows = []
    for unit in _pilot_units(snapshot):
        s = unit["summary"]
        seed, _, _, condition = _pilot_identity(s)
        fits = sorted(r["update_seconds"] for r in _rows(s.get("fit_metrics")) if _number(r.get("update_seconds")))
        intervals = [r for e in _rows(s.get("episodes")) for r in _rows(e.get("intervals"))]
        waits = [_mapping(r.get("timing")).get("model_lease_wait_s") for r in intervals]
        waits = [x for x in waits if _number(x)]
        misses = [_mapping(r.get("timing")).get("inference_deadline_miss") for r in intervals]
        misses = [x for x in misses if _number(x)]
        fallback = _mapping(s.get("fallback_counts"))
        stops = sum(v for k, v in fallback.items() if k not in ("policy", "startup") and _number(v))
        seconds, acquired = _pilot_seconds(s), s.get("acquired_count")
        rows.append({"Seed / method": f"{seed} / {'FD' if condition == 'stm_fd' else 'Replay/EWC'}",
            "Mean / p95 fit s": f"{sum(fits)/len(fits):.3f} / {fits[max(0, math.ceil(.95*len(fits))-1)]:.3f}" if fits else PENDING,
            "Max actor lease wait s": f"{max(waits):.3f}" if waits else PENDING,
            "Deadline misses": int(sum(misses)) if misses else PENDING,
            "Non-startup fallback commands": stops if fallback else PENDING,
            "Acquired / pilot h": f"{3600*acquired/seconds:.1f}" if _number(seconds) and seconds > 0 and _number(acquired) else PENDING})
    return rows


def diagnostic_outputs(snapshot, *, notebook_dir=None):
    units = _units(snapshot)
    directory = Path(notebook_dir or DEFAULT_NOTEBOOK.parent)
    outputs = [_note("Declared recipes (plan settings, not measured outcomes). Live-gradient second moment only; no first-moment momentum. "
                     "Ridge is proximal at the current center in all recipes; only lagged_1 delays its EWC center."),
               _table(_settings_rows(snapshot)), _note("Recorded comparison: same held-out pools, conditional on historical behavior; not live efficacy replicas.")]
    if _correction_note(snapshot):
        outputs.insert(0, _note(_correction_note(snapshot)))
    outputs.append(_table(_brief_comparison(snapshot)) if _recorded(snapshot) else _note("Recorded comparison: " + PENDING))
    for title, plot in (("Quality versus compute", _quality), ("EWC / update diagnostics", _diagnostics)):
        outputs.append(plot(snapshot) or _note(title + ": " + PENDING))
    outputs.append(_note("Timing and selection. Isolated GPU execution and online blocking are distinct; overlapping timers are not summed. "
                         "Per-event mean/median/p95 exclude initial warm-up and extra-gradient diagnostic events under the recorded timing policy. "
                         "All-event fit seconds and quality-versus-compute include charged diagnostics. Missing GPU telemetry remains unavailable; "
                         "cost ratios use measured mean time for the same condition/alpha, never step counts. "
                         "Torch peak is allocated tensor memory; device residency also includes reserved pools and CUDA overhead."))
    outputs.append(_table(_brief_timing(snapshot)) if _recorded(snapshot) else _note("Measured timing: " + PENDING))
    selection = _latest(units, "solver_selection")
    outputs.append(_table(_selection_rows(selection)))
    outputs.append(_note("Selected-solver live development: four trajectories planned (two paired seeds), not six solver campaigns. "
                         "Boolean fixed-layout outcomes only; repeated panels are not independent training replicas. "
                         "The progress forecast extrapolates 48-interval live checks plus sealed Fix 1 evaluation cost, "
                         "then uses completed pilot times as available. Its 0.75--1.5 planning multiplier is not a confidence interval "
                         "and counts an unfinished pilot in full until completion."))
    outputs.append(_capability(snapshot) or _note("Paired A retention / B success: " + PENDING))
    rows = _brief_pilots(snapshot)
    outputs.append(_table(rows) if rows else _note("Live pilot outcomes: " + PENDING))
    runtime = operational_rows(snapshot)
    if runtime:
        output = _table(runtime)
        output.data["text/html"] = ("<details><summary>Live runtime health (expand)</summary>"
            "<p>Pilot throughput uses checkpointed cumulative runtime across resumes, includes evaluation and excludes the shared model load. Fit service includes cooperative actor waits. "
            "Deadline misses and fallback commands can describe the same interval; do not add them. Startup stops are excluded from the fallback count.</p>"
            + output.data["text/html"] + "</details>")
        outputs.append(output)
    if rows and not _paired_pilots(snapshot):
        outputs.append(_note("Pilot outcomes are unpaired; no paired effect is inferred."))
    for unit in _pilot_units(snapshot)[-4:]:
        videos = _rows(unit["summary"].get("videos"))
        if videos:
            selected = [v for v in videos if v.get("selected") is True or v.get("selection") == "preselected"] or videos
            selected = [v for v in selected if v.get("checkpoint") == "final"] or selected
            bounded = {**unit, "summary": {**unit["summary"], "videos": selected[:2]}}
            campaign = next((p for p in Path(unit["path"]).parents if p.name.startswith("solver-fix2-v")), Path(snapshot["root"]))
            outputs.append(_videos(bounded, campaign, directory))
    if units:
        outputs.append(_artifact_links(units, directory))
    return outputs


def _display(outputs):
    from IPython.display import display
    for output in outputs:
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def display_progress(snapshot):
    _display([_progress_output(snapshot)])


def display_diagnostics(snapshot, *, notebook_dir=None):
    _display(diagnostic_outputs(snapshot, notebook_dir=notebook_dir))


def conclusion(snapshot):
    units = _units(snapshot)
    selection = _latest(units, "solver_selection")
    recorded = [u for u in units if u["summary"]["kind"] in ("solver_recorded", "solver_comparison")]
    failed = sum(u["summary"].get("status") == "numerical_failure" for u in recorded)
    pilots = _pilot_units(snapshot)
    choice = selection.get("selected_solver")
    decision = (f"Recorded selection: {choice}; alpha={_shown(selection.get('selected_learning_rate'))}. "
                + selection.get("reason", "")) if choice else "Solver selection remains unresolved; no winner is inferred from pending results."
    lines = ["## Conclusion", "Phase 3 Fix 2: " + decision]
    if recorded or pilots:
        lines.append(f"{len(recorded)} sealed recorded-data forks ({failed} numerical failures) and {len(pilots)} sealed development pilots.")
    else:
        lines.append("Recorded comparison and paired live outcomes are pending.")
    forks = [r for r in _plot_forks(snapshot) if r.get("solver_recipe") == choice]
    events = [e for r in forks for e in _rows(r.get("rows"))]
    measured = [e for e in events if _number(e.get("first_step_ewc_proposal_difference_norm"))]
    if measured:
        changed = sum(e["first_step_ewc_proposal_difference_norm"] > 0 for e in measured)
        lines.append(f"**Numerical repair:** EWC changed the first-step proposal in {changed}/{len(measured)} "
                     "recorded events for the selected recipe. This demonstrates curvature use, not successful retention "
                     "or nonlinear convergence. A zero initial centered gradient is no longer an EWC blind spot.")
    if choice:
        costs = [r.get("cost_ratio_to_proximal_1")
                 for fork, r in zip(_fork_summaries(snapshot), timing_rows(snapshot))
                 if fork.get("solver_recipe") == choice
                 and fork.get("learning_rate") == selection.get("selected_learning_rate")]
        costs = [v for v in costs if _number(v)]
        if costs:
            lines.append(f"**Cost:** selected-recipe isolated updates cost {min(costs):.2f}--{max(costs):.2f}x "
                         "one-step proximal updates across the two objectives. Whole live-run costs appear separately above.")
    if pilots:
        lines.append("FD and Replay/EWC share trainable STM/LTM banks, replay, EWC and ridge; FD adds the STM "
                     "sensitivity penalty. Inherited penalty strengths were not freshly retention-matched under "
                     "this solver, so these pilots do not isolate FD efficacy at equal retention.")
        endpoint = []
        for unit in pilots:
            summary = unit["summary"]
            seed, _, _, condition = _pilot_identity(summary)
            panels = _panels(summary)
            counts = []
            for domain in ("easy", "hard"):
                rate, count, _ = initialreport._panel(panels.get((domain, "final"), {}))
                counts.append(f"{round(rate*count)}/{count}" if rate is not None else PENDING)
            endpoint.append(f"| {seed} | {'FD' if condition == 'stm_fd' else 'Replay/EWC'} | {counts[0]} | {counts[1]} |")
        lines.append("**Live development:** endpoint held-out capability (A = initial/easy task, B = hard task). "
                     "Initial/midpoint results and measured costs are in the preceding table.\n\n"
                     "| Seed | Method | Final A | Final B |\n|---|---|---:|---:|\n"+"\n".join(endpoint))
        hard = [initialreport._panel(_panels(u["summary"]).get(("hard", "final"), {})) for u in pilots]
        if hard and all(rate == 0 and count for rate, count, _ in hard):
            lines.append("No completed pilot succeeded on its endpoint hard-task panel. The numerical repair therefore "
                         "does not yet establish hard-task acquisition or FD efficacy. Better exploration/demonstration "
                         "coverage remains a hypothesis for a separately agreed next step, not a diagnosed cause.")
    if not _latest(units, "solver_campaign"):
        lines.append("Execution is still in progress. Unfinished or budget-limited work is unresolved, not a scientific rejection.")
    lines.append("**Scope:** two development seeds sharing one initializer cannot establish fresh significance, "
                 "retention equivalence or sample savings. Recorded-history forks are conditional diagnostics, "
                 "not counterfactual navigation replicas. Stop for review; no automatic Phase 4 or physical motion.")
    return "\n\n".join(lines)


def augment_notebook(book, snapshot=None, *, root=None, notebook_dir=None):
    """Idempotently append Fix 2, preserving every prior cell and conclusion.

    ``snapshot`` may be the parent's already validated shared snapshot. New
    notebook execution reloads this report's narrow campaign-only JSON view.
    """
    import nbformat

    snapshot = load_report(root) if snapshot is None else snapshot
    campaign = _campaign_root(root if root is not None else snapshot.get("root"))
    outputs = diagnostic_outputs(snapshot, notebook_dir=notebook_dir)
    progress = _progress_output(snapshot)
    book.cells = [c for c in book.cells if not c.metadata.get(_TAG)]
    code = ("from numerical_experiment.picar_solver_repair import report as solver_report\n"
            f"solver_snapshot = solver_report.load_report({str(campaign)!r})\n")
    progress_cell = next((c for c in book.cells if c.metadata.get("report_section") == "progress"), None)
    if progress_cell is not None:
        source = "\n" + code + "solver_report.display_progress(solver_snapshot)"
        if source not in progress_cell.source:
            progress_cell.source += source
        progress_cell.outputs = [o for o in progress_cell.get("outputs", []) if not o.get("metadata", {}).get(_TAG + "_progress")]
        progress_cell.outputs.append(progress)
    else:
        cell = nbformat.v4.new_code_cell(code + "solver_report.display_progress(solver_snapshot)", outputs=[progress])
        cell.id = "plan19-phase3-fix2-progress"
        cell.metadata[_TAG] = True
        book.cells.insert(min(1, len(book.cells)), cell)
    section = [nbformat.v4.new_markdown_cell(_INTRO + f"\n\nReporting campaign: **{campaign.name}** (versions are not pooled)."), nbformat.v4.new_code_cell(
        code + f"solver_report.display_diagnostics(solver_snapshot, notebook_dir={str(Path(notebook_dir).resolve()) if notebook_dir else None!r})",
        outputs=outputs)]
    for i, cell in enumerate(section):
        cell.id = f"plan19-phase3-fix2-{i}"
        cell.metadata[_TAG] = True
    for cell in book.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## Conclusion"):
            cell.source = cell.source.replace("## Conclusion", "#### Historical Closeout Before Phase 3 Fix 2", 1)
    book.cells.extend(section)
    from numerical_experiment import picar_reward_review
    reward_section = [nbformat.v4.new_markdown_cell(picar_reward_review.INTRO),
        nbformat.v4.new_code_cell(
            "from numerical_experiment import picar_reward_review\n"
            "picar_reward_review.display_review(solver_snapshot)",
            outputs=picar_reward_review.outputs(snapshot))]
    for i, cell in enumerate(reward_section):
        cell.id = f"plan19-phase3-fix2-rewards-{i}"
        cell.metadata[_TAG] = True
    book.cells.extend(reward_section)
    final = nbformat.v4.new_markdown_cell(conclusion(snapshot))
    final.id = "plan19-phase3-fix2-conclusion"
    final.metadata[_TAG] = True
    book.cells.append(final)
    return book


def _historical_fingerprint(root):
    """Completion seals are immutable; solver completions cannot stale history."""
    root = Path(root).resolve()
    entries = []
    for marker in sorted(root.rglob("COMPLETED")) if root.is_dir() else []:
        if not any(part.startswith("solver-fix2-v") for part in marker.relative_to(root).parts):
            entries.append((str(marker.relative_to(root)), hashlib.sha256(shared._read(marker, shared.MAX_JSON_BYTES)).hexdigest()))
    return str(root), tuple(entries)


def build_notebook(root=None, output=DEFAULT_NOTEBOOK, *, use_cache=True):
    """Build prior chronology then augment; never writes the canonical by default."""
    import nbformat

    snapshot = load_report(root)
    _units(snapshot)
    output = Path(output)
    if any((p / "COMPLETED").exists() for p in (output.parent, *output.parents)):
        raise shared.ArtifactError("cannot write a notebook inside a completed unit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".plan19-solver-report-", suffix=".ipynb", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        campaign = _campaign_root(root)
        historical_root = campaign.parent.parent if campaign.name.startswith("solver-fix2-v") else shared.resolve_root(root)
        key = _historical_fingerprint(historical_root)
        if not use_cache or key not in _BASE_CACHE:
            initialreport.build_notebook(historical_root, temporary)
            base = nbformat.read(temporary, as_version=4)
            if use_cache:
                _BASE_CACHE.clear()
                _BASE_CACHE[key] = copy.deepcopy(base)
        else:
            base = _BASE_CACHE[key]
        book = copy.deepcopy(base)
        augment_notebook(book, snapshot, root=campaign, notebook_dir=output.parent)
        nbformat.validate(book)
        nbformat.write(book, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
