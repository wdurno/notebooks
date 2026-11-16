"""Artifact-only Phase 3 Fix 1 reporting, independent of the numerical runtime."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.phase2_report import (
    PENDING, _mapping, _note, _number, _rows, _table, _videos,
)


VERSION = "plan19-phase3-fix1-v1"
REFERENCE_VERSION = "plan19-initialization-mc-reference-v1"
CAMPAIGN = "initialization-fix1-v1"
_PREFIX = "initialization_"
_KINDS = {"dataset", "collection", "fit", "bootstrap", "probe", "pilot", "growth", "mc_reference",
          "evaluation", "campaign", "software", "provenance", "integrity"}
_INTRO = (
    "### Phase 3 Fix 1: Stationary Initialization and Bulk Fisher\n\n"
    "The earlier Phase 3 results remain historical. This separately versioned repair fits fixed, "
    "whole-episode train/validation sets with STM neutral, then builds an equal-weight model-score "
    "Fisher at the accepted anchor. Repeated passes and synthetic score draws are not new observations; "
    "validation guides development, not confirmation. Only sealed results are plotted. Committed live "
    "counters are operational progress; missing results are **pending**, never zero.\n\n"
    "The repair has its own **8--10 active-hour planning range and 10-hour ceiling**, separate from "
    "the prior study ledger (about 10.719 h); the historical 20-hour target does not limit this repair."
)


def _is_repair(path):
    return any(p.name == CAMPAIGN for p in (Path(path), *Path(path).parents))


def _committed_progress(path, *, strict=True):
    """Follow the sealed CURRENT transaction, never a lagging progress mirror."""
    path = Path(path)
    if not (path / "CURRENT.json").is_file():
        return {}
    pointer = shared._json(path / "CURRENT.json")
    shared._version(pointer.get("storage_schema_version"), 1, "live storage schema")
    commit = pointer.get("commit_id")
    if not isinstance(commit, str) or Path(commit).name != commit or commit in (".", ".."):
        raise shared.ArtifactError("unsafe live progress commit")
    transaction = path / "transactions" / commit
    shared._verify_seal(transaction / "manifest.json", pointer.get("manifest"))
    manifest = shared._json(transaction / "manifest.json")
    progress = transaction / "progress.json"
    shared._verify_seal(progress, manifest.get("files", {}).get("progress.json"))
    value = shared._json(progress)
    if strict or "schema_version" in value:
        shared._version(value.get("schema_version"), 1, "Phase 3 Fix 1 progress schema")
    return value


def historical_phase3_snapshot(snapshot):
    """Keep the historical renderer on its own units and committed live counters.

    An empty root disables its legacy recursive progress-mirror scan. Absolute
    unit paths still identify each historical campaign and its media correctly.
    """
    live = []
    for item in snapshot.get("live", []):
        if not _is_repair(item["path"]):
            value = _committed_progress(item["path"], strict=False) if Path(item["path"]).name == "orchestration" else {}
            live.append({**item, **value})
    return {**snapshot, "root": "", "live": live, "complete_units": [
        u for u in snapshot.get("complete_units", [])
        if not str(u["summary"].get("kind", "")).startswith(_PREFIX)]}


def _count(value):
    return type(value) is int and value >= 0


def _curve(row):
    """Accept the numerical fitter's split dictionaries and earlier flat rows."""
    if not isinstance(row, dict):
        return {}
    result = dict(row)
    for split in ("train", "validation"):
        for source, target in (("joint_nll", "nll"), ("teacher_kl", "kl"),
                               ("actor_nll", "actor_nll"), ("critic_nll", "critic_nll")):
            if source in _mapping(row.get(split)):
                result[split + "_" + target] = row[split][source]
    for key in ("gradient_norm", "update_norm"):
        if key + "_mean" in _mapping(row.get("diagnostics")):
            result[key] = row["diagnostics"][key + "_mean"]
    if "optimization_exposures" in row:
        result["exposures"] = row["optimization_exposures"]
    return result


def _validate(unit):
    summary = unit["summary"]
    kind = summary["kind"][len(_PREFIX):]

    def require(condition, message):
        if not condition:
            raise shared.ArtifactError(f"Phase 3 Fix 1 schema-incompatible {kind}: {message}; {unit['path']}")

    require(kind in _KINDS, "unknown initialization kind")
    version = _mapping(unit.get("config")).get("version")
    expected_version = REFERENCE_VERSION if kind == "mc_reference" else VERSION
    require(version is None or version == expected_version,
            "unexpected initialization version; do not pool campaigns")
    if kind == "dataset":
        for key in ("train_count", "validation_count", "acquired_count", "new_acquisitions"):
            require(_count(summary.get(key)), f"{key} must be a nonnegative count")
        for key, count in (("train_ids", "train_count"), ("validation_ids", "validation_count")):
            ids = summary.get(key)
            require(isinstance(ids, list) and len(ids) == summary[count]
                    and all(type(v) in (str, int) for v in ids), f"{key} must match {count}")
            require(len(set(ids)) == len(ids), f"duplicate {key}")
        require(not set(summary["train_ids"]) & set(summary["validation_ids"]), "train/validation IDs overlap")
    if kind in ("dataset", "collection"):
        coverage = summary.get("coverage")
        require(isinstance(coverage, dict) and bool(coverage), "nonempty coverage required")
        require(all(_count(coverage.get(k)) for k in ("records", "episodes")), "coverage records/episodes required")
    if kind == "fit":
        curves = summary.get("curves")
        require(isinstance(curves, list) and bool(curves), "nonempty fixed-set curves required")
        for original in curves:
            row = _curve(original)
            require(isinstance(row, dict) and isinstance(row.get("stage"), str)
                    and bool(row["stage"]) and any(_number(row.get(k)) for k in ("epoch", "exposures")),
                    "curve rows require stage and epoch/exposures")
            require(any(_number(row.get(k)) for k in ("train_nll", "train_kl"))
                    and any(_number(row.get(k)) for k in ("validation_nll", "validation_kl")),
                    "curve rows require finite train and validation metrics")
        readiness = summary.get("readiness")
        require(isinstance(readiness, dict) and bool(readiness), "nonempty readiness required")
        if "ready" in readiness:
            require(type(readiness["ready"]) is bool, "ready must be a boolean")
    if kind == "bootstrap":
        require(any(summary.get(k) for k in ("fisher_stability", "stability", "stability_rows", "fisher_diagnostics", "diagnostics"))
                or _number(summary.get("trace")), "nonempty Fisher diagnostics required")
    if kind in ("evaluation", "pilot"):
        require(bool(_rows(summary.get("evaluations"))), "nonempty recorded evaluation panels required")
    if kind == "evaluation":
        require(summary.get("checkpoint") in ("distillation", "accepted_nll"), "unknown initialization checkpoint")
    if kind == "campaign":
        require(isinstance(summary.get("status"), str) and bool(summary["status"]), "status required")
    if kind == "mc_reference":
        require(summary.get("primary_unchanged") is True and summary.get("reference_is_oracle") is False,
                "reference must be diagnostic-only with unchanged primary Fisher")
        require(bool(_mapping(summary.get("comparisons")).get("primary_4_vs_reference")),
                "independent reference comparison required")
    return kind


def _units(snapshot, kind=None):
    result = []
    for unit in snapshot.get("complete_units", []):
        if str(unit["summary"].get("kind", "")).startswith(_PREFIX):
            found = _validate(unit)
            if kind is None or kind == found:
                result.append(unit)
    return sorted(result, key=lambda u: (str(u.get("manifest", {}).get("completed_at", "")), u["path"]))


def _latest(units, kind):
    return next((u["summary"] for u in reversed(units) if u["summary"]["kind"] == _PREFIX + kind), {})


def _progress(snapshot):
    parents = [u for u in snapshot.get("live", [])
               if _is_repair(u["path"]) and Path(u["path"]).name == "orchestration"]
    if len(parents) > 1:
        raise shared.ArtifactError("multiple initialization repair parents; select one campaign root")
    return _committed_progress(parents[0]["path"]) if parents else {}


def _live_fit(snapshot):
    """Only fixed-set epoch diagnostics from a sealed live transaction qualify."""
    candidates = [u for u in snapshot.get("live", []) if _is_repair(u["path"])
                  and Path(u["path"]).name.startswith("fit")]
    active = _progress(snapshot).get("active_unit")
    if isinstance(active, str):
        candidates = [u for u in candidates if Path(u["path"]).resolve() == Path(active).resolve()]
    result = []
    for item in candidates:
        progress = _committed_progress(item["path"])
        curves = progress.get("diagnostic_curves")
        if curves is None or curves == []:
            continue
        unit = {"path": item["path"], "summary": {"kind": "initialization_fit", "curves": curves,
                "readiness": {"status": "fit/selection not sealed"}, "provisional": True}}
        _validate(unit)
        result.append(unit)
    return result


def _shown(value):
    return PENDING if value is None else str(value)


def _hours(value):
    return f"{value / 3600:.3f}" if _number(value) and value >= 0 else PENDING


def progress_rows(snapshot):
    """One compact row; uncommitted numerical results never enter this view."""
    units = _units(snapshot)
    progress = {**_progress(snapshot), **_latest(units, "campaign")}
    child = _mapping(progress.get("active_unit_progress"))
    counts = _mapping(child.get("counts"))
    dataset, fit = _latest(units, "dataset"), _latest(units, "fit")
    curves = _rows(fit.get("curves"))
    provisional = _live_fit(snapshot)
    if provisional:
        curves = provisional[-1]["summary"]["curves"]
    passes = "; ".join(f"{stage}: {max(r['epoch'] for r in curves if r['stage'] == stage and _number(r.get('epoch')))}"
                       for stage in dict.fromkeys(r["stage"] for r in curves)
                       if any(r["stage"] == stage and _number(r.get("epoch")) for r in curves)) or None
    fisher = _latest(units, "bootstrap")
    fisher_progress = child.get("status") if "fisher" in str(progress.get("status", "")).lower() else None
    if counts:
        counter = "/".join(_shown(counts.get(k)) for k in ("step", "target"))
        fisher_progress = f"{fisher_progress}: {counter}" if fisher_progress else None
    if _count(counts.get("fisher_histories")) and (counts.get("fisher_histories", 0) > 0 or fisher_progress):
        fisher_progress = f"{counts['fisher_histories']} histories / {_shown(counts.get('model_score_draws'))} score draws"
    completed = progress.get("completed_trajectories")
    if completed is None and any(u["summary"]["kind"] == _PREFIX + "pilot" for u in units):
        completed = sum(u["summary"]["kind"] == _PREFIX + "pilot" for u in units)
    remaining = progress.get("remaining_trajectories", progress.get("remaining_pilots"))
    final = _latest(units, "campaign")
    if remaining is None and _count(completed) and (final or "pilot" in str(progress.get("status", "")).lower()):
        remaining = 0 if final else max(0, 4 - completed)
    estimate = progress.get("remaining_estimate_seconds", progress.get("remaining_time_range_seconds"))
    if isinstance(estimate, list) and len(estimate) == 2 and all(_number(v) and v >= 0 for v in estimate):
        forecast = f"{estimate[0] / 3600:.2f}--{estimate[1] / 3600:.2f}"
    else:
        forecast = _hours(estimate)
    if estimate is None and "pilot" in str(progress.get("status", "")).lower() and _count(remaining):
        durations = [u["summary"].get("timings", {}).get("active_seconds") for u in units
                     if u["summary"]["kind"] == _PREFIX + "pilot"]
        durations = [v for v in durations if _number(v) and v > 0]
        typical = sum(durations) / len(durations) if durations else 3600.
        fraction = min(1., counts["step"] / counts["target"]) if _count(counts.get("step")) and counts.get("target", 0) > 0 else 0.
        projected = max(0., remaining - fraction) * typical + 600.
        forecast = f"{.8 * projected / 3600:.2f}--{1.4 * projected / 3600:.2f} (planning)"
    reference = _latest(units, "mc_reference")
    combined = reference.get("combinedrepaircharged_seconds")
    if _number(combined):
        progress["repair_active_seconds"] = combined
        main = reference.get("mainrepaircharged_seconds", 0.)
        if _number(progress.get("cumulative_study_seconds")):
            progress["cumulative_study_seconds"] += combined - main
    if final:
        progress["remaining_budget_seconds"] = max(0., 36000. - progress.get("repair_active_seconds", 0.))
        forecast = "0.00"
    data_counts = {**dataset, **{k: child[k] for k in
        ("acquired_count", "new_acquisitions", "train_count", "validation_count") if _count(child.get(k))}}
    for source, target in (("train_histories", "train_count"), ("validation_histories", "validation_count"),
                           ("acquired_count", "acquired_count")):
        if _count(counts.get(source)):
            data_counts[target] = counts[source]
    stage = progress.get("status", "incomplete; committed progress pending" if any(
        _is_repair(u["path"]) for u in snapshot.get("live", [])) else PENDING)
    return [{"Stage": stage, "Acquired / new": f"{_shown(data_counts.get('acquired_count'))} / {_shown(data_counts.get('new_acquisitions'))}",
        "Usable train / validation": f"{_shown(data_counts.get('train_count'))} / {_shown(data_counts.get('validation_count'))}",
        "Passes": passes if passes is not None else (f"epoch {child['epoch']} in progress" if _count(child.get("epoch")) else PENDING),
        "Fisher": "sealed bulk asset" if fisher else fisher_progress or PENDING,
        "Pilots complete / remaining": f"{_shown(completed)} / {_shown(remaining)}",
        "Charged repair / study h": f"{_hours(progress.get('repair_active_seconds'))} / {_hours(progress.get('cumulative_study_seconds'))}",
        "Remaining forecast / ceiling h": f"{forecast} / {_hours(progress.get('remaining_budget_seconds'))}"}]


def _figure(figure, description):
    import nbformat
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=105)
    figure.clear()
    return nbformat.v4.new_output("display_data", data={
        "image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": description})


def _finish(axes):
    for axis in axes:
        axis.grid(alpha=.2)
        if axis.lines or axis.patches or axis.collections:
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, PENDING, ha="center", va="center", transform=axis.transAxes, fontsize=9)


def _coverage(units):
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    datasets = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "dataset"]
    collections = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "collection"]
    if not datasets and not collections:
        return None
    figure = Figure(figsize=(10, 5), layout="constrained")
    axes = figure.subplots(2, 3).ravel()
    for key, label, color in (("train_count", "Train", "#287d68"), ("validation_count", "Validation", "#ae4e69")):
        if datasets:
            axes[0].plot(range(1, len(datasets) + 1), [d[key] for d in datasets], "o-", label=label, color=color)
    axes[0].set(xlabel="Sealed nested dataset rung", ylabel="Usable histories")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    coverage = (datasets or collections)[-1]["coverage"]
    for axis, key, title in ((axes[1], "dominant_actions", "Teacher dominant action"),
                             (axes[2], "return_bin_counts", "Recorded return bin"),
                             (axes[3], "outcomes", "Episode outcome")):
        values = _mapping(coverage.get(key))
        if values:
            if key == "return_bin_counts":
                bins = sorted((int(label), count) for label, count in values.items())
                axis.bar([label for label, _ in bins], [count for _, count in bins], color="#4b789d")
                axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            else:
                axes_labels = sorted(values, key=str)
                axis.bar([str(v) for v in axes_labels], [values[v] for v in axes_labels], color="#4b789d")
            axis.tick_params(axis="x", labelsize=7)
        axis.set(xlabel=title, ylabel="Episodes" if key == "outcomes" else "Histories")
    geometry = _rows(coverage.get("episode_geometry"))
    for axis, key, label in ((axes[4], "initial_bearing_degrees", "Initial bearing (degrees)"),
                             (axes[5], "initial_distance_m", "Initial distance (m)")):
        values = [r[key] for r in geometry if _number(r.get(key))]
        if values:
            axis.hist(values, bins=min(8, len(values)), color="#ae4e69")
        axis.set(xlabel=label, ylabel="Episodes")
    _finish(axes)
    return _figure(figure, "Sealed nested data sizes, dominant teacher actions, return-bin counts and episode outcomes.")


def _optimization(units):
    from matplotlib.figure import Figure
    fits = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "fit"]
    if not fits:
        return None
    figure = Figure(figsize=(9, 4.7), layout="constrained")
    axes = figure.subplots(2, 2).ravel()
    for index, fit in enumerate(fits, 1):
        count = _mapping(fit.get("counts")).get("train_histories")
        prefix = "Live" if fit.get("provisional") else f"N={count}" if _count(count) else f"Fit {index}"
        stages = dict.fromkeys(row["stage"] for row in fit["curves"])
        for stage in stages:
            rows = [_curve(r) for r in fit["curves"] if r["stage"] == stage]
            exposure = all(_number(r.get("exposures")) for r in rows)
            xkey = "exposures" if exposure else "epoch"
            for axis, fields in zip(axes, (("train_nll", "validation_nll"), ("train_kl", "validation_kl"),
                    ("train_actor_nll", "validation_actor_nll", "train_critic_nll", "validation_critic_nll"),
                    ("gradient_norm", "update_norm", "ltm_gradient_norm", "stm_update_norm"))):
                if axis is axes[2] and fit is not fits[-1]:
                    continue
                for key in fields:
                    points = [(r[xkey], r[key]) for r in rows if _number(r.get(xkey)) and _number(r.get(key))]
                    if points:
                        friendly = key.replace("validation_", "Validation ").replace("train_", "Train ")
                        friendly = friendly.replace("_nll", "").replace("_kl", "").replace("_norm", " norm")
                        label = f"{prefix}: {friendly.replace('_', ' ')}"
                        existing = axis.get_legend_handles_labels()[1]
                        line = next((line for line in axis.lines if line.get_label() == label), None)
                        axis.plot(*zip(*points), "o-" if not key.startswith("validation") else "s--", markersize=3,
                                  color=line.get_color() if line else None,
                                  label="_nolegend_" if label in existing else label)
                axis.set_xlabel("Optimization exposures" if exposure else "Pass / epoch (within stage)")
        nll_rows = [_curve(r) for r in fit["curves"] if r["stage"] == "nll"]
        if nll_rows and _number(nll_rows[0].get("exposures")):
            axes[0].axvline(nll_rows[0]["exposures"], color="gray", linestyle=":",
                           linewidth=1, alpha=.65, label=f"{prefix}: NLL starts")
    for axis, title in zip(axes, ("Joint NLL per fixed history", "Teacher KL per fixed history",
                                "Actor / critic NLL (latest fit)", "Per-update norms")):
        axis.set_title(title, fontsize=10)
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[3].set_yscale("symlog", linthresh=1e-7)
    _finish(axes)
    return _figure(figure, "Fixed-set train/validation NLL and teacher KL; components and update diagnostics where recorded. "
                   "KL and norm axes are logarithmic outside a tiny neighborhood of zero. "
                   "Gray guides mark the change from KL fitting to joint-NLL fitting; earlier NLL values are diagnostics. "
                   "Norms describe optimizer events, not a full-dataset stationarity test. "
                   "Fits and stages are separate curves, not new samples.")


def _diagnostic_rows(summary, *keys):
    for key in keys:
        value = summary.get(key)
        rows = _rows(value) or _rows(_mapping(value).get("rows"))
        if rows:
            return rows
    return []


def _fisher(units):
    from matplotlib.figure import Figure
    assets = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "bootstrap"]
    if not assets:
        return None
    figure = Figure(figsize=(10, 5), layout="constrained")
    axes = figure.subplots(2, 3).ravel()
    for index, asset in enumerate(assets, 1):
        stability = _mapping(asset.get("fisher_stability"))
        rows = _rows(stability.get("nested_draws")) or _diagnostic_rows(asset, "fisher_stability", "stability_rows", "stability")
        for key, axis in (("trace", axes[0]), ("relative_diagonal_disagreement", axes[2]),
                          ("relative_error", axes[2]), ("relative_l2", axes[2]),
                          ("absolute_error", axes[3]), ("absolute_l2", axes[3])):
            points = [(r.get("draws", r.get("draws_per_history")), r.get(key)) for r in rows]
            points = [(x, y) for x, y in points if _number(x) and _number(y)]
            if points:
                axis.plot(*zip(*points), "o-", markersize=3, label=f"asset {index} / {key}")
        history = _rows(stability.get("history_count"))
        if history:
            axes[1].plot([r["history_count"] for r in history], [r["trace"] for r in history], "o-",
                         markersize=3, label=f"asset {index}")
        independent = _mapping(stability.get("independent_subset"))
        for key, axis in (("relative_l2", axes[2]), ("absolute_l2", axes[3])):
            if _number(independent.get(key)) and _number(independent.get("draws_per_history")):
                axis.scatter([independent["draws_per_history"]], [independent[key]], marker="x",
                             label=f"independent labels; N={independent.get('history_count', PENDING)}")
        local = _diagnostic_rows(asset, "local_quadratic", "local_quadratic_rows", "quadratic_checks")
        for key in ("observed_remainder", "corrected_observed_nll_change", "diagonal_quadratic",
                    "exact_kl", "exact_kl_quadratic", "exact_predictive_kl"):
            for direction in dict.fromkeys(str(r.get("direction", "recorded")) for r in local):
                points = [(r.get("scale"), r.get(key)) for r in local if str(r.get("direction", "recorded")) == direction]
                points = [(x, y) for x, y in points if _number(x) and _number(y)]
                if points:
                    targets = axes[4:] if key == "diagonal_quadratic" else [axes[5] if "kl" in key else axes[4]]
                    for axis in targets:
                        axis.plot(*zip(*points), "o-", markersize=3, label=f"asset {index} / {key}")
    for i, (axis, title) in enumerate(zip(axes, ("Trace vs label budget", "Trace vs history count", "Relative L2 disagreement",
                                               "Absolute L2 disagreement", "First-order-corrected NLL", "Exact predictive KL"))):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Actual LTM direction scale" if i >= 4 else "Training histories" if i == 1 else "Model draws per history")
    _finish(axes)
    return _figure(figure, "Bulk per-history Fisher trace/stability and local quadratic diagnostics at the fixed accepted anchor.")


def _panel(panel):
    rows = _rows(panel.get("rows"))
    seeds = [r.get("seed") for r in rows]
    declared = panel.get("seeds")
    valid = bool(rows) and all(type(r.get("success")) is bool for r in rows) and all(_count(s) for s in seeds)
    valid = valid and len(set(seeds)) == len(seeds)
    if isinstance(declared, list):
        valid = valid and set(seeds) == set(declared) and len(seeds) == len(declared)
    return (sum(r["success"] for r in rows) / len(rows), len(rows), tuple(sorted(seeds))) if valid else (None, len(rows), ())


def _capability(units):
    from matplotlib.figure import Figure
    evaluations = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "evaluation"]
    pilots = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "pilot"]
    if not evaluations and not pilots:
        return None
    figure = Figure(figsize=(10, 3.3), layout="constrained")
    axes = figure.subplots(1, 3)
    small = [s for s in evaluations if s.get("small_panel_not_confirmation") is True and _count(s.get("train_count"))]
    if small:
        for checkpoint in ("distillation", "accepted_nll"):
            panels = {}
            for summary in small:
                for panel in summary["evaluations"]:
                    rate, count, ids = _panel(panel)
                    if summary["checkpoint"] == checkpoint and panel.get("difficulty") == "easy" and rate is not None:
                        panels.setdefault(ids, []).append((summary["train_count"], rate, count))
            for panel_index, points in enumerate(panels.values(), 1):
                points.sort()
                axes[0].plot([p[0] for p in points], [p[1] for p in points], "o-", markersize=3,
                             label=f"{'After KL' if checkpoint == 'distillation' else 'After NLL'}; n={points[0][2]}")
        axes[0].set_xlabel("Usable training histories (5-layout diagnostic)")
    else:
        for summary in evaluations:
            for panel in summary["evaluations"]:
                rate, count, _ = _panel(panel)
                if panel.get("difficulty") == "easy" and rate is not None:
                    axes[0].scatter([summary["checkpoint"]], [rate], label=f"{summary['checkpoint']} (n={count})")
    groups = {}
    for pilot in pilots:
        settings = _mapping(pilot.get("settings"))
        seed, asset = settings.get("seed"), settings.get("initial_asset")
        if _count(seed) and isinstance(asset, str) and asset:
            groups.setdefault((seed, asset), {}).setdefault(settings.get("name"), []).append(pilot)
    for (seed, _), conditions in groups.items():
        if any(len(conditions.get(c, [])) != 1 for c in ("replay_ewc", "stm_fd")):
            continue
        arms = {c: conditions[c][0] for c in ("replay_ewc", "stm_fd")}
        for axis, difficulty in zip(axes[1:], ("easy", "hard")):
            points = {}
            for condition, arm in arms.items():
                points[condition] = {}
                for panel in arm["evaluations"]:
                    rate, _, ids = _panel(panel)
                    n = panel.get("acquired_count")
                    if panel.get("difficulty") == difficulty and rate is not None and _count(n):
                        points[condition][panel.get("checkpoint")] = (n, rate, ids)
            checkpoints = set(points["replay_ewc"])
            paired = (checkpoints == set(points["stm_fd"]) and {"initial", "final"} <= checkpoints
                      and all(points["replay_ewc"][c][2] == points["stm_fd"][c][2] for c in checkpoints))
            for condition in arms:
                if paired:
                    measured = sorted(points[condition].values())
                    axis.plot([r[0] for r in measured], [r[1] for r in measured], "o-", markersize=3,
                              label=f"{'Replay/EWC' if condition == 'replay_ewc' else 'FD'} / {seed}")
    for axis, title in zip(axes, ("Initial easy-A capability", "Paired A retention", "Paired B acquisition")):
        axis.set(title=title, ylabel="Held-out success fraction", ylim=(-.03, 1.03))
    for axis in axes[1:]:
        axis.set_xlabel("Unique acquired transitions")
    _finish(axes)
    return _figure(figure, "Initial checkpoint capability regardless of readiness; paired pilot curves require common training seed, "
                   "initial asset, checkpoint and evaluation layout IDs. Each line uses its own measured exposure, "
                   "which can differ slightly at episode boundaries. Unpaired pilots are not pooled.")


def _pilot_rows(units):
    rows = []
    for unit in units:
        summary = unit["summary"]
        if summary["kind"] != _PREFIX + "pilot":
            continue
        settings = _mapping(summary.get("settings"))
        panels = {(p.get("difficulty"), p.get("checkpoint")): p
                  for p in _rows(summary.get("evaluations"))}

        def endpoint(difficulty):
            values = []
            for checkpoint in ("initial", "final"):
                rate, count, _ = _panel(panels.get((difficulty, checkpoint), {}))
                if rate is None:
                    return PENDING
                values.append(f"{round(rate * count)}/{count}")
            return " -> ".join(values)

        seconds = _mapping(summary.get("timings")).get("active_seconds")
        rows.append({"Seed": settings.get("seed"),
            "Method": {"replay_ewc": "Replay/EWC", "stm_fd": "STM FD"}.get(settings.get("name"), settings.get("name")),
            "Incoming intervals": summary.get("acquired_count"),
            "A successes: initial -> final": endpoint("easy"),
            "B successes: initial -> final": endpoint("hard"),
            "Active min": round(seconds / 60, 1) if _number(seconds) else None})
    return rows


def diagnostic_outputs(snapshot, *, notebook_dir=None):
    units = _units(snapshot)
    provisional = _live_fit(snapshot)
    outputs = []
    groups = (
        ("1. Coverage / data size", _coverage, "Counts distinguish acquired transitions, usable histories and episodes. "
         "Coverage includes all available collected histories, not just a fitted prefix and the fixed validation panel. "
         "Dominant actions are recorded teacher argmax labels; outcome bars retain failures. Overlapping contexts are not iid."),
        ("2. Fixed-set optimization", _optimization, "NLL is the declared joint return/action negative log likelihood; "
         "KL measures divergence from recorded teacher actions. Lower is better on the same fixed histories; "
         "validation is diagnostic, not fresh confirmation. Finite or flat loss alone does not prove an MLE."),
        ("3. Bulk Fisher / local quadratic", _fisher, "D0 is the average per-history model Fisher diagonal; accumulated "
         "information is N D0, not a second Fisher. More label draws reduce Monte Carlo error, not increase data count. "
         "Relative diagonal disagreement needs absolute error near zero. NLL remainder subtracts its first-order term; "
         "exact predictive KL is a separate curvature reference, not proof of LAN or exact diagonal agreement."),
        ("4. Initial capability / paired progression", _capability, "A success is retained initial capability; B success is "
         "acquisition on held-out layouts. Only explicit Boolean outcomes enter rates. Repeated panels are not independent "
         "training replicas. Development cannot establish confirmatory sample savings."),
    )
    for title, plot, definition in groups:
        outputs.append(_note(f"{title}. {definition}"))
        if title.startswith("2.") and provisional:
            outputs.append(_note("Initialization in progress: committed diagnostic epochs; fit/selection not sealed. "
                                 "These fixed-set diagnostics do not establish readiness or trajectory efficacy."))
        figure = plot(units + provisional if title.startswith("2.") else units)
        outputs.append(figure if figure is not None else _note(title + ": " + PENDING))
        if title.startswith("1."):
            dataset = _latest(units, "dataset")
            coverage = _mapping(dataset.get("coverage"))
            if dataset:
                outputs.append(_table([{"Acquired": dataset["acquired_count"], "New acquisitions": dataset["new_acquisitions"],
                    "Usable train / validation": f"{dataset['train_count']} / {dataset['validation_count']}",
                    "Episodes": coverage.get("episodes"), "Successes": coverage.get("successes"),
                    "Degenerate teacher coverage": coverage.get("degenerate_teacher_action_coverage"),
                    "Control ranges": {k: v for k, v in _mapping(coverage.get("controls")).items() if k.endswith("_range")}}]))
        if title.startswith("2."):
            fit = _latest(units, "fit")
            readiness = _mapping(fit.get("readiness"))
            if readiness:
                fisher_readiness = _mapping(_latest(units, "bootstrap").get("readiness"))
                outputs.append(_table([{"Readiness": label, "Recorded": value} for label, value in (
                    ("Declared fit rule", readiness.get("ready")),
                    ("Optimization", readiness.get("optimization_ready")),
                    ("Data adequacy", _mapping(readiness.get("data_adequacy")).get("status")),
                    ("Fisher numerical / stability", f"{_shown(fisher_readiness.get('fisher_numerically_ready'))} / "
                     + _shown(fisher_readiness.get("fisher_stability_within_tolerance"))),
                    ("Recorded cautions", "; ".join(str(v) for v in readiness.get("flags", [])[:4]) or None))]))
        if title.startswith("3."):
            reference = _latest(units, "mc_reference")
            if reference:
                comparison = reference["comparisons"]["primary_4_vs_reference"]
                counts = reference.get("counts", {})
                outputs.append(_table([{"Histories": counts.get("N"), "Reference draws / history": counts.get("M"),
                    "Primary trace": comparison.get("primary_trace"), "Reference trace": comparison.get("reference_trace"),
                    "Relative diagonal L2 difference": comparison.get("relative_l2"),
                    "Combined repair h": reference.get("combinedrepaircharged_seconds", 0.) / 3600}]))
                outputs.append(_note("This independent higher-draw estimate is a reference, not truth. "
                    "It isolates model-label Monte Carlo sensitivity on the same training histories and anchor; "
                    "the primary Fisher and all learning comparisons are unchanged."))
        if title.startswith("4."):
            panels = [{"Checkpoint": u["summary"]["checkpoint"], "Domain": panel.get("difficulty"),
                "Success / evaluated": (f"{round(rate * count)} / {count}" if rate is not None else PENDING)}
                for u in units if u["summary"]["kind"] == _PREFIX + "evaluation"
                and u["summary"].get("small_panel_not_confirmation") is not True
                for panel in u["summary"]["evaluations"] for rate, count, _ in [_panel(panel)]]
            if panels:
                outputs.append(_table(panels[-4:]))
            pilots = _pilot_rows(units)
            if pilots:
                outputs.append(_table(pilots))
                outputs.append(_note("Only completed development trajectories appear in this table. "
                    "A and B use their fixed held-out layout panels; initial and final successes are counts, "
                    "not training returns. Active minutes include evaluation. Missing trajectories remain pending."))
            outputs.append(_note("Five-layout rung panels have 20-percentage-point resolution; "
                                 "they are development diagnostics, not fresh significance tests or the full selected panels."))
            probe = _latest(units, "probe")
            if probe:
                outputs.append(_table([{key: probe.get(key) for key in ("selected_learning_rate", "selected_optimizer_steps",
                    "selected_fd_step", "fd_convergence_pass")}]))
    for unit in [u for u in units if u["summary"]["kind"] in (_PREFIX + "evaluation", _PREFIX + "pilot")][-4:]:
        if _rows(unit["summary"].get("videos")):
            campaign = next((p for p in Path(unit["path"]).parents if p.name == CAMPAIGN), Path(unit["path"]).parent)
            bounded = {**unit, "summary": {**unit["summary"], "videos": unit["summary"]["videos"][:2]}}
            outputs.append(_videos(bounded, campaign, notebook_dir or shared.DEFAULT_NOTEBOOK.parent))
    final = _latest(units, "campaign")
    omitted = _rows(final.get("omitted"))
    if omitted:
        outputs.append(_note("Omitted development units (not failures): " + "; ".join(
            f"{r.get('condition', 'unit')}: {r.get('reason', PENDING)}" for r in omitted[:4])))
    outputs.append(_note("Detailed histories, settings and journals remain in the sealed artifacts; missing panels and readiness stay pending."))
    return outputs


def _progress_output(snapshot):
    output = _table(progress_rows(snapshot))
    title = "Phase 3 Fix 1 committed progress: separate 10 h repair ceiling; pending is not zero."
    output.data["text/plain"] = title + "\n" + output.data["text/plain"]
    output.data["text/html"] = "<p><strong>" + title + "</strong></p>" + output.data["text/html"]
    output.metadata["plan19_phase3_fix1_progress"] = True
    return output


def display_progress(snapshot):
    from IPython.display import display
    display(_progress_output(snapshot).data, raw=True)


def display_diagnostics(snapshot, *, notebook_dir=None):
    from IPython.display import display
    for output in diagnostic_outputs(snapshot, notebook_dir=notebook_dir):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_fix1(book, snapshot, *, notebook_dir=None, historical_conclusion=None):
    """Add concise progress and a chronological, idempotent two-cell section."""
    import nbformat
    book.cells = [c for c in book.cells if not c.metadata.get("plan19_phase3_fix1")]
    progress_cell = next((c for c in book.cells if c.metadata.get("report_section") == "progress"), None)
    if progress_cell is not None:
        source = ("\nfrom numerical_experiment.picar_initialization.report import display_progress\n"
                  "display_progress(snapshot)")
        if source not in progress_cell.source:
            progress_cell.source += source
        output = _progress_output(snapshot)
        progress_cell.outputs = [o for o in progress_cell.outputs if not o.get("metadata", {}).get("plan19_phase3_fix1_progress")]
        progress_cell.outputs.append(output)
    directory = str(Path(notebook_dir).resolve()) if notebook_dir is not None else None
    section = [nbformat.v4.new_markdown_cell(_INTRO), nbformat.v4.new_code_cell(
        "from numerical_experiment.picar_initialization.report import display_diagnostics\n"
        f"display_diagnostics(snapshot, notebook_dir={directory!r})",
        outputs=diagnostic_outputs(snapshot, notebook_dir=notebook_dir))]
    active = _units(snapshot) or any(_is_repair(u["path"]) for u in snapshot.get("live", []))
    if active and historical_conclusion:
        section.insert(0, nbformat.v4.new_markdown_cell(historical_conclusion.replace(
            "## Conclusion", "#### Historical Phase 3 Closeout", 1)))
    for i, cell in enumerate(section):
        cell.id = f"plan19-phase3-fix1-{i}"
        cell.metadata["plan19_phase3_fix1"] = True
    end = next((i for i, c in enumerate(book.cells) if c.cell_type == "markdown" and c.source.startswith("## Conclusion")), len(book.cells))
    after = max((i + 1 for i, c in enumerate(book.cells) if c.metadata.get("plan19_phase3")), default=end)
    book.cells[min(after, end):min(after, end)] = section
    return book


def conclusion(snapshot):
    units = _units(snapshot)
    fit, final = _latest(units, "fit"), _latest(units, "campaign")
    readiness = _mapping(fit.get("readiness"))
    ready = readiness.get("ready")
    decision = "Initialization readiness remains pending; no accepted fit is inferred from progress counters."
    if ready is False:
        decision = "Initialization readiness is unresolved. This is not a negative FD efficacy result."
    elif ready is True:
        decision = "The fit's declared readiness rule passed; optimization, data, Fisher and capability evidence remain distinct."
    data_status = _mapping(readiness.get("data_adequacy")).get("status", PENDING)
    fisher = _mapping(_latest(units, "bootstrap").get("readiness"))
    stability = fisher.get("fisher_stability_within_tolerance")
    fisher_status = "within recorded diagnostic tolerance" if stability is True else "unresolved" if stability is False else PENDING
    capability = any(u["summary"]["kind"] == _PREFIX + "evaluation" for u in units)
    evidence = (f"Data adequacy: {data_status}. Fisher stability: {fisher_status}. "
                f"Initial capability: {'recorded panels, not an automatic readiness pass' if capability else PENDING}.")
    pilots = [u for u in units if u["summary"]["kind"] == _PREFIX + "pilot"]
    status = (f"{len(pilots)} sealed development trajectories are descriptive, not confirmation." if pilots else
              "Paired acquisition and retention evidence is pending; no FD winner is inferred.")
    assessment = _mapping(final.get("decision"))
    if assessment.get("retention_matched") is False:
        status += " The recorded comparison is not retention matched."
    elif assessment.get("retention_matched") is True:
        status += " The campaign records retention matching, not confirmatory superiority."
    if pilots:
        endpoints = [p for u in pilots for p in u["summary"]["evaluations"]
                     if p.get("difficulty") == "hard" and p.get("checkpoint") == "final"]
        if len(endpoints) == len(pilots) and all(_panel(p)[0] == 0 for p in endpoints):
            status += " Every completed pilot ended at zero hard-B successes; useful new-task learning is not established."
    coverage = _mapping(_latest(units, "dataset").get("coverage"))
    if coverage.get("degenerate_teacher_action_coverage") is True:
        evidence += " Teacher argmax-action coverage remained degenerate; fitting more histories did not establish diverse demonstrations."
    outcomes = []
    if final:
        for name, label in (("replay_ewc", "Replay/EWC"), ("stm_fd", "STM FD")):
            counts = []
            for unit in pilots:
                if _mapping(unit["summary"].get("settings")).get("name") != name:
                    continue
                for panel in unit["summary"]["evaluations"]:
                    if panel.get("difficulty") == "easy" and panel.get("checkpoint") == "final":
                        rate, count, _ = _panel(panel)
                        if rate is not None:
                            counts.append(f"{round(rate * count)}/{count}")
            if counts:
                outcomes.append(label + " " + ", ".join(counts))
    measured = ("\n\nEndpoint A successes, one result per completed development trajectory: "
                + "; ".join(outcomes) + ".") if outcomes else ""
    next_action = (" Before confirmation, prioritize a separately versioned diagnostic of steering coverage "
                   "and successful exploration; the present pilots do not isolate the acquisition bottleneck.") \
        if final and assessment.get("control_learning_observed") is False else ""
    return ("## Conclusion\n\nPhase 3 Fix 1: " + decision + " " + evidence + " " + status + "\n\n"
        "No sample-savings claim is supported by this development repair. Review fixed-set fitting, demonstration "
        "coverage, bulk-Fisher stability and initial capability separately before choosing further work. "
        "A budget stop is not scientific rejection; more data is warranted only by the diagnostics." + measured + next_action + "\n\n"
        "Stop for review; no automatic Phase 4 or physical motion.")


def build_notebook(root=None, output=shared.DEFAULT_NOTEBOOK):
    """Coordinator-compatible wrapper; generating a report never starts a run."""
    return shared.build_notebook(root, output)
