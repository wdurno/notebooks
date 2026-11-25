"""Bounded, JSON-only Plan 19 Fix 5 reporting.

``build_notebook(root, output, base_notebook=None)`` reads root/progress.json,
sealed unit summaries, and an optional sealed scalar action-analysis sibling.
No manifests, tensors,
checkpoints, runtime modules, or incomplete summaries are opened. Conditions
use settings.condition (or method); replicas use settings.replica (or seed).
Optional forecast.remaining_seconds is a [low, high] range. Unknown diagnostic
fields may coexist with this scalar schema; explicit schema_version must be 1.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import html
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from numerical_experiment.picar import report as shared


VERSION = "plan19-phase3-fix5-v1"
ArtifactError = shared.ArtifactError
MAX_UNITS = 128
MAX_SCAN_DIRS = 2048
MAX_JSON_BYTES = 8 * 1024 * 1024
# The full 28-unit campaign is about 73 MiB; keep a finite reporting bound.
MAX_TOTAL_BYTES = 96 * 1024 * 1024
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100_000
MAX_TABLE_ROWS = 18
MAX_PLOT_GROUPS = 8
_TAG = "plan19_phase3_fix5"
_INTRO = """### Phase 3 Fix 5: Frozen-Head AWAC And Stronger Replay

Earlier predictive fits did not establish improved observed returns. Fix 5
tests cheaper frozen-head learning and stronger replay on A only, with a small
EWC rank survey and one shared signed-square reward recipe. The operational
likelihood explicitly freezes teacher targets, weights and conditioning during
each solve; a stronger self-consistent MLE-efficiency claim remains deferred.

Curves below use completed trajectory summaries only. Raw observed discounted
returns, encoded returns and raw Qwen scores have different units. Training
data NLL excludes FD/EWC penalties: changes in teacher, targets, conditioning,
weights or variance can invalidate absolute comparisons. Held-out NLL is a
common frozen-reference diagnostic only when settings establish that common
reference. Critic MSE is calibration against realized evaluation returns, not
training TD loss. Missing diagnostics are not zero.
"""


def _require(ok, message):
    if not ok:
        raise ArtifactError("Fix 5 schema-incompatible: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _optional_numbers(row, keys, context):
    for key in keys:
        _require(row.get(key) is None or _number(row[key]), context + "." + key + " must be finite or null")


def _rows(value, context):
    _require(isinstance(value, list) and len(value) <= MAX_ROWS, context + " must be a bounded list")
    _require(all(isinstance(row, dict) for row in value), context + " entries must be objects")
    return value


def _schema(value, context):
    _require("schema_version" not in value or type(value["schema_version"]) is int
             and value["schema_version"] == 1, context + " schema_version must be 1")
    _require("version" not in value or value["version"] == VERSION,
             context + " version must be " + VERSION + "; campaigns must not be pooled")


def _validate_summary(summary, context):
    _schema(summary, context)
    operation = summary.get("operation")
    _require(isinstance(operation, str) and bool(operation), context + " requires operation")
    _require(summary.get("kind") == "head_awac_" + operation, context + " kind must match operation")
    _require(isinstance(summary.get("settings"), dict), context + " settings must be an object")
    _require(_number(summary.get("active_seconds")) and summary["active_seconds"] >= 0,
             context + " active_seconds must be nonnegative")
    if operation == "initial_fit":
        result = summary.get("result")
        _require(isinstance(result, dict), context + " initial_fit.result must be an object")
        _require(_count(result.get("epoch")), context + " initial_fit.result.epoch must be nonnegative")
        previous = 0
        for row in _rows(result.get("curves", []), context + ".result.curves"):
            _require(_count(row.get("epoch")) and previous < row["epoch"] <= result["epoch"],
                     context + " initial_fit curve epochs must increase through the stopping epoch")
            previous = row["epoch"]
            _require(all(_number(row.get(key)) and row[key] >= 0 for key in ("train_mse", "validation_mse")),
                     context + " initial_fit curve MSEs must be finite and nonnegative")
        for row in _rows(result.get("inventory", []), context + ".result.inventory"):
            _require(isinstance(row.get("bank"), str) and _count(row.get("count")),
                     context + " initial_fit inventory requires bank and nonnegative count")
    for row in _rows(summary.get("metrics", []), context + ".metrics"):
        _require(_number(row.get("n")) and row["n"] >= 0, context + " metric requires nonnegative n")
        _optional_numbers(row, ("data_nll", "fd", "ewc", "post_step_ewc", "update_seconds", "replay_count", "age_mean",
                                "age_saturated", "visits", "ewc_proposal_difference", "optimizer_exposures"),
                          context + ".metrics")
    for panel in _rows(summary.get("evaluations", []), context + ".evaluations"):
        _require(_number(panel.get("n")) and panel["n"] >= 0, context + " evaluation requires nonnegative n")
        _require(panel.get("acquired_count") is None or _count(panel["acquired_count"]),
                 context + " evaluation acquired_count must be nonnegative or null")
        _require(isinstance(panel.get("environment"), str), context + " evaluation requires environment")
        for episode in _rows(panel.get("episodes"), context + ".episodes"):
            _require(type(episode.get("success")) is bool, context + " episode.success must be Boolean")
            _optional_numbers(episode, ("raw_return", "coded_return", "raw_qwen_mean", "critic_mse", "heldout_nll"),
                              context + ".episodes")
            for video in _rows(episode.get("videos", []), context + ".episode.videos"):
                _require(isinstance(video.get("path"), str) and bool(video["path"]),
                         context + " episode video requires a path")
    if summary.get("archive") is not None:
        archive = summary["archive"]
        _require(isinstance(archive, dict), context + " archive must be an object")
        values = _eigenvalues(archive)
        _require(isinstance(values, list) and len(values) <= 10 and all(_number(x) for x in values),
                 context + " archive.eigenvalues must contain at most ten finite LL^T eigenvalues")
        _optional_numbers(archive, ("trace", "correlated_trace_fraction"), context + ".archive")
        _require(archive.get("rank") is None or _count(archive["rank"]), context + " archive.rank must be nonnegative")


def _eigenvalues(archive):
    return archive.get("eigenvalues", archive.get("factor_eigenvalues", []))


def _validate_active(progress):
    active = progress.get("active_progress", {})
    _require(isinstance(active, dict), "progress.active_progress must be an object")
    latest = active.get("latest_fit")
    if latest is not None:
        _require(isinstance(latest, dict), "active_progress.latest_fit must be an object")
        _optional_numbers(latest, ("n", "data_nll", "fd", "ewc", "update_seconds", "replay_count",
                                  "age_mean", "age_saturated", "visits", "optimizer_exposures",
                                  "post_step_ewc", "ewc_proposal_difference"), "active_progress.latest_fit")
    for row in _rows(active.get("evaluation_preview", []), "active_progress.evaluation_preview"):
        _require(_number(row.get("n")) and row["n"] >= 0, "evaluation_preview.n must be nonnegative")
        _require(isinstance(row.get("environment"), str), "evaluation_preview.environment must be text")
        _require(_count(row.get("episodes")), "evaluation_preview.episodes must be a nonnegative integer")
        _optional_numbers(row, ("success", "raw_return"), "evaluation_preview")
        _require(row.get("success") is None or 0 <= row["success"] <= 1,
                 "evaluation_preview.success must be a fraction")


def load_report(root, *, action_report=None):
    """Read scalar artifacts once, without repairing or inspecting runtime data."""
    root = Path(root)
    total = 0

    def read(path):
        nonlocal total
        value = shared._json(path, limit=MAX_JSON_BYTES)
        total += path.stat().st_size
        _require(total <= MAX_TOTAL_BYTES, "total scalar JSON size limit exceeded")
        return value

    progress_path = root / "progress.json"
    progress = read(progress_path) if progress_path.exists() else {}
    if progress_path.exists():
        _schema(progress, "progress")
        for key in ("status", "stage"):
            _require(isinstance(progress.get(key), str), "progress." + key + " must be text")
        for key in ("completed_units", "planned_units"):
            _require(_count(progress.get(key)), "progress." + key + " must be a nonnegative integer")
        _require(progress["completed_units"] <= progress["planned_units"], "progress unit counts disagree")
        _require(_number(progress.get("charged_seconds")) and progress["charged_seconds"] >= 0,
                 "progress.charged_seconds must be nonnegative")
        _require(progress.get("active_unit") is None or isinstance(progress["active_unit"], (str, dict)),
                 "progress.active_unit must be text, object or null")
        _validate_active(progress)
    units = []
    directories = root / "units"
    if directories.exists():
        _require(not directories.is_symlink(), "units directory must not be a symlink")
        children = []
        for child in directories.iterdir():
            children.append(child)
            _require(len(children) <= MAX_SCAN_DIRS, "unit directory scan limit exceeded")
        for child in sorted(children):
            if not child.is_dir() or not (child / "COMPLETED").is_file():
                continue
            _require(not child.is_symlink(), "completed unit must not be a symlink")
            _require(len(units) < MAX_UNITS, "completed unit limit exceeded")
            summary = read(child / "summary.json")
            _validate_summary(summary, str(child / "summary.json"))
            units.append({"name": child.name, "summary": summary})
    complete = (progress.get("status") in ("complete", "completed", "complete_for_review")
                and progress.get("completed_units") == progress.get("planned_units")
                and progress.get("completed_units") == len(units))
    snapshot = {"root": str(root.resolve()), "progress": progress, "units": units, "complete": complete}
    snapshot["action_diagnostics"] = _load_action_diagnostics(root, units, action_report)
    return snapshot


def _load_action_diagnostics(root, units, directory=None):
    explicit = directory is not None
    directory = Path(directory) if explicit else root.with_name(root.name + "-action-diagnostics-v2")
    marker, path = directory / "COMPLETED", directory / "summary.json"
    _require(not explicit or marker.is_file(), "explicit action report is not complete")
    if not marker.is_file():
        return None
    _require(not directory.is_symlink() and not marker.is_symlink() and marker.stat().st_size <= 128,
             "invalid action-report completion marker")
    value = shared._json(path, limit=MAX_JSON_BYTES)
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == marker.read_text().strip(),
             "action-report checksum mismatch")
    identity = value.get("identity", {})
    _require(identity.get("version") == "plan19-fix5-turn-bouts-v2" and identity.get("schema_version") == 1,
             "unsupported action-report schema")
    _require(all(_number(identity.get(k)) and identity[k] > 0 for k in ("turn_degrees", "moving_mps")),
             "invalid action thresholds")
    known = {u["name"]: u for u in units}
    sources = _rows(identity.get("sources"), "action-report.sources")
    names = [row.get("unit") for row in sources]
    _require(all(isinstance(name, str) for name in names)
             and len(names) == len(set(names)) and all(name in known for name in names),
             "action-report unit identities differ")
    for row in sources:
        summary_path = root / "units" / row["unit"] / "summary.json"
        _require(row.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                 "action-report source summary differs")
    analyses = _rows(value.get("units"), "action-report.units")
    _require([u.get("name") for u in analyses] == names, "action-report coverage identities differ")
    for unit in analyses:
        expected = {e["episode_id"]: e for e in known[unit["name"]]["summary"].get("episodes", [])}
        episodes = _rows(unit.get("episodes"), "action-report.episodes")
        _require([e.get("episode_id") for e in episodes] == list(expected), "action-report episode identities differ")
        for row in episodes:
            original = expected[row["episode_id"]]
            _require(_number(row.get("n")) and row["n"] == original["n"] and row["n"] >= 0,
                     "invalid action-report acquisition")
            _require(_number(row.get("sigma_m")) and row["sigma_m"] > 0,
                     "invalid action-report difficulty")
            _require(_count(row.get("recovered_intervals")) and _count(row.get("expected_intervals"))
                     and row["recovered_intervals"] <= row["expected_intervals"]
                     and row["expected_intervals"] == len(original["raw_stage_rewards"]),
                     "invalid action-report interval coverage")
            complete = row["expected_intervals"] > 0 and row["recovered_intervals"] == row["expected_intervals"]
            _require(type(row.get("complete_history")) is bool and row["complete_history"] == complete,
                     "invalid action-report complete-history flag")
            if complete:
                _require(_count(row.get("turn_bouts")) and row["turn_bouts"] <= row["expected_intervals"],
                         "invalid action-report bout count")
                _optional_numbers(row, ("mean_abs_steering_deg",), "action-report")
                _require(_count(row.get("moving_commands")) and row["moving_commands"] <= row["expected_intervals"],
                         "invalid action-report moving-command count")
            else:
                _require(all(row.get(k) is None for k in ("turn_bouts", "mean_abs_steering_deg", "moving_commands")),
                         "partial action histories cannot become full-episode outcomes")
    return value


def _shown(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    elif isinstance(value, float):
        value = f"{value:.4g}"
    value = str(value)
    return value if len(value) <= 220 else value[:217] + "..."


def _table(rows, title, *, note=None):
    import nbformat

    columns = list(rows[0])
    esc = lambda value: html.escape(_shown(value))
    body = "".join("<tr>" + "".join("<td>" + esc(row.get(col)) + "</td>" for col in columns) + "</tr>"
                   for row in rows[:MAX_TABLE_ROWS])
    content = ("<h4>" + html.escape(title) + "</h4><table><thead><tr>"
               + "".join("<th>" + html.escape(col) + "</th>" for col in columns)
               + "</tr></thead><tbody>" + body + "</tbody></table>")
    if len(rows) > MAX_TABLE_ROWS:
        content += f"<p>Showing {MAX_TABLE_ROWS}/{len(rows)} rows; full records remain in JSON artifacts.</p>"
    if note:
        content += "<p>" + html.escape(note) + "</p>"
    return nbformat.v4.new_output("display_data", data={"text/html": content,
        "text/plain": title + "\n" + "\n".join(str({key: _shown(value) for key, value in row.items()})
                                                  for row in rows[:MAX_TABLE_ROWS]) + ("\n" + note if note else "")})


def progress_outputs(snapshot):
    p = snapshot["progress"]
    active = p.get("active_unit")
    counters = p.get("active_progress", active if isinstance(active, dict) else {})
    forecast = p.get("forecast")
    remaining = forecast.get("remaining_seconds") if isinstance(forecast, dict) else None
    if isinstance(remaining, list) and len(remaining) == 2 and all(_number(x) and x >= 0 for x in remaining):
        historical = forecast.get("admission_note")
        forecast = (f"Historical short-pilot range: {remaining[0] / 3600:.2f}--{remaining[1] / 3600:.2f} h; "
                    "not a current remaining-work estimate" if historical else
                    f"{remaining[0] / 3600:.2f}--{remaining[1] / 3600:.2f} h remaining")
    outputs = [_table([{
        "Status": p.get("status", "Not started / incomplete"), "Stage": p.get("stage"),
        "Completed / planned": f"{p.get('completed_units', len(snapshot['units']))} / {p.get('planned_units', '?')}",
        "Sealed units": len(snapshot["units"]), "Active unit": active,
        "Acquisitions": p.get("acquisitions", counters.get("n", counters.get("acquisitions"))),
        "Accepted fits": p.get("accepted_fits", counters.get("fits", counters.get("accepted_fits"))),
        "Charged h": p["charged_seconds"] / 3600 if "charged_seconds" in p else None,
        "Recorded forecast": forecast,
    }], "Fix 5 Current Progress (earlier progress blocks are historical)")]
    if counters.get("latest_fit"):
        outputs.append(_table([{key: counters["latest_fit"].get(key) for key in
            ("n", "data_nll", "fd", "ewc", "update_seconds", "replay_count", "age_mean", "age_saturated",
             "visits", "optimizer_exposures", "ewc_proposal_difference")}],
             "Latest Fit: Incomplete Live Preview (not a sealed trajectory)"))
    preview = counters.get("evaluation_preview", [])
    if preview:
        outputs.append(_table([{key: row.get(key) for key in ("n", "environment", "success", "raw_return", "episodes")}
                               for row in preview], "Evaluation: Incomplete Live Preview (excluded from sealed curves)"))
    return outputs


def _condition(summary):
    settings = summary["settings"]
    return str(settings.get("condition", settings.get("method", summary.get("method", "unspecified"))))


def _group_key(summary):
    settings = {key: value for key, value in summary["settings"].items() if key not in ("seed", "replica")}
    return json.dumps({"condition": _condition(summary), "settings": settings}, sort_keys=True, separators=(",", ":"))


def _phase(summary):
    return summary["settings"].get("phase", summary["settings"].get("stage", "unspecified"))


def _label(summary):
    # Keep development and fresh assessment distinct, including differing ranks.
    settings = summary["settings"]
    pieces = [_condition(summary)]
    for key in ("phase", "stage", "ewc_rank"):
        if key in settings:
            pieces.append(f"{key}={settings[key]}")
    recipe = settings.get("recipe")
    if isinstance(recipe, dict):
        names = {"capacity": "C", "batch_size": "b", "rounds": "rounds", "fd_strength": "FD", "ridge": "ridge"}
        pieces.append(", ".join(f"{label}={recipe[key]}" for key, label in names.items() if key in recipe))
    return " / ".join(pieces)


def _resource(summary):
    kinds = ("resource_pilot", "resource_development", "timing")
    return _phase(summary) in kinds or summary["operation"] in kinds


def _legend_labels(units):
    groups = {_group_key(u["summary"]): u["summary"] for u in units}
    labels = {}
    for key, summary in sorted(groups.items()):
        settings = summary["settings"]
        recipe = settings.get("recipe", settings)
        method = {"mc_replay": "MC replay", "td_replay": "TD replay", "td_fd": "TD + FD"}.get(
            _condition(summary), _condition(summary)[:20])
        rank = settings.get("ewc_rank", recipe.get("rank", summary.get("archive", {}).get("rank")))
        if summary["operation"] in ("initial_fisher", "geometry_fisher"):
            method = "Initial Fisher" if summary["operation"] == "initial_fisher" else "Geometry Fisher"
        phase = {"development": "dev", "retention_calibration": "cal"}.get(_phase(summary))
        if phase:
            method += f" ({phase})"
        parts = [method]
        for name, value in (("r", rank), ("FD", recipe.get("fd_strength")), ("ridge", recipe.get("ridge"))):
            if value is not None:
                parts.append(f"{name}={value:.3g}" if _number(value) else f"{name}={str(value)[:12]}")
        labels[key] = " ".join(parts)
    duplicates = defaultdict(list)
    for key, label in labels.items():
        duplicates[label].append(key)
    for label, keys in duplicates.items():
        if len(keys) > 1:
            for index, key in enumerate(keys, 1):
                labels[key] = f"{label} (recipe {index})"
    return labels


def _mean(values):
    return math.fsum(x / len(values) for x in values) if values else None


def evaluation_rows(snapshot):
    """Episode means within trajectory, then equal-trajectory means at exact n."""
    grouped = defaultdict(list)
    for unit in snapshot["units"]:
        summary = unit["summary"]
        if summary["operation"] != "trajectory":
            continue
        panels = defaultdict(list)
        for panel in summary.get("evaluations", []):
            if panel["environment"] in ("initial_A", "moderate_A"):
                panels[panel["environment"], panel["n"]].extend(panel["episodes"])
        for (environment, n), episodes in panels.items():
            if episodes:
                grouped[_group_key(summary), environment, n].append((unit, episodes))
    rows = []
    for (_, environment, n), entries in sorted(grouped.items()):
        row = {"Condition": _label(entries[0][0]["summary"]), "_group": _group_key(entries[0][0]["summary"]),
               "Environment": environment, "n": n,
               "Episodes": sum(len(episodes) for _, episodes in entries),
               "Trajectories": len(entries)}
        replicas = [u["summary"]["settings"].get("replica", u["summary"]["settings"].get("seed", u["summary"].get("seed")))
                    for u, _ in entries]
        row["Replicas"] = len({str(r) for r in replicas}) if all(r is not None for r in replicas) else None
        acquired = [p["acquired_count"] for u, _ in entries for p in u["summary"].get("evaluations", [])
                    if p["environment"] == environment and p["n"] == n and p.get("acquired_count") is not None]
        row["Actual acquisition range"] = [min(acquired), max(acquired)] if acquired else None
        for key in ("success", "raw_return", "coded_return", "raw_qwen_mean", "critic_mse", "heldout_nll"):
            means = [_mean([float(ep[key]) for ep in episodes if ep.get(key) is not None]) for _, episodes in entries]
            row[key] = _mean([value for value in means if value is not None])
            row[key + " count"] = sum(ep.get(key) is not None for _, episodes in entries for ep in episodes)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["Condition"], r["Environment"], r["n"]))


def _replica_key(summary):
    identity = []
    for key in ("replica", "seed"):
        if key not in summary["settings"] and key not in summary:
            continue
        value = summary["settings"].get(key, summary.get(key))
        if type(value) not in (int, str) or value == "":
            return None
        if key in summary and (type(summary[key]) is not type(value) or summary[key] != value):
            return None
        identity.append((key, type(value).__name__, value))
    return tuple(identity) or None


def _confirmation_pairs(snapshot):
    """Unique trajectories per exact identity, grouped by both methods' recipes."""
    if not snapshot["complete"]:
        return []
    indexed = defaultdict(list)
    for unit in snapshot["units"]:
        summary = unit["summary"]
        if summary["operation"] != "trajectory" or _phase(summary) != "confirmation" or _resource(summary):
            continue
        identity = _replica_key(summary)
        if identity is not None:
            indexed[_condition(summary), identity].append(summary)
    # Repeated trajectories, including differing recipes for one seed, are ambiguous.
    unique = {key: entries[0] for key, entries in indexed.items() if len(entries) == 1}
    grouped = defaultdict(list)
    for (method, identity), left in unique.items():
        label = {"mc_replay": "MC replay - TD replay (Bellman control)",
                 "td_fd": "TD + FD - TD replay"}.get(method)
        right = unique.get(("td_replay", identity))
        if label is not None and right is not None:
            grouped[label, _group_key(left), _group_key(right)].append((left, right))
    counts, indices = defaultdict(int), defaultdict(int)
    for label, _, _ in grouped:
        counts[label] += 1
    result = []
    for (label, _, _), pairs in sorted(grouped.items()):
        indices[label] += 1
        name = label if counts[label] == 1 else f"{label} (recipe pair {indices[label]})"
        result.append((name, pairs))
    return result


def _paired_budget(left, right):
    budget = left["settings"].get("total")
    other = right["settings"].get("total")
    return budget if _count(budget) and budget > 0 and _count(other) and budget == other else None


def _endpoint_mean(summary, environment, n, metric):
    # Shared initialization is not an endpoint or an independent replica.
    episodes = [episode for panel in summary.get("evaluations", [])
                if panel["environment"] == environment and panel["n"] == n
                and not panel.get("shared_initial_source") for episode in panel["episodes"]]
    return _mean([float(ep[metric]) for ep in episodes if ep.get(metric) is not None])


def paired_endpoint_rows(snapshot):
    """Final confirmation gaps: episode means, then equal paired-replica means."""
    rows = []
    for comparison, pairs in _confirmation_pairs(snapshot):
        n = _paired_budget(*pairs[0])
        if n is None:
            continue
        for environment, endpoint in (("initial_A", "initial_A retention"), ("moderate_A", "moderate_A acquisition")):
            for metric, label, scale in (("raw_return", "Raw observed return", 1.), ("success", "Success (pp)", 100.)):
                gaps = []
                for left, right in pairs:
                    a, b = (_endpoint_mean(summary, environment, n, metric) for summary in (left, right))
                    if a is not None and b is not None:
                        gaps.append(scale * (a - b))
                if gaps:
                    rows.append({"Comparison": comparison, "Endpoint": endpoint, "Metric": label,
                                 "Nominal n": n, "Pairs": len(gaps), "Mean gap": _mean(gaps),
                                 "Individual-gap range (not CI)": [min(gaps), max(gaps)]})
    return rows


def paired_cost_rows(snapshot):
    """Mean of paired active-time ratios, never a ratio of pooled durations."""
    rows = []
    for comparison, pairs in _confirmation_pairs(snapshot):
        ratios = [left["active_seconds"] / right["active_seconds"] for left, right in pairs
                  if _paired_budget(left, right) is not None
                  and left["active_seconds"] > 0 and right["active_seconds"] > 0]
        if ratios:
            rows.append({"Comparison": comparison, "Nominal n": _paired_budget(*pairs[0]), "Pairs": len(ratios),
                         "Mean active-time ratio (left/TD)": _mean(ratios),
                         "Individual-ratio range (not CI)": [min(ratios), max(ratios)]})
    return rows


def rank_development_rows(snapshot):
    """Keep each full development trajectory separate from fresh assessment."""
    rows = []
    for unit in snapshot["units"]:
        summary = unit["summary"]
        if summary["operation"] != "trajectory" or _phase(summary) != "development":
            continue
        settings = summary["settings"]
        n = settings.get("total")
        if not _count(n) or not n:
            continue
        row = {"Method": {"td_replay": "TD replay", "td_fd": "TD + FD"}.get(
                   _condition(summary), _condition(summary)),
               "EWC rank": settings.get("ewc_rank"), "Seed": settings.get("seed"),
               "Nominal n": n, "Active h": summary["active_seconds"] / 3600}
        for environment, label in (("initial_A", "Initial A"), ("moderate_A", "Moderate A")):
            episodes = [ep for p in summary.get("evaluations", [])
                        if p["n"] == n and p["environment"] == environment
                        and not p.get("shared_initial_source") for ep in p["episodes"]]
            row[label + " return"] = _mean([ep["raw_return"] for ep in episodes
                                            if ep.get("raw_return") is not None])
            row[label + " wins"] = f"{sum(ep['success'] for ep in episodes)}/{len(episodes)}" if episodes else None
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["Method"]), str(row["EWC rank"]), str(row["Seed"])))


def rank_selection_notes(snapshot):
    notes = """#### EWC Rank: Development And Selection

Plan 18's four-replica development comparison favored rank-64-plus-diagonal
EWC with STM FD over its diagonal counterpart: novel risk 2.2227 versus
2.3275, with better retention. That motivates this representation test.
EWC rank controls retained **correlations between LTM parameters**, not the
number of trainable parameters; the full diagonal is preserved. STM LoRA
rank remains one in every condition. See [Plan 18, Phase 9E](../../mnist_experiment/plan18.md).

Fix 5 screened ranks 0, 3, 5 and 10 on cached data. Only 0 and 10 received
full learning trajectories, one development replica per method/rank, with
two final evaluation episodes per environment. Ranks 3 and 5 have no full
learning comparison. The fresh assessment subsequently used rank zero only.
The development plot and table below are therefore separate from fresh results.
"""
    selections = [u["summary"]["selected"] for u in snapshot["units"]
                  if u["summary"]["operation"] == "selection" and isinstance(u["summary"].get("selected"), dict)]
    if len(selections) == 1:
        selected = selections[0]
        scores = selected.get("rank_scores", {})
        values = ", ".join(f"rank {rank}: {score:.4f}" for rank, score in sorted(scores.items()) if _number(score))
        if values:
            notes += (f"\nThe recorded selection scores were **{values}**. Each score sums four "
                      "endpoint mean raw returns: initial A and moderate A, under TD replay and TD + FD. "
                      "It is not a sample-efficiency statistic or an uncertainty-adjusted comparison. "
                      f"The selected common rank was {selected.get('recipe', {}).get('rank')}. "
                      "Retention refinement occurred after rank selection, only at that selected rank. "
                      "This small screen does not establish that diagonal EWC is preferable, or that "
                      "a larger correlated archive cannot help STM FD.\n")
    return notes


def rank_outputs(snapshot):
    from matplotlib.figure import Figure

    rows = rank_development_rows(snapshot)
    if not rows:
        return []
    units = [u for u in snapshot["units"] if u["summary"]["operation"] == "trajectory"
             and _phase(u["summary"]) == "development"]
    figure = Figure(figsize=(10, 3.8))
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.09, right=.98, bottom=.18, top=.73, wspace=.32)
    legends = _legend_labels(units)
    evaluations = evaluation_rows({"units": units})
    for axis, environment, title in zip(axes, ("initial_A", "moderate_A"),
                                        ("Initial A: retention", "Moderate A: acquisition")):
        for unit in units[:MAX_PLOT_GROUPS]:
            summary = unit["summary"]
            key = _group_key(summary)
            points = [r for r in evaluations if r["_group"] == key and r["Environment"] == environment]
            if points:
                axis.plot([r["n"] for r in points], [r["raw_return"] for r in points],
                          marker="o", linestyle="-" if summary["settings"].get("ewc_rank") == 0 else "--",
                          color="#2463a2" if _condition(summary) == "td_replay" else "#23845b",
                          label=legends[key])
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Acquired intervals (nominal)")
        axis.set_ylabel("Mean raw discounted return")
    _finish(axes, legend=False)
    _shared_legend(figure)
    outputs = [_figure(figure, "EWC rank development: solid rank zero, dashed positive rank; TD replay blue, TD + FD green. One development replica, two held-out episodes per environment/checkpoint; no confidence intervals.")]
    columns = ("Method", "EWC rank", "Initial A return", "Initial A wins",
               "Moderate A return", "Moderate A wins", "Active h")
    outputs.append(_table([{key: row[key] for key in columns} for row in rows],
                          "Full-Curriculum Rank Development", note=(
                              "Raw discounted return: higher is better. One trajectory per row; the same development "
                              "seed and nominal 1,024-interval budget. All four endpoint success counts tie. "
                              "Active hours include whole trajectories and evaluation, not isolated rank overhead.")))
    cached = []
    for unit in snapshot["units"]:
        if unit["summary"]["operation"] != "cached_screen":
            continue
        for candidate in unit["summary"].get("candidates", []):
            recipe = candidate.get("recipe", {})
            if (recipe.get("capacity"), recipe.get("batch_size"), recipe.get("rounds")) == (256, 64, 4):
                cached.append(candidate)
    timing = []
    for rank in sorted({c["recipe"]["rank"] for c in cached}):
        group = [c for c in cached if c["recipe"]["rank"] == rank]
        # Ambiguous duplicate screens must not be silently pooled.
        control = [c for c in group if c["recipe"].get("fd_strength") == 0]
        fd = [c for c in group if c["recipe"].get("fd_strength") == 1]
        if len(control) == len(fd) == 1:
            timing.append({"EWC rank": rank, "Fit without FD (ms)": 1000 * control[0]["fit_seconds"],
                           "Fit with FD=1 (ms)": 1000 * fd[0]["fit_seconds"],
                           "Retirement with FD=1 (ms)": 1000 * fd[0]["retirement_seconds"]})
    if timing:
        outputs.append(_table(timing, "Cached Rank Cost: 65 Records, Three Inner Steps", note=(
            "Median of three fits; one retirement timing per candidate. This screen used default likelihood "
            "scales and FD strength 1, not the selected online strength 12.14. It supports feasibility, "
            "not a prediction of sample efficiency or a substitute for full learning trajectories.")))
    return outputs


def _plot_units(snapshot):
    trajectories = [u for u in snapshot["units"] if u["summary"]["operation"] == "trajectory" and not _resource(u["summary"])]
    confirmation = [u for u in trajectories if _phase(u["summary"]) == "confirmation"]
    candidates = confirmation or [u for u in trajectories if _phase(u["summary"]) not in
                                  ("resource_development", "resource_pilot", "timing")]
    keys = sorted({_group_key(u["summary"]) for u in candidates})[:MAX_PLOT_GROUPS]
    return [u for u in candidates if _group_key(u["summary"]) in keys]


def _figure(figure, description):
    import base64
    import io
    import nbformat

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    figure.clear()
    return nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode(),
                                                       "text/plain": description})


def _finish(axes, *, legend=True):
    for axis in axes:
        axis.grid(alpha=.2)
        if legend and axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
        elif not axis.has_data():
            axis.text(.5, .5, "Not collected", ha="center", va="center", transform=axis.transAxes)


def _shared_legend(figure):
    handles = {}
    for axis in figure.axes:
        items, labels = axis.get_legend_handles_labels()
        for handle, label in zip(items, labels):
            handles.setdefault(label, handle)
    if handles:
        figure.legend(list(handles.values()), list(handles), loc="upper left", bbox_to_anchor=(.07, .99),
                      ncol=2, fontsize=7, frameon=False)


def _initialization_outputs(snapshot):
    fits = [u for u in snapshot["units"] if u["summary"]["operation"] == "initial_fit"
            and u["summary"].get("result", {}).get("curves")]
    if not fits:
        return []
    from matplotlib.figure import Figure

    # One panel, never an overlay of separately fitted initializers.
    unit = fits[-1]
    summary, result = unit["summary"], unit["summary"]["result"]
    curves = result["curves"]
    first, last = curves[0], curves[-1]
    best = min(curves, key=lambda row: row["validation_mse"])
    figure = Figure(figsize=(10, 4))
    axis = figure.subplots()
    figure.subplots_adjust(left=.08, right=.98, top=.86, bottom=.30)
    for key, label in (("train_mse", "Train (pre-step)"), ("validation_mse", "Validation (post-step)")):
        axis.plot([r["epoch"] for r in curves], [r[key] for r in curves], label=label)
    axis.scatter([best["epoch"]], [best["validation_mse"]], color="#287d68", zorder=3,
                 label=f"Lowest recorded validation: epoch {best['epoch']}")
    axis.set(title=f"Initial critic fit: stopped at epoch {result['epoch']}",
             xlabel="Epoch", ylabel="Twin-critic MSE (cached return targets)")
    _finish([axis])
    inventory = result.get("inventory", [])
    counts = {bank: sum(row["count"] for row in inventory if row["bank"] == bank) for bank in ("ltm", "stm")}
    dimensions = (f"LTM {counts['ltm']:,} / STM {counts['stm']:,} inventory coordinates (not fitted-parameter counts)."
                  if inventory else "Inventory not recorded.")
    description = (
        f"Train MSE {first['train_mse']:.4g} -> {last['train_mse']:.4g}; "
        f"validation {first['validation_mse']:.4g} -> {last['validation_mse']:.4g}; "
        f"lowest validation {best['validation_mse']:.4g} at epoch {best['epoch']}.\n"
        f"{_shown(summary.get('train_count'))} train / {_shown(summary.get('validation_count'))} validation records; " + dimensions + "\n"
        "Best-validation weights restored, not final-epoch weights. This curve alone establishes neither convergence nor navigation quality."
    )
    if len(fits) > 1:
        description += f"\nShowing {unit['name']} only; {len(fits)} completed initializer fits are not pooled."
    figure.text(.08, .035, description, fontsize=8, va="bottom")
    return [_figure(figure, "Initialization Diagnostic\n" + description)]


def _fit_diagnostic_glossary(visit_key):
    import nbformat

    visits = (
        "- **visits**: Mean number of accepted fits that the selected transitions have participated in, "
        "including this fit. These are reuse counts, not new observations or individual optimizer steps.\n"
        if visit_key == "visits" else
        "- **optimizer exposures**: Batch size times optimizer steps in this fit; repeated uses are "
        "not additional acquired observations.\n"
    )
    text = (
        "#### Training Diagnostic Headers\n\n"
        "Each point is an accepted head fit (three optimizer steps in the selected recipe). "
        "Data NLL, FD and EWC proposal difference are recorded at its first optimizer step; "
        "EWC after fit uses the accepted parameters relative to the anchor from before that fit. "
        "The former first-step EWC curve was exactly zero at every fit because each accepted fit "
        "recenters the anchor. This was an uninformative measurement time, not evidence of no EWC use.\n\n"
        "- **data nll**: Mean conditional actor-plus-critic negative log likelihood of the fitting batch, "
        "excluding FD, EWC, ridge and critic-disagreement penalties. Targets and advantage weights change, "
        "so this is not a fixed-reference test loss.\n"
        "- **replay count**: Number of transitions used in this fit, including the current transition "
        "and sampled replay. This is the fitting batch, not total buffer occupancy (up to 65 versus "
        "a 256-transition buffer here).\n"
        "- **age mean**: Mean transition age in the fitting batch, divided by replay capacity and clipped "
        "to [0, 1]. Zero means newly acquired; one means at least one buffer-capacity's worth of incoming "
        "intervals old. FD uses the square of each transition's age as its weight.\n"
        "- **age saturated**: Fraction of the fitting batch whose clipped age has reached one, "
        "and therefore receives the full FD age weight.\n"
        + visits +
        "- **update seconds**: Elapsed time inside one accepted fitting routine, including its optimizer "
        "steps; not end-to-end control, visual encoding or Qwen reward-scoring time.\n"
        "- **fd**: Age-weighted STM finite-difference KL penalty, including the selected penalty "
        "coefficient. Zero in the no-FD controls.\n"
        "- **ewc after fit**: Saved post-fit LTM quadratic penalty, including the archive weight, "
        "measured against that fit's original anchor (not the newly recentered one). "
        "It covers the represented archive, not just the newest retired observation. "
        "It measures local LTM movement under the archive metric, not memory volume; "
        "a moving anchor and adaptive archive do not imply monotone growth. "
        "A small value alone does not establish STM-to-LTM transfer. Missing post-fit measurements "
        "are not replaced by first-step zeros.\n"
        "- **ewc proposal difference**: Euclidean norm of the difference between the first LTM step "
        "with EWC and the otherwise identical step without EWC. A positive value shows EWC changes "
        "the update even when its starting penalty is zero."
    )
    return nbformat.v4.new_output("display_data", data={"text/markdown": text, "text/plain": text})


def fit_diagnostic_glossary(snapshot):
    trajectories = [u for u in _plot_units(snapshot) if u["summary"].get("metrics")]
    if not trajectories:
        return ""
    visit_key = "visits" if any("visits" in row for u in trajectories for row in u["summary"]["metrics"]) else "optimizer_exposures"
    return _fit_diagnostic_glossary(visit_key)["data"]["text/markdown"]


def recording_index(snapshot, notebook_directory):
    """HTML index of existing scalar-linked videos; never rerun an episode."""
    root, directory = Path(snapshot["root"]), Path(notebook_directory).resolve()
    units = _plot_units(snapshot)
    labels = _legend_labels(units)
    groups = defaultdict(lambda: defaultdict(list))
    seen_shared = set()
    for unit in units:
        summary = unit["summary"]
        for panel in summary.get("evaluations", []):
            source = panel.get("shared_initial_source")
            if source:
                identity = (source, panel["environment"], panel["n"])
                if identity in seen_shared:
                    continue
                seen_shared.add(identity)
            condition = "Shared initialization (all methods)" if source else labels[_group_key(summary)]
            seed = "Shared" if source else "Seed " + _shown(summary["settings"].get("seed", summary.get("seed")))
            title = f"{seed}; n={_shown(panel['n'])}"
            outcomes = {True: [], False: []}
            for episode in panel["episodes"]:
                links = []
                seen = set()
                for video in episode.get("videos", []):
                    path = Path(video["path"])
                    path = path if path.is_absolute() else root / path
                    path = path.resolve()
                    if path in seen or path.suffix.lower() != ".mp4" or not path.is_file():
                        continue
                    seen.add(path)
                    relative = Path(os.path.relpath(path, directory)).as_posix()
                    href = html.escape(quote(relative, safe="/"), quote=True)
                    layout = html.escape(_shown(episode.get("seed")))
                    links.append(f'<a href="{href}">Watch layout {layout}</a>')
                outcomes[episode["success"]].append(links)
            groups[condition][panel["environment"]].append((title, outcomes))
    if not groups:
        return ""
    text = [
        "<h4>Episode Recordings</h4>\n",
        "<p>These clips correspond to the same sealed cohort as the success curves above. "
        "Each saved MP4 combines the robot view and a reader-only top-down view. "
        "The recording policy preselected the first layout at initialization and at the final checkpoint, "
        "not winners or representative episodes; midpoint episodes were not recorded. "
        "Counts below distinguish missing recordings from evaluated failures. "
        "Shared initialization is listed once, not as independent runs for each method. "
        "Use the checkpoint and outcome labels below, not the legacy renderer's final/failure captions.</p>\n",
    ]
    for condition in sorted(groups, key=lambda key: (not key.startswith("Shared initialization"), key)):
        text.append(f"<details><summary>{html.escape(condition)}</summary>\n")
        for environment, panels in sorted(groups[condition].items()):
            text.append(f"<details><summary>{html.escape(environment)}</summary>\n")
            for success, name in ((True, "Wins"), (False, "Losses")):
                evaluated = sum(len(outcomes[success]) for _, outcomes in panels)
                recorded = sum(bool(links) for _, outcomes in panels for links in outcomes[success])
                text.append(f"<details><summary>{name}: {recorded}/{evaluated} evaluated episodes recorded</summary>\n<ul>")
                for title, outcomes in panels:
                    episodes = outcomes[success]
                    if not episodes:
                        continue
                    links = [link for clips in episodes for link in clips]
                    coverage = f"{sum(bool(clips) for clips in episodes)}/{len(episodes)} recorded"
                    detail = "; ".join(links) if links else "No saved video available."
                    text.append(f"<li>{html.escape(title)}: {coverage}. {detail}</li>")
                if not evaluated:
                    text.append("<li>No evaluated episodes with this outcome.</li>")
                text.append("</ul>\n</details>\n")
            text.append("</details>\n")
        text.append("</details>\n")
    return "\n".join(text)


def recording_outputs(snapshot, notebook_directory):
    import nbformat

    content = recording_index(snapshot, notebook_directory)
    if not content:
        return []
    return [nbformat.v4.new_output("display_data", data={
        "text/html": content,
        "text/plain": "Episode Recordings: expand condition, environment and win/loss groups to view saved video links.",
    })]


def action_notes(snapshot):
    identity = snapshot["action_diagnostics"]["identity"]
    return (
        "#### Turning During The Curriculum\n\n"
        "**Coverage limitation:** these are recovered action histories from the final replay windows, "
        "not the entire experiment. Earlier histories were pruned and cannot be reconstructed. "
        "The shaded range means unavailable, not zero turns; only episodes with every executed "
        "interval recovered enter the averages.\n\n"
        f"A **turn** here is one continuous same-direction steering bout exceeding "
        f"{identity['turn_degrees']:g} degrees while moving faster than "
        f"{identity['moving_mps']:g} m/s in either direction. Repeating the same turn over several "
        "decisions counts once; returning to straight/stopped control or reversing steering ends it. "
        "Camera motion does not count. These are saved policy commands mapped through the episode's "
        "calibration, not measured yaw or successful detours. Commands avoid the simulator's post-stop "
        "neutral readout masking a final turn. Every post-startup policy command must be available; "
        "episodes with gaps are excluded. The second panel shows mean absolute commanded steering "
        "while commanding motion, including turns smaller than the threshold. Episode means are "
        "averaged within replica/bin, then replicas "
        "are weighted equally. Bin coverage varies; these are descriptive, exploration-on training "
        "episodes, not greedy held-out evaluations.\n\n"
        "Difficulty rises through the middle ramp as barrier placement becomes more concentrated "
        "on the direct path. Smaller barrier spread (sigma) means harder layouts, not guaranteed "
        "blockage. The recovered episodes cover only the end of that ramp and the moderate hold, "
        "so they cannot establish the full turning-versus-difficulty trend."
    )


def action_outputs(snapshot):
    analysis = snapshot.get("action_diagnostics")
    if analysis is None:
        return []
    selected = _plot_units(snapshot)
    known = {unit["name"]: unit["summary"] for unit in selected}
    legends = _legend_labels(selected)
    colors = {key: f"C{index}" for index, key in enumerate(sorted(legends))}
    pooled = defaultdict(list)
    coverage = []
    first_observed = []
    for unit in analysis["units"]:
        if unit["name"] not in known:
            continue
        summary = known[unit["name"]]
        total = summary["settings"].get("total", summary.get("n"))
        _require(_number(total) and total > 0, "action plot requires positive acquisition budget")
        episodes = [row for row in unit["episodes"] if row["complete_history"]]
        ranges = lambda key: f"{min(e[key] for e in episodes):.4g}--{max(e[key] for e in episodes):.4g}" if episodes else "Not available"
        coverage.append({"Condition": legends[_group_key(summary)], "Seed": summary["settings"].get("seed"),
                         "Complete histories / training episodes": f"{len(episodes)} / {len(unit['episodes'])}",
                         "Episode-start acquisition range": ranges("n"), "Barrier spread sigma (m)": ranges("sigma_m"),
                         "Commanded turns": sum(e["turn_bouts"] for e in episodes),
                         "Turns / episode": _mean([e["turn_bouts"] for e in episodes])})
        bins = defaultdict(list)
        for row in episodes:
            progress = row["n"] / total
            first_observed.append(progress)
            bins[min(7, int(progress * 8))].append(row)
        for index, rows in sorted(bins.items()):
            pooled[_group_key(summary), index].append({
                "progress": _mean([row["n"] / total for row in rows]),
                "turn_bouts": _mean([row["turn_bouts"] for row in rows]),
                "mean_abs_steering_deg": _mean([row["mean_abs_steering_deg"] for row in rows
                                                if row["mean_abs_steering_deg"] is not None]),
            })
    outputs = []
    if pooled:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10, 4.8))
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(left=.08, right=.98, top=.73, bottom=.15, wspace=.3)
        figure.text(.08, .81, "Initial hold: 0--12.5%  |  Continuous barrier ramp: 12.5--87.5%  |  Moderate hold: 87.5--100%", fontsize=8)
        for axis, key, title, ylabel in zip(axes, ("turn_bouts", "mean_abs_steering_deg"),
                ("Commanded turns per training episode", "Commanded steering while driving"),
                ("Mean steering bouts / episode", "Mean absolute steering (degrees)")):
            for group in sorted({key[0] for key in pooled}):
                means = [(index, {name: _mean([r[name] for r in records if r[name] is not None])
                                  for name in ("progress", key)})
                         for (g, index), records in pooled.items() if g == group]
                means.sort()
                axis.plot([r["progress"] for _, r in means],
                          [r[key] if r[key] is not None else math.nan for _, r in means],
                          marker="o", label=legends[group], color=colors[group])
            axis.axvspan(0, min(first_observed), color=".94", zorder=0)
            axis.text(.27, .5, "Earlier action histories\nnot retained", color=".4", ha="center", va="center", transform=axis.transAxes)
            for boundary in (.125, .875):
                axis.axvline(boundary, color=".7", linestyle=":", linewidth=.8)
            axis.set(title=title, ylabel=ylabel, xlabel="Training progress (incoming intervals / nominal budget)",
                     xlim=(0, 1.02), xticks=(0, .25, .5, .75, 1), xticklabels=("0%", "25%", "50%", "75%", "100%"))
            axis.set_ylim(bottom=0)
        _finish(axes, legend=False)
        _shared_legend(figure)
        outputs.append(_figure(figure, "Recovered training policy commands: mean turn bouts and steering angle while commanding motion in eight progress bins, equal replica means. Only complete late replay-window histories; missing early actions are not zeros. No held-out behavior or full-ramp trend inferred."))
    if coverage:
        outputs.append(_table(coverage, "Recovered Action Coverage (exploration-on training only)", note=(
            "Counts describe only recovered complete episodes. Their outcomes and duration can differ; "
            "more turns do not by themselves establish better navigation. Missing early/partial histories "
            "are excluded, not assigned zero turns.")))
    return outputs


def result_outputs(snapshot, *, section=None):
    _require(section in (None, "prediction", "training", "archive"), "unknown reporting section")
    units = snapshot["units"]
    if not units:
        return []
    from matplotlib.figure import Figure

    prediction = section in (None, "prediction")
    training = section in (None, "training")
    archive = section in (None, "archive")
    outputs = _initialization_outputs(snapshot) if prediction else []
    paired = paired_endpoint_rows(snapshot) if prediction else []
    if paired:
        outputs.append(_table(paired, "Confirmation: Final Paired Endpoint Gaps (left minus TD replay)", note=(
            "Episode means within trajectory, then equal means across exact seed/replica pairs. "
            "Shared initial panels, unmatched identities and ambiguous repeats are omitted; recipes are not pooled. "
            "Individual-gap ranges are not confidence intervals. Own training/evaluation NLL is excluded "
            "from causal cross-method performance because labels differ.")))
    costs = paired_cost_rows(snapshot) if prediction else []
    if costs:
        outputs.append(_table(costs, "Confirmation: Measured Whole-Trajectory Compute Ratios", note=(
            "Left / TD replay active seconds, averaged across pairs with positive durations and equal declared "
            "nominal budgets. These are measured costs, not sample efficiency or guaranteed savings.")))
    features = [u["summary"] for u in units if u["summary"]["operation"] == "feature_health"]
    if prediction and features:
        health = features[-1]
        outputs.append(_table([{"Frozen feature dimension": health.get("feature_dimension"),
                                "Centered variance trace": health.get("centered_variance_trace"),
                                "Leading component variance share": health.get("leading_variance_share"),
                                "Median channel SD": health.get("median_channel_std")}],
                              "Frozen Representation Health (small components may still carry task information)"))
    plot_units = _plot_units(snapshot)
    legends = _legend_labels(plot_units)
    colors = {key: f"C{index}" for index, key in enumerate(sorted(legends))}
    evaluations = evaluation_rows({**snapshot, "units": plot_units})
    if prediction and plot_units:
        ranks = sorted({u["summary"]["settings"].get("ewc_rank", u["summary"]["settings"].get("rank"))
                        for u in plot_units} - {None})
        outputs.append(_table([{"Cohort": "confirmation" if any(_phase(u["summary"]) == "confirmation" for u in plot_units)
                                else "development / unclassified", "Trajectories": len(plot_units),
                                "EWC ranks": ranks,
                                "Recipe groups": len({_group_key(u["summary"]) for u in plot_units}),
                                "Display limit": MAX_PLOT_GROUPS}],
                              "Main Curves: Sealed Cohort (resource pilots remain table-only)", note=(
                                  "Outcome and training-fit curves use this cohort only. Earlier initialization / "
                                  "geometry spectra below are separate diagnostics, not additional fresh trajectories. "
                                  "EWC rank r is not STM LoRA rank.")))
    if prediction and evaluations:
        figure = Figure(figsize=(10, 6))
        axes = figure.subplots(2, 2)
        figure.subplots_adjust(left=.08, right=.98, bottom=.09, top=.77, wspace=.3, hspace=.65)
        for i, environment in enumerate(("initial_A", "moderate_A")):
            for j, (key, title) in enumerate((("raw_return", "Observed discounted return (raw reward units)"),
                                             ("success", "Success fraction"))):
                axis = axes[i, j]
                for group in sorted({r["_group"] for r in evaluations}):
                    rows = [r for r in evaluations if r["_group"] == group and r["Environment"] == environment]
                    axis.plot([r["n"] for r in rows], [r[key] if r[key] is not None else math.nan for r in rows],
                              marker="o", label=legends[group], color=colors[group])
                axis.set_title(environment + ": " + ("Observed raw return" if key == "raw_return" else title), fontsize=9)
                axis.set_xlabel("Nominal acquisition checkpoint")
                if key == "success":
                    axis.set_ylim(-.05, 1.05)
        _finish(axes.flat, legend=False)
        _shared_legend(figure)
        outputs.append(_figure(figure, "Observed raw return and success; equal trajectory means at nominal acquisition checkpoints. Episodes finish naturally; actual acquisition ranges are tabulated. No interpolation or confidence claims."))
    if training and evaluations:
        endpoints = {}
        for row in evaluations:
            endpoints[row["_group"], row["Environment"]] = row
        outputs.append(_table([{
            **{k: r[k] for k in ("Condition", "Environment", "n", "success", "raw_return", "coded_return",
                                 "raw_qwen_mean", "heldout_nll", "critic_mse")},
            "Actual acquisition range": r["Actual acquisition range"],
            "Episodes / trajectories / replicas": f"{r['Episodes']} / {r['Trajectories']} / {_shown(r['Replicas'])}",
            "Return / NLL / MSE coverage": f"{r['raw_return count']} / {r['heldout_nll count']} / {r['critic_mse count']}",
        } for r in endpoints.values()], "Latest Observed Outcomes (raw and encoded units kept separate)"))
    trajectories = [u for u in plot_units if u["summary"].get("metrics")]
    if training and trajectories:
        figure = Figure(figsize=(10, 8))
        axes = figure.subplots(3, 3).ravel()
        figure.subplots_adjust(left=.07, right=.98, bottom=.07, top=.81, wspace=.4, hspace=.7)
        visit_key = "visits" if any("visits" in row for u in trajectories for row in u["summary"]["metrics"]) else "optimizer_exposures"
        for axis, key in zip(axes, ("data_nll", "replay_count", "age_mean", "age_saturated", visit_key,
                                    "update_seconds", "fd", "post_step_ewc", "ewc_proposal_difference")):
            for unit in trajectories:
                summary = unit["summary"]
                rows = sorted(summary["metrics"], key=lambda r: r["n"])
                if any(r.get(key) is not None for r in rows):
                    axis.plot([r["n"] for r in rows], [r.get(key) if r.get(key) is not None else math.nan for r in rows],
                              label=legends[_group_key(summary)], color=colors[_group_key(summary)], alpha=.75)
            axis.set_title("ewc after fit" if key == "post_step_ewc" else key.replace("_", " "), fontsize=9)
            axis.set_xlabel("Acquired observations", fontsize=8)
        _finish(axes, legend=False)
        _shared_legend(figure)
        outputs.append(_figure(figure, "Training data NLL (target/weight dependent), replay, age, visits or optimizer exposures, update seconds, FD at the first step, EWC after the accepted fit against its original anchor, and first-step EWC proposal influence. Held-out NLL comparability requires a common frozen reference."))
        if section is None:
            outputs.append(_fit_diagnostic_glossary(visit_key))
    archives = [u for u in units if u["summary"].get("archive") and not _resource(u["summary"]) and
                (u in plot_units or u["summary"]["operation"] != "trajectory")][:MAX_PLOT_GROUPS]
    if archive and archives:
        figure = Figure(figsize=(10, 4))
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(left=.08, right=.98, bottom=.15, top=.65, wspace=.35)
        archive_labels = _legend_labels(archives)
        for index, unit in enumerate(archives):
            summary = unit["summary"]
            values = sorted(_eigenvalues(summary["archive"]), reverse=True)
            cohort = "Initial/geometry" if summary["operation"] != "trajectory" else _phase(summary)
            label = cohort + ": " + archive_labels[_group_key(summary)]
            if values:
                axes[0].plot(range(1, len(values) + 1), values, marker="o", label=label)
            axes[1].bar(index, summary["active_seconds"], label=label)
        axes[0].set_title("Correlated LL^T spectrum (not full archive)", fontsize=9)
        axes[0].set_xlabel("Eigenvalue index")
        axes[1].set_title("Whole-unit active cost (not compression-only)", fontsize=9)
        axes[1].set_ylabel("Seconds")
        axes[1].set_xticks(range(len(archives)), [str(i + 1) for i in range(len(archives))])
        axes[1].set_xlabel("Archive snapshot")
        _finish(axes, legend=False)
        _shared_legend(figure)
        outputs.append(_figure(figure, "EWC correlated-factor spectrum and measured whole-unit cost; no dense Fisher reconstruction."))
        outputs.append(_table([{"Unit": u["name"], "Condition": _label(u["summary"]),
                                "Cohort": "initialization / geometry" if u["summary"]["operation"] != "trajectory"
                                else _phase(u["summary"]),
                                "EWC rank": u["summary"]["archive"].get("rank"),
                                "STM LoRA rank (unchanged)": u["summary"]["settings"].get("stm_rank"),
                                "Trace": u["summary"]["archive"].get("trace"),
                                "Correlated trace fraction": u["summary"]["archive"].get("correlated_trace_fraction"),
                                "Active s": u["summary"]["active_seconds"]} for u in archives], "EWC Rank, Trace And Cost", note=(
                                    "Includes earlier initialization / geometry snapshots as well as the main cohort. "
                                    "An initialization spectrum does not imply fresh outcome curves at that rank exist. "
                                    "Rank zero has no correlated factor to plot; diagonal EWC remains present. "
                                    "Whole-unit costs cover different operations and are not matched method timing comparisons.")))
    def endpoint(summary, environment):
        panels = [p for p in summary.get("evaluations", []) if p["environment"] == environment and p["episodes"]]
        if not panels:
            return None
        panel = max(panels, key=lambda p: p["n"])
        return {"n": panel["n"], "raw_return": _mean([e["raw_return"] for e in panel["episodes"] if e.get("raw_return") is not None]),
                "success": _mean([float(e["success"]) for e in panel["episodes"]])}

    if archive:
        outputs.append(_table([{"Unit": u["name"], "Operation": u["summary"]["operation"],
                            "Condition / recipe": _label(u["summary"]), "Settings": u["summary"]["settings"],
                            "Initial A endpoint": endpoint(u["summary"], "initial_A"),
                            "Moderate A endpoint": endpoint(u["summary"], "moderate_A"),
                            "Active s": u["summary"]["active_seconds"],
                            "Selected / forecast": u["summary"].get("selected", u["summary"].get("forecast"))}
                           for u in units], "Executed Settings And Sealed Unit Costs"))
    return outputs


def conclusion(snapshot):
    state = "Approved scope complete" if snapshot["complete"] else "Provisional / incomplete"
    evidence = ("No completed units yet." if not snapshot["units"] else
                "Observed outcomes and measured costs above are descriptive; missing panels are not failures.")
    if snapshot["complete"]:
        paired = paired_endpoint_rows(snapshot)
        if paired:
            comparisons = defaultdict(list)
            for row in paired:
                unit = "raw return" if row["Metric"] == "Raw observed return" else "pp success"
                comparisons[row["Comparison"]].append(
                    f"{row['Endpoint']} {row['Mean gap']:+.4g} {unit} ({row['Pairs']} pairs)")
            evidence = "Completed confirmation endpoint gaps (left minus TD replay): " + "; ".join(
                name + ": " + ", ".join(values) for name, values in comparisons.items()) + "."
        else:
            evidence += " No unambiguous matched-budget confirmation endpoint pairs are available."
        costs = paired_cost_rows(snapshot)
        if costs:
            evidence += " Measured mean active-time ratios (left / TD replay): " + "; ".join(
                f"{r['Comparison']} {r['Mean active-time ratio (left/TD)']:.4g} ({r['Pairs']} pairs)"
                for r in costs) + "."
    elif snapshot["units"]:
        evidence += " Final paired confirmation comparisons await completion of the approved scope."
    if rank_development_rows(snapshot):
        evidence += (" Fresh assessment covers only the selected EWC rank. The separate development rank "
                     "comparison above does not settle the value of correlated EWC or its interaction with FD.")
    return (f"#### Fix 5 Conclusion\n\n**{state}.** {evidence} "
            "Own training/evaluation NLL is not causal cross-method performance: labels differ. "
            "No acquired-sample equivalence is estimated without an observed common capability crossing and "
            "retention qualification. Two paired replicas do not establish significance or broad efficacy; "
            "individual-gap ranges are not confidence intervals, and measured ratios do not guarantee savings. "
            "Development is not confirmation. B and physical learning remain outside this A-only scope.")


def display_progress(snapshot):
    from IPython.display import display

    for output in progress_outputs(snapshot):
        display(output["data"], raw=True)


def display_results(snapshot, *, section=None):
    from IPython.display import display

    for output in result_outputs(snapshot, section=section):
        display(output["data"], raw=True)


def display_rank_development(snapshot):
    from IPython.display import display

    for output in rank_outputs(snapshot):
        display(output["data"], raw=True)


def display_recordings(snapshot, notebook_directory):
    from IPython.display import display

    for output in recording_outputs(snapshot, notebook_directory):
        display(output["data"], raw=True)


def display_actions(snapshot):
    from IPython.display import display

    for output in action_outputs(snapshot):
        display(output["data"], raw=True)


def build_notebook(root, output, base_notebook=None, *, action_report=None):
    """Atomically refresh only own tagged cells; preserve all history verbatim.

    Base precedence: explicit base, existing output, root/history.ipynb, canonical
    Plan 19 notebook. If none exists, create a minimal report. Reusing an output
    never reruns historical builders or normalizes their cells/outputs/metadata.
    """
    import nbformat

    snapshot = load_report(root, action_report=action_report)
    output, root = Path(output), Path(root)
    _require(not output.is_symlink(), "output must not be a symlink")
    _require(not any((p / "COMPLETED").exists() for p in output.resolve().parents),
             "output must not be inside a completed run")
    candidates = [output, root / "history.ipynb", shared.DEFAULT_NOTEBOOK]
    base = Path(base_notebook) if base_notebook is not None else next((p for p in candidates if p.exists()), None)
    if base is None:
        book = dict(nbformat.v4.new_notebook())
    else:
        book = shared._json(base, limit=MAX_HISTORY_BYTES)
        _require(book.get("nbformat") == 4 and isinstance(book.get("cells"), list), "base must be a v4 notebook")
    book = copy.deepcopy(book)
    history = [c for c in book["cells"] if not c.get("metadata", {}).get(_TAG)]
    identifiers = [c.get("id") for c in history if c.get("id") is not None]
    _require(len(identifiers) == len(set(identifiers)), "historical cell ids must be unique")
    used = set(identifiers)

    def own(cell, name):
        stem = "plan19-phase3-fix5-" + name
        identifier, suffix = stem, 0
        while identifier in used:
            suffix += 1
            identifier = f"{stem}-{suffix}"
        used.add(identifier)
        cell["id"] = identifier
        cell["metadata"][_TAG] = True
        return cell

    project = Path(__file__).resolve().parents[2]
    try:
        relative_root = root.resolve().relative_to(project)
    except ValueError:
        # Explicit external-run overrides remain usable for local reports.
        root_expression = repr(snapshot["root"])
    else:
        root_expression = f"project_root / {str(relative_root)!r}"
    try:
        notebook_directory = f"project_root / {str(output.parent.resolve().relative_to(project))!r}"
    except ValueError:
        notebook_directory = repr(str(output.parent.resolve()))
    action_argument = ""
    if action_report is not None:
        try:
            relative_action = Path(action_report).resolve().relative_to(project)
        except ValueError:
            expression = repr(str(Path(action_report).resolve()))
        else:
            expression = f"project_root / {str(relative_action)!r}"
        action_argument = f", action_report={expression}"
    bootstrap = (
        "from pathlib import Path\nimport sys\n"
        "for parent in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (parent / 'numerical_experiment/picar_head_awac/report.py').is_file():\n"
        "        project_root = parent\n"
        "        if str(parent) not in sys.path:\n            sys.path.insert(0, str(parent))\n"
        "        break\n"
        "else:\n    raise RuntimeError('Open this notebook from within the project checkout.')\n"
        "from numerical_experiment.picar_head_awac import report as head_awac_report\n"
        f"head_awac_snapshot = head_awac_report.load_report({root_expression}{action_argument})\n"
    )
    progress = own(nbformat.v4.new_code_cell(bootstrap + "head_awac_report.display_progress(head_awac_snapshot)",
                                           outputs=progress_outputs(snapshot)), "progress")
    results = [own(nbformat.v4.new_markdown_cell(_INTRO), "intro")]
    rank_results = rank_outputs(snapshot)
    if rank_results:
        results.extend([
            own(nbformat.v4.new_markdown_cell(rank_selection_notes(snapshot)), "rank-intro"),
            own(nbformat.v4.new_code_cell(
                bootstrap + "head_awac_report.display_rank_development(head_awac_snapshot)",
                outputs=rank_results), "rank-development"),
        ])
    for section, name in (("prediction", "results"), ("training", "training"), ("archive", "archive")):
        outputs = result_outputs(snapshot, section=section)
        if section != "prediction" and not outputs:
            continue
        if section == "archive":
            results.append(own(nbformat.v4.new_markdown_cell(
                "#### Archive Diagnostics: Initialization And Fresh Assessment\n\n"
                "Unlike the outcome curves, this section also includes earlier initialization / geometry "
                "snapshots. Their spectra are not fresh assessment results at those ranks. "
                "The cohort is labeled in the legend and table; EWC rank is distinct from STM LoRA rank."
            ), "archive-intro"))
        results.append(own(nbformat.v4.new_code_cell(
            bootstrap + f"head_awac_report.display_results(head_awac_snapshot, section={section!r})",
            outputs=outputs), name))
        if section == "prediction":
            recordings = recording_outputs(snapshot, output.parent)
            if recordings:
                results.append(own(nbformat.v4.new_code_cell(
                    bootstrap + f"head_awac_report.display_recordings(head_awac_snapshot, {notebook_directory})",
                    outputs=recordings, metadata={"jupyter": {"source_hidden": True}}), "recordings"))
            actions = action_outputs(snapshot)
            if actions:
                results.append(own(nbformat.v4.new_markdown_cell(action_notes(snapshot)), "actions-intro"))
                results.append(own(nbformat.v4.new_code_cell(
                    bootstrap + "head_awac_report.display_actions(head_awac_snapshot)",
                    outputs=actions), "actions"))
        elif section == "training":
            glossary = fit_diagnostic_glossary(snapshot)
            if glossary:
                results.append(own(nbformat.v4.new_markdown_cell(glossary), "training-glossary"))
    results.append(own(nbformat.v4.new_markdown_cell(conclusion(snapshot)), "conclusion"))
    # Current counters precede history, even if the historical first cell is code.
    position = 1 if history and history[0].get("cell_type") == "markdown" else 0
    book["cells"] = history[:position] + [progress] + history[position:] + results
    try:
        nbformat.validate(copy.deepcopy(book))
    except nbformat.ValidationError as exc:
        raise ArtifactError("Fix 5 schema-incompatible notebook: " + str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".head-awac-report-", suffix=".ipynb", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(book, handle, ensure_ascii=True, allow_nan=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
