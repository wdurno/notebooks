"""Bounded JSON-only reporting for the exploratory Plan 19 Fix 5A extension.

Only progress, small configurations, sealed scalar summaries and notebooks are
read. Importing this module never imports the experiment runtime or Torch.
"""

from __future__ import annotations

from collections import defaultdict
import base64
import copy
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from . import VERSION


SOURCE_VERSION = "plan19-phase3-fix5-v1"
_TAG = "plan19_phase3_fix5a"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 160 * 1024 * 1024
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_UNITS = 128
MAX_SCAN_DIRS = 2048
MAX_ROWS = 100_000
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "picar/plan19_results.ipynb"
OPERATIONS = {"initial_archives", "resource_audit", "rank_admission", "evaluation_expansion",
              "trajectory", "correction_decision", "selection"}
METHODS = ("td_replay", "td_fd")
ENVIRONMENTS = ("initial_A", "moderate_A")


class ArtifactError(ValueError):
    """A scalar artifact is incompatible, malformed, or outside reporting bounds."""


def _require(ok, message):
    if not ok:
        raise ArtifactError("Fix 5A: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _read(path, limit):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), f"missing/nonregular artifact: {path}")
    _require(path.stat().st_size <= limit, f"artifact exceeds size bound: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    _require(len(data) <= limit, f"artifact exceeds size bound: {path}")
    return data


def _json(path, limit=MAX_JSON_BYTES, *, data=None):
    def invalid(value):
        raise ArtifactError(f"Fix 5A: nonfinite JSON value in {path}: {value}")

    def finite(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)

    try:
        result = json.loads(_read(path, limit) if data is None else data,
                            parse_constant=invalid, parse_float=finite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ArtifactError(f"Fix 5A: invalid JSON: {path}: {exc}") from exc
    _require(isinstance(result, dict), f"JSON must contain an object: {path}")
    return result


def _rows(value, context):
    _require(isinstance(value, list) and len(value) <= MAX_ROWS
             and all(isinstance(row, dict) for row in value), context + " must be a bounded object list")
    return value


def _optional_numbers(row, keys, context):
    for key in keys:
        _require(row.get(key) is None or _number(row[key]), context + "." + key + " must be finite or null")


def _archive(value, context):
    _require(isinstance(value, dict), context + " must be an object")
    for key in ("rank", "realized_rank", "factor_columns"):
        _require(value.get(key) is None or _count(value[key]) and value[key] <= 64,
                 context + "." + key + " must be an integer <=64")
    for key in ("eigenvalues", "factor_eigenvalues"):
        if key in value:
            values = value[key]
            _require(isinstance(values, list) and len(values) <= 64 and all(_number(x) for x in values),
                     context + "." + key + " must contain at most 64 finite LL^T eigenvalues")
    _optional_numbers(value, ("trace", "correlated_trace_fraction", "minimum_residual",
                              "residual_trace", "storage_bytes", "trace_error"), context)
    if value.get("allocation") is not None:
        _allocation(value["allocation"])


def _allocation(value):
    _require(isinstance(value, dict), "allocation must be an object")
    for block in ("actor", "critic"):
        if block not in value:
            continue
        row = value[block]
        _require(isinstance(row, dict), "allocation block must be an object")
        _optional_numbers(row, ("trace", "correlated_trace", "residual_trace", "correlated_trace_fraction"),
                          "allocation." + block)
        energy = row.get("factor_column_energy", [])
        _require(isinstance(energy, list) and len(energy) <= 64 and all(_number(x) and x >= 0 for x in energy),
                 "allocation factor_column_energy must contain <=64 nonnegative values")


def _final_state(value):
    remaining = 4096

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        _require(remaining >= 0 and depth <= 8, "final state exceeds scalar bounds")
        if isinstance(item, dict):
            _require(len(item) <= 128, "final state object exceeds scalar bounds")
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            _require(len(item) <= 512, "final state array exceeds scalar bounds")
            for child in item:
                visit(child, depth + 1)
        else:
            _require(item is None or type(item) is bool or _number(item)
                     or isinstance(item, str) and len(item) <= 4096, "invalid scalar final state")

    visit(value, 0)


def _summary(value, *, source=False):
    expected = SOURCE_VERSION if source else VERSION
    _require(value.get("version") == expected, "summary version must be " + expected)
    _require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
             "summary schema_version must be 1")
    operation = value.get("operation")
    _require(operation == "trajectory" if source else operation in OPERATIONS, "unsupported operation")
    _require(isinstance(value.get("settings"), dict), "summary.settings must be an object")
    if "kind" in value:
        _require(value["kind"] == ("head_awac_" if source else "head_rank_") + operation,
                 "summary kind disagrees with operation")
    _optional_numbers(value, ("active_seconds", "n"), "summary")
    if operation in ("trajectory", "evaluation_expansion"):
        settings = value["settings"]
        _require(settings.get("method") in METHODS and _count(settings.get("seed")), "trajectory method/seed required")
        _require(settings.get("phase") in ("development", "retention_calibration", "confirmation", "development_expansion"),
                 "unknown trajectory phase")
        _require(_count(settings.get("ewc_rank")) and settings["ewc_rank"] <= 64, "settings.ewc_rank must be <=64")
        _require(_count(settings.get("total")) and settings["total"] > 0, "settings.total must be positive")
        _require(isinstance(settings.get("recipe", {}), dict), "recipe must be an object")
    if value.get("allocation") is not None:
        _allocation(value["allocation"])
    if value.get("archive") is not None:
        _archive(value["archive"], "archive")
    for row in _rows(value.get("archive_metrics", []), "archive_metrics"):
        _archive(row, "archive_metrics")
    for row in _rows(value.get("metrics", []), "metrics"):
        _require(_number(row.get("n")) and row["n"] >= 0, "metric.n must be nonnegative")
        _optional_numbers(row, ("data_nll", "fd", "ewc", "post_step_ewc", "post_step_fd",
                                "ewc_proposal_difference", "update_seconds"), "metric")
    panel_ids = set()
    for panel in _rows(value.get("evaluations", []), "evaluations"):
        _require(_count(panel.get("n")) and panel.get("environment") in ENVIRONMENTS,
                 "evaluation requires nonnegative integer n and a known environment")
        identity = panel["n"], panel["environment"]
        _require(identity not in panel_ids, "duplicate evaluation panel")
        panel_ids.add(identity)
        _require(panel.get("acquired_count") is None or _count(panel["acquired_count"]),
                 "evaluation acquired_count must be nonnegative or null")
        for episode in _rows(panel.get("episodes"), "evaluation.episodes"):
            _require(type(episode.get("success")) is bool, "episode.success must be Boolean")
            _optional_numbers(episode, ("raw_return", "coded_return", "heldout_nll", "critic_mse"), "episode")
            for video in _rows(episode.get("videos", []), "videos"):
                _require(isinstance(video.get("path"), str) and bool(video["path"]), "video path required")
            for key in ("final_state", "final_states"):
                if key in episode:
                    _final_state(episode[key])
    for episode in _rows(value.get("episodes", []), "training episodes"):
        for key in ("final_state", "final_states"):
            if key in episode:
                _final_state(episode[key])
    for key in ("final_state", "final_states"):
        if key in value:
            _final_state(value[key])
    for row in _rows(value.get("action_metrics", []), "action_metrics"):
        _optional_numbers(row, ("n", "sigma_m", "turn_bouts", "mean_abs_steering_deg", "moving_commands"),
                          "action_metrics")
        _require(type(row.get("complete_history")) is bool, "action_metrics complete_history must be Boolean")
        if row["complete_history"]:
            _require(_count(row.get("turn_bouts")), "complete action history requires turn_bouts")
    diagnostics = value.get("diagnostics", {})
    _require(isinstance(diagnostics, dict), "diagnostics must be an object")
    ranks = diagnostics.get("ranks", diagnostics if operation == "initial_archives" else {})
    _require(isinstance(ranks, dict), "diagnostic ranks must be an object")
    for rank, row in ranks.items():
        if str(rank).isdigit():
            _require(int(rank) <= 64, "diagnostic rank must be <=64")
            _archive(row, "diagnostics.ranks." + str(rank))
    if operation == "resource_audit":
        result = value.get("result", {})
        _require(isinstance(result, dict), "resource_audit.result must be an object")
        for row in _rows(result.get("timings", []), "resource timings"):
            if "archive" in row:
                _archive(row["archive"], "resource archive")
            if "allocation" in row:
                _allocation(row["allocation"])
            _optional_numbers(row, ("fit_seconds", "retirement_seconds", "compression_seconds",
                                    "factor_generation_seconds", "solve_seconds"), "resource timings")


def _children(root):
    if not root.exists():
        return []
    _require(not root.is_symlink(), "units directory must not be a symlink")
    children = []
    for child in root.iterdir():
        children.append(child)
        _require(len(children) <= MAX_SCAN_DIRS, "unit scan bound exceeded")
    return sorted(children)


def load_report(root):
    """Load sealed JSON only; incomplete units remain explicitly pending."""
    root = Path(root)
    total = 0

    def read(path, limit=MAX_JSON_BYTES, receipt=None):
        nonlocal total
        data = _read(path, limit)
        total += len(data)
        _require(total <= MAX_TOTAL_BYTES, "total scalar JSON bound exceeded")
        if receipt and receipt.get("summary_sha256"):
            _require(hashlib.sha256(data).hexdigest() == receipt["summary_sha256"],
                     "source summary checksum mismatch: " + str(path))
        return _json(path, limit, data=data)

    progress = read(root / "progress.json", MAX_CONFIG_BYTES) if (root / "progress.json").exists() else {}
    if progress:
        _require(progress.get("version") == VERSION, "progress version mismatch")
        _require(all(isinstance(progress.get(k), str) for k in ("status", "stage")), "progress status/stage required")
        _require(all(_count(progress.get(k)) for k in ("completed_units", "planned_units")), "progress counts invalid")
        _require(_number(progress.get("charged_seconds")) and progress["charged_seconds"] >= 0,
                 "progress charged_seconds invalid")
        _require(isinstance(progress.get("active_progress", {}), dict), "active_progress must be an object")
    config_path = root / "orchestration/config.json"
    config = read(config_path, MAX_CONFIG_BYTES) if config_path.exists() else {}
    units, controls = [], []
    for child in _children(root / "units"):
        if not child.is_dir() or not (child / "COMPLETED").is_file():
            continue
        _require(not child.is_symlink() and not (child / "COMPLETED").is_symlink(), "symlinked sealed unit")
        _require(len(units) < MAX_UNITS, "unit limit exceeded")
        value = read(child / "summary.json")
        _summary(value)
        units.append({"name": child.name, "path": str(child.resolve()), "summary": value, "reused": False})
    source = config.get("source")
    if source is not None:
        _require(isinstance(source, str) and bool(source), "source must be a path string")
        source = Path(source)
        if not source.is_absolute():
            source = root / source
        receipts = config.get("source_receipts", [])
        _rows(receipts, "source_receipts")
        receipts = {str(Path(r["unit"]).resolve()): r for r in receipts if isinstance(r.get("unit"), str)}
        for child in _children(source / "units"):
            if not child.is_dir() or not (child / "COMPLETED").is_file():
                continue
            _require(not child.is_symlink() and not (child / "COMPLETED").is_symlink(), "symlinked source unit")
            # Configurations are small; do not open MC, pilot or rank-survey summaries.
            cfg = read(child / "config.json", MAX_CONFIG_BYTES)
            settings = cfg.get("settings", {})
            if not (cfg.get("operation") == "trajectory" and settings.get("method") in METHODS
                    and settings.get("ewc_rank") == 0 and settings.get("phase") in ("development", "confirmation")):
                continue
            _require(len(controls) < MAX_UNITS, "source control limit exceeded")
            value = read(child / "summary.json", receipt=receipts.get(str(child.resolve())))
            _summary(value, source=True)
            _require(value["settings"] == settings, "source summary/config settings differ")
            controls.append({"name": child.name, "path": str(child.resolve()), "summary": value, "reused": True})
    active_settings = {}
    if progress.get("active_unit"):
        active_path = Path(progress["active_unit"]).resolve()
        _require(active_path.parent == (root / "units").resolve(), "active unit is outside extension units")
        active_config = read(active_path / "config.json", MAX_CONFIG_BYTES)
        _require(active_config.get("version") == VERSION, "active configuration version mismatch")
        active_settings = active_config.get("settings", {})
        _require(isinstance(active_settings, dict), "active settings must be an object")
    complete = (progress.get("status") in ("complete", "completed")
                and progress.get("completed_units") == progress.get("planned_units") == len(units))
    return {"root": str(root.resolve()), "source": str(source.resolve()) if source else None,
            "progress": progress, "config": config, "units": units, "controls": controls,
            "active_settings": active_settings, "complete": complete}


def _shown(value):
    if value is None:
        return "Pending / unavailable"
    if isinstance(value, float):
        return f"{value:.5g}"
    return json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else str(value)


def _table(rows, title, note=""):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    plain = title + "\n" + ("\n".join(str({k: _shown(v) for k, v in row.items()}) for row in rows)
                              if rows else "Pending / unavailable") + ("\n" + note if note else "")
    content = "<h4>" + html.escape(title) + "</h4>"
    if rows:
        content += '<div style="overflow-x:auto"><table><thead><tr>' + "".join(
            "<th>" + html.escape(k) + "</th>" for k in columns) + "</tr></thead><tbody>"
        content += "".join("<tr>" + "".join("<td>" + html.escape(_shown(row.get(k))) + "</td>"
                                           for k in columns) + "</tr>" for row in rows) + "</tbody></table></div>"
    else:
        content += "<p>Pending / unavailable</p>"
    content += "<p>" + html.escape(note) + "</p>" if note else ""
    return {"output_type": "display_data", "data": {"text/plain": plain, "text/html": content}, "metadata": {}}


def remaining_forecast(snapshot):
    """Coarse operational range, updated after a completed new learning path."""
    if snapshot["complete"]:
        return [0., 0.]
    measured = [u["summary"] for u in snapshot["units"]
                if u["summary"]["operation"] == "trajectory" and _number(u["summary"].get("active_seconds"))
                and u["summary"]["active_seconds"] > 0]
    if not measured:
        return None
    planned = {(phase, seed, method) for phase, seeds in
               (("development", (199301,)), ("confirmation", (199101, 199102)))
               for seed in seeds for method in METHODS}
    finished = {(s["settings"]["phase"], s["settings"]["seed"], s["settings"]["method"]) for s in measured}
    remaining = planned - finished
    durations = {}
    for method in METHODS:
        costs = [s["active_seconds"] / 3600 for s in measured if s["settings"]["method"] == method]
        if not costs:
            costs = [u["summary"]["active_seconds"] / 3600 for u in snapshot["controls"]
                     if u["summary"]["settings"]["method"] == method
                     and u["summary"]["settings"]["phase"] == "confirmation"
                     and u["summary"].get("active_seconds", 0) > 0]
        durations[method] = sum(costs) / len(costs) if costs else 1.7
    hours = sum(durations[method] for _, _, method in remaining)
    settings = snapshot.get("active_settings", {})
    identity = settings.get("phase"), settings.get("seed"), settings.get("method")
    active = snapshot["progress"].get("active_progress", {})
    if identity in remaining and _number(active.get("n")):
        total = settings.get("total", 1024)
        _require(_count(total) and total > 0, "active trajectory budget must be positive")
        fraction = min(1., max(0., active["n"] / total))
        hours -= .8 * fraction * durations[settings["method"]]
    optional = 0. if _selection(snapshot).get("selected") else 1.8
    return [max(0., .85 * hours) + .15, max(0., 1.2 * hours) + optional + .25]


def progress_outputs(snapshot):
    p = snapshot["progress"]
    active = p.get("active_progress", {})
    estimate = remaining_forecast(snapshot)
    outputs = [_table([{"Status": p.get("status", "Not started / incomplete"), "Stage": p.get("stage"),
                     "New units": f"{p.get('completed_units', 0)} / {p.get('planned_units', '?')}",
                     "Acquisitions (unsealed preview)": active.get("n", active.get("acquisitions")),
                     "Charged incremental h": p["charged_seconds"] / 3600 if "charged_seconds" in p else None,
                     "Remaining forecast": f"{estimate[0]:.1f}--{estimate[1]:.1f} h" if estimate is not None else p.get("forecast"),
                     "Active unit": p.get("active_unit")}],
                   "Fix 5A Current Progress", "Live counters are unsealed previews, not completed outcome evidence. "
                   "After the first completed learning path, remaining hours use measured path costs and acquisition "
                   "progress, reserving evaluation/closeout and one possible calibration correction. This is a planning range, not a confidence interval.")]
    preview = active.get("evaluation_preview", [])
    if preview:
        _rows(preview, "live evaluation preview")
        outputs.append(_table([{key: row.get(key) for key in ("n", "environment", "episodes", "success", "raw_return")}
                               for row in preview], "Live Held-Out Preview (Unsealed)",
                              "Episodes and panels may be incomplete. These observations are excluded from sealed comparison tables."))
    return outputs


def _mean(values):
    return math.fsum(values) / len(values) if values and all(_number(v) for v in values) else None


def _range(values):
    return [min(values), max(values)] if values and all(_number(v) for v in values) else None


def _trajectories(snapshot, phase):
    return [u for u in snapshot["controls"] + snapshot["units"]
            if u["summary"]["operation"] == "trajectory" and u["summary"]["settings"].get("phase") == phase]


def _selection(snapshot):
    rows = [u["summary"] for u in snapshot["units"] if u["summary"]["operation"] == "selection"]
    _require(len(rows) <= 1, "ambiguous sealed selection")
    return rows[0] if rows else {}


def _admission(snapshot):
    rows = [u["summary"] for u in snapshot["units"] if u["summary"]["operation"] == "rank_admission"]
    _require(len(rows) <= 1, "ambiguous rank admission")
    return rows[0] if rows else {}


def _condition(unit):
    settings = unit["summary"]["settings"]
    return ("F" if settings["method"] == "td_fd" else "T") + ("0" if settings["ewc_rank"] == 0 else "+")


def _recipe(settings):
    recipe = settings.get("recipe", {})
    return {k: recipe[k] for k in ("ridge", "fd_strength", "capacity", "batch_size", "rounds", "steps") if k in recipe}


def _panel(unit, environment, n=None):
    summary = unit["summary"]
    n = summary["settings"].get("total", 1024) if n is None else n
    rows = [p for p in summary.get("evaluations", []) if p["environment"] == environment and p["n"] == n]
    return rows[0] if len(rows) == 1 else None


def _outcome(panel, metric, count):
    if panel is None or len(panel["episodes"]) < count:
        return None
    return _mean([float(e["success"]) if metric == "success" else e.get(metric)
                  for e in panel["episodes"][:count]])


def settings_rows(snapshot):
    chosen, rank = _selection(snapshot).get("selected", {}), _admission(snapshot).get("rank")
    rows = []
    for condition, method, reused in (("T0", "td_replay", True), ("F0", "td_fd", True),
                                       ("T+", "td_replay", False), ("F+", "td_fd", False)):
        candidates = [u for u in _trajectories(snapshot, "confirmation") if _condition(u) == condition]
        recipe = (candidates[0]["summary"]["settings"].get("recipe") if candidates else
                  chosen.get(method, {}).get("recipe") if not reused else None)
        rows.append({"Condition": condition, "Learner": "TD + FD" if method == "td_fd" else "TD replay",
                     "Evidence": "reused Fix 5" if reused else "new Fix 5A",
                     "EWC rank": 0 if reused else rank, "STM rank": 1,
                     "Assessment seeds": [199101, 199102], "Sealed paths": len(candidates),
                     "Recipe": _recipe({"recipe": recipe}) if recipe else None})
    return rows


def development_rows(snapshot):
    """Eight-layout endpoints only; never compare two old layouts to eight new."""
    units = list(_trajectories(snapshot, "development")) + _trajectories(snapshot, "retention_calibration")
    expanded = [u for u in snapshot["units"] if u["summary"]["operation"] == "evaluation_expansion"]
    by_source = defaultdict(list)
    for unit in expanded:
        summary = unit["summary"]
        by_source[summary.get("source_unit", summary["settings"].get("source_unit"))].append(unit)
    replaced = []
    for unit in units:
        additions = by_source.get(unit["path"], []) if unit["reused"] else []
        _require(len(additions) <= 1, "ambiguous development expansion")
        replaced.append(additions[0] if additions else unit)
    # A sealed expansion can be rendered even when its original summary is unavailable.
    units = replaced + [u for u in expanded if u not in replaced]
    reference = [_outcome(_panel(u, "initial_A"), "raw_return", 8) for u in units if _condition(u) == "T0"]
    reference = reference[0] if len(reference) == 1 else None
    chosen = _selection(snapshot).get("selected", {})
    rows = []
    for unit in units:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        initial, moderate = _panel(unit, "initial_A"), _panel(unit, "moderate_A")
        retention = _outcome(initial, "raw_return", 8)
        gap = retention - reference if retention is not None and reference is not None else None
        selected_path = chosen.get(settings["method"], {}).get("development_unit")
        rows.append({"Condition": _condition(unit), "Phase": settings["phase"], "Unit": unit["name"],
                     "Recipe": _recipe(settings), "Selected": unit["path"] == selected_path,
                     "Initial A return (8)": retention,
                     "Initial A successes / 8": sum(e["success"] for e in initial["episodes"][:8])
                     if initial and len(initial["episodes"]) >= 8 else None,
                     "Retention gap vs expanded T0": gap, "Within +/-5": abs(gap) <= 5 if gap is not None else None,
                     "Moderate A return (8)": _outcome(moderate, "raw_return", 8),
                     "Moderate A successes / 8": sum(e["success"] for e in moderate["episodes"][:8])
                     if moderate and len(moderate["episodes"]) >= 8 else None,
                     "Actual acquisitions": initial.get("acquired_count") if initial else None})
    return rows


def evaluation_rows(snapshot, phase="confirmation"):
    """Equal trajectory means on the common first four (development: two) layouts."""
    count = 4 if phase == "confirmation" else 2
    groups = defaultdict(list)
    for unit in _trajectories(snapshot, phase):
        settings = unit["summary"]["settings"]
        recipe = json.dumps(settings.get("recipe", {}), sort_keys=True)
        for panel in unit["summary"].get("evaluations", []):
            groups[(_condition(unit), recipe, panel["environment"], panel["n"])].append((unit, panel))
    rows = []
    for (condition, recipe, environment, n), pairs in sorted(groups.items()):
        seeds = [u["summary"]["settings"]["seed"] for u, _ in pairs]
        _require(len(set(seeds)) == len(seeds), "duplicate seed within trajectory recipe")
        raw = [_outcome(panel, "raw_return", count) for _, panel in pairs]
        success = [_outcome(panel, "success", count) for _, panel in pairs]
        rows.append({"Condition": condition, "Recipe": recipe, "Phase": phase, "Environment": environment,
                     "n": n, "raw_return": _mean(raw), "success": _mean(success),
                     "Return seed range": _range(raw), "Success seed range": _range(success),
                     "Trajectories": len(pairs), "Layouts per trajectory": count,
                     "Actual acquisitions": [p.get("acquired_count") for _, p in pairs]})
    return rows


_CONTRASTS = {
    "Rank, TD: T+ - T0": {"T+": 1, "T0": -1},
    "Rank, FD: F+ - F0": {"F+": 1, "F0": -1},
    "FD, diagonal: F0 - T0": {"F0": 1, "T0": -1},
    "FD, correlated: F+ - T+": {"F+": 1, "T+": -1},
    "Interaction: (F+ - T+) - (F0 - T0)": {"F+": 1, "T+": -1, "F0": -1, "T0": 1},
}


def paired_endpoint_rows(snapshot):
    """Within-seed final full-eight contrasts; never substitute another budget."""
    indexed = defaultdict(list)
    recipes = defaultdict(set)
    for unit in _trajectories(snapshot, "confirmation"):
        settings = unit["summary"]["settings"]
        indexed[settings["seed"], _condition(unit)].append(unit)
        recipes[_condition(unit)].add(json.dumps(settings.get("recipe", {}), sort_keys=True))
    seeds = snapshot["config"].get("protocol", {}).get("assessment_seeds", [199101, 199102])
    rows = []
    for environment in ENVIRONMENTS:
        for metric in ("raw_return", "success"):
            for contrast, weights in _CONTRASTS.items():
                effects, counts, fitting, whole = {}, {}, {}, {}
                for seed in seeds:
                    units = {c: indexed[seed, c][0] for c in weights if len(indexed[seed, c]) == 1
                             and len(recipes[c]) == 1}
                    effect = None
                    if len(units) == len(weights):
                        budgets = {u["summary"]["settings"].get("total") for u in units.values()}
                        panels = {c: _panel(u, environment, n=u["summary"]["settings"]["total"])
                                  for c, u in units.items()}
                        values = {c: _outcome(p, metric, 8) for c, p in panels.items()}
                        layouts = [tuple(e.get("seed") for e in p["episodes"][:8])
                                   for p in panels.values() if p is not None and len(p["episodes"]) >= 8]
                        if budgets == {1024} and all(v is not None for v in values.values()) and len(set(layouts)) == 1:
                            effect = math.fsum(weights[c] * values[c] for c in weights) * (100 if metric == "success" else 1)
                        counts[str(seed)] = {c: p.get("acquired_count") if p else None for c, p in panels.items()}
                        fitting[str(seed)] = {c: u["summary"].get("timings", {}).get("fitting") for c, u in units.items()}
                        whole[str(seed)] = {c: u["summary"].get("active_seconds") for c, u in units.items()}
                    effects[str(seed)] = effect
                available = [v for v in effects.values() if v is not None]
                rows.append({"Environment": environment, "Metric": "Success (pp)" if metric == "success" else "Raw return",
                             "Contrast": contrast, **effects, "Mean gap": _mean(available),
                             "Seed range (not CI)": _range(available), "Pairs": len(available),
                             "Status": "descriptive" if len(available) == len(seeds) else "Pending pairs",
                             "Actual acquisitions": counts or None, "Fitting s": fitting or None,
                             "Whole-path s": whole or None})
    return rows


def _figure(figure, description):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buffer = io.BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": description}}


def _curves(snapshot, phase):
    from matplotlib.figure import Figure

    rows = evaluation_rows(snapshot, phase)
    if not rows:
        return []
    figure = Figure(figsize=(9, 5.3), layout="constrained")
    axes = figure.subplots(2, 2)
    for i, environment in enumerate(ENVIRONMENTS):
        for j, metric in enumerate(("raw_return", "success")):
            axis = axes[i, j]
            for condition, recipe in sorted({(r["Condition"], r["Recipe"]) for r in rows}):
                points = [r for r in rows if (r["Condition"], r["Recipe"], r["Environment"]) ==
                          (condition, recipe, environment)]
                axis.plot([r["n"] for r in points], [r[metric] if r[metric] is not None else math.nan for r in points],
                          label=condition, color="#b23a48" if condition.startswith("F") else "#167d8d",
                          linestyle="--" if condition.endswith("+") else "-", marker="o", markersize=4)
            axis.set_title(environment + (": return" if metric == "raw_return" else ": success"), fontsize=10)
            axis.set_xlabel("Nominal acquired intervals", fontsize=9)
            axis.grid(alpha=.2)
            if metric == "success":
                axis.set_ylim(-.05, 1.05)
    axes[0, 0].legend(fontsize=8, ncol=2)
    return [_figure(figure, ("Exploratory assessment" if phase == "confirmation" else "Development, original recipes")
                    + ": equal training-seed means, common first " + ("four" if phase == "confirmation" else "two")
                    + " layouts at each checkpoint. Endpoint tables use all eight layouts, so their success fractions can differ. "
                    "No interpolation, CIs or pooled development/assessment.")]


def _archives(snapshot):
    for unit in snapshot["units"]:
        summary = unit["summary"]
        if summary["operation"] == "initial_archives":
            diagnostics = summary.get("diagnostics", {})
            for rank, archive in diagnostics.get("ranks", {}).items():
                yield "Initial r=" + rank, archive, archive.get("allocation", {}), summary.get("active_seconds")
        elif summary.get("archive"):
            settings = summary["settings"]
            yield (_condition(unit) + " " + settings["phase"] + " seed=" + str(settings["seed"]),
                   summary["archive"], summary.get("allocation", {}), summary.get("active_seconds"))


def diagnostic_outputs(snapshot):
    from matplotlib.figure import Figure

    archives = list(_archives(snapshot))
    outputs = []
    if archives:
        figure = Figure(figsize=(9, 3.2), layout="constrained")
        spectrum, coverage = figure.subplots(1, 2)
        # Keep the small panel legible; all snapshots remain in the scalar table.
        initial = [a[0] for a in archives if a[0].startswith("Initial") and a[1].get("rank")]
        final = [a[0] for a in archives if not a[0].startswith("Initial")]
        spectra = set(initial[:3] + final[-3:])
        rows = []
        for i, (label, archive, allocation, seconds) in enumerate(archives):
            values = sorted(archive.get("factor_eigenvalues", archive.get("eigenvalues", [])), reverse=True)
            if values and label in spectra:
                spectrum.plot(range(1, len(values) + 1), values, label=label, linewidth=1)
            for offset, block, color in ((-.17, "actor", "#167d8d"), (.17, "critic", "#b23a48")):
                value = allocation.get(block, {}).get("correlated_trace_fraction")
                if value is not None:
                    coverage.bar(i + offset, value, width=.32, color=color,
                                 label=block if i == 0 else None)
            rows.append({"Snapshot": label, "EWC rank": archive.get("rank"), "Trace": archive.get("trace"),
                         "Correlated fraction": archive.get("correlated_trace_fraction"),
                         "Actor fraction": allocation.get("actor", {}).get("correlated_trace_fraction"),
                         "Critic fraction": allocation.get("critic", {}).get("correlated_trace_fraction"),
                         "Residual min (before correction)": archive.get("minimum_residual_before_correction"),
                         "Trace error": archive.get("trace_error"), "Storage bytes": archive.get("storage_bytes"),
                         "Whole-unit s": seconds})
        spectrum.set(title="LL^T factor spectrum, not full Fisher", xlabel="Eigenvalue index")
        coverage.set(title="Correlated trace coverage", ylim=(0, 1.05))
        coverage.set_xticks(range(len(archives)), [str(i + 1) for i in range(len(archives))])
        coverage.legend(fontsize=8)
        if spectrum.lines:
            spectrum.legend(fontsize=6, ncol=2)
        outputs += [_figure(figure, "Initial and final archives are labeled separately; at most six spectra are drawn. Streaming thin-SVD is not dense recompression."),
                    _table(rows, "Rank, Allocation And Measured Cost", "Snapshot numbers follow table order; missing coverage is not zero. Initial costs are shared, not per-rank benchmarks.")]
    timing_rows = []
    for unit in snapshot["units"]:
        if unit["summary"]["operation"] != "resource_audit":
            continue
        result = unit["summary"].get("result", {})
        for row in result.get("timings", [])[:16]:
            if not all(k in row for k in ("rank", "method", "fit_seconds")):
                continue
            timing_rows.append({"Rank": row["rank"], "Method": row["method"], "Healthy": row.get("health", {}).get("healthy"),
                                **{key: row.get(key) if row.get("timing_available", True) else None
                                   for key in ("fit_seconds", "retirement_seconds", "compression_seconds",
                                               "factor_generation_seconds", "solve_seconds")},
                                "Warmups / repeats": [result.get("warmups"), result.get("repeats")]})
    if timing_rows:
        outputs.append(_table(timing_rows, "Cached Resource Audit", "Transfer-inclusive whole-fit and isolated component timings overlap; do not sum them."))
    fits = [u for u in snapshot["units"] if u["summary"].get("metrics")]
    if fits:
        figure = Figure(figsize=(9, 3.2), layout="constrained")
        axes = figure.subplots(1, 3)
        for axis, key, title in zip(axes, ("post_step_ewc", "fd", "ewc_proposal_difference"),
                                    ("EWC after accepted fit", "FD at fit (saved scalar)", "First-proposal EWC influence")):
            for unit in fits:
                rows = sorted(unit["summary"]["metrics"], key=lambda r: r["n"])
                if any(row.get(key) is not None for row in rows):
                    axis.plot([r["n"] for r in rows], [r.get(key) if r.get(key) is not None else math.nan for r in rows],
                              label=_condition(unit) + " " + unit["summary"]["settings"]["phase"], linewidth=1)
            axis.set(title=title, xlabel="Acquired intervals")
            if not axis.lines:
                axis.text(.5, .5, "Not collected", ha="center", transform=axis.transAxes)
        outputs.append(_figure(figure, "Separate per-trajectory diagnostics, not pooled outcomes. Missing post-fit EWC is not replaced by the zero first-step EWC scalar. Training NLL is not a common navigation-risk metric."))
    return outputs


def recording_outputs(snapshot, notebook_directory):
    rows, seen = [], set()
    for unit in _trajectories(snapshot, "confirmation"):
        for panel in unit["summary"].get("evaluations", []):
            if panel["n"] not in (0, unit["summary"]["settings"]["total"]) or not panel["episodes"]:
                continue
            # Only preselected first-layout clips; do not choose videos on outcomes.
            episode = panel["episodes"][0]
            origin = Path(panel.get("shared_initial_source", panel.get("source_unit", unit["path"])))
            for video in episode.get("videos", [])[:2]:
                path = Path(video["path"])
                path = path if path.is_absolute() else origin / path
                path = path.resolve()
                if str(path) in seen:
                    continue
                seen.add(str(path))
                rows.append({"Condition": "Shared initialization" if panel["n"] == 0 else _condition(unit),
                             "Seed": "shared" if panel["n"] == 0 else unit["summary"]["settings"]["seed"],
                             "Environment": panel["environment"], "n": panel["n"],
                             "Raw return": episode.get("raw_return"), "Success": episode["success"],
                             "Video": quote(os.path.relpath(path, notebook_directory), safe="/.") if path.is_file() else None})
    gallery = Path(snapshot["root"]).with_name(Path(snapshot["root"]).name + "-success-example-v1")
    examples = []
    if (gallery / "COMPLETED").is_file():
        marker = _json(gallery / "COMPLETED", MAX_CONFIG_BYTES)
        manifest_data = _read(gallery / "manifest.json", MAX_CONFIG_BYTES)
        _require(hashlib.sha256(manifest_data).hexdigest() == marker.get("manifest_sha256"),
                 "success example completion seal mismatch")
        manifest = _json(gallery / "manifest.json", MAX_CONFIG_BYTES, data=manifest_data)
        summary_data = _read(gallery / "summary.json", MAX_CONFIG_BYTES)
        _require(hashlib.sha256(summary_data).hexdigest() ==
                 manifest.get("final_files", {}).get("summary.json", {}).get("sha256"),
                 "success example summary checksum mismatch")
        example = _json(gallery / "summary.json", MAX_CONFIG_BYTES, data=summary_data)
        _require(example.get("version") == "plan19-saved-success-example-v1"
                 and example.get("success") is True and example.get("new_evaluation") is False,
                 "success example schema mismatch")
        matches = [u for u in _trajectories(snapshot, "confirmation")
                   if Path(u["path"]).resolve() == Path(example["source_unit"]).resolve()]
        _require(len(matches) == 1, "success example source is not an assessed trajectory")
        unit = matches[0]
        episodes = [e for p in unit["summary"]["evaluations"]
                    if p["n"] == example["n"] and p["environment"] == "moderate_A"
                    for e in p["episodes"] if e.get("episode_id") == example["episode_id"] and e["success"]]
        _require(len(episodes) == 1, "success example does not match a saved successful endpoint")
        path = (gallery / example["video"]).resolve()
        _require(path.parent == gallery.resolve() and path.is_file(), "success example video unavailable")
        examples.append({"Condition": _condition(unit) + " (selected success)", "Seed": example["seed"],
                         "Environment": "moderate_A", "n": example["n"],
                         "Raw return": example["raw_return"], "Success": True,
                         "Video": quote(os.path.relpath(path, notebook_directory), safe="/.")})
    rows = rows[:12] + examples
    if not rows:
        return []
    note = "Shared initial recordings are shown once; clips are links, not embedded payloads."
    if examples:
        note += (" Outcome-selected success example: not representative or a new evaluation. "
                 "Original camera observations play once per second; no top-down recording was saved. "
                 "This final moderate-A layout has "
                 + ("a blocked" if example.get("direct_path_blocked") else "an open")
                 + " direct path.")
    output = _table(rows, "Preselected Initial / Final Videos", note)
    for row in rows:
        if row["Video"]:
            escaped = html.escape(row["Video"], quote=True)
            output["data"]["text/html"] = output["data"]["text/html"].replace(
                "<td>" + escaped + "</td>", '<td><a href="' + escaped + '">video</a></td>')
    return [output]


def action_outputs(snapshot):
    from matplotlib.figure import Figure

    units = [u for u in snapshot["units"] if u["summary"].get("action_metrics")]
    if not units:
        return []
    figure = Figure(figsize=(8, 3), layout="constrained")
    axis = figure.subplots()
    coverage = []
    for unit in units:
        rows = [r for r in unit["summary"]["action_metrics"] if r.get("complete_history")
                and _number(r.get("sigma_m")) and _number(r.get("turn_bouts"))]
        label = _condition(unit) + " " + unit["summary"]["settings"]["phase"] + " " + str(unit["summary"]["settings"]["seed"])
        if rows:
            axis.scatter([r["sigma_m"] for r in rows], [r["turn_bouts"] for r in rows], label=label, s=12, alpha=.65)
        coverage.append({"Trajectory": label, "Complete episodes": len(rows),
                         "Saved episode summaries": len(unit["summary"]["action_metrics"])})
    axis.set(xlabel="Barrier difficulty sigma (m)", ylabel="Commanded same-sign turn bouts")
    if axis.collections:
        axis.legend(fontsize=7, ncol=2)
    return [_figure(figure, "Commanded steering bouts, not measured yaw. Complete new training episodes only; reporting does not consume policy RNG."),
            _table(coverage, "Turning Coverage", "Early control action history is unavailable. Partial late-history controls are not directly compared to complete new trajectories.")]


def result_outputs(snapshot, notebook_directory=None):
    admission, selection = _admission(snapshot), _selection(snapshot)
    outputs = [_table(settings_rows(snapshot), "Four Conditions", "Two paired assessment seeds, one shared initialization; reused controls are not new replicas."),
               _table([{"Rank": admission.get("rank"), "Admission reason": admission.get("reason"),
                        "Forecast": admission.get("forecast"), "Correction": selection.get("correction")}],
                      "Rank Admission And Calibration"),
               _table(development_rows(snapshot), "Development Retention (Full Eight Layouts)",
                      "Expanded T0 is the reference. +/-5 raw-return units is a descriptive matching rule, not an equivalence interval. Every original/corrected recipe remains visible.")]
    selected = selection.get("selected", {})
    if selected:
        outputs.append(_table([{"Method": method, "Recipe": _recipe(values),
                                "Retention gap": values.get("retention_gap"),
                                "Retention matched": values.get("retention_matched"),
                                "Development unit": Path(values["development_unit"]).name if values.get("development_unit") else None}
                               for method, values in selected.items()],
                              "Frozen Development Selection", "A ridge-adjusted contrast describes calibrated policies, not an isolated rank/FD mechanism."))
        gaps = [selected.get(method, {}).get("retention_gap") for method in METHODS]
        if all(_number(gap) for gap in gaps):
            outputs.append(_table([{"Selected F+ minus T+ retention gap": gaps[1] - gaps[0]}],
                                  "Direct Positive-Rank Retention Gap", "Both being close to T0 does not guarantee they match one another."))
    outputs += _curves(snapshot, "development") + _curves(snapshot, "confirmation")
    paired = paired_endpoint_rows(snapshot)
    seeds = [str(s) for s in snapshot["config"].get("protocol", {}).get("assessment_seeds", [199101, 199102])]
    columns = ["Environment", "Metric", "Contrast", *seeds, "Mean gap", "Seed range (not CI)", "Pairs"]
    outputs.append(_table([{k: row[k] for k in columns} for row in paired], "Paired Assessment Endpoints (Eight Layouts)",
                          "Within-seed effects first, then equal paired means and seed ranges, not CIs. Missing, ambiguous or unmatched-budget/layout pairs remain pending. Development is never pooled. Whole-path and fitting seconds are recorded observations, not guaranteed savings."))
    costs = [{"Condition": _condition(u), "Seed": u["summary"]["settings"]["seed"],
              "Evidence": "reused" if u["reused"] else "new", "Nominal endpoint": u["summary"]["settings"]["total"],
              "Actual acquisitions": u["summary"].get("n"),
              "Fitting s": u["summary"].get("timings", {}).get("fitting"),
              "Whole-path s": u["summary"].get("active_seconds")}
             for u in _trajectories(snapshot, "confirmation")]
    if costs:
        outputs.append(_table(costs, "Assessment Acquisition And Cost", "Replay draws and optimizer steps are not acquisitions. Source costs are not charged again to the extension."))
    outputs += diagnostic_outputs(snapshot) + action_outputs(snapshot)
    if notebook_directory is not None:
        outputs += recording_outputs(snapshot, notebook_directory)
    visible = []
    for output in outputs:
        visible.append(output)
        if "image/png" in output.get("data", {}):
            caption = output["data"].get("text/plain", "")
            visible.append({"output_type": "display_data", "metadata": {},
                            "data": {"text/plain": caption,
                                     "text/html": "<p>" + html.escape(caption) + "</p>"}})
    return visible


def display_progress(snapshot):
    from IPython.display import display
    for output in progress_outputs(snapshot):
        display(output["data"], raw=True)


def display_results(snapshot, notebook_directory=None):
    from IPython.display import display
    for output in result_outputs(snapshot, notebook_directory):
        display(output["data"], raw=True)


def conclusion(snapshot):
    paired = paired_endpoint_rows(snapshot)
    if not snapshot["complete"] or not paired or any(row["Pairs"] != 2 for row in paired):
        return ("#### Fix 5A Conclusion\n\n**Provisional / incomplete.** "
                "The full paired rank/FD assessment remains pending. Missing outcomes are not zero. "
                "Development and unsealed previews do not establish an assessment winner.")

    effects = {(r["Environment"], r["Metric"], r["Contrast"]): r["Mean gap"] for r in paired}
    rank_td = effects["moderate_A", "Raw return", "Rank, TD: T+ - T0"]
    rank_fd = effects["moderate_A", "Raw return", "Rank, FD: F+ - F0"]
    success_fd = effects["moderate_A", "Success (pp)", "Rank, FD: F+ - F0"]
    interaction = effects["moderate_A", "Raw return", "Interaction: (F+ - T+) - (F0 - T0)"]
    groups = {condition: [u for u in _trajectories(snapshot, "confirmation")
                          if _condition(u) == condition] for condition in ("T0", "F0", "T+", "F+")}
    rank = groups["F+"][0]["summary"]["settings"]["ewc_rank"]
    labels = {"T0": "TD replay, diagonal", "F0": "TD + FD, diagonal",
              "T+": f"TD replay, rank {rank}", "F+": f"TD + FD, rank {rank}"}
    table = ["| Condition | Initial A successes | Moderate A successes | Moderate A return |",
             "| --- | ---: | ---: | ---: |"]
    for condition, units in groups.items():
        counts, returns = {}, []
        for environment in ENVIRONMENTS:
            episodes = [e for unit in units for e in _panel(unit, environment, n=1024)["episodes"]]
            counts[environment] = f"{sum(e['success'] for e in episodes)}/{len(episodes)}"
            if environment == "moderate_A":
                returns = [e["raw_return"] for e in episodes]
        table.append(f"| {labels[condition]} | {counts['initial_A']} | {counts['moderate_A']} | {_mean(returns):.3f} |")

    headline = ("No demonstrated FD acquisition benefit from wider EWC."
                if rank_fd <= 0 and success_fd <= 0 else "Exploratory paired assessment complete.")
    text = ("#### Fix 5A Conclusion\n\n**Scope complete. " + headline + "**\n\n"
            + "\n".join(table)
            + "\n\nCounts combine two training seeds evaluated on the same eight layouts, "
              "not 16 independent replicas. Every condition has rank-one STM; FD is the additional age-weighted penalty.\n\n"
            + f"Wider EWC changed moderate-A return by **{rank_td:+.3f} for TD replay** and "
              f"**{rank_fd:+.3f} for FD**, versus their diagonal controls. "
              f"The return interaction is **{interaction:+.3f}**. ")
    if interaction > 0 and rank_td < 0 and rank_fd <= 0:
        text += ("That positive interaction comes from reduced TD performance, not improved FD performance; "
                 "it must not be called an FD gain. ")
    if _selection(snapshot).get("correction") is None:
        text += "Development retention matched without changing either ridge coefficient. "
    else:
        text += "The ridge correction makes these calibrated-policy contrasts, not isolated rank effects. "

    ratios = []
    for low, high, label in (("T0", "T+", "TD replay"), ("F0", "F+", "FD")):
        before = [u["summary"].get("active_seconds") for u in groups[low]]
        after = [u["summary"].get("active_seconds") for u in groups[high]]
        if all(_number(v) and v > 0 for v in before + after):
            ratios.append(f"{label} {100 * (sum(after) / sum(before) - 1):+.1f}%")
    if ratios:
        text += ("\n\nObserved whole-path time changes were " + ", ".join(ratios)
                 + "; these include collection and evaluation, not just archive operations. ")
    text += ("The wider archive passed the numerical audit and materially expanded critic correlation coverage, "
             "but that did not establish a navigation or sample-efficiency benefit here. "
             "No samples-saved/S estimate is supported by a common capability crossing.\n\n"
             "**Practical choice for this A-only pipeline:** "
             + ("retain diagonal TD replay as the incumbent; wider EWC has not earned its extra complexity. "
                if rank_fd <= 0 and rank_td <= 0 else
                "treat any observed advantage as a lead requiring further assessment. ")
             + "Two paired seeds, reused controls/layouts and one "
             "initialization do not establish significance or equivalence; seed ranges are not confidence intervals. "
             "This does not negate Plan 18's result or test B/physical navigation.")
    return text


def build_notebook(root, output, base_notebook=None):
    """Atomically replace only Fix5A-tagged cells; preserve all other history."""
    import nbformat

    root, output = Path(root), Path(output)
    snapshot = load_report(root)
    _require(output.suffix == ".ipynb" and not output.is_symlink(), "output must be a nonsymlink notebook")
    _require(not any((p / "COMPLETED").exists() for p in output.resolve().parents),
             "output is inside a completed run")
    if snapshot["source"]:
        _require(not output.resolve().is_relative_to(Path(snapshot["source"])), "source artifacts are read-only")
    candidates = [output, Path(base_notebook) if base_notebook is not None else root / "history.ipynb",
                  root / "history.ipynb", DEFAULT_NOTEBOOK]
    base = next((p for p in candidates if p.exists()), None)
    book = _json(base, MAX_HISTORY_BYTES) if base else dict(nbformat.v4.new_notebook())
    _require(book.get("nbformat") == 4 and isinstance(book.get("cells"), list), "base must be a v4 notebook")
    book = copy.deepcopy(book)
    history = [c for c in book["cells"] if _TAG not in c.get("metadata", {})]
    used = {c["id"] for c in history if "id" in c}
    _require(len(used) == sum("id" in c for c in history), "historical cell ids must be unique")

    def own(cell, name):
        stem, suffix = "plan19-phase3-fix5a-" + name, 0
        identifier = stem
        while identifier in used:
            suffix += 1
            identifier = f"{stem}-{suffix}"
        used.add(identifier)
        cell["id"] = identifier
        cell["metadata"][_TAG] = True
        return cell

    project = Path(__file__).resolve().parents[2]
    root_path = root.resolve()
    expression = (f"project_root / {str(root_path.relative_to(project))!r}" if root_path.is_relative_to(project)
                  else repr(str(root_path)))
    bootstrap = ("from pathlib import Path\nimport sys\n"
                 "for parent in (Path.cwd(), *Path.cwd().parents):\n"
                 "    if (parent / 'numerical_experiment/picar_head_rank/report.py').is_file():\n"
                 "        project_root = parent\n"
                 "        if str(parent) not in sys.path:\n            sys.path.insert(0, str(parent))\n"
                 "        break\n"
                 "else:\n    raise RuntimeError('Open from within the project checkout.')\n"
                 "from numerical_experiment.picar_head_rank import report as rank_report\n"
                 f"rank_snapshot = rank_report.load_report({expression})\n")
    progress = own(nbformat.v4.new_code_cell(bootstrap + "rank_report.display_progress(rank_snapshot)",
                                           outputs=progress_outputs(snapshot)), "progress")
    intro = ("### Phase 3 Fix 5A: Correlated EWC And STM FD\n\n"
             "Plan 18 motivates testing wider correlated EWC after Fix 5 assessed only rank zero. "
             "This A-only extension compares reused T0/F0 with new T+/F+ from the same initialized model. "
             "EWC correlation rank is distinct from unchanged STM LoRA rank 1. "
             "Development and assessment are separate. Two paired seeds and reused layouts do not "
             "constitute independent confirmation or establish significance.")
    directory = output.parent.resolve()
    directory_expression = (f"project_root / {str(directory.relative_to(project))!r}"
                            if directory.is_relative_to(project) else repr(str(directory)))
    results = [own(nbformat.v4.new_markdown_cell(intro), "intro"),
               own(nbformat.v4.new_code_cell(bootstrap + f"rank_report.display_results(rank_snapshot, {directory_expression})",
                                             outputs=result_outputs(snapshot, directory)), "results"),
               own(nbformat.v4.new_markdown_cell(conclusion(snapshot)), "conclusion")]
    position = 1 if history and history[0].get("cell_type") == "markdown" else 0
    book["cells"] = history[:position] + [progress] + history[position:] + results
    nbformat.validate(copy.deepcopy(book))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".head-rank-report-", suffix=".ipynb", dir=output.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(book, handle, allow_nan=False, ensure_ascii=True, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
