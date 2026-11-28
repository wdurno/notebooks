"""Bounded, artifact-only Plan 19 Fix 6 reporting (no runtime imports).

Only sealed scalar summaries enter comparisons. Unit progress is a separately
labelled provisional view; it never repairs or supplies scientific results.
"""

from __future__ import annotations

from collections import defaultdict
import base64
import copy
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


VERSION = "plan19-phase3-fix6-v1"
_TAG = "plan19_phase3_fix6"
_TASK_ONLY_TAG = "plan19_phase3_fix6_task_only"
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "picar/plan19_results.ipynb"
OPERATIONS = {"software", "readiness_geometry", "readiness_compliance", "readiness_teacher", "distillation_features",
              "resource_profile", "collection", "distillation", "features",
              "initial_fit", "initial_archive", "trajectory", "final_audit"}
PHASES = {"development", "assessment", "initialization", "readiness"}
ENVIRONMENTS = ("initial_A", "moderate_A")
DIFFICULTIES = ("initial", "medium", "final")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PROGRESS_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 160 * 1024 * 1024
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_UNITS = 128
MAX_SCAN_DIRS = 2048
MAX_ROWS = 100_000


class ArtifactError(ValueError):
    """Missing, incompatible, or out-of-bounds reporting artifact."""


def _require(ok, message):
    if not ok:
        raise ArtifactError("Fix 6: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _task_only(snapshot):
    protocol = {**snapshot.get("protocol", {}), **snapshot.get("progress", {}).get("protocol", {})}
    return (protocol.get("reward_mode") == "task_only" or protocol.get("adherence_enabled") is False
            or Path(snapshot["root"]).name == "coaching-fix6-v4")


def _read(path, limit):
    path = Path(path)
    _require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(),
             f"missing/nonregular artifact: {path}")
    _require(path.stat().st_size <= limit, f"artifact exceeds size bound: {path}")
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    _require(len(value) <= limit, f"artifact exceeds size bound: {path}")
    return value


def _json(path, limit=MAX_JSON_BYTES, *, data=None):
    def invalid(value):
        raise ArtifactError(f"Fix 6: nonfinite JSON in {path}: {value}")

    try:
        value = json.loads(_read(path, limit) if data is None else data, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ArtifactError(f"Fix 6: invalid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    remaining = 500_000

    def visit(item, depth=0):
        nonlocal remaining
        remaining -= 1
        _require(remaining >= 0 and depth <= 24, "scalar JSON complexity bound exceeded")
        if isinstance(item, dict):
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            _require(len(item) <= MAX_ROWS, "scalar JSON array bound exceeded")
            for child in item:
                visit(child, depth + 1)
        else:
            _require(item is None or type(item) is bool or isinstance(item, str) or _number(item),
                     "nonfinite or nonscalar JSON value")

    visit(value)
    return value


def _rows(value, context):
    _require(isinstance(value, list) and len(value) <= MAX_ROWS
             and all(isinstance(row, dict) for row in value), context + " must be a bounded object list")
    return value


def _optional_numbers(value, keys, context):
    for key in keys:
        _require(value.get(key) is None or _number(value[key]), f"{context}.{key} must be finite or null")


def _version(value, *, required=False):
    if required or "schema_version" in value:
        _require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
                 "schema_version must be 1")
    if "version" in value:
        _require(value["version"] == VERSION, "version mismatch")


def _episode(episode, *, heldout=False):
    _require(type(episode.get("success")) is bool, "episode.success must be Boolean")
    _optional_numbers(episode, ("raw_return", "task_return", "guidance_return", "coded_return",
                                "difficulty_start", "difficulty_end", "start_difficulty", "end_difficulty"),
                      "episode")
    for video in _rows(episode.get("videos", []), "episode.videos"):
        _require(isinstance(video.get("path"), str) and bool(video["path"]), "video path required")
    if heldout:
        _require(episode.get("guidance_return", 0) in (None, 0), "held-out guidance reward must be zero")
        _require(not episode.get("guidance_active", False), "held-out episode must be unassisted")


def _summary(value):
    _version(value, required=True)
    _require(value.get("operation") in OPERATIONS, "unsupported operation")
    settings = value.get("settings")
    _require(isinstance(settings, dict), "settings must be an object")
    _require(settings.get("condition") in (None, "U", "G"), "unknown condition")
    _require(settings.get("phase", "readiness" if value["operation"] in ("software", "final_audit") else None) in PHASES,
             "unknown phase")
    _require(settings.get("seed") is None or _count(settings["seed"]), "settings.seed invalid")
    _require(_count(settings.get("run_order", 0)), "settings.run_order invalid")
    _optional_numbers(value, ("active_seconds", "n"), "summary")
    if value["operation"] == "trajectory":
        _require(settings.get("condition") in ("U", "G") and _count(settings.get("seed")),
                 "trajectory condition/seed required")
        _require(_count(settings.get("total")) and settings["total"] > 0, "positive trajectory total required")
    for row in _rows(value.get("metrics", []), "metrics"):
        _require(_number(row.get("n")) and row["n"] >= 0, "metric.n required")
        _optional_numbers(row, ("post_step_ewc", "post_step_fd", "fd", "age_mean", "data_nll"), "metric")
    panels = set()
    for panel in _rows(value.get("evaluations", []), "evaluations"):
        _require(_count(panel.get("n")) and panel.get("environment") in ENVIRONMENTS,
                 "evaluation n/environment invalid")
        identity = panel["n"], panel["environment"]
        _require(identity not in panels, "duplicate evaluation panel")
        panels.add(identity)
        _require(panel.get("acquired_count") is None or _count(panel["acquired_count"]),
                 "evaluation acquired_count invalid")
        for episode in _rows(panel.get("episodes"), "evaluation.episodes"):
            _episode(episode, heldout=True)
    for episode in _rows(value.get("episodes", []), "episodes"):
        _episode(episode)
    for curve in _rows(value.get("curves", []), "curves"):
        _require(_count(curve.get("epoch")), "distillation epoch required")
        _optional_numbers(curve, ("train_kl", "validation_kl", "train_turn_kl", "validation_turn_kl",
                                  "train_turn_agreement", "validation_turn_agreement", "train_mse", "validation_mse"), "curve")
    for row in _rows(value.get("action_metrics", []), "action_metrics"):
        _require(type(row.get("complete_history", True)) is bool, "action complete_history must be Boolean")
        _optional_numbers(row, ("n", "difficulty", "turn_bouts", "moving_seconds", "mean_abs_steering_deg",
                                "time_above_turn_threshold_s", "measured_yaw_change_deg"), "action metric")
    if "ready" in value:
        _require(type(value["ready"]) is bool, "readiness.ready must be Boolean")
    for key in ("audit_clear_correct", "audit_clear_total", "correct_left_turns", "correct_right_turns"):
        _require(value.get(key) is None or _count(value[key]), key + " must be a nonnegative integer")
    if "audit_clear_correct" in value and "audit_clear_total" in value:
        _require(value["audit_clear_correct"] <= value["audit_clear_total"], "audit correct exceeds total")
    if value["operation"].startswith("readiness_"):
        _rows(value.get("rows", []), "readiness rows")
    if "confusion" in value:
        matrix = value["confusion"]
        _require(isinstance(matrix, dict) and all(isinstance(row, dict) and all(_count(v) for v in row.values())
                                                 for row in matrix.values()), "confusion must contain nonnegative counts")


def load_report(root):
    """Read campaign progress and sealed summaries; never open a checkpoint.

    An absent root is a not-started report. An unsealed summary is deliberately
    not opened, even if it happens to contain parseable JSON.
    """
    root = Path(root)
    _require(not any(p.is_symlink() for p in (root, *root.parents)), "symlinked campaign root")
    total = 0

    def read(path, limit=MAX_JSON_BYTES):
        nonlocal total
        data = _read(path, limit)
        total += len(data)
        _require(total <= MAX_TOTAL_BYTES, "total scalar JSON bound exceeded")
        return _json(path, limit, data=data)

    progress = read(root / "progress.json", MAX_PROGRESS_BYTES) if (root / "progress.json").exists() else {}
    if (root / "progress.json").exists():
        _version(progress)
        _require(all(isinstance(progress.get(k), str) for k in ("status", "stage")), "progress status/stage required")
        _require(all(_count(progress.get(k)) for k in ("completed_units", "planned_units")), "progress counts invalid")
        _require(progress["completed_units"] <= progress["planned_units"], "completed exceeds planned units")
        _require(_number(progress.get("charged_seconds")) and progress["charged_seconds"] >= 0,
                 "progress charged_seconds invalid")
        _require(isinstance(progress.get("active_progress", {}), dict), "active_progress must be an object")
    config_path = root / "orchestration/config.json"
    identity = read(config_path) if config_path.exists() else {}
    _version(identity)
    _require(isinstance(identity.get("protocol", {}), dict), "orchestration protocol must be an object")
    _require(isinstance(progress.get("protocol", {}), dict), "progress protocol must be an object")
    protocol = {**identity.get("protocol", {}), **progress.get("protocol", {})}
    units, pending = [], []
    directory = root / "units"
    if directory.exists():
        _require(not directory.is_symlink() and directory.is_dir(), "invalid units directory")
        children = []
        for child in directory.iterdir():
            children.append(child)
            _require(len(children) <= MAX_SCAN_DIRS, "unit scan bound exceeded")
        for child in sorted(children):
            _require(not child.is_symlink(), "symlinked unit")
            if not child.is_dir():
                continue
            _require(len(units) + len(pending) < MAX_UNITS, "unit limit exceeded")
            marker = child / "COMPLETED"
            if marker.exists():
                _require(marker.is_file() and not marker.is_symlink(), "invalid completion marker")
                value = read(child / "summary.json")
                _summary(value)
                units.append({"name": child.name, "path": str(child.resolve()), "summary": value})
            else:
                preview = read(child / "progress.json", MAX_PROGRESS_BYTES) if (child / "progress.json").exists() else {}
                _version(preview)
                pending.append({"name": child.name, "path": str(child.resolve()), "progress": preview})
    complete = (progress.get("status") == "completed" and progress.get("planned_units", 0) > 0
                and progress.get("completed_units") == progress.get("planned_units") == len(units) and not pending)
    return {"root": str(root.resolve()), "progress": progress, "protocol": protocol, "units": units,
            "pending": pending, "complete": complete}


def _display(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _html(markup, plain):
    return {"output_type": "display_data", "metadata": {},
            "data": {"text/html": markup, "text/plain": plain}}


def _table(rows, title, note=""):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    esc = lambda v: html.escape(_display(v), quote=True)
    markup = f"<h4>{html.escape(title)}</h4><p>{html.escape(note)}</p>"
    if not rows:
        markup += "<p>Not yet available.</p>"
    else:
        markup += "<div style='overflow-x:auto'><table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in columns)
        markup += "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(r.get(c))}</td>" for c in columns)
                                                  + "</tr>" for r in rows) + "</tbody></table></div>"
    plain = title + "\n" + note + "\n" + "\n".join("; ".join(f"{k}: {_display(v)}" for k, v in r.items()) for r in rows)
    return _html(markup, plain)


def progress_outputs(snapshot):
    progress = snapshot["progress"]
    active = progress.get("active_progress", {})
    unit_preview = next((u["progress"] for u in snapshot["pending"]
                         if u["path"] == progress.get("active_unit") or u["name"] == progress.get("active_unit")), {})
    active = {**unit_preview, **active}
    settings = active.get("settings", {})
    row = {"Status": progress.get("status", "Not started / incomplete"),
           "Stage": progress.get("stage", "Not started"),
           "Units": f"{progress.get('completed_units', 0)} / {progress.get('planned_units', '?')}",
           "Condition": active.get("condition", settings.get("condition")),
           "Seed": active.get("seed", settings.get("seed")),
           "Captured intervals": active.get("n", active.get("captured_intervals", active.get("count"))),
           "Charged hours": progress.get("charged_seconds", 0) / 3600,
           "Remaining forecast": progress.get("forecast"), "Active unit": progress.get("active_unit")}
    if _task_only(snapshot):
        row.update({"Reward": "Task-only; identical U/G task reward; coaching is the treatment",
                    "Adherence": "disabled by design; no compliance readiness gate or bonus"})
    failed = [r for r in readiness_rows(snapshot) if r["Status"] == "Failed"]
    if failed:
        row["Readiness stop"] = "; ".join(r["Gate"] + ": " + r["Reason / qualification"] for r in failed)
    if row.get("Active unit"):
        row["Active unit"] = Path(row["Active unit"]).name
    if isinstance(row.get("Remaining forecast"), dict) and row["Remaining forecast"].get("reason"):
        row["Remaining forecast"] = row["Remaining forecast"]["reason"]
    outputs = [_table([{"Item": key, "Value": value} for key, value in row.items() if value is not None],
                      "Fix 6 Progress", "Active unit values are an unsealed preview; only sealed units enter comparisons.")]
    previews = active.get("evaluation_preview", [])
    if previews:
        outputs.append(_table(_rows(previews, "evaluation_preview"), "Provisional Held-Out Preview",
                              "Unsealed, possibly partial. Excluded from all result curves and paired tables."))
    return outputs


def readiness_rows(snapshot):
    """Describe sealed admission failures without claiming later learning ran."""
    result = []
    for operation, title in (("readiness_geometry", "Geometry"), ("readiness_compliance", "Qwen compliance"),
                             ("readiness_teacher", "Teacher left / right"), ("resource_profile", "Resources")):
        if operation == "readiness_compliance" and _task_only(snapshot):
            result.append({"Gate": title, "Status": "disabled by design",
                           "Evidence": {"adherence_enabled": False, "compliance_raw_magnitude": 0,
                                        "reward_mode": "task_only"},
                           "Reason / qualification": "No compliance readiness gate, evaluation or bonus is expected. "
                               "Text coaching and cue logs remain enabled for G."})
            continue
        summaries = [u["summary"] for u in snapshot["units"] if u["summary"]["operation"] == operation]
        if not summaries:
            result.append({"Gate": title, "Status": "Pending", "Evidence": None,
                           "Reason / qualification": "No sealed audit yet."})
        for summary in summaries:
            ready, rows = summary.get("ready"), summary.get("rows", [])
            details, reason = {}, summary.get("failure_reason", summary.get("reason"))
            if operation == "readiness_geometry":
                collisions = sum(r.get("collision") is True for r in rows)
                details = {"cases": len(rows), "collisions": collisions,
                           "minimum_clearance_m": min((r["minimum_clearance_m"] for r in rows
                                                       if _number(r.get("minimum_clearance_m"))), default=None)}
                if ready is False and not reason:
                    reason = f"Geometry gate failed: {collisions}/{len(rows)} audited cases had contact."
            elif operation == "readiness_compliance":
                audit = [r for r in rows if r.get("panel") == "audit"]
                clear = [r for r in audit if r.get("expected") != "uncertain"]
                correct = summary.get("audit_clear_correct", sum(r.get("correct") is True for r in clear) if clear else None)
                total = summary.get("audit_clear_total", len(clear) if audit else None)
                details = {"clear_correct": correct, "clear_total": total, "clear_target": ">=90%",
                           "audit_uncertain": sum(r.get("judgment") == "uncertain" for r in audit) if audit else None,
                           "audit_windows": len(audit) if audit else None, "inference_errors": summary.get("errors")}
                if ready is False and not reason:
                    reason = f"Compliance gate failed: {correct}/{total} clear audit cases correct; target >=90%."
            elif operation == "readiness_teacher":
                left, right = summary.get("correct_left_turns"), summary.get("correct_right_turns")
                details = {"correct_left_turns": left, "correct_right_turns": right,
                           "paired_diagnostic_trials": len(rows),
                           "collisions": sum(r.get("collision") is True for r in rows)}
                if ready is False and not reason:
                    reason = f"Teacher gate failed: correct left/right turns {left}/{right}; both directions are required."
            else:
                details = {k: summary.get(k) for k in ("median_teacher_seconds", "median_compliance_seconds", "forecast")}
                if ready is False and not reason:
                    reason = "Resource/protocol admission failed; inspect the preceding audit gates and saved forecast."
            result.append({"Gate": title, "Status": "Passed" if ready is True else "Failed" if ready is False else "Not recorded",
                           "Evidence": details, "Reason / qualification": str(reason or summary.get("qualification")
                                or "Bounded diagnostic, not task mastery or a population guarantee.")})
    return result


def readiness_outputs(snapshot):
    outputs = [_table(readiness_rows(snapshot), "Readiness Gates", "Diagnostic-only cases never enter demonstrations, replay or the initial Fisher. "
                      "A failed gate is a readiness result, not a failed unassisted-learning comparison.")]
    if _task_only(snapshot):
        return outputs
    confusion, examples = [], []
    for unit in snapshot["units"]:
        summary = unit["summary"]
        if summary["operation"] != "readiness_compliance":
            continue
        audit = [r for r in summary.get("rows", []) if r.get("panel") == "audit"]
        matrix = summary.get("confusion")
        if matrix is None and audit:
            matrix = {expected: {observed: sum(r.get("expected") == expected and r.get("judgment") == observed for r in audit)
                                 for observed in ("adherent", "violated", "uncertain")}
                      for expected in ("adherent", "violated", "uncertain")}
        for expected, counts in (matrix or {}).items():
            confusion.append({"Unit": unit["name"], "Expected (sealed audit)": expected,
                              **{"Judged " + label: counts.get(label) for label in ("adherent", "violated", "uncertain")}})
        selected = [r for r in audit if r.get("correct") is False or r.get("judgment") == "uncertain"][:4]
        for row in selected:
            examples.append({"Audit index": row.get("index"), "Expected": row.get("expected"),
                             "Qwen judgment": row.get("judgment"), "Reason": row.get("reason"),
                             "Issued instructions": [c.get("text") for c in row.get("evidence", {}).get("instructions", [])],
                             "Before / after camera refs": [row.get("before_ref"), row.get("after_ref")]})
    if confusion:
        outputs.append(_table(confusion, "Compliance Confusion", "Sealed audit panel only; prompt-development examples are not included. "
                              "Uncertainty remains its own category and is not silently counted as correct on clear cases."))
    if examples:
        outputs.append(_table(examples, "Compliance Audit Examples", "First four audit errors/uncertainties in saved order; no new inference or relabelling."))
    outputs += readiness_image_outputs(snapshot)
    return outputs


def readiness_image_outputs(snapshot):
    """Two saved audit examples; no model inference or checkpoint loading."""
    if _task_only(snapshot):
        return []
    from matplotlib.figure import Figure
    from numerical_experiment.picar.storage import FrameStore
    import textwrap

    outputs = []
    for unit in snapshot["units"]:
        if unit["summary"]["operation"] != "readiness_compliance" or not (Path(unit["path"]) / "frames").is_dir():
            continue
        audit = [r for r in unit["summary"].get("rows", []) if r.get("panel") == "audit"]
        selected = []
        for correct in (False, True):
            row = next((r for r in audit if r.get("correct") is correct and r.get("before_ref") and r.get("after_ref")), None)
            if row:
                selected.append(row)
        if not selected:
            continue
        frames = FrameStore(Path(unit["path"]) / "frames", readonly=True)
        figure = Figure(figsize=(8, 3.0 * len(selected)), layout="constrained")
        axes = figure.subplots(len(selected), 2, squeeze=False)
        for index, row in enumerate(selected):
            for axis, ref, title in zip(axes[index], (row["before_ref"], row["after_ref"]), ("Before", "After")):
                axis.imshow(frames.get(ref))
                axis.set(xticks=[], yticks=[], title=f"{title}: audit window {row['index']}")
            instruction = "; ".join(c["text"] for c in row["evidence"]["instructions"])
            controls = row["evidence"].get("controls", [])
            actual = controls[0]["executed"] if controls else None
            telemetry = f"Steering {math.degrees(actual[2]):.1f} degrees; drive {actual[3]:+.2f} m/s" if actual else "No eligible policy response"
            axes[index, 0].set_xlabel(textwrap.fill(instruction, 48) + "\n" + telemetry, fontsize=8)
            axes[index, 1].set_xlabel(f"Expected: {row['expected']}; Qwen: {row['judgment']}\n" +
                                      textwrap.fill(row.get("reason", ""), 48), fontsize=8)
        outputs.append(_figure(figure, "Saved audit images: first incorrect and first correct window. "
                               "Instruction, actual controls and judgments are recorded evidence; no new evaluation."))
    return outputs


def pilot_rows(snapshot):
    rows = []
    for name in ("coaching-fix6-v1", "coaching-fix6-v2", "coaching-fix6-v3"):
        root = Path(snapshot["root"]).parent / name
        summaries = [_json(path) for path in (root / "units").glob("*/summary.json")
                     if (path.parent / "COMPLETED").is_file()]
        by_operation = {s["operation"]: s for s in summaries}
        compliance = by_operation.get("readiness_compliance", {})
        teacher = by_operation.get("readiness_teacher", {})
        if not compliance:
            continue
        rows.append({"Pilot": name.rsplit("-", 1)[-1],
            "Clear audit correct": f"{compliance.get('audit_clear_correct')}/{compliance.get('audit_clear_total')}",
            "Correct coached left/right turns": f"{teacher.get('correct_left_turns')}/{teacher.get('correct_right_turns')}",
            "Median teacher seconds": by_operation.get("resource_profile", {}).get("median_teacher_seconds"),
            "Teacher gate": teacher.get("ready"), "Compliance gate": compliance.get("ready")})
    return rows


def settings_rows(snapshot):
    rows = []
    for condition in ("U", "G"):
        candidates = [u["summary"]["settings"] for u in snapshot["units"]
                      if u["summary"]["settings"].get("condition") == condition]
        recipes = {json.dumps(c.get("recipe", {}), sort_keys=True) for c in candidates if c.get("recipe")}
        rows.append({"Condition": condition, "Learner": "FD + TD; rank-64 EWC + residual diagonal; STM rank 1",
                     "Initialization guidance": "0%" if condition == "U" else "50% availability",
                     "Online guidance": "None" if condition == "U" else "50% to zero by n/N=.75",
                     "Assessment": "Unassisted; original task reward",
                     "Recipe": [json.loads(r) for r in sorted(recipes)] or None})
        if _task_only(snapshot):
            rows[-1].update({"Training reward": "Task-only; identical task reward for U and G",
                             "Adherence": "disabled by design",
                             "Compliance bonus (raw)": 0,
                             "Treatment": "Coached initialization and tapered text cues"})
    return rows


def _mean(values):
    return math.fsum(values) / len(values) if values and all(_number(v) for v in values) else None


def _trajectories(snapshot):
    return [u for u in snapshot["units"] if u["summary"]["operation"] == "trajectory"]


def _task(episode):
    return episode.get("task_return", episode.get("raw_return"))


def evaluation_rows(snapshot):
    """Per-trained-agent common-four curves, never episode-weighted seed means."""
    rows = []
    for unit in _trajectories(snapshot):
        summary, settings = unit["summary"], unit["summary"]["settings"]
        count = settings.get("curve_layouts", 4)
        _require(type(count) is int and count > 0, "curve_layouts must be positive")
        for panel in sorted(summary.get("evaluations", []), key=lambda p: (p["environment"], p["n"])):
            episodes = panel["episodes"][:count]
            full = len(episodes) == count
            rows.append({"Condition": settings["condition"], "Cohort": settings["phase"], "Seed": settings["seed"],
                         "Unit": unit["name"], "Environment": panel["environment"], "n": panel["n"],
                         "Actual acquisitions": panel.get("acquired_count"), "Layouts": len(episodes),
                         "task_return": _mean([_task(e) for e in episodes]) if full else None,
                         "success": _mean([float(e["success"]) for e in episodes]) if full else None})
    return rows


def _panel(unit, environment, n):
    panels = [p for p in unit["summary"].get("evaluations", []) if p["n"] == n and p["environment"] == environment]
    return panels[0] if len(panels) == 1 else None


def _layout_ids(episodes):
    ids = [e.get("seed") for e in episodes]
    return tuple(ids) if all(_count(seed) for seed in ids) and len(set(ids)) == len(ids) else None


def _auc(unit, environment):
    settings = unit["summary"]["settings"]
    count, total = settings.get("curve_layouts", 4), settings["total"]
    checkpoints = settings.get("evaluation_checkpoints", [0, total // 4, total // 2, 3 * total // 4, total])
    if not isinstance(checkpoints, list) or len(set(checkpoints)) != len(checkpoints):
        return None
    points = []
    for n in sorted(checkpoints):
        panel = _panel(unit, environment, n)
        if panel is None or len(panel["episodes"]) < count:
            return None
        episodes = panel["episodes"][:count]
        value = _mean([_task(e) for e in episodes])
        identity = _layout_ids(episodes)
        if value is None or identity is None:
            return None
        points.append((n, value, identity))
    if not points or points[0][0] != 0 or points[-1][0] != total or len({p[2] for p in points}) != 1:
        return None
    return math.fsum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:])) / total


def paired_endpoint_rows(snapshot):
    """Exact phase/seed/budget/layout pairs, with per-stream effects first."""
    indexed = defaultdict(list)
    for unit in _trajectories(snapshot):
        settings = unit["summary"]["settings"]
        indexed[settings["phase"], settings["seed"], settings["condition"]].append(unit)
    rows = []
    for phase, seed in sorted({(p, s) for p, s, _ in indexed}):
        if not all(len(indexed[phase, seed, c]) == 1 for c in ("U", "G")):
            continue
        left, right = (indexed[phase, seed, c][0] for c in ("U", "G"))
        su, sg = left["summary"]["settings"], right["summary"]["settings"]
        if (su["total"] != sg["total"] or su.get("recipe", {}) != sg.get("recipe", {})
                or su.get("replica") != sg.get("replica")):
            continue
        total = su["total"]
        count = su.get("endpoint_layouts", 8)
        if type(count) is not int or count < 1 or count != sg.get("endpoint_layouts", 8):
            continue
        for environment in ENVIRONMENTS:
            panels = [_panel(u, environment, total) for u in (left, right)]
            if any(p is None or len(p["episodes"]) < count for p in panels):
                continue
            episodes = [p["episodes"][:count] for p in panels]
            identities = [_layout_ids(es) for es in episodes]
            if identities[0] is None or identities[0] != identities[1]:
                continue
            row = {"Cohort": phase, "Seed": seed, "Environment": environment, "Budget": total,
                   "Recipe": su.get("recipe", {})}
            returns, aucs, changes = [], [], []
            for condition, unit, panel, es in zip(("U", "G"), (left, right), panels, episodes):
                value = _mean([_task(e) for e in es])
                returns.append(value)
                aucs.append(_auc(unit, environment))
                baseline = _panel(unit, "initial_A", 0)
                final_initial = _panel(unit, "initial_A", total)
                common = unit["summary"]["settings"].get("curve_layouts", 4)
                change = None
                if baseline and final_initial and min(len(baseline["episodes"]), len(final_initial["episodes"])) >= common:
                    before, after = baseline["episodes"][:common], final_initial["episodes"][:common]
                    b, a = _mean([_task(e) for e in before]), _mean([_task(e) for e in after])
                    if b is not None and a is not None and _layout_ids(before) is not None and _layout_ids(before) == _layout_ids(after):
                        change = a - b
                changes.append(change)
                row.update({condition + " task return": value,
                            condition + " successes/episodes": f"{sum(e['success'] for e in es)}/{count}",
                            condition + " collisions": sum(e.get("end_reason") == "collision" for e in es)
                            if all("end_reason" in e for e in es) else None,
                            condition + " path seconds": unit["summary"].get("active_seconds"),
                            condition + " actual acquisitions": panel.get("acquired_count"),
                            condition + " budget-averaged task return": aucs[-1],
                            condition + " initial-A change (common layouts)": change})
            row["G-U task return"] = returns[1] - returns[0] if all(_number(v) for v in returns) else None
            row["G-U success (pp)"] = 100 * (sum(e["success"] for e in episodes[1]) - sum(e["success"] for e in episodes[0])) / count
            row["G-U budget-averaged return"] = aucs[1] - aucs[0] if all(_number(v) for v in aucs) else None
            rows.append(row)
    return rows


def paired_mean_rows(snapshot):
    groups = defaultdict(list)
    for row in paired_endpoint_rows(snapshot):
        groups[row["Cohort"], row["Environment"], json.dumps(row["Recipe"], sort_keys=True)].append(row)
    return [{"Cohort": phase, "Environment": environment, "Recipe": json.loads(recipe), "Pairs": len(rows),
             **{metric: _mean([r[metric] for r in rows]) for metric in
                ("G-U task return", "G-U success (pp)", "G-U budget-averaged return")},
             "Return range (not CI)": [min(values), max(values)] if (values := [r["G-U task return"] for r in rows
                                                                             if _number(r["G-U task return"])]) else None}
            for (phase, environment, recipe), rows in sorted(groups.items())]


def _figure(figure, description):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buffer = io.BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": description}}


def schedule_outputs():
    from matplotlib.figure import Figure

    def smooth(z):
        z = min(1, max(0, z))
        return 6 * z**5 - 15 * z**4 + 10 * z**3

    xs = [i / 200 for i in range(201)]
    figure = Figure(figsize=(9, 3.2), layout="constrained")
    axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
    axes[0].plot([0, 1], [.5, .5], color="#bc4a55", label="G: 50%")
    axes[0].plot([0, 1], [0, 0], color="#147d83", label="U: 0%")
    axes[0].set(title="Stationary initial A", xlabel="Collection (physical-time gates)", ylim=(-.03, .6), xticks=[])
    axes[0].legend(fontsize=8)
    axis = axes[1]
    axis.plot(xs, [.5 * (1 - smooth((x - .125) / .625)) for x in xs], color="#bc4a55", label="G availability")
    axis.plot(xs, [0] * len(xs), color="#147d83", label="U availability")
    axis.axvspan(.75, 1, color="#dcebdc", label="Fresh guidance off")
    axis.set(xlabel="Newly captured online intervals n / N", ylabel="Enabled-window probability", ylim=(-.03, .6))
    twin = axis.twinx()
    twin.plot(xs, [.5**smooth((x - .125) / .75) for x in xs], color="#555555", linestyle="--", label="Barrier sigma (m)")
    twin.set(ylabel="Barrier sigma (m)", ylim=(.45, 1.05))
    axis.legend(loc="upper right", fontsize=7)
    return [_figure(figure, "Acquisition clock: N=1024 online captures. G availability fades from .5 to zero at .75N; "
                    "barrier sigma tightens 1.0 to .5 m by .875N. Physical-time gate: one draw per frozen four-second "
                    "window (or pilot-selected six seconds); resets each episode. Availability is not active cue coverage. "
                    "All held-out assessments are unassisted. Final means moderate A, never category B.")]


def _learning_outputs(snapshot):
    from matplotlib.figure import Figure

    rows = evaluation_rows(snapshot)
    outputs = []
    for phase in ("development", "assessment"):
        selected = [r for r in rows if r["Cohort"] == phase]
        if not selected:
            continue
        figure = Figure(figsize=(9, 5.5), layout="constrained")
        axes = figure.subplots(2, 2)
        for i, environment in enumerate(ENVIRONMENTS):
            for j, metric in enumerate(("task_return", "success")):
                axis = axes[i, j]
                for unit in _trajectories(snapshot):
                    settings = unit["summary"]["settings"]
                    points = [r for r in selected if r["Unit"] == unit["name"] and r["Environment"] == environment]
                    if not points:
                        continue
                    axis.plot([r["n"] for r in points], [r[metric] if r[metric] is not None else math.nan for r in points],
                              marker="o", markersize=3, linewidth=1,
                              label=f"{settings['condition']} seed {settings['seed']}",
                              color="#147d83" if settings["condition"] == "U" else "#bc4a55")
                    axis.axvline(.75 * settings["total"], color="gray", linestyle=":", linewidth=.7)
                axis.set(title=environment + (" task return" if j == 0 else " success"), xlabel="Online acquisitions")
                if j:
                    axis.set_ylim(-.05, 1.05)
                axis.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7, ncol=2)
        outputs.append(_figure(figure, phase.title() + ": original unassisted task outcomes, common first four layouts. "
                               "Each line is one trained stream, not independent checkpoints. Dotted line: fresh guidance off. "
                               "Full-eight endpoint denominators differ from curve denominators; actual acquisitions include overshoot."))
    return outputs


def _initialization_outputs(snapshot):
    from matplotlib.figure import Figure

    outputs, coverage, fitting = [], [], []
    for unit in snapshot["units"]:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        if summary["operation"] == "collection":
            episodes = summary.get("episodes", [])
            actions = summary.get("action_metrics", [])
            turning = [r for r in actions if _number(r.get("time_above_turn_threshold_s"))]
            coverage.append({"Condition": settings.get("condition"), "Captured intervals": summary.get("count", summary.get("n")),
                             "Usable histories": summary.get("records"),
                             "Episodes": len(episodes), "Successes": sum(e["success"] for e in episodes),
                             "Turn episodes / measured": f"{sum(r['time_above_turn_threshold_s'] > 0 for r in turning)}/{len(turning)}"
                             if turning else None, "Seconds": summary.get("active_seconds")})
        if summary["operation"] in ("distillation", "initial_fit", "features"):
            fitting.append({"Condition": settings.get("condition"), "Operation": summary["operation"],
                            "Train histories": summary.get("train_count"), "Validation histories": summary.get("validation_count"),
                            "Selected epoch": summary.get("selected_epoch"), "Fitting exposures": summary.get("exposures"),
                            "Neutral STM": summary.get("stm_neutral"), "Trainable inventory": summary.get("inventory")})
        curves = summary.get("curves", []) if summary["operation"] == "distillation" else []
        if curves:
            figure = Figure(figsize=(9, 2.8), layout="constrained")
            axes = figure.subplots(1, 3)
            for axis, keys, title in zip(axes, (("train_kl", "validation_kl"), ("train_turn_kl", "validation_turn_kl"),
                                               ("train_turn_agreement", "validation_turn_agreement")),
                                         ("All-example KL", "Turn-example KL", "Turn action agreement")):
                for key in keys:
                    axis.plot([r["epoch"] for r in curves], [r.get(key) if r.get(key) is not None else math.nan for r in curves], label=key)
                selected = summary.get("selected_epoch", summary.get("best_epoch"))
                if _number(selected):
                    axis.axvline(selected, linestyle=":", color="gray")
                axis.set(title=title, xlabel="Epoch (including zero)")
                axis.legend(fontsize=6)
            outputs.append(_figure(figure, f"Condition {settings.get('condition')}: recorded distillation curves; "
                                    "turn examples are shown separately. Missing values stay unavailable. "
                                    "Selection uses whole-episode validation, not the last fitted weights."))
    if fitting:
        outputs.append(_table(fitting, "Initialization Fit", "Whole-episode training/validation split. Condition-specific fitted representations "
                              "and Fishers are intentionally different; accepted initial assets are reused across later streams."))
    return [_table(coverage, "Initialization Coverage", "Historical Fix 5A: 628/628 dominant actions were forward; zero commanded steering. "
                   "That historical dataset is motivation, not a new matched U control. Turn episodes contain executed steering above five degrees "
                   "while moving above 0.04 m/s; successes are collection outcomes, not held-out results. Captures are dependent, not iid histories.")] + outputs


def _estimator_outputs(snapshot):
    """One shared figure, keeping fresh and development streams separate."""
    from matplotlib.figure import Figure

    trajectories = _trajectories(snapshot)
    phase = "assessment" if any(u["summary"]["settings"]["phase"] == "assessment" for u in trajectories) else "development"
    units = [u for u in trajectories if u["summary"]["settings"]["phase"] == phase and u["summary"].get("metrics")]
    if not units:
        return []
    seeds = sorted({u["summary"]["settings"]["seed"] for u in units})
    styles = ("-", "--", "-.", ":")
    figure = Figure(figsize=(9, 2.9), layout="constrained")
    axes = figure.subplots(1, 3)
    for axis, key, title, ylabel in zip(axes, ("post_step_ewc", "fd", "age_mean"),
                                      ("Post-fit EWC", "First-step FD (weighted)", "Replay age / capacity"),
                                      ("Penalty", "Loss contribution", "Mean age / replay capacity")):
        for unit in units:
            settings = unit["summary"]["settings"]
            metrics = sorted(unit["summary"]["metrics"], key=lambda r: r["n"])
            if not any(r.get(key) is not None for r in metrics):
                continue
            axis.plot([r["n"] for r in metrics], [r.get(key) if r.get(key) is not None else math.nan for r in metrics],
                      label=f"{settings['condition']} seed {settings['seed']}", linewidth=1,
                      color="#147d83" if settings["condition"] == "U" else "#bc4a55",
                      linestyle=styles[seeds.index(settings["seed"]) % len(styles)])
        axis.set(title=title, xlabel="Online acquisitions", ylabel=ylabel)
        axis.grid(alpha=.2)
        if not axis.lines:
            axis.text(.5, .5, "Not recorded", ha="center", va="center", transform=axis.transAxes)
    handles = {}
    for axis in axes:
        for line in axis.lines:
            handles.setdefault(line.get_label(), line)
    if handles:
        figure.legend(handles.values(), handles.keys(), loc="outside upper center", ncol=min(4, len(handles)), fontsize=7)
    return [_figure(figure, ("Fresh assessment" if phase == "assessment" else "Development (until fresh assessment appears)")
                    + ": one line per condition/seed, no development/fresh pooling. "
                    "EWC uses saved post_step_ewc, not first-step zeros. FD uses fd: strength-weighted first-step contribution, "
                    "not a post-fit value. Replay age uses age_mean in age/capacity units, not acquisition counts. "
                    "Missing diagnostics stay unavailable. Own-target TD/NLL is not a common navigation metric.")]


def diagnostic_outputs(snapshot):
    """Compact per-stream binned behavior and component accounting."""
    outputs, guidance, health, costs = [], [], [], []
    task_only = _task_only(snapshot)
    reward_rows = []
    for unit in snapshot["units"]:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        identity = {"Condition": settings.get("condition"), "Cohort": settings.get("phase", "readiness"), "Seed": settings.get("seed")}
        if isinstance(summary.get("coaching_metrics"), list):
            coaching = summary["coaching_metrics"]
            totals = {key: sum(r[key] for r in coaching) if coaching and all(_number(r.get(key)) for r in coaching) else None
                      for key in ("windows", "enabled", "cues", "assessable", "scored", "abstentions", "active_instruction_seconds")}
            if task_only:
                totals = {k: v for k, v in totals.items() if k not in ("assessable", "scored")}
                totals["Adherence"] = "disabled by design"
            windows, enabled, acquired = totals["windows"], totals["enabled"], summary.get("n")
            guidance.append({**identity, **totals,
                             "Enabled window fraction": enabled / windows if _number(enabled) and windows else None,
                             "Cues per acquired interval": totals["cues"] / acquired
                             if _number(totals["cues"]) and _number(acquired) and acquired > 0 else None})
        elif summary.get("coaching_metrics") is not None:
            guidance.append({**identity, "Instruction coverage" if task_only else "Instruction coverage / adherence": summary["coaching_metrics"]})
        if summary.get("archive") is not None or summary["operation"] == "initial_archive":
            health.append({**identity, "Operation": summary["operation"], "Archive": summary.get("archive", summary.get("diagnostics")),
                           "Allocation": summary.get("allocation")})
        if summary.get("timings") is not None or summary.get("active_seconds") is not None:
            costs.append({**identity, "Operation": summary["operation"], "Seconds": summary.get("active_seconds"),
                          "Components": summary.get("timings")})
        if summary["operation"] != "trajectory":
            continue
        total = settings["total"]
        for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, math.inf)):
            episodes = [e for e in summary.get("episodes", []) if _number(e.get("n")) and lo * total <= e["n"] < hi * total]
            if episodes:
                reward_rows.append({**identity, "Acquisition bin": f"{int(lo * total)}-{int(hi * total) if math.isfinite(hi) else 'end'}",
                                    "Episodes": len(episodes), "Task raw return": _mean([e.get("task_return") for e in episodes]),
                                    "Guidance raw return": _mean([e.get("guidance_return") for e in episodes]),
                                    "Total raw return": _mean([e.get("raw_return") for e in episodes]),
                                    "Total encoded return": _mean([e.get("coded_return") for e in episodes])})
    outputs += _estimator_outputs(snapshot)
    behavior = action_rows(snapshot)
    coaching_note = ("Report enabled windows, active time, delivered cues and geometric-coach abstentions. Adherence: disabled by design; "
                     "no judgments or scored windows are expected. Missing cue diagnostics are unavailable, not zero."
                     if task_only else "Report enabled windows, active time, cues, abstentions, unassessable windows, "
                     "judgment errors and adherence separately. Missing diagnostics are unavailable, not zero.")
    reward_note = ("Whole-episode returns, binned by episode-start acquisitions. U and G use identical task-only reward; "
                   "compliance bonus is zero. Coaching is the treatment, not a reward change."
                   if task_only else "Whole-episode returns, binned by episode-start acquisitions. Detours need not "
                   "improve the next image score. Guidance is +/-2 raw, +/-0.2 encoded, paid once per assessed window.")
    cost_note = ("Initialization collection/fitting and text coaching are additional costs; adherence evaluation is disabled by design. "
                 if task_only else "Initialization collection/fitting and compliance scoring are additional costs; ")
    return ([_table(guidance, "Coaching And Readiness", coaching_note),
             _table(reward_rows, "Episode Reward Parts", reward_note),
             _table(behavior, "Action Coverage", "Means across complete episode histories within each bin; missing values are not zero. "
                    "Commanded steering during motion is in degrees, turn duration in seconds, bouts per episode; "
                    "measured yaw change is distinct from commanded steering. Histories survive replay eviction."),
             _table(health, "Fisher Health", "Rank-64 correlations plus residual diagonal; allocation, PSD and compression are numerical diagnostics."),
             _table(costs, "Supervision And Compute", cost_note +
                    "instructions are supervision, not extra environment observations. Shared initialization is counted once, not once per stream.")]
            + outputs)


def action_rows(snapshot):
    result = []
    for unit in _trajectories(snapshot):
        settings = unit["summary"]["settings"]
        groups = defaultdict(list)
        for row in unit["summary"].get("action_metrics", []):
            if row.get("complete_history", True) is not True:
                continue
            difficulty = difficulty_group(row, None)
            groups[difficulty or "outside declared windows"].append(row)
        for difficulty, rows in sorted(groups.items()):
            result.append({"Condition": settings["condition"], "Cohort": settings["phase"], "Seed": settings["seed"],
                           "Difficulty bin": difficulty, "Complete episodes": len(rows),
                           **{key: _mean([r.get(key) for r in rows]) for key in
                              ("mean_abs_steering_deg", "moving_seconds", "time_above_turn_threshold_s", "turn_bouts", "measured_yaw_change_deg")}})
    return result


def _difficulty_value(episode):
    value = episode.get("difficulty_start", episode.get("start_difficulty", episode.get("difficulty")))
    if value is None:
        value = episode.get("schedule", {}).get("difficulty")
    curriculum = episode.get("layout", {}).get("curriculum", {})
    if value is None:
        value = curriculum.get("difficulty", curriculum.get("d"))
    if value is None and _number(episode.get("sigma_m")) and episode["sigma_m"] > 0:
        value = math.log(episode["sigma_m"]) / math.log(.5)
    return value


def difficulty_group(episode, panel):
    """Scene difficulty, never the model checkpoint; no extra held-out panels."""
    if panel is not None:
        return {"initial_A": "initial", "moderate_A": "final"}.get(panel.get("environment"))
    value = _difficulty_value(episode)
    if not _number(value):
        return None
    for lo, hi, name in ((0, .1, "initial"), (.4, .6, "medium"), (.9, 1, "final")):
        if lo <= value <= hi:
            return name
    return None


def _clips(snapshot):
    result = []
    for unit in snapshot["units"]:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        if settings.get("condition") not in ("U", "G"):
            continue
        sources = [(None, e) for e in summary.get("episodes", [])]
        sources += [(panel, e) for panel in summary.get("evaluations", []) for e in panel["episodes"]]
        seen = set()
        for index, (panel, episode) in enumerate(sources):
            provenance = "held-out" if panel is not None else "collection" if summary["operation"] == "collection" else "training"
            checkpoint = panel["n"] if panel is not None else episode.get("n")
            episode_id = episode.get("episode_id", str(index))
            identity = provenance, checkpoint, episode_id
            _require(identity not in seen, "duplicate episode identity within unit")
            seen.add(identity)
            phase = settings["phase"]
            order = (settings.get("run_order", 0), {"initialization": 0, "development": 1, "assessment": 2}.get(phase, 3),
                     settings.get("seed", -1), unit["name"], index)
            result.append({"unit": unit, "settings": settings, "episode": episode, "panel": panel,
                           "provenance": provenance, "checkpoint": checkpoint, "episode_id": episode_id,
                           "difficulty": difficulty_group(episode, panel), "order": order})
    return sorted(result, key=lambda r: r["order"])


def _links(clip, directory):
    links = []
    for video in clip["episode"].get("videos", []):
        path = Path(video["path"])
        path = path if path.is_absolute() else Path(clip["unit"]["path"]) / path
        if path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)):
            href = quote(os.path.relpath(path, directory), safe="/.")
            links.append('<a href="' + html.escape(href, quote=True) + '">camera + top-down</a>')
    return links


def _clip_label(clip):
    episode, settings, panel = clip["episode"], clip["settings"], clip["panel"]
    blocked = episode.get("layout", {}).get("curriculum", {}).get("direct_path_blocked", episode.get("direct_path_blocked"))
    barrier = "blocked" if blocked is True else "open" if blocked is False else "unknown"
    start = _difficulty_value(episode)
    end = episode.get("difficulty_end", episode.get("end_difficulty"))
    if panel is not None:
        start = end = 0 if panel["environment"] == "initial_A" else 1
    guidance = "off (unassisted)" if panel is not None or settings["condition"] == "U" else episode.get("guidance_status", "see instruction timeline")
    denominator = (f"; same-cohort evaluated successes/episodes={sum(e['success'] for e in panel['episodes'])}/{len(panel['episodes'])}"
                   if panel is not None else "")
    return (f"{clip['provenance']}; cohort={settings['phase']}; stream seed={settings.get('seed')}; "
            f"layout seed={episode.get('seed')}; episode={clip['episode_id']}; learning checkpoint={clip['checkpoint']}; "
            f"actual difficulty={_display(start)} to {_display(end)}; guidance={guidance}; direct path={barrier}{denominator}")


def recording_outputs(snapshot, notebook_directory):
    """Rendered HTML, campaign-wide first-three episodes, not first-three files."""
    clips = _clips(snapshot)
    preselected = [c for c in clips if c["episode"].get("preselected") is True
                   or any(v.get("selection") == "preselected" for v in c["episode"].get("videos", []))]
    esc = html.escape
    common = "<h4>Common Preselected Recordings</h4><ul>"
    for clip in preselected:
        common += "<li>" + esc(_clip_label(clip)) + ": " + ("; ".join(_links(clip, notebook_directory)) or "Missing recording") + "</li>"
    common += "</ul>" + ("" if preselected else "<p>Pending or no common preselected recordings saved.</p>")
    gallery = ("<h4>Outcome-Selected Gallery</h4><p>Target: three distinct successes and three failures per condition and difficulty, "
               "campaign-wide. Stable order: declared run_order, initialization/development/assessment, stream seed, unit name, "
               "saved episode order. Select first eligible episodes even when their recording is missing. "
               "Balanced examples do not estimate a success rate; medium clips come only from training.</p>")
    plain = []
    for condition in ("U", "G"):
        gallery += f"<details><summary>Condition {condition}</summary>"
        for difficulty in DIFFICULTIES:
            gallery += f"<details><summary>{difficulty}</summary>"
            for success, label in ((True, "success"), (False, "failure")):
                candidates = [c for c in clips if c["settings"]["condition"] == condition and c["difficulty"] == difficulty
                              and c["episode"]["success"] is success
                              and (difficulty != "medium" or c["provenance"] == "training")]
                selected = candidates[:3]
                recorded = sum(bool(_links(c, notebook_directory)) for c in selected)
                text = f"{recorded}/{len(selected)} selected recorded; {len(candidates)} available outcomes; target 3; shortfall {3 - recorded}"
                gallery += f"<details><summary>{label}: {esc(text)}</summary>"
                if not candidates:
                    gallery += "<p>" + ("No successes observed." if success else "No failures observed.")
                    gallery += "</p>"
                if not snapshot["complete"]:
                    gallery += "<p>Pending runs: coverage is incomplete.</p>"
                gallery += "<ul>"
                for clip in selected:
                    gallery += "<li>" + esc(_clip_label(clip)) + ": " + ("; ".join(_links(clip, notebook_directory)) or "Missing recording (original evidence may still be retained)") + "</li>"
                gallery += "</ul></details>"
                plain.append(f"{condition}/{difficulty}/{label}: {text}")
            gallery += "</details>"
        gallery += "</details>"
    return [_html(common + gallery, "Common preselected clips are separate from outcome-selected examples.\n" + "\n".join(plain))]


def conclusion(snapshot):
    task_only = _task_only(snapshot)
    text = "### Fix 6 Task-Only Conclusion\n\n" if task_only else "### Fix 6 Conclusion\n\n"
    if task_only:
        text += ("U and G use identical task-only reward. Adherence is disabled by design: no Qwen compliance evaluation, "
                 "bonus or readiness gate. Coached initialization and tapered text cues are the treatment; "
                 "cue logs and video overlays remain available. Earlier failed adherence pilots are historical, "
                 "not learning comparisons or admission gates for this revision. ")
    failed = [r for r in readiness_rows(snapshot) if r["Status"] == "Failed"]
    if snapshot.get("progress", {}).get("status") == "readiness_failed" and not _trajectories(snapshot):
        if task_only:
            return (text + "**Stopped at readiness, before initialization or learner training.** "
                    + "; ".join(r["Gate"] + ": " + r["Reason / qualification"] for r in failed)
                    + ". No sealed online-learning trajectories are available. No coaching benefit, sample savings "
                    "or S statistic has been measured.")
        rows = readiness_rows(snapshot)
        compliance = next((r for r in rows if r["Gate"] == "Qwen compliance"), None)
        evidence = compliance["Evidence"] if compliance else {}
        passed = [r["Gate"] for r in rows if r["Status"] == "Passed"]
        passed_text = "Passed checks: " + ", ".join(passed) + ". " if passed else ""
        return (text + "**Stopped at readiness, before initialization or learner training.** "
            f"The adherence judge got {evidence.get('clear_correct')}/{evidence.get('clear_total')} clear audit cases correct "
            "against the 90% target. " + passed_text +
            "Compliance judgments remained brittle under prompt changes. All pilot sources, images and outputs are preserved.\n\n"
            "No sealed online-learning trajectories are available. No unassisted learning comparison, sample savings, or S statistic has been measured. "
            "The coaching hypothesis remains untested. The next required repair is reliable adherence evaluation; "
            "the planned U/G initialization and learning scope remains pending. " +
            ("The GPU is released." if snapshot.get("progress", {}).get("node_free") else ""))
    if not snapshot["complete"]:
        text += "Provisional / incomplete: pending work and unsealed previews are not final evidence. "
    if failed:
        text += "Readiness failed: " + "; ".join(r["Reason / qualification"] for r in failed) + " "
        if not _trajectories(snapshot):
            text += "No sealed online-learning trajectories are available, so no coaching benefit or failure is inferred. "
    pairs = paired_mean_rows(snapshot)
    assessment = [r for r in pairs if r["Cohort"] == "assessment"]
    if assessment:
        text += "Fresh-stream paired unassisted differences (G minus U): "
        text += "; ".join(f"{r['Environment']} task return {_display(r['G-U task return'])}, success {_display(r['G-U success (pp)'])} pp, {r['Pairs']} pairs"
                          for r in assessment) + ". "
    else:
        text += "No matched fresh-stream endpoints yet. "
    if task_only:
        if not _trajectories(snapshot):
            text += "No sealed online-learning trajectories are available; the revised coaching hypothesis is untested. "
        return text + ("Initialization and online coaching are a combined treatment, not isolated effects. "
                       "Assess unassisted learning and initial-A retention separately; include initialization and coaching costs. "
                       "Shared initial assets are not independent full-pipeline replicas. "
                       "No sample savings or measured compute speedup is established here.")
    return text + ("Guided execution, turn distillation, persistent unassisted learning and absolute initial-A retention are separate questions. "
                   "Initial capability is part of the treatment, not proof of faster subsequent learning. "
                   "U and G intentionally use different fitted initializations/Fishers, reused across streams; these are not independent full-pipeline replicas. "
                   "Two fresh stream pairs give descriptive ranges, not 95% confidence intervals. "
                   "Include initial collection/fitting, instructions, scored windows and compliance compute in cost comparisons. "
                   "No samples-saved estimate is available without an observed common capability crossing and interpretable retention.")


def result_outputs(snapshot, notebook_directory=None):
    outputs = readiness_outputs(snapshot)
    if _task_only(snapshot):
        pilots = pilot_rows(snapshot)
        if pilots:
            outputs += [_table(pilots, "Preserved Readiness Pilots",
                "V1-V3 failed-pilot history is preserved, not pooled with task-only learning results. "
                "Their compliance gates do not apply to this revision.")]
    if snapshot.get("progress", {}).get("status") == "readiness_failed" and not _trajectories(snapshot):
        if not _task_only(snapshot):
            outputs += [_table(pilot_rows(snapshot), "Preserved Readiness Pilots",
                "Different prompt versions and fresh audit inputs, not independent learning replicas. V2's reverse and teacher-right "
                "development examples motivated V3; its audit labels did not select the refinement. No learner trained in any pilot.")]
        return outputs
    outputs += [_table(settings_rows(snapshot), "Two Conditions", "Same FD+TD recipe and acquisition budget; coached initialization is part of G's treatment.")]
    outputs += schedule_outputs() + _initialization_outputs(snapshot) + _learning_outputs(snapshot)
    outputs += [_table(paired_endpoint_rows(snapshot), "Paired Endpoints: Development / Fresh Assessment",
                       "Per-seed effects first. Full eight held-out layouts, exact matching budgets/layout seeds; missing partners are unavailable. "
                       "Budget-averaged return uses trapezoids over all five common-four checkpoints, never extrapolation."),
                _table(paired_mean_rows(snapshot), "Equal-Weight Paired Summary", "Development and fresh assessment remain separate. Ranges are descriptive, not CIs.")]
    outputs += recording_outputs(snapshot, notebook_directory or Path.cwd()) + diagnostic_outputs(snapshot)
    return outputs


def display_progress(snapshot):
    from IPython.display import display

    for output in progress_outputs(snapshot):
        display(output["data"], raw=True)


def display_results(snapshot, notebook_directory=None):
    from IPython.display import display

    for output in result_outputs(snapshot, notebook_directory):
        display(output["data"], raw=True)


def build_notebook(root, output, base_notebook=None):
    """Append Fix 6 to saved history, refreshing only this builder's own cells."""
    import nbformat

    root, output = Path(root), Path(output)
    snapshot = load_report(root)
    _require(output.suffix == ".ipynb" and not any(p.is_symlink() for p in (output, *output.parents)),
             "output must be a nonsymlink notebook")
    _require(not any((p / "COMPLETED").exists() for p in output.resolve().parents), "output is inside a completed run")
    candidates = [output, Path(base_notebook) if base_notebook is not None else root / "history.ipynb",
                  root / "history.ipynb", DEFAULT_NOTEBOOK]
    base = next((p for p in candidates if p.exists()), None)
    book = _json(base, MAX_HISTORY_BYTES) if base else dict(nbformat.v4.new_notebook())
    _require(book.get("nbformat") == 4 and isinstance(book.get("cells"), list), "base must be a v4 notebook")
    book = copy.deepcopy(book)
    task_only = _task_only(snapshot)
    tag = _TASK_ONLY_TAG if task_only else _TAG
    prefix = "plan19-phase3-fix6-task-only-" if task_only else "plan19-phase3-fix6-"
    history = [c for c in book["cells"] if tag not in c.get("metadata", {})]
    used = {c["id"] for c in history if "id" in c}
    _require(len(used) == sum("id" in c for c in history), "duplicate historical cell ids")

    def own(cell, name):
        identifier, index = prefix + name, 0
        while identifier in used:
            index += 1
            identifier = f"{prefix}{name}-{index}"
        used.add(identifier)
        cell["id"], cell["metadata"][tag] = identifier, True
        return cell

    project = Path(__file__).resolve().parents[2]

    def expression(path):
        path = Path(path).resolve()
        return f"project_root / {str(path.relative_to(project))!r}" if path.is_relative_to(project) else repr(str(path))

    bootstrap = ("from pathlib import Path\nimport sys\n"
                 "for parent in (Path.cwd(), *Path.cwd().parents):\n"
                 "    if (parent / 'numerical_experiment/picar_coaching/report.py').is_file():\n"
                 "        project_root = parent\n"
                 "        if str(parent) not in sys.path:\n            sys.path.insert(0, str(parent))\n"
                 "        break\n"
                 "else:\n    raise RuntimeError('Open from within the project checkout.')\n"
                 "from numerical_experiment.picar_coaching import report as coaching_report\n"
                 f"coaching_snapshot = coaching_report.load_report({expression(root)})\n")
    progress = own(nbformat.v4.new_code_cell(bootstrap + "coaching_report.display_progress(coaching_snapshot)",
                                           outputs=progress_outputs(snapshot)), "progress")
    intro = ("### Phase 3 Fix 6: Tapered Geometric Coaching\n\n"
             "Can coached initialization and tapered text guidance improve later unassisted navigation at matched acquired experience? "
             "Fix 5A's selected initialization had no steering examples. U and G share the FD+TD learner, not fitted initial assets. "
             "Privileged geometry supplies external teaching text only; overlays and top-down views never enter policy inputs.")
    if task_only:
        intro = ("### Phase 3 Fix 6: Task-Reward-Only Coaching (V4)\n\n"
                 "Earlier pilots stopped before learning because Qwen adherence judgments were unreliable. "
                 "This authorized revision tests coached initialization and tapered text cues with identical task-only reward "
                 "for U and G. Adherence is disabled by design: no compliance evaluation, bonus or readiness gate. "
                 "Coaching is the treatment. Text/cue logs, the taper and video overlays remain; "
                 "privileged geometry and top-down views never enter policy inputs. "
                 "The V1-V3 pilot cells and failed-gate outputs are preserved as historical evidence.")
    elif snapshot.get("progress", {}).get("decisions"):
        prior = root.parent / "coaching-fix6-v1/readiness_report.ipynb"
        href = quote(os.path.relpath(prior, output.parent), safe="/.")
        intro += ("\n\nThe [first readiness pilot](" + href + ") is preserved: geometry passed, but Qwen judged every "
                  "compliance example a violation and made no teacher turns; its median 1.66-second response exceeded "
                  "the one-second control deadline. No learner was trained. The revised pilot clarifies action/telemetry "
                  "prompts and tests cue publication, expiry, actual exposure, execution segments and reward delivery "
                  "through fallback intervals before collection. Both conditions receive the same software corrections.")
    appended = [own(nbformat.v4.new_markdown_cell(intro), "intro"),
                own(nbformat.v4.new_code_cell(bootstrap + f"coaching_report.display_results(coaching_snapshot, {expression(output.parent)})",
                                             outputs=result_outputs(snapshot, output.parent.resolve())), "results"),
                own(nbformat.v4.new_markdown_cell(conclusion(snapshot)), "conclusion")]
    position = 1 if history and history[0].get("cell_type") == "markdown" else 0
    book["cells"] = history[:position] + [progress] + history[position:] + appended
    nbformat.validate(copy.deepcopy(book))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".coaching-report-", suffix=".ipynb", dir=output.parent)
    temporary = Path(filename)
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
