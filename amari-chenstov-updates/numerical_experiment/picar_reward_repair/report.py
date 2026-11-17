"""Bounded, scalar-only Plan 19 Phase 3 Fix 3 reporting.

Sealed RunStore envelopes supply results; orchestration progress supplies only
operational counters. No checkpoints, images, model code or historical campaign
scans are needed. Initial panels use ``initial_asset_id`` (or ``asset_id``) to
identify a shared reference, not an independent online replica.
"""

from __future__ import annotations

import copy
import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.phase2_report import PENDING, _mapping, _number, _rows, _table, _note
from numerical_experiment.picar_initialization.report import _committed_progress, _curve, _figure, _finish


VERSION = "plan19-phase3-fix3-v1"
CAMPAIGN = "reward-fix3-v1"
CONDITIONS = ("width6_linear", "width6_signed_square")
MAX_UNITS = 128
MAX_SCAN_DIRS = 2048
MAX_SCALAR_BYTES = 128 * 1024 * 1024
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_CURVES = 64
_TAG = "plan19_phase3_fix3"
_KINDS = {"reward_" + name for name in (
    "dataset", "references", "scoring", "fit", "bootstrap", "initial_evaluation", "readiness",
    "pilot", "campaign", "integrity", "provenance", "software", "audit")}
_CONDITION_KINDS = {"reward_" + name for name in ("fit", "bootstrap", "initial_evaluation", "readiness", "pilot")}
_LABELS = dict(zip(CONDITIONS, ("Linear", "Signed square")))
_COLORS = dict(zip(CONDITIONS, ("#147d92", "#ba4670")))
_INTRO = r"""### Phase 3 Fix 3: Six-Example Visual Rewards

Earlier rewards often distinguished seeing the ball poorly from approaching it.
The frozen six-example width evaluator modestly improved the fresh diagnostic
(reward MAE 1.1875 to 1.0801); this is not learning evidence.
Compare $F(x)=x$ and $F(x)=x|x|$ with
$r^{raw}=\Delta t[.01F(q)-.04]+F(b)$, using Qwen's returned $q$.
Raw success/collision/timeout bonuses are +10/-5/-1 versus +100/-25/-1;
coded returns divide raw returns by 1 versus 10. Larger returns alone do not
establish better behavior. Both recipes use Replay/EWC, three proximal steps,
and **FD off**. Reward-consistent initialization is part of the comparison.

Only sealed results enter task outcomes; committed incomplete-fit diagnostics
are explicitly labeled provisional and establish no accepted checkpoint.
This is paired development, not fresh
significance, retention equivalence or sample-savings evidence. Initial panels
may be shared between online seeds; missing or unscored work remains pending.
The evaluator is imperfect: a zero score is not proof that the ball is absent.
Inspect the saved clips alongside reward curves, especially for small or
partly occluded targets.
"""


def _require(ok, message):
    if not ok:
        raise shared.ArtifactError("Phase 3 Fix 3 schema-incompatible: " + message)


def _count(value):
    return type(value) is int and value >= 0


def _campaign_root(root):
    path = shared.resolve_root(root)
    if path.name.startswith("reward-fix3-"):
        _require(path.name == CAMPAIGN, "wrong campaign; versions must not be pooled")
        return path
    if any((path / name).exists() for name in ("orchestration", "units", "history.ipynb")):
        return path
    return path / "phase3" / CAMPAIGN


def _identity(value):
    shared._version(value.get("schema_version"), 1, "Fix 3 schema")
    _require(type(value.get("phase")) is int and value["phase"] == 3, "phase must be 3")
    if "version" in value:
        _require(value["version"] == VERSION, "wrong source version; versions must not be pooled")


def _condition(summary):
    return _mapping(summary.get("settings")).get("condition")


def _asset(panel, summary):
    return (panel.get("asset_id") or panel.get("source_asset_id") or panel.get("initial_asset_id")
            or _mapping(summary.get("settings")).get("initial_asset_id")
            or _mapping(summary.get("settings")).get("initial_asset"))


def _validate_episode(row, *, evaluation=False):
    _require(isinstance(row, dict), "episode must be an object")
    if evaluation:
        _require(row.get("complete") is True and not row.get("truncated")
                 and row.get("learning") is False and row.get("explore") is False
                 and row.get("reward_scoring") is True and row.get("evaluation") is True,
                 "evaluation must be complete, scored, no-learning and no-explore")
        _require(type(row.get("success")) is bool and _count(row.get("seed")),
                 "evaluation requires seed and Boolean success")
    intervals, scores = row.get("intervals"), row.get("visual_scores")
    _require(isinstance(intervals, list) and bool(intervals), "episode intervals required")
    _require(isinstance(scores, list) and len(scores) == len(intervals), "visual_scores must match intervals")
    for interval, score in zip(intervals, scores):
        _require(isinstance(interval, dict) and _number(interval.get("elapsed_s"))
                 and interval["elapsed_s"] > 0 and isinstance(interval.get("frame_ref"), str)
                 and bool(interval["frame_ref"]), "interval requires elapsed_s and frame_ref")
        _require(isinstance(score, dict) and _number(score.get("score")) and 0 <= score["score"] <= 10,
                 "completed episode requires finite Qwen score in [0,10]; invalid is not zero")
        _require(_number(score.get("width_percent")) and 0 <= score["width_percent"] <= 100,
                 "finite width_percent required")
    for key in ("stage_rewards", "returns", "raw_stage_rewards", "raw_returns", "terminal_bonuses"):
        values = row.get(key)
        _require(isinstance(values, list) and len(values) == len(intervals)
                 and all(_number(v) for v in values), key + " must match intervals with finite values")
    counterfactual = row.get("counterfactual_rewards")
    if counterfactual is not None:
        _require(isinstance(counterfactual, dict) and set(counterfactual) == set(CONDITIONS),
                 "counterfactual arithmetic must identify both reward conditions")
        for values in counterfactual.values():
            _require(isinstance(values, dict), "counterfactual reward arrays required")
            for key in ("stage_rewards", "returns", "raw_stage_rewards", "raw_returns"):
                array = values.get(key, values.get("coded_" + key))
                _require(isinstance(array, list) and len(array) == len(intervals)
                         and all(_number(v) for v in array), "invalid counterfactual " + key)


def _validate(unit):
    summary, config = unit["summary"], unit["config"]
    _identity(summary)
    _require(summary.get("kind") in _KINDS, "unknown reward unit kind")
    _require(summary.get("development_only") is True, "development_only must be true")
    for obj in (config, summary):
        if "version" in obj:
            _require(obj["version"] == VERSION, "wrong source version")
        for settings in (obj, _mapping(obj.get("settings")), _mapping(obj.get("learning"))):
            if "fd_strength" in settings:
                _require(settings["fd_strength"] == 0, "FD must be off")
    condition = _condition(summary)
    if summary["kind"] in _CONDITION_KINDS or condition is not None:
        _require(condition in CONDITIONS, "settings.condition must identify an admitted reward method")
        for declared in (summary.get("condition"), config.get("condition"), _condition(config)):
            _require(declared is None or declared == condition, "source/summary condition mismatch")
        _require(all(row.get("condition") in (None, condition) for row in unit["metrics"]),
                 "metric source condition mismatch")
    kind = summary["kind"]
    marker = summary.get("reward_repair")
    if marker is not None:
        _require(isinstance(marker, dict) and marker.get("version") == VERSION
                 and marker.get("condition") == condition, "wrong reward source identity")
    if kind == "reward_scoring":
        _require(all(_count(summary.get(k)) for k in ("unique_images", "scored_images")), "scoring counts required")
        _require(summary["scored_images"] <= summary["unique_images"], "scored_images exceeds unique_images")
        _require(_number(summary.get("seconds")) and summary["seconds"] >= 0, "scoring seconds required")
    if kind == "reward_fit":
        curves = summary.get("curves")
        _require(isinstance(curves, list) and bool(curves), "fit curves required")
        for original in curves:
            row = _curve(original)
            _require(isinstance(row.get("stage"), str) and _number(row.get("epoch")), "curve stage/epoch required")
            _require(all(_number(row.get(k)) for k in ("train_nll", "validation_nll")), "train/validation joint NLL required")
        _require(isinstance(summary.get("selection"), dict) and isinstance(summary.get("readiness"), dict),
                 "fit selection/readiness required")
    if kind == "reward_bootstrap":
        _require(bool(_mapping(summary.get("fisher_stability"))) and bool(_mapping(summary.get("normalization"))),
                 "Fisher stability/normalization required")
    if kind in ("reward_pilot", "reward_initial_evaluation"):
        if kind == "reward_pilot":
            _require(_count(_mapping(summary.get("settings")).get("seed")), "pilot settings.seed required")
            _require(_count(summary.get("acquired_count")), "pilot acquired_count required")
            _require(isinstance(summary.get("episodes"), list), "pilot episodes required")
            for row in summary["episodes"]:
                _validate_episode(row)
                _require(row.get("reward_condition", condition) == condition, "episode reward condition mismatch")
        panels = summary.get("evaluations")
        _require(isinstance(panels, list) and bool(panels), "evaluation panels required")
        seen = set()
        for panel in panels:
            _require(isinstance(panel, dict), "evaluation panel must be an object")
            cp, domain = panel.get("checkpoint"), panel.get("difficulty")
            _require(cp in ("initial", "midpoint", "final") and domain in ("easy", "hard"), "unknown evaluation checkpoint/domain")
            _require((cp, domain) not in seen, "duplicate evaluation panel")
            seen.add((cp, domain))
            _require(kind != "reward_initial_evaluation" or cp == "initial", "initial unit contains noninitial panel")
            _require(_count(panel.get("acquired_count")), "panel acquired_count required")
            if cp == "initial":
                _require(isinstance(_asset(panel, summary), str) and bool(_asset(panel, summary)), "initial source asset ID required")
                expected = _mapping(summary.get("settings")).get("initial_asset_id")
                _require(expected is None or panel.get("initial_asset_id", expected) == expected,
                         "wrong initial source asset")
            declared = panel.get("condition")
            _require(declared is None or declared == condition, "evaluation source condition mismatch")
            rows = panel.get("rows")
            _require(isinstance(rows, list) and bool(rows), "nonempty evaluation rows required")
            for row in rows:
                _validate_episode(row, evaluation=True)
                _require(row.get("reward_condition", condition) == condition, "evaluation reward condition mismatch")
            seeds = [row["seed"] for row in rows]
            _require(len(set(seeds)) == len(seeds), "duplicate evaluation seeds")
            if "seeds" in panel:
                _require(panel["seeds"] == seeds, "panel seeds do not match recorded rows")
    return unit


def read_completed_unit(path):
    """Read and validate sealed JSON/CSV only, never referenced model assets."""
    return _validate(shared.read_completed_unit(path))


def _progress(path):
    committed = (path / "CURRENT.json").is_file()
    if committed:
        value = _committed_progress(path)
    elif (path / "progress.json").is_file():
        value = shared._json(path / "progress.json")
    else:
        return {}, False
    _identity(value)
    _require(value.get("version") == VERSION, "progress version required")
    _require(isinstance(value.get("status"), str) and bool(value["status"]), "progress status required")
    for key in ("repair_active_seconds", "remaining_budget_seconds"):
        if value.get(key) is not None:
            _require(_number(value[key]) and value[key] >= 0, "invalid progress " + key)
    if "max_hours" in value:
        _require(value["max_hours"] == 12, "repair ceiling must be twelve hours")
    if "completed_trajectories" in value:
        _require(_count(value["completed_trajectories"]) and value["completed_trajectories"] <= 4,
                 "invalid completed_trajectories")
    child = value.get("active_unit_progress", {})
    _require(isinstance(child, dict), "active_unit_progress must be an object")
    counts = child.get("counts", {})
    _require(isinstance(counts, dict), "progress counts must be an object")
    _require(all(_count(v) for v in counts.values()), "progress counts must be nonnegative integers")
    if "scored" in counts and "target" in counts:
        _require(counts["scored"] <= counts["target"], "scored progress exceeds target")
    return value, committed


def load_report(root=None):
    """Bounded scan: sealed outcomes and explicitly provisional fit diagnostics."""
    campaign = _campaign_root(root)
    units, pending, scanned, size = [], [], 0, 0
    if campaign.exists():
        _require(campaign.is_dir() and not campaign.is_symlink(), "campaign must be a regular directory")
        for parent, directories, files in os.walk(campaign, followlinks=False):
            scanned += 1
            _require(scanned <= MAX_SCAN_DIRS, "campaign scan limit exceeded")
            path = Path(parent)
            directories[:] = sorted(d for d in directories if d not in {
                "transactions", "frames", "videos", "checkpoints", "score-cache", "score_cache", "reference_bundle", "inputs", ".git"}
                and not (path / d).is_symlink())
            if path == campaign:
                directories[:] = [d for d in directories if d in {"units", "orchestration"}]
            if "COMPLETED" in files:
                _require(len(units) < MAX_UNITS, "completed unit limit exceeded")
                for name in ("config.json", "summary.json", "manifest.json", "metrics.csv", "COMPLETED"):
                    artifact = path / name
                    size += artifact.stat().st_size if artifact.is_file() else 0
                _require(size <= MAX_SCALAR_BYTES, "campaign scalar size limit exceeded")
                units.append(read_completed_unit(path))
                directories[:] = []
            elif "config.json" in files or path.name == "orchestration":
                pending.append(str(path))
                directories[:] = []
    units.sort(key=lambda u: (str(u["manifest"].get("completed_at", "")), u["path"]))
    progress, committed = _progress(campaign / "orchestration")
    snapshot = {"root": str(campaign), "version": VERSION, "complete_units": units,
                "pending_units": pending, "progress": progress, "progress_committed": committed}
    active = snapshot["progress"].get("active_unit")
    snapshot["active_settings"] = {}
    if isinstance(active, str) and active in pending and (Path(active) / "config.json").is_file():
        config = shared._json(Path(active) / "config.json")
        snapshot["active_settings"] = _mapping(config.get("config", config).get("settings"))
    _fit_preview(snapshot)
    evaluation_panels(snapshot)  # Conflicting reused references must fail before rendering.
    pilots = [(_condition(u["summary"]), u["summary"]["settings"]["seed"])
              for u in units if u["summary"]["kind"] == "reward_pilot"]
    _require(len(set(pilots)) == len(pilots), "duplicate condition/replica pilot")
    scorer_sources = {json.dumps(u["summary"]["reward_repair"]["scorer"], sort_keys=True)
                      for u in units if _mapping(u["summary"].get("reward_repair")).get("scorer") is not None}
    _require(len(scorer_sources) <= 1, "different frozen scorer sources; conditions must share the evaluator")
    return snapshot


def _units(snapshot, kind):
    return [u for u in snapshot["complete_units"] if u["summary"]["kind"] == "reward_" + kind]


def _latest(snapshot, kind, condition=None):
    return next((u["summary"] for u in reversed(_units(snapshot, kind))
                 if condition is None or _condition(u["summary"]) == condition), {})


def _fit_preview(snapshot):
    """Read only the bounded diagnostics embedded in the parent's sealed commit.

    The active unit must still be incomplete. Neither its progress mirror nor
    its checkpoint is opened, and previews never enter completed-unit tables.
    """
    if not snapshot.get("progress_committed"):
        return None
    progress = snapshot["progress"]
    active = progress.get("active_unit")
    if active not in snapshot["pending_units"] or not Path(active).name.startswith("fit-"):
        return None
    child = _mapping(progress.get("active_unit_progress"))
    curves = child.get("diagnostic_curves")
    if curves is None or curves == []:
        return None
    _require(isinstance(curves, list) and len(curves) <= MAX_PREVIEW_CURVES,
             "provisional fit diagnostic_curves exceeds bounded row limit or is not a list")
    condition = snapshot["active_settings"].get("condition")
    _require(condition in CONDITIONS, "provisional fit requires active_settings.condition")
    _require(child.get("condition", condition) == condition, "provisional fit condition mismatch")
    result = []
    for original in curves:
        row = _curve(original)
        _require(row.get("stage") == "nll" and _count(row.get("epoch"))
                 and all(_number(row.get(k)) for k in ("train_nll", "validation_nll")),
                 "provisional fit requires NLL stage, epoch and finite train/validation metrics")
        for key in ("train_kl", "validation_kl", "gradient_norm", "update_norm"):
            _require(row.get(key) is None or _number(row[key]), "nonfinite provisional fit " + key)
        result.append({key: row.get(key) for key in (
            "stage", "epoch", "train_nll", "validation_nll", "train_kl", "validation_kl", "gradient_norm", "update_norm")})
    return {"condition": condition, "curves": result}


def _shown(value):
    return PENDING if value is None else str(value)


def _hours(value):
    return f"{value / 3600:.2f}" if _number(value) and value >= 0 else PENDING


def progress_rows(snapshot):
    progress = {**_latest(snapshot, "campaign"), **snapshot["progress"]}
    child = _mapping(progress.get("active_unit_progress"))
    counts = _mapping(child.get("counts"))
    scoring = _latest(snapshot, "scoring")
    scored = counts.get("scored", scoring.get("scored_images"))
    target = counts.get("target") if "scored" in counts else scoring.get("unique_images")
    pilots = _units(snapshot, "pilot")
    completed = progress.get("completed_trajectories", len(pilots) if pilots else None)
    forecast = progress.get("forecast")
    forecast_note = ""
    if isinstance(forecast, dict):
        if "four_pilots_hours" in forecast:
            hours = forecast["four_pilots_hours"]
            if isinstance(hours, list) and len(hours) == 2 and all(_number(h) and h >= 0 for h in hours):
                forecast = [h * 3600 * (4-completed)/4 for h in hours] if _count(completed) else [h*3600 for h in hours]
                forecast_note = " (scaled four-pilot forecast, not an updated measurement)"
            else:
                forecast = None
        else:
            forecast = forecast.get("remaining_time_range_seconds", forecast.get("remaining_seconds"))
    forecast_text = "--".join(_hours(v) for v in forecast) if isinstance(forecast, list) and len(forecast) == 2 else _hours(forecast)
    charged = progress.get("repair_active_seconds", progress.get("charged_seconds"))
    budget = progress.get("remaining_budget_seconds")
    if budget is None and _number(charged):
        budget = max(0., 12*3600-charged)
    rows = [{"Item": "Status / active unit", "State": progress.get("status", "not yet started") + " / " + _shown(progress.get("active_unit"))},
            {"Item": "Scoring done / remaining", "State": _shown(scored) + " / " + _shown(target - scored if _count(target) and _count(scored) else None)},
            {"Item": "Pilots completed / planned", "State": _shown(completed) + " / 4"},
            {"Item": "Charged / remaining ceiling h (12 h cap)", "State": _hours(charged) + " / " + _hours(budget)},
            {"Item": "Measured remaining h", "State": forecast_text + forecast_note}]
    for condition in CONDITIONS:
        fit, fisher = _latest(snapshot, "fit", condition), _latest(snapshot, "bootstrap", condition)
        state = _mapping(fit.get("readiness")).get("status", "sealed" if fit else PENDING)
        fisher_state = "sealed" if fisher else PENDING
        active = str(progress.get("active_unit", ""))
        if child.get("condition", snapshot.get("active_settings", {}).get("condition")) == condition or condition in active:
            live = "running: " + str(child.get("stage", child.get("status", "active")))
            if counts:
                live += " " + _shown(counts.get("step")) + "/" + _shown(counts.get("target"))
            if "fisher" in active or "bootstrap" in active:
                fisher_state = live
            elif "fit" in active:
                state = live
        rows.append({"Item": _LABELS[condition] + " fit / Fisher", "State": state + " / " + fisher_state})
    evaluations = _rows(child.get("evaluations"))
    if evaluations:
        rows.append({"Item": "Active held-out episode scoring", "State": "; ".join(
            f"{p.get('checkpoint', '?')}/{p.get('difficulty', '?')}: {_shown(p.get('scored'))}/{_shown(p.get('episodes'))}"
            for p in evaluations[:6])})
    checks = [u["summary"]["kind"].removeprefix("reward_") for u in snapshot["complete_units"]
              if u["summary"]["kind"] in {"reward_software", "reward_integrity", "reward_audit", "reward_provenance"}]
    rows.append({"Item": "Sealed checks / incomplete units", "State": (", ".join(checks) or PENDING) + " / " + str(len(snapshot["pending_units"]))})
    return rows


def evaluation_panels(snapshot):
    """Deduplicate initial references by condition/checkpoint/asset/domain."""
    seen, result, initial_sources = {}, [], {}
    for unit in snapshot["complete_units"]:
        summary = unit["summary"]
        if summary["kind"] not in {"reward_pilot", "reward_initial_evaluation"}:
            continue
        for panel in summary["evaluations"]:
            initial = panel["checkpoint"] == "initial"
            if initial:
                source = (_condition(summary), panel["difficulty"])
                initial_sources.setdefault(source, _asset(panel, summary))
                _require(initial_sources[source] == _asset(panel, summary), "different initial source assets; do not pool initializers")
            replica = "shared initial" if initial else summary["settings"]["seed"]
            key = (_condition(summary), panel["checkpoint"], _asset(panel, summary), panel["difficulty"], replica)
            # Noninitial models from different units are never accidentally pooled.
            content = json.dumps({"rows": panel["rows"], "acquired_count": panel["acquired_count"]}, sort_keys=True)
            if key in seen:
                _require(seen[key] == content, "conflicting evaluation source/asset reuse")
                continue
            seen[key] = content
            result.append({"condition": _condition(summary), "replica": replica, "panel": panel, "unit": unit})
    return result


def _domain(episode):
    difficulty = episode.get("difficulty", _mapping(episode.get("config")).get("difficulty"))
    if difficulty in ("easy", "hard"):
        return difficulty
    layout = _mapping(episode.get("layout"))
    if layout.get("domain") in ("A", "B"):
        return "easy" if layout["domain"] == "A" else "hard"
    blocked = _mapping(layout.get("curriculum")).get("direct_path_blocked")
    if type(blocked) is bool:
        return "hard" if blocked else "easy"
    barrier = _mapping(episode.get("config")).get("barrier_density")
    return "hard" if _number(barrier) and barrier > 0 else "easy" if barrier == 0 else None


def _episode_value(row, t, *, cumulative=False, coded=False):
    elapsed, total, observed = 0., 0., None
    for i, interval in enumerate(row["intervals"]):
        elapsed += interval["elapsed_s"]
        if cumulative:
            if elapsed > t + 1e-9:
                return total
            total += row["stage_rewards" if coded else "raw_stage_rewards"][i]
        else:
            if elapsed > t + 1e-9:
                return observed
            observed = row["visual_scores"][i]["score"]
    return total if cumulative else observed if t <= elapsed + 1e-9 else None


def reward_curves(snapshot, *, grid=None, cumulative=False, coded=False):
    """Equal replica weights, with episode counts varying after termination.

    Post-action scores are held until the next observation, never extended
    before the first observation or after termination. Earned (undiscounted)
    cumulative rewards are step functions, held flat after termination.
    """
    grid = [i / 2 for i in range(41)] if grid is None else list(grid)
    groups = {}

    def add(mode, domain, condition, label, replica, episode):
        if domain in ("easy", "hard"):
            groups.setdefault((mode, domain, condition, label), {}).setdefault(replica, []).append(episode)

    for unit in _units(snapshot, "pilot"):
        summary = unit["summary"]
        for row in summary["episodes"]:
            acquired = row.get("reset_exposure", row.get("acquired_count", row.get("end_acquired_count")))
            label = row.get("window")
            if label not in ("early", "middle", "late"):
                if not _number(acquired):
                    continue
                label = "early" if acquired < 640 / 3 else "middle" if acquired < 1280 / 3 else "late"
            add("training", _domain(row), _condition(summary), label, summary["settings"]["seed"], row)
    for item in evaluation_panels(snapshot):
        panel = item["panel"]
        for row in panel["rows"]:
            add("held-out", panel["difficulty"], item["condition"], panel["checkpoint"], item["replica"], row)
    curves = []
    for (mode, domain, condition, label), replicas in groups.items():
        points = []
        for t in grid:
            means, count = [], 0
            for episodes in replicas.values():
                values = [_episode_value(row, t, cumulative=cumulative, coded=coded) for row in episodes]
                values = [v for v in values if v is not None]
                if values:
                    means.append(sum(values) / len(values))
                    count += len(values)
            if means:
                points.append({"elapsed_s": t, "mean": sum(means) / len(means), "episodes": count, "replicas": len(means)})
        curves.append({"mode": mode, "difficulty": domain, "condition": condition, "label": label, "points": points})
    return curves


def endpoint_rows(snapshot):
    panels = evaluation_panels(snapshot)
    rows = []
    for item in panels:
        panel, condition = item["panel"], item["condition"]
        if panel["checkpoint"] != "final":
            continue
        outcomes = panel["rows"]
        initial = next((p["panel"] for p in panels if p["condition"] == condition
                        and p["panel"]["checkpoint"] == "initial" and p["panel"]["difficulty"] == panel["difficulty"]), None)
        matched = initial is not None and {r["seed"] for r in initial["rows"]} == {r["seed"] for r in outcomes}
        rate = sum(r["success"] for r in outcomes) / len(outcomes)
        change = rate - sum(r["success"] for r in initial["rows"]) / len(initial["rows"]) if matched else None
        rows.append({"Method": _LABELS[condition], "Replica": item["replica"], "Task": "A" if panel["difficulty"] == "easy" else "B",
                     "Success / episodes": f"{sum(r['success'] for r in outcomes)} / {len(outcomes)}",
                     "Collisions": sum(r["collision"] for r in outcomes) if all(type(r.get("collision")) is bool for r in outcomes) else None,
                     "Timeouts": sum(r["end_reason"] in ("timeout", "deadline") for r in outcomes) if all(isinstance(r.get("end_reason"), str) for r in outcomes) else None,
                     "A change vs shared initial": change if panel["difficulty"] == "easy" else None,
                     "Raw earned / coded earned": f"{sum(sum(r['raw_stage_rewards']) for r in outcomes)/len(outcomes):.3g} / {sum(sum(r['stage_rewards']) for r in outcomes)/len(outcomes):.3g}",
                     "Raw / coded discounted start return": f"{sum(r['raw_returns'][0] for r in outcomes)/len(outcomes):.3g} / {sum(r['returns'][0] for r in outcomes)/len(outcomes):.3g}"})
    return rows


def cost_rows(snapshot):
    rows = []
    for unit in snapshot["complete_units"]:
        summary = unit["summary"]
        if summary["kind"] not in {"reward_pilot", "reward_readiness", "reward_scoring", "reward_fit", "reward_bootstrap"}:
            continue
        timing = _mapping(summary.get("timings"))
        intervals = [r for e in _rows(summary.get("episodes")) for r in e.get("intervals", [])]
        steering = [abs(r["controls"]["steering_rad"]) for r in intervals
                    if _number(_mapping(r.get("controls")).get("steering_rad"))]
        drive = [r["controls"]["drive_mps"] for r in intervals
                 if _number(_mapping(r.get("controls")).get("drive_mps"))]
        fallbacks = _mapping(summary.get("fallback_counts"))
        rows.append({"Stage": summary["kind"].removeprefix("reward_"), "Method": _condition(summary),
                     "Seed": _mapping(summary.get("settings")).get("seed"), "Observations": summary.get("acquired_count"),
                     "Unique / scored": f"{summary['unique_images']} / {summary['scored_images']}" if "unique_images" in summary else None,
                     **{key: timing.get(key, summary.get(key)) for key in ("active_seconds", "seconds", "scoring_seconds", "inference_seconds", "fit_seconds", "deadline_misses", "fallback_count", "peak_vram_bytes")},
                     "Fallback intervals": sum(v for k, v in fallbacks.items() if k != "policy") if fallbacks and all(_count(v) for v in fallbacks.values()) else None,
                     "Mean |steering| rad": sum(steering)/len(steering) if steering else None,
                     "Mean drive m/s": sum(drive)/len(drive) if drive else None})
    return rows


def initialization_rows(snapshot):
    rows = []
    for condition in CONDITIONS:
        fit, fisher = _latest(snapshot, "fit", condition), _latest(snapshot, "bootstrap", condition)
        if not (fit or fisher):
            continue
        selection = _mapping(_mapping(fit.get("selection")).get("nll"))
        accepted = next((_curve(r) for r in fit.get("curves", [])
                         if r.get("stage") == "nll" and r.get("epoch") == selection.get("epoch")), {})
        stability = _mapping(fisher.get("fisher_stability"))
        normalization = _mapping(fisher.get("normalization"))
        influences = [m["first_step_ewc_suppression"] for u in _units(snapshot, "pilot")
                      if _condition(u["summary"]) == condition for m in _rows(u["summary"].get("fit_metrics"))
                      if _number(m.get("first_step_ewc_suppression"))]
        rows.append({"Method": _LABELS[condition], "Accepted NLL pass": selection.get("epoch"),
                     "Accepted train / validation teacher KL": f"{_shown(accepted.get('train_kl'))} / {_shown(accepted.get('validation_kl'))}",
                     "Fisher histories / draws": f"{_shown(normalization.get('history_count'))} / {_shown(normalization.get('draws_per_history'))}",
                     "Independent Fisher relative L2": _mapping(stability.get("independent_subset")).get("relative_l2"),
                     "Fisher within diagnostic tolerance": stability.get("within_diagnostic_tolerance"),
                     "Mean first-step EWC suppression (events)": sum(influences)/len(influences) if influences else None})
    return rows


def _reward_figure(snapshot):
    from matplotlib.figure import Figure

    curves = reward_curves(snapshot)
    if not curves:
        return None
    figure = Figure(figsize=(10, 6), layout="constrained")
    axes = figure.subplots(2, 2)
    styles = {"early": ":", "middle": "--", "late": "-", "initial": ":", "midpoint": "--", "final": "-"}
    for curve in curves:
        axis = axes[("easy", "hard").index(curve["difficulty"]), ("training", "held-out").index(curve["mode"])]
        points = curve["points"]
        if points:
            counts = [(p["episodes"], p["replicas"]) for p in points]
            label = f"{_LABELS[curve['condition']]} {curve['label']} (e={min(v[0] for v in counts)}--{max(v[0] for v in counts)}, r={min(v[1] for v in counts)}--{max(v[1] for v in counts)})"
            axis.plot([p["elapsed_s"] for p in points], [p["mean"] for p in points],
                      color=_COLORS[curve["condition"]], linestyle=styles[curve["label"]], label=label)
    for i, task in enumerate(("A", "B")):
        for j, mode in enumerate(("Training windows", "Held-out checkpoints")):
            title = ("Clear / A-labeled" if i == 0 else "Blocked / B-labeled") + ": " + mode if j == 0 else task + ": " + mode
            axes[i, j].set(title=title, xlabel="Elapsed episode seconds", ylabel="Raw Qwen score (0--10)", ylim=(-.2, 10.2))
    _finish(axes.ravel())
    return _figure(figure, "Scores stop at termination; equal replica weights after within-replica episode means. Legend e/r: contributing episodes/replicas (range over time). Continuous-curriculum training is stratified by its saved clear/blocked flag when no task label exists; these are not fixed A/B tests.")


def _capability_figure(snapshot):
    from matplotlib.figure import Figure

    panels = evaluation_panels(snapshot)
    if not panels:
        return None
    figure = Figure(figsize=(9, 3), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, domain in zip(axes, ("easy", "hard")):
        for condition in CONDITIONS:
            grouped = {}
            for item in panels:
                p = item["panel"]
                if p["difficulty"] == domain and item["condition"] == condition:
                    grouped.setdefault(p["acquired_count"], []).append(sum(r["success"] for r in p["rows"]) / len(p["rows"]))
            if grouped:
                axis.plot(sorted(grouped), [sum(grouped[x])/len(grouped[x]) for x in sorted(grouped)], "o-", color=_COLORS[condition], label=_LABELS[condition])
        axis.set(title="A" if domain == "easy" else "B", xlabel="Incoming intervals", ylabel="Held-out success rate", ylim=(-.05, 1.05))
    _finish(axes)
    return _figure(figure, "Initial/midpoint/final task success; initial reference counted once per condition, not once per online seed. No significance inference.")


def _initialization_figure(snapshot):
    from matplotlib.figure import Figure

    preview = _fit_preview(snapshot)
    if not (_units(snapshot, "fit") or _units(snapshot, "bootstrap") or preview):
        return None
    figure = Figure(figsize=(10, 3), layout="constrained")
    left, right = figure.subplots(1, 2)
    for condition in CONDITIONS:
        curves = [_curve(r) for r in _latest(snapshot, "fit", condition).get("curves", []) if r.get("stage") == "nll"]
        for split, style in (("train", "-"), ("validation", "--")):
            if curves:
                left.plot([r["epoch"] for r in curves], [r[split + "_nll"] for r in curves], style,
                          color=_COLORS[condition], label=_LABELS[condition] + " " + split + " sealed")
        stability = _mapping(_latest(snapshot, "bootstrap", condition).get("fisher_stability"))
        points = [r for r in _rows(stability.get("nested_draws")) if _number(r.get("relative_l2")) and _number(r.get("draws_per_history"))]
        if points:
            right.plot([r["draws_per_history"] for r in points], [r["relative_l2"] for r in points], "o-", color=_COLORS[condition], label=_LABELS[condition])
    if preview:
        condition, curves = preview["condition"], preview["curves"]
        for split, style in (("train", ":"), ("validation", "-.")):
            left.plot([r["epoch"] for r in curves], [r[split + "_nll"] for r in curves],
                      linestyle=style, marker="o", markersize=3, alpha=.75,
                      color=_COLORS[condition], label=_LABELS[condition] + " " + split + " provisional/incomplete")
        left.set_title("Includes committed provisional/incomplete fit", fontsize=10)
    left.set(xlabel="NLL pass", ylabel="Joint NLL (different targets)")
    right.set(xlabel="Nested label draws/history", ylabel="Fisher relative L2 vs four draws")
    _finish((left, right))
    return _figure(figure, "Reward models have different NLL targets. "
                   + ("Provisional/incomplete fit curves are committed diagnostics only, not an accepted checkpoint or completed outcome. " if preview else "")
                   + "Nested Fisher prefixes are sensitivity diagnostics, not independent uncertainty estimates; full teacher KL/selection/gradient diagnostics remain linked.")


def _links(snapshot, notebook_dir):
    links, clips, seen_clips = [], [], set()
    root = Path(snapshot["root"]).resolve()
    for unit in snapshot["complete_units"]:
        path = Path(unit["path"]) / "summary.json"
        href = quote(os.path.relpath(path, notebook_dir))
        links.append(f'<a href="{href}">{html.escape(Path(unit["path"]).name)}</a>')
        for video in _rows(unit["summary"].get("videos")):
            if video.get("selection") not in ("preselected", "failure", "failure_gallery", "reward_spike"):
                continue
            reference = video.get("path")
            if not isinstance(reference, str):
                continue
            path = Path(reference)
            path = path if path.is_absolute() else Path(unit["path"]) / path
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or resolved in seen_clips:
                continue
            seen_clips.add(resolved)
            label = " / ".join(str(v) for v in (_condition(unit["summary"]), video.get("checkpoint"), video.get("difficulty"), video.get("view"), video["selection"]) if v is not None)
            clips.append(f'<a href="{quote(os.path.relpath(path, notebook_dir))}">{html.escape(label)}</a>')
    import nbformat
    return nbformat.v4.new_output("display_data", data={"text/plain": "Scalar artifacts and preselected clips; failure/reward-spike clips are diagnostic, not representative samples.",
        "text/html": "<p>Scalar artifacts: " + (", ".join(links) or PENDING) + "</p><p>Preselected clips / failure gallery: " + ("; ".join(clips[:12]) or PENDING) + "</p>"})


def result_outputs(snapshot, *, notebook_dir=None):
    outputs = []
    for figure in (_reward_figure(snapshot), _capability_figure(snapshot), _initialization_figure(snapshot)):
        if figure is not None:
            outputs.append(figure)
    preview = _fit_preview(snapshot)
    if preview:
        last = preview["curves"][-1]
        outputs.append(_note(f"{_LABELS[preview['condition']]} provisional/incomplete fit: committed NLL pass {last['epoch']}; "
                             f"train/validation NLL {_shown(last['train_nll'])} / {_shown(last['validation_nll'])}, "
                             f"teacher KL {_shown(last['train_kl'])} / {_shown(last['validation_kl'])}, "
                             f"gradient/update norm {_shown(last['gradient_norm'])} / {_shown(last['update_norm'])}. "
                             "No accepted checkpoint, completed fit or Fisher result is inferred."))
    endpoints = endpoint_rows(snapshot)
    outputs.append(_table(endpoints) if endpoints else _note("Endpoint outcomes: " + PENDING))
    costs = cost_rows(snapshot)
    outputs.append(_table(costs) if costs else _note("Measured costs: " + PENDING))
    initializers = initialization_rows(snapshot)
    if initializers:
        outputs.append(_table(initializers))
    outputs.append(_note("Raw/coded earned totals are undiscounted; start returns use physical-time discounting. Counterfactual reward arithmetic is not a counterfactual learned-policy rollout. Visual curves stop at termination; cumulative earned rewards stay flat thereafter. Missing train domains/windows are not inferred from test panels."))
    outputs.append(_links(snapshot, notebook_dir or Path.cwd()))
    return outputs


def _display(outputs):
    from IPython.display import display
    for output in outputs:
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def display_progress(snapshot):
    _display([_table(progress_rows(snapshot))])


def display_results(snapshot, *, notebook_dir=None):
    _display(result_outputs(snapshot, notebook_dir=notebook_dir))


def conclusion(snapshot):
    pilots = _units(snapshot, "pilot")
    panels = evaluation_panels(snapshot)
    methods, outcomes = [], []

    def counts(panel):
        return f"{sum(r['success'] for r in panel['rows'])}/{len(panel['rows'])}" if panel else "pending"

    for condition in CONDITIONS:
        method = _LABELS[condition]
        seeds = sorted(u["summary"]["settings"]["seed"] for u in pilots if _condition(u["summary"]) == condition)
        initial = {d: next((p["panel"] for p in panels if p["condition"] == condition
                           and p["panel"]["checkpoint"] == "initial" and p["panel"]["difficulty"] == d), None)
                   for d in ("easy", "hard")}
        final = {(p["panel"]["difficulty"], p["replica"]): p["panel"] for p in panels
                 if p["condition"] == condition and p["panel"]["checkpoint"] == "final"}
        complete = len(seeds) == 2 and all(initial.values()) and all(
            (d, seed) in final for d in ("easy", "hard") for seed in seeds)
        changes = []
        for seed in seeds:
            before, after = initial["easy"], final.get(("easy", seed))
            if before and after and {r["seed"] for r in before["rows"]} == {r["seed"] for r in after["rows"]}:
                changes.append(sum(r["success"] for r in after["rows"]) / len(after["rows"])
                               - sum(r["success"] for r in before["rows"]) / len(before["rows"]))
        retained = len(changes) == 2 and all(change >= 0 for change in changes)
        b_success_seeds = sum(any(r["success"] for r in final[("hard", seed)]["rows"])
                              for seed in seeds if ("hard", seed) in final)
        methods.append({"name": method, "seeds": set(seeds), "complete": complete,
                        "retained": retained, "a_matched": len(changes) == len(seeds),
                        "a_loss_seeds": sum(change < 0 for change in changes), "b_success_seeds": b_success_seeds})
        comparisons = []
        for domain, label in (("easy", "A retention"), ("hard", "B acquisition")):
            finals = [counts(final.get((domain, seed))) for seed in seeds]
            if len(seeds) < 2:
                finals.append("pending")
            comparisons.append(label + " " + counts(initial[domain]) + " -> " + ", ".join(finals))
        order = " (final seeds " + ", ".join(map(str, seeds)) + ")" if seeds else ""
        outcomes.append(method + order + ": " + "; ".join(comparisons) + ".")
    paired = len(methods[0]["seeds"] & methods[1]["seeds"])
    complete = len(pilots) == 4 and paired == 2 and all(m["complete"] for m in methods)
    matched = all(m["a_matched"] for m in methods)
    if not complete or not matched:
        guidance = "Defer recipe selection until matched paired A/B panels are complete. Missing or budget-omitted work is not evidence of inability to learn."
    elif not any(m["b_success_seeds"] for m in methods):
        guidance = "Hard-task acquisition remains unestablished for both conditions. Neither is ready for promotion; review A retention and reward/coverage limitations before extending."
    else:
        candidates = [m["name"] for m in methods if m["b_success_seeds"] and m["retained"]]
        guidance = ("Provisional follow-up candidates with B successes and no observed A decline: " + ", ".join(candidates) + "."
                    if candidates else "B successes accompany A losses; review that trade-off before promoting either recipe.")
        guidance += " B successes are development evidence only, not proof of learning beyond initialization."
    coverage = "; ".join(f"{m['name']}: A declined in {m['a_loss_seeds']}/{len(m['seeds'])} seeds, "
                         f"B successes in {m['b_success_seeds']}/{len(m['seeds'])} evaluated seeds"
                         for m in methods if m["seeds"] and m["complete"] and m["a_matched"])
    if coverage:
        guidance = coverage + ". " + guidance
    status = "Complete development panels." if complete and matched else "Partial development results; unevaluated panels remain pending."
    return ("## Phase 3 Fix 3 Conclusion\n\n" + f"{len(pilots)}/4 sealed pilot trajectories; {paired}/2 complete paired seeds. "
            + status + " Counts are successes/episodes; shared initial panels count once.\n\n"
            + "\n\n".join(outcomes) + "\n\n" + guidance
            + " FD was off: no FD benefit, significance, causal proof, sample-savings or retention-equivalence claim follows. "
            "Stop for review; Phase 4 and physical motion remain deferred.")


def build_notebook(root, output, base_notebook=None):
    """Append Fix 3 to an immutable history snapshot, without rebuilding history.

    Defaults to ``campaign/history.ipynb``. Missing history fails clearly;
    callers may explicitly supply another saved notebook. Refreshes replace
    only cells tagged by this report. Output publication is atomic.
    """
    import nbformat

    snapshot = load_report(root)
    campaign = Path(snapshot["root"])
    base = Path(base_notebook) if base_notebook is not None else campaign / "history.ipynb"
    output = Path(output)
    _require(base.resolve() != output.resolve(), "output must not overwrite immutable history")
    _require(not output.is_symlink(), "output must not be a symlink")
    _require(not any((p / "COMPLETED").exists() for p in output.resolve().parents), "output is inside a completed unit")
    book = nbformat.reads(shared._read(base, MAX_HISTORY_BYTES).decode("utf-8"), as_version=4)
    book = copy.deepcopy(book)
    book.cells = [c for c in book.cells if not c.metadata.get(_TAG)]
    code = ("from numerical_experiment.picar_reward_repair import report as reward_report\n"
            f"reward_snapshot = reward_report.load_report({str(campaign.resolve())!r})\n")
    bootstrap = (
        "from pathlib import Path\nimport sys\n"
        "for parent in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (parent / 'numerical_experiment/picar_reward_repair/report.py').is_file():\n"
        "        sys.path.insert(0, str(parent))\n"
        "        break\n"
        "else:\n"
        "    raise RuntimeError('Run this notebook from within the project checkout.')\n")
    progress = nbformat.v4.new_code_cell(bootstrap + code + "reward_report.display_progress(reward_snapshot)", outputs=[_table(progress_rows(snapshot))])
    progress.id = "plan19-phase3-fix3-progress"
    progress.metadata[_TAG] = True
    book.cells.insert(min(1, len(book.cells)), progress)
    section = [nbformat.v4.new_markdown_cell(_INTRO), nbformat.v4.new_code_cell(
        code + f"reward_report.display_results(reward_snapshot, notebook_dir={str(output.parent.resolve())!r})",
        outputs=result_outputs(snapshot, notebook_dir=output.parent)), nbformat.v4.new_markdown_cell(conclusion(snapshot))]
    for index, cell in enumerate(section):
        cell.id = f"plan19-phase3-fix3-{index}"
        cell.metadata[_TAG] = True
    book.cells.extend(section)
    nbformat.validate(book)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".reward-report-", suffix=".ipynb", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        nbformat.write(book, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
