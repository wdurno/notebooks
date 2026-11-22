"""Bounded, scalar-only Fix 4 reporting; never load a runtime or checkpoint.

Only sealed trajectory units supply outcome curves. Timing and calibration
units remain separate. Live orchestration counters are provisional, not model
results. Historical notebooks are preserved rather than rerendered.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import html
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.phase2_report import _mapping, _note, _number, _table
from numerical_experiment.picar_initialization.report import _committed_progress, _figure
from . import VERSION


CAMPAIGN = "curriculum-fix4-v1"
METHODS = ("replay_ewc", "stm_fd")
ENVIRONMENTS = ("A", "B_initial", "B_final")
MAX_UNITS = 128
MAX_SCAN_DIRS = 2048
MAX_SCALAR_BYTES = 512 * 1024 * 1024
MAX_HISTORY_BYTES = 64 * 1024 * 1024
_TAG = "plan19_phase3_fix4"
_DRAFT_TAG = "plan19_phase3_fix4_draft"
_OPERATIONS = {"trajectory", "timing", "calibration_run", "selection", "forecast", "integrity",
               "references", "reference_bank", "initial_evaluation", "provenance", "software",
               "cpu_smoke", "audit", "readiness", "fd_probe"}
_LABELS = {"replay_ewc": "Replay/EWC", "stm_fd": "STM FD"}
_COLORS = {"replay_ewc": "#167d91", "stm_fd": "#b14368"}
_INTRO = """### Phase 3 Fix 4: Staged A-To-B Learning And Predictive Loss

Fix 3's barrier and steering/texture clocks did not settle at the same B
endpoint. Fix 4 gives one uninterrupted learner additional acquisition and
adaptation time: A hold, blocking ramp, B initial hold, steering/texture ramp,
then B final hold. Replay/EWC control and positive STM FD share the linear
six-example reward design and the inherited stationary initializer. Its
unresolved training plateau is not a global-MLE claim.

The diagram is the proposed 2,048-interval schedule, not measured results;
sealed settings and actual acquisition counts identify the executed schedule.

![Proposed Phase 3 Fix 4 schedule]({diagram})

Training NLL is first-inner-step **data-only** NLL in 32-observation bins:
events are averaged within replica, then replicas receive equal weight.
EWC, ridge and FD penalties are excluded. Spikes remain visible; changing
return/advantage weights are not automatically numerical failure.

Fixed-reference joint and return NLLs use the same sealed histories, executed
logits, completed labels and pre-action baselines at each checkpoint. Within
replica they are equal-record means, then replicas receive equal weight.
These off-policy reference fits are not navigation success. Current-policy
NLLs are kept separate: deterministic own-action Gaussian densities include
all normalizers and are not imitation accuracy. Per-record scalar artifacts
retain the outputs and likelihood design needed for exact recomputation.

Only sealed main trajectories enter the curves below. Timing pilots,
calibration and live previews do not. Repeatedly inspected development panels
and two paired seeds do not establish confirmation or retention equivalence.
Missing, not collected, stopped and not yet run are distinct from zero.
"""


def _require(ok, message):
    if not ok:
        raise shared.ArtifactError("Phase 3 Fix 4 schema-incompatible: " + message)


def _count(value):
    return type(value) is int and value >= 0


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


def _identity(value):
    shared._version(value.get("schema_version"), 1, "Fix 4 schema")
    _require(value.get("version") == VERSION, "wrong version; campaigns must not be pooled")


def _campaign_root(root):
    path = shared.resolve_root(root)
    if path.name.startswith("curriculum-fix4-"):
        _require(path.name == CAMPAIGN, "wrong campaign version")
        return path
    if any((path / name).exists() for name in ("units", "orchestration", "history.ipynb")):
        return path
    return path / "phase3" / CAMPAIGN


def _nll(row):
    _require(isinstance(row, dict), "predictive row must be an object")
    _require(all(_number(row.get(key)) for key in ("return_nll", "action_nll", "joint_nll", "weight"))
             and row["weight"] > 0, "finite predictive components and positive weight required")
    _require(math.isclose(row["return_nll"] + row["action_nll"], row["joint_nll"], rel_tol=2e-6, abs_tol=2e-6),
             "joint NLL must equal return NLL plus action NLL")
    if "log_weight" in row:
        _require(_number(row["log_weight"]) and math.isclose(math.log(row["weight"]), row["log_weight"],
                 rel_tol=2e-6, abs_tol=2e-6), "weight and log_weight disagree")


def _episode(row, *, evaluation=False):
    _require(isinstance(row, dict) and row.get("complete") is True and row.get("truncated", False) is False,
             "episode must be complete and nontruncated")
    _require(type(row.get("success")) is bool and row.get("reward_scoring") is True,
             "episode requires Boolean success and complete reward scoring")
    intervals = row.get("intervals")
    _require(isinstance(intervals, list) and bool(intervals), "episode intervals required")
    for interval in intervals:
        _require(isinstance(interval, dict) and _number(interval.get("elapsed_s")) and interval["elapsed_s"] > 0,
                 "finite positive interval elapsed_s required")
    for key in ("raw_stage_rewards", "raw_returns"):
        values = row.get(key)
        _require(isinstance(values, list) and len(values) == len(intervals) and all(_number(v) for v in values),
                 "complete finite " + key + " must match intervals")
    if evaluation:
        _require(row.get("evaluation") is True and row.get("learning") is False and row.get("explore") is False,
                 "evaluation must explicitly be nonlearning and nonexploring")
        _require(_count(row.get("seed")), "evaluation seed required")
        _require(row.get("predictive_scoring") is True, "completed evaluation requires predictive scoring")
        scores, missing = row.get("predictive_scores"), row.get("predictive_missing_intervals")
        _require(isinstance(scores, list) and _count(missing) and len(scores) + missing == len(intervals),
                 "predictive coverage must account for every interval")
        indices = set()
        for score in scores:
            _nll(score)
            index = score.get("index")
            _require(_count(index) and index < len(intervals) and index not in indices,
                     "predictive interval indices must be unique and in range")
            indices.add(index)


def _validate(unit):
    summary, config = unit["summary"], unit["config"]
    _identity(summary)
    if "version" in config:
        _require(config["version"] == VERSION, "configuration version differs")
    kind = summary.get("kind", "")
    _require(kind in {"curriculum_" + name for name in _OPERATIONS}, "unknown curriculum operation")
    operation = kind.removeprefix("curriculum_")
    if "operation" in config:
        _require(config["operation"] == operation, "configuration operation differs")
    settings = summary.get("settings")
    _require(isinstance(settings, dict), "settings must be an object")
    if "settings" in config:
        _require(config["settings"] == settings, "configuration settings differ")
    _require(_number(summary.get("active_seconds")) and summary["active_seconds"] >= 0,
             "finite nonnegative active_seconds required")
    if operation in {"trajectory", "timing", "calibration_run"}:
        _require(settings.get("method") in METHODS and _count(settings.get("seed")), "method and replica seed required")
        _require(_count(summary.get("acquired_count")), "actual acquired_count required")
        for row in summary.get("episodes", []):
            _episode(row)
        if "fd_strength" in summary:
            strength = summary["fd_strength"]
            _require(_number(strength) and (strength > 0 if settings["method"] == "stm_fd" else strength == 0),
                     "FD strength must match method")
    panels = summary.get("evaluations", [])
    _require(isinstance(panels, list), "evaluations must be a list")
    seen = set()
    for panel in panels:
        _require(isinstance(panel, dict), "evaluation panel must be an object")
        target, domain = panel.get("target"), panel.get("difficulty")
        _require(_count(target) and domain in ENVIRONMENTS and panel.get("checkpoint") == f"n{target}",
                 "evaluation target/checkpoint/environment identity differs")
        _require(_count(panel.get("acquired_count")) and panel["acquired_count"] >= target,
                 "evaluation actual count must reach its target")
        _require((target, domain) not in seen, "duplicate evaluation panel")
        seen.add((target, domain))
        if operation == "initial_evaluation":
            _require(target == 0, "shared initialization contains a noninitial panel")
        rows = panel.get("rows")
        _require(isinstance(rows, list) and bool(rows), "nonempty scored evaluation rows required")
        for row in rows:
            _episode(row, evaluation=True)
        seeds = [row["seed"] for row in rows]
        _require(len(set(seeds)) == len(seeds), "duplicate evaluation layout seeds")
        if "seeds" in panel:
            _require(panel["seeds"] == seeds, "declared evaluation seeds differ from rows")
    references = summary.get("reference_scores", [])
    _require(isinstance(references, list), "reference_scores must be a list")
    targets = set()
    for checkpoint in references:
        _require(isinstance(checkpoint, dict) and _count(checkpoint.get("target"))
                 and _count(checkpoint.get("acquired_count")) and checkpoint["acquired_count"] >= checkpoint["target"],
                 "reference target/actual count required")
        _require(checkpoint["target"] not in targets, "duplicate reference checkpoint")
        targets.add(checkpoint["target"])
        rows = checkpoint.get("rows")
        _require(isinstance(rows, list) and bool(rows), "nonempty reference scores required")
        ids = set()
        for row in rows:
            _nll(row)
            _require(row.get("environment") in ENVIRONMENTS and all(isinstance(row.get(k), str) and row[k]
                     for k in ("episode_id", "record_id")), "reference record identity required")
            _require(row["record_id"] not in ids, "duplicate reference record ID")
            ids.add(row["record_id"])
    if operation == "reference_bank":
        for key in ("histories", "episodes", "captured"):
            if key in summary:
                _require(_count(summary[key]), "reference bank " + key + " must be a count")
    return unit


def read_completed_unit(path):
    """Validate shared completion seals and Fix 4 JSON/CSV, without unpickling."""
    return _validate(shared.read_completed_unit(path))


def load_report(root=None):
    """Bounded scan of units only; incomplete directories supply no outcomes."""
    campaign = _campaign_root(root)
    units, incomplete, scanned, size = [], [], 0, 0
    if campaign.exists():
        _require(campaign.is_dir() and not campaign.is_symlink(), "campaign must be a regular directory")
    unit_root = campaign / "units"
    _require(not unit_root.is_symlink(), "units must not be a symlink")
    for parent, directories, files in os.walk(unit_root, followlinks=False):
        scanned += 1
        _require(scanned <= MAX_SCAN_DIRS, "directory scan bound exceeded")
        path = Path(parent)
        directories[:] = sorted(d for d in directories if not d.startswith(".") and not (path / d).is_symlink())
        if "COMPLETED" in files:
            _require(len(units) < MAX_UNITS, "completed unit bound exceeded")
            for name in ("manifest.json", "config.json", "summary.json", "metrics.csv", "COMPLETED"):
                artifact = path / name
                size += artifact.stat().st_size if artifact.is_file() else 0
            _require(size <= MAX_SCALAR_BYTES, "scalar byte bound exceeded")
            units.append(read_completed_unit(path))
            directories[:] = []
        elif "config.json" in files or "manifest.json" in files:
            incomplete.append(str(path))
            directories[:] = []
    units.sort(key=lambda u: (str(u["manifest"].get("completed_at", "")), u["path"]))
    orchestration = campaign / "orchestration"
    _require(not orchestration.is_symlink(), "orchestration must not be a symlink")
    progress = (_committed_progress(orchestration) if (orchestration / "CURRENT.json").exists()
                else shared._json(orchestration / "progress.json") if (orchestration / "progress.json").exists() else {})
    if progress:
        _identity(progress)
        _require(isinstance(progress.get("status"), str) and bool(progress["status"]), "progress status required")
        _require(progress.get("max_hours", 28) == 28, "ceiling must be 28 active hours")
        if "charged_seconds" in progress:
            _require(_number(progress["charged_seconds"]) and progress["charged_seconds"] >= 0, "invalid charged_seconds")
        child = progress.get("active_unit_progress", {})
        _require(isinstance(child, dict), "active_unit_progress must be an object")
        if "acquired_count" in child:
            _require(_count(child["acquired_count"]), "invalid active acquisition clock")
        preview = child.get("preview_training", [])
        _require(isinstance(preview, list) and len(preview) <= 32, "live training preview exceeds 32 rows")
        _require(all(isinstance(r, dict) and _count(r.get("n")) and _number(r.get("data")) for r in preview),
                 "invalid live training preview")
        for key in ("checkpoint_preview", "reference_preview"):
            preview = child.get(key, [])
            _require(isinstance(preview, list) and len(preview) <= 64, "invalid or oversized " + key)
            for row in preview:
                _require(isinstance(row, dict) and _count(row.get("target")) and row.get("environment") in ENVIRONMENTS
                         and _count(row.get("acquired_count", row["target"])), "invalid " + key + " identity")
                if key == "checkpoint_preview":
                    _require(_count(row.get("episodes")) and row["episodes"] > 0 and _number(row.get("success"))
                             and 0 <= row["success"] <= 1 and _number(row.get("raw_return")), "invalid checkpoint preview metrics")
                else:
                    _require(_count(row.get("records")) and row["records"] > 0 and _number(row.get("joint_nll")),
                             "invalid reference preview metrics")
    snapshot = {"root": str(campaign), "version": VERSION, "complete_units": units,
                "incomplete_units": incomplete, "progress": progress, "active_settings": {}}
    active = progress.get("active_unit")
    if isinstance(active, str) and active in incomplete:
        cfg = shared._json(Path(active) / "config.json")
        cfg = cfg.get("config", cfg)
        snapshot["active_settings"] = _mapping(cfg.get("settings"))
        snapshot["active_operation"] = cfg.get("operation")
    pairs = [(u["summary"]["settings"]["method"], u["summary"]["settings"]["seed"]) for u in _trajectories(snapshot)]
    _require(len(pairs) == len(set(pairs)), "duplicate main method/replica trajectory")
    evaluation_panels(snapshot)
    _reference_identity(snapshot)
    training_nll_curves(snapshot)
    return snapshot


def _trajectories(snapshot):
    return [u for u in snapshot["complete_units"] if u["summary"]["kind"] == "curriculum_trajectory"]


def _pairs(snapshot):
    seeds = defaultdict(set)
    for unit in _trajectories(snapshot):
        settings = unit["summary"]["settings"]
        seeds[settings["seed"]].add(settings["method"])
    return seeds


def progress_rows(snapshot):
    progress, settings = snapshot["progress"], snapshot["active_settings"]
    child = _mapping(progress.get("active_unit_progress"))
    curriculum = _mapping(child.get("curriculum"))
    charged = progress.get("charged_seconds")
    forecast = _mapping(progress.get("forecast"))
    eta = forecast.get("remaining_hours", forecast.get("remaining_hours_range"))
    main_range = forecast.get("main_hours")
    if eta is None and isinstance(main_range, list) and len(main_range) == 2 and all(_number(v) for v in main_range):
        done = len(_trajectories(snapshot))
        live_fraction = 0.
        if snapshot.get("active_operation") == "trajectory":
            n = child.get("acquired_count", 0)
            scored = sum(p.get("scored", 0) for p in child.get("evaluations", []) if isinstance(p, dict))
            live_fraction = min(.98, .9*min(1., n/2048)+.1*min(1., scored/156))
        remaining = max(0., 4-done-live_fraction)/4
        eta = [round(v*remaining, 2) for v in main_range]
    seeds = _pairs(snapshot)
    started = set(seeds)
    if snapshot.get("active_operation") == "trajectory" and _count(settings.get("seed")):
        started.add(settings["seed"])
    return [{"Status": progress.get("status", "sealed partial results" if snapshot["complete_units"] else "not yet run"),
        "Method": _LABELS.get(settings.get("method"), "shared preparation"),
        "FD Strength": settings.get("fd_strength", "not applicable"),
        "Stage": curriculum.get("name", child.get("stage", "not yet run")),
        "Within Stage": curriculum.get("within", "not collected"),
        "Fresh Observations": child.get("acquired_count", "not collected"),
        "Started / Completed Pairs": f"{len(started)} / {sum(v == set(METHODS) for v in seeds.values())}",
        "Sealed / Planned Trajectories": f"{len(_trajectories(snapshot))} / {progress.get('planned_trajectories', 4)}",
        "Charged Active Hours": charged / 3600 if charged is not None else "not collected",
        "Remaining Budget Hours": max(0., 28 - charged / 3600) if charged is not None else "not collected",
        "ETA Hours (Recorded)": eta if eta is not None else "remaining ETA not yet estimated" if forecast else "not yet profiled",
        "Whole-Main Forecast Hours": forecast.get("main_hours", "not yet profiled"),
        "Forecast (Recorded)": forecast or "not collected",
        "Active Unit": progress.get("active_unit") or "not applicable",
        "Live Preview": "provisional; not outcome evidence" if child.get("preview_training") else "not collected"}]


def _progress_table(snapshot):
    row = progress_rows(snapshot)[0]
    items = []
    for name, value in row.items():
        if name == "Forecast (Recorded)":
            continue
        if name == "ETA Hours (Recorded)":
            name = "Remaining Main Hours (Rough)"
        if name == "Active Unit" and value != "not applicable":
            value = Path(value).name
        items.append({"Item": name, "Value": value})
    return _table(items)


def training_nll_curves(snapshot, *, window=32):
    """First-step data NLL: event mean within replica/bin, equal replica mean."""
    _require(type(window) is int and window > 0, "training window must be positive integer")
    groups = defaultdict(lambda: defaultdict(list))
    for unit in _trajectories(snapshot):
        summary = unit["summary"]
        metrics = summary.get("fit_metrics", [])
        _require(isinstance(metrics, list), "fit_metrics must be a list")
        if not metrics:
            continue
        origin = summary.get("arrival_offset")
        if origin is None:
            arrivals = [r.get("arrival") for e in summary.get("episodes", []) for r in e["intervals"]]
            _require(len(arrivals) == summary["acquired_count"] and bool(arrivals)
                     and all(_count(n) for n in arrivals), "training NLL requires explicit arrival_offset or a complete captured-arrival ledger")
            origin = arrivals[0]
            _require(arrivals == list(range(origin, origin+len(arrivals))), "training arrival ledger must be contiguous")
        _require(_count(origin), "training NLL requires explicit arrival_offset")
        method, seed = (summary["settings"][key] for key in ("method", "seed"))
        seen = set()
        for event in metrics:
            _require(isinstance(event, dict), "fit event must be an object")
            steps, acquired, event_id = event.get("inner_steps"), event.get("acquired_count"), event.get("event_id")
            _require(isinstance(steps, list) and bool(steps) and isinstance(steps[0], dict) and steps[0].get("inner_step") == 1
                     and _number(steps[0].get("data")), "finite first-inner-step data NLL required")
            _require(_count(event_id) and event_id not in seen, "unique fit event_id required")
            seen.add(event_id)
            _require(_count(acquired) and 0 < acquired - origin <= summary["acquired_count"],
                     "training acquisition clock outside trajectory")
            groups[(method, (acquired - origin - 1) // window)][seed].append(steps[0]["data"])
    curves = []
    for method in METHODS:
        points = []
        for (source, index), replicas in sorted(groups.items()):
            if source == method:
                points.append({"observations": (index + .5) * window, "window_start": index * window,
                    "window_end": (index + 1) * window, "mean": _mean([_mean(v) for v in replicas.values()]),
                    "replicas": len(replicas), "events": sum(len(v) for v in replicas.values())})
        if points:
            curves.append({"method": method, "points": points})
    return curves


def _reference_identity(snapshot):
    banks, sources = [], set()
    conditioning = ("label", "action", "baseline", "bin_values", "temperature", "action_variance")
    for unit in _trajectories(snapshot):
        summary = unit["summary"]
        reference_id = _mapping(summary.get("fix4")).get("reference_id")
        if reference_id is not None:
            sources.add(reference_id)
        for checkpoint in summary.get("reference_scores", []):
            bank = {}
            for row in checkpoint["rows"]:
                _require(all(k in row for k in conditioning), "reference conditioning fields required")
                bank[row["record_id"]] = json.dumps({k: row[k] for k in
                    (*conditioning, "environment", "episode_id")}, sort_keys=True, allow_nan=False)
            banks.append(bank)
    _require(len(sources) <= 1, "main trajectories use different reference banks")
    _require(not banks or all(b == banks[0] for b in banks), "fixed-reference records/labels/conditioning differ")


def reference_nll_curves(snapshot):
    """Equal record means within replicas; no weighting by advantage precision."""
    groups = defaultdict(list)
    for unit in _trajectories(snapshot):
        summary = unit["summary"]
        for checkpoint in summary.get("reference_scores", []):
            for domain in ENVIRONMENTS:
                rows = [r for r in checkpoint["rows"] if r["environment"] == domain]
                if rows:
                    groups[(summary["settings"]["method"], domain, checkpoint["target"])].append({
                        "observations": checkpoint["acquired_count"], "records": len(rows),
                        "episodes": len({r["episode_id"] for r in rows}),
                        **{name: _mean([r[name] for r in rows]) for name in ("joint_nll", "return_nll", "action_nll")}})
    return [{"method": method, "environment": domain, "target": target,
             "replicas": len(replicas), "observations": _mean([r["observations"] for r in replicas]),
             "records": sum(r["records"] for r in replicas), "episodes": sum(r["episodes"] for r in replicas),
             **{name: _mean([r[name] for r in replicas]) for name in ("joint_nll", "return_nll", "action_nll")}}
            for (method, domain, target), replicas in sorted(groups.items())]


def evaluation_panels(snapshot):
    """Main panels and explicitly reused initialization panels, counted once.

    A common initializer is not evidence that independently acquired n0 panels
    are one shared observation. Reused panels declare shared_panel_id; a
    standalone initial_evaluation unit is its own shared source by default.
    """
    panels, initial, assets = [], {}, set()
    for unit in snapshot["complete_units"]:
        summary = unit["summary"]
        if summary["kind"] not in {"curriculum_trajectory", "curriculum_initial_evaluation"}:
            continue
        settings = summary["settings"]
        for panel in summary.get("evaluations", []):
            item = {"panel": panel, "method": settings.get("method"), "seed": settings.get("seed"), "unit": unit["path"]}
            if panel["target"] == 0:
                asset = panel.get("initial_asset_id", settings.get("initial_asset_id"))
                _require(isinstance(asset, str) and bool(asset), "initial panel requires initial_asset_id")
                assets.add(asset)
                source = panel.get("shared_panel_id")
                if source is None and summary["kind"] == "curriculum_initial_evaluation":
                    source = unit["path"]
                if source is None:
                    panels.append(item)
                    continue
                _require(isinstance(source, str) and bool(source), "shared_panel_id must be a nonempty string")
                # Exclude timing/media/episode IDs; retain task, scoring and outcomes.
                signature = {"config": panel.get("config"), "seeds": [r["seed"] for r in panel["rows"]],
                    "rows": [{k: r.get(k) for k in ("success", "raw_stage_rewards", "raw_returns", "predictive_scores",
                                                    "predictive_missing_intervals")} for r in panel["rows"]]}
                key = (source, panel["difficulty"])
                identity = json.dumps(signature, sort_keys=True, allow_nan=False)
                _require(key not in initial or initial[key][0] == identity, "conflicting shared initial panel identity")
                item.update(method=None, seed=asset)
                initial[key] = (identity, item)
            else:
                panels.append(item)
    _require(len(assets) <= 1, "different initial assets must not be pooled")
    return [value[1] for value in initial.values()] + panels


def task_curves(snapshot):
    groups = defaultdict(list)
    panels = evaluation_panels(snapshot)
    own_initial = {(p["method"], p["panel"]["difficulty"]) for p in panels
                   if p["method"] is not None and p["panel"]["target"] == 0}
    for item in panels:
        panel = item["panel"]
        rows = panel["rows"]
        scored = [r for r in rows if r["predictive_scores"]]
        point = {"observations": panel["acquired_count"], "success": _mean([float(r["success"]) for r in rows]),
                 "raw_reward": _mean([math.fsum(r["raw_stage_rewards"]) for r in rows]), "episodes": len(rows),
                 "predictive_records": sum(len(r["predictive_scores"]) for r in rows),
                 "intervals": sum(len(r["intervals"]) for r in rows),
                 **{name: _mean([_mean([s[field] for s in r["predictive_scores"]]) for r in scored])
                    for name, field in (("policy_joint_nll", "joint_nll"), ("policy_return_nll", "return_nll"))}}
        for method in METHODS if item["method"] is None else (item["method"],):
            if item["method"] is None and (method, panel["difficulty"]) in own_initial:
                continue
            groups[(method, panel["difficulty"], panel["target"])].append(point)
    return [{"method": method, "environment": domain, "target": target, "replicas": len(replicas),
             **{name: _mean([r[name] for r in replicas]) for name in ("observations", "success", "raw_reward")},
             "episodes": sum(r["episodes"] for r in replicas),
             **{name: _mean([r[name] for r in replicas if r[name] is not None])
                for name in ("policy_joint_nll", "policy_return_nll")},
             **{name: sum(r[name] for r in replicas) for name in ("predictive_records", "intervals")}}
            for (method, domain, target), replicas in sorted(groups.items())]


def outcome_rows(snapshot):
    curves = task_curves(snapshot)
    rows = []
    for method in METHODS:
        for domain in ENVIRONMENTS:
            points = [p for p in curves if p["method"] == method and p["environment"] == domain and p["target"] > 0]
            if points:
                p = max(points, key=lambda p: p["target"])
                rows.append({"Method": _LABELS[method], "Environment": domain, "Target": p["target"],
                    "Actual Observations": p["observations"], "Replicas": p["replicas"], "Episodes": p["episodes"],
                    "Success Rate": p["success"], "Raw Linear Reward": p["raw_reward"],
                    "Current Return NLL": p["policy_return_nll"] if p["policy_return_nll"] is not None else "not collected",
                    "Own-Action Joint NLL": p["policy_joint_nll"] if p["policy_joint_nll"] is not None else "not collected",
                    "Predictive Coverage": f"{p['predictive_records']}/{p['intervals']} intervals"})
    return rows


def cost_rows(snapshot):
    """Main trajectories only; shared setup has a separate narrow table."""
    rows = []
    for unit in snapshot["complete_units"]:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        operation = summary["kind"].removeprefix("curriculum_")
        if operation != "trajectory":
            continue
        rows.append({"Operation": operation, "Method": _LABELS[settings["method"]], "Seed": settings["seed"],
            "Fresh Observations": summary["acquired_count"], "Optimizer Exposures": summary.get("optimizer_exposures", "not collected"),
            "FD Strength": summary.get("fd_strength", "not collected"), "Ridge": summary.get("ridge", "not collected"),
            "First-Step EWC Events": summary.get("first_step_ewc_events", "not collected"),
            "Sealed Unit Active Hours": summary["active_seconds"] / 3600})
    return rows


def _budget_average(points, budget):
    """Trapezoidal average on [0,budget], clipping an observed final overshoot."""
    points = sorted(points)
    _require(points and points[0][0] == 0 and points[-1][0] >= budget > 0,
             "budget average requires observed coverage from zero through the budget")
    area = 0.
    for (x, y), (right, value) in zip(points, points[1:]):
        _require(right > x, "budget-average acquisition counts must increase")
        if x >= budget:
            break
        end = min(budget, right)
        endpoint = y+(value-y)*(end-x)/(right-x)
        area += .5*(y+endpoint)*(end-x)
    return area/budget


def paired_effect_rows(snapshot):
    """Differences across complete paired trajectories, never across images.

    Student intervals concern the two independent stream replicas conditional
    on their common initializer/panels. They are exploratory and assume normal
    paired effects; no layout/image pseudo-replication or significance stopping.
    """
    from statistics import stdev
    pairs = defaultdict(dict)
    for unit in _trajectories(snapshot):
        s = unit["summary"]
        pairs[s["settings"]["seed"]][s["settings"]["method"]] = s
    values = defaultdict(list)
    for methods in pairs.values():
        if set(methods) != set(METHODS):
            continue
        control, fd = methods["replay_ewc"], methods["stm_fd"]
        for environment in ENVIRONMENTS:
            effects = []
            for summary in (control, fd):
                refs = summary["reference_scores"]
                budget = max(p["target"] for p in refs)
                points = [(p["acquired_count"], _mean([r["joint_nll"] for r in p["rows"]
                    if r["environment"] == environment])) for p in refs]
                _require(all(v is not None for _, v in points), "paired reference risk requires each fixed environment")
                risk = _budget_average(points, budget)
                final = max((p for p in summary["evaluations"] if p["difficulty"] == environment),
                    key=lambda p: p["target"])
                success = _mean([float(e["success"]) for e in final["rows"]])
                effects.append((budget, risk, success))
            _require(effects[0][0] == effects[1][0], "paired budgets must match")
            values[f"{environment}: budget-averaged reference NLL"].append(effects[1][1]-effects[0][1])
            values[f"{environment}: final success (percentage points)"].append(100*(effects[1][2]-effects[0][2]))
        values["Trajectory compute multiplier (FD / control)"].append(fd["active_seconds"]/control["active_seconds"])
    rows = []
    for label, effects in values.items():
        mean = _mean(effects)
        if "success" in label:
            interval = "not estimated from two discrete panel rates"
        elif len(effects) == 2:
            # With two replicas, Student t_1 is Cauchy. Its .975 quantile is
            # exact tan(.475*pi), avoiding another reporting dependency.
            half = math.tan(.475*math.pi)*stdev(effects)/math.sqrt(2)
            interval = f"[{mean-half:.3g}, {mean+half:.3g}]"
        else:
            interval = "one pair; not estimable"
        rows.append({"Paired Metric": label, "Pairs": len(effects), "Mean FD - Control / Ratio": mean,
            "Exploratory 95% CI": interval})
    return rows


def setup_cost_rows(snapshot):
    units = [u for u in snapshot["complete_units"] if u["summary"]["kind"] != "curriculum_trajectory"]
    return [{"Scope": "Shared setup / timing / calibration (not outcomes)", "Sealed Units": len(units),
             "Sealed Unit Active Hours": math.fsum(u["summary"]["active_seconds"] for u in units) / 3600}] if units else []


def _preview_outputs(snapshot):
    child = _mapping(snapshot["progress"].get("active_unit_progress"))
    checkpoints, references = child.get("checkpoint_preview", []), child.get("reference_preview", [])
    if not checkpoints and not references:
        return []
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 2.6), layout="constrained")
    axes = figure.subplots(1, 3, squeeze=False)[0]
    actual = {r["target"]: r["acquired_count"] for r in checkpoints if "acquired_count" in r}
    has_actual = all("acquired_count" in r or r["target"] in actual for r in checkpoints+references)
    horizon = max([snapshot.get("active_settings", {}).get("budget", 2048),
        *[r.get("acquired_count", actual.get(r["target"], r["target"])) for r in checkpoints+references]])
    for domain, color in zip(ENVIRONMENTS, ("#167d91", "#b14368", "#52813d")):
        for axis, source, field in zip(axes, (checkpoints, checkpoints, references), ("success", "raw_return", "joint_nll")):
            points = sorted([r for r in source if r["environment"] == domain], key=lambda r: r["target"])
            if points:
                axis.plot([r.get("acquired_count", actual.get(r["target"], r["target"])) for r in points], [r[field] for r in points],
                          marker="o", linestyle="--", color=color, label=domain)
    for axis, label in zip(axes, ("Panel success", "Raw start return", "Reference joint NLL")):
        axis.set(xlabel="New training observations" if has_actual else "Checkpoint target observations", ylabel=label,
            xlim=(-.02*horizon, 1.02*horizon))
        axis.set_xticks([horizon*i/4 for i in range(5)])
    _finish(axes)
    return [_note("Provisional active-unit checkpoint/reference previews only. Fully scored panels, but the trajectory is incomplete; not paired outcome evidence."),
            _figure(figure, "Provisional active-unit checkpoint/reference preview; never pooled with sealed trajectories.")]


def _plots(snapshot):
    from matplotlib.figure import Figure

    outputs = []
    training = training_nll_curves(snapshot)
    preview = _mapping(snapshot["progress"].get("active_unit_progress")).get("preview_training", [])
    if training or preview:
        figure = Figure(figsize=(9, 2.6), layout="constrained")
        axis = figure.subplots()
        for curve in training:
            points, method = curve["points"], curve["method"]
            axis.plot([p["observations"] for p in points], [p["mean"] for p in points], marker=".",
                      label=_LABELS[method], color=_COLORS[method])
        if preview:
            axis.plot([r["n"] for r in preview], [r["data"] for r in preview], linestyle="--", color="#727272",
                      label="Live preview (provisional; up to 32 events)")
        axis.set(xlabel="New training observations (32-observation bins)", ylabel="First-step data NLL")
        _finish([axis])
        outputs.append(_figure(figure, "Training NLL: equal replica means; spikes retained; no penalties."))
    references, tasks = reference_nll_curves(snapshot), task_curves(snapshot)
    for mode in ("reference", "task"):
        if not (references if mode == "reference" else tasks):
            continue
        figure = Figure(figsize=(11, 4), layout="constrained")
        axes = figure.subplots(2, 3, squeeze=False)
        for column, domain in enumerate(ENVIRONMENTS):
            for row, quantity in enumerate(("joint_nll", "return_nll") if mode == "reference" else ("success", "raw_reward")):
                axis = axes[row, column]
                for method in METHODS:
                    points = sorted([p for p in (references if mode == "reference" else tasks)
                                     if p["method"] == method and p["environment"] == domain], key=lambda p: p["target"])
                    if points:
                        axis.plot([p["observations"] for p in points], [p[quantity] for p in points], marker="o",
                                  label=_LABELS[method], color=_COLORS[method])
                axis.set(title=domain if row == 0 else "", xlabel="Actual new training observations" if row else "",
                         ylabel={"joint_nll": "Reference joint NLL", "return_nll": "Reference return NLL",
                                 "success": "Success rate", "raw_reward": "Raw linear episode reward"}[quantity])
                if quantity == "success":
                    axis.set_ylim(-.03, 1.03)
        _finish(axes.flat)
        outputs.append(_figure(figure, "Fixed-reference predictive NLL, not task success." if mode == "reference"
                               else "Fixed A, B initial, B final task outcomes; equal replica means."))
    return outputs


def _finish(axes):
    for axis in axes:
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "not collected", transform=axis.transAxes, ha="center", va="center")


def _video_links(snapshot, directory, *, limit=6):
    """First available preselected endpoint clip per method/environment, bounded.

    Selection uses only checkpoint/method/environment/seed, never outcomes.
    Stat referenced files, but never read, decode or reconstruct their content.
    """
    candidates = []
    campaign = Path(snapshot["root"]).resolve()
    for unit in snapshot["complete_units"]:
        summary, settings = unit["summary"], unit["summary"]["settings"]
        videos = summary.get("videos", [])
        _require(isinstance(videos, list), "videos must be a list")
        for index, video in enumerate(videos):
            if not isinstance(video, dict) or not isinstance(video.get("path"), str):
                continue
            reference = Path(video["path"])
            possibilities = [reference] if reference.is_absolute() else [Path(unit["path"]) / reference, campaign / reference]
            path = next((p for p in possibilities if p.resolve().is_relative_to(campaign) and p.is_file()), None)
            if path is None:
                continue
            checkpoint = str(video.get("checkpoint", ""))
            target = int(checkpoint[1:]) if checkpoint.startswith("n") and checkpoint[1:].isdigit() else -1
            method = settings.get("method", summary["kind"].removeprefix("curriculum_"))
            domain = str(video.get("difficulty", "not specified"))
            key = (summary["kind"] != "curriculum_trajectory", method, domain, -target, settings.get("seed", 0), index)
            candidates.append((key, path, f"{_LABELS.get(method, method)} | {domain} | {checkpoint} | seed={settings.get('seed', 'not applicable')}"))
    links, selected = [], set()
    for key, path, label in sorted(candidates, key=lambda c: c[0]):
        group = key[:3]
        if group in selected:
            continue
        selected.add(group)
        target = quote(os.path.relpath(path.resolve(), directory), safe="/")
        links.append(f'<a href="{html.escape(target, quote=True)}">{html.escape(label)}</a>')
        if len(links) == limit:
            break
    return links


def result_outputs(snapshot, *, notebook_dir=None):
    import nbformat

    outputs = [_note("Sealed main trajectories only. Unit timing/calibration is separate; current-policy likelihood is not imitation accuracy.")]
    selections = [u["summary"] for u in snapshot["complete_units"] if u["summary"]["kind"] == "curriculum_selection"]
    if selections:
        selection = selections[-1]
        chosen = selection["selected"]
        outputs.append(_note(f"Frozen development selection: FD strength {chosen['strength']:.6g}, ridge {chosen['ridge']:.6g}; control FD=0, ridge=.1. "
            + ("A-retention matching diagnostic passed; this is not equivalence." if chosen["retention_match_diagnostic"] else
               "A retention was not matched; interpret main differences as a trade-off.")
            + " Selection used separate compressed calibration streams, not the main outcomes."))
    outputs.extend(_preview_outputs(snapshot))
    outputs.extend(_plots(snapshot))
    outcomes, costs = outcome_rows(snapshot), cost_rows(snapshot)
    outputs.append(_table(outcomes) if outcomes else _note("Main trajectory outcomes: not yet run / not collected; no zero imputation."))
    if outcomes:
        outputs.append(_note("Current-policy NLL: equal scored-interval means within episode, then equal episode and replica means. Missing predictive intervals are explicitly counted; own deterministic actions are not imitation accuracy."))
    effects = paired_effect_rows(snapshot)
    if effects:
        outputs.append(_table(effects))
        outputs.append(_note("Paired effects: lower reference NLL is better; higher success is better. Budget-averaged NLL is the trapezoidal mean across observed observation budgets, not samples saved or navigation success. Two-stream Student intervals assume normal paired effects and are highly imprecise; this remains development, not confirmation. Compute includes each trajectory's training, evaluations and scoring; shared preparation is separate."))
    outputs.append(_table(costs) if costs else _note("Sealed unit costs: not collected. Charged time comes from orchestration, not overlapping timer sums."))
    setup = setup_cost_rows(snapshot)
    if setup:
        outputs.append(_table(setup))
    directory = Path(notebook_dir or Path.cwd()).resolve()
    links = []
    for unit in snapshot["complete_units"]:
        path = Path(unit["path"]) / "summary.json"
        target = quote(os.path.relpath(path.resolve(), directory), safe="/")
        links.append(f'<a href="{html.escape(target, quote=True)}">{html.escape(Path(unit["path"]).name)} scalar summary / event diagnostics</a>')
    videos = _video_links(snapshot, directory)
    if videos:
        outputs.append(nbformat.v4.new_output("display_data", data={"text/plain": "Preselected recorded clips (diagnostics, not representative samples).",
            "text/html": "<p>Preselected recorded clips (not representative samples):<br>" + "<br>".join(videos) + "</p>"}))
    if links:
        outputs.append(nbformat.v4.new_output("display_data", data={"text/plain": "Per-record outputs and per-event diagnostics remain in linked scalar artifacts.",
            "text/html": f"<details><summary>Scalar artifacts and event diagnostics ({len(links)} units)</summary><ul>" +
                         "".join("<li>" + link + "</li>" for link in links) + "</ul></details>"}))
    return outputs


def _display(outputs):
    from IPython.display import display
    for output in outputs:
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def display_progress(snapshot):
    _display([_progress_table(snapshot)])


def display_results(snapshot, *, notebook_dir=None):
    _display(result_outputs(snapshot, notebook_dir=notebook_dir))


def conclusion(snapshot):
    trajectories = _trajectories(snapshot)
    completed_pairs = sum(methods == set(METHODS) for methods in _pairs(snapshot).values())
    rows = outcome_rows(snapshot)
    state = "Partial development results" if len(trajectories) < 4 or completed_pairs < 2 else "Sealed development trajectories"
    b = [r for r in rows if r["Environment"] != "A"]
    finding = ("No success was observed on the latest sealed B panels." if b and all(r["Success Rate"] == 0 for r in b)
               else "B capability and A retention must be read together on the measured curves." if b
               else "B capability and A retention are not collected; missing panels are not failures.")
    return (f"#### Fix 4 Conclusion\n\n**{state}:** {len(trajectories)}/4 main trajectories and {completed_pairs}/2 complete method pairs. "
            + finding + " Fixed-reference NLL measures fit to frozen off-policy histories; it does not establish better navigation. "
            "Timing and calibration results are not added to the outcome sample. No samples-saved estimate is made without an observed common capability target; "
            "flat curves are not extrapolated. Two development seeds do not establish efficacy or retention equivalence. "
            "The current progress block reports charged time and recorded forecasts separately from sealed unit costs. "
            "Phase 4 and physical robot operation remain outside this report.")


def build_notebook(root, output, base_notebook=None):
    """Preserve history cells; append tagged Fix 4, with current progress near top.

    Only own/draft Fix 4 cells are replaced. Historical progress is labeled by
    an adjacent new cell, without altering its source, outputs or metadata.
    The history snapshot and completed artifacts are never overwritten.
    """
    import nbformat

    snapshot = load_report(root)
    campaign, output = Path(snapshot["root"]), Path(output)
    base = Path(base_notebook) if base_notebook is not None else campaign / "history.ipynb"
    _require(base.resolve() != output.resolve(), "output must not overwrite history")
    _require(not output.is_symlink(), "output must not be a symlink")
    _require(not any((p / "COMPLETED").exists() for p in output.resolve().parents), "output is inside a completed unit")
    book = copy.deepcopy(nbformat.reads(shared._read(base, MAX_HISTORY_BYTES).decode("utf-8"), as_version=4))
    history = [c for c in book.cells if not c.metadata.get(_TAG) and not c.metadata.get(_DRAFT_TAG)]
    book.cells = []
    for index, cell in enumerate(history):
        if cell.metadata.get("report_section") == "progress" or "progress" in cell.get("id", ""):
            note = nbformat.v4.new_markdown_cell("**Historical progress snapshot:** these earlier counters are superseded for current status by the Fix 4 block above.")
            note.metadata[_TAG] = True
            note.id = f"plan19-phase3-fix4-history-{index}"
            book.cells.append(note)
        book.cells.append(cell)
    bootstrap = ("from pathlib import Path\nimport sys\n"
        "for parent in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (parent / 'numerical_experiment/picar_curriculum_repair/report.py').is_file():\n"
        "        sys.path.insert(0, str(parent))\n        break\n"
        "else:\n    raise RuntimeError('Run this notebook from within the project checkout.')\n")
    code = ("from numerical_experiment.picar_curriculum_repair import report as curriculum_report\n"
            f"curriculum_snapshot = curriculum_report.load_report({str(campaign.resolve())!r})\n")
    progress = nbformat.v4.new_code_cell(bootstrap + code + "curriculum_report.display_progress(curriculum_snapshot)",
                                       outputs=[_progress_table(snapshot)])
    progress.metadata[_TAG], progress.id = True, "plan19-phase3-fix4-progress"
    book.cells.insert(min(1, len(book.cells)), progress)
    diagram = Path(__file__).resolve().parents[2] / "docs/figures/plan19_phase3_fix4_transition.png"
    section = [nbformat.v4.new_markdown_cell(_INTRO.format(diagram=quote(os.path.relpath(diagram, output.parent.resolve()), safe="/"))),
        nbformat.v4.new_code_cell(code + f"curriculum_report.display_results(curriculum_snapshot, notebook_dir={str(output.parent.resolve())!r})",
                                 outputs=result_outputs(snapshot, notebook_dir=output.parent)),
        nbformat.v4.new_markdown_cell(conclusion(snapshot))]
    for index, cell in enumerate(section):
        cell.metadata[_TAG], cell.id = True, f"plan19-phase3-fix4-{index}"
    book.cells.extend(section)
    nbformat.validate(book)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".curriculum-report-", suffix=".ipynb", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        nbformat.write(book, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
