"""Foreground, resumable Plan 16 scheduler; no agent connection is required."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit, utc_now
from .config import Config, content_hash
from .curvature import CONDITIONS
from .data import load_data, make_assets
from .runtime import REPO, open_store, read_json
from .trajectory import initialize, run_trajectory


def stream_view(assets, schedule):
    prefix = "stream" if schedule == "main" else "stationary_stream"
    segment_names = [x["name"] for x in assets[f"{prefix}_segments"]]
    return {"ids": assets[f"{prefix}_ids"], "p": assets[f"{prefix}_p"],
            "segments": [segment_names[int(i)] for i in assets[f"{prefix}_segment_ids"]],
            "expected_cumulative_nines": assets[f"{prefix}_expected_cumulative_9s"]}


def averaged(rows, name):
    area = sum((b["arrivals"] - a["arrivals"]) * (a[name] + b[name]) / 2
               for a, b in zip(rows, rows[1:]))
    return area / rows[-1]["arrivals"]


class Scheduler:
    def __init__(self, cfg, store, budget, data_root, *, max_units=None):
        self.cfg, self.store, self.budget = cfg, store, budget
        self.data = load_data(data_root)
        self.max_units, self.new_units = max_units, 0
        self.assets_cache, self.initial_cache = {}, {}
        self.last_pulse = self.last_report = 0.
        self.phase, self.current = "acceptance", {}
        self.complete = {}
        self.selection = read_json(store.root / "selection.json") if (store.root / "selection.json").exists() else {}
        self.changes = read_json(store.root / "changes.json") if (store.root / "changes.json").exists() else []
        for record in store.iter_units(completed_only=True):
            if record["metadata"].get("kind") == "trajectory":
                uid = record["unit_id"]
                self.complete[uid] = self.read_trajectory(uid, record["metadata"])

    def read_trajectory(self, uid, metadata):
        path = self.store.unit_path(uid)
        return {"unit_id": uid, **metadata, "completed": True,
                "metrics": read_json(path / "metrics.json"),
                "summary": read_json(path / "summary.json")}

    def pulse(self, *, force=False, report=False, **info):
        self.current.update({k: v for k, v in info.items() if k != "accepted_metrics"})
        if "accepted_metrics" in info:
            self.current_metrics = info["accepted_metrics"]
        now = time.monotonic()
        if not force and now - self.last_pulse < 10:
            return
        self.budget.pulse(force=True)
        block_arms = {}
        for item in self.complete.values():
            if item["stage"] == "fresh":
                block_arms.setdefault((item["schedule"], item["replica"]), set()).add(item["condition"])
        required = set(self.selection.get("conditions", CONDITIONS))
        completed_blocks = sum(required <= arms for arms in block_arms.values())
        planned_blocks = (self.selection.get("main_replicas", 0) +
                          self.selection.get("stationary_replicas", 0)) or None
        fresh_target = self.selection.get("fresh_trajectory_target")
        development_count = sum(t["stage"] == "development" for t in self.complete.values())
        remaining_forecast = None
        if fresh_target is not None:
            costs = self.selection["seconds_by_condition"]
            remaining_forecast = sum(max(0, self.selection["main_replicas"] + self.selection["stationary_replicas"] -
                                        sum(t["stage"] == "fresh" and t["condition"] == cond
                                            for t in self.complete.values())) * costs[cond] for cond in required)
            if getattr(self, "current_metadata", None) and self.current_metadata["stage"] == "fresh":
                remaining_forecast -= costs[self.current_metadata["condition"]] * min(1.,
                    self.current.get("step", 0) / max(1, self.current.get("total_steps", 1)))
        progress = {"phase": self.phase, "status": "running", "heartbeat": utc_now(),
                    "completed_trajectories": len(self.complete),
                    "completed_fresh": sum(t["stage"] == "fresh" for t in self.complete.values()),
                    "planned_fresh": fresh_target,
                    "planned_trajectories": None if fresh_target is None else development_count + fresh_target,
                    "completed_blocks": completed_blocks, "planned_blocks": planned_blocks,
                    "forecast_seconds": remaining_forecast,
                    "revision": f"{self.cfg.protocol}/{self.store.root.name}",
                    "active_hours": self.budget.used / 3600,
                    "remaining_hours": self.budget.remaining / 3600,
                    "allocation_hours": self.budget.allocation_seconds / 3600,
                    "storage": self.store.storage_status(), **self.current}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.last_pulse = now
        if report or now - self.last_report >= 900:
            self.publish(progress)

    def publish(self, progress=None, *, status="running"):
        progress = progress or (read_json(self.store.root / "progress.json") if
                                (self.store.root / "progress.json").exists() else {})
        progress.update(status=status, phase=self.phase, heartbeat=utc_now(), active_hours=self.budget.used / 3600)
        items = list(self.complete.values())
        active = self.current.get("active_unit")
        if active and active not in self.complete and getattr(self, "current_metadata", None) is not None:
            rows = getattr(self, "current_metrics", [])
            if rows:
                items.append({"unit_id": active, **self.current_metadata, "completed": False,
                              "metrics": rows, "summary": {}})
        from .analysis import summarize
        from .report import build_report
        analysis = summarize(items, bootstrap_draws=self.cfg.bootstrap_draws, seed=self.cfg.seed)
        snapshot = {"schema": 1, "study_id": self.store.root.name, "status": status,
                    "phase": self.phase, "config": self.cfg.mapping(), "progress": progress,
                    "changes": self.changes, "trajectories": items, "selection": self.selection,
                    "analysis": analysis}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        with self.store.reserve(32 * 1024**2, finalizing=True):
            build_report(self.store.root)
        self.last_report = time.monotonic()
        self.store.reconcile(persist=True, finalizing=True)

    def count_unit(self):
        self.new_units += 1
        if self.max_units is not None and self.new_units >= self.max_units:
            raise StopRequested("max-units reached after a sealed unit")

    def assets(self, cohort, replica):
        key = (cohort, replica)
        if key not in self.assets_cache:
            uid = f"assets-{cohort}-r{replica:03}"
            metadata = {"kind": "assets", "cohort": cohort, "replica": replica}
            path = self.store.begin(uid, metadata)
            if path is not None:
                assets = make_assets(self.cfg, replica, cohort, self.data)
                assets["non9_weights"] = assets["non9_probabilities"]
                self.store.save_checkpoint(uid, assets)
                self.store.write_json(uid, "summary.json", assets["metadata"])
                self.store.complete(uid)
                self.count_unit()
            self.assets_cache[key] = self.store.load_checkpoint(uid)
        return self.assets_cache[key]

    def initial(self, cohort, replica):
        key = (cohort, replica)
        if key not in self.initial_cache:
            assets = self.assets(cohort, replica)
            uid = f"initial-{cohort}-r{replica:03}"
            was_complete = self.store.is_complete(uid)
            self.initial_cache[key] = initialize(self.store, self.budget, self.cfg, uid,
                                                 assets, replica, cohort, pulse=self.pulse)
            if not was_complete:
                self.count_unit()
        return self.initial_cache[key]

    def trajectory(self, cohort, replica, schedule, condition, c, ridge):
        meta = {"kind": "trajectory", "stage": "development" if cohort == "development" else "fresh",
                "replica": replica, "schedule": schedule, "condition": condition,
                "c": c, "ridge": ridge, "cohort": cohort}
        uid = f"{meta['stage']}-{schedule}-r{replica:03}-{condition}-{content_hash(meta)[:10]}"
        if uid in self.complete:
            return self.complete[uid]
        initial, assets = self.initial(cohort, replica), self.assets(cohort, replica)
        meta["asset_unit"] = f"assets-{cohort}-r{replica:03}"
        meta["initial_unit"] = f"initial-{cohort}-r{replica:03}"
        meta["stream_hash"] = assets["metadata"]["streams"]["stream" if schedule == "main" else "stationary_stream"]["stream_hash"]
        self.current_metadata, self.current_metrics = meta, []
        self.current = {"active_unit": uid}
        self.pulse(force=True)
        run_trajectory(self.store, self.budget, self.cfg, uid, meta, initial, assets,
                       stream_view(assets, schedule), self.data["train_x"], self.data["train_y"], pulse=self.pulse)
        self.complete[uid] = self.read_trajectory(uid, meta)
        self.current_metadata, self.current_metrics = None, []
        self.pulse(force=True, report=True, active_unit=None)
        summary = self.complete[uid]["summary"]
        print(json.dumps({"completed": uid, "seconds": summary["wall_seconds"],
                          "nine_recall": summary["final_nine_recall"],
                          "old_accuracy": summary["final_old_accuracy"]}), flush=True)
        self.count_unit()
        return self.complete[uid]

    def save_selection(self):
        self.store.write_json(None, "selection.json", self.selection)
        self.pulse(force=True, report=True)

    def run(self):
        cfg = self.cfg
        self.phase = "1: resource pilot"
        self.pulse(force=True, report=True)
        pilot_c = .025 if .025 in cfg.control_cs else cfg.control_cs[0]
        for condition in CONDITIONS:
            self.trajectory("development", 0, "main", condition, pilot_c, cfg.ridge_ratio)
        self.phase = "2: development"
        if "common_c" not in self.selection:
            controls = [self.trajectory("development", replica, "main", "ema", c, cfg.ridge_ratio)
                        for replica in range(max(1, cfg.dev_replicas)) for c in cfg.control_cs]
            candidates = []
            for c in cfg.control_cs:
                same = [t for t in controls if t["c"] == c]
                candidates.append({"c": c, "replicas": len(same),
                                   "nine_recall": statistics.mean(averaged(t["metrics"], "nine_recall") for t in same),
                                   "old_accuracy": statistics.mean(averaged(t["metrics"], "old_accuracy") for t in same),
                                   "seconds": statistics.mean(t["summary"]["wall_seconds"] for t in same)})
            best = max(candidates, key=lambda item: (.5 * (item["nine_recall"] + item["old_accuracy"]),
                                                     -item["seconds"]))
            self.selection.update(common_c=best["c"], common_ridge=cfg.ridge_ratio,
                                  selection_stage="development",
                                  heuristic="equal average acquisition and retention; cost tie-break",
                                  candidates=candidates)
            self.save_selection()
        c, ridge = self.selection["common_c"], self.selection["common_ridge"]
        for replica in range(max(1, cfg.dev_replicas)):
            for condition in CONDITIONS:
                self.trajectory("development", replica, "main", condition, c, ridge)
        for condition in CONDITIONS:
            self.trajectory("development", 0, "stationary", condition, c, ridge)
        if "main_replicas" not in self.selection:
            durations = {cond: statistics.median([t["summary"]["wall_seconds"] for t in self.complete.values()
                         if t["condition"] == cond]) for cond in CONDITIONS}
            init_times = [v["wall_seconds"] for v in self.initial_cache.values()]
            init_seconds = statistics.median(init_times) if init_times else 120.
            conditions = list(CONDITIONS)
            available = max(0., self.budget.remaining - 1800.) / 1.25
            nmain, nstation = cfg.main_replicas, cfg.stationary_replicas
            def forecast():
                return (nmain + nstation) * sum(durations[k] for k in conditions) + max(nmain, nstation) * init_seconds
            while forecast() > available and nmain + nstation > 2:
                if nmain > 2 * max(nstation, 1):
                    nmain -= 1
                elif nstation > 1:
                    nstation -= 1
                else:
                    nmain -= 1
            if forecast() > available:
                conditions.remove("analytic_lfu")
            if forecast() > available:
                raise ResourceLimit("remaining allocation cannot cover a full fresh paired block and closeout")
            self.selection.update(main_replicas=nmain, stationary_replicas=nstation,
                                  conditions=conditions, seconds_by_condition=durations,
                                  forecast_fresh_seconds=forecast(), contingency=1.25,
                                  fresh_trajectory_target=(nmain + nstation) * len(conditions),
                                  frozen_at=utc_now())
            self.save_selection()
        self.phase = "3: fresh paired exploration"
        for replica in range(max(self.selection["main_replicas"], self.selection["stationary_replicas"])):
            for schedule, count in (("main", self.selection["main_replicas"]),
                                    ("stationary", self.selection["stationary_replicas"])):
                if replica >= count:
                    continue
                conditions = self.selection["conditions"]
                order = conditions[replica % len(conditions):] + conditions[:replica % len(conditions)]
                for condition in order:
                    self.trajectory("fresh", replica, schedule, condition, c, ridge)
        self.phase = "4: closeout"
        self.pulse(force=True)
        self.publish(status="completed")
        self.store.write_json(None, "study_completed.json", {"completed_at": utc_now(),
                              "active_hours": self.budget.used / 3600, "selection": self.selection})
        print(json.dumps({"status": "completed", "active_hours": self.budget.used / 3600,
                          "root": str(self.store.root)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO / "cache/mnist_experiment/datasets")
    parser.add_argument("--allocation-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.allocation_hours <= 20:
        parser.error("Plan 16's authorized cumulative allocation is at most 20 hours")
    cfg = Config.from_json(args.config)
    store, budget = open_store(cfg, args.root, args.data_root, resume=args.resume,
                               clear_stop=args.clear_stop, allocation_hours=args.allocation_hours,
                               max_wall_seconds=args.max_wall_seconds)
    scheduler = None
    try:
        with store.control.signal_handlers():
            if (store.root / "study_completed.json").is_file():
                print(json.dumps({"status": "already_completed", "root": str(store.root),
                                  "active_hours": budget.used / 3600}), flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, args.data_root, max_units=args.max_units)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler is not None:
            scheduler.publish(status="paused")
        print(json.dumps({"status": "paused", "reason": str(error), "active_hours": budget.used / 3600}), flush=True)
        return 0
    finally:
        try:
            budget.close()
            store.close()
        finally:
            store.plan16_allocation_lock.close()


if __name__ == "__main__":
    main()
