"""Single-writer, resumable execution of all authorized Plan 17 phases."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit, utc_now
from .config import Config, content_hash
from .data import load_data, make_assets
from .policies import PolicySpec
from .runtime import REPO, open_store, read_json
from .trajectory import initialize, calibrate, run_trajectory


def stream_view(assets, schedule):
    prefix = "stream" if schedule == "main" else "stationary_stream"
    names = [s["name"] for s in assets[f"{prefix}_segments"]]
    return {"ids": assets[f"{prefix}_ids"], "p": assets[f"{prefix}_p"],
            "segments": [names[int(i)] for i in assets[f"{prefix}_segment_ids"]],
            "expected_cumulative_nines": assets[f"{prefix}_expected_cumulative_9s"]}


def averaged(rows, name):
    return sum((b["arrivals"] - a["arrivals"]) * (a[name] + b[name]) / 2
               for a, b in zip(rows, rows[1:])) / rows[-1]["arrivals"]


def accuracy_rank(trajectories):
    return statistics.mean(.5 * (averaged(t["metrics"], "nine_recall") +
                                 averaged(t["metrics"], "old_accuracy")) for t in trajectories)


def shadow_records(unit_id, metadata, metrics, *, completed):
    records = []
    for condition, rows in metrics.items():
        segments = sorted({r["segment"] for r in rows})
        scores = {segment: statistics.mean(r["innovation_log_score"] for r in rows if r["segment"] == segment)
                  for segment in segments}
        summary = {"regime_log_scores": scores,
                   "log_score": statistics.mean(scores.values()) if scores else 0.,
                   "pi_mean": statistics.mean(r["applied_pi"] for r in rows) if rows else 0.,
                   "innovation_ratio": statistics.mean(r["innovation_ratio"] for r in rows) if rows else 0.,
                   "boundary_fraction": statistics.mean(r["boundary_low"] or r["boundary_high"] for r in rows) if rows else 0.,
                   "mapping_error_ratio_mean": statistics.mean(r["mapping_error_ratio"] for r in rows) if rows else 0.}
        records.append({"unit_id": unit_id + "-" + condition, "driver_unit": unit_id,
                        "replica": metadata["replica"], "schedule": metadata["schedule"],
                        "condition": condition, "completed": completed, "metrics": rows, "summary": summary})
    return records


class Scheduler:
    def __init__(self, cfg, store, budget, data_root, *, max_units=None):
        self.cfg, self.store, self.budget = cfg, store, budget
        self.data = load_data(data_root)
        self.max_units, self.new_units = max_units, 0
        self.assets_cache, self.initial_cache, self.calibration_cache = {}, {}, {}
        self.initial_times = {}
        self.last_pulse = self.last_report = 0.
        self.phase, self.current, self.current_metadata = "0: acceptance", {}, None
        self.current_metrics, self.current_shadows = [], {}
        self.complete, self.drivers, self.shadows = {}, {}, []
        self.selection = read_json(store.root / "selection.json") if (store.root / "selection.json").exists() else {}
        self.changes = read_json(store.root / "changes.json") if (store.root / "changes.json").exists() else []
        for record in store.iter_units(completed_only=True):
            kind = record["metadata"].get("kind")
            if kind == "initialization":
                self.initial_times[record["unit_id"]] = read_json(store.unit_path(record["unit_id"]) / "summary.json")["wall_seconds"]
            if kind not in ("trajectory", "shadow_driver"):
                continue
            uid, metadata = record["unit_id"], record["metadata"]
            item = self.read_trajectory(uid, metadata)
            if kind == "trajectory":
                self.complete[uid] = item
            else:
                self.drivers[uid] = item
                raw = read_json(store.unit_path(uid) / "shadow_metrics.json")
                self.shadows.extend(shadow_records(uid, metadata, raw, completed=True))

    def read_trajectory(self, uid, metadata):
        path = self.store.unit_path(uid)
        return {"unit_id": uid, **metadata, "completed": True,
                "metrics": read_json(path / "metrics.json"), "summary": read_json(path / "summary.json")}

    def pulse(self, *, force=False, report=False, **info):
        self.current.update({k: v for k, v in info.items() if k not in ("accepted_metrics", "accepted_shadow_metrics")})
        if "accepted_metrics" in info:
            self.current_metrics = info["accepted_metrics"]
        if "accepted_shadow_metrics" in info:
            self.current_shadows = info["accepted_shadow_metrics"]
        now = time.monotonic()
        if not force and now - self.last_pulse < 10:
            return
        self.budget.pulse(force=True)
        blocks = {}
        for item in self.complete.values():
            if item["stage"] == "fresh":
                blocks.setdefault((item["schedule"], item["replica"]), set()).add(item["condition"])
        required = set(self.selection.get("conditions", []))
        fresh_target = self.selection.get("fresh_trajectory_target")
        fresh = sum(t["stage"] == "fresh" for t in self.complete.values())
        forecast = None
        if fresh_target is not None:
            costs = self.selection["seconds_by_condition"]
            forecast = sum((self.selection["main_replicas"] + self.selection["stationary_replicas"] -
                            sum(t["stage"] == "fresh" and t["condition"] == condition for t in self.complete.values())) *
                           costs[condition] * max(self.cfg.schedule_factors) for condition in required)
            if self.current_metadata and self.current_metadata.get("stage") == "fresh":
                forecast -= costs[self.current_metadata["condition"]] * max(self.cfg.schedule_factors) * min(
                    1., self.current.get("step", 0) / max(1, self.current.get("total_steps", 1)))
        progress = {"phase": self.phase, "status": "running", "heartbeat": utc_now(),
                    "completed_trajectories": len(self.complete), "completed_fresh": fresh,
                    "planned_fresh": fresh_target,
                    "planned_trajectories": None if fresh_target is None else len(self.complete) - fresh + fresh_target,
                    "completed_blocks": sum(required <= arms for arms in blocks.values()) if required else 0,
                    "planned_blocks": self.selection.get("main_replicas", 0) + self.selection.get("stationary_replicas", 0) or None,
                    "completed_shadow_drivers": len(self.drivers),
                    "forecast_seconds": max(0., forecast) if forecast is not None else None,
                    "revision": f"{self.cfg.protocol}/{self.store.root.name}",
                    "active_hours": self.budget.used / 3600, "remaining_hours": self.budget.remaining / 3600,
                    "allocation_hours": self.budget.allocation_seconds / 3600,
                    "storage": self.store.storage_status(), **self.current}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.last_pulse = now
        if report or now - self.last_report >= 900:
            self.publish(progress)

    def publish(self, progress=None, *, status="running"):
        progress = dict(progress or (read_json(self.store.root / "progress.json") if
                        (self.store.root / "progress.json").exists() else {}))
        progress.update(status=status, phase=self.phase, heartbeat=utc_now(), active_hours=self.budget.used / 3600)
        items, shadows = list(self.complete.values()), list(self.shadows)
        uid, metadata = self.current.get("active_unit"), self.current_metadata
        if uid and metadata and uid not in self.complete and uid not in self.drivers:
            if metadata["kind"] == "trajectory" and self.current_metrics:
                items.append({"unit_id": uid, **metadata, "completed": False,
                              "metrics": self.current_metrics, "summary": {}})
            elif metadata["kind"] == "shadow_driver" and self.current_shadows:
                shadows.extend(shadow_records(uid, metadata, self.current_shadows, completed=False))
        from .analysis import summarize
        from .report import build_report
        analysis = summarize(items, bootstrap_draws=self.cfg.bootstrap_draws, seed=self.cfg.seed,
                             required_conditions=self.selection.get("conditions"))
        snapshot = {"schema": 1, "study_id": self.store.root.name, "status": status, "phase": self.phase,
                    "config": self.cfg.mapping(), "progress": progress, "selection": self.selection,
                    "changes": self.changes, "trajectories": items, "shadows": shadows, "analysis": analysis}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        with self.store.reserve(32 * 1024**2, finalizing=True):
            build_report(self.store.root, self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()
        self.store.reconcile(persist=True, finalizing=True)

    def count_unit(self):
        self.new_units += 1
        if self.max_units is not None and self.new_units >= self.max_units:
            raise StopRequested("max-units reached after a sealed unit")

    def assets(self, cohort, replica):
        key = cohort, replica
        if key not in self.assets_cache:
            uid = f"assets-{cohort}-r{replica:03}"
            path = self.store.begin(uid, {"kind": "assets", "cohort": cohort, "replica": replica})
            if path is not None:
                assets = make_assets(self.cfg, replica, cohort, self.data)
                self.store.save_checkpoint(uid, assets)
                self.store.write_json(uid, "summary.json", assets["metadata"])
                self.store.complete(uid)
                self.count_unit()
            self.assets_cache[key] = self.store.load_checkpoint(uid)
        return self.assets_cache[key]

    def initial(self, cohort, replica):
        key = cohort, replica
        if key not in self.initial_cache:
            assets = self.assets(cohort, replica)
            uid = f"initial-{cohort}-r{replica:03}"
            done = self.store.is_complete(uid)
            self.initial_cache[key] = initialize(self.store, self.budget, self.cfg, uid, assets,
                                                 replica, cohort, pulse=self.pulse)
            self.initial_times[uid] = self.initial_cache[key]["wall_seconds"]
            if not done:
                self.count_unit()
        return self.initial_cache[key]

    def calibration(self, cohort, replica):
        key = cohort, replica
        if key not in self.calibration_cache:
            initial, assets = self.initial(cohort, replica), self.assets(cohort, replica)
            uid = f"calibration-{cohort}-r{replica:03}"
            done = self.store.is_complete(uid)
            self.calibration_cache[key] = calibrate(self.store, self.budget, self.cfg, uid,
                                                    initial, assets, pulse=self.pulse)
            if not done:
                self.count_unit()
        return self.calibration_cache[key]

    def trajectory(self, cohort, replica, schedule, condition, c, *, shadow_specs=()):
        kind = "shadow_driver" if shadow_specs else "trajectory"
        meta = {"kind": kind, "stage": "development" if cohort == "development" else "fresh",
                "cohort": cohort, "replica": replica, "schedule": schedule, "condition": condition,
                "c": c, "ridge": self.cfg.ridge_ratio, "anchor_refresh": self.cfg.anchor_refresh,
                "shadow_conditions": [s.condition for s in shadow_specs]}
        uid = f"{'shadow' if shadow_specs else meta['stage']}-{schedule}-r{replica:03}-{condition}-{content_hash(meta)[:10]}"
        target = self.drivers if shadow_specs else self.complete
        if uid in target:
            return target[uid]
        initial, assets = self.initial(cohort, replica), self.assets(cohort, replica)
        calibration = self.calibration(cohort, replica) if shadow_specs or PolicySpec.parse(condition).movement else None
        meta.update(asset_unit=f"assets-{cohort}-r{replica:03}", initial_unit=f"initial-{cohort}-r{replica:03}",
                    schedule_factor=assets["schedule_factor"],
                    stream_hash=assets["metadata"]["streams"]["stream" if schedule == "main" else "stationary_stream"]["stream_hash"])
        self.current, self.current_metadata = {"active_unit": uid}, meta
        self.current_metrics, self.current_shadows = [], {}
        self.pulse(force=True)
        run_trajectory(self.store, self.budget, self.cfg, uid, meta, initial, assets, stream_view(assets, schedule),
                       self.data["train_x"], self.data["train_y"], calibration=calibration,
                       shadow_specs=shadow_specs, pulse=self.pulse)
        target[uid] = self.read_trajectory(uid, meta)
        if shadow_specs:
            raw = read_json(self.store.unit_path(uid) / "shadow_metrics.json")
            self.shadows.extend(shadow_records(uid, meta, raw, completed=True))
        self.current_metadata, self.current_metrics, self.current_shadows = None, [], {}
        self.pulse(force=True, report=True, active_unit=None)
        summary = target[uid]["summary"]
        print({"completed": uid, "seconds": summary["wall_seconds"],
               "nine_recall": summary["final_nine_recall"], "old_accuracy": summary["final_old_accuracy"]}, flush=True)
        self.count_unit()
        return target[uid]

    def save_selection(self):
        self.store.write_json(None, "selection.json", self.selection)
        self.pulse(force=True, report=True)

    def run(self):
        cfg = self.cfg
        self.phase = "1: common calibration and timing"
        self.pulse(force=True, report=True)
        pilot_c = .025 if .025 in cfg.control_cs else cfg.control_cs[0]
        self.trajectory("development", 0, "main", "fixed", pilot_c)
        shadow_specs = [*[PolicySpec("ema", h) for h in cfg.ema_half_lives],
                        *[PolicySpec("kalman", eta) for eta in cfg.process_scales]]
        for replica in range(cfg.shadow_replicas):
            for schedule in ("main", "stationary"):
                self.trajectory("development", replica, schedule, "fixed", pilot_c, shadow_specs=shadow_specs)
        self.phase = "2: bounded observer development"
        if "common_c" not in self.selection:
            controls = [self.trajectory("development", r, "main", "fixed", c)
                        for r in range(max(1, cfg.dev_replicas)) for c in cfg.control_cs]
            candidates = [{"c": c, "score": accuracy_rank([t for t in controls if t["c"] == c]),
                           "nine_recall": statistics.mean(averaged(t["metrics"], "nine_recall") for t in controls if t["c"] == c),
                           "old_accuracy": statistics.mean(averaged(t["metrics"], "old_accuracy") for t in controls if t["c"] == c),
                           "seconds": statistics.mean(t["summary"]["wall_seconds"] for t in controls if t["c"] == c)}
                          for c in cfg.control_cs]
            best = max(candidates, key=lambda t: (t["score"], -t["seconds"]))
            self.selection.update(common_c=best["c"], common_ridge=cfg.ridge_ratio, candidates=candidates,
                                  heuristic="equal observation-averaged acquisition and retention; cost tie-break")
            self.save_selection()
        c = self.selection["common_c"]
        if "shadow_finalists" not in self.selection:
            scores = []
            for spec in shadow_specs:
                rows = [s for s in self.shadows if s["condition"] == spec.condition]
                scores.append({"condition": spec.condition,
                               "log_score": statistics.mean(s["summary"]["log_score"] for s in rows),
                               "innovation_ratio": statistics.mean(s["summary"]["innovation_ratio"] for s in rows),
                               "boundary_fraction": statistics.mean(s["summary"]["boundary_fraction"] for s in rows),
                               "mapping_error_ratio_mean": statistics.mean(s["summary"]["mapping_error_ratio_mean"] for s in rows)})
            finalists = []
            for kind in ("ema", "kalman"):
                family = [s for s in scores if PolicySpec.parse(s["condition"]).kind == kind]
                finalists += [s["condition"] for s in sorted(family, key=lambda s: s["log_score"], reverse=True)[:cfg.candidates_per_family]]
            self.selection.update(shadow_scores=scores, shadow_finalists=finalists)
            self.save_selection()
        finalists = self.selection["shadow_finalists"]
        for replica in range(max(1, cfg.dev_replicas)):
            for condition in ("fixed", "covariance", *finalists):
                self.trajectory("development", replica, "main", condition, c)
        for condition in ("fixed", "covariance", *finalists):
            self.trajectory("development", 0, "stationary", condition, c)
        if "conditions" not in self.selection:
            rows = [t for t in self.complete.values() if t["stage"] == "development" and t["c"] == c]
            chosen = []
            for kind in ("ema", "kalman"):
                candidates = [condition for condition in finalists if PolicySpec.parse(condition).kind == kind]
                def rank(condition):
                    paths = [t for t in rows if t["condition"] == condition]
                    return accuracy_rank(paths), -statistics.mean(t["summary"]["wall_seconds"] for t in paths)
                chosen.append(max(candidates, key=rank))
            self.selection.update(conditions=["fixed", "covariance", *chosen],
                                  observer_selection="equal acquisition/retention average across main/stationary development paths")
            self.save_selection()
        if "main_replicas" not in self.selection:
            conditions = self.selection["conditions"]
            durations = {condition: statistics.median(t["summary"]["wall_seconds"] for t in self.complete.values()
                         if t["condition"] == condition and t["c"] == c) for condition in conditions}
            available = max(0., self.budget.remaining - 1800.) / 1.25
            nmain, nstationary = cfg.main_replicas, cfg.stationary_replicas
            init_times = list(self.initial_times.values())
            initialization = statistics.median(init_times) if init_times else 120.
            def forecast():
                return (nmain + nstationary) * max(cfg.schedule_factors) * sum(durations.values()) + max(nmain, nstationary) * initialization
            while forecast() > available and nmain + nstationary > 2:
                if nstationary > 1 and nmain <= 2 * nstationary:
                    nstationary -= 1
                else:
                    nmain -= 1
            if forecast() > available:
                raise ResourceLimit("remaining allocation cannot cover a complete four-policy paired block")
            self.selection.update(main_replicas=nmain, stationary_replicas=nstationary,
                                  seconds_by_condition=durations, forecast_fresh_seconds=forecast(), contingency=1.25,
                                  fresh_trajectory_target=(nmain + nstationary) * len(conditions), frozen_at=utc_now())
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
                    self.trajectory("fresh", replica, schedule, condition, c)
        self.phase = "4: closeout"
        self.pulse(force=True)
        self.publish(status="completed")
        self.store.write_json(None, "study_completed.json", {"completed_at": utc_now(),
                              "active_hours": self.budget.used / 3600, "selection": self.selection})
        print({"status": "completed", "active_hours": self.budget.used / 3600}, flush=True)


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
        parser.error("Plan 17's cumulative allocation must be in (0, 20] hours")
    cfg = Config.from_json(args.config)
    store, budget = open_store(cfg, args.root, args.data_root, resume=args.resume, clear_stop=args.clear_stop,
                               allocation_hours=args.allocation_hours, max_wall_seconds=args.max_wall_seconds)
    scheduler = None
    try:
        with store.control.signal_handlers():
            if (store.root / "study_completed.json").exists():
                print({"status": "already_completed", "active_hours": budget.used / 3600}, flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, args.data_root, max_units=args.max_units)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler is not None:
            scheduler.publish(status="paused")
        print({"status": "paused", "reason": str(error), "active_hours": budget.used / 3600}, flush=True)
        return 0
    finally:
        try:
            budget.close()
            store.close()
        finally:
            store.plan17_allocation_lock.close()


if __name__ == "__main__":
    main()
