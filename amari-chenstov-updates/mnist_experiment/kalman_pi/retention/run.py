"""Autonomous, resumable Plan 17 Phase 5 calibration and fresh confirmation."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.kalman_pi import run as original
from mnist_experiment.kalman_pi.policies import PolicySpec
from mnist_experiment.kalman_pi.runtime import read_json, file_digest
from mnist_experiment.kalman_pi.trajectory import run_trajectory
from .config import Config, CONDITIONS, content_hash
from .data import make_assets
from .runtime import REPO, open_store


class Scheduler(original.Scheduler):
    def __init__(self, cfg, store, budget, data_root, *, historical_root=None, max_units=None):
        super().__init__(cfg, store, budget, data_root, max_units=max_units)
        self.phase = "5A: recovery acceptance"
        self.failures = read_json(store.root / "failed_candidates.json") if (store.root / "failed_candidates.json").exists() else {}
        if store.is_complete("frozen-selection"):
            sealed = read_json(store.unit_path("frozen-selection") / "selection.json")
            if (store.root / "selection.json").exists() and self.selection != sealed:
                raise ValueError("the mutable selection differs from its immutable decision artifact")
            self.selection = sealed
            if not (store.root / "selection.json").exists():
                store.write_json(None, "selection.json", sealed)
        if not self.selection:
            self.selection = {"conditions": list(CONDITIONS), "common_c": .025,
                "reference_ridge": .1, "tolerance_pp": cfg.tolerance_pp,
                "development_replicas": {"main": cfg.development_main, "stationary": cfg.development_stationary},
                "initial_ridges": list(cfg.ridges), "refinement_jobs": [], "ridges": {},
                "development_selection": {}, "bootstrap_seed": cfg.seed_for("analysis", 0, "bootstrap")}
            store.write_json(None, "selection.json", self.selection)
        historical_path = store.root / "historical_report.json"
        if historical_path.exists():
            self.historical = read_json(historical_path)
        elif cfg.smoke:
            self.historical = {"report_dir": None, "notebook_sha256": None}
            store.write_json(None, "historical_report.json", self.historical)
        else:
            if historical_root is None:
                raise ValueError("production requires a sealed historical report")
            historical_root = Path(historical_root).resolve()
            if not (historical_root / "study_completed.json").is_file():
                raise ValueError("historical study must be completed")
            manifest = read_json(historical_root / "report/latest.json")
            directory = Path(manifest["report_dir"]).resolve()
            if not directory.is_relative_to(historical_root / "report"):
                raise ValueError("historical report must belong to its sealed study")
            self.historical = {"report_dir": str(directory),
                "notebook_sha256": file_digest(directory / "report.ipynb", budget.check)}
            store.write_json(None, "historical_report.json", self.historical)
        if self.historical["report_dir"] is not None:
            recorded = file_digest(Path(self.historical["report_dir"]) / "report.ipynb", budget.check)
            if recorded != self.historical["notebook_sha256"]:
                raise ValueError("pinned historical report has changed")

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

    def planned_development(self):
        initial = sum(self.selection["development_replicas"].values()) * (1 + 3 * len(self.cfg.ridges))
        return initial + sum(self.selection["development_replicas"][j["schedule"]]
                             for j in self.selection["refinement_jobs"])

    def pulse(self, *, force=False, report=False, **info):
        self.current.update({k: v for k, v in info.items() if k not in ("accepted_metrics", "accepted_shadow_metrics")})
        if "accepted_metrics" in info:
            self.current_metrics = info["accepted_metrics"]
        now = time.monotonic()
        if not force and now - self.last_pulse < 10:
            return
        self.budget.pulse(force=True)
        fresh = [t for t in self.complete.values() if t["stage"] == "fresh"]
        counts = {}
        for stage in ("development", "fresh"):
            counts[stage] = {s: sum(t["stage"] == stage and t["schedule"] == s for t in self.complete.values())
                             for s in ("main", "stationary")}
        target = self.selection.get("fresh_trajectory_target")
        progress = {"phase": self.phase, "status": "running", "heartbeat": utc_now(),
            "revision": f"{self.cfg.protocol}/{self.store.root.name}",
            "completed_trajectories": len(self.complete), "completed_fresh": len(fresh),
            "planned_fresh": target, "completed_by_stage_schedule": counts,
            "planned_development": self.planned_development(), "failed_candidates": len(self.failures),
            "planned_trajectories": self.planned_development() + (target or 0),
            "forecast_seconds": self.remaining_forecast(),
            "active_hours": self.budget.used / 3600, "remaining_hours": self.budget.remaining / 3600,
            "allocation_hours": self.budget.allocation_seconds / 3600,
            "storage": self.store.storage_status(), **self.current}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.last_pulse = now
        if report or now - self.last_report >= 900:
            self.publish(progress)

    def publish(self, progress=None, *, status="running"):
        from .analysis import summarize
        from .report import build_report
        progress = dict(progress or (read_json(self.store.root / "progress.json")
                        if (self.store.root / "progress.json").exists() else {}))
        progress.update(status=status, phase=self.phase, heartbeat=utc_now(), active_hours=self.budget.used / 3600)
        items = list(self.complete.values())
        uid, meta = self.current.get("active_unit"), self.current_metadata
        if uid and meta and uid not in self.complete and uid not in self.failures and self.current_metrics:
            items.append({"unit_id": uid, **meta, "completed": False, "metrics": self.current_metrics, "summary": {}})
        result = summarize(items, self.selection, bootstrap_draws=self.cfg.bootstrap_draws,
                           seed=self.selection["bootstrap_seed"])
        snapshot = {"schema": 1, "kind": "plan17-retention-snapshot", "study_id": self.store.root.name,
            "status": status, "phase": self.phase, "config": self.cfg.mapping(), "progress": progress,
            "selection": self.selection, "changes": self.changes, "failed_candidates": self.failures,
            "trajectories": items, "analysis": result,
            "historical_report_dir": self.historical["report_dir"],
            "historical_notebook_sha256": self.historical["notebook_sha256"]}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        with self.store.reserve(32 * 1024**2, finalizing=True):
            build_report(self.store.root, self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()
        self.store.reconcile(persist=True, finalizing=True)

    def trajectory(self, cohort, replica, schedule, condition, ridge):
        meta = {"kind": "trajectory", "stage": "development" if cohort == "development" else "fresh",
            "cohort": cohort, "replica": replica, "schedule": schedule, "condition": condition,
            "c": .025, "ridge": float(ridge), "anchor_refresh": 0}
        uid = f"{meta['stage']}-{schedule}-r{replica:03}-{condition}-{content_hash(meta)[:10]}"
        if uid in self.complete:
            return self.complete[uid]
        if uid in self.failures:
            return None
        initial, assets = self.initial(cohort, replica), self.assets(cohort, replica)
        calibration = self.calibration(cohort, replica) if PolicySpec.parse(condition).movement else None
        meta.update(asset_unit=f"assets-{cohort}-r{replica:03}", initial_unit=f"initial-{cohort}-r{replica:03}",
            schedule_factor=assets["schedule_factor"],
            stream_hash=assets["metadata"]["streams"]["stream" if schedule == "main" else "stationary_stream"]["stream_hash"])
        self.current, self.current_metadata = {"active_unit": uid}, meta
        self.current_metrics, self.current_shadows = [], {}
        self.pulse(force=True)
        try:
            run_trajectory(self.store, self.budget, self.cfg, uid, meta, initial, assets,
                original.stream_view(assets, schedule), self.data["train_x"], self.data["train_y"],
                calibration=calibration, pulse=self.pulse)
        except FloatingPointError as error:
            if cohort != "development" or condition == "fixed":
                raise
            self.failures[uid] = {"unit_id": uid, **meta, "reason": str(error), "recorded_at": utc_now()}
            self.store.write_json(uid, "numerical_failure.json", self.failures[uid], finalizing=True)
            self.store.write_json(None, "failed_candidates.json", self.failures, finalizing=True)
            self.current_metadata, self.current_metrics = None, []
            self.pulse(force=True, report=True, active_unit=None)
            print({"failed": uid, "reason": str(error)}, flush=True)
            return None
        item = self.complete[uid] = self.read_trajectory(uid, meta)
        self.current_metadata, self.current_metrics, self.current_shadows = None, [], {}
        self.pulse(force=True, active_unit=None)
        print({"completed": uid, "seconds": item["summary"]["wall_seconds"],
               "nine_recall": item["summary"]["final_nine_recall"],
               "old_accuracy": item["summary"]["final_old_accuracy"]}, flush=True)
        self.count_unit()
        return item

    def durations(self, ridges=None, *, conservative=False):
        ridges = self.selection.get("ridges") if ridges is None else ridges
        result = {}
        for condition in CONDITIONS:
            if ridges:
                by_schedule = []
                for schedule, mapping in ridges.items():
                    values = [t["summary"]["wall_seconds"] / t.get("schedule_factor", 1.)
                              for t in self.complete.values() if t["condition"] == condition
                              and t["schedule"] == schedule and t["ridge"] == mapping[condition]]
                    if values:
                        by_schedule.append(statistics.median(values))
                result[condition] = max(by_schedule or [200.])
            else:
                values = [t["summary"]["wall_seconds"] / (t.get("schedule_factor", 1.) if conservative else 1.)
                          for t in self.complete.values() if t["condition"] == condition]
                result[condition] = (max if conservative else statistics.median)(values or [200.])
        return result

    def fresh_forecast(self, main=None, stationary=None, *, ridges=None, conservative=False):
        main = self.cfg.main_replicas if main is None else main
        stationary = self.cfg.stationary_replicas if stationary is None else stationary
        # Extra horizon margin and new initialization/calibration are intentionally conservative.
        init = statistics.median(list(self.initial_times.values()) or [90.]) + (20. if not self.cfg.smoke else 1.)
        return ((main + stationary) * max(self.cfg.schedule_factors)
                * sum(self.durations(ridges, conservative=conservative).values()) + max(main, stationary) * init)

    def remaining_forecast(self):
        if "fresh_trajectory_target" not in self.selection:
            left = max(0, self.planned_development() - sum(t["stage"] == "development" for t in self.complete.values()))
            return (left * max(self.cfg.schedule_factors) * statistics.mean(self.durations(conservative=True).values())
                    + self.fresh_forecast(conservative=True))
        costs = self.durations()
        done = {(t["schedule"], t["replica"], t["condition"]) for t in self.complete.values() if t["stage"] == "fresh"}
        pending, pending_initial = 0., set()
        for schedule in ("main", "stationary"):
            for replica in range(self.selection[schedule + "_replicas"]):
                factor = self.cfg.schedule_factors[replica % len(self.cfg.schedule_factors)]
                for condition in CONDITIONS:
                    if (schedule, replica, condition) not in done:
                        pending += factor * costs[condition]
                        if f"initial-fresh-r{replica:03}" not in self.initial_times:
                            pending_initial.add(replica)
        init = statistics.median(list(self.initial_times.values()) or [90.]) + (20. if not self.cfg.smoke else 1.)
        return pending + len(pending_initial) * init

    def run_development_candidate(self, condition, ridge, schedule=None):
        schedules = (schedule,) if schedule else ("main", "stationary")
        for s in schedules:
            for replica in range(self.selection["development_replicas"][s]):
                self.trajectory("development", replica, s, condition, ridge)
        self.pulse(force=True, report=True)

    def freeze_selection(self):
        from .analysis import candidate_table, select_candidate
        ridges, selected = {}, {}
        for schedule, count in self.selection["development_replicas"].items():
            ridges[schedule], selected[schedule] = {"fixed": .1}, {}
            for condition in CONDITIONS[1:]:
                table = candidate_table(list(self.complete.values()), schedule, condition, replicas=range(count))
                if not table:
                    raise ResourceLimit(f"no numerically valid complete development candidate: {schedule}/{condition}; repair or report non-estimable")
                chosen = select_candidate(table, tolerance_pp=self.cfg.tolerance_pp)
                selected[schedule][condition] = chosen
                ridges[schedule][condition] = chosen["ridge"]
        nmain, nstat = self.cfg.main_replicas, self.cfg.stationary_replicas
        while 1.25 * self.fresh_forecast(nmain, nstat, ridges=ridges) + 1800 > self.budget.remaining:
            if nmain == nstat == 0:
                break
            step = 1 if self.cfg.smoke else 2
            if nstat > (1 if self.cfg.smoke else 2) and nmain <= 2 * nstat:
                nstat = max(0, nstat - step)
            elif nmain > (1 if self.cfg.smoke else 4):
                nmain = max(0, nmain - step)
            elif nstat:
                nstat = max(0, nstat - step)
            else:
                nmain = max(0, nmain - step)
        self.selection.update(ridges=ridges, development_selection=selected,
            main_replicas=nmain, stationary_replicas=nstat,
            fresh_trajectory_target=4 * (nmain + nstat),
            forecast_fresh_seconds=self.fresh_forecast(nmain, nstat, ridges=ridges),
            seconds_by_condition=self.durations(ridges), frozen_at=utc_now(), contingency=1.25,
            resource_limited=(nmain, nstat) != (self.cfg.main_replicas, self.cfg.stationary_replicas))
        self.save_selection()

    def seal_selection(self):
        uid = "frozen-selection"
        path = self.store.begin(uid, {"kind": "selection", "selection_hash": content_hash(self.selection)})
        if path is not None:
            self.store.write_json(uid, "selection.json", self.selection)
            self.store.complete(uid)
            self.count_unit()
        if read_json(self.store.unit_path(uid) / "selection.json") != self.selection:
            raise ValueError("frozen selection integrity failure")

    def run(self):
        from .analysis import candidate_table, refinement_candidate
        self.phase = "5B: bounded retention calibration"
        self.pulse(force=True, report=True)
        if "frozen_at" not in self.selection:
            self.run_development_candidate("fixed", .1)
            order = sorted(self.cfg.ridges, key=lambda r: (r != .1, r))
            for ridge in order:
                for condition in CONDITIONS[1:]:
                    self.run_development_candidate(condition, ridge)
            for schedule, count in self.selection["development_replicas"].items():
                for condition in CONDITIONS[1:]:
                    jobs = [j for j in self.selection["refinement_jobs"]
                            if j["schedule"] == schedule and j["condition"] == condition]
                    for job in jobs:
                        self.run_development_candidate(condition, job["ridge"], schedule)
                    while len(jobs) < self.cfg.max_refinements:
                        table = candidate_table(list(self.complete.values()), schedule, condition, replicas=range(count))
                        ridge = refinement_candidate(table, tolerance_pp=self.cfg.tolerance_pp)
                        if ridge is None or any(j["ridge"] == ridge for j in jobs):
                            break
                        refinement_cost = count * max(self.cfg.schedule_factors) * self.durations(conservative=True)[condition]
                        reserve = 1.25 * (self.fresh_forecast(conservative=True) + refinement_cost) + 1800
                        if reserve > self.budget.remaining:
                            break
                        job = {"schedule": schedule, "condition": condition, "ridge": ridge,
                               "decided_at": utc_now(), "ordinal": len(jobs) + 1}
                        self.selection["refinement_jobs"].append(job)
                        jobs.append(job)
                        self.save_selection()
                        self.run_development_candidate(condition, ridge, schedule)
            self.freeze_selection()
        self.seal_selection()
        self.phase = "5C: fresh paired confirmation"
        self.pulse(force=True, report=True)
        for replica in range(max(self.selection["main_replicas"], self.selection["stationary_replicas"])):
            for schedule in ("main", "stationary"):
                if replica >= self.selection[schedule + "_replicas"]:
                    continue
                order = list(CONDITIONS[replica % 4:] + CONDITIONS[:replica % 4])
                for condition in order:
                    self.trajectory("fresh", replica, schedule, condition,
                                    self.selection["ridges"][schedule][condition])
                self.pulse(force=True, report=True)
        self.phase = "5D: interpretation and closeout"
        self.pulse(force=True)
        self.publish(status="completed")
        self.store.write_json(None, "study_completed.json", {"completed_at": utc_now(),
            "active_hours": self.budget.used / 3600, "selection": self.selection})
        print({"status": "completed", "active_hours": self.budget.used / 3600}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, default=REPO / "cache/mnist_experiment/plan17/production_v4")
    parser.add_argument("--data-root", type=Path, default=REPO / "cache/mnist_experiment/datasets")
    parser.add_argument("--allocation-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config.from_json(args.config)
    store, budget = open_store(cfg, args.root, args.data_root, resume=args.resume, clear_stop=args.clear_stop,
        allocation_hours=args.allocation_hours, max_wall_seconds=args.max_wall_seconds)
    scheduler = None
    try:
        with store.control.signal_handlers():
            if (store.root / "study_completed.json").exists():
                print({"status": "already_completed", "active_hours": budget.used / 3600}, flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, args.data_root,
                                  historical_root=args.historical_root, max_units=args.max_units)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler is not None:
            scheduler.publish(status="paused")
        print({"status": "paused", "reason": str(error), "active_hours": budget.used / 3600}, flush=True)
        return 0
    finally:
        budget.close()
        store.close()
        store.plan17_allocation_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
