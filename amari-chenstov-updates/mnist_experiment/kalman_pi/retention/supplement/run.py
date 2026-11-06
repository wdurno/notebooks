"""Resumable three-arm Phase 5E, following the completed primary calibration."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.kalman_pi import run as original
from mnist_experiment.kalman_pi.policies import PolicySpec
from mnist_experiment.kalman_pi.runtime import file_digest, read_json
from mnist_experiment.kalman_pi.trajectory import run_trajectory
from mnist_experiment.kalman_pi.retention.data import make_assets
from .config import Config, ARMS, content_hash
from .runtime import REPO, open_store


class Scheduler(original.Scheduler):
    def __init__(self, cfg, store, budget, data_root, *, historical_root=None, max_units=None):
        super().__init__(cfg, store, budget, data_root, max_units=max_units)
        self.phase = "5E: supplemental zero-ridge confirmation"
        if store.is_complete("frozen-selection"):
            sealed = read_json(store.unit_path("frozen-selection") / "selection.json")
            if (store.root / "selection.json").exists() and self.selection != sealed:
                raise ValueError("supplemental selection differs from its immutable seal")
            self.selection = sealed
            if not (store.root / "selection.json").exists():
                store.write_json(None, "selection.json", sealed)
        historical_path = store.root / "historical_report.json"
        if historical_path.exists():
            self.historical = read_json(historical_path)
        elif cfg.smoke:
            self.historical = {"report_dir": None, "notebook_sha256": None}
            store.write_json(None, "historical_report.json", self.historical)
        else:
            if historical_root is None:
                raise ValueError("production supplement requires the completed primary report")
            historical_root = Path(historical_root).resolve()
            if not (historical_root / "study_completed.json").is_file():
                raise ValueError("finish the approved primary study before running the supplement")
            directory = Path(read_json(historical_root / "report/latest.json")["report_dir"]).resolve()
            if not directory.is_relative_to(historical_root / "report"):
                raise ValueError("primary report must belong to its study")
            self.historical = {"report_dir": str(directory),
                "notebook_sha256": file_digest(directory / "report.ipynb", budget.check)}
            store.write_json(None, "historical_report.json", self.historical)
        if self.historical["report_dir"] is not None:
            if file_digest(Path(self.historical["report_dir"]) / "report.ipynb", budget.check) != self.historical["notebook_sha256"]:
                raise ValueError("pinned primary report has changed")
        if self.selection:
            self.validate_selection()

    def design_fields(self):
        return {"schema": 1, "protocol": self.cfg.protocol, "frozen": True,
            "arms": ARMS, "main_replicas": self.cfg.main_replicas, "stationary_replicas": 0,
            "fresh_trajectory_target": len(ARMS) * self.cfg.main_replicas,
            "bootstrap_seed": self.cfg.seed_for("analysis", 0, "bootstrap"),
            "config_hash": self.cfg.scientific_hash,
            "sources_hash": content_hash(self.store.identity.get("sources", {})),
            "data_partition": {"protocol": "plan17-retention-calibration-v1",
                "seed": self.cfg.partition_seed,
                **{name + "_seed": self.cfg.seed_for("study-partition", 0, name) for name in ("train", "test")}},
            "schedule_factors": list(self.cfg.schedule_factors)}

    def validate_selection(self):
        expected = self.design_fields()
        if any(self.selection.get(key) != value for key, value in expected.items()):
            raise ValueError("supplemental design changed; use a new protocol and root")
        if not isinstance(self.selection.get("frozen_at"), str) or not self.selection["frozen_at"]:
            raise ValueError("supplemental design lacks its pre-confirmation freeze")

    def freeze_design(self):
        if not self.selection:
            if self.complete:
                raise ValueError("confirmation trajectories exist without a frozen design")
            self.selection = {**self.design_fields(), "frozen_at": utc_now()}
            self.store.write_json(None, "selection.json", self.selection)
        self.validate_selection()
        if not self.store.is_complete("frozen-selection"):
            self.store.begin("frozen-selection", {"kind": "selection", "selection_hash": content_hash(self.selection)})
            self.store.write_json("frozen-selection", "selection.json", self.selection)
            self.store.complete("frozen-selection")
            self.count_unit()

    def assets(self, cohort, replica):
        if cohort != "fresh":
            raise ValueError("supplemental assets are confirmation-only")
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

    def costs(self):
        return {arm: statistics.median([t["summary"]["wall_seconds"] / t["schedule_factor"]
                    for t in self.complete.values() if t["arm"] == arm] or [230.]) for arm in ARMS}

    def remaining_forecast(self):
        costs = self.costs()
        done = {(t["replica"], t["arm"]) for t in self.complete.values()}
        pending = sum(costs[arm] * self.cfg.schedule_factors[r % len(self.cfg.schedule_factors)]
            for r in range(self.cfg.main_replicas) for arm in ARMS if (r, arm) not in done)
        initial = statistics.median(list(self.initial_times.values()) or [90.]) + 20.
        pending += sum(initial for r in range(self.cfg.main_replicas)
                       if f"initial-fresh-r{r:03}" not in self.initial_times)
        if self.current_metadata and self.current_metadata.get("kind") == "trajectory":
            pending -= costs[self.current_metadata["arm"]] * self.current_metadata["schedule_factor"] * min(
                1., self.current.get("step", 0) / max(1, self.current.get("total_steps", 1)))
        return max(0., pending)

    def pulse(self, *, force=False, report=False, **info):
        self.current.update({k: v for k, v in info.items() if k not in ("accepted_metrics", "accepted_shadow_metrics")})
        if "accepted_metrics" in info:
            self.current_metrics = info["accepted_metrics"]
        now = time.monotonic()
        if not force and now - self.last_pulse < 10:
            return
        self.budget.pulse(force=True)
        blocks = {}
        for item in self.complete.values():
            blocks.setdefault(item["replica"], set()).add(item["arm"])
        progress = {"status": "running", "phase": self.phase, "heartbeat": utc_now(),
            "revision": f"{self.cfg.protocol}/{self.store.root.name}",
            "completed_trajectories": len(self.complete), "completed_fresh": len(self.complete),
            "planned_trajectories": 3 * self.cfg.main_replicas, "planned_fresh": 3 * self.cfg.main_replicas,
            "completed_blocks": sum(set(ARMS) <= arms for arms in blocks.values()),
            "planned_blocks": self.cfg.main_replicas, "forecast_seconds": self.remaining_forecast(),
            "active_hours": self.budget.used / 3600, "remaining_hours": self.budget.remaining / 3600,
            "allocation_hours": self.budget.allocation_seconds / 3600,
            "storage": self.store.storage_status(), **self.current}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.last_pulse = now
        if self.selection and (report or now - self.last_report >= 900):
            self.publish(progress)

    def publish(self, progress=None, *, status="running"):
        from .analysis import summarize
        from .report import build_report
        progress = dict(progress or (read_json(self.store.root / "progress.json")
            if (self.store.root / "progress.json").exists() else {}))
        progress.update(status=status, phase=self.phase, heartbeat=utc_now(), active_hours=self.budget.used / 3600)
        items = list(self.complete.values())
        uid, meta = self.current.get("active_unit"), self.current_metadata
        if uid and meta and uid not in self.complete and self.current_metrics:
            items.append({"unit_id": uid, **meta, "completed": False, "metrics": self.current_metrics, "summary": {}})
        analysis = summarize(items, self.selection, bootstrap_draws=self.cfg.bootstrap_draws,
                             seed=self.selection["bootstrap_seed"])
        snapshot = {"schema": 1, "kind": "plan17-zero-ridge-snapshot", "study_id": self.store.root.name,
            "status": status, "phase": self.phase, "config": self.cfg.mapping(), "progress": progress,
            "selection": self.selection, "trajectories": items, "analysis": analysis,
            "historical_report_dir": self.historical["report_dir"],
            "historical_notebook_sha256": self.historical["notebook_sha256"]}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        with self.store.reserve(32 * 1024**2, finalizing=True):
            build_report(self.store.root, self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()
        self.store.reconcile(persist=True, finalizing=True)

    def trajectory(self, replica, arm):
        spec = ARMS[arm]
        meta = {"kind": "trajectory", "stage": "fresh", "cohort": "fresh", "replica": replica,
            "schedule": "main", "arm": arm, "condition": spec["condition"],
            "c": .025, "ridge": spec["ridge"], "anchor_refresh": 0}
        uid = f"fresh-main-r{replica:03}-{arm}-{content_hash(meta)[:10]}"
        if uid in self.complete:
            return self.complete[uid]
        initial, assets = self.initial("fresh", replica), self.assets("fresh", replica)
        calibration = self.calibration("fresh", replica) if PolicySpec.parse(spec["condition"]).movement else None
        meta.update(asset_unit=f"assets-fresh-r{replica:03}", initial_unit=f"initial-fresh-r{replica:03}",
            schedule_factor=assets["schedule_factor"], stream_hash=assets["metadata"]["streams"]["stream"]["stream_hash"])
        self.current, self.current_metadata = {"active_unit": uid}, meta
        self.current_metrics, self.current_shadows = [], {}
        self.pulse(force=True)
        run_trajectory(self.store, self.budget, self.cfg, uid, meta, initial, assets,
            original.stream_view(assets, "main"), self.data["train_x"], self.data["train_y"],
            calibration=calibration, pulse=self.pulse)
        item = self.complete[uid] = self.read_trajectory(uid, meta)
        self.current_metadata, self.current_metrics = None, []
        self.pulse(force=True, active_unit=None)
        print({"completed": uid, "seconds": item["summary"]["wall_seconds"],
            "nine_recall": item["summary"]["final_nine_recall"],
            "old_accuracy": item["summary"]["final_old_accuracy"]}, flush=True)
        self.count_unit()
        return item

    def run(self):
        self.freeze_design()
        self.pulse(force=True, report=True)
        arms = list(ARMS)
        for replica in range(self.cfg.main_replicas):
            if not any(t["replica"] == replica for t in self.complete.values()) and not self.cfg.smoke:
                factor = self.cfg.schedule_factors[replica % len(self.cfg.schedule_factors)]
                block = 1.25 * (sum(self.costs().values()) * factor + 110.)
                if self.budget.remaining < block + 1800.:
                    raise ResourceLimit("preserve final reporting reserve before an unstarted supplemental block")
            for arm in arms[replica % 3:] + arms[:replica % 3]:
                self.trajectory(replica, arm)
            self.pulse(force=True, report=True)
        self.phase = "5E: interpretation and closeout"
        self.pulse(force=True)
        self.publish(status="completed")
        self.store.write_json(None, "study_completed.json", {"completed_at": utc_now(),
            "active_hours": self.budget.used / 3600, "selection": self.selection})
        print({"status": "completed", "active_hours": self.budget.used / 3600}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, default=REPO / "cache/mnist_experiment/plan17/retention_v1")
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
        if scheduler is not None and scheduler.selection:
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
