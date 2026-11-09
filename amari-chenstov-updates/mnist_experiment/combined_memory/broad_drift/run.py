"""Execute Plan 18 Phase 5 with immutable units and cumulative resource limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, request_stop
from ..run import Scheduler as OriginalScheduler
from ..config import content_hash, Recipe
from .config import Config
from .data import make_assets
from .detectability import run_diagnostic
from .runtime import FAMILY, DATA_ROOT, read_json, open_store
from .trajectory import Reader, run_trajectory


class Scheduler(OriginalScheduler):
    def status(self, state="running"):
        result = super().status(state)
        number = self.phase.split(":",1)[0]
        subphase = {"0":"5A", "1":"5B", "2":"5C", "3":"5D", "4":"5E"}.get(number,"5")
        result.update(phase=subphase+": "+self.phase.split(":",1)[-1].strip(),
            acquisition_clock="outbound grids", analysis_outbound_count=self.cfg.analysis_novel_count)
        return result

    def publish(self, status="running", *, figures=True):
        self.store.write_json(None, "status.json", self.status(status), finalizing=True)
        if (not figures or self.store.control.stop_requested or (self.store.root/"STOP").exists()
                or self.budget.remaining < 30):
            return
        from .report import build_report
        build_report(self.store.root, cfg=self.cfg, check=self.budget.check,
            notebook_path=self.store.root/"smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()

    def assets(self, cohort, replica):
        key = (cohort, replica)
        if key not in self.assets_cache:
            uid = f"assets-{cohort}-{replica:03}"
            metadata = dict(kind="assets", cohort=cohort, replica=replica)
            if self.store.begin(uid, metadata) is None:
                result = self.store.load_checkpoint(uid)
            else:
                result = make_assets(self.cfg, replica, cohort, self.raw)
                self.store.save_checkpoint(uid, result)
                self.store.write_json(uid, "summary.json", result["metadata"])
                self.store.complete(uid)
            self.assets_cache[key] = result
        return self.assets_cache[key]

    def reader(self, cohort, replica):
        key = (cohort, replica)
        if key not in self.readers:
            self.readers.clear()
            self.readers[key] = Reader(self.raw, self.assets(cohort, replica), self.cfg, self.budget.check)
        return self.readers[key]

    def trajectory(self, cohort, replica, recipe, *, readiness=False):
        cfg = self.effective_cfg()
        numeric = dict(fd_dtype=cfg.fd_dtype if recipe.strength else "unused",
                       max_iter=cfg.max_iter, max_eval=cfg.max_eval)
        uid = f"path-{cohort}-{replica:03}-{recipe.key}-{content_hash(numeric)[:8]}"
        if uid in self.records:
            return self.records[uid]
        if self.max_units is not None and self.new_paths >= self.max_units:
            raise StopRequested("completed the requested invocation trajectory count")
        assets = self.assets(cohort, replica)
        initial = self.initial(cohort, replica)
        metadata = dict(kind="trajectory", cohort=cohort, replica=replica,
            protocol=cfg.protocol, acquisition_clock="outbound_grids", recipe=recipe.mapping(),
            pairing_asset_hash=assets["metadata"]["asset_hash"],
            model_seed=cfg.seed_for(cohort, replica, "initialization_model"),
            factor_seed=cfg.seed_for(cohort, replica, "factors"),
            numerical_policy=numeric, global_sign=assets["metadata"]["assignment"]["global_sign"],
            seed_derivation="config.seed_for(cohort,replica,component); separate per-batch replay/probe seeds")
        reader = self.reader(cohort, replica)
        reader.cfg = cfg
        run_trajectory(self.store, self.budget, cfg, uid, metadata, initial, reader,
            pulse=self.pulse, audit=readiness and replica == 0, shadows=readiness)
        result = self.load_record(uid, metadata)
        self.records[uid] = result
        self.new_paths += 1
        self.current = {}
        self.pulse(force=True, report=True)
        return result

    def readiness_extras(self):
        replicas = [run_diagnostic(self, i) for i in range(2)]
        return dict(detectability=dict(replicas=replicas), capability=None,
            analysis_outbound_count=self.cfg.analysis_novel_count,
            observer_unchanged=True,
            legacy_scalar_aliases="novel_count and H100 mean outbound count and endpoint H_N in this protocol")

    def integrity_audit(self):
        result = super().integrity_audit()
        for record in self.records.values():
            if record["metadata"]["cohort"] == "development":
                continue
            self.budget.check()
            stationary = record["metadata"]["cohort"] == "stationary"
            expected = 0 if stationary else self.cfg.analysis_novel_count
            if record["summary"]["novel_count"] != expected:
                raise ValueError("final outbound acquisition count differs from the sealed design")
            for row in record["metrics"]:
                outbound = min(self.cfg.ramp_steps,max(0,row["accepted_step"]-self.cfg.hold_steps))
                count = 0 if stationary else outbound*self.cfg.samples_per_step
                if row["novel_count"] != count or row["outbound_count"] != count:
                    raise ValueError("outbound metric clock disagrees with its accepted batch")
        result.update(acquisition_clock="outbound_grids", analysis_outbound_count=self.cfg.analysis_novel_count,
                      stationary_sample_equivalence=False)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=FAMILY/"broad_drift_v1")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-active-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args(argv)
    cfg = Config.read(args.config)
    if args.request_stop:
        request_stop(args.root)
        print("Stop requested; execution will exit at a recoverable boundary.", flush=True)
        return 0
    if args.report_only:
        from .report import build_report
        build_report(args.root, cfg=cfg,
            notebook_path=args.root/"smoke_results.ipynb" if cfg.smoke else None)
        return 0
    try:
        store, budget = open_store(cfg, args.root, args.data_root, resume=args.resume,
            clear_stop=args.clear_stop, allocation_hours=args.max_active_hours,
            max_wall_seconds=args.max_wall_seconds)
    except (StopRequested, ResourceLimit) as error:
        print(json.dumps(dict(status="paused", reason=str(error), phase="store startup")), flush=True)
        return 0
    scheduler = None
    try:
        with store.control.signal_handlers():
            if (store.root/"study_completed.json").exists():
                print(json.dumps(dict(status="already_completed", active_hours=budget.used/3600)), flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, load_data(args.data_root), max_units=args.max_units)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler:
            scheduler.publish("paused" if isinstance(error, StopRequested) else "resource_limited", figures=False)
        print(json.dumps(dict(status="paused", reason=str(error), active_hours=budget.used/3600)), flush=True)
        return 0
    finally:
        budget.close()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
