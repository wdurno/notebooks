"""Run the retention-calibrated Plan 18 follow-up with per-batch recovery."""

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, request_stop, utc_now
from ..analysis import summarize_path
from ..config import ARMS, Recipe, content_hash
from ..run import Scheduler as BaseScheduler
from ..broad_drift.run import Scheduler as BroadScheduler
from .config import Config
from .runtime import (DATA_ROOT, DEFAULT_ROOT, HISTORY_ROOT, REPO, open_store, read_json,
                      read_exported_records, scalar_record, load_readonly_checkpoint, checked_scalar)
from . import calibration


class Scheduler(BroadScheduler):
    def __init__(self, cfg, store, budget, raw, *, history_root=HISTORY_ROOT, max_units=None):
        super().__init__(cfg, store, budget, raw, max_units=max_units)
        self.history_root = Path(history_root).resolve()
        self.phase = "6A: acceptance and historical development validation"
        self.calibration_status = "pending"
        self.reference_ridge = 10.
        self.member_summaries, self.rounds = [], []
        self.current_check = None
        self.records = {}
        self.refresh_records()
        if self.selection:
            self.calibration_status = "passed"
            self.validate_selection()

    def refresh_records(self):
        records, _ = read_exported_records(self.store.root, self.budget.check)
        self.records.update({r["unit_id"]: r for r in records})

    def status(self, state="running"):
        result = BaseScheduler.status(self, state)
        result.update(calibration_status=self.calibration_status,
            imported_development=sum(bool(r["metadata"].get("imported")) for r in self.records.values()),
            expected_development_new=10, development_members=list(range(4)),
            reference_ridge=self.reference_ridge, calibration_check=self.current_check,
            confirmation_gate="passed" if self.selection else "closed",
            acquisition_clock="outbound grids", analysis_outbound_count=self.cfg.analysis_novel_count)
        return result

    def publish(self, status="running", *, figures=True):
        self.store.write_json(None, "status.json", self.status(status), finalizing=True)
        if (not figures or self.store.control.stop_requested or (self.store.root / "STOP").exists()
                or self.budget.remaining < 30):
            return
        from .report import build_report
        build_report(self.store.root, cfg=self.cfg, check=self.budget.check,
            notebook_path=self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()

    def import_history(self):
        uid = "import-development"
        metadata = dict(kind="historical_development_import", members=[0, 1])
        if self.store.begin(uid, metadata) is not None:
            identity = artifacts._identity(self.history_root)
            references, dependencies = [], {}
            for unit in artifacts.iter_units(self.history_root, completed_only=True,
                                             check=self.budget.check, full=True):
                meta = unit["metadata"]
                if meta.get("cohort") != "development" or meta.get("replica") not in (0, 1):
                    continue
                if meta.get("kind") == "trajectory":
                    row, seals = scalar_record(self.history_root, unit["unit_id"], self.budget.check)
                    expected = summarize_path(row["metrics"], self.cfg)
                    for key in ("Havg", "H100", "novel_auc"):
                        if not math.isclose(row["summary"][key], expected[key], rel_tol=1e-10, abs_tol=1e-10):
                            raise artifacts.ArtifactError("imported summary differs from its scalar trajectory")
                    references.append(dict(unit_id=unit["unit_id"], seals=seals))
                elif meta.get("kind") in ("assets", "initialization"):
                    dependencies[unit["unit_id"]] = artifacts._file_hash(unit["path"] / "COMPLETED")
            if {r["unit_id"].split("-")[2] for r in references} != {"000", "001"}:
                raise artifacts.ArtifactError("both historical development members are required")
            payload = dict(schema=1, root=str(self.history_root.relative_to(REPO)),
                identity_hash=identity["identity_hash"], units=references, dependencies=dependencies,
                roles="development 0/1 only; no historical confirmation imports")
            self.store.write_json(uid, "references.json", payload)
            self.store.complete(uid)
        self.refresh_records()
        numeric = read_json(self.history_root / "numerics.json")
        if (numeric["fd_dtype"], numeric["max_iter"], numeric["max_eval"]) != (
                self.cfg.fd_dtype, self.cfg.max_iter, self.cfg.max_eval):
            raise artifacts.ArtifactError("historical numerical policy differs from the Phase 6 policy")
        self.numeric = numeric
        self.store.write_json(None, "numerics.json", numeric)
        self.development_view()

    def historical_checkpoint(self, uid):
        references, _ = checked_scalar(self.store.root, "import-development", "references.json", self.budget.check)
        expected = references["dependencies"].get(uid)
        if expected != artifacts._file_hash(self.history_root / "units" / uid / "COMPLETED", self.budget.check):
            raise artifacts.ArtifactError("historical asset or initialization reference changed")
        return load_readonly_checkpoint(self.history_root, uid, self.budget.check)

    def assets(self, cohort, replica):
        if cohort == "development" and replica < 2:
            key = (cohort, replica)
            if key not in self.assets_cache:
                self.assets_cache[key] = self.historical_checkpoint(f"assets-development-{replica:03}")
            return self.assets_cache[key]
        return super().assets(cohort, replica)

    def initial(self, cohort, replica):
        if cohort == "development" and replica < 2:
            key = (cohort, replica)
            if key not in self.initial_cache:
                self.initial_cache[key] = self.historical_checkpoint(f"initial-development-{replica:03}")
            return self.initial_cache[key]
        return super().initial(cohort, replica)

    def compatible(self, recipe):
        target = calibration.initial_recipes(self.cfg.fd_strength)[recipe.arm]
        return recipe.strength == target.strength and recipe.process_scale == target.process_scale

    def trajectory(self, cohort, replica, recipe, *, readiness=False):
        if not self.compatible(recipe):
            raise ValueError("Phase 6 permits only the frozen FD strength and process multiplier")
        if cohort != "development":
            self.validate_selection()
            if self.selection["recipes"][recipe.arm] != recipe.mapping():
                raise artifacts.ArtifactError("confirmation recipe differs from its calibration seal")
        for row in self.records.values():
            meta = row["metadata"]
            if (meta["cohort"] == cohort and meta["replica"] == replica
                    and meta["recipe"] == recipe.mapping()):
                return row
        row = super().trajectory(cohort, replica, recipe, readiness=False)
        self.development_view()
        return row

    def development_view(self):
        self.member_summaries = []
        for row in self.records.values():
            if row["metadata"]["cohort"] != "development":
                continue
            self.member_summaries.append(dict(recipe=row["metadata"]["recipe"],
                replica=row["metadata"]["replica"], imported=bool(row["metadata"].get("imported")),
                **{k: row["summary"].get(k) for k in ("Havg", "H100", "novel_auc", "learning_seconds")}))
        self.store.write_json(None, "development.json", dict(members=list(range(4)),
            member_summaries=self.member_summaries, current_check=self.current_check,
            reference_ridge=self.reference_ridge, refinement_rounds=self.rounds,
            candidate_allowances=self.used_allowances()))

    def used_allowances(self):
        counts = {a: 0 for a in ARMS if a != "A"}
        for unit in self.store.iter_units(completed_only=True):
            if unit["metadata"].get("name", "").startswith("refinement-"):
                payload = read_json(self.store.unit_path(unit["unit_id"]) / "decision.json")
                for recipe in payload["recipes"]:
                    counts[recipe["arm"]] += 1
        return counts

    def all_candidate_rows(self):
        result = {a: [] for a in ARMS}
        recipes = {Recipe(**r["metadata"]["recipe"]) for r in self.records.values()
                   if r["metadata"]["cohort"] == "development"}
        for recipe in sorted(recipes, key=lambda r: r.key):
            if not self.compatible(recipe):
                continue
            try:
                row = calibration.aggregate(list(self.records.values()), recipe)
            except ValueError:
                continue
            if recipe.arm != "A" or recipe.ridge == self.reference_ridge:
                result[recipe.arm].append(row)
        return result

    def best_available(self):
        pools = self.all_candidate_rows()
        if any(not pools[a] for a in ARMS):
            raise ValueError("starting tuple lacks one or more complete four-member recipes")
        tuples = []
        for rows in itertools.product(*(pools[a] for a in ARMS)):
            mapped = dict(zip(ARMS, rows))
            check = calibration.matching(mapped, tolerance=self.cfg.retention_tolerance)
            violation = max(0., *(v-self.cfg.retention_tolerance for v in check["ranges"].values()))
            rank = (violation, statistics.mean(r["novel_auc"] for r in rows),
                    tuple(r["recipe"]["ridge"] for r in rows))
            tuples.append((rank, mapped, check))
        _, rows, check = min(tuples, key=lambda x: x[0])
        self.current_check = check
        return rows

    def path_cost(self, arm):
        rows = [r for r in self.records.values() if r["metadata"]["recipe"]["arm"] == arm
                and not r["metadata"].get("imported")]
        if not rows:
            rows = [r for r in self.records.values() if r["metadata"]["recipe"]["arm"] == arm]
        return statistics.median(r["summary"].get("online_wall_seconds", 360.) for r in rows) if rows else 360.

    def initial_cost(self):
        values = [v["wall_seconds"] for v in self.initial_cache.values()]
        for path in (self.store.root / "units").glob("initial-*/summary.json"):
            if (path.parent / "COMPLETED").exists():
                values.append(read_json(path)["wall_seconds"])
        return statistics.median(values) if values else (.1 if self.cfg.smoke else 263.)

    def closeout_reserve(self):
        return 15. if self.cfg.smoke else .35 * 3600

    def can_admit(self, recipes):
        needed = 0.
        for recipe in recipes:
            for replica in range(4):
                found = any(r["metadata"]["cohort"] == "development"
                    and r["metadata"]["replica"] == replica
                    and r["metadata"]["recipe"] == recipe.mapping() for r in self.records.values())
                if not found:
                    needed += self.path_cost(recipe.arm)
        return needed * 1.25 + self.closeout_reserve() <= self.budget.remaining

    def proposal_round(self):
        rows = self.best_available()
        reference = rows["A"]
        counts = self.used_allowances()
        proposals = []
        self.proposal_details = []
        for arm in ARMS[1:]:
            if counts[arm] >= self.cfg.refinements_per_arm:
                continue
            row = rows[arm]
            gap = max(abs(row[k]-reference[k]) for k in ("Havg", "H100"))
            if gap <= self.cfg.retention_tolerance / 2:
                continue
            base = Recipe(**row["recipe"])
            siblings = [r for r in self.records.values() if r["metadata"]["cohort"] == "development"
                and r["metadata"]["recipe"]["arm"] == arm
                and self.compatible(Recipe(**r["metadata"]["recipe"]))]
            completed = {r["recipe"]["ridge"] for r in self.all_candidate_rows()[arm]}
            historical = []
            for ridge in sorted({r["metadata"]["recipe"]["ridge"] for r in siblings} - completed):
                recipe = Recipe(arm, ridge, base.strength, base.process_scale)
                try:
                    candidate = calibration.aggregate(siblings, recipe, members=(0, 1))
                except ValueError:
                    continue
                distance = max(abs(candidate[k]-reference[k]) for k in ("Havg", "H100"))
                historical.append((distance, ridge, recipe))
            if historical and min(historical)[0] < gap:
                _, _, candidate = min(historical)
                proposals.append(candidate)
                self.proposal_details.append(dict(arm=arm, ridge=candidate.ridge,
                    reason="extend an observed historical candidate on all four members",
                    historical_distance=min(historical)[0], current_distance=gap,
                    reference={k: reference[k] for k in ("Havg", "H100")}))
                continue
            seen = sorted({r["metadata"]["recipe"]["ridge"] for r in siblings})
            proposal = calibration.ridge_proposal(arm, list(self.records.values()), base,
                reference, seen, tolerance=self.cfg.retention_tolerance, strength=self.cfg.fd_strength)
            if proposal is None:
                continue
            proposals.append(Recipe(arm, proposal["ridge"], base.strength, base.process_scale))
            self.proposal_details.append(dict(arm=arm, **proposal))
        return proposals

    def selection_payload(self, selected):
        recipes = selected["recipes"]
        rows = {a: calibration.aggregate(list(self.records.values()), Recipe(**recipes[a])) for a in ARMS}
        check = calibration.matching(rows, tolerance=self.cfg.retention_tolerance)
        if not check["passed"] or self.cfg.force_calibration_failure:
            raise artifacts.ArtifactError("refusing to seal an unmatched calibration")
        per_path = statistics.mean(self.path_cost(a) for a in ARMS)
        init_seconds = self.initial_cost()
        forecast = ((self.cfg.main_replicas + self.cfg.stationary_replicas)
                    * (sum(self.path_cost(a) for a in ARMS) + init_seconds) * 1.25
                    + self.closeout_reserve())
        evidence = sorted(r["unit_id"] for r in self.records.values()
            if r["metadata"]["cohort"] == "development"
            and r["metadata"]["recipe"] == recipes[r["metadata"]["recipe"]["arm"]])
        return dict(frozen=True, frozen_at=utc_now(), calibration_status="passed", recipes=recipes,
            calibration=rows, matching=check, members=list(range(4)), reference_ridge=self.reference_ridge,
            candidate_allowances=self.used_allowances(), evidence_units=evidence,
            source_identity=self.store.identity_hash, config_hash=self.cfg.scientific_hash,
            main_replicas=self.cfg.main_replicas, stationary_replicas=self.cfg.stationary_replicas,
            seconds_per_path=per_path, forecast_fresh_seconds=forecast,
            bootstrap_seed=self.cfg.seed_for("fresh", 0, "bootstrap"), numerical_policy=self.numeric)

    def validate_selection(self):
        if not self.selection or not self.store.is_complete("decision-selection"):
            raise artifacts.ArtifactError("confirmation requires a sealed passing calibration")
        sealed, _ = checked_scalar(self.store.root, "decision-selection", "decision.json", self.budget.check)
        if sealed != self.selection or sealed.get("source_identity") != self.store.identity_hash:
            raise artifacts.ArtifactError("mutable selection differs from the sealed scientific decision")
        rows = {a: calibration.aggregate(list(self.records.values()), Recipe(**sealed["recipes"][a])) for a in ARMS}
        check = calibration.matching(rows, tolerance=self.cfg.retention_tolerance)
        if (not check["passed"] or check != sealed["matching"] or rows != sealed["calibration"]
                or sealed.get("calibration_status") != "passed" or sealed.get("members") != list(range(4))):
            raise artifacts.ArtifactError("calibration seal no longer proves matching on all four members")
        self.current_check = check
        return sealed

    def stop_calibration(self, reason, detail):
        self.calibration_status = reason
        self.phase = "6B: calibration closed without confirmation"
        self.current = {}
        self.development_view()
        self.store.write_json(None, "calibration_completed.json", dict(status=reason, detail=detail,
            completed_at=utc_now(), active_seconds=self.budget.used, matching=self.current_check,
            launched_confirmation=any(r["metadata"]["cohort"] != "development" for r in self.records.values())))
        self.publish(reason)

    def calibration(self):
        self.phase = "6B: four-member retention calibration"
        starting = calibration.initial_recipes(self.cfg.fd_strength)
        for replica in range(4):
            for arm in ARMS:
                self.trajectory("development", replica, starting[arm])
        if self.cfg.force_calibration_failure:
            self.best_available()
            self.stop_calibration("calibration_infeasible", "deliberate smoke failure; confirmation gate stays closed")
            return False
        for reference_index, ridge in enumerate((10., 1., .1, 100.)):
            if self.store.is_complete(f"decision-reference-closed-{reference_index}"):
                continue
            self.reference_ridge = ridge
            reference = Recipe("A", ridge=ridge)
            if not self.can_admit([reference]):
                self.stop_calibration("calibration_budget_limited", "next complete reference cannot fit with closeout reserve")
                return False
            for replica in range(4):
                self.trajectory("development", replica, reference)
            for round_index in range(3):
                name = f"refinement-{reference_index}-{round_index}"
                uid = "decision-" + name
                if self.store.is_complete(uid):
                    proposal = read_json(self.store.unit_path(uid) / "decision.json")
                    self.rounds.append(proposal)
                    for member in range(4):
                        for mapping in proposal["recipes"]:
                            self.trajectory("development", member, Recipe(**mapping))
                    continue
                self.best_available()
                self.development_view()
                chosen = calibration.select_tuple(list(self.records.values()), reference_ridge=ridge,
                    tolerance=self.cfg.retention_tolerance, strength=self.cfg.fd_strength)
                if chosen is not None:
                    self.selection = self.decision("selection", lambda: self.selection_payload(chosen))
                    self.store.write_json(None, "selection.json", self.selection)
                    self.calibration_status = "passed"
                    self.validate_selection()
                    self.publish()
                    return True
                if round_index == 2:
                    break
                recipes = self.proposal_round()
                if not recipes:
                    break
                if not self.can_admit(recipes):
                    self.stop_calibration("calibration_budget_limited", "the complete balanced refinement round cannot fit")
                    return False
                proposal = self.decision(name, lambda: dict(reference_ridge=ridge,
                    recipes=[r.mapping() for r in recipes], prior_allowances=self.used_allowances(),
                    evidence=self.proposal_details,
                    reason="observed retention response and complete four-member matching; no confirmation outcomes"))
                self.rounds.append(proposal)
                for member in range(4):
                    for mapping in proposal["recipes"]:
                        self.trajectory("development", member, Recipe(**mapping))
            self.decision(f"reference-closed-{reference_index}", lambda: dict(
                reference_ridge=ridge, matching=self.current_check,
                reason="no passing tuple at the completed bounded search stage"))
        self.stop_calibration("calibration_infeasible", "no common tuple within bounded candidates and reference fallbacks")
        return False

    def run(self):
        self.import_history()
        self.publish()
        if not self.selection and not self.calibration():
            return
        self.validate_selection()
        fresh_started = any(r["metadata"]["cohort"] in ("fresh", "stationary") for r in self.records.values())
        if not fresh_started and self.selection["forecast_fresh_seconds"] > self.budget.remaining:
            self.stop_calibration("calibration_budget_limited", "calibration passed; the fixed confirmation allocation needs more compute")
            return
        self.phase = "6C: fresh paired comparison"
        blocks = [("fresh", i) for i in range(self.cfg.main_replicas)]
        for i in range(self.cfg.stationary_replicas):
            blocks.insert(min(len(blocks), 2 + 2*i), ("stationary", i))
        for cohort, replica in blocks:
            offset = self.cfg.seed_for(cohort, replica, "condition_order") % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                self.trajectory(cohort, replica, Recipe(**self.selection["recipes"][arm]))
        self.phase = "6D: inference and final integrity audit"
        self.current = {}
        self.publish()
        audit = self.decision("final-integrity", self.integrity_audit)
        self.store.write_json(None, "study_completed.json", dict(completed_at=utc_now(),
            active_seconds=self.budget.used, selection=self.selection, paths=len(self.records), integrity=audit))
        self.publish("completed")
        print(json.dumps(dict(status="completed", active_hours=self.budget.used/3600)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
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
        print(json.dumps(build_report(args.root, cfg=cfg)), flush=True)
        return 0
    store = budget = scheduler = None
    try:
        store, budget = open_store(cfg, args.root, args.history_root, args.data_root, resume=args.resume,
            clear_stop=args.clear_stop, allocation_hours=args.max_active_hours,
            max_wall_seconds=args.max_wall_seconds)
        with store.control.signal_handlers():
            for name in ("study_completed.json", "calibration_completed.json"):
                if (store.root / name).exists():
                    print(json.dumps(dict(status="already_closed", completion=read_json(store.root/name))), flush=True)
                    return 0
            scheduler = Scheduler(cfg, store, budget, load_data(args.data_root),
                                  history_root=args.history_root, max_units=args.max_units)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler:
            scheduler.publish("paused" if isinstance(error, StopRequested) else "resource_limited", figures=False)
        print(json.dumps(dict(status="paused", reason=str(error),
                             active_hours=budget.used/3600 if budget else None)), flush=True)
        return 0
    finally:
        if budget:
            budget.close()
        if store:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
