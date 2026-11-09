"""Execute Plan 18 Phase 7 with fixed STM ridge and atomic batch recovery."""

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
from ..config import ARMS, Recipe
from ..run import Scheduler as OriginalScheduler
from ..broad_drift.run import Scheduler as BroadScheduler
from ..retention_match.run import Scheduler as HistoryScheduler
from ..retention_match import runtime as history_runtime
from . import calibration
from .config import Config
from .runtime import (DATA_ROOT, DEFAULT_ROOT, HISTORY_ROOT, REPO, authorization, open_store,
                      read_json, read_exported_records, scalar_record, checked_scalar,
                      load_readonly_checkpoint)

TUNABLE = ("A", "C", "D", "E")


def validate_import_pairing(record, assets, initial_metadata, cfg):
    meta, recipe = record["metadata"], record["metadata"]["recipe"]
    member = meta["replica"]
    for name, seed in assets["seeds"].items():
        expected_seed = (cfg.seed_for("study-partition", 0, "test") if name == "evaluation_partition"
                         else cfg.seed_for("development", member, name))
        if seed != expected_seed:
            raise artifacts.ArtifactError(f"historical component seed differs: {name}")
    expected = dict(pairing_asset_hash=assets["asset_hash"],
        model_seed=cfg.seed_for("development", member, "initialization_model"),
        factor_seed=cfg.seed_for("development", member, "factors"))
    if (any(meta.get(k) != value or initial_metadata.get(k) != value for k,value in expected.items())
            or meta.get("global_sign") != assets["assignment"]["global_sign"]):
        raise artifacts.ArtifactError("historical trajectory asset, initialization or component seeds differ")
    numeric = dict(fd_dtype=cfg.fd_dtype if recipe["strength"] else "unused",
                   max_iter=cfg.max_iter, max_eval=cfg.max_eval)
    if meta.get("numerical_policy") != numeric:
        raise artifacts.ArtifactError("historical trajectory numerical policy differs")


class Scheduler(HistoryScheduler):
    def __init__(self, cfg, store, budget, raw, *, history_root=HISTORY_ROOT, max_units=None):
        super().__init__(cfg, store, budget, raw, history_root=history_root, max_units=max_units)
        self.phase, self.reference_ridge = "7A: historical validation and acceptance", 1.
        self.reference_arm = "B"

    def load_record(self, uid, metadata):
        return scalar_record(self.store.root, uid, self.budget.check)[0]

    def refresh_records(self):
        records, _ = read_exported_records(self.store.root, self.budget.check)
        self.records = {r["unit_id"]: r for r in records}

    def status(self, state="running"):
        result = OriginalScheduler.status(self, state)
        approved = getattr(self.budget, "authorization", {})
        baseline = approved.get("baseline_seconds", 0.)
        planned = {(member, recipe) for member in range(4)
                   for recipe in calibration.initial_recipes(self.cfg.fd_strength).values()}
        for decision in self.rounds:
            planned.update((member, Recipe(**mapping)) for member in range(4) for mapping in decision["recipes"])
        development = [r for r in self.records.values() if r["metadata"]["cohort"] == "development"]
        completed = {(r["metadata"]["replica"], Recipe(**r["metadata"]["recipe"])) for r in development}
        result.update(calibration_status=self.calibration_status,
            imported_development=sum(bool(r["metadata"].get("imported")) for r in self.records.values()),
            development_members=list(range(4)), reference_arm="B", reference_ridge=1.,
            current_check=self.current_check, calibration_check=self.current_check,
            candidate_allowances=self.used_allowances(), refinement_rounds=self.rounds,
            confirmation_gate="passed" if self.selection else "closed",
            acquisition_clock="outbound grids", analysis_outbound_count=self.cfg.analysis_novel_count,
            baseline_seconds=baseline, additional_active_seconds=max(0., self.budget.used-baseline),
            incremental_allowance_seconds=approved.get("incremental_allowance_seconds", 72000.))
        result["expected_development_paths"] = len(development) + len(planned-completed)
        return result

    def publish(self, status="running", *, figures=True):
        self.store.write_json(None, "status.json", self.status(status), finalizing=True)
        if (not figures or self.store.control.stop_requested or (self.store.root / "STOP").exists()
                or self.budget.remaining < 30):
            return
        from .report import build_report
        build_report(self.store.root, cfg=self.cfg, check=self.budget.check,
            notebook_path=self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None,
            history_root=self.history_root)
        self.last_report = time.monotonic()

    def import_history(self):
        uid = "import-development"
        if self.store.begin(uid, dict(kind="historical_development_import", members=list(range(4)))) is not None:
            rows, _ = history_runtime.read_exported_records(self.history_root, self.budget.check)
            references, seen = [], set()
            for record in rows:
                meta = record["metadata"]
                if meta.get("cohort") != "development" or meta.get("replica") not in range(4):
                    continue
                origin = REPO / meta.get("origin_root", str(self.history_root.relative_to(REPO)))
                origin_uid = meta.get("origin_unit", record["unit_id"])
                key = (str(origin.resolve()), origin_uid)
                if key in seen:
                    continue
                seen.add(key)
                original, seals = scalar_record(origin, origin_uid, self.budget.check)
                identity = artifacts._identity(origin)
                # Full initial import verification is recorded once; subsequent scalar reads
                # validate only the completed exports consumed by analysis.
                artifacts._unit_record(origin, origin_uid, identity["identity_hash"], {},
                                       self.budget.check)
                expected = summarize_path(original["metrics"], self.cfg)
                for field in ("Havg", "H100", "novel_auc"):
                    if not math.isclose(original["summary"][field], expected[field], rel_tol=1e-10, abs_tol=1e-10):
                        raise artifacts.ArtifactError("historical summary differs from sealed metrics")
                references.append(dict(root=str(origin.relative_to(REPO)), unit_id=origin_uid,
                    identity_hash=identity["identity_hash"], seals=seals))
            if {r["metadata"]["replica"] for r in rows if r["metadata"].get("cohort") == "development"} != set(range(4)):
                raise artifacts.ArtifactError("all four historical development members required")
            h_identity = artifacts._identity(self.history_root)
            old_root = REPO / h_identity["identity"]["historical_development"]["root"]
            dependencies = []
            pairing = {}
            for member in range(4):
                origin = old_root if member < 2 else self.history_root
                origin_id = artifacts._identity(origin)["identity_hash"]
                for kind in ("assets", "initial"):
                    name = f"{kind}-development-{member:03}"
                    artifacts._unit_record(origin, name, origin_id, {}, self.budget.check)
                    dependencies.append(dict(member=member, kind=kind, root=str(origin.relative_to(REPO)),
                        unit_id=name, identity_hash=origin_id,
                        completion=history_runtime.original.file_digest(origin / "units" / name / "COMPLETED", self.budget.check)))
                assets, _ = checked_scalar(origin, f"assets-development-{member:03}", "summary.json", self.budget.check)
                manifest, _ = checked_scalar(origin, f"initial-development-{member:03}", "manifest.json", self.budget.check)
                pairing[member] = (assets, manifest["metadata"])
            for record in rows:
                meta = record["metadata"]
                if meta.get("cohort") == "development":
                    validate_import_pairing(record, *pairing[meta["replica"]], self.cfg)
            self.store.write_json(uid, "references.json", dict(schema=1, units=references,
                dependencies=dependencies, roles="development 0--3 only; transitive origins deduplicated"))
            self.store.complete(uid)
        self.refresh_records()
        numeric = read_json(self.history_root / "numerics.json")
        if (numeric["fd_dtype"], numeric["max_iter"], numeric["max_eval"]) != (
                self.cfg.fd_dtype, self.cfg.max_iter, self.cfg.max_eval):
            raise artifacts.ArtifactError("historical numerical policy differs")
        self.numeric = numeric
        self.store.write_json(None, "numerics.json", numeric)
        self.development_view()

    def historical_state(self, member, kind):
        refs, _ = checked_scalar(self.store.root, "import-development", "references.json", self.budget.check)
        matches = [r for r in refs["dependencies"] if r["member"] == member and r["kind"] == kind]
        if len(matches) != 1:
            raise artifacts.ArtifactError("missing unique historical initialization/asset")
        reference = matches[0]
        origin = REPO / reference["root"]
        if (artifacts._identity(origin)["identity_hash"] != reference["identity_hash"]
                or history_runtime.original.file_digest(origin / "units" / reference["unit_id"] / "COMPLETED",
                    self.budget.check) != reference["completion"]):
            raise artifacts.ArtifactError("historical initialization/asset seal changed")
        return load_readonly_checkpoint(origin, reference["unit_id"], self.budget.check)

    def assets(self, cohort, replica):
        if cohort == "development":
            key = (cohort, replica)
            if key not in self.assets_cache:
                self.assets_cache[key] = self.historical_state(replica, "assets")
            return self.assets_cache[key]
        return BroadScheduler.assets(self, cohort, replica)

    def initial(self, cohort, replica):
        if cohort == "development":
            key = (cohort, replica)
            if key not in self.initial_cache:
                self.initial_cache[key] = self.historical_state(replica, "initial")
            return self.initial_cache[key]
        return OriginalScheduler.initial(self, cohort, replica)

    def compatible(self, recipe):
        return calibration.compatible(recipe, strength=self.cfg.fd_strength)

    def development_view(self):
        self.member_summaries = [dict(recipe=r["metadata"]["recipe"], replica=r["metadata"]["replica"],
            imported=bool(r["metadata"].get("imported")),
            **{k:r["summary"].get(k) for k in ("Havg", "H100", "novel_auc", "learning_seconds")})
            for r in self.records.values() if r["metadata"]["cohort"] == "development"]
        self.store.write_json(None, "development.json", dict(members=list(range(4)),
            member_summaries=self.member_summaries, current_check=self.current_check,
            reference_arm="B", reference_ridge=1., refinement_rounds=self.rounds,
            candidate_allowances=self.used_allowances()))

    def used_allowances(self):
        counts = dict.fromkeys(TUNABLE, 0)
        # Status must remain writable after a stop request. Read sealed decisions
        # without re-entering the training control's stop check during finalization.
        for unit in artifacts.iter_units(self.store.root, completed_only=True, verify=False):
            if unit["metadata"].get("name", "").startswith("refinement-"):
                decision, _ = checked_scalar(self.store.root, unit["unit_id"], "decision.json")
                for recipe in decision["recipes"]:
                    if recipe["arm"] not in counts:
                        raise artifacts.ArtifactError("fixed B cannot consume a refinement slot")
                    counts[recipe["arm"]] += 1
        return counts

    def all_candidate_rows(self):
        result = {arm:[] for arm in ARMS}
        recipes = {Recipe(**r["metadata"]["recipe"]) for r in self.records.values()
                   if r["metadata"]["cohort"] == "development"}
        for recipe in sorted(recipes, key=lambda r:r.key):
            if not self.compatible(recipe):
                continue
            try:
                result[recipe.arm].append(calibration.aggregate(list(self.records.values()), recipe))
            except ValueError as error:
                if "missing or incomplete" not in str(error):
                    raise
        return result

    def closeout_reserve(self):
        return 15. if self.cfg.smoke else .5*3600

    def confirmation_reserve(self, *, remaining=False):
        seconds = 0.
        for cohort, count in (("fresh", self.cfg.main_replicas), ("stationary", self.cfg.stationary_replicas)):
            for member in range(count):
                done = {r["metadata"]["recipe"]["arm"] for r in self.records.values()
                    if remaining and r["metadata"]["cohort"] == cohort and r["metadata"]["replica"] == member}
                seconds += sum(self.path_cost(a) for a in ARMS if a not in done)
                if (not remaining or (not done and (cohort,member) not in self.initial_cache
                        and not self.store.is_complete(f"initial-{cohort}-{member:03}"))):
                    seconds += self.initial_cost()
        return seconds * 1.25 + self.closeout_reserve()

    def can_admit(self, recipes):
        needed = sum(self.path_cost(recipe.arm) for recipe in recipes for member in range(4)
            if not any(r["metadata"]["cohort"] == "development" and r["metadata"]["replica"] == member
                and r["metadata"]["recipe"] == recipe.mapping() for r in self.records.values()))
        return needed*1.25 + self.confirmation_reserve() <= self.budget.remaining

    def proposal_round(self):
        rows = self.best_available()
        reference = rows["B"]
        counts = self.used_allowances()
        self.proposal_details = []
        proposals = []
        for arm in TUNABLE:
            if counts[arm] >= self.cfg.refinements_per_arm:
                continue
            # Half-band proximity to B guarantees the full all-arm range.
            if max(abs(rows[arm][k]-reference[k]) for k in ("Havg", "H100")) <= self.cfg.retention_tolerance/2:
                continue
            used = {r["recipe"]["ridge"] for r in self.all_candidate_rows()[arm]}
            for decision in self.rounds:
                used.update(r["ridge"] for r in decision["recipes"] if r["arm"] == arm)
            detail = calibration.ridge_proposal(arm, list(self.records.values()), used,
                tolerance=self.cfg.retention_tolerance, strength=self.cfg.fd_strength)
            if detail is not None:
                base = calibration.initial_recipes(self.cfg.fd_strength)[arm]
                proposals.append(Recipe(arm, detail["ridge"], base.strength, base.process_scale))
                self.proposal_details.append(dict(arm=arm, **detail))
        return proposals

    def selection_payload(self, selected):
        recipes = selected["recipes"]
        if recipes["B"] != calibration.initial_recipes(self.cfg.fd_strength)["B"].mapping():
            raise artifacts.ArtifactError("B must remain fixed at ridge 1")
        rows = {a:calibration.aggregate(list(self.records.values()), Recipe(**recipes[a])) for a in ARMS}
        check = calibration.matching(rows, tolerance=self.cfg.retention_tolerance)
        if not check["passed"] or self.cfg.force_calibration_failure:
            raise artifacts.ArtifactError("refusing to seal unmatched low-ridge calibration")
        return dict(frozen=True, frozen_at=utc_now(), calibration_status="passed", recipes=recipes,
            calibration=rows, matching=check, members=list(range(4)), reference_arm="B", reference_ridge=1.,
            efficiency_reference_arm="A", candidate_allowances=self.used_allowances(),
            evidence_units=sorted(r["unit_id"] for r in self.records.values()
                if r["metadata"]["cohort"] == "development"
                and r["metadata"]["recipe"] == recipes[r["metadata"]["recipe"]["arm"]]),
            source_identity=self.store.identity_hash, config_hash=self.cfg.scientific_hash,
            main_replicas=self.cfg.main_replicas, stationary_replicas=self.cfg.stationary_replicas,
            seconds_per_path=statistics.mean(self.path_cost(a) for a in ARMS),
            forecast_fresh_seconds=self.confirmation_reserve(),
            bootstrap_seed=self.cfg.seed_for("fresh", 0, "bootstrap"), numerical_policy=self.numeric)

    def validate_selection(self):
        if not self.selection or not self.store.is_complete("decision-selection"):
            raise artifacts.ArtifactError("confirmation requires a sealed passing calibration")
        sealed, _ = checked_scalar(self.store.root, "decision-selection", "decision.json", self.budget.check)
        if sealed != self.selection or sealed.get("source_identity") != self.store.identity_hash:
            raise artifacts.ArtifactError("selection differs from its sealed scientific decision")
        if (sealed.get("reference_arm") != "B" or sealed.get("efficiency_reference_arm") != "A"
                or sealed["recipes"]["B"] != calibration.initial_recipes(self.cfg.fd_strength)["B"].mapping()
                or sealed.get("main_replicas") != self.cfg.main_replicas
                or sealed.get("stationary_replicas") != self.cfg.stationary_replicas):
            raise artifacts.ArtifactError("fixed B, reference role, or confirmation allocation changed")
        rows = {a:calibration.aggregate(list(self.records.values()), Recipe(**sealed["recipes"][a])) for a in ARMS}
        check = calibration.matching(rows, tolerance=self.cfg.retention_tolerance)
        if (not check["passed"] or check != sealed["matching"] or rows != sealed["calibration"]
                or sealed.get("calibration_status") != "passed" or sealed.get("members") != list(range(4))):
            raise artifacts.ArtifactError("calibration seal no longer proves four-member matching")
        self.current_check = check
        return sealed

    def stop_calibration(self, reason, detail):
        self.calibration_status, self.phase, self.current = reason, "7C: calibration closed without confirmation", {}
        self.development_view()
        self.store.write_json(None, "calibration_completed.json", dict(status=reason, detail=detail,
            completed_at=utc_now(), active_seconds=self.budget.used, matching=self.current_check,
            launched_confirmation=any(r["metadata"]["cohort"] != "development" for r in self.records.values())))
        self.publish(reason)

    def calibrate(self):
        starting = calibration.initial_recipes(self.cfg.fd_strength)
        self.phase = "7B: fixed STM ridge-1 reference"
        if not self.can_admit([starting["B"]]):
            self.stop_calibration("calibration_budget_limited", "fixed B reference plus confirmation reserve cannot fit")
            return False
        for member in range(4):
            self.trajectory("development", member, starting["B"])
        reference = self.decision("retention-reference", lambda: dict(reference_arm="B", reference_ridge=1.,
            summary=calibration.aggregate(list(self.records.values()), starting["B"])))
        if reference["summary"] != calibration.aggregate(list(self.records.values()), starting["B"]):
            raise artifacts.ArtifactError("fixed B retention reference changed")
        self.phase = "7C: four-member recalibration"
        if not self.can_admit([starting[a] for a in TUNABLE]):
            self.stop_calibration("calibration_budget_limited", "starting recipes plus confirmation reserve cannot fit")
            return False
        for member in range(4):
            for arm in TUNABLE:
                self.trajectory("development", member, starting[arm])
        if self.cfg.force_calibration_failure:
            self.best_available()
            self.stop_calibration("calibration_infeasible", "deliberate smoke failure; confirmation gate stays closed")
            return False
        for index in range(self.cfg.refinements_per_arm + 1):
            name, uid = f"refinement-{index}", f"decision-refinement-{index}"
            if self.store.is_complete(uid):
                decision = read_json(self.store.unit_path(uid) / "decision.json")
                if not any(d == decision for d in self.rounds):
                    self.rounds.append(decision)
                for member in range(4):
                    for mapping in decision["recipes"]:
                        self.trajectory("development", member, Recipe(**mapping))
                continue
            self.best_available()
            self.development_view()
            chosen = calibration.select_tuple(list(self.records.values()),
                tolerance=self.cfg.retention_tolerance, strength=self.cfg.fd_strength)
            if chosen is not None:
                self.selection = self.decision("selection", lambda:self.selection_payload(chosen))
                self.store.write_json(None, "selection.json", self.selection)
                self.calibration_status = "passed"
                self.validate_selection()
                self.publish()
                return True
            if index == self.cfg.refinements_per_arm:
                break
            proposals = self.proposal_round()
            if not proposals:
                break
            if not self.can_admit(proposals):
                self.stop_calibration("calibration_budget_limited", "complete refinement round would spend the confirmation reserve")
                return False
            decision = self.decision(name, lambda:dict(reference_arm="B", reference_ridge=1.,
                recipes=[r.mapping() for r in proposals], prior_allowances=self.used_allowances(),
                evidence=self.proposal_details, reason="development retention response; no confirmation outcomes"))
            self.rounds.append(decision)
            for member in range(4):
                for mapping in decision["recipes"]:
                    self.trajectory("development", member, Recipe(**mapping))
        self.stop_calibration("calibration_infeasible", "no matched five-arm tuple at fixed B/ridge1 within bounded search")
        return False

    def integrity_audit(self):
        self.validate_selection()
        read_exported_records(self.store.root, self.budget.check)
        for name, digest in self.store.identity["sources"].items():
            if history_runtime.original.file_digest(REPO / name, self.budget.check) != digest:
                raise artifacts.ArtifactError(f"frozen scientific source changed: {name}")
        return BroadScheduler.integrity_audit(self)

    def run(self):
        self.import_history()
        self.publish()
        if not self.selection and not self.calibrate():
            return
        self.validate_selection()
        started = any(r["metadata"]["cohort"] != "development" for r in self.records.values())
        if not started and self.confirmation_reserve() > self.budget.remaining:
            self.stop_calibration("calibration_budget_limited", "passing calibration; full confirmation plus reserve cannot fit")
            return
        self.phase = "7D: fresh paired comparison"
        blocks = [("fresh", i) for i in range(self.cfg.main_replicas)]
        for i in range(self.cfg.stationary_replicas):
            blocks.insert(min(len(blocks), 2+3*i), ("stationary", i))
        for cohort, member in blocks:
            offset = self.cfg.seed_for(cohort, member, "condition_order") % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                done = any(r["metadata"]["cohort"] == cohort and r["metadata"]["replica"] == member
                    and r["metadata"]["recipe"]["arm"] == arm for r in self.records.values())
                if not done and self.confirmation_reserve(remaining=True) > self.budget.remaining:
                    raise ResourceLimit("remaining confirmation paths and closeout reserve need more compute")
                self.trajectory(cohort, member, Recipe(**self.selection["recipes"][arm]))
        self.phase, self.current = "7E: final inference and integrity audit", {}
        self.publish()
        audit = self.decision("final-integrity", self.integrity_audit)
        self.store.write_json(None, "study_completed.json", dict(completed_at=utc_now(),
            active_seconds=self.budget.used, selection=self.selection, paths=len(self.records), integrity=audit))
        self.publish("completed")
        print(json.dumps(dict(status="completed", cumulative_active_hours=self.budget.used/3600,
            additional_active_hours=(self.budget.used-self.budget.authorization["baseline_seconds"])/3600)), flush=True)


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
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args(argv)
    cfg = Config.read(args.config)
    if args.request_stop:
        request_stop(args.root)
        print("Stop requested; recoverable boundary exit pending.", flush=True)
        return 0
    if args.report_only:
        from .report import build_report
        print(json.dumps(build_report(args.root, cfg=cfg, history_root=args.history_root)), flush=True)
        return 0
    store = budget = scheduler = None
    try:
        store, budget = open_store(cfg, args.root, args.history_root, args.data_root,
            resume=args.resume, clear_stop=args.clear_stop, max_wall_seconds=args.max_wall_seconds)
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
