"""Resumable Plan 18 Phase 9E calibration and eight fresh R/K pairs."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit, utc_now
from mnist_experiment.fisher_memory.learner import StopRequested as LearnerStop
from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
from mnist_experiment.fisher_memory_regression.runtime import (
    read_json, setup_torch, provenance, dataset_manifest, content_hash,
)
from mnist_experiment.combined_memory.reproduction.audit import load_checkpoint, compare_array
from mnist_experiment.combined_memory.replicated_combination.run import Study as PreviousStudy
from mnist_experiment.combined_memory.replicated_combination.runner import record
from . import PROTOCOL
from .config import (REPO, ROOT, PREVIOUS, HISTORY, FAMILY, DATA, Policy, POLICIES,
                     configuration, PROCESS_SCALE, BOOTSTRAP_SEED)
from .runtime import (authorize, sources, peer_seconds, external_bytes, RuntimeBudget,
                      ReuseStore, check_external, atomic_json)
from .runner import Runner
from . import calibration as cal
from .analysis import analyze, validate_record

TERMINAL = ("completed", "calibration_infeasible", "calibration_budget_limited", "fidelity_passed")


class Study(PreviousStudy):
    def __init__(self, args, store, budget, history):
        super().__init__(args, store, budget, history)
        self.stage = "9E.1 acceptance"
        self.view = ReuseStore(store, reuse=not args.smoke)
        self.last_report = 0.
        if not args.smoke:
            refs, seals = [], {}
            for path in sorted((PREVIOUS / "units").glob("development-R--*trajectory*")):
                checked = check_external(path.name, budget.check)
                if checked["status"] != "completed":
                    raise ValueError("incomplete reused R reference")
                row = record(type("Reference", (), {"unit_path": lambda _, u: PREVIOUS / "units" / u})(), path.name)
                validate_record(row)
                refs.append(row)
                seals[path.name] = checked["completion"]["seal_hash"]
            if sorted(r["replica"] for r in refs) != list(range(4)):
                raise ValueError("exactly four original development R references required")
            self.seal("reused-references", dict(root=str(PREVIOUS.relative_to(REPO)),
                identity_hash=artifacts._identity(PREVIOUS)["identity_hash"], seals=seals))
            self.records.extend(refs)
            shadow = read_json(PREVIOUS / "units/decision-process-scale/result.json")
            if shadow["process_scale"] != PROCESS_SCALE:
                raise ValueError("frozen observer scale changed")
            if any(e["metadata"].get("namespace") == "fresh" and
                   e["metadata"].get("kind") == "grid_registration_trajectory"
                   for e in artifacts.iter_units(PREVIOUS, verify=False)):
                raise ValueError("original fresh namespace has exposed outcomes; use a new explicit seed version")

    def runner(self, namespace):
        if namespace not in self.runners:
            cfg = configuration(self.history, namespace, self.args.device, self.args.smoke)
            self.runners[namespace] = Runner(cfg, self.view, self.source, self.budget, namespace,
                progress=self.progress, stop_at=self.args.smoke_stop_at)
        return self.runners[namespace]

    def costs(self):
        result = dict(R=600., K=900.)
        for p in POLICIES:
            values = [r["active_seconds"] for r in self.records if r["policy"] == p]
            if values:
                result[p] = max(values)
        return result

    def forecast(self):
        costs = self.costs()
        done = {(r["replica"], r["policy"]) for r in self.records if r["namespace"] == "fresh"}
        return sum(costs[p] for r in range(self.args.replicas) for p in POLICIES if (r, p) not in done)+1800.

    def reserve(self, extra=(), validation=True):
        costs = self.costs()
        seconds = self.forecast()-1800 + self.args.replicas*120
        seconds += sum(costs[p] for p in extra) + (2*costs["K"] if validation else 0.)
        required = 1.25*seconds+1800
        return dict(passed=required <= self.budget.remaining, required_seconds=required,
                    remaining_seconds=self.budget.remaining, measured_path_costs=costs)

    def progress(self, force=False, **updates):
        self.budget.check()
        self.current = updates.get("current", self.current)
        fresh = [r for r in self.records if r["namespace"] == "fresh"]
        development = [r for r in self.records if r["namespace"] == "development"]
        blocks = sum({r["policy"] for r in fresh if r["replica"] == i} == set(POLICIES)
                     for i in range(self.args.replicas))
        status = dict(status=self.state, stage=self.stage, current=self.current, reason=self.reason,
            pid=os.getpid() if self.state == "running" else None, heartbeat=utc_now(),
            fresh_completed=len(fresh), planned_fresh=2*self.args.replicas,
            fresh_blocks_completed=blocks, planned_blocks=self.args.replicas,
            development_completed=sum(r["policy"] == "K" for r in development),
            development_references_reused=0 if self.args.smoke else 4,
            accepted_checkpoint=updates.get("novel_count"), charged_hours=self.budget.used/3600,
            remaining_hours=self.budget.remaining/3600,
            forecast_hours=0. if self.state in TERMINAL else self.forecast()/3600)
        self.store.write_json(None, "status.json", status, finalizing=True)
        if not force and time.monotonic()-self.last_report < 900:
            return
        design = self.decision("design")
        selection = (design or {}).get("selection")
        candidates = cal.candidates(development) if development else []
        if selection is None and development and any(r["policy"] == "K" for r in development):
            selection = cal.selection(development, require_match=False)
        ridge = selection["ridge"] if selection else 1.
        curves = {}
        selected = {p: [r for r in development if r["policy"] == p and r["recipe"]["ridge"] == (1. if p=="R" else ridge)]
                    for p in POLICIES}
        common = sorted(set(r["replica"] for r in selected["R"]) & set(r["replica"] for r in selected["K"]))
        for p, records in selected.items():
            rows = [r for r in records if r["replica"] in common]
            if rows:
                curves[p] = dict(n=[m["n"] for m in rows[0]["metrics"]], **{
                    domain: dict(mean_mse=np.mean([[m[field] for m in r["metrics"]] for r in rows], axis=0).tolist())
                    for domain, field in (("query", "novel_mse"), ("base", "base_mse"))})
        analysis = self.decision("analysis")
        if analysis is None and blocks >= 2:
            analysis = analyze(fresh, draws=100, partial=True, check=self.budget.check)
        text = "Calibration is pending. Both mean retention gaps must be within .01 pixels."
        if selection:
            text = (f"{(design or {}).get('status', 'Development only')}; ridge c={ridge:.6g}. "
                f"Kalman-minus-fixed retention gaps: average {selection['gaps']['Havg']:+.5f} px; "
                f"endpoint {selection['gaps']['H100']:+.5f} px. "
                "These descriptive values reuse four development controls; they are not fresh efficacy evidence.")
        conclusion = "The R/K comparison is underway. The earlier A–E gate remains closed."
        if self.state in ("calibration_infeasible", "calibration_budget_limited"):
            conclusion = ("The bounded R/K retention calibration did not authorize fresh confirmation. "
                "The common-ridge and calibrated development paths show the observed acquisition/retention trade-off. "
                "No fresh effect or sample-saving conclusion is available; this is not a negative confirmation result.")
        elif analysis:
            d = analysis["contrast"]
            ci = d["acquisition_difference"]["ci"]
            finding = "favors Kalman pi" if ci[1] < 0 else "favors fixed pi" if ci[0] > 0 else "does not resolve a difference"
            conclusion = (f"The {'completed' if analysis['is_final'] else 'interim'} paired novel-risk interval {finding}. "
                f"Both fresh retention equivalence checks {'pass' if d['retention_equivalent'] else 'are not established'}. "
                "Compare retention and measured compute alongside prediction; the result concerns this frozen policy and regime. "
                "Ridge calibration, if needed, means the contrast is between policies rather than adaptive timing alone.")
        snapshot = dict(protocol=PROTOCOL, progress=status, ridge=ridge,
            development=dict(candidates=candidates, curves=curves, paired_members=common),
            calibration=design, calibration_text=text, analysis=analysis,
            diagnostics=self.diagnostics(fresh or [r for r in development if r["policy"]=="K" and r["recipe"]["ridge"]==ridge]),
            conclusion=conclusion)
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        from .report import publish
        publish(snapshot, self.args.notebook, self.store)
        self.last_report = time.monotonic()

    def diagnostics(self, rows):
        maps = []
        for row in rows:
            if row["policy"] != "K":
                continue
            grouped = {}
            for fit in read_json(self.store.unit_path(row["physical_unit"]) / "fits.json"):
                grouped.setdefault(fit["group"], []).append(fit)
            maps.append(grouped)
        if not maps:
            return {}
        groups = sorted(set.intersection(*(set(m) for m in maps)))
        result = dict(observations=[sum(max(f["observations"] for f in m[g]) for m in maps)/len(maps) for g in groups])
        for field in ("applied_pi", "kalman_level_gain_mean", "kalman_velocity_gain_mean", "motion_energy", "noise_scale"):
            if all(any(field in f for f in m[g]) for m in maps for g in groups):
                result[field] = [sum(next(f[field] for f in m[g] if field in f) for m in maps)/len(maps) for g in groups]
        return result if "applied_pi" in result else {}

    def fidelity(self):
        if self.decision("fidelity"):
            return
        self.stage = "9E.1 forced-pi F1 fidelity"
        runner = self.runner("acceptance")
        runner.forced_pi = .025
        if self.args.smoke:
            reference = self.job("acceptance", 0, Policy("R"))
            reference_root, reference_unit = self.store.root, reference["physical_unit"]
        else:
            matches = list((PREVIOUS / "units").glob("acceptance-R--*trajectory*"))
            if len(matches) != 1:
                raise ValueError("unique sealed original F1 acceptance reference required")
            reference_root, reference_unit = PREVIOUS, matches[0].name
            check_external(reference_unit, self.budget.check)
        current = self.job("acceptance", 0, Policy("K"))
        runner.forced_pi = None
        checks = []
        ref_path = reference_root / "units" / reference_unit
        old_metrics = read_json(ref_path / "metrics.json")
        new_metrics = current["metrics"]
        for field in ("n", "observations", "base_observations"):
            if [r[field] for r in old_metrics] != [r[field] for r in new_metrics]:
                raise ValueError("forced-pi fidelity clock mismatch")
        for field in ("base_mse", "novel_mse", "base_nll", "novel_nll"):
            checks.append(compare_array([r[field] for r in old_metrics], [r[field] for r in new_metrics], field))
        old = load_checkpoint(reference_root, reference_unit)
        new = self.store.load_checkpoint(current["physical_unit"])
        for field in ("theta", "anchor_L", "ltm_indices"):
            checks.append(compare_array(old["learner"][field], new["learner"][field], field))
        for field in ("factor", "residual_diagonal"):
            checks.append(compare_array(old["learner"]["fisher_L"][field], new["learner"]["fisher_L"][field], "archive_"+field))
        if old["retirements"] != new["retirements"] or old["learner"]["replay"] != new["learner"]["replay"]:
            raise ValueError("forced-pi FIFO or retirement mismatch")
        old_fits = read_json(ref_path / "fits.json")
        fits = read_json(self.store.unit_path(current["physical_unit"]) / "fits.json")
        for field in ("step", "n", "observations", "selected_replay_ids", "solver_reason"):
            if [f.get(field) for f in old_fits] != [f.get(field) for f in fits]:
                raise ValueError("forced-pi solve scheduling mismatch: "+field)
        for field in ("initial_objective", "final_objective", "gradient_inf_norm"):
            checks.append(compare_array([f[field] for f in old_fits], [f[field] for f in fits], field))
        self.seal("fidelity", dict(status="passed", checks=checks, reference_root=str(reference_root),
            reference_unit=reference_unit, adaptive_unit=current["physical_unit"], smoke_only=self.args.smoke))

    def development_records(self):
        return [r for r in self.records if r["namespace"] == "development"]

    def calibrate(self):
        if self.decision("design"):
            return self.decision("design")
        self.stage = "9E.2 common-ridge development"
        for replica in range(4):
            self.job("development", replica, Policy("K"))
            self.progress(force=True)
        selected = cal.selection(self.development_records())
        if selected is None:
            for i in range(2):
                self.stage = "9E.2 first-pair ridge calibration"
                name = f"proposal-first-{i}"
                proposal = self.decision(name)
                if proposal is None:
                    proposal = cal.propose(self.development_records())
                    reservation = self.reserve(extra=("K", "K"), validation=True)
                    proposal = proposal if proposal and reservation["passed"] else dict(skipped="no proposal or resource reserve", reservation=reservation)
                    self.seal(name, proposal)
                if "skipped" not in proposal:
                    for replica in (0, 1):
                        self.job("development", replica, Policy("K", proposal["ridge"]))
                    self.progress(force=True)
                if cal.selection(self.development_records(), members=(0, 1)):
                    break
            proposal = self.decision("validation-proposal")
            if proposal is None:
                choice = cal.selection(self.development_records(), members=(0, 1), require_match=False)
                proposal = self.seal("validation-proposal", dict(ridge=choice["ridge"]))
            self.stage = "9E.2 four-member validation"
            for replica in (2, 3):
                self.job("development", replica, Policy("K", proposal["ridge"]))
            selected = cal.selection(self.development_records())
        if selected is None:
            self.stage = "9E.2 final bounded repair"
            repair = self.decision("proposal-repair")
            if repair is None:
                rows = cal.candidates(self.development_records())
                incomplete = [r for r in rows if r["policy"]=="K" and not r["complete"]]
                first = cal.selection(self.development_records(), members=(0, 1), require_match=False)
                if incomplete and first["passed"]:
                    repair = dict(ridge=first["ridge"], reason="complete first-pair candidate")
                else:
                    repair = cal.propose(self.development_records(), members=(0, 1, 2, 3))
                reservation = self.reserve(extra=("K",)*4, validation=False)
                repair = repair if repair and reservation["passed"] else dict(skipped="no repair or resource reserve", reservation=reservation)
                self.seal("proposal-repair", repair)
            if "skipped" not in repair:
                for replica in range(4):
                    self.job("development", replica, Policy("K", repair["ridge"]))
                self.progress(force=True)
            selected = cal.selection(self.development_records())
        if selected is None:
            return self.seal("design", dict(status="calibration_infeasible",
                selection=cal.selection(self.development_records(), require_match=False)))
        reservation = self.reserve(validation=False)
        return self.seal("design", dict(status="passed" if reservation["passed"] else "calibration_budget_limited",
            selection=selected, reservation=reservation, source_hash=self.store.identity["source_hash"],
            fresh_seed=self.runner("fresh").cfg.seed, replicas=8))

    def execute(self):
        self.progress(force=True)
        self.seal("cohort", dict(protocol=PROTOCOL, policies=list(POLICIES),
            fresh_replicas=list(range(self.args.replicas)), process_scale=PROCESS_SCALE,
            development_seed=self.runner("development").cfg.seed,
            fresh_seed=self.runner("fresh").cfg.seed, smoke=self.args.smoke))
        self.fidelity()
        if self.args.stage == "fidelity":
            self.state, self.stage = "fidelity_passed", "9E.1 fidelity passed"
            return
        if self.args.smoke:
            ridge = 1.
        else:
            design = self.calibrate()
            if design["status"] != "passed":
                self.state, self.stage, self.current = design["status"], "9E.2 calibration closeout", None
                return
            ridge = design["selection"]["ridge"]
        self.stage = "9E.3 fresh paired confirmation"
        for replica in range(self.args.replicas):
            for name in (POLICIES if replica % 2 == 0 else POLICIES[::-1]):
                self.job("fresh", replica, Policy(name, ridge if name=="K" else 1.))
            self.progress(force=True)
        self.stage = "9E.4 inference and closeout"
        if self.decision("analysis") is None:
            result = analyze([r for r in self.records if r["namespace"]=="fresh"],
                draws=50 if self.args.smoke else 2000, check=self.budget.check,
                load_bootstrap=lambda i: self.decision(f"bootstrap-{i:04d}"),
                save_bootstrap=lambda i, v: self.seal(f"bootstrap-{i:04d}", v))
            self.seal("analysis", result)
        self.integrity()
        self.state, self.stage, self.current = "completed", "Phase 9E complete", None
        self.progress(force=True)
        self.store.write_json(None, "study_completed.json", dict(status="completed", protocol=PROTOCOL,
            completed_at=utc_now(), charged_seconds=self.budget.used, smoke=self.args.smoke))

    def integrity(self):
        if self.decision("integrity"):
            return
        if content_hash(sources()) != self.store.identity["source_hash"]:
            raise artifacts.ImmutableError("scientific sources changed during execution")
        units = list(artifacts.iter_units(self.store.root, full=True, check=self.budget.check))
        if any(u["status"] != "completed" for u in units):
            raise ValueError("unsealed final unit")
        for row in self.records:
            if row["namespace"] != "fresh":
                continue
            state = self.store.load_checkpoint(row["physical_unit"])
            retired = [r["exposure"] for r in state["retirements"]]
            archive = state["learner"]["fisher_L"]
            if len(set(retired)) != len(retired) or not all(torch.isfinite(archive[f]).all() for f in ("factor", "residual_diagonal")):
                raise ValueError("invalid retirement/archive state")
            if row["policy"]=="K" and state["pending_scores"]:
                raise ValueError("unassimilated final group")
        self.seal("integrity", dict(status="passed", units=len(units), source_hash=self.store.identity["source_hash"]))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--notebook", type=Path, default=REPO/"mnist_experiment/plan18_results.ipynb")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--stage", choices=("all", "fidelity", "status", "stop", "report"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clear-stop", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--max-paths", type=int)
    p.add_argument("--smoke-stop-at", choices=("pretrain", "accepted", "mid_solve", "eviction", "split_group"))
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.stage in ("status", "stop", "report"):
        if args.stage == "status":
            print(json.dumps(read_json(args.root/"status.json"), indent=2))
        elif args.stage == "stop":
            (args.root/"STOP").touch()
        else:
            from .report import publish
            publish(read_json(args.root/"report_snapshot.json"), args.notebook,
                    type("ReadOnlyRoot", (), {"root": args.root})())
        return 0
    if (args.device != "cuda" or args.replicas != 8) and not args.smoke:
        raise ValueError("production requires CUDA and eight pairs")
    if args.smoke_stop_at and not args.smoke:
        raise ValueError("artificial stops are smoke-only")
    if args.smoke and (args.root.resolve()==ROOT.resolve() or args.notebook.resolve()==(REPO/"mnist_experiment/plan18_results.ipynb").resolve()):
        raise ValueError("smoke requires separate root/notebook")
    if args.replicas < 2 or (args.max_paths is not None and args.max_paths < 1):
        raise ValueError("invalid replica/path count")
    approval = authorize()
    history = read_json(HISTORY/"resolved_config.json")
    cfg = configuration(history, "fresh", args.device, args.smoke)
    setup_torch(cfg)
    code, data = sources(), dataset_manifest(args.data_root)
    if data != read_json(HISTORY/"provenance.json")["dataset_manifest"]:
        raise ValueError("historical data changed")
    identity = dict(protocol=PROTOCOL, allocation_family="plan18-phase9", role="smoke" if args.smoke else "production",
        source_hash=content_hash(code), sources=code, data_hash=content_hash(data),
        config_hash=content_hash(dict(config=cfg.mapping(), replicas=args.replicas)),
        authorization_sha256=approval["sha256"], previous_identity=artifacts._identity(PREVIOUS)["identity_hash"])
    if not args.smoke and args.stage == "all":
        accepted = read_json(FAMILY/"phase9e_acceptance.json")
        if accepted.get("status") != "passed" or accepted.get("source_hash") != identity["source_hash"]:
            raise ValueError("matching Phase 9E acceptance required")
    other = peer_seconds(args.root)
    with artifacts.Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
            external_bytes=external_bytes(args.root, args.data_root), free_reserve_bytes=20_000_000_000,
            finalization_reserve_bytes=64_000_000, max_active_seconds=max(0., 72000.-other)) as store:
        budget = RuntimeBudget(store, other_seconds=other, allocation_seconds=72000.)
        study = None
        try:
            if not (args.root/"provenance.json").exists():
                detail = provenance(cfg, data)
                detail.update(protocol=PROTOCOL, sources=code, source_hash=identity["source_hash"], authorization=approval)
                store.write_json(None, "provenance.json", detail)
                store.write_json(None, "resolved_config.json", cfg.mapping())
            with store.control.signal_handlers():
                study = Study(args, store, budget, history)
                study.execute()
        except (StopRequested, LearnerStop, ResourceLimit) as exc:
            if study:
                study.state, study.reason = "paused", str(exc)
            print(json.dumps(dict(status="paused", reason=str(exc))), flush=True)
            return 2
        except BaseException as exc:
            if study:
                study.state, study.reason = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if study:
                try:
                    study.progress(force=True)
                except (StopRequested, LearnerStop, ResourceLimit):
                    atomic_json(args.root/"status.json", dict(status=study.state, stage=study.stage, reason=study.reason))
            budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
