"""Autonomous Plan 14 execution with immutable phase decisions and rolling reports."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from mnist_experiment.rotated_mnist.artifacts import _read_json, _write_json, _utc_now
from src.mnist_model import configure_torch_runtime

from .artifacts import UnitStore, RunLimit, StopExecution, freeze_json, run_lock, file_hash
from .assets import ensure_assets
from .calibration import candidate_grid, candidate_id, balanced_score, select_candidate, out_of_fold_selection, preassigned_steps
from .config import Study, REPO_ROOT, canonical_hash, policies
from .trajectory import ensure_trajectory, REQUIRED as TRAJECTORY_REQUIRED


def read_summary(path: Path) -> dict:
    return _read_json(Path(path) / "summary.json")


def clean_fixed_policy(policy: dict) -> dict:
    policy = dict(policy)
    if policy["gate"] is None:
        policy["c"], policy["cadence"] = 1., 1
    return policy


class Runner:
    def __init__(self, store: UnitStore, notebook: Path):
        self.store, self.study = store, store.study
        self.notebook = notebook
        self.profile = None
        self.repair_profiles = {}
        self.selection = None
        self.last_refresh = 0.
        path = store.root / "progress.json"
        self.progress = _read_json(path) if path.exists() else {"schema_version": 1, "phases": {}}

    def refresh(self, *, force: bool = False):
        _write_json(self.store.root / "progress.json", self.progress)
        if force or time.monotonic() - self.last_refresh >= 900:
            from .refresh_notebook import refresh
            refresh(self.store.root, self.notebook)
            self.last_refresh = time.monotonic()

    def phase(self, number: int, status: str):
        name = f"phase{number}"
        value = self.progress["phases"].setdefault(name, {})
        value.update(status=status, updated_at=_utc_now())
        if status == "running":
            value.setdefault("started_at", _utc_now())
            for key in ("error", "error_type", "interruption"):
                self.progress.pop(key, None)
        if status == "completed":
            value["completed_at"] = _utc_now()
        self.progress.update(phase=name, status=status)
        self.refresh(force=True)
        print(f"[{_utc_now()}] {name}: {status}", flush=True)

    def checkpoint_progress(self):
        completed = list(self.store.root.glob("phase*/**/COMPLETED"))
        self.progress["completed_units"] = sum(".incomplete" not in p.parts for p in completed)
        self.progress["active_invocation_seconds"] = time.monotonic() - self.store.limit.started
        self.refresh()

    def decision(self, phase: str, name: str, payload: dict, parents: list[Path] | None = None) -> dict:
        path = self.store.root / "decisions" / f"{name}.json"
        resolved = {"schema_version": 1, "study_hash": self.study.config_hash, **payload}
        parent_ids = {str(p.relative_to(self.store.root)): file_hash(p / "config.json")
                      for p in parents or []}
        unit = self.store.unit(phase, "decision", 0, policy=name,
                               detail={"decision": resolved, "parents": parent_ids})
        session = self.store.begin(unit, ("summary.json", "decision.json"), resume=True)
        if session is not None:
            session.write_json("summary.json", {"phase": phase, "decision": name, **resolved})
            session.write_json("decision.json", resolved)
            self.store.finish(session, ("summary.json", "decision.json"))
        freeze_json(path, resolved)
        self.store.log(f"Frozen {name}", phase=phase, selection=resolved)
        self.refresh(force=True)
        return resolved

    def load_decision(self, phase: str, name: str):
        pointer = self.store.root / "decisions" / f"{name}.json"
        if not pointer.exists():
            return None
        value = _read_json(pointer)
        if value.get("study_hash") != self.study.config_hash:
            raise RuntimeError("incompatible frozen decision study")
        matches = []
        for path in (self.store.root / phase).glob("decision__*/config.json"):
            unit = _read_json(path)
            if unit.get("policy") == name and unit.get("detail", {}).get("decision") == value:
                matches.append((path.parent, unit))
        if len(matches) != 1:
            raise RuntimeError("frozen decision must match exactly one sealed unit")
        path, unit = matches[0]
        if unit.get("source_hashes") != self.store.sources:
            raise RuntimeError("frozen decision source hashes changed")
        if self.store.completed(unit, ("summary.json", "decision.json")) != path:
            raise RuntimeError("frozen decision is not complete")
        if _read_json(path / "decision.json") != value:
            raise RuntimeError("decision pointer differs from immutable decision")
        for relative, digest in unit["detail"]["parents"].items():
            parent = (self.store.root / relative).resolve()
            if not parent.is_relative_to(self.store.root.resolve()):
                raise RuntimeError("invalid decision parent path")
            if file_hash(parent / "config.json") != digest:
                raise RuntimeError("frozen decision parent identity changed")
            config = _read_json(parent / "config.json")
            required = tuple(_read_json(parent / "integrity.json"))
            if self.store.completed(config, required) != parent:
                raise RuntimeError("frozen decision parent is not complete")
        return value

    def assets(self, phase, index, family):
        return ensure_assets(self.store, phase, index, family, self.selection["width"],
            resume=True, extended_epochs=self.selection.get("extended_epochs", False))

    def trajectory(self, phase, index, env, policy, *, tier="principal", episode_arrivals=None,
                   snapshots=False, retention_override=None):
        asset = self.assets(phase, index, env["family"])
        profile = self.profile
        if (policy.get("gate") is not None and
                (policy.get("score_samples", self.study.score_samples) != self.study.score_samples
                 or policy.get("gate_rank", policy["rank"]) != policy["rank"])):
            identity = canonical_hash(policy)
            if identity not in self.repair_profiles:
                from .phase0 import ensure_repair_profile
                path = ensure_repair_profile(self.store, self.profile, policy,
                                             self.selection["width"], resume=True)
                self.repair_profiles[identity] = _read_json(path / "profile.json")
            profile = self.repair_profiles[identity]
        total = self.study.rotation_transitions_per_arrow * 3 if env["family"] == "rotation" else self.study.mixture_steps
        snapshot_steps = preassigned_steps(total, self.study.calibration_shadow_positions) if snapshots else ()
        path = ensure_trajectory(self.store, phase, index, env, clean_fixed_policy(policy), asset,
            width=self.selection["width"], retention=retention_override or self.selection["retention"][env["family"]],
            profile=profile, budget_tier=tier, snapshot_steps=snapshot_steps,
            episode_arrivals=episode_arrivals, resume=True)
        self.checkpoint_progress()
        return path, asset

    def candidate_runs(self, candidate, indices, *, snapshots=False):
        condition_list = policies(**candidate, rank=self.study.rank, score_samples=self.study.score_samples)
        chosen = [p for p in condition_list if p["name"].startswith("asg_") and p["name"] != "asg_uncalibrated"]
        controls = [p for p in condition_list if p["depth"] == 6]
        result, parents, drivers = [], [], []
        for index in indices:
            for env in self.study.environments():
                for policy in chosen + controls:
                    path, asset = self.trajectory("phase3", index, env, policy, snapshots=snapshots and policy["gate"] is not None)
                    parents.append(path)
                    if policy["gate"] is not None:
                        result.append(read_summary(path))
                        drivers.append({"index": index, "environment": env, "policy": policy,
                                        "path": str(path), "asset_path": str(asset)})
                self.refresh(force=True)
        return result, parents, drivers

    def run_phase3(self):
        previous = self.load_decision("phase3", "phase3_calibration")
        if previous is not None:
            return previous
        nprobe = min(self.study.calibration_probe_replicas, self.study.phase_replicas["phase3"])
        candidates = candidate_grid()
        freeze_json(self.store.root / "phase3_candidates.json", candidates)
        scores, rows_by_candidate, parents, all_drivers = {}, {}, [], {}
        for candidate in candidates:
            key = candidate_id(candidate)
            rows, paths, drivers = self.candidate_runs(candidate, range(nprobe), snapshots=True)
            scores[key], rows_by_candidate[key] = balanced_score(rows), rows
            parents.extend(paths)
            all_drivers[key] = drivers
        base = candidate_id(candidates[0])
        ranked = sorted(scores, key=lambda key: (not scores[key]["admissible"],
            scores[key]["nll_auc"] + (0 if scores[key]["admissible"] else scores[key]["retention_nll"])))
        shortlist = [base] + [key for key in ranked if key != base][:3]
        shortlist_candidates = {candidate_id(c): c for c in candidates if candidate_id(c) in shortlist}
        freeze_json(self.store.root / "phase3_shortlist.json", shortlist_candidates)
        for key, candidate in shortlist_candidates.items():
            rows, paths, drivers = self.candidate_runs(candidate,
                range(nprobe, self.study.phase_replicas["phase3"]), snapshots=True)
            rows_by_candidate[key].extend(rows)
            all_drivers[key].extend(drivers)
            parents.extend(paths)
        # Complete current-batch-only protocol, not merely an altered gate ESS.
        for index in range(nprobe):
            for env in self.study.environments():
                for policy in policies(rank=self.study.rank, score_samples=self.study.score_samples):
                    if policy["ewc"] == "lr" and (policy["depth"] in (1, 4, 6) or policy["name"] == "asg_lr_lr"):
                        path, _ = self.trajectory("phase3", index, env, policy,
                                                  episode_arrivals=1, snapshots=True)
                        parents.append(path)
                self.refresh(force=True)
        final_scores = {key: balanced_score(rows_by_candidate[key]) for key in shortlist}
        selected, detail = select_candidate(final_scores)
        from .calibration_audit import ensure_calibration_audit, ensure_shadow_audit
        audit_paths = []
        # Every baseline ASG trajectory receives the six preassigned local shadows.
        for ordinal, driver in enumerate(all_drivers[base]):
            path, asset = Path(driver["path"]), Path(driver["asset_path"])
            snapshots = torch.load(path / "snapshots.pt", map_location="cpu", weights_only=False)
            gates = _read_json(path / "gates.json")
            for snap_index, snapshot in enumerate(snapshots):
                matching = [g for g in gates if g["step"] == snapshot["step"]]
                frontier = min(5, matching[-1]["depth"] if matching else 1)
                case = {"environment": driver["environment"], "step": snapshot["step"],
                        "frontier": frontier, "source_depth": None, "historical": False}
                audit_paths.append(ensure_shadow_audit(self.store,
                    ordinal * self.study.calibration_shadow_positions + snap_index,
                    case, asset, path, resume=True))
                self.checkpoint_progress()
        # Sixteen predetermined history-variance cases: four replicas by four conditions.
        variance_drivers = [d for d in all_drivers[base] if d["policy"]["name"] == "asg_lr_lr"]
        for ordinal, driver in enumerate(variance_drivers[:self.study.history_variance_states]):
            path, asset = Path(driver["path"]), Path(driver["asset_path"])
            snapshots = torch.load(path / "snapshots.pt", map_location="cpu", weights_only=False)
            snapshot = snapshots[len(snapshots)//2]
            case = {"environment": driver["environment"], "step": snapshot["step"],
                    "frontier": 1 + ordinal % 5, "source_depth": None, "historical": False}
            audit_paths.append(ensure_calibration_audit(self.store, ordinal, case, asset, path, resume=True))
            self.checkpoint_progress()
        payload = {"selected": shortlist_candidates[selected], "selected_id": selected,
                   "shortlist": shortlist_candidates, **detail,
                   "out_of_fold": out_of_fold_selection({key: rows_by_candidate[key] for key in shortlist},
                                                         screening_replicas=tuple(range(nprobe))),
                   "out_of_fold_scope": "only replicas unused in initial candidate screening are scoring folds",
                   "audit_paths": [str(p) for p in audit_paths]}
        return self.decision("phase3", "phase3_calibration", payload, parents + audit_paths)

    def run_phase4(self, calibration):
        previous = self.load_decision("phase4", "phase4_selection")
        if previous is not None:
            return previous
        chosen = policies(**calibration["selected"], rank=self.study.rank, score_samples=self.study.score_samples)
        rows, parents = [], []
        for index in range(self.study.phase_replicas["phase4"]):
            for env in self.study.environments():
                seen = set()
                for policy in chosen:
                    path, _ = self.trajectory("phase4", index, env, policy)
                    if path not in seen:
                        rows.append(read_summary(path)); parents.append(path); seen.add(path)
                self.refresh(force=True)
        for index in range(min(self.study.data_panel_replicas, self.study.phase_replicas["phase4"])):
            for env in self.study.environments(scarcity=True):
                if env["family"] != "mixture":
                    continue
                for tier in ("strict", "principal"):
                    for policy in chosen:
                        path, _ = self.trajectory("phase4", index, env, policy, tier=tier)
                        parents.append(path)
                    self.refresh(force=True)
        candidates = {p["name"]: p for p in chosen if p["name"].startswith("asg_")}
        adaptive_scores = {name: balanced_score([r for r in rows if r["policy"] == name])
                           for name in candidates if any(r["policy"] == name for r in rows)}
        best, _ = select_candidate(adaptive_scores)
        baseline = candidates[best]
        # Bound repairs to measured current-curvature noise and measured overhead.
        diagnostics = []
        for path in (self.store.root / "phase2").glob("*/audit_metrics.json"):
            if (path.parent / "COMPLETED").exists():
                diagnostics.extend(_read_json(path))
        noisy = any(r.get("decision", {}).get("gate_flip") is True
                    and not r.get("reference_unresolved", True)
                    and r.get("representation") == "dense_same_samples" for r in diagnostics)
        compression_limited = any(r.get("same_bank_decision", {}).get("gate_flip") is True
                                  and r.get("representation") == "lr8_diag" for r in diagnostics)
        best_rows = [r for r in rows if r["policy"] == best]
        overhead = statistics.mean(r["probe_seconds"] / max(r["online_seconds"], 1e-12) for r in best_rows)
        repairs = []
        if noisy:
            repairs.append({**baseline, "name": "repair_score128", "score_samples": 128 if not self.study.smoke else 8})
        if compression_limited and baseline["gate"] == "lr":
            repairs.append({**baseline, "name": "repair_gate_rank16", "gate_rank": 16})
        if overhead > .2 and baseline["cadence"] < 4:
            repairs.append({**baseline, "name": "repair_cadence", "cadence": min(4, max(2, baseline["cadence"] * 2))})
        if not repairs:
            self.store.log("No authorized repair trigger established; retaining the original candidate", phase="phase4")
        freeze_json(self.store.root / "phase4_repair_candidates.json", repairs[:2])
        repair_rows = {}
        repair_audits = {}
        repairs = repairs[:2]
        for repair_index, repair in enumerate(repairs):
            values = []
            for index in range(min(self.study.repair_replicas, self.study.phase_replicas["phase4"])):
                for env_index, env in enumerate(self.study.environments()):
                    needs_local_check = repair["name"] in {"repair_score128", "repair_gate_rank16"}
                    path, asset = self.trajectory("phase4", index, env, repair,
                                                  snapshots=needs_local_check and index == 0)
                    values.append(read_summary(path)); parents.append(path)
                    if needs_local_check and index == 0:
                        from .calibration_audit import ensure_shadow_audit
                        snapshots = torch.load(path / "snapshots.pt", map_location="cpu", weights_only=False)
                        snapshot = snapshots[len(snapshots) // 2]
                        matching = [g for g in _read_json(path / "gates.json") if g["step"] == snapshot["step"]]
                        case = {"environment": env, "step": snapshot["step"],
                                "frontier": min(5, matching[-1]["depth"] if matching else 1),
                                "source_depth": None, "historical": False}
                        audit = ensure_shadow_audit(self.store, repair_index * 4 + env_index,
                            case, asset, path, phase="phase4", resume=True)
                        parents.append(audit)
                        repair_audits.setdefault(repair["name"], []).append(str(audit))
                        self.checkpoint_progress()
                self.refresh(force=True)
            repair_rows[repair["name"]] = values
        restricted_baseline = [r for r in best_rows if r["replica_index"] < self.study.repair_replicas]
        repair_scores = {best: balanced_score(restricted_baseline), **{key: balanced_score(values) for key, values in repair_rows.items()}}
        promoted_name, promotion = select_candidate(repair_scores)
        promoted = next((p for p in repairs if p["name"] == promoted_name), None)
        comparisons = {}
        for env in self.study.environments():
            env_rows = [r for r in rows if r["environment"] == env["name"]]
            fixed_scores = {p["name"]: balanced_score([r for r in env_rows if r["policy"] == p["name"]])
                            for p in chosen if p["depth"] in range(1, 7)}
            fixed, _ = select_candidate(fixed_scores)
            asg_scores = {key: balanced_score([r for r in env_rows if r["policy"] == key]) for key in adaptive_scores}
            asg, _ = select_candidate(asg_scores)
            comparisons[env["name"]] = {"fixed": fixed, "asg": asg}
        return self.decision("phase4", "phase4_selection", {"comparators": comparisons,
            "principal_policies": chosen, "promoted_repair": promoted, "promotion": promotion,
            "repair_local_calibration": repair_audits,
            "repair_triggers": {"gate_disagreement_detected": noisy,
                                "compression_decision_error_detected": compression_limited,
                                "probe_time_fraction": overhead},
            "scope": "gate-only repairs; EWC objective, archive samples, and solver unchanged"}, parents)

    def run_phase5(self, development):
        conditions = development["principal_policies"] + ([development["promoted_repair"]] if development["promoted_repair"] else [])
        ledger = {"schema_version": 1, "study_hash": self.study.config_hash,
            "replica_indices": list(range(self.study.phase_replicas["phase5"])),
            "environments": self.study.environments(), "policies": conditions,
            "comparators": development["comparators"], "architecture": self.selection,
            "profile_hash": canonical_hash(self.profile), "quality_targets": self.study.quality_targets,
            "seeds": {str(i): {env["name"]: self.study.seed(f"phase5:{env['family']}:initialization", i)
                       for env in self.study.environments()} for i in range(self.study.phase_replicas["phase5"])},
            "nominal_trajectory_assignments": self.study.phase_replicas["phase5"] * 4 * len(conditions),
            "uncalibrated_alias_when_identical": True, "no_significance_stopping": True}
        freeze_json(self.store.root / "final_ledger.json", ledger)
        parents = []
        for index in ledger["replica_indices"]:
            for env in ledger["environments"]:
                for policy in conditions:
                    path, _ = self.trajectory("phase5", index, env, policy)
                    parents.append(path)
                self.refresh(force=True)
        return parents


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default_float64.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=REPO_ROOT / "mnist_experiment/akaike_score_gate_results.ipynb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--through-phase", type=int, choices=range(-1, 7), default=6)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    study = Study.from_path(args.config)
    if study.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Plan 14 requires CUDA; run outside the restricted sandbox, never silently substitute CPU")
    torch.set_num_threads(study.cpu_threads)
    configure_torch_runtime(deterministic_algorithms=True, warn_only=study.device == "cuda")
    root = args.output_root.resolve()
    limit = RunLimit(args.max_wall_seconds, args.max_units)
    with run_lock(root), limit.signal_handlers():
        freeze_json(root / "study.json", {**study.mapping(), "study_hash": study.config_hash}, resume=args.resume)
        store = UnitStore(root, study, limit=limit)
        runner = Runner(store, args.notebook.resolve())
        runner.refresh(force=True)
        try:
            if args.through_phase < 0:
                return
            runner.phase(0, "running")
            from .phase0 import ensure_phase0
            path = ensure_phase0(store, resume=True)
            runner.profile = _read_json(path / "profile.json")
            from .validation import ensure_pipeline_validation
            ensure_pipeline_validation(store, runner.profile, resume=True)
            runner.phase(0, "completed")
            if args.through_phase == 0:
                return
            runner.phase(1, "running")
            from .development import run_phase1, run_phase2
            runner.selection = run_phase1(store, resume=True, refresh=lambda: runner.refresh(force=True))
            runner.phase(1, "completed")
            if args.through_phase == 1:
                return
            runner.phase(2, "running")
            run_phase2(store, runner.selection, resume=True, refresh=lambda: runner.refresh(force=True))
            runner.phase(2, "completed")
            if args.through_phase == 2:
                return
            runner.phase(3, "running")
            calibration = runner.run_phase3()
            runner.phase(3, "completed")
            if args.through_phase == 3:
                return
            runner.phase(4, "running")
            development = runner.run_phase4(calibration)
            runner.phase(4, "completed")
            if args.through_phase == 4:
                return
            runner.phase(5, "running")
            paths = runner.run_phase5(development)
            runner.phase(5, "completed")
            if args.through_phase == 5:
                return
            runner.phase(6, "running")
            from .closeout import ensure_closeout
            ensure_closeout(store, paths, development, resume=True)
            runner.phase(6, "completed")
            runner.progress["status"] = "completed"
        except StopExecution as error:
            runner.progress["status"] = "interrupted"
            runner.progress["interruption"] = str(error)
            store.log(str(error), phase=runner.progress.get("phase"))
            print(str(error), flush=True)
        except BaseException as error:
            runner.progress.update(status="blocked", error_type=type(error).__name__, error=str(error))
            store.log("Execution needs repair before the dependent work can continue", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            runner.checkpoint_progress()
            runner.refresh(force=True)


if __name__ == "__main__":
    main()
