"""Versioned cold-subspace execution wrapper; the Phase 11 v1 engine stays frozen."""

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
import math
from pathlib import Path
import statistics
import sys
import time

import torch

from mnist_experiment.fisher_memory.artifacts import _identity, _unit_record, _hash_json
from mnist_experiment.fisher_memory_grid_registration import run, learner, acceptance
from mnist_experiment.fisher_memory_grid_registration.compression import compress_fisher as _exact
from mnist_experiment.fisher_memory_grid_registration.data import plan_data
from mnist_experiment.fisher_memory_regression.runtime import read_json, file_digest
from mnist_experiment import plan15_grid_compression_probe as probe
from src.representations import representation_from_artifact

BaseConfig, BaseRunner, BaseStudy, base_sources = run.Config, run.Runner, run.Study, run.sources
base_process_alive = run.process_alive
POLICY = "plan15-grid-subspace-cold-v1"
ROOTS = ContextVar("grid_subspace_roots", default=(None, None))
POLICY_FIELDS = ("numerical_policy", "power_steps", "oversample", "ritz_tol", "cluster_tol", "archive_rank")


@dataclass(frozen=True)
class Config(BaseConfig):
    numerical_policy: str = POLICY
    power_steps: int = 4
    oversample: int = 16
    ritz_tol: float = 1e-4
    cluster_tol: float = 1e-6
    archive_rank: int = 64

    def validate(self):
        super().validate()
        if (tuple(getattr(self, k) for k in POLICY_FIELDS) != (POLICY, 4, 16, 1e-4, 1e-6, 64)
                or any(type(getattr(self, k)) is not int for k in ("power_steps", "oversample", "archive_rank"))):
            raise ValueError("changing the frozen subspace policy requires a new version")


def policy(cfg):
    return {k: getattr(cfg, k) for k in POLICY_FIELDS}


def sources():
    return dict(sorted({**base_sources(), **{str(p.relative_to(run.ROOT)): file_digest(p)
        for p in (Path(__file__).resolve(), Path(probe.__file__).resolve())}}.items()))


def process_alive(root, pid):
    if base_process_alive(root, pid):
        return True
    if type(pid) is not int or pid <= 0:
        return False
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if "mnist_experiment.plan15_grid_subspace" not in parts or "--root" not in parts:
            return False
        requested = Path(parts[parts.index("--root")+1])
        return (requested if requested.is_absolute() else Path(f"/proc/{pid}/cwd").resolve()/requested).resolve() == Path(root).resolve()
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


def compress_fisher(matrix, rank=64, seed=0, check_stop=None, backend="exact_cpu"):
    """Cold starts only; preserve source diagonal/trace, never absorb cancellation."""
    if backend not in ("subspace_cpu", "subspace_cuda"):
        return _exact(matrix, rank, seed, check_stop, backend)
    started = time.perf_counter()
    attempts, reason, chosen = [], None, None
    for steps in (4, 8, 16):
        learner._stop(check_stop)
        try:
            artifact, diagnostics = probe.approximate_fisher(matrix.to(backend.split("_")[1]),
                rank=rank, seed=seed, steps=steps, oversample=16, check_stop=check_stop)
            c = diagnostics["compression"]
            tolerance = c["diagonal_roundoff_tolerance"]
            diagonal = representation_from_artifact(artifact).diagonal_vector()
            if (not math.isfinite(c["retained_directions_orthogonality_error_fro"])
                    or c["retained_directions_orthogonality_error_fro"] > 1e-8
                    or c["residual_diagonal_minimum_before_clipping"] < -tolerance
                    or not torch.isfinite(diagonal).all()
                    or not math.isfinite(tolerance) or tolerance < 0
                    or abs(float(diagonal.sum())-float(matrix.trace())) > len(matrix)*tolerance
                    or not torch.allclose(diagonal.cpu(), matrix.diagonal().double().cpu(), rtol=0, atol=tolerance)):
                raise ValueError("invalid orthogonality, residual diagonal, source diagonal or trace")
            omitted, gap = c.get("largest_omitted_core_eigenvalue"), c.get("cutoff_relative_gap")
            if omitted is not None and omitted > 0 and (gap is None or not math.isfinite(gap) or gap <= 1e-6):
                reason = "fixed-rank cutoff splits a positive eigenvalue cluster (gap <= 1e-6)"
                attempts.append(dict(steps=steps, failure=reason))
                break
            ritz = c["ritz_residual_relative_fro"]
            attempts.append(dict(steps=steps, ritz=ritz if math.isfinite(ritz) else None))
            if math.isfinite(ritz) and ritz <= 1e-4:
                chosen = steps
                break
            reason = "Ritz residual exceeds 1e-4"
        except (learner.StopRequested, run.StopRequested, run.ResourceLimit):
            raise
        except (ValueError, RuntimeError) as error:
            reason = f"{type(error).__name__}: {error}"
            attempts.append(dict(steps=steps, failure=reason))
    if chosen is None:
        learner._stop(check_stop)
        artifact, diagnostics = _exact(matrix, rank, seed, check_stop, "exact_" + backend.split("_")[1])
    diagnostics["compression"].update(numerical_policy=POLICY, requested_backend=backend,
        chosen_steps=chosen, policy_attempts=attempts, fallback_reason=reason if chosen is None else None,
        dispatch_elapsed_seconds=time.perf_counter()-started)
    return artifact, diagnostics


def sealed(root, name, check=lambda: None):
    """Verify a selected unit fully, without opening a writable source Store."""
    root = Path(root)
    if (root / "units" / name).is_symlink():
        raise ValueError("source unit must not be a symlink")
    record = _unit_record(root, name, _identity(root)["identity_hash"], {}, check)
    if record["completion"] is None:
        raise ValueError(f"incomplete source evidence: {name}")
    return record


def checkpoint(record, identity_hash):
    pointer = read_json(record["path"] / "latest.json")
    descriptor = pointer["current"]
    if pointer["schema"] != 1 or descriptor != dict(file=descriptor["file"], **record["completion"]["files"][descriptor["file"]]):
        raise ValueError("sealed checkpoint descriptor mismatch")
    envelope = torch.load(record["path"] / descriptor["file"], map_location="cpu", weights_only=False)
    if (envelope["identity_hash"] != identity_hash or envelope["unit_id"] != record["unit_id"]
            or envelope["schema"] != 1 or envelope["manifest_hash"] != _hash_json(record["manifest"])):
        raise ValueError("source checkpoint identity mismatch")
    return envelope["payload"]


def parent_config(cfg, root):
    original = BaseConfig.read(root / "resolved_config.json")
    identity, provenance = _identity(root), read_json(root / "provenance.json")
    if (any(getattr(cfg, f.name) != getattr(original, f.name) for f in fields(BaseConfig) if f.name != "archive_rank")
            or original.scientific_hash != identity["identity"]["config_hash"]
            or provenance["sources"] != base_sources()
            or run.content_hash(provenance["sources"]) != identity["identity"]["source_hash"]
            or provenance["source_hash"] != identity["identity"]["source_hash"]):
        raise ValueError("parent configuration or frozen sources differ")
    return original, identity


class Runner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_backend = "subspace_" + self.cfg.device

    def assets(self, family, replica, partition):
        root = ROOTS.get()[0]
        if root is not None and not self.cfg.smoke:
            original, identity = parent_config(self.cfg, root)
            stage, task = family.split("--", 1)
            stem = f"assets-{stage}-{task}-{replica:04d}-c{self.channels[0]}x{self.channels[1]}-e{self.init_epochs:04d}"
            unit, parent = stem + f"-r{self.archive_rank:02d}", stem + f"-r{original.archive_rank:02d}"
            if not self.store.is_complete(unit) and (root / "units" / parent / "COMPLETED").exists():
                source = sealed(root, parent, self.check)
                plan = plan_data(self.source, self.cfg, stage, replica, partition)
                metadata = dict(kind="grid_registration_assets", schema_version=1, family=stage, replica=replica,
                    partition=partition, plan_hash=run.content_hash(plan), init_epochs=self.init_epochs,
                    channels=list(self.channels), output_dim=54, archive_rank=self.archive_rank,
                    compression_backend=self.compression_backend)
                if (any(source["metadata"].get(k) != v for k, v in metadata.items()
                        if k not in ("archive_rank", "compression_backend"))
                        or read_json(source["path"] / "plan.json") != plan):
                    raise ValueError("pretrained source plan/asset metadata differ")
                lineage = dict(root=str(root), identity_hash=identity["identity_hash"], unit_id=parent,
                               seal_hash=source["completion"]["seal_hash"])
                self.store.begin(unit, metadata)
                state = self.store.load_checkpoint(unit)
                if state is None:
                    state = checkpoint(source, identity["identity_hash"])
                    if state["stage"] != "ready" or state["plan_hash"] != metadata["plan_hash"] or state["epoch"] != self.init_epochs:
                        raise ValueError("source is not the requested pretrained asset")
                    state["learner"] = learner.initial_state(state["theta"], state["learner"]["ltm_indices"],
                        state["reference_fisher"], capacity=self.cfg.reference_capacity, rank=self.archive_rank,
                        seed=self.cfg.seed_for(family, replica, "initial_compression"),
                        backend=self.compression_backend, check_stop=self.check)
                    state.update(source_parent=lineage, numerical_policy=policy(self.cfg))
                    self.store.write_json(unit, "plan.json", plan)
                    self.store.save_checkpoint(unit, state)
                if state.get("source_parent") != lineage or state["learner"]["compression_backend"] != self.compression_backend:
                    raise ValueError("incomplete reused asset lineage/backend changed")
                self.store.write_json(unit, "health.json", read_json(source["path"] / "health.json"))
                self.store.complete(unit)
        return super().assets(family, replica, partition)


class Study(BaseStudy):
    def __init__(self, cfg, args, store, budget):
        reuse, evidence = ROOTS.get()
        self.evidence, self.parent_rank = None, None
        lineage = dict(policy=policy(cfg))
        if not cfg.smoke:
            if reuse is None or evidence is None or reuse == store.root.resolve() or evidence == store.root.resolve():
                raise ValueError("production needs distinct reuse/evidence roots")
            if reuse.parent != store.root.resolve().parent or evidence.parent != reuse.parent:
                raise ValueError("source and assessment must share the cumulative peer budget")
            _, parent = parent_config(cfg, reuse)
            assessed = _identity(evidence)
            if (assessed["identity"]["source_identity"] != parent["identity_hash"]
                    or assessed["identity"].get("role") != "compression_assessment"
                    or assessed["identity"].get("config_hash") != parent["identity"]["config_hash"]
                    or assessed["identity"]["code_hashes"].get(Path(probe.__file__).name) != file_digest(Path(probe.__file__))
                    or store.identity["data_hash"] != parent["identity"]["data_hash"]):
                raise ValueError("assessment probe/source/data identity mismatch")
            records = {n: sealed(reuse, "decision-" + n, budget.check) for n in ("initialization", "readiness", "archive-rank")}
            self.parent_rank = read_json(records["archive-rank"]["path"] / "result.json")
            summary = sealed(evidence, "summary", budget.check)
            self.evidence = read_json(summary["path"] / "result.json")
            cases = self.evidence["cases"].get("cold-s4", [])
            if (self.evidence["selected_steps"].get("cold") != 4 or not self.evidence["all_cases_passed"]
                    or {r["case"] for r in cases} != {"initial", "n25", "n75"} or len(cases) != 3
                    or any(not r["passed"] or r["rank"] != 64 or r["steps"] != 4 or r["mode"] != "cold"
                           or not r["matrix_audit"]["passed"] or not 0 <= r["matrix_audit"]["matrix_relative_error"] <= .01
                           or not 0 <= r["matrix_audit"].get("source_relative_error", math.inf) <= .1
                           or not 0 <= r["matrix_audit"].get("reference_source_relative_error", math.inf) <= .1
                           or not 0 <= r["candidate_profile"]["diagnostics"]["compression"]["ritz_residual_relative_fro"] <= cfg.ritz_tol
                           or (r["case"] != "initial" and not r["fit_audit"]["passed"]) for r in cases)):
                raise ValueError("sealed complete passing cold-four assessment required")
            evidence_record = sealed(evidence, "source-evidence", budget.check)
            if read_json(evidence_record["path"] / "result.json") != self.evidence["source_evidence"]:
                raise ValueError("assessment source evidence differs from summary")
            if self.evidence["source_evidence"]["source_identity"] != parent:
                raise ValueError("assessment embeds a different parent identity")
            seals = {n: r["completion"]["seal_hash"] for n, r in records.items()}
            if any(self.evidence["source_evidence"]["source_seals"].get("decision-" + n) != seal for n, seal in seals.items()):
                raise ValueError("parent decision seals differ from assessed sources")
            lineage.update(parent_root=str(reuse), parent_identity=parent, decision_seals=seals,
                evidence_root=str(evidence), evidence_identity=assessed,
                evidence_seals={"summary": summary["completion"]["seal_hash"],
                                "source-evidence": evidence_record["completion"]["seal_hash"]})
        super().__init__(cfg, args, store, budget)
        self.seal("lineage", lineage)

    def resource_assessment(self, channels=None):
        if channels is not None and tuple(channels) != self.cfg.channels:
            raise ValueError("a different architecture requires new actual-snapshot evidence")
        backend = "subspace_" + self.cfg.device
        if not self.decision("resource-assessment"):
            if self.cfg.smoke:
                matrix = torch.diag(torch.arange(1, 13, dtype=torch.float64))
                started = time.perf_counter()
                packed, diagnostics = compress_fisher(matrix, rank=64, backend=backend, check_stop=self.budget.check)
                error = float((representation_from_artifact(packed).to_dense()-matrix).norm()/matrix.norm())
                if error > 1e-8:
                    raise ValueError("small synthetic compression proof failed")
                cases = [dict(case="small_synthetic", candidate_profile=dict(timings=[
                    dict(warmup=False, seconds=time.perf_counter()-started)]))]
                proof = dict(matrix_relative_error=error, diagnostics=diagnostics)
            else:
                cases, proof = self.evidence["cases"]["cold-s4"], None
            attempts = [dict(case=r["case"], backend=backend, validated=True, rows=[
                dict(warmup=t["warmup"], transaction_seconds=t["seconds"]) for t in r["candidate_profile"]["timings"]]) for r in cases]
            self.seal("resource-assessment", dict(selected_backend=backend, status="complete", **policy(self.cfg),
                backends={backend: dict(validated=True, median_transaction_seconds=statistics.median(
                    t["transaction_seconds"] for r in attempts for t in r["rows"] if not t["warmup"]))},
                attempts=attempts, synthetic_proof=proof, lineage_unit="decision-lineage",
                numerical_comparison=None if self.cfg.smoke else self.numerical_comparison(cases),
                whole_update_profile=dict(status="pending_full_trajectory_profile")))
        self.runner.compression_backend = backend

    def numerical_comparison(self, cases):
        timings = [dict(case=r["case"], **{name + "_median_seconds": statistics.median(
            t["seconds"] for t in r[key]["timings"] if not t["warmup"])
            for name, key in (("candidate", "candidate_profile"), ("exact", "exact_profile"))}) for r in cases]
        return dict(candidate_median_seconds=statistics.median(t["candidate_median_seconds"] for t in timings),
            exact_median_seconds=statistics.median(t["exact_median_seconds"] for t in timings), cold_case_timings=timings,
            max_matrix_relative_error=max(r["matrix_audit"]["matrix_relative_error"] for r in cases),
            max_prediction_relative_error=max(r["fit_audit"]["prediction_difference_relative_to_reference_change"]
                for r in cases if r["case"] != "initial"),
            max_quadratic_relative_error=max(r["matrix_audit"]["quadratic_relative_error"] for r in cases),
            max_damped_solve_relative_error=max(v for r in cases for v in r["matrix_audit"]["damped_solve_relative_errors"].values()),
            max_source_relative_error=max(r["matrix_audit"]["source_relative_error"] for r in cases),
            proof_root=str(ROOTS.get()[1]), proof_seal=self.decision("lineage")["evidence_seals"]["summary"],
            reference="exact_same_rank", rank=64)

    def rank_calibration(self):
        if not self.decision("archive-rank"):
            if self.cfg.smoke:
                self.resource_assessment()
                result = dict(rank=64, adequate=True, rows=[], synthetic_proof=self.decision("resource-assessment")["synthetic_proof"])
            else:
                result = dict(self.parent_rank)
                selected = [r for r in result["rows"] if r["rank"] == 64]
                if (result["rank"] != 64 or not result["adequate"]
                        or {r["snapshot"] for r in selected} != {"initial", "n25", "n75"}
                        or max(r["relative_frobenius"] for r in selected) >= .01):
                    raise ValueError("sealed adequate rank64 audit with error below .01 required")
                result.update(source_evidence=self.decision("lineage"), numerical_discrepancy_rows=[
                    dict(snapshot=r["case"], relative_frobenius_to_exact_same_rank=r["matrix_audit"]["matrix_relative_error"],
                         candidate_relative_frobenius_to_source=r["matrix_audit"]["source_relative_error"],
                         exact_same_rank_relative_frobenius_to_source=r["matrix_audit"]["reference_source_relative_error"])
                    for r in self.evidence["cases"]["cold-s4"]])
            self.seal("archive-rank", result)
        self.runner.archive_rank = 64


@contextmanager
def bindings(reuse_root=None, evidence_root=None):
    """Explicit process-local substitution; nested contexts and failures restore v1."""
    token = ROOTS.set(tuple(Path(p).resolve() if p is not None else None for p in (reuse_root, evidence_root)))
    replacements = [(run, "Config", Config), (run, "Runner", Runner), (run, "Study", Study),
                    (run, "sources", sources), (learner, "compress_fisher", compress_fisher),
                    (acceptance, "sources", sources), (run, "process_alive", process_alive)]
    previous = [(module, name, getattr(module, name)) for module, name, value in replacements]
    try:
        for module, name, value in replacements:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(previous):
            setattr(module, name, value)
        ROOTS.reset(token)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["accept"]:
        with bindings():
            return acceptance.main(argv[1:])
    if argv[:1] == ["run"]:
        argv = argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    options, rest = parser.parse_known_args(argv)
    with bindings(options.reuse_root, options.evidence_root):
        args = run.arguments(rest)
        if args.stage not in ("status", "stop", "report"):
            cfg = Config.read(args.config) if args.config else Config.smoke_config(args.device or "cpu") if args.smoke else Config()
            if not cfg.smoke and (not options.reuse_root or not options.evidence_root or not args.acceptance_root):
                raise ValueError("production requires --reuse-root, --evidence-root and --acceptance-root")
            if cfg.smoke and (options.reuse_root or options.evidence_root):
                raise ValueError("smokes must not reuse production assets/evidence")
            roots = [p.resolve() for p in (options.reuse_root, options.evidence_root) if p is not None]
            if args.root.resolve() in roots or any(p.parent != args.budget_root.resolve() for p in roots):
                raise ValueError("use distinct peer roots under the shared budget root")
        return run.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
