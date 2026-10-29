"""Bounded, assessment-only audits against sealed Phase 11 model snapshots.

No source Store is opened for writing, no matrices are saved, and no production
backend is selected. Run separately from the paused production process.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from mnist_experiment.fisher_memory.artifacts import Store, iter_units, _hash_json
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.fisher_memory_grid_registration import ALLOCATION_FAMILY
from mnist_experiment.fisher_memory_grid_registration.compression import compress_fisher
from mnist_experiment.fisher_memory_grid_registration.config import Config, Recipe, ROOT, content_hash
from mnist_experiment.fisher_memory_grid_registration.curvature import reference_fisher
from mnist_experiment.fisher_memory_grid_registration.data import (
    MNISTSource, affine_from_parameters, blur_pairs, render_pairs,
)
from mnist_experiment.fisher_memory_grid_registration.learner import fit_step, prepare_objective
from mnist_experiment.fisher_memory_grid_registration.model import build_model, flat_prediction, memory_indices
from mnist_experiment.fisher_memory_grid_registration.run import budget_lock, peer_seconds, sources
from mnist_experiment.fisher_memory_grid_registration.runner import settings
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, dataset_manifest, file_bytes, file_digest, read_json, setup_torch, synchronize,
)
from mnist_experiment.plan15_grid_compression_probe import approximate_fisher
from src.representations import representation_from_artifact


PILOT = "trajectory-pilot--grid-0000-replay_ewc-b814039271843e24-n0100"


def code_provenance(identity, previous=None):
    snapshot = {name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                for name in identity["code_hashes"]}
    if {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in snapshot.items()} != identity["code_hashes"]:
        raise ValueError("assessment source snapshot does not match identity code hashes")
    if previous is None:
        def git(*arguments):
            return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True, timeout=5).strip()
        git_metadata = dict(git_commit=git("rev-parse", "HEAD"),
                            git_dirty=bool(git("status", "--porcelain")))
    else:
        # Resume retains the original run provenance, even after unrelated commits.
        git_metadata = {key: previous[key] for key in ("git_commit", "git_dirty")}
    return dict(source_snapshot=snapshot, **git_metadata)


def checkpoint(unit, identity_hash):
    """Read the latest envelope only after iter_units verified every sealed byte."""
    pointer = read_json(unit["path"] / "latest.json")
    descriptor = pointer["current"]
    if pointer["schema"] != 1 or descriptor != dict(
            file=descriptor["file"], **unit["completion"]["files"][descriptor["file"]]):
        raise ValueError("sealed checkpoint descriptor mismatch")
    envelope = torch.load(unit["path"] / descriptor["file"], map_location="cpu", weights_only=False)
    if (envelope["schema"] != 1 or envelope["unit_id"] != unit["unit_id"]
            or envelope["identity_hash"] != identity_hash
            or envelope["manifest_hash"] != _hash_json(unit["manifest"])):
        raise ValueError("checkpoint identity mismatch")
    return envelope["payload"]


def measure(matrix, function, device, check):
    rows, diagnostics, warm = [], None, None
    for repeat in range(3):
        check()
        synchronize(device)
        started = time.perf_counter()
        artifact, diagnostics = function(matrix)
        synchronize(device)
        rows.append(dict(warmup=repeat == 0, seconds=time.perf_counter() - started))
        if repeat == 0:
            warm = artifact
    return warm, dict(timings=rows, diagnostics=diagnostics)


def matrix_audit(matrix, exact, candidate, probes):
    reference, actual = map(representation_from_artifact, (exact, candidate))
    dense = reference.to_dense()
    candidate_dense = actual.to_dense()
    matrix_error = float((candidate_dense - dense).norm() / dense.norm().clamp_min(1e-30))
    source_norm = matrix.norm().clamp_min(1e-30)
    source_error = float((candidate_dense - matrix).norm() / source_norm)
    reference_source_error = float((dense - matrix).norm() / source_norm)
    q = (probes * reference.matvec(probes)).sum(0)
    quadratic_error = float((((probes * actual.matvec(probes)).sum(0) - q).abs()
                             / q.abs().clamp_min(1e-30)).max())
    solve_errors = {}
    for c in (.1, 1., 10.):
        damping = max(c * float(matrix.trace()) / len(matrix), 1e-12)
        expected = reference.damped_solve(probes, damping)
        solve_errors[str(c)] = float(((actual.damped_solve(probes, damping) - expected).norm(dim=0)
                                      / expected.norm(dim=0).clamp_min(1e-30)).max())
    diagonal_error = float((actual.diagonal_vector() - matrix.diagonal()).abs().max())
    trace_error = abs(float(actual.diagonal_vector().sum() - matrix.trace()))
    tolerance = 100 * len(matrix) * torch.finfo(torch.float64).eps * float(matrix.abs().sum(1).max())
    passed = (matrix_error <= .01 and quadratic_error <= .05 and max(solve_errors.values()) <= .05
              and diagonal_error <= tolerance and trace_error <= len(matrix) * tolerance)
    return dict(passed=passed, matrix_relative_error=matrix_error, quadratic_relative_error=quadratic_error,
                source_relative_error=source_error, reference_source_relative_error=reference_source_error,
                damped_solve_relative_errors=solve_errors, diagonal_max_absolute_error=diagonal_error,
                trace_absolute_error=trace_error, diagonal_roundoff_tolerance=tolerance)


def fit_audit(snapshot, exact, candidate, cfg, model, layout, check):
    fit_cfg = replace(settings(cfg, Recipe.from_mapping(snapshot["recipe"])), max_iter=100, max_eval=150)
    values = [snapshot[k].to(device=cfg.device, dtype=torch.float32)
              for k in ("theta", "new_x", "new_y", "replay_x", "replay_y", "ages")]
    theta, new_x, new_y, replay_x, replay_y, ages = values
    arguments = (model, layout, theta, new_x, new_y, replay_x, replay_y, ages)
    tail = (snapshot["anchor"], "replay_ewc", fit_cfg, snapshot["seed"], check)
    started = time.perf_counter()
    accepted, reference_diagnostics = fit_step(*arguments, exact, *tail)
    proposed, candidate_diagnostics = fit_step(*arguments, candidate, *tail)
    objective, _ = prepare_objective(*arguments, exact, *tail)
    with torch.no_grad():
        original_value = float(objective(accepted)[0])
        gap = abs(float(objective(proposed)[0]) - original_value) / max(
            abs(original_value - model.output_dim * math.log(2 * math.sqrt(2 * math.pi))), 1e-12)
        x = torch.cat((new_x, replay_x))
        baseline = flat_prediction(model, layout, theta, x).double()
        expected = flat_prediction(model, layout, accepted, x).double()
        proposed_predictions = flat_prediction(model, layout, proposed, x).double()
        change = float((expected - baseline).norm())
        prediction_error = float((proposed_predictions - expected).norm()) / max(change, 1e-12)
    synchronize(cfg.device)
    passed = (not reference_diagnostics["solver_failed"] and not candidate_diagnostics["solver_failed"]
              and gap <= .001 and prediction_error <= .05)
    return dict(passed=passed, reference_objective_relative_gap=gap,
        prediction_difference_relative_to_reference_change=prediction_error,
        reference_prediction_change_norm=change, reference_fit=reference_diagnostics,
        candidate_fit=candidate_diagnostics, elapsed_seconds=time.perf_counter() - started)


def execute(args, cfg, provenance, store, budget):
    check, source = budget.check, args.source_root
    units = {u["unit_id"]: u for u in iter_units(source, completed_only=True, full=True, check=check)}
    if store.is_complete("summary"):
        return read_json(store.unit_path("summary") / "result.json")
    initialization = read_json(units["decision-initialization"]["path"] / "result.json")
    rank_decision = read_json(units["decision-archive-rank"]["path"] / "result.json")
    if rank_decision["rank"] != 64 or not rank_decision["adequate"]:
        raise ValueError("assessment requires the sealed adequate rank-64 decision")
    blur = read_json(units["decision-readiness"]["path"] / "result.json")["blur_max"]
    channels = tuple(initialization["channels"])
    asset_id = f"assets-development-grid-0000-c{channels[0]}x{channels[1]}-e0045-r08"
    identity_hash = read_json(source / "identity.json")["identity_hash"]
    asset, pilot = (checkpoint(units[key], identity_hash) for key in (asset_id, PILOT))
    if pilot["n"] != 100 or initialization["epochs"] != 45:
        raise ValueError("expected complete 100-grid pilot and selected 45-epoch asset")
    snapshots = {f"n{s['n']}": s for s in pilot["snapshots"] if s["n"] in (25, 75)}
    if set(snapshots) != {"n25", "n75"}:
        raise ValueError("sealed pilot is missing required snapshots")
    pilot_asset_id = read_json(units[PILOT]["path"] / "summary.json")["asset_id"]
    selected_units = (asset_id, PILOT, pilot_asset_id, "decision-initialization", "decision-readiness", "decision-archive-rank")
    previous = read_json(store.unit_path("source-evidence") / "result.json") \
        if store.is_complete("source-evidence") else None
    evidence = dict(source_root=str(source), source_identity=read_json(source / "identity.json"),
        source_seals={k: units[k]["completion"]["seal_hash"] for k in selected_units},
        config=cfg.mapping(), assessment_identity=store.identity, dataset_manifest=provenance["dataset_manifest"],
        **code_provenance(store.identity, previous),
        environment=dict(python=platform.python_version(), pytorch=torch.__version__, cuda=torch.version.cuda,
            device=cfg.device, device_name=torch.cuda.get_device_name(0), dtype="float32/float64"))
    if not store.is_complete("source-evidence"):
        store.begin("source-evidence", dict(kind="compression_assessment_sources"))
        store.write_json("source-evidence", "result.json", evidence)
        store.complete("source-evidence")
    elif previous != evidence:
        raise ValueError("sealed assessment source evidence changed")
    if dataset_manifest(args.data_root, check) != provenance["dataset_manifest"]:
        raise ValueError("cached dataset differs from the sealed source")
    model, layout = build_model(cfg.seed_for("pilot", 0, "initialization_model"), channels=channels,
        factor_seed=cfg.seed_for("pilot", 0, "factors"), device=cfg.device, dtype=torch.float32)
    layout.assert_metadata(asset["layout"])
    for snapshot in snapshots.values():
        layout.assert_metadata(snapshot["layout"])
    ltm, _ = memory_indices(layout)
    plan = read_json(units[pilot_asset_id]["path"] / "plan.json")
    ids = torch.tensor(plan["diagnostic_reserve_ids"], dtype=torch.long)
    if ids.shape != (128, 9):
        raise ValueError("assessment requires the reserved 128-grid diagnostic panel")
    data = MNISTSource(args.data_root)
    raw = data.images(ids.flatten().tolist()).reshape(128, 9, 1, 28, 28).float()
    parameters = torch.tensor(plan["diagnostic_reserve_transforms"], dtype=torch.float64)
    panel = render_pairs(raw, *affine_from_parameters(parameters), check)
    panel_seed = cfg.seed_for("pilot", 0, "rank-panel-strengths")
    strengths = .5 + .5 * torch.rand(128, generator=torch.Generator().manual_seed(panel_seed), dtype=torch.float64)
    panel = blur_pairs(panel, strengths, blur, check)
    matrices, references, results = {"initial": asset["reference_fisher"].double().cpu()}, {}, {}
    modes = ("cold", "warm") if args.warm_start else ("cold",)
    for mode in modes:
        for steps in (4, 8):
            case_results = []
            for name in ("initial", "n25", "n75"):
                unit = f"{name}-cold-s{steps}" if name == "initial" else f"{name}-{mode}-s{steps}"
                if store.is_complete(unit):
                    result = read_json(store.unit_path(unit) / "result.json")
                else:
                    check()
                    if name not in matrices:
                        matrices[name] = reference_fisher(model, layout, snapshots[name]["theta"].to(
                            device=cfg.device, dtype=torch.float32), panel.to(cfg.device), indices=ltm,
                            chunk_size=cfg.fisher_chunk, check_stop=check).cpu()
                    matrix = matrices[name]
                    if name not in references:
                        references[name] = measure(matrix, lambda m: compress_fisher(m, rank=64,
                            backend="exact_cuda", check_stop=check), cfg.device, check)
                    exact, exact_profile = references[name]
                    seed = cfg.seed_for("compression-assessment", 0, name + ":subspace")
                    warm = snapshots[name]["fisher"]["factor"] if mode == "warm" and name != "initial" else None
                    candidate, profile = measure(matrix, lambda m: approximate_fisher(m.to(cfg.device),
                        rank=64, seed=seed, steps=steps, warm_start=warm, check_stop=check), cfg.device, check)
                    probe_seed = cfg.seed_for("compression-assessment", 0, name + ":quadratics_solves")
                    probes = torch.randn(len(matrix), 16, generator=torch.Generator().manual_seed(probe_seed), dtype=torch.float64)
                    audit = matrix_audit(matrix, exact, candidate, probes)
                    fit = fit_audit(snapshots[name], exact, candidate, cfg, model, layout, check) \
                        if name != "initial" and audit["passed"] else None
                    result = dict(case=name, mode="cold" if name == "initial" else mode, steps=steps,
                        rank=64, panel_seed=panel_seed, compression_seed=seed, probe_seed=probe_seed,
                        exact_profile=exact_profile, candidate_profile=profile, matrix_audit=audit,
                        fit_audit=fit, passed=audit["passed"] and (name == "initial" or bool(fit and fit["passed"])))
                    store.begin(unit, dict(kind="compression_assessment_case", case=name, mode=result["mode"], steps=steps))
                    store.write_json(unit, "result.json", result)
                    store.complete(unit)
                    print(json.dumps(dict(event="case_complete", unit=unit, passed=result["passed"])), flush=True)
                case_results.append(result)
            results[f"{mode}-s{steps}"] = case_results
            if all(r["passed"] for r in case_results):
                break
    selected = {mode: next((steps for steps in (4, 8) if f"{mode}-s{steps}" in results
        and all(r["passed"] for r in results[f"{mode}-s{steps}"])), None) for mode in modes}
    summary = dict(selected_steps=selected, all_cases_passed=any(s is not None for s in selected.values()),
        status="assessment_only_no_production_admission", cases=results, source_evidence=evidence,
        cumulative_assessment_seconds=store.control.active_seconds, aggregate_active_seconds=budget.used)
    if not store.is_complete("summary"):
        store.begin("summary", dict(kind="compression_assessment_summary"))
        store.write_json("summary", "result.json", summary)
        store.complete("summary")
    return read_json(store.unit_path("summary") / "result.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--max-wall-seconds", type=float, default=600.)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    args.source_root, args.root = args.source_root.resolve(), args.root.resolve()
    budget_root = args.source_root.parent
    if args.root.parent != budget_root or args.root == args.source_root or not 0 < args.max_wall_seconds <= 1200:
        raise ValueError("use a unique peer root and a wall limit in (0,1200]")
    cfg = replace(Config.read(args.source_root / "resolved_config.json"), device=args.device)
    provenance = read_json(args.source_root / "provenance.json")
    source_identity = read_json(args.source_root / "identity.json")["identity"]
    if (cfg.scientific_hash != source_identity["config_hash"]
            or provenance["source_hash"] != source_identity["source_hash"]
            or content_hash(provenance["dataset_manifest"]) != source_identity["data_hash"]):
        raise ValueError("source configuration/provenance identity mismatch")
    hashes = {p.name: file_digest(p) for p in (Path(__file__), Path(__file__).with_name("plan15_grid_compression_probe.py"))}
    identity = dict(allocation_family=ALLOCATION_FAMILY, role="compression_assessment", schema_version=1,
        source_root=str(args.source_root), source_identity=read_json(args.source_root / "identity.json")["identity_hash"],
        code_hashes=hashes, config_hash=cfg.scientific_hash, warm_start=args.warm_start, rank=64)
    with budget_lock(budget_root):
        other = peer_seconds(budget_root, args.root)
        external = file_bytes(budget_root, exclude=args.root) + file_bytes(args.data_root)
        preflight = time.monotonic() - started
        with Store(args.root, identity, resume=args.resume, external_bytes=external,
            max_bytes=cfg.max_artifact_bytes, free_reserve_bytes=cfg.free_reserve_bytes,
            finalization_reserve_bytes=32_000_000, max_active_seconds=min(1200., max(0., 72000-other)),
            max_wall_seconds=max(0., args.max_wall_seconds-preflight)) as store:
            store.control.restore_active_seconds(store.control.active_seconds + preflight)
            budget = RuntimeBudget(store, other_seconds=other)
            try:
                with store.control.signal_handlers():
                    budget.check()
                    if sources() != provenance["sources"] or content_hash(provenance["sources"]) != provenance["source_hash"]:
                        raise ValueError("current scientific source differs from the sealed source")
                    setup_torch(cfg)
                    result = execute(args, cfg, provenance, store, budget)
                    print(json.dumps(result, allow_nan=False), flush=True)
                    return 0 if result["all_cases_passed"] else 1
            except (StopRequested, ResourceLimit) as error:
                print(json.dumps(dict(status="paused", reason=str(error))), flush=True)
                return 2
            finally:
                budget.close()


if __name__ == "__main__":
    raise SystemExit(main())
