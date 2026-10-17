"""Small paired Phase 0 precision audit on already sealed prefit states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .artifacts import UnitStore, file_hash, freeze_json, run_lock
from .config import Study
from .model import build_model, active_indices
from .optimizer import fit_restricted
from .trajectory import scalar_tree, weighted_gradient, synchronize


def audit_precision(store, drivers, *, resume=True):
    for path in drivers:
        if not (path / "COMPLETED").is_file():
            raise ValueError("precision audit requires sealed drivers")
        integrity = json.loads((path / "integrity.json").read_text())
        for name in ("config.json", "snapshots.pt"):
            if integrity.get(name) != file_hash(path / name):
                raise ValueError("precision audit parent checksum mismatch")
    parents = {str(path): {name: file_hash(path / name) for name in ("config.json", "snapshots.pt")}
               for path in drivers}
    variants = [(dtype, tolerance) for dtype in ("float32", "float64")
                for tolerance in (1e-9, 1e-12)]
    required = ("summary.json", "metrics.json")
    unit = store.unit("phase0", "numerical_precision", 0, detail={
        "parents": parents, "variants": [list(value) for value in variants], "gradient_tolerance": 1e-5,
        "max_iter": 200, "max_eval": 300, "fit_depth": 1,
        "pairing": "identical prefit parameters, evidence, archive, anchor, and precision",
        "scope": "finite-precision diagnostic, not independent efficacy replicas"})
    session = store.begin(unit, required, resume=resume)
    if session is None:
        return store.completed(unit, required)
    rows = []
    for driver_index, driver in enumerate(drivers):
        config = json.loads((driver / "config.json").read_text())
        snapshots = torch.load(driver / "snapshots.pt", map_location="cpu", weights_only=False)
        for snapshot_index, snapshot in enumerate(snapshots):
            for dtype_name, tolerance in variants:
                name = f"parts/{driver_index}_{snapshot_index}_{dtype_name}_{tolerance}.json"
                part = session.path / name
                if part.exists():
                    rows.append(json.loads(part.read_text()))
                    continue
                store.limit.check()
                dtype, device = getattr(torch, dtype_name), torch.device(store.study.device)
                model, layout = build_model(store.study.seed("precision_audit", driver_index),
                    width=config["detail"]["width"], device=device, dtype=dtype)
                model.load_state_dict(snapshot["model"])
                inputs, targets = snapshot["inputs"].to(device=device, dtype=dtype), snapshot["targets"].to(device)
                weights = snapshot["weights"].to(device)
                anchor, precision = snapshot["anchor"].to(device=device, dtype=dtype), snapshot["precision"].to(device)
                indices, _ = active_indices(layout, 1)
                synchronize(device)
                begin = time.perf_counter()
                result = fit_restricted(model, layout, inputs, targets, weights, anchor, precision,
                    indices.to(device), max_iter=200, max_eval=300, tolerance_grad=1e-5,
                    residual_tolerance=1e-5, tolerance_change=tolerance)
                synchronize(device)
                elapsed = time.perf_counter() - begin
                gradient, _ = weighted_gradient(model, layout, inputs, targets, weights, indices.to(device))
                theta = layout.flatten_module(model, detach=True).double()
                residual = gradient + precision @ (theta - anchor.double())
                row = {"driver": str(driver), "environment": config["environment"],
                       "step": snapshot["step"], "dtype": dtype_name,
                       "tolerance_change": tolerance, "seconds": elapsed,
                       "verified_residual_max": float(residual[indices.to(device)].abs().max()),
                       "same_start_and_objective": True, **scalar_tree(result)}
                session.write_json(name, row)
                rows.append(row)
    cells = []
    for dtype_name, tolerance in variants:
        group = [row for row in rows if row["dtype"] == dtype_name and row["tolerance_change"] == tolerance]
        cells.append({"dtype": dtype_name, "tolerance_change": tolerance, "fits": len(group),
                      "converged": sum(row["converged"] for row in group),
                      "max_verified_residual": max(row["verified_residual_max"] for row in group),
                      "fit_seconds": sum(row["seconds"] for row in group)})
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", {"phase": "phase0", "kind": "numerical_precision",
        "cells": cells, "status": "completed", "same_start_and_objective": True,
        "scope": "numerical validation only; no efficacy or uncertainty claim"})
    return store.finish(session, required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    study = Study.from_path(args.config)
    if study.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(study.cpu_threads)
    root = args.output_root.resolve()
    with run_lock(root):
        freeze_json(root / "study.json", {**study.mapping(), "study_hash": study.config_hash}, resume=args.resume)
        path = audit_precision(UnitStore(root, study), [path.resolve() for path in args.trajectory], resume=args.resume)
        print((path / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
