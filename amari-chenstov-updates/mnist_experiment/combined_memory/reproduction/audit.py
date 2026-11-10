"""Read-only historical seals and explicit numerical reproduction comparisons."""

import math
from pathlib import Path

import numpy as np
import torch

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory_regression.runtime import read_json, dataset_manifest
from .config import POLICIES, TOLERANCES, recipe
from .runtime import historical_sources, check_historical_unit


def load_checkpoint(root, unit, previous=False):
    path = Path(root) / "units" / unit
    pointer = read_json(path / "latest.json")
    descriptor = pointer.get("previous" if previous else "current")
    if descriptor is None:
        return None
    if artifacts._file_hash(path / descriptor["file"]) != descriptor["sha256"]:
        raise artifacts.ArtifactError("checkpoint checksum mismatch")
    envelope = torch.load(path / descriptor["file"], map_location="cpu", weights_only=False)
    if (envelope["identity_hash"] != artifacts._identity(Path(root))["identity_hash"]
            or envelope["unit_id"] != unit
            or envelope["manifest_hash"] != artifacts._hash_json(read_json(path / "manifest.json"))):
        raise artifacts.ArtifactError("historical checkpoint binding changed")
    return envelope["payload"]


def historical_records(history, cfg, check=lambda: None):
    from mnist_experiment.fisher_memory_stm_capacity.runner import trajectory_id
    rows = []
    seals = {}
    units = set()
    for replica in range(8):
        for policy, (chart, _) in POLICIES.items():
            unit = trajectory_id(chart, "confirmation--grid", replica, recipe(cfg, policy), 100)
            record = check_historical_unit(unit, history, check)
            seals[unit] = record["completion"]["seal_hash"]
            path = Path(history) / "units" / unit
            summary = read_json(path / "summary.json")
            if summary["recipe"] != recipe(cfg, policy).mapping():
                raise ValueError("historical recipe differs from frozen policy")
            units.add(summary["asset_id"])
            rows.append({**summary, "policy": policy, "context": "blur",
                         "metrics": read_json(path / "metrics.json")})
    for unit in sorted(units):
        check()
        record = check_historical_unit(unit, history, check)
        seals[unit] = record["completion"]["seal_hash"]
        parent = record["metadata"].get("parent")
        if parent and parent not in seals:
            value = check_historical_unit(parent, history, check)
            seals[parent] = value["completion"]["seal_hash"]
    return rows, seals


def preflight(history, data_root, cfg, check=lambda: None):
    details = historical_sources(history, check)
    if dataset_manifest(data_root, check) != details["dataset_manifest"]:
        raise ValueError("historical MNIST data hash mismatch")
    rows, seals = historical_records(history, cfg, check)
    lineage = read_json(Path(history) / "units/decision-design/result.json")
    return dict(status="passed", source_files=len(details["sources"]),
                historical_identity=artifacts._identity(Path(history))["identity_hash"],
                historical_units=seals, parent_lineage=lineage,
                recorded_environment={k: details[k] for k in ("python", "torch", "cuda", "device_name")},
                tolerances=TOLERANCES, records=rows)


def compare_array(old, new, label, *, atol=None, rtol=None):
    old, new = torch.as_tensor(old, dtype=torch.float64), torch.as_tensor(new, dtype=torch.float64)
    if old.shape != new.shape or not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise ValueError(f"nonfinite or incompatible parity field: {label}")
    absolute = float((old - new).abs().max()) if old.numel() else 0.
    relative = float((old - new).norm() / old.norm().clamp_min(1e-30))
    atol = TOLERANCES["atol"] if atol is None else atol
    rtol = TOLERANCES["rtol"] if rtol is None else rtol
    if not torch.allclose(old, new, atol=atol, rtol=rtol):
        raise ValueError(f"parity mismatch {label}: max_abs={absolute:.8g}, relative={relative:.8g}")
    return dict(field=label, max_absolute=absolute, relative_norm=relative,
                absolute_tolerance=atol, relative_tolerance=rtol)


def compare_paths(history, historical_unit, store, new_unit):
    old_path, new_path = Path(history) / "units" / historical_unit, store.unit_path(new_unit)
    old_rows, new_rows = read_json(old_path / "metrics.json"), read_json(new_path / "metrics.json")
    if len(old_rows) != len(new_rows):
        raise ValueError("parity metric count mismatch")
    checks = []
    for field in ("n", "observations", "base_observations"):
        if [row[field] for row in old_rows] != [row[field] for row in new_rows]:
            raise ValueError(f"parity clock mismatch: {field}")
    for field in ("base_mse", "novel_mse", "base_nll", "novel_nll", "base_accuracy1", "novel_accuracy1"):
        checks.append(compare_array([r[field] for r in old_rows], [r[field] for r in new_rows],
                                    field, atol=TOLERANCES["risk_atol"]))
    old_fits, new_fits = read_json(old_path / "fits.json"), read_json(new_path / "fits.json")
    if len(old_fits) != len(new_fits):
        raise ValueError("parity accepted-update count differs")
    for field in ("step", "n", "observations", "selected_replay_ids", "selected_count", "solver_reason"):
        if [r.get(field) for r in old_fits] != [r.get(field) for r in new_fits]:
            raise ValueError(f"parity update/replay mismatch: {field}")
    for field in ("initial_objective", "final_objective", "gradient_inf_norm"):
        if all(isinstance(r.get(field), (int, float)) for r in old_fits + new_fits):
            checks.append(compare_array([r[field] for r in old_fits], [r[field] for r in new_fits], field))
    old_retire, new_retire = read_json(old_path / "retirement.json"), read_json(new_path / "retirement.json")
    if old_retire["retirements"] != new_retire["retirements"]:
        raise ValueError("parity eviction ledger differs")
    old = load_checkpoint(history, historical_unit)
    new = store.load_checkpoint(new_unit)
    if old["layout"] != new["layout"]:
        raise ValueError("parity parameter chart differs")
    checks.append(compare_array(old["learner"]["theta"], new["learner"]["theta"], "final_theta"))
    for key in ("anchor_L", "ltm_indices"):
        checks.append(compare_array(old["learner"][key], new["learner"][key], key))
    if old["learner"]["replay"] != new["learner"]["replay"]:
        raise ValueError("final replay states differ")
    # Check the represented operator; eigenvector signs themselves are arbitrary.
    fisher_old, fisher_new = old["learner"]["fisher_L"], new["learner"]["fisher_L"]
    if isinstance(fisher_old, dict):
        generator = torch.Generator().manual_seed(18110908)
        probes = torch.randn((len(old["learner"]["ltm_indices"]), 8), generator=generator,
                             dtype=torch.float64)
        def apply(value):
            factor, diagonal = value["factor"].double(), value["residual_diagonal"].double()
            return factor @ (factor.T @ probes) + diagonal[:, None] * probes
        checks.append(compare_array(apply(fisher_old), apply(fisher_new), "fisher_operator"))
        checks.append(compare_array(fisher_old["factor"].double().square().sum(1)
                                    + fisher_old["residual_diagonal"].double(),
                                    fisher_new["factor"].double().square().sum(1)
                                    + fisher_new["residual_diagonal"].double(), "fisher_diagonal"))
    elif torch.is_tensor(fisher_old):
        checks.append(compare_array(fisher_old, fisher_new, "fisher"))
    old_asset = read_json(old_path / "summary.json")["asset_id"]
    new_asset = read_json(new_path / "summary.json")["asset_id"]
    namespace = new_unit.split("-blur--", 1)[0]
    new_asset_unit = f"{namespace}-shared--{new_asset}"
    old_initial = load_checkpoint(history, old_asset)
    new_initial = store.load_checkpoint(new_asset_unit)
    checks.append(compare_array(old_initial["theta"], new_initial["theta"], "initial_theta"))
    if old_initial["plan_hash"] != new_initial["plan_hash"]:
        raise ValueError("regenerated historical source/transform plan changed")
    return dict(status="passed", historical_unit=historical_unit, rerun_unit=new_unit,
                accepted_updates=len(new_fits), retirements=len(new_retire["retirements"]),
                plan_hash=old_initial["plan_hash"], checks=checks)
