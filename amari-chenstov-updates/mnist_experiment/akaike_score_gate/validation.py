"""Explicit Phase 0 CPU/CUDA pipeline smokes and safe-boundary resume audits.

This module runs real cached-MNIST assets only when invoked by orchestration.
The timing profile is recorded as provenance, not applied to smaller smokes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .artifacts import RunLimit, StopExecution, UnitStore, file_hash
from .assets import ensure_assets
from .config import Study, canonical_hash
from .trajectory import ensure_trajectory


REQUIRED = ("summary.json", "metrics.json", "validation_manifest.json")
VALIDATION_SCHEMA_VERSION = 1


class _ValidationInterruption(StopExecution):
    pass


class _InterruptAfterChecks(RunLimit):
    """Inject one halt without swallowing or replacing the parent's limit."""

    def __init__(self, parent: RunLimit, checks: int = 3):
        self.parent = parent
        self.threshold = checks
        self.checks = 0

    @property
    def units(self):
        return self.parent.units

    def completed(self):
        self.parent.completed()

    def check(self):
        self.parent.check()
        self.checks += 1
        if self.checks == self.threshold:
            raise _ValidationInterruption("intentional validation halt after a saved fit")


def _child_study(parent: Study, device: str) -> Study:
    name = "cuda_smoke.json" if device == "cuda" else "smoke.json"
    raw = Study.from_path(Path(__file__).parent / "configs" / name).mapping()
    raw.update(device=device, data_root=parent.data_root, dtype=parent.dtype,
               seed_root=parent.seed_root, tolerance_change=parent.tolerance_change)
    return Study(raw)


def _child_store(root: Path, study: Study, parent: UnitStore, *, limit=None) -> UnitStore:
    child = UnitStore(root, study, limit=parent.limit if limit is None else limit)
    child.sources = dict(parent.sources)
    return child


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare(actual: Any, expected: Any, path: str = "artifact") -> None:
    """Compare science, including charged work; exclude only measured times."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6,
                                   equal_nan=True, msg=lambda message: f"{path}: {message}")
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: incompatible scientific fields")
        for key in actual:
            if key in {"started_unix", "peak_cuda_bytes", "peak_cpu_bytes"} or (
                key.endswith("_seconds") and key != "charged_seconds"
            ):
                continue
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: unequal scientific sequence lengths")
        for i, (one, two) in enumerate(zip(actual, expected)):
            _compare(one, two, f"{path}[{i}]")
    elif isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(f"{path}: unequal scientific scalars {actual}, {expected}")
    elif actual != expected:
        raise AssertionError(f"{path}: unequal scientific values {actual}, {expected}")


def _compare_trajectories(resumed: Path, reference: Path) -> None:
    for name in ("final_state.pt", "snapshots.pt"):
        actual, expected = _load(resumed / name), _load(reference / name)
        if name == "final_state.pt":
            for final in (actual, expected):
                if not all(bool(torch.isfinite(value).all()) for value in final["model"].values()):
                    raise AssertionError("validation produced nonfinite model tensors")
        _compare(actual, expected, name)
    for name in ("metrics.json", "gates.json", "summary.json"):
        _compare(json.loads((resumed / name).read_text()),
                 json.loads((reference / name).read_text()), name)


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(file): file_hash(file) for path in paths
            for file in sorted(path.iterdir()) if file.is_file()}


def ensure_pipeline_validation(store: UnitStore, profile: dict | None, resume: bool = True) -> Path:
    """Seal real-data pipeline validation, with CPU mandatory for CUDA parents.

    All child stores share the parent's wall/unit limit. Source fitting, assets,
    and trajectories live below ``validation/<device>`` and use smoke configs.
    No deployment work charges are inferred from these unbudgeted audits.
    """
    devices = ["cpu"] + (["cuda"] if store.study.device == "cuda" else [])
    studies = {device: _child_study(store.study, device) for device in devices}
    unit = store.unit("phase0", "pipeline_validation", 0, detail={
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "devices": devices, "child_config_hashes": {d: s.config_hash for d, s in studies.items()},
        "profile_hash": None if profile is None else canonical_hash(profile),
        "trajectory_allowance": "unlimited_audit", "interruption_check": 3,
        "comparison_atol": 1e-6, "comparison_rtol": 1e-6,
        "adaptive_gate_validation": {"ridge_ratio": 100., "max_iter": 200, "max_eval": 300,
                                     "requires_real_gate": True, "scientific_settings_unchanged": True},
    })
    session = store.begin(unit, REQUIRED, resume=resume)
    if session is None:
        return store.completed(unit, REQUIRED)
    identity = canonical_hash(unit)
    progress_path = session.path / "validation_checkpoint.json"
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        if saved["identity"] != identity or saved["payload_hash"] != canonical_hash(saved["payload"]):
            raise RuntimeError("corrupt or incompatible pipeline-validation checkpoint")
        progress = saved["payload"]
    else:
        progress = {"schema_version": VALIDATION_SCHEMA_VERSION, "cases": {},
                    "cpu_status": "pending", "cuda_status": "pending" if "cuda" in devices else "not_requested"}

    def save_progress():
        session.write_json("validation_checkpoint.json", {
            "identity": identity, "payload": progress, "payload_hash": canonical_hash(progress),
        })
        session.write_json("validation_manifest.json", progress)

    previous_threads = torch.get_num_threads()
    try:
        for device, child_study in studies.items():
            store.limit.check()
            if device == "cuda" and not torch.cuda.is_available():
                progress["cuda_status"] = "unavailable"
                save_progress()
                raise RuntimeError("CUDA pipeline validation was requested but CUDA is unavailable")
            torch.set_num_threads(child_study.cpu_threads)
            child_root = store.root / "validation" / device
            asset_store = _child_store(child_root, child_study, store)
            environments = [env for env in child_study.environments()
                            if env["name"] in {"rotation_linear", "mixture_m2"}]
            if {env["name"] for env in environments} != {"rotation_linear", "mixture_m2"}:
                raise RuntimeError("pipeline smoke requires rotation_linear and mixture_m2")
            for environment in environments:
                family = environment["family"]
                asset_path = ensure_assets(asset_store, "phase0", 0, family, 5, resume=True)
                asset = _load(asset_path / "assets.pt")
                steps = len(asset["streams"][environment["name"]]["positions"])
                snapshots = tuple(range(steps))
                retention = {"omega": 39. if family == "rotation" else 19.,
                             "ridge_ratio": 0.1 if family == "rotation" else 0.}
                common = {"c": 1., "half_life": 8., "cadence": 1,
                          "rank": 8, "score_samples": child_study.score_samples}
                policies = [
                    {**common, "name": "depth1_diag", "depth": 1, "ewc": "diag", "gate": None},
                    {**common, "name": "asg_lr_lr", "depth": None, "ewc": "lr", "gate": "lr"},
                ]
                for policy in policies:
                    store.limit.check()
                    label = f"{environment['name']}__{policy['name']}"
                    case_id = f"{device}/{label}"
                    record = progress["cases"].setdefault(case_id, {})
                    reference_store = _child_store(child_root / "uninterrupted" / label, child_study, store)
                    resumed_store = _child_store(child_root / "interrupted" / label, child_study, store)
                    fit_retention = (retention if policy["gate"] is None else
                                     {**retention, "ridge_ratio": 100., "max_iter": 200, "max_eval": 300})
                    kwargs = dict(width=5, retention=fit_retention, profile=None,
                                  budget_tier="unlimited", snapshot_steps=snapshots, resume=True)

                    def run(child):
                        return ensure_trajectory(child, "phase0", 0, environment, policy, asset_path, **kwargs)

                    reference = run(reference_store)
                    if policy["gate"] is not None:
                        gates = json.loads((reference / "gates.json").read_text())
                        if not gates:
                            raise AssertionError("adaptive restart validation must exercise a real gate")
                    if "interruption" not in record:
                        limiter = _InterruptAfterChecks(store.limit)
                        resumed_store.limit = limiter
                        try:
                            run(resumed_store)
                        except _ValidationInterruption:
                            # Prior immutable code revisions can coexist here.
                            reference_unit = json.loads((reference / "config.json").read_text())
                            _, working = resumed_store.paths(reference_unit)
                            checkpoint_path = working / "checkpoint.pt"
                            if not checkpoint_path.is_file():
                                raise RuntimeError("interruption must leave the current accepted fit checkpoint")
                            checkpoint = _load(checkpoint_path)
                            # A parent wall limit may already have left a later
                            # accepted boundary before this deliberate halt.
                            if checkpoint["fit_calls"] < 1 or checkpoint["stage"] not in {
                                "gate", "evaluate", "consolidate", "advance", "arrival", "fit",
                            }:
                                raise RuntimeError("validation interruption did not reach a completed fit boundary")
                            name = f"interruptions/{device}__{label}.pt"
                            session.write_torch(name, checkpoint)
                            record["interruption"] = {"checks": limiter.checks, "artifact": name,
                                "artifact_hash": file_hash(session.path / name),
                                "stage": checkpoint["stage"], "fit_calls": checkpoint["fit_calls"],
                                "budget": checkpoint["budget"]}
                            save_progress()
                        else:
                            raise RuntimeError("validation failed to exercise an interrupted fit")
                        finally:
                            resumed_store.limit = store.limit
                    certificate = record["interruption"]
                    if file_hash(session.path / certificate["artifact"]) != certificate["artifact_hash"]:
                        raise RuntimeError("corrupt accepted-state interruption certificate")
                    resumed = run(resumed_store)
                    _compare_trajectories(resumed, reference)

                    source = ensure_assets(asset_store, "phase0", 0, family, 5, source_only=True, resume=True)
                    complete_paths = [source, asset_path, reference, resumed]
                    before_hashes, completed_units = _hashes(complete_paths), store.limit.units
                    if ensure_assets(asset_store, "phase0", 0, family, 5, source_only=True, resume=True) != source:
                        raise AssertionError("completed source identity changed on resume")
                    if ensure_assets(asset_store, "phase0", 0, family, 5, resume=True) != asset_path:
                        raise AssertionError("completed asset identity changed on resume")
                    if run(reference_store) != reference or run(resumed_store) != resumed:
                        raise AssertionError("completed trajectory identity changed on resume")
                    if store.limit.units != completed_units or _hashes(complete_paths) != before_hashes:
                        raise AssertionError("completed resume performed work or changed artifacts")
                    record["result"] = {"metric_kind": "pipeline_validation", "device": device,
                        "environment": environment["name"], "policy": policy["name"], "status": "passed",
                        "arrival_count": steps, "tensor_and_metric_resume_equal": True,
                        "budget_resume_equal": True, "completed_resume_units": 0,
                        "retention": fit_retention,
                        "real_gate_required": policy["gate"] is not None,
                        "asset_path": str(asset_path.relative_to(store.root)),
                        "reference_path": str(reference.relative_to(store.root)),
                        "resumed_path": str(resumed.relative_to(store.root))}
                    save_progress()
            progress[device + "_status"] = "passed"
            save_progress()
    finally:
        torch.set_num_threads(previous_threads)

    rows = [record["result"] for record in progress["cases"].values()]
    summary = {"phase": "phase0", "kind": "pipeline_validation", "status": "passed",
               "cpu_status": progress["cpu_status"], "cuda_status": progress["cuda_status"],
               "validated_cases": len(rows), "development": True,
               "profile_hash": unit["detail"]["profile_hash"],
               "budget_scope": "unlimited audit; deterministic caps tested separately",
               "comparison_atol": 1e-6, "comparison_rtol": 1e-6}
    session.write_json("summary.json", summary)
    session.write_json("metrics.json", rows)
    certificates = tuple(record["interruption"]["artifact"] for record in progress["cases"].values())
    return store.finish(session, (*REQUIRED, "validation_checkpoint.json", *certificates))
