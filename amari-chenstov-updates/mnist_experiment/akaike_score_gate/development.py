"""Bounded, immutable Phase 1 selection and preassigned Phase 2 dispatch.

Only Phase 1 source-validation and development panels may select settings.
The caller owns the run lock. ``refresh`` is an optional zero-argument callback,
invoked after paired blocks, at 15-minute safe boundaries, and on exit.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError, _read_json

from .artifacts import UnitStore, file_hash, freeze_json
from .assets import ASSET_REQUIRED, SOURCE_REQUIRED, ensure_assets
from .config import Study, canonical_hash
from .fisher_audit import ensure_audit, ensure_depth_audit
from .trajectory import REQUIRED as TRAJECTORY_REQUIRED, ensure_trajectory


DECISION_REQUIRED = ("summary.json", "decision.json")
FAMILIES = ("rotation", "mixture")
SOURCE_WIDTHS = (5, 6, 0)
PILOT_DEPTHS = (1, 4, 6)


def audit_positions(num_steps: int, count: int = 3) -> tuple[int, ...]:
    """Evenly spaced early/middle/late arrivals, excluding the terminal arrival."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (num_steps, count)):
        raise ValueError("num_steps and count must be positive integers")
    last = max(0, num_steps - 2)
    if count == 1:
        return (last // 2,)
    return tuple(sorted({round(i * last / (count - 1)) for i in range(count)}))


def retention_candidates(family: str) -> list[dict[str, Any]]:
    """Five one-axis conditions, with duplicate starting ridge values removed."""
    if family not in FAMILIES:
        raise ValueError("unknown environment family")
    omega, ridge = (39.0, 0.1) if family == "rotation" else (19.0, 0.0)
    proposals = [("default", omega, ridge), ("omega_half", omega / 2, ridge),
                 ("omega_double", omega * 2, ridge),
                 *((f"ridge_{value:g}", omega, value) for value in (0.0, 0.01, 0.1))]
    result, seen = [], set()
    for name, strength, ratio in proposals:
        if (strength, ratio) not in seen:
            result.append({"name": name, "omega": strength, "ridge_ratio": ratio})
            seen.add((strength, ratio))
    return result


def optimizer_ceilings(study: Study) -> tuple[int, ...]:
    if study.smoke:
        return (study.max_iter,)
    return (study.max_iter, *(value for value in (100, 200) if value > study.max_iter))


def solver_settings(study: Study, max_iter: int) -> dict[str, Any]:
    return {"max_iter": max_iter,
            "max_eval": math.ceil(study.max_eval * max_iter / study.max_iter),
            "residual_tolerance": study.tolerance_grad}


def build_ledger(study: Study) -> dict[str, Any]:
    """Preassign every source comparison, pilot, driver, and audit case."""
    environments = study.environments()
    replicas = study.phase_replicas["phase1"]
    snapshots = {env["name"]: list(audit_positions(
        3 * study.rotation_transitions_per_arrow if env["family"] == "rotation" else study.mixture_steps,
        study.audit_positions)) for env in environments}
    cases = []
    for replica in range(replicas):
        for environment_index, environment in enumerate(environments):
            for stage, step in enumerate(snapshots[environment["name"]]):
                ordinal = len(cases)
                cases.append({"ordinal": ordinal, "replica_index": replica,
                              "environment": environment, "stage": stage, "step": step,
                              "source_depth": PILOT_DEPTHS[(replica + environment_index + stage) % 3],
                              "frontier": 1 + ordinal % 5,
                              "audit_initialization_seed": study.seed("audit_initialization", ordinal)})
    historical_count = min(study.historical_audits, len(cases))
    historical = ({len(cases) // 2} if historical_count == 1 else
                  {round(i * (len(cases) - 1) / (historical_count - 1)) for i in range(historical_count)})
    for case in cases:
        case["historical"] = case["ordinal"] in historical
    pilots = [{"index": index, "environment": env, "depth": depth,
               "candidate": retention_candidates(env["family"])[candidate]}
              for index in range(min(2, replicas)) for candidate in range(5)
              for depth in PILOT_DEPTHS for env in environments]
    drivers = [{"index": index, "environment": env, "depth": depth}
               for index in range(replicas) for depth in (*range(1, 7), 0) for env in environments]
    return {
        "schema_version": 1, "study_hash": study.config_hash,
        "sources": [{"index": index, "width": width, "family": family}
                    for index in range(replicas) for width in SOURCE_WIDTHS for family in FAMILIES],
        "source_extension": "extend every width/family/replica if any source is unsettled and budget remains",
        "optimizer_ceilings": list(optimizer_ceilings(study)),
        "optimizer_rule": "raise common ceiling iff unresolved fits exceed 5 percent; residual tolerance fixed",
        "pilots": pilots, "drivers": drivers, "cases": cases, "snapshot_steps": snapshots,
        "retention_margin": 0.02, "default_tie_tolerance": 0.005,
        "source_width_rule": "equal-family validation means; prefer five within .02 nat and .01 accuracy of six",
        "selection_data": "phase1 source_validation and development_train only",
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing/nonfinite {name}; do not select from surviving rows")
    return float(value)


def select_architecture(rows: list[dict], replicas: int) -> dict:
    """Select width on paired source-only validation; width zero is diagnostic."""
    expected = {(family, width, index) for family in FAMILIES for width in SOURCE_WIDTHS
                for index in range(replicas)}
    keys = [(row["family"], row["width"], row["index"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("source comparison must contain every paired width/family/replica exactly once")
    means = {}
    for width in SOURCE_WIDTHS:
        family_means = {}
        for family in FAMILIES:
            selected = [row for row in rows if row["width"] == width and row["family"] == family]
            family_means[family] = {field: sum(_finite(row[field], field) for row in selected) / replicas
                                    for field in ("source_nll", "source_accuracy")}
        means[str(width)] = {field: sum(family_means[f][field] for f in FAMILIES) / len(FAMILIES)
                             for field in ("source_nll", "source_accuracy")}
        means[str(width)]["families"] = family_means
    five, six, old = (means[str(width)] for width in SOURCE_WIDTHS)
    width = 5 if (five["source_nll"] <= six["source_nll"] + 0.02 + 1e-12
                  and five["source_accuracy"] >= six["source_accuracy"] - 0.01 - 1e-12) else 6
    limited = all(means[str(w)]["source_nll"] > old["source_nll"] + 0.02
                  or means[str(w)]["source_accuracy"] < old["source_accuracy"] - 0.01 for w in (5, 6))
    return {"width": width, "means": means, "capacity_limitation": limited,
            "criterion": "paired source-validation means, equal weight per family",
            "source_reference_role": "diagnostic_only_not_a_candidate",
            "capacity_flag_rule": "both new widths miss the old reference by .02 NLL or .01 accuracy",
            "replicas_per_family": replicas}


def optimizer_health(rows: list[dict]) -> dict:
    calls = unresolved = 0
    for row in rows:
        n, failed = row["fit_calls"], row["unresolved_fits"]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (n, failed)) or failed > n:
            raise ValueError("invalid optimizer-health counts")
        calls += n
        unresolved += failed
    if not calls:
        raise ValueError("optimizer pilot has no fit diagnostics")
    return {"fit_calls": calls, "unresolved_fits": unresolved,
            "unresolved_fraction": unresolved / calls,
            "raise_ceiling": unresolved > 0.05 * calls,
            "trajectories": len(rows)}


def select_retention(rows: list[dict], family: str) -> dict:
    """Compare complete paired candidates, retaining unresolved finite fits."""
    candidates = retention_candidates(family)
    selected = [row for row in rows if row["family"] == family]
    records, reference_keys = [], None
    expected_names = {candidate["name"] for candidate in candidates}
    if {row["candidate"] for row in selected} != expected_names:
        raise ValueError("retention grid has missing or unknown candidates")
    expected_cells = {(row["index"], other["environment"], depth)
                      for row in selected for other in selected for depth in PILOT_DEPTHS}
    for candidate in candidates:
        members = [row for row in selected if row["candidate"] == candidate["name"]]
        keys = [(row["index"], row["environment"], row["depth"]) for row in members]
        if (len(keys) != len(set(keys)) or set(keys) != expected_cells
                or (reference_keys is not None and set(keys) != reference_keys)):
            raise ValueError("retention candidates must have the same complete paired cells")
        reference_keys = set(keys)
        nll = sum(_finite(row["nll_auc"], "nll_auc") for row in members) / len(members)
        retained = sum(_finite(row["retention_nll"], "retention_nll") for row in members) / len(members)
        change = sum(_finite(row["retention_nll"], "retention_nll")
                     - _finite(row["initial_retention_nll"], "initial_retention_nll")
                     for row in members) / len(members)
        records.append({**candidate, "mean_current_nll": nll, "mean_retention_nll": retained,
                        "mean_retention_change": change, "admissible": change <= 0.02 + 1e-12,
                        "mean_current_plus_retention_nll": nll + retained,
                        "trajectories": len(members),
                        "unresolved_fits": sum(row.get("unresolved_fits", 0) for row in members)})
    feasible = [row for row in records if row["admissible"]]
    field = "mean_current_nll" if feasible else "mean_current_plus_retention_nll"
    eligible = feasible or records
    best = min(eligible, key=lambda row: row[field])
    original = next(row for row in records if row["name"] == "default")
    if original in eligible and original[field] <= best[field] + 0.005 + 1e-12:
        best = original
    return {"family": family, "selected": {key: best[key] for key in ("name", "omega", "ridge_ratio")},
            "constraint_failure": not feasible, "criterion": field, "candidates": records,
            "reference": "each paired trajectory's initial no-update source-distribution NLL"}


class _Refresh:
    def __init__(self, callback):
        self.callback, self.last = callback, time.monotonic()

    def __call__(self, *, block=False):
        if self.callback is not None and (block or time.monotonic() - self.last >= 900):
            self.callback()
            self.last = time.monotonic()


def _identity(store: UnitStore) -> dict:
    return {"schema_version": 1, "study_hash": store.study.config_hash, "source_hashes": store.sources}


def _parent(store, path, required, *, expected=None) -> dict:
    path = Path(path).resolve()
    unit = _read_json(path / "config.json")
    if unit.get("study_hash") != store.study.config_hash or unit.get("source_hashes") != store.sources:
        raise RotatedArtifactError(f"incompatible parent configuration/source hashes: {path}")
    for key, value in (expected or {}).items():
        if unit.get(key) != value:
            raise RotatedArtifactError(f"parent {key} mismatch: {path}")
    sealed = store.completed(unit, tuple(required))
    if sealed is None or sealed.resolve() != path:
        raise RotatedArtifactError(f"parent must be the expected sealed unit: {path}")
    integrity = _read_json(path / "integrity.json")
    hashes = {}
    for name in required:
        if name not in integrity or file_hash(path / name) != integrity[name]:
            raise RotatedArtifactError(f"corrupt required parent artifact: {path / name}")
        hashes[name] = integrity[name]
    return {"path": str(path), "config_hash": canonical_hash(unit), "artifacts": hashes}


def _verify_parent(store, reference):
    current = _parent(store, reference["path"], tuple(reference["artifacts"]))
    if current != reference:
        raise RotatedArtifactError(f"frozen parent hash changed: {reference['path']}")


def _log_once(store, event_id, message, **detail):
    path = store.root / "lab_log.json"
    rows = _read_json(path) if path.exists() else []
    if not any(row.get("event_id") == event_id for row in rows):
        store.log(message, event_id=event_id, **detail)


def _decision(store, kind, ledger_hash, parents, compute, *, resume):
    unit = store.unit("phase1", kind, 0, detail={"ledger_hash": ledger_hash, "parents": parents})
    session = store.begin(unit, DECISION_REQUIRED, resume=resume)
    if session is None:
        path = store.paths(unit)[0]
        return _read_json(path / "decision.json"), path
    result = compute()
    session.write_json("decision.json", result)
    session.write_json("summary.json", {"phase": "phase1", "kind": kind,
                                        "status": "completed", **result})
    return result, store.finish(session, DECISION_REQUIRED)


def _source_rows(store, ledger, *, extended, resume, pulse):
    rows, parents = [], []
    for ordinal, assignment in enumerate(ledger["sources"]):
        path = ensure_assets(store, "phase1", assignment["index"], assignment["family"],
                             assignment["width"], source_only=True, extended_epochs=extended, resume=resume)
        parent = _parent(store, path, SOURCE_REQUIRED, expected={"phase": "phase1", "kind": "source",
                          "index": assignment["index"], "environment": assignment["family"]})
        detail = _read_json(path / "config.json")["detail"]
        if detail["width"] != assignment["width"] or detail["extended_epochs"] != extended:
            raise RotatedArtifactError("source architecture/epoch protocol mismatch")
        summary = _read_json(path / "summary.json")
        rows.append({**assignment, "path": str(path), "source_nll": summary["source_nll"],
                     "source_accuracy": summary["source_accuracy"],
                     "settled": summary["convergence"]["settled"],
                     "selected_epoch": summary["selected_epoch"], "completed_epochs": summary["completed_epochs"]})
        parents.append(parent)
        pulse(block=(ordinal + 1) % len(FAMILIES) == 0)
    return rows, parents


def _policy(study, depth):
    return {"name": "no_update" if depth == 0 else f"depth{depth}_lr",
            "depth": depth, "ewc": "lr", "gate": None, "c": 1.0,
            "half_life": 8.0, "cadence": 1, "rank": 8, "score_samples": study.score_samples}


def _driver(store, assignment, asset_paths, width, retention, ledger, *, resume):
    index, env, depth = assignment["index"], assignment["environment"], assignment["depth"]
    asset_path = asset_paths[(index, env["family"])]
    path = ensure_trajectory(store, "phase1", index, env, _policy(store.study, depth), asset_path,
                             width=width, retention=retention, profile=None, budget_tier="unlimited",
                             snapshot_steps=tuple(ledger["snapshot_steps"][env["name"]]), resume=resume)
    parent = _parent(store, path, TRAJECTORY_REQUIRED, expected={"phase": "phase1", "kind": "trajectory",
                                                               "index": index, "environment": env["name"]})
    config = _read_json(path / "config.json")
    detail = config["detail"]
    if (detail["width"] != width or detail["retention"] != retention
            or detail["policy"] != _policy(store.study, depth)
            or detail["asset_hash"] != file_hash(asset_path / "config.json")):
        raise RotatedArtifactError("driver does not match its assigned source/policy/retention")
    entry = {"index": index, "environment": env, "depth": depth,
             "path": str(path.resolve()), "asset_path": str(asset_path.resolve())}
    return entry, _read_json(path / "summary.json"), parent


def _frozen_selection(store, ledger_hash, *, resume):
    pointer = store.root / "phase1_selection.json"
    if not pointer.exists():
        return None
    if not resume:
        raise RotatedArtifactError("frozen Phase 1 selection exists; use resume")
    result = _read_json(pointer)
    if any(result.get(key) != value for key, value in _identity(store).items()) or result.get("ledger_hash") != ledger_hash:
        raise RotatedArtifactError("frozen Phase 1 study/source/ledger hashes changed")
    path = Path(result["selection_path"])
    reference = _parent(store, path, DECISION_REQUIRED,
                        expected={"phase": "phase1", "kind": "phase1_selection"})
    stored = _read_json(path / "decision.json")
    expected = {**stored, "selection_path": str(path), "selection_hash": reference["config_hash"]}
    if result != expected:
        raise RotatedArtifactError("root Phase 1 selection differs from its sealed decision")
    for parent in result["parents"]:
        _verify_parent(store, parent)
    return result


def run_phase1(store: UnitStore, *, resume: bool = True,
               refresh: Callable[[], Any] | None = None) -> dict:
    """Freeze source architecture, common solver and retention; run paired drivers.

    Default counts are 24 source fits (plus a paired extension when needed),
    120 pilot trajectories per admitted optimizer ceiling, 112 starting drivers,
    replacement drivers only for changed retention, and 48 local depth audits.
    Smoke configurations reduce replicas, arrivals and audit positions explicitly.
    """
    pulse = _Refresh(refresh)
    try:
        ledger = {**build_ledger(store.study), "source_hashes": store.sources}
        ledger_hash = canonical_hash(ledger)
        freeze_json(store.root / "phase1_ledger.json", ledger, resume=resume)
        previous = _frozen_selection(store, ledger_hash, resume=resume)
        if previous is not None:
            return previous
        study = store.study
        parents = {}

        def keep(reference):
            parents[reference["path"]] = reference

        base_rows, source_parents = _source_rows(store, ledger, extended=False, resume=resume, pulse=pulse)
        for reference in source_parents:
            keep(reference)
        extension, extension_path = _decision(store, "source_epoch_protocol", ledger_hash, source_parents,
            lambda: {"extended_epochs": any(not row["settled"] for row in base_rows)
                     and study.source_epoch_ceiling > study.source_epochs,
                     "all_sources_settled": all(row["settled"] for row in base_rows),
                     "initial_epochs": study.source_epochs, "ceiling_epochs": study.source_epoch_ceiling,
                     "selection_panel": "source_validation", "source_rows": base_rows}, resume=resume)
        keep(_parent(store, extension_path, DECISION_REQUIRED))
        extended = extension["extended_epochs"]
        _log_once(store, f"source_epochs:{ledger_hash}", "Source epoch protocol selected on source-validation only",
                  **extension)
        source_rows = base_rows
        if extended:
            source_rows, source_parents = _source_rows(store, ledger, extended=True, resume=resume, pulse=pulse)
            for reference in source_parents:
                keep(reference)
        architecture, architecture_path = _decision(store, "architecture_selection", ledger_hash,
            [*source_parents, parents[str(extension_path.resolve())]],
            lambda: {**select_architecture(source_rows, study.phase_replicas["phase1"]),
                     "extended_epochs": extended, "source_rows": source_rows}, resume=resume)
        keep(_parent(store, architecture_path, DECISION_REQUIRED))
        width = architecture["width"]
        _log_once(store, f"architecture:{ledger_hash}", "Source-only architecture selection frozen",
                  width=width, extended_epochs=extended, means=architecture["means"],
                  capacity_limitation=architecture["capacity_limitation"])
        asset_paths = {}
        for index in range(study.phase_replicas["phase1"]):
            for family in FAMILIES:
                path = ensure_assets(store, "phase1", index, family, width, resume=resume,
                                     extended_epochs=extended)
                reference = _parent(store, path, ASSET_REQUIRED,
                                    expected={"phase": "phase1", "kind": "assets", "index": index,
                                              "environment": family})
                summary = _read_json(path / "summary.json")
                detail = _read_json(path / "config.json")["detail"]
                if (summary["evaluation_split"] != "development_train" or summary["width"] != width
                        or detail["extended_epochs"] != extended):
                    raise RotatedArtifactError("Phase 1 assets must use selected width and development-only evaluation")
                keep(reference)
                asset_paths[(index, family)] = Path(path)
            pulse(block=True)
        driver_cache = {}

        def get_driver(assignment, setting):
            key = (assignment["index"], assignment["environment"]["name"], assignment["depth"],
                   canonical_hash(setting))
            if key not in driver_cache:
                driver_cache[key] = _driver(store, assignment, asset_paths, width, setting, ledger, resume=resume)
            return driver_cache[key]

        pilot_history = []
        pilot_rows = []
        for ceiling in optimizer_ceilings(study):
            solver = solver_settings(study, ceiling)
            pilot_rows, pilot_parents = [], []
            for ordinal, assignment in enumerate(ledger["pilots"]):
                setting = {key: assignment["candidate"][key] for key in ("omega", "ridge_ratio")}
                entry, summary, reference = get_driver(assignment, {**setting, **solver})
                keep(reference)
                pilot_parents.append(reference)
                pilot_rows.append({"index": entry["index"], "environment": entry["environment"]["name"],
                    "family": entry["environment"]["family"], "depth": entry["depth"],
                    "candidate": assignment["candidate"]["name"], "path": entry["path"],
                    **{key: summary[key] for key in ("nll_auc", "retention_nll", "initial_retention_nll",
                                                    "fit_calls", "unresolved_fits", "online_seconds")}})
                pulse(block=(ordinal + 1) % len(study.environments()) == 0)
            health, health_path = _decision(store, "optimizer_pilot", ledger_hash, pilot_parents,
                lambda: {**optimizer_health(pilot_rows), "solver": solver,
                         "online_seconds": sum(row["online_seconds"] for row in pilot_rows),
                         "rows": pilot_rows}, resume=resume)
            keep(_parent(store, health_path, DECISION_REQUIRED))
            pilot_history.append({"path": str(health_path.resolve()), **{key: health[key] for key in (
                "solver", "fit_calls", "unresolved_fits", "unresolved_fraction", "raise_ceiling", "online_seconds")}})
            _log_once(store, f"optimizer:{ledger_hash}:{ceiling}", "Common optimizer pilot completed",
                      **pilot_history[-1], at_maximum=ceiling == optimizer_ceilings(study)[-1])
            if not health["raise_ceiling"]:
                break
        optimizer, optimizer_path = _decision(store, "optimizer_selection", ledger_hash,
            [parents[row["path"]] for row in pilot_history],
            lambda: {"solver": solver, "pilot_history": pilot_history,
                     "unresolved_at_ceiling": health["raise_ceiling"],
                     "residual_threshold_unchanged": True}, resume=resume)
        keep(_parent(store, optimizer_path, DECISION_REQUIRED))
        default_settings = {family: {**{key: retention_candidates(family)[0][key]
                                        for key in ("omega", "ridge_ratio")}, **solver} for family in FAMILIES}
        default_drivers = []
        for ordinal, assignment in enumerate(ledger["drivers"]):
            entry, _, reference = get_driver(assignment, default_settings[assignment["environment"]["family"]])
            default_drivers.append(entry)
            keep(reference)
            pulse(block=(ordinal + 1) % len(study.environments()) == 0)
        retention_choice, retention_path = _decision(store, "retention_selection", ledger_hash,
            [parents[str(optimizer_path.resolve())], *pilot_parents],
            lambda: {"families": {family: select_retention(pilot_rows, family) for family in FAMILIES},
                     "solver": solver, "selection_panel": "development_train"}, resume=resume)
        keep(_parent(store, retention_path, DECISION_REQUIRED))
        _log_once(store, f"retention:{ledger_hash}", "Shared family retention selected; failed constraints retained",
                  **retention_choice)
        retention = {family: {**{key: retention_choice["families"][family]["selected"][key]
                                for key in ("omega", "ridge_ratio")}, **solver} for family in FAMILIES}
        drivers = []
        for ordinal, (assignment, default) in enumerate(zip(ledger["drivers"], default_drivers, strict=True)):
            family = assignment["environment"]["family"]
            if retention[family] == default_settings[family]:
                drivers.append(default)
            else:
                entry, _, reference = get_driver(assignment, retention[family])
                drivers.append(entry)
                keep(reference)
            pulse(block=(ordinal + 1) % len(study.environments()) == 0)
        driver_map = {(row["index"], row["environment"]["name"], row["depth"]): row for row in drivers}
        depth_audits = []
        for case in ledger["cases"]:
            driver = driver_map[(case["replica_index"], case["environment"]["name"], case["source_depth"])]
            path = ensure_depth_audit(store, case["ordinal"], case, Path(driver["asset_path"]),
                                      Path(driver["path"]), resume=resume)
            keep(_parent(store, path, ("summary.json", "audit_metrics.json", "diagnostics.pt"),
                         expected={"phase": "phase1", "kind": "depth_audit"}))
            depth_audits.append(str(path.resolve()))
            pulse(block=True)
        result, selection_path = _decision(store, "phase1_selection", ledger_hash, list(parents.values()),
            lambda: {**_identity(store), "ledger_hash": ledger_hash, "width": width,
                     "extended_epochs": extended, "retention": retention, "drivers": drivers,
                     "default_drivers": default_drivers, "cases": ledger["cases"],
                     "depth_audits": depth_audits, "parents": list(parents.values()),
                     "architecture_decision": str(architecture_path.resolve()),
                     "optimizer_decision": str(optimizer_path.resolve()),
                     "retention_decision": str(retention_path.resolve()),
                     "constraint_failures": {family: retention_choice["families"][family]["constraint_failure"]
                                              for family in FAMILIES}}, resume=resume)
        selection_hash = canonical_hash(_read_json(selection_path / "config.json"))
        result = {**result, "selection_path": str(selection_path.resolve()), "selection_hash": selection_hash}
        freeze_json(store.root / "phase1_selection.json", result, resume=resume)
        _log_once(store, f"phase1_selection:{selection_hash}",
                  "Phase 1 architecture, epoch protocol, optimizer and retention frozen",
                  width=width, extended_epochs=extended, retention=retention,
                  optimizer_pilots=pilot_history, constraint_failures=result["constraint_failures"],
                  capacity_limitation=architecture["capacity_limitation"], selection_hash=selection_hash)
        return result
    finally:
        pulse(block=True)


def run_phase2(store: UnitStore, phase1_result: dict, *, resume: bool = True,
               refresh: Callable[[], Any] | None = None) -> list[Path]:
    """Dispatch the same preassigned cases; never choose states from audit outcomes."""
    pulse = _Refresh(refresh)
    try:
        ledger = {**build_ledger(store.study), "source_hashes": store.sources}
        ledger_hash = canonical_hash(ledger)
        if _read_json(store.root / "phase1_ledger.json") != ledger:
            raise RotatedArtifactError("Phase 2 ledger differs from predeclared Phase 1 cases")
        frozen = _frozen_selection(store, ledger_hash, resume=True)
        if frozen is None or frozen != phase1_result:
            raise RotatedArtifactError("Phase 2 requires the exact frozen Phase 1 selection")
        driver_map = {(row["index"], row["environment"]["name"], row["depth"]): row
                      for row in frozen["drivers"]}
        assignments = [{"case": case, "driver": driver_map[(case["replica_index"],
                        case["environment"]["name"], case["source_depth"])]["path"]}
                       for case in ledger["cases"]]
        freeze_json(store.root / "phase2_ledger.json", {**_identity(store), "phase1_ledger_hash": ledger_hash,
                    "phase1_selection_hash": frozen["selection_hash"], "assignments": assignments}, resume=resume)
        paths = []
        for case in ledger["cases"]:
            driver = driver_map[(case["replica_index"], case["environment"]["name"], case["source_depth"])]
            path = ensure_audit(store, case["ordinal"], case, Path(driver["asset_path"]),
                                Path(driver["path"]), resume=resume)
            _parent(store, path, ("summary.json", "audit_metrics.json", "diagnostics.pt"),
                    expected={"phase": "phase2", "kind": "fisher_audit"})
            paths.append(path)
            pulse(block=True)
        return paths
    finally:
        pulse(block=True)
