"""Read-only, scalar analysis of sealed Plan 14 artifacts (schema 1).

No study/store class, tensors, dataset adapters, or execution modules are imported.
Missing optional measurements remain missing; they are never zero-filled.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 140061
BOOTSTRAP_SAMPLES = 512
_HASH_CACHE: dict[Path, tuple[tuple[int, ...], str]] = {}
PHASE_ORDER = {f"phase{index}": index for index in range(7)}
PHASE_LABELS = dict(zip(PHASE_ORDER, (
    "Phase 0: Mathematical, numerical, and execution validation",
    "Phase 1: Architecture, retention, and depth opportunity",
    "Phase 2: Fisher geometry and decision sensitivity",
    "Phase 3: Risk calibration and evidence horizon",
    "Phase 4: Causal development and bounded repair",
    "Phase 5: Fresh fixed-cohort evaluation",
    "Phase 6: Artifact synthesis and closeout",
)))
GROUP_FIELDS = (
    "phase", "environment", "schedule", "expected_nines", "total_observations",
    "budget_tier", "protocol_variant",
)
METRIC_FIELDS = (
    "step", "observations", "expected_nines", "realized_nines", "current_nll",
    "current_accuracy", "retention_nll", "retention_accuracy", "fixed_nll",
    "nine_precision", "nine_recall", "nine_false_positive_rate", "selected_depth",
    "active_parameters", "n_eff", "d_eff", "online_seconds", "charged_seconds",
)


class ArtifactError(ValueError):
    """A purportedly sealed result is missing, incompatible, or corrupt."""


def phase_sort_key(phase: str) -> tuple[int, str]:
    return PHASE_ORDER.get(phase, len(PHASE_ORDER)), phase


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Cannot read artifact {path}: {error}") from error


def _object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object: {path}")
    return value


def _check_versions(value: dict[str, Any], path: Path, *, required: bool = False) -> None:
    for name in ("schema_version", "manifest_schema_version", "artifact_schema_version",
                 "metric_schema_version"):
        if name in value and (type(value[name]) is not int or value[name] != SCHEMA_VERSION):
            raise ArtifactError(f"Incompatible {name} in {path}: expected {SCHEMA_VERSION}")
    if required:
        for name in ("artifact_schema_version", "metric_schema_version"):
            if name not in value:
                raise ArtifactError(f"Missing {name} in {path}")


def _fingerprint(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hash_file(path: Path, *, force_fresh: bool = False) -> str:
    """Cache only byte hashes, never completion/schema/integrity decisions."""
    digest = hashlib.sha256()
    try:
        path = path.resolve()
        before = _fingerprint(path)
        cached = _HASH_CACHE.get(path)
        if not force_fresh and cached is not None and cached[0] == before:
            return cached[1]
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if _fingerprint(path) != before:
            raise ArtifactError(f"Artifact changed while being checksum-validated: {path}")
    except OSError as error:
        raise ArtifactError(f"Missing integrity artifact {path}") from error
    value = digest.hexdigest()
    _HASH_CACHE[path] = before, value
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ArtifactError(f"Expected a list of metric objects: {path}")
    for row in value:
        _check_versions(row, path)
        for field in METRIC_FIELDS:
            number = row.get(field)
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ArtifactError(f"Invalid numeric metric {field} in {path}")
        flag = row.get("nine_precision_defined")
        if flag is not None and not isinstance(flag, bool):
            raise ArtifactError(f"Invalid nine_precision_defined flag in {path}")
    return value


def _load_unit(path: Path, phase: str, study: dict[str, Any], *, force_fresh: bool = False) -> dict[str, Any]:
    config = _object(path / "config.json")
    manifest = _object(path / "manifest.json")
    _check_versions(config, path / "config.json")
    _check_versions(manifest, path / "manifest.json", required=True)
    if manifest.get("status") != "completed":
        raise ArtifactError(f"COMPLETED unit has non-completed manifest: {path}")
    if config.get("phase") != phase:
        raise ArtifactError(f"Unit phase disagrees with directory: {path}")
    for name in ("kind", "index", "environment"):
        if name not in config:
            raise ArtifactError(f"Missing unit metadata {name}: {path}")
    if not isinstance(config.get("detail", {}), dict):
        raise ArtifactError(f"Unit detail must be an object: {path}")
    for name in ("phase", "environment", "policy", "kind"):
        if name in manifest and name in config and manifest[name] != config[name]:
            raise ArtifactError(f"Manifest/config {name} mismatch: {path}")
    if "config_hash" in manifest:
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        if manifest["config_hash"] != digest:
            raise ArtifactError(f"Configuration hash mismatch: {path}")
    expected_study = study.get("study_hash", study.get("config_hash"))
    for source in (config, manifest):
        if expected_study is not None and "study_hash" in source:
            if source["study_hash"] != expected_study:
                raise ArtifactError(f"Study hash mismatch: {path}")
    integrity = _object(path / "integrity.json")
    # Both plain SHA256 maps and explicitly versioned file maps are accepted.
    if "files" in integrity:
        _check_versions(integrity, path / "integrity.json")
        integrity = integrity["files"]
    if not isinstance(integrity, dict):
        raise ArtifactError(f"Invalid integrity map: {path}")
    required = {"summary.json"}
    if config.get("kind") == "trajectory":
        required.add("metrics.json")
    if config.get("kind") in {"audit", "fisher_audit"}:
        required.add("audit_metrics.json")
    declared = manifest.get("required", manifest.get("required_artifacts", []))
    if not isinstance(declared, list) or any(not isinstance(name, str) for name in declared):
        raise ArtifactError(f"Invalid required artifact list: {path}")
    required.update(name for name in declared if name not in {
        "config.json", "manifest.json", "integrity.json", "COMPLETED",
    })
    required.update(name for name in ("metrics.json", "audit_metrics.json", "pr_curves.json", "decision.json")
                    if (path / name).exists())
    if not required.issubset(integrity):
        raise ArtifactError(f"Missing integrity coverage in {path}: {sorted(required - integrity.keys())}")
    for name, entry in integrity.items():
        relative = Path(name)
        candidate = path / relative
        if relative.is_absolute() or ".." in relative.parts or not candidate.resolve().is_relative_to(path.resolve()):
            raise ArtifactError(f"Unsafe integrity path in {path}: {name}")
        digest = entry.get("sha256") if isinstance(entry, dict) else entry
        if not isinstance(digest, str) or len(digest) != 64 or _hash_file(candidate, force_fresh=force_fresh) != digest:
            raise ArtifactError(f"Corrupt artifact (SHA256 mismatch): {candidate}")
    summary = _object(path / "summary.json")
    _check_versions(summary, path / "summary.json")
    for name in ("phase", "environment", "policy", "kind"):
        if name in summary and name in config and summary[name] != config[name]:
            raise ArtifactError(f"Summary/config {name} mismatch: {path}")
    result = {
        "path": str(path), "config": config, "manifest": manifest, "summary": summary,
        "decision": _object(path / "decision.json") if (path / "decision.json").is_file() else None,
        **{name: _rows(path / f"{name}.json") if (path / f"{name}.json").is_file() else None
           for name in ("metrics", "audit_metrics", "pr_curves")},
    }
    if config["kind"] == "trajectory":
        rows = result["metrics"]
        if not rows or any(row.get("step") is None or row.get("observations") is None for row in rows):
            raise ArtifactError(f"Trajectory requires recorded step/observations evaluations: {path}")
        try:
            _ordered_rows(rows)
        except ValueError as error:
            raise ArtifactError(f"Invalid trajectory exposure in {path}: {error}") from error
    return result


def load_report(root: Path, *, force_fresh: bool = False) -> dict[str, Any]:
    """Validate all sealed units; count but never read or repair incomplete units.

    A missing root/study is an error. Progress/log files may be absent before
    orchestration's first checkpoint and are then explicitly marked unavailable.
    Per-process SHA256 reuse requires identical inode/device/size/mtime/ctime;
    final audits can bypass it with force_fresh=True. Nothing is cached on disk.
    """
    root = Path(root).resolve()
    study = _object(root / "study.json")
    _check_versions(study, root / "study.json")
    progress = _object(root / "progress.json") if (root / "progress.json").exists() else None
    lab_log = _read_json(root / "lab_log.json") if (root / "lab_log.json").exists() else None
    if lab_log is not None and not isinstance(lab_log, list):
        raise ArtifactError(f"Expected a list of decisions: {root / 'lab_log.json'}")
    units, phase_progress = [], []
    for directory in root.glob("phase*"):
        if directory.is_dir() and directory.name not in PHASE_ORDER:
            raise ArtifactError(f"Unknown phase directory: {directory}")
    for phase in PHASE_ORDER:
        directory = root / phase
        complete, incomplete = 0, 0
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if not path.is_dir():
                    continue
                if path.name == ".incomplete":
                    incomplete += sum(child.is_dir() for child in path.iterdir())
                elif path.name.startswith(".") or path.name.endswith(".incomplete"):
                    incomplete += 1
                elif not (path / "COMPLETED").is_file():
                    incomplete += 1
                else:
                    units.append(_load_unit(path, phase, study, force_fresh=force_fresh))
                    complete += 1
        phase_progress.append({"phase": phase, "completed": complete, "incomplete": incomplete})
    decisions = {}
    for unit in units:
        name = unit["summary"].get("decision", unit["config"].get("kind"))
        if name in {"phase1_selection", "optimizer_selection", "phase3_calibration"}:
            decisions[name] = unit["decision"] or unit["summary"]
    pointer = root / "decisions" / "phase3_calibration.json"
    if pointer.exists():
        frozen = _object(pointer)
        _check_versions(frozen, pointer)
        expected = study.get("study_hash", study.get("config_hash"))
        if expected is not None and frozen.get("study_hash") != expected:
            raise ArtifactError(f"Frozen calibration belongs to another study: {pointer}")
        if "phase3_calibration" in decisions and frozen != decisions["phase3_calibration"]:
            raise ArtifactError(f"Frozen calibration pointer disagrees with sealed decision: {pointer}")
        # This frozen root file controls presentation only; it supplies no outcomes.
        decisions["phase3_calibration"] = frozen
    return {"root": str(root), "study": study, "progress": progress, "lab_log": lab_log,
            "decisions": decisions,
            "units": units, "phase_progress": phase_progress,
            "incomplete": sum(row["incomplete"] for row in phase_progress)}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def metric_value(row: dict[str, Any], field: str) -> float | None:
    """Precision without a defined flag is unknown, not evidence of zero precision."""
    if field in {"nine_precision", "standardized_nine_precision"}:
        if row.get("nine_precision_defined") is not True:
            return None
    if field == "standardized_nine_precision":
        recall, fpr = _finite(row.get("nine_recall")), _finite(row.get("nine_false_positive_rate"))
        if recall is None or fpr is None:
            return None
        denominator = .1 * recall + .9 * fpr
        return .1 * recall / denominator if denominator > 0 else None
    if field == "effective_dimension_ratio":
        dimension, ess = _finite(row.get("d_eff")), _finite(row.get("n_eff"))
        return dimension / ess if dimension is not None and ess is not None and ess > 0 else None
    return _finite(row.get(field))


def mean_interval(values: Iterable[float | None], *, seed: int = BOOTSTRAP_SEED,
                  samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    """Percentile bootstrap of replica means, never of evaluation time points."""
    raw = list(values)
    valid = np.asarray([number for value in raw if (number := _finite(value)) is not None])
    result = {"mean": None, "ci95_low": None, "ci95_high": None, "n": len(valid),
              "n_total": len(raw), "missing": len(raw) - len(valid)}
    if len(valid):
        result["mean"] = float(valid.mean())
    if len(valid) >= 2:
        if samples < 2:
            raise ValueError("At least two bootstrap samples are required")
        rng = np.random.default_rng(seed)
        means = rng.choice(valid, size=(samples, len(valid)), replace=True).mean(axis=1)
        result["ci95_low"], result["ci95_high"] = map(float, np.quantile(means, [.025, .975]))
    return result


def exposure_normalized_auc(observations: Sequence[float], values: Sequence[float], *,
                            origin: float, stop: float | None = None) -> float:
    """Trapezoidal integral divided by observed exposure span, including origin.

    No origin imputation or extrapolation. An interior stop uses the declared
    piecewise-linear quadrature, not a new measured evaluation.
    """
    x, y = np.asarray(observations, dtype=float), np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("AUC needs equal one-dimensional vectors with at least two evaluations")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("AUC requires finite values and strictly increasing exposures")
    if not math.isfinite(origin) or x[0] != origin:
        raise ValueError("AUC requires an explicitly observed exposure origin")
    end = x[-1] if stop is None else float(stop)
    if not math.isfinite(end) or not origin < end <= x[-1]:
        raise ValueError("AUC stop must follow origin and cannot extrapolate")
    mask = x < end
    xs = np.append(x[mask], end)
    ys = np.append(y[mask], np.interp(end, x, y))
    return float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2) / (end - origin))


def early_exposure_summary(rows: list[dict[str, Any]], *, origin: float,
                           total_observations: float | None = None) -> list[dict[str, Any]]:
    """Fixed prefixes use the first complete evaluation at or beyond each fraction."""
    ordered = _ordered_rows(rows)
    if not ordered:
        return []
    end = ordered[-1]["observations"] if total_observations is None else total_observations
    result = []
    for fraction in (.25, .5, 1.):
        threshold = origin + fraction * (end - origin)
        endpoint = next((row for row in ordered if row["observations"] >= threshold), None)
        auc, reason = None, None
        prefix = [] if endpoint is None else [row for row in ordered if row["observations"] <= endpoint["observations"]]
        try:
            if any(metric_value(row, "current_nll") is None for row in prefix):
                raise ValueError("Missing current_nll in prefix")
            auc = exposure_normalized_auc([row["observations"] for row in prefix],
                                          [row["current_nll"] for row in prefix], origin=origin)
        except ValueError as error:
            reason = str(error)
        result.append({"fraction": fraction, "observations": None if endpoint is None else endpoint["observations"],
                       "current_nll": None if endpoint is None else metric_value(endpoint, "current_nll"),
                       "expected_nines": None if endpoint is None else endpoint.get("expected_nines"),
                       "nll_auc": auc, "missing_reason": reason})
    return result


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(_finite(row.get("observations")) is None for row in rows):
        raise ValueError("Exposure analysis requires recorded observations at every evaluation")
    ordered = sorted(rows, key=lambda row: row["observations"])
    if any(left["observations"] >= right["observations"] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Duplicate evaluation exposure in trajectory")
    return ordered


def target_attainment(rows: list[dict[str, Any]], target: float, *, origin: float,
                      metric: str = "fixed_nll") -> dict[str, Any]:
    """Count through the second consecutive passing evaluation; retain censoring.

    Missing evaluations break a consecutive pair. Censored resource values are
    follow-up endpoints, NOT estimates of the unobserved time to attainment.
    """
    if not math.isfinite(target):
        raise ValueError("Target must be finite")
    ordered = [row for row in _ordered_rows(rows) if row["observations"] >= origin]
    values = [metric_value(row, metric) for row in ordered]
    event = next((index for index in range(1, len(values))
                  if values[index - 1] is not None and values[index] is not None
                  and values[index - 1] <= target and values[index] <= target), None)
    reached = event is not None
    endpoint = ordered[event] if reached else (ordered[-1] if ordered else {})
    baseline = (values[0] <= target if ordered and ordered[0]["observations"] == origin
                and values[0] is not None else None)
    available = sum(value is not None for value in values)
    return {"target": target, "metric": metric, "reached": reached,
            "censored": not reached, "status": "attained" if reached else (
                "right_censored" if available == len(values) and available else "missing_metrics"),
            "baseline_satisfied": baseline,
            "lost_after_attainment": any(value is not None and value > target for value in values[event + 1:])
            if reached else False,
            "missing_evaluations": len(values) - available,
            "observations": endpoint.get("observations"),
            **{name: endpoint.get(name) for name in ("expected_nines", "realized_nines", "online_seconds", "charged_seconds")}}


def unit_metadata(unit: dict[str, Any]) -> dict[str, Any]:
    """Recorded design coordinates for grouping; no inference from outcomes."""
    config, summary = unit["config"], unit["summary"]
    detail = config.get("detail", {})
    metadata = {**detail, **config, **summary}
    metadata["replica_index"] = summary.get("replica_index", config.get("index"))
    metadata["schedule"] = metadata.get("schedule", metadata.get("schedule_kind"))
    # Keep distinct evidence/architecture generations out of the same mean.
    policy = detail.get("policy", {})
    if isinstance(policy, dict):
        for name in ("c", "half_life", "cadence", "rank", "score_samples"):
            if name in policy:
                metadata[name] = policy[name]
    evidence = {key: metadata.get(key) for key in (
        "evidence_protocol", "c", "half_life", "episode_length", "probe_interval", "rank", "score_samples",
        "driver_generation", "architecture", "protocol_hash", "width", "retention",
        "episode_arrivals", "cadence", "profile_hash",
    ) if metadata.get(key) is not None}
    metadata["evidence_protocol"] = json.dumps(evidence, sort_keys=True) if evidence else None
    metadata["protocol_variant"] = hashlib.sha256(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]
    return metadata


def _key(row: dict[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    # Rotation's recorded nine count is stream composition, not a treatment scale.
    rotation = row.get("family") == "rotation" or str(row.get("environment", "")).startswith("rotation")
    return tuple(json.dumps(None if field == "expected_nines" and rotation else row.get(field), sort_keys=True)
                 for field in fields)


def trajectory_groups(report: dict[str, Any], phase: str | None = None) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for unit in report["units"]:
        if unit["config"].get("kind") == "trajectory" and unit["metrics"] is not None and (
                phase is None or unit["config"]["phase"] == phase):
            groups[_key(unit_metadata(unit), GROUP_FIELDS)].append(unit)
    result = [groups[key] for key in sorted(groups)]
    fixed = [unit for members in result for unit in members if _fixed_policy(unit)]
    # Fixed-depth controls canonically store c=1/cadence=1: those gate-only
    # settings do not change their learning process. Reuse, never re-average,
    # these same sealed controls in each otherwise compatible candidate panel.
    for members in result:
        if not any(_active_gate(unit) for unit in members):
            continue
        base = _comparison_key(members[0])
        seen = {unit["path"] for unit in members}
        members.extend(unit for unit in fixed if unit["path"] not in seen and _comparison_key(unit) == base)
    return result


def _fixed_policy(unit: dict[str, Any]) -> bool:
    policy = unit["config"].get("detail", {}).get("policy", {})
    return isinstance(policy, dict) and "gate" in policy and policy["gate"] is None


def _active_gate(unit: dict[str, Any]) -> bool:
    policy = unit["config"].get("detail", {}).get("policy", {})
    return isinstance(policy, dict) and policy.get("gate") is not None


def _comparison_key(unit: dict[str, Any]) -> tuple[str, ...]:
    metadata = unit_metadata(unit)
    evidence = json.loads(metadata["evidence_protocol"] or "{}")
    for field in ("c", "cadence"):
        evidence.pop(field, None)
    metadata["protocol_variant"] = evidence
    return _key(metadata, GROUP_FIELDS)


def paired_effects(rows: list[dict[str, Any]], treatment: str, control: str, metric: str,
                   *, group_fields: Sequence[str] = GROUP_FIELDS, seed: int = BOOTSTRAP_SEED,
                   samples: int = BOOTSTRAP_SAMPLES) -> list[dict[str, Any]]:
    """Treatment minus control; pair replica IDs within the exact design cell."""
    groups: dict[tuple[str, ...], dict[str, dict[Any, dict[str, Any]]]] = defaultdict(
        lambda: {treatment: {}, control: {}})
    if treatment == control:
        raise ValueError("Treatment and control must differ")
    for row in rows:
        policy, replica = row.get("policy"), row.get("replica_index")
        if policy not in {treatment, control}:
            continue
        if replica is None:
            raise ValueError("Paired effects require a recorded replica_index")
        arm = groups[_key(row, group_fields)][policy]
        if replica in arm:
            raise ValueError(f"Duplicate replica {replica} for policy {policy}; refine grouping")
        arm[replica] = row
    result = []
    for key, arms in sorted(groups.items()):
        left, right = arms[treatment], arms[control]
        paired = sorted(left.keys() & right.keys(), key=str)
        differences = [None if (a := metric_value(left[index], metric)) is None
                       or (b := metric_value(right[index], metric)) is None else a - b for index in paired]
        result.append({**dict(zip(group_fields, map(json.loads, key))),
                       "treatment": treatment, "control": control, "metric": metric,
                       "effect_convention": "treatment minus control",
                       **mean_interval(differences, seed=seed, samples=samples),
                       "paired_replicas": paired, "differences": differences,
                       "unpaired_treatment": len(left.keys() - right.keys()),
                       "unpaired_control": len(right.keys() - left.keys())})
    return result


def pointwise_summary(units: list[dict[str, Any]], field: str, *, x_field: str = "observations",
                       seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:
    """Pointwise bands over replicas aligned by scheduled step, never interpolation.

    For stochastic/time x axes, x is the replica mean at that step. The report
    labels these axes accordingly; missing replica points remain in n_total.
    """
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_policy[unit_metadata(unit).get("policy")].append(unit)
    result = []
    for policy, members in sorted(by_policy.items(), key=lambda item: str(item[0])):
        replicas = [unit_metadata(unit).get("replica_index") for unit in members]
        if len(set(replicas)) != len(replicas):
            raise ValueError("Duplicate replica trajectory; design cells must not be pooled")
        indexed = []
        for unit in members:
            index = {}
            for row in unit["metrics"]:
                step = row.get("step", row.get("observations"))
                if step is None or step in index:
                    raise ValueError("Missing or duplicate scheduled evaluation step")
                index[step] = row
            indexed.append(index)
        steps = sorted(set().union(*(index.keys() for index in indexed)))
        for step in steps:
            rows = [index.get(step, {}) for index in indexed]
            xs = [metric_value(row, x_field) for row in rows]
            values = [metric_value(row, field) if x is not None else None for row, x in zip(rows, xs)]
            stats = mean_interval(values, seed=seed)
            valid_x = [x for x in xs if x is not None]
            result.append({"policy": policy, "step": step, "x": float(np.mean(valid_x)) if valid_x else None,
                           "x_min": min(valid_x) if valid_x else None,
                           "x_max": max(valid_x) if valid_x else None,
                           "field": field, **stats,
                           "defined_fraction": stats["n"] / len(rows) if rows else None})
    return result


def target_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Include every replica, including censored and missing outcomes, in counts."""
    return {"replicas": len(outcomes), "attained": sum(row["reached"] for row in outcomes),
            "censored": sum(row["censored"] for row in outcomes),
            "baseline_satisfied": sum(row["baseline_satisfied"] is True for row in outcomes),
            "baseline_unknown": sum(row["baseline_satisfied"] is None for row in outcomes),
            "later_loss": sum(row["lost_after_attainment"] for row in outcomes),
            "missing_metrics": sum(row["status"] == "missing_metrics" for row in outcomes),
            "attainment_fraction": sum(row["reached"] for row in outcomes) / len(outcomes) if outcomes else None}


def pr_curve_summary(units: list[dict[str, Any]], *, points: int = 101) -> list[dict[str, Any]]:
    """Final standardized PR on a common recall grid, using scalar plot samples.

    Linear interpolation is for visualization only, within each replica's
    recorded, defined recall support. At tied recall retain the last point in
    the descending-threshold sweep (the lowest threshold). No extrapolation, precision envelope,
    production-threshold selection, or AP estimate from sampled points.
    """
    if points < 2:
        raise ValueError("A PR plotting grid requires at least two points")
    by_policy = defaultdict(list)
    for unit in units:
        by_policy[unit_metadata(unit).get("policy")].append(unit)
    result = []
    for policy, members in sorted(by_policy.items(), key=lambda item: str(item[0])):
        curves = []
        for unit in members:
            rows = unit.get("pr_curves") or []
            last_step = max((row.get("step", 0) for row in rows), default=0)
            valid = [row for row in rows if row.get("step", 0) == last_step
                     and row.get("precision_defined") is True
                     and all(_finite(row.get(field)) is not None for field in ("threshold", "precision", "recall"))]
            valid.sort(key=lambda row: row["threshold"], reverse=True)
            thresholds = [row["threshold"] for row in valid]
            if len(set(thresholds)) != len(thresholds):
                raise ArtifactError("Duplicate thresholds in final scalar PR curve")
            by_recall = {row["recall"]: row["precision"] for row in valid}
            curves.append(sorted(by_recall.items()))
        for recall in np.linspace(0., 1., points):
            precisions = []
            for curve in curves:
                if not curve or not curve[0][0] <= recall <= curve[-1][0]:
                    precisions.append(None)
                    continue
                x, y = zip(*curve)
                precisions.append(float(np.interp(recall, x, y)))
            stats = mean_interval(precisions)
            result.append({"policy": policy, "recall": float(recall), **stats,
                           "defined_fraction": stats["n"] / len(members)})
    return result
