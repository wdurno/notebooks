"""Immutable orchestration for Plan 2's nested low-data screen."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .initialization import (
    ReplicaBundleError,
    derive_replica_bundle,
    load_replica_bundle_for_config,
    replica_bundle_id,
)
from .phase9 import build_phase9_bundle, load_phase9_spec, parse_replica_indices

PLAN2_SPEC_SCHEMA_VERSION = 1
PLAN2_BUNDLE_SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Plan2Error(RuntimeError):
    """Raised when a Plan 2 specification or artifact is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan2Error(f"could not read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise Plan2Error(
            f"invalid {context}: missing={missing}, unknown={unknown}"
        )


def _name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise Plan2Error(
            f"{context} must use lowercase letters, digits, and hyphens"
        )
    return value


def _deep_override(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key not in result:
            raise Plan2Error(f"{context} refers to unknown key {key!r}")
        if isinstance(value, Mapping):
            if not isinstance(result[key], Mapping):
                raise Plan2Error(f"{context}.{key} cannot be an object")
            result[key] = _deep_override(
                result[key], value, context=f"{context}.{key}"
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclasses.dataclass(frozen=True)
class Plan2Condition:
    name: str
    kind: str
    policy: str
    overrides: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Plan2Spec:
    path: Path
    name: str
    phase9_spec: str
    master_profile: str
    master_cell: str
    bundle_root: str
    replica_root: str
    run_cache_root: str
    default_samples_per_step: tuple[int, ...]
    default_replica_start: int
    default_replica_count: int
    fixed_seconds_per_run: float
    seconds_per_optimizer_observation: float
    estimated_run_bytes: int
    base_overrides: Mapping[str, Any]
    conditions: tuple[Plan2Condition, ...]
    raw: Mapping[str, Any]

    @property
    def content_hash(self) -> str:
        return _content_hash(self.raw)


@dataclasses.dataclass(frozen=True)
class Plan2Bundle:
    path: Path
    manifest: Mapping[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["entries"])


def parse_samples_per_step(value: str | None, defaults: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(defaults)
    try:
        samples = tuple(sorted({int(token.strip()) for token in value.split(",")}))
    except ValueError as exc:
        raise Plan2Error("samples-per-step must be comma-separated integers") from exc
    if not samples or samples[0] < 1:
        raise Plan2Error("samples-per-step values must be positive")
    return samples


def load_plan2_spec(path: str | Path) -> Plan2Spec:
    spec_path = Path(path)
    raw = _read_json(spec_path)
    if not isinstance(raw, Mapping):
        raise Plan2Error("Plan 2 specification must be an object")
    _require_keys(
        raw,
        {
            "schema_version",
            "name",
            "phase9_spec",
            "master_profile",
            "master_cell",
            "bundle_root",
            "replica_root",
            "run_cache_root",
            "default_samples_per_step",
            "default_replica_start",
            "default_replica_count",
            "cost_model",
            "base_overrides",
            "conditions",
        },
        "Plan 2 specification",
    )
    if raw["schema_version"] != PLAN2_SPEC_SCHEMA_VERSION:
        raise Plan2Error("unsupported Plan 2 specification schema")
    for field in (
        "phase9_spec",
        "master_profile",
        "master_cell",
        "bundle_root",
        "replica_root",
        "run_cache_root",
    ):
        if not isinstance(raw[field], str) or not raw[field]:
            raise Plan2Error(f"{field} must be a nonempty string")
    samples = raw["default_samples_per_step"]
    if (
        not isinstance(samples, list)
        or not samples
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in samples
        )
        or len(samples) != len(set(samples))
    ):
        raise Plan2Error("default_samples_per_step must be unique positive integers")
    start = raw["default_replica_start"]
    count = raw["default_replica_count"]
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise Plan2Error("default_replica_start must be nonnegative")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise Plan2Error("default_replica_count must be positive")
    cost = raw["cost_model"]
    if not isinstance(cost, Mapping):
        raise Plan2Error("cost_model must be an object")
    _require_keys(
        cost,
        {
            "fixed_seconds_per_run",
            "seconds_per_optimizer_observation",
            "estimated_run_bytes",
        },
        "cost_model",
    )
    for field in ("fixed_seconds_per_run", "seconds_per_optimizer_observation"):
        value = cost[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise Plan2Error(f"cost_model.{field} must be finite and nonnegative")
    estimated_bytes = cost["estimated_run_bytes"]
    if (
        not isinstance(estimated_bytes, int)
        or isinstance(estimated_bytes, bool)
        or estimated_bytes < 0
    ):
        raise Plan2Error("cost_model.estimated_run_bytes must be nonnegative")
    if not isinstance(raw["base_overrides"], Mapping):
        raise Plan2Error("base_overrides must be an object")

    conditions_raw = raw["conditions"]
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise Plan2Error("conditions must be a nonempty list")
    conditions = []
    names = set()
    for index, value in enumerate(conditions_raw):
        if not isinstance(value, Mapping):
            raise Plan2Error(f"conditions[{index}] must be an object")
        _require_keys(
            value, {"name", "kind", "policy", "overrides"}, f"conditions[{index}]"
        )
        condition_name = _name(value["name"], f"conditions[{index}].name")
        if condition_name in names:
            raise Plan2Error(f"duplicate condition: {condition_name}")
        names.add(condition_name)
        if value["kind"] not in {"control", "treatment"}:
            raise Plan2Error("condition kind must be control or treatment")
        if not isinstance(value["policy"], str) or not value["policy"]:
            raise Plan2Error("condition policy must be nonempty")
        if not isinstance(value["overrides"], Mapping):
            raise Plan2Error("condition overrides must be an object")
        conditions.append(
            Plan2Condition(
                name=condition_name,
                kind=value["kind"],
                policy=value["policy"],
                overrides=value["overrides"],
            )
        )
    controls = [condition for condition in conditions if condition.kind == "control"]
    if len(controls) > 1:
        raise Plan2Error("Plan 2 permits at most one primary control condition")

    return Plan2Spec(
        path=spec_path,
        name=_name(raw["name"], "name"),
        phase9_spec=raw["phase9_spec"],
        master_profile=raw["master_profile"],
        master_cell=raw["master_cell"],
        bundle_root=raw["bundle_root"],
        replica_root=raw["replica_root"],
        run_cache_root=raw["run_cache_root"],
        default_samples_per_step=tuple(sorted(samples)),
        default_replica_start=start,
        default_replica_count=count,
        fixed_seconds_per_run=float(cost["fixed_seconds_per_run"]),
        seconds_per_optimizer_observation=float(
            cost["seconds_per_optimizer_observation"]
        ),
        estimated_run_bytes=estimated_bytes,
        base_overrides=raw["base_overrides"],
        conditions=tuple(conditions),
        raw=raw,
    )


def _source_configs(
    spec: Plan2Spec,
    repo_root: Path,
    replica_indices: Sequence[int],
) -> dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    phase9_spec = load_phase9_spec(repo_root / spec.phase9_spec)
    manifest, configs = build_phase9_bundle(
        phase9_spec,
        repo_root,
        profile_names=(spec.master_profile,),
        replica_indices=replica_indices,
    )
    result = {}
    for replica_index in replica_indices:
        matches = [
            entry
            for entry in manifest["entries"]
            if entry["replica_index"] == replica_index
            and entry["profile"] == spec.master_profile
            and entry["cell"] == spec.master_cell
            and entry["kind"] == "treatment"
        ]
        if len(matches) != 1:
            raise Plan2Error(
                f"master Phase 9 cell is not unique for replica {replica_index}"
            )
        entry = matches[0]
        result[replica_index] = (entry, configs[entry["entry_id"]])
    return result


def build_plan2_bundle(
    spec: Plan2Spec,
    repo_root: str | Path,
    *,
    samples_per_step: Sequence[int] | None = None,
    replica_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    samples = tuple(samples_per_step or spec.default_samples_per_step)
    if (
        not samples
        or len(samples) != len(set(samples))
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in samples
        )
    ):
        raise Plan2Error("samples_per_step must be unique positive integers")
    replicas = tuple(
        replica_indices
        or range(
            spec.default_replica_start,
            spec.default_replica_start + spec.default_replica_count,
        )
    )
    if not replicas or len(replicas) != len(set(replicas)) or min(replicas) < 0:
        raise Plan2Error("replica indices must be unique and nonnegative")
    sources = _source_configs(spec, root, replicas)

    entries = []
    configs: dict[str, dict[str, Any]] = {}
    control_name = next(
        (
            condition.name
            for condition in spec.conditions
            if condition.kind == "control"
        ),
        None,
    )
    for replica_index in replicas:
        source_entry, source_mapping = sources[replica_index]
        source_config = ExperimentConfig.from_mapping(source_mapping)
        master_size = source_config.data.samples_per_step
        if max(samples) > master_size:
            raise Plan2Error(
                f"requested m exceeds replica {replica_index}'s master m={master_size}"
            )
        reference_artifact = str(
            Path(source_config.cache_root)
            / source_entry["run_id"]
            / "phase8_reference_optimum.pt"
        )
        for sample_size in sorted(samples):
            cell_entries = []
            for condition in spec.conditions:
                mapping = _deep_override(
                    source_mapping,
                    spec.base_overrides,
                    context=f"{spec.name}.base_overrides",
                )
                mapping = _deep_override(
                    mapping,
                    condition.overrides,
                    context=f"{spec.name}.{condition.name}",
                )
                mapping["schema_version"] = 12
                mapping["artifact_schema_version"] = 4
                mapping["metric_schema_version"] = 8
                mapping["experiment"] = (
                    f"mnist_lfu_{spec.name}_m{sample_size:03d}_{condition.name}"
                )
                mapping["data"]["samples_per_step"] = sample_size
                mapping["cache_root"] = spec.run_cache_root
                mapping["controller"]["policy"] = condition.policy
                mapping["controller"]["reference_optimum_artifact"] = (
                    reference_artifact
                )
                config = ExperimentConfig.from_mapping(mapping)
                entry_id = (
                    f"r{replica_index:04d}:m{sample_size:03d}:{condition.name}"
                )
                entry = {
                    "entry_id": entry_id,
                    "kind": condition.kind,
                    "condition": condition.name,
                    "replica_index": replica_index,
                    "replica_id": config.replica_id,
                    "replica_seed": config.replica_seed,
                    "samples_per_step": sample_size,
                    "master_samples_per_step": master_size,
                    "master_replica_bundle_id": source_entry["replica_bundle_id"],
                    "replica_bundle_id": replica_bundle_id(config),
                    "stream_relation": (
                        "master"
                        if sample_size == master_size
                        else "first_m_ordered_observations_at_each_p_step"
                    ),
                    "reference_source_run_id": source_entry["run_id"],
                    "reference_optimum_artifact": reference_artifact,
                    "run_id": config.run_id,
                    "config_hash": config.config_hash,
                    "config_file": str(
                        Path("configs")
                        / config.replica_id
                        / f"m{sample_size:03d}__{condition.name}.json"
                    ),
                    "cache_root": config.cache_root,
                    "control_run_id": None,
                    "estimated_trajectory_seconds": (
                        spec.fixed_seconds_per_run
                        + spec.seconds_per_optimizer_observation
                        * (config.data.num_p_steps - 1)
                        * sample_size
                    ),
                    "estimated_run_bytes": spec.estimated_run_bytes,
                }
                entries.append(entry)
                cell_entries.append(entry)
                configs[entry_id] = mapping
            if control_name is not None:
                control_run_id = next(
                    entry["run_id"]
                    for entry in cell_entries
                    if entry["condition"] == control_name
                )
                for entry in cell_entries:
                    if entry["kind"] == "treatment":
                        entry["control_run_id"] = control_run_id

    entries.sort(
        key=lambda row: (
            row["replica_index"],
            row["samples_per_step"],
            row["kind"] != "control",
            row["condition"],
        )
    )
    selection = {
        "spec_hash": spec.content_hash,
        "samples_per_step": list(sorted(samples)),
        "replica_indices": list(sorted(replicas)),
        "generated_config_schema_version": 12,
        "generated_metric_schema_version": 8,
        "generated_artifact_schema_version": 4,
    }
    digest = _content_hash(selection)[:12]
    replica_label = (
        f"r{replicas[0]:04d}"
        if len(replicas) == 1
        else f"r{min(replicas):04d}-r{max(replicas):04d}"
    )
    manifest = {
        "bundle_schema_version": PLAN2_BUNDLE_SCHEMA_VERSION,
        "bundle_id": f"{spec.name}__{replica_label}__{digest}",
        "spec_name": spec.name,
        "spec_hash": spec.content_hash,
        "selection": selection,
        "entry_count": len(entries),
        "replica_count": len(replicas),
        "samples_per_step_count": len(samples),
        "condition_count": len(spec.conditions),
        "derived_bundle_count": len(
            {
                row["replica_bundle_id"]
                for row in entries
                if row["samples_per_step"] < row["master_samples_per_step"]
            }
        ),
        "new_initialization_fit_count": 0,
        "estimated_seconds": sum(
            row["estimated_trajectory_seconds"] for row in entries
        ),
        "estimated_bytes": sum(row["estimated_run_bytes"] for row in entries),
        "cost_model": {
            "fixed_seconds_per_run": spec.fixed_seconds_per_run,
            "seconds_per_optimizer_observation": (
                spec.seconds_per_optimizer_observation
            ),
            "basis": "fixed_overhead_plus_optimizer_consumed_observations",
        },
        "replica_root": spec.replica_root,
        "entries": entries,
    }
    return manifest, configs


def prepare_plan2_bundle(
    spec: Plan2Spec,
    repo_root: str | Path,
    *,
    samples_per_step: Sequence[int] | None = None,
    replica_indices: Sequence[int] | None = None,
) -> Plan2Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan2_bundle(
        spec,
        root,
        samples_per_step=samples_per_step,
        replica_indices=replica_indices,
    )
    unique_designs = {}
    for entry in manifest["entries"]:
        unique_designs.setdefault(entry["replica_bundle_id"], entry)
    for entry in unique_designs.values():
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        parent = root / spec.replica_root / entry["master_replica_bundle_id"]
        if entry["stream_relation"] == "master":
            load_replica_bundle_for_config(
                root / spec.replica_root, config, device="cpu"
            )
            continue
        try:
            derive_replica_bundle(
                parent,
                root / spec.replica_root,
                config,
                repo_root=root,
            )
        except ReplicaBundleError as exc:
            raise Plan2Error(
                f"could not derive {entry['replica_bundle_id']}: {exc}"
            ) from exc

    destination = root / spec.bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan2_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan2Error(
                f"existing bundle does not match generated content: {destination}"
            )
        return loaded
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{manifest['bundle_id']}.", dir=destination.parent)
    )
    try:
        for entry in manifest["entries"]:
            _write_json(
                temporary / entry["config_file"], configs[entry["entry_id"]]
            )
        _write_json(temporary / "bundle.json", manifest)
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan2Bundle(destination, manifest)


def load_plan2_bundle(path: str | Path) -> Plan2Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan2Error(f"Plan 2 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if not isinstance(manifest, Mapping):
        raise Plan2Error("Plan 2 bundle manifest must be an object")
    if manifest.get("bundle_schema_version") != PLAN2_BUNDLE_SCHEMA_VERSION:
        raise Plan2Error("unsupported Plan 2 bundle schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get("entry_count"):
        raise Plan2Error("Plan 2 bundle entry count is invalid")
    run_ids = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise Plan2Error("Plan 2 bundle entry must be an object")
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.run_id != entry["run_id"]:
            raise Plan2Error(f"run ID mismatch in {config_path}")
        if config.config_hash != entry["config_hash"]:
            raise Plan2Error(f"config hash mismatch in {config_path}")
        if config.run_id in run_ids:
            raise Plan2Error(f"duplicate run ID in bundle: {config.run_id}")
        run_ids.add(config.run_id)
    for entry in entries:
        control = entry.get("control_run_id")
        if control is not None and control not in run_ids:
            raise Plan2Error(f"missing paired control dependency: {control}")
    return Plan2Bundle(bundle_path, manifest)


def discover_plan2_bundles(root: str | Path) -> tuple[Plan2Bundle, ...]:
    bundle_root = Path(root)
    if not bundle_root.is_dir():
        return ()
    return tuple(
        load_plan2_bundle(path)
        for path in sorted(bundle_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    )


def _artifact_state(final_path: Path, incomplete_path: Path) -> str:
    if (final_path / "COMPLETED").is_file():
        return "completed"
    if final_path.exists():
        return "invalid"
    if incomplete_path.exists():
        return "incomplete"
    return "missing"


def plan2_status_rows(
    bundle: Plan2Bundle, repo_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in bundle.entries:
        config_path = bundle.path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        run_root = root / config.cache_root
        replica_path = root / bundle.manifest["replica_root"]
        replica_path = replica_path / entry["replica_bundle_id"]
        reference_path = root / entry["reference_optimum_artifact"]
        rows.append(
            {
                **entry,
                "bundle_id": bundle.bundle_id,
                "bundle_path": str(bundle.path),
                "config_path": str(config_path),
                "run_state": _artifact_state(
                    run_root / config.run_id,
                    run_root / ".incomplete" / config.run_id,
                ),
                "run_path": str(run_root / config.run_id),
                "derived_bundle_state": (
                    "completed"
                    if (replica_path / "COMPLETED").is_file()
                    else "invalid" if replica_path.exists() else "missing"
                ),
                "replica_path": str(replica_path),
                "reference_state": (
                    "completed"
                    if reference_path.is_file()
                    and (reference_path.parent / "COMPLETED").is_file()
                    else "missing"
                ),
            }
        )
    return rows


__all__ = [
    "Plan2Bundle",
    "Plan2Error",
    "Plan2Spec",
    "build_plan2_bundle",
    "discover_plan2_bundles",
    "load_plan2_bundle",
    "load_plan2_spec",
    "parse_replica_indices",
    "parse_samples_per_step",
    "plan2_status_rows",
    "prepare_plan2_bundle",
]
