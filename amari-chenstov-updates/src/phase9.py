"""Immutable Phase 9 experiment-bundle generation and status inspection."""

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
from .initialization import replica_bundle_id
from .seeding import derive_component_seed

PHASE9_SPEC_SCHEMA_VERSION = 2
PHASE9_BUNDLE_SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Phase9Error(RuntimeError):
    """Raised when a Phase 9 specification or bundle is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase9Error(f"could not read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise Phase9Error(
            f"invalid {context}: missing={missing}, unknown={unknown}"
        )


def _validate_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise Phase9Error(
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
            raise Phase9Error(f"{context} refers to unknown key {key!r}")
        current = result[key]
        if isinstance(value, Mapping):
            if not isinstance(current, Mapping):
                raise Phase9Error(f"{context}.{key} cannot be an object")
            result[key] = _deep_override(
                current,
                value,
                context=f"{context}.{key}",
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclasses.dataclass(frozen=True)
class Phase9Cell:
    name: str
    overrides: Mapping[str, Any]
    policy: str | None = None
    kind: str = "treatment"
    control_ref: str | None = None


@dataclasses.dataclass(frozen=True)
class Phase9Profile:
    name: str
    template: str
    control_scope: str
    estimated_trajectory_seconds: float
    estimated_run_bytes: int
    overrides: Mapping[str, Any]
    cells: tuple[Phase9Cell, ...]


@dataclasses.dataclass(frozen=True)
class Phase9Spec:
    path: Path
    name: str
    base_replica_seed: int
    default_replica_start: int
    default_replica_count: int
    fixed_pi: float
    bundle_root: str
    default_profiles: tuple[str, ...]
    reference_seconds_per_p: float
    initialization_seconds: float
    profiles: tuple[Phase9Profile, ...]
    raw: Mapping[str, Any]

    @property
    def content_hash(self) -> str:
        return _content_hash(self.raw)

    def profile(self, name: str) -> Phase9Profile:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise Phase9Error(f"unknown Phase 9 profile: {name}")


@dataclasses.dataclass(frozen=True)
class Phase9Bundle:
    path: Path
    manifest: Mapping[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["entries"])


def load_phase9_spec(path: str | Path) -> Phase9Spec:
    spec_path = Path(path)
    raw = _read_json(spec_path)
    if not isinstance(raw, Mapping):
        raise Phase9Error("Phase 9 specification must be an object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "name",
            "base_replica_seed",
            "default_replica_start",
            "default_replica_count",
            "fixed_pi",
            "bundle_root",
            "default_profiles",
            "cost_model",
            "profiles",
        },
        "Phase 9 specification",
    )
    if raw["schema_version"] != PHASE9_SPEC_SCHEMA_VERSION:
        raise Phase9Error(
            f"unsupported Phase 9 spec schema: {raw['schema_version']}"
        )
    name = _validate_name(raw["name"], "specification name")
    base_seed = raw["base_replica_seed"]
    if (
        not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
        or not 0 <= base_seed < 2**63
    ):
        raise Phase9Error("base_replica_seed must be an integer in [0, 2**63)")
    start = raw["default_replica_start"]
    count = raw["default_replica_count"]
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise Phase9Error("default_replica_start must be a nonnegative integer")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise Phase9Error("default_replica_count must be a positive integer")
    fixed_pi = raw["fixed_pi"]
    if (
        not isinstance(fixed_pi, (int, float))
        or isinstance(fixed_pi, bool)
        or not math.isfinite(float(fixed_pi))
        or not 0.0 < float(fixed_pi) < 1.0
    ):
        raise Phase9Error("fixed_pi must be finite and in (0, 1)")
    if not isinstance(raw["bundle_root"], str) or not raw["bundle_root"]:
        raise Phase9Error("bundle_root must be a nonempty path")

    cost_model = raw["cost_model"]
    if not isinstance(cost_model, Mapping):
        raise Phase9Error("cost_model must be an object")
    _require_exact_keys(
        cost_model,
        {"reference_seconds_per_p", "initialization_seconds"},
        "cost_model",
    )
    for key in ("reference_seconds_per_p", "initialization_seconds"):
        value = cost_model[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise Phase9Error(f"cost_model.{key} must be finite and nonnegative")

    profiles_raw = raw["profiles"]
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise Phase9Error("profiles must be a nonempty list")
    profiles = []
    profile_names = set()
    for index, profile_raw in enumerate(profiles_raw):
        context = f"profiles[{index}]"
        if not isinstance(profile_raw, Mapping):
            raise Phase9Error(f"{context} must be an object")
        _require_exact_keys(
            profile_raw,
            {
                "name",
                "template",
                "control_scope",
                "estimated_trajectory_seconds",
                "estimated_run_bytes",
                "overrides",
                "cells",
            },
            context,
        )
        profile_name = _validate_name(profile_raw["name"], f"{context}.name")
        if profile_name in profile_names:
            raise Phase9Error(f"duplicate profile name: {profile_name}")
        profile_names.add(profile_name)
        if profile_raw["control_scope"] not in {
            "profile",
            "cell",
            "explicit",
        }:
            raise Phase9Error(
                f"{context}.control_scope must be profile, cell, or explicit"
            )
        if not isinstance(profile_raw["template"], str):
            raise Phase9Error(f"{context}.template must be a path string")
        if not isinstance(profile_raw["overrides"], Mapping):
            raise Phase9Error(f"{context}.overrides must be an object")
        cells_raw = profile_raw["cells"]
        if not isinstance(cells_raw, list) or not cells_raw:
            raise Phase9Error(f"{context}.cells must be a nonempty list")
        cells = []
        cell_names = set()
        for cell_index, cell_raw in enumerate(cells_raw):
            cell_context = f"{context}.cells[{cell_index}]"
            if not isinstance(cell_raw, Mapping):
                raise Phase9Error(f"{cell_context} must be an object")
            explicit = profile_raw["control_scope"] == "explicit"
            _require_exact_keys(
                cell_raw,
                (
                    {"name", "overrides", "policy", "kind", "control_ref"}
                    if explicit
                    else {"name", "overrides"}
                ),
                cell_context,
            )
            cell_name = _validate_name(cell_raw["name"], f"{cell_context}.name")
            if cell_name in cell_names:
                raise Phase9Error(
                    f"duplicate cell name in {profile_name}: {cell_name}"
                )
            cell_names.add(cell_name)
            if not isinstance(cell_raw["overrides"], Mapping):
                raise Phase9Error(f"{cell_context}.overrides must be an object")
            if explicit:
                policy = cell_raw["policy"]
                kind = cell_raw["kind"]
                control_ref = cell_raw["control_ref"]
                if policy not in {"fixed_unified", "optimal_plugin"}:
                    raise Phase9Error(
                        f"{cell_context}.policy must be fixed_unified or "
                        "optimal_plugin"
                    )
                if kind not in {"control", "treatment"}:
                    raise Phase9Error(
                        f"{cell_context}.kind must be control or treatment"
                    )
                if control_ref is not None:
                    if not isinstance(control_ref, str) or control_ref.count(":") != 1:
                        raise Phase9Error(
                            f"{cell_context}.control_ref must be null or profile:cell"
                        )
                    ref_profile, ref_cell = control_ref.split(":", 1)
                    _validate_name(ref_profile, f"{cell_context}.control_ref profile")
                    _validate_name(ref_cell, f"{cell_context}.control_ref cell")
                if kind == "control" and control_ref is not None:
                    raise Phase9Error(
                        f"{cell_context} control cells cannot name a control_ref"
                    )
            else:
                policy = None
                kind = "treatment"
                control_ref = None
            cells.append(
                Phase9Cell(
                    cell_name,
                    dict(cell_raw["overrides"]),
                    policy=policy,
                    kind=kind,
                    control_ref=control_ref,
                )
            )
        for cost_name in (
            "estimated_trajectory_seconds",
            "estimated_run_bytes",
        ):
            value = profile_raw[cost_name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise Phase9Error(f"{context}.{cost_name} must be nonnegative")
        profiles.append(
            Phase9Profile(
                name=profile_name,
                template=profile_raw["template"],
                control_scope=profile_raw["control_scope"],
                estimated_trajectory_seconds=float(
                    profile_raw["estimated_trajectory_seconds"]
                ),
                estimated_run_bytes=int(profile_raw["estimated_run_bytes"]),
                overrides=dict(profile_raw["overrides"]),
                cells=tuple(cells),
            )
        )

    default_profiles = raw["default_profiles"]
    if (
        not isinstance(default_profiles, list)
        or not default_profiles
        or any(name not in profile_names for name in default_profiles)
        or len(default_profiles) != len(set(default_profiles))
    ):
        raise Phase9Error("default_profiles must name unique defined profiles")
    return Phase9Spec(
        path=spec_path,
        name=name,
        base_replica_seed=base_seed,
        default_replica_start=start,
        default_replica_count=count,
        fixed_pi=float(fixed_pi),
        bundle_root=raw["bundle_root"],
        default_profiles=tuple(default_profiles),
        reference_seconds_per_p=float(cost_model["reference_seconds_per_p"]),
        initialization_seconds=float(cost_model["initialization_seconds"]),
        profiles=tuple(profiles),
        raw=raw,
    )


def parse_replica_indices(
    value: str | None,
    *,
    default_start: int,
    default_count: int,
) -> tuple[int, ...]:
    if value is None:
        return tuple(range(default_start, default_start + default_count))
    indices = set()
    try:
        for part in value.split(","):
            token = part.strip()
            if not token:
                raise ValueError
            if "-" in token:
                left, right = token.split("-", 1)
                start, stop = int(left), int(right)
                if start > stop:
                    raise ValueError
                indices.update(range(start, stop + 1))
            else:
                indices.add(int(token))
    except ValueError as exc:
        raise Phase9Error(
            "replicas must be comma-separated nonnegative indices or ranges"
        ) from exc
    if not indices or min(indices) < 0:
        raise Phase9Error("replica indices must be nonnegative")
    return tuple(sorted(indices))


def _replica_seed(base_seed: int, replica_index: int) -> int:
    return derive_component_seed(base_seed, f"phase9_replica:{replica_index}")


def _experiment_name(profile: str, cell: str, policy: str) -> str:
    return f"mnist_lfu_phase9_{profile}_{cell}_{policy}"


def _factors(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "pi_min": config.controller.pi_min,
        "pi_max": config.controller.pi_max,
        "trend_half_life_p": config.controller.trend_half_life_p,
        "samples_per_step": config.data.samples_per_step,
        "num_p_steps": config.data.num_p_steps,
        "optimizer_inner_steps": config.optimizer.inner_steps,
        "representation": config.estimator.representation,
        "low_rank": config.estimator.low_rank,
        "controller_methods": config.estimator.controller_methods,
    }


def _config_with_identity(
    mapping: Mapping[str, Any],
    *,
    experiment: str,
    replica_index: int,
    replica_seed: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(mapping))
    result["experiment"] = experiment
    result["replica_id"] = f"replica-{replica_index:04d}"
    result["replica_seed"] = replica_seed
    result["controller"]["reference_optimum_artifact"] = None
    return result


def _load_template(repo_root: Path, profile: Phase9Profile) -> dict[str, Any]:
    path = repo_root / profile.template
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise Phase9Error(f"template must be an object: {path}")
    merged = _deep_override(raw, profile.overrides, context=profile.name)
    ExperimentConfig.from_mapping(merged)
    return merged


def _run_artifact_path(config: ExperimentConfig) -> str:
    return str(
        Path(config.cache_root)
        / config.run_id
        / "phase8_reference_optimum.pt"
    )


def _selection_payload(
    spec: Phase9Spec,
    profile_names: Sequence[str],
    replica_indices: Sequence[int],
) -> dict[str, Any]:
    return {
        "spec_hash": spec.content_hash,
        "profiles": list(profile_names),
        "replica_indices": list(replica_indices),
    }


def build_phase9_bundle(
    spec: Phase9Spec,
    repo_root: str | Path,
    *,
    profile_names: Sequence[str] | None = None,
    replica_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    selected_names = tuple(profile_names or spec.default_profiles)
    if not selected_names or len(selected_names) != len(set(selected_names)):
        raise Phase9Error("selected profiles must be unique and nonempty")
    selected_profiles = tuple(spec.profile(name) for name in selected_names)
    generation_profiles = [
        (profile, profile.cells) for profile in selected_profiles
    ]
    dependency_cells_by_profile: dict[str, set[str]] = {}
    if "dense-confirm" in selected_names:
        dependency_cells_by_profile.setdefault("controller-screen", set()).add(
            "center"
        )
    if "lfu-isolation" in selected_names:
        dependency_cells_by_profile.setdefault("controller-screen", set()).update(
            {"center", "trend-h-010"}
        )
    if "adaptation-screen" in selected_names:
        dependency_cells_by_profile.setdefault("controller-screen", set()).update(
            {"center", "trend-h-010"}
        )
        dependency_cells_by_profile.setdefault("lfu-isolation", set()).add(
            "adaptive-h010-no-lfu"
        )
    for dependency_name, dependency_names in reversed(
        tuple(dependency_cells_by_profile.items())
    ):
        if dependency_name in selected_names:
            continue
        dependency_profile = spec.profile(dependency_name)
        dependency_cells = tuple(
            cell
            for cell in dependency_profile.cells
            if cell.name in dependency_names
        )
        if {cell.name for cell in dependency_cells} != dependency_names:
            raise Phase9Error(
                f"{dependency_name} is missing a required dependency cell"
            )
        generation_profiles.insert(0, (dependency_profile, dependency_cells))
    selected_replicas = tuple(
        replica_indices
        or range(
            spec.default_replica_start,
            spec.default_replica_start + spec.default_replica_count,
        )
    )
    if (
        not selected_replicas
        or len(selected_replicas) != len(set(selected_replicas))
        or min(selected_replicas) < 0
    ):
        raise Phase9Error("replica indices must be unique and nonnegative")

    candidates: list[dict[str, Any]] = []
    profile_templates = {
        profile.name: _load_template(root, profile)
        for profile, _ in generation_profiles
    }
    for replica_index in selected_replicas:
        seed = _replica_seed(spec.base_replica_seed, replica_index)
        for profile, cells in generation_profiles:
            template = profile_templates[profile.name]
            for cell_index, cell in enumerate(cells):
                mapping = _deep_override(
                    template,
                    cell.overrides,
                    context=f"{profile.name}.{cell.name}",
                )
                policy = cell.policy or "optimal_plugin"
                policy_suffix = (
                    "fixed" if policy == "fixed_unified" else "plugin"
                )
                mapping = _config_with_identity(
                    mapping,
                    experiment=_experiment_name(
                        profile.name, cell.name, policy_suffix
                    ),
                    replica_index=replica_index,
                    replica_seed=seed,
                )
                mapping["controller"]["policy"] = policy
                config = ExperimentConfig.from_mapping(mapping)
                candidates.append(
                    {
                        "kind": "treatment",
                        "profile": profile,
                        "cell": cell,
                        "cell_index": cell_index,
                        "replica_index": replica_index,
                        "config": config,
                        "mapping": mapping,
                        "policy": policy,
                        "policy_suffix": policy_suffix,
                    }
                )

    anchors: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        design = replica_bundle_id(candidate["config"])
        current = anchors.get(design)
        priority = (
            0 if candidate["profile"].name == "controller-screen" else 1,
            candidate["cell_index"],
            candidate["profile"].name,
        )
        if current is None or priority < current["priority"]:
            anchors[design] = {"candidate": candidate, "priority": priority}

    for anchor in anchors.values():
        anchor["config"] = anchor["candidate"]["config"]
        anchor["run_id"] = anchor["config"].run_id
        anchor["artifact"] = _run_artifact_path(anchor["config"])

    configs: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    treatment_entries: dict[tuple[int, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        config = candidate["config"]
        mapping = copy.deepcopy(candidate["mapping"])
        design = replica_bundle_id(config)
        anchor = anchors[design]
        is_anchor = config.run_id == anchor["run_id"]
        if not is_anchor:
            mapping["controller"]["reference_optimum_artifact"] = anchor[
                "artifact"
            ]
            config = ExperimentConfig.from_mapping(mapping)
        profile = candidate["profile"]
        cell = candidate["cell"]
        factors = _factors(config)
        if profile.control_scope == "explicit":
            factors = {
                **factors,
                "fisher_update_method": config.estimator.method,
                "controller_policy": config.controller.policy,
                "fixed_pi": config.controller.fixed_pi,
            }
        entry = {
            "entry_id": (
                f"r{candidate['replica_index']:04d}:{profile.name}:"
                f"{cell.name}:{candidate['policy_suffix']}"
            ),
            "kind": cell.kind,
            "profile": profile.name,
            "cell": cell.name,
            "policy": candidate["policy"],
            "replica_index": candidate["replica_index"],
            "replica_id": config.replica_id,
            "replica_seed": config.replica_seed,
            "replica_bundle_id": design,
            "run_id": config.run_id,
            "config_hash": config.config_hash,
            "config_file": "",
            "cache_root": config.cache_root,
            "oracle_anchor_run_id": anchor["run_id"],
            "is_oracle_anchor": is_anchor,
            "control_run_id": None,
            "factors": factors,
            "estimated_trajectory_seconds": profile.estimated_trajectory_seconds
            * config.data.num_p_steps
            / 100.0
            * config.data.samples_per_step
            / 128.0,
            "estimated_run_bytes": int(
                profile.estimated_run_bytes
                * config.data.num_p_steps
                / 100.0
            ),
        }
        entries.append(entry)
        treatment_entries[
            (candidate["replica_index"], profile.name, cell.name)
        ] = entry
        configs[entry["entry_id"]] = mapping

    controls: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        profile = candidate["profile"]
        if profile.control_scope == "explicit":
            continue
        cell = candidate["cell"]
        config = candidate["config"]
        scope_key = (
            candidate["replica_index"],
            profile.name,
            cell.name if profile.control_scope == "cell" else "profile",
        )
        if scope_key not in controls:
            if profile.control_scope == "profile":
                control_base = profile_templates[profile.name]
            else:
                control_base = _deep_override(
                    profile_templates[profile.name],
                    cell.overrides,
                    context=f"{profile.name}.{cell.name}.control",
                )
            control_cell = (
                cell.name if profile.control_scope == "cell" else "control"
            )
            control_mapping = _config_with_identity(
                control_base,
                experiment=_experiment_name(
                    profile.name, control_cell, "fixed"
                ),
                replica_index=candidate["replica_index"],
                replica_seed=config.replica_seed,
            )
            control_mapping["controller"].update(
                {
                    "policy": "fixed_unified",
                    "fixed_pi": spec.fixed_pi,
                    "pi_min": min(spec.fixed_pi, 0.05),
                    "pi_max": max(spec.fixed_pi, 0.95),
                    "reference_optimum_artifact": anchors[
                        replica_bundle_id(config)
                    ]["artifact"],
                }
            )
            control_config = ExperimentConfig.from_mapping(control_mapping)
            entry = {
                "entry_id": (
                    f"r{candidate['replica_index']:04d}:{profile.name}:"
                    f"{control_cell}:fixed"
                ),
                "kind": "control",
                "profile": profile.name,
                "cell": control_cell,
                "policy": "fixed_unified",
                "replica_index": candidate["replica_index"],
                "replica_id": control_config.replica_id,
                "replica_seed": control_config.replica_seed,
                "replica_bundle_id": replica_bundle_id(control_config),
                "run_id": control_config.run_id,
                "config_hash": control_config.config_hash,
                "config_file": "",
                "cache_root": control_config.cache_root,
                "oracle_anchor_run_id": anchors[
                    replica_bundle_id(config)
                ]["run_id"],
                "is_oracle_anchor": False,
                "control_run_id": None,
                "factors": _factors(control_config),
                "estimated_trajectory_seconds": (
                    profile.estimated_trajectory_seconds
                    * control_config.data.num_p_steps
                    / 100.0
                    * control_config.data.samples_per_step
                    / 128.0
                ),
                "estimated_run_bytes": int(
                    profile.estimated_run_bytes
                    * control_config.data.num_p_steps
                    / 100.0
                ),
            }
            entries.append(entry)
            configs[entry["entry_id"]] = control_mapping
            controls[scope_key] = entry
        treatment_entries[
            (candidate["replica_index"], profile.name, cell.name)
        ]["control_run_id"] = controls[scope_key]["run_id"]

    entries_by_cell: dict[tuple[int, str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (entry["replica_index"], entry["profile"], entry["cell"])
        # Per-cell generated controls intentionally share the treatment cell's
        # name. Prefer the control when resolving explicit cross-profile refs.
        if key not in entries_by_cell or entry["kind"] == "control":
            entries_by_cell[key] = entry
    for candidate in candidates:
        profile = candidate["profile"]
        cell = candidate["cell"]
        if profile.control_scope != "explicit" or cell.control_ref is None:
            continue
        control_profile, control_cell = cell.control_ref.split(":", 1)
        control = entries_by_cell.get(
            (candidate["replica_index"], control_profile, control_cell)
        )
        if control is None:
            raise Phase9Error(
                f"{profile.name}.{cell.name} refers to missing control "
                f"{cell.control_ref}"
            )
        treatment_entries[
            (candidate["replica_index"], profile.name, cell.name)
        ]["control_run_id"] = control["run_id"]

    entries.sort(
        key=lambda row: (
            row["replica_index"],
            not row["is_oracle_anchor"],
            row["profile"],
            row["kind"],
            row["cell"],
        )
    )
    selection = _selection_payload(spec, selected_names, selected_replicas)
    bundle_digest = _content_hash(selection)[:12]
    replica_label = (
        f"r{selected_replicas[0]:04d}"
        if len(selected_replicas) == 1
        else f"r{selected_replicas[0]:04d}-r{selected_replicas[-1]:04d}"
    )
    bundle_id = f"{spec.name}__{replica_label}__{bundle_digest}"
    for entry in entries:
        entry["config_file"] = str(
            Path("configs")
            / entry["replica_id"]
            / f"{entry['profile']}__{entry['cell']}__{entry['policy']}.json"
        )

    unique_designs = {entry["replica_bundle_id"] for entry in entries}
    anchor_entries = [entry for entry in entries if entry["is_oracle_anchor"]]
    manifest = {
        "bundle_schema_version": PHASE9_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "spec_name": spec.name,
        "spec_hash": spec.content_hash,
        "selection": selection,
        "fixed_pi": spec.fixed_pi,
        "entry_count": len(entries),
        "replica_count": len(selected_replicas),
        "initialization_count": len(unique_designs),
        "oracle_anchor_count": len(anchor_entries),
        "estimated_seconds": (
            len(unique_designs) * spec.initialization_seconds
            + sum(
                spec.reference_seconds_per_p * entry["factors"]["num_p_steps"]
                for entry in anchor_entries
            )
            + sum(entry["estimated_trajectory_seconds"] for entry in entries)
        ),
        "estimated_bytes": sum(entry["estimated_run_bytes"] for entry in entries),
        "entries": entries,
    }
    return manifest, configs


def prepare_phase9_bundle(
    spec: Phase9Spec,
    repo_root: str | Path,
    *,
    profile_names: Sequence[str] | None = None,
    replica_indices: Sequence[int] | None = None,
) -> Phase9Bundle:
    root = Path(repo_root)
    manifest, configs = build_phase9_bundle(
        spec,
        root,
        profile_names=profile_names,
        replica_indices=replica_indices,
    )
    bundle_root = root / spec.bundle_root
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_phase9_bundle(destination)
        if loaded.manifest != manifest:
            raise Phase9Error(
                f"existing bundle does not match generated content: {destination}"
            )
        return loaded

    bundle_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{manifest['bundle_id']}.", dir=bundle_root)
    )
    try:
        for entry in manifest["entries"]:
            _write_json(
                temporary / entry["config_file"],
                configs[entry["entry_id"]],
            )
        _write_json(temporary / "bundle.json", manifest)
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Phase9Bundle(destination, manifest)


def load_phase9_bundle(path: str | Path) -> Phase9Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Phase9Error(f"Phase 9 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if not isinstance(manifest, Mapping):
        raise Phase9Error("Phase 9 bundle manifest must be an object")
    if manifest.get("bundle_schema_version") != PHASE9_BUNDLE_SCHEMA_VERSION:
        raise Phase9Error("unsupported Phase 9 bundle schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get("entry_count"):
        raise Phase9Error("Phase 9 bundle entry count is invalid")
    run_ids = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise Phase9Error("Phase 9 bundle entry must be an object")
        config_path = bundle_path / entry["config_file"]
        config_raw = _read_json(config_path)
        config = ExperimentConfig.from_mapping(config_raw)
        if config.run_id != entry["run_id"]:
            raise Phase9Error(f"run ID mismatch in {config_path}")
        if config.config_hash != entry["config_hash"]:
            raise Phase9Error(f"config hash mismatch in {config_path}")
        if config.run_id in run_ids:
            raise Phase9Error(f"duplicate run ID in bundle: {config.run_id}")
        run_ids.add(config.run_id)
    for entry in entries:
        anchor = entry["oracle_anchor_run_id"]
        if anchor not in run_ids:
            raise Phase9Error(f"missing oracle anchor dependency: {anchor}")
        control = entry.get("control_run_id")
        if control is not None and control not in run_ids:
            raise Phase9Error(f"missing paired control dependency: {control}")
    return Phase9Bundle(bundle_path, manifest)


def discover_phase9_bundles(root: str | Path) -> tuple[Phase9Bundle, ...]:
    bundle_root = Path(root)
    if not bundle_root.is_dir():
        return ()
    bundles = []
    for path in sorted(bundle_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            bundles.append(load_phase9_bundle(path))
    return tuple(bundles)


def _artifact_state(final_path: Path, incomplete_path: Path) -> str:
    if (final_path / "COMPLETED").is_file():
        return "completed"
    if final_path.exists():
        return "invalid"
    if incomplete_path.exists():
        return "incomplete"
    return "missing"


def phase9_status_rows(
    bundle: Phase9Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in bundle.entries:
        config = ExperimentConfig.from_mapping(
            _read_json(bundle.path / entry["config_file"])
        )
        run_root = root / config.cache_root
        run_state = _artifact_state(
            run_root / config.run_id,
            run_root / ".incomplete" / config.run_id,
        )
        replica_root = root / Path(config.cache_root).parent / "replicas"
        replica_path = replica_root / entry["replica_bundle_id"]
        rows.append(
            {
                **entry,
                "bundle_id": bundle.bundle_id,
                "bundle_path": str(bundle.path),
                "config_path": str(bundle.path / entry["config_file"]),
                "run_state": run_state,
                "run_path": str(run_root / config.run_id),
                "initialization_state": (
                    "completed"
                    if (replica_path / "COMPLETED").is_file()
                    else "invalid"
                    if replica_path.exists()
                    else "missing"
                ),
                "replica_path": str(replica_path),
            }
        )
    return rows
