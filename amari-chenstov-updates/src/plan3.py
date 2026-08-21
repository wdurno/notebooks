"""Validated, planning-only expansion of the Plan 3 handoff design."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .phase9 import (
    Phase9Error,
    load_phase9_bundle,
    load_phase9_spec,
    phase9_status_rows,
)
from .plan2 import Plan2Error, load_plan2_bundle, plan2_status_rows
from .config import ExperimentConfig
from .initialization import replica_bundle_id
from .seeding import derive_component_seed


PLAN3_SPEC_SCHEMA_VERSION = 1
PLAN3_PHASE1_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE2_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE3_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE4_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE5_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE6_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE7_BUNDLE_SCHEMA_VERSION = 1
PLAN3_PHASE1_ACCEPTED_BUNDLE_ID = "plan3-phase1__r0006__3930db574a45"
PLAN3_PHASE2_ACCEPTED_BUNDLE_ID = (
    "plan3-replay-screen__r0006-r0010__16a259169db6"
)
PLAN3_PHASE4_ACCEPTED_BUNDLE_ID = (
    "plan3-history-frontier__r0006-r0010__12582f2d297a"
)
PLAN3_PHASE5_ACCEPTED_BUNDLE_ID = (
    "plan3-lfu-isolation__r0006-r0010__6d8c4ad67265"
)
PLAN3_PHASE6_ACCEPTED_BUNDLE_ID = (
    "plan3-deployment-frontier__r0006-r0010__f21931201ff8"
)


class Plan3Error(RuntimeError):
    """Raised when the Plan 3 planning specification is invalid."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan3Error(f"could not read {path}: {exc}") from exc


def _require_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise Plan3Error(
            f"invalid {context}: missing={missing}, unknown={unknown}"
        )


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Plan3Error(f"{context} must be a positive integer")
    return value


def _nonnegative_number(value: Any, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise Plan3Error(f"{context} must be finite and nonnegative")
    return float(value)


@dataclasses.dataclass(frozen=True)
class Plan3Condition:
    name: str
    role: str
    source: str
    data_mode: str
    replay_budget: int | str | None
    uses_ewc: bool
    fisher_update: str
    controller: str
    comparison: str | None
    estimand: str

    @property
    def is_new(self) -> bool:
        return self.source == "new"


@dataclasses.dataclass(frozen=True)
class Plan3Stage:
    name: str
    plan_phase: int
    decision: str
    check_in: str
    conditions: tuple[Plan3Condition, ...]


@dataclasses.dataclass(frozen=True)
class Plan3Spec:
    path: Path
    name: str
    source_anchor_bundle: str
    confirmation_bundle: str
    replica_indices: tuple[int, ...]
    samples_per_step: int
    num_p_steps: int
    optimizer_inner_steps: int
    parameter_count: int
    fisher_rank: int
    selected_replay_budget: int
    fixed_seconds_per_run: float
    seconds_per_optimizer_observation: float
    estimated_artifact_bytes: int
    replay_fixed_seconds_per_run: float
    replay_seconds_per_optimizer_observation: float
    replay_estimated_artifact_bytes: int
    observation_payload_bytes: int
    replay_fixed_metadata_bytes: int
    parameter_scalar_bytes: int
    stages: tuple[Plan3Stage, ...]

    @property
    def ewc_summary_bytes(self) -> int:
        return (
            self.parameter_count
            * (self.fisher_rank + 2)
            * self.parameter_scalar_bytes
        )

    @property
    def memory_matched_replay_budget(self) -> int:
        available = self.ewc_summary_bytes - self.replay_fixed_metadata_bytes
        return max(0, available // self.observation_payload_bytes)


def load_plan3_spec(path: str | Path) -> Plan3Spec:
    spec_path = Path(path)
    raw = _read_json(spec_path)
    if not isinstance(raw, Mapping):
        raise Plan3Error("Plan 3 specification must be an object")
    _require_keys(
        raw,
        {
            "schema_version",
            "name",
            "handoff",
            "shared_design",
            "planning_placeholders",
            "cost_model",
            "stages",
        },
        "Plan 3 specification",
    )
    if raw["schema_version"] != PLAN3_SPEC_SCHEMA_VERSION:
        raise Plan3Error("unsupported Plan 3 specification schema")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise Plan3Error("name must be a nonempty string")

    handoff = raw["handoff"]
    if not isinstance(handoff, Mapping):
        raise Plan3Error("handoff must be an object")
    _require_keys(
        handoff,
        {"source_anchor_bundle", "confirmation_bundle", "replica_indices"},
        "handoff",
    )
    for field in ("source_anchor_bundle", "confirmation_bundle"):
        if not isinstance(handoff[field], str) or not handoff[field]:
            raise Plan3Error(f"handoff.{field} must be a nonempty string")
    replicas = handoff["replica_indices"]
    if (
        not isinstance(replicas, list)
        or not replicas
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in replicas
        )
        or len(replicas) != len(set(replicas))
    ):
        raise Plan3Error("handoff.replica_indices must be unique nonnegative integers")

    design = raw["shared_design"]
    if not isinstance(design, Mapping):
        raise Plan3Error("shared_design must be an object")
    _require_keys(
        design,
        {
            "samples_per_step",
            "num_p_steps",
            "optimizer_inner_steps",
            "parameter_count",
            "fisher_rank",
        },
        "shared_design",
    )
    design_values = {
        key: _positive_int(value, f"shared_design.{key}")
        for key, value in design.items()
    }
    if design_values["fisher_rank"] > design_values["parameter_count"]:
        raise Plan3Error("fisher_rank cannot exceed parameter_count")

    placeholders = raw["planning_placeholders"]
    if not isinstance(placeholders, Mapping):
        raise Plan3Error("planning_placeholders must be an object")
    _require_keys(
        placeholders,
        {"selected_replay_budget"},
        "planning_placeholders",
    )
    selected_budget = _positive_int(
        placeholders["selected_replay_budget"],
        "planning_placeholders.selected_replay_budget",
    )
    cost = raw["cost_model"]
    if not isinstance(cost, Mapping):
        raise Plan3Error("cost_model must be an object")
    _require_keys(
        cost,
        {
            "fixed_seconds_per_run",
            "seconds_per_optimizer_observation",
            "estimated_artifact_bytes",
            "replay_fixed_seconds_per_run",
            "replay_seconds_per_optimizer_observation",
            "replay_estimated_artifact_bytes",
            "observation_payload_bytes",
            "replay_fixed_metadata_bytes",
            "parameter_scalar_bytes",
        },
        "cost_model",
    )
    fixed_seconds = _nonnegative_number(
        cost["fixed_seconds_per_run"], "cost_model.fixed_seconds_per_run"
    )
    per_observation = _nonnegative_number(
        cost["seconds_per_optimizer_observation"],
        "cost_model.seconds_per_optimizer_observation",
    )
    artifact_bytes = _positive_int(
        cost["estimated_artifact_bytes"], "cost_model.estimated_artifact_bytes"
    )
    replay_fixed_seconds = _nonnegative_number(
        cost["replay_fixed_seconds_per_run"],
        "cost_model.replay_fixed_seconds_per_run",
    )
    replay_per_observation = _nonnegative_number(
        cost["replay_seconds_per_optimizer_observation"],
        "cost_model.replay_seconds_per_optimizer_observation",
    )
    replay_artifact_bytes = _positive_int(
        cost["replay_estimated_artifact_bytes"],
        "cost_model.replay_estimated_artifact_bytes",
    )
    observation_bytes = _positive_int(
        cost["observation_payload_bytes"], "cost_model.observation_payload_bytes"
    )
    replay_metadata_bytes = _positive_int(
        cost["replay_fixed_metadata_bytes"],
        "cost_model.replay_fixed_metadata_bytes",
    )
    scalar_bytes = _positive_int(
        cost["parameter_scalar_bytes"], "cost_model.parameter_scalar_bytes"
    )

    stages_raw = raw["stages"]
    if not isinstance(stages_raw, list) or not stages_raw:
        raise Plan3Error("stages must be a nonempty list")
    stages = []
    stage_names = set()
    known_conditions = set()
    allowed_roles = {
        "primary-control",
        "mechanism-reference",
        "unconstrained-memory-control",
        "treatment",
        "diagnostic",
    }
    for stage_index, stage_raw in enumerate(stages_raw):
        if not isinstance(stage_raw, Mapping):
            raise Plan3Error(f"stages[{stage_index}] must be an object")
        _require_keys(
            stage_raw,
            {"name", "plan_phase", "decision", "check_in", "conditions"},
            f"stages[{stage_index}]",
        )
        stage_name = stage_raw["name"]
        if not isinstance(stage_name, str) or not stage_name:
            raise Plan3Error(f"stages[{stage_index}].name must be nonempty")
        if stage_name in stage_names:
            raise Plan3Error(f"duplicate stage: {stage_name}")
        stage_names.add(stage_name)
        plan_phase = _positive_int(
            stage_raw["plan_phase"], f"stages[{stage_index}].plan_phase"
        )
        for field in ("decision", "check_in"):
            if not isinstance(stage_raw[field], str) or not stage_raw[field]:
                raise Plan3Error(f"stages[{stage_index}].{field} must be nonempty")
        conditions_raw = stage_raw["conditions"]
        if not isinstance(conditions_raw, list) or not conditions_raw:
            raise Plan3Error(f"stages[{stage_index}].conditions must be nonempty")
        conditions = []
        stage_condition_names = set()
        for condition_index, condition_raw in enumerate(conditions_raw):
            context = f"stages[{stage_index}].conditions[{condition_index}]"
            if not isinstance(condition_raw, Mapping):
                raise Plan3Error(f"{context} must be an object")
            _require_keys(
                condition_raw,
                {
                    "name",
                    "role",
                    "source",
                    "data_mode",
                    "replay_budget",
                    "uses_ewc",
                    "fisher_update",
                    "controller",
                    "comparison",
                    "estimand",
                },
                context,
            )
            name = condition_raw["name"]
            if not isinstance(name, str) or not name:
                raise Plan3Error(f"{context}.name must be nonempty")
            if name in stage_condition_names:
                raise Plan3Error(f"duplicate condition {name} in {stage_name}")
            stage_condition_names.add(name)
            role = condition_raw["role"]
            if role not in allowed_roles:
                raise Plan3Error(f"{context}.role is unsupported")
            source = condition_raw["source"]
            if not isinstance(source, str) or not source:
                raise Plan3Error(f"{context}.source must be nonempty")
            data_mode = condition_raw["data_mode"]
            if data_mode not in {
                "current",
                "replay",
                "hybrid",
                "gate-selected-ewc",
            }:
                raise Plan3Error(f"{context}.data_mode is unsupported")
            replay_budget = condition_raw["replay_budget"]
            if data_mode == "current" and replay_budget is not None:
                raise Plan3Error(f"{context} current data cannot have replay budget")
            if data_mode != "current" and not (
                (
                    isinstance(replay_budget, int)
                    and not isinstance(replay_budget, bool)
                    and replay_budget > 0
                )
                or replay_budget in {"selected", "memory-matched", "unbounded"}
            ):
                raise Plan3Error(f"{context}.replay_budget is unsupported")
            if not isinstance(condition_raw["uses_ewc"], bool):
                raise Plan3Error(f"{context}.uses_ewc must be boolean")
            fisher_update = condition_raw["fisher_update"]
            if fisher_update not in {
                "none",
                "ac-only",
                "full-lfu",
                "gate-selected",
            }:
                raise Plan3Error(f"{context}.fisher_update is unsupported")
            if fisher_update != "none" and not condition_raw["uses_ewc"]:
                raise Plan3Error(f"{context} cannot update an absent Fisher summary")
            controller = condition_raw["controller"]
            if controller not in {"fixed-005", "adaptive-h020", "not-applicable"}:
                raise Plan3Error(f"{context}.controller is unsupported")
            if condition_raw["uses_ewc"] and controller == "not-applicable":
                raise Plan3Error(f"{context} EWC condition needs a controller")
            comparison = condition_raw["comparison"]
            if comparison is not None and (
                not isinstance(comparison, str) or not comparison
            ):
                raise Plan3Error(f"{context}.comparison must be null or nonempty")
            if role != "primary-control" and comparison is None:
                raise Plan3Error(f"{context} must name its comparison")
            estimand = condition_raw["estimand"]
            if not isinstance(estimand, str) or not estimand:
                raise Plan3Error(f"{context}.estimand must be nonempty")
            conditions.append(
                Plan3Condition(
                    name=name,
                    role=role,
                    source=source,
                    data_mode=data_mode,
                    replay_budget=replay_budget,
                    uses_ewc=condition_raw["uses_ewc"],
                    fisher_update=fisher_update,
                    controller=controller,
                    comparison=comparison,
                    estimand=estimand,
                )
            )
            known_conditions.add(name)
        stages.append(
            Plan3Stage(
                name=stage_name,
                plan_phase=plan_phase,
                decision=stage_raw["decision"],
                check_in=stage_raw["check_in"],
                conditions=tuple(conditions),
            )
        )

    missing_comparisons = sorted(
        {
            condition.comparison
            for stage in stages
            for condition in stage.conditions
            if condition.comparison is not None
            and condition.comparison not in known_conditions
        }
    )
    if missing_comparisons:
        raise Plan3Error(
            f"conditions refer to unknown comparisons: {missing_comparisons}"
        )

    return Plan3Spec(
        path=spec_path,
        name=raw["name"],
        source_anchor_bundle=handoff["source_anchor_bundle"],
        confirmation_bundle=handoff["confirmation_bundle"],
        replica_indices=tuple(replicas),
        samples_per_step=design_values["samples_per_step"],
        num_p_steps=design_values["num_p_steps"],
        optimizer_inner_steps=design_values["optimizer_inner_steps"],
        parameter_count=design_values["parameter_count"],
        fisher_rank=design_values["fisher_rank"],
        selected_replay_budget=selected_budget,
        fixed_seconds_per_run=fixed_seconds,
        seconds_per_optimizer_observation=per_observation,
        estimated_artifact_bytes=artifact_bytes,
        replay_fixed_seconds_per_run=replay_fixed_seconds,
        replay_seconds_per_optimizer_observation=replay_per_observation,
        replay_estimated_artifact_bytes=replay_artifact_bytes,
        observation_payload_bytes=observation_bytes,
        replay_fixed_metadata_bytes=replay_metadata_bytes,
        parameter_scalar_bytes=scalar_bytes,
        stages=tuple(stages),
    )


def parse_stage_names(value: str | None, spec: Plan3Spec) -> tuple[str, ...]:
    available = {stage.name for stage in spec.stages}
    if value is None:
        return tuple(stage.name for stage in spec.stages)
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected or len(selected) != len(set(selected)):
        raise Plan3Error("stages must be a unique comma-separated list")
    unknown = sorted(set(selected) - available)
    if unknown:
        raise Plan3Error(f"unknown Plan 3 stages: {unknown}")
    return selected


def _resolved_replay_budget(condition: Plan3Condition, spec: Plan3Spec) -> int:
    budget = condition.replay_budget
    if isinstance(budget, int):
        return budget
    if budget == "selected":
        return spec.selected_replay_budget
    if budget == "memory-matched":
        return spec.memory_matched_replay_budget
    if budget == "unbounded":
        return spec.samples_per_step * spec.num_p_steps
    if budget is None:
        return 0
    raise AssertionError(f"unhandled replay budget: {budget}")


def _optimizer_observations(condition: Plan3Condition, spec: Plan3Spec) -> int:
    updates = spec.num_p_steps - 1
    if condition.data_mode == "current":
        return updates * spec.samples_per_step
    budget = _resolved_replay_budget(condition, spec)
    return sum(
        spec.samples_per_step + min(budget, step * spec.samples_per_step)
        for step in range(updates)
    )


def _persistent_bytes(condition: Plan3Condition, spec: Plan3Spec) -> int:
    total = 0
    if condition.uses_ewc:
        total += spec.ewc_summary_bytes
    if condition.data_mode != "current":
        total += spec.replay_fixed_metadata_bytes
        total += _resolved_replay_budget(condition, spec) * spec.observation_payload_bytes
    if condition.controller == "adaptive-h020":
        total += spec.parameter_count * spec.parameter_scalar_bytes
    return total


def preview_plan3(
    spec: Plan3Spec,
    *,
    stage_names: Sequence[str] | None = None,
    replica_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    selected_names = tuple(stage_names or (stage.name for stage in spec.stages))
    if not selected_names or len(selected_names) != len(set(selected_names)):
        raise Plan3Error("selected stages must be unique and nonempty")
    stages_by_name = {stage.name: stage for stage in spec.stages}
    unknown = sorted(set(selected_names) - set(stages_by_name))
    if unknown:
        raise Plan3Error(f"unknown Plan 3 stages: {unknown}")
    replicas = tuple(replica_indices or spec.replica_indices)
    if (
        not replicas
        or len(replicas) != len(set(replicas))
        or any(value < 0 for value in replicas)
    ):
        raise Plan3Error("replica indices must be unique and nonnegative")

    stage_rows = []
    condition_rows = []
    for stage_name in selected_names:
        stage = stages_by_name[stage_name]
        stage_seconds = 0.0
        stage_bytes = 0
        new_runs = 0
        reused_runs = 0
        for condition in stage.conditions:
            observations = _optimizer_observations(condition, spec)
            uses_replay_cost = not condition.uses_ewc
            seconds_per_run = (
                (
                    spec.replay_fixed_seconds_per_run
                    + spec.replay_seconds_per_optimizer_observation * observations
                )
                if uses_replay_cost
                else (
                    spec.fixed_seconds_per_run
                    + spec.seconds_per_optimizer_observation * observations
                )
            )
            if stage.plan_phase == 6:
                deployment_estimates = {
                    row["condition"]: float(row["estimated_seconds"])
                    for row in _phase6_profiles()
                }
                seconds_per_run = deployment_estimates[condition.name]
            run_count = len(replicas)
            condition_new_runs = run_count if condition.is_new else 0
            condition_reused_runs = 0 if condition.is_new else run_count
            if condition.is_new:
                stage_seconds += seconds_per_run * run_count
                stage_bytes += (
                    spec.replay_estimated_artifact_bytes
                    if uses_replay_cost
                    else spec.estimated_artifact_bytes
                ) * run_count
            new_runs += condition_new_runs
            reused_runs += condition_reused_runs
            condition_rows.append(
                {
                    "stage": stage.name,
                    "plan_phase": stage.plan_phase,
                    "condition": condition.name,
                    "role": condition.role,
                    "source": condition.source,
                    "comparison": condition.comparison,
                    "estimand": condition.estimand,
                    "data_mode": condition.data_mode,
                    "configured_replay_budget": condition.replay_budget,
                    "planning_replay_budget": _resolved_replay_budget(condition, spec),
                    "uses_ewc": condition.uses_ewc,
                    "fisher_update": condition.fisher_update,
                    "controller": condition.controller,
                    "new_runs": condition_new_runs,
                    "reused_runs": condition_reused_runs,
                    "optimizer_observations_per_run": observations,
                    "estimated_seconds_per_new_run": seconds_per_run,
                    "estimated_incremental_persistent_bytes": _persistent_bytes(
                        condition, spec
                    ),
                }
            )
        stage_rows.append(
            {
                "stage": stage.name,
                "plan_phase": stage.plan_phase,
                "decision": stage.decision,
                "check_in": stage.check_in,
                "condition_count": len(stage.conditions),
                "new_runs": new_runs,
                "reused_runs": reused_runs,
                "estimated_wall_hours": stage_seconds / 3600.0,
                "estimated_storage_gib": stage_bytes / 2**30,
            }
        )

    return {
        "spec_name": spec.name,
        "planning_only": True,
        "replica_indices": list(replicas),
        "source_anchor_bundle": spec.source_anchor_bundle,
        "confirmation_bundle": spec.confirmation_bundle,
        "shared_design": {
            "samples_per_step": spec.samples_per_step,
            "num_p_steps": spec.num_p_steps,
            "optimizer_inner_steps": spec.optimizer_inner_steps,
            "parameter_count": spec.parameter_count,
            "fisher_rank": spec.fisher_rank,
        },
        "selected_stages": list(selected_names),
        "new_runs": sum(row["new_runs"] for row in stage_rows),
        "reused_condition_runs": sum(row["reused_runs"] for row in stage_rows),
        "estimated_wall_hours": sum(row["estimated_wall_hours"] for row in stage_rows),
        "estimated_storage_gib": sum(
            row["estimated_storage_gib"] for row in stage_rows
        ),
        "cost_basis": (
            "phase1_pure_replay_pilot_or_plan2_fisher_runner_by_condition"
        ),
        "planning_placeholders": {
            "selected_replay_budget": spec.selected_replay_budget,
        },
        "derived_memory_contract": {
            "ewc_summary_bytes": spec.ewc_summary_bytes,
            "observation_payload_bytes": spec.observation_payload_bytes,
            "replay_fixed_metadata_bytes": spec.replay_fixed_metadata_bytes,
            "memory_matched_replay_budget": spec.memory_matched_replay_budget,
        },
        "warnings": [
            "Pure replay timing is calibrated at capacity 32; larger buffers are linear extrapolations.",
            "The selected replay budget is a cost placeholder, not a decision.",
            "Phase 6 timing uses measured replay and hybrid deployment-runner calibrations.",
            "Phase 6 adaptive timing assumes scalar controller work is negligible relative to fitting.",
        ],
        "stages": stage_rows,
        "conditions": condition_rows,
    }


def validate_plan3_handoff_manifests(
    spec: Plan3Spec,
    source_manifest: Mapping[str, Any],
    confirmation_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the exact paired identities Plan 3 inherits from Plan 2."""

    if source_manifest.get("bundle_id") != spec.source_anchor_bundle:
        raise Plan3Error("source anchor bundle ID does not match Plan 3")
    if confirmation_manifest.get("bundle_id") != spec.confirmation_bundle:
        raise Plan3Error("confirmation bundle ID does not match Plan 3")
    expected_replicas = list(spec.replica_indices)
    if source_manifest.get("selection", {}).get("replica_indices") != expected_replicas:
        raise Plan3Error("source anchor replicas do not match Plan 3")
    confirmation_selection = confirmation_manifest.get("selection", {})
    if confirmation_selection.get("replica_indices") != expected_replicas:
        raise Plan3Error("confirmation replicas do not match Plan 3")
    if confirmation_selection.get("samples_per_step") != [spec.samples_per_step]:
        raise Plan3Error("confirmation sample size does not match Plan 3")

    source_entries = source_manifest.get("entries")
    confirmation_entries = confirmation_manifest.get("entries")
    if not isinstance(source_entries, list) or not isinstance(
        confirmation_entries, list
    ):
        raise Plan3Error("handoff manifests must contain entry lists")
    anchors = [entry for entry in source_entries if entry.get("is_oracle_anchor")]
    anchors_by_replica: dict[int, Mapping[str, Any]] = {}
    for entry in anchors:
        replica = entry.get("replica_index")
        if replica in anchors_by_replica:
            raise Plan3Error(f"multiple source anchors for replica {replica}")
        anchors_by_replica[replica] = entry
    if set(anchors_by_replica) != set(spec.replica_indices):
        raise Plan3Error("source bundle does not contain one anchor per replica")

    expected_conditions = {
        "no-ewc-pi100",
        "fixed-ewc-pi005",
        "adaptive-ewc-h020",
        "fixed-ewc-pi010",
    }
    frozen_rows = []
    for replica in spec.replica_indices:
        anchor = anchors_by_replica[replica]
        rows = [
            entry
            for entry in confirmation_entries
            if entry.get("replica_index") == replica
        ]
        by_condition = {entry.get("condition"): entry for entry in rows}
        if len(rows) != len(by_condition) or set(by_condition) != expected_conditions:
            raise Plan3Error(
                f"confirmation conditions are invalid for replica {replica}"
            )
        if any(entry.get("samples_per_step") != spec.samples_per_step for entry in rows):
            raise Plan3Error(f"sample size mismatch for replica {replica}")
        master_ids = {entry.get("master_replica_bundle_id") for entry in rows}
        derived_ids = {entry.get("replica_bundle_id") for entry in rows}
        references = {entry.get("reference_source_run_id") for entry in rows}
        if master_ids != {anchor.get("replica_bundle_id")}:
            raise Plan3Error(f"master initialization mismatch for replica {replica}")
        if len(derived_ids) != 1 or None in derived_ids:
            raise Plan3Error(f"derived stream mismatch for replica {replica}")
        if references != {anchor.get("run_id")}:
            raise Plan3Error(f"reference path mismatch for replica {replica}")
        control = by_condition["no-ewc-pi100"]
        if control.get("control_run_id") is not None:
            raise Plan3Error(f"primary control is mislinked for replica {replica}")
        if any(
            by_condition[name].get("control_run_id") != control.get("run_id")
            for name in expected_conditions - {"no-ewc-pi100"}
        ):
            raise Plan3Error(f"paired control mismatch for replica {replica}")
        frozen_rows.append(
            {
                "replica_index": replica,
                "master_replica_bundle_id": anchor["replica_bundle_id"],
                "derived_replica_bundle_id": next(iter(derived_ids)),
                "reference_source_run_id": anchor["run_id"],
                "reference_source_config_hash": anchor["config_hash"],
                "control_run_id": control["run_id"],
                "control_config_hash": control["config_hash"],
                "fixed_ewc_run_id": by_condition["fixed-ewc-pi005"]["run_id"],
                "fixed_ewc_config_hash": by_condition["fixed-ewc-pi005"][
                    "config_hash"
                ],
                "adaptive_ewc_run_id": by_condition["adaptive-ewc-h020"][
                    "run_id"
                ],
                "adaptive_ewc_config_hash": by_condition[
                    "adaptive-ewc-h020"
                ]["config_hash"],
            }
        )
    return frozen_rows


def audit_plan3_handoff(spec: Plan3Spec, repo_root: str | Path) -> dict[str, Any]:
    """Read and validate frozen source artifacts without modifying them."""

    root = Path(repo_root)
    source_path = (
        root
        / "cache"
        / "mnist_experiment"
        / "phase9"
        / "bundles"
        / spec.source_anchor_bundle
    )
    confirmation_path = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan2"
        / "bundles"
        / spec.confirmation_bundle
    )
    try:
        source = load_phase9_bundle(source_path)
        confirmation = load_plan2_bundle(confirmation_path)
    except (Phase9Error, Plan2Error) as exc:
        raise Plan3Error(f"could not load frozen handoff: {exc}") from exc

    frozen_rows = validate_plan3_handoff_manifests(
        spec, source.manifest, confirmation.manifest
    )
    source_status = {
        row["run_id"]: row
        for row in phase9_status_rows(source, root)
        if row["is_oracle_anchor"]
    }
    confirmation_status = {
        row["run_id"]: row for row in plan2_status_rows(confirmation, root)
    }
    invalid = []
    for row in confirmation_status.values():
        if row["run_state"] != "completed":
            invalid.append(row["run_id"])
        if row["derived_bundle_state"] != "completed":
            invalid.append(row["replica_bundle_id"])
        if row["reference_state"] != "completed":
            invalid.append(row["reference_source_run_id"])
    for frozen in frozen_rows:
        anchor = source_status.get(frozen["reference_source_run_id"])
        if anchor is None or anchor["run_state"] != "completed":
            invalid.append(frozen["reference_source_run_id"])
        if anchor is None or anchor["initialization_state"] != "completed":
            invalid.append(frozen["master_replica_bundle_id"])
        for key in ("control_run_id", "fixed_ewc_run_id", "adaptive_ewc_run_id"):
            row = confirmation_status.get(frozen[key])
            if row is None or row["run_state"] != "completed":
                invalid.append(frozen[key])
    if invalid:
        raise Plan3Error(
            "frozen handoff contains incomplete dependencies: "
            + ", ".join(sorted(set(invalid)))
        )

    return {
        "validated": True,
        "source_anchor_bundle": source.bundle_id,
        "source_manifest_sha256": hashlib.sha256(
            (source.path / "bundle.json").read_bytes()
        ).hexdigest(),
        "confirmation_bundle": confirmation.bundle_id,
        "confirmation_manifest_sha256": hashlib.sha256(
            (confirmation.path / "bundle.json").read_bytes()
        ).hexdigest(),
        "replica_count": len(frozen_rows),
        "completed_anchor_count": len(source_status),
        "completed_confirmation_run_count": len(confirmation_status),
        "replicas": frozen_rows,
    }


@dataclasses.dataclass(frozen=True)
class Plan3Phase1Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _phase1_profiles() -> tuple[dict[str, Any], ...]:
    profiles = []
    for device in ("cpu", "cuda"):
        for capacity, label in ((0, "b000"), (8, "b008"), ("unbounded", "unbounded")):
            profiles.append(
                {
                    "name": f"{device}-{label}-smoke",
                    "device": device,
                    "capacity": capacity,
                    "max_steps": 3,
                    "role": "smoke",
                }
            )
    profiles.append(
        {
            "name": "cuda-b032-production-pilot",
            "device": "cuda",
            "capacity": 32,
            "max_steps": None,
            "role": "production-timing-pilot",
        }
    )
    return tuple(profiles)


def build_plan3_phase1_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    audit = audit_plan3_handoff(spec, root)
    replica_index = spec.replica_indices[0]
    confirmation_path = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan2"
        / "bundles"
        / spec.confirmation_bundle
    )
    confirmation = load_plan2_bundle(confirmation_path)
    source_entry = next(
        (
            entry
            for entry in confirmation.manifest["entries"]
            if entry["replica_index"] == replica_index
            and entry["condition"] == "no-ewc-pi100"
        ),
        None,
    )
    if source_entry is None:
        raise Plan3Error("Phase 1 source control configuration is missing")
    source_config_path = confirmation.path / source_entry["config_file"]
    source_mapping = _read_json(source_config_path)
    configs: dict[str, dict[str, Any]] = {}
    entries = []
    for profile in _phase1_profiles():
        mapping = json.loads(json.dumps(source_mapping))
        mapping.update(
            {
                "schema_version": 13,
                "artifact_schema_version": 5,
                "metric_schema_version": 9,
                "cache_root": "cache/mnist_experiment/plan3_runs",
                "experiment": f"mnist_lfu_plan3-phase1_{profile['name']}",
                "replay": {
                    "capacity": profile["capacity"],
                    "policy": "fifo",
                    "max_steps": profile["max_steps"],
                },
            }
        )
        mapping["runtime"]["device"] = profile["device"]
        config = ExperimentConfig.from_mapping(mapping)
        relative_path = f"configs/{profile['name']}.json"
        configs[relative_path] = config.to_mapping()
        entries.append(
            {
                **profile,
                "config_file": relative_path,
                "config_hash": config.config_hash,
                "run_id": config.run_id,
                "cache_root": config.cache_root,
                "replica_id": config.replica_id,
                "replica_bundle_id": source_entry["replica_bundle_id"],
                "source_control_run_id": source_entry["run_id"],
                "source_control_config_hash": source_entry["config_hash"],
            }
        )
    identity = {
        "schema_version": PLAN3_PHASE1_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "source_confirmation_bundle": spec.confirmation_bundle,
        "replica_index": replica_index,
        "profiles": list(_phase1_profiles()),
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-phase1__r{replica_index:04d}__{digest[:12]}"
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "config_count": len(configs),
        "audit": {
            "source_manifest_sha256": audit["source_manifest_sha256"],
            "confirmation_manifest_sha256": audit["confirmation_manifest_sha256"],
        },
        "entries": entries,
    }
    return manifest, configs


def prepare_plan3_phase1_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase1Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase1_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase1_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 1 bundle has incompatible contents")
        return loaded
    bundle_root.mkdir(parents=True, exist_ok=True)
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            destination_path = temporary / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase1Bundle(destination, manifest)


def load_plan3_phase1_bundle(path: str | Path) -> Plan3Phase1Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 1 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE1_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 1 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 1 bundle ID does not match its directory")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get("config_count"):
        raise Plan3Error("Phase 1 bundle entry count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 1 config identity mismatch: {config_path}")
    return Plan3Phase1Bundle(bundle_path, manifest)


def plan3_phase1_status_rows(
    bundle: Plan3Phase1Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state = "completed"
            run_path = final_path
        elif incomplete_path.exists():
            state = "incomplete"
            run_path = incomplete_path
        elif final_path.exists():
            state = "invalid"
            run_path = final_path
        else:
            state = "missing"
            run_path = final_path
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_state": state,
                "run_path": str(run_path),
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase2Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _replay_capacity_label(capacity: int | str) -> str:
    return "unbounded" if capacity == "unbounded" else f"b{int(capacity):03d}"


def _replay_optimizer_observations(
    capacity: int | str,
    spec: Plan3Spec,
) -> int:
    resolved = (
        spec.samples_per_step * spec.num_p_steps
        if capacity == "unbounded"
        else int(capacity)
    )
    return sum(
        spec.samples_per_step + min(resolved, step * spec.samples_per_step)
        for step in range(spec.num_p_steps - 1)
    )


def build_plan3_phase2_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    audit = audit_plan3_handoff(spec, root)
    confirmation = load_plan2_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan2"
        / "bundles"
        / spec.confirmation_bundle
    )
    phase1 = load_plan3_phase1_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE1_ACCEPTED_BUNDLE_ID
    )
    phase1_pilot = next(
        (
            entry
            for entry in phase1.manifest["entries"]
            if entry["role"] == "production-timing-pilot"
        ),
        None,
    )
    if phase1_pilot is None:
        raise Plan3Error("accepted Phase 1 bundle has no production pilot")
    phase1_pilot_config = ExperimentConfig.from_mapping(
        _read_json(phase1.path / phase1_pilot["config_file"])
    )
    phase1_pilot_path = root / phase1_pilot["cache_root"] / phase1_pilot["run_id"]
    if not (phase1_pilot_path / "COMPLETED").is_file():
        raise Plan3Error("accepted Phase 1 production pilot is incomplete")

    confirmation_by_replica_condition = {
        (entry["replica_index"], entry["condition"]): entry
        for entry in confirmation.manifest["entries"]
    }
    configs: dict[str, dict[str, Any]] = {}
    replay_entries = []
    control_entries = []
    capacities: tuple[int | str, ...] = (8, 32, 128, "unbounded")
    for replica in spec.replica_indices:
        current = confirmation_by_replica_condition.get(
            (replica, "no-ewc-pi100")
        )
        ewc = confirmation_by_replica_condition.get(
            (replica, "fixed-ewc-pi005")
        )
        if current is None or ewc is None:
            raise Plan3Error(f"Phase 2 controls are missing for replica {replica}")
        for condition, source in (
            ("current-only", current),
            ("ewc-fixed005-no-lfu", ewc),
        ):
            control_entries.append(
                {
                    "replica_index": replica,
                    "replica_id": source["replica_id"],
                    "condition": condition,
                    "run_id": source["run_id"],
                    "config_hash": source["config_hash"],
                    "cache_root": source["cache_root"],
                    "replica_bundle_id": source["replica_bundle_id"],
                    "source": "plan2-confirmation",
                }
            )
        source_mapping = _read_json(confirmation.path / current["config_file"])
        for capacity in capacities:
            label = _replay_capacity_label(capacity)
            condition = f"replay-{label}"
            reuses_pilot = replica == spec.replica_indices[0] and capacity == 32
            if reuses_pilot:
                config = phase1_pilot_config
                source = "phase1-production-pilot"
            else:
                mapping = json.loads(json.dumps(source_mapping))
                mapping.update(
                    {
                        "schema_version": 13,
                        "artifact_schema_version": 5,
                        "metric_schema_version": 9,
                        "cache_root": "cache/mnist_experiment/plan3_runs",
                        "experiment": f"mnist_lfu_plan3-replay-screen_{label}",
                        "replay": {
                            "capacity": capacity,
                            "policy": "fifo",
                            "max_steps": None,
                        },
                    }
                )
                mapping["runtime"]["device"] = "cuda"
                config = ExperimentConfig.from_mapping(mapping)
                source = "new"
            if config.replay is None or config.replay.capacity != capacity:
                raise Plan3Error("Phase 2 replay capacity was not encoded exactly")
            relative_path = f"configs/replica-{replica:04d}/{condition}.json"
            configs[relative_path] = config.to_mapping()
            optimizer_observations = _replay_optimizer_observations(capacity, spec)
            estimated_seconds = (
                spec.replay_fixed_seconds_per_run
                + spec.replay_seconds_per_optimizer_observation
                * optimizer_observations
            )
            replay_entries.append(
                {
                    "replica_index": replica,
                    "replica_id": config.replica_id,
                    "condition": condition,
                    "capacity": capacity,
                    "source": source,
                    "config_file": relative_path,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": current["replica_bundle_id"],
                    "paired_current_run_id": current["run_id"],
                    "paired_ewc_run_id": ewc["run_id"],
                    "optimizer_observations": optimizer_observations,
                    "estimated_seconds": estimated_seconds,
                    "estimated_artifact_bytes": spec.replay_estimated_artifact_bytes,
                }
            )

    pilot_entries = [entry for entry in replay_entries if entry["source"] != "new"]
    if len(pilot_entries) != 1 or pilot_entries[0]["run_id"] != phase1_pilot["run_id"]:
        raise Plan3Error("Phase 2 must reuse exactly one accepted Phase 1 pilot")
    identity = {
        "schema_version": PLAN3_PHASE2_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "stage": "replay-screen",
        "source_confirmation_bundle": spec.confirmation_bundle,
        "source_phase1_bundle": phase1.bundle_id,
        "replica_indices": list(spec.replica_indices),
        "capacities": list(capacities),
        "replay_config_hashes": [entry["config_hash"] for entry in replay_entries],
        "control_config_hashes": [entry["config_hash"] for entry in control_entries],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-replay-screen__r0006-r0010__{digest[:12]}"
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "replay_entry_count": len(replay_entries),
        "new_run_count": sum(entry["source"] == "new" for entry in replay_entries),
        "reused_pilot_count": len(pilot_entries),
        "control_entry_count": len(control_entries),
        "estimated_remaining_seconds": sum(
            entry["estimated_seconds"]
            for entry in replay_entries
            if entry["source"] == "new"
        ),
        "estimated_remaining_bytes": sum(
            entry["estimated_artifact_bytes"]
            for entry in replay_entries
            if entry["source"] == "new"
        ),
        "audit": {
            "source_manifest_sha256": audit["source_manifest_sha256"],
            "confirmation_manifest_sha256": audit["confirmation_manifest_sha256"],
            "phase1_manifest_sha256": hashlib.sha256(
                (phase1.path / "bundle.json").read_bytes()
            ).hexdigest(),
        },
        "controls": control_entries,
        "entries": replay_entries,
    }
    return manifest, configs


def prepare_plan3_phase2_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase2Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase2_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase2_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 2 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase2Bundle(destination, manifest)


def load_plan3_phase2_bundle(path: str | Path) -> Plan3Phase2Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 2 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE2_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 2 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 2 bundle ID does not match its directory")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get(
        "replay_entry_count"
    ):
        raise Plan3Error("Phase 2 replay entry count is invalid")
    controls = manifest.get("controls")
    if not isinstance(controls, list) or len(controls) != manifest.get(
        "control_entry_count"
    ):
        raise Plan3Error("Phase 2 control entry count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 2 config identity mismatch: {config_path}")
    return Plan3Phase2Bundle(bundle_path, manifest)


def plan3_phase2_status_rows(
    bundle: Plan3Phase2Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    controls_by_run = {}
    for control in bundle.manifest["controls"]:
        path = root / control["cache_root"] / control["run_id"]
        controls_by_run[control["run_id"]] = (
            "completed" if (path / "COMPLETED").is_file() else "missing"
        )
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state = "completed"
            run_path = final_path
        elif incomplete_path.exists():
            state = "incomplete"
            run_path = incomplete_path
        elif final_path.exists():
            state = "invalid"
            run_path = final_path
        else:
            state = "missing"
            run_path = final_path
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": state,
                "current_control_state": controls_by_run.get(
                    entry["paired_current_run_id"], "missing"
                ),
                "ewc_control_state": controls_by_run.get(
                    entry["paired_ewc_run_id"], "missing"
                ),
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase3Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _phase3_profiles() -> tuple[dict[str, Any], ...]:
    profiles = []
    for device in ("cpu", "cuda"):
        for capacity, label, max_steps, archived_events in (
            (0, "b000", 3, 16),
            (8, "b008", 3, 8),
            (25, "b025", 5, 7),
            ("unbounded", "unbounded", 3, 0),
        ):
            profiles.append(
                {
                    "name": f"{device}-{label}-smoke",
                    "device": device,
                    "capacity": capacity,
                    "max_steps": max_steps,
                    "expected_archived_online_events": archived_events,
                    "role": "hybrid-transaction-smoke",
                }
            )
    return tuple(profiles)


def build_plan3_phase3_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    phase2_path = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE2_ACCEPTED_BUNDLE_ID
    )
    phase2 = load_plan3_phase2_bundle(phase2_path)
    phase2_rows = plan3_phase2_status_rows(phase2, root)
    if any(row["run_state"] != "completed" for row in phase2_rows):
        raise Plan3Error("accepted Phase 2 bundle is not complete")
    replica_index = spec.replica_indices[0]
    source = next(
        (
            row
            for row in phase2.manifest["controls"]
            if row["replica_index"] == replica_index
            and row["condition"] == "ewc-fixed005-no-lfu"
        ),
        None,
    )
    if source is None:
        raise Plan3Error("Phase 3 fixed-EWC archive source is missing")
    source_run_path = root / source["cache_root"] / source["run_id"]
    source_config_path = source_run_path / "config.json"
    source_artifact_path = source_run_path / "phase8_checkpoints.pt"
    if not (source_run_path / "COMPLETED").is_file():
        raise Plan3Error("Phase 3 fixed-EWC archive source is incomplete")
    if not source_config_path.is_file() or not source_artifact_path.is_file():
        raise Plan3Error("Phase 3 fixed-EWC archive source files are missing")
    source_mapping = _read_json(source_config_path)
    relative_source_artifact = str(source_artifact_path.relative_to(root))
    configs: dict[str, dict[str, Any]] = {}
    entries = []
    for profile in _phase3_profiles():
        mapping = json.loads(json.dumps(source_mapping))
        mapping.update(
            {
                "schema_version": 14,
                "artifact_schema_version": 6,
                "metric_schema_version": 10,
                "cache_root": "cache/mnist_experiment/plan3_runs",
                "experiment": f"mnist_lfu_plan3-phase3_{profile['name']}",
                "replay": {
                    "capacity": profile["capacity"],
                    "policy": "fifo",
                    "max_steps": profile["max_steps"],
                    "mode": "hybrid",
                    "archive_initialization_artifact": relative_source_artifact,
                },
            }
        )
        mapping["runtime"]["device"] = profile["device"]
        config = ExperimentConfig.from_mapping(mapping)
        relative_path = f"configs/{profile['name']}.json"
        configs[relative_path] = config.to_mapping()
        entries.append(
            {
                **profile,
                "config_file": relative_path,
                "config_hash": config.config_hash,
                "run_id": config.run_id,
                "cache_root": config.cache_root,
                "replica_id": config.replica_id,
                "replica_index": replica_index,
                "replica_bundle_id": source["replica_bundle_id"],
                "archive_source_run_id": source["run_id"],
                "archive_source_config_hash": source["config_hash"],
                "archive_source_artifact": relative_source_artifact,
                "archive_source_artifact_sha256": hashlib.sha256(
                    source_artifact_path.read_bytes()
                ).hexdigest(),
            }
        )
    identity = {
        "schema_version": PLAN3_PHASE3_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "phase": 3,
        "source_phase2_bundle": phase2.bundle_id,
        "replica_index": replica_index,
        "profiles": [entry["name"] for entry in entries],
        "config_hashes": [entry["config_hash"] for entry in entries],
        "archive_source_artifact_sha256": entries[0][
            "archive_source_artifact_sha256"
        ],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-phase3__r{replica_index:04d}__{digest[:12]}"
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "config_count": len(entries),
        "entries": entries,
    }
    return manifest, configs


def prepare_plan3_phase3_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase3Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase3_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase3_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 3 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase3Bundle(destination, manifest)


def load_plan3_phase3_bundle(path: str | Path) -> Plan3Phase3Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 3 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE3_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 3 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 3 bundle ID does not match its directory")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get("config_count"):
        raise Plan3Error("Phase 3 config count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 3 config identity mismatch: {config_path}")
    return Plan3Phase3Bundle(bundle_path, manifest)


def plan3_phase3_status_rows(
    bundle: Plan3Phase3Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state = "completed"
            run_path = final_path
        elif incomplete_path.exists():
            state = "incomplete"
            run_path = incomplete_path
        elif final_path.exists():
            state = "invalid"
            run_path = final_path
        else:
            state = "missing"
            run_path = final_path
        source_artifact = root / entry["archive_source_artifact"]
        source_state = "missing"
        if source_artifact.is_file():
            digest = hashlib.sha256(source_artifact.read_bytes()).hexdigest()
            source_state = (
                "completed"
                if digest == entry["archive_source_artifact_sha256"]
                else "invalid"
            )
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": state,
                "archive_source_state": source_state,
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase4Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _phase4_new_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "condition": "replay-memory-matched",
            "runner": "replay",
            "capacity": 25,
            "comparison": "ewc-fixed005-no-lfu",
        },
        {
            "condition": "hybrid-b008-no-lfu",
            "runner": "hybrid",
            "capacity": 8,
            "comparison": "replay-b008",
        },
        {
            "condition": "hybrid-memory-matched-no-lfu",
            "runner": "hybrid",
            "capacity": 25,
            "comparison": "replay-memory-matched",
        },
        {
            "condition": "hybrid-selected-no-lfu",
            "runner": "hybrid",
            "capacity": 32,
            "comparison": "replay-selected",
        },
    )


def build_plan3_phase4_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    phase2 = load_plan3_phase2_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE2_ACCEPTED_BUNDLE_ID
    )
    phase2_status = plan3_phase2_status_rows(phase2, root)
    if any(row["run_state"] != "completed" for row in phase2_status):
        raise Plan3Error("accepted Phase 2 replay bundle is incomplete")
    if any(
        row["current_control_state"] != "completed"
        or row["ewc_control_state"] != "completed"
        for row in phase2_status
    ):
        raise Plan3Error("accepted Phase 2 control dependencies are incomplete")

    replay_by_key = {
        (entry["replica_index"], entry["condition"]): entry
        for entry in phase2.manifest["entries"]
    }
    control_by_key = {
        (entry["replica_index"], entry["condition"]): entry
        for entry in phase2.manifest["controls"]
    }
    reused_conditions = (
        ("ewc-fixed005-no-lfu", "ewc-fixed005-no-lfu"),
        ("replay-b008", "replay-b008"),
        ("replay-selected", "replay-b032"),
        ("replay-unbounded", "replay-unbounded"),
    )
    configs: dict[str, dict[str, Any]] = {}
    controls = []
    entries = []
    for replica in spec.replica_indices:
        ewc = control_by_key.get((replica, "ewc-fixed005-no-lfu"))
        replay_b032 = replay_by_key.get((replica, "replay-b032"))
        if ewc is None or replay_b032 is None:
            raise Plan3Error(f"Phase 4 sources are missing for replica {replica}")
        for condition, source_condition in reused_conditions:
            source = (
                control_by_key.get((replica, source_condition))
                if source_condition == "ewc-fixed005-no-lfu"
                else replay_by_key.get((replica, source_condition))
            )
            if source is None:
                raise Plan3Error(
                    f"Phase 4 reused {source_condition} is missing for replica {replica}"
                )
            controls.append(
                {
                    "replica_index": replica,
                    "replica_id": source["replica_id"],
                    "condition": condition,
                    "source_condition": source_condition,
                    "artifact_kind": (
                        "control"
                        if source_condition == "ewc-fixed005-no-lfu"
                        else "replay"
                    ),
                    "run_id": source["run_id"],
                    "config_hash": source["config_hash"],
                    "cache_root": source["cache_root"],
                    "replica_bundle_id": source["replica_bundle_id"],
                    "source": "accepted-phase2",
                }
            )

        replay_source_mapping = _read_json(
            phase2.path / replay_b032["config_file"]
        )
        ewc_run_path = root / ewc["cache_root"] / ewc["run_id"]
        ewc_config_path = ewc_run_path / "config.json"
        archive_artifact_path = ewc_run_path / "phase8_checkpoints.pt"
        if not ewc_config_path.is_file() or not archive_artifact_path.is_file():
            raise Plan3Error(f"Phase 4 archive source is missing for replica {replica}")
        ewc_source_mapping = _read_json(ewc_config_path)
        relative_archive_artifact = str(archive_artifact_path.relative_to(root))
        archive_sha256 = hashlib.sha256(archive_artifact_path.read_bytes()).hexdigest()

        for profile in _phase4_new_profiles():
            condition = profile["condition"]
            capacity = profile["capacity"]
            runner = profile["runner"]
            if runner == "replay":
                mapping = json.loads(json.dumps(replay_source_mapping))
                mapping.update(
                    {
                        "cache_root": "cache/mnist_experiment/plan3_runs",
                        "experiment": "mnist_lfu_plan3-phase4_replay-memory-matched",
                    }
                )
                mapping["replay"]["capacity"] = capacity
                archive_source = None
                archive_source_sha256 = None
            else:
                mapping = json.loads(json.dumps(ewc_source_mapping))
                mapping.update(
                    {
                        "schema_version": 14,
                        "artifact_schema_version": 6,
                        "metric_schema_version": 10,
                        "cache_root": "cache/mnist_experiment/plan3_runs",
                        "experiment": f"mnist_lfu_plan3-phase4_{condition}",
                        "replay": {
                            "capacity": capacity,
                            "policy": "fifo",
                            "max_steps": None,
                            "mode": "hybrid",
                            "archive_initialization_artifact": (
                                relative_archive_artifact
                            ),
                        },
                    }
                )
                archive_source = relative_archive_artifact
                archive_source_sha256 = archive_sha256
            mapping["runtime"]["device"] = "cuda"
            config = ExperimentConfig.from_mapping(mapping)
            if config.replay is None or config.replay.capacity != capacity:
                raise Plan3Error("Phase 4 capacity was not encoded exactly")
            relative_path = f"configs/replica-{replica:04d}/{condition}.json"
            configs[relative_path] = config.to_mapping()
            optimizer_observations = _replay_optimizer_observations(capacity, spec)
            estimated_seconds = (
                spec.replay_fixed_seconds_per_run
                + spec.replay_seconds_per_optimizer_observation
                * optimizer_observations
                if runner == "replay"
                else spec.fixed_seconds_per_run
                + spec.seconds_per_optimizer_observation * optimizer_observations
            )
            entries.append(
                {
                    "replica_index": replica,
                    "replica_id": config.replica_id,
                    "condition": condition,
                    "runner": runner,
                    "capacity": capacity,
                    "comparison": profile["comparison"],
                    "config_file": relative_path,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": ewc["replica_bundle_id"],
                    "archive_source_run_id": (
                        ewc["run_id"] if runner == "hybrid" else None
                    ),
                    "archive_source_artifact": archive_source,
                    "archive_source_artifact_sha256": archive_source_sha256,
                    "optimizer_observations": optimizer_observations,
                    "estimated_seconds": estimated_seconds,
                    "estimated_artifact_bytes": (
                        spec.replay_estimated_artifact_bytes
                        if runner == "replay"
                        else spec.estimated_artifact_bytes
                    ),
                }
            )

    identity = {
        "schema_version": PLAN3_PHASE4_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "phase": 4,
        "stage": "memory-hybrid",
        "source_phase2_bundle": phase2.bundle_id,
        "replica_indices": list(spec.replica_indices),
        "condition_order": [
            "ewc-fixed005-no-lfu",
            "replay-b008",
            "replay-memory-matched",
            "replay-selected",
            "hybrid-b008-no-lfu",
            "hybrid-memory-matched-no-lfu",
            "hybrid-selected-no-lfu",
            "replay-unbounded",
        ],
        "new_config_hashes": [entry["config_hash"] for entry in entries],
        "reused_config_hashes": [entry["config_hash"] for entry in controls],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-history-frontier__r0006-r0010__{digest[:12]}"
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "new_entry_count": len(entries),
        "reused_entry_count": len(controls),
        "estimated_remaining_seconds": sum(
            float(entry["estimated_seconds"]) for entry in entries
        ),
        "estimated_remaining_bytes": sum(
            int(entry["estimated_artifact_bytes"]) for entry in entries
        ),
        "controls": controls,
        "entries": entries,
    }
    return manifest, configs


def prepare_plan3_phase4_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase4Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase4_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase4_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 4 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase4Bundle(destination, manifest)


def load_plan3_phase4_bundle(path: str | Path) -> Plan3Phase4Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 4 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE4_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 4 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 4 bundle ID does not match its directory")
    entries = manifest.get("entries")
    controls = manifest.get("controls")
    if not isinstance(entries, list) or len(entries) != manifest.get(
        "new_entry_count"
    ):
        raise Plan3Error("Phase 4 new-entry count is invalid")
    if not isinstance(controls, list) or len(controls) != manifest.get(
        "reused_entry_count"
    ):
        raise Plan3Error("Phase 4 reused-entry count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 4 config identity mismatch: {config_path}")
    return Plan3Phase4Bundle(bundle_path, manifest)


def plan3_phase4_status_rows(
    bundle: Plan3Phase4Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    controls_complete = all(
        (root / row["cache_root"] / row["run_id"] / "COMPLETED").is_file()
        for row in bundle.manifest["controls"]
    )
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state = "completed"
            run_path = final_path
        elif incomplete_path.exists():
            state = "incomplete"
            run_path = incomplete_path
        elif final_path.exists():
            state = "invalid"
            run_path = final_path
        else:
            state = "missing"
            run_path = final_path
        archive_source_state = "not-applicable"
        if entry["runner"] == "hybrid":
            source = root / entry["archive_source_artifact"]
            if not source.is_file():
                archive_source_state = "missing"
            else:
                archive_source_state = (
                    "completed"
                    if hashlib.sha256(source.read_bytes()).hexdigest()
                    == entry["archive_source_artifact_sha256"]
                    else "invalid"
                )
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": state,
                "reused_controls_state": (
                    "completed" if controls_complete else "missing"
                ),
                "archive_source_state": archive_source_state,
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase5Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _phase5_new_profiles() -> tuple[dict[str, str], ...]:
    return (
        {
            "condition": "ewc-fixed005-ac-only",
            "runner": "controller",
            "method": "ac_only",
            "comparison": "ewc-fixed005-no-lfu",
        },
        {
            "condition": "ewc-fixed005-full-lfu",
            "runner": "controller",
            "method": "full_lfu",
            "comparison": "ewc-fixed005-no-lfu",
        },
        {
            "condition": "hybrid-selected-full-lfu",
            "runner": "hybrid",
            "method": "full_lfu",
            "comparison": "hybrid-selected-no-lfu",
        },
    )


def build_plan3_phase5_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    phase4 = load_plan3_phase4_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE4_ACCEPTED_BUNDLE_ID
    )
    statuses = plan3_phase4_status_rows(phase4, root)
    if any(row["run_state"] != "completed" for row in statuses):
        raise Plan3Error("accepted Phase 4 treatment bundle is incomplete")
    if any(row["reused_controls_state"] != "completed" for row in statuses):
        raise Plan3Error("accepted Phase 4 controls are incomplete")

    ewc_by_replica = {
        int(row["replica_index"]): row
        for row in phase4.manifest["controls"]
        if row["condition"] == "ewc-fixed005-no-lfu"
    }
    hybrid_by_replica = {
        int(row["replica_index"]): row
        for row in phase4.manifest["entries"]
        if row["condition"] == "hybrid-selected-no-lfu"
    }
    controls = []
    entries = []
    configs: dict[str, dict[str, Any]] = {}
    for replica in spec.replica_indices:
        ewc = ewc_by_replica.get(replica)
        hybrid = hybrid_by_replica.get(replica)
        if ewc is None or hybrid is None:
            raise Plan3Error(f"Phase 5 sources are missing for replica {replica}")
        for condition, source, artifact_kind in (
            ("ewc-fixed005-no-lfu", ewc, "control"),
            ("hybrid-selected-no-lfu", hybrid, "hybrid"),
        ):
            controls.append(
                {
                    "replica_index": replica,
                    "replica_id": source["replica_id"],
                    "condition": condition,
                    "artifact_kind": artifact_kind,
                    "run_id": source["run_id"],
                    "config_hash": source["config_hash"],
                    "cache_root": source["cache_root"],
                    "replica_bundle_id": source["replica_bundle_id"],
                    "source": "accepted-phase4",
                }
            )

        ewc_mapping = _read_json(root / ewc["cache_root"] / ewc["run_id"] / "config.json")
        hybrid_mapping = _read_json(phase4.path / hybrid["config_file"])
        for profile in _phase5_new_profiles():
            runner = profile["runner"]
            mapping = json.loads(
                json.dumps(ewc_mapping if runner == "controller" else hybrid_mapping)
            )
            mapping.update(
                {
                    "cache_root": "cache/mnist_experiment/plan3_runs",
                    "experiment": f"mnist_lfu_plan3-phase5_{profile['condition']}",
                }
            )
            mapping["estimator"]["method"] = profile["method"]
            if runner == "hybrid":
                mapping.update(
                    {
                        "schema_version": 15,
                        "artifact_schema_version": 7,
                        "metric_schema_version": 11,
                    }
                )
            mapping["runtime"]["device"] = "cuda"
            config = ExperimentConfig.from_mapping(mapping)
            relative_path = (
                f"configs/replica-{replica:04d}/{profile['condition']}.json"
            )
            configs[relative_path] = config.to_mapping()
            entries.append(
                {
                    "replica_index": replica,
                    "replica_id": config.replica_id,
                    "condition": profile["condition"],
                    "runner": runner,
                    "method": profile["method"],
                    "comparison": profile["comparison"],
                    "config_file": relative_path,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": ewc["replica_bundle_id"],
                    "archive_source_artifact": (
                        None
                        if runner == "controller"
                        else mapping["replay"]["archive_initialization_artifact"]
                    ),
                    "estimated_seconds": spec.fixed_seconds_per_run,
                    "estimated_artifact_bytes": spec.estimated_artifact_bytes,
                }
            )

    identity = {
        "schema_version": PLAN3_PHASE5_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "phase": 5,
        "stage": "lfu-isolation",
        "source_phase4_bundle": phase4.bundle_id,
        "replica_indices": list(spec.replica_indices),
        "condition_order": [
            "ewc-fixed005-no-lfu",
            "ewc-fixed005-ac-only",
            "ewc-fixed005-full-lfu",
            "hybrid-selected-no-lfu",
            "hybrid-selected-full-lfu",
        ],
        "new_config_hashes": [entry["config_hash"] for entry in entries],
        "reused_config_hashes": [entry["config_hash"] for entry in controls],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-lfu-isolation__r0006-r0010__{digest[:12]}"
    return {
        **identity,
        "bundle_id": bundle_id,
        "new_entry_count": len(entries),
        "reused_entry_count": len(controls),
        "estimated_remaining_seconds": sum(
            float(entry["estimated_seconds"]) for entry in entries
        ),
        "estimated_remaining_bytes": sum(
            int(entry["estimated_artifact_bytes"]) for entry in entries
        ),
        "controls": controls,
        "entries": entries,
    }, configs


def prepare_plan3_phase5_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase5Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase5_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase5_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 5 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase5Bundle(destination, manifest)


def load_plan3_phase5_bundle(path: str | Path) -> Plan3Phase5Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 5 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE5_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 5 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 5 bundle ID does not match its directory")
    entries = manifest.get("entries")
    controls = manifest.get("controls")
    if not isinstance(entries, list) or len(entries) != manifest.get("new_entry_count"):
        raise Plan3Error("Phase 5 new-entry count is invalid")
    if not isinstance(controls, list) or len(controls) != manifest.get("reused_entry_count"):
        raise Plan3Error("Phase 5 reused-entry count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 5 config identity mismatch: {config_path}")
    return Plan3Phase5Bundle(bundle_path, manifest)


def plan3_phase5_status_rows(
    bundle: Plan3Phase5Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    controls_complete = all(
        (root / row["cache_root"] / row["run_id"] / "COMPLETED").is_file()
        for row in bundle.manifest["controls"]
    )
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state, run_path = "completed", final_path
        elif incomplete_path.exists():
            state, run_path = "incomplete", incomplete_path
        elif final_path.exists():
            state, run_path = "invalid", final_path
        else:
            state, run_path = "missing", final_path
        archive_state = "not-applicable"
        source = entry.get("archive_source_artifact")
        if source is not None:
            archive_state = "completed" if (root / source).is_file() else "missing"
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": state,
                "reused_controls_state": (
                    "completed" if controls_complete else "missing"
                ),
                "archive_source_state": archive_state,
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase6Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _phase6_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "condition": "deployment-current-only",
            "runner": "replay",
            "capacity": 0,
            "controller": "none",
            "comparison": None,
            "estimated_seconds": 50.0,
        },
        {
            "condition": "deployment-ewc-fixed005",
            "runner": "hybrid",
            "capacity": 0,
            "controller": "fixed",
            "comparison": "deployment-current-only",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "deployment-hybrid-b032-fixed005",
            "runner": "hybrid",
            "capacity": 32,
            "controller": "fixed",
            "comparison": "deployment-ewc-fixed005",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "deployment-ewc-adaptive-h020",
            "runner": "hybrid",
            "capacity": 0,
            "controller": "adaptive",
            "comparison": "deployment-ewc-fixed005",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "deployment-hybrid-b032-adaptive-h020",
            "runner": "hybrid",
            "capacity": 32,
            "controller": "adaptive",
            "comparison": "deployment-hybrid-b032-fixed005",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "deployment-replay-b032",
            "runner": "replay",
            "capacity": 32,
            "controller": "none",
            "comparison": "deployment-current-only",
            "estimated_seconds": 60.0,
        },
        {
            "condition": "deployment-replay-unbounded",
            "runner": "replay",
            "capacity": "unbounded",
            "controller": "none",
            "comparison": "deployment-replay-b032",
            "estimated_seconds": 75.0,
        },
    )


def build_plan3_phase6_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    phase4 = load_plan3_phase4_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE4_ACCEPTED_BUNDLE_ID
    )
    phase4_statuses = plan3_phase4_status_rows(phase4, root)
    if any(row["run_state"] != "completed" for row in phase4_statuses):
        raise Plan3Error("accepted Phase 4 treatment bundle is incomplete")
    if any(row["reused_controls_state"] != "completed" for row in phase4_statuses):
        raise Plan3Error("accepted Phase 4 controls are incomplete")
    phase5 = load_plan3_phase5_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE5_ACCEPTED_BUNDLE_ID
    )
    phase5_statuses = plan3_phase5_status_rows(phase5, root)
    pilot = [row for row in phase5_statuses if int(row["replica_index"]) == 6]
    if len(pilot) != 3 or any(row["run_state"] != "completed" for row in pilot):
        raise Plan3Error("Phase 6 requires the completed Phase 5 preflight gate")

    phase4_controls = {
        (int(row["replica_index"]), str(row["condition"])): row
        for row in phase4.manifest["controls"]
    }
    phase4_hybrids = {
        (int(row["replica_index"]), str(row["condition"])): row
        for row in phase4.manifest["entries"]
        if row["runner"] == "hybrid"
    }
    configs: dict[str, dict[str, Any]] = {}
    entries = []
    for replica in spec.replica_indices:
        ewc = phase4_controls.get((replica, "ewc-fixed005-no-lfu"))
        replay = phase4_controls.get((replica, "replay-selected"))
        hybrid = phase4_hybrids.get((replica, "hybrid-selected-no-lfu"))
        if ewc is None or replay is None or hybrid is None:
            raise Plan3Error(f"Phase 6 sources are missing for replica {replica}")
        ewc_path = root / ewc["cache_root"] / ewc["run_id"]
        replay_path = root / replay["cache_root"] / replay["run_id"]
        archive_path = ewc_path / "phase8_checkpoints.pt"
        if not archive_path.is_file():
            raise Plan3Error(f"Phase 6 archive source is missing for replica {replica}")
        archive_relative = str(archive_path.relative_to(root))
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        ewc_mapping = _read_json(ewc_path / "config.json")
        replay_mapping = _read_json(replay_path / "config.json")

        for profile in _phase6_profiles():
            runner = profile["runner"]
            mapping = json.loads(
                json.dumps(replay_mapping if runner == "replay" else ewc_mapping)
            )
            mapping.update(
                {
                    "cache_root": "cache/mnist_experiment/plan3_runs",
                    "experiment": f"mnist_lfu_plan3-phase6_{profile['condition']}",
                }
            )
            mapping["runtime"]["device"] = "cuda"
            if runner == "replay":
                mapping.update(
                    {
                        "schema_version": 13,
                        "artifact_schema_version": 5,
                        "metric_schema_version": 9,
                        "replay": {
                            "capacity": profile["capacity"],
                            "policy": "fifo",
                            "max_steps": None,
                        },
                    }
                )
                mapping["controller"].update(
                    {
                        "oracle_mode": "none",
                        "reference_optimum_artifact": None,
                    }
                )
                source_artifact = None
                source_sha256 = None
            else:
                adaptive = profile["controller"] == "adaptive"
                mapping.update(
                    {
                        "schema_version": 16,
                        "artifact_schema_version": 8,
                        "metric_schema_version": 12,
                        "replay": {
                            "capacity": profile["capacity"],
                            "policy": "fifo",
                            "max_steps": None,
                            "mode": "hybrid",
                            "archive_initialization_artifact": archive_relative,
                        },
                    }
                )
                mapping["estimator"]["method"] = "ema"
                mapping["controller"].update(
                    {
                        "policy": "optimal_plugin" if adaptive else "fixed_unified",
                        "fixed_pi": 0.05,
                        "pi_min": 0.05,
                        "pi_max": 0.95 if adaptive else 0.05,
                        "trend_half_life_p": 0.2,
                        "oracle_mode": "none",
                        "reference_optimum_artifact": None,
                    }
                )
                source_artifact = archive_relative
                source_sha256 = archive_sha256
            config = ExperimentConfig.from_mapping(mapping)
            relative_path = (
                f"configs/replica-{replica:04d}/{profile['condition']}.json"
            )
            configs[relative_path] = config.to_mapping()
            entries.append(
                {
                    "replica_index": replica,
                    "replica_id": config.replica_id,
                    "condition": profile["condition"],
                    "runner": runner,
                    "capacity": profile["capacity"],
                    "controller": profile["controller"],
                    "comparison": profile["comparison"],
                    "fisher_update": "ema_no_lfu" if runner == "hybrid" else "none",
                    "oracle_free": True,
                    "expected_hvp_count": 0,
                    "config_file": relative_path,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": ewc["replica_bundle_id"],
                    "archive_source_artifact": source_artifact,
                    "archive_source_artifact_sha256": source_sha256,
                    "estimated_seconds": profile["estimated_seconds"],
                    "estimated_artifact_bytes": (
                        spec.replay_estimated_artifact_bytes
                        if runner == "replay"
                        else spec.estimated_artifact_bytes
                    ),
                }
            )

    identity = {
        "schema_version": PLAN3_PHASE6_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "phase": 6,
        "stage": "deployment-frontier",
        "source_phase4_bundle": phase4.bundle_id,
        "source_phase5_bundle": phase5.bundle_id,
        "fisher_update_decision": "ema_no_lfu",
        "replica_indices": list(spec.replica_indices),
        "condition_order": [row["condition"] for row in _phase6_profiles()],
        "config_hashes": [entry["config_hash"] for entry in entries],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = f"plan3-deployment-frontier__r0006-r0010__{digest[:12]}"
    return {
        **identity,
        "bundle_id": bundle_id,
        "entry_count": len(entries),
        "estimated_seconds": sum(float(row["estimated_seconds"]) for row in entries),
        "estimated_artifact_bytes": sum(
            int(row["estimated_artifact_bytes"]) for row in entries
        ),
        "phase4_manifest_sha256": hashlib.sha256(
            (phase4.path / "bundle.json").read_bytes()
        ).hexdigest(),
        "phase5_manifest_sha256": hashlib.sha256(
            (phase5.path / "bundle.json").read_bytes()
        ).hexdigest(),
        "entries": entries,
    }, configs


def prepare_plan3_phase6_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
) -> Plan3Phase6Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase6_bundle(spec, root)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase6_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 6 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase6Bundle(destination, manifest)


def load_plan3_phase6_bundle(path: str | Path) -> Plan3Phase6Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 6 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE6_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 6 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 6 bundle ID does not match its directory")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != manifest.get("entry_count"):
        raise Plan3Error("Phase 6 entry count is invalid")
    for entry in entries:
        config_path = bundle_path / entry["config_file"]
        config = ExperimentConfig.from_mapping(_read_json(config_path))
        if config.config_hash != entry["config_hash"] or config.run_id != entry["run_id"]:
            raise Plan3Error(f"Phase 6 config identity mismatch: {config_path}")
    return Plan3Phase6Bundle(bundle_path, manifest)


def plan3_phase6_status_rows(
    bundle: Plan3Phase6Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in bundle.manifest["entries"]:
        run_root = root / entry["cache_root"]
        final_path = run_root / entry["run_id"]
        incomplete_path = run_root / ".incomplete" / entry["run_id"]
        if (final_path / "COMPLETED").is_file():
            state, run_path = "completed", final_path
        elif incomplete_path.exists():
            state, run_path = "incomplete", incomplete_path
        elif final_path.exists():
            state, run_path = "invalid", final_path
        else:
            state, run_path = "missing", final_path
        source_state = "not-applicable"
        source = entry.get("archive_source_artifact")
        if source is not None:
            source_path = root / source
            if not source_path.is_file():
                source_state = "missing"
            else:
                source_state = (
                    "completed"
                    if hashlib.sha256(source_path.read_bytes()).hexdigest()
                    == entry["archive_source_artifact_sha256"]
                    else "invalid"
                )
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": state,
                "archive_source_state": source_state,
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class Plan3Phase7Bundle:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])


def _phase7_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "condition": "confirm-current-only",
            "source_condition": "deployment-current-only",
            "runner": "replay",
            "estimated_seconds": 55.0,
        },
        {
            "condition": "confirm-ewc-fixed005",
            "source_condition": "deployment-ewc-fixed005",
            "runner": "hybrid",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "confirm-hybrid-b032-fixed005",
            "source_condition": "deployment-hybrid-b032-fixed005",
            "runner": "hybrid",
            "estimated_seconds": 100.0,
        },
        {
            "condition": "confirm-replay-b032",
            "source_condition": "deployment-replay-b032",
            "runner": "replay",
            "estimated_seconds": 60.0,
        },
        {
            "condition": "confirm-replay-unbounded",
            "source_condition": "deployment-replay-unbounded",
            "runner": "replay",
            "estimated_seconds": 75.0,
        },
    )


def _phase7_identity(
    mapping: Mapping[str, Any],
    *,
    experiment: str,
    replica_index: int,
    replica_seed: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(mapping))
    result["experiment"] = experiment
    result["replica_id"] = f"replica-{replica_index:04d}"
    result["replica_seed"] = replica_seed
    result["controller"]["oracle_mode"] = "none"
    result["controller"]["reference_optimum_artifact"] = None
    return result


def build_plan3_phase7_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
    replica_indices: Sequence[int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    replicas = tuple(sorted(set(int(value) for value in replica_indices)))
    if not replicas or replicas[0] < 0:
        raise Plan3Error("Phase 7 requires nonnegative replica indices")
    if len(replicas) % 5:
        raise Plan3Error("Phase 7 replicas must be predeclared in blocks of five")
    phase6 = load_plan3_phase6_bundle(
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "bundles"
        / PLAN3_PHASE6_ACCEPTED_BUNDLE_ID
    )
    phase6_statuses = plan3_phase6_status_rows(phase6, root)
    if any(row["run_state"] != "completed" for row in phase6_statuses):
        raise Plan3Error("accepted Phase 6 bundle is incomplete")
    templates: dict[str, dict[str, Any]] = {}
    for entry in phase6.manifest["entries"]:
        condition = str(entry["condition"])
        if condition in templates:
            continue
        templates[condition] = _read_json(
            root / entry["cache_root"] / entry["run_id"] / "config.json"
        )
    expected_sources = {row["source_condition"] for row in _phase7_profiles()}
    if set(templates).isdisjoint(expected_sources) or not expected_sources.issubset(
        templates
    ):
        raise Plan3Error("Phase 7 source templates are incomplete")

    phase9 = load_phase9_spec(spec.path.parent / "phase9_profiles.json")
    configs: dict[str, dict[str, Any]] = {}
    anchors = []
    entries = []
    for replica in replicas:
        replica_seed = derive_component_seed(
            phase9.base_replica_seed,
            f"phase9_replica:{replica}",
        )
        anchor_mapping = _phase7_identity(
            templates["deployment-current-only"],
            experiment="mnist_lfu_plan3-phase7_initial-archive",
            replica_index=replica,
            replica_seed=replica_seed,
        )
        anchor_mapping["cache_root"] = (
            "cache/mnist_experiment/plan3_initial_archives"
        )
        anchor_mapping["replay"] = {
            "capacity": 0,
            "policy": "fifo",
            "max_steps": None,
        }
        anchor_config = ExperimentConfig.from_mapping(anchor_mapping)
        anchor_relative = f"anchors/replica-{replica:04d}.json"
        configs[anchor_relative] = anchor_config.to_mapping()
        archive_relative = str(
            Path(anchor_config.cache_root)
            / anchor_config.run_id
            / "initial_archive.pt"
        )
        anchors.append(
            {
                "replica_index": replica,
                "replica_id": anchor_config.replica_id,
                "replica_seed": replica_seed,
                "config_file": anchor_relative,
                "config_hash": anchor_config.config_hash,
                "run_id": anchor_config.run_id,
                "cache_root": anchor_config.cache_root,
                "replica_bundle_id": replica_bundle_id(anchor_config),
                "archive_artifact": archive_relative,
                "estimated_seconds": 45.0,
            }
        )
        for profile in _phase7_profiles():
            mapping = _phase7_identity(
                templates[profile["source_condition"]],
                experiment=f"mnist_lfu_plan3-phase7_{profile['condition']}",
                replica_index=replica,
                replica_seed=replica_seed,
            )
            mapping["cache_root"] = "cache/mnist_experiment/plan3_runs"
            if profile["runner"] == "hybrid":
                mapping["replay"]["archive_initialization_artifact"] = (
                    archive_relative
                )
            config = ExperimentConfig.from_mapping(mapping)
            relative = f"configs/replica-{replica:04d}/{profile['condition']}.json"
            configs[relative] = config.to_mapping()
            entries.append(
                {
                    "replica_index": replica,
                    "replica_id": config.replica_id,
                    "condition": profile["condition"],
                    "runner": profile["runner"],
                    "source_phase6_condition": profile["source_condition"],
                    "config_file": relative,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": replica_bundle_id(config),
                    "archive_source_artifact": (
                        archive_relative if profile["runner"] == "hybrid" else None
                    ),
                    "estimated_seconds": profile["estimated_seconds"],
                }
            )

    identity = {
        "schema_version": PLAN3_PHASE7_BUNDLE_SCHEMA_VERSION,
        "spec_name": spec.name,
        "phase": 7,
        "stage": "fresh-confirmation",
        "source_phase6_bundle": phase6.bundle_id,
        "replica_indices": list(replicas),
        "condition_order": [row["condition"] for row in _phase7_profiles()],
        "block_size": 5,
        "initial_replica_target": 10,
        "maximum_replica_count": len(replicas),
        "config_hashes": [row["config_hash"] for row in anchors]
        + [row["config_hash"] for row in entries],
        "analysis_contract": {
            "path": "unchanged 100-point linear p trajectory from 0 to 1",
            "samples_per_step": 8,
            "principal_plot_domain": "0<=p<0.5",
            "auc_domain": "0<=p<0.5",
            "primary_contrast": (
                "confirm-hybrid-b032-fixed005 minus confirm-replay-b032"
            ),
            "primary_metric": "environment_accuracy AUC",
            "primary_auc_ci_half_width_target": 0.02,
            "median_pointwise_ci_half_width_target": 0.03,
            "nine_ovr_practical_equivalence_margin": 0.03,
            "ci": "two-sided 95% Student-t intervals across paired replicas",
        },
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    bundle_id = (
        f"plan3-fresh-confirmation__r{replicas[0]:04d}-r{replicas[-1]:04d}__"
        f"{digest[:12]}"
    )
    return {
        **identity,
        "bundle_id": bundle_id,
        "anchor_count": len(anchors),
        "entry_count": len(entries),
        "estimated_seconds": sum(float(row["estimated_seconds"]) for row in anchors)
        + sum(float(row["estimated_seconds"]) for row in entries),
        "phase6_manifest_sha256": hashlib.sha256(
            (phase6.path / "bundle.json").read_bytes()
        ).hexdigest(),
        "anchors": anchors,
        "entries": entries,
    }, configs


def prepare_plan3_phase7_bundle(
    spec: Plan3Spec,
    repo_root: str | Path,
    replica_indices: Sequence[int],
) -> Plan3Phase7Bundle:
    root = Path(repo_root)
    manifest, configs = build_plan3_phase7_bundle(spec, root, replica_indices)
    bundle_root = root / "cache" / "mnist_experiment" / "plan3" / "bundles"
    destination = bundle_root / manifest["bundle_id"]
    if destination.exists():
        loaded = load_plan3_phase7_bundle(destination)
        if loaded.manifest != manifest:
            raise Plan3Error("existing Phase 7 bundle has incompatible contents")
        return loaded
    incomplete = bundle_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{manifest['bundle_id']}.", dir=incomplete))
    try:
        for relative_path, mapping in configs.items():
            config_path = temporary / relative_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Plan3Phase7Bundle(destination, manifest)


def load_plan3_phase7_bundle(path: str | Path) -> Plan3Phase7Bundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise Plan3Error(f"Phase 7 bundle is incomplete: {bundle_path}")
    manifest = _read_json(bundle_path / "bundle.json")
    if manifest.get("schema_version") != PLAN3_PHASE7_BUNDLE_SCHEMA_VERSION:
        raise Plan3Error("unsupported Phase 7 bundle schema")
    if manifest.get("bundle_id") != bundle_path.name:
        raise Plan3Error("Phase 7 bundle ID does not match its directory")
    if len(manifest.get("anchors", [])) != manifest.get("anchor_count"):
        raise Plan3Error("Phase 7 anchor count is invalid")
    if len(manifest.get("entries", [])) != manifest.get("entry_count"):
        raise Plan3Error("Phase 7 entry count is invalid")
    for row in (*manifest["anchors"], *manifest["entries"]):
        config = ExperimentConfig.from_mapping(_read_json(bundle_path / row["config_file"]))
        if config.config_hash != row["config_hash"] or config.run_id != row["run_id"]:
            raise Plan3Error(f"Phase 7 config identity mismatch: {row['config_file']}")
    return Plan3Phase7Bundle(bundle_path, manifest)


def _phase7_run_state(root: Path, cache_root: str, run_id: str) -> tuple[str, Path]:
    run_root = root / cache_root
    final = run_root / run_id
    incomplete = run_root / ".incomplete" / run_id
    if (final / "COMPLETED").is_file():
        return "completed", final
    if incomplete.exists():
        return "incomplete", incomplete
    if final.exists():
        return "invalid", final
    return "missing", final


def plan3_phase7_status_rows(
    bundle: Plan3Phase7Bundle,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    anchors = {int(row["replica_index"]): row for row in bundle.manifest["anchors"]}
    rows = []
    for entry in bundle.manifest["entries"]:
        replica = int(entry["replica_index"])
        anchor = anchors[replica]
        replica_path = root / "cache" / "mnist_experiment" / "replicas" / anchor[
            "replica_bundle_id"
        ]
        if (replica_path / "COMPLETED").is_file():
            replica_state = "completed"
        elif replica_path.exists():
            replica_state = "invalid"
        else:
            replica_state = "missing"
        anchor_state, anchor_path = _phase7_run_state(
            root, anchor["cache_root"], anchor["run_id"]
        )
        archive_path = root / anchor["archive_artifact"]
        archive_state = (
            "completed"
            if anchor_state == "completed" and archive_path.is_file()
            else "invalid"
            if anchor_state == "completed"
            else anchor_state
        )
        run_state, run_path = _phase7_run_state(
            root, entry["cache_root"], entry["run_id"]
        )
        rows.append(
            {
                **entry,
                "config_path": str(bundle.path / entry["config_file"]),
                "run_path": str(run_path),
                "run_state": run_state,
                "replica_config_path": str(bundle.path / anchor["config_file"]),
                "replica_bundle_path": str(replica_path),
                "replica_bundle_state": replica_state,
                "anchor_run_id": anchor["run_id"],
                "anchor_run_path": str(anchor_path),
                "anchor_run_state": anchor_state,
                "archive_source_state": (
                    archive_state
                    if entry["runner"] == "hybrid"
                    else "not-applicable"
                ),
            }
        )
    return rows
