"""Validated, planning-only expansion of the Plan 3 handoff design."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PLAN3_SPEC_SCHEMA_VERSION = 1


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
    memory_matched_replay_budget: int
    fixed_seconds_per_run: float
    seconds_per_optimizer_observation: float
    estimated_artifact_bytes: int
    observation_payload_bytes: int
    parameter_scalar_bytes: int
    stages: tuple[Plan3Stage, ...]


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
        {"selected_replay_budget", "memory_matched_replay_budget"},
        "planning_placeholders",
    )
    selected_budget = _positive_int(
        placeholders["selected_replay_budget"],
        "planning_placeholders.selected_replay_budget",
    )
    memory_budget = _positive_int(
        placeholders["memory_matched_replay_budget"],
        "planning_placeholders.memory_matched_replay_budget",
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
            "observation_payload_bytes",
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
    observation_bytes = _positive_int(
        cost["observation_payload_bytes"], "cost_model.observation_payload_bytes"
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
        "upper-control",
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
        memory_matched_replay_budget=memory_budget,
        fixed_seconds_per_run=fixed_seconds,
        seconds_per_optimizer_observation=per_observation,
        estimated_artifact_bytes=artifact_bytes,
        observation_payload_bytes=observation_bytes,
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
        fisher_and_anchor_scalars = spec.parameter_count * (spec.fisher_rank + 2)
        total += fisher_and_anchor_scalars * spec.parameter_scalar_bytes
    if condition.data_mode != "current":
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
            seconds_per_run = (
                spec.fixed_seconds_per_run
                + spec.seconds_per_optimizer_observation * observations
            )
            run_count = len(replicas)
            condition_new_runs = run_count if condition.is_new else 0
            condition_reused_runs = 0 if condition.is_new else run_count
            if condition.is_new:
                stage_seconds += seconds_per_run * run_count
                stage_bytes += spec.estimated_artifact_bytes * run_count
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
        "cost_basis": "plan2_fixed_overhead_plus_optimizer_objective_observations",
        "planning_placeholders": {
            "selected_replay_budget": spec.selected_replay_budget,
            "memory_matched_replay_budget": spec.memory_matched_replay_budget,
        },
        "warnings": [
            "Replay timing is not calibrated; Phase 1 must replace this planning proxy.",
            "Selected and memory-matched replay budgets are count/cost placeholders, not decisions.",
            "LFU and deployment diagnostic savings are not yet assigned separate cost factors.",
            "Gate-selected deployment rows use the selected hybrid budget as an upper-cost placeholder.",
        ],
        "stages": stage_rows,
        "conditions": condition_rows,
    }
