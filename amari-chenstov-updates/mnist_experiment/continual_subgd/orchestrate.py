"""Bounded, resumable orchestration for Plan 13."""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

from mnist_experiment.rotated_mnist.artifacts import _read_json
from mnist_experiment.rotated_mnist.plan12.assets import ASSET_REQUIRED

from .analysis import (
    ANALYSIS_REQUIRED,
    _burn_in_unit,
    _trajectory_unit,
    run_phase0_analysis,
    run_phase1_analysis,
    run_phase2_analysis,
    run_phase3_probe_analysis,
    run_phase3_stage_analysis,
    run_phase4_analysis,
    run_phase5_analysis,
)
from .artifacts import UnitStore, freeze_json
from .conditions import (
    Condition,
    adaptive_floor_conditions,
    condition_from_mapping,
    default_probe_conditions,
    geometry_rate_conditions,
    phase2_conditions,
)
from .config import Plan13Study
from .environments.digit9_mixture import (
    MIXTURE_ASSET_REQUIRED,
    ensure_mixture_assets,
    mixture_asset_unit,
)
from .mixture import (
    MIXTURE_BURN_REQUIRED,
    MIXTURE_TRAJECTORY_REQUIRED,
    ensure_mixture_burn_in,
    mixture_burn_unit,
    mixture_trajectory_unit,
    run_mixture_trajectory,
)
from .refresh_notebook import DEFAULT_NOTEBOOK, refresh
from .rotation import ensure_rotation_assets
from .trajectory import (
    BURN_IN_REQUIRED,
    TRAJECTORY_REQUIRED,
    ensure_burn_in,
    run_rotation_trajectory,
)


DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
PHASES = ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5")


def _item(
    action: str,
    unit: dict[str, Any],
    required: tuple[str, ...],
    **parameters: Any,
) -> dict[str, Any]:
    return {
        "action": action,
        "unit": unit,
        "required": list(required),
        "parameters": parameters,
    }


def _asset_item(store: UnitStore, phase: str, index: int) -> dict[str, Any]:
    return _item(
        "assets",
        store.unit(phase, "assets", index),
        ASSET_REQUIRED,
        phase=phase,
        index=index,
    )


def _burn_item(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    burn_in_steps: int,
) -> dict[str, Any]:
    return _item(
        "burn_in",
        _burn_in_unit(store, phase, index, schedule, burn_in_steps),
        BURN_IN_REQUIRED,
        phase=phase,
        index=index,
        schedule=schedule,
        burn_in_steps=burn_in_steps,
    )


def _trajectory_item(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    condition: Condition,
    burn_in_steps: int,
    rank: int,
) -> dict[str, Any]:
    return _item(
        "trajectory",
        _trajectory_unit(
            store,
            phase,
            index,
            schedule,
            condition,
            burn_in_steps=burn_in_steps,
            rank=rank,
        ),
        TRAJECTORY_REQUIRED,
        phase=phase,
        index=index,
        schedule=schedule,
        condition=condition.mapping(),
        burn_in_steps=burn_in_steps,
        rank=rank,
    )


def _phase0_conditions(study: Plan13Study) -> tuple[Condition, ...]:
    probe = default_probe_conditions()[0]
    floor = dataclasses.replace(probe, name="adaptive_probe_floor005", epsilon=0.05)
    return (
        Condition("no_update", "no_update"),
        Condition("full_space", "full_space"),
        Condition("head_only", "head_only"),
        Condition("random_rank_matched", "random_rank_matched"),
        Condition("static_projector", "static_projector"),
        Condition("static_subgd", "static_subgd"),
        Condition(
            f"online_subgd_h{study.covariance_half_lives[0]:g}",
            "online_subgd",
            covariance_half_life=study.covariance_half_lives[0],
        ),
        probe,
        floor,
    )


def build_phase0_ledger(store: UnitStore) -> dict[str, Any]:
    phase = "phase0"
    burn_in = min(store.study.burn_in_candidates)
    rank = min(store.study.rank_candidates)
    conditions = _phase0_conditions(store.study)
    items = [_asset_item(store, phase, 1)]
    for schedule in ("linear", "sigmoid"):
        items.append(_burn_item(store, phase, 1, schedule, burn_in))
        items.extend(
            _trajectory_item(store, phase, 1, schedule, condition, burn_in, rank)
            for condition in conditions
        )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "conditions": [condition.mapping() for condition in conditions],
    }
    items.append(
        _item(
            "phase0_analysis",
            store.unit(phase, "analysis", 1, detail=detail),
            ANALYSIS_REQUIRED,
            conditions=detail["conditions"],
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "items": items,
    }


def build_phase1_ledger(store: UnitStore) -> dict[str, Any]:
    phase = "phase1"
    burn_in = min(store.study.burn_in_candidates)
    rank = min(store.study.rank_candidates)
    items = []
    full = Condition("full_space", "full_space")
    for index in range(1, store.study.phase1_replicas + 1):
        items.append(_asset_item(store, phase, index))
        for schedule in ("linear", "sigmoid"):
            items.append(_burn_item(store, phase, index, schedule, burn_in))
            items.append(_trajectory_item(store, phase, index, schedule, full, burn_in, rank))
    items.append(
        _item(
            "phase1_analysis",
            store.unit(phase, "analysis", 1),
            ANALYSIS_REQUIRED,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "items": items,
    }


def _phase1_selection(store: UnitStore) -> dict[str, Any]:
    path = store.completed(store.unit("phase1", "analysis", 1), ANALYSIS_REQUIRED)
    if path is None:
        raise RuntimeError("Phase 1 selection is not complete")
    return _read_json(path / "selection.json")


def _ledger_selection(store: UnitStore, ledger_id: str) -> dict[str, Any]:
    ledger_path = _ledger_path(store, ledger_id)
    if not ledger_path.exists():
        raise RuntimeError(f"missing prerequisite ledger: {ledger_id}")
    ledger = _read_json(ledger_path)
    analyses = [item for item in ledger["items"] if item["action"].endswith("analysis")]
    if len(analyses) != 1:
        raise RuntimeError(f"{ledger_id} must contain exactly one analysis item")
    completed = store.completed(analyses[0]["unit"], tuple(analyses[0]["required"]))
    if completed is None:
        raise RuntimeError(f"{ledger_id} analysis is not complete")
    return _read_json(completed / "selection.json")


def build_phase2_ledger(store: UnitStore) -> dict[str, Any]:
    phase = "phase2"
    selection = _phase1_selection(store)
    burn_in = int(selection["burn_in_steps"])
    rank = int(selection["rank"])
    conditions = phase2_conditions(store.study.covariance_half_lives)
    items = []
    for index in range(1, store.study.phase2_replicas + 1):
        items.append(_asset_item(store, phase, index))
        for schedule in ("linear", "sigmoid"):
            items.append(_burn_item(store, phase, index, schedule, burn_in))
            items.extend(
                _trajectory_item(store, phase, index, schedule, condition, burn_in, rank)
                for condition in conditions
            )
    items.append(
        _item(
            "phase2_analysis",
            store.unit(
                phase,
                "analysis",
                1,
                detail={"burn_in_steps": burn_in, "adaptation_rank": rank},
            ),
            ANALYSIS_REQUIRED,
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "items": items,
    }


def _development_controls(store: UnitStore) -> tuple[Condition, ...]:
    phase2 = _ledger_selection(store, "phase2")
    return (
        Condition("full_space", "full_space"),
        Condition("static_subgd", "static_subgd"),
        Condition(
            str(phase2["online_condition"]),
            "online_subgd",
            covariance_half_life=float(phase2["online_half_life"]),
        ),
    )


def _unique_conditions(conditions: tuple[Condition, ...]) -> tuple[Condition, ...]:
    result = []
    names = set()
    for condition in conditions:
        if condition.name not in names:
            result.append(condition)
            names.add(condition.name)
    return tuple(result)


def _rotation_stage_items(
    store: UnitStore,
    *,
    artifact_phase: str,
    replica_count: int,
    conditions: tuple[Condition, ...],
    burn_in_steps: int,
    rank: int,
) -> list[dict[str, Any]]:
    items = []
    for index in range(1, replica_count + 1):
        items.append(_asset_item(store, artifact_phase, index))
        for schedule in ("linear", "sigmoid"):
            items.append(_burn_item(store, artifact_phase, index, schedule, burn_in_steps))
            items.extend(
                _trajectory_item(
                    store,
                    artifact_phase,
                    index,
                    schedule,
                    condition,
                    burn_in_steps,
                    rank,
                )
                for condition in conditions
            )
    return items


def build_phase3a_ledger(store: UnitStore) -> dict[str, Any]:
    ledger_id = "phase3a"
    phase1 = _phase1_selection(store)
    burn_in = int(phase1["burn_in_steps"])
    rank = int(phase1["rank"])
    probes = tuple(condition_from_mapping(value) for value in phase1["probe_conditions"])
    floors = adaptive_floor_conditions(probes[0].controller, prefix="adaptive_probe_floor")
    conditions = _unique_conditions((*_development_controls(store), *probes, *floors))
    items = _rotation_stage_items(
        store,
        artifact_phase=ledger_id,
        replica_count=store.study.phase3_probe_replicas,
        conditions=conditions,
        burn_in_steps=burn_in,
        rank=rank,
    )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "conditions": [condition.mapping() for condition in conditions],
    }
    items.append(
        _item(
            "phase3_probe_analysis",
            store.unit(ledger_id, "analysis", 1, detail=detail),
            ANALYSIS_REQUIRED,
            phase=ledger_id,
            replica_count=store.study.phase3_probe_replicas,
            conditions=detail["conditions"],
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": ledger_id,
        "items": items,
    }


def build_phase3a_repair_ledger(store: UnitStore) -> dict[str, Any]:
    ledger_id = "phase3a_repair"
    first = _ledger_selection(store, "phase3a")
    if not first["fallback_used"]:
        return {
            "study_hash": store.study.config_hash,
            "source_hashes": store.sources,
            "phase": ledger_id,
            "items": [],
            "skipped": "initial_probe_had_mechanically_healthy_survivors",
        }
    phase1 = _phase1_selection(store)
    burn_in = int(phase1["burn_in_steps"])
    rank = int(phase1["rank"])
    carried = condition_from_mapping(first["survivor_conditions"][0])
    assert carried.controller is not None
    controller = carried.controller
    repaired = (
        Condition(
            "adaptive_repair_faster",
            "adaptive_subgd",
            controller=dataclasses.replace(
                controller,
                innovation_half_life=max(controller.innovation_half_life / 2, 1e-6),
                alpha_scale=controller.alpha_scale * 2,
                beta_scale=controller.beta_scale * 2,
            ),
        ),
        Condition(
            "adaptive_repair_slower",
            "adaptive_subgd",
            controller=dataclasses.replace(
                controller,
                innovation_half_life=controller.innovation_half_life * 2,
                alpha_scale=max(controller.alpha_scale / 2, 1e-6),
                beta_scale=max(controller.beta_scale / 2, 1e-6),
            ),
        ),
        Condition(
            "adaptive_repair_floor005",
            "adaptive_subgd",
            epsilon=0.05,
            controller=dataclasses.replace(
                controller,
                alpha_scale=controller.alpha_scale * 2,
                beta_scale=controller.beta_scale * 2,
            ),
        ),
    )
    conditions = _unique_conditions((*_development_controls(store), *repaired))
    items = _rotation_stage_items(
        store,
        artifact_phase=ledger_id,
        replica_count=store.study.phase3_probe_replicas,
        conditions=conditions,
        burn_in_steps=burn_in,
        rank=rank,
    )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "conditions": [condition.mapping() for condition in conditions],
    }
    items.append(
        _item(
            "phase3_probe_analysis",
            store.unit(ledger_id, "analysis", 1, detail=detail),
            ANALYSIS_REQUIRED,
            phase=ledger_id,
            replica_count=store.study.phase3_probe_replicas,
            conditions=detail["conditions"],
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": ledger_id,
        "items": items,
        "repair_generation": 1,
    }


def _phase3_probe_selection(store: UnitStore) -> dict[str, Any]:
    first = _ledger_selection(store, "phase3a")
    if not first["fallback_used"]:
        return first
    return _ledger_selection(store, "phase3a_repair")


def _phase3_stage_ledger(
    store: UnitStore,
    *,
    ledger_id: str,
    conditions: tuple[Condition, ...],
    candidate_names: tuple[str, ...],
    final_stage: bool,
) -> dict[str, Any]:
    phase1 = _phase1_selection(store)
    burn_in = int(phase1["burn_in_steps"])
    rank = int(phase1["rank"])
    trajectory_phase = "phase3b"
    items = _rotation_stage_items(
        store,
        artifact_phase=trajectory_phase,
        replica_count=store.study.phase3_replicas,
        conditions=conditions,
        burn_in_steps=burn_in,
        rank=rank,
    )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "candidate_names": list(candidate_names),
        "trajectory_phase": trajectory_phase,
    }
    items.append(
        _item(
            "phase3_stage_analysis",
            store.unit(ledger_id, "analysis", 1, detail=detail),
            ANALYSIS_REQUIRED,
            phase=ledger_id,
            trajectory_phase=trajectory_phase,
            replica_count=store.study.phase3_replicas,
            conditions=[condition.mapping() for condition in conditions],
            candidate_names=list(candidate_names),
            burn_in_steps=burn_in,
            rank=rank,
            final_stage=final_stage,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": ledger_id,
        "items": items,
    }


def build_phase3b_trust_ledger(store: UnitStore) -> dict[str, Any]:
    probe = _phase3_probe_selection(store)
    candidates = tuple(
        condition_from_mapping(value) for value in probe["survivor_conditions"]
    )
    conditions = _unique_conditions((*_development_controls(store), *candidates))
    return _phase3_stage_ledger(
        store,
        ledger_id="phase3b_trust",
        conditions=conditions,
        candidate_names=tuple(condition.name for condition in candidates),
        final_stage=False,
    )


def build_phase3b_geometry_ledger(store: UnitStore) -> dict[str, Any]:
    trust = _ledger_selection(store, "phase3b_trust")
    selected = condition_from_mapping(trust["selected_condition"])
    assert selected.controller is not None
    candidates = geometry_rate_conditions(selected.controller)
    conditions = _unique_conditions((*_development_controls(store), *candidates))
    return _phase3_stage_ledger(
        store,
        ledger_id="phase3b_geometry",
        conditions=conditions,
        candidate_names=tuple(condition.name for condition in candidates),
        final_stage=False,
    )


def build_phase3b_floor_ledger(store: UnitStore) -> dict[str, Any]:
    geometry = _ledger_selection(store, "phase3b_geometry")
    selected = condition_from_mapping(geometry["selected_condition"])
    assert selected.controller is not None
    candidates = adaptive_floor_conditions(selected.controller)
    conditions = _unique_conditions((*_development_controls(store), *candidates))
    return _phase3_stage_ledger(
        store,
        ledger_id="phase3b_floor",
        conditions=conditions,
        candidate_names=tuple(condition.name for condition in candidates),
        final_stage=True,
    )


def build_phase4_ledger(store: UnitStore) -> dict[str, Any]:
    phase = "phase4"
    phase1 = _phase1_selection(store)
    phase3 = _ledger_selection(store, "phase3b_floor")
    burn_in = int(phase1["burn_in_steps"])
    rank = int(phase1["rank"])
    adaptive = condition_from_mapping(phase3["selected_condition"])
    nonadaptive = condition_from_mapping(phase3["selected_nonadaptive_condition"])
    conditions = (
        Condition("full_space", "full_space"),
        Condition("head_only", "head_only"),
        nonadaptive,
        adaptive,
        Condition("random_rank_matched", "random_rank_matched"),
    )
    replica_count = int(phase3["phase4_replica_count"])
    items = _rotation_stage_items(
        store,
        artifact_phase=phase,
        replica_count=replica_count,
        conditions=conditions,
        burn_in_steps=burn_in,
        rank=rank,
    )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "replica_count": replica_count,
        "conditions": [condition.mapping() for condition in conditions],
    }
    items.append(
        _item(
            "phase4_analysis",
            store.unit(phase, "analysis", 1, detail=detail),
            ANALYSIS_REQUIRED,
            replica_count=replica_count,
            conditions=detail["conditions"],
            adaptive_name=adaptive.name,
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "items": items,
    }


def build_phase5_ledger(store: UnitStore) -> dict[str, Any]:
    phase = "phase5"
    phase1 = _phase1_selection(store)
    phase4 = _ledger_selection(store, "phase4")
    burn_in = int(phase1["burn_in_steps"])
    rank = int(phase1["rank"])
    dynamic = condition_from_mapping(phase4["dynamic_transport_condition"])
    conditions = _unique_conditions(
        (
            Condition("full_space", "full_space"),
            Condition("head_only", "head_only"),
            Condition("random_rank_matched", "random_rank_matched"),
            Condition("static_subgd", "static_subgd"),
            dynamic,
        )
    )
    if len(conditions) != 5:
        raise RuntimeError("Phase 5 requires exactly five distinct conditions")
    items = []
    for index in range(1, store.study.phase5_replicas + 1):
        items.append(
            _item(
                "mixture_assets",
                mixture_asset_unit(store, index),
                MIXTURE_ASSET_REQUIRED,
                index=index,
            )
        )
        items.append(
            _item(
                "mixture_burn_in",
                mixture_burn_unit(store, index, burn_in),
                MIXTURE_BURN_REQUIRED,
                index=index,
                burn_in_steps=burn_in,
            )
        )
        items.extend(
            _item(
                "mixture_trajectory",
                mixture_trajectory_unit(
                    store,
                    index,
                    condition,
                    burn_in_steps=burn_in,
                    rank=rank,
                ),
                MIXTURE_TRAJECTORY_REQUIRED,
                index=index,
                condition=condition.mapping(),
                burn_in_steps=burn_in,
                rank=rank,
            )
            for condition in conditions
        )
    detail = {
        "burn_in_steps": burn_in,
        "adaptation_rank": rank,
        "replica_count": store.study.phase5_replicas,
        "conditions": [condition.mapping() for condition in conditions],
    }
    items.append(
        _item(
            "phase5_analysis",
            store.unit(
                phase,
                "analysis",
                1,
                environment="digit9_mixture",
                detail=detail,
            ),
            ANALYSIS_REQUIRED,
            conditions=detail["conditions"],
            dynamic_name=dynamic.name,
            burn_in_steps=burn_in,
            rank=rank,
        )
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "items": items,
    }


def _ledger_path(store: UnitStore, phase: str) -> Path:
    return store.root / "ledgers" / f"{phase}.json"


def _execute_item(
    store: UnitStore,
    item: dict[str, Any],
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    parameters = item["parameters"]
    action = item["action"]
    if action == "assets":
        return ensure_rotation_assets(
            store,
            parameters["phase"],
            int(parameters["index"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "burn_in":
        return ensure_burn_in(
            store,
            parameters["phase"],
            int(parameters["index"]),
            parameters["schedule"],
            int(parameters["burn_in_steps"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "trajectory":
        return run_rotation_trajectory(
            store,
            parameters["phase"],
            int(parameters["index"]),
            parameters["schedule"],
            condition_from_mapping(parameters["condition"]),
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "mixture_assets":
        return ensure_mixture_assets(
            store,
            int(parameters["index"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "mixture_burn_in":
        return ensure_mixture_burn_in(
            store,
            int(parameters["index"]),
            int(parameters["burn_in_steps"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "mixture_trajectory":
        return run_mixture_trajectory(
            store,
            int(parameters["index"]),
            condition_from_mapping(parameters["condition"]),
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            data_root=data_root,
            resume=resume,
        )
    if action == "phase1_analysis":
        return run_phase1_analysis(store, resume=resume)
    if action == "phase0_analysis":
        return run_phase0_analysis(
            store,
            conditions=tuple(
                condition_from_mapping(value) for value in parameters["conditions"]
            ),
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            resume=resume,
        )
    if action == "phase2_analysis":
        return run_phase2_analysis(
            store,
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            resume=resume,
        )
    if action == "phase3_probe_analysis":
        return run_phase3_probe_analysis(
            store,
            phase=parameters["phase"],
            replica_count=int(parameters["replica_count"]),
            conditions=tuple(
                condition_from_mapping(value) for value in parameters["conditions"]
            ),
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            resume=resume,
        )
    if action == "phase3_stage_analysis":
        return run_phase3_stage_analysis(
            store,
            phase=parameters["phase"],
            trajectory_phase=parameters["trajectory_phase"],
            replica_count=int(parameters["replica_count"]),
            conditions=tuple(
                condition_from_mapping(value) for value in parameters["conditions"]
            ),
            candidate_names=tuple(parameters["candidate_names"]),
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            final_stage=bool(parameters["final_stage"]),
            resume=resume,
        )
    if action == "phase4_analysis":
        return run_phase4_analysis(
            store,
            replica_count=int(parameters["replica_count"]),
            conditions=tuple(
                condition_from_mapping(value) for value in parameters["conditions"]
            ),
            adaptive_name=parameters["adaptive_name"],
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            resume=resume,
        )
    if action == "phase5_analysis":
        return run_phase5_analysis(
            store,
            conditions=tuple(
                condition_from_mapping(value) for value in parameters["conditions"]
            ),
            dynamic_name=parameters["dynamic_name"],
            burn_in_steps=int(parameters["burn_in_steps"]),
            rank=int(parameters["rank"]),
            resume=resume,
        )
    raise ValueError(f"unsupported Plan 13 action: {action}")


def run_ledger(
    store: UnitStore,
    ledger: dict[str, Any],
    *,
    data_root: Path,
    notebook: Path,
    resume: bool,
    max_units: int | None,
    max_wall_seconds: float | None,
) -> tuple[int, bool]:
    path = _ledger_path(store, ledger["phase"])
    freeze_json(path, ledger, resume=resume)
    refresh(store, notebook)
    started = time.monotonic()
    last_refresh = started
    completed_now = 0
    exhausted = False
    for item in ledger["items"]:
        required = tuple(item["required"])
        if store.completed(item["unit"], required) is not None:
            continue
        if max_units is not None and completed_now >= max_units:
            exhausted = True
            break
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            exhausted = True
            break
        try:
            _execute_item(store, item, data_root=data_root, resume=resume)
        except BaseException as error:
            store.record_failure(item["unit"], error)
            refresh(store, notebook)
            raise
        completed_now += 1
        if item["action"] in {
            "phase0_analysis",
            "phase1_analysis",
            "phase2_analysis",
            "phase3_probe_analysis",
            "phase3_stage_analysis",
            "phase4_analysis",
            "phase5_analysis",
        } or time.monotonic() - last_refresh >= 300:
            refresh(store, notebook)
            last_refresh = time.monotonic()
    refresh(store, notebook)
    return completed_now, exhausted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("cache/mnist_experiment/datasets"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--through-phase", choices=PHASES, default="phase5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.max_units is not None and arguments.max_units < 1:
        raise ValueError("--max-units must be positive")
    if arguments.max_wall_seconds is not None and arguments.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    builders = {
        "phase0": (build_phase0_ledger,),
        "phase1": (build_phase1_ledger,),
        "phase2": (build_phase2_ledger,),
        "phase3": (
            build_phase3a_ledger,
            build_phase3a_repair_ledger,
            build_phase3b_trust_ledger,
            build_phase3b_geometry_ledger,
            build_phase3b_floor_ledger,
        ),
        "phase4": (build_phase4_ledger,),
        "phase5": (build_phase5_ledger,),
    }
    total_units = 0
    started = time.monotonic()
    for phase in PHASES[: PHASES.index(arguments.through_phase) + 1]:
        if phase not in builders:
            raise RuntimeError(f"{phase} implementation is not yet available")
        phase_exhausted = False
        for builder in builders[phase]:
            remaining_units = (
                None
                if arguments.max_units is None
                else arguments.max_units - total_units
            )
            remaining_seconds = (
                None
                if arguments.max_wall_seconds is None
                else arguments.max_wall_seconds - (time.monotonic() - started)
            )
            if remaining_units is not None and remaining_units <= 0:
                phase_exhausted = True
                break
            if remaining_seconds is not None and remaining_seconds <= 0:
                phase_exhausted = True
                break
            completed, exhausted = run_ledger(
                store,
                builder(store),
                data_root=arguments.data_root,
                notebook=arguments.notebook,
                resume=arguments.resume,
                max_units=remaining_units,
                max_wall_seconds=remaining_seconds,
            )
            total_units += completed
            if exhausted:
                phase_exhausted = True
                break
        if phase_exhausted:
            break


if __name__ == "__main__":
    main()
