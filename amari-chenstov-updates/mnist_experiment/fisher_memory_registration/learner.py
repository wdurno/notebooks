"""Registration FIFO transactions and frozen-objective, bounded local fits.

Only actual retirements update the likelihood-Fisher archive, at accepted
parameters. Split segments retain the nominal four-pair retirement clock.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from mnist_experiment.fisher_memory.learner import (
    FitSettings,
    StopRequested,
    _bounded_solve,
    _exposure_ids,
    _fisher_matrix,
    _indices,
    _integer,
    _settings,
    _stop,
    _vector,
    retention_terms,
)
from mnist_experiment.fisher_memory_regression.learner import (
    age_weights,
    compress_fisher,
    consolidate,
)
from src.seeding import derive_component_seed

from .curvature import SIGMA2, fd_values, gaussian_nll
from .model import GAUSSIAN_SCALE_PIXELS, flat_prediction, memory_indices


STATE_SCHEMA = 1
LIKELIHOOD = "gaussian_fixed_scale_2_pixels"
NOMINAL_ARRIVAL_SIZE = 4
CONDITIONS = frozenset({"fd_replay_ewc", "replay_ewc", "ewc_only"})
PROBE_SEED_COMPONENT = "registration_training_curvature_probes"


def initial_state(theta: Tensor, ltm_indices: Tensor, fisher: Any, *,
                  capacity: int = 32, rank: int = 8, seed: int = 0,
                  check_stop: Callable | None = None) -> dict:
    """Create an independently versioned pixel-likelihood learner checkpoint."""
    _stop(check_stop)
    theta = _vector(theta, "theta").cpu().clone()
    indices = _indices(ltm_indices, theta.numel())
    _integer(capacity, "capacity", 0)
    source = _fisher_matrix(fisher)
    if source.shape != (indices.numel(), indices.numel()):
        raise ValueError("initial Fisher must be the LTM principal block")
    archive, diagnostics = compress_fisher(source, rank=rank, seed=seed, check_stop=check_stop)
    return {
        "schema_version": STATE_SCHEMA, "likelihood": LIKELIHOOD,
        "nominal_arrival_size": NOMINAL_ARRIVAL_SIZE,
        "theta": theta, "ltm_indices": indices, "anchor_L": theta[indices].clone(),
        "fisher_L": archive, "capacity": capacity, "replay": [],
        "consolidated_ids": [], "cumulative_count": 0, "accepted_arrivals": 0,
        "archive_updates": 0, "archive_diagnostics": diagnostics,
    }


def validate_state(state: Mapping) -> None:
    if (state.get("schema_version") != STATE_SCHEMA or state.get("likelihood") != LIKELIHOOD
            or state.get("nominal_arrival_size") != NOMINAL_ARRIVAL_SIZE):
        raise ValueError("incompatible registration learner state schema or likelihood")
    theta = _vector(state["theta"], "theta")
    indices = _indices(state["ltm_indices"], theta.numel())
    anchor = _vector(state["anchor_L"], "anchor_L")
    if (anchor.numel() != len(indices)
            or _fisher_matrix(state["fisher_L"]).shape != (len(indices), len(indices))):
        raise ValueError("archive and anchor must use LTM coordinates")
    capacity = _integer(state["capacity"], "capacity", 0)
    count = _integer(state["cumulative_count"], "cumulative_count", 0)
    accepted = _integer(state["accepted_arrivals"], "accepted_arrivals", 0)
    updates = _integer(state["archive_updates"], "archive_updates", 0)
    replay = state["replay"]
    live = _exposure_ids([record["exposure_id"] for record in replay])
    archived = _exposure_ids(state["consolidated_ids"])
    insertions = [record["insertion_count"] for record in replay]
    age_weights(insertions, count, capacity)
    if (len(replay) > capacity or set(live).intersection(archived)
            or len(live) + len(archived) != count or insertions != sorted(insertions)
            or updates > accepted or accepted > count):
        raise ValueError("inconsistent FIFO, exposure ledger, or arrival counts")
    if any("input_ref" not in record for record in replay):
        raise ValueError("replay entries require input references")


def stage_arrival(state: Mapping, exposure_ids: Sequence, input_refs: Sequence) -> dict:
    """Stage FIFO retirement while leaving all old residents eligible to replay."""
    validate_state(state)
    ids, refs = _exposure_ids(exposure_ids), list(input_refs)
    if not ids or len(ids) != len(refs):
        raise ValueError("arrival needs one input reference per exposure")
    seen = set(state["consolidated_ids"]) | {r["exposure_id"] for r in state["replay"]}
    if seen.intersection(ids):
        raise ValueError("exposure was already replayed or consolidated")
    count = state["cumulative_count"] + len(ids)
    new = [{"exposure_id": uid, "input_ref": copy.deepcopy(ref), "insertion_count": count}
           for uid, ref in zip(ids, refs, strict=True)]
    working = copy.deepcopy(state["replay"]) + new
    overflow = max(0, len(working) - state["capacity"])
    return {
        "replay": working[overflow:], "evicted": working[:overflow], "new": new,
        "cumulative_count": count, "working_set_size": len(working),
        "ages": age_weights([r["insertion_count"] for r in state["replay"]],
                            count, state["capacity"]),
    }


def replay_indices(state: Mapping, b: int, seed: int) -> Tensor:
    """Sample resident CPU positions once per solve, with paired nested prefixes."""
    validate_state(state)
    _integer(b, "b", 0)
    if b > state["capacity"] or (state["capacity"] > 0 and b == 0):
        raise ValueError("replay arms need 1 <= b <= capacity; no replay uses C=b=0")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("replay seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(len(state["replay"]), generator=generator)[:b]


def commit_arrival(state: Mapping, accepted_theta: Tensor, exposure_ids: Sequence,
                   input_refs: Sequence, evicted_fisher: Tensor | None = None, *,
                   pi: float, arrival_size: int = NOMINAL_ARRIVAL_SIZE, rank: int = 8,
                   seed: int = 0, check_stop: Callable | None = None) -> tuple[dict, dict]:
    """Atomically archive all actual evictees exactly once, independent of replay.

    The caller supplies their model-integrated Fisher at accepted_theta, not an
    observed-residual outer product. arrival_size stays four even for split groups.
    """
    _stop(check_stop)
    _integer(arrival_size, "arrival_size")
    if arrival_size != NOMINAL_ARRIVAL_SIZE:
        raise ValueError("registration archive arrival_size is fixed at the nominal value 4")
    staged = stage_arrival(state, exposure_ids, input_refs)
    accepted = _vector(accepted_theta, "accepted_theta").cpu().clone()
    if accepted.shape != state["theta"].shape:
        raise ValueError("accepted parameter layout changed")
    updated = consolidate(
        state["fisher_L"], state["anchor_L"], accepted[state["ltm_indices"]],
        evicted_fisher, evicted_ids=[r["exposure_id"] for r in staged["evicted"]],
        consolidated_ids=state["consolidated_ids"], pi=pi,
        arrival_size=NOMINAL_ARRIVAL_SIZE, rank=rank, seed=seed, check_stop=check_stop,
    )
    result = copy.deepcopy(dict(state))
    for key in ("fisher_L", "anchor_L", "consolidated_ids", "archive_diagnostics"):
        if key in updated:
            result[key] = updated[key]
    result.update(theta=accepted, replay=staged["replay"],
                  cumulative_count=staged["cumulative_count"],
                  accepted_arrivals=state["accepted_arrivals"] + 1,
                  archive_updates=state["archive_updates"] + bool(staged["evicted"]))
    validate_state(result)
    _stop(check_stop)
    return result, {"eta": updated["eta"], "evicted_count": len(staged["evicted"]),
                    "working_set_size": staged["working_set_size"],
                    "replay_size": len(staged["replay"]),
                    "nominal_arrival_size": NOMINAL_ARRIVAL_SIZE}


def prepare_objective(model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
                      replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any,
                      anchor: Tensor, condition: str, settings: FitSettings | Mapping,
                      seed: int, check_stop: Callable | None = None) -> tuple[Callable, dict]:
    """Freeze data, probes, anchors and Fisher scale throughout a local solve.

    Targets and predictions are (N, d) physical pixels. Coordinate NLLs sum;
    replay and new losses average only over observations. FD additionally
    averages probes and divides by STM dimension, never by output dimension.
    """
    _stop(check_stop)
    cfg = _settings(settings)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown registration learner condition: {condition}")
    start = _vector(theta, "theta").clone()
    if start.numel() != layout.total_numel or start.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must match the registration layout in float32 or float64")
    ltm, stm = (value.to(start.device) for value in memory_indices(layout))
    new_x, new_y = new_x.detach().clone(), new_y.detach().clone()
    replay_x, replay_y = replay_x.detach().clone(), replay_y.detach().clone()
    if (new_y.ndim != 2 or new_y.shape != (len(new_y), model.output_dim) or not len(new_y)
            or replay_y.ndim != 2 or replay_y.shape != (len(replay_y), model.output_dim)
            or new_x.shape != (len(new_y), 2, 28, 28)
            or replay_x.shape != (len(replay_y), 2, 28, 28)
            or not new_y.is_floating_point() or not replay_y.is_floating_point()):
        raise ValueError("new and replay pairs need matching (N, d) physical-pixel targets")
    if (new_x.dtype != start.dtype or replay_x.dtype != start.dtype
            or any(value.device != start.device for value in (new_x, replay_x, new_y, replay_y))):
        raise ValueError("images and theta must share dtype; all solve data must share a device")
    if not all(torch.isfinite(value).all() for value in (new_x, replay_x, new_y, replay_y)):
        raise ValueError("solve pairs and pixel targets must be finite")
    if condition == "ewc_only" and len(replay_y):
        raise ValueError("ewc_only cannot receive replay observations")
    ages = torch.as_tensor(ages, device=start.device, dtype=torch.float64).detach().clone()
    if (ages.shape != (len(replay_y),) or not torch.isfinite(ages).all()
            or ((ages < 0) | (ages > 1)).any()):
        raise ValueError("ages must be one fixed squared age in [0,1] per replay pair")
    archive = _fisher_matrix(fisher, device=start.device)
    fixed_anchor = anchor.detach().to(start.device).clone()
    _, _, kappa = retention_terms(start, ltm, stm, archive, fixed_anchor, start[stm],
                                 pi=cfg.pi, ridge_ratio=cfg.ridge_ratio)
    probe_seed = derive_component_seed(seed, PROBE_SEED_COMPONENT)
    generator = torch.Generator(device="cpu").manual_seed(probe_seed)
    probes = (2 * torch.randint(2, (cfg.num_probes, stm.numel()), generator=generator)
              - 1).to(device=start.device, dtype=start.dtype)
    strength = cfg.lambda_f if condition == "fd_replay_ewc" else 0.0

    def objective(candidate: Tensor) -> tuple[Tensor, dict]:
        _stop(check_stop)
        zero = candidate.double().sum() * 0.0

        def mean_nll(x, y):
            value = zero
            for offset in range(0, len(y), cfg.chunk_size):
                _stop(check_stop)
                predictions = flat_prediction(model, layout, candidate, x[offset:offset + cfg.chunk_size])
                value = value + gaussian_nll(predictions, y[offset:offset + cfg.chunk_size]).sum() / len(y)
            return value

        new_loss = mean_nll(new_x, new_y)
        replay_loss = mean_nll(replay_x, replay_y) if len(replay_y) else zero
        data_loss = ((1 - cfg.replay_weight) * new_loss + cfg.replay_weight * replay_loss
                     if len(replay_y) else new_loss)
        ewc, ridge, _ = retention_terms(candidate, ltm, stm, archive, fixed_anchor,
                                       start[stm], pi=cfg.pi, ridge_ratio=cfg.ridge_ratio)
        auxiliary = zero
        if strength:
            for offset in range(0, len(replay_y), cfg.chunk_size):
                _stop(check_stop)
                values = fd_values(model, layout, candidate,
                                   replay_x[offset:offset + cfg.chunk_size], probes, cfg.fd_step)
                values = values.mean(0) / stm.numel()
                auxiliary = auxiliary + (ages[offset:offset + cfg.chunk_size] * values).sum() / len(replay_y)
        total = data_loss + ewc + ridge + strength * auxiliary
        return total, {"new_nll": new_loss, "replay_nll": replay_loss,
                       "data_loss": data_loss, "ewc": ewc, "ridge": ridge,
                       "auxiliary": auxiliary, "weighted_auxiliary": strength * auxiliary}

    return objective, {
        "probe_seed": probe_seed, "kappa": kappa, "num_probes": cfg.num_probes,
        "fd_step": cfg.fd_step, "fd_perturbation_norm": cfg.fd_step * math.sqrt(stm.numel()),
        "fd_branch_perturbation_norm": cfg.fd_step * math.sqrt(stm.numel()) / 2,
        "gaussian_scale_pixels": GAUSSIAN_SCALE_PIXELS, "sigma2": SIGMA2,
        "output_dim": model.output_dim, "stm_dimension": stm.numel(),
        "new_count": len(new_y), "replay_count": len(replay_y),
        "fd_active": bool(strength and len(replay_y)), "lambda_f": strength,
    }


def fit_step(model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
             replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any,
             anchor: Tensor, condition: str, settings: FitSettings | Mapping,
             seed: int, check_stop: Callable | None = None, *,
             objective_observer: Callable | None = None) -> tuple[Tensor, dict]:
    """Fit without mutation using the tested bounded solver; never commit here."""
    _stop(check_stop)
    cfg = _settings(settings)
    start = _vector(theta, "theta").clone()
    started = time.perf_counter()
    objective, metadata = prepare_objective(
        model, layout, start, new_x, new_y, replay_x, replay_y, ages, fisher,
        anchor, condition, cfg, seed, check_stop,
    )
    accepted, diagnostics = _bounded_solve(start, objective, cfg, check_stop)
    _stop(check_stop)
    evaluations = diagnostics["objective_evaluations"]
    diagnostics.update(metadata)
    diagnostics.update(
        condition=condition, seed=seed, elapsed_seconds=time.perf_counter() - started,
        total_closure_calls=diagnostics["closure_calls"],
        new_presentations=evaluations * len(new_y),
        replay_presentations=evaluations * len(replay_y),
        fd_branch_presentations=(evaluations * 2 * cfg.num_probes * len(replay_y)
                                 if metadata["fd_active"] else 0),
    )
    if objective_observer is not None:
        objective_observer(accepted.detach().clone(), objective, metadata, diagnostics)
    _stop(check_stop)
    return accepted.detach(), diagnostics
