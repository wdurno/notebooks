"""Pure regression FIFO transactions and fixed-objective, bounded local fits.

The caller materializes one pre-arrival replay subset, then persists the whole
accepted transaction. An interruption cannot partially retire observations.
The archive uses four nominal arrivals as its exposure unit even when a solve
ends after a shorter, exact-novel-count segment.
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
    age_weights as _legacy_age_weights,
    retention_terms,
)
from src.representations import LowRankDiagonalFisher
from src.seeding import derive_component_seed

from .curvature import fd_values, gaussian_nll
from .model import ANGLE_SCALE_DEGREES, flat_prediction, memory_indices


STATE_SCHEMA = 1
LIKELIHOOD = "gaussian_fixed_scale_45_degrees"
NOMINAL_ARRIVAL_SIZE = 4
CONDITIONS = frozenset({"fd_replay_ewc", "replay_ewc", "ewc_only"})


def compress_fisher(matrix: Tensor, *, rank: int = 8, seed: int = 0,
                    check_stop: Callable | None = None) -> tuple[dict, dict]:
    """Keep top PSD modes and the residual diagonal without inflating curvature.

    This experiment already materializes the small LTM block. Exact eigh avoids
    duplicated Ritz directions near Lanczos breakdown. The seed is recorded for
    interface compatibility; this deterministic projection consumes no RNG.
    """
    _integer(rank, "rank", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("compression seed must lie in [0, 2**63)")
    _stop(check_stop)
    source = _fisher_matrix(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(source)
    _stop(check_stop)
    scale = max(float(eigenvalues.abs().max()), torch.finfo(source.dtype).tiny)
    tolerance = 100 * len(source) * torch.finfo(source.dtype).eps * scale
    minimum = float(eigenvalues.min())
    if minimum < -tolerance:
        raise ValueError("cannot archive a materially indefinite Fisher")
    count = min(rank, len(source))
    selected = torch.arange(len(source) - count, len(source)) if count else torch.empty(0, dtype=torch.long)
    selected = selected[eigenvalues[selected] > 0]
    factor = eigenvectors[:, selected] * eigenvalues[selected].sqrt()
    residual = source.diagonal() - factor.square().sum(1)
    residual_minimum = float(residual.min())
    if residual_minimum < -tolerance:
        raise ValueError("spectral factor materially exceeds the source diagonal")
    representation = LowRankDiagonalFisher(factor, residual.clamp_min(0))
    diagonal = representation.diagonal_vector()
    if not torch.allclose(diagonal, source.diagonal(), rtol=0, atol=tolerance):
        raise ValueError("Fisher compression did not preserve the diagonal")
    _stop(check_stop)
    return representation.artifact_mapping(), {
        "pre_compression_min_eigenvalue": minimum,
        "pre_compression_negative_mass": float((-eigenvalues[eigenvalues < 0]).sum()),
        "source_trace": float(source.trace()), "represented_trace": float(diagonal.sum()),
        "compression": dict(algorithm="exact_symmetric_eigh", requested_rank=rank,
            realized_rank=representation.rank, seed=seed, randomized=False,
            retained_eigenvalues=eigenvalues[selected].flip(0).tolist(),
            residual_diagonal_minimum_before_clipping=residual_minimum,
            represented_diagonal_relative_error=float((diagonal - source.diagonal()).norm()
                / source.diagonal().norm().clamp_min(torch.finfo(source.dtype).tiny))),
    }


def consolidate(fisher: Any, anchor: Tensor, accepted_ltm: Tensor,
                evicted_fisher: Tensor | None, *, evicted_ids: Sequence,
                consolidated_ids: Sequence, pi: float, arrival_size: int,
                rank: int = 8, seed: int = 0, check_stop: Callable | None = None) -> dict:
    """Same eviction EMA and anchors, using the trace-preserving compressor."""
    if not math.isfinite(pi) or not 0 < pi <= 1:
        raise ValueError("pi must lie in (0, 1]")
    _integer(arrival_size, "arrival_size")
    ids, old_ids = _exposure_ids(evicted_ids), _exposure_ids(consolidated_ids)
    if set(ids).intersection(old_ids):
        raise ValueError("exposure already consolidated")
    _stop(check_stop)
    old = _fisher_matrix(fisher)
    anchor, accepted_ltm = _vector(anchor, "anchor").cpu(), _vector(accepted_ltm, "accepted_ltm").cpu()
    if anchor.shape != accepted_ltm.shape or old.shape != (len(anchor), len(anchor)):
        raise ValueError("archive and LTM anchor dimensions must agree")
    if not ids:
        if evicted_fisher is not None:
            raise ValueError("a Fisher update requires actual evicted exposures")
        return dict(fisher_L=copy.deepcopy(fisher), anchor_L=anchor.clone(),
                    consolidated_ids=old_ids, eta=0., evicted_count=0)
    if evicted_fisher is None:
        raise ValueError("eviction requires the accepted-state model-label Fisher")
    fresh = _fisher_matrix(evicted_fisher)
    if fresh.shape != old.shape:
        raise ValueError("evicted Fisher must be the matching LTM block")
    eta = 1. if pi == 1 else -math.expm1(len(ids) / arrival_size * math.log1p(-pi))
    archived, diagnostics = compress_fisher((1 - eta) * old + eta * fresh,
        rank=rank, seed=seed, check_stop=check_stop)
    return dict(fisher_L=archived, anchor_L=accepted_ltm.clone(), consolidated_ids=old_ids + ids,
                eta=eta, evicted_count=len(ids), archive_diagnostics=diagnostics)


def age_weights(insertion_counts: Sequence[int] | Tensor, cumulative_count: int,
                capacity: int, *, device="cpu", dtype=torch.float64) -> Tensor:
    """Squared arrival ages, allowing an empty zero-capacity boundary."""
    _integer(capacity, "capacity", 0)
    _integer(cumulative_count, "cumulative_count", 0)
    if capacity == 0:
        counts = torch.as_tensor(insertion_counts)
        if counts.ndim != 1 or counts.numel():
            raise ValueError("zero capacity cannot have resident insertion counts")
        return torch.empty(0, device=device, dtype=dtype)
    return _legacy_age_weights(insertion_counts, cumulative_count, capacity,
                               device=device, dtype=dtype)


def initial_state(theta: Tensor, ltm_indices: Tensor, fisher: Any, *,
                  capacity: int = 32, rank: int = 8, seed: int = 0,
                  check_stop: Callable | None = None) -> dict:
    """Create a regression checkpoint; pretraining is not an online exposure."""
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
        raise ValueError("incompatible regression learner state schema or likelihood")
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
    """Propose FIFO retirement; all old residents remain eligible for this solve."""
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
    """CPU resident positions sampled once, with nested prefixes for different b.

    ``seed`` is the caller's independently named replay seed for this arrival.
    It must not depend on b when pairing different amounts at the same capacity.
    Global random streams, state, and the data-order generator are untouched.
    """
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
    """Return an atomic accepted transaction, archiving every evictee once.

    The caller computes ``evicted_fisher`` at accepted_theta on *all* staged
    evictees, whether selected for replay or not. ``arrival_size`` is the fixed
    nominal four-image exposure unit, never the current split-segment length.
    """
    _stop(check_stop)
    _integer(arrival_size, "arrival_size")
    if arrival_size != NOMINAL_ARRIVAL_SIZE:
        raise ValueError("regression archive arrival_size is fixed at the nominal value 4")
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
    """Freeze replay, probes, anchors and degree targets for every closure.

    Returns the legacy (objective, metadata) pair. Objective outputs are the
    scalar penalized loss and a dictionary of differentiable component losses.
    Mean replay normalization does not change with the selected subset size b.
    """
    _stop(check_stop)
    cfg = _settings(settings)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown regression learner condition: {condition}")
    start = _vector(theta, "theta").clone()
    ltm, stm = (value.to(start.device) for value in memory_indices(layout))
    new_x, new_y = new_x.detach().clone(), new_y.detach().clone()
    replay_x, replay_y = replay_x.detach().clone(), replay_y.detach().clone()
    if (new_y.ndim != 1 or not new_y.numel() or replay_y.ndim != 1
            or new_x.shape != (len(new_y), 1, 28, 28)
            or replay_x.shape != (len(replay_y), 1, 28, 28)):
        raise ValueError("new and replay images need matching degree-target vectors")
    if (new_x.dtype != start.dtype or replay_x.dtype != start.dtype
            or any(value.device != start.device for value in (new_x, replay_x, new_y, replay_y))):
        raise ValueError("images and theta must share dtype; all solve data must share a device")
    if not all(torch.isfinite(value).all() for value in (new_x, replay_x, new_y, replay_y)):
        raise ValueError("solve images and degree targets must be finite")
    if condition == "ewc_only" and len(replay_y):
        raise ValueError("ewc_only cannot receive replay observations")
    ages = torch.as_tensor(ages, device=start.device, dtype=torch.float64).detach().clone()
    if (ages.shape != replay_y.shape or not torch.isfinite(ages).all()
            or ((ages < 0) | (ages > 1)).any()):
        raise ValueError("ages must be one fixed squared age in [0,1] per replay input")
    archive = _fisher_matrix(fisher, device=start.device)
    fixed_anchor = anchor.detach().to(start.device).clone()
    _, _, kappa = retention_terms(start, ltm, stm, archive, fixed_anchor, start[stm],
                                 pi=cfg.pi, ridge_ratio=cfg.ridge_ratio)
    probe_seed = derive_component_seed(seed, "regression_training_curvature_probes")
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
                auxiliary = auxiliary + (
                    ages[offset:offset + cfg.chunk_size] * values
                ).sum() / len(replay_y)
        total = data_loss + ewc + ridge + strength * auxiliary
        return total, {"new_nll": new_loss, "replay_nll": replay_loss,
                       "data_loss": data_loss, "ewc": ewc, "ridge": ridge,
                       "auxiliary": auxiliary, "weighted_auxiliary": strength * auxiliary}

    return objective, {
        "probe_seed": probe_seed, "kappa": kappa, "num_probes": cfg.num_probes,
        "fd_step": cfg.fd_step, "fd_perturbation_norm": cfg.fd_step * math.sqrt(stm.numel()),
        "fd_branch_perturbation_norm": cfg.fd_step * math.sqrt(stm.numel()) / 2,
        "gaussian_scale_degrees": ANGLE_SCALE_DEGREES, "stm_dimension": stm.numel(),
        "new_count": len(new_y), "replay_count": len(replay_y),
        "fd_active": bool(strength and len(replay_y)), "lambda_f": strength,
    }


def fit_step(model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
             replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any,
             anchor: Tensor, condition: str, settings: FitSettings | Mapping,
             seed: int, check_stop: Callable | None = None, *,
             objective_observer: Callable | None = None) -> tuple[Tensor, dict]:
    """Fit without mutating inputs, using the frozen legacy bounded solver.

    The optional observer receives (accepted_theta, objective, metadata,
    diagnostics), including the exact frozen objective for numerical audits.
    No state is committed here, including when cancellation interrupts a solve.
    """
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
