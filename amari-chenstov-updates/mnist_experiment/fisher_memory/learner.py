"""Plan 15's pure FIFO/EWC transactions and bounded per-arrival optimization.

The caller owns input loading, evaluation, and atomic checkpoint persistence.
``fit_step`` never mutates its inputs or the model. Evaluate its returned theta,
then ``stage_arrival``/``commit_arrival`` archives only the evicted exposure IDs.
Cancellation raises without committing any part of an arrival.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed


STATE_SCHEMA = 1
CONDITIONS = frozenset({"replay_ewc", "jvp", "finite_difference", "directional",
                        "distillation", "replay_compute_matched"})


class StopRequested(RuntimeError):
    """An uncommitted arrival was interrupted; retain the previous checkpoint."""


class _CompressionStop(Exception):
    """Keep cancellation out of the legacy helper's numerical retry handler."""


def _stop(check_stop: Callable | None) -> None:
    if check_stop is not None and check_stop():
        raise StopRequested("stop requested before arrival commit")


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _indices(value: Tensor, dimension: int) -> Tensor:
    value = torch.as_tensor(value).detach().cpu()
    if (value.ndim != 1 or value.dtype != torch.int64 or not value.numel()
            or value.min() < 0 or value.max() >= dimension
            or value.unique().numel() != value.numel()):
        raise ValueError("LTM indices must be unique int64 coordinates in range")
    return value.clone()


def _vector(value: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value).detach()
    if (value.ndim != 1 or not value.numel() or not value.is_floating_point()
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite nonempty floating vector")
    return value


def _fisher_matrix(value: Any, *, device="cpu") -> Tensor:
    if isinstance(value, Mapping):
        value = representation_from_artifact(dict(value), device=device)
    if isinstance(value, Tensor):
        matrix = value.detach().to(device=device, dtype=torch.float64)
    elif hasattr(value, "to_dense"):
        matrix = value.to_dense().detach().to(device=device, dtype=torch.float64)
    elif hasattr(value, "matrix"):
        matrix = value.matrix.detach().to(device=device, dtype=torch.float64)
    else:
        raise TypeError("Fisher must be a square tensor or representation artifact")
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not matrix.shape[0] or not torch.isfinite(matrix).all()):
        raise ValueError("Fisher must be finite, nonempty, and square")
    if not torch.allclose(matrix, matrix.mT, atol=1e-10, rtol=1e-7):
        raise ValueError("Fisher must be symmetric")
    if (matrix.diagonal() < -1e-12).any():
        raise ValueError("Fisher has negative diagonal curvature")
    return ((matrix + matrix.mT) / 2).clone()


def compress_fisher(matrix: Tensor, *, rank: int = 8, seed: int = 0,
                    check_stop: Callable | None = None) -> tuple[dict, dict]:
    """Use the existing seeded rank-plus-diagonal helper; never insert ridge."""
    _integer(rank, "rank", 0)
    _stop(check_stop)
    source = _fisher_matrix(matrix)
    spectrum = torch.linalg.eigvalsh(source)
    minimum = float(spectrum.min())
    tolerance = 100 * source.shape[0] * torch.finfo(source.dtype).eps * max(
        float(spectrum.abs().max()), torch.finfo(source.dtype).tiny)
    if minimum < -tolerance:
        raise ValueError("cannot archive a materially indefinite Fisher")
    _stop(check_stop)
    if not source.count_nonzero():
        representation = LowRankDiagonalFisher(
            source.new_empty((len(source), 0)), source.new_zeros(len(source)))
        compression = {"realized_rank": 0, "seed": seed}
    else:
        def operator(vector):
            try:
                _stop(check_stop)
            except Exception as error:
                raise _CompressionStop from error
            return source @ vector

        # The legacy helper forks its own RNG. Include initialized CUDA devices
        # here as well because torch.manual_seed also seeds CUDA generators.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices):
                result = approximate_low_rank_diagonal(
                    operator, source.diagonal(), rank=min(rank, len(source)), seed=seed)
        except _CompressionStop as error:
            raise error.__cause__
        representation = result.representation
        compression = result.diagnostics.mapping()
    _stop(check_stop)
    return representation.artifact_mapping(), {
        "pre_compression_min_eigenvalue": minimum,
        "pre_compression_negative_mass": float((-spectrum[spectrum < 0]).sum()),
        "source_trace": float(source.trace()),
        "represented_trace": float(representation.diagonal_vector().sum()),
        "compression": compression,
    }


def age_weights(insertion_counts: Sequence[int] | Tensor, cumulative_count: int,
                capacity: int, *, device="cpu", dtype=torch.float64) -> Tensor:
    """Squared clipped observation age, not replay-pass age or normalized mass."""
    _integer(cumulative_count, "cumulative_count", 0)
    _integer(capacity, "capacity")
    counts = torch.as_tensor(insertion_counts, device=device)
    if (counts.ndim != 1 or not torch.isfinite(counts).all()
            or (counts < 0).any() or (counts > cumulative_count).any()
            or (counts != counts.round()).any()):
        raise ValueError("insertion counts must be integral and no later than arrival")
    return ((cumulative_count - counts.to(dtype)) / capacity).clamp(0, 1).square()


def initial_state(theta: Tensor, ltm_indices: Tensor, fisher: Any, *,
                  capacity: int = 32, rank: int = 8, seed: int = 0,
                  check_stop: Callable | None = None) -> dict:
    """Create a checkpoint-friendly state; pretraining is not an online exposure."""
    theta = _vector(theta, "theta").cpu().clone()
    indices = _indices(ltm_indices, theta.numel())
    _integer(capacity, "capacity")
    source = _fisher_matrix(fisher)
    if source.shape != (indices.numel(), indices.numel()):
        raise ValueError("initial Fisher must be the LTM principal block")
    archive, diagnostics = compress_fisher(source, rank=rank, seed=seed,
                                         check_stop=check_stop)
    return {"schema_version": STATE_SCHEMA, "theta": theta,
            "ltm_indices": indices, "anchor_L": theta[indices].clone(),
            "fisher_L": archive, "capacity": capacity, "replay": [],
            "consolidated_ids": [], "cumulative_count": 0,
            "accepted_arrivals": 0, "archive_updates": 0,
            "archive_diagnostics": diagnostics}


def _exposure_ids(values: Sequence) -> list:
    values = list(values)
    if (any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in values)
            or len(values) != len(set(values))):
        raise ValueError("exposure IDs must be distinct strings or integers")
    return values


def validate_state(state: Mapping) -> None:
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("incompatible learner state schema")
    theta = _vector(state["theta"], "theta")
    indices = _indices(state["ltm_indices"], theta.numel())
    anchor = _vector(state["anchor_L"], "anchor_L")
    if (anchor.numel() != indices.numel()
            or _fisher_matrix(state["fisher_L"]).shape != (len(indices), len(indices))):
        raise ValueError("archive and anchor must use LTM coordinates")
    capacity = _integer(state["capacity"], "capacity")
    count = _integer(state["cumulative_count"], "cumulative_count", 0)
    _integer(state["accepted_arrivals"], "accepted_arrivals", 0)
    _integer(state["archive_updates"], "archive_updates", 0)
    replay = state["replay"]
    live = _exposure_ids([record["exposure_id"] for record in replay])
    archived = _exposure_ids(state["consolidated_ids"])
    insertions = [record["insertion_count"] for record in replay]
    age_weights(insertions, count, capacity)
    if (len(replay) > capacity or set(live).intersection(archived)
            or len(live) + len(archived) != count
            or insertions != sorted(insertions)):
        raise ValueError("inconsistent FIFO, exposure ledger, or arrival counts")
    if any("input_ref" not in record for record in replay):
        raise ValueError("replay entries require input references")


def stage_arrival(state: Mapping, exposure_ids: Sequence,
                  input_refs: Sequence) -> dict:
    """Pure FIFO proposal. Ages refer to pre-arrival replay after the new count.

    Every member of one incoming batch has its insertion_count equal to the
    count after that arrival, hence zero age on its initial fit.
    """
    validate_state(state)
    ids = _exposure_ids(exposure_ids)
    refs = list(input_refs)
    if not ids or len(ids) != len(refs):
        raise ValueError("arrival needs one input reference per exposure")
    seen = set(state["consolidated_ids"]) | {r["exposure_id"] for r in state["replay"]}
    if seen.intersection(ids):
        raise ValueError("exposure was already replayed or consolidated")
    count = state["cumulative_count"] + len(ids)
    new = [{"exposure_id": uid, "input_ref": copy.deepcopy(ref),
            "insertion_count": count} for uid, ref in zip(ids, refs, strict=True)]
    working = copy.deepcopy(state["replay"]) + new
    overflow = max(0, len(working) - state["capacity"])
    return {"replay": working[overflow:], "evicted": working[:overflow],
            "new": new, "cumulative_count": count, "working_set_size": len(working),
            "ages": age_weights([r["insertion_count"] for r in state["replay"]],
                                count, state["capacity"])}


def consolidate(fisher: Any, anchor: Tensor, accepted_ltm: Tensor,
                evicted_fisher: Tensor | None, *, evicted_ids: Sequence,
                consolidated_ids: Sequence, pi: float, arrival_size: int,
                rank: int = 8, seed: int = 0,
                check_stop: Callable | None = None) -> dict:
    """Pure exactly-once eviction update, with eta=1-(1-pi)**(e/m)."""
    if not math.isfinite(pi) or not 0 < pi <= 1:
        raise ValueError("pi must lie in (0, 1]")
    _integer(arrival_size, "arrival_size")
    ids, old_ids = _exposure_ids(evicted_ids), _exposure_ids(consolidated_ids)
    if set(ids).intersection(old_ids):
        raise ValueError("exposure already consolidated")
    _stop(check_stop)
    old = _fisher_matrix(fisher)
    anchor = _vector(anchor, "anchor").cpu()
    accepted_ltm = _vector(accepted_ltm, "accepted_ltm").cpu()
    if anchor.shape != accepted_ltm.shape or old.shape != (len(anchor), len(anchor)):
        raise ValueError("archive and LTM anchor dimensions must agree")
    if not ids:
        if evicted_fisher is not None:
            raise ValueError("a Fisher update requires actual evicted exposures")
        return {"fisher_L": copy.deepcopy(fisher), "anchor_L": anchor.clone(),
                "consolidated_ids": old_ids, "eta": 0., "evicted_count": 0}
    if evicted_fisher is None:
        raise ValueError("eviction requires the accepted-state model-label Fisher")
    fresh = _fisher_matrix(evicted_fisher)
    if fresh.shape != old.shape:
        raise ValueError("evicted Fisher must be the matching LTM block")
    eta = 1. if pi == 1. else -math.expm1(len(ids) / arrival_size * math.log1p(-pi))
    archived, diagnostics = compress_fisher((1 - eta) * old + eta * fresh,
                                          rank=rank, seed=seed, check_stop=check_stop)
    return {"fisher_L": archived, "anchor_L": accepted_ltm.clone(),
            "consolidated_ids": old_ids + ids, "eta": eta,
            "evicted_count": len(ids), "archive_diagnostics": diagnostics}


def commit_arrival(state: Mapping, accepted_theta: Tensor, exposure_ids: Sequence,
                   input_refs: Sequence, evicted_fisher: Tensor | None = None, *,
                   pi: float, arrival_size: int, rank: int = 8, seed: int = 0,
                   check_stop: Callable | None = None) -> tuple[dict, dict]:
    """Return the whole accepted transaction; never modify the prior checkpoint.

    ``evicted_fisher`` must be calculated on stage_arrival()['evicted'] at
    accepted_theta, after the caller's predictive evaluation. Persist the
    returned state atomically together with the corresponding metric chunk.
    """
    _stop(check_stop)
    staged = stage_arrival(state, exposure_ids, input_refs)
    accepted = _vector(accepted_theta, "accepted_theta").cpu().clone()
    if accepted.shape != state["theta"].shape:
        raise ValueError("accepted parameter layout changed")
    updated = consolidate(
        state["fisher_L"], state["anchor_L"], accepted[state["ltm_indices"]],
        evicted_fisher, evicted_ids=[r["exposure_id"] for r in staged["evicted"]],
        consolidated_ids=state["consolidated_ids"], pi=pi, arrival_size=arrival_size,
        rank=rank, seed=seed, check_stop=check_stop)
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
                    "replay_size": len(staged["replay"])}


@dataclasses.dataclass(frozen=True)
class FitSettings:
    """Frozen local solve contract; first_order is an explicit fallback cohort."""

    pi: float = .025
    ridge_ratio: float = .1
    replay_weight: float = .5
    lambda_f: float = 0.
    lambda_d: float = 0.
    num_probes: int = 4
    fd_step: float = .001
    direction_max_norm: float = .1
    direction_min_norm: float = 1e-12
    shadow_max_eval: int = 5
    max_iter: int = 50
    max_eval: int = 75
    history_size: int = 20
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    solver: str = "lbfgs"
    learning_rate: float = 1.
    first_order_lr: float = .01
    chunk_size: int = 32

    def validate(self) -> None:
        for name in ("num_probes", "shadow_max_eval", "max_iter", "max_eval",
                     "history_size", "chunk_size"):
            _integer(getattr(self, name), name)
        if self.shadow_max_eval > 5:
            raise ValueError("directional shadow proposal is bounded at five closures")
        if self.solver not in {"lbfgs", "first_order", "adam"}:
            raise ValueError("solver must be lbfgs or the explicit first_order/adam fallback")
        for name in ("pi", "fd_step", "direction_max_norm", "learning_rate", "first_order_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.pi > 1:
            raise ValueError("pi must not exceed one")
        if not math.isfinite(self.replay_weight) or not 0 < self.replay_weight < 1:
            raise ValueError("replay_weight must lie strictly between zero and one")
        for name in ("ridge_ratio", "lambda_f", "lambda_d", "direction_min_norm",
                     "tolerance_grad", "tolerance_change"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.direction_min_norm >= self.direction_max_norm:
            raise ValueError("direction minimum must be below the radius cap")


def _settings(value: FitSettings | Mapping) -> FitSettings:
    result = value if isinstance(value, FitSettings) else FitSettings(**dict(value))
    result.validate()
    return result


def retention_terms(theta: Tensor, ltm_indices: Tensor, stm_indices: Tensor,
                    fisher: Any, anchor: Tensor, stm_anchor: Tensor, *,
                    pi: float, ridge_ratio: float = .1) -> tuple[Tensor, Tensor, float]:
    """EWC protects LTM only; ridge uses archive-LTM and pre-step-STM anchors."""
    matrix = _fisher_matrix(fisher, device=theta.device)
    ltm, stm = ltm_indices.to(theta.device), stm_indices.to(theta.device)
    anchor = anchor.detach().to(device=theta.device, dtype=torch.float64)
    stm_anchor = stm_anchor.detach().to(device=theta.device, dtype=torch.float64)
    if (matrix.shape != (ltm.numel(), ltm.numel()) or anchor.shape != (ltm.numel(),)
            or stm_anchor.shape != (stm.numel(),)):
        raise ValueError("retention terms require separate LTM and STM anchor coordinates")
    if not 0 < pi <= 1 or not math.isfinite(ridge_ratio) or ridge_ratio < 0:
        raise ValueError("invalid retention strength")
    omega = (1 - pi) / pi
    dl = theta[ltm].double() - anchor
    ds = theta[stm].double() - stm_anchor
    kappa = ridge_ratio * float(matrix.trace() / len(ltm))
    return omega / 2 * (dl @ matrix @ dl), omega * kappa / 2 * (
        dl.square().sum() + ds.square().sum()), kappa


def prepare_objective(model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
                      replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any,
                      anchor: Tensor, condition: str, settings: FitSettings | Mapping,
                      seed: int, check_stop: Callable | None = None, *,
                      direction: Tensor | None = None) -> tuple[Callable, dict]:
    """Freeze all solve inputs, returning a differentiable objective and metadata.

    Callable output is (scalar loss, component tensor dictionary). Curvature
    functions differentiate probabilities and perturbations into both blocks.
    Only the distillation teacher and a proposed STM direction are detached.
    """
    from .model import flat_logits, memory_indices
    from .curvature import jvp_values, kl_values, step_kl_values

    cfg = _settings(settings)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown learner condition: {condition}")
    start = _vector(theta, "theta").clone()
    ltm, stm = (index.to(start.device) for index in memory_indices(layout))
    new_x, replay_x = new_x.detach(), replay_x.detach()
    new_y, replay_y = new_y.detach(), replay_y.detach()
    if (new_y.ndim != 1 or new_y.numel() == 0 or replay_y.ndim != 1
            or new_x.shape[0] != new_y.numel() or replay_x.shape[0] != replay_y.numel()):
        raise ValueError("new and replay input/label counts must match, with nonempty new data")
    ages = torch.as_tensor(ages, device=start.device, dtype=torch.float64).detach().clone()
    if (ages.shape != replay_y.shape or not torch.isfinite(ages).all()
            or ((ages < 0) | (ages > 1)).any()):
        raise ValueError("ages are fixed squared age weights, one value in [0,1] per replay input")
    archive = _fisher_matrix(fisher, device=start.device)
    fixed_anchor = anchor.detach().to(start.device).clone()
    # Resolve and validate retention inputs once, not by mutating a closure anchor.
    _, _, kappa = retention_terms(start, ltm, stm, archive, fixed_anchor, start[stm],
                                 pi=cfg.pi, ridge_ratio=cfg.ridge_ratio)
    probe_seed = derive_component_seed(seed, "training_curvature_probes")
    generator = torch.Generator(device="cpu").manual_seed(probe_seed)
    probes = (2 * torch.randint(2, (cfg.num_probes, stm.numel()), generator=generator)
              - 1).to(device=start.device, dtype=start.dtype)
    fixed_direction = None if direction is None else direction.detach().to(start).clone()
    if condition == "directional":
        if fixed_direction is None or fixed_direction.shape != (stm.numel(),):
            raise ValueError("directional objective requires a detached STM-coordinate proposal")
        if not torch.isfinite(fixed_direction).all():
            raise ValueError("direction must be finite")
    teacher = None
    if condition == "distillation" and len(replay_y) and cfg.lambda_d:
        parts = []
        with torch.no_grad():
            for x in replay_x.split(cfg.chunk_size):
                _stop(check_stop)
                parts.append(F.log_softmax(flat_logits(model, layout, start, x).double(), dim=-1))
        teacher = torch.cat(parts).detach()

    def objective(candidate: Tensor) -> tuple[Tensor, dict]:
        _stop(check_stop)
        zero = candidate.sum() * 0.

        def mean_nll(x, y):
            value = zero
            for offset in range(0, len(y), cfg.chunk_size):
                _stop(check_stop)
                logits = flat_logits(model, layout, candidate, x[offset:offset + cfg.chunk_size])
                value = value + F.cross_entropy(
                    logits, y[offset:offset + cfg.chunk_size], reduction="none").double().sum() / len(y)
            return value

        new_loss = mean_nll(new_x, new_y)
        replay_loss = mean_nll(replay_x, replay_y) if len(replay_y) else zero
        data_loss = ((1 - cfg.replay_weight) * new_loss + cfg.replay_weight * replay_loss
                     if len(replay_y) else new_loss)
        ewc, ridge, _ = retention_terms(candidate, ltm, stm, archive, fixed_anchor,
                                       start[stm], pi=cfg.pi, ridge_ratio=cfg.ridge_ratio)
        auxiliary = zero
        strength = cfg.lambda_d if condition == "distillation" else cfg.lambda_f
        if condition in {"jvp", "finite_difference", "directional", "distillation"} and strength:
            for offset in range(0, len(replay_y), cfg.chunk_size):
                _stop(check_stop)
                x = replay_x[offset:offset + cfg.chunk_size]
                weights = ages[offset:offset + cfg.chunk_size]
                if condition == "jvp":
                    values = jvp_values(model, layout, candidate, x, probes).mean(0) / stm.numel()
                elif condition == "finite_difference":
                    values = kl_values(model, layout, candidate, x, probes, cfg.fd_step).mean(0) / stm.numel()
                elif condition == "directional":
                    if float(fixed_direction.norm()) <= cfg.direction_min_norm:
                        values = zero.expand(len(x))
                    else:
                        values = step_kl_values(model, layout, candidate, x, fixed_direction,
                                                min_norm=cfg.direction_min_norm)
                else:
                    log_student = F.log_softmax(flat_logits(
                        model, layout, candidate, x, stm_scale=0.).double(), dim=-1)
                    log_teacher = teacher[offset:offset + cfg.chunk_size]
                    values = (log_teacher.exp() * (log_teacher - log_student)).sum(-1)
                auxiliary = auxiliary + (weights * values.double()).sum() / len(replay_y)
        total = data_loss + ewc + ridge + strength * auxiliary
        return total, {"new_nll": new_loss, "replay_nll": replay_loss,
                       "data_loss": data_loss, "ewc": ewc, "ridge": ridge,
                       "auxiliary": auxiliary, "weighted_auxiliary": strength * auxiliary}

    return objective, {"probe_seed": probe_seed, "kappa": kappa,
                       "num_probes": cfg.num_probes, "fd_step": cfg.fd_step,
                       "fd_perturbation_norm": cfg.fd_step * math.sqrt(stm.numel()),
                       "teacher_detached": teacher is not None,
                       "direction_norm": 0. if fixed_direction is None else float(fixed_direction.norm())}


class _SolveLimit(Exception):
    pass


def _bounded_solve(start: Tensor, objective: Callable, cfg: FitSettings,
                   check_stop: Callable | None) -> tuple[Tensor, dict]:
    """Accept only completed, evaluated nonincreasing steps, never Wolfe trials."""
    parameter = torch.nn.Parameter(start.detach().clone())
    accepted = start.detach().clone()
    accepted_record = initial_record = None
    evaluations = calls = iterations = accepted_steps = 0
    reason = "max_iterations"
    rejected_increase = roundoff_tolerance = 0.
    started = time.perf_counter()

    def evaluate():
        nonlocal evaluations
        _stop(check_stop)
        evaluations += 1
        if not torch.isfinite(parameter).all():
            raise _SolveLimit("nonfinite_parameters")
        value, parts = objective(parameter)
        gradient = torch.autograd.grad(value, parameter)[0]
        if not torch.isfinite(value).all() or not torch.isfinite(gradient).all():
            raise _SolveLimit("nonfinite_objective_or_gradient")
        return {"theta": parameter.detach().clone(), "value": float(value.detach()),
                "gradient": gradient.detach().clone(),
                "parts": {key: float(v.detach()) for key, v in parts.items()}}

    optimizer = (torch.optim.LBFGS(
        [parameter], lr=cfg.learning_rate, max_iter=1, max_eval=cfg.max_eval,
        tolerance_grad=cfg.tolerance_grad, tolerance_change=cfg.tolerance_change,
        history_size=cfg.history_size, line_search_fn="strong_wolfe")
        if cfg.solver == "lbfgs" else torch.optim.Adam([parameter], lr=cfg.first_order_lr))
    try:
        for _ in range(cfg.max_iter):
            _stop(check_stop)
            records = [] if accepted_record is None else [accepted_record]

            def closure():
                nonlocal calls, initial_record, accepted_record
                _stop(check_stop)
                if calls >= cfg.max_eval:
                    raise _SolveLimit("max_evaluations")
                calls += 1
                record = next((r for r in records if torch.equal(r["theta"], parameter.detach())), None)
                if record is None:
                    record = evaluate()
                    records.append(record)
                if accepted_record is None:
                    initial_record = accepted_record = record
                parameter.grad = record["gradient"].clone()
                return parameter.new_tensor(record["value"], dtype=torch.float64)

            if cfg.solver == "lbfgs":
                optimizer.step(closure)
            else:
                closure()
                if float(parameter.grad.abs().max()) <= cfg.tolerance_grad:
                    reason = "gradient_tolerance"
                    break
                optimizer.step()
                closure()  # Never accept an unevaluated final Adam displacement.
            iterations += 1
            candidate = next((r for r in records if torch.equal(r["theta"], parameter.detach())), None)
            if candidate is None:
                raise _SolveLimit("unevaluated_candidate")
            if candidate["value"] > accepted_record["value"]:
                rejected_increase = candidate["value"] - accepted_record["value"]
                roundoff_tolerance = (16 * torch.finfo(parameter.dtype).eps
                                      * max(1., abs(accepted_record["value"])))
                # Keep the previous nonincreasing iterate even for a rounding-sized increase.
                raise _SolveLimit("roundoff_stagnation" if rejected_increase <= roundoff_tolerance
                                  else "objective_increase")
            moved = not torch.equal(accepted, candidate["theta"])
            change = abs(candidate["value"] - accepted_record["value"])
            accepted = candidate["theta"].clone()
            accepted_record = candidate
            accepted_steps += int(moved)
            if float(candidate["gradient"].abs().max()) <= cfg.tolerance_grad:
                reason = "gradient_tolerance"
                break
            if not moved or change <= cfg.tolerance_change:
                reason = "change_tolerance"
                break
        else:
            reason = "max_iterations"
    except _SolveLimit as error:
        reason = str(error)
    _stop(check_stop)
    final = {} if accepted_record is None else accepted_record["parts"]
    return accepted, {
        "solver": cfg.solver, "solver_reason": reason, "closure_calls": calls,
        "objective_evaluations": evaluations, "iterations": iterations,
        "accepted_steps": accepted_steps, "initial_objective": None if initial_record is None else initial_record["value"],
        "final_objective": None if accepted_record is None else accepted_record["value"],
        "objective_decrease": None if initial_record is None else initial_record["value"] - accepted_record["value"],
        "gradient_inf_norm": None if accepted_record is None else float(accepted_record["gradient"].abs().max()),
        "converged": accepted_record is not None and float(accepted_record["gradient"].abs().max()) <= cfg.tolerance_grad,
        "evaluation_limit_reached": reason == "max_evaluations",
        "parameter_change_norm": float((accepted - start).double().norm()),
        "rejected_objective_increase": rejected_increase,
        "objective_roundoff_tolerance": roundoff_tolerance,
        "solver_failed": reason in {"objective_increase", "unevaluated_candidate",
                                      "nonfinite_parameters", "nonfinite_objective_or_gradient"},
        "solver_seconds": time.perf_counter() - started,
        **{f"final_{key}": value for key, value in final.items()},
    }


def fit_step(model, layout, theta: Tensor, new_x: Tensor, new_y: Tensor,
             replay_x: Tensor, replay_y: Tensor, ages: Tensor, fisher: Any,
             anchor: Tensor, condition: str, settings: FitSettings | Mapping,
             seed: int, check_stop: Callable | None = None, *,
             objective_observer: Callable | None = None) -> tuple[Tensor, dict]:
    """Fit one arrival without mutation or I/O; ``ages`` are squared age weights.

    ``fisher`` and ``anchor`` contain LTM coordinates only. All settings, probes,
    teacher probabilities, and shadow directions remain fixed within the solve.
    The caller must not commit an interrupted fit. Explicit first_order/adam
    settings provide a versionable fallback, never a hidden mid-cohort switch.
    """
    from .model import memory_indices

    _stop(check_stop)
    cfg = _settings(settings)
    start = _vector(theta, "theta").clone()
    started = time.perf_counter()
    direction = None
    shadow_diagnostics = {}
    if condition == "directional":
        ordinary, _ = prepare_objective(
            model, layout, start, new_x, new_y, replay_x, replay_y, ages, fisher,
            anchor, "replay_ewc", cfg, seed, check_stop)
        shadow_cfg = dataclasses.replace(cfg, max_eval=cfg.shadow_max_eval,
                                         max_iter=min(cfg.max_iter, cfg.shadow_max_eval))
        proposed, shadow_diagnostics = _bounded_solve(start, ordinary, shadow_cfg, check_stop)
        _, stm = memory_indices(layout)
        direction = (proposed - start)[stm.to(start.device)].detach()
        norm = float(direction.norm())
        if norm <= cfg.direction_min_norm:
            direction = torch.zeros_like(direction)
        elif norm > cfg.direction_max_norm:
            direction = direction * (cfg.direction_max_norm / norm)
        shadow_diagnostics["uncapped_direction_norm"] = norm
    objective, metadata = prepare_objective(
        model, layout, start, new_x, new_y, replay_x, replay_y, ages, fisher,
        anchor, condition, cfg, seed, check_stop, direction=direction)
    accepted, diagnostics = _bounded_solve(start, objective, cfg, check_stop)
    _stop(check_stop)
    diagnostics.update(metadata)
    diagnostics.update(condition=condition, seed=seed,
                       elapsed_seconds=time.perf_counter() - started,
                       unsupported_direction=condition == "directional" and not bool(direction.count_nonzero()),
                       total_closure_calls=diagnostics["closure_calls"] + shadow_diagnostics.get("closure_calls", 0))
    diagnostics.update({f"shadow_{key}": value for key, value in shadow_diagnostics.items()})
    if objective_observer is not None:
        objective_observer(accepted.detach(), objective, metadata, diagnostics)
    return accepted.detach(), diagnostics
