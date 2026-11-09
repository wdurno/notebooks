"""Mean likelihood, diagonal LTM protection, and differentiable STM sensitivity."""

from __future__ import annotations

import math
import time

import torch

from mnist_experiment.kalman_pi.policies import concentration
from .config import Recipe
from .curvature import fd_values
from .model import flat_prediction, gaussian_nll, memory_indices


def archive_update(diagonal, delta, fresh, *, pi, evicted_count, nominal_arrival,
                   q_archive, transport):
    """Transport old evidence before adding disjoint, actually retired evidence."""
    if not 0 < pi < 1 or evicted_count < 0 or nominal_arrival < 1:
        raise ValueError("invalid retirement weight")
    old = diagonal.double()
    if old.ndim != 1 or not torch.isfinite(old).all() or (old < 0).any():
        raise ValueError("finite nonnegative archived diagonal required")
    correction = torch.zeros_like(old) if not transport else delta.double()
    if correction.shape != old.shape or not torch.isfinite(correction).all():
        raise FloatingPointError("invalid finite-change correction")
    eta = -math.expm1(evicted_count / nominal_arrival * math.log1p(-pi)) if evicted_count else 0.
    raw = (1 - eta) * (old + correction)
    if evicted_count:
        if fresh is None or fresh.shape != old.shape or not torch.isfinite(fresh).all() or (fresh < 0).any():
            raise ValueError("accepted-parameter evictee Fisher is required")
        raw = raw + eta * fresh.double()
    elif fresh is not None:
        raise ValueError("Fisher evidence cannot be added without eviction")
    if not torch.isfinite(raw).all():
        raise FloatingPointError("nonfinite unprojected archive")
    projected = raw.clamp_min(0)
    scale = max(float(old.sum()), 1e-30)
    diagnostics = dict(eta=eta, evicted_count=evicted_count,
        negative_fraction=float((raw < 0).double().mean()),
        negative_mass=float((-raw.clamp_max(0)).sum()) / scale,
        projection_norm=float((projected - raw).norm()),
        correction_norm=float(correction.norm()), archive_trace=float(projected.sum()))
    next_q = ((1 - eta)**2 * q_archive + eta**2 / evicted_count
              if evicted_count else q_archive)
    return projected, next_q, diagnostics


def prepare_objective(model, layout, start, diagonal, anchor, new_x, new_y,
                      replay_x, replay_y, ages, probes, *, pi, recipe, cfg):
    ltm, stm = (indices.to(start.device) for indices in memory_indices(layout))
    d = diagonal.detach().to(start.device, torch.float64).clone()
    alpha = anchor.detach().to(start.device, torch.float64).clone()
    previous_s = start[stm].detach().double().clone()
    omega = (1 - pi) / pi
    scale_fisher = float(d.mean())
    kappa = recipe.ridge * scale_fisher
    fixed_ages = ages.to(start.device, torch.float64).detach().clone()
    fixed_probes = probes.to(start.device).detach().clone()
    if d.shape != ltm.shape or alpha.shape != ltm.shape:
        raise ValueError("LTM archive and anchor dimensions differ from the layout")
    if fixed_ages.shape != (len(replay_y),):
        raise ValueError("one age is required per replay grid")
    if ((fixed_ages < 0) | (fixed_ages > 1)).any():
        raise ValueError("squared clipped ages must lie in [0,1]")

    def objective(candidate):
        new_loss = gaussian_nll(flat_prediction(model, layout, candidate, new_x), new_y).double().mean()
        zero = candidate.double().sum() * 0
        replay_loss = (gaussian_nll(flat_prediction(model, layout, candidate, replay_x), replay_y)
                       .double().mean() if len(replay_y) else zero)
        data_loss = ((1 - cfg.replay_weight) * new_loss + cfg.replay_weight * replay_loss
                     if len(replay_y) else new_loss)
        dl, ds = candidate[ltm].double() - alpha, candidate[stm].double() - previous_s
        ewc = omega / 2 * (d * dl.square()).sum()
        ridge = omega * kappa / 2 * (dl.square().sum() + ds.square().sum())
        auxiliary = zero
        if recipe.strength and len(replay_y):
            fd_parameter = candidate.double() if cfg.fd_dtype == "float64" else candidate
            values = fd_values(model, layout, fd_parameter, replay_x.to(fd_parameter.dtype),
                               fixed_probes.to(fd_parameter.dtype), cfg.fd_step)
            auxiliary = (fixed_ages * values.double().mean(0) / len(stm)).mean()
        value = data_loss + ewc + ridge + recipe.strength * auxiliary
        return value, dict(data_loss=data_loss, new_nll=new_loss, replay_nll=replay_loss,
                           ewc=ewc, ridge=ridge, auxiliary=auxiliary,
                           weighted_auxiliary=recipe.strength * auxiliary)

    return objective, dict(ltm=ltm, stm=stm, s=scale_fisher, kappa=kappa, omega=omega)


class _TrialLimit(Exception):
    pass


def fit_step(model, layout, start, diagonal, anchor, new_x, new_y,
             replay_x, replay_y, ages, probes, *, pi, recipe, cfg,
             initial_fisher_scale, check=lambda: None):
    """Optimize in fixed scaled coordinates and return a finite evaluated best."""
    objective, info = prepare_objective(model, layout, start, diagonal, anchor,
        new_x, new_y, replay_x, replay_y, ages, probes, pi=pi, recipe=recipe, cfg=cfg)
    ltm, stm = info["ltm"], info["stm"]
    floor = max(1e-12, 1e-8 * initial_fisher_scale)
    scales = torch.empty_like(start, dtype=torch.float64)
    d = diagonal.to(start.device, torch.float64)
    scales[ltm] = (d + info["omega"] * (d + info["kappa"]) + floor).rsqrt()
    scales[stm] = (info["s"] + info["omega"] * info["kappa"] + floor)**-.5
    clipped = float(((scales < 1e-4) | (scales > 1e4)).double().mean())
    scales = scales.clamp(1e-4, 1e4).to(start.dtype).detach()
    coordinate = torch.zeros_like(start, requires_grad=True)
    optimizer = torch.optim.LBFGS([coordinate], lr=1., max_iter=1, max_eval=cfg.max_eval,
        history_size=cfg.history_size, tolerance_grad=0., tolerance_change=cfg.tolerance_change,
        line_search_fn="strong_wolfe")
    calls, nonfinite, best, initial_value = 0, 0, None, None
    reason, started = "iteration_limit", time.perf_counter()

    def closure():
        nonlocal calls, nonfinite, best, initial_value
        check()
        if calls >= cfg.max_eval:
            raise _TrialLimit("evaluation_limit")
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        theta = start + scales * coordinate
        value, parts = objective(theta)
        gradient = torch.autograd.grad(value, coordinate)[0]
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            nonfinite += 1
            if best is None:
                raise FloatingPointError("invalid initial physical objective/gradient")
            raise _TrialLimit("nonfinite_trial")
        coordinate.grad = gradient
        number = float(value.detach())
        if initial_value is None:
            initial_value = number
        if best is None or number < best["objective"]:
            best = dict(theta=theta.detach().clone(), coordinate=coordinate.detach().clone(),
                objective=number, gradient=float((gradient.detach() / scales).abs().max()),
                parts={key: float(item.detach()) for key, item in parts.items()})
        return value

    closure()
    try:
        for _ in range(cfg.max_iter):
            check()
            previous = best["objective"]
            optimizer.step(closure)
            if not torch.equal(coordinate.detach(), best["coordinate"]):
                with torch.no_grad():
                    coordinate.copy_(best["coordinate"])
                optimizer.state.clear()
            if best["gradient"] <= cfg.tolerance_grad:
                reason = "gradient_tolerance"
                break
            if abs(previous - best["objective"]) <= cfg.tolerance_change:
                reason = "change_tolerance"
                break
    except _TrialLimit as error:
        reason = str(error)
    if best is None:
        raise FloatingPointError("solve produced no evaluated finite point")
    return best["theta"], dict(solver_seconds=time.perf_counter() - started,
        closure_count=calls, solver_reason=reason, gradient_inf_norm=best["gradient"],
        initial_objective=initial_value, final_objective=best["objective"],
        objective_decrease=initial_value - best["objective"],
        moved=not torch.equal(best["theta"], start), nonfinite_trials=nonfinite,
        scale_clip_fraction=clipped, s=info["s"], kappa=info["kappa"], **best["parts"])
