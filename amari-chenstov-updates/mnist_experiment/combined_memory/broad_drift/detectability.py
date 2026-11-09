"""Bounded offline common-anchor score comparisons on diagnostic-role grids."""

from __future__ import annotations

import math

import torch

from ..curvature import observed_scores
from ..model import build_model, merge_parameters
from .data import render_panel, sample_transforms


def score_summary(nominal, center, minus, plus, metric, *, p, pminus, pplus,
                  lag, batch_size, bootstrap_seed, bootstrap_draws=500, check=lambda: None):
    """Analytic mixture moments and paired mean-difference uncertainty.

    Rows are mapped positive observed-label scores; metrics are fixed across
    the compared states. Bootstrap rows preserve pairing across all endpoints.
    A split-half cross product exposes squared sample-mean noise inflation.
    """
    arrays = [value.detach().cpu().double() for value in (nominal, center, minus, plus)]
    nominal, center, minus, plus = arrays
    metric = metric.detach().cpu().double()
    if (nominal.ndim != 2 or len(nominal) < 4 or any(v.shape != nominal.shape for v in arrays)
            or metric.shape != (nominal.shape[1],) or lag <= 0 or batch_size < 1
            or not all(0 <= v <= 1 for v in (p, pminus, pplus))):
        raise ValueError("invalid paired detectability inputs")
    if not all(torch.isfinite(v).all() for v in [*arrays, metric]) or (metric < 0).any():
        raise ValueError("finite scores and a nonnegative fixed metric required")
    mu0, mur = nominal.mean(0), center.mean(0)
    variance0 = nominal.var(0, unbiased=True)
    variancer = center.var(0, unbiased=True)
    v0, vr = float((metric * variance0).sum()), float((metric * variancer).sum())
    between = float((metric * (mur-mu0).square()).sum())
    mixture_v = (1-p)*v0 + p*vr + p*(1-p)*between
    differences = {"rare": (pplus*(plus-nominal)-pminus*(minus-nominal))/lag,
                   "broad": (plus-minus)/lag}
    rng = torch.Generator().manual_seed(bootstrap_seed)
    # Share draws across laws; no large draws-by-samples-by-coordinates tensor.
    draws = torch.randint(len(nominal), (bootstrap_draws, len(nominal)), generator=rng)
    result = {}
    for name, delta in differences.items():
        mean = delta.mean(0)
        energy = float((metric * mean.square()).sum())
        middle = len(delta)//2
        cross = float((metric * delta[:middle].mean(0) * delta[middle:].mean(0)).sum())
        monte_carlo_variance = float((metric * delta.var(0, unbiased=True)/len(delta)).sum())
        estimates = []
        for index, indices in enumerate(draws):
            if index % 25 == 0:
                check()
            estimates.append(float((metric*delta[indices].mean(0).square()).sum()))
        estimates = torch.tensor(estimates, dtype=torch.float64)
        noise = (mixture_v if name == "rare" else vr)/batch_size
        result[name] = dict(motion_energy=energy, split_motion_energy=cross,
            mean_difference_mc_variance=monte_carlo_variance, batch_noise=noise,
            motion_to_batch_noise=energy/noise if noise > 0 else None,
            split_motion_to_batch_noise=cross/noise if noise > 0 else None,
            motion_ci=torch.quantile(estimates, torch.tensor([.025,.975], dtype=torch.float64)).tolist(),
            diagnostic_samples=len(delta), bootstrap_draws=bootstrap_draws)
    result["mixture_components"] = dict(nominal_variance=v0, rotated_variance=vr,
        between_component_energy=between, mixture_variance=mixture_v)
    return result


def run_diagnostic(scheduler, replica):
    cfg, store, budget = scheduler.cfg, scheduler.store, scheduler.budget
    uid = f"diagnostic-detectability-{replica:03}"
    metadata = dict(kind="diagnostic", name="paired-score-detectability", replica=replica,
        grids=cfg.diagnostic_size, training=False, live_controller_input=False)
    if store.begin(uid, metadata) is None:
        from .runtime import read_json
        return read_json(store.unit_path(uid)/"summary.json")
    assets = scheduler.assets("development", replica)
    initial = scheduler.initial("development", replica)
    model, layout = build_model(cfg.seed_for("development", replica, "initialization_model"),
        device=cfg.device, dtype=getattr(torch, cfg.dtype),
        factor_seed=cfg.seed_for("development", replica, "factors"))
    model, layout, anchor = merge_parameters(model, layout, initial["theta"].to(cfg.device))
    metric = initial["diagonal"].double()
    floor = max(cfg.observer_floor_absolute, cfg.observer_floor_relative*float(metric.mean()))
    metric = metric.clamp_min(floor)
    panel = assets["diagnostic"]

    def scores(h):
        means = (h*cfg.angle_max*assets["signs"]).expand(cfg.diagnostic_size,9)
        transforms, _ = sample_transforms(means.reshape(-1), assets["seeds"]["diagnostic:transforms"],
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
            uniforms=panel["uniforms"].reshape(-1,6))
        x, y = render_panel(scheduler.raw, dict(panel, transforms=transforms.reshape(-1,9,6)),
                            check=budget.check)
        s = observed_scores(model, layout, anchor, x.to(cfg.device), y.to(cfg.device),
                            chunk_size=cfg.fisher_chunk, check=budget.check)
        return s.cpu()/metric

    state = store.load_checkpoint(uid) or dict(replica=replica, states=[])
    nominal = scores(0.)
    centers = (64,128,192,256) if not cfg.smoke else (max(1,cfg.ramp_steps//2),)
    distance = 4 if not cfg.smoke else 1
    for j in centers[len(state["states"]):]:
        budget.check()
        left, right = max(0,j-distance), min(cfg.ramp_steps,j+distance)
        h = lambda k: (1-math.cos(math.pi*k/cfg.ramp_steps))/2
        center, minus, plus = scores(h(j)), scores(h(left)), scores(h(right))
        result = score_summary(nominal, center, minus, plus, metric,
            p=.125*h(j), pminus=.125*h(left), pplus=.125*h(right),
            lag=right-left, batch_size=cfg.samples_per_step,
            bootstrap_seed=cfg.seed_for("development",replica,f"detectability-bootstrap:{j}"),
            bootstrap_draws=100 if cfg.smoke else 500, check=budget.check)
        state["states"].append(dict(outbound_batch=j, severity=h(j), **result))
        store.save_checkpoint(uid, state)
        scheduler.pulse(unit=uid, batch=len(state["states"]), total=len(centers))
    state.update(metric="fixed floored initial likelihood Fisher diagonal",
        uncertainty="paired diagnostic-row bootstrap; not trained-trajectory intervals",
        qualification="Raw energy includes Monte Carlo bias; inspect split-half energy and MC variance. "
            "Finite-population source rows are not exactly independent.",
        controller_input=False)
    store.write_json(uid, "summary.json", state)
    store.complete(uid)
    return state
