"""Development-only scales and reversible numerical probes, never fresh outcomes."""

from dataclasses import replace
import math
import time

import torch

from numerical_experiment.picar.fix3_student import _batch
from numerical_experiment.picar.learner import MemoryLearner
from src.continual import picar_math as pm
from src.robotics.actions import ACTION_VECTORS


def bootstrap_statistics(records):
    if len(records) < 8:
        raise ValueError("a nontrivial recorded initialization dataset is required")
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float64)
    baselines = torch.tensor([r["baseline"] for r in records], dtype=torch.float64)
    if not torch.isfinite(returns).all() or not torch.isfinite(baselines).all():
        raise ValueError("nonfinite recorded return or baseline")
    if returns.min() < -6 or returns.max() > 12:
        raise ValueError("recorded returns violate the unchanged reward bound")
    advantage = returns-baselines
    quantiles = advantage.quantile(torch.tensor([.1, .25, .5, .75, .9], dtype=torch.float64))
    spread = max(float((quantiles[-1]-quantiles[0])/2.5631031311), 18/161)
    rows = []
    for multiplier in (.5, 1., 2.):
        temperature = spread*multiplier
        logw = advantage/temperature
        probabilities = logw.softmax(0)
        effective_fraction = float(1/probabilities.square().sum()/len(records))
        rows.append({"temperature": temperature, "spread_multiplier": multiplier,
            "weight_effective_fraction": effective_fraction,
            "maximum_weight_share": float(probabilities.max()),
            "weight_p90_to_p10": float(torch.exp((quantiles[-1]-quantiles[0])/temperature)),
            "log_weight_min": float(logw.min()), "log_weight_max": float(logw.max()),
            "numeric_weight_safe": bool(logw.abs().max() <= 60),
            "cap_hits": 0})
    admissible = [r for r in rows if r["weight_effective_fraction"] >= .25
                  and r["maximum_weight_share"] <= .1 and r["numeric_weight_safe"]]
    if not admissible:
        raise ValueError("no bounded, nondegenerate temperature in the admitted three-point bracket")
    selected = min(admissible, key=lambda r: r["temperature"])
    z = torch.stack([r["logits"].reshape(7).double() for r in records])
    return {"records": len(records), "return_min": float(returns.min()), "return_max": float(returns.max()),
        "return_mean": float(returns.mean()), "return_std": float(returns.std()),
        "advantage_quantiles": quantiles.tolist(), "robust_spread": spread,
        "temperature_rows": rows, "selected_temperature": selected["temperature"],
        "selection_rule": "Smallest half/base/double robust spread with effective weight fraction >=.25 and maximum share <=.1.",
        "action_variance": 1., "teacher_logit_variance_max": float(z.var(0).max()),
        "action_scale_reason": "Teacher actions are constant; retain the documented isotropic unit scale rather than estimate a singular empirical covariance.",
        "weight_effective_count_is_iid_sample_size": False,
        "new_observations": 0, "return_range": [-6., 12.], "return_bins": 161}


def explore_controls(means, calibration, *, seed=193031, draws=2048):
    """Sample hypothetical Gaussian logits, not environment transitions."""
    means = torch.as_tensor(means, dtype=torch.float64, device="cpu")
    if means.ndim != 2 or means.shape[1] != 7 or not len(means) or not torch.isfinite(means).all():
        raise ValueError("finite retained-history means [N,7] are required")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(draws, len(means), 7, dtype=torch.float64, generator=generator)
    vectors = torch.tensor(ACTION_VECTORS, dtype=torch.float64)
    rows = []
    for std in (2., 4., 6., 8.):
        z = means[None]+std*noise
        q = torch.cat((z, z.new_zeros(*z.shape[:-1], 1)), -1).softmax(-1)
        controls = q@vectors
        steering = controls[..., 2]*calibration.steering_limit_rad
        drive = controls[..., 3]*torch.where(controls[..., 3] >= 0,
            calibration.forward_limit_mps, calibration.reverse_limit_mps)
        pan = controls[..., 0]*calibration.pan_limit_rad
        rows.append({"exploration_std": std, "drive_mean_mps": float(drive.mean()),
            "drive_fraction_above_004_mps": float((drive > .04).double().mean()),
            "forward_steering_fraction": float(((drive > .04)&(steering.abs() > .12)).double().mean()),
            "left_steering_fraction": float(((drive > .04)&(steering < -.12)).double().mean()),
            "right_steering_fraction": float(((drive > .04)&(steering > .12)).double().mean()),
            "large_pan_fraction": float((pan.abs() > .4).double().mean()),
            "steering_std_rad": float(steering.std()),
            "dominant_probability_mean": float(q.max(-1).values.mean())})
    useful = [r for r in rows if r["forward_steering_fraction"] >= .20
              and r["drive_fraction_above_004_mps"] >= .5 and r["large_pan_fraction"] <= .3]
    if useful:
        selected = min(useful, key=lambda r: r["exploration_std"])
    else:
        selected = max(rows, key=lambda r: r["forward_steering_fraction"]-r["large_pan_fraction"])
    return {"exploration_rows": rows, "selected_exploration_std": selected["exploration_std"],
        "seed": seed, "hypothetical_draws": draws*len(means), "new_observations": 0,
        "selection_rule": "Smallest std with >=20% moving-and-steering, >=50% forward coverage, <=30% large pan; otherwise best steering-minus-pan score.",
        "coverage_only_not_reward_selection": True}


def _vector(loss, params):
    gradients = (torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
                 if loss.requires_grad else (None,)*len(params))
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).float().flatten()
                      for p, g in zip(params.values(), gradients)])


def _bank_norms(vector, params, banks):
    pieces,offset = {},0
    for name,p in params.items():
        pieces[name] = vector[offset:offset+p.numel()]
        offset += p.numel()
    return {bank:float(torch.cat([pieces[name] for name in names]).norm()) for bank,names in banks.items()}


def optimizer_probe(model, config, initial):
    """Restore model, Adam and named RNGs after each scratch initialization probe."""
    rows, records = [], initial["eligible"]
    if len(records) < 2:
        raise ValueError("two distinct retained histories are required")
    learner = MemoryLearner(model, config.learning, config.seeds(0))
    learner.load_state_dict(initial["learner"])
    snapshot = learner.state_dict()
    gradients = {name:None if p.grad is None else p.grad.detach().clone() for name,p in learner.params.items()}
    new, replay = _batch(records[0], records), _batch(records[-1], records)
    with torch.no_grad():
        before_loss = float((learner.nll(learner.params, new).mean()+learner.nll(learner.params, replay).mean())/2)
        prediction = {key: value.detach().clone() for key, value in learner.predict(learner.params, replay).items()}
    try:
        for lr, steps in ((.0003, 1), (.001, 1), (.001, 2), (.003, 1), (.003, 2), (.001, 4)):
            learner.load_state_dict(snapshot)
            learner.config = replace(config.learning, learning_rate=lr, optimizer_steps=steps, fd_strength=0.)
            for group in learner.optimizer.param_groups:
                group["lr"] = lr
            started = time.monotonic()
            result = learner.fit(new=new, replay=replay,
                acquired_count=initial["acquired_count"], steps=steps)
            with torch.no_grad():
                after_loss = float((learner.nll(learner.params, new).mean()+learner.nll(learner.params, replay).mean())/2)
                kl = float(pm.symmetric_kl(prediction, learner.predict(learner.params, replay), learner.design(replay)).mean())
            seconds = time.monotonic()-started
            rows.append({"learning_rate": lr, "optimizer_steps": steps,
                "data_nll_before": before_loss, "data_nll_after": after_loss,
                "data_nll_decrease": before_loss-after_loss, "symmetric_model_kl": kl,
                "seconds": seconds, "update_seconds": result["update_seconds"],
                "finite": all(math.isfinite(v) for v in (after_loss, kl, result["total"]))})
        stable = [r for r in rows if r["finite"] and r["data_nll_decrease"] > 0 and r["symmetric_model_kl"] <= .25]
        selected = max(stable, key=lambda r: r["data_nll_decrease"]/r["optimizer_steps"]) if stable else rows[1]
        return {"optimizer_rows": rows, "selected_learning_rate": selected["learning_rate"],
            "selected_optimizer_steps": selected["optimizer_steps"],
            "selection_rule": "Largest positive retained-data NLL decrease per step among finite scratch fits with symmetric KL <=.25; fallback .001/one step.",
            "purpose": "common numerical calibration, not evidence of task improvement",
            "scratch_state_restored": True, "new_observations": 0}
    finally:
        learner.load_state_dict(snapshot)
        for name,p in learner.params.items():
            p.grad = gradients[name]


def fd_probe(model, config, initial):
    """Half/base/double epsilon with one shared, explicitly seeded probe."""
    learner = MemoryLearner(model, config.learning, config.seeds(0))
    learner.load_state_dict(initial["learner"])
    snapshot = learner.state_dict()
    gradients = {name:None if p.grad is None else p.grad.detach().clone() for name,p in learner.params.items()}
    records = initial["eligible"]
    batch = _batch(records[0], records)
    params = learner.params
    generator = torch.Generator(device=learner.device).manual_seed(193033)
    probes = pm.rademacher_probes(params, learner.banks["stm"], 1, generator=generator)
    try:
        data_norm = float(_vector(learner.nll(params, batch).mean(), params).norm())
        groups = []
        chosen = None
        for epsilons in ((.005, .01, .02), (.0025, .005, .01), (.01, .02, .04)):
            rows, previous = [], None
            for eps in epsilons:
                started = time.monotonic()
                loss = pm.stm_fd_penalty(lambda p: learner.predict(p, batch), params, learner.design(batch),
                    learner.banks["stm"], probes, torch.ones(1, device=learner.device), eps)
                gradient = _vector(loss, params)
                norm = float(gradient.norm())
                bank_norms = _bank_norms(gradient,params,learner.banks)
                cosine = None if previous is None else float(torch.nn.functional.cosine_similarity(gradient, previous, dim=0))
                previous = gradient
                rows.append({"epsilon": eps, "penalty": float(loss.detach()),
                    "gradient_norm": norm, "cosine_to_previous": cosine,
                    "ltm_gradient_norm":bank_norms["ltm"],"stm_gradient_norm":bank_norms["stm"],
                    "seconds": time.monotonic()-started})
            penalties, norms = [r["penalty"] for r in rows], [r["gradient_norm"] for r in rows]
            finite = all(math.isfinite(v) for v in penalties+norms)
            passed = (finite and data_norm > 0 and math.isfinite(data_norm) and min(penalties) > 0 and min(norms) > 0
                and max(penalties)/min(penalties) <= 1.25 and max(norms)/min(norms) <= 1.5
                and all(r["cosine_to_previous"] >= .95 for r in rows[1:])
                and all(r["ltm_gradient_norm"]>0 and r["stm_gradient_norm"]>0 for r in rows))
            for row in rows:
                for key,value in row.items():
                    if isinstance(value,float) and not math.isfinite(value):
                        row[key] = None
            groups.append({"rows": rows, "passed": passed})
            if passed:
                chosen = rows[1]
                break
        strengths = [] if chosen is None else [{"target_gradient_ratio": ratio,
            "fd_strength": ratio*data_norm/chosen["gradient_norm"]} for ratio in (.01, .1, 1.)]
        cost_rows = []
        if chosen is not None:
            for strength in (0., strengths[1]["fd_strength"]):
                times = []
                for _ in range(3):
                    learner.load_state_dict(snapshot)
                    learner.config = replace(config.learning, fd_strength=strength, fd_step=chosen["epsilon"])
                    for group in learner.optimizer.param_groups:
                        group["lr"] = config.learning.learning_rate
                    fit = learner.fit(new=batch, replay=batch,
                        acquired_count=initial["acquired_count"]+config.learning.replay_capacity,
                        steps=config.learning.optimizer_steps)
                    times.append(fit["update_seconds"])
                cost_rows.append({"fd_strength":strength,"optimizer_steps":config.learning.optimizer_steps,
                    "median_update_seconds":float(torch.tensor(times).median()),"repetitions":3})
        return {"fd_resolution_groups": groups, "fd_convergence_pass": chosen is not None,
            "selected_fd_step": None if chosen is None else chosen["epsilon"],
            "data_gradient_norm": data_norm if math.isfinite(data_norm) else None, "fd_strength_rows": strengths,
            "fit_cost_rows": cost_rows,
            "probe_seed": 193033, "probe_age_weight": 1., "probe_count": 1,
            "purpose": "Numerical resolution and initial gradient ratios, not equal downstream effects.",
            "scratch_state_restored": True, "new_observations": 0}
    finally:
        learner.load_state_dict(snapshot)
        for name,p in learner.params.items():
            p.grad = gradients[name]
