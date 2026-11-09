"""Checkpointed initialization and atomic four-grid scientific transactions."""

from __future__ import annotations

import copy
import math
import time

import torch

from mnist_experiment.diagonal_lfu.trajectory import cpu_tree, rng_state, restore_rng
from mnist_experiment.kalman_pi.observations import combine_moments
from mnist_experiment.kalman_pi.policies import Controller, PolicySpec, concentration
from . import data as data_module
from .curvature import fisher_diagonal, observed_scores, paired_change, fd_values
from .learner import fit_step, archive_update
from .model import build_model, flat_prediction, memory_indices, merge_parameters, gaussian_nll
from .diagnostics import eviction_diagnostics


def synchronize(cfg):
    if cfg.device == "cuda":
        torch.cuda.synchronize()


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    return value


class Reader:
    def __init__(self, raw, assets, cfg, check=lambda: None):
        self.raw, self.assets, self.cfg, self.check = raw, assets, cfg, check
        self.cache = {}

    def panel(self, role, indices=None, *, device=True):
        if role not in self.cache:
            panel = self.assets[role] if role in self.assets else self.assets["panels"][role]
            self.cache[role] = data_module.render_panel(self.raw, panel, check=self.check)
        x, y = self.cache[role]
        if indices is not None:
            x, y = x[indices], y[indices]
        if device:
            return x.to(self.cfg.device, getattr(torch, self.cfg.dtype)), y.to(self.cfg.device, getattr(torch, self.cfg.dtype))
        return x, y

    def stream(self, indices):
        panel = self.assets["stream"]
        x, y = data_module.render_panel(self.raw, panel, torch.as_tensor(indices, dtype=torch.long), check=self.check)
        return x.to(self.cfg.device, getattr(torch, self.cfg.dtype)), y.to(self.cfg.device, getattr(torch, self.cfg.dtype))

    def current(self, step):
        panel = data_module.current_panel(self.assets, step, self.cfg)
        x, y = data_module.render_panel(self.raw, panel, check=self.check)
        return x.to(self.cfg.device, getattr(torch, self.cfg.dtype)), y.to(self.cfg.device, getattr(torch, self.cfg.dtype))


def initial_fit(store, budget, cfg, uid, metadata, reader, *, pulse=lambda **kw: None):
    if store.begin(uid, metadata) is None:
        return store.load_checkpoint(uid)
    model, layout = build_model(metadata["model_seed"], device=cfg.device,
                                dtype=getattr(torch, cfg.dtype), factor_seed=metadata["factor_seed"])
    ltm, stm = memory_indices(layout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)
    saved = store.load_checkpoint(uid)
    state = saved or dict(epoch=0, offset=0, phase="fit", history=[], best_loss=None,
        best_model=None, reference_offset=0, reference_sum=torch.zeros(len(ltm), dtype=torch.float64),
        score_count=0, score_mean=torch.zeros(len(ltm), dtype=torch.float64),
        score_m2=torch.zeros(len(ltm), dtype=torch.float64), wall_seconds=0.,
        epoch_loss_sum=0., epoch_examples=0)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    started, base = time.perf_counter(), state["wall_seconds"]

    def checkpoint():
        state.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     rng=rng_state(), wall_seconds=base+time.perf_counter()-started)
        store.save_checkpoint(uid, state)

    while state["phase"] == "fit" and state["epoch"] < cfg.init_epochs:
        order = torch.randperm(cfg.init_size, generator=torch.Generator().manual_seed(
            cfg.seed_for(metadata["cohort"], metadata["replica"], f"initialization_order:epoch:{state['epoch']}")))
        for offset in range(state["offset"], cfg.init_size, cfg.init_batch):
            budget.check()
            x, y = reader.panel("init", order[offset:offset+cfg.init_batch])
            optimizer.zero_grad(set_to_none=True)
            loss = gaussian_nll(model(x), y).mean()
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                             for p in model.parameters()):
                raise FloatingPointError("invalid Plan 18 pretraining objective or gradient")
            optimizer.step()
            state["offset"] = offset + len(y)
            state["epoch_loss_sum"] += float(loss.detach()) * len(y)
            state["epoch_examples"] += len(y)
            checkpoint()
            pulse(unit=uid, batch=state["offset"], total=cfg.init_size, epoch=state["epoch"]+1)
        epoch = state["epoch"] + 1
        row = dict(epoch=epoch, training_nll=state["epoch_loss_sum"]/state["epoch_examples"],
            order_seed=cfg.seed_for(metadata["cohort"], metadata["replica"], f"initialization_order:epoch:{epoch-1}"))
        if epoch in (15, 30, 45, 60, cfg.init_epochs):
            total, count = 0., 0
            with torch.no_grad():
                for offset in range(0, cfg.validation_size, cfg.evaluation_chunk):
                    budget.check()
                    x, y = reader.panel("validation", slice(offset, offset+cfg.evaluation_chunk))
                    total += float(gaussian_nll(model(x), y).double().sum())
                    count += len(y)
            row["validation_nll"] = total / count
            if state["best_loss"] is None or row["validation_nll"] < state["best_loss"]:
                state["best_loss"], state["best_model"] = row["validation_nll"], cpu_tree(model.state_dict())
        state["history"].append(row)
        state.update(epoch=epoch, offset=0, epoch_loss_sum=0., epoch_examples=0)
        checkpoint()
    if state["phase"] == "fit":
        model.load_state_dict(state["best_model"])
        state.update(theta=layout.flatten_module(model, detach=True).cpu(), phase="fisher")
        checkpoint()
    theta = state["theta"].to(cfg.device)
    merged, merged_layout, merged_theta = merge_parameters(model, layout, theta)
    for offset in range(state["reference_offset"], cfg.reference_size, cfg.fisher_chunk):
        budget.check()
        x, y = reader.panel("reference", slice(offset, offset+cfg.fisher_chunk))
        diagonal = fisher_diagonal(model, layout, theta, x, indices=ltm,
            chunk_size=cfg.fisher_chunk, output_chunk_size=cfg.output_chunk, check=budget.check)
        scores = observed_scores(merged, merged_layout, merged_theta, x, y,
                                 chunk_size=cfg.fisher_chunk, check=budget.check).cpu()
        state["reference_sum"] += diagonal.cpu() * len(y)
        state["score_count"], state["score_mean"], state["score_m2"] = combine_moments(
            state["score_count"], state["score_mean"], state["score_m2"], scores)
        state["reference_offset"] = offset + len(y)
        if state["reference_offset"] == cfg.reference_size // 2:
            state["nested_diagonal"] = state["reference_sum"] / state["reference_offset"]
        checkpoint()
        pulse(unit=uid, batch=state["reference_offset"], total=cfg.reference_size, epoch="Fisher")
    state.update(diagonal=state["reference_sum"] / cfg.reference_size,
        variance=state["score_m2"] / (state["score_count"]-1),
        merged_anchor=merged_theta.detach().cpu(), parameter_layout=layout.metadata(), phase="completed")
    checkpoint()
    store.write_json(uid, "summary.json", dict(history=state["history"],
        best_validation_nll=state["best_loss"], wall_seconds=state["wall_seconds"],
        initial_fisher_trace=float(state["diagonal"].sum()), parameter_count=layout.total_numel,
        initialization_grids=cfg.init_size, reference_grids=cfg.reference_size,
        nested_diagonal_relative_error=float((state["nested_diagonal"]-state["diagonal"]).norm()
            / state["diagonal"].norm().clamp_min(1e-30)) if "nested_diagonal" in state else None))
    store.complete(uid)
    return state


def evaluate_panel(model, layout, theta, x, y, cfg, check):
    squared, hits, nll, count = 0., 0., 0., 0
    with torch.no_grad():
        for offset in range(0, len(y), cfg.evaluation_chunk):
            check()
            prediction = flat_prediction(model, layout, theta, x[offset:offset+cfg.evaluation_chunk])
            target = y[offset:offset+cfg.evaluation_chunk]
            errors = (prediction.double()-target.double()).reshape(-1, 27, 2).square().sum(-1)
            squared += float(errors.sum())
            hits += float((errors <= 1).sum())
            nll += float(gaussian_nll(prediction, target).double().sum())
            count += len(target)
    return dict(mse=squared/(27*count), accuracy=hits/(27*count), nll=nll/count)


def run_trajectory(store, budget, cfg, uid, metadata, initial, reader,
                   *, pulse=lambda **kw: None, audit=False, shadows=False):
    if store.begin(uid, metadata) is None:
        return store.load_checkpoint(uid)
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    recipe = metadata["recipe_object"] if "recipe_object" in metadata else None
    if recipe is None:
        from .config import Recipe
        recipe = Recipe(**metadata["recipe"])
    model, layout = build_model(metadata["model_seed"], device=cfg.device,
                                dtype=getattr(torch, cfg.dtype), factor_seed=metadata["factor_seed"])
    ltm, stm = (x.to(cfg.device) for x in memory_indices(layout))
    observer_model, observer_layout, observer_anchor = merge_parameters(
        model, layout, initial["theta"].to(cfg.device))
    saved = store.load_checkpoint(uid)
    state = saved or dict(step=0, theta=initial["theta"], diagonal=initial["diagonal"],
        anchor=initial["theta"][ltm.cpu()], curvature_reference=initial["theta"],
        replay=[], retired_ids=[], q_ref=1/cfg.init_size, q_archive=1/cfg.init_size,
        novel_count=0, metrics=[], diagnostics=[], audit_states=[], controller=None,
        shadow_states={}, shadow_logs={str(v): [] for v in cfg.process_scales},
        learning_seconds=0., online_wall_seconds=0., evaluation_seconds=0.,
        checkpoint_seconds=0., shadow_seconds=0., rng=rng_state())
    state.setdefault("evicted_probes", [])
    state.setdefault("diagnostic_seconds", 0.)
    restore_rng(state["rng"])
    theta = state["theta"].to(cfg.device)
    diagonal = state["diagonal"].to(cfg.device)
    anchor = state["anchor"].to(cfg.device)
    initial_scale = float(initial["diagonal"].mean())
    if initial_scale <= 0:
        raise FloatingPointError("initial likelihood Fisher has no positive scale")
    controller = Controller(PolicySpec("kalman", recipe.process_scale), cfg, cfg.fixed_pi,
                            device_tree(state["controller"], cfg.device) if state["controller"] else None)
    if state["controller"] is None:
        controller.initialize(initial["merged_anchor"].to(cfg.device),
                              initial["diagonal"].to(cfg.device), initial["variance"].to(cfg.device))
    shadow_filters = {}
    if shadows:
        for eta in cfg.process_scales:
            shadow = Controller(PolicySpec("kalman", eta), cfg, cfg.fixed_pi,
                device_tree(state["shadow_states"].get(str(eta)), cfg.device))
            if str(eta) not in state["shadow_states"]:
                shadow.initialize(initial["merged_anchor"].to(cfg.device),
                                  initial["diagonal"].to(cfg.device), initial["variance"].to(cfg.device))
            shadow_filters[str(eta)] = shadow
    call_started, wall_base = time.perf_counter(), state["online_wall_seconds"]

    def metric(at_theta, step, novel, kind, learning, current):
        begin = time.perf_counter()
        row = dict(accepted_step=step, incoming_grids=step*cfg.samples_per_step,
                   novel_count=novel, kind=kind, learning_seconds=learning)
        for name in ("old", "novel"):
            x, y = reader.panel(name)
            result = evaluate_panel(model, layout, at_theta, x, y, cfg, budget.check)
            row.update({f"{name}_{key}": value for key, value in result.items()})
        if current:
            x, y = reader.current(step)
            result = evaluate_panel(model, layout, at_theta, x, y, cfg, budget.check)
            row.update({f"current_{key}": value for key, value in result.items()})
            frozen = evaluate_panel(model, layout, initial["theta"].to(cfg.device), x, y, cfg, budget.check)
            row["frozen_current_mse"] = frozen["mse"]
        else:
            row.update(current_mse=None, current_accuracy=None)
        synchronize(cfg)
        return row, time.perf_counter()-begin

    if not state["metrics"]:
        row, seconds = metric(theta, 0, 0, "initial", 0., True)
        state["metrics"].append(row)
        state["evaluation_seconds"] += seconds
        store.save_checkpoint(uid, cpu_tree(state))
    stream = reader.assets["stream"]
    novel_flags = torch.as_tensor(stream["novel"], dtype=torch.bool).flatten()
    audit_batches = (32, 312, 464, 616)
    for step in range(state["step"], cfg.num_p_steps):
        budget.check()
        if recipe.adaptive:
            for name in ("mean", "covariance", "raw_variance", "mapping", "k0"):
                if not torch.isfinite(controller.state[name]).all():
                    raise FloatingPointError(f"nonfinite observer {name}; refusing covariance fallback")
        synchronize(cfg)
        learning_start = time.perf_counter()
        action = (controller.choose(state["q_ref"], diagonal, cfg.samples_per_step, step)
                  if recipe.adaptive else dict(raw_pi=cfg.fixed_pi, applied_pi=cfg.fixed_pi,
                    cold_start=False, motion_energy=0., uncertainty_energy=0., noise_scale=None,
                    rho=None, boundary_low=False, boundary_high=False, fallback=""))
        pi = action["applied_pi"]
        positions = list(range(step*cfg.samples_per_step, (step+1)*cfg.samples_per_step))
        new_x, new_y = reader.stream(positions)
        new_novel = int(novel_flags[positions].sum())
        extra_evaluation = 0.
        pending_rows = []
        if state["novel_count"] == 0 and new_novel:
            row, seconds = metric(theta, step, 0, "zero_novel", state["learning_seconds"], False)
            pending_rows.append(row)
            extra_evaluation += seconds
        residents = state["replay"]
        replay_rng = torch.Generator().manual_seed(cfg.seed_for(metadata["cohort"], metadata["replica"], f"replay:{step}"))
        selected = (torch.randperm(len(residents), generator=replay_rng)[:cfg.replay_count].tolist()
                    if residents else [])
        replay_ids = [residents[i] for i in selected]
        if replay_ids:
            replay_x, replay_y = reader.stream(replay_ids)
        else:
            replay_x, replay_y = new_x[:0], new_y[:0]
        cumulative = (step+1)*cfg.samples_per_step
        ages = torch.tensor([min(1., max(0., (cumulative-(i+1))/cfg.capacity))**2 for i in replay_ids],
                            dtype=torch.float64, device=cfg.device)
        probe_rng = torch.Generator().manual_seed(cfg.seed_for(metadata["cohort"], metadata["replica"], f"probes:{step}"))
        probes = (2*torch.randint(2, (cfg.probes, len(stm)), generator=probe_rng)-1).to(cfg.device, theta.dtype)
        accepted, solve = fit_step(model, layout, theta, diagonal, anchor,
            new_x, new_y, replay_x, replay_y, ages, probes, pi=pi, recipe=recipe,
            cfg=cfg, initial_fisher_scale=initial_scale, check=budget.check)
        combined = residents + positions
        evicted = combined[:max(0, len(combined)-cfg.capacity)]
        next_replay = combined[len(evicted):]
        if any(i in state["retired_ids"] for i in evicted):
            raise ValueError("online grid was consolidated twice")
        delta = torch.zeros_like(diagonal)
        fresh, change_diagnostics = None, {}
        if evicted:
            qx, _ = reader.stream(evicted)
            if recipe.transport:
                delta, fresh, change_diagnostics = paired_change(model, layout, theta, accepted, qx,
                    indices=ltm, chunk_size=cfg.fisher_chunk, output_chunk_size=cfg.output_chunk,
                    check=budget.check)
            else:
                fresh = fisher_diagonal(model, layout, accepted, qx, indices=ltm,
                    chunk_size=cfg.fisher_chunk, output_chunk_size=cfg.output_chunk, check=budget.check)
        elif recipe.transport:
            delta, _, change_diagnostics = paired_change(model, layout, theta, accepted, new_x,
                indices=ltm, chunk_size=cfg.fisher_chunk, output_chunk_size=cfg.output_chunk,
                check=budget.check)
        next_diagonal, q_archive, archive = archive_update(diagonal, delta, fresh, pi=pi,
            evicted_count=len(evicted), nominal_arrival=cfg.nominal_arrival,
            q_archive=state["q_archive"], transport=recipe.transport)
        next_anchor = accepted[ltm].detach().clone() if evicted else anchor
        observer_diagnostics, shadow_seconds = {}, 0.
        if recipe.adaptive or shadows:
            score_started = time.perf_counter()
            scores = observed_scores(observer_model, observer_layout, observer_anchor,
                new_x, new_y, chunk_size=cfg.fisher_chunk, check=budget.check)
            if recipe.adaptive:
                observer_diagnostics = controller.assimilate(scores, step)
            if shadows:
                for eta, shadow in shadow_filters.items():
                    result = shadow.assimilate(scores, step)
                    result.update(step=step+1, segment=segment_name(reader.assets, step))
                    state["shadow_logs"][eta].append(result)
                synchronize(cfg)
                shadow_seconds = time.perf_counter()-score_started
        synchronize(cfg)
        learning_delta = time.perf_counter()-learning_start-extra_evaluation-shadow_seconds
        learning = state["learning_seconds"] + learning_delta
        novel = state["novel_count"] + new_novel
        regular = ((step+1) % cfg.eval_every == 0 or step+1 == cfg.num_p_steps
                   or step+1 in (cfg.hold_steps, cfg.hold_steps+cfg.ramp_steps,
                                 cfg.hold_steps+2*cfg.ramp_steps))
        needs_novel = bool(new_novel and state["novel_count"] < cfg.analysis_novel_count)
        if regular or needs_novel:
            row, seconds = metric(accepted, step+1, novel, "regular" if regular else "novel", learning, regular)
            pending_rows.append(row)
            extra_evaluation += seconds
        probe_started = time.perf_counter()
        evicted_probes, probe_metrics = eviction_diagnostics(model, layout, accepted, reader,
            state["evicted_probes"], evicted, novel_flags, step+1,
            cfg=cfg, replay_x=replay_x, probes=probes, check=budget.check)
        synchronize(cfg)
        probe_seconds = time.perf_counter()-probe_started
        if audit and step+1 in audit_batches:
            state["audit_states"].append(cpu_tree(dict(batch=step+1, theta=theta, accepted=accepted,
                diagonal=diagonal, anchor=anchor, replay_ids=replay_ids, positions=positions,
                ages=ages, probes=probes, pi=pi, solve=solve)))
        state.update(step=step+1, theta=accepted.detach().cpu(), diagonal=next_diagonal.detach().cpu(),
            anchor=next_anchor.detach().cpu(), curvature_reference=accepted.detach().cpu(),
            replay=next_replay, retired_ids=state["retired_ids"]+evicted,
            q_ref=concentration(state["q_ref"], pi, cfg.samples_per_step), q_archive=q_archive,
            novel_count=novel, learning_seconds=learning,
            online_wall_seconds=wall_base+time.perf_counter()-call_started,
            evaluation_seconds=state["evaluation_seconds"]+extra_evaluation,
            shadow_seconds=state["shadow_seconds"]+shadow_seconds,
            diagnostic_seconds=state["diagnostic_seconds"]+probe_seconds, evicted_probes=evicted_probes,
            metrics=state["metrics"]+pending_rows, rng=rng_state(),
            controller=cpu_tree(controller.state) if recipe.adaptive else None,
            shadow_states={eta: cpu_tree(shadow.state) for eta, shadow in shadow_filters.items()})
        diagnostic = dict(batch=step+1, incoming_grids=cumulative, novel_count=novel,
            learning_seconds=learning, q_ref=state["q_ref"], q_archive=q_archive,
            replay_seed=replay_rng.initial_seed(), probe_seed=probe_rng.initial_seed(),
            paired_difference_error=change_diagnostics.get("direct_difference_error"),
            ltm_move=float((accepted[ltm]-theta[ltm]).double().norm()),
            stm_move=float((accepted[stm]-theta[stm]).double().norm()),
            **action, **solve, **archive, **observer_diagnostics, **probe_metrics)
        state["diagnostics"] = state["diagnostics"] + [diagnostic]
        write_started = time.perf_counter()
        store.save_checkpoint(uid, cpu_tree(state))
        state["checkpoint_seconds"] += time.perf_counter()-write_started
        theta, diagonal, anchor = accepted, next_diagonal, next_anchor
        pulse(unit=uid, batch=step+1, total=cfg.num_p_steps, novel_count=novel,
              metrics=state["metrics"], recipe=metadata["recipe"])
    from .analysis import summarize_path
    summary = summarize_path(state["metrics"], cfg)
    summary.update(learning_seconds=state["learning_seconds"], online_wall_seconds=state["online_wall_seconds"],
        evaluation_seconds=state["evaluation_seconds"], checkpoint_seconds=state["checkpoint_seconds"],
        shadow_seconds=state["shadow_seconds"], actual_novel_grids=state["novel_count"],
        diagnostic_seconds=state["diagnostic_seconds"],
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated() if cfg.device == "cuda" else 0,
        fallback_fraction=sum(bool(x["fallback"]) for x in state["diagnostics"])/cfg.num_p_steps,
        nonfinite_trial_fraction=sum(x["nonfinite_trials"]>0 for x in state["diagnostics"])/cfg.num_p_steps,
        no_move_fraction=sum(not x["moved"] for x in state["diagnostics"])/cfg.num_p_steps,
        median_closure_count=float(torch.tensor([x["closure_count"] for x in state["diagnostics"]]).double().median()),
        valid=True,
        applied_pi_mean=sum(x["applied_pi"] for x in state["diagnostics"])/cfg.num_p_steps)
    store.write_json(uid, "metrics.json", state["metrics"])
    store.write_json(uid, "diagnostics.json", state["diagnostics"])
    store.write_json(uid, "summary.json", summary)
    probe_exports=[{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in r.items()}
                   for r in state["evicted_probes"]]
    store.write_json(uid, "evicted_probes.json", probe_exports)
    store.save_checkpoint(uid, cpu_tree(state))
    store.complete(uid)
    return state


def segment_name(assets, step):
    segments = assets["stream"].get("segments", [])
    for segment in segments:
        if segment["start"] <= step < segment["end"]:
            return segment["name"]
    return "unknown"
