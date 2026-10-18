"""Resumable, paired initialization; never invoked from the results notebook."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.nn import functional as F

from .config import Config, Treatment, content_hash
from .curvature import reference_fisher
from .data import MNISTSource, materialize, plan_data
from .learner import FitSettings, commit_arrival, fit_step, initial_state
from .model import build_model, flat_logits, memory_indices


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def fit_settings(cfg: Config, environment: str, treatment: Treatment) -> FitSettings:
    return FitSettings(
        pi=cfg.mixture_pi if environment == "digit9_mixture" else cfg.rotation_pi,
        ridge_ratio=treatment.ridge_multiplier, replay_weight=treatment.replay_weight,
        lambda_f=treatment.strength if treatment.condition != "distillation" else 0.,
        lambda_d=treatment.strength if treatment.condition == "distillation" else 0.,
        num_probes=treatment.probes, fd_step=treatment.fd_step,
        direction_max_norm=cfg.direction_radius, max_iter=treatment.inner_steps,
        max_eval=treatment.max_evaluations, solver=cfg.optimizer,
        learning_rate=cfg.learning_rate, chunk_size=cfg.fisher_chunk)


def initialize(store, source: MNISTSource, cfg: Config, family: str, replica: int,
               kind: str, *, on_progress=lambda: None) -> tuple[str, dict]:
    unit = f"assets-{family}-{kind}-{replica:04d}"
    plan = plan_data(source, cfg, family, replica, kind)
    metadata = dict(kind="assets", schema_version=1, family=family, replica=replica,
                    plan_hash=content_hash(plan), seeds=plan["seeds"])
    if store.unit_path(unit).exists() and store.is_complete(unit):
        state = store.load_checkpoint(unit)
        if state["assets"]["plan_hash"] != metadata["plan_hash"]:
            raise ValueError("paired asset plan changed")
        return unit, state
    store.begin(unit, metadata)
    state = store.load_checkpoint(unit)
    if store.is_complete(unit):
        return unit, state
    dtype = getattr(torch, cfg.dtype)
    model, layout = build_model(plan["seeds"]["initialization_model"], device=cfg.device,
                                dtype=dtype, factor_seed=plan["seeds"]["factors"])
    ltm, stm = memory_indices(layout)
    if state is None:
        state = dict(stage="pretrain", epoch=0, offset=0, theta=layout.flatten_module(model, detach=True).cpu(),
                     optimizer=None, layout=layout.metadata(), active_seconds=0.)
        store.save_checkpoint(unit, state)
    layout.assert_metadata(state["layout"])
    check = store.check
    if state["stage"] == "pretrain":
        theta = state["theta"].to(cfg.device).requires_grad_()
        optimizer = torch.optim.Adam([theta], lr=cfg.init_learning_rate)
        if state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        images = source.images(plan["initialization_ids"])
        labels = source.train_labels[plan["initialization_ids"]]
        for epoch in range(state["epoch"], cfg.init_epochs):
            generator = torch.Generator().manual_seed(cfg.seed_for(family, replica, f"{kind}:epoch:{epoch}"))
            order = torch.randperm(len(images), generator=generator)
            offset = state["offset"] if epoch == state["epoch"] else 0
            for start in range(offset, len(images), cfg.init_batch):
                check()
                tick = time.perf_counter()
                ids = order[start:start + cfg.init_batch]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(flat_logits(model, layout, theta, images[ids].to(cfg.device, dtype)),
                                       labels[ids].to(cfg.device))
                loss.backward()
                theta.grad[stm.to(cfg.device)] = 0
                optimizer.step()
                if not torch.isfinite(theta).all():
                    raise ValueError("nonfinite initialization parameters")
                state["active_seconds"] += time.perf_counter() - tick
                state.update(epoch=epoch, offset=start + len(ids))
                # A bounded restart loss of at most sixteen pretraining batches.
                if ((start // cfg.init_batch + 1) % 16 == 0 or start + len(ids) == len(images)):
                    state.update(theta=cpu_tree(theta), optimizer=cpu_tree(optimizer.state_dict()))
                    store.save_checkpoint(unit, state)
                    on_progress()
            state.update(epoch=epoch + 1, offset=0)
        state.update(stage="reference", theta=cpu_tree(theta), optimizer=None,
                     reference_offset=0, fisher_sum=torch.zeros(len(ltm), len(ltm), dtype=torch.float64))
        store.save_checkpoint(unit, state)
    if state["stage"] == "reference":
        theta = state["theta"].to(cfg.device)
        images = source.images(plan["reference_ids"])
        for start in range(state["reference_offset"], len(images), cfg.fisher_chunk):
            check()
            tick = time.perf_counter()
            x = images[start:start + cfg.fisher_chunk].to(cfg.device, dtype)
            fisher = reference_fisher(model, layout, theta, x, indices=ltm, chunk_size=cfg.fisher_chunk)
            state["fisher_sum"] += len(x) * fisher.cpu()
            state["reference_offset"] = start + len(x)
            if "reference_prefix_sum" not in state and len(images) // 2 <= start + len(x) < len(images):
                state["reference_prefix_sum"] = state["fisher_sum"].clone()
                state["reference_prefix_count"] = start + len(x)
            state["active_seconds"] += time.perf_counter() - tick
            store.save_checkpoint(unit, state)
            on_progress()
        full = state["fisher_sum"] / len(images)
        health = dict(reference_samples=len(images), trace=float(full.trace()),
                      label_integration="exact ten-model-label probabilities", convergence_available=False)
        if "reference_prefix_sum" in state:
            prefix = state.pop("reference_prefix_sum") / state["reference_prefix_count"]
            denominator = float(full.norm())
            health.update(convergence_available=True, prefix_samples=state["reference_prefix_count"],
                          nested_relative_frobenius=float((prefix - full).norm()) / denominator if denominator > 0 else None,
                          prefix_trace=float(prefix.trace()),
                          interpretation="Nested prefix-versus-full sensitivity, not independent error to truth.")
        learner = initial_state(theta, ltm, full,
                                capacity=cfg.replay_capacity,
                                seed=cfg.seed_for(family, replica, f"{kind}:initial_compression"),
                                check_stop=check)
        state.update(stage="burn", learner=learner, assets=materialize(source, cfg, plan), initial_fisher_health=health)
        del state["fisher_sum"]
        store.save_checkpoint(unit, state)
    if state["stage"] == "burn" and kind == "mixture":
        # This common p=0 online arrival is replay-resident, not archived at initialization.
        assets, learner = state["assets"], state["learner"]
        width = cfg.mixture_batch
        if width > cfg.replay_capacity:
            raise ValueError("common p=0 burn batch must fit in the replay buffer")
        x = assets["streams"]["digit9_mixture"][:width].to(cfg.device, dtype)
        y = assets["targets"][:width].to(cfg.device)
        settings = fit_settings(cfg, "digit9_mixture", Treatment.default(cfg, "replay_ewc"))
        theta, diagnostics = fit_step(
            model, layout, learner["theta"].to(cfg.device), x, y, x[:0], y[:0],
            torch.empty(0, device=cfg.device), learner["fisher_L"], learner["anchor_L"],
            "replay_ewc", settings, cfg.seed_for(family, replica, "mixture:common_burn"), check)
        learner, _ = commit_arrival(learner, theta, list(range(width)), list(range(width)),
                                    pi=cfg.mixture_pi, arrival_size=width, check_stop=check)
        state.update(learner=learner, burn_diagnostics=diagnostics)
    state["stage"] = "ready"
    store.save_checkpoint(unit, state)
    store.write_json(unit, "data_plan.json", plan)
    store.write_json(unit, "layout.json", layout.metadata())
    store.write_json(unit, "initial_fisher_health.json", state["initial_fisher_health"])
    store.complete(unit)
    on_progress()
    return unit, state
