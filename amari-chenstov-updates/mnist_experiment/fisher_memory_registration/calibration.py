"""Development-only capability checks and fixed-snapshot FD calibration."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from mnist_experiment.fisher_memory.initialize import cpu_tree
from .curvature import gaussian_nll
from .model import flat_prediction
from .numerical import calibrate_fd
from mnist_experiment.fisher_memory_regression.runtime import read_json, synchronize

from .runner import panel_metrics


def diagnostics(runner, context, replica=0):
    """Fit a separate, relaxed-memory reference; never seed an online learner."""
    cfg, store = runner.cfg, runner.store
    family = "development--" + context.key
    asset_id, asset, data, model, layout = runner.assets(family, replica, "development")
    unit = f"diagnostic-{context.key}-{replica:04d}-b{runner.blur_max:g}-c{runner.channels[0]}x{runner.channels[1]}-e{runner.init_epochs}"
    if store.is_complete(unit):
        return read_json(store.unit_path(unit) / "result.json")
    store.begin(unit, dict(kind="registration_learnability", schema_version=1,
        context=context.mapping(), replica=replica, asset_id=asset_id,
        blur_max=runner.blur_max,
        asset_seal=read_json(store.unit_path(asset_id) / "COMPLETED")["seal_hash"]))
    state = store.load_checkpoint(unit)
    if state is None:
        state = dict(theta=asset["theta"].clone(), optimizer=None, epoch=0, offset=0,
                     evaluations=[], learning_seconds=0.)
        store.save_checkpoint(unit, state)
    x = torch.cat((data["diagnostic_base_x"], data["diagnostic_novel_x"]))
    y = torch.cat((data["diagnostic_base_y"], data["diagnostic_novel_y"]))
    from .data import blur_pairs
    count = len(data["validation_y"])
    generator = torch.Generator().manual_seed(cfg.seed_for(family, replica, "diagnostic_validation_strengths"))
    strengths = .5 + .5 * torch.rand(count, generator=generator, dtype=torch.float64)
    val_x = data["validation_x"]
    validation = dict(base=dict(x=val_x, y=data["validation_y"],
        digits=torch.as_tensor(runner.source.train_labels)[data["plan"]["validation_ids"]],
        weights=torch.full((count,), 1/count, dtype=torch.float64)))
    validation["novel"] = {**validation["base"],
        "x": blur_pairs(val_x, strengths, runner.blur_max, check_stop=runner.check)}
    initial = {d: panel_metrics(model, layout, asset["theta"].to(cfg.device), p,
                               cfg, runner.check) for d, p in validation.items()}
    theta = state["theta"].to(cfg.device).requires_grad_()
    optimizer = torch.optim.Adam([theta], lr=.001)
    if state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    endpoints = tuple(sorted(set((cfg.diagnostic_epochs, cfg.max_diagnostic_epochs))))
    for epoch in range(state["epoch"], max(endpoints)):
        generator = torch.Generator().manual_seed(cfg.seed_for(
            "development", replica, f"diagnostic_order:{epoch}"))
        order = torch.randperm(len(y), generator=generator)
        offset = state["offset"] if epoch == state["epoch"] else 0
        for start in range(offset, len(y), cfg.diagnostic_batch):
            runner.check()
            tick = time.perf_counter()
            ids = order[start:start + cfg.diagnostic_batch]
            optimizer.zero_grad(set_to_none=True)
            loss = gaussian_nll(flat_prediction(model, layout, theta, x[ids].to(theta)),
                                y[ids].to(theta)).mean()
            loss.backward()
            optimizer.step()
            if not torch.isfinite(theta).all():
                raise ValueError("nonfinite learnability diagnostic")
            synchronize(cfg.device)
            state["learning_seconds"] += time.perf_counter() - tick
            state.update(epoch=epoch, offset=start + len(ids), theta=cpu_tree(theta),
                         optimizer=cpu_tree(optimizer.state_dict()))
            store.save_checkpoint(unit, state)
        state.update(epoch=epoch + 1, offset=0)
        if epoch + 1 in endpoints:
            measured = {d: panel_metrics(model, layout, theta, p, cfg, runner.check)
                        for d, p in validation.items()}
            recovered = (measured["novel"]["mse"] <= .9 * initial["novel"]["mse"]
                         and measured["base"]["rmse"] - initial["base"]["rmse"] <= cfg.retention_margin)
            item = dict(epochs=epoch + 1, metrics=measured, recovered=recovered)
            order_key = lambda row: (not row["recovered"], row["metrics"]["novel"]["mse"], row["epochs"])
            if not state["evaluations"] or order_key(item) < min(map(order_key, state["evaluations"])):
                state["chosen_theta"] = cpu_tree(theta)
            state["evaluations"].append(item)
            store.save_checkpoint(unit, state)
            runner.on_progress(current=unit)
            if recovered:
                break
    chosen = min(state["evaluations"], key=lambda row: (
        not row["recovered"], row["metrics"]["novel"]["mse"], row["epochs"]))
    heldout_initial = {d: panel_metrics(model, layout, asset["theta"].to(cfg.device), p,
                        cfg, runner.check) for d, p in data["panels"].items()}
    heldout_chosen = {d: panel_metrics(model, layout, state["chosen_theta"].to(cfg.device), p,
                       cfg, runner.check) for d, p in data["panels"].items()}
    result = dict(context=context.mapping(), replica=replica, initial=initial,
        blur_max=runner.blur_max, heldout_initial=heldout_initial, heldout_chosen=heldout_chosen,
        selection_population="independent development validation, never the fixed test panels",
        chosen=chosen, evaluations=state["evaluations"],
        frozen_degradation_ratio=initial["novel"]["rmse"] / initial["base"]["rmse"],
        meaningful_degradation=initial["novel"]["rmse"] >= 1.1 * initial["base"]["rmse"],
        recovered=chosen["recovered"], diagnostic_training_images=len(y),
        diagnostic_novel_images=len(data["diagnostic_novel_y"]),
        learning_seconds=state["learning_seconds"],
        limitation="Extra-label relaxed-memory diagnostic; weights never enter online learning.")
    result["preview"] = dict(
        reference=(data["diagnostic_novel_x"][0, 0] * 255).round().byte().tolist(),
        query=(data["diagnostic_novel_x"][0, 1] * 255).round().byte().tolist(),
        nominal=(runner._nominal["diagnostic_novel_x"][:3, 0] * 255).round().byte().tolist(),
        transformed=(data["diagnostic_novel_x"][:3, 1] * 255).round().byte().tolist(),
        targets=data["diagnostic_novel_y"][:3].tolist(),
        strengths=data["plan"]["diagnostic_novel_strengths"][:3],
        interpretation="Display-only quantization; all learning uses the original floating images.")
    if model.output_dim == 6:
        from .data import ANCHORS
        anchors=torch.tensor(ANCHORS,dtype=torch.float64)
        with torch.no_grad():
            initial_theta=asset["theta"].to(cfg.device)
            predicted=flat_prediction(model,layout,initial_theta,
                data["diagnostic_novel_x"][:1].to(initial_theta))
        result["preview"].update(true_landmarks=(anchors+data["diagnostic_novel_y"][0].double().reshape(3,2)).tolist(),
            predicted_landmarks=(anchors+predicted[0].cpu().double().reshape(3,2)).tolist(),
            model="frozen nominal initialization; illustration only")
    store.write_json(unit, "result.json", result)
    store.complete(unit)
    return result


def coefficient(runner, context, reference_record):
    """Audit FD in native/double precision and estimate the declared norm scale."""
    cfg, store = runner.cfg, runner.store
    unit = f"numerics-{context.key}-r{runner.archive_rank:02d}"
    if store.is_complete(unit):
        return read_json(store.unit_path(unit) / "result.json")
    state = store.load_checkpoint(reference_record["unit_id"])
    snapshots = [s for s in state["snapshots"] if s["n"] in (25, 75)]
    if cfg.smoke:
        snapshots = [s for s in state["snapshots"] if len(s["replay_y"])]
    if not snapshots:
        raise ValueError("FD calibration needs recorded reference replay snapshots")
    _, _, _, model, layout = runner.assets("development--" + context.key, 0, "development")
    audit = calibrate_fd(cfg, snapshots, model, layout, runner.check)
    ratios = [row for row in audit["gradient_ratios"] if row["status"] == "valid"]
    a = float(np.median([row["data_gradient_norm"] for row in ratios]))
    b = float(np.median([row["fd_gradient_norm"] for row in ratios]))
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("unresolved FD gradient scale; no denominator clipping")
    result = dict(context=context.mapping(), numerical=audit,
        lambda0=.1 * a / b, data_gradient_norm_median=a, fd_gradient_norm_median=b,
        selected_step=audit["selected_step"], reference_unit=reference_record["unit_id"],
        reference_seal=read_json(store.unit_path(reference_record["unit_id"]) / "COMPLETED")["seal_hash"])
    store.begin(unit, dict(kind="registration_numerical_calibration", schema_version=1,
                           context=context.mapping()))
    store.write_json(unit, "result.json", result)
    store.complete(unit)
    return result
