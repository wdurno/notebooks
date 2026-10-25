"""Paired regression assets and atomic, exactly event-aligned trajectories."""

from __future__ import annotations

import copy
import dataclasses
import math
import resource
import time

import torch

from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory.initialize import cpu_tree
from mnist_experiment.fisher_memory.learner import _bounded_solve

from .config import Config, Recipe, content_hash
from .runtime import read_json, synchronize


def trajectory_id(family: str, replica: int, recipe: Recipe, target: int) -> str:
    return f"trajectory-{family}-{replica:04d}-{recipe.key}-n{target:04d}"


def settings(cfg, recipe):
    from .learner import FitSettings
    return FitSettings(pi=cfg.pi, ridge_ratio=recipe.ridge, lambda_f=recipe.strength,
                       replay_weight=.5, num_probes=recipe.probes, fd_step=recipe.fd_step,
                       max_iter=recipe.max_iter, max_eval=recipe.max_eval,
                       chunk_size=cfg.fisher_chunk)


def predict(model, layout, theta, x, chunk, check=lambda: None):
    from .model import flat_prediction
    outputs = []
    with torch.no_grad():
        for part in x.split(chunk):
            check()
            outputs.append(flat_prediction(model, layout, theta, part.to(theta)).detach().cpu())
    return torch.cat(outputs)


def panel_metrics(model, layout, theta, panel, cfg, check=lambda: None):
    means = predict(model, layout, theta, panel["x"], cfg.evaluation_chunk, check).double()
    errors = (cfg.angle_scale * means - panel["y"].double()).square()
    weights = panel["weights"].double()
    if errors.shape != weights.shape or not torch.isfinite(errors).all():
        raise ValueError("invalid regression predictions or panel weights")
    mse = float((weights * errors).sum())
    per_digit = {}
    for digit in range(10):
        take = panel["digits"] == digit
        if take.any():
            per_digit[str(digit)] = float(errors[take].mean())
    return dict(mse=mse, rmse=math.sqrt(mse),
                nll=mse / (2 * cfg.angle_scale ** 2) + math.log(cfg.angle_scale * math.sqrt(2 * math.pi)),
                per_digit_mse=per_digit)


class Runner:
    def __init__(self, cfg: Config, store, source, budget, *, on_progress=lambda **kwargs: None,
                 after_commit=None, stop_at=None):
        self.cfg, self.store, self.source, self.budget = cfg, store, source, budget
        self.on_progress, self.after_commit = on_progress, after_commit
        self.stop_at = stop_at
        self._cache_key, self._cache = None, None

    def check(self):
        self.budget.check()

    def _model(self, family, replica):
        from .model import build_model
        return build_model(self.cfg.seed_for(family, replica, "initialization_model"),
                           factor_seed=self.cfg.seed_for(family, replica, "factors"),
                           device=self.cfg.device, dtype=getattr(torch, self.cfg.dtype))

    def assets(self, family, replica, partition):
        from .curvature import reference_fisher, gaussian_nll
        from .data import plan_data, materialize
        from .learner import initial_state
        from .model import flat_prediction, memory_indices

        cfg, store = self.cfg, self.store
        key = (family, replica, partition)
        unit = f"assets-{family}-{replica:04d}"
        plan = plan_data(self.source, cfg, family, replica, partition)
        metadata = dict(kind="regression_assets", schema_version=1, family=family,
                        replica=replica, partition=partition, plan_hash=content_hash(plan))
        model, layout = self._model(family, replica)
        ltm, stm = memory_indices(layout)
        if self._cache_key != key:
            self._cache = materialize(self.source, cfg, plan, check_stop=self.check)
            self._cache_key = key
        data = self._cache
        if store.is_complete(unit):
            state = store.load_checkpoint(unit)
            if state["plan_hash"] != metadata["plan_hash"]:
                raise ValueError("shared asset plan changed")
            layout.assert_metadata(state["layout"])
            return unit, state, data, model, layout
        store.begin(unit, metadata)
        state = store.load_checkpoint(unit)
        if state is None:
            state = dict(stage="pretrain", epoch=0, offset=0,
                         theta=layout.flatten_module(model, detach=True).cpu(), optimizer=None,
                         layout=layout.metadata(), plan_hash=metadata["plan_hash"],
                         pretrain_seconds=0., reference_seconds=0.)
            store.write_json(unit, "plan.json", plan)
            store.save_checkpoint(unit, state)
        layout.assert_metadata(state["layout"])
        theta = state["theta"].to(cfg.device).requires_grad_()
        if state["stage"] == "pretrain":
            optimizer = torch.optim.Adam([theta], lr=cfg.init_learning_rate)
            if state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            for epoch in range(state["epoch"], cfg.init_epochs):
                generator = torch.Generator().manual_seed(plan["seeds"]["epochs"][epoch])
                order = torch.randperm(cfg.init_size, generator=generator)
                offset = state["offset"] if epoch == state["epoch"] else 0
                for start in range(offset, cfg.init_size, cfg.init_batch):
                    self.check()
                    tick = time.perf_counter()
                    ids = order[start:start + cfg.init_batch]
                    optimizer.zero_grad(set_to_none=True)
                    normalized = flat_prediction(model, layout, theta, data["pretrain_x"][ids].to(theta))
                    loss = gaussian_nll(normalized, data["pretrain_y"][ids].to(theta)).mean()
                    loss.backward()
                    theta.grad[stm.to(theta.device)] = 0
                    optimizer.step()
                    if not torch.isfinite(theta).all():
                        raise ValueError("nonfinite regression initialization")
                    synchronize(cfg.device)
                    state["pretrain_seconds"] += time.perf_counter() - tick
                    state.update(epoch=epoch, offset=start + len(ids))
                    if ((start // cfg.init_batch + 1) % 16 == 0
                            or start + len(ids) == cfg.init_size):
                        state.update(theta=cpu_tree(theta), optimizer=cpu_tree(optimizer.state_dict()))
                        store.save_checkpoint(unit, state)
                        self.on_progress(stage="initialization", current=unit)
                        if self.stop_at == "pretrain":
                            raise StopRequested("acceptance stop after pretraining checkpoint")
                state.update(epoch=epoch + 1, offset=0)
            state.update(theta=cpu_tree(theta), optimizer=None, stage="reference", reference_offset=0,
                         fisher_sum=torch.zeros((len(ltm), len(ltm)), dtype=torch.float64))
            store.save_checkpoint(unit, state)
        theta = state["theta"].to(cfg.device)
        if state["stage"] == "reference":
            for start in range(state["reference_offset"], cfg.reference_size, cfg.fisher_chunk):
                self.check()
                tick = time.perf_counter()
                inputs = data["reference_x"][start:start + cfg.fisher_chunk].to(theta)
                matrix = reference_fisher(model, layout, theta, inputs, indices=ltm,
                                          chunk_size=cfg.fisher_chunk, check_stop=self.check)
                state["fisher_sum"] += len(inputs) * matrix.detach().cpu().double()
                state["reference_offset"] = start + len(inputs)
                if ("prefix_sum" not in state and cfg.reference_size // 2 <= start + len(inputs)
                        < cfg.reference_size):
                    state["prefix_sum"] = state["fisher_sum"].clone()
                    state["prefix_count"] = start + len(inputs)
                synchronize(cfg.device)
                state["reference_seconds"] += time.perf_counter() - tick
                store.save_checkpoint(unit, state)
                self.on_progress(stage="initial_fisher", current=unit)
            full = state["fisher_sum"] / cfg.reference_size
            health = dict(samples=cfg.reference_size, trace=float(full.trace()),
                          target_integration="exact fixed-variance Gaussian", nested_relative_frobenius=None)
            if "prefix_sum" in state:
                prefix = state.pop("prefix_sum") / state.pop("prefix_count")
                health["nested_relative_frobenius"] = float((prefix - full).norm() / full.norm()) if full.norm() > 0 else None
            state["learner"] = initial_state(theta, ltm, full, capacity=cfg.reference_capacity,
                seed=cfg.seed_for(family, replica, "initial_compression"), check_stop=self.check)
            state["fisher_health"] = health
            state["initial_panel_metrics"] = {
                domain: panel_metrics(model, layout, theta, panel, cfg, self.check)
                for domain, panel in data["panels"].items()}
            val_mean = predict(model, layout, theta, data["validation_x"], cfg.evaluation_chunk, self.check)
            val_error = (cfg.angle_scale * val_mean.double() - data["validation_y"].double()).square()
            state["initialization_validation"] = dict(
                mse=float(val_error.mean()), zero_predictor_mse=float(data["validation_y"].double().square().mean()),
                count=len(val_error))
            del state["fisher_sum"]
            state["stage"] = "ready"
            store.save_checkpoint(unit, state)
        store.write_json(unit, "health.json", dict(fisher=state["fisher_health"],
            initialization_validation=state["initialization_validation"],
            initial_panel_metrics=state["initial_panel_metrics"],
            pretrain_seconds=state["pretrain_seconds"], reference_seconds=state["reference_seconds"]))
        store.complete(unit)
        return unit, state, data, model, layout

    def _row(self, model, layout, theta, data, state, n, observations):
        measured = {domain: panel_metrics(model, layout, theta, panel, self.cfg, self.check)
                    for domain, panel in data["panels"].items()}
        row = dict(n=n, observations=observations, base_observations=observations - n,
                   learning_seconds=state["learning_seconds"])
        for domain, values in measured.items():
            row.update({f"{domain}_{name}": values[name] for name in ("mse", "rmse", "nll")})
        row["per_digit_mse"] = {domain: values["per_digit_mse"] for domain, values in measured.items()}
        return row

    def run(self, family: str, replica: int, partition: str, recipe: Recipe, *, target=None,
            parent_target=None, capture=False, audit=False):
        from .curvature import reference_fisher
        from .learner import stage_arrival, commit_arrival, fit_step, replay_indices
        from .model import memory_indices

        cfg, store = self.cfg, self.store
        target = cfg.novel_count if target is None else target
        unit = trajectory_id(family, replica, recipe, target)
        if store.is_complete(unit):
            requested = dict(family=family, partition=partition, replica=replica,
                recipe=recipe.mapping(), target=target, capture=capture, audit=audit,
                parent=trajectory_id(family, replica, recipe, parent_target) if parent_target else None)
            persisted = read_json(store.unit_path(unit) / "manifest.json")["metadata"]
            if any(persisted.get(key) != value for key, value in requested.items()):
                raise ValueError("completed trajectory request changes scientific metadata")
            return self.record(unit)
        asset_id, shared, data, model, layout = self.assets(family, replica, partition)
        ltm, stm = memory_indices(layout)
        parent = trajectory_id(family, replica, recipe, parent_target) if parent_target else None
        if parent and not store.is_complete(parent):
            raise ValueError("continuation requires a completed identical-policy prefix")
        parent_seal = read_json(store.unit_path(parent) / "COMPLETED")["seal_hash"] if parent else None
        meta = dict(schema_version=1, kind="regression_trajectory", family=family, partition=partition,
                    replica=replica, recipe=recipe.mapping(), recipe_key=recipe.key,
                    asset_id=asset_id, plan_hash=shared["plan_hash"], target=target,
                    parent=parent, parent_seal=parent_seal, capture=capture, audit=audit)
        store.begin(unit, meta)
        state = store.load_checkpoint(unit)
        if state is None:
            if parent:
                state = copy.deepcopy(store.load_checkpoint(parent))
                if state["n"] != parent_target or parent_target >= target:
                    raise ValueError("invalid prefix checkpoint boundary")
            else:
                learner = copy.deepcopy(shared["learner"])
                learner["capacity"] = recipe.capacity
                state = dict(learner=learner, next_segment=0, n=0, rows=[], fit_records=[],
                             retirements=[], retired_audit=[], snapshots=[], audits=[],
                             learning_seconds=0., evaluation_seconds=0., audit_seconds=0.,
                             input_seconds=0., fit_seconds=0., eviction_seconds=0.,
                             layout=layout.metadata(), active_seconds=0.)
                state["peak_vram_bytes"] = 0
                state["rows"].append(self._row(model, layout, learner["theta"].to(cfg.device), data, state, 0, 0))
            store.save_checkpoint(unit, state)
        layout.assert_metadata(state["layout"])
        fit_cfg = settings(cfg, recipe)
        segments = data["plan"]["segments"]
        flags = data["plan"]["stream_novel"]
        audit_ids = [i for i, value in enumerate(flags) if value][:8]
        audit_events = {1, max(1, cfg.screen_novel_count // 2), cfg.screen_novel_count,
                        max(1, cfg.novel_count // 2), cfg.novel_count}
        if cfg.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(state["next_segment"], len(segments)):
            if state["n"] >= target:
                break
            self.check()
            tick = time.perf_counter()
            batch = segments[step]
            ids = list(range(batch["start"], batch["end"]))
            new_n = batch["novel_count"]
            if new_n > target:
                raise ValueError("event segment overshot exact budget")
            old = state["learner"]
            staged = stage_arrival(old, ids, ids)
            selected = replay_indices(old, recipe.replay_count,
                cfg.seed_for(family, replica, f"replay:{step}"))
            selected = torch.as_tensor(selected, dtype=torch.long).tolist()
            replay = [old["replay"][i]["input_ref"] for i in selected]
            theta = old["theta"].to(cfg.device)
            x, y = data["stream_x"], data["stream_y"]
            new_x, new_y = x[ids].to(theta), y[ids].to(theta)
            replay_x, replay_y = x[replay].to(theta), y[replay].to(theta)
            ages = staged["ages"][selected].to(theta.device)
            seed = cfg.seed_for(family, replica, f"probes:{step}")
            synchronize(cfg.device)
            input_seconds = time.perf_counter() - tick
            snapshot = None
            if capture and new_n > state["n"] and new_n in audit_events:
                snapshot = cpu_tree(dict(theta=theta, new_x=new_x, new_y=new_y,
                    replay_x=replay_x, replay_y=replay_y, ages=ages,
                    fisher=old["fisher_L"], anchor=old["anchor_L"], seed=seed, n=new_n,
                    step=step, recipe=recipe.mapping(), layout=layout.metadata()))
            branch = []

            def observer(accepted, objective, metadata, diagnostics):
                if audit and new_n > state["n"] and new_n in audit_events:
                    from .model import flat_prediction
                    audit_tick = time.perf_counter()
                    enlarged = dataclasses.replace(fit_cfg, max_iter=fit_cfg.max_iter * cfg.audit_multiplier,
                                                    max_eval=fit_cfg.max_eval * cfg.audit_multiplier)
                    candidate, result = _bounded_solve(theta, objective, enlarged, self.check)
                    base_value = diagnostics["final_objective"]
                    improved_value = result["final_objective"]
                    gain = base_value - improved_value if base_value is not None and improved_value is not None else None
                    constant = math.log(cfg.angle_scale * math.sqrt(2 * math.pi))
                    relative_gain = gain / max(abs(base_value - constant), 1e-12) if gain is not None else None
                    with torch.no_grad():
                        audit_x = torch.cat((new_x, replay_x))
                        change = cfg.angle_scale * (flat_prediction(model, layout, candidate, audit_x)
                                                    - flat_prediction(model, layout, accepted, audit_x))
                    synchronize(cfg.device)
                    branch.append(dict(n=new_n, original_objective=base_value,
                        enlarged_objective=improved_value, objective_gain=gain,
                        relative_objective_gain=relative_gain,
                        gap_normalization="objective minus Gaussian additive constant",
                        prediction_change_rmse_degrees=float(change.double().square().mean().sqrt()),
                        original_gradient_inf_norm=diagnostics["gradient_inf_norm"],
                        enlarged_gradient_inf_norm=result["gradient_inf_norm"],
                        original_reason=diagnostics["solver_reason"], enlarged_reason=result["solver_reason"],
                        parameter_difference=float((candidate - accepted).double().norm()),
                        enlarged_failed=bool(result["solver_failed"]) or gain is None,
                        seconds=time.perf_counter() - audit_tick))

            fit_checks = 0

            def fit_check():
                nonlocal fit_checks
                self.check()
                fit_checks += 1
                if self.stop_at == "mid_solve" and fit_checks == 9:
                    raise StopRequested("acceptance interruption inside bounded fit")

            fit_tick = time.perf_counter()
            accepted, diagnostics = fit_step(model, layout, theta, new_x, new_y,
                replay_x, replay_y, ages, old["fisher_L"], old["anchor_L"], recipe.method,
                fit_cfg, seed, fit_check, objective_observer=observer)
            synchronize(cfg.device)
            audit_seconds = sum(item["seconds"] for item in branch)
            fit_seconds = time.perf_counter() - fit_tick - audit_seconds
            eviction_tick = time.perf_counter()
            evicted = [entry["input_ref"] for entry in staged["evicted"]]
            fresh = reference_fisher(model, layout, accepted, x[evicted].to(accepted), indices=ltm,
                                     chunk_size=cfg.fisher_chunk, check_stop=self.check) if evicted else None
            if self.stop_at == "eviction" and evicted:
                raise StopRequested("acceptance interruption before retirement transaction")
            learner, transaction = commit_arrival(old, accepted, ids, ids, fresh,
                pi=cfg.pi, arrival_size=cfg.nominal_batch,
                seed=cfg.seed_for(family, replica, f"compression:{step}"), check_stop=self.check)
            synchronize(cfg.device)
            eviction_seconds = time.perf_counter() - eviction_tick
            candidate = copy.deepcopy(state)
            candidate.update(learner=learner, next_segment=step + 1, n=new_n)
            candidate["input_seconds"] += input_seconds
            candidate["fit_seconds"] += fit_seconds
            candidate["eviction_seconds"] += eviction_seconds
            candidate["learning_seconds"] += input_seconds + fit_seconds + eviction_seconds
            candidate["audit_seconds"] += audit_seconds
            candidate["audits"].extend(branch)
            if snapshot is not None:
                candidate["snapshots"].append(snapshot)
            for exposure in evicted:
                candidate["retirements"].append(dict(exposure=exposure, observations=batch["end"],
                                                       n=new_n, novel=bool(flags[exposure])))
            evaluation_tick = time.perf_counter()
            if new_n > state["n"]:
                if new_n != state["n"] + 1:
                    raise ValueError("a segment must expose exactly one novel event")
                candidate["rows"].append(self._row(model, layout, accepted, data, candidate, new_n, batch["end"]))
                retired = {row["exposure"]: row for row in candidate["retirements"]}
                audit_now = [i for i in audit_ids if i in retired]
                if audit_now:
                    values = predict(model, layout, accepted, x[audit_now], cfg.evaluation_chunk, self.check)
                    errors = (cfg.angle_scale * values.double() - y[audit_now].double()).square().tolist()
                    candidate["retired_audit"].extend(dict(n=new_n, exposure=i, squared_error=error,
                        observations=batch["end"], retirement_observations=retired[i]["observations"],
                        age_since_retirement=batch["end"] - retired[i]["observations"])
                        for i, error in zip(audit_now, errors, strict=True))
            synchronize(cfg.device)
            candidate["evaluation_seconds"] += time.perf_counter() - evaluation_tick
            candidate["fit_records"].append(dict(step=step, n=new_n, observations=batch["end"],
                selected_replay_ids=replay, selected_count=len(replay),
                input_seconds=input_seconds, fit_seconds=fit_seconds, eviction_seconds=eviction_seconds,
                archive_scale=float(torch.as_tensor(fresh).trace()) / len(ltm) if fresh is not None else None,
                **cpu_tree(diagnostics), **transaction))
            candidate["active_seconds"] += time.perf_counter() - tick
            if cfg.device == "cuda":
                candidate["peak_vram_bytes"] = max(candidate["peak_vram_bytes"], torch.cuda.max_memory_allocated())
            store.save_checkpoint(unit, candidate)
            state = candidate
            self.on_progress(stage=family, current=unit, novel_count=state["n"], target=target)
            if self.after_commit:
                self.after_commit(unit, state)
            if self.stop_at == "accepted":
                raise StopRequested("acceptance interruption after accepted arrival")
        if state["n"] != target or len(state["rows"]) != target + 1:
            raise ValueError("trajectory ended without every exact-budget prediction")
        rows = state["rows"]
        summary = dict(schema_version=1, unit_id=unit, family=family, partition=partition,
            replica=replica, method=recipe.method, recipe=recipe.mapping(), recipe_key=recipe.key,
            capacity=recipe.capacity, replay_count=recipe.replay_count, target=target,
            novel_auc=sum(row["novel_mse"] for row in rows[1:]) / target,
            base_auc=sum(row["base_mse"] for row in rows[1:]) / target,
            frozen_novel_mse=rows[0]["novel_mse"], frozen_base_mse=rows[0]["base_mse"],
            observations=rows[-1]["observations"], learning_seconds=state["learning_seconds"],
            input_seconds=state["input_seconds"], fit_seconds=state["fit_seconds"],
            eviction_seconds=state["eviction_seconds"], evaluation_seconds=state["evaluation_seconds"],
            active_seconds=state["active_seconds"], audit_seconds=state["audit_seconds"],
            solver_failures=sum(bool(row["solver_failed"]) for row in state["fit_records"]),
            unconverged_updates=sum(not row["converged"] for row in state["fit_records"]),
            peak_vram_bytes=state["peak_vram_bytes"],
            process_peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            asset_id=asset_id, plan_hash=shared["plan_hash"], pairing_id=shared["plan_hash"],
            peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            update_latency_p95_seconds=float(torch.quantile(torch.tensor([
                item["input_seconds"] + item["fit_seconds"] + item["eviction_seconds"]
                for item in state["fit_records"]], dtype=torch.float64), .95)),
            parent=parent, parent_seal=parent_seal)
        summary["valid"] = summary["solver_failures"] == 0 and all(
            math.isfinite(row[key]) for row in rows for key in ("novel_mse", "base_mse"))
        store.write_json(unit, "summary.json", summary)
        store.write_json(unit, "metrics.json", rows)
        store.write_csv(unit, "metrics.csv", [{k: v for k, v in row.items() if k != "per_digit_mse"} for row in rows])
        store.write_json(unit, "fits.json", state["fit_records"])
        store.write_json(unit, "retirement.json", dict(retirements=state["retirements"], audit=state["retired_audit"]))
        store.write_json(unit, "audits.json", state["audits"])
        store.complete(unit)
        return {**summary, "metrics": rows, "retired_audit": state["retired_audit"]}

    def record(self, unit):
        if not self.store.is_complete(unit):
            raise ValueError("incomplete trajectory cannot supply final results")
        path = self.store.unit_path(unit)
        return {**read_json(path / "summary.json"), "metrics": read_json(path / "metrics.json"),
                "retired_audit": read_json(path / "retirement.json")["audit"]}

    def snapshots(self, family, replica, recipe, target):
        unit = trajectory_id(family, replica, recipe, target)
        if not self.store.is_complete(unit):
            raise ValueError("numerical calibration needs a sealed trajectory")
        return self.store.load_checkpoint(unit)["snapshots"]
