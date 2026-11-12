"""Original F1 numerical operations with a causal, resumable group observer."""

import copy
from dataclasses import replace
import resource
import time

import torch

from mnist_experiment.fisher_memory.initialize import cpu_tree
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_grid_registration import learner as original
from mnist_experiment.fisher_memory_grid_registration.curvature import reference_fisher
from mnist_experiment.fisher_memory_grid_registration.runner import settings
from mnist_experiment.fisher_memory_stm_capacity import model as capacity
from mnist_experiment.fisher_memory_stm_capacity.run import bindings
from mnist_experiment.fisher_memory_stm_capacity.runner import trajectory_id
from mnist_experiment.fisher_memory_regression.runtime import synchronize
from mnist_experiment.combined_memory.broad_drift.observer import Controller
from mnist_experiment.kalman_pi.policies import PolicySpec, concentration
from mnist_experiment.combined_memory.replicated_combination.runner import (
    Runner as PreviousRunner, record,
)
from mnist_experiment.combined_memory.replicated_combination import curvature
from . import PROTOCOL
from .config import observer_config


def archive_diagonal(archive):
    values = archive["factor"].double().square().sum(1) + archive["residual_diagonal"].double()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise FloatingPointError("invalid represented archive diagonal")
    return values


class Runner(PreviousRunner):
    def __init__(self, *args, forced_pi=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_pi = forced_pi

    def engine(self, policy):
        engine = super().engine(policy)
        if policy.name == "R":
            # Artificial recovery probes target the new integration, not R.
            engine.stop_at = None
        return engine

    def run(self, policy, replica, partition):
        if policy.name == "R":
            return super().run(policy, replica, partition)
        with bindings(), capacity.use_chart("head_r1"):
            return self.run_adaptive(policy, replica, partition)

    def run_adaptive(self, policy, replica, partition):
        cfg, base = self.cfg, self.store
        engine, (asset, shared, data, model, layout) = self.assets(replica, partition, policy)
        store, recipe = engine.store.base, policy.recipe(cfg)
        family = partition + "--grid"
        unit = trajectory_id("head_r1", family, replica, recipe, cfg.novel_count)
        physical = store.key(unit)
        if base.is_complete(physical):
            return record(base, physical)
        ltm, stm = capacity.memory_indices(layout)
        store.begin(unit, dict(kind="grid_registration_trajectory", family=family,
            partition=partition, replica=replica, recipe=recipe.mapping(), recipe_key=recipe.key,
            asset_id=asset, plan_hash=shared["plan_hash"], target=cfg.novel_count,
            forced_pi=self.forced_pi, **engine.store.chart_identity, **policy.mapping()))
        state = store.load_checkpoint(unit)
        if state is None:
            learner = copy.deepcopy(shared["learner"])
            learner["capacity"] = recipe.capacity
            state = dict(learner=learner, next_segment=0, n=0, rows=[], fit_records=[],
                retirements=[], retired_audit=[], audits=[], layout=layout.metadata(),
                q_ref=1/cfg.init_size, q_archive=1/cfg.init_size, group=None, group_pi=None,
                pending_scores=[], controller=None, group_diagnostics={}, peak_vram_bytes=0,
                **{k: 0. for k in ("active_seconds", "learning_seconds", "input_seconds", "fit_seconds",
                    "eviction_seconds", "jacobian_fisher_seconds", "compression_seconds",
                    "evaluation_seconds", "audit_seconds", "observer_seconds")})
            state["rows"].append(engine._row(model, layout, learner["theta"].to(cfg.device), data, state, 0, 0))
            store.save_checkpoint(unit, state)
        layout.assert_metadata(state["layout"])
        merged, ml, anchor, initial = self.observer_asset(engine, asset, shared, data, model, layout)
        controller = Controller(PolicySpec("kalman", policy.process_scale), observer_config(cfg),
                                .025, state["controller"])
        if state["controller"] is None:
            controller.initialize(initial["anchor"], initial["diagonal"], initial["variance"])
        x, y = data["stream_x"], data["stream_y"]
        segments, groups = data["plan"]["segments"], data["plan"]["stream_groups"]
        if cfg.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(state["next_segment"], len(segments)):
            if state["n"] == cfg.novel_count:
                break
            self.budget.check()
            tick = time.perf_counter()
            batch = segments[step]
            ids = list(range(batch["start"], batch["end"]))
            candidate, old = copy.deepcopy(state), state["learner"]
            group = batch["group"] - 1
            if state["group"] != group:
                if state["pending_scores"]:
                    raise ValueError("new group would discard unassimilated scores")
                action = controller.choose(state["q_ref"], archive_diagonal(old["fisher_L"]), 4, group)
                if self.forced_pi is not None:
                    action.update(applied_pi=self.forced_pi, acceptance_forced=True)
                candidate.update(group=group, group_pi=action["applied_pi"], group_diagnostics=action)
            pi = candidate["group_pi"]
            fit_cfg = replace(settings(cfg, recipe), pi=pi)
            staged = original.stage_arrival(old, ids, ids)
            selected = original.replay_indices(old, recipe.replay_count,
                cfg.seed_for(family, replica, f"replay:{step}")).tolist()
            replay = [old["replay"][i]["input_ref"] for i in selected]
            theta = old["theta"].to(cfg.device)
            new_x, new_y = x[ids].to(theta), y[ids].to(theta)
            replay_x, replay_y = x[replay].to(theta), y[replay].to(theta)
            ages = staged["ages"][selected].to(theta.device)
            seed = cfg.seed_for(family, replica, f"probes:{step}")
            synchronize(cfg.device)
            input_seconds = time.perf_counter()-tick + len(ids)*shared["stream_preparation_seconds"]/len(x)
            checks = 0

            def fit_check():
                nonlocal checks
                self.budget.check()
                checks += 1
                if self.stop_at == "mid_solve" and checks == 9:
                    raise StopRequested("Phase 9E interruption inside solve")

            fit_tick = time.perf_counter()
            accepted, diagnostics = original.fit_step(model, layout, theta, new_x, new_y,
                replay_x, replay_y, ages, old["fisher_L"], old["anchor_L"], recipe.method,
                fit_cfg, seed, fit_check)
            if diagnostics["solver_failed"]:
                raise FloatingPointError("Phase 9E numerical solve failed: " + str(diagnostics["solver_reason"]))
            synchronize(cfg.device)
            fit_seconds = time.perf_counter()-fit_tick
            eviction_tick = time.perf_counter()
            evicted = [r["input_ref"] for r in staged["evicted"]]
            fresh = (reference_fisher(model, layout, accepted, x[evicted].to(accepted), indices=ltm,
                chunk_size=cfg.fisher_chunk, check_stop=self.budget.check) if evicted else None)
            synchronize(cfg.device)
            jacobian_seconds = time.perf_counter()-eviction_tick
            if self.stop_at == "eviction" and evicted:
                raise StopRequested("Phase 9E interruption before eviction")
            compression_tick = time.perf_counter()
            accepted_state, transaction = original.commit_arrival(old, accepted, ids, ids, fresh,
                pi=pi, arrival_size=cfg.nominal_batch, rank=engine.archive_rank,
                seed=cfg.seed_for(family, replica, f"compression:{step}"), check_stop=self.budget.check)
            synchronize(cfg.device)
            compression_seconds = time.perf_counter()-compression_tick
            eviction_seconds = time.perf_counter()-eviction_tick
            observer_tick = time.perf_counter()
            group_end = batch["end"] == len(groups) or groups[batch["end"]] != batch["group"]
            scores = curvature.observed_scores(merged, ml, anchor, new_x.to(anchor), new_y.to(anchor),
                                               check=self.budget.check).cpu()
            candidate["pending_scores"].append(scores)
            observed = {}
            if group_end:
                all_scores = torch.cat(candidate["pending_scores"])
                observed = controller.assimilate(all_scores, group)
                candidate["q_ref"] = concentration(state["q_ref"], pi, len(all_scores))
                candidate["pending_scores"] = []
            candidate["controller"] = cpu_tree(controller.state)
            synchronize(cfg.device)
            observer_seconds = time.perf_counter()-observer_tick
            candidate.update(learner=accepted_state, next_segment=step+1, n=batch["novel_count"],
                q_archive=concentration(state["q_archive"], transaction["eta"], len(evicted))
                if evicted else state["q_archive"])
            for key, value in dict(input_seconds=input_seconds, fit_seconds=fit_seconds,
                    eviction_seconds=eviction_seconds, jacobian_fisher_seconds=jacobian_seconds,
                    compression_seconds=compression_seconds, observer_seconds=observer_seconds,
                    learning_seconds=input_seconds+fit_seconds+eviction_seconds+observer_seconds).items():
                candidate[key] += value
            for exposure in evicted:
                candidate["retirements"].append(dict(exposure=exposure, observations=batch["end"],
                    n=candidate["n"], novel=bool(data["plan"]["stream_novel"][exposure])))
            evaluation_tick = time.perf_counter()
            if candidate["n"] > state["n"]:
                if candidate["n"] != state["n"] + 1:
                    raise ValueError("novel event skipped")
                row = engine._row(model, layout, accepted, data, candidate, candidate["n"], batch["end"])
                row.update(candidate["group_diagnostics"])
                candidate["rows"].append(row)
            synchronize(cfg.device)
            candidate["evaluation_seconds"] += time.perf_counter()-evaluation_tick
            candidate["fit_records"].append(dict(step=step, n=candidate["n"], observations=batch["end"],
                selected_replay_ids=replay, selected_count=len(replay), group=group, group_end=group_end,
                input_seconds=input_seconds, fit_seconds=fit_seconds, eviction_seconds=eviction_seconds,
                compression_seconds=compression_seconds, observer_seconds=observer_seconds,
                **cpu_tree(diagnostics), **transaction, **candidate["group_diagnostics"], **observed))
            candidate["active_seconds"] += time.perf_counter()-tick
            if cfg.device == "cuda":
                candidate["peak_vram_bytes"] = max(candidate["peak_vram_bytes"], torch.cuda.max_memory_allocated())
            store.save_checkpoint(unit, candidate)
            state = candidate
            self.progress(current=physical, novel_count=state["n"])
            if self.stop_at == "split_group" and state["pending_scores"]:
                raise StopRequested("Phase 9E interruption with pending group scores")
            if self.stop_at == "accepted":
                raise StopRequested("Phase 9E interruption after accepted segment")
        if state["n"] != cfg.novel_count or state["pending_scores"] or len(state["rows"]) != cfg.novel_count+1:
            raise ValueError("incomplete adaptive F1 trajectory")
        rows = state["rows"]
        summary = dict(schema_version=1, protocol=PROTOCOL, unit_id=unit, family=family,
            partition=partition, context_key="grid", task="grid", output_dim=54,
            archive_rank=engine.archive_rank, channels=[4, 6], blur_max=cfg.blur_max,
            replica=replica, method=recipe.method, recipe=recipe.mapping(), recipe_key=recipe.key,
            capacity=recipe.capacity, replay_count=recipe.replay_count, target=cfg.novel_count,
            novel_auc=sum(r["novel_mse"] for r in rows[1:])/cfg.novel_count,
            base_auc=sum(r["base_mse"] for r in rows[1:])/cfg.novel_count,
            frozen_novel_mse=rows[0]["novel_mse"], frozen_base_mse=rows[0]["base_mse"],
            observations=rows[-1]["observations"], asset_id=asset, plan_hash=shared["plan_hash"],
            pairing_id=shared["plan_hash"], arm="head_r1--fd_replay_ewc", valid=True,
            status="completed", **engine.store.chart_identity, **policy.mapping(),
            stream_preparation_seconds=shared["stream_preparation_seconds"], parent=None, parent_seal=None,
            peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            solver_failures=0, unconverged_updates=sum(not r["converged"] for r in state["fit_records"]))
        for key in ("active_seconds", "learning_seconds", "input_seconds", "fit_seconds", "eviction_seconds",
                    "jacobian_fisher_seconds", "compression_seconds", "evaluation_seconds", "observer_seconds",
                    "audit_seconds", "peak_vram_bytes"):
            summary[key] = state[key]
        store.write_json(unit, "summary.json", summary)
        store.write_json(unit, "metrics.json", rows)
        store.write_csv(unit, "metrics.csv", [{k: v for k, v in r.items()
            if k not in ("per_digit_mse", "per_cell_mse", "coordinate_mse")} for r in rows])
        store.write_json(unit, "fits.json", state["fit_records"])
        store.write_json(unit, "retirement.json", dict(retirements=state["retirements"], audit=state["retired_audit"]))
        store.write_json(unit, "audits.json", state["audits"])
        store.complete(unit)
        return record(base, physical)
