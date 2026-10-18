"""Accepted-arrival execution and bounded measurement for Plan 15."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import resource
import time

import numpy as np
import torch

from .config import Config, Treatment, content_hash
from .curvature import jvp_values, reference_fisher
from .data import audit_cohorts, class_weights, panel
from .initialize import cpu_tree, fit_settings, initialize
from .learner import commit_arrival, fit_step, stage_arrival
from .metrics import (cohort_retention_auc, cohort_summary, dependence_metrics,
                      normalized_auc, predictive_metrics, standardized_pr_curve)
from .model import build_model, effective_residuals, flat_logits, memory_indices


def common_fd_step(audits, preferred):
    """A step must pass at every checkpoint; one invalid FD method cannot veto A/C."""
    eligible = [{row["h"] for row in audit["methods"]["finite_difference"].get("candidates", [])
                 if row.get("valid")} for audit in audits]
    common = set.intersection(*eligible) if eligible else set()
    return min(common, key=lambda h: (abs(math.log(h / preferred)), h)) if common else None


class Runner:
    def __init__(self, cfg: Config, store, source, *, on_progress=lambda: None,
                 after_commit=None):
        self.cfg, self.store, self.source = cfg, store, source
        self.on_progress, self.after_commit = on_progress, after_commit
        self.completed_this_invocation = 0

    def _assets(self, family, env, replica):
        kind = "mixture" if env == "digit9_mixture" else "rotation"
        return initialize(self.store, self.source, self.cfg, family, replica, kind,
                          on_progress=self.on_progress)

    def _model(self, shared):
        seeds = shared["assets"]["plan"]["seeds"]
        model, layout = build_model(seeds["initialization_model"], device=self.cfg.device,
                                    dtype=getattr(torch, self.cfg.dtype), factor_seed=seeds["factors"])
        layout.assert_metadata(shared["layout"])
        return model, layout

    def _logits(self, model, layout, theta, x, *, tau=1.):
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                self.store.check()
                parts.append(flat_logits(model, layout, theta, x[start:start + 256].to(theta),
                                         stm_scale=tau).double().cpu())
        return torch.cat(parts)

    def _fisher(self, model, layout, theta, x, indices):
        total = torch.zeros(len(indices), len(indices), dtype=torch.float64)
        for start in range(0, len(x), self.cfg.fisher_chunk):
            self.store.check()
            block = x[start:start + self.cfg.fisher_chunk].to(theta)
            total += len(block) * reference_fisher(model, layout, theta, block,
                                                  indices=indices, chunk_size=self.cfg.fisher_chunk).cpu()
        return total / len(x)

    def _historical_panel(self, assets, env, cohort):
        # This panel is independent of training; its mixture matches the cohort's
        # entire insertion window, not merely its last environment coordinate.
        blocks = [panel(assets, env, coordinate) for coordinate in cohort["coordinates"]]
        return (torch.cat([b[0] for b in blocks]), torch.cat([b[1] for b in blocks]),
                torch.cat([b[2] / len(blocks) for b in blocks]))

    def _evaluate(self, model, layout, theta, assets, env, step, state, rows, base):
        cfg, width = self.cfg, self.cfg.batch_size(env)
        burn = width if env == "digit9_mixture" else 0
        exposure = burn + step * width
        values = cfg.schedule(env)
        coordinate = values[step - 1] if step else 0.
        common = dict(base, step=step, exposure=exposure, coordinate=coordinate,
                      expected_nines=width * sum(values[:step]) if burn else None,
                      realized_nines=int((assets["targets"][:exposure] == 9).sum()))

        def measure(split, x, y, weights=None, **extra):
            logits = self._logits(model, layout, theta, x)
            conditional = None
            if env == "digit9_mixture" and split in ("current", "initial", "historical", "historical_pre_eviction"):
                probabilities = assets["plan"]["class_probabilities"]
                non_nine = sum(probabilities[:9])
                conditional = class_weights(y, [.9 * p / non_nine for p in probabilities[:9]] + [.1])
            rows.append(dict(common, split=split, **predictive_metrics(
                logits, y, weights, conditional_weights=conditional), **extra))
            return logits

        def pr_rows(logits, y, weights, population, **extra):
            curve = standardized_pr_curve(logits.softmax(-1), y, weights)
            for threshold, recall, precision, fpr in zip(curve["thresholds"], curve["recall"],
                                                        curve["precision"], curve["fpr"], strict=True):
                rows.append(dict(common, split="pr", population=population, threshold=threshold,
                                 nine_recall=recall, nine_precision=precision, nine_fpr=fpr,
                                 average_precision=curve["average_precision"], p_ref=.1, **extra))

        def reliance_rows(full, hx, hy, weights, info):
            for tau in (1., .9, .5, 0.):
                scaled = self._logits(model, layout, theta, hx, tau=tau)
                diagnostic = dependence_metrics(full, scaled, hy, weights)
                rows.append(dict(common, split="dependence", tau=tau, **info,
                                 reliance_nll=diagnostic["nll_difference"],
                                 predictive_kl=diagnostic["predictive_kl"],
                                 ltm_only_nll=diagnostic["intervened_nll"] if tau == 0 else None))

        if step == 0 or step == len(values) or step % cfg.evaluation_every == 0:
            x, y, weights = panel(assets, env, coordinate)
            logits = measure("current", x, y, weights)
            frozen = self._logits(model, layout, state["initial_theta"].to(theta), x)
            rows[-1]["frozen_nll"] = predictive_metrics(frozen, y, weights)["nll"]
            measure("initial", *panel(assets, env, 0.))
            if env == "digit9_mixture" and step == len(values):
                pr_rows(logits, y, weights, "current")
        archived = set(state["learner"]["consolidated_ids"])
        for cohort in audit_cohorts(cfg, env):
            if step == cohort["first_eviction_step"] - 1:
                ids = cohort["ids"]
                if max(ids) < exposure:
                    measure("cohort_pre_eviction", assets["streams"][env][ids], assets["targets"][ids],
                            cohort_id=cohort["cohort_id"], cohort_age=(step - cohort["eviction_step"]) * width)
                    hx, hy, weights = self._historical_panel(assets, env, cohort)
                    full = measure("historical_pre_eviction", hx, hy, weights,
                            cohort_id=cohort["cohort_id"], cohort_age=(step - cohort["eviction_step"]) * width)
                    reliance_rows(full, hx, hy, weights, dict(cohort_id=cohort["cohort_id"],
                                  cohort_age=(step - cohort["eviction_step"]) * width,
                                  horizon=cohort["horizon"], pre_eviction=True))
            if not set(cohort["ids"]).issubset(archived):
                continue
            age = (step - cohort["eviction_step"]) * width
            if age < 0 or age > cohort["horizon"]:
                continue
            followups = (0, min(cfg.replay_capacity, cohort["horizon"]), cohort["horizon"])
            if age not in followups and step % cfg.evaluation_every:
                continue
            info = dict(cohort_id=cohort["cohort_id"], cohort_age=age, horizon=cohort["horizon"],
                        eviction_exposure=burn + cohort["eviction_step"] * width,
                        end_exposure=burn + len(values) * width, capacity=cfg.replay_capacity)
            ids = cohort["ids"]
            x, y = assets["streams"][env][ids], assets["targets"][ids]
            measure("evicted_train", x, y, **info)
            hx, hy, weights = self._historical_panel(assets, env, cohort)
            full = measure("historical", hx, hy, weights, **info)
            if env == "digit9_mixture" and age == cohort["horizon"]:
                pr_rows(full, hy, weights, "historical", **info)
            if age in followups:
                reliance_rows(full, hx, hy, weights, info)
                _, stm = memory_indices(layout)
                if age == 0:
                    state["cohort_snapshots"][cohort["cohort_id"]] = theta.detach().cpu()
                elif cohort["cohort_id"] in state["cohort_snapshots"]:
                    earlier = state["cohort_snapshots"][cohort["cohort_id"]].to(theta)
                    swap = earlier.clone()
                    swap[stm.to(theta.device)] = theta[stm.to(theta.device)]
                    before = self._logits(model, layout, earlier, hx)
                    after = self._logits(model, layout, swap, hx)
                    diagnostic = dependence_metrics(before, after, hy, weights)
                    rows.append(dict(common, split="dependence", **info,
                                     vulnerability_nll=diagnostic["nll_difference"],
                                     predictive_kl=diagnostic["predictive_kl"],
                                     stm_update_norm=float((swap - earlier).norm())))
                generator = torch.Generator().manual_seed(cfg.seed_for(
                    base["family"], base["replica"], f"{env}:diagnostic:{cohort['cohort_id']}:{step}"))
                probes = (torch.randint(0, 2, (cfg.diagnostic_probes, len(stm)), generator=generator) * 2 - 1).to(theta)
                self.store.check()
                audit_x = x[:cfg.diagnostic_inputs]
                independent = jvp_values(model, layout, theta, audit_x.to(theta), probes).detach().double()
                trace = float(independent.mean())
                full_fisher = self._fisher(model, layout, theta, audit_x, torch.arange(layout.total_numel))
                ltm, stm = memory_indices(layout)
                diagonal = full_fisher.diagonal()
                residuals = effective_residuals(model, layout, theta)
                rows.append(dict(common, split="curvature", **info, trace_stm=trace,
                                 trace_stm_per_parameter=trace / len(stm), probe_count=len(probes),
                                 exact_trace_stm=float(diagonal[stm].sum()),
                                 exact_trace_ltm=float(diagonal[ltm].sum()),
                                 trace_ltm=float(diagonal[ltm].sum()),
                                 factor_norm=float(theta[stm.to(theta.device)].norm()),
                                 effective_residual_norm=float(sum(r.square().sum() for r in residuals.values()).sqrt()),
                                 exact_trace_stm_per_parameter=float(diagonal[stm].mean()),
                                 exact_trace_ltm_per_parameter=float(diagonal[ltm].mean()),
                                 audit_inputs=len(audit_x),
                                 measurement="independent_model_label_random_probe"))

    def run_trajectory(self, family: str, phase: str, env: str, replica: int,
                       treatment: Treatment) -> dict:
        cfg, store = self.cfg, self.store
        unit = f"trajectory-{family}-{env}-{replica:04d}-{treatment.key}"
        asset_id, shared = self._assets(family, env, replica)
        metadata = dict(schema_version=1, kind="trajectory", family=family, phase=str(phase),
                        environment=env, replica=replica, condition=treatment.condition,
                        config=treatment.mapping(), asset_id=asset_id,
                        expected_nines=cfg.expected_nines(env), smoke=cfg.smoke,
                        scientific_revision=cfg.protocol, protocol_revision=cfg.protocol,
                        source_hash=store.identity.get("source_hash"), config_hash=cfg.scientific_hash,
                        treatment_id=treatment.key,
                        seeds={name: [cfg.seed_for(family, replica, f"{env}:{name}:{step}")
                                      for step in range(1, len(cfg.schedule(env)) + 1)]
                               for name in ("training_probes", "compression")})
        if store.unit_path(unit).exists() and store.is_complete(unit):
            recorded = json.loads((store.unit_path(unit) / "metadata.json").read_text())
            if recorded != metadata:
                raise ValueError("completed trajectory metadata changed")
            return json.loads((store.unit_path(unit) / "summary.json").read_text())
        store.begin(unit, metadata)
        store.write_json(unit, "metadata.json", metadata)
        model, layout = self._model(shared)
        ltm, _ = memory_indices(layout)
        assets = shared["assets"]
        state = store.load_checkpoint(unit)
        base = dict(schema_version=1, family=family, phase=str(phase), environment=env,
                    replica=replica, condition=treatment.condition)
        if state is None:
            state = dict(step=0, learner=copy.deepcopy(shared["learner"]), rows=[],
                         initial_theta=shared["learner"]["theta"].clone(),
                         calibration_snapshots=[], cohort_snapshots={}, fit_records=[],
                         active_seconds=0., layout=layout.metadata())
            self._evaluate(model, layout, state["learner"]["theta"].to(cfg.device), assets,
                           env, 0, state, state["rows"], base)
            store.save_checkpoint(unit, state)
        layout.assert_metadata(state["layout"])
        width = cfg.batch_size(env)
        burn = width if env == "digit9_mixture" else 0
        x, y = assets["streams"][env], assets["targets"]
        settings = fit_settings(cfg, env, treatment)
        nsteps = len(cfg.schedule(env))
        calibration_steps = {2, max(2, math.ceil(cfg.replay_capacity / width)),
                             min(nsteps, math.ceil(2 * cfg.replay_capacity / width) + 1)}
        for step in range(state["step"] + 1, nsteps + 1):
            store.check()
            tick = time.perf_counter()
            old = state["learner"]
            ids = list(range(burn + (step - 1) * width, burn + step * width))
            replay = [item["input_ref"] for item in old["replay"]]
            staged = stage_arrival(old, ids, ids)
            before = old["theta"].to(cfg.device)
            prequential = predictive_metrics(self._logits(model, layout, before, x[ids]), y[ids])
            theta, diagnostics = fit_step(
                model, layout, before, x[ids].to(before), y[ids].to(cfg.device),
                x[replay].to(before), y[replay].to(cfg.device), staged["ages"].to(cfg.device),
                old["fisher_L"], old["anchor_L"], treatment.condition, settings,
                cfg.seed_for(family, replica, f"{env}:training_probes:{step}"), store.check)
            evicted = [item["input_ref"] for item in staged["evicted"]]
            fresh = self._fisher(model, layout, theta, x[evicted], ltm) if evicted else None
            learner, transaction = commit_arrival(
                old, theta, ids, ids, fresh, pi=settings.pi, arrival_size=width,
                seed=cfg.seed_for(family, replica, f"{env}:compression:{step}"), check_stop=store.check)
            # Nothing in this candidate state is visible until its checkpoint commits.
            candidate = copy.deepcopy(state)
            candidate.update(step=step, learner=learner)
            if step in calibration_steps and treatment.condition == "replay_ewc" and family in ("calibration", "policy_development"):
                candidate["calibration_snapshots"].append(cpu_tree(dict(
                    learner=old, new_ids=ids, replay_ids=replay, ages=staged["ages"], step=step)))
            candidate["rows"].append(dict(base, step=step, exposure=burn + step * width,
                                           split="prequential", **prequential))
            candidate["fit_records"].append(dict(step=step, **diagnostics, **transaction))
            self._evaluate(model, layout, theta, assets, env, step, candidate, candidate["rows"], base)
            candidate["active_seconds"] += time.perf_counter() - tick
            for row in candidate["rows"][len(state["rows"]):]:
                row["cumulative_active_seconds"] = candidate["active_seconds"]
            store.save_checkpoint(unit, candidate)
            state = candidate
            store.write_json(unit, "progress.json", dict(schema_version=1, sealed=True,
                             metadata=metadata, rows=state["rows"], accepted_step=step))
            store.heartbeat()
            self.on_progress()
            if self.after_commit is not None:
                self.after_commit(unit, state)
        summary = self._summary(state, unit, metadata)
        summary["initial_fisher_health"] = shared["initial_fisher_health"]
        store.write_json(unit, "metrics.json", state["rows"])
        store.write_csv(unit, "metrics.csv", state["rows"])
        store.write_json(unit, "fit.json", state["fit_records"])
        store.write_json(unit, "summary.json", summary)
        store.write_json(unit, "audit_cohorts.json", audit_cohorts(cfg, env))
        store.complete(unit)
        self.completed_this_invocation += 1
        self.on_progress()
        return summary

    def _summary(self, state, unit, metadata):
        rows = state["rows"]
        def auc(split):
            selected = [r for r in rows if r["split"] == split]
            return normalized_auc([r["exposure"] for r in selected], [r["nll"] for r in selected])
        result = dict(unit_id=unit, current_auc=auc("current"), prequential_auc=auc("prequential"),
                      initial_auc=auc("initial"), active_seconds=state["active_seconds"],
                      fit_seconds=sum(r["elapsed_seconds"] for r in state["fit_records"]),
                      closure_calls=sum(r["total_closure_calls"] for r in state["fit_records"]),
                      nonfinite=any("nonfinite" in r["solver_reason"] for r in state["fit_records"]),
                      solver_failures=sum(r["solver_failed"] for r in state["fit_records"]),
                      expected_nines=metadata["expected_nines"])
        for split in ("historical", "evicted_train"):
            cohorts = []
            for name in sorted({r.get("cohort_id") for r in rows if r["split"] == split}):
                series = [r for r in rows if r["split"] == split and r["cohort_id"] == name]
                r = series[0]
                cohorts.append(dict(cohort_id=name, **cohort_retention_auc(
                    [s["exposure"] for s in series], [s["nll"] for s in series],
                    eviction_exposure=r["eviction_exposure"], end_exposure=r["end_exposure"],
                    capacity=r["capacity"])))
            result[f"{split}_auc"] = cohort_summary(cohorts)["mean_auc"]
            result[f"{split}_cohorts"] = cohorts
        last = [r for r in rows if r["split"] == "current"][-1]
        result.update({key: last[key] for key in ("nine_precision", "nine_recall", "realized_nines")})
        norms = [r["gradient_inf_norm"] for r in state["fit_records"] if r["gradient_inf_norm"] is not None]
        result["mean_final_gradient_norm"] = float(np.mean(norms)) if norms else None
        current = [r for r in rows if r["split"] == "current"]
        result["frozen_current_auc"] = normalized_auc([r["exposure"] for r in current],
                                                     [r["frozen_nll"] for r in current])
        result["acquisition_ok"] = result["current_auc"] < result["frozen_current_auc"]
        before = {r["cohort_id"]: r["nll"] for r in rows if r["split"] == "historical_pre_eviction"}
        changes = [r["auc"] - before[r["cohort_id"]] for r in result["historical_cohorts"]
                   if r["auc"] is not None and r["cohort_id"] in before]
        result["retention_change_from_pre_eviction"] = float(np.mean(changes)) if changes else None
        result["retention_ok"] = bool(np.mean(changes) <= self.cfg.retention_margin) if changes else None
        result["solver_ok"] = not result["nonfinite"] and result["solver_failures"] == 0
        result["under_solved_fraction"] = float(np.mean([
            not r["converged"] and r["gradient_inf_norm"] is not None and r["gradient_inf_norm"] > 1e-3
            for r in state["fit_records"]]))
        dependence = [r for r in rows if r["split"] == "dependence" and r.get("tau") == 0. and not r.get("pre_eviction")]
        before_dependence = [r for r in rows if r["split"] == "dependence" and r.get("tau") == 0. and r.get("pre_eviction")]
        vulnerability = [r["vulnerability_nll"] for r in rows if r.get("vulnerability_nll") is not None]
        result["mechanism"] = {
            "mean_stm_removal_harm": float(np.mean([r["reliance_nll"] for r in dependence])) if dependence else None,
            "mean_ltm_only_nll": float(np.mean([r["ltm_only_nll"] for r in dependence])) if dependence else None,
            "mean_future_stm_swap_harm": float(np.mean(vulnerability)) if vulnerability else None,
            "mean_pre_eviction_stm_removal_harm": float(np.mean([r["reliance_nll"] for r in before_dependence])) if before_dependence else None,
            "interpretation": "Descriptive historical-cohort interventions; compare paired controls before interpreting transfer."}
        result["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if self.cfg.device == "cuda" else 0
        return result

    def calibrate(self, family, env, replica, baseline_treatment):
        from .numerical import audit, aggregate_audits
        from .scheduler import validate_calibration
        summary = self.run_trajectory(family, "phase1", env, replica, baseline_treatment)
        unit = f"calibration-{family}-{env}-{replica:04d}-{baseline_treatment.key}"
        store, cfg = self.store, self.cfg
        if store.unit_path(unit).exists() and store.is_complete(unit):
            return json.loads((store.unit_path(unit) / "result.json").read_text())
        store.begin(unit, dict(kind="calibration", environment=env, replica=replica,
                              family=family, treatment=baseline_treatment.mapping()))
        partial = store.load_checkpoint(unit) or dict(audits=[])
        trajectory = store.load_checkpoint(summary["unit_id"])
        _, shared = self._assets(family, env, replica)
        model, layout = self._model(shared)
        assets = shared["assets"]
        x, y = assets["streams"][env], assets["targets"]
        settings = fit_settings(cfg, env, baseline_treatment)
        if "common_fd_step" in partial:
            settings = dataclasses.replace(settings, fd_step=partial["common_fd_step"])
        snapshots = trajectory["calibration_snapshots"]
        if len(snapshots) < 2:
            raise ValueError("calibration needs shared young and post-eviction checkpoints")
        def measure_pending():
            for snapshot in snapshots[len(partial["audits"]):]:
                learner = snapshot["learner"]
                theta = learner["theta"].to(cfg.device)
                new, replay = snapshot["new_ids"], snapshot["replay_ids"]
                result = audit(model, layout, theta, x[new].to(theta), y[new].to(cfg.device),
                               x[replay].to(theta), y[replay].to(cfg.device), snapshot["ages"].to(cfg.device),
                               learner["fisher_L"], learner["anchor_L"], settings,
                               cfg.seed_for(family, replica, f"{env}:calibration:{snapshot['step']}"),
                               cfg=cfg, checkpoint_label=f"step-{snapshot['step']}", check_stop=store.check)
                partial["audits"].append(result)
                store.save_checkpoint(unit, partial)
                self.on_progress()
        measure_pending()
        steps = {row["selected_fd_step"] for row in partial["audits"]}
        if len(steps) > 1 and "common_fd_step" not in partial:
            chosen = common_fd_step(partial["audits"], settings.fd_step)
            partial["original_audits"] = copy.deepcopy(partial["audits"])
            if chosen is not None:
                partial.update(common_fd_step=chosen, audits=[])
                store.save_checkpoint(unit, partial)
                settings = dataclasses.replace(settings, fd_step=chosen)
                measure_pending()
        combined = copy.deepcopy(partial["audits"])
        fd_unavailable = len({row["selected_fd_step"] for row in combined}) > 1
        if fd_unavailable:
            for row in combined:
                row["calibration_signature"]["fd_step"] = settings.fd_step
                row["methods"]["finite_difference"]["valid"] = False
                row["calibration"]["finite_difference"]["calibrated"] = False
            partial["fd_resolution"] = "No common checked FD step; B unavailable, other methods retained."
        aggregated = aggregate_audits(combined)
        if "common_fd_step" in partial:
            aggregated["elapsed_seconds"] += sum(row["elapsed_seconds"] for row in partial["original_audits"])
        methods = {}
        for name, value in aggregated["methods"].items():
            scale = value.get("unit_ratio_coefficient")
            valid = value["valid"] and scale is not None and scale > 0
            methods[name] = dict(value, valid=valid, unit_gradient_ratio=1 / scale if valid else None,
                                 gradient_ratio_ltm=None, gradient_ratio_stm=None,
                                 probes=aggregated["selected_probes"], fd_step=aggregated["selected_fd_step"],
                                 reason=None if valid else "Numerical check failed or no nondegenerate gradient bracket.")
            value_errors, gradient_errors, seconds = [], [], []
            for measured in partial["audits"]:
                diagnostic = measured["methods"][name]
                agreement = diagnostic.get("agreement", {})
                if name == "finite_difference":
                    candidate = next((c for c in diagnostic.get("candidates", [])
                                      if c.get("h") == measured["selected_fd_step"]), {})
                    agreement = candidate.get("agreement", {})
                    if candidate.get("seconds") is not None:
                        seconds.append(candidate["seconds"])
                if agreement.get("values", {}).get("relative_error") is not None:
                    value_errors.append(agreement["values"]["relative_error"])
                for check in agreement.get("gradients", {}).values():
                    if check.get("relative_error") is not None:
                        gradient_errors.append(check["relative_error"])
                for check in diagnostic.get("gradient_check", {}).get("blocks", {}).values():
                    if check.get("relative_error") is not None:
                        gradient_errors.append(check["relative_error"])
                if diagnostic.get("seconds") is not None:
                    seconds.append(diagnostic["seconds"])
            methods[name].update(relative_error=max(value_errors) if value_errors else None,
                                 gradient_error=max(gradient_errors) if gradient_errors else None,
                                 seconds=sum(seconds) if seconds else None)
            for block in ("ltm", "stm"):
                ratios = []
                for measured in partial["audits"]:
                    bracket = measured["calibration"][name]
                    auxiliary = bracket.get("auxiliary_gradient_norms", {}).get(block)
                    data_norm = bracket.get("data_gradient_norms", {}).get(block)
                    if auxiliary is not None and data_norm is not None and data_norm > 1e-10:
                        ratios.append(auxiliary / data_norm)
                methods[name][f"gradient_ratio_{block}"] = float(np.median(ratios)) if ratios else None
        result = dict(aggregated, methods=methods, trajectory_summary=summary,
                      fd_resolution=partial.get("fd_resolution", "Common checked FD contract."))
        validate_calibration(result, cfg)
        store.write_json(unit, "result.json", result)
        store.write_json(unit, "audits.json", partial["audits"])
        if "original_audits" in partial:
            store.write_json(unit, "original_audits.json", partial["original_audits"])
        store.complete(unit)
        self.on_progress()
        return result
