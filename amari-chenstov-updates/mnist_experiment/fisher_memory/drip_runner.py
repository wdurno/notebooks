"""Event-aligned rare-domain trajectories for Plan 15 Phase 5B."""

from __future__ import annotations

import copy
import json
import time

import torch

from .config import Config, Treatment, content_hash
from .data import MNISTSource, panel
from .initialize import fit_settings, initialize
from .learner import commit_arrival, fit_step, stage_arrival
from .metrics import predictive_metrics
from .model import memory_indices
from .runner import Runner


def trajectory_id(family: str, replica: int, stream: str, treatment: Treatment) -> str:
    if stream not in ("mixed", "base"):
        raise ValueError("unknown drip stream")
    return f"trajectory-{family}-{replica:04d}-{stream}-{treatment.key}"


class DripRunner(Runner):
    def __init__(self, cfg: Config, store, source: MNISTSource, *, on_progress=lambda: None,
                 after_commit=None, phase="5B", protocol="plan15-drip-development-v1",
                 audit=None):
        super().__init__(cfg, store, source, on_progress=on_progress, after_commit=after_commit)
        self.phase, self.protocol = phase, protocol
        self.audit = audit

    def _assets(self, family: str, replica: int):
        return initialize(self.store, self.source, self.cfg, family, replica, "drip",
                          on_progress=self.on_progress)

    def _evaluate(self, model, layout, theta, panels, step: int, exposure: int,
                  event_start: int, event_end: int, stream: str, rows: list[dict]):
        if event_end == event_start and step:
            return
        scores = {}
        for domain, (x, y, weights) in panels.items():
            logits = self._logits(model, layout, theta, x)
            metric = predictive_metrics(logits, y, weights)
            scores[f"{domain}_nll"] = metric["nll"]
            scores[f"{domain}_accuracy"] = metric["accuracy"]
        counts = [0] if step == 0 else range(event_start + 1, event_end + 1)
        for event in counts:
            rows.append(dict(step=step, exposure=exposure, paired_rotated_event=event,
                             actual_rotated_events=event if stream == "mixed" else 0,
                             **scores))

    def run_trajectory(self, family: str, replica: int, stream: str,
                       treatment: Treatment) -> dict:
        cfg, store = self.cfg, self.store
        unit = trajectory_id(family, replica, stream, treatment)
        asset_id, shared = self._assets(family, replica)
        plan = shared["assets"]["plan"]
        metadata = dict(schema_version=1, kind="drip_trajectory", family=family,
                        phase=self.phase, replica=replica, stream=stream,
                        condition=treatment.condition, treatment=treatment.mapping(),
                        treatment_id=treatment.key, asset_id=asset_id,
                        plan_hash=content_hash(plan), event_target=plan["event_target"],
                        protocol=self.protocol,
                        source_hash=store.identity["source_hash"],
                        config_hash=cfg.scientific_hash)
        if store.unit_path(unit).exists() and store.is_complete(unit):
            recorded = json.loads((store.unit_path(unit) / "metadata.json").read_text())
            if recorded != metadata:
                raise ValueError("completed drip trajectory metadata changed")
            return json.loads((store.unit_path(unit) / "summary.json").read_text())
        store.begin(unit, metadata)
        store.write_json(unit, "metadata.json", metadata)
        model, layout = self._model(shared)
        ltm, stm = memory_indices(layout)
        assets = shared["assets"]
        x = assets["streams"]["drip_mixed" if stream == "mixed" else "drip_base"]
        rotated = assets["streams"]["drip_mixed"]
        y = assets["targets"]
        panels = {"rotated": panel(assets, "rotation_linear", 30.),
                  "base": panel(assets, "rotation_linear", 0.)}
        state = store.load_checkpoint(unit)
        if state is None:
            state = dict(step=0, learner=copy.deepcopy(shared["learner"]),
                         rows=[], fit_records=[], active_seconds=0.,
                         layout=layout.metadata())
            self._evaluate(model, layout, state["learner"]["theta"].to(cfg.device),
                           panels, 0, 0, 0, 0, stream, state["rows"])
            store.save_checkpoint(unit, state)
        layout.assert_metadata(state["layout"])
        settings = fit_settings(cfg, "rotation_linear", treatment)
        if self.audit is not None:
            self.audit.drain(unit, state)
        for step in range(state["step"] + 1, len(plan["event_batches"]) + 1):
            store.check()
            tick = time.perf_counter()
            batch = plan["event_batches"][step - 1]
            ids = list(range(batch["start"], batch["end"]))
            old = state["learner"]
            replay = [item["input_ref"] for item in old["replay"]]
            staged = stage_arrival(old, ids, ids)
            before = old["theta"].to(cfg.device)
            flags = batch["rotated"]
            event_ids = [index for index, flag in zip(ids, flags, strict=True) if flag]
            previous_events = plan["event_batches"][step - 2]["cumulative_events"] if step > 1 else 0
            seed = cfg.seed_for(family, replica, f"drip:training_probes:{step}")
            observer = None
            if self.audit is not None:
                observer = self.audit.observer(dict(
                    unit=unit, family=family, replica=replica, stream=stream,
                    treatment=treatment.mapping(), step=step,
                    previous_events=previous_events, events=batch["cumulative_events"],
                    before=before, fisher=old["fisher_L"], anchor=old["anchor_L"],
                    new_x=x[ids], new_y=y[ids], replay_x=x[replay], replay_y=y[replay],
                    ages=staged["ages"], settings=settings, seed=seed,
                    new_ids=ids, replay_ids=replay, panels=panels,
                    layout=layout.metadata()))
            prequential = None
            if event_ids:
                prequential = predictive_metrics(
                    self._logits(model, layout, before, rotated[event_ids]), y[event_ids])
            theta, diagnostics = fit_step(
                model, layout, before, x[ids].to(before), y[ids].to(cfg.device),
                x[replay].to(before), y[replay].to(cfg.device),
                staged["ages"].to(cfg.device), old["fisher_L"], old["anchor_L"],
                treatment.condition, settings,
                seed, store.check, objective_observer=observer)
            eviction_tick = time.perf_counter()
            evicted = [item["input_ref"] for item in staged["evicted"]]
            fresh = self._fisher(model, layout, theta, x[evicted], ltm) if evicted else None
            learner, transaction = commit_arrival(
                old, theta, ids, ids, fresh, pi=settings.pi, arrival_size=len(ids),
                seed=cfg.seed_for(family, replica, f"drip:compression:{step}"),
                check_stop=store.check)
            eviction_seconds = time.perf_counter() - eviction_tick
            candidate = copy.deepcopy(state)
            candidate.update(step=step, learner=learner,
                             paired_events=batch["cumulative_events"])
            evaluation_tick = time.perf_counter()
            self._evaluate(model, layout, theta, panels, step, batch["end"],
                           previous_events, batch["cumulative_events"], stream,
                           candidate["rows"])
            evaluation_seconds = time.perf_counter() - evaluation_tick
            candidate["fit_records"].append(dict(step=step, exposure=batch["end"],
                                                 paired_rotated_events=batch["cumulative_events"],
                                                 rotated_prequential_nll=prequential["nll"] if prequential else None,
                                                 rotated_prequential_accuracy=prequential["accuracy"] if prequential else None,
                                                 stm_parameter_norm=float(theta[stm.to(theta.device)].norm()),
                                                 ltm_displacement_norm=float((theta - before)[ltm.to(theta.device)].double().norm()),
                                                 stm_displacement_norm=float((theta - before)[stm.to(theta.device)].double().norm()),
                                                 fisher_scale=(diagnostics["kappa"] / treatment.ridge_multiplier
                                                               if treatment.ridge_multiplier else None),
                                                 effective_ridge_weight=(1 - settings.pi) / settings.pi * diagnostics["kappa"],
                                                 eviction_seconds=eviction_seconds,
                                                 evaluation_seconds=evaluation_seconds,
                                                 **diagnostics, **transaction))
            candidate["active_seconds"] += (time.perf_counter() - tick
                                            - diagnostics.get("audit_capture_seconds", 0.))
            for row in candidate["rows"][len(state["rows"]):]:
                row["cumulative_active_seconds"] = candidate["active_seconds"]
            store.save_checkpoint(unit, candidate)
            store.write_json(unit, "progress.json", dict(schema_version=1,
                             accepted_step=step, paired_rotated_events=batch["cumulative_events"],
                             rows=candidate["rows"]))
            state = candidate
            store.heartbeat()
            self.on_progress()
            if self.after_commit is not None:
                self.after_commit(unit, state)
            if self.audit is not None:
                self.audit.drain(unit, state)
        rows = [row for row in state["rows"] if row["paired_rotated_event"] > 0]
        if len(rows) != plan["event_target"]:
            raise ValueError("drip event AUC has missing checkpoints")
        summary = dict(unit_id=unit, schema_version=1, replica=replica, stream=stream,
                       condition=treatment.condition, strength=treatment.strength,
                       treatment=treatment.mapping(), treatment_id=treatment.key,
                       initial_rotated_nll=state["rows"][0]["rotated_nll"],
                       initial_base_nll=state["rows"][0]["base_nll"],
                       rotated_auc=sum(row["rotated_nll"] for row in rows) / len(rows),
                       base_auc=sum(row["base_nll"] for row in rows) / len(rows),
                       final_rotated_nll=rows[-1]["rotated_nll"],
                       final_base_nll=rows[-1]["base_nll"],
                       final_rotated_accuracy=rows[-1]["rotated_accuracy"],
                       final_base_accuracy=rows[-1]["base_accuracy"],
                       event_target=plan["event_target"], batches=len(plan["event_batches"]),
                       exposure=plan["event_batches"][-1]["end"],
                       per_class_events={str(k): sum(plan["stream_labels"][i] == k
                           for batch in plan["event_batches"] for i, flag in
                           zip(range(batch["start"], batch["end"]), batch["rotated"], strict=True)
                           if flag) for k in range(10)},
                       active_seconds=state["active_seconds"],
                       fit_seconds=sum(r["elapsed_seconds"] for r in state["fit_records"]),
                       eviction_seconds=sum(r.get("eviction_seconds", 0.) for r in state["fit_records"]),
                       evaluation_seconds=sum(r.get("evaluation_seconds", 0.) for r in state["fit_records"]),
                       closure_calls=sum(r["total_closure_calls"] for r in state["fit_records"]),
                       solver_failures=sum(r["solver_failed"] for r in state["fit_records"]),
                       nonfinite=any("nonfinite" in r["solver_reason"] for r in state["fit_records"]),
                       under_solved=sum(r.get("gradient_inf_norm") is not None and
                                        r["gradient_inf_norm"] > 1e-3 for r in state["fit_records"]),
                       evictions=sum(r["evicted_count"] for r in state["fit_records"]))
        store.write_json(unit, "metrics.json", state["rows"])
        store.write_csv(unit, "metrics.csv", state["rows"])
        store.write_json(unit, "fit.json", state["fit_records"])
        store.write_json(unit, "summary.json", summary)
        store.complete(unit)
        self.completed_this_invocation += 1
        self.on_progress()
        return summary
