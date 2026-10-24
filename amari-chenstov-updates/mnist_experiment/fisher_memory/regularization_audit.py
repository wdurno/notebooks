"""Immutable, off-trajectory re-solves of selected frozen Phase 7 objectives."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time

import torch

from .config import content_hash
from .initialize import cpu_tree
from .learner import FitSettings, _bounded_solve, prepare_objective
from .metrics import predictive_metrics
from .model import build_model, flat_logits, memory_indices


def scientific_state_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(str((str(array.dtype), list(array.shape))).encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


class FrozenObjectiveAudit:
    def __init__(self, cfg, store, events, *, on_progress=lambda: None):
        self.cfg, self.store = cfg, store
        self.events = tuple(events)
        self.on_progress = on_progress
        self.pending = {}

    def observer(self, context):
        events = [event for event in self.events
                  if context["previous_events"] < event <= context["events"]]
        if not events:
            return None

        def capture(accepted, objective, metadata, diagnostics):
            tick = time.perf_counter()
            point = accepted.detach().clone().requires_grad_()
            value, _ = objective(point)
            gradient = torch.autograd.grad(value, point)[0]
            if not math.isclose(float(value.detach()), diagnostics["final_objective"], rel_tol=1e-9, abs_tol=1e-11):
                raise ValueError("ordinary accepted objective differs from its solve diagnostic")
            snapshot = {key: item for key, item in context.items()
                        if key not in ("settings", "panels")}
            snapshot.update(settings=dataclasses.asdict(context["settings"]),
                            panels=context["panels"], accepted=accepted,
                            original_objective=float(value.detach()),
                            original_gradient=gradient, objective_metadata=metadata)
            snapshot = cpu_tree(snapshot)
            parent_hash = scientific_state_hash(snapshot)
            for event in events:
                unit = f"audit-{context['unit']}-event-{event:03d}"
                meta = dict(schema_version=1, kind="regularization_audit",
                            parent_id=context["unit"], parent_state_hash=parent_hash,
                            replica=context["replica"], stream=context["stream"],
                            treatment=context["treatment"], event=event,
                            policy_iterations=context["settings"].max_iter,
                            step=context["step"])
                if self.store.is_complete(unit):
                    continue
                self.store.begin(unit, meta)
                prior = self.store.load_checkpoint(unit)
                if prior is None:
                    self.store.save_checkpoint(unit, snapshot)
                elif scientific_state_hash(prior) != parent_hash:
                    raise ValueError("recomputed audit objective differs from saved snapshot")
                self.pending.setdefault(context["unit"], set()).add(unit)
            diagnostics["audit_capture_seconds"] = time.perf_counter() - tick
        return capture

    def drain(self, parent, state):
        if parent not in self.pending:
            self.pending[parent] = {
                path.name for path in (self.store.root / "units").glob(f"audit-{parent}-event-*")
                if (path / "manifest.json").exists() and not self.store.is_complete(path.name)}
        for unit in sorted(self.pending[parent]):
            self.store.check()
            snapshot = self.store.load_checkpoint(unit)
            if snapshot is None:
                meta = json.loads((self.store.unit_path(unit) / "manifest.json").read_text())["metadata"]
                if state["step"] < meta["step"]:
                    continue
                raise ValueError("committed parent audit lacks its frozen objective snapshot")
            if state["step"] < snapshot["step"]:
                continue
            self._run(unit, snapshot)
        self.pending[parent] = {unit for unit in self.pending[parent]
                                if not self.store.is_complete(unit)}

    def _run(self, unit, snapshot):
        if self.store.is_complete(unit):
            return
        cfg, store = self.cfg, self.store
        devices = [torch.cuda.current_device()] if cfg.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            tick = time.perf_counter()
            model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
            layout.assert_metadata(snapshot["layout"])
            before = snapshot["before"].to(cfg.device)
            settings = FitSettings(**snapshot["settings"])
            objective, metadata = prepare_objective(
                model, layout, before, snapshot["new_x"].to(before),
                snapshot["new_y"].to(cfg.device), snapshot["replay_x"].to(before),
                snapshot["replay_y"].to(cfg.device), snapshot["ages"].to(cfg.device),
                snapshot["fisher"], snapshot["anchor"], snapshot["treatment"]["condition"],
                settings, snapshot["seed"], store.check)
            accepted = snapshot["accepted"].to(cfg.device)
            point = accepted.detach().clone().requires_grad_()
            value, parts = objective(point)
            gradient = torch.autograd.grad(value, point)[0]
            if (metadata != snapshot["objective_metadata"]
                    or not torch.isclose(value.detach().cpu(), torch.tensor(
                        snapshot["original_objective"], dtype=value.dtype), rtol=1e-9, atol=1e-11)
                    or not torch.allclose(gradient.cpu(), snapshot["original_gradient"],
                                          rtol=1e-7, atol=1e-9)):
                raise ValueError("frozen objective value/gradient reconstruction failed")
            extended_settings = dataclasses.replace(settings,
                max_iter=2 * cfg.inner_steps if cfg.smoke else 200,
                max_eval=2 * cfg.max_evaluations if cfg.smoke else 300)
            extended, diagnostics = _bounded_solve(accepted, objective, extended_settings, store.check)
            if diagnostics["solver_failed"] or diagnostics["final_objective"] is None:
                raise ValueError(f"invalid frozen-objective audit: {diagnostics['solver_reason']}")
            ordinary, after, tv = {}, {}, {}
            with torch.no_grad():
                for domain, (x, y, weights) in snapshot["panels"].items():
                    store.check()
                    x, y = x.to(before), y.to(cfg.device)
                    first = flat_logits(model, layout, accepted, x)
                    second = flat_logits(model, layout, extended, x)
                    ordinary[domain] = predictive_metrics(first, y, weights)
                    after[domain] = predictive_metrics(second, y, weights)
                    distances = .5 * (first.double().softmax(-1) - second.double().softmax(-1)).abs().sum(-1)
                    weight = torch.as_tensor(weights, device=distances.device, dtype=distances.dtype)
                    tv[domain] = float((distances * weight / weight.sum()).sum())
            ltm, stm = (index.to(before.device) for index in memory_indices(layout))
            gain = float(value.detach()) - diagnostics["final_objective"]
            result = dict(schema_version=1, parent_id=snapshot["unit"],
                parent_state_hash=scientific_state_hash(snapshot), step=snapshot["step"],
                ordinary_objective=float(value.detach()), extended_objective=diagnostics["final_objective"],
                objective_gain=gain, relative_objective_gain=gain / max(1., abs(float(value.detach()))),
                original_gradient_inf=float(gradient.abs().max()),
                extended_gradient_inf=diagnostics["gradient_inf_norm"],
                ordinary_parts={key: float(item.detach()) for key, item in parts.items()},
                ltm_motion=float((extended - accepted)[ltm].double().norm()),
                stm_motion=float((extended - accepted)[stm].double().norm()),
                ordinary=ordinary, extended=after, prediction_tv=tv,
                ordinary_history="reset per arrival", extended_history="fresh L-BFGS history",
                diagnostics=diagnostics, seconds=time.perf_counter() - tick,
                scientific_snapshot_hash=content_hash(snapshot["objective_metadata"]))
            store.write_json(unit, "result.json", result)
            store.complete(unit)
            store.heartbeat()
            self.on_progress()
