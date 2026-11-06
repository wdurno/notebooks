"""Resumable Phase 7 scheduler; smoke studies belong to implementation acceptance."""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import statistics
import time

import torch

from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from .policies import PolicySpec

from .adapters import BatchReader, stream_view
from .analysis import summarize, summarize_path
from .config import Config, content_hash
from .data import load_cached, make_assets
from .learner import evaluate, fit_batch, mean_nll
from .model import build_model, observed_scores
from .runtime import REPO, file_digest, open_store, read_json
from .trajectory import calibrate, cpu_tree, initialize, run_trajectory


HISTORICAL_ROOT = REPO / "cache/mnist_experiment/plan17/registration_v1"


def pin_historical(store, cfg, historical_root):
    if cfg.smoke:
        return {"report_dir": None, "notebook_sha256": None}
    pointer = store.root / "historical_report.json"
    if pointer.exists():
        result = read_json(pointer)
        if file_digest(Path(result["report_dir"]) / "report.ipynb") != result["notebook_sha256"]:
            raise ValueError("pinned historical notebook changed")
        return result
    latest = read_json(Path(historical_root) / "report/latest.json")
    name = latest.get("generation", latest.get("generation_id"))
    if name is None:
        name = Path(latest["report_dir"]).name
    directory = Path(historical_root) / "report" / name
    if not (directory / "COMPLETED").is_file():
        raise ValueError("historical report must be a completed immutable generation")
    result = {"report_dir": str(directory.resolve()),
              "notebook_sha256": file_digest(directory / "report.ipynb")}
    store.write_json(None, "historical_report.json", result)
    return result


class Scheduler:
    def __init__(self, cfg, store, budget, raw_data, *, max_units=None, historical_root=HISTORICAL_ROOT):
        self.cfg, self.store, self.budget, self.data = cfg, store, budget, raw_data
        self.max_units, self.new_units = max_units, 0
        self.historical = pin_historical(store, cfg, historical_root)
        self.phase = "7A: implementation acceptance" if cfg.smoke else "7B: readiness and timing"
        self.current, self.current_metadata, self.current_metrics = {}, None, []
        self.last_pulse, self.last_report = 0., 0.
        self.complete, self.assets_cache, self.initial_cache, self.calibration_cache = {}, {}, {}, {}
        self.context, self.selection, self.readiness, self.diagnostics = {}, {}, [], {}
        self.frozen_cache = {}
        self.visual_cache = {}
        for record in store.iter_units(completed_only=True):
            uid, meta = record["unit_id"], record["metadata"]
            if meta.get("kind") == "trajectory":
                self.complete[uid] = self.read_trajectory(uid, meta)
            elif meta.get("kind") == "decision":
                payload = read_json(store.unit_path(uid) / "decision.json")
                if meta["name"] == "context":
                    self.context = payload
                elif meta["name"] == "selection":
                    self.selection = payload
            elif meta.get("kind") == "diagnostics":
                self.diagnostics.update(read_json(store.unit_path(uid) / "summary.json"))
            elif meta.get("kind") == "capability":
                self.readiness.append(read_json(store.unit_path(uid) / "summary.json"))

    def count_unit(self):
        self.new_units += 1
        if self.max_units is not None and self.new_units >= self.max_units:
            raise StopRequested("max-units reached after a sealed unit")

    def read_trajectory(self, uid, metadata):
        root = self.store.unit_path(uid)
        return {"unit_id": uid, **metadata, "completed": True,
                "metrics": read_json(root / "metrics.json"), "summary": read_json(root / "summary.json")}

    def decision(self, name, produce):
        uid = "decision-" + name
        path = self.store.begin(uid, {"kind": "decision", "name": name})
        target = self.store.unit_path(uid) / "decision.json"
        if path is None:
            return read_json(target)
        payload = read_json(target) if target.exists() else produce()
        if not target.exists():
            self.store.write_json(uid, "decision.json", payload)
        self.store.complete(uid)
        self.count_unit()
        return payload

    def assets(self, cohort, replica, angle=None, hold_factor=None):
        angle = float(angle if angle is not None else self.cfg.amplitudes[0])
        hold_factor = hold_factor if hold_factor is not None else self.context.get("hold_factor", 1)
        key = cohort, replica, angle, hold_factor
        if key not in self.assets_cache:
            uid = f"assets-{cohort}-r{replica:03}-a{angle:g}-h{hold_factor:g}"
            metadata = {"kind": "assets", "cohort": cohort, "replica": replica,
                        "angle_max": angle, "hold_factor": hold_factor}
            if self.store.begin(uid, metadata) is not None:
                value = make_assets(self.cfg, replica, cohort, self.data, angle=angle, hold_factor=hold_factor)
                self.store.save_checkpoint(uid, value)
                self.store.write_json(uid, "summary.json", value["metadata"])
                self.store.complete(uid)
                self.count_unit()
            value = self.store.load_checkpoint(uid)
            value["unit_id"] = uid
            self.assets_cache[key] = value
        return self.assets_cache[key]

    def reader(self, assets):
        return BatchReader(self.cfg, assets, self.data, self.budget.check)

    def initial(self, cohort, replica, assets=None):
        key = cohort, replica
        if key not in self.initial_cache:
            assets = self.assets(cohort, replica) if assets is None else assets
            plan = assets["panels"]["init"]
            reference = assets["panels"]["reference"]
            metadata = {"kind": "initialization", "cohort": cohort, "replica": replica,
                "model_seed": assets["seeds"]["initialization_model"],
                "order_seed": assets["seeds"]["initialization_order"],
                "initial_plan_hash": content_hash({"ids": plan["ids"].tolist(), "transforms": plan["transforms"].tolist()}),
                "reference_plan_hash": content_hash({"ids": reference["ids"].tolist(), "transforms": reference["transforms"].tolist()}),
                "parameter_count": 448}
            uid = f"initial-{cohort}-r{replica:03}"
            done = self.store.is_complete(uid)
            reader = self.reader(assets)
            if not done:
                reader.preload_initialization()
            result = initialize(self.store, self.budget, self.cfg, uid, metadata, reader, pulse=self.pulse)
            self.initial_cache[key] = result
            if not done:
                self.count_unit()
        return self.initial_cache[key]

    def calibration(self, cohort, replica, assets, initial):
        key = cohort, replica
        if key not in self.calibration_cache:
            uid = f"calibration-{cohort}-r{replica:03}"
            metadata = {"kind": "calibration", "cohort": cohort, "replica": replica,
                        "count": self.cfg.calibration_size, "additional_labels": 0}
            done = self.store.is_complete(uid)
            self.calibration_cache[key] = calibrate(self.store, self.budget, self.cfg, uid, metadata,
                initial, self.reader(assets), pulse=self.pulse)
            if not done:
                self.count_unit()
        return self.calibration_cache[key]

    def pulse(self, *, force=False, report=False, **info):
        if "accepted_metrics" in info:
            self.current_metrics = info.pop("accepted_metrics")
        info.pop("accepted_shadow_metrics", None)
        self.current.update(info)
        now = time.monotonic()
        if not force and now - self.last_pulse < 10:
            return
        self.budget.pulse(force=True)
        fresh = [t for t in self.complete.values() if t["stage"] == "fresh"]
        required = {a["arm"] for a in self.selection.get("arms", [])}
        blocks = {}
        for t in fresh:
            blocks.setdefault((t["schedule"], t["angle_max"], t["replica"]), set()).add(t["arm"])
        planned_blocks = 2 * self.selection.get("main_replicas", 0) + self.selection.get("stationary_replicas", 0)
        progress = {"status": "running", "phase": self.phase, "heartbeat": utc_now(),
            "revision": f"{self.cfg.protocol}/{self.store.root.name}",
            "completed_trajectories": len(self.complete), "completed_fresh": len(fresh),
            "planned_fresh": len(required) * planned_blocks if self.selection else None,
            "completed_blocks": sum(required <= v for v in blocks.values()) if required else 0,
            "planned_blocks": planned_blocks or None,
            "active_hours": self.budget.used / 3600, "phase7_active_hours": self.budget.used / 3600,
            "historical_active_hours": getattr(self.budget, "historical_seconds", 0.) / 3600,
            "total_family_active_hours": (self.budget.used + getattr(self.budget, "historical_seconds", 0.)) / 3600,
            "remaining_hours": self.budget.remaining / 3600,
            "allocation_hours": self.budget.allocation_seconds / 3600,
            "forecast_seconds": self.forecast_remaining() if self.selection else None,
            "storage": self.store.storage_status(), **self.current}
        self.store.write_json(None, "progress.json", progress, finalizing=True)
        self.last_pulse = now
        if report or now - self.last_report >= 900:
            self.publish(progress)

    def publish(self, progress=None, *, status="running"):
        from .report import build_report
        progress = dict(progress or (read_json(self.store.root / "progress.json")
            if (self.store.root / "progress.json").exists() else {}))
        progress.update(status=status, phase=self.phase, heartbeat=utc_now(), active_hours=self.budget.used / 3600,
            phase7_active_hours=self.budget.used / 3600,
            historical_active_hours=getattr(self.budget, "historical_seconds", 0.) / 3600,
            total_family_active_hours=(self.budget.used + getattr(self.budget, "historical_seconds", 0.)) / 3600)
        items = list(self.complete.values())
        if self.current_metadata and self.current_metrics and self.current.get("active_unit") not in self.complete:
            items.append({"unit_id": self.current["active_unit"], **self.current_metadata,
                          "completed": False, "metrics": self.current_metrics, "summary": {}})
        analysis = summarize(items, self.selection, bootstrap_draws=self.cfg.bootstrap_draws,
                             seed=self.selection.get("bootstrap_seed", self.cfg.seed), config=self.cfg.mapping())
        report_config = {**self.cfg.mapping(), "bootstrap_seed": self.cfg.seed}
        snapshot = {"schema": 1, "kind": "plan17-discrepancy-snapshot", "study_id": self.store.root.name,
            "status": status, "phase": self.phase, "config": report_config, "progress": progress,
            "selection": self.selection, "trajectories": items, "analysis": analysis,
            "readiness": {"probes": self.readiness, **self.visuals()}, "diagnostics": self.diagnostics,
            "historical_report_dir": self.historical["report_dir"],
            "historical_notebook_sha256": self.historical["notebook_sha256"]}
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        with self.store.reserve(32 * 1024**2, finalizing=True):
            build_report(self.store.root, self.store.root / "smoke_results.ipynb" if self.cfg.smoke else None)
        self.last_report = time.monotonic()

    def visuals(self):
        result = {"contexts": []}
        for angle in self.cfg.amplitudes:
            assets = next((a for a in self.assets_cache.values() if a["metadata"]["angle_max"] == angle), None)
            if assets is None:
                continue
            key = assets["unit_id"]
            if key not in self.visual_cache:
                v = self.reader(assets)("shifted", torch.tensor([0]))
                anchors = torch.tensor([[-6., -4.], [6., -4.], [0., 8.]])
                self.visual_cache[key] = {"angle_max": angle, "sign": assets["sign"],
                    "reference": v["inputs"][0, 0].cpu().tolist(),
                    "query": v["inputs"][0, 1].cpu().tolist(),
                    "reference_landmarks": anchors.tolist(),
                    "query_landmarks": (anchors + v["targets"][0].cpu().reshape(3, 2)).tolist(),
                    "schedule": [{"arrivals": (i + 1) * self.cfg.samples_per_step, "h": float(h),
                                  "mu": float(assets["streams"]["main"]["mu"][i])}
                                 for i, h in enumerate(assets["streams"]["main"]["h"])],
                    "angle_diagnostics": assets["metadata"]["transform_diagnostics"]}
            result["contexts"].append(self.visual_cache[key])
        return result

    def frozen_reference(self, assets, initial, schedule):
        key = assets["unit_id"], schedule
        if key in self.frozen_cache:
            return self.frozen_cache[key]
        uid = f"frozen-{assets['unit_id']}-{schedule}"
        metadata = {"kind": "frozen_evaluation", "asset_unit": assets["unit_id"], "schedule": schedule}
        if self.store.begin(uid, metadata) is None:
            values = read_json(self.store.unit_path(uid) / "metrics.json")
        else:
            saved = self.store.load_checkpoint(uid) or {"rows": {}}
            values = saved["rows"]
            reader = self.reader(assets)
            model, layout = build_model(0, self.cfg.device, getattr(torch, self.cfg.dtype))
            theta = initial["theta"].to(self.cfg.device)
            segments = assets["streams"][schedule]["segments"]
            count = len(segments)
            checkpoints = [0] + [j for j in range(1, count + 1) if
                j % self.cfg.eval_every == 0 or j == count or segments[j] != segments[j-1]]
            for j in checkpoints:
                if str(j) in values:
                    continue
                self.budget.check()
                v = reader(f"current:{schedule}:{j}")
                values[str(j)] = evaluate(model, layout, theta, v["inputs"], v["targets"],
                                         v["digits"], assets["class_probabilities"], check=self.budget.check)
                self.store.save_checkpoint(uid, {"rows": values})
                self.pulse(active_unit=uid, step=j, total_steps=count)
            self.store.write_json(uid, "metrics.json", values)
            self.store.complete(uid)
            self.count_unit()
        self.frozen_cache[key] = values
        return values

    def trajectory(self, cohort, replica, schedule, arm, condition, ridge, *, angle=None, shadow_specs=()):
        angle = float(angle if angle is not None else self.cfg.amplitudes[0])
        stage = "development" if cohort == "development" else "fresh"
        base = {"kind": "trajectory", "stage": stage, "cohort": cohort, "replica": replica,
            "schedule": schedule, "arm": arm, "condition": condition, "ridge": float(ridge), "c": .025,
            "angle_max": 0. if schedule == "stationary" else angle,
            "shadow_conditions": [x.condition for x in shadow_specs]}
        uid = f"{stage}-{schedule}-a{base['angle_max']:g}-r{replica:03}-{arm}-{content_hash(base)[:10]}"
        if uid in self.complete:
            return self.complete[uid]
        assets = self.assets(cohort, replica, angle)
        initial = self.initial(cohort, replica, assets)
        calibration = self.calibration(cohort, replica, assets, initial) if (
            PolicySpec.parse(condition).movement or shadow_specs) else None
        frozen = self.frozen_reference(assets, initial, schedule)
        metadata = {**base, "sign": assets["sign"], "schedule_factor": assets["schedule_factor"],
            "hold_steps": assets["metadata"]["resolved_hold_steps"],
            "ramp_steps": assets["metadata"]["resolved_ramp_steps"],
            "stream_hash": assets["streams"][schedule]["hash"], "asset_unit": assets["unit_id"],
            "initial_unit": f"initial-{cohort}-r{replica:03}",
            "solver_audit": stage == "development" and arm == "fixed"}
        self.current, self.current_metadata, self.current_metrics = {"active_unit": uid}, metadata, []
        self.pulse(force=True)
        run_trajectory(self.store, self.budget, self.cfg, uid, metadata, initial, self.reader(assets),
            stream_view(assets, schedule), assets["class_probabilities"], calibration=calibration,
            shadow_specs=shadow_specs, frozen_reference=frozen, pulse=self.pulse)
        value = self.complete[uid] = self.read_trajectory(uid, metadata)
        self.current_metadata, self.current_metrics, self.current = None, [], {}
        self.pulse(force=True)
        print({"completed": uid, "current_budget_rms": self._current_area(value)**.5,
               "learning_seconds": value["summary"]["learning_seconds"]}, flush=True)
        self.count_unit()
        return value

    @staticmethod
    def _current_area(value):
        rows = value["metrics"]
        return sum((b["arrivals"]-a["arrivals"]) * (a["current_risk"]+b["current_risk"]) / 2
                   for a,b in zip(rows, rows[1:])) / rows[-1]["arrivals"]

    def path_summary(self, value):
        return summarize_path(value["metrics"], value["hold_steps"], value["ramp_steps"])

    def capability(self, replica, angle):
        uid = f"capability-r{replica:03}-a{angle:g}"
        metadata = {"kind": "capability", "replica": replica, "angle_max": float(angle)}
        assets = self.assets("development", replica, angle)
        initial = self.initial("development", replica, assets)
        if self.store.begin(uid, metadata) is None:
            return read_json(self.store.unit_path(uid) / "summary.json")
        model, layout = build_model(0, device=self.cfg.device, dtype=getattr(torch, self.cfg.dtype))
        reader = self.reader(assets)
        weights = assets["class_probabilities"]
        state = self.store.load_checkpoint(uid)
        if state is None:
            layout.copy_vector_to_module(model, initial["theta"].to(self.cfg.device))
            state = {"epoch": 0, "wall_seconds": 0.}
        else:
            model.load_state_dict(state["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.initial_lr)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

        def measure(role):
            v = reader(role)
            return evaluate(model, layout, layout.flatten_module(model, detach=True),
                            v["inputs"], v["targets"], v["digits"], weights, check=self.budget.check)

        began, base = time.perf_counter(), state["wall_seconds"]
        if "frozen_old" not in state:
            state.update(frozen_old=measure("old"), frozen_shifted=measure("shifted"))
        values = reader("capability")
        for epoch in range(state["epoch"], self.cfg.capability_epochs):
            self.budget.check()
            optimizer.zero_grad(set_to_none=True)
            loss = mean_nll(model(values["inputs"]), values["targets"])
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite capability fit")
            state.update(epoch=epoch + 1, model=cpu_tree(model.state_dict()),
                         optimizer=cpu_tree(optimizer.state_dict()), wall_seconds=base + time.perf_counter() - began)
            self.store.save_checkpoint(uid, state)
        fitted = measure("shifted")
        old, shifted = state["frozen_old"]["risk"], state["frozen_shifted"]["risk"]
        deficit = shifted - old
        recovery = (shifted - fitted["risk"]) / max(shifted, 1e-30)
        result = {**metadata, "frozen_old": state["frozen_old"], "frozen_shifted": state["frozen_shifted"],
            "fitted_shifted": fitted, "positive_deficit": deficit > 0,
            "relative_deficit": deficit / max(old, 1e-30), "recovery_fraction": recovery,
            "recovered_deficit_fraction": (shifted - fitted["risk"]) / deficit if deficit > 0 else None,
            "readiness_flags": deficit >= .05 * old and recovery >= .05,
            "wall_seconds": base + time.perf_counter() - began,
            "additional_training_labels": self.cfg.capability_size}
        self.store.write_json(uid, "summary.json", result)
        self.store.complete(uid)
        self.readiness.append(result)
        self.count_unit()
        return result

    def choose_context(self):
        self.context = self.decision("context", lambda: {
            "amplitudes": list(self.cfg.amplitudes), "hold_factor": 1,
            "reason": "both predeclared excursions retained irrespective of efficacy",
            "readiness": [self.capability(r, angle) for angle in self.cfg.amplitudes
                          for r in range(self.cfg.dev_replicas)]})
        return self.context

    def _rank(self, values):
        return statistics.mean(self._current_area(v) for v in values)

    def audits(self, drivers):
        uid = "diagnostics-score-and-solver"
        metadata = {"kind": "diagnostics", "drivers": [v["unit_id"] for v in drivers]}
        if self.store.begin(uid, metadata) is None:
            self.diagnostics.update(read_json(self.store.unit_path(uid) / "summary.json"))
            return
        saved = self.store.load_checkpoint(uid) or {"solver_audit": [], "anchor_audit": [], "noise_audit": []}
        model, layout = build_model(0, device=self.cfg.device, dtype=getattr(torch, self.cfg.dtype))
        for driver_number, driver in enumerate(drivers):
            frozen = self.store.load_checkpoint(driver["unit_id"])
            probes = frozen.get("audit_states", [])
            # Four common-ridge checks cover nominal, mid-ramp, and turning states.
            position = (0, 1, 1, -1)[driver_number % 4]
            selected_step = probes[min(position, len(probes)-1)]["step"] if probes else None
            for probe in probes:
                key = [driver["unit_id"], probe["step"]]
                if (probe["step"] == selected_step and len(saved["solver_audit"]) < 4 and
                        not any(x["key"] == key for x in saved["solver_audit"])):
                    self.budget.check()
                    candidate, solve = fit_batch(model, layout, probe["theta"].to(self.cfg.device),
                        probe["diagonal"].to(self.cfg.device), probe["inputs"].to(self.cfg.device),
                        probe["targets"].to(self.cfg.device), pi=probe["pi"], ridge=probe["ridge"],
                        max_iter=2*self.cfg.max_iter, max_eval=2*self.cfg.max_eval,
                        learning_rate=self.cfg.learning_rate, history_size=self.cfg.history_size,
                        tolerance_grad=self.cfg.tolerance_grad, tolerance_change=self.cfg.tolerance_change,
                        check=self.budget.check)
                    saved["solver_audit"].append({"key": key,
                        "objective_gap": probe["solve"]["objective"]-solve["objective"],
                        "parameter_disagreement": float((candidate.cpu()-probe["accepted"]).double().norm())})
                    self.store.save_checkpoint(uid, saved)
                if any(x["key"] == key for x in saved["anchor_audit"]):
                    continue
                # Reused arrived examples provide a consistency check, never new evidence.
                x,y = probe["inputs"].to(self.cfg.device), probe["targets"].to(self.cfg.device)
                a,theta = probe["observer_anchor"].to(self.cfg.device), probe["theta"].to(self.cfg.device)
                da = probe["anchor_diagonal"].to(self.cfg.device).double()
                from .model import observation
                dt = observation(model, layout, theta, x, chunk_size=self.cfg.chunk_size, check=self.budget.check)
                sa = observed_scores(model, layout, a, x,y, chunk_size=self.cfg.chunk_size, check=self.budget.check)
                st = observed_scores(model, layout, theta, x,y, chunk_size=self.cfg.chunk_size, check=self.budget.check)
                floor_a = max(self.cfg.observer_floor_absolute,self.cfg.observer_floor_relative*float(da.mean()))
                floor_t = max(self.cfg.observer_floor_absolute,self.cfg.observer_floor_relative*float(dt.mean()))
                residual = sa/da.clamp_min(floor_a)-st/dt.clamp_min(floor_t)-(theta-a).double()
                energy = float((da*residual.mean(0).square()).sum())
                noise = float((da*residual.var(0,unbiased=True)/len(x)).sum())
                saved["anchor_audit"].append({"key":key,"examples":len(x),"additional_labels":0,
                    "residual_energy":energy,"sampling_noise":noise,"ratio":energy/max(noise,1e-30)})
                self.store.save_checkpoint(uid, saved)
            r = driver["replica"]
            if any(x["replica"] == r for x in saved["noise_audit"]):
                continue
            assets=self.assets("development",r,driver["angle_max"])
            initial=self.initial("development",r,assets)
            calibration=self.calibration("development",r,assets,initial)
            v=self.reader(assets)("calibration",torch.arange(max(2,self.cfg.calibration_size//2)))
            scores=observed_scores(model,layout,initial["theta"].to(self.cfg.device),v["inputs"],v["targets"],
                                  chunk_size=self.cfg.chunk_size,check=self.budget.check)
            full=calibration["variance"].to(self.cfg.device)
            da=initial["diagonal"].to(self.cfg.device)
            floor=max(self.cfg.observer_floor_absolute,self.cfg.observer_floor_relative*float(da.mean()))
            weight=da/da.clamp_min(floor).square()
            nested=scores.var(0,unbiased=True)
            saved["noise_audit"].append({"replica":r,"additional_labels":0,"reference_is_truth":False,
                "trace_ratio":float(nested.sum()/full.sum()),
                "weighted_ratio":float((weight*nested).sum()/(weight*full).sum())})
            self.store.save_checkpoint(uid,saved)
        self.store.write_json(uid,"summary.json",saved)
        self.store.complete(uid)
        self.diagnostics.update(saved)
        self.count_unit()

    def develop(self):
        self.phase="7B: readiness, common ridge and timing"
        self.pulse(force=True,report=True)
        self.choose_context()
        shadows=[PolicySpec("velocity",eta) for eta in self.cfg.process_scales]
        fixed={}
        for ridge in reversed(self.cfg.ridge_candidates):
            fixed[ridge]=[self.trajectory("development",r,"main","fixed","fixed",ridge,
                angle=angle,shadow_specs=shadows) for angle in self.cfg.amplitudes
                for r in range(self.cfg.dev_replicas)]
            self.pulse(force=True,report=True)
        common=self.decision("common-ridge",lambda:{"ridge":min(fixed,
            key=lambda ridge:(self._rank(fixed[ridge]),
                statistics.mean(v["summary"]["learning_seconds"] for v in fixed[ridge]),ridge))})["ridge"]
        self.audits(fixed[common])
        self.phase="7C: observer calibration and frozen design"
        ranking=[]
        for spec in shadows:
            values=[]
            for driver in fixed[common]:
                rows=read_json(self.store.unit_path(driver["unit_id"])/"shadow_metrics.json")[spec.condition]
                segments=sorted({row["segment"] for row in rows})
                values.append(statistics.mean(statistics.mean(row["innovation_log_score"] for row in rows
                    if row["segment"]==segment) for segment in segments))
            ranking.append((spec.value,statistics.mean(values)))
        candidates=self.decision("observer-candidates",lambda:{"etas":[eta for eta,score in
            sorted(ranking,key=lambda x:(-x[1],x[0]))[:self.cfg.candidates_per_family]],"ranking":ranking})
        best={}
        for family in ("velocity","discrepancy"):
            outcomes={}
            for eta in candidates["etas"]:
                condition=PolicySpec(family,eta).condition
                outcomes[condition]=[self.trajectory("development",r,"main",family,condition,common,angle=angle)
                    for angle in self.cfg.amplitudes for r in range(self.cfg.dev_replicas)]
                self.pulse(force=True,report=True)
            best[family]=min(outcomes,key=lambda c:(self._rank(outcomes[c]),
                statistics.mean(v["summary"]["learning_seconds"] for v in outcomes[c]),PolicySpec.parse(c).value))
        for angle in self.cfg.amplitudes:
            for r in range(self.cfg.dev_replicas):
                self.trajectory("development",r,"main","covariance","covariance",common,angle=angle)
        arms=[{"arm":"fixed","condition":"fixed","ridge":common},
              {"arm":"covariance","condition":"covariance","ridge":common},
              {"arm":"velocity","condition":best["velocity"],"ridge":common},
              {"arm":"discrepancy","condition":best["discrepancy"],"ridge":common}]
        for r in range(self.cfg.dev_replicas):
            for arm in arms:
                self.trajectory("development",r,"stationary",arm["arm"],arm["condition"],common)
        self.diagnostics.update(shadow_ranking=ranking)
        def select():
            nmain,nstationary=self.cfg.main_replicas,self.cfg.stationary_replicas
            per_path=statistics.median(v["summary"]["online_wall_seconds"] for v in self.complete.values())
            per_init=statistics.median(x["wall_seconds"] for x in self.initial_cache.values())
            def forecast(n):
                return 4*(2*n+nstationary)*per_path*1.25+(n+nstationary)*(per_init+180)
            estimate=forecast(nmain)
            if not self.cfg.smoke and 1.25*estimate+3600>self.budget.remaining:
                nmain=min(nmain,4)
                estimate=forecast(nmain)
            if not self.cfg.smoke and 1.25*estimate+3600>self.budget.remaining:
                raise ResourceLimit("remaining Phase 7 allowance cannot admit balanced fresh cohort")
            return {"frozen":True,"frozen_at":utc_now(),"common_ridge":common,
                "velocity_condition":best["velocity"],"discrepancy_condition":best["discrepancy"],
                "arms":arms,"main_replicas":nmain,"stationary_replicas":nstationary,
                "amplitudes":list(self.cfg.amplitudes),"hold_steps":self.cfg.hold_steps,
                "ramp_steps":self.cfg.ramp_steps,"forecast_seconds":estimate,
                "per_path_seconds":per_path,"per_initial_seconds":per_init,
                "bootstrap_seed":self.cfg.seed_for("fresh",0,"bootstrap"),"shadow_ranking":ranking,
                "observer_candidates":candidates}
        self.selection=self.decision("selection",select)
        self.store.write_json(None,"selection.json",self.selection)
        self.pulse(force=True,report=True)

    def forecast_remaining(self):
        if not self.selection:
            return None
        done = sum(v["stage"] == "fresh" for v in self.complete.values())
        target = len(self.selection["arms"]) * (2*self.selection["main_replicas"] + self.selection["stationary_replicas"])
        return max(0, target - done) * self.selection["per_path_seconds"] * 1.25

    def run(self):
        self.pulse(force=True,report=True)
        if not self.selection:
            self.develop()
        self.phase="7D: fresh paired exploration" if not self.cfg.smoke else "7A: CPU pipeline acceptance"
        allocations=[("main",r,angle) for r in range(self.selection["main_replicas"])
                     for angle in self.cfg.amplitudes]
        allocations += [("stationary",r,self.cfg.amplitudes[0])
                        for r in range(self.selection["stationary_replicas"])]
        allocations.sort(key=lambda x:(x[1],x[0]!="main",x[2]))
        for schedule,replica,angle in allocations:
            cohort="fresh" if schedule=="main" else "stationary"
            arms=self.selection["arms"]
            for arm in arms[replica%len(arms):]+arms[:replica%len(arms)]:
                self.trajectory(cohort,replica,schedule,arm["arm"],arm["condition"],arm["ridge"],angle=angle)
            self.pulse(force=True,report=True)
        self.phase="7E: interpretation and closeout"
        self.pulse(force=True,report=True)
        self.publish(status="completed")
        self.store.write_json(None,"study_completed.json",{"completed_at":utc_now(),
            "phase7_active_hours":self.budget.used/3600,"selection":self.selection})
        print({"status":"completed","phase7_active_hours":self.budget.used/3600},flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO / "cache/mnist_experiment/datasets")
    parser.add_argument("--allocation-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config.from_json(args.config)
    store, budget = open_store(cfg, args.root, args.data_root, resume=args.resume, clear_stop=args.clear_stop,
        allocation_hours=args.allocation_hours, max_wall_seconds=args.max_wall_seconds)
    scheduler = None
    try:
        with store.control.signal_handlers():
            if (store.root / "study_completed.json").exists():
                print({"status": "already_completed", "phase7_active_hours": budget.used / 3600}, flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, load_cached(args.data_root),
                                  max_units=args.max_units, historical_root=args.historical_root)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler is not None:
            scheduler.publish(status="paused")
        print({"status": "paused", "reason": str(error), "phase7_active_hours": budget.used / 3600}, flush=True)
        return 0
    finally:
        budget.close()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

