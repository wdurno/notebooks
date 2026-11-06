"""Resumable Phase 6 scheduler; smoke studies belong to implementation acceptance."""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import statistics
import time

import torch

from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.kalman_pi.policies import PolicySpec

from .adapters import BatchReader, stream_view
from .analysis import summarize, summarize_path
from .config import Config, content_hash
from .data import load_cached, make_assets
from .learner import evaluate, fit_batch, mean_nll
from .model import build_model, observed_scores
from .runtime import REPO, file_digest, open_store, read_json
from .trajectory import calibrate, cpu_tree, initialize, run_trajectory


HISTORICAL_ROOT = REPO / "cache/mnist_experiment/plan17/retention_supplement_v1"


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
        self.phase = "6A: implementation acceptance" if cfg.smoke else "6B: readiness and timing"
        self.current, self.current_metadata, self.current_metrics = {}, None, []
        self.last_pulse, self.last_report = 0., 0.
        self.complete, self.assets_cache, self.initial_cache, self.calibration_cache = {}, {}, {}, {}
        self.context, self.selection, self.readiness, self.diagnostics = {}, {}, [], {}
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
        angle = float(angle if angle is not None else self.context.get("angle_max", self.cfg.angle_max))
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
            blocks.setdefault((t["schedule"], t["replica"]), set()).add(t["arm"])
        planned_blocks = self.selection.get("main_replicas", 0) + self.selection.get("stationary_replicas", 0)
        progress = {"status": "running", "phase": self.phase, "heartbeat": utc_now(),
            "revision": f"{self.cfg.protocol}/{self.store.root.name}",
            "completed_trajectories": len(self.complete), "completed_fresh": len(fresh),
            "planned_fresh": len(required) * planned_blocks if self.selection else None,
            "completed_blocks": sum(required <= v for v in blocks.values()) if required else 0,
            "planned_blocks": planned_blocks or None,
            "active_hours": self.budget.used / 3600, "phase6_active_hours": self.budget.used / 3600,
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
            phase6_active_hours=self.budget.used / 3600,
            historical_active_hours=getattr(self.budget, "historical_seconds", 0.) / 3600,
            total_family_active_hours=(self.budget.used + getattr(self.budget, "historical_seconds", 0.)) / 3600)
        items = list(self.complete.values())
        if self.current_metadata and self.current_metrics and self.current.get("active_unit") not in self.complete:
            items.append({"unit_id": self.current["active_unit"], **self.current_metadata,
                          "completed": False, "metrics": self.current_metrics, "summary": {}})
        analysis = summarize(items, self.selection, bootstrap_draws=self.cfg.bootstrap_draws,
                             seed=self.selection.get("bootstrap_seed", self.cfg.seed), config=self.cfg.mapping())
        report_config = {**self.cfg.mapping(), "bootstrap_seed": self.cfg.seed}
        snapshot = {"schema": 1, "kind": "plan17-registration-snapshot", "study_id": self.store.root.name,
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
        angle = float(self.context.get("angle_max", self.cfg.angle_max))
        available = [a for a in self.assets_cache.values() if a["metadata"]["angle_max"] == angle]
        if not available:
            return {}
        assets = available[0]
        key = assets["unit_id"]
        if key not in self.visual_cache:
            values = self.reader(assets)("shifted", torch.tensor([0]))
            anchors = torch.tensor([[-6., -4.], [6., -4.], [0., 8.]])
            target = values["targets"][0].detach().cpu().reshape(3, 2)
            edges = torch.linspace(-50., 50., 51, dtype=torch.float64)
            centers = (edges[:-1] + edges[1:]) / 2
            proposed, accepted, acceptance = {}, {}, {}
            for role, strength in (("diagnostic_nominal", 0.), ("diagnostic_mid", .5), ("diagnostic_shifted", 1.)):
                label = f"h={strength:g}"
                mu = assets["sign"] * angle * strength
                normalizer = .5 * (math.erf((50 - mu) / (6 * math.sqrt(2))) - math.erf((-50 - mu) / (6 * math.sqrt(2))))
                proposed[label] = ((-(centers - mu).square() / 72).exp() / (6 * math.sqrt(2 * math.pi) * normalizer)).tolist()
                draws = assets["panels"][role]["transforms"][:, 0]
                accepted[label] = (torch.histogram(draws.double(), bins=edges).hist / (len(draws) * 2)).tolist()
                info = assets["metadata"]["transform_diagnostics"][role]
                acceptance[label] = info["accepted"] / info["proposed"]
            self.visual_cache[key] = {"examples": [{"reference": values["inputs"][0, 0].cpu().tolist(),
                "query": values["inputs"][0, 1].cpu().tolist(), "reference_landmarks": anchors.tolist(),
                "query_landmarks": (anchors + target).tolist()}],
                "angle_law": {"angles": centers.tolist(), "proposed": proposed, "accepted": accepted,
                              "acceptance_fraction": acceptance},
                "schedule": [{"step": i, "h": float(value)} for i, value in enumerate(assets["streams"]["main"]["h"])]}
        return self.visual_cache[key]

    def trajectory(self, cohort, replica, schedule, arm, condition, ridge, *, shadow_specs=()):
        stage = "development" if cohort == "development" else "fresh"
        base = {"kind": "trajectory", "stage": stage, "cohort": cohort, "replica": replica,
            "schedule": schedule, "arm": arm, "condition": condition, "ridge": float(ridge), "c": .025}
        uid = f"{stage}-{schedule}-r{replica:03}-{arm}-{content_hash(base)[:10]}"
        if uid in self.complete:
            return self.complete[uid]
        assets = self.assets(cohort, replica)
        initial = self.initial(cohort, replica, assets)
        calibration = self.calibration(cohort, replica, assets, initial) if (
            PolicySpec.parse(condition).movement or shadow_specs) else None
        metadata = {**base, "sign": assets["sign"], "schedule_factor": assets["schedule_factor"],
            "hold_steps": assets["metadata"]["resolved_hold_steps"],
            "ramp_steps": assets["metadata"]["resolved_ramp_steps"],
            "stream_hash": assets["streams"][schedule]["hash"], "asset_unit": assets["unit_id"],
            "initial_unit": f"initial-{cohort}-r{replica:03}",
            "shadow_conditions": [s.condition for s in shadow_specs],
            "solver_audit": stage == "development" and arm == "fixed" and ridge == max(self.cfg.ridge_candidates)}
        self.current, self.current_metadata, self.current_metrics = {"active_unit": uid}, metadata, []
        self.pulse(force=True)
        run_trajectory(self.store, self.budget, self.cfg, uid, metadata, initial, self.reader(assets),
            stream_view(assets, schedule), assets["class_probabilities"], calibration=calibration,
            shadow_specs=shadow_specs, pulse=self.pulse)
        value = self.complete[uid] = self.read_trajectory(uid, metadata)
        self.current_metadata, self.current_metrics = None, []
        self.current = {}
        self.pulse(force=True)
        print({"completed": uid, **summarize_path(value["metrics"], metadata["hold_steps"], metadata["ramp_steps"])}, flush=True)
        self.count_unit()
        return value

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
        def choose():
            default = [self.capability(r, self.cfg.angle_max) for r in range(self.cfg.dev_replicas)]
            if not default:
                return {"angle_max": self.cfg.angle_max, "hold_factor": 1, "weak_opportunity": True}
            candidate_angles = [self.cfg.angle_max]
            if not all(x["readiness_flags"] for x in default):
                alternate = 35. if statistics.mean(x["relative_deficit"] for x in default) < .05 else 15.
                candidate_angles.append(alternate)
                for r in range(self.cfg.dev_replicas):
                    self.capability(r, alternate)
            candidates = []
            for angle in candidate_angles:
                rows = [x for x in self.readiness if x["angle_max"] == angle]
                flags = all(x["readiness_flags"] for x in rows)
                positive = [x["recovered_deficit_fraction"] for x in rows if x["positive_deficit"]]
                recovery = statistics.mean(positive) if positive else None
                candidates.append((angle, flags, recovery, any(x["positive_deficit"] for x in rows)))
            passing = [x for x in candidates if x[1]]
            selected = passing[0] if passing else max(candidates, key=lambda x: (x[3], x[2] if x[2] is not None else 0.))
            if not any(x[3] for x in candidates):
                selected = candidates[0]
            return {"angle_max": selected[0], "hold_factor": 1, "weak_opportunity": not bool(passing),
                    "reason": "development task readiness before policy screening", "candidates": candidates}
        self.context = self.decision("context", choose)
        return self.context

    def _rank(self, values):
        return statistics.mean(self.path_summary(v)["normalized_score"] for v in values)

    def audits(self, drivers):
        uid = "diagnostics-score-and-solver"
        metadata = {"kind": "diagnostics", "drivers": [v["unit_id"] for v in drivers]}
        if self.store.begin(uid, metadata) is None:
            self.diagnostics.update(read_json(self.store.unit_path(uid) / "summary.json"))
            return
        model, layout = build_model(0, device=self.cfg.device, dtype=getattr(torch, self.cfg.dtype))
        results = {"solver_audit": [], "score_audit": []}
        for driver in drivers:
            self.budget.check()
            frozen = self.store.load_checkpoint(driver["unit_id"])
            for probe in frozen.get("audit_states", []):
                candidate, solve = fit_batch(model, layout, probe["theta"].to(self.cfg.device),
                    probe["diagonal"].to(self.cfg.device), probe["inputs"].to(self.cfg.device),
                    probe["targets"].to(self.cfg.device), pi=probe["pi"], ridge=probe["ridge"],
                    max_iter=2 * self.cfg.max_iter, max_eval=2 * self.cfg.max_eval,
                    learning_rate=self.cfg.learning_rate, history_size=self.cfg.history_size,
                    tolerance_grad=self.cfg.tolerance_grad, tolerance_change=self.cfg.tolerance_change,
                    check=self.budget.check)
                results["solver_audit"].append({"driver": driver["unit_id"], "step": probe["step"],
                    "original": probe["solve"], "doubled": solve,
                    "objective_gap": probe["solve"]["objective"] - solve["objective"],
                    "parameter_disagreement": float((candidate.cpu() - probe["accepted"]).double().norm())})
            assets = self.assets("development", driver["replica"])
            initial = self.initial("development", driver["replica"], assets)
            reader = self.reader(assets)
            for role in ("diagnostic_nominal", "diagnostic_mid", "diagnostic_shifted"):
                values = reader(role)
                scores = observed_scores(model, layout, initial["theta"].to(self.cfg.device),
                    values["inputs"], values["targets"], chunk_size=self.cfg.chunk_size, check=self.budget.check)
                results["score_audit"].append({"replica": driver["replica"], "role": role,
                    "labels": len(scores), "mean_score": scores.mean(0).cpu().tolist(),
                    "variance": scores.var(0, unbiased=True).cpu().tolist(),
                    "mean_norm": float(scores.mean(0).norm()), "within_score_variance": float(scores.var(0).sum())})
        self.store.write_json(uid, "summary.json", results)
        self.store.complete(uid)
        self.diagnostics.update(results)
        self.count_unit()

    def develop(self):
        self.phase = "6B: task readiness and timing"
        self.pulse(force=True, report=True)
        self.choose_context()
        shadows = [PolicySpec("ema", h) for h in self.cfg.ema_half_lives]
        shadows += [PolicySpec("kalman", eta) for eta in self.cfg.process_scales]
        fixed = {}
        for ridge in reversed(self.cfg.ridge_candidates):
            fixed[ridge] = [self.trajectory("development", r, "main", "fixed", "fixed", ridge,
                shadow_specs=shadows if ridge == max(self.cfg.ridge_candidates) else ())
                for r in range(self.cfg.dev_replicas)]
            self.pulse(force=True, report=True)
        self.audits(fixed[max(self.cfg.ridge_candidates)])
        common = self.decision("common-ridge", lambda: {"ridge": min(fixed,
            key=lambda r: (self._rank(fixed[r]), statistics.mean(v["summary"]["learning_seconds"] for v in fixed[r]), r))})["ridge"]
        self.phase = "6C: observer and retention development"
        ranking = {"ema": [], "kalman": []}
        for spec in shadows:
            scores = []
            for driver in fixed[max(self.cfg.ridge_candidates)]:
                rows = read_json(self.store.unit_path(driver["unit_id"]) / "shadow_metrics.json")[spec.condition]
                means = [statistics.mean(x["innovation_log_score"] for x in rows if x["segment"] == segment)
                         for segment in sorted({x["segment"] for x in rows})]
                scores.append(statistics.mean(means))
            ranking[spec.kind].append((spec.condition, statistics.mean(scores)))
        candidates = self.decision("observer-candidates", lambda: {
            family: [c for c, score in sorted(rows, key=lambda x: (-x[1], x[0]))[:self.cfg.candidates_per_family]]
            for family, rows in ranking.items()})
        best = {}
        for family in ("ema", "kalman"):
            outcomes = {}
            for condition in candidates[family]:
                outcomes[condition] = [self.trajectory("development", r, "main", family, condition, common)
                                       for r in range(self.cfg.dev_replicas)]
            best[family] = min(outcomes, key=lambda c: (self._rank(outcomes[c]), c))
        for r in range(self.cfg.dev_replicas):
            self.trajectory("development", r, "main", "covariance", "covariance", common)
        references = fixed[common]
        kalman = [self.trajectory("development", r, "main", "kalman", best["kalman"], common)
                  for r in range(self.cfg.dev_replicas)]
        gap = statistics.mean(self.path_summary(k)["old_rms"] - self.path_summary(f)["old_rms"]
                              for k, f in zip(kalman, references, strict=True))
        acquisition = statistics.mean(self.path_summary(k)["Ashifted"] - self.path_summary(f)["Ashifted"]
                                      for k, f in zip(kalman, references, strict=True))
        calibrated = common
        retention = [{"ridge": common, "old_gap": gap, "acquisition_gap": acquisition,
                      "learning_seconds": statistics.mean(v["summary"]["learning_seconds"] for v in kalman)}]
        if abs(gap) > self.cfg.retention_tolerance_pixels and not (gap < 0 and acquisition < 0):
            grid = list(self.cfg.retention_ridges)
            index = grid.index(common)
            alternatives = [grid[index - 1], grid[index + 1]] if index else [.01, .1]
            for ridge in alternatives:
                rows = [self.trajectory("development", r, "main", "kalman_matched", best["kalman"], ridge)
                        for r in range(self.cfg.dev_replicas)]
                retention.append({"ridge": ridge,
                    "learning_seconds": statistics.mean(v["summary"]["learning_seconds"] for v in rows),
                    "old_gap": statistics.mean(self.path_summary(k)["old_rms"] - self.path_summary(f)["old_rms"]
                        for k, f in zip(rows, references, strict=True)),
                    "acquisition_gap": statistics.mean(self.path_summary(k)["Ashifted"] - self.path_summary(f)["Ashifted"]
                        for k, f in zip(rows, references, strict=True))})
            eligible = [x for x in retention if abs(x["old_gap"]) <= self.cfg.retention_tolerance_pixels]
            calibrated = min(eligible, key=lambda x: (x["acquisition_gap"], abs(x["old_gap"]), x["learning_seconds"], x["ridge"]))["ridge"] if eligible else min(
                retention, key=lambda x: (abs(x["old_gap"]), x["acquisition_gap"], x["ridge"]))["ridge"]
        arms = [{"arm": "fixed", "condition": "fixed", "ridge": common},
                {"arm": "covariance", "condition": "covariance", "ridge": common},
                {"arm": "ema", "condition": best["ema"], "ridge": common},
                {"arm": "kalman", "condition": best["kalman"], "ridge": common}]
        if calibrated != common:
            arms += [{"arm": "kalman_matched", "condition": best["kalman"], "ridge": calibrated},
                     {"arm": "fixed_matched", "condition": "fixed", "ridge": calibrated}]
        self.diagnostics.update(shadow_ranking=ranking, retention_candidates=retention)

        def select():
            nmain, nstationary = self.cfg.main_replicas, self.cfg.stationary_replicas
            per_path = statistics.median(v["summary"]["online_wall_seconds"] for v in self.complete.values())
            per_init = statistics.median(x["wall_seconds"] for x in self.initial_cache.values())
            forecast = len(arms) * (nmain + nstationary) * per_path * 1.25 + (nmain + nstationary) * (per_init + 30)
            if not self.cfg.smoke and 1.25 * forecast + 3600 > self.budget.remaining:
                nmain = min(nmain, 4)
                nstationary = min(nstationary, 4)
                forecast = len(arms) * (nmain + nstationary) * per_path * 1.25 + (nmain + nstationary) * (per_init + 30)
            if not self.cfg.smoke and 1.25 * forecast + 3600 > self.budget.remaining:
                raise ResourceLimit("remaining Phase 6 allocation cannot admit the balanced fresh cohort")
            return {"frozen": True, "frozen_at": utc_now(), "common_ridge": common,
                "ema_condition": best["ema"], "kalman_condition": best["kalman"],
                "calibrated_ridge": calibrated, "arms": arms, "main_replicas": nmain,
                "stationary_replicas": nstationary, "angle_max": self.context["angle_max"],
                "hold_factor": self.context["hold_factor"], "hold_steps": self.cfg.hold_steps,
                "ramp_steps": self.cfg.ramp_steps, "forecast_seconds": forecast,
                "per_path_seconds": per_path, "per_initial_seconds": per_init,
                "bootstrap_seed": self.cfg.seed_for("fresh", 0, "bootstrap"),
                "retention_candidates": retention, "shadow_ranking": ranking}
        self.selection = self.decision("selection", select)
        self.store.write_json(None, "selection.json", self.selection)
        self.pulse(force=True, report=True)

    def forecast_remaining(self):
        if not self.selection:
            return None
        done = sum(v["stage"] == "fresh" for v in self.complete.values())
        target = len(self.selection["arms"]) * (self.selection["main_replicas"] + self.selection["stationary_replicas"])
        return max(0, target - done) * self.selection["per_path_seconds"] * 1.25

    def run(self):
        self.pulse(force=True, report=True)
        if not self.selection:
            self.develop()
        self.phase = "6D: fresh paired exploration" if not self.cfg.smoke else "6A: CPU pipeline acceptance"
        allocations = [("main", r) for r in range(self.selection["main_replicas"])]
        allocations += [("stationary", r) for r in range(self.selection["stationary_replicas"])]
        allocations.sort(key=lambda v: (v[1], v[0] != "main"))
        for schedule, replica in allocations:
            cohort = "fresh" if schedule == "main" else "stationary"
            arms = self.selection["arms"]
            for arm in arms[replica % len(arms):] + arms[:replica % len(arms)]:
                self.trajectory(cohort, replica, schedule, arm["arm"], arm["condition"], arm["ridge"])
            self.pulse(force=True, report=True)
        self.phase = "6E: interpretation and closeout"
        self.pulse(force=True, report=True)
        self.publish(status="completed")
        self.store.write_json(None, "study_completed.json", {"completed_at": utc_now(),
            "phase6_active_hours": self.budget.used / 3600, "selection": self.selection})
        print({"status": "completed", "phase6_active_hours": self.budget.used / 3600}, flush=True)


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
                print({"status": "already_completed", "phase6_active_hours": budget.used / 3600}, flush=True)
                return 0
            scheduler = Scheduler(cfg, store, budget, load_cached(args.data_root),
                                  max_units=args.max_units, historical_root=args.historical_root)
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler is not None:
            scheduler.publish(status="paused")
        print({"status": "paused", "reason": str(error), "phase6_active_hours": budget.used / 3600}, flush=True)
        return 0
    finally:
        budget.close()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
