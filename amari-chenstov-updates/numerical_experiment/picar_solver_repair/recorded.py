"""Transactional short recorded-sequence solver comparisons."""

import copy
import math
import statistics
import time

import torch

from numerical_experiment.picar.fix3_student import _batch, _trim_eligible
from numerical_experiment.picar.learner import LearningBatch
from src.seeding import derive_seed_map
from .config import SCREEN_SEED, VERSION, solver_steps
from .learner import SolverLearner


def evaluate(learner, pools):
    values = {}
    with torch.no_grad():
        for name, rows in pools.items():
            losses = []
            for row in rows:
                batch = LearningBatch([row["history"]], torch.tensor([row["baseline"]]),
                    torch.tensor([row["bin"]]), row["logits"].reshape(1, 7),
                    torch.tensor([row["arrival"]]))
                losses.append(float(learner.nll(learner.params, batch).mean()))
            if not losses or not all(math.isfinite(x) for x in losses):
                raise FloatingPointError("nonfinite or empty recorded validation pool")
            values[name + "_nll"] = statistics.mean(losses)
    return values


def timing(rows):
    measured = [r for r in rows if not r.get("diagnostic_gradient_evaluations", 0)]
    values = sorted(r["update_seconds"] for r in measured)
    if not values:
        return {"events": 0}
    return {"events": len(values), "all_events": len(rows), "fit_seconds": sum(values),
        "diagnostic_fit_seconds": sum(r["update_seconds"] for r in rows if r.get("diagnostic_gradient_evaluations", 0)),
        "timing_policy": "exclude_initial_warmup_and_extra_gradient_diagnostic_events_from_per_event_comparison",
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values), "p95_seconds": values[min(len(values)-1, math.ceil(.95*len(values))-1)],
        "gradient_evaluations": sum(r.get("gradient_evaluations", r.get("optimizer_steps", 1)) for r in measured),
        "gpu_seconds": sum(r.get("gpu_seconds", 0.) or 0. for r in measured),
        "gpu_mean_seconds": statistics.mean(r.get("gpu_seconds", 0.) or 0. for r in measured),
        "peak_vram_bytes": max(r.get("peak_vram_bytes", 0) or 0 for r in rows)}


class RecordedRun:
    def __init__(self, model, config, initial, segments, validation, store, progress,
                 *, solver_recipe, condition):
        self.model, self.config, self.initial = model, config, initial
        self.segments, self.validation = segments, validation
        self.store, self.progress = store, progress
        self.solver, self.condition = solver_recipe, condition
        self.s = store.load_checkpoint(trusted=True) or {"schema_version": 1, "segment_index": 0,
            "cursor": 0, "eligible": [], "rows": [], "validations": [],
            "segment_summaries": [], "active_seconds": 0., "learner": None, "status": "running"}
        self.base_seconds, self.started = self.s["active_seconds"], time.monotonic()
        self.last_save = self.started

    def save(self):
        self.s["active_seconds"] = self.base_seconds + time.monotonic() - self.started
        self.store.checkpoint(self.s, progress={"phase": 3, "version": VERSION,
            "status": "recorded solver comparison", "solver_recipe": self.solver,
            "condition": self.condition, "learning_rate": self.config.learning.learning_rate,
            "completed_events": len(self.s["rows"]),
            "counts": {"step": len(self.s["rows"]), "target": sum(map(len, self.segments.values()))}})
        self.last_save = time.monotonic()
        self.progress()

    def run(self):
        names = list(self.segments)
        for segment_index in range(self.s["segment_index"], len(names)):
            name = names[segment_index]
            rows = self.segments[name]
            seeds = derive_seed_map(SCREEN_SEED + segment_index,
                ("fisher_labels", "fd_probes", "exploration", "replay"))
            learner = SolverLearner(self.model, self.config.learning, seeds, recipe=self.solver)
            if self.s["learner"] is None:
                learner.load_state_dict(self.initial["learner"])
                learner.version = learner.update_count = 0
                for key, generator in learner.generators.items():
                    generator.manual_seed(seeds[key])
                self.s["eligible"] = []
                self.s["validations"].append({"segment": name, "events": 0,
                    "cumulative_fit_seconds": sum(r["update_seconds"] for r in self.s["rows"]),
                    **evaluate(learner, self.validation)})
                self.s["learner"] = learner.state_dict()
                self.save()
            else:
                learner.load_state_dict(self.s["learner"])
            replay_rng = torch.Generator().manual_seed(seeds["replay"])
            if self.s.get("replay_rng") is not None:
                replay_rng.set_state(self.s["replay_rng"])
            for index in range(self.s["cursor"], len(rows)):
                item = copy.deepcopy(rows[index])
                eligible = self.s["eligible"]
                replay_candidates = [r for r in eligible if r["learned"]]
                other = replay_candidates[int(torch.randint(len(replay_candidates), (), generator=replay_rng))] if replay_candidates else None
                eligible.append(item)
                tick = time.monotonic()
                gpu_start = gpu_end = None
                if learner.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(learner.device)
                    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    gpu_start.record()
                try:
                    metrics = learner.fit(new=_batch(item, eligible),
                        replay=_batch(other, eligible) if other is not None else None,
                        acquired_count=item["arrival"] + 1, steps=solver_steps(self.solver),
                        diagnostics=(index == 0 or (index + 1) % 16 == 0))
                    if gpu_end is not None:
                        gpu_end.record()
                        gpu_end.synchronize()
                        metrics.update(gpu_seconds=gpu_start.elapsed_time(gpu_end)/1000,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(learner.device),
                            isolated_completed_seconds=time.monotonic()-tick)
                    item["learned"] = True
                    while len(eligible) > self.config.learning.replay_capacity:
                        learner.retire(_batch(eligible[0], eligible))
                        eligible.pop(0)
                        _trim_eligible(eligible)
                except (FloatingPointError, ValueError) as exc:
                    if not isinstance(exc, FloatingPointError) and "finite" not in str(exc).lower():
                        raise
                    self.s.update(status="numerical_failure", reason=str(exc), learner=learner.state_dict())
                    self.save()
                    return self.summary()
                self.s["rows"].append({**metrics, "event_wall_seconds": time.monotonic() - tick,
                    "segment": name, "event": index + 1, "arrival": item["arrival"],
                    "optimizer_steps": solver_steps(self.solver), "solver_recipe": self.solver,
                    "condition": self.condition, "source_exposure": item["source_exposure"]})
                self.s["cursor"] = index + 1
                if (index + 1) % 16 == 0 or index + 1 == len(rows):
                    self.s["validations"].append({"segment": name, "events": index + 1,
                        "cumulative_fit_seconds": sum(r["update_seconds"] for r in self.s["rows"]),
                        **evaluate(learner, self.validation)})
                if (index + 1) % 8 == 0 or time.monotonic() - self.last_save > 40:
                    self.s.update(learner=learner.state_dict(), replay_rng=replay_rng.get_state())
                    self.save()
                else:
                    self.progress()
            segment_validation = [r for r in self.s["validations"] if r["segment"] == name]
            self.s["segment_summaries"].append({"segment": name, "events": len(rows),
                "validation_initial": segment_validation[0], "validation_final": segment_validation[-1],
                "start_state": "common_pre_stream_asset", "historical_policy_continuation": False})
            self.s.update(segment_index=segment_index + 1, cursor=0, learner=None, replay_rng=None)
            self.save()
        self.s["status"] = "complete"
        self.save()
        return self.summary()

    def summary(self):
        old = [s["validation_initial"]["old_nll"] for s in self.s["segment_summaries"]]
        live = [s["validation_initial"]["live_nll"] for s in self.s["segment_summaries"]]
        final_old = [s["validation_final"]["old_nll"] for s in self.s["segment_summaries"]]
        final_live = [s["validation_final"]["live_nll"] for s in self.s["segment_summaries"]]
        avg = lambda xs: statistics.mean(xs) if xs else None
        return {"status": self.s["status"], "reason": self.s.get("reason"),
            "solver_recipe": self.solver, "condition": self.condition,
            "steps_per_event": solver_steps(self.solver), "completed_events": len(self.s["rows"]),
            "fit_seconds": sum(r["update_seconds"] for r in self.s["rows"]),
            "rows": self.s["rows"], "validation_rows": self.s["validations"],
            "segments": self.s["segment_summaries"], "timing": timing(self.s["rows"]),
            "validation": {"initial": {"old_nll": avg(old), "live_nll": avg(live)},
                "final": {"old_nll": avg(final_old), "live_nll": avg(final_live)}},
            "new_observations": 0, "teacher_calls": 0,
            "limitations": ["Conditional recorded-data diagnostic, not counterfactual live efficacy.",
                "Early and late segments each reset to the same pre-stream initialization.",
                "Validation episodes are disjoint from training and archive admissions."]}


def select_solver(results):
    """Prefer a stable low-cost method among competitive validation outcomes."""
    candidates = []
    for solver in ("proximal_1", "lagged_1", "proximal_3"):
        runs = [r for r in results if r["solver_recipe"] == solver]
        stable = len(runs) == 2 and all(r.get("status") == "complete" for r in runs)
        candidate = {"solver_recipe": solver, "stable": stable, "runs": len(runs)}
        if stable:
            candidate.update(old_gap=statistics.mean(r["validation"]["final"]["old_nll"]-
                r["validation"]["initial"]["old_nll"] for r in runs),
                live_nll=statistics.mean(r["validation"]["final"]["live_nll"] for r in runs),
                mean_seconds=statistics.mean(r["timing"]["mean_seconds"] for r in runs))
        candidates.append(candidate)
    good = [r for r in candidates if r["stable"]]
    if not good:
        return {"selected_solver": None, "reason": "No numerical candidate completed both objectives.", "candidates": candidates}
    # Retention/live prediction are paired diagnostic coordinates, not efficacy tests.
    best_old = min(r["old_gap"] for r in good)
    retained = [r for r in good if r["old_gap"] <= best_old + .1]
    best_live = min(r["live_nll"] for r in retained)
    competitive = [r for r in retained if r["live_nll"] <= best_live + .05]
    selected = min(competitive, key=lambda r: (r["mean_seconds"], r["solver_recipe"] != "proximal_1"))
    return {"selected_solver": selected["solver_recipe"], "candidates": candidates,
        "reason": "Among finite completed methods within 0.1 old-domain NLL of best retention, choose the fastest within 0.05 live NLL of best; review curves and stability before live admission.",
        "thresholds_are_development_guides": True, "statistical_equivalence_claim": False}
