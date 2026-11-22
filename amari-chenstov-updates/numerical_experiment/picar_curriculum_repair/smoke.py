"""Bounded CPU simulator smoke, explicitly not a Qwen capability experiment."""

from dataclasses import replace

import torch

from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from .config import configuration, simulation
from .student import CurriculumRun


class SmokeScorer:
    provenance = {"kind": "constant_score_cpu_pipeline_smoke_only"}

    def state_dict(self):
        return dict(self.provenance)

    def load_state_dict(self, state):
        if state != self.provenance:
            raise ValueError("smoke scorer identity differs")

    def score_rgb(self, rgb):
        return {"score": 1., "width_percent": 8., "raw_text": "CPU smoke only"}

    def score(self, history, pan):
        return self.score_rgb(None)


def run(store, stopped):
    torch.set_num_threads(2)
    cfg = configuration(strength=.1)
    cfg = replace(cfg, image_width=32, image_height=24,
        model=replace(cfg.model, hidden_size=4, context_length=2),
        learning=replace(cfg.learning, replay_capacity=2))
    model = build_tiny_model(198001, hidden_size=4, width=4, return_bins=161, context_length=2).eval()
    learner = MemoryLearner(model, cfg.learning, cfg.seeds(0))
    state = learner.state_dict()
    state["diagonal"] = {n: v+1 for n, v in state["diagonal"].items()}
    state["retired_count"] = 2
    initial = {"stage": "done", "learner": state, "eligible": [], "retired": [0, 1], "acquired_count": 2,
        "distill_receipt": {"kind": "synthetic_cpu_smoke_not_scientific_initialization"}, "distinct_bootstrap_samples": 2}
    sim = replace(simulation(), width=32, height=24, deadline_s=1., control_interval_s=.25,
        total_training_intervals=12, curriculum_total_intervals=12)
    result = CurriculumRun(model, cfg, sim, store, stopped, scorer=SmokeScorer(), seed=198001,
        max_transitions=12, initial=initial, initial_asset_id="synthetic_smoke", checkpoints=(0, 5, 12),
        evaluations=[]).run()
    if (result["acquired_count"] < 12 or not result["fit_metrics"] or result["fd_evaluations"] <= 0
            or result["first_step_ewc_events"] != len(result["fit_metrics"])):
        raise ValueError("CPU simulator smoke did not exercise learning, FD and first-step EWC")
    return {"passed": True, "smoke_only": True, "new_scientific_observations": 0,
        "acquired_count": result["acquired_count"], "fd_evaluations": result["fd_evaluations"],
        "events": len(result["fit_metrics"]), "first_step_ewc_events": result["first_step_ewc_events"]}
