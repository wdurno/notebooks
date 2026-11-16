"""Reuse the admitted control scheduler with a separately versioned learner."""

import copy

from numerical_experiment.picar_phase3.student import DevelopmentRun
from .config import VERSION


class _ConstructingStore:
    """Give the historical constructor a temporary compatible optimizer view."""

    def __init__(self, store, initial_optimizer):
        self.store = store
        self.initial_optimizer = initial_optimizer
        self.constructing = True
        self.saved = None

    def __getattr__(self, name):
        return getattr(self.store, name)

    def load_checkpoint(self, **kwargs):
        saved = self.store.load_checkpoint(**kwargs)
        if saved is None or not self.constructing:
            return saved
        self.saved = saved
        learner = copy.deepcopy(saved["learner"])
        learner["schema_version"] = 1
        learner["optimizer"] = copy.deepcopy(self.initial_optimizer)
        return {**saved, "learner": learner}


class SolverDevelopmentRun(DevelopmentRun):
    def __init__(self, *args, solver_recipe, **kwargs):
        from .learner import SolverLearner

        positional = list(args)
        store = positional[3] if len(positional) > 3 else kwargs["store"]
        initial = kwargs["initial"]
        config = positional[1] if len(positional) > 1 else kwargs["config"]
        initial_optimizer = copy.deepcopy(initial["learner"]["optimizer"])
        for group in initial_optimizer["param_groups"]:
            group["lr"] = config.learning.learning_rate
        adapter = _ConstructingStore(store, initial_optimizer)
        if len(positional) > 3:
            positional[3] = adapter
        else:
            kwargs["store"] = adapter
        super().__init__(*positional, **kwargs)
        adapter.constructing = False
        resumed = adapter.saved is not None
        learner = SolverLearner(self.model, self.config.learning, self.seeds, recipe=solver_recipe)
        learner.load_state_dict(adapter.saved["learner"] if resumed else initial["learner"])
        if not resumed:
            learner.version = 0
            learner.update_count = 0
            self.model.encoder.publication = 0
            for name, generator in learner.generators.items():
                generator.manual_seed(self.seeds[name])
        self.learner = learner
        self.s["learner"] = learner.state_dict()
        self.s["solver_recipe"] = solver_recipe
        self.s["solver_version"] = VERSION

    def run(self):
        result = super().run()
        return {**result, "kind": "solver_pilot", "solver_recipe": self.s["solver_recipe"],
            "limitations": ["Development only; no fresh significance or sample-savings claim.",
                "Repaired solver; common sealed initialization and primary Fisher.",
                "Actual wall-clock control cadence retained; policy-dependent histories differ."],
            "solver_center_version": self.learner.version}
