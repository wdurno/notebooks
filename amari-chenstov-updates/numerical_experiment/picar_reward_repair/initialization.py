"""Reuse the tested offline numerical kernels, starting at common recorded KL."""

from __future__ import annotations

import copy

from numerical_experiment.picar_initialization.numerics import InitializationFit
from .config import CONDITIONS, reward_version


class _ConditionStore:
    def __init__(self, store, condition):
        self.store, self.condition = store, condition

    def __getattr__(self, name):
        return getattr(self.store, name)

    def checkpoint(self, state, *, metrics=(), progress=None, **kwargs):
        return self.store.checkpoint(state,
            metrics=[{**r, "condition": self.condition} for r in metrics], progress=progress, **kwargs)


def pristine_from_distilled(data):
    state = copy.deepcopy(data["distilled_learner"])
    state["retired_count"] = state["update_count"] = 0
    state["diagonal"] = {n: value.detach().clone().zero_() for n, value in state["diagonal"].items()}
    state["anchor"] = {n: state["parameters"][n].clone() for n in state["anchor"]}
    state["optimizer"]["state"] = {}
    return state


class RewardInitialization(InitializationFit):
    def __init__(self, *args, distill_receipt, shared_kl_selection, **kwargs):
        positional = list(args)
        config = positional[1] if len(positional) > 1 else kwargs["config"]
        store = positional[4] if len(positional) > 4 else kwargs["store"]
        condition = next(c for c in CONDITIONS if reward_version(c) == config.learning.reward_version)
        if len(positional) > 4:
            positional[4] = _ConditionStore(store, condition)
        else:
            kwargs["store"] = _ConditionStore(store, condition)
        super().__init__(*positional, **kwargs)
        if self.s["stage"] == "kl":
            if self.s["curves"] or self.s["optimization_steps"]["kl"]:
                raise ValueError("reward repair cannot repeat teacher distillation")
            self.s["distill_receipt"] = copy.deepcopy(distill_receipt)
            self.s["selection"]["kl"] = copy.deepcopy(shared_kl_selection)
            self.s["distilled_learner"] = self.learner.state_dict()
            self.s["stage"] = "nll"
            self._start_stage()
            self.model.distillation_receipt = copy.deepcopy(distill_receipt)

    def _progress_counts(self):
        counts = super()._progress_counts()
        counts["target"] = len(self.train) * self.policy["nll_max_passes"]
        return counts

    def _summary(self):
        return {**super()._summary(), "shared_KL_reused": True,
            "conditioning": "reward-compatible coded returns; train-only fixed mean baseline; full causal histories",
            "reward_version": self.config.learning.reward_version}
