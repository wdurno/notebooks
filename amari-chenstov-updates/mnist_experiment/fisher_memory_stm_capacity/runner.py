"""Reuse the frozen grid engine with explicit chart and artifact identities."""

import copy
from contextlib import contextmanager
import json

import torch

from mnist_experiment.fisher_memory_grid_registration import runner as grid
from mnist_experiment.fisher_memory_regression.runtime import read_json
from mnist_experiment.fisher_memory_regression.config import content_hash
from . import PROTOCOL
from .config import arm
from . import model

_trajectory_id = grid.trajectory_id


def trajectory_id(chart, family, replica, recipe, target):
    return f"{chart}-" + _trajectory_id(family, replica, recipe, target)


class ChartStore:
    """Enrich artifacts before sealing; never amend a completed engine artifact."""

    def __init__(self, store):
        self.base = store
        self.chart_identity = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def begin(self, unit, metadata):
        if metadata.get("kind") == "grid_registration_trajectory":
            metadata = {**metadata, "stm_config": model.CHART.get(),
                        "phase_protocol": PROTOCOL, **self.chart_identity}
        elif metadata.get("kind") in ("grid_registration_assets","stm_capacity_assets"):
            metadata = {**metadata, **self.chart_identity}
        return self.base.begin(unit, metadata)

    def write_json(self, unit, filename, value, **kwargs):
        if filename == "summary.json":
            value = {**value, "stm_config": model.CHART.get(),
                     "arm": arm(model.CHART.get(), value["method"]), "protocol": PROTOCOL,
                     **self.chart_identity}
        return self.base.write_json(unit, filename, value, **kwargs)

    def save_checkpoint(self, unit, state, **kwargs):
        state = {**state, "stm_config": model.CHART.get()}
        fits = state.get("fit_records")
        if fits and state.get("learner", {}).get("archive_diagnostics"):
            fits[-1] = {**fits[-1], "archive_compression": state["learner"][
                "archive_diagnostics"].get("compression", {})}
            state["fit_records"] = fits
        return self.base.save_checkpoint(unit, state, **kwargs)

    def load_checkpoint(self, unit, **kwargs):
        state = self.base.load_checkpoint(unit, **kwargs)
        if state is not None and state.get("stm_config", model.CHART.get()) != model.CHART.get():
            raise ValueError("checkpoint STM chart differs from requested chart")
        return state


class Runner(grid.Runner):
    def __init__(self, cfg, store, *args, **kwargs):
        super().__init__(cfg, ChartStore(store), *args, **kwargs)
        self.compression_backend = "subspace_" + cfg.device

    def _model(self, family, replica):
        fitted, layout = super()._model(family,replica)
        self.store.chart_identity = dict(stm_config=fitted.stm_config,
            chart_hash=content_hash(dict(name=fitted.stm_config,layout=layout.metadata())),
            layout_hash=content_hash(layout.metadata()), factor_mapping=fitted.factor_mapping,
            factor_seeds=fitted.factor_seeds)
        return fitted, layout

    def assets(self, family, replica, partition):
        chart = model.CHART.get()
        if chart == "all_r2":
            return super().assets(family, replica, partition)
        with model.use_chart("all_r2"):
            parent, original, data, old_model, old_layout = super().assets(family, replica, partition)
        unit = chart + "-" + parent
        fitted, layout = self._model(family, replica)
        if self.store.is_complete(unit):
            state = self.store.load_checkpoint(unit)
            layout.assert_metadata(state["layout"])
            if state["plan_hash"] != original["plan_hash"]:
                raise ValueError("adapted asset plan mismatch")
            return unit, state, data, fitted, layout
        metadata = dict(kind="stm_capacity_assets", schema_version=1, stm_config=chart,
            parent=parent, parent_seal=read_json(self.store.unit_path(parent)/"COMPLETED")["seal_hash"],
            family=family, replica=replica, partition=partition, plan_hash=original["plan_hash"])
        self.store.begin(unit, metadata)
        values = layout.unflatten_named(layout.flatten_module(fitted, detach=True).cpu())
        canonical = old_layout.unflatten_named(original["theta"])
        for name in values:
            if "lora_" not in name:
                values[name] = canonical[name].clone()
        theta = torch.cat([values[s.name].reshape(-1) for s in layout.specs])
        ltm, stm = model.memory_indices(layout)
        old_ltm, _ = model.memory_indices(old_layout)
        if not torch.equal(theta[ltm], original["theta"][old_ltm]):
            raise ValueError("initial LTM is not exactly paired")
        learner = copy.deepcopy(original["learner"])
        learner.update(theta=theta.clone(), ltm_indices=ltm)
        # The dense initial Fisher stays in the canonical asset only.
        state = {k: copy.deepcopy(v) for k, v in original.items()
                 if k not in ("theta", "learner", "layout", "reference_fisher", "training_optimizer")}
        state.update(theta=theta, learner=learner, layout=layout.metadata(), parent_asset=parent,
                     parent_seal=metadata["parent_seal"], stm_config=chart)
        self.store.save_checkpoint(unit, state)
        self.store.write_json(unit, "health.json", {"parent": parent, "stm_config": chart,
            "ltm_identical": True, "initial_residual_zero": all(
                not values[s.name].count_nonzero() for s in layout.specs if s.name.endswith("lora_B")),
            "shared_initial_fisher": parent})
        self.store.complete(unit)
        return unit, state, data, fitted, layout

    def run_chart(self, chart, family, replica, partition, recipe, **kwargs):
        previous = grid.trajectory_id
        try:
            grid.trajectory_id = lambda f, r, p, n: trajectory_id(chart, f, r, p, n)
            with model.use_chart(chart):
                result = super().run(family, replica, partition, recipe, **kwargs)
                return {**result, "stm_config": chart, "arm": arm(chart, recipe.method),
                        "protocol": PROTOCOL}
        finally:
            grid.trajectory_id = previous

    def record(self, unit):
        result = super().record(unit)
        if result.get("stm_config") not in model.CONFIGURATIONS:
            raise ValueError("trajectory lacks its STM chart")
        return result

    def captured(self, chart, family, replica, recipe):
        unit = trajectory_id(chart, family, replica, recipe, self.cfg.novel_count)
        with model.use_chart(chart):
            return self.store.load_checkpoint(unit)["snapshots"]

