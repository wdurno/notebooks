"""Phase 5 numerical neutrality, score diagnostics, and acquisition recovery."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.combined_memory.analysis import novel_curve
from mnist_experiment.combined_memory.config import Recipe
from mnist_experiment.combined_memory.model import build_model, memory_indices, merge_parameters
from mnist_experiment.combined_memory.runtime import setup
from mnist_experiment.combined_memory.broad_drift.config import Config
from mnist_experiment.combined_memory.broad_drift.data import make_assets
from mnist_experiment.combined_memory.broad_drift.detectability import score_summary, run_diagnostic
from mnist_experiment.combined_memory.broad_drift.observer import Controller
from mnist_experiment.combined_memory.broad_drift.trajectory import Reader, run_trajectory
from mnist_experiment.kalman_pi.policies import Controller as OriginalController, PolicySpec
from mnist_experiment.fisher_memory.control import StopRequested
from test_combined_memory_recovery import (
    raw, one_cpu_thread, DummyBudget, assert_tree_equal, without_timing, store,
)


def test_instrumented_gain_preserves_exact_original_filter_and_action():
    cfg = Config.smoke_config()
    rng = torch.Generator().manual_seed(525)
    diagonal = torch.rand(17, generator=rng, dtype=torch.float64)+.1
    variance = torch.rand(17, generator=rng, dtype=torch.float64)+.2
    anchor = torch.randn(17, generator=rng, dtype=torch.float64)
    actual = Controller(PolicySpec("kalman", 1e-4), cfg, .025)
    expected = OriginalController(PolicySpec("kalman", 1e-4), cfg, .025)
    actual.initialize(anchor, diagonal, variance)
    expected.initialize(anchor, diagonal, variance)
    q = .01
    for step in range(15):
        assert actual.choose(q,diagonal,4,step) == expected.choose(q,diagonal,4,step)
        scores = torch.randn(4,17,generator=rng,dtype=torch.float64)+step*.02
        result = actual.assimilate(scores,step)
        original = expected.assimilate(scores,step)
        assert {k:result[k] for k in original} == original
        assert_tree_equal(actual.state, expected.state)
        assert 0 < result["kalman_level_gain_mean"] < 1
        assert result["kalman_velocity_gain_mean"] > 0
        for name in ("level","velocity"):
            assert result[f"kalman_{name}_gain_p10"] <= result[f"kalman_{name}_gain_p90"]


def test_frozen_score_mixture_variance_and_per_batch_motion_units():
    nominal = torch.tensor([[-1.],[1.],[-1.],[1.]],dtype=torch.float64)
    center = nominal+2
    metric = torch.tensor([3.],dtype=torch.float64)
    result = score_summary(nominal,center,center-.4,center+.4,metric,
        p=.5,pminus=.5,pplus=.5,lag=8,batch_size=4,bootstrap_seed=4,bootstrap_draws=20)
    # Within-component unbiased variance is 4/3; between-mean variance is 1.
    assert result["mixture_components"]["mixture_variance"] == pytest.approx(7.)
    assert result["broad"]["batch_noise"] == pytest.approx(1.)
    assert result["rare"]["batch_noise"] == pytest.approx(7/4)
    assert result["broad"]["motion_energy"] == pytest.approx(3*.1**2)
    assert result["rare"]["motion_energy"] == pytest.approx(3*.05**2)
    assert result["broad"]["split_motion_energy"] == pytest.approx(3*.1**2)
    assert result["broad"]["mean_difference_mc_variance"] == pytest.approx(0.,abs=1e-25)


def test_score_diagnostic_keeps_mixture_weight_motion_and_unresolved_energy():
    zero = torch.zeros(4,1,dtype=torch.float64)
    rotated = torch.ones_like(zero)*2
    result = score_summary(zero,rotated,rotated,rotated,torch.ones(1,dtype=torch.float64),
        p=.2,pminus=.1,pplus=.3,lag=2,batch_size=4,bootstrap_seed=9,bootstrap_draws=20)
    assert result["broad"]["motion_energy"] == 0
    assert result["rare"]["motion_energy"] == pytest.approx(.2**2)
    noisy = torch.tensor([[1.],[1.],[-1.],[-1.]],dtype=torch.float64)
    unresolved = score_summary(zero,noisy,zero,noisy,torch.ones(1,dtype=torch.float64),
        p=.2,pminus=.1,pplus=.3,lag=1,batch_size=4,bootstrap_seed=5,bootstrap_draws=20)
    assert unresolved["broad"]["motion_energy"] == 0
    assert unresolved["broad"]["split_motion_energy"] < 0


def test_frozen_diagnostic_completes_without_training_and_reuses_sealed_result(tmp_path, raw):
    cfg = Config.smoke_config()
    setup(cfg)
    assets = make_assets(cfg,0,"development",raw)
    model, layout = build_model(cfg.seed_for("development",0,"initialization_model"),
        dtype=torch.float64,factor_seed=cfg.seed_for("development",0,"factors"))
    theta = layout.flatten_module(model,detach=True)
    initial = dict(theta=theta,diagonal=torch.ones(3268,dtype=torch.float64))
    before = deepcopy(initial)
    with store(tmp_path/"diagnostic") as storage:
        scheduler = SimpleNamespace(cfg=cfg,store=storage,budget=DummyBudget(),raw=raw,
            assets=lambda *_:assets,initial=lambda *_:initial,pulse=lambda **_:None)
        result = run_diagnostic(scheduler,0)
        assert storage.is_complete("diagnostic-detectability-000")
        assert len(result["states"]) == 1 and result["controller_input"] is False
        assert result["states"][0]["broad"]["batch_noise"] > 0
    with store(tmp_path/"diagnostic",resume=True) as storage:
        scheduler.store = storage
        scheduler.initial = lambda *_: pytest.fail("sealed diagnostic must not rebuild a model")
        assert run_diagnostic(scheduler,0) == result
    assert_tree_equal(before,initial)


@pytest.mark.parametrize("recipe", [Recipe("A"), Recipe("E",strength=.1)])
def test_outbound_clock_and_gain_state_survive_restart(tmp_path, raw, recipe):
    cfg = replace(Config.smoke_config(device=os.environ.get("PLAN18_TEST_DEVICE","cpu")),
        hold_steps=1,ramp_steps=2,capacity=2,max_iter=2,max_eval=6,analysis_novel_count=4)
    cfg.validate()
    setup(cfg)
    assets = make_assets(cfg,0,"development",raw)
    model_seed = cfg.seed_for("development",0,"initialization_model")
    factor_seed = cfg.seed_for("development",0,"factors")
    model,layout = build_model(model_seed,dtype=torch.float64,factor_seed=factor_seed)
    theta = layout.flatten_module(model,detach=True)
    ltm,_ = memory_indices(layout)
    _,_,merged = merge_parameters(model,layout,theta)
    initial = dict(theta=theta,diagonal=torch.ones(len(ltm),dtype=torch.float64),
        variance=torch.ones(len(ltm),dtype=torch.float64),merged_anchor=merged)
    metadata = dict(cohort="development",replica=0,model_seed=model_seed,factor_seed=factor_seed,
                    recipe=recipe.mapping())

    def run(storage,budget,pulse=lambda **kw:None):
        return run_trajectory(storage,budget,cfg,"broad-path",metadata,initial,
            Reader(raw,assets,cfg,check=budget.check),pulse=pulse)

    with store(tmp_path/"whole") as storage:
        whole = run(storage,DummyBudget())
    budget = DummyBudget()
    with store(tmp_path/"resumed") as storage:
        def pulse(**progress):
            if progress["batch"] == cfg.hold_steps+1:
                budget.stop = True
        with pytest.raises(StopRequested):
            run(storage,budget,pulse)
        checkpoint = storage.load_checkpoint("broad-path")
        assert checkpoint["novel_count"] == cfg.samples_per_step
    with store(tmp_path/"resumed",resume=True) as storage:
        resumed = run(storage,DummyBudget())
        summary = json.loads((storage.unit_path("broad-path")/"summary.json").read_text())
    assert_tree_equal(without_timing(resumed),without_timing(whole))
    assert summary["analysis_outbound_count"] == cfg.analysis_novel_count
    assert resumed["novel_count"] == cfg.ramp_steps*cfg.samples_per_step
    curve = novel_curve(resumed["metrics"],cfg)
    assert curve["supported"]
    assert curve["observed_steps"] == [cfg.hold_steps,cfg.hold_steps+1,cfg.hold_steps+2]
    assert max(curve["observed_steps"]) <= cfg.hold_steps+cfg.ramp_steps
    assert resumed["retired_ids"]+resumed["replay"] == list(range(cfg.stream_size))
    if recipe.adaptive:
        assert resumed["controller"]["assimilations"] == cfg.num_p_steps
        assert all("kalman_velocity_gain_mean" in row for row in resumed["diagnostics"])
