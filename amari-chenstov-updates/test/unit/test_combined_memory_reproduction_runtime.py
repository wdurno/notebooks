"""Reproduction identities and fidelity gates protect the scientific contrast."""

from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store, ImmutableError
from mnist_experiment.combined_memory.reproduction.config import POLICIES, configuration, recipe
from mnist_experiment.combined_memory.reproduction.runner import NamespaceStore
from mnist_experiment.combined_memory.reproduction.audit import compare_array
from mnist_experiment.combined_memory.reproduction.run import Study


def test_policies_are_the_historical_recipes_and_distinct_from_phase7():
    cfg = configuration({}, smoke=True, device="cpu")
    assert recipe(cfg,"F1").strength == 3.34056320604949
    assert recipe(cfg,"F1").ridge == 1
    assert recipe(cfg,"R1").ridge == recipe(cfg,"R2").ridge == 10**.5
    assert POLICIES["R1"][0] == POLICIES["F1"][0] == "head_r1"
    assert configuration({},fresh=True,smoke=True,device="cpu").seed != cfg.seed


def test_contexts_share_assets_but_never_trajectories(tmp_path):
    with Store(tmp_path/"study",{"protocol":"test"}) as store:
        blur, clear = NamespaceStore(store,"fresh","blur"), NamespaceStore(store,"fresh","no_blur")
        assert blur.key("assets-confirmation-grid-0000") == clear.key("assets-confirmation-grid-0000")
        assert blur.key("head_r1-assets-confirmation-grid-0000") == clear.key("head_r1-assets-confirmation-grid-0000")
        assert blur.key("head_r1-trajectory-test") != clear.key("head_r1-trajectory-test")
        blur.begin("assets-test",{"kind":"asset"})
        blur.save_checkpoint("assets-test",{"theta":torch.tensor([1.])})
        blur.complete("assets-test")
        assert clear.is_complete("assets-test")
        assert torch.equal(clear.load_checkpoint("assets-test")["theta"],torch.tensor([1.]))
        with pytest.raises(ImmutableError):
            clear.write_json("assets-test","changed.json",{})
        assert not NamespaceStore(store,"historical","blur").is_complete("assets-test")


def test_numerical_parity_rejects_scientifically_changed_values():
    assert compare_array([1.,2.],[1.,2.],"risk")["max_absolute"] == 0.
    with pytest.raises(ValueError,match="parity mismatch"):
        compare_array([1.,2.],[1.,2.01],"risk")
    with pytest.raises(ValueError,match="nonfinite"):
        compare_array([1.],[float("nan")],"risk")


def test_failed_fidelity_blocks_fresh_paths():
    study = object.__new__(Study)
    study.args = Namespace(stage="all")
    study.progress = lambda **kw: None
    def fail():
        raise ValueError("fidelity mismatch")
    study.fidelity_checks = fail
    study.reserve = lambda: pytest.fail("fresh resource reservation before fidelity passed")
    study.job = lambda *a: pytest.fail("fresh path launched before fidelity passed")
    with pytest.raises(ValueError,match="fidelity mismatch"):
        study.execute()
