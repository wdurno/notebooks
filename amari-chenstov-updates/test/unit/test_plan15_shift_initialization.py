"""Small deterministic initialization extension, with identical paired inputs."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory_regression.config import ROOT
from mnist_experiment.fisher_memory_regression.runtime import setup_torch
from mnist_experiment.fisher_memory_shift.config import Config
from mnist_experiment.fisher_memory_shift.data import MNISTSource
from mnist_experiment.fisher_memory_shift.runner import Runner


def test_epoch_extension_preserves_adam_and_matches_uninterrupted_pretraining(tmp_path):
    cfg = replace(Config.smoke_config(), max_init_epochs=2, init_epoch_stages=(1,2))
    setup_torch(cfg)
    source = MNISTSource(ROOT/"cache/mnist_experiment/datasets")
    states = []
    for name, staged in (("extended",True),("direct",False)):
        with Store(tmp_path/name, {"protocol":cfg.protocol,"test":"epoch-extension"},
                   max_bytes=500_000_000, free_reserve_bytes=0) as store:
            runner = Runner(cfg, store, source, SimpleNamespace(check=store.check))
            if staged:
                runner.assets("development", 0, "development")
            runner.init_epochs = 2
            _, state, _, _, _ = runner.assets("development", 0, "development")
            states.append(state)
    assert states[0]["plan_hash"] == states[1]["plan_hash"]
    torch.testing.assert_close(states[0]["theta"], states[1]["theta"], rtol=0, atol=0)
    assert states[0]["training_optimizer"]["state"][0]["step"] == states[1]["training_optimizer"]["state"][0]["step"]
