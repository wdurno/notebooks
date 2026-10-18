import dataclasses

import pytest
import torch

from mnist_experiment.fisher_memory.config import Config
from mnist_experiment.fisher_memory.data import audit_cohorts, class_weights, plan_data


class Source:
    train_labels = torch.arange(10000) % 10
    test_labels = torch.arange(4000) % 10


def test_pairing_disjointness_and_evaluation_isolation():
    cfg = dataclasses.replace(Config(), init_size=100, reference_size=20, eval_per_class=5)
    a = plan_data(Source(), cfg, "pilot", 1, "mixture")
    assert a == plan_data(Source(), cfg, "pilot", 1, "mixture")
    assert len(a["stream_ids"]) == len(set(a["stream_ids"])) == 202
    assert not (set(a["initialization_ids"]) & set(a["stream_ids"]))
    assert not (set(a["reference_ids"]) & set(a["stream_ids"]))
    assert all(Source.train_labels[i] != 9 for i in a["initialization_ids"])
    b = plan_data(Source(), cfg, "confirmation", 1, "mixture")
    assert not (set(a["evaluation_ids"]) & set(b["evaluation_ids"]))


def test_weights_and_censoring():
    weights = class_weights(torch.arange(10).repeat(3), [.1] * 10)
    assert weights.sum() == pytest.approx(1)
    assert audit_cohorts(Config(), "digit9_mixture")[-1]["horizon"] == 28
    assert audit_cohorts(Config(), "rotation_linear")[0]["eviction_step"] == 20
