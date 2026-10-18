import dataclasses

import pytest

from mnist_experiment.fisher_memory.config import Config, Treatment


def test_protocol_and_schedules():
    cfg = Config()
    cfg.validate()
    assert len(cfg.schedule("rotation_linear")) == 120
    assert cfg.schedule("rotation_linear")[39] == 30
    assert cfg.schedule("rotation_linear")[79] == 0
    assert cfg.schedule("rotation_sigmoid")[-1] == 30
    assert cfg.expected_nines("digit9_mixture") == pytest.approx(10.1)
    assert dataclasses.replace(cfg, mixture_batch=20).expected_nines("digit9_mixture") == pytest.approx(101)


def test_budget_is_not_scientific_identity_but_cannot_exceed_cap():
    cfg = Config()
    assert cfg.scientific_hash == dataclasses.replace(cfg, compute_allocation_seconds=72000).scientific_hash
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, max_artifact_bytes=500_000_000_001).validate()


def test_seeds_and_conditions_are_stable_and_distinct():
    cfg = Config()
    assert cfg.seed_for("pilot", 1, "stream") == cfg.seed_for("pilot", 1, "stream")
    assert len({cfg.seed_for(f, r, c) for f in ("pilot", "confirmation")
                for r in (1, 2) for c in ("stream", "probes")}) == 8
    assert Treatment.default(cfg, "jvp", .1).key != Treatment.default(cfg, "jvp", .2).key
