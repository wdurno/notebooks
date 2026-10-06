from __future__ import annotations

import math
from pathlib import Path

from mnist_experiment.continual_subgd.artifacts import UnitStore
from mnist_experiment.continual_subgd.config import Plan13Study
from mnist_experiment.continual_subgd.phase6_low_prevalence import (
    PHASE6A,
    PHASE6B,
    PHASE6B_SCHEMA_VERSION,
    asset_unit,
    build_ledger,
    expected_schedule,
    low_prevalence_data_config,
    phase6b_conditions,
)


REPO_ROOT = Path(__file__).parents[3]
CONFIG = REPO_ROOT / "mnist_experiment/continual_subgd/configs/default.json"


def test_phase6b_schedule_expects_about_one_hundred_nines() -> None:
    schedule = expected_schedule(phase=PHASE6B)
    data = low_prevalence_data_config(smoke=False, phase=PHASE6B)
    smoke_data = low_prevalence_data_config(smoke=True, phase=PHASE6B)

    assert data.samples_per_step == 182
    assert smoke_data.samples_per_step == 182
    assert math.isclose(data.samples_per_step * sum(schedule[1:]), 100.1)
    assert [condition.name for condition in phase6b_conditions()] == [
        "no_update",
        "full_space",
        "digit9_bias_only",
        "head_only",
        "adaptive_floor_0.1",
    ]


def test_phase6b_ledger_is_distinct_and_replica_incremental(tmp_path: Path) -> None:
    study = Plan13Study.from_path(CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)

    ledger = build_ledger(store, (1, 2), phase=PHASE6B, smoke=False)
    repeated = build_ledger(store, (1, 2), phase=PHASE6B, smoke=False)
    phase6a_asset = asset_unit(store, 1, phase=PHASE6A, smoke=False)
    phase6b_asset = asset_unit(store, 1, phase=PHASE6B, smoke=False)

    assert ledger == repeated
    assert len(ledger["items"]) == 15
    assert ledger["schema_version"] == PHASE6B_SCHEMA_VERSION
    assert ledger["items"][-1]["action"] == "phase6_analysis"
    assert phase6b_asset["detail"]["schema_version"] == PHASE6B_SCHEMA_VERSION
    assert phase6b_asset != phase6a_asset
    assert store.paths(phase6b_asset) != store.paths(phase6a_asset)
