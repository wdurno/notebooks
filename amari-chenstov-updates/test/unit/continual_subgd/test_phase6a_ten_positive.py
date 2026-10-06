from __future__ import annotations

import math
from pathlib import Path

from mnist_experiment.continual_subgd.artifacts import UnitStore
from mnist_experiment.continual_subgd.config import Plan13Study
from mnist_experiment.continual_subgd.phase6_low_prevalence import (
    PHASE,
    PHASE6A,
    PHASE6A_SCHEMA_VERSION,
    asset_unit,
    build_ledger,
    expected_schedule,
    low_prevalence_data_config,
    phase6a_conditions,
)


REPO_ROOT = Path(__file__).parents[3]
CONFIG = REPO_ROOT / "mnist_experiment/continual_subgd/configs/default.json"


def test_phase6a_schedule_expects_about_ten_nines() -> None:
    schedule = expected_schedule(phase=PHASE6A)
    data = low_prevalence_data_config(smoke=False, phase=PHASE6A)
    smoke_data = low_prevalence_data_config(smoke=True, phase=PHASE6A)

    assert data.samples_per_step == 18
    assert smoke_data.samples_per_step == 18
    assert math.isclose(data.samples_per_step * sum(schedule[1:]), 9.9)
    assert [condition.name for condition in phase6a_conditions()] == [
        "no_update",
        "full_space",
        "digit9_bias_only",
        "head_only",
        "adaptive_floor_0.1",
    ]


def test_phase6a_ledger_is_distinct_and_replica_incremental(tmp_path: Path) -> None:
    study = Plan13Study.from_path(CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)

    ledger = build_ledger(store, (1, 2), phase=PHASE6A, smoke=False)
    repeated = build_ledger(store, (1, 2), phase=PHASE6A, smoke=False)
    phase6_asset = asset_unit(store, 1, phase=PHASE, smoke=False)
    phase6a_asset = asset_unit(store, 1, phase=PHASE6A, smoke=False)

    assert ledger == repeated
    assert len(ledger["items"]) == 15
    assert ledger["schema_version"] == PHASE6A_SCHEMA_VERSION
    assert ledger["items"][-1]["action"] == "phase6_analysis"
    assert phase6a_asset["detail"]["schema_version"] == PHASE6A_SCHEMA_VERSION
    assert phase6a_asset != phase6_asset
    assert store.paths(phase6a_asset) != store.paths(phase6_asset)
