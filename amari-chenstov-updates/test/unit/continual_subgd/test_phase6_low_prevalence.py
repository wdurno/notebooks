from __future__ import annotations

import math
from pathlib import Path

import torch

from mnist_experiment.continual_subgd.artifacts import UnitStore
from mnist_experiment.continual_subgd.config import Plan13Study
from mnist_experiment.continual_subgd.geometry import AdaptationGeometry
from mnist_experiment.continual_subgd.phase6_low_prevalence import (
    PHASE,
    build_ledger,
    digit9_bias_functional_error,
    expected_schedule,
    fixed_prevalence_precision,
    low_prevalence_data_config,
    phase6_conditions,
    standardized_precision_recall_curve,
)
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model


REPO_ROOT = Path(__file__).parents[3]
CONFIG = REPO_ROOT / "mnist_experiment/continual_subgd/configs/default.json"


def test_phase6_schedule_and_conditions_are_frozen() -> None:
    schedule = expected_schedule()
    assert len(schedule) == 11
    assert schedule[0] == 0.0
    assert schedule[-1] == 0.1
    assert math.isclose(8 * sum(schedule[1:]), 4.4)
    assert low_prevalence_data_config(smoke=False).samples_per_step == 8
    assert [condition.name for condition in phase6_conditions()] == [
        "no_update",
        "full_space",
        "digit9_bias_only",
        "head_only",
        "random_rank_one",
        "static_tiny_burn_subgd",
        "adaptive_floor_0.1",
    ]


def test_fixed_prevalence_precision_reconstructs_from_conditional_rates() -> None:
    precision, undefined = fixed_prevalence_precision(0.8, 0.9)
    assert not undefined
    assert math.isclose(precision, 8 / 17)
    precision, undefined = fixed_prevalence_precision(0.0, 1.0)
    assert precision == 0.0
    assert undefined


def test_standardized_average_precision_uses_reference_prevalence() -> None:
    result = standardized_precision_recall_curve(
        torch.tensor([0.9, 0.8, 0.7, 0.1], dtype=torch.float64),
        torch.tensor([9, 0, 9, 0]),
        threshold_count=5,
    )
    assert math.isclose(result["average_precision"], 13 / 22)
    assert result["reference_prevalence"] == 0.1
    assert len(result["precision"]) == 5
    assert len(result["recall"]) == 5


def test_digit9_bias_coordinate_is_functionally_exact() -> None:
    model, layout = build_gauge_fixed_model(41, dtype=torch.float64)
    inputs = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    assert digit9_bias_functional_error(model, layout, inputs) < 1e-12


def test_growing_geometry_matches_untruncated_dense_ema() -> None:
    geometry = AdaptationGeometry(
        torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    observation = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    actual = geometry.update_growing(observation, 0.25, rank_cap=2)
    expected = torch.diag(torch.tensor([1.5, 0.25, 0.0], dtype=torch.float64))
    assert actual.rank == 2
    assert torch.allclose(actual.dense(), expected, atol=1e-12, rtol=1e-12)


def test_phase6_ledger_is_replica_incremental_and_deterministic(tmp_path: Path) -> None:
    study = Plan13Study.from_path(CONFIG)
    store = UnitStore(tmp_path / "runs", study, REPO_ROOT)
    first = build_ledger(store, (1, 2), phase=PHASE, smoke=False)
    second = build_ledger(store, (1, 2), phase=PHASE, smoke=False)
    assert first == second
    assert len(first["items"]) == 19
    assert [item["action"] for item in first["items"][:2]] == [
        "phase6_assets",
        "phase6_burn_in",
    ]
    assert first["items"][-1]["action"] == "phase6_analysis"
