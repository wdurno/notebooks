import json
from pathlib import Path

import pytest
import torch

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase3_config import (
    PHASE3_CONDITIONS,
    RotatedPhase3Config,
    load_phase3_config,
)
from mnist_experiment.rotated_mnist.run_phase3 import (
    _classwise_metrics,
    _normalized_auc,
    _persistent_thresholds,
)


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase3_smoke.json"
)


def test_phase3_config_round_trips_and_freezes_pairing() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_phase3_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == RotatedPhase3Config.from_mapping(raw)
    assert config.conditions == PHASE3_CONDITIONS
    assert config.rotation.knots_degrees == (0.0, 15.0, 30.0)
    assert config.fisher.rank == 8
    assert config.fisher.fixed_pi == 0.05


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditions", ["current_only"], "conditions"),
        ("rotation.knots_degrees", [0.0, 30.0], "first rotation ascent"),
        ("data.samples_per_step", 4, "m=8"),
        ("fisher.rank", 4, "rank is frozen"),
        ("fisher.fixed_pi", 0.1, "fixed pi"),
    ],
)
def test_phase3_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        RotatedPhase3Config.from_mapping(raw)


def test_classwise_metrics_keep_precision_and_recall_distinct() -> None:
    confusion = torch.eye(10, dtype=torch.long) * 5
    confusion[0, 1] = 3
    metrics = _classwise_metrics(confusion)

    assert metrics["per_class_recall"][0] == pytest.approx(5 / 8)
    assert metrics["per_class_precision"][0] == pytest.approx(1.0)
    assert metrics["per_class_precision"][1] == pytest.approx(5 / 8)
    assert metrics["worst_class_label"] == 0
    assert metrics["worst_class_recall"] == pytest.approx(5 / 8)


def test_phase3_trajectory_summaries_use_observation_exposure() -> None:
    rows = [
        {
            "observations_before_evaluation": exposure,
            "current_environment_accuracy": accuracy,
        }
        for exposure, accuracy in ((0, 0.5), (8, 0.7), (16, 0.8), (24, 0.75))
    ]

    assert _normalized_auc(rows, "current_environment_accuracy") == pytest.approx(
        (0.6 + 0.75 + 0.775) / 3
    )
    assert _persistent_thresholds(rows) == {
        "persistent_env_accuracy_0.6": 8,
        "persistent_env_accuracy_0.7": 8,
        "persistent_env_accuracy_0.8": None,
    }
