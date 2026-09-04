import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase5_double_lap_artifacts import (
    RotatedDoubleLapRunStore,
)
from mnist_experiment.rotated_mnist.phase5_double_lap_config import (
    DOUBLE_LAP_CONDITIONS,
    DOUBLE_LAP_SCHEDULES,
    RotatedDoubleLapConfig,
    load_double_lap_config,
)
from mnist_experiment.rotated_mnist.run_phase5_double_lap import (
    EDR_CONDITION,
    _classify_retry,
)


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase5_double_lap_smoke.json"
)


def test_double_lap_config_round_trips_and_freezes_treatment() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_double_lap_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == RotatedDoubleLapConfig.from_mapping(raw)
    assert config.conditions == DOUBLE_LAP_CONDITIONS
    assert config.schedule_kinds == DOUBLE_LAP_SCHEDULES
    assert config.data.samples_per_step == 4
    assert config.controller.cold_start_steps == 8
    assert config.controller.trend_half_life_degrees == 1.875
    assert config.controller.action_half_life_steps == 8.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditions", ["current_only"], "conditions"),
        ("schedule_kinds", ["sigmoid"], "linear and sigmoid"),
        ("sigmoid_kappa", 4.0, "kappa"),
        ("data.samples_per_step", 2, "m=4"),
        ("controller.cold_start_steps", 4, "cold start"),
        ("controller.trend_half_life_degrees", 3.75, "frozen"),
    ],
)
def test_double_lap_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        RotatedDoubleLapConfig.from_mapping(raw)


def _summary(accuracy: float, *, action_lift: float = 0.0) -> dict[str, object]:
    return {
        "environment_accuracy_auc": accuracy,
        "environment_nll_auc": 1.0,
        "final_upright_environment_accuracy": 0.8,
        "final_worst_class_recall": 0.6,
        "fast_minus_slow_action": action_lift,
        "lagged_speed_action_correlation": 0.5,
    }


def test_double_lap_classification_requires_favorable_schedule_interaction() -> None:
    summaries = {}
    for schedule in DOUBLE_LAP_SCHEDULES:
        summaries[schedule] = {
            "current_only": _summary(0.6),
            "fixed_pi0025": _summary(0.70),
            "fixed_pi005": _summary(0.72),
            "fixed_pi0075": _summary(0.71),
            "fixed_pi010": _summary(0.69),
            EDR_CONDITION: _summary(
                0.72 if schedule == "linear" else 0.74,
                action_lift=0.01 if schedule == "sigmoid" else 0.0,
            ),
        }

    result = _classify_retry(summaries)

    assert result["classification"] == "dynamic_value"
    assert result["schedule_by_policy_interaction"] == pytest.approx(0.02)


def test_double_lap_classification_stops_predictively_harmful_response() -> None:
    summaries = {}
    for schedule in DOUBLE_LAP_SCHEDULES:
        summaries[schedule] = {
            "current_only": _summary(0.6),
            "fixed_pi0025": _summary(0.70),
            "fixed_pi005": _summary(0.70),
            "fixed_pi0075": _summary(0.69),
            "fixed_pi010": _summary(0.68),
            EDR_CONDITION: {
                **_summary(0.40, action_lift=0.01),
                "environment_nll_auc": 4.0,
                "final_upright_environment_accuracy": 0.4,
            },
        }

    result = _classify_retry(summaries)

    assert result["classification"] == "stop"
    assert result["no_material_secondary_regression"] is False


def test_double_lap_completed_run_store_is_immutable(tmp_path: Path) -> None:
    config = load_double_lap_config(SMOKE_CONFIG)
    store = RotatedDoubleLapRunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(Exception, match="completed"):
        store.begin(config, REPO_ROOT)
