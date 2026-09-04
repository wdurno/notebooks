import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase5_single_lap_artifacts import (
    RotatedSingleLapRunStore,
    SINGLE_LAP_RUN_KIND,
)
from mnist_experiment.rotated_mnist.phase5_single_lap_config import (
    SINGLE_LAP_CONDITIONS,
    SINGLE_LAP_EDR_CONDITION,
    SINGLE_LAP_SCHEDULES,
    RotatedSlowSingleLapConfig,
    load_single_lap_config,
)
from mnist_experiment.rotated_mnist.run_phase5_single_lap import (
    classify_single_lap_retry,
)


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase5_single_lap_smoke.json"
)


def test_single_lap_config_round_trips_and_freezes_slow_treatment() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_single_lap_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == RotatedSlowSingleLapConfig.from_mapping(raw)
    assert config.conditions == SINGLE_LAP_CONDITIONS
    assert config.schedule_kinds == SINGLE_LAP_SCHEDULES
    assert config.rotation.knots_degrees == (0.0, 30.0, 0.0)
    assert config.data.samples_per_step == 4
    assert config.controller.cold_start_steps == 8
    assert config.controller.trend_half_life_degrees == 7.5
    assert config.controller.action_half_life_steps == 8.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditions", ["current_only"], "conditions"),
        ("schedule_kinds", ["sigmoid"], "linear and sigmoid"),
        ("sigmoid_kappa", 4.0, "kappa"),
        ("rotation.knots_degrees", [0.0, 30.0], "knots"),
        ("data.samples_per_step", 2, "m=4"),
        ("controller.trend_half_life_degrees", 1.875, "frozen"),
    ],
)
def test_single_lap_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        RotatedSlowSingleLapConfig.from_mapping(raw)


def _summary(
    accuracy: float,
    *,
    nll: float = 1.0,
    action_min: float = 0.03,
    action_max: float = 0.06,
) -> dict[str, object]:
    return {
        "environment_accuracy_auc": accuracy,
        "environment_nll_auc": nll,
        "final_upright_environment_accuracy": 0.8,
        "final_worst_class_recall": 0.6,
        "action_min": action_min,
        "action_max": action_max,
        "action_span": action_max - action_min,
    }


def _summaries(edr: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    return {
        schedule: {
            "current_only": _summary(0.5),
            "fixed_pi0025": _summary(0.72),
            "fixed_pi005": _summary(0.70),
            SINGLE_LAP_EDR_CONDITION: dict(edr),
        }
        for schedule in SINGLE_LAP_SCHEDULES
    }


def test_single_lap_classifier_accepts_stable_automatic_selection() -> None:
    result = classify_single_lap_retry(_summaries(_summary(0.715)))

    assert result["classification"] == "automatic_selection_value"
    assert result["stable_action_bracket"] is True


def test_single_lap_classifier_stops_action_escape_and_harm() -> None:
    result = classify_single_lap_retry(
        _summaries(_summary(0.40, nll=4.0, action_min=0.15, action_max=0.60))
    )

    assert result["classification"] == "stop"
    assert result["stable_action_bracket"] is False
    assert result["no_material_secondary_regression"] is False


def test_single_lap_classifier_marks_cold_only_smoke_as_integration() -> None:
    summaries = _summaries(_summary(0.715))
    summaries["linear"][SINGLE_LAP_EDR_CONDITION][
        "action_statistics_include_cold_start"
    ] = True

    result = classify_single_lap_retry(summaries)

    assert result["classification"] == "integration_only"
    assert result["action_statistics_include_cold_start"] is True


def test_single_lap_run_store_uses_distinct_immutable_manifest(
    tmp_path: Path,
) -> None:
    config = load_single_lap_config(SMOKE_CONFIG)
    store = RotatedSingleLapRunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    manifest = json.loads(
        (session.working_path / "manifest.json").read_text(encoding="utf-8")
    )
    path = session.complete(required=())

    assert manifest["run_kind"] == SINGLE_LAP_RUN_KIND
    assert (path / "COMPLETED").is_file()
    with pytest.raises(Exception, match="completed"):
        store.begin(config, REPO_ROOT)
