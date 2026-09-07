import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase8_artifacts import Phase8RunStore
from mnist_experiment.rotated_mnist.phase8_config import (
    Phase8Config,
    load_phase8_config,
)


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase8_smoke.json"
)


def test_phase8_config_round_trips_and_freezes_rechallenge() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_phase8_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == Phase8Config.from_mapping(raw)
    assert config.source_kind == "single_lap"
    assert config.conditions == (
        "fixed_pi005_sentinel",
        "tracked_q_covariance",
        "decomposed_edr",
    )
    assert config.controller.cold_start_steps == 1
    assert config.controller.movement_half_life_steps == 8.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_kind", "unknown", "source kind"),
        ("conditions", ["fixed_pi005_sentinel"], "decomposed_edr"),
        ("controller.pi_min", 0.02, "frozen"),
        ("controller.movement_half_life_steps", 4.0, "frozen"),
        ("controller.trend_half_life_degrees", 1.875, "trend half-life"),
        ("controller.cold_start_steps", 8, "cold_start_steps"),
    ],
)
def test_phase8_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        Phase8Config.from_mapping(raw)


def test_phase8_run_store_is_distinct_and_immutable(tmp_path: Path) -> None:
    config = load_phase8_config(SMOKE_CONFIG)
    session = Phase8RunStore(tmp_path).begin(config, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(Exception, match="completed"):
        Phase8RunStore(tmp_path).begin(config, REPO_ROOT)
