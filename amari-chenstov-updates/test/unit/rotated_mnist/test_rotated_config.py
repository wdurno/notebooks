import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import (
    RotatedConfigError,
    RotatedExperimentConfig,
    load_config,
)


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase1_smoke.json"
)


def test_smoke_config_round_trips_with_stable_hash() -> None:
    config = load_config(SMOKE_CONFIG)
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))

    assert config.to_mapping() == raw
    assert config == RotatedExperimentConfig.from_mapping(raw)
    assert config.config_hash == (
        "d478561f92b548bb1995195a587ee5288bb01b629c5ff6e430c3b2e7d520ba16"
    )
    assert config.run_id.endswith(config.config_hash[:16])
    assert config.data.samples_per_step == 2
    assert config.data.stream_width == 8


def test_config_rejects_unknown_and_nonmodular_experiment() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["surprise"] = True
    with pytest.raises(RotatedConfigError, match="unknown"):
        RotatedExperimentConfig.from_mapping(raw)

    raw.pop("surprise")
    raw["experiment"] = "mnist_experiment"
    with pytest.raises(RotatedConfigError, match="rotated_mnist"):
        RotatedExperimentConfig.from_mapping(raw)


def test_config_rejects_active_batch_wider_than_master_stream() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["samples_per_step"] = 9
    with pytest.raises(RotatedConfigError, match="stream_width"):
        RotatedExperimentConfig.from_mapping(raw)
