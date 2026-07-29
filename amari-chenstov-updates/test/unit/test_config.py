import json
from pathlib import Path

import pytest

from src.config import ConfigError, ExperimentConfig, load_config


SMOKE_CONFIG = (
    Path(__file__).parents[2] / "mnist_experiment" / "configs" / "smoke.json"
)


def test_smoke_config_loads_and_has_stable_hash() -> None:
    config = load_config(SMOKE_CONFIG)
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    reordered = dict(reversed(tuple(raw.items())))
    same_config = ExperimentConfig.from_mapping(reordered)

    assert config == same_config
    assert config.config_hash == same_config.config_hash
    assert config.run_id.endswith(config.config_hash[:16])
    assert config.data.num_p_steps == 3


def test_unknown_configuration_keys_are_rejected() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["surprise"] = True

    with pytest.raises(ConfigError, match="unknown"):
        ExperimentConfig.from_mapping(raw)


def test_invalid_representation_rank_is_rejected() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["estimator"]["low_rank"] = 2

    with pytest.raises(ConfigError, match="must be null"):
        ExperimentConfig.from_mapping(raw)


def test_nonfinite_values_are_rejected_before_hashing() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["estimator"]["ema_gain"] = float("nan")

    with pytest.raises(ConfigError, match="ema_gain"):
        ExperimentConfig.from_mapping(raw)

    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["optimizer"]["ewc_strength"] = float("nan")

    with pytest.raises(ConfigError, match="ewc_strength"):
        ExperimentConfig.from_mapping(raw)


def test_boolean_is_not_accepted_as_an_integer() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["samples_per_step"] = True

    with pytest.raises(ConfigError, match="samples_per_step"):
        ExperimentConfig.from_mapping(raw)
