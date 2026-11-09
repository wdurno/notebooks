from dataclasses import replace

import pytest

from numerical_experiment.picar.config import PicarConfig, config_from_dict, load_config


def test_round_trip_and_pairing():
    config = PicarConfig()
    assert config_from_dict(config.to_dict()).config_hash == config.config_hash
    baseline = config.for_condition("replay_ewc")
    treatment = config.for_condition("stm_fd")
    assert baseline.learning.fd_strength == 0
    assert treatment.learning.fd_strength > 0
    assert baseline.model == treatment.model
    assert baseline.seeds(0) == treatment.seeds(0)
    assert len(set(config.seeds(0).values())) == len(config.seeds(0))
    assert config.seeds(1) != config.seeds(0)


@pytest.mark.parametrize("mutation", [
    {"typo": 1}, {"learning": {"pi": 0}}, {"learning": {"fd_strength": float("nan")}},
    {"runtime": {"threads": True}}, {"mode": "production"},
    {"conditions": ["stm_fd", "stm_fd"]}, {"model": {"top_layers": 0}},
    {"learning": {"retired_archive_replay": True}},
])
def test_invalid_configuration_rejected(mutation):
    with pytest.raises((ValueError, TypeError)):
        config_from_dict(mutation)


def test_hash_tracks_scientific_settings():
    config = PicarConfig()
    assert replace(config, transitions=13).config_hash != config.config_hash


def test_shipped_configs():
    for name in ("smoke", "plan19"):
        load_config(f"numerical_experiment/picar/configs/{name}.json").validate()
