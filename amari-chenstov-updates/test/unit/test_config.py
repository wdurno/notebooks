import json
from pathlib import Path

import pytest

from src.config import ConfigError, ExperimentConfig, load_config
from src.initialization import replica_bundle_id


SMOKE_CONFIG = (
    Path(__file__).parents[2] / "mnist_experiment" / "configs" / "smoke.json"
)
RIDGE_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase4_ridge_smoke.json"
)
CONTROLLER_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_controller_smoke.json"
)
CONVERGENCE_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_convergence.json"
)
ORACLE_CONVERGENCE_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_oracle_convergence.json"
)
CORRECTED_PLUGIN_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_corrected_plugin.json"
)
CORRECTED_ORACLE_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_corrected_oracle.json"
)
K100_PLUGIN_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k100_plugin.json"
)
K100_FIXED_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k100_fixed.json"
)
K100_ORACLE_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k100_oracle.json"
)
K50_PLUGIN_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k50_plugin_rank8.json"
)
K50_FIXED_CONFIG = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k50_fixed_rank8.json"
)


def test_smoke_config_loads_and_has_stable_hash() -> None:
    config = load_config(SMOKE_CONFIG)
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    reordered = dict(reversed(tuple(raw.items())))
    same_config = ExperimentConfig.from_mapping(reordered)

    assert config == same_config
    assert config.config_hash == same_config.config_hash
    assert config.run_id.endswith(config.config_hash[:16])
    assert config.config_hash == (
        "072faff78d1bf6cd2c1c395f9eae47a55f5aca0e4b536ebbfce6d3c75e10871d"
    )
    assert config.data.num_p_steps == 3
    assert config.to_mapping() == raw
    assert config.estimator.ridge_half_life_steps is None


def test_version_five_requires_and_loads_explicit_ridge_fields() -> None:
    config = load_config(RIDGE_CONFIG)
    raw = json.loads(RIDGE_CONFIG.read_text(encoding="utf-8"))

    assert config.to_mapping() == raw
    assert config.estimator.ridge_half_life_steps == 2.0
    assert config.estimator.ridge_amplitude_epsilon == 1e-6
    assert config.estimator.ridge_coherence_threshold == 0.75

    raw["estimator"].pop("ridge_amplitude_epsilon")
    with pytest.raises(ConfigError, match="missing"):
        ExperimentConfig.from_mapping(raw)


def test_partial_ridge_configuration_is_rejected() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["estimator"]["ridge_half_life_steps"] = 2.0

    with pytest.raises(ConfigError, match="all null or all supplied"):
        ExperimentConfig.from_mapping(raw)


def test_unknown_configuration_keys_are_rejected() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["surprise"] = True

    with pytest.raises(ConfigError, match="unknown"):
        ExperimentConfig.from_mapping(raw)


def test_explicit_schedule_round_trips_without_changing_legacy_hash() -> None:
    legacy = load_config(CONTROLLER_CONFIG)
    legacy_hash = legacy.config_hash
    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["schedule"] = {
        "kind": "normalized_logistic",
        "p_start": 0.0,
        "p_end": 0.2,
        "center_fraction": 0.5,
        "steepness": 16.0,
    }

    scheduled = ExperimentConfig.from_mapping(raw)

    assert scheduled.to_mapping() == raw
    assert scheduled.config_hash != legacy_hash
    assert load_config(CONTROLLER_CONFIG).config_hash == legacy_hash


def test_explicit_schedule_is_strictly_parsed() -> None:
    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["schedule"] = {
        "kind": "normalized_logistic",
        "p_start": 0.0,
        "p_end": 0.2,
        "center_fraction": 0.5,
        "steepness": 16.0,
        "surprise": True,
    }

    with pytest.raises(ConfigError, match="data.schedule.*unknown"):
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


def test_version_six_requires_a_rank_grid_field() -> None:
    raw = json.loads(RIDGE_CONFIG.read_text(encoding="utf-8"))
    raw["schema_version"] = 6

    with pytest.raises(ConfigError, match="low_rank_grid"):
        ExperimentConfig.from_mapping(raw)

    raw["estimator"]["low_rank_grid"] = None
    config = ExperimentConfig.from_mapping(raw)
    assert config.estimator.low_rank_grid is None
    assert config.to_mapping() == raw


def test_low_rank_grid_contract_is_validated() -> None:
    raw = json.loads(RIDGE_CONFIG.read_text(encoding="utf-8"))
    raw["schema_version"] = 6
    raw["estimator"].update(
        {
            "representation": "low_rank_diagonal",
            "low_rank": 8,
            "low_rank_grid": [0, 4, 8],
        }
    )

    config = ExperimentConfig.from_mapping(raw)
    assert config.estimator.low_rank_grid == [0, 4, 8]

    raw["estimator"]["low_rank_grid"] = [0, 8, 4]
    with pytest.raises(ConfigError, match="unique increasing"):
        ExperimentConfig.from_mapping(raw)

    raw["estimator"]["low_rank_grid"] = [0, 4]
    with pytest.raises(ConfigError, match="largest rank"):
        ExperimentConfig.from_mapping(raw)


def test_schema_seven_unified_controller_round_trips_without_ema_gain() -> None:
    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_mapping(raw)

    assert config.schema_version == 7
    assert config.estimator.ema_gain is None
    assert config.controller.pi_min == 0.05
    assert config.controller.trend_half_life_p == 0.2
    assert config.to_mapping() == raw


def test_schema_seven_rejects_decoupled_and_legacy_controller_fields() -> None:
    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["estimator"]["ema_gain"] = 0.25
    with pytest.raises(ConfigError, match="must not specify ema_gain"):
        ExperimentConfig.from_mapping(raw)

    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["controller"]["damping"] = 1e-6
    with pytest.raises(ConfigError, match="must not specify damping"):
        ExperimentConfig.from_mapping(raw)


def test_schema_seven_requires_new_artifact_versions_and_unified_policy() -> None:
    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["artifact_schema_version"] = 1
    with pytest.raises(ConfigError, match="artifact_schema_version must be 2"):
        ExperimentConfig.from_mapping(raw)

    raw = json.loads(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    raw["controller"]["policy"] = "fixed"
    with pytest.raises(ConfigError, match="unified policy"):
        ExperimentConfig.from_mapping(raw)


def test_schema_eight_requires_and_round_trips_convergence_contract() -> None:
    raw = json.loads(CONVERGENCE_CONFIG.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_mapping(raw)

    assert config.schema_version == 8
    assert config.reference.convergence_min_chunks == 8
    assert config.reference.calibration_min_fits == 8
    assert config.reference.calibration_max_fits == 32
    assert config.to_mapping() == raw

    raw["reference"].pop("convergence_sigma")
    with pytest.raises(ConfigError, match="missing"):
        ExperimentConfig.from_mapping(raw)


def test_schema_eight_rejects_an_insufficient_fisher_budget() -> None:
    raw = json.loads(CONVERGENCE_CONFIG.read_text(encoding="utf-8"))
    raw["reference"]["sample_size"] = 2048
    raw["reference"]["convergence_sample_sizes"] = [2048]

    with pytest.raises(ConfigError, match="at least"):
        ExperimentConfig.from_mapping(raw)


def test_schema_eight_plugin_and_oracle_configs_are_replica_paired() -> None:
    plugin = load_config(CONVERGENCE_CONFIG)
    oracle = load_config(ORACLE_CONVERGENCE_CONFIG)

    assert plugin.controller.policy == "optimal_plugin"
    assert oracle.controller.policy == "optimal_oracle"
    assert replica_bundle_id(plugin) == replica_bundle_id(oracle)


def test_schema_nine_corrected_configs_share_the_exact_reference_path() -> None:
    plugin = load_config(CORRECTED_PLUGIN_CONFIG)
    oracle = load_config(CORRECTED_ORACLE_CONFIG)

    assert plugin.schema_version == 9
    assert plugin.artifact_schema_version == 4
    assert plugin.metric_schema_version == 5
    assert plugin.controller.reference_optimum_artifact is not None
    assert (
        plugin.controller.reference_optimum_artifact
        == oracle.controller.reference_optimum_artifact
    )
    assert replica_bundle_id(plugin) == replica_bundle_id(oracle)


def test_schema_nine_oracle_requires_an_external_reference_path() -> None:
    raw = json.loads(CORRECTED_ORACLE_CONFIG.read_text(encoding="utf-8"))
    raw["controller"]["reference_optimum_artifact"] = None

    with pytest.raises(ConfigError, match="reference_optimum_artifact"):
        ExperimentConfig.from_mapping(raw)


def test_schema_ten_round_trips_explicit_lbfgs_controls() -> None:
    raw = json.loads(K100_PLUGIN_CONFIG.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_mapping(raw)

    assert config.schema_version == 10
    assert config.metric_schema_version == 6
    assert config.optimizer.name == "lbfgs"
    assert config.optimizer.inner_steps == 100
    assert config.optimizer.lbfgs_history_size == 20
    assert config.optimizer.lbfgs_line_search_fn == "strong_wolfe"
    assert config.estimator.controller_methods == [
        "dense",
        "diagonal",
        "low_rank_diagonal",
    ]
    assert config.to_mapping() == raw


def test_schema_eleven_versions_exposure_and_calibration_metrics() -> None:
    raw = json.loads(K100_PLUGIN_CONFIG.read_text(encoding="utf-8"))
    raw["schema_version"] = 11
    raw["metric_schema_version"] = 7

    config = ExperimentConfig.from_mapping(raw)

    assert config.schema_version == 11
    assert config.artifact_schema_version == 4
    assert config.metric_schema_version == 7
    assert config.to_mapping() == raw


def test_schema_twelve_versions_nine_classification_metrics() -> None:
    raw = json.loads(K100_PLUGIN_CONFIG.read_text(encoding="utf-8"))
    raw["schema_version"] = 12
    raw["metric_schema_version"] = 8

    config = ExperimentConfig.from_mapping(raw)

    assert config.schema_version == 12
    assert config.artifact_schema_version == 4
    assert config.metric_schema_version == 8
    assert config.to_mapping() == raw


def test_schema_ten_principal_configs_are_replica_paired() -> None:
    configs = [
        load_config(path)
        for path in (K100_PLUGIN_CONFIG, K100_FIXED_CONFIG, K100_ORACLE_CONFIG)
    ]

    assert {config.controller.policy for config in configs} == {
        "optimal_plugin",
        "fixed_unified",
        "optimal_oracle",
    }
    assert len({replica_bundle_id(config) for config in configs}) == 1
    assert len(
        {config.controller.reference_optimum_artifact for config in configs}
    ) == 1


def test_schema_ten_rejects_duplicate_controller_methods() -> None:
    raw = json.loads(K100_PLUGIN_CONFIG.read_text(encoding="utf-8"))
    raw["estimator"]["controller_methods"] = ["dense", "dense"]

    with pytest.raises(ConfigError, match="controller_methods"):
        ExperimentConfig.from_mapping(raw)


def test_schema_ten_rank_sensitivity_selects_only_rank_eight() -> None:
    from mnist_experiment.run_controller import _phase8_methods, _validate_config

    path = (
        Path(__file__).parents[2]
        / "mnist_experiment"
        / "configs"
        / "phase8_gpu_k50_plugin_rank8.json"
    )
    config = load_config(path)
    selected_rank = _validate_config(config)

    assert _phase8_methods(config, selected_rank) == (
        "low_rank_diagonal_r8",
    )


def test_schema_ten_k50_plugin_and_fixed_configs_are_replica_paired() -> None:
    plugin = load_config(K50_PLUGIN_CONFIG)
    fixed = load_config(K50_FIXED_CONFIG)

    assert plugin.controller.policy == "optimal_plugin"
    assert fixed.controller.policy == "fixed_unified"
    assert fixed.controller.fixed_pi == 0.05
    assert plugin.estimator.controller_methods == ["low_rank_diagonal"]
    assert fixed.estimator.controller_methods == ["low_rank_diagonal"]
    assert replica_bundle_id(plugin) == replica_bundle_id(fixed)
    assert (
        plugin.controller.reference_optimum_artifact
        == fixed.controller.reference_optimum_artifact
    )
