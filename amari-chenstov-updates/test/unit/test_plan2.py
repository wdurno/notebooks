from pathlib import Path

import pytest

from src.config import ExperimentConfig
from src.plan2 import (
    Plan2Error,
    build_plan2_bundle,
    load_plan2_spec,
    parse_samples_per_step,
)

REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "mnist_experiment" / "plan2_profiles.json"
PHASE4_HALF_LIFE_SPEC_PATH = (
    REPO_ROOT / "mnist_experiment" / "plan2_phase4_half_life.json"
)
PHASE4_PI_MIN_SPEC_PATH = (
    REPO_ROOT / "mnist_experiment" / "plan2_phase4_pi_min.json"
)
PHASE4_FIXED005_SPEC_PATH = (
    REPO_ROOT / "mnist_experiment" / "plan2_phase4_fixed005.json"
)
PHASE5_CONFIRMATION_SPEC_PATH = (
    REPO_ROOT / "mnist_experiment" / "plan2_phase5_confirmation.json"
)


def test_samples_per_step_parser_is_explicit_and_sorted() -> None:
    assert parse_samples_per_step(None, (1, 2, 4)) == (1, 2, 4)
    assert parse_samples_per_step("8,1,2,2", (4,)) == (1, 2, 8)
    with pytest.raises(Plan2Error, match="positive"):
        parse_samples_per_step("0,1", (1,))


def test_plan2_build_has_only_three_named_conditions_per_m_and_replica() -> None:
    spec = load_plan2_spec(SPEC_PATH)
    manifest, configs = build_plan2_bundle(
        spec,
        REPO_ROOT,
        samples_per_step=(1, 2),
        replica_indices=(1, 2),
    )

    assert manifest["entry_count"] == 12
    assert manifest["condition_count"] == 3
    assert manifest["derived_bundle_count"] == 4
    assert manifest["new_initialization_fit_count"] == 0
    assert manifest["selection"]["samples_per_step"] == [1, 2]
    assert manifest["selection"]["generated_config_schema_version"] == 12
    assert manifest["selection"]["generated_metric_schema_version"] == 8
    for replica in (1, 2):
        replica_rows = [
            row for row in manifest["entries"] if row["replica_index"] == replica
        ]
        assert len({row["master_replica_bundle_id"] for row in replica_rows}) == 1
        assert len({row["reference_source_run_id"] for row in replica_rows}) == 1
        for sample_size in (1, 2):
            rows = [
                row
                for row in replica_rows
                if row["samples_per_step"] == sample_size
            ]
            assert {row["condition"] for row in rows} == {
                "no-ewc-pi100",
                "fixed-ewc-pi010",
                "adaptive-ewc-h010",
            }
            assert len({row["replica_bundle_id"] for row in rows}) == 1
            control = next(row for row in rows if row["kind"] == "control")
            assert {
                row["control_run_id"]
                for row in rows
                if row["kind"] == "treatment"
            } == {control["run_id"]}

    for entry in manifest["entries"]:
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        assert config.schema_version == 12
        assert config.metric_schema_version == 8
        assert config.data.samples_per_step == entry["samples_per_step"]
        assert config.estimator.method == "ema"
        assert config.estimator.controller_methods == ["low_rank_diagonal"]
        assert config.controller.reference_optimum_artifact == entry[
            "reference_optimum_artifact"
        ]
        if entry["condition"] == "no-ewc-pi100":
            assert config.controller.fixed_pi == 1.0
            assert config.controller.pi_min == config.controller.pi_max == 1.0
        elif entry["condition"] == "fixed-ewc-pi010":
            assert config.controller.policy == "fixed_unified"
            assert config.controller.fixed_pi == 0.1
        else:
            assert config.controller.policy == "optimal_plugin"
            assert config.controller.trend_half_life_p == 0.1


def test_plan2_rejects_m_larger_than_master_stream() -> None:
    spec = load_plan2_spec(SPEC_PATH)
    with pytest.raises(Plan2Error, match="exceeds"):
        build_plan2_bundle(
            spec,
            REPO_ROOT,
            samples_per_step=(129,),
            replica_indices=(1,),
        )


def test_plan2_cost_model_preserves_fixed_overhead() -> None:
    spec = load_plan2_spec(SPEC_PATH)
    manifest, _ = build_plan2_bundle(
        spec,
        REPO_ROOT,
        samples_per_step=(1, 2),
        replica_indices=(1,),
    )
    by_m = {}
    for row in manifest["entries"]:
        by_m.setdefault(row["samples_per_step"], row["estimated_trajectory_seconds"])

    assert by_m[2] > by_m[1]
    assert by_m[2] < 2.0 * by_m[1]
    assert manifest["cost_model"]["basis"] == (
        "fixed_overhead_plus_optimizer_consumed_observations"
    )


def test_phase4_half_life_screen_contains_only_missing_axial_conditions() -> None:
    phase4_spec = load_plan2_spec(PHASE4_HALF_LIFE_SPEC_PATH)
    phase4_manifest, configs = build_plan2_bundle(
        phase4_spec,
        REPO_ROOT,
        samples_per_step=(8,),
        replica_indices=(1,),
    )

    assert phase4_manifest["entry_count"] == 3
    assert phase4_manifest["condition_count"] == 3
    assert {row["control_run_id"] for row in phase4_manifest["entries"]} == {None}

    half_lives = {}
    for entry in phase4_manifest["entries"]:
        if entry["condition"].startswith("adaptive-ewc"):
            config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
            half_lives[entry["condition"]] = config.controller.trend_half_life_p
    assert half_lives == {
        "adaptive-ewc-h005": 0.05,
        "adaptive-ewc-h020": 0.2,
        "adaptive-ewc-h040": 0.4,
    }


def test_plan2_treatment_only_extension_preserves_derived_stream_count() -> None:
    spec = load_plan2_spec(PHASE4_HALF_LIFE_SPEC_PATH)
    manifest, _ = build_plan2_bundle(
        spec,
        REPO_ROOT,
        samples_per_step=(8,),
        replica_indices=(1, 2),
    )

    assert manifest["derived_bundle_count"] == 2
    assert manifest["new_initialization_fit_count"] == 0


def test_phase4_pi_min_screen_is_axial_at_selected_half_life() -> None:
    spec = load_plan2_spec(PHASE4_PI_MIN_SPEC_PATH)
    manifest, configs = build_plan2_bundle(
        spec,
        REPO_ROOT,
        samples_per_step=(8,),
        replica_indices=(1,),
    )

    assert manifest["entry_count"] == 2
    settings = {}
    for entry in manifest["entries"]:
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        settings[entry["condition"]] = (
            config.controller.trend_half_life_p,
            config.controller.pi_min,
        )
    assert settings == {
        "adaptive-ewc-h020-pimin001": (0.2, 0.01),
        "adaptive-ewc-h020-pimin010": (0.2, 0.1),
    }


def test_phase4_fixed005_control_is_explicit_and_isolated() -> None:
    spec = load_plan2_spec(PHASE4_FIXED005_SPEC_PATH)
    manifest, configs = build_plan2_bundle(
        spec,
        REPO_ROOT,
        samples_per_step=(8,),
        replica_indices=(1,),
    )

    assert manifest["entry_count"] == 1
    entry = manifest["entries"][0]
    config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
    assert entry["condition"] == "fixed-ewc-pi005"
    assert config.controller.policy == "fixed_unified"
    assert config.controller.fixed_pi == 0.05
    assert config.controller.pi_min == 0.05


def test_phase5_confirmation_uses_four_paired_fresh_replica_conditions() -> None:
    spec = load_plan2_spec(PHASE5_CONFIRMATION_SPEC_PATH)
    manifest, configs = build_plan2_bundle(spec, REPO_ROOT)

    assert manifest["selection"]["samples_per_step"] == [8]
    assert manifest["selection"]["replica_indices"] == [6, 7, 8, 9, 10]
    assert manifest["entry_count"] == 20
    assert manifest["replica_count"] == 5
    assert manifest["condition_count"] == 4
    assert manifest["derived_bundle_count"] == 5
    for replica in range(6, 11):
        rows = [
            row for row in manifest["entries"] if row["replica_index"] == replica
        ]
        control = next(row for row in rows if row["condition"] == "no-ewc-pi100")
        assert {row["control_run_id"] for row in rows if row["kind"] == "treatment"} == {
            control["run_id"]
        }

    settings = {}
    for entry in manifest["entries"]:
        if entry["replica_index"] != 6:
            continue
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        settings[entry["condition"]] = (
            config.controller.policy,
            config.controller.fixed_pi,
            config.controller.pi_min,
            config.controller.trend_half_life_p,
            config.estimator.method,
            config.data.samples_per_step,
        )
    assert settings == {
        "no-ewc-pi100": ("fixed_unified", 1.0, 1.0, 0.2, "ema", 8),
        "fixed-ewc-pi005": ("fixed_unified", 0.05, 0.05, 0.2, "ema", 8),
        "adaptive-ewc-h020": ("optimal_plugin", 0.5, 0.05, 0.2, "ema", 8),
        "fixed-ewc-pi010": ("fixed_unified", 0.1, 0.1, 0.2, "ema", 8),
    }
