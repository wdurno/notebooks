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
