import json
from pathlib import Path

from mnist_experiment.plan4_command_center import (
    DEFAULT_CONFIGS,
    PHASE3_CONFIGS,
    _audit_completed,
    _load_configs,
    _run_state,
)
from src.initialization import replica_bundle_id, replica_design_hash
from src.schedules import resolve_schedule


def test_plan4_smoke_configs_are_paired_and_explicit() -> None:
    loaded = _load_configs(list(DEFAULT_CONFIGS))
    fixed = loaded[0][1]
    adaptive = loaded[1][1]
    schedule = resolve_schedule(fixed.data)

    assert fixed.controller.policy == "fixed_unified"
    assert adaptive.controller.policy == "optimal_plugin"
    assert fixed.estimator.method == adaptive.estimator.method == "ema"
    assert fixed.config_hash != adaptive.config_hash
    assert replica_design_hash(fixed) == replica_design_hash(adaptive)
    assert replica_bundle_id(fixed) == replica_bundle_id(adaptive)
    assert schedule.p_values[0] == 0.0
    assert schedule.p_values[-1] == 0.2
    assert fixed.data.samples_per_step * sum(schedule.p_values) == 1.0


def test_plan4_phase3_smoke_is_paired_and_varies_only_controller_treatment() -> None:
    loaded = _load_configs(list(PHASE3_CONFIGS))
    configs = [config for _, config in loaded]

    assert {config.schema_version for config in configs} == {17}
    assert {config.artifact_schema_version for config in configs} == {9}
    assert {config.metric_schema_version for config in configs} == {13}
    assert {config.estimator.method for config in configs} == {"ema"}
    assert {
        tuple(config.estimator.controller_methods or []) for config in configs
    } == {("low_rank_diagonal",)}
    assert {replica_design_hash(config) for config in configs} == {
        replica_design_hash(configs[0])
    }
    assert {replica_bundle_id(config) for config in configs} == {
        replica_bundle_id(configs[0])
    }
    assert [config.controller.policy for config in configs] == [
        "fixed_unified",
        "optimal_plugin",
        "optimal_plugin",
    ]
    assert [config.controller.risk_metric for config in configs] == [
        "euclidean",
        "euclidean",
        "fisher",
    ]


def test_plan4_run_state_checks_stored_configuration(tmp_path: Path) -> None:
    config = _load_configs([DEFAULT_CONFIGS[0]])[0][1]
    config = config.__class__.from_mapping(
        {**config.to_mapping(), "cache_root": str(tmp_path / "runs")}
    )

    state, path = _run_state(config)
    assert state == "missing"

    incomplete = path.parent / ".incomplete" / config.run_id
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text(
        json.dumps(config.to_mapping()), encoding="utf-8"
    )
    assert _run_state(config)[0] == "incomplete"

    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    assert _run_state(config)[0] == "invalid"


def test_plan4_completion_audit_requires_pairing_and_zero_hvps(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(2):
        run_path = tmp_path / f"run-{index}"
        run_path.mkdir()
        (run_path / "schedule_trajectory.json").write_text(
            json.dumps(
                {
                    "schedule_hash": "a" * 64,
                    "uniform_stream_hash": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        (run_path / "phase8_metrics.json").write_text(
            json.dumps(
                {
                    "condition_steps": [
                        {"hvp_count": 0, "same_pi_consumed": True}
                    ]
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "run_state": "completed",
                "run_id": f"run-{index}",
                "run_path": str(run_path),
            }
        )

    audit = _audit_completed(rows, expected_schedule_hash="a" * 64)

    assert audit["completed_runs"] == 2
    assert audit["total_hvp_count"] == 0
    assert audit["schedule_hash"] == "a" * 64
    assert audit["uniform_stream_hash"] == "b" * 64
