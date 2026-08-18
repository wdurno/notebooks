import dataclasses
import json
from pathlib import Path

import pytest

from mnist_experiment.command_center import _execution_entries
from src.config import ExperimentConfig
from src.phase9 import (
    Phase9Bundle,
    Phase9Error,
    build_phase9_bundle,
    load_phase9_bundle,
    load_phase9_spec,
    parse_replica_indices,
    phase9_status_rows,
    prepare_phase9_bundle,
)
from src.results_analysis import (
    PHASE9_PAIRED_METRICS,
    phase9_paired_aggregates,
    phase9_paired_rows,
)


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "mnist_experiment" / "phase9_profiles.json"


def test_replica_index_parser_supports_ranges_and_defaults() -> None:
    assert parse_replica_indices(
        None, default_start=1, default_count=5
    ) == (1, 2, 3, 4, 5)
    assert parse_replica_indices(
        "5,1-3,3", default_start=0, default_count=1
    ) == (1, 2, 3, 5)

    with pytest.raises(Phase9Error, match="replicas"):
        parse_replica_indices("5-2", default_start=0, default_count=1)


def test_controller_screen_generates_one_shared_control_per_replica() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    manifest, configs = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("controller-screen",),
        replica_indices=(1, 2),
    )

    assert manifest["entry_count"] == 16
    assert manifest["replica_count"] == 2
    assert manifest["initialization_count"] == 2
    assert manifest["oracle_anchor_count"] == 2
    for replica_index in (1, 2):
        rows = [
            row
            for row in manifest["entries"]
            if row["replica_index"] == replica_index
        ]
        controls = [row for row in rows if row["kind"] == "control"]
        treatments = [row for row in rows if row["kind"] == "treatment"]
        assert len(controls) == 1
        assert len(treatments) == 7
        assert {row["control_run_id"] for row in treatments} == {
            controls[0]["run_id"]
        }
        assert len({row["replica_bundle_id"] for row in rows}) == 1
        assert sum(row["is_oracle_anchor"] for row in rows) == 1

    for entry in manifest["entries"]:
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        assert config.run_id == entry["run_id"]
        if entry["is_oracle_anchor"]:
            assert config.controller.reference_optimum_artifact is None
        else:
            assert config.controller.reference_optimum_artifact is not None


def test_anchor_only_execution_selects_one_reference_path_per_replica() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    manifest, _ = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("controller-screen",),
        replica_indices=(6, 7),
    )
    selected = _execution_entries(
        Phase9Bundle(path=Path("<test>"), manifest=manifest),
        oracle_anchors_only=True,
    )

    assert len(selected) == 2
    assert {row["replica_index"] for row in selected} == {6, 7}
    assert all(row["is_oracle_anchor"] for row in selected)


def test_data_screen_keeps_cell_specific_fixed_controls() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    manifest, _ = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("data-screen",),
        replica_indices=(1,),
    )

    controls = [row for row in manifest["entries"] if row["kind"] == "control"]
    treatments = [
        row for row in manifest["entries"] if row["kind"] == "treatment"
    ]
    assert len(controls) == len(treatments) == 4
    assert len({row["control_run_id"] for row in treatments}) == 4
    assert manifest["oracle_anchor_count"] == 4


def test_dense_confirmation_reuses_canonical_controller_anchor() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    dense_manifest, _ = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("dense-confirm",),
        replica_indices=(3,),
    )
    combined_manifest, _ = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("controller-screen", "dense-confirm"),
        replica_indices=(3,),
    )

    dense_only = next(
        row
        for row in dense_manifest["entries"]
        if row["profile"] == "dense-confirm" and row["kind"] == "treatment"
    )
    dense_combined = next(
        row
        for row in combined_manifest["entries"]
        if row["profile"] == "dense-confirm" and row["kind"] == "treatment"
    )

    assert dense_manifest["entry_count"] == 4
    assert {row["profile"] for row in dense_manifest["entries"]} == {
        "controller-screen",
        "dense-confirm",
    }
    assert dense_only["run_id"] == dense_combined["run_id"]
    assert dense_only["config_hash"] == dense_combined["config_hash"]
    assert dense_only["oracle_anchor_run_id"] == dense_combined[
        "oracle_anchor_run_id"
    ]


def test_lfu_isolation_uses_explicit_paired_comparisons() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    manifest, configs = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("lfu-isolation",),
        replica_indices=(4,),
    )
    rows = {
        (row["profile"], row["cell"]): row
        for row in manifest["entries"]
    }

    assert manifest["entry_count"] == 9
    assert manifest["oracle_anchor_count"] == 1
    assert set(rows) == {
        ("controller-screen", "center"),
        ("controller-screen", "trend-h-010"),
        ("controller-screen", "control"),
        ("lfu-isolation", "fixed-005-no-lfu"),
        ("lfu-isolation", "fixed-010-no-lfu"),
        ("lfu-isolation", "fixed-010-ac-only"),
        ("lfu-isolation", "fixed-010-full-lfu"),
        ("lfu-isolation", "adaptive-h010-no-lfu"),
        ("lfu-isolation", "adaptive-h010-ac-only"),
    }

    fixed_005_full = rows[("controller-screen", "control")]
    adaptive_h010_full = rows[("controller-screen", "trend-h-010")]
    fixed_010_none = rows[("lfu-isolation", "fixed-010-no-lfu")]
    assert rows[("lfu-isolation", "fixed-005-no-lfu")][
        "control_run_id"
    ] == fixed_005_full["run_id"]
    for cell in ("fixed-010-ac-only", "fixed-010-full-lfu"):
        assert rows[("lfu-isolation", cell)]["control_run_id"] == (
            fixed_010_none["run_id"]
        )
    for cell in ("adaptive-h010-no-lfu", "adaptive-h010-ac-only"):
        assert rows[("lfu-isolation", cell)]["control_run_id"] == (
            adaptive_h010_full["run_id"]
        )

    expected = {
        "fixed-005-no-lfu": ("fixed_unified", "ema", 0.05),
        "fixed-010-no-lfu": ("fixed_unified", "ema", 0.10),
        "fixed-010-ac-only": ("fixed_unified", "ac_only", 0.10),
        "fixed-010-full-lfu": ("fixed_unified", "full_lfu", 0.10),
        "adaptive-h010-no-lfu": ("optimal_plugin", "ema", 0.50),
        "adaptive-h010-ac-only": ("optimal_plugin", "ac_only", 0.50),
    }
    for cell, (policy, method, fixed_pi) in expected.items():
        entry = rows[("lfu-isolation", cell)]
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        assert config.controller.policy == policy
        assert config.estimator.method == method
        assert config.controller.fixed_pi == fixed_pi


def test_adaptation_screen_reuses_no_lfu_h010_and_adds_three_conditions() -> None:
    spec = load_phase9_spec(SPEC_PATH)
    manifest, configs = build_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("adaptation-screen",),
        replica_indices=(4,),
    )
    rows = {
        (row["profile"], row["cell"]): row
        for row in manifest["entries"]
    }

    assert manifest["entry_count"] == 7
    assert manifest["oracle_anchor_count"] == 1
    assert set(rows) == {
        ("controller-screen", "center"),
        ("controller-screen", "trend-h-010"),
        ("controller-screen", "control"),
        ("lfu-isolation", "adaptive-h010-no-lfu"),
        ("adaptation-screen", "adaptive-h020-no-lfu"),
        ("adaptation-screen", "adaptive-h040-no-lfu"),
        ("adaptation-screen", "fixed-100-no-ewc"),
    }

    baseline = rows[("lfu-isolation", "adaptive-h010-no-lfu")]
    for cell in (
        "adaptive-h020-no-lfu",
        "adaptive-h040-no-lfu",
        "fixed-100-no-ewc",
    ):
        assert rows[("adaptation-screen", cell)]["control_run_id"] == (
            baseline["run_id"]
        )

    expected = {
        "adaptive-h020-no-lfu": ("optimal_plugin", 0.2, 0.5, 0.05, 0.95),
        "adaptive-h040-no-lfu": ("optimal_plugin", 0.4, 0.5, 0.05, 0.95),
        "fixed-100-no-ewc": ("fixed_unified", 0.2, 1.0, 1.0, 1.0),
    }
    for cell, values in expected.items():
        entry = rows[("adaptation-screen", cell)]
        config = ExperimentConfig.from_mapping(configs[entry["entry_id"]])
        assert config.controller.policy == values[0]
        assert config.controller.trend_half_life_p == values[1]
        assert config.controller.fixed_pi == values[2]
        assert config.controller.pi_min == values[3]
        assert config.controller.pi_max == values[4]
        assert config.estimator.method == "ema"


def test_prepared_bundle_is_immutable_and_strict(tmp_path: Path) -> None:
    spec = dataclasses.replace(
        load_phase9_spec(SPEC_PATH),
        bundle_root=str(tmp_path / "bundles"),
    )
    first = prepare_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("smoke",),
        replica_indices=(9876,),
    )
    second = prepare_phase9_bundle(
        spec,
        REPO_ROOT,
        profile_names=("smoke",),
        replica_indices=(9876,),
    )

    assert first.path == second.path
    assert first.manifest == second.manifest
    assert (first.path / "COMPLETED").is_file()
    statuses = phase9_status_rows(first, REPO_ROOT)
    assert {row["run_state"] for row in statuses} == {"missing"}

    config_path = first.path / first.entries[0]["config_file"]
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["optimizer"]["inner_steps"] += 1
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(Phase9Error, match="mismatch"):
        load_phase9_bundle(first.path)


def _summary_row(
    replica: int,
    *,
    kind: str,
    run_id: str,
    control_run_id: str | None,
    shift: float,
) -> dict:
    row = {
        "profile": "controller-screen",
        "cell": "center" if kind == "treatment" else "control",
        "kind": kind,
        "method": "low_rank_diagonal_r8",
        "replica_index": replica,
        "replica_id": f"replica-{replica:04d}",
        "replica_seed": replica,
        "replica_bundle_id": f"bundle-{replica}",
        "run_id": run_id,
        "run_path": f"runs/{run_id}",
        "control_run_id": control_run_id,
        "factors": {"pi_min": 0.05},
    }
    row.update({metric: 1.0 + shift for metric in PHASE9_PAIRED_METRICS})
    return row


def test_phase9_pairing_uses_replicas_as_the_statistical_unit() -> None:
    rows = []
    for replica, shift in ((1, 0.1), (2, 0.3)):
        control_id = f"control-{replica}"
        rows.extend(
            (
                _summary_row(
                    replica,
                    kind="control",
                    run_id=control_id,
                    control_run_id=None,
                    shift=0.0,
                ),
                _summary_row(
                    replica,
                    kind="treatment",
                    run_id=f"treatment-{replica}",
                    control_run_id=control_id,
                    shift=shift,
                ),
            )
        )

    pairs = phase9_paired_rows(rows)
    aggregates = phase9_paired_aggregates(pairs)
    accuracy = next(
        row for row in aggregates if row["metric"] == "nine_accuracy_auc"
    )

    assert len(pairs) == 2
    assert accuracy["replica_count"] == 2
    assert accuracy["mean_paired_effect"] == pytest.approx(0.2)
    assert accuracy["standard_error"] == pytest.approx(0.1)
    assert accuracy["ci95_low"] < 0.0 < accuracy["ci95_high"]
    assert accuracy["initial_replica_target_met"] is False


def test_phase9_pairing_can_use_an_explicit_treatment_as_control() -> None:
    baseline = _summary_row(
        1,
        kind="treatment",
        run_id="fixed-010-no-lfu",
        control_run_id=None,
        shift=0.0,
    )
    treatment = _summary_row(
        1,
        kind="treatment",
        run_id="fixed-010-full-lfu",
        control_run_id=baseline["run_id"],
        shift=0.25,
    )

    pairs = phase9_paired_rows((baseline, treatment))

    assert len(pairs) == 1
    assert pairs[0]["control_run_id"] == baseline["run_id"]
    assert pairs[0]["delta_nine_accuracy_auc"] == pytest.approx(0.25)
