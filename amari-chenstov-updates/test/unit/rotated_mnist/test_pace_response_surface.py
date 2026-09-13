from pathlib import Path

import pytest
import torch

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.pace_control.response_surface import (
    analyze_response_surface,
    regret_interval,
)
from mnist_experiment.rotated_mnist.pace_control.response_surface_config import (
    load_response_surface_config,
)


REPO_ROOT = Path(__file__).parents[3]
CONFIG_ROOT = (
    REPO_ROOT / "mnist_experiment" / "rotated_mnist" / "pace_control" / "configs"
)


def test_response_surface_configs_freeze_stages_and_prerequisites() -> None:
    smoke = load_response_surface_config(CONFIG_ROOT / "phase1d_smoke.json")
    coarse = load_response_surface_config(CONFIG_ROOT / "phase1d_coarse.json")
    full = load_response_surface_config(CONFIG_ROOT / "phase1d_full.json")
    expansion = load_response_surface_config(CONFIG_ROOT / "phase1d_expansion.json")

    assert smoke.prerequisite_run_path is None
    assert coarse.prerequisite_run_id == smoke.run_id
    assert full.prerequisite_run_id == coarse.run_id
    assert expansion.prerequisite_run_id == full.run_id
    assert coarse.replicate_count == 16
    assert full.anchor_steps == (20, 40, 60, 100)
    assert expansion.replicate_count == 128


def test_response_surface_config_rejects_treatment_drift() -> None:
    config = load_response_surface_config(CONFIG_ROOT / "phase1d_smoke.json")
    value = config.to_mapping()
    value["pace_degrees"][2] = 0.8
    with pytest.raises(RotatedConfigError, match="grid drifted"):
        type(config).from_mapping(value)


def test_paired_regret_interval_cancels_common_batch_noise() -> None:
    nll = torch.tensor(
        [
            [1.0, 0.9, 0.8],
            [2.0, 1.9, 1.8],
            [3.0, 2.9, 2.8],
            [4.0, 3.9, 3.8],
        ],
        dtype=torch.float64,
    )
    result = regret_interval(
        nll,
        (0.025, 0.05, 0.10),
        fixed_pi=0.05,
        confidence_level=0.95,
        bootstrap_replicates=500,
        seed=7,
    )
    assert result["point_regret"] == pytest.approx(0.1)
    assert result["lower_regret"] == pytest.approx(0.1)
    assert result["upper_regret"] == pytest.approx(0.1)


def _synthetic_records(*, flat: bool) -> tuple[list[dict], list[dict]]:
    config = load_response_surface_config(CONFIG_ROOT / "phase1d_full.json")
    directions = {20: 1, 40: -1, 60: -1, 100: 1}
    optima = {
        0.0: 0.0125,
        0.375: 0.025,
        0.75: 0.0375,
        1.125: 0.05,
        1.5: 0.075,
        3.0: 0.15,
    }
    records = []
    for anchor in config.anchor_steps:
        for pace in config.pace_degrees:
            for replicate in range(config.replicate_count):
                common = (replicate % 5) * 0.0001
                for pi in config.pi_values:
                    treatment = 0.0 if flat else 5.0 * (pi - optima[pace]) ** 2
                    records.append(
                        {
                            "anchor_step": anchor,
                            "pace_degrees": pace,
                            "replicate_index": replicate,
                            "pi": pi,
                            "next_nll": 1.0 + common + treatment,
                            "failure": None,
                        }
                    )
    anchors = [
        {"step": anchor, "direction_to_next": directions[anchor]}
        for anchor in config.anchor_steps
    ]
    return records, anchors


def test_response_surface_gate_accepts_resolved_interior_crossing() -> None:
    config = load_response_surface_config(CONFIG_ROOT / "phase1d_full.json")
    records, anchors = _synthetic_records(flat=False)
    _, gate, summary = analyze_response_surface(records, anchors, config)
    assert summary["gate_pass"]
    assert summary["resolved_crossing_both_directions"]
    assert summary["interior_upper_competitive_anchor_count"] == 4
    assert all(gate["summary"]["gate_checks"].values())


def test_response_surface_gate_rejects_uninformative_flat_surface() -> None:
    config = load_response_surface_config(CONFIG_ROOT / "phase1d_full.json")
    records, anchors = _synthetic_records(flat=True)
    _, _, summary = analyze_response_surface(records, anchors, config)
    assert not summary["gate_pass"]
    assert not summary["resolved_crossing_both_directions"]
    assert not summary["expansion_recommended"]
