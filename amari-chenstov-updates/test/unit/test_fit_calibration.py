import json
from pathlib import Path

import pytest
import torch

from src.artifacts import RunStore
from src.config import ConfigError
from src.fit_calibration import (
    FitCalibrationConfig,
    displacement_comparison,
    load_fit_calibration_config,
)


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_fit_calibration.json"
)
LBFGS_CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_fit_calibration_lbfgs.json"
)


def test_fit_calibration_config_round_trips_and_hashes_stably() -> None:
    config = load_fit_calibration_config(CONFIG_PATH)
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config.to_mapping() == raw
    assert config.checkpoint_steps == [0, 50, 99]
    assert config.inner_step_budgets == [20, 50, 100, 200]
    assert config.run_id.endswith(config.config_hash[:16])


def test_schema_two_fit_calibration_records_lbfgs_controls() -> None:
    config = load_fit_calibration_config(LBFGS_CONFIG_PATH)
    raw = json.loads(LBFGS_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config.schema_version == 2
    assert config.metric_schema_version == 2
    assert config.optimizer is not None
    assert config.optimizer.name == "lbfgs"
    assert config.optimizer.lbfgs_line_search_fn == "strong_wolfe"
    assert config.to_mapping() == raw

@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_steps", [50, 0]),
        ("checkpoint_steps", [0, 0]),
        ("inner_step_budgets", [20, 10]),
        ("inner_step_budgets", [0, 20]),
    ],
)
def test_fit_calibration_config_rejects_unpaired_grids(
    field: str,
    value: list[int],
) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(ConfigError, match=field):
        FitCalibrationConfig.from_mapping(raw)


def test_displacement_comparison_reports_distance_scale_and_angle() -> None:
    estimate = torch.tensor([1.0, 1.0], dtype=torch.float64)
    reference = torch.tensor([2.0, 0.0], dtype=torch.float64)

    result = displacement_comparison(estimate, reference)

    assert result["distance"] == pytest.approx(2.0**0.5)
    assert result["relative_distance_to_reference"] == pytest.approx(2.0**-0.5)
    assert result["cosine"] == pytest.approx(2.0**-0.5)


def test_fit_calibration_config_uses_immutable_run_lifecycle(
    tmp_path: Path,
) -> None:
    config = load_fit_calibration_config(CONFIG_PATH)
    session = RunStore(tmp_path).begin(
        config,
        Path(__file__).parents[2],
    )
    session.write_json("fit_calibration_metrics.json", {"complete": True})

    destination = session.complete(["fit_calibration_metrics.json"])

    assert destination.name == config.run_id
    assert (destination / "COMPLETED").is_file()
