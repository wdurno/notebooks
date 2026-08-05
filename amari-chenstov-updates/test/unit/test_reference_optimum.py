import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.config import ExperimentConfig
from src.mnist_data import DatasetPartitions
from src.reference_optimum import (
    ReferenceOptimumPath,
    build_reference_optimum_path,
)


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_convergence.json"
)


def test_reference_optimum_path_rejects_inconsistent_displacements() -> None:
    path = ReferenceOptimumPath(
        p_values=(0.0, 1.0),
        parameters=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        displacements=torch.tensor([[2.0]], dtype=torch.float64),
        rows=({}, {}),
        sample_plans=(None, None),
        content_hash="a" * 64,
    )

    with pytest.raises(ValueError, match="do not match"):
        path.validate()


def test_adaptive_reference_path_records_independent_displacement_radius() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["runtime"].update(
        {"device": "cpu", "training_dtype": "float64"}
    )
    raw["initialization"]["num_workers"] = 0
    raw["data"].update(
        {
            "initialization_size": 1,
            "online_pool_size": 1,
            "reference_pool_size": 20,
            "evaluation_size": 1,
            "num_p_steps": 2,
            "samples_per_step": 1,
        }
    )
    raw["reference"].update(
        {
            "sample_size": 4,
            "chunk_size": 2,
            "convergence_sample_sizes": [4],
            "convergence_min_chunks": 2,
            "calibration_steps": 1,
            "calibration_batch_size": 2,
            "calibration_min_fits": 2,
            "calibration_max_fits": 2,
            "calibration_validation_chunks": 2,
        }
    )
    config = ExperimentConfig.from_mapping(raw)
    targets = torch.tensor([0, 9] * 10, dtype=torch.long)
    dataset = TensorDataset(
        torch.linspace(-1.0, 1.0, targets.numel(), dtype=torch.float64).unsqueeze(1),
        targets,
    )
    partitions = DatasetPartitions(
        initialization=(),
        online=(),
        reference=tuple(range(targets.numel())),
        evaluation=(),
        train_size=targets.numel(),
        test_size=0,
        seed=1,
        initialization_seed=2,
        evaluation_seed=3,
    )
    torch.manual_seed(4)
    model = nn.Linear(1, 10, dtype=torch.float64)

    path = build_reference_optimum_path(
        model,
        dataset,
        targets,
        partitions,
        (0.0, 1.0),
        config,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert path.schema_version == 2
    assert path.replicate_parameters.shape == (2, 2, 20)
    assert len(path.fit_rows) == 4
    assert path.displacement_diagnostics[1]["count"] == 2
    assert path.displacement_diagnostics[1]["estimand"] == (
        "adjacent_reference_displacement"
    )
    torch.testing.assert_close(
        path.displacements,
        path.parameters[1:] - path.parameters[:-1],
    )

    restored = ReferenceOptimumPath.from_artifact_mapping(
        path.artifact_mapping()
    )
    assert restored.content_hash == path.content_hash
    torch.testing.assert_close(restored.parameters, path.parameters)
    assert restored.sample_plans == path.sample_plans
