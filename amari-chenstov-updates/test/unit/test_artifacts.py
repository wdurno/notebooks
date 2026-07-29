import json
from pathlib import Path

import pytest
import torch

from src.artifacts import (
    CompletedRunError,
    IncompleteRunError,
    RunStateError,
    RunStore,
)
from src.config import load_config


REPO_ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = REPO_ROOT / "mnist_experiment" / "configs" / "smoke.json"


def test_run_completion_is_immutable(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    store = RunStore(tmp_path / "runs")
    session = store.begin(config, REPO_ROOT)

    session.write_json("metrics/trajectory.json", [{"p": 0.0}])
    session.write_torch(
        "checkpoints/state.pt",
        {"weights": torch.tensor([1.0, 2.0])},
    )
    completed_path = session.complete(
        ["metrics/trajectory.json", "checkpoints/state.pt"]
    )

    assert session.path == completed_path
    assert (completed_path / "COMPLETED").is_file()
    assert json.loads(
        (completed_path / "manifest.json").read_text(encoding="utf-8")
    )["status"] == "completed"
    loaded = torch.load(
        completed_path / "checkpoints/state.pt",
        weights_only=True,
    )
    assert loaded["weights"].tolist() == [1.0, 2.0]

    with pytest.raises(CompletedRunError):
        session.write_json("metrics/other.json", {})
    with pytest.raises(CompletedRunError):
        store.begin(config, REPO_ROOT)


def test_incomplete_run_requires_explicit_resume(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    store = RunStore(tmp_path / "runs")
    original = store.begin(config, REPO_ROOT)

    with pytest.raises(IncompleteRunError):
        store.begin(config, REPO_ROOT)

    resumed = store.begin(config, REPO_ROOT, resume=True)
    assert resumed.working_path == original.working_path
    resumed.write_json("metrics.json", {"ok": True})
    assert resumed.complete(["metrics.json"]).is_dir()


def test_completion_requires_declared_artifacts(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    session = RunStore(tmp_path / "runs").begin(config, REPO_ROOT)

    with pytest.raises(RunStateError, match="missing artifacts"):
        session.complete(["metrics.json"])


def test_artifact_paths_cannot_escape_run(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    session = RunStore(tmp_path / "runs").begin(config, REPO_ROOT)

    with pytest.raises(ValueError, match="relative"):
        session.write_json("../outside.json", {})
