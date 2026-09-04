from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.artifacts import (
    RotatedCompletedRunError,
    RotatedIncompleteRunError,
    RotatedRunStore,
    RotatedAuditRunStore,
    load_completed_run,
)
from mnist_experiment.rotated_mnist.audit_config import load_audit_config
from mnist_experiment.rotated_mnist.config import load_config


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase1_smoke.json"
)
AUDIT_SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase2_smoke.json"
)


def test_completed_run_is_immutable(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    store = RotatedRunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(RotatedCompletedRunError, match="completed"):
        store.begin(config, REPO_ROOT)
    with pytest.raises(RotatedCompletedRunError, match="complete"):
        session.write_json("late.json", {})


def test_incomplete_run_requires_explicit_resume(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    store = RotatedRunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)

    with pytest.raises(RotatedIncompleteRunError, match="resume"):
        store.begin(config, REPO_ROOT)
    resumed = store.begin(config, REPO_ROOT, resume=True)
    assert resumed.working_path == session.working_path


def test_loader_rejects_completed_marker_without_required_artifacts(
    tmp_path: Path,
) -> None:
    config = load_config(SMOKE_CONFIG)
    session = RotatedRunStore(tmp_path).begin(config, REPO_ROOT)
    path = session.complete(required=())

    with pytest.raises(Exception, match="missing partitions.json"):
        load_completed_run(path)


def test_completed_audit_is_immutable(tmp_path: Path) -> None:
    config = load_audit_config(AUDIT_SMOKE_CONFIG)
    store = RotatedAuditRunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(RotatedCompletedRunError, match="completed"):
        store.begin(config, REPO_ROOT)
