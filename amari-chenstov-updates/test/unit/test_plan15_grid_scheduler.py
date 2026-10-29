"""Scheduler identity and selected-architecture resume checks without learning."""

from pathlib import Path
from types import SimpleNamespace

from mnist_experiment.fisher_memory_grid_registration.config import Config
from mnist_experiment.fisher_memory_grid_registration.run import Study, process_alive


def test_live_process_verification_requires_new_module_and_same_root(monkeypatch, tmp_path):
    root = tmp_path / "run"
    command = ["python", "-m", "mnist_experiment.fisher_memory_grid_registration.run", "--root", str(root)]
    monkeypatch.setattr(Path, "read_bytes", lambda p: "\0".join(command).encode())
    assert process_alive(root, 123)
    assert not process_alive(tmp_path / "different", 123)
    command[2] = "mnist_experiment.fisher_memory_registration.run"
    assert not process_alive(root, 123)
    assert not process_alive(root, None)


def test_resume_keeps_the_selected_wider_architecture_backend():
    study = Study.__new__(Study)
    study.cfg = Config()
    study.runner = SimpleNamespace(channels=(4, 8), compression_backend="exact_cuda")
    decisions = {"resource-assessment": {"selected_backend": "exact_cpu"},
                 "resource-assessment-wide": {"selected_backend": "exact_cuda"}}
    study.decision = decisions.get
    study.resource_assessment()
    assert study.runner.compression_backend == "exact_cuda"
    study.runner.channels = (4, 6)
    study.resource_assessment()
    assert study.runner.compression_backend == "exact_cpu"
