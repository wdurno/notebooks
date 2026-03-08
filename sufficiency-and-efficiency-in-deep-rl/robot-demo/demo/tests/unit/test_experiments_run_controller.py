from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.run_controller import _resolve_picar_host, build_training_config
from experiments.schemas import ExperimentRunConfig
from experiments.snapshot_store import resolve_snapshot_path


def test_build_training_config_for_init_disables_training():
    config = ExperimentRunConfig(phase="init")

    training, training_enabled, traversing = build_training_config(config)

    assert training_enabled is False
    assert traversing is False
    assert training.fixed_t == 0.0
    assert training.train_every_steps == 0
    assert training.memorize_every_steps == 0


def test_build_training_config_for_tune_uses_linear_ramp_step():
    config = ExperimentRunConfig(
        phase="tune",
        fixed_t=None,
        t_step=0.001,
    )

    training, training_enabled, traversing = build_training_config(config)

    assert training_enabled is True
    assert traversing is True
    assert training.fixed_t is None
    assert training.t_start == 0.0
    assert training.t_end == 1.0
    assert training.t_ramp_steps == 1000


def test_build_training_config_for_retask_defaults_to_t_one():
    config = ExperimentRunConfig(phase="retask")

    training, training_enabled, traversing = build_training_config(config)

    assert training_enabled is True
    assert traversing is False
    assert training.fixed_t == 1.0
    assert training.t_start == 1.0
    assert training.t_end == 1.0


def test_build_training_config_rejects_bad_t_log_every():
    config = ExperimentRunConfig(phase="tune", t_log_every=0)

    with pytest.raises(ValueError, match="--t-log-every"):
        build_training_config(config)


def test_build_training_config_rejects_bad_history_window():
    config = ExperimentRunConfig(phase="tune", history_window=0)

    with pytest.raises(ValueError, match="--history-window"):
        build_training_config(config)


def test_resolve_snapshot_path_accepts_file_or_directory(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    first = snapshots_dir / "snapshot-step-000010-initial.pt"
    second = snapshots_dir / "snapshot-step-000020-memorize.pt"
    first.write_bytes(b"x")
    second.write_bytes(b"y")

    assert resolve_snapshot_path(first) == first
    assert resolve_snapshot_path(snapshots_dir) == second

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    nested = run_dir / "snapshots"
    nested.mkdir(parents=True, exist_ok=True)
    nested_file = nested / "snapshot-step-000030-memorize.pt"
    nested_file.write_bytes(b"z")
    assert resolve_snapshot_path(run_dir) == nested_file


def test_resolve_picar_host_prefers_cli_then_env_then_default(monkeypatch):
    monkeypatch.delenv("PICAR_V_HOST", raising=False)
    config = ExperimentRunConfig(phase="init")
    assert _resolve_picar_host(config) == "127.0.0.1:5000"

    monkeypatch.setenv("PICAR_V_HOST", "10.0.0.22:5000")
    assert _resolve_picar_host(config) == "10.0.0.22:5000"

    config_cli = ExperimentRunConfig(phase="init", picar_host="10.0.0.44:6000")
    assert _resolve_picar_host(config_cli) == "10.0.0.44:6000"
