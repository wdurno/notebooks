from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.persistence import create_experiment_paths, save_model_artifacts


class FakePersistedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor([1.0]))
        self.frozen = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.value_head = nn.Linear(1, 1)
        self.saved_base = None

    def save(self, path):
        self.saved_base = path
        Path(path + ".state.pt").write_bytes(b"state")
        Path(path + ".ssr.pt").write_bytes(b"ssr")


def test_create_experiment_paths_makes_expected_tree(tmp_path):
    paths = create_experiment_paths(tmp_path, "robot-run")

    assert paths.run_dir.exists()
    assert paths.blobs_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.metrics_dir.exists()
    assert paths.artifacts_dir.exists()


def test_save_model_artifacts_exports_trainable_weights(tmp_path):
    model = FakePersistedModel()

    save_model_artifacts(model, tmp_path, "checkpoint")

    assert (tmp_path / "checkpoint.state.pt").exists()
    assert (tmp_path / "checkpoint.ssr.pt").exists()
    assert (tmp_path / "checkpoint.trainable.pt").exists()
    assert (tmp_path / "checkpoint.value_head.pt").exists()
