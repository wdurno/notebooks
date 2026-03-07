from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.observation_store import ObservationStore
from experiments.snapshot_store import SnapshotStore
from model.schemas import ModelObservation


class FakeReplayBuffer:
    def __init__(self, *, capacity: int = 128, n: int = 0):
        self.capacity = capacity
        self.n = n

    def __len__(self):
        return self.n


class FakeSnapshotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.frozen = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=False)
        self._loaded_ssr = None

    def ssr_dict(self):
        return {
            "ssr_rank": 1,
            "ssr_low_rank_matrix": torch.tensor([[1.0]], dtype=torch.float32),
            "ssr_residual_diagonal": torch.tensor([0.0], dtype=torch.float32),
            "ssr_center": torch.tensor([[0.0]], dtype=torch.float32),
            "ssr_prev_center": None,
            "ssr_n": 1,
            "ssr_cov_trace": 0.0,
            "ssr_cov_n": 0,
            "ssr_model_dimension": 1,
            "dt_mean_N": 10,
            "dt_mean_trend": torch.tensor(0.0),
            "dt_mean_norm_trend": 0.0,
            "dt_mean_trace_cov": 0.0,
            "dt_prev_pi": 0.5,
            "device": torch.device("cpu"),
        }

    def load_ssr_dict(self, payload):
        self._loaded_ssr = payload


def _make_observation(step_index: int, t: float = 0.0) -> ModelObservation:
    return ModelObservation(
        image_rgb=np.full((4, 4, 3), fill_value=step_index, dtype=np.uint8),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        t=t,
        step_index=step_index,
    )


def test_observation_store_writes_jsonl_and_compressed_images(tmp_path):
    store = ObservationStore(tmp_path / "run")

    store.append_observation(
        observation=_make_observation(0, t=0.0),
        reward=None,
        user_texts=[],
        action=None,
        training=None,
        action_receipt=None,
        source="reset",
    )
    store.append_observation(
        observation=_make_observation(1, t=0.1),
        reward=1.5,
        user_texts=["find the red ball"],
        action=None,
        training={"triggered": False},
        action_receipt={"status": "ok"},
        source="step",
    )

    lines = (store.observations_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"image_path": "images/step_000000.npz"' in lines[0]
    assert '"image_path": "images/step_000001.npz"' in lines[1]
    assert (store.images_dir / "step_000000.npz").exists()
    assert (store.images_dir / "step_000001.npz").exists()

    first_image = np.load(store.images_dir / "step_000000.npz")["image"]
    assert first_image.shape == (4, 4, 3)
    assert first_image.dtype == np.uint8


def test_snapshot_store_saves_trainable_only_and_enforces_retention(tmp_path):
    store = SnapshotStore(tmp_path / "run", max_keep=3)
    replay_buffer = FakeReplayBuffer(capacity=512, n=17)
    model = FakeSnapshotModel()

    first_snapshot = store.save_snapshot(
        model=model,
        replay_buffer=replay_buffer,
        step_index=1,
        t=0.0,
        reason="initial",
    )
    payload = torch.load(first_snapshot, map_location="cpu")
    assert set(payload["tunable_state_dict"]) == {"trainable"}
    assert payload["replay_metadata"]["size"] == 17
    assert payload["replay_metadata"]["capacity"] == 512

    for idx in range(2, 6):
        store.save_snapshot(
            model=model,
            replay_buffer=replay_buffer,
            step_index=idx,
            t=0.1 * idx,
            reason="memorize",
            memorize_count=64,
        )

    snapshots = sorted(store.snapshots_dir.glob("*.pt"))
    assert len(snapshots) == 3
    assert all("snapshot-step-" in path.name for path in snapshots)


def test_snapshot_store_loads_trainable_state_and_ssr(tmp_path):
    store = SnapshotStore(tmp_path / "run", max_keep=3)
    replay_buffer = FakeReplayBuffer(capacity=64, n=8)
    model = FakeSnapshotModel()
    model.trainable.data.fill_(3.5)
    model.frozen.data.fill_(9.0)
    snapshot_path = store.save_snapshot(
        model=model,
        replay_buffer=replay_buffer,
        step_index=12,
        t=0.7,
        reason="memorize",
    )

    model.trainable.data.fill_(1.0)
    model.frozen.data.fill_(5.0)
    load_result = store.load_into_model(snapshot_path=snapshot_path, model=model)

    assert load_result.path == snapshot_path
    assert load_result.has_ssr_state is True
    assert load_result.loaded_trainable_keys == ["trainable"]
    assert float(model.trainable.item()) == 3.5
    assert float(model.frozen.item()) == 5.0
    assert model._loaded_ssr is not None
