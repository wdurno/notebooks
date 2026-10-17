import json

import pytest
import torch

from mnist_experiment.akaike_score_gate.artifacts import UnitStore
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.model import build_model
from mnist_experiment.akaike_score_gate.numerical_audit import audit_precision


def test_paired_precision_audit_is_sealed_and_resumable(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        study = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")
        store = UnitStore(tmp_path, study)
        model, layout = build_model(18, dtype=torch.float32)
        theta = layout.flatten_module(model, detach=True)
        generator = torch.Generator().manual_seed(91)
        snapshot = {"step": 0, "model": model.state_dict(), "anchor": theta,
                    "inputs": torch.randn(4, 1, 28, 28, generator=generator),
                    "targets": torch.tensor([0, 1, 2, 3]), "weights": torch.ones(4) / 4,
                    "precision": torch.eye(layout.total_numel, dtype=torch.float64)}
        required = ("snapshots.pt", "summary.json")
        unit = store.unit("phase0", "test_driver", 0, detail={"width": 5})
        session = store.begin(unit, required, resume=True)
        session.write_torch("snapshots.pt", [snapshot])
        session.write_json("summary.json", {})
        parent = store.finish(session, required)
        result = audit_precision(store, [parent])
        summary = json.loads((result / "summary.json").read_text())
        rows = json.loads((result / "metrics.json").read_text())
        assert len(rows) == len(summary["cells"]) == 4
        assert all(row["same_start_and_objective"] for row in rows)
        assert max(row["objective_before"] for row in rows) - min(row["objective_before"] for row in rows) < 1e-6
        count = store.limit.units
        assert audit_precision(store, [parent]) == result
        assert store.limit.units == count
        (parent / "snapshots.pt").write_bytes(b"invalid")
        with pytest.raises(ValueError, match="checksum"):
            audit_precision(store, [parent])
    finally:
        torch.set_num_threads(threads)
