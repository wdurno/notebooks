"""Tiny synthetic-input acceptance of the complete supplemental scheduler."""

from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.kalman_pi.retention.supplement.config import Config
from mnist_experiment.kalman_pi.retention.supplement import run


@pytest.mark.parametrize("publication", (False, True))
def test_three_arm_cpu_resume_preserves_completed_units_and_full_pairing(tmp_path, monkeypatch, publication):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(23)
    data = {"train_x": torch.rand(2000, 1, 28, 28, generator=generator),
            "train_y": torch.arange(2000) % 10,
            "test_x": torch.rand(100, 1, 28, 28, generator=generator), "test_y": torch.arange(100) % 10}
    monkeypatch.setattr(run.original, "load_data", lambda _: data)
    if not publication:
        monkeypatch.setattr(run.Scheduler, "publish", lambda *a, **k: None)
    cfg = Config.from_json("mnist_experiment/kalman_pi/retention/supplement/configs/smoke.json")
    root, identity = tmp_path / "study", {"test": "supplement-cpu"}

    def budget(storage):
        return SimpleNamespace(used=0., remaining=72_000., allocation_seconds=72_000.,
                               check=storage.check, pulse=lambda **kw: None)

    with Store(root, identity, free_reserve_bytes=0) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(storage), tmp_path, max_units=6)
        with pytest.raises(StopRequested, match="max-units"):
            scheduler.run()
        assert len(scheduler.complete) == 2
        completed = {p: p.read_bytes() for record in storage.iter_units(completed_only=True)
                     for p in storage.unit_path(record["unit_id"]).iterdir()}
    with Store(root, identity, resume=True, free_reserve_bytes=0) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(storage), tmp_path)
        scheduler.run()
        assert len(scheduler.complete) == 6
        for path, content in completed.items():
            assert path.read_bytes() == content
        for replica in range(2):
            items = [t for t in scheduler.complete.values() if t["replica"] == replica]
            assert {t["arm"] for t in items} == {"fixed_ridge", "fixed_zero", "kalman_zero"}
            for key in ("asset_unit", "initial_unit", "stream_hash", "schedule_factor"):
                assert len({t[key] for t in items}) == 1
            initials = [{k: t["metrics"][0][k] for k in ("old_accuracy", "nine_recall", "nll")} for t in items]
            assert initials[0] == initials[1] == initials[2]
        assert all(storage.is_complete(record["unit_id"], full=True)
                   for record in storage.iter_units(completed_only=True))
        if publication:
            import nbformat
            from mnist_experiment.kalman_pi.runtime import read_json
            snapshot = read_json(root / "report_snapshot.json")
            assert snapshot["analysis"]["n_pairs"] == 2
            assert snapshot["status"] == "completed"
            book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
            assert book.metadata.plan17_zero_ridge_report.smoke is True
            assert len([c for c in book.cells if c.cell_type == "code"]) == 2
