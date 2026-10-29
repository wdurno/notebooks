"""Cooperative cold-subspace recovery, using cached MNIST and no production roots.

GRID_TEST_DEVICE selects cpu (default) or cuda. Historical hard-kill drills
already cover atomic writes; these checks exercise the new numerical dispatch.
"""

import os

import pytest
import torch

from mnist_experiment import plan15_grid_subspace as wrapper
import test_plan15_grid_recovery as helpers


@pytest.fixture
def inputs(monkeypatch):
    missing = [name for name in helpers.RAW_FILES
               if not (helpers.DATA_ROOT / "MNIST/raw" / name).is_file()]
    if missing:
        pytest.skip(f"cached MNIST required; no download: {', '.join(missing)}")
    cfg = wrapper.Config.smoke_config(device=os.environ.get("GRID_TEST_DEVICE", "cpu"))
    assert cfg.smoke and cfg.archive_rank == 64
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with wrapper.bindings():
        monkeypatch.setattr(helpers, "Config", wrapper.Config)
        monkeypatch.setattr(helpers, "Runner", wrapper.Runner)
        assert wrapper.ROOTS.get() == (None, None)
        try:
            helpers.setup_torch(cfg)
            yield cfg, helpers.MNISTSource(helpers.DATA_ROOT)
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture(scope="module")
def baseline_root(tmp_path_factory):
    return tmp_path_factory.mktemp("subspace_uninterrupted")


@pytest.fixture
def baseline(baseline_root, inputs):
    result = helpers.execute(baseline_root, inputs, "grid",
        resume=(baseline_root / "identity.json").exists(),
        numerical_backend="subspace_" + inputs[0].device)
    helpers.assert_exact_recovery(result, result, inputs[0], "grid")
    return result


@pytest.mark.parametrize("stop", ("pretrain", "mid_solve", "eviction", "accepted"))
def test_subspace_interruption_resumes_exactly(tmp_path, inputs, baseline, stop):
    cfg, _ = inputs
    backend = "subspace_" + cfg.device
    with pytest.raises(helpers.StopRequested):
        helpers.execute(tmp_path, inputs, "grid", stop=stop, numerical_backend=backend)
    before = helpers.completed_files(tmp_path)
    with helpers.Store(tmp_path, helpers._identity(cfg), resume=True, free_reserve_bytes=0) as store:
        if stop == "pretrain":
            asset = next(u for u in store.iter_units() if u["metadata"].get("kind") == "grid_registration_assets")
            checkpoint = store.load_checkpoint(asset["unit_id"])
            assert checkpoint["stage"] == "pretrain" and checkpoint["offset"] > 0
            assert checkpoint["optimizer"] is not None and asset["status"] == "incomplete"
        else:
            unit = helpers.trajectory_id("acceptance--grid", 0, helpers._recipe(cfg), cfg.novel_count)
            checkpoint = store.load_checkpoint(unit)
            assert not store.is_complete(unit)
            assert checkpoint["learner"]["compression_backend"] == backend
            if stop in ("mid_solve", "eviction"):
                assert checkpoint["next_segment"] == checkpoint["n"] == 0
                assert checkpoint["learner"]["cumulative_count"] == 0
                assert not checkpoint["learner"]["consolidated_ids"]
            else:
                assert checkpoint["next_segment"] > 0
                assert checkpoint["learner"]["accepted_arrivals"] > 0
    result = helpers.execute(tmp_path, inputs, "grid", resume=True, numerical_backend=backend)
    helpers.assert_exact_recovery(result, baseline, cfg, "grid")
    compression = result[1]["learner"]["archive_diagnostics"]["compression"]
    assert compression["requested_backend"] == backend and compression["requested_rank"] == 64
    assert compression["numerical_policy"] == wrapper.POLICY
    assert compression.get("warm_start_columns", 0) == 0
    after = helpers.completed_files(tmp_path)
    assert before.items() <= after.items()
    assert all(u["status"] == "completed" for u in helpers.iter_units(tmp_path, full=True))
    repeated = helpers.execute(tmp_path, inputs, "grid", resume=True, numerical_backend=backend)
    helpers.equal_tree(repeated[1], result[1])
    assert after == helpers.completed_files(tmp_path)
