"""Explicit Phase 0 pipeline validation; never connects to real hardware."""

from dataclasses import replace
import json

import pytest
import torch

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.engine import SmokeEngine, execution_seeds, run_smoke
from numerical_experiment.picar.runtime import StopController
from numerical_experiment.picar.storage import RunStore


def tiny():
    config = PicarConfig()
    return replace(config, transitions=4, bootstrap_transitions=4, episode_steps=2,
        image_width=24, image_height=16,
        model=replace(config.model, hidden_size=8, context_length=2, return_bins=5),
        learning=replace(config.learning, replay_capacity=2, replay_batch=1),
        runtime=replace(config.runtime, reserve_bytes=0, local_quota_bytes=1_000_000_000))


def test_paired_complete_resume_is_immutable(tmp_path):
    config = tiny()
    assert run_smoke(config, tmp_path)["status"] == "completed"
    summaries = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("replica-*/*/summary.json"))]
    assert len(summaries) == 2
    assert summaries[0]["initial_parameter_sha256"] == summaries[1]["initial_parameter_sha256"]
    assert all(s["new_transitions"] == 4 and s["retired_transitions"] == 6 for s in summaries)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert run_smoke(config, tmp_path, resume=True)["status"] == "completed"
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_bootstrap_and_replica_streams_are_distinct_and_pairable():
    config = tiny()
    initial = execution_seeds(config, 0, bootstrap=True)
    first, second = execution_seeds(config, 0), execution_seeds(config, 1)
    assert all(initial[k] != first[k] != second[k] for k in first)
    assert execution_seeds(config.for_condition("replay_ewc"), 1) == execution_seeds(config.for_condition("stm_fd"), 1)


@pytest.mark.parametrize("stage", ["collect", "labels", "distill", "joint", "retire"])
@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
def test_interrupted_bootstrap_recovers_exact_numerics(tmp_path, stage, boundary):
    config = tiny()
    states = []
    for tag in ("reference", "interrupted"):
        root = tmp_path / tag
        injected = False
        def fault(current, point):
            nonlocal injected
            if not injected and current == stage and point == boundary:
                injected = True
                raise InterruptedError("injected halt")
        with RunStore(root, config.to_dict(), reserve_bytes=0, local_quota_bytes=1_000_000_000) as store:
            engine = SmokeEngine(config, store, condition="bootstrap", replica=0, bootstrap=True)
            try:
                if tag == "reference":
                    engine.run(StopController(root))
                else:
                    with pytest.raises(InterruptedError):
                        engine.run(StopController(root), fault=fault)
            finally:
                engine.close()
        with RunStore(root, config.to_dict(), resume=True, reserve_bytes=0,
                      local_quota_bytes=1_000_000_000) as store:
            if not store.completed:
                engine = SmokeEngine(config, store, condition="bootstrap", replica=0, bootstrap=True)
                try:
                    engine.run(StopController(root))
                finally:
                    engine.close()
            states.append(store.load_checkpoint(trusted=True))
    for name in states[0]["learner"]["parameters"]:
        torch.testing.assert_close(states[0]["learner"]["parameters"][name],
                                   states[1]["learner"]["parameters"][name], rtol=0, atol=0)
    assert states[0]["replay"] == states[1]["replay"]
    assert states[0]["acquired"] == states[1]["acquired"]


def test_production_cannot_use_accelerated_smoke(tmp_path):
    config = tiny()
    config = replace(config, mode="production", model=replace(config.model, kind="qwen"))
    with pytest.raises(ValueError, match="accelerated acceptance only"):
        run_smoke(config, tmp_path)


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
def test_optimizer_publication_recovers_without_duplicate_evidence(tmp_path, boundary):
    config = tiny()
    reference, interrupted = tmp_path / "reference", tmp_path / "interrupted"
    run_smoke(config, reference)
    def fault(stage, point):
        if stage == "update" and point == boundary:
            raise InterruptedError("lost power at optimizer publication")
    with pytest.raises(InterruptedError):
        run_smoke(config, interrupted, fault=fault)
    run_smoke(config, interrupted, resume=True)
    for condition in config.conditions:
        a, b = [json.loads((root / "replica-000" / condition / "summary.json").read_text())
                for root in (reference, interrupted)]
        for key in ("initial_parameter_sha256", "final_parameter_sha256", "new_transitions",
                    "retired_transitions", "replay_exposures"):
            assert a[key] == b[key]


def test_recurrent_replay_masks_retired_predecessors(tmp_path):
    config = tiny()
    with RunStore(tmp_path / "unit", config.to_dict(), reserve_bytes=0,
                  local_quota_bytes=1_000_000_000) as store:
        engine = SmokeEngine(config, store, condition="bootstrap", replica=0, bootstrap=True)
        try:
            engine.run(StopController(tmp_path))
            rows = engine.replay.eligible()
            history = engine._batch([rows[0]]).histories[0]
            assert history["images"].shape[1] == 1
            assert engine.learner.retired_count == 2
        finally:
            engine.close()
