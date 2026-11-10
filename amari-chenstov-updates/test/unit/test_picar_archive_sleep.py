"""CPU-only acceptance for Plan19 Fix2 item 5 supplemental rehearsal."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from numerical_experiment.picar.archive_sleep import (
    ArchiveSleepSession, LikelihoodIdentity, ReadBudget, RetiredArchive, SleepPermission,
)
from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.learner import LearningBatch, MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar.runtime import ArchiveUnavailable, BudgetExceeded
from numerical_experiment.picar.schemas import TransitionRecord
from numerical_experiment.picar.storage import FIFOReplay, IntegrityError, RunStore
from src.continual import picar_math as pm


CONFIG = {"test": "retired-sleep-cpu-v1"}
PERMISSION = SleepPermission("stopped-fixture-0", True)
CONTEXT = {"episode_deadline": 20., "decision_time": [.1], "status_events": []}


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_learner(enabled=True, **overrides):
    config = PicarConfig()
    learning = replace(config.learning, retired_archive_replay=enabled,
                       retired_replay_protocol="supplemental-rehearsal-v1" if enabled else "off",
                       replay_capacity=4, sleep_batch=4, sleep_steps=1, **overrides)
    model = build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=4)
    return MemoryLearner(model, learning, config.seeds(0))


def identity(learner):
    return LikelihoodIdentity.from_learner(learner, task_version="approach-v1",
        model_version="tiny-fixture-seed31-v1", label_version="returns-v1", baseline_version="frozen-policy-v1")


def record(index, ref):
    return TransitionRecord(
        transition_id=f"t{index}", episode_id=f"e{index}", arrival_index=index,
        observation_ref=ref, next_observation_ref=ref, context_refs=(ref,),
        instruction="Approach and stop", action_logits=(.2,) * 7,
        action_probabilities=(.125,) * 8, requested_control={"drive_mps": .1},
        executed_control={"drive_mps": .1}, baseline=.05 * index, model_version=0,
        elapsed_s=.1, reward=-.02, discount=.99, terminated=False, truncated=False,
        end_reason=None, instruction_events=({"text": "Approach and stop", "issued_at": 0.},),
        action_history=((0., 0., 0., 0.),), context_arrivals=(index,), context_elapsed=(.1,),
        return_value=-.1, return_bin=1)


def direct_batch(records, store):
    histories = [dict(images=torch.from_numpy(store.frames.get(r.observation_ref).copy())
                      .permute(2, 0, 1)[None, None].float() / 255,
                      executed_actions=torch.zeros(1, 1, 4), elapsed=torch.tensor([[[.1]]]),
                      contexts=[list(r.instruction_events)], start_time=torch.zeros(1),
                      episode_deadline=torch.tensor([20.]), decision_time=torch.tensor([[.1]]),
                      status_events=[[]]) for r in records]
    return LearningBatch(histories, torch.tensor([r.baseline for r in records]),
        torch.tensor([r.return_bin for r in records]), torch.tensor([r.action_logits for r in records]),
        torch.tensor([r.arrival_index for r in records]))


class Owner:
    def __init__(self, store, learner=None):
        self.store = store
        self.learner = learner or make_learner()
        self.replay = FIFOReplay(4)
        self.clocks = {"new_transitions": 6, "drift_exposure": 6, "active_seconds": 1.25}
        self.pending = {"labels": ["not-ready"], "queue": [2, 3]}

    def capture(self):
        return dict(learner=self.learner.state_dict(), replay=self.replay.state_dict(),
                    clocks=self.clocks, pending_labels=self.pending)

    def restore(self, state):
        self.learner.load_state_dict(state["learner"])
        self.replay.load_state_dict(state["replay"])
        self.clocks, self.pending = state["clocks"], state["pending_labels"]

    def session(self, archive, **kwargs):
        return ArchiveSleepSession(learner=self.learner, replay=self.replay, store=self.store,
            identity=identity(self.learner), archive=archive, seed=19371,
            capture_owner=self.capture, restore_owner=self.restore,
            eligible_context=lambda r: CONTEXT, **kwargs)


def prepare(tmp_path, *, retired=2, eligible=4, cache_bytes=192, **learning):
    store = RunStore(tmp_path / "run", CONFIG, checkpoint_policy="journal-v1")
    owner = Owner(store, make_learner(**learning))
    archive = RetiredArchive(tmp_path / "archive", identity(owner.learner), create=True,
                             cache_bytes=cache_bytes)
    records = []
    for i in range(retired + eligible):
        ref = store.frames.put(np.full((8, 8, 3), i * 17, dtype=np.uint8))
        r = record(i, ref)
        records.append(r)
        owner.replay.admit(r)
    if retired:
        owner.learner.retire(direct_batch(records[:retired], store))
        owner.replay.retire([r.transition_id for r in records[:retired]], transaction_id="retirement-0")
        for r in records[:retired]:
            archive.put(r, replay=owner.replay, context=CONTEXT)
    session = owner.session(archive)
    state = owner.capture()
    state[session.state_key] = session.state_dict()
    store.checkpoint(state, records=records, transaction_id="bootstrap")
    return store, owner, archive, session, records


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_default_off_never_accesses_archive_checkpoint_or_learner(tmp_path):
    learner = make_learner(False)
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail(f"disabled path accessed {name}")
    def forbidden(*args):
        pytest.fail("disabled path opened archive or captured owner")
    session = ArchiveSleepSession(learner=learner, replay=Forbidden(), store=Forbidden(),
        identity=identity(learner), archive=forbidden, seed=21,
        capture_owner=forbidden, restore_owner=forbidden)
    before = learner.state_dict()
    assert session.step(index=0, permission=None, acquired_count=0) == {"status": "disabled", "protocol": "off"}
    assert_tree_equal(before, learner.state_dict())
    assert not list(tmp_path.iterdir())


def test_real_updates_keep_ewc_ledger_counts_and_frozen_design(tmp_path, monkeypatch):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        before = owner.capture()
        assert sum(float(x.sum()) for x in owner.learner.diagonal.values()) > 0
        seen, ages = [], []
        fit, fd = owner.learner.fit, pm.stm_fd_penalty
        def inspect_fit(**kwargs):
            seen.append(kwargs)
            return fit(**kwargs)
        def inspect_fd(*args, **kwargs):
            ages.append(args[5].clone())
            return fd(*args, **kwargs)
        monkeypatch.setattr(owner.learner, "fit", inspect_fit)
        monkeypatch.setattr(pm, "stm_fd_penalty", inspect_fd)
        receipt = session.step(index=0, permission=PERMISSION, acquired_count=6)
        assert receipt["archived_count"] == 1 and receipt["eligible_count"] == 3
        assert receipt["realized_batch_size"] == receipt["requested_batch_size"] == 4
        assert receipt["new_data_count"] == receipt["drift_exposure"] == receipt["ewc_admissions"] == 0
        assert set(receipt["archived_ids"]) <= set(owner.replay.retired_ids)
        assert seen[0]["new"] is None and seen[0]["steps"] == 1
        batch = seen[0]["replay"]
        assert batch.archived.tolist() == [False, False, False, True]
        assert ages[0][-1].item() == 1
        chosen = next(r for r in records if r.transition_id == receipt["archived_ids"][0])
        assert batch.baselines[-1].item() == pytest.approx(chosen.baseline)
        assert batch.labels[-1].item() == chosen.return_bin
        assert batch.histories[-1]["episode_deadline"].item() == 20
        assert batch.histories[-1]["contexts"][0] == list(chosen.instruction_events)
        after = owner.capture()
        assert any(not torch.equal(before["learner"]["parameters"][k], value)
                   for k, value in after["learner"]["parameters"].items())
        for key in ("diagonal", "anchor", "retired_count"):
            assert_tree_equal(before["learner"][key], after["learner"][key])
        for key in ("replay", "clocks", "pending_labels"):
            assert_tree_equal(before[key], after[key])
        assert torch.equal(before["learner"]["rng"]["fisher_labels"], after["learner"]["rng"]["fisher_labels"])
        second = session.step(index=1, permission=PERMISSION, acquired_count=6)
        assert second["metrics"]["ewc"] > 0
        assert second["metrics"]["fd"] >= 0
        assert receipt["io"]["frame_read_bytes"] > 0 and receipt["io"]["entry_read_bytes"] > 0
        assert session.io_totals["decode_seconds"] > 0
        assert archive.cache_bytes <= archive.cache_limit == 192
        assert store.progress()["archive_sleep"]["receipt"] == second
        checkpoint = store.load_checkpoint(trusted=True)
        assert checkpoint["archive_sleep"]["last_receipt"] == second
        assert checkpoint["pending_labels"] == before["pending_labels"]
    finally:
        store.close()


@pytest.mark.parametrize("retired,eligible,expected", [(0, 4, (0, 3)), (2, 1, (1, 1)), (1, 0, (1, 0))])
def test_source_shortages_underfill_without_exceeding_fixed_budget(tmp_path, retired, eligible, expected):
    store, owner, archive, session, _ = prepare(tmp_path, retired=retired, eligible=eligible)
    try:
        receipt = session.step(index=0, permission=PERMISSION, acquired_count=retired + eligible)
        assert (receipt["archived_count"], receipt["eligible_count"]) == expected
        assert receipt["realized_batch_size"] == sum(expected) <= 4
        assert receipt["archived_target"] == 1 and receipt["eligible_target"] == 3
    finally:
        store.close()


def test_stopped_permission_and_ledger_are_required_before_reads(tmp_path):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        before = dict(archive.metrics)
        with pytest.raises(PermissionError, match="stopped"):
            session.step(index=0, permission=SleepPermission("moving", False), acquired_count=6)
        assert archive.metrics == before and session.pending is None
        with pytest.raises(PermissionError, match="ledger"):
            archive.put(records[-1], replay=owner.replay)
        with pytest.raises(PermissionError, match="ledger"):
            archive.get(records[-1].transition_id, replay=owner.replay, permission=PERMISSION,
                        budget=ReadBudget(10000, 1))
        with pytest.raises(PermissionError, match="stopped"):
            archive.get(records[0].transition_id, replay=owner.replay,
                        permission=SleepPermission("moving", False), budget=ReadBudget(10000, 1))
        assert archive.metrics == before
    finally:
        store.close()


@pytest.mark.parametrize("field,value", [("task_version", "other-task"), ("reward_version", "other-reward"),
    ("model_version", "other-model"), ("label_version", "other-labels"),
    ("baseline_version", "recomputed-baseline"), ("temperature", 7.), ("action_variance", 2.),
    ("return_edges", (-10., 0., 5., 20.))])
def test_frozen_identity_rejects_incompatible_archived_entries(tmp_path, field, value):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        other = RetiredArchive(archive.root, replace(archive.identity, **{field: value}))
        with pytest.raises(IntegrityError, match="identity"):
            other.get(records[0].transition_id, replay=owner.replay, permission=PERMISSION,
                      budget=ReadBudget(10000, 1))
    finally:
        store.close()


def test_immutable_entries_and_hashchecked_labels_context_and_planned_seals(tmp_path):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        original = records[0]
        path = archive._path(original.transition_id)
        data, mtime = path.read_bytes(), path.stat().st_mtime_ns
        seal = archive.put(original, replay=owner.replay, context=CONTEXT)
        assert path.read_bytes() == data and path.stat().st_mtime_ns == mtime
        with pytest.raises(IntegrityError, match="immutable"):
            archive.put(replace(original, baseline=999.), replay=owner.replay, context=CONTEXT)
        changed = json.loads(data)
        changed["payload"]["record"]["return_bin"] = 0
        changed["payload"]["context"]["episode_deadline"] = 10.
        path.write_text(json.dumps(changed))
        with pytest.raises(IntegrityError, match="checksum"):
            archive.get(original.transition_id, replay=owner.replay, permission=PERMISSION,
                        budget=ReadBudget(10000, 1))
        from numerical_experiment.picar.config import canonical_json
        changed["sha256"] = hashlib.sha256(canonical_json(changed["payload"]).encode()).hexdigest()
        path.write_text(json.dumps(changed))
        with pytest.raises(IntegrityError, match="checksum"):
            archive.get(original.transition_id, replay=owner.replay, permission=PERMISSION,
                        budget=ReadBudget(10000, 1), expected_sha256=seal)
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["missing-entry", "missing-mount", "missing-frame", "corrupt-frame", "bytes", "frames"])
def test_failures_pause_with_recorded_reason_without_updates(tmp_path, failure):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        before = owner.capture()
        if failure == "missing-entry":
            for r in records[:2]:
                archive._path(r.transition_id).unlink()
        elif failure == "missing-mount":
            archive.root.rename(archive.root.with_name("unmounted"))
        elif failure in {"missing-frame", "corrupt-frame"}:
            for path in store.frames.root.glob("*.npy"):
                if failure == "missing-frame":
                    path.unlink()
                else:
                    path.write_bytes(b"corrupt")
        elif failure == "bytes":
            session.max_read_bytes = 1
        else:
            session.max_frames = 1
        with pytest.raises((ArchiveUnavailable, IntegrityError, BudgetExceeded)):
            session.step(index=0, permission=PERMISSION, acquired_count=6)
        assert session.pause_reason and session.pending and session.next_index == 0
        assert store.progress()["archive_sleep"]["pause_reason"] == session.pause_reason
        assert_tree_equal(before, owner.capture())
        assert store.load_checkpoint(trusted=True)["archive_sleep"]["pause_reason"] == session.pause_reason
    finally:
        store.close()


def test_explicit_loader_bounded_cache_and_warm_cache_missing_file(tmp_path):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        r = records[0]
        reads = []
        def loader(ref):
            reads.append(ref)
            return (store.frames.root / (ref[7:] + ".npy")).read_bytes()
        for _ in range(2):
            history = archive.history(r, replay=owner.replay, permission=PERMISSION,
                budget=ReadBudget(10000, 2), archived=True, loader=loader, context=CONTEXT)
            assert history["images"].shape == (1, 1, 3, 8, 8)
        assert reads == [r.observation_ref] * 2 and archive.metrics["cache_hits"] == 1
        assert archive.cache_bytes == archive.cache_limit == 192
        assert archive.metrics["peak_cache_bytes"] == 192
        (store.frames.root / (r.observation_ref[7:] + ".npy")).unlink()
        with pytest.raises(ArchiveUnavailable, match="frame"):
            archive.history(r, replay=owner.replay, permission=PERMISSION,
                budget=ReadBudget(10000, 2), archived=True, loader=loader)
        assert len(reads) == 3
    finally:
        store.close()


def test_eligible_context_suffix_does_not_gain_retired_frames(tmp_path):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        a, b = records[0], records[2]
        extended = replace(b, context_refs=(a.observation_ref, b.observation_ref),
                           action_history=((0.,) * 4,) * 2, context_arrivals=(0, 2), context_elapsed=(.1, .1))
        # Use a fresh ledger with the original same retired and eligible IDs.
        ledger = FIFOReplay(4)
        for r in (a, records[1], extended):
            ledger.admit(r)
        ledger.retire([a.transition_id, records[1].transition_id], transaction_id="fixture-retirement")
        context = {**CONTEXT, "decision_time": [.1, .2]}
        history = archive.history(extended, replay=ledger, permission=PERMISSION,
            budget=ReadBudget(10000, 4), frames=store.frames, context=context)
        assert history["images"].shape[1] == 1
        assert history["start_time"].item() == pytest.approx(.1)
        assert history["decision_time"].tolist() == [[pytest.approx(.2)]]
    finally:
        store.close()


class Halt(BaseException):
    pass


@pytest.mark.parametrize("stage", ["plan", "sealed", "commit"])
@pytest.mark.parametrize("after", [False, True])
def test_exact_uninterrupted_and_resumed_parameters_around_commits(tmp_path, monkeypatch, stage, after):
    baseline_dir = tmp_path / "baseline"
    interrupted_dir = tmp_path / "interrupted"
    baseline_dir.mkdir()
    interrupted_dir.mkdir()
    store, owner, archive, session, _ = prepare(baseline_dir)
    try:
        receipts = [session.step(index=i, permission=PERMISSION, acquired_count=6) for i in range(3)]
        expected, expected_rng = owner.capture(), session.rng.getstate()
    finally:
        store.close()
    store, owner, archive, session, _ = prepare(interrupted_dir)
    checkpoint = store.checkpoint
    def halt(*args, **kwargs):
        if kwargs.get("transaction_id", "").endswith(f"0-{stage}"):
            if after:
                checkpoint(*args, **kwargs)
            raise Halt(f"{stage}: after={after}")
        return checkpoint(*args, **kwargs)
    monkeypatch.setattr(store, "checkpoint", halt)
    try:
        with pytest.raises(Halt):
            session.step(index=0, permission=PERMISSION, acquired_count=6)
    finally:
        store.close()
    with RunStore(interrupted_dir / "run", CONFIG, checkpoint_policy="journal-v1", resume=True) as resumed_store:
        resumed_owner = Owner(resumed_store)
        resumed_archive = RetiredArchive(interrupted_dir / "archive", identity(resumed_owner.learner), cache_bytes=192)
        resumed = resumed_owner.session(resumed_archive)
        assert resumed.restore()
        for i, receipt in enumerate(receipts):
            actual = resumed.step(index=i, permission=PERMISSION, acquired_count=6)
            for key in ("archived_ids", "eligible_ids", "learner_version", "update_count"):
                assert actual[key] == receipt[key]
        assert_tree_equal(expected, resumed_owner.capture())
        assert resumed.rng.getstate() == expected_rng
        assert resumed_owner.learner.update_count == 3
        before = resumed_owner.capture()
        last = resumed.step(index=2, permission=PERMISSION, acquired_count=6)
        assert last == resumed.last_receipt
        assert resumed.step(index=0, permission=PERMISSION, acquired_count=6)["status"] == "already_committed"
        assert_tree_equal(before, resumed_owner.capture())
        assert resumed.pending is None and resumed.next_index == 3
        assert set(resumed.state_dict()) == {"schema_version", "contract", "rng", "next_index",
                                             "pending", "last_receipt", "pause_reason", "io_totals"}
        assert len(json.dumps(resumed.state_dict())) < 15_000


def test_restore_rejects_changed_session_contract(tmp_path):
    store, owner, archive, session, _ = prepare(tmp_path)
    try:
        state = session.state_dict()
        state["contract"]["fraction"] = .5
        with pytest.raises(IntegrityError, match="incompatible"):
            session.load_state_dict(state)
        with pytest.raises(ArchiveUnavailable, match="root"):
            RetiredArchive(tmp_path / "absent", identity(owner.learner))
        assert not (tmp_path / "absent").exists()
    finally:
        store.close()


def test_missing_entry_repair_reuses_the_persisted_batch_plan(tmp_path):
    store, owner, archive, session, records = prepare(tmp_path)
    try:
        for r in records[:2]:
            archive._path(r.transition_id).unlink()
        with pytest.raises(ArchiveUnavailable):
            session.step(index=0, permission=PERMISSION, acquired_count=6)
        planned = session.state_dict()
        for r in records[:2]:
            archive.put(r, replay=owner.replay, context=CONTEXT)
        restored = owner.session(archive)
        assert restored.restore()
        receipt = restored.step(index=0, permission=PERMISSION, acquired_count=6)
        assert receipt["eligible_ids"] == planned["pending"]["eligible_ids"]
        assert receipt["archived_ids"] == planned["pending"]["archived_ids"]
        assert restored.rng.getstate() == planned["rng"]
        assert restored.pause_reason is None and owner.learner.update_count == 1
        assert restored.io_totals["frame_read_bytes"] > receipt["io"]["frame_read_bytes"]
    finally:
        store.close()


def test_sealed_entry_replacement_is_rejected_before_resumed_optimizer(tmp_path, monkeypatch):
    store, owner, archive, session, _ = prepare(tmp_path)
    try:
        checkpoint = store.checkpoint
        def halt_after_seal(*args, **kwargs):
            result = checkpoint(*args, **kwargs)
            if kwargs.get("transaction_id", "").endswith("0-sealed"):
                raise Halt("sealed")
            return result
        monkeypatch.setattr(store, "checkpoint", halt_after_seal)
        with pytest.raises(Halt):
            session.step(index=0, permission=PERMISSION, acquired_count=6)
        before = owner.capture()
        key = session.pending["archived_ids"][0]
        path = archive._path(key)
        changed = json.loads(path.read_bytes())
        changed["payload"]["record"]["baseline"] = 999.
        from numerical_experiment.picar.config import canonical_json
        changed["sha256"] = hashlib.sha256(canonical_json(changed["payload"]).encode()).hexdigest()
        path.write_text(json.dumps(changed))
        with pytest.raises(IntegrityError, match="checksum"):
            session.step(index=0, permission=PERMISSION, acquired_count=6)
        assert_tree_equal(before, owner.capture())
        assert owner.learner.update_count == 0
        assert "checksum" in session.pause_reason
    finally:
        store.close()
