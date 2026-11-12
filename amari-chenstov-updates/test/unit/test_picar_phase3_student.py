"""Phase 3 CPU numerics and fake-session recovery; no GPU, VLM or Bullet."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from numerical_experiment.picar import fix3_student as frozen
from numerical_experiment.picar.storage import FrameStore, RunStore
from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
from numerical_experiment.picar_phase3 import student
import test_picar_fix3_student as fixtures
from test_picar_fix3_student import setup  # noqa: F401 (shared fixture)
from test_picar_phase2_student import assert_equal, runtime  # noqa: F401 (shared fixture)


class MemoryStore:
    """Exercise checkpoint boundaries without fsync on every evaluation frame."""

    def __init__(self, root):
        self.root = root
        self.frames = FrameStore(root / "frames")
        self.state, self.rows, self.progress = None, [], None

    def load_checkpoint(self, **kwargs):
        return copy.deepcopy(self.state)

    def checkpoint(self, state, *, metrics=(), progress=None):
        self.state = copy.deepcopy(state)
        self.rows.extend(copy.deepcopy(list(metrics)))
        self.progress = copy.deepcopy(progress)


@pytest.fixture
def initial(tmp_path, runtime):
    return fixtures.boot(tmp_path / "initial", runtime)[1]


def configuration(runtime, **changes):
    base, _ = runtime
    learning = dict(fd_strength=.05, optimizer_steps=2, learning_rate=.003, fd_step=.02)
    learning.update(changes)
    return replace(base, learning=replace(base.learning, **learning))


def panels(sim, full=False):
    return [{"difficulty": label, "config": replace(sim, domain_override=domain),
             "seeds": tuple(range(193501, 193501 + (count if full else 1)))}
            for label, domain, count in (("easy", "A", 20), ("hard", "B", 10))]


def runner(store, runtime, initial, stopped=lambda: None, *, config=None, model=None, **kwargs):
    _, sim = runtime
    options = dict(seed=193101, max_transitions=20, initial=initial, evaluations=panels(sim))
    options.update(kwargs)
    return student.DevelopmentRun(model if model is not None else fixtures.model(),
        config if config is not None else configuration(runtime), sim, store, stopped, **options)


def test_real_positive_fd_identity_adam_lr_moments_and_named_seeds(tmp_path, runtime, initial):
    untouched = copy.deepcopy(initial)
    store = MemoryStore(tmp_path / "identity")
    run = runner(store, runtime, initial)
    state = run.learner.state_dict()
    for key in ("parameters", "diagonal", "anchor"):
        assert_equal(state[key], initial["learner"][key])
    assert_equal(state["optimizer"]["state"], initial["learner"]["optimizer"]["state"])
    assert all(g["lr"] == .003 for g in state["optimizer"]["param_groups"])
    assert all(g["lr"] == .001 for g in initial["learner"]["optimizer"]["param_groups"])
    for name, generator in run.learner.generators.items():
        expected = torch.Generator().manual_seed(run.seeds[name]).get_state()
        assert_equal(generator.get_state(), expected)
    run.save()
    assert store.state["identity"]["config"] == asdict(configuration(runtime))
    assert store.state["identity"]["phase3"]["confirmatory"] is False
    assert store.state["phase"] == store.progress["phase"] == 3
    assert "phase2" not in store.state
    assert_equal(run.store.load_checkpoint(trusted=True)["identity"], run.identity)
    resumed = runner(store, runtime, initial)
    assert_equal(state, resumed.learner.state_dict())
    assert resumed.config.learning.fd_strength == resumed.learner.config.fd_strength == .05
    assert_equal(initial, untouched)


def test_actual_fd_steps_online_sleep_midpoint_and_phase3_metrics(tmp_path, runtime, initial):
    store = MemoryStore(tmp_path / "positive-fd")
    run = runner(store, runtime, initial)
    fd_rng = run.learner.generators["fd_probes"].get_state().clone()
    teacher_calls = fixtures.FakeTeacher.calls
    summary = run.run()
    metrics = summary["fit_metrics"]
    assert summary["kind"] == "phase3_pilot" and summary["condition"] == "stm_fd"
    assert summary["acquired_count"] == 16 and summary["midpoint_acquired_count"] == 12
    assert [p["acquired_count"] for p in summary["evaluations"]] == [0, 0, 12, 12, 16, 16]
    assert [p["checkpoint"] for p in summary["evaluations"]] == [
        label for label in ("initial", "midpoint", "final") for _ in range(2)]
    assert summary["online_updates"] > 0 and summary["sleep_updates"] > 0
    assert len(metrics) == summary["online_updates"] + summary["sleep_updates"]
    assert all(m["optimizer_steps"] == 2 and m["learning_rate"] == .003 for m in metrics)
    assert summary["online_optimizer_steps"] == 2 * summary["online_updates"]
    assert summary["sleep_optimizer_steps"] == 2 * summary["sleep_updates"]
    assert summary["learner_optimizer_step_count"] == initial["learner"]["update_count"] + 2 * len(metrics)
    assert any(m["fd"] > 0 for m in metrics if m["replay_arrival"] is not None)
    assert not torch.equal(fd_rng, run.learner.generators["fd_probes"].get_state())
    assert summary["fd_strength"] == .05 and summary["fd_enabled_fit_seconds"] > 0
    assert summary["fd_overhead_measured"] is False
    assert all(m["phase"] == 3 and m["condition"] == "stm_fd" for m in store.rows)
    assert run.s["kl_step"] == run.s["nll_step"] == 0
    assert_equal(run.s["distill_receipt"], initial["distill_receipt"])
    assert fixtures.FakeTeacher.calls == teacher_calls
    assert not run.s["ready"] and not run.s["pending"] and run.s["midpoint_done"]
    assert len(run.s["retired"]) == len(set(run.s["retired"]))
    assert not set(run.s["retired"]) & {r["arrival"] for r in run.s["eligible"]}


def interrupted_boundary(s, boundary, initial_retired):
    current = s["current"] or {}
    mid = s["stage"] == "midpoint_drain"
    pending = s.get("drain_retire_pending", False)
    return {
        "ready": s["stage"] == "collect" and current.get("label_complete") and bool(s["ready"]),
        "online_retire": s["stage"] == "collect" and current and not current.get("evaluation")
            and len(s["retired"]) > initial_retired,
        "midpoint_pending": mid and bool(s["ready"]) and not pending,
        "midpoint_fit": mid and pending and len(s["ready"]) == 2 and len(s["eligible"]) == 5,
        "midpoint_retire": mid and pending and len(s["ready"]) == 2 and len(s["eligible"]) == 4,
        "midpoint_evaluation": s["stage"] == "midpoint_evaluation"
            and len(current.get("intervals", [])) == 2,
        "midpoint_row": s["stage"] == "midpoint_evaluation" and current == {}
            and any(p["checkpoint"] == "midpoint" and p["rows"] for p in s["evaluations"]),
        "final_fit": s["stage"] == "drain" and pending and len(s["ready"]) == 2,
    }[boundary]


@pytest.mark.parametrize("boundary", ["ready", "online_retire", "midpoint_pending", "midpoint_fit",
    "midpoint_retire", "midpoint_evaluation", "midpoint_row", "final_fit"])
def test_journal_resume_exact_at_fit_retire_and_midpoint(tmp_path, runtime, initial, boundary):
    reference = MemoryStore(tmp_path / "reference")
    runner(reference, runtime, initial).run()
    expected = reference.state
    path = tmp_path / "interrupted"
    with RunStore(path, {"test": "phase3"}, checkpoint_policy="journal-v1") as store:
        def stop():
            if interrupted_boundary(store.load_checkpoint(trusted=True), boundary, len(initial["retired"])):
                raise InterruptedError(boundary)
        with pytest.raises(InterruptedError, match=boundary):
            runner(store, runtime, initial, stop).run()
    with RunStore(path, {"test": "phase3"}, resume=True, checkpoint_policy="journal-v1") as store:
        summary = runner(store, runtime, initial).run()
        actual = store.load_checkpoint(trusted=True)
        assert store.progress()["phase"] == 3
    for key in ("learner", "identity", "episodes", "evaluations", "eligible", "retired", "ready",
                "intervals", "videos", "replay_rng", "rng", "seeds", "midpoint_done",
                "midpoint_acquired_count", "drain_retire_pending"):
        assert_equal(expected[key], actual[key])
    stable_metrics = lambda s: [{k: v for k, v in m.items() if k != "update_seconds"}
                                for m in s["fit_metrics"]]
    assert_equal(stable_metrics(expected), stable_metrics(actual))
    assert summary["pending_updates"] == 0


def test_full_panels_frozen_initial_midpoint_final_and_first_seed_videos(tmp_path, runtime, initial, monkeypatch):
    store = MemoryStore(tmp_path / "panels")
    run = runner(store, runtime, initial, evaluations=panels(runtime[1], full=True))
    learner = copy.deepcopy(run.learner.state_dict())
    replay = copy.deepcopy({k: run.s[k] for k in ("eligible", "retired", "ready")})
    rng = frozen._rng_state(run.learner.device)
    replay_rng = run.replay_rng.get_state().clone()
    calls = fixtures.FakeTeacher.calls, fixtures.FakeReward.calls

    def forbidden(*args, **kwargs):
        pytest.fail("frozen evaluation must not fit, retire or score")

    monkeypatch.setattr(run.learner, "fit", forbidden)
    monkeypatch.setattr(run.learner, "retire", forbidden)
    monkeypatch.setattr(run.scorer, "score", forbidden)
    for label, count in (("initial", 0), ("midpoint", 12), ("final", 16)):
        run.s.update(acquired_count=count, stage=label + "_evaluation")
        run.evaluate(label)
    assert [len(p["rows"]) for p in run.s["evaluations"]] == [20, 10] * 3
    assert [p["acquired_count"] for p in run.s["evaluations"]] == [0, 0, 12, 12, 16, 16]
    for panel in run.s["evaluations"]:
        assert [r["seed"] for r in panel["rows"]] == panel["seeds"]
        assert all(r["acquired_count"] == panel["acquired_count"] for r in panel["rows"])
        assert all(r["physical_seconds"] == pytest.approx(.4) for r in panel["rows"])
    assert [(v["checkpoint"], v["difficulty"], v["seed"]) for v in run.s["videos"]] == [
        (c, d, 193501) for c in ("initial", "midpoint", "final") for d in ("easy", "hard")]
    assert_equal(learner, run.learner.state_dict())
    assert_equal(replay, {k: run.s[k] for k in replay})
    assert_equal(rng, frozen._rng_state(run.learner.device))
    assert_equal(replay_rng, run.replay_rng.get_state())
    assert calls == (fixtures.FakeTeacher.calls, fixtures.FakeReward.calls)
    assert run.s["timings"]["evaluation_seconds"] > 0


@pytest.mark.parametrize("field,value", [("fd_strength", 0.), ("fd_strength", .1),
    ("optimizer_steps", 3), ("learning_rate", .007), ("temperature", 6.),
    ("action_variance", 2.), ("fd_step", .03), ("fd_probes", 2)])
def test_full_real_configuration_rejects_changed_resume_without_writes(tmp_path, runtime, initial, field, value):
    store = MemoryStore(tmp_path / "mismatch")
    runner(store, runtime, initial).save()
    before = store.load_checkpoint()
    with pytest.raises(ValueError, match="Phase 3 resume identity differs"):
        runner(store, runtime, initial, config=configuration(runtime, **{field: value}))
    assert_equal(before, store.load_checkpoint())


@pytest.mark.parametrize("mode,eps", [("feature_rms", 1e-6), ("feature_and_projection_rms", 1e-5)])
def test_normalization_provenance_is_strict_resume_identity(tmp_path, runtime, mode, eps):
    store = MemoryStore(tmp_path / "normalized")
    options = dict(mode="bootstrap", evaluations=())
    runner(store, runtime, None, model=NormalizedStudent(fixtures.model(), mode="feature_and_projection_rms"),
           **options).save()
    before = store.load_checkpoint()
    with pytest.raises(ValueError, match="Phase 3 resume identity differs"):
        runner(store, runtime, None, model=NormalizedStudent(fixtures.model(), mode=mode, eps=eps), **options)
    assert_equal(before, store.load_checkpoint())


def test_initial_panel_reuse_exact_config_asset_identity_and_no_alias(tmp_path, runtime, initial):
    asset_id = "a" * 64
    first = runner(MemoryStore(tmp_path / "first"), runtime, initial,
                   config=configuration(runtime, fd_strength=0.), initial_asset_id=asset_id)
    first.evaluate("initial")
    panels_before = copy.deepcopy(first.s["evaluations"])
    store = MemoryStore(tmp_path / "reused")
    calls = fixtures.FakeTeacher.calls, fixtures.FakeReward.calls
    run = runner(store, runtime, initial, initial_evaluations=panels_before, initial_asset_id=asset_id)
    assert run.s["stage"] == "collect" and run.s["acquired_count"] == 0
    assert run.s["initial_evaluation_source"]["initial_asset_id"] == asset_id
    assert_equal(run.s["evaluations"], first.s["evaluations"])
    assert_equal(run.learner.state_dict()["parameters"], initial["learner"]["parameters"])
    assert calls == (fixtures.FakeTeacher.calls, fixtures.FakeReward.calls)
    run.save()
    resumed = runner(store, runtime, initial, initial_evaluations=panels_before, initial_asset_id=asset_id)
    assert_equal(run.identity, resumed.identity)
    run.s["evaluations"][0]["rows"][0]["actor"] = "changed"
    assert_equal(panels_before, first.s["evaluations"])


@pytest.mark.parametrize("change", ["checkpoint", "seed", "config", "learning", "asset", "duplicate"])
def test_unverified_initial_panel_reuse_rejected(tmp_path, runtime, initial, change):
    first = runner(MemoryStore(tmp_path / "first"), runtime, initial, initial_asset_id="asset-sha")
    first.evaluate("initial")
    supplied = copy.deepcopy(first.s["evaluations"])
    panel = supplied[0]
    if change == "checkpoint":
        panel["checkpoint"] = "midpoint"
    elif change == "seed":
        panel["rows"][0]["seed"] += 1
    elif change == "config":
        panel["config"]["domain_override"] = "B"
    elif change == "learning":
        panel["rows"][0]["learning"] = True
    elif change == "asset":
        panel["initial_asset_id"] = "wrong"
    else:
        supplied[1] = copy.deepcopy(supplied[0])
    store = MemoryStore(tmp_path / "bad")
    with pytest.raises(ValueError, match="initial panel"):
        runner(store, runtime, initial, initial_evaluations=supplied, initial_asset_id="asset-sha")
    assert store.state is None


def test_failed_multistep_fit_rolls_back_learner_and_replay_rng(tmp_path, runtime, initial, monkeypatch):
    run = runner(MemoryStore(tmp_path / "rollback"), runtime, initial)
    item = copy.deepcopy(run.s["eligible"][-1])
    item.update(arrival=initial["acquired_count"], learned=False)
    run.s["eligible"].append(item)
    run.s["ready"] = [item["arrival"]]
    run.s["acquired_count"] = 1
    before = copy.deepcopy(run.learner.state_dict())
    replay_rng = run.replay_rng.get_state().clone()
    step, calls = run.learner.optimizer.step, []

    def fail_second_step(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("second step interrupted")
        return step(*args, **kwargs)

    monkeypatch.setattr(run.learner.optimizer, "step", fail_second_step)
    with pytest.raises(RuntimeError, match="second step interrupted"):
        run.work_one()
    assert len(calls) == 2
    assert_equal(before, run.learner.state_dict())
    assert_equal(replay_rng, run.replay_rng.get_state())
    assert run.s["ready"] == [item["arrival"]] and not item["learned"]
    assert run.s["fit_metrics"] == [] and run.s["sleep_updates"] == 0


def test_normalized_fresh_bootstrap_and_completed_journal_resume(tmp_path, runtime):
    path = tmp_path / "bootstrap"
    options = dict(mode="bootstrap", max_transitions=12, evaluations=(),
                   config=configuration(runtime, fd_strength=0., optimizer_steps=1, learning_rate=.001))
    with RunStore(path, {"test": "phase3-bootstrap"}, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, None,
            model=NormalizedStudent(fixtures.model(), mode="feature_and_projection_rms"), **options)
        result = run.run()
        assert result["kind"] == "phase3_bootstrap" and result["acquired_count"] == 8
        assert run.s["kl_step"] == run.s["nll_step"] == 6
        assert not run.s["midpoint_done"] and result["midpoint_acquired_count"] is None
        before = store.load_checkpoint(trusted=True)
        store.complete(result)
    with RunStore(path, {"test": "phase3-bootstrap"}, resume=True, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, None,
            model=NormalizedStudent(fixtures.model(), mode="feature_and_projection_rms"), **options)
        assert run.run()["kind"] == "phase3_bootstrap"
        assert_equal(before, store.load_checkpoint(trusted=True))
