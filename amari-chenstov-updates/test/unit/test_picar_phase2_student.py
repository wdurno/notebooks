"""CPU-only Phase 2 recovery and isolation checks, with no Bullet/VLM launch."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from numerical_experiment.picar import fix3_student as frozen
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase2 import student
import test_picar_fix3_student as fixtures
from test_picar_fix3_student import setup  # noqa: F401 (shared fixture)


REAL_RENDER = student.ReadinessRun.render_video


class SnapshotWorker(fixtures.FakeWorker):
    index = 0

    def snapshot(self):
        return {"backend": {"seed": self.seed, "time_s": self.index * .1}}


class SnapshotSession(fixtures.FakeSession):
    def next_interval(self, timeout):
        result = super().next_interval(timeout)
        self.worker.index = self.index
        return result

    def snapshot(self, **kwargs):
        return {**super().snapshot(**kwargs), "worker": self.worker.snapshot()}

    def restore(self, state):
        super().restore(state)
        self.worker.index = self.index


class SnapshotRenderer:
    def __init__(self, snapshot):
        self.restore(snapshot)

    @classmethod
    def from_snapshot(cls, snapshot):
        return cls(snapshot)

    def restore(self, snapshot):
        self.time_s, self.seed = snapshot["time_s"], snapshot["seed"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def diagnostics(self):
        return {"geometry": {"clearance_m": .25, "heading_error_rad": -.2},
            "target_contact": False, "unsafe_contact": False,
            "physical_time_s": self.time_s, "end_reason": "deadline"}

    def render(self):
        return np.full((8, 8, 3), round(self.time_s * 100), dtype=np.uint8)

    def render_topdown(self):
        return 255 - self.render()


def fake_video(self, current, checkpoint, difficulty):
    return [{"checkpoint": checkpoint, "difficulty": difficulty, "seed": current["seed"],
             "times": [x["time_s"] for x in current["frames"]]}]


@pytest.fixture
def runtime(setup, monkeypatch):
    monkeypatch.setattr(frozen, "SimulatorWorker", SnapshotWorker)
    monkeypatch.setattr(frozen, "ControlSession", SnapshotSession)
    monkeypatch.setattr(student, "PyBulletBackend", SnapshotRenderer)
    monkeypatch.setattr(student.ReadinessRun, "render_video", fake_video)
    return setup


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def panels(sim):
    # Non-easy first and unsorted seeds detect hard-coded/easy-only selection.
    return [{"difficulty": name, "config": sim, "seeds": seeds}
            for name, seeds in (("hard", (9, 4)), ("middle", (11,)), ("easy", (7, 2)))]


def runner(store, runtime, initial, stopped=lambda: None, **kwargs):
    config, sim = runtime
    options = dict(mode="pilot", seed=192101, max_transitions=12, initial=initial,
                   evaluations=panels(sim))
    options.update(kwargs)
    return student.ReadinessRun(fixtures.model(), config, sim, store, stopped, **options)


def execute(path, runtime, initial, *, resume=False):
    with RunStore(path, {"test": "phase2"}, resume=resume, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, initial)
        summary = run.run()
        assert store.progress()["phase"] == 2
        assert store.progress()["status"].startswith("Phase 2 pilot")
        return summary, store.load_checkpoint(trusted=True), store.metrics()


def test_full_panels_initial_only_isolation_and_paired_videos(tmp_path, runtime, monkeypatch):
    _, initial, _ = fixtures.boot(tmp_path / "initial", runtime)
    original = copy.deepcopy(initial)
    _, sim = runtime
    evaluations = [{"difficulty": name, "config": replace(sim, domain_override=domain),
                    "seeds": tuple(range(192501, 192501 + n))}
                   for name, n, domain in (("easy", 20, "A"), ("hard", 10, "B"))]
    with RunStore(tmp_path / "panels", {"test": "panels"}, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, initial, evaluations=evaluations)
        before = run.learner.state_dict()
        replay = copy.deepcopy({k: run.s[k] for k in ("eligible", "retired", "ready")})
        rng = frozen._rng_state(run.learner.device)
        replay_rng = run.replay_rng.get_state().clone()
        calls = fixtures.FakeTeacher.calls, fixtures.FakeReward.calls

        def forbidden(*args, **kwargs):
            pytest.fail("evaluation must not fit, retire or score")

        monkeypatch.setattr(run.learner, "fit", forbidden)
        monkeypatch.setattr(run.learner, "retire", forbidden)
        monkeypatch.setattr(run.scorer, "score", forbidden)
        run.evaluate("initial")
        assert run.s["stage"] == "collect" and run.s["acquired_count"] == 0
        assert [len(x["rows"]) for x in run.s["evaluations"]] == [20, 10]
        run.evaluate("final")
        summary = run.run()
        assert_equal(before, run.learner.state_dict())
        assert_equal(replay, {k: run.s[k] for k in replay})
        assert_equal(rng, frozen._rng_state(run.learner.device))
        assert_equal(replay_rng, run.replay_rng.get_state())
        assert calls == (fixtures.FakeTeacher.calls, fixtures.FakeReward.calls)
        assert summary["timings"]["evaluation_seconds"] > 0
        assert summary["timings"]["fit_seconds"] == summary["timings"]["scoring_seconds"] == 0
        assert summary["phase"] == 2 and summary["mechanism"] == "exploratory"
        assert not summary["stm_efficacy"]
        assert summary["online_updates"] == summary["sleep_updates"] == run.s["kl_step"] == 0
        assert_equal(summary["distill_receipt"], initial["distill_receipt"])
        for first, last in zip(summary["evaluations"][:2], summary["evaluations"][2:]):
            assert [r["seed"] for r in first["rows"]] == [r["seed"] for r in last["rows"]]
            for row in first["rows"] + last["rows"]:
                assert row["physical_seconds"] == pytest.approx(.4)
                assert row["terminal_diagnostics"]["physical_seconds"] == pytest.approx(.4)
                assert row["terminal_diagnostics"]["source"] == "terminal_worker_snapshot"
                assert row["terminal_diagnostics"]["clearance_m"] == .25
                assert row["terminal_diagnostics"]["heading_error_rad"] == -.2
                assert row["terminal_diagnostics"]["target_contact"] is False
                assert row["terminal_diagnostics"]["unsafe_contact"] is False
                assert row["coverage"]["policy_intervals"] == 3
                assert sum(row["coverage"]["dominant_action_counts"]) == 3
                assert row["coverage"]["steering_rad_range"] == [0., 0.]
        assert [(v["checkpoint"], v["difficulty"], v["seed"]) for v in summary["videos"]] == [
            (c, d, 192501) for c in ("initial", "final") for d in ("easy", "hard")]
    assert_equal(original, initial)


@pytest.mark.parametrize("boundary", ["evaluation_capture", "evaluation_terminal", "evaluation_row",
                                     "collection", "scoring", "drain", "final_evaluation"])
def test_resume_same_identity_matches_uninterrupted(tmp_path, runtime, boundary):
    _, initial, _ = fixtures.boot(tmp_path / "initial", runtime)
    _, expected, _ = execute(tmp_path / "reference", runtime, initial)
    with RunStore(tmp_path / "interrupted", {"test": "phase2"}, checkpoint_policy="journal-v1") as store:
        def stop():
            s = store.load_checkpoint(trusted=True)
            current = s["current"] or {}
            hit = {"evaluation_capture": s["stage"] == "initial_evaluation"
                       and len(current.get("intervals", [])) == 2,
                "evaluation_terminal": s["stage"] == "initial_evaluation" and current.get("complete"),
                "evaluation_row": s["stage"] == "initial_evaluation"
                       and s["evaluations"] and bool(s["evaluations"][0]["rows"]),
                "collection": s["stage"] == "collect" and s["acquired_count"] == 2,
                "scoring": bool(s["pending"] and s["pending"][0]["score"]),
                "drain": s["stage"] == "drain" and bool(s["ready"]),
                "final_evaluation": s["stage"] == "final_evaluation" and current.get("complete")}[boundary]
            if hit:
                raise InterruptedError(boundary)
        with pytest.raises(InterruptedError):
            runner(store, runtime, initial, stop).run()
    summary, actual, metrics = execute(tmp_path / "interrupted", runtime, initial, resume=True)
    assert_equal(expected["learner"], actual["learner"])
    for key in ("identity", "episodes", "evaluations", "eligible", "retired", "ready", "intervals", "videos"):
        assert_equal(expected[key], actual[key])
    assert summary["online_updates"] > 0 and summary["pending_updates"] == 0
    assert actual["kl_step"] == actual["nll_step"] == 0
    assert_equal(actual["distill_receipt"], initial["distill_receipt"])
    assert actual["stage"] == "done" and actual["phase2"]["schema_version"] == student.PHASE2_SCHEMA
    assert all(m["phase"] == 2 for m in metrics)
    assert [(v["difficulty"], v["seed"]) for v in summary["videos"]] == [
        (d, s) for _ in range(2) for d, s in (("hard", 9), ("middle", 11), ("easy", 7))]


@pytest.mark.parametrize("change", ["panel_seed", "panel_config", "teacher_variant", "runtime_schema"])
def test_changed_identity_rejected_without_writes(tmp_path, runtime, monkeypatch, change):
    _, initial, _ = fixtures.boot(tmp_path / "initial", runtime)
    with RunStore(tmp_path / "identity", {"test": "identity"}, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, initial)
        run.save()
        before = store.load_checkpoint(trusted=True)
        assert_equal(run.identity, runner(store, runtime, initial).identity)
        options = {}
        if change.startswith("panel"):
            changed = panels(runtime[1])
            if change == "panel_seed":
                changed[0]["seeds"] = (9, 5)
            else:
                changed[0]["config"] = replace(runtime[1], deadline_s=.5)
            options["evaluations"] = changed
        elif change == "teacher_variant":
            options["teacher_variant"] = "baseline"
        else:
            monkeypatch.setattr(student, "PHASE2_SCHEMA", student.PHASE2_SCHEMA + 1)
        with pytest.raises(ValueError, match="resume identity differs"):
            runner(store, runtime, initial, **options)
        assert_equal(before, store.load_checkpoint(trusted=True))


def test_bootstrap_reuses_frozen_math_and_completed_resume_is_readonly(tmp_path, runtime):
    _, expected, _ = fixtures.boot(tmp_path / "reference", runtime)
    config, sim = runtime
    sim = replace(sim, curriculum_exposure_override=0., domain_override="A")
    for name in ("collect", "distill", "initial_nll", "drain", "calibration", "work_one", "retire_one", "episode"):
        assert getattr(student.ReadinessRun, name) is getattr(frozen._Run, name)
    with RunStore(tmp_path / "bootstrap", {"test": "bootstrap"}, checkpoint_policy="journal-v1") as store:
        run = student.ReadinessRun(fixtures.model(), config, sim, store, lambda: None,
                                  mode="bootstrap", seed=config.seed, max_transitions=12)
        summary = run.run()
        assert_equal(expected["learner"], run.s["learner"])
        assert_equal(expected["distill_receipt"], summary["distill_receipt"])
        store.complete(summary)
    with RunStore(tmp_path / "bootstrap", {"test": "bootstrap"}, resume=True,
                  checkpoint_policy="journal-v1") as store:
        before = (store.root / "CURRENT.json").read_bytes()
        resumed = student.ReadinessRun(fixtures.model(), config, sim, store, lambda: None,
                                      mode="bootstrap", seed=config.seed, max_transitions=12)
        assert resumed.run()["kind"] == "phase2_bootstrap"
        assert (store.root / "CURRENT.json").read_bytes() == before


def test_language_diagnostics_same_histories_no_optimization(tmp_path, runtime, monkeypatch):
    _, initial, _ = fixtures.boot(tmp_path / "initial", runtime)
    with RunStore(tmp_path / "language", {"test": "language"}, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, initial)
        before, eligible = run.learner.state_dict(), copy.deepcopy(run.s["eligible"])
        rng = frozen._rng_state(run.learner.device)
        seen = []
        choose = run.learner.choose_action

        def track(history, *, explore):
            assert explore is False
            seen.append(copy.deepcopy(history))
            return choose(history, explore=explore)

        monkeypatch.setattr(run.learner, "choose_action", track)
        result = student.language_readiness(run.learner, run.s["eligible"])
        assert len(result["rows"]) == 4 * len(eligible)
        assert not result["learning"] and result["phase"] == 2
        for offset in range(0, len(seen), 4):
            group = result["rows"][offset:offset + 4]
            assert [r["condition"] for r in group] == ["forward", "left", "right", "stop"]
            assert len({r["source_history_digest"] for r in group}) == 1
            assert all(len(r["logits"]) == 7 and len(r["probabilities"]) == 8 and r["features"] for r in group)
            for changed in seen[offset + 1:offset + 4]:
                assert_equal({k: v for k, v in seen[offset].items() if k != "contexts"},
                             {k: v for k, v in changed.items() if k != "contexts"})
        assert_equal(before, run.learner.state_dict())
        assert_equal(eligible, run.s["eligible"])
        assert_equal(rng, frozen._rng_state(run.learner.device))
        assert_equal(result, student.language_readiness(run))
        with pytest.raises(ValueError, match="eligible"):
            student.language_readiness(run.learner, [])


def test_video_labels_and_synchronized_snapshot_views(tmp_path, runtime, monkeypatch):
    from numerical_experiment.picar import report

    _, initial, _ = fixtures.boot(tmp_path / "initial", runtime)
    with RunStore(tmp_path / "video", {"test": "video"}, checkpoint_policy="journal-v1") as store:
        run = runner(store, runtime, initial)

        def assemble(rows, path, *, seed, fps):
            assert seed == 192501 and fps == 5.
            assert len(rows) == 2
            for row in rows:
                assert row["time_s"] == row["ego_time_s"] == row["topdown_time_s"]
                assert "learned student | initial | hard | seed=192501" in row["outcome"]
                from PIL import Image
                with Image.open(row["ego_path"]) as ego, Image.open(row["topdown_path"]) as top:
                    assert np.asarray(ego)[0, 0, 0] + int(np.asarray(top)[0, 0, 0]) == 255
            return {"path": str(path), "seed": seed, "selection": "preselected"}

        monkeypatch.setattr(report, "assemble_video", assemble)
        result = REAL_RENDER(run, {"seed": 192501, "frames": [
            {"seed": 192501, "time_s": t} for t in (.1, .1, .2)]}, "initial", "hard")
        assert result[0]["actor"] == "learned_student"
        assert result[0]["views"] == "synchronized_reconstruction_of_recorded_physics_snapshots"


def test_missing_terminal_evidence_is_explicit_not_fabricated():
    result = student._terminal_diagnostics({"terminal": True})
    assert result["source"].startswith("unavailable")
    assert result["target_contact"] is result["unsafe_contact"] is result["clearance_m"] is None


@pytest.mark.parametrize("field,value", [("pending", [{"score": None}]), ("ready", [1]),
                                       ("current", {"evaluation": False})])
def test_evaluation_rejects_unfinished_training_without_mutating(field, value):
    run = object.__new__(student.ReadinessRun)
    run.s = {"pending": [], "ready": [], "current": None, field: value}
    before = copy.deepcopy(run.s)
    with pytest.raises(RuntimeError, match="drained training"):
        run.evaluate("initial")
    assert_equal(run.s, before)
