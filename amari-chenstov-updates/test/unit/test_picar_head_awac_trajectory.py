"""Tiny CPU histories and transactional scheduling, without physics or downloads."""

import copy
from dataclasses import asdict
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from numerical_experiment.picar.fix2_data import history_batch
from numerical_experiment.picar.models import PiCarStudent, TinyContextEncoder, batch_digest
from numerical_experiment.picar_head_awac.config import simulator
from numerical_experiment.picar_head_awac.data import append_history, pack_history, unpack_history
from numerical_experiment.picar_head_awac.learner import Learner, Recipe
from numerical_experiment.picar_head_awac.model import FrozenFeatures, Heads
from numerical_experiment.picar_head_awac.trajectory import Trajectory
from numerical_experiment.picar_head_awac import trajectory as trajectory_module
from src.robotics.simulator import PyBulletBackend


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            exact(left, right)
    else:
        assert actual == expected


def observation(n):
    return {"rgb": np.full((8, 8, 3), 20 * n, dtype=np.uint8), "time": .5 * n,
            "elapsed": .5, "action": [n / 10, 0., 0., 0.]}


def history(count=2):
    return history_batch([observation(i) for i in range(1, count + 1)],
                         contexts=[{"text": "Drive toward the ball.", "issued_at": 0.}],
                         statuses=[{"text": "Ready", "available_at": 0., "kind": "robot_status"}],
                         deadline=20., width=8, height=8, length=4)


class Frames:
    def __init__(self):
        self.values = {}

    def put(self, value):
        key = hashlib.sha256(value.tobytes()).hexdigest()
        self.values[key] = value.copy()
        return key

    def get(self, key):
        return self.values[key].copy()


class Pause(RuntimeError):
    pass


class Store:
    def __init__(self):
        self.frames = Frames()
        self.saved = None
        self.progress = None
        self.commits = 0
        self.stop_after = None

    def load_checkpoint(self, *, trusted=False):
        assert trusted
        return copy.deepcopy(self.saved)

    def checkpoint(self, state, *, progress):
        self.saved = copy.deepcopy(state)
        self.progress = copy.deepcopy(progress)
        self.commits += 1
        if self.commits == self.stop_after:
            raise Pause("after durable checkpoint")


class Features:
    def __init__(self):
        self.calls = []
        self.cleared = 0

    def __call__(self, value):
        self.calls.append(batch_digest(value))
        return torch.tensor([value["images"].shape[1], float(value["decision_time"][0, -1]),
                             float(value["images"].mean())])

    def clear(self):
        self.cleared += 1


class SchedulerLearner:
    def __init__(self, *, capacity=2, rounds=2, batch_size=1):
        self.recipe = SimpleNamespace(capacity=capacity, rounds=rounds, batch_size=batch_size, rank=0)
        self.method, self.version = "td_replay", 0
        self.calls, self.retired = [], []

    def state_dict(self):
        return copy.deepcopy({"calls": self.calls, "retired": self.retired, "version": self.version})

    def load_state_dict(self, state):
        self.calls, self.retired, self.version = copy.deepcopy((state["calls"], state["retired"], state["version"]))

    def fit(self, records, acquired, check):
        check()
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))
        assert all(r["visits"] > 0 for r in records[1:])
        self.calls.append((ids, acquired))
        self.version += 1
        return {"n": acquired, "update_seconds": .01, "replay_count": len(records)}

    def retire(self, record):
        assert record["visits"] > 0
        self.retired.append(record["id"])


def runner(store=None, learner=None, features=None, scorer=None, check=lambda: None, **kwargs):
    return Trajectory(learner or SchedulerLearner(), features or Features(), scorer or Mock(),
                      store or Store(), check, seed=91, total=8, evaluation_episodes=1,
                      initial_episodes=1, **kwargs)


def replay_record(index, frames):
    return {"id": f"episode:{index}", "episode_id": "episode", "arrival": index - 1, "visits": 0,
            "feature": torch.tensor([.2, -.1, .7]), "next_feature": torch.tensor([.3, -.2, .6]),
            "action": torch.zeros(7), "reward": torch.tensor(.1), "return": torch.tensor(.2),
            "discount": torch.tensor(.9), "history": pack_history(history(index), frames),
            "next_history": pack_history(history(index + 1), frames)}


def seeded_work(count=3, **recipe):
    result = runner(learner=SchedulerLearner(**recipe))
    result.s.update(stage="fit", n=count, replay=[replay_record(i, result.store.frames) for i in range(1, count + 1)],
                    ready=[f"episode:{i}" for i in range(1, count + 1)])
    result.save()
    return result


def pending_episode(result, *, evaluation=False):
    rows = []
    for i in range(1, 3):
        record = replay_record(i, result.store.frames)
        rows.append({"frame_ref": result.store.frames.put(observation(i)["rgb"]), "elapsed": .5,
                     "bonus": 1. if i == 2 else 0., "terminal": i == 2, "truncated": False,
                     "action": record["action"], "feature": record["feature"], "next_feature": None,
                     "history": record["history"], "next_history": record["next_history"], "arrival": i - 1})
    return {"episode_id": "train:91:0", "evaluation": evaluation, "complete": True,
            "rows": rows, "scores": [None, None], "frames": [], "success": True}


def test_pack_unpack_history_exact_frozen_roundtrip_and_no_aliasing():
    frames, original = Frames(), history()
    before = copy.deepcopy(original)
    packed = pack_history(original, frames)
    assert "images" not in packed and len(packed["frame_refs"]) == 2
    restored = unpack_history(packed, frames)
    exact(restored, original)
    assert batch_digest(restored) == batch_digest(original)
    restored["executed_actions"].add_(1)
    restored["contexts"][0][0]["text"] = "Changed"
    exact(original, before)
    exact(unpack_history(packed, frames), before)


@pytest.mark.parametrize("post_time, expected", [(1.5, [.5, 1., 1.5]), (1., [.5, 1.])])
def test_append_history_preserves_causal_past_and_does_not_mutate_input(post_time, expected):
    original, post = history(), observation(3)
    post["time"] = post_time
    before, post_before = copy.deepcopy(original), copy.deepcopy(post)
    appended = append_history(original, post, context_length=4)
    assert appended["decision_time"][0].tolist() == expected
    torch.testing.assert_close(appended["images"][0, 0, :, 0, 0], original["images"][0, 0, :, 0, 0])
    exact(appended["contexts"], original["contexts"])
    exact(appended["status_events"], original["status_events"])
    exact(original, before)
    exact(post, post_before)


def test_append_history_bounds_context_and_retains_recent_order():
    appended = append_history(history(3), observation(4), context_length=2)
    assert appended["decision_time"].tolist() == [[1.5, 2.]]
    assert appended["start_time"].tolist() == [1.]
    assert appended["images"].shape[1] == 2


@pytest.mark.parametrize("environment, sigma", [("initial_A", 1.), ("moderate_A", .5)])
def test_fixed_panels_use_declared_sigma_and_A_textures_without_physics(environment, sigma):
    cfg = simulator(900, 1024, fixed=environment)
    assert cfg.barrier_sigma_easy == cfg.barrier_sigma_hard == sigma
    assert cfg.domain_override == "A" and not cfg.staged_curriculum
    backend = SimpleNamespace(config=cfg, exposure_intervals=900, _last_alpha=None,
                              _tiles=[(11, False), (12, True)], _p=Mock(), client_id=3)
    PyBulletBackend._update_drift(backend)
    assert backend.alpha == 0.
    colors = [call.kwargs["rgbaColor"] for call in backend._p.changeVisualShape.call_args_list]
    assert colors == [(.58, .58, .58, 1), (.72, .72, .72, 1)]


def test_training_sigma_varies_but_domain_remains_A():
    assert simulator(0, 8).barrier_sigma_easy == 1.
    assert simulator(8, 8).barrier_sigma_easy == .5
    assert .5 < simulator(4, 8).barrier_sigma_easy < 1.
    assert simulator(4, 8).domain_override == "A"


@pytest.fixture
def frozen():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        source = PiCarStudent(TinyContextEncoder(width=4, depth=1, top_layers=1, ltm_rank=1,
                                                stm_rank=1, context_length=4), hidden_size=3, return_bins=3)
    return FrozenFeatures(source, "frozen-initializer-v1", capacity=2)


def test_frozen_features_disable_every_parameter_and_return_independent_cache_values(frozen):
    assert not frozen.model.training
    assert not any(p.requires_grad for p in frozen.model.parameters())
    before = copy.deepcopy(frozen.model.state_dict())
    value = frozen(history())
    assert value.shape == (3,) and value.device.type == "cpu" and not value.requires_grad
    expected = value.clone()
    value.add_(100)
    exact(frozen(history()), expected)
    assert (frozen.hits, frozen.misses) == (1, 1)
    exact(frozen.model.state_dict(), before)


@pytest.mark.parametrize("field", ["images", "executed_actions", "elapsed", "episode_deadline",
                                   "decision_time", "contexts", "status_events", "identity"])
def test_cache_invalidates_on_visible_history_or_frozen_identity_change(frozen, field):
    original = history()
    frozen(original)
    changed = copy.deepcopy(original)
    if field == "identity":
        frozen.identity = "new-initializer"
    elif field == "contexts":
        changed["contexts"][0][0]["text"] = "Turn left."
    elif field == "status_events":
        changed[field][0][0]["text"] = "Obstacle ahead"
    elif field == "elapsed":
        changed[field][0, 0, 0] -= .1
    else:
        changed[field].flatten()[-1] += .01
    frozen(changed)
    assert (frozen.hits, frozen.misses) == (0, 2)


def test_cache_lru_eviction_and_clear(frozen):
    a, b, c = history(1), history(2), history(3)
    frozen(a)
    frozen(b)
    frozen(a)
    frozen(c)
    frozen(b)
    assert (frozen.hits, frozen.misses) == (1, 4)
    assert len(frozen.cache) == 2
    frozen.clear()
    assert not frozen.cache


@pytest.mark.parametrize("count, capacity, retired", [(1, 2, []), (2, 2, []), (3, 2, ["episode:1"])])
def test_work_replay_ids_visits_and_retirement_only_above_capacity(count, capacity, retired):
    result = seeded_work(count=count, capacity=capacity)
    result.work()
    assert result.s["ready"] == [] and result.s["round"] == 0
    assert result.s["retired_ids"] == retired == result.learner.retired
    assert len(result.s["metrics"]) == count * result.learner.recipe.rounds
    assert all(r["visits"] >= 2 for r in result.s["replay"])
    assert result.learner.calls[0][0] == ["episode:1"]
    assert len(result.s["replay"]) <= capacity
    exact(result.store.saved["learner"], result.learner.state_dict())


@pytest.mark.parametrize("stop_commit", [2, 3, 4, 5, 6, 7])
def test_work_resume_reproduces_committed_rng_visits_retirement_and_fit_order(stop_commit):
    uninterrupted = seeded_work()
    uninterrupted.work()
    interrupted = seeded_work()
    interrupted.store.stop_after = stop_commit
    with pytest.raises(Pause):
        interrupted.work()
    interrupted.store.stop_after = None
    resumed = runner(store=interrupted.store)
    resumed.work()
    exact(resumed.s, uninterrupted.s)
    exact(resumed.learner.state_dict(), uninterrupted.learner.state_dict())


@pytest.mark.parametrize("stop_commit", [2, 3, 4, 5])
def test_label_resume_does_not_repeat_committed_scoring_or_next_features(stop_commit):
    scorer = Mock()
    scorer.score_rgb.return_value = {"score": 5.}
    result = runner(scorer=scorer)
    result.s.update(stage="label", current=pending_episode(result), n=2)
    result.save()
    result.store.stop_after = stop_commit
    with pytest.raises(Pause):
        result.label()
    result.store.stop_after = None
    resumed = runner(store=result.store, scorer=scorer, features=result.features)
    records = resumed.label()
    assert scorer.score_rgb.call_count == 2 and len(result.features.calls) == 2
    assert [r["id"] for r in records] == ["train:91:0:0", "train:91:0:1"]
    assert all(r["visits"] == 0 for r in records)
    ep = resumed.s["current"]
    np.testing.assert_allclose(ep["raw_stage_rewards"], [.105, 1.105])
    np.testing.assert_allclose(ep["coded_stage_rewards"], [.0105, .1105])
    assert records[-1]["discount"] == 0
    assert ep["coded_return"] == pytest.approx(ep["raw_return"] / 10)
    assert resumed.s["replay"] == [] and resumed.s["retired_ids"] == []


def test_evaluation_label_restores_tiny_head_state_rng_and_never_enters_replay():
    learner = Learner(Heads(hidden=3, width=2, seed=33), Recipe(integration_draws=2, rank=0), 91, "td_replay")
    scorer = Mock()
    scorer.score_rgb.return_value = {"score": 5.}
    result = runner(learner=learner, scorer=scorer)
    result.s["current"] = pending_episode(result, evaluation=True)
    before = copy.deepcopy(learner.state_dict())
    result.label()
    exact(learner.state_dict(), before)
    assert np.isfinite(result.s["current"]["heldout_nll"])
    assert result.s["current"]["critic_mse"] >= 0
    assert result.s["replay"] == [] and result.s["ready"] == []


def test_label_rejects_incomplete_or_truncated_episode_before_scoring():
    result = runner()
    result.s["current"] = pending_episode(result)
    result.s["current"]["complete"] = False
    with pytest.raises(RuntimeError, match="cannot receive task returns"):
        result.label()
    result.s["current"]["complete"] = True
    result.s["current"]["rows"][-1]["truncated"] = True
    with pytest.raises(RuntimeError, match="cannot receive task returns"):
        result.label()
    result.scorer.score_rgb.assert_not_called()


def test_save_captures_session_history_and_learner_in_one_checkpoint():
    result = runner()
    causal = Mock()
    causal.snapshot.return_value = {"live": ["causal-past"]}
    session = Mock()
    session.snapshot.side_effect = lambda capture_external: {"terminal": True, "external": capture_external()}
    result.save(session=session, history=causal)
    exact(result.store.saved["history"], {"live": ["causal-past"]})
    exact(result.store.saved["session"]["external"]["learner"], result.store.saved["learner"])
    result.s["history"]["live"].append("not committed")
    assert result.store.saved["history"]["live"] == ["causal-past"]


@pytest.mark.parametrize("stop_commit", [6, 7, 8, 9])
def test_run_label_to_fit_handoff_resumes_without_duplicate_episode_or_ready_ids(stop_commit, monkeypatch):
    monkeypatch.setattr(trajectory_module.time, "monotonic", lambda: 0.)

    def skip_final_evaluation(self):
        assert self.s["target"] == self.total
        self.s["stage"] = "done"
        self.save()

    monkeypatch.setattr(Trajectory, "evaluate", skip_final_evaluation)

    def prepare():
        scorer = Mock()
        scorer.score_rgb.return_value = {"score": 5.}
        result = runner(scorer=scorer)
        result.total = 2
        result.learner.archive = SimpleNamespace(diagnostics=lambda: {})
        result.s.update(stage="label", n=2, current=pending_episode(result))
        result.save()
        return result

    baseline, interrupted = prepare(), prepare()
    baseline.run()
    interrupted.store.stop_after = stop_commit
    with pytest.raises(Pause):
        interrupted.run()
    interrupted.store.stop_after = None
    resumed = runner(store=interrupted.store, scorer=interrupted.scorer, features=interrupted.features)
    resumed.total = 2
    resumed.learner.archive = SimpleNamespace(diagnostics=lambda: {})
    resumed.run()
    exact(resumed.s, baseline.s)
    assert len(resumed.s["episodes"]) == 1 and len(resumed.s["metrics"]) == 4
    assert resumed.s["ready"] == []
    assert interrupted.scorer.score_rgb.call_count == 2


@pytest.mark.parametrize("acquired,milestone", [(4, 4), (5, 4), (8, 8), (10, 8)])
@pytest.mark.parametrize("complete", [False, True], ids=["active", "terminal"])
def test_collect_resume_finishes_current_training_transaction_before_milestone_evaluation(
    acquired, milestone, complete, monkeypatch
):
    scorer = Mock()
    scorer.score_rgb.return_value = {"score": 5.}
    original = runner(scorer=scorer)
    episode = pending_episode(original)
    terminal_row = episode["rows"].pop()
    episode["scores"].pop()
    episode["rows"][0]["arrival"] = acquired - (2 if complete else 1)
    terminal_row["arrival"] = acquired - 1 if complete else acquired
    if complete:
        episode["rows"].append(terminal_row)
        episode["scores"].append(None)
    episode["complete"] = complete
    original.s.update(stage="collect", n=acquired, current=episode,
                      session={"terminal": complete}, history={"live": ["saved training history"]})
    original.save()
    resumed = runner(store=original.store, scorer=scorer)
    resumed.learner.archive = SimpleNamespace(diagnostics=lambda: {})
    saved_episode = resumed.s["current"]
    episode_id = saved_episode["episode_id"]
    saved_session = resumed.s["session"]
    saved_history = resumed.s["history"]
    events = []
    label, work = resumed.label, resumed.work

    def continue_training(seed, *, environment=None, video=False):
        events.append("continue")
        assert environment is None and not video
        assert resumed.s["current"] is saved_episode
        assert resumed.s["session"] is saved_session
        assert resumed.s["history"] is saved_history
        assert saved_episode["episode_id"] == episode_id
        if not saved_episode["complete"]:
            # This is the episode's real terminal receipt, not a budget cutoff.
            saved_episode["rows"].append(terminal_row)
            saved_episode["scores"].append(None)
            saved_episode["complete"] = True
            resumed.s["n"] += 1
        return saved_episode

    def label_training():
        events.append("label")
        assert resumed.s["current"] is saved_episode
        assert saved_episode["complete"] and not saved_episode["evaluation"]
        return label()

    def work_training():
        events.append("work")
        assert resumed.s["current"] is None
        assert resumed.s["ready"] == [episode_id + ":0", episode_id + ":1"]
        work()

    def evaluate_after_training():
        events.append("evaluate")
        assert events == ["continue", "label", "work", "evaluate"]
        assert resumed.s["target"] == milestone
        assert resumed.s["current"] is resumed.s["session"] is resumed.s["history"] is None
        assert resumed.s["ready"] == []
        assert len(resumed.s["episodes"]) == 1
        assert resumed.s["episodes"][0]["episode_id"] == episode_id
        assert len(resumed.s["metrics"]) == 2 * resumed.learner.recipe.rounds
        # Stop after this panel boundary; no further physics is needed here.
        resumed.s["stage"] = "done"
        resumed.save()

    monkeypatch.setattr(resumed, "episode", continue_training)
    monkeypatch.setattr(resumed, "label", label_training)
    monkeypatch.setattr(resumed, "work", work_training)
    monkeypatch.setattr(resumed, "evaluate", evaluate_after_training)
    summary = resumed.run()
    assert events == ["continue", "label", "work", "evaluate"]
    assert summary["n"] == acquired + (not complete)
    assert [r["id"] for r in resumed.s["replay"]] == [episode_id + ":0", episode_id + ":1"]
    assert [r["arrival"] for r in resumed.s["replay"]] == (
        [acquired - 2, acquired - 1] if complete else [acquired - 1, acquired]
    )
    assert all(metric["n"] == summary["n"] for metric in summary["metrics"])


def test_terminal_episode_continuation_returns_saved_current_without_new_acquisitions(monkeypatch):
    result = runner()
    current = pending_episode(result)
    current["simulator_config"] = asdict(simulator(8, result.total))
    result.s.update(stage="collect", n=8, current=current, session={"terminal": True})
    before = copy.deepcopy(result.s)
    session = Mock()
    session_factory = Mock(return_value=session)
    worker_factory = Mock()
    monkeypatch.setattr(trajectory_module, "ControlSession", session_factory)
    monkeypatch.setattr(trajectory_module, "SimulatorWorker", worker_factory)
    monkeypatch.setattr(trajectory_module, "_History", Mock())
    returned = result.episode(seed=91)
    assert returned is current and result.s["current"] is current
    exact(result.s, before)
    session.restore.assert_called_once_with(before["session"])
    session.start.assert_not_called()
    session.resume.assert_not_called()
    session.next_interval.assert_not_called()
    session.close.assert_called_once_with(timeout=120.)
    assert result.store.commits == 0


@pytest.mark.parametrize("target,actual", [(4, 6), (8, 10)])
def test_evaluation_panels_report_actual_acquisitions_separately_from_milestone(target, actual, monkeypatch):
    result = runner()
    result.s.update(stage="evaluate", target=target, n=actual)
    visited = []

    def heldout_episode(seed, *, environment=None, video=False):
        assert result.s["current"] is None
        assert result.s["n"] == actual
        visited.append(environment)
        result.s["current"] = {
            "episode_id": f"eval:{seed}:{result.s['episode_counter']}",
            "evaluation": True, "complete": True, "environment": environment,
            "success": False, "raw_return": 1.0, "frames": [], "rows": [],
        }
        return result.s["current"]

    monkeypatch.setattr(result, "episode", heldout_episode)
    monkeypatch.setattr(result, "label", lambda: [])
    result.evaluate()
    assert visited == ["initial_A", "moderate_A"]
    assert result.s["n"] == actual
    assert result.s["stage"] == ("done" if target == result.total else "collect")
    assert len(result.s["evaluations"]) == 2
    for panel in result.s["evaluations"]:
        assert panel["n"] == target
        assert panel["acquired_count"] == actual
        assert len(panel["episodes"]) == 1
    exact(result.store.saved["evaluations"], result.s["evaluations"])
    assert result.store.progress["n"] == actual
    assert result.s["replay"] == result.s["ready"] == result.s["retired_ids"] == []
