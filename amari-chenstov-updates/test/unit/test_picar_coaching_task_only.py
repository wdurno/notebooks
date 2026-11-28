"""Bounded Fix 6 v4 regressions: CPU stubs, no Qwen, simulator, or training."""

import ast
import copy
from dataclasses import asdict, replace
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar.fix2_data import TASK, history_batch
from numerical_experiment.picar_head_awac.data import pack_history, unpack_history
from numerical_experiment.picar_coaching import campaign, config, readiness, trajectory
from numerical_experiment.picar_coaching.coach import CoachingWindows, GeometricCoach
from numerical_experiment.picar_coaching.scoring import PendingComplianceError
from numerical_experiment.picar_coaching.trajectory import CoachingTrajectory, episode_summary
from src.robotics.protocol import CommandReceipt, Control


def forbidden(*args, **kwargs):
    raise PendingComplianceError("task-only path attempted compliance inference or reward claim")


class Frames:
    def __init__(self):
        self.values = {}

    def put(self, rgb):
        key = str(len(self.values))
        self.values[key] = rgb.copy()
        return key

    def get(self, key):
        return self.values[key].copy()


class Store:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state)
        self._current = None if state is None else "checkpoint"
        self.frames = Frames()
        self.progress = None

    def load_checkpoint(self, *, trusted):
        assert trusted
        return copy.deepcopy(self.state)

    def checkpoint(self, state, *, progress):
        self.state, self.progress = copy.deepcopy(state), copy.deepcopy(progress)
        self._current = "checkpoint"


class Learner:
    version = 0

    def state_dict(self):
        return {"version": self.version}

    def load_state_dict(self, state):
        self.version = state["version"]


class Teacher:
    provenance = {"test_only": True}
    last_diagnostic = {"test_only": True}
    histories = []

    def __init__(self, encoder):
        pass

    def distribution(self, history, probability_floor):
        self.histories.append(copy.deepcopy(history))
        text = " ".join(c["text"] for c in history["contexts"][0]).lower()
        action = 0 if "steer left" in text else 1 if "steer right" in text else 2
        p = torch.nn.functional.one_hot(torch.tensor([action]), 8).float()
        return SimpleNamespace(executed_probabilities=p, original_probabilities=p,
                               reference_logits=torch.zeros(1, 7))


class FixedCoach(GeometricCoach):
    def propose(self, diagnostic, cfg):
        side = "left" if diagnostic.get("seed", 0) % 2 == 0 else "right"
        return {"text": f"Steer {side} while driving forward.", "reason": "test only",
                "maneuver": f"steer_{side}", "safety": False, "diagnostics": {}}


def diagnostic(now=0., seed=0, yaw=0.):
    return {"privileged": True, "seed": seed, "simulator_time_s": now,
            "layout": {"start": (.5, 0.), "target": (2.9, 0.), "obstacle": (1.575, 0.), "mirror": 1},
            "position_m": (.5, 0., .045), "orientation_xyzw": (0., 0., math.sin(yaw / 2), math.cos(yaw / 2)),
            "geometry": {"speed_mps": 0., "clearance_m": 2.}, "steering_gain": 1., "end_reason": None}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr("numerical_experiment.picar.models.load_qwen_model", forbidden)
    monkeypatch.setattr(trajectory, "SimulatorWorker", forbidden)
    monkeypatch.setattr(readiness, "PyBulletBackend", forbidden)
    monkeypatch.setattr(trajectory, "CenteringTeacher", Teacher)
    Teacher.histories = []


def runner(*, condition="G", collection=False, store=None, **kwargs):
    scores = iter([0., 4., 8.])
    scorer = SimpleNamespace(score_rgb=lambda rgb: {"score": next(scores)})
    return CoachingTrajectory(Learner(), lambda h: torch.zeros(64), scorer,
        store if store is not None else Store(), lambda: None,
        model=SimpleNamespace(encoder=None), condition=condition,
        phase="initialization" if collection else "assessment", seed=17, total=16,
        collection=collection, **kwargs)


def labeled_episode(value, *, bonus=10.):
    cfg = replace(trajectory.simulator(fixed="initial_A"), width=8, height=8)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    ref = value.store.frames.put(rgb)
    coach = CoachingWindows(1, condition=value.condition, total=value.total,
                            initialization=value.collection, coach=FixedCoach())
    first = coach.advance(0., 0., diagnostic(), cfg)
    if first["instruction_id"]:
        receipt = CommandReceipt("test", "session", 0, "frame", .5,
                                 Control(steering_rad=-.2, drive_mps=.1), True, "executing", 1.)
        coach.record_action(receipt, generated_at_s=.3, decision_time_s=.3,
                            exposed_instruction_ids=[first["instruction_id"]], source="teacher" if value.collection else "student")
    coach.finish(3., transition_id="r2")
    rows, end = [], 0.
    for i, elapsed in enumerate((.5, 1., 1.5)):
        end += elapsed
        row = {"id": f"r{i}", "frame_ref": ref, "elapsed": elapsed,
               "bonus": bonus if i == 2 else 0., "terminal": i == 2, "truncated": False,
               "end_time": end, "observed_controls": [0., 0., 0., 0.],
               "action": torch.zeros(7) if i < 2 else None,
               "feature": None if value.collection or i == 2 else torch.zeros(64), "arrival": i}
        if i < 2:
            context = coach.policy_context(end, scheduled_expiry=True)
            history = history_batch([{"rgb": rgb, "time": end, "elapsed": elapsed,
                "action": [0., 0., 0., 0.]}], contexts=[{"text": TASK, "issued_at": 0.}, *context["instructions"]],
                statuses=context["statuses"], deadline=3., width=8, height=8)
            row["history"] = pack_history(history, value.store.frames)
            if value.collection:
                row["teacher_probabilities"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        rows.append(row)
    value.coach = coach
    value.s["current"] = {"complete": True, "evaluation": False, "episode_id": "test-episode",
        "rows": rows, "scores": [None] * 3, "compliance": {}, "initial_frame_ref": ref,
        "initial_frame_time": 0., "initial_controls": [0., 0., 0., 0.], "simulator_config": asdict(cfg),
        "adherence_enabled": False, "reward_mode": "task_only"}
    return value.s["current"]


@pytest.mark.parametrize("condition", config.CONDITIONS)
@pytest.mark.parametrize("collection", [False, True])
def test_default_constructor_needs_no_compliance_and_persists_treatment(condition, collection):
    value = runner(condition=condition, collection=collection)
    assert value.compliance is None and value.adherence_enabled is False
    assert value.reward_mode == value.s["reward_mode"] == "task_only"
    value.save()
    resumed = runner(condition=condition, collection=collection, store=value.store)
    assert resumed.s["reward_mode"] == "task_only" and not resumed.adherence_enabled


@pytest.mark.parametrize("condition", config.CONDITIONS)
@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("bonus", [10., -5., -1.])
def test_task_only_labels_never_score_or_claim_guidance_and_preserve_arithmetic(monkeypatch, condition, collection, bonus):
    value = runner(condition=condition, collection=collection,
                   adherence_enabled=False, compliance=SimpleNamespace(score=forbidden))
    ep = labeled_episode(value, bonus=bonus)
    monkeypatch.setattr(value, "_compliance", forbidden)
    monkeypatch.setattr(value.coach, "claim_reward", forbidden)
    windows = copy.deepcopy(value.coach.windows)
    if condition == "G":
        assert windows[0]["assessable"] and windows[0]["instructions"]
    records = value.label()
    task = [-.02, .12, .9 + bonus * abs(bonus)]
    d0, d1 = .99, .99 ** 2
    returns = [task[0] + d0 * (task[1] + d1 * task[2]), task[1] + d1 * task[2], task[2]]
    assert ep["raw_stage_rewards"] == pytest.approx(task)
    assert ep["task_stage_rewards"] == pytest.approx(task)
    assert ep["coded_stage_rewards"] == pytest.approx([r / 10 for r in task])
    assert ep["raw_returns"] == pytest.approx(returns)
    assert ep["coded_returns"] == pytest.approx([r / 10 for r in returns])
    assert ep["raw_return"] == ep["task_return"] == pytest.approx(returns[0])
    assert ep["coded_return"] == pytest.approx(returns[0] / 10)
    assert ep["guidance_return"] == 0. and ep["guidance_stage_rewards"] == [0., 0., 0.]
    assert ep["compliance"] == {} and value.coach.windows == windows
    assert all(w["reward_receipt"] is None for w in ep["coaching_windows"])
    assert episode_summary(ep)["scored_guidance_windows"] == 0
    assert [r["id"] for r in records] == ["r0", "r1"]
    assert [r["physical_rows"] for r in records] == [1, 2]
    assert [float(r["reward"]) for r in records] == pytest.approx([task[0] / 10, returns[1] / 10])
    assert [float(r["return"]) for r in records] == pytest.approx([r / 10 for r in returns[:2]])
    assert [float(r["discount"]) for r in records] == pytest.approx([d0, 0.])
    assert all(r["reward"].device.type == "cpu" for r in records)
    if collection:
        assert all("teacher_probabilities" in r for r in records)
    if condition == "G":
        history = unpack_history(records[0]["history"], value.store.frames)
        assert any("Steer left" in c["text"] for c in history["contexts"][0])
    # Re-labeling uses cached width scores and cannot create a pending reward.
    again = value.label()
    assert ep["compliance"] == {} and value.coach.windows == windows
    assert [float(r["reward"]) for r in again] == [float(r["reward"]) for r in records]


def test_direct_compliance_entrypoint_is_inert_when_disabled():
    value = runner(compliance=SimpleNamespace(score=forbidden))
    value.coach = SimpleNamespace(windows=property(forbidden))
    value._compliance()
    assert value.s["current"] is None


@pytest.mark.parametrize("condition", config.CONDITIONS)
def test_collection_entrypoint_saves_task_only_metadata_and_labels_without_compliance(monkeypatch, condition):
    class Worker:
        def __init__(self, cfg, **kwargs):
            self.config, self.now = cfg, 0.

        def observe(self):
            return SimpleNamespace(capture_time_s=self.now, rgb=bytes(8 * 8 * 3), height=8, width=8,
                                   executed_controls=Control())

        def diagnostics(self):
            return diagnostic(self.now)

        def snapshot(self):
            return {"backend": {"test_only": True}}

    class Session:
        def __init__(self, worker, access, *, infer, before_submit, **kwargs):
            self.worker, self.infer, self.before_submit = worker, infer, before_submit
            self._open, self._index = None, 0

        def start(self, **kwargs):
            self.initial = self.before_submit(self, self.worker.observe())
            self.action = self.infer(self.initial, 0)
            return self.initial

        def next_interval(self, **kwargs):
            receipt = CommandReceipt("test", "session", 0, "frame", .5,
                                     Control(drive_mps=.1), True, "executing", 1.5)
            self._open = {"action": self.action, "receipt": receipt}
            self.worker.now = .5
            self.before_submit(self, self.worker.observe())
            self.worker.now, self._index = 1., 1
            return SimpleNamespace(observation=self.initial, action=self.action, receipt=receipt,
                command=SimpleNamespace(controls=receipt.executed_controls), fallback_reason=None, timing={},
                result=SimpleNamespace(observation=self.worker.observe(), terminated=True, truncated=False,
                    end_reason="success", reward_components=SimpleNamespace(elapsed_s=1., terminal_bonus=10.)))

        def snapshot(self, *, capture_external, **kwargs):
            return {"external": capture_external(), "terminal": True}

        def pause(self, **kwargs):
            pass

        def close(self, **kwargs):
            pass

    real_simulator = trajectory.simulator
    monkeypatch.setattr(trajectory, "simulator", lambda *a, **kw: replace(real_simulator(*a, **kw), width=8, height=8))
    monkeypatch.setattr(trajectory, "SimulatorWorker", Worker)
    monkeypatch.setattr(trajectory, "CoachingSession", Session)
    monkeypatch.setattr(trajectory, "snapshot_at_capture", lambda snapshot, at: snapshot)
    monkeypatch.setattr(trajectory, "execution_segments", lambda *a: [])
    monkeypatch.setattr(trajectory, "CoachingWindows", lambda seed, **kw: CoachingWindows(1, coach=FixedCoach(), **kw))
    monkeypatch.setattr(CoachingWindows, "claim_reward", forbidden)
    value = runner(condition=condition, collection=True, compliance=SimpleNamespace(score=forbidden))
    monkeypatch.setattr(value, "_compliance", forbidden)
    ep = value.episode(23)
    assert ep["complete"] and ep["reward_mode"] == "task_only" and ep["adherence_enabled"] is False
    assert len(ep["frames"]) == 1 and ep["rows"][0]["history"]
    records = value.label()
    assert len(records) == 1 and float(records[0]["reward"]) == pytest.approx(9.996)
    assert ep["compliance"] == {} and ep["guidance_return"] == 0.
    assert ep["raw_return"] == ep["task_return"]
    assert episode_summary(ep)["reward_mode"] == "task_only"
    if condition == "G":
        assert ep["coaching_windows"][0]["assessable"]
        assert ep["rows"][0]["instruction_ids_seen"] and ep["coaching_events"]
    assert all(w["reward_receipt"] is None for w in ep["coaching_windows"])


@pytest.mark.parametrize("raw", [-2., 0., 2.])
def test_task_only_rejects_existing_compliance_instead_of_relabeling_as_neutral(raw):
    value = runner()
    ep = labeled_episode(value)
    ep["compliance"] = {"legacy": {"closing_row": "r2", "receipt": {"raw": raw}}}
    with pytest.raises(ValueError, match="task-only.*adherence"):
        value.label()
    assert "raw_returns" not in ep


@pytest.mark.parametrize("saved_enabled", [False, True])
def test_resume_rejects_reward_treatment_switch(saved_enabled):
    scorer = SimpleNamespace(score=forbidden)
    value = runner(adherence_enabled=saved_enabled, compliance=scorer)
    value.save()
    with pytest.raises(ValueError, match="reward mode.*recovery"):
        runner(store=value.store, adherence_enabled=not saved_enabled, compliance=scorer)


def test_resume_rejects_unversioned_checkpoint_with_mixed_rewards():
    value = runner()
    ep = labeled_episode(value)
    ep.update(task_return=1., raw_return=3., guidance_return=2.,
              compliance={"old": {"closing_row": "r2", "receipt": {"raw": 2.}}})
    value.save()
    value.store.state.pop("reward_mode")
    with pytest.raises(ValueError, match="reward mode.*recovery"):
        runner(store=value.store)


def test_explicit_legacy_adherence_requires_scorer_and_resumes_only_same_mode():
    with pytest.raises(ValueError, match="require.*scorer"):
        runner(adherence_enabled=True)
    value = runner(adherence_enabled=True, compliance=SimpleNamespace(score=forbidden))
    value.save()
    restored = runner(store=value.store, adherence_enabled=True, compliance=SimpleNamespace(score=forbidden))
    assert restored.adherence_enabled and restored.s["reward_mode"] == "task_and_adherence"


@pytest.mark.parametrize("geometry_ready,teacher_ready", [(True, True), (False, True), (True, False)])
def test_resource_profile_without_audit_retains_real_gates_and_no_compliance_cost(geometry_ready, teacher_ready):
    store, calls = Store(), []
    result = readiness.resource_profile(None, {"ready": geometry_ready}, None,
        {"ready": teacher_ready, "rows": [{"inference_seconds": s} for s in (.1, .3, .2)]},
        store, lambda: calls.append(True))
    assert result["ready"] is (geometry_ready and teacher_ready)
    assert result["median_teacher_seconds"] == .2
    assert result["median_compliance_seconds"] is None and result["adherence_enabled"] is False
    assert result["cuda_reserved_gib"] == 0.
    assert store.state == result and len(calls) == 1
    assert all(math.isfinite(x) and x >= 0 for x in result["forecast"]["remaining_hours"])


@pytest.mark.parametrize("audit_ready", [True, False])
def test_resource_profile_legacy_audit_cost_and_gate_are_not_silently_dropped(audit_ready):
    result = readiness.resource_profile(None, {"ready": True},
        {"ready": audit_ready, "median_scoring_seconds": 1.25},
        {"ready": True, "rows": [{"inference_seconds": .1}]}, Store(), lambda: None)
    assert result["ready"] is audit_ready
    assert result["adherence_enabled"] is True and result["median_compliance_seconds"] == 1.25


@pytest.mark.parametrize("compliance", [None, SimpleNamespace(score=forbidden)])
def test_teacher_audit_retains_paired_cue_histories_without_compliance(monkeypatch, compliance):
    class Backend:
        def __init__(self, cfg, **kwargs):
            self.config, self.time_s, self.yaw, self.seed = cfg, 0., 0., 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def observe(self):
            return SimpleNamespace(capture_time_s=self.time_s, rgb=bytes(8 * 8 * 3), height=8, width=8,
                                   executed_controls=Control())

        def reset(self, *, seed):
            self.seed = seed
            return self.observe()

        def diagnostics(self):
            return {**diagnostic(self.time_s, self.seed, self.yaw), "unsafe_contact": False}

    def command(sim, observation, controls, sequence, seconds):
        sim.time_s += seconds
        sim.yaw = -.2 if controls.steering_rad > .05 else .2 if controls.steering_rad < -.05 else 0.
        return None, SimpleNamespace(observation=sim.observe())

    real_simulator = readiness.simulator
    monkeypatch.setattr(readiness, "simulator", lambda **kw: replace(real_simulator(**kw), width=8, height=8))
    monkeypatch.setattr(readiness, "PyBulletBackend", Backend)
    monkeypatch.setattr(readiness, "CenteringTeacher", Teacher)
    monkeypatch.setattr(readiness, "GeometricCoach", FixedCoach)
    monkeypatch.setattr(readiness, "_command", command)
    store = Store()
    result = readiness.teacher_audit(SimpleNamespace(encoder=None), SimpleNamespace(score_rgb=forbidden),
                                    compliance, store, lambda: None)
    assert result["ready"] and result["total_calls"] == 12
    assert result["correct_left_turns"] and result["correct_right_turns"]
    assert result["learning_samples"] == 0 and len(Teacher.histories) == 12
    for i in range(0, 12, 2):
        u, g = result["rows"][i:i + 2]
        assert (u["condition"], g["condition"]) == ("U", "G") and u["seed"] == g["seed"]
        assert u["cue"] is None and g["cue"]
        assert len(Teacher.histories[i]["contexts"][0]) == 1
        assert len(Teacher.histories[i + 1]["contexts"][0]) == 2


def test_v4_campaign_has_no_compliance_evaluator_construction_or_gate():
    assert config.OUTPUT.name == "coaching-fix6-v4"
    assert config.CONDITIONS == ("U", "G")
    tree = ast.parse(inspect.getsource(campaign.run))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    assert "ComplianceScorer" not in names | imports
    assert "compliance_audit" not in names | imports
    components = ("collection", "heads", "development", "assessment:0", "assessment:1")
    seeds = [config.seed(name) for name in components]
    assert seeds == [config.seed(name) for name in components] and len(set(seeds)) == len(seeds)


def test_campaign_identity_records_paired_task_only_protocol_before_any_model_load(tmp_path, monkeypatch):
    class CapturedIdentity(Exception):
        pass

    source = tmp_path / "source"
    marker = source / "units/features-test/COMPLETED"
    marker.parent.mkdir(parents=True)
    marker.touch()
    identities = []

    def capture(root, identity, **kwargs):
        identities.append(copy.deepcopy(identity))
        raise CapturedIdentity

    monkeypatch.setattr(campaign, "RunStore", capture)
    monkeypatch.setattr(campaign, "selected_recipe", lambda source: config.recipe())
    monkeypatch.setattr(campaign, "fingerprints", lambda: {})
    monkeypatch.setattr(campaign, "receipt", lambda path: {"source": str(path)})
    monkeypatch.setattr(campaign, "sha256_file", lambda path: "test-only")
    monkeypatch.setattr(campaign, "StopController", lambda root: SimpleNamespace(install=lambda: None, close=lambda: None))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    for _ in range(2):
        with pytest.raises(CapturedIdentity):
            campaign.run(output=tmp_path / "output", source=source, stage="software")
    assert identities[0] == identities[1]
    identity = identities[0]
    assert identity["conditions"] == ["U", "G"]
    protocol = identity["protocol"]
    assert protocol["reward_mode"] == "task_only" and protocol["adherence_enabled"] is False
    assert protocol["compliance_raw_magnitude"] == 0. and protocol["reward_coding"] == 10.
    assert protocol["coaching_seconds"] == 4. and not protocol["physical_motion"]
    assert protocol["assessment_seeds"] == [config.seed("assessment:0"), config.seed("assessment:1")]
    assert protocol["development_seed"] not in protocol["assessment_seeds"]
