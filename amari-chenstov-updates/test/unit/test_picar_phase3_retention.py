"""CPU-only conditional handover, immutable references and scheduler recovery."""

import copy
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

import pytest

from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase3.config import Recipe, configuration, likelihood_key, panels, recipe_dict
from numerical_experiment.picar_phase3_retention import campaign as c
import test_picar_fix3_student as fixtures
from test_picar_fix3_student import setup  # noqa: F401 (shared fixture)
from test_picar_phase2_student import assert_equal, runtime  # noqa: F401 (shared fixture)


class Clock:
    def __init__(self):
        self.seconds = 0.

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


def summary(recipe, seed=193101, *, easy=(.25, 0., 0.), hard=(0., 0., 0.), asset_id="asset"):
    evaluations = []
    for checkpoint, count, index in (("initial", 0, 0), ("midpoint", 320, 1), ("final", 640, 2)):
        for spec, rates in zip(panels(recipe), (easy, hard)):
            n = len(spec["seeds"])
            rows = [{"seed": s, "success": j < round(n*rates[index]), "complete": True,
                "truncated": False, "evaluation": True, "learning": False, "explore": False,
                "reward_scoring": False, "actor": "learned_student", "videos": [],
                "acquired_count": count} for j, s in enumerate(spec["seeds"])]
            evaluations.append({"checkpoint": checkpoint, "difficulty": spec["difficulty"],
                "seeds": list(spec["seeds"]), "config": asdict(spec["config"]),
                "acquired_count": count, "initial_asset_id": asset_id, "rows": rows})
    return {"schema_version": 1, "settings": {**recipe_dict(recipe), "seed": seed},
        "seed": seed, "acquired_count": 640, "evaluations": evaluations,
        "timings": {"active_seconds": 2400., "evaluation_seconds": 600.}}


def complete(path, config, result):
    with RunStore(path, config, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"stage": "done"}, progress={"phase": 3, "counts": {"step": 640}})
        store.complete({**result, "unit_path": str(path)})


@pytest.fixture
def source(tmp_path, monkeypatch):
    frozen = "numerical_experiment/picar_phase3/config.py"
    hashes = {frozen: sha256_file(c.PROJECT / frozen)}
    monkeypatch.setattr(c, "old_sources", lambda: dict(hashes))
    root = tmp_path / "old"
    teacher = tmp_path / "original-demonstrations"
    complete(teacher, {"schema_version": 1}, {"provenance": "original-data-only"})
    config = {"schema_version": 1, "phase": 3, "version": c.SOURCE_VERSION,
        "source_sha256": hashes, "prior_study_seconds": c.PRIOR_STUDY_SECONDS,
        "max_hours": 6., "model_source": str((tmp_path / "model").resolve()),
        "teacher_source": str(teacher), "teacher_completion_sha256": sha256_file(teacher / "COMPLETED")}
    base = Recipe("common", temperature=2.8375266913896, action_variance=1., exploration_std=8.,
                  learning_rate=.003, optimizer_steps=2)
    asset = root / "units/bootstrap-initial"
    complete(asset, {**config, "operation": "bootstrap", "settings": {"likelihood_key": likelihood_key(base)}},
             {"execution": {"model_provenance": "fake-pinned-Qwen"}})
    asset_id = sha256_file(asset / "COMPLETED")
    for ridge, name in zip((.1, .03, .3), c.CONTROL_NAMES):
        recipe = replace(base, name=name, ridge=ridge)
        result = summary(recipe, asset_id=asset_id)
        complete(root / f"units/pilot-{name}",
                 {**config, "operation": "pilot", "settings": result["settings"]}, result)
    with RunStore(root / "orchestration", config, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"charged_seconds": 9000.}, progress={"phase": 3, "phase3_active_seconds": 9000.})
    return root


def tree_hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*")
            if p.is_file() and p.name != ".writer.lock"}


def test_missing_or_wrong_seed_screen_never_requests_stop(source, tmp_path):
    last = source / f"units/pilot-{c.CONTROL_NAMES[-1]}"
    (last / "COMPLETED").unlink()
    with pytest.raises(RuntimeError, match="all three completed"):
        c.seal_source(source, tmp_path / "new")
    assert not (source / "STOP").exists()


def test_feasible_screen_leaves_old_writer_running(source, tmp_path, monkeypatch):
    _, values = c.screen_controls(source)
    values[0] = {**summary(c._recipe(values[0]), easy=(.25, .25, .25), hard=(0., .1, .1)),
                 "unit_path": values[0]["unit_path"]}
    original = c.screen_controls
    monkeypatch.setattr(c, "screen_controls", lambda p: (original(p)[0], values))
    before = tree_hashes(source)
    result = c.seal_source(source, tmp_path / "new")
    assert result["status"] == "original_campaign_retained" and not result["takeover"]
    assert tree_hashes(source) == before


def test_retained_but_absent_learning_is_also_eligible_for_takeover(source):
    _, values = c.screen_controls(source)
    for i, value in enumerate(values):
        values[i] = {**summary(c._recipe(value), easy=(.25, .25, .25)), "unit_path": value["unit_path"]}
    assert c.takeover_required(values)


def test_wait_handover_uses_durable_parent_charge_and_authoritative_lock(source, tmp_path, monkeypatch):
    path = source / "orchestration"
    writer = RunStore(path, c.read_json(path / "config.json"), resume=True, checkpoint_policy="journal-v1")
    sleeps = []

    def release(_):
        assert (source / "STOP").exists()
        sleeps.append(True)
        writer.checkpoint({"charged_seconds": 9123.})
        writer.close()

    markers = {str(p): sha256_file(p) for p in source.glob("units/*/COMPLETED")}
    monkeypatch.setattr(c.time, "sleep", release)
    result = c.seal_source(source, tmp_path / "new")
    assert len(sleeps) == 1 and result["phase3_active_seconds"] == 9123.
    assert result["status"] == "superseded_after_ridge_screen"
    marker = sha256_file(path / "COMPLETED")
    assert c.seal_source(source, tmp_path / "new") == result
    assert sha256_file(path / "COMPLETED") == marker
    assert all(sha256_file(Path(p)) == h for p, h in markers.items())
    with pytest.raises(ValueError, match="different supervisor"):
        c.seal_source(source, tmp_path / "another")


@pytest.mark.parametrize("acquired,spent", [(0, 0), (7, 1)])
def test_preserve_partial_old_repair_and_count_only_acquired_attempts(source, tmp_path, acquired, spent):
    config, controls = c.screen_controls(source)
    settings = {**recipe_dict(replace(c._recipe(controls[0]), name="common_plasticity_repair")), "seed": 193101}
    path = source / "units/pilot-accidental-repair"
    with RunStore(path, {**config, "operation": "pilot", "settings": settings}, checkpoint_policy="journal-v1") as store:
        for i in range(4):
            store.checkpoint({"acquired_count": acquired, "cursor": i},
                             progress={"counts": {"step": acquired}})
    before = tree_hashes(path)
    result = c.seal_source(source, tmp_path / "new")
    assert len(result["spent_repair_rounds"]) == spent
    assert result["preserved_trajectories"][0]["acquired_count"] == acquired
    assert tree_hashes(path) == before
    assert not (path / "COMPLETED").exists()


def test_shared_budget_wait_overlap_resume_partial_and_single_closeout():
    clock = Clock()
    clock.advance(5000.)  # Waiting before construction is not new processing.
    budget = c.SharedBudget(9000., clock=clock)
    assert budget.new() == 0 and budget.total() == 9000.
    clock.advance(50.)
    assert budget.new() == 50. and budget.total() == 9050.
    resumed = c.SharedBudget(9000., budget.new(), clock=clock)
    clock.advance(25.)
    assert resumed.total() == 9075. and resumed.new() == 75.
    late = c.SharedBudget(21000., clock=clock)
    assert not late.admit(2400.) and late.admit(2400., partial=True)
    clock.advance(211.)
    assert not late.admit(0., partial=True)
    assert resumed.total() + c.CLOSEOUT_SECONDS == resumed.old + resumed.new() + 300.


def test_initial_marker_full_panel_reuse_and_gentle_choice(source):
    _, values = c.screen_controls(source)
    asset, marker, initial, _ = c.source_initial(source, values)
    assert marker == sha256_file(asset / "COMPLETED")
    assert [len(p["rows"]) for p in initial] == [20, 10]
    old = c._recipe(values[0])
    gentle = replace(old, name="control_retention_repair", learning_rate=.001, optimizer_steps=1,
                     pi=.01, exploration_std=4., action_variance=16.)
    improved = {**summary(gentle, easy=(.25, .2, .15), asset_id=marker), "unit_path": "gentle"}
    assert c.choose_control(values, improved).name == gentle.name
    worse = {**summary(gentle, asset_id=marker), "unit_path": "gentle"}
    assert c.choose_control(values, worse).name == values[0]["settings"]["name"]
    assert likelihood_key(gentle) != likelihood_key(old)
    cfg = configuration(gentle)
    assert cfg.version == c.SOURCE_VERSION
    assert (cfg.learning.exploration_std, cfg.learning.temperature, cfg.learning.action_variance) == (
        4., 2.8375266913896, 16.)
    invalid = copy.deepcopy(values)
    invalid[0]["evaluations"][0]["initial_asset_id"] = "wrong"
    with pytest.raises(ValueError, match="initial asset"):
        c.source_initial(source, invalid)


@pytest.fixture
def fake_execution(monkeypatch):
    clock, calls, fail = Clock(), [], {"seed": None, "used": False, "mismatch": False,
        "assets": [], "bootstrap_interrupt": False, "bootstrap_used": False, "gentle_bad": False}
    monkeypatch.setattr(c.time, "monotonic", clock)
    monkeypatch.setattr(c, "check_storage", lambda *a, **kw: None)
    monkeypatch.setattr("numerical_experiment.picar.report.build_notebook", lambda *a: None)

    def software(store):
        calls.append(("software",))
        clock.advance(20.)
        return {"passed": True}

    class Execution:
        def __init__(self, model_path, offline, **kwargs):
            pass

        def bootstrap(self, recipe, store, progress, teacher_source):
            saved = store.load_checkpoint(trusted=True)
            calls.append(("bootstrap", recipe.action_variance, bool(saved)))
            clock.advance(840.)
            if fail["bootstrap_interrupt"] and not fail["bootstrap_used"]:
                fail["bootstrap_used"] = True
                store.checkpoint({"stage": "initial_nll", "nll_step": 120}, progress={"phase": 3})
                raise InterruptedError("bootstrap interrupted")
            store.checkpoint({"stage": "done"}, progress={"phase": 3})
            return {"old_fisher_reused": False, "new_teacher_transitions": 0,
                    "likelihood_key": likelihood_key(recipe), "timings": {"active_seconds": 840.}}

        def pilot(self, recipe, seed, store, progress, asset, initial_evaluations):
            assert ("software",) in calls
            marker = sha256_file(asset / "COMPLETED")
            fail["assets"].append((recipe.action_variance, marker, bool(initial_evaluations)))
            if initial_evaluations:
                assert [len(p["rows"]) for p in initial_evaluations] == [20, 10]
                assert all(p["initial_asset_id"] == marker for p in initial_evaluations)
            saved = store.load_checkpoint(trusted=True)
            calls.append(("pilot", recipe.name, seed, bool(saved)))
            clock.advance(100.)
            if seed == fail["seed"] and not fail["used"]:
                fail["used"] = True
                store.checkpoint({"stage": "collect", "acquired_count": 320},
                                 progress={"phase": 3, "counts": {"step": 320}})
                raise InterruptedError("fake pilot interrupted")
            bad = fail["mismatch"] and recipe.fd_strength or fail["gentle_bad"] and recipe.name == "control_retention_repair"
            easy = (.25, .25, 0.) if bad else (.25, .25, .25)
            result = summary(recipe, seed, easy=easy, hard=(0., .1, .1), asset_id=marker)
            store.checkpoint({"stage": "done", "acquired_count": 640},
                             progress={"phase": 3, "counts": {"step": 640}})
            return result

        def probe(self, recipe, asset):
            calls.append(("probe", recipe_dict(recipe)))
            clock.advance(30.)
            return {"fd_convergence_pass": True, "selected_fd_step": .01,
                "initial_asset_marker_sha256": sha256_file(asset / "COMPLETED"),
                "fd_strength_rows": [{"target_gradient_ratio": r, "fd_strength": r*10} for r in (.01, .1, 1.)],
                "fit_cost_rows": [{"fd_strength": 0., "median_update_seconds": 1.},
                                  {"fd_strength": 1., "median_update_seconds": 1.5}]}

    monkeypatch.setattr(c, "_software", software)
    monkeypatch.setattr(c, "_Execution", Execution)
    return clock, calls, fail


def test_complete_scheduler_order_references_and_exact_once_accounting(source, tmp_path, fake_execution):
    clock, calls, flags = fake_execution
    output = tmp_path / "new"
    result = c.run(output, source)
    assert [x[:3] for x in calls if x[0] == "pilot"] == [
        ("pilot", "control_retention_repair", 193101), ("pilot", "control_retention_repair", 193102),
        ("pilot", "fd_retention_ratio_01", 193101), ("pilot", "fd_retention_ratio_01", 193102)]
    probe = next(x[1] for x in calls if x[0] == "probe")
    assert (probe["learning_rate"], probe["optimizer_steps"], probe["pi"], probe["exploration_std"],
            probe["action_variance"]) == (.001, 1, .01, 4., 16.)
    assert calls.count(("bootstrap", 16., False)) == 1
    assert flags["assets"][0][2] is False and all(row[2] for row in flags["assets"][1:])
    assert len({row[1] for row in flags["assets"]}) == 1
    old_marker = sha256_file(source / "units/bootstrap-initial/COMPLETED")
    assert flags["assets"][0][1] != old_marker
    assert result["total_active_seconds"] == 1590.  # 20 software + 840 bootstrap + 400 pilots + 30 probe + 300 closeout.
    assert result["phase3_active_seconds"] == 10590.
    assert result["cumulative_study_seconds"] == c.PRIOR_STUDY_SECONDS + 10590.
    assert result["selection"]["ready_to_propose_confirmation"]
    assert result["completed_trajectories"] == 4 and result["historical_reference_trajectories"] == 3
    assert not result["next_phase_authorized"] and len(result["selection"]["repair_rounds"]) == 1
    references = c.read_completed_unit(result["reference_controls"])["summary"]
    assert references["kind"] == "phase3_reference_controls"
    assert len(references["candidate_rows"]) == 3 and references["new_trajectories"] is False
    assert references["already_charged"] is True and len(result["pilot_paths"]) == 4
    marker = sha256_file(output / "orchestration/COMPLETED")
    calls_before = list(calls)
    clock.advance(5000.)
    assert c.run(output, source, resume=True) == result
    assert calls == calls_before and sha256_file(output / "orchestration/COMPLETED") == marker


def test_completed_resume_is_artifact_only_without_torch(source, tmp_path, fake_execution):
    output = tmp_path / "no-torch"
    c.run(output, source)
    script = '''
import json, sys
from pathlib import Path
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("completed resume imported Torch")
sys.meta_path.insert(0, NoTorch())
from numerical_experiment.picar_phase3_retention import campaign as c
source, output = map(Path, sys.argv[1:])
config = json.loads((source / "orchestration/config.json").read_text())
c.old_sources = lambda: config["source_sha256"]
result = c.run(output, source, resume=True)
assert result["status"] == "ready_for_checkin"
assert "torch" not in sys.modules
'''
    before = tree_hashes(output)
    result = subprocess.run([sys.executable, "-c", script, str(source), str(output)],
                            cwd=c.PROJECT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert tree_hashes(output) == before


def test_interrupted_scheduler_resumes_partial_without_repeating_completed_units(source, tmp_path, fake_execution):
    clock, calls, fail = fake_execution
    output = tmp_path / "resume"
    fail["seed"] = 193102
    with pytest.raises(InterruptedError, match="fake pilot interrupted"):
        c.run(output, source)
    progress = c.read_json(output / "orchestration/progress.json")
    assert progress["total_active_seconds"] == 1060.
    assert progress["phase3_active_seconds"] == 10060.
    assert progress["completed_trajectories"] == 1 and progress["historical_reference_trajectories"] == 3
    assert progress["counts"]["step"] == 5 and progress["counts"]["target"] == 13
    clock.advance(5000.)  # Offline time is not processing.
    result = c.run(output, source, resume=True)
    resumed = [x for x in calls if x[0] == "pilot" and x[1] == "control_retention_repair"]
    assert resumed == [("pilot", "control_retention_repair", 193101, False),
                       ("pilot", "control_retention_repair", 193102, False),
                       ("pilot", "control_retention_repair", 193102, True)]
    assert calls.count(("software",)) == 1
    assert calls.count(("bootstrap", 16., False)) == 1
    assert result["total_active_seconds"] == 1690. and result["phase3_active_seconds"] == 10690.


def test_spent_old_repair_forbids_adjacent_fd_adjustment(source, tmp_path, fake_execution):
    _, _, fail = fake_execution
    config, controls = c.screen_controls(source)
    old = {**recipe_dict(replace(c._recipe(controls[0]), name="common_plasticity_repair")), "seed": 193101}
    with RunStore(source / "units/pilot-partial-repair", {**config, "operation": "pilot", "settings": old},
                  checkpoint_policy="journal-v1") as store:
        store.checkpoint({"acquired_count": 1}, progress={"counts": {"step": 1}})
    fail["mismatch"] = True
    result = c.run(tmp_path / "new", source)
    decision = result["selection"]
    assert len(decision["repair_rounds"]) == 2
    assert decision["repair_rounds"][0]["origin"] == "source_campaign"
    assert any("two shared repair rounds" in r["reason"] for r in decision["budget_skips"])
    assert not decision["retention_matched"]


def test_budget_skips_preserved_and_no_fd_when_not_admitted(source, tmp_path, fake_execution):
    path = source / "orchestration"
    with RunStore(path, c.read_json(path / "config.json"), resume=True, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"charged_seconds": 20000.})
    result = c.run(tmp_path / "new", source)
    assert result["pilot_paths"] == []
    assert len(result["selection"]["budget_skips"]) >= 2
    assert result["phase3_active_seconds"] <= 21600.
    assert not result["selection"]["ready_to_propose_confirmation"]


def test_software_stage_never_calls_model_or_probe(source, tmp_path, fake_execution):
    _, calls, _ = fake_execution
    result = c.run(tmp_path / "software", source, stage_limit="software")
    assert result["status"] == "software_complete"
    assert result["phase3_active_seconds"] == 9020. and result["total_active_seconds"] == 20.
    assert calls == [("software",)]
    assert not (tmp_path / "software/orchestration/COMPLETED").exists()


def test_source_or_model_identity_mismatch_precedes_stop_and_writes(source, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="model source"):
        c.run(tmp_path / "bad", source, model_path=tmp_path / "different-model")
    assert not (source / "STOP").exists()
    monkeypatch.setattr(c, "old_sources", lambda: {"wrong": "hash"})
    with pytest.raises(ValueError, match="source campaign"):
        c.run(tmp_path / "bad", source)
    assert not (source / "STOP").exists()


def test_resume_config_change_rejected_without_touching_completed_artifacts(source, tmp_path, fake_execution, monkeypatch):
    output = tmp_path / "sealed"
    c.run(output, source)
    before = tree_hashes(output)
    monkeypatch.setattr(c, "VERSION", c.VERSION + "-changed")
    with pytest.raises(Exception, match="configuration or schemas differ"):
        c.run(output, source, resume=True)
    assert tree_hashes(output) == before


def test_wait_for_third_control_does_not_stop_early(source, tmp_path, monkeypatch):
    marker = source / f"units/pilot-{c.CONTROL_NAMES[-1]}/COMPLETED"
    data = marker.read_bytes()
    marker.unlink()
    waited, validated = [], []
    original = c.screen_controls

    def validate(path):
        validated.append(True)
        return original(path)

    def finish(seconds):
        assert not (source / "STOP").exists()
        assert not validated
        waited.append(seconds)
        if len(waited) == 3:
            marker.write_bytes(data)

    monkeypatch.setattr(c, "screen_controls", validate)
    monkeypatch.setattr(c.time, "sleep", finish)
    result = c.seal_source(source, tmp_path / "new", wait=True)
    assert waited == [10., 10., 10.] and len(validated) == 1 and result["takeover"]


def test_wrong_control_seed_rejected_before_stop(source, tmp_path):
    config, values = c.screen_controls(source)
    recipe = c._recipe(values[0])
    bad = summary(recipe, seed=193999)
    complete(source / "units/pilot-wrong-seed", {**config, "settings": bad["settings"]}, bad)
    with pytest.raises(ValueError, match="fixed development seeds"):
        c.seal_source(source, tmp_path / "new")
    assert not (source / "STOP").exists()


def test_old_fallback_and_fd_use_only_old_likelihood_asset(source, tmp_path, fake_execution):
    _, calls, flags = fake_execution
    flags["gentle_bad"] = True
    result = c.run(tmp_path / "fallback", source)
    assert result["selection"]["selected_control"] == c.CONTROL_NAMES[0]
    probe = next(x[1] for x in calls if x[0] == "probe")
    assert probe["action_variance"] == 1. and probe["exploration_std"] == 8.
    marker = sha256_file(source / "units/bootstrap-initial/COMPLETED")
    assert flags["assets"][0][0] == 16. and flags["assets"][0][1] != marker
    assert all(var == 1. and asset == marker and reuse for var, asset, reuse in flags["assets"][1:])
    assert len(result["selection"]["selected_control_rows"]) == 2
    for path in map(Path, result["pilot_paths"]):
        stored = c.read_json(path / "config.json")
        assert stored["config"]["learning"]["action_variance"] == stored["settings"]["action_variance"]


def test_new_bootstrap_recovery_does_not_restart_or_change_asset(source, tmp_path, fake_execution):
    _, calls, flags = fake_execution
    flags["bootstrap_interrupt"] = True
    output = tmp_path / "bootstrap-resume"
    with pytest.raises(InterruptedError, match="bootstrap interrupted"):
        c.run(output, source)
    result = c.run(output, source, resume=True)
    boots = [r for r in calls if r[0] == "bootstrap"]
    assert boots == [("bootstrap", 16., False), ("bootstrap", 16., True)]
    assert len(list((output / "units").glob("bootstrap-*/COMPLETED"))) == 1
    assert len(result["selection"]["repair_rounds"]) == 1
    assert result["total_active_seconds"] == 2430.  # The interrupted attempt's 840 seconds remain charged.


def test_completion_checkpoint_recovery_charges_closeout_once(source, tmp_path, fake_execution, monkeypatch):
    output = tmp_path / "completion-resume"
    original, failed = RunStore.complete, []

    def interrupt(store, result, **kwargs):
        if store.root == output / "orchestration" and not failed:
            failed.append(True)
            raise InterruptedError("completion publication interrupted")
        return original(store, result, **kwargs)

    monkeypatch.setattr(RunStore, "complete", interrupt)
    with pytest.raises(InterruptedError, match="completion publication"):
        c.run(output, source)
    result = c.run(output, source, resume=True)
    assert result["total_active_seconds"] == 1590. and result["phase3_active_seconds"] == 10590.
    assert result["closeout_reserve_seconds"] == 300.


def test_changed_teacher_seal_prevents_new_initialization(source, tmp_path, fake_execution):
    config, _ = c.screen_controls(source)
    marker = Path(config["teacher_source"]) / "COMPLETED"
    marker.write_text("corrupt")
    with pytest.raises(ValueError, match="demonstration completion marker"):
        c.run(tmp_path / "teacher-mismatch", source)
    assert not any(row[0] == "bootstrap" for row in fake_execution[1])


def test_real_cpu_bootstrap_uses_restored_data_and_fresh_fisher_not_old_asset(tmp_path, runtime, monkeypatch):
    from numerical_experiment.picar_phase2_rescue import data as restoration
    from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent

    old = fixtures.boot(tmp_path / "old-initial", runtime)[1]
    untouched = copy.deepcopy(old)
    records = []
    for i in range(4):
        item = copy.deepcopy(old["eligible"][i % 2])
        item.update(arrival=i, episode_id=f"restored-{i}", learned=False)
        records.append(item)
    restored = {"records": records, "acquired_count": 8, "episodes": copy.deepcopy(old["episodes"]),
                "provenance": {"kind": "synthetic-original-records"}}
    monkeypatch.setattr(restoration, "restore_demonstrations", lambda source: restored)
    config, sim = runtime

    def tiny_configuration(recipe, *, bootstrap=False):
        return replace(config, learning=replace(config.learning, fd_strength=0., optimizer_steps=1,
            action_variance=recipe.action_variance, temperature=recipe.temperature, learning_rate=.001))

    monkeypatch.setattr(c, "configuration", tiny_configuration)
    monkeypatch.setattr(c, "simulation", lambda *a, **kw: sim)
    executor = c._Execution(None, True)
    model = NormalizedStudent(fixtures.model(), mode="feature_and_projection_rms")
    monkeypatch.setattr(executor, "model_for", lambda recipe: model)
    calls = fixtures.FakeTeacher.calls, fixtures.FakeReward.calls
    recipe = Recipe("new-likelihood", action_variance=16., exploration_std=4.)
    with RunStore(tmp_path / "new-initial", {"schema_version": 1}, checkpoint_policy="journal-v1") as store:
        result = executor.bootstrap(recipe, store, lambda: None, tmp_path / "teacher")
        actual = store.load_checkpoint(trusted=True)
    assert actual["kl_step"] == actual["nll_step"] == actual["learner"]["update_count"] == 4
    assert actual["learner"]["retired_count"] == 2 and old["learner"]["retired_count"] == 4
    assert sum(float(d.sum()) for d in actual["learner"]["diagonal"].values()) > 0
    assert actual["identity"]["config"]["learning"]["action_variance"] == 16.
    assert result["old_fisher_reused"] is False and result["new_teacher_transitions"] == 0
    assert calls == (fixtures.FakeTeacher.calls, fixtures.FakeReward.calls)
    assert_equal(old, untouched)


def test_model_content_and_compute_identity_not_personal_paths():
    expected = {"kind": "qwen2_5_vl", "checkpoint_manifest": {"sha256": "bits"},
                "compute_policy": {"top_dtype": "torch.float32"}, "source": "/original/local/copy"}
    actual = {**expected, "source": "pinned-repository", "resolved_path": "/project/cache"}
    c._check_model(actual, expected)
    for field, changed in (("checkpoint_manifest", {"sha256": "different"}),
                           ("compute_policy", {"top_dtype": "torch.bfloat16"})):
        with pytest.raises(ValueError, match="frozen model bits"):
            c._check_model({**actual, field: changed}, expected)


def test_reported_costs_share_one_clock_snapshot(source, tmp_path, fake_execution, monkeypatch):
    clock = fake_execution[0]

    def ticking():
        clock.advance(.001)
        return clock.seconds

    monkeypatch.setattr(c.time, "monotonic", ticking)
    result = c.run(tmp_path / "clock-snapshot", source)
    assert result["phase3_active_seconds"] == result["old_phase3_seconds"] + result["total_active_seconds"]
    assert result["cumulative_study_seconds"] == c.PRIOR_STUDY_SECONDS + result["phase3_active_seconds"]
