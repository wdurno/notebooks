"""Report-only resume safety, with temporary artifacts and generated CPU images."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mnist_experiment import akaike_score_gate_resume as driver
from mnist_experiment.akaike_score_gate import analysis, assets, orchestrate, trajectory
from mnist_experiment.akaike_score_gate import refresh_notebook as report_view
from mnist_experiment.akaike_score_gate.artifacts import UnitStore, file_hash, source_hashes
from mnist_experiment.akaike_score_gate.budget import WorkBudget
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError

if __package__:
    from .test_trajectory import (
        assert_same_artifacts, assert_same_science, deterministic_test_runtime,
        profile, tiny_case,
    )
else:
    from test_trajectory import (
        assert_same_artifacts, assert_same_science, deterministic_test_runtime,
        profile, tiny_case,
    )


FROZEN_RUNNER = orchestrate.Runner
REAL_REPORT_REFRESH = report_view.refresh


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def rng_state():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {"torch": torch.random.get_rng_state().clone(), "python": random.getstate(),
            "numpy": (algorithm, keys.tolist(), position, has_gauss, cached)}


@pytest.fixture(autouse=True)
def cpu_only_no_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Resume tests must not use CUDA or download data")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    monkeypatch.setattr(assets, "load_mnist_datasets", forbidden)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    study = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(orchestrate.time, "monotonic", lambda: clock.now)

    def render(root, notebook):
        write_json(notebook, {"rendered_progress": json.loads((root / "progress.json").read_text())})
        return notebook

    renderer = Mock(side_effect=render)
    monkeypatch.setattr(report_view, "refresh", renderer)
    parent_calls = []
    original = FROZEN_RUNNER.refresh

    def parent_refresh(self, *, force=False):
        parent_calls.append(force)
        return original(self, force=force)

    monkeypatch.setattr(FROZEN_RUNNER, "refresh", parent_refresh)
    runner = driver.CoalescedReportRunner(UnitStore(root, study), tmp_path / "report.ipynb")
    runner.progress.update(phase="phase5", status="running", completed_units=0)
    return SimpleNamespace(runner=runner, renderer=renderer, clock=clock, parent_calls=parent_calls)


def test_initial_render_and_duplicate_forces_still_write_progress(harness):
    runner = harness.runner
    runner.refresh(force=True)
    assert runner.notebook.is_file()
    assert runner.report_renders == 1 and runner.coalesced_refreshes == 0
    assert runner.last_refresh == harness.clock.now
    for elapsed in (1., 2., 3.):
        runner.progress["active_invocation_seconds"] = elapsed
        runner.refresh(force=True)
        assert json.loads((runner.store.root / "progress.json").read_text()) == runner.progress
    assert harness.parent_calls == [True, False, False, False]
    assert harness.renderer.call_count == runner.report_renders == 1
    assert runner.coalesced_refreshes == 3
    assert not (runner.store.root / "execution").exists()


def test_identity_is_order_independent_recursive_and_does_not_mutate():
    progress = {"status": "running", "completed_units": 3, "active_invocation_seconds": 1.,
                "updated_at": "old", "started_at": "old", "completed_at": None,
                "phases": {"phase5": {"status": "running", "updated_at": "old"}},
                "details": [{"started_at": "old", "completed_at": None, "value": 4}]}
    before = copy.deepcopy(progress)
    key = driver.report_identity(progress, 2)
    assert isinstance(key, str) and key
    assert progress == before
    changed = copy.deepcopy(progress)
    for field in ("active_invocation_seconds", "updated_at", "started_at", "completed_at"):
        changed[field] = "new"
    changed["phases"]["phase5"]["updated_at"] = "new"
    changed["details"][0].update(started_at="new", completed_at="new")
    assert driver.report_identity(dict(reversed(list(changed.items()))), 2) == key
    assert driver.report_identity(progress, 3) != key


@pytest.mark.parametrize("update", [
    {"phase": "phase4"}, {"status": "completed"}, {"error": "fit failed"},
    {"error_type": "RuntimeError"}, {"interruption": "unit limit"},
    {"completed_units": 1}, {"schema_version": 2}, {"new_diagnostic": {"value": 7}},
    {"charged_seconds": 1.}, {"phases": {"phase4": {"status": "completed"}}},
])
def test_nonvolatile_progress_changes_render_then_coalesce(harness, update):
    runner = harness.runner
    runner.refresh(force=True)
    previous = driver.report_identity(runner.progress, runner.store.limit.units)
    runner.progress.update(update)
    assert driver.report_identity(runner.progress, runner.store.limit.units) != previous
    runner.refresh(force=True)
    runner.refresh(force=True)
    assert harness.parent_calls == [True, True, False]
    assert harness.renderer.call_count == runner.report_renders == 2
    assert runner.coalesced_refreshes == 1


def test_phase_transitions_and_timestamp_only_replay(harness):
    runner = harness.runner
    runner.progress.update(error="old failure", error_type="RuntimeError", interruption="stopped")
    runner.refresh(force=True)
    runner.phase(5, "running")
    assert not {"error", "error_type", "interruption"} & runner.progress.keys()
    runner.phase(5, "running")
    runner.phase(5, "completed")
    runner.phase(5, "completed")
    runner.phase(6, "running")
    assert harness.parent_calls == [True, True, False, True, False, True]
    assert runner.report_renders == 4 and runner.coalesced_refreshes == 2


def test_newly_sealed_units_and_checkpoint_counts_invalidate_identity(harness):
    runner = harness.runner
    runner.refresh(force=True)
    unit = runner.store.unit("phase5", "report_fixture", 0)
    session = runner.store.begin(unit, ("summary.json",), resume=True)
    session.write_json("summary.json", {"schema_version": 1, "nll": 1.})
    runner.refresh(force=True)
    path = runner.store.finish(session, ("summary.json",))
    sealed = snapshot(path)
    # The completed count is still stale; the invocation's unit count must suffice.
    assert runner.progress["completed_units"] == 0 and runner.store.limit.units == 1
    runner.refresh(force=True)
    runner.refresh(force=True)
    runner.checkpoint_progress()
    assert runner.progress["completed_units"] == 1
    runner.refresh(force=True)
    assert harness.renderer.call_count == runner.report_renders == 3
    assert runner.coalesced_refreshes == 2
    assert snapshot(path) == sealed


@pytest.mark.parametrize("force", [False, True])
def test_900_second_heartbeat_is_preserved(harness, force):
    runner = harness.runner
    runner.refresh(force=True)
    harness.clock.now = 999.
    runner.refresh(force=force)
    assert harness.renderer.call_count == 1
    harness.clock.now = 1000.
    runner.refresh(force=force)
    assert harness.renderer.call_count == runner.report_renders == 2
    assert runner.last_refresh == 1000.
    assert harness.parent_calls == [True, False, False]
    harness.clock.now = 1899.
    runner.refresh()
    assert harness.renderer.call_count == 2


def test_missing_notebook_forces_rebuild_despite_unchanged_identity(harness):
    runner = harness.runner
    runner.refresh(force=True)
    runner.notebook.unlink()
    runner.refresh(force=True)
    assert runner.notebook.is_file()
    assert harness.parent_calls == [True, True]
    assert runner.report_renders == 2 and runner.coalesced_refreshes == 0


@pytest.mark.parametrize("previous_success", [False, True])
def test_failed_renderer_does_not_cache_identity_or_advance_timer(harness, previous_success):
    runner = harness.runner
    runner.notebook.write_text("previous notebook", encoding="utf-8")
    if previous_success:
        runner.refresh(force=True)
        runner.progress["completed_units"] += 1
    previous = (runner._last_report_identity, runner.last_refresh,
                runner.report_renders, runner.coalesced_refreshes)
    notebook = runner.notebook.read_bytes()
    original_render = harness.renderer.side_effect
    harness.renderer.side_effect = OSError("renderer failed")
    harness.clock.now += 1.
    with pytest.raises(OSError, match="renderer failed"):
        runner.refresh(force=True)
    assert (runner._last_report_identity, runner.last_refresh,
            runner.report_renders, runner.coalesced_refreshes) == previous
    assert runner.notebook.read_bytes() == notebook
    assert json.loads((runner.store.root / "progress.json").read_text()) == runner.progress
    harness.renderer.side_effect = original_render
    runner.refresh(force=True)
    runner.refresh(force=True)
    assert runner.report_renders == int(previous_success) + 1
    assert runner.coalesced_refreshes == 1


@pytest.mark.parametrize("phase,status", [
    ("phase5", "blocked"), ("phase5", "interrupted"), ("phase6", "completed"),
])
def test_terminal_duplicate_and_final_forces_are_never_coalesced(harness, phase, status):
    runner = harness.runner
    assert runner.terminal_phase == 6
    runner.progress.update(phase=phase, status=status)
    for _ in range(3):
        runner.refresh(force=True)
    assert harness.parent_calls == [True, True, True]
    assert runner.report_renders == 3 and runner.coalesced_refreshes == 0


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_context_tracks_runners_sets_terminal_phase_and_always_restores(harness, failure):
    runner = harness.runner
    before = snapshot(runner.store.root)
    runners = None
    try:
        with driver.coalesced_reporting(terminal_phase=5) as runners:
            assert runners == []
            created = orchestrate.Runner(runner.store, runner.notebook)
            assert isinstance(created, driver.CoalescedReportRunner)
            assert runners == [created] and created.terminal_phase == 5
            created.progress.update(phase="phase5", status="completed")
            created.refresh(force=True)
            created.refresh(force=True)
            assert created.report_renders == 2 and created.coalesced_refreshes == 0
            if failure:
                raise failure("context exit")
    except BaseException as error:
        assert failure is not None and type(error) is failure
    else:
        assert failure is None
    assert orchestrate.Runner is FROZEN_RUNNER
    assert len(runners) == 1
    assert not (runner.store.root / "execution").exists()
    assert {name: data for name, data in snapshot(runner.store.root).items()
            if name != "progress.json"} == before


def test_every_scientific_method_is_inherited_unchanged():
    assert issubclass(driver.CoalescedReportRunner, FROZEN_RUNNER)
    for name, method in vars(FROZEN_RUNNER).items():
        if callable(method) and name not in {"__init__", "refresh"}:
            assert getattr(driver.CoalescedReportRunner, name) is method, name


def test_nested_context_is_rejected_without_displacing_outer_runner(harness):
    with driver.coalesced_reporting() as outer:
        installed = orchestrate.Runner
        with pytest.raises(RuntimeError, match="override is already installed"):
            with driver.coalesced_reporting():
                pytest.fail("nested reporting context was accepted")
        assert orchestrate.Runner is installed
        created = orchestrate.Runner(harness.runner.store, harness.runner.notebook)
        assert outer == [created]
    assert orchestrate.Runner is FROZEN_RUNNER


@pytest.mark.parametrize("phase,status,study_complete", [
    ("phase5", "interrupted", False), ("phase5", "completed", False),
    ("phase6", "completed", True),
])
def test_finished_invocation_is_not_necessarily_a_finished_cohort(harness, phase, status, study_complete):
    with driver.coalesced_reporting(provenance={"reporting_only": True}) as runners:
        created = orchestrate.Runner(harness.runner.store, harness.runner.notebook)
        created.progress.update(phase=phase, status=status)
        created.refresh(force=True)
        created.refresh(force=True)
    assert runners == [created] and orchestrate.Runner is FROZEN_RUNNER
    marker, = created.store.root.glob("execution/report_resume/*/COMPLETED")
    summary = json.loads((marker.parent / "summary.json").read_text())
    assert summary["invocation_status"] == "finished"
    assert summary["study_status"] == status and summary["study_complete"] is study_complete
    assert summary["report_renders"] == created.report_renders
    assert summary["coalesced_forced_refreshes"] == created.coalesced_refreshes
    if status == "interrupted" or study_complete:
        assert created.report_renders == 2 and created.coalesced_refreshes == 0


def test_corruption_still_fails_frozen_store_and_real_report_heartbeat(harness, monkeypatch):
    runner = harness.runner
    write_json(runner.store.root / "study.json",
               {**runner.study.mapping(), "study_hash": runner.study.config_hash})
    unit = runner.store.unit("phase5", "report_fixture", 0)
    session = runner.store.begin(unit, ("summary.json",), resume=True)
    session.write_json("summary.json", {"schema_version": 1, "nll": 1.})
    path = runner.store.finish(session, ("summary.json",))
    # Exercise the frozen validator, not the harness's lightweight renderer.
    real_renderer = Mock(wraps=REAL_REPORT_REFRESH)
    monkeypatch.setattr(report_view, "refresh", real_renderer)
    runner.refresh(force=True)
    notebook = runner.notebook.read_bytes()
    write_json(path / "summary.json", {"schema_version": 1, "nll": 9.})
    corrupted = snapshot(path)
    fresh_store = UnitStore(runner.store.root, runner.study)
    with pytest.raises(RotatedArtifactError, match="corrupt immutable artifact"):
        fresh_store.completed(unit, ("summary.json",))
    harness.clock.now = 999.
    runner.refresh(force=True)
    assert real_renderer.call_count == 1
    previous = (runner._last_report_identity, runner.last_refresh, runner.report_renders)
    harness.clock.now = 1000.
    with pytest.raises(analysis.ArtifactError, match="Corrupt artifact"):
        runner.refresh(force=True)
    assert real_renderer.call_count == 2
    assert (runner._last_report_identity, runner.last_refresh, runner.report_renders) == previous
    assert snapshot(path) == corrupted and runner.notebook.read_bytes() == notebook


@pytest.fixture
def frozen_resume(tmp_path):
    study = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")
    root = tmp_path / "frozen"
    config = tmp_path / "config.json"
    write_json(config, study.mapping())
    write_json(root / "study.json", {**study.mapping(), "study_hash": study.config_hash})
    write_json(root / "phase1_ledger.json", {"source_hashes": source_hashes()})
    for path in (root / "final_ledger.json", root / "decisions/phase4_selection.json"):
        write_json(path, {"schema_version": 1, "study_hash": study.config_hash})
    # The frozen executor replays status transitions; running is not missing science.
    write_json(root / "progress.json", {"phase": "phase0", "status": "running", "phases": {}})
    argv = ["--config", str(config), "--output-root", str(root), "--resume",
            "--notebook", str(tmp_path / "report.ipynb"), "--through-phase", "5",
            "--max-units", "1", "--max-wall-seconds", "2"]
    return SimpleNamespace(root=root, study=study, argv=argv,
                           args=orchestrate.parse_arguments(argv))


def test_validate_resume_accepts_exact_frozen_settings_without_writes(frozen_resume):
    case = frozen_resume
    before = snapshot(case.root)
    result = driver.validate_resume(case.args)
    assert result["reporting_only"] is True and result["scientific_settings_changed"] is False
    assert result["heartbeat_seconds"] == 900 and result["through_phase"] == 5
    assert result["resolved_config"] == case.study.mapping()
    assert result["driver_source_sha256"] == file_hash(Path(driver.__file__))
    assert snapshot(case.root) == before


@pytest.mark.parametrize("bad", [
    "no_resume", "missing_study", "changed_study", "changed_sources",
    "missing_final_ledger", "wrong_final_study", "missing_selection", "wrong_selection_study",
])
def test_invalid_resume_fails_before_dispatch_without_artifact_changes(frozen_resume, monkeypatch, bad):
    case = frozen_resume
    argv = list(case.argv)
    if bad == "no_resume":
        argv.remove("--resume")
    elif bad.startswith("missing_"):
        relative = {"missing_study": "study.json", "missing_final_ledger": "final_ledger.json",
                    "missing_selection": "decisions/phase4_selection.json"}[bad]
        (case.root / relative).unlink()
    elif bad == "changed_study":
        value = {**case.study.mapping(), "study_hash": case.study.config_hash, "seed_root": -1}
        write_json(case.root / "study.json", value)
    elif bad == "changed_sources":
        write_json(case.root / "phase1_ledger.json", {"source_hashes": {"altered.py": "wrong"}})
    else:
        name = "final_ledger.json" if bad == "wrong_final_study" else "decisions/phase4_selection.json"
        write_json(case.root / name, {"study_hash": "different"})
    before = snapshot(case.root)
    dispatch = Mock(side_effect=AssertionError("invalid resume dispatched scientific executor"))
    monkeypatch.setattr(orchestrate, "main", dispatch)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        driver.main(argv)
    dispatch.assert_not_called()
    assert orchestrate.Runner is FROZEN_RUNNER and snapshot(case.root) == before


@pytest.mark.parametrize("failure", [False, True])
def test_cli_forwards_arguments_and_seals_separate_immutable_provenance(
    frozen_resume, monkeypatch, failure,
):
    case = frozen_resume
    before = snapshot(case.root)
    calls = []

    def original_main(argv):
        calls.append(list(argv))
        assert orchestrate.Runner is not FROZEN_RUNNER
        value = orchestrate.Runner(UnitStore(case.root, case.study), case.args.notebook)
        assert value.terminal_phase == case.args.through_phase
        value.progress.update(phase="phase5", status="interrupted" if failure else "completed")
        if failure:
            raise RuntimeError("simulated executor failure")

    monkeypatch.setattr(orchestrate, "main", original_main)
    for invocation in range(2):
        if failure:
            with pytest.raises(RuntimeError, match="simulated executor failure"):
                driver.main(case.argv)
        else:
            driver.main(case.argv)
        assert orchestrate.Runner is FROZEN_RUNNER
        for name, data in before.items():
            assert (case.root / name).read_bytes() == data
        provenance = case.root / "execution/report_resume"
        completed = list(provenance.glob("*/COMPLETED"))
        assert len(completed) == invocation + 1
        for marker in completed:
            path = marker.parent
            manifest = json.loads((path / "manifest.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            config = json.loads((path / "config.json").read_text())
            assert manifest["status"] == "completed"
            assert summary["invocation_status"] == ("failed" if failure else "finished")
            assert summary["study_complete"] is False
            assert summary["scientific_sources_unchanged"] is True
            assert summary["error_type"] == ("RuntimeError" if failure else None)
            assert config["command"][-len(case.argv):] == case.argv
            integrity = json.loads((path / "integrity.json").read_text())
            assert all(file_hash(path / name) == digest for name, digest in integrity.items())
        if invocation == 0:
            first_path, first_bytes = completed[0].parent, snapshot(completed[0].parent)
        else:
            assert snapshot(first_path) == first_bytes
        assert not list(provenance.glob(".incomplete/*/manifest.json"))
    assert calls == [case.argv, case.argv]


@pytest.mark.parametrize("record_provenance", [False, True])
@pytest.mark.parametrize("policy_name", ["depth1_diag", "asg_lr_lr"])
def test_generated_cpu_trajectory_equivalence_and_completed_resume(
    tiny_case, tmp_path, monkeypatch, record_provenance, policy_name,
):
    policy = dict(tiny_case["policy"])
    if policy_name == "asg_lr_lr":
        policy.update(name=policy_name, depth=None, gate="lr", ewc="lr")
    renderer = Mock(side_effect=lambda root, notebook: notebook.write_text("fixture report", encoding="utf-8"))
    monkeypatch.setattr(report_view, "refresh", renderer)
    fixture_assets = Mock(return_value=tiny_case["asset_path"])
    monkeypatch.setattr(orchestrate, "ensure_assets", fixture_assets)

    def make_runner(cls, name, *, root=None):
        root = tmp_path / name if root is None else root
        root.mkdir(exist_ok=True)
        value = cls(UnitStore(root, tiny_case["study"]), tmp_path / f"{name}.ipynb")
        value.profile = profile()
        value.selection = {"width": 5, "retention": {"mixture": tiny_case["retention"]}}
        return value

    def run(value):
        value.refresh(force=True)
        value.phase(5, "running")
        path, asset = value.trajectory("phase5", 0, tiny_case["environment"],
                                       policy, tier="strict", snapshots=True)
        assert asset == tiny_case["asset_path"]
        value.refresh(force=True)
        value.refresh(force=True)
        return path

    def reporting():
        if record_provenance:
            return driver.coalesced_reporting(provenance={"reporting_only": True,
                                                         "fixture": "generated-image-equivalence"})
        return nullcontext()

    reference_runner = make_runner(FROZEN_RUNNER, "reference")
    reference = run(reference_runner)
    before_provenance = rng_state()
    with reporting() as runners:
        cls = orchestrate.Runner if record_provenance else driver.CoalescedReportRunner
        wrapped_runner = make_runner(cls, "wrapped")
        assert_same_science(rng_state(), before_provenance)
        wrapped = run(wrapped_runner)
        sealed_before_finalization = snapshot(wrapped)
        before_finalization = rng_state()
        if record_provenance:
            assert runners == [wrapped_runner]
    assert orchestrate.Runner is FROZEN_RUNNER
    assert_same_science(rng_state(), before_finalization)
    assert snapshot(wrapped) == sealed_before_finalization
    assert_same_artifacts(wrapped, reference)
    assert_same_science(json.loads((wrapped / "pr_curves.json").read_text()),
                        json.loads((reference / "pr_curves.json").read_text()))
    assert_same_science(torch.load(wrapped / "pr_curves.pt", map_location="cpu", weights_only=False),
                        torch.load(reference / "pr_curves.pt", map_location="cpu", weights_only=False))
    assert (wrapped / "config.json").read_bytes() == (reference / "config.json").read_bytes()
    summary = json.loads((wrapped / "summary.json").read_text())
    assert summary["policy"] == policy_name and summary["fit_calls"] > 0
    gates = json.loads((wrapped / "gates.json").read_text())
    if policy["gate"] is not None:
        assert gates and summary["gate_events"] == len(gates)
        for gate in gates:
            scores = gate["score_diagnostics"]
            assert scores["estimator"] == "sampled_model_labels"
            assert scores["computed_score_count"] == policy["score_samples"]
            assert scores["score_cost_scope"] == "full_network"
            assert "schur" in gate["solve_diagnostics"]
    else:
        assert gates == [] and summary["gate_events"] == 0
    final = torch.load(wrapped / "final_state.pt", map_location="cpu", weights_only=False)
    assert WorkBudget.from_state_dict(final["budget"]).spent > 0
    assert sum(float(units) for key, units in final["budget"]["units"].items()
               if key.startswith("probe:")) == len(gates)
    assert wrapped_runner.coalesced_refreshes > 0
    assert fixture_assets.call_count == 2
    before = snapshot(wrapped)
    first_records = snapshot(wrapped_runner.store.root / "execution")
    if record_provenance:
        assert len(list(wrapped_runner.store.root.glob("execution/report_resume/*/COMPLETED"))) == 1
    else:
        assert first_records == {}

    def forbidden(*args, **kwargs):
        pytest.fail("completed resume performed model construction, fitting, or Fisher work")

    monkeypatch.setattr(trajectory, "build_model", forbidden)
    monkeypatch.setattr(trajectory, "fit_restricted", forbidden)
    monkeypatch.setattr(trajectory, "model_fisher", forbidden)
    monkeypatch.setattr(trajectory, "evaluate_gate", forbidden)
    with reporting():
        cls = orchestrate.Runner if record_provenance else driver.CoalescedReportRunner
        resumed = make_runner(cls, "resumer", root=wrapped_runner.store.root)
        assert run(resumed) == wrapped
    assert resumed.store.limit.units == 0 and snapshot(wrapped) == before
    for name, data in first_records.items():
        assert (wrapped_runner.store.root / "execution" / name).read_bytes() == data
    if record_provenance:
        assert len(list(wrapped_runner.store.root.glob("execution/report_resume/*/COMPLETED"))) == 2
