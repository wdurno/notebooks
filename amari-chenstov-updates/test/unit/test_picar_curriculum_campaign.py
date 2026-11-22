"""Bounded control-plane checks: no campaign.run, model, rollout or GPU work."""

import ast
import copy
from dataclasses import asdict, replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import RunCollisionError, RunStore
from numerical_experiment.picar_curriculum_repair import calibration, campaign, config


def _campaign_tree():
    return ast.parse(inspect.getsource(campaign.run))


def _closure(name, **bindings):
    """Exercise a private closure in isolation, never the campaign entry point.

    These functions have no public test boundary. Turning nonlocal bindings
    into globals preserves their ledger mutations with injected clocks/stores.
    """
    nodes = [node for node in ast.walk(_campaign_tree())
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1, f"expected one isolated campaign closure: {name}"
    node = copy.deepcopy(nodes[0])
    node.body = [ast.copy_location(ast.Global(names=item.names), item)
                 if isinstance(item, ast.Nonlocal) else item for item in node.body]
    namespace = {**vars(campaign), **bindings}
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(campaign.__file__), "exec"), namespace)
    return namespace[name]


def _complete(path, summary, state=None):
    with RunStore(path, {"schema_version": 1, "version": "unit-test-only"},
                  checkpoint_policy="journal-v1") as store:
        store.checkpoint(state or {"stage": "done"})
        store.complete(summary)
    return path


def _source(tmp_path, *, outside=False, bad_seal=False, kind="reward_campaign"):
    source = tmp_path / "fix3"
    asset_path = (tmp_path / "outside" if outside else source / "units" / "linear")
    _complete(asset_path, {"kind": "reward_bootstrap", "condition": "width6_linear"})
    asset = {"path": str(asset_path), "id": "0" * 64 if bad_seal else sha256_file(asset_path / "COMPLETED")}
    _complete(source / "orchestration", {"kind": kind, "assets": {"width6_linear": asset}})
    return source, asset


def test_source_asset_reuses_exact_sealed_linear_initializer_read_only(tmp_path):
    source, asset = _source(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert campaign.source_asset(source) == asset
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change,match", [
    ({"outside": True}, "inside"),
    ({"bad_seal": True}, "seal differs"),
    ({"kind": "solver_campaign"}, "completed Fix 3"),
])
def test_source_asset_rejects_wrong_campaign_external_asset_or_stale_seal(tmp_path, change, match):
    source, _ = _source(tmp_path, **change)
    with pytest.raises(ValueError, match=match):
        campaign.source_asset(source)


def test_source_asset_rejects_corrupt_completed_summary(tmp_path):
    source, asset = _source(tmp_path)
    # Corruption injection in a throwaway envelope, not a historical artifact.
    (Path(asset["path"]) / "summary.json").write_text('{"kind":"replacement"}\n')
    with pytest.raises(ValueError):
        campaign.source_asset(source)


def _initial():
    return {"stage": "done", "eligible": [], "retired": list(range(515)),
            "identity": {"config": asdict(config.configuration()),
                         "policy": {"fisher_draws": 4}},
            "learner": {"retired_count": 515}, "fisher": {"count": 515},
            "distill_receipt": {"stage": "recorded_one_time_KL"},
            "train_plateau": {"reached": False, "stale_epochs": 0}}


def _integrity(initial):
    return _closure("integrity", initial=initial, asset={"id": "sealed-linear-asset"})(None)


def test_initializer_reuse_audit_starts_with_empty_replay_and_no_new_observations():
    initial = _initial()
    before = copy.deepcopy(initial)
    result = _integrity(initial)
    assert initial == before
    assert result["asset"]["id"] == "sealed-linear-asset"
    assert result["train_histories"] == 515
    assert result["validation_histories"] == 65
    assert result["new_observations"] == 0
    assert result["equal_weight_four_draw_initial_fisher"]
    assert result["initial_pi_used"] is False
    assert result["training_plateau"]["reached"] is False


@pytest.mark.parametrize("change", ["eligible", "duplicate_retirement", "receipt", "signed_square"])
def test_initializer_audit_rejects_recounted_evidence_or_different_likelihood(change):
    initial = _initial()
    if change == "eligible":
        initial["eligible"] = [{"arrival": 0}]
    elif change == "duplicate_retirement":
        initial["retired"][-1] = initial["retired"][0]
    elif change == "receipt":
        initial["distill_receipt"] = None
    else:
        initial["identity"]["config"]["learning"]["reward_version"] = "six-example-width-2-v1"
    with pytest.raises(ValueError):
        _integrity(initial)


@pytest.mark.parametrize("change", ["single_draw", "incomplete_fisher", "return_time_units"])
def test_initializer_audit_verifies_the_fisher_and_return_units_it_certifies(change):
    initial = _initial()
    if change == "single_draw":
        initial["identity"]["policy"]["fisher_draws"] = 1
    elif change == "incomplete_fisher":
        initial["fisher"]["count"] = 514
    else:
        initial["identity"]["config"]["learning"]["nominal_period_s"] *= 2
    with pytest.raises(ValueError):
        _integrity(initial)


def _summary(name, *, a_nll=10., b_nll=5., successes=8, strength=.1):
    return {"unit_path": name, "fd_strength": strength, "ridge": .1,
            "settings": {"seed": config.CALIBRATION_SEED, "reference_id": "shared-bank",
                         "initial_asset_id": "shared-initializer"},
            "reference_scores": [
                {"target": 0, "rows": [{"environment": "A", "joint_nll": 100.}]},
                {"target": 128, "rows": [{"environment": "A", "joint_nll": a_nll},
                                          {"environment": "B_initial", "joint_nll": b_nll}]}],
            "evaluations": [
                {"target": 128, "difficulty": "A", "rows": [{"success": i < successes} for i in range(8)]},
                {"target": 128, "difficulty": "B_initial", "rows": [{"success": False} for _ in range(8)]}]}


def test_calibration_matches_control_retention_before_ranking_b_fit():
    control = _summary("control", strength=0.)
    candidates = [_summary("lost-A", a_nll=9., b_nll=1., successes=0),
                  _summary("matched", a_nll=10.1, b_nll=4., successes=7),
                  _summary("bad-A-nll", a_nll=11., b_nll=2., successes=8)]
    before = copy.deepcopy((control, candidates))
    result = calibration.select(control, candidates)
    assert result["selected"]["path"] == "matched"
    assert result["control_A"]["nll"] == 10.
    assert result["selected"]["A_nll_gap"] == pytest.approx(.1)
    assert result["warning"] is None
    assert result["development_only"] and result["retention_equivalence"] is False
    assert (control, candidates) == before


def test_unmatched_calibration_reports_tradeoff_not_equivalence():
    result = calibration.select(_summary("control", strength=0.), [
        _summary("small-A-gap", a_nll=10.3, b_nll=20.),
        _summary("fast-B", a_nll=12., b_nll=1.)])
    assert result["selected"]["path"] == "small-A-gap"
    assert result["warning"] and not result["retention_equivalence"]
    with pytest.raises(ValueError, match="candidate"):
        calibration.select(_summary("control", strength=0.), [])


def test_calibration_does_not_accept_a_missing_b_initial_task_panel():
    candidate = _summary("missing-B-panel")
    candidate["evaluations"] = [candidate["evaluations"][0]]
    with pytest.raises(ValueError, match="panels"):
        calibration.select(_summary("control", strength=0.), [candidate])


def _pair_scope(tmp_path, *, elapsed=0., max_hours=28., completed_control=False):
    calls = []

    def unit_path(operation, settings):
        return tmp_path / f"{operation}-{settings['seed']}-{settings['method']}"

    def trajectory(store, **kwargs):
        return kwargs

    def unit(operation, settings, callback):
        calls.append((copy.deepcopy(settings), callback(None)))
        return {"unit_path": str(unit_path(operation, settings))}

    if completed_control:
        path = unit_path("trajectory", {"seed": config.MAIN_SEEDS[0], "method": "replay_ewc"})
        path.mkdir()
        (path / "COMPLETED").touch()
    return {**vars(campaign), "calls": calls, "completed": [], "unit_path": unit_path,
            "unit": unit, "trajectory": trajectory, "chosen": {"strength": .3, "ridge": .1},
            "asset": {"id": "common-initializer"}, "bank_id": "common-reference-bank",
            "estimate": 20000., "control": {"cost": 1000.}, "fd_candidate": {"cost": 9000.},
            "projected": lambda summary: summary["cost"],
            "elapsed": lambda: elapsed, "max_hours": max_hours}


def _admit_pairs(namespace):
    # Execute only the admission/dispatch loop with pure callbacks. In particular,
    # do not call campaign.run or its model-loading and reference-collection code.
    loops = [node for node in ast.walk(_campaign_tree()) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == "enumerate" and node.iter.args
             and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == "MAIN_SEEDS"]
    assert len(loops) == 1
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loops[0])], type_ignores=[]))
    exec(compile(tree, str(campaign.__file__), "exec"), namespace)


def test_main_pairs_share_initializer_reference_and_named_seeds_and_reverse_order(tmp_path):
    namespace = _pair_scope(tmp_path)
    _admit_pairs(namespace)
    calls = namespace["calls"]
    assert [(s["seed"], s["method"]) for s, _ in calls] == [
        (198101, "replay_ewc"), (198101, "stm_fd"),
        (198102, "stm_fd"), (198102, "replay_ewc")]
    maps = {}
    for settings, dispatched in calls:
        assert settings["initial_asset_id"] == "common-initializer"
        assert settings["reference_id"] == "common-reference-bank"
        assert settings["budget"] == dispatched["budget"] == 2048
        assert dispatched["main"] is True
        assert dispatched["seed"] == settings["seed"]
        cfg = config.configuration(strength=dispatched["strength"], ridge=dispatched["ridge"])
        seeds = replace(cfg, seed=settings["seed"]).seeds(0)
        assert {"layouts", "exploration", "replay", "fisher_labels", "fd_probes"} <= seeds.keys()
        assert len(set(seeds.values())) == len(seeds)
        assert maps.setdefault(settings["seed"], seeds) == seeds
    assert maps[198101] != maps[198102]
    separated = {config.PILOT_SEED, config.CALIBRATION_SEED, config.REFERENCE_SEED,
                 *config.MAIN_SEEDS, *config.EVALUATION_SEEDS}
    assert len(separated) == 3 + len(config.MAIN_SEEDS) + len(config.EVALUATION_SEEDS)


def test_calibration_control_and_all_fd_candidates_dispatch_the_same_development_seed(tmp_path):
    namespace = _pair_scope(tmp_path)
    namespace.update(candidates=[], probe={"strengths": [.03, .3, 3.]})
    nodes = list(ast.walk(_campaign_tree()))
    candidates = [node for node in nodes if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name) and node.target.id == "strength"
                  and isinstance(node.iter, ast.Subscript)
                  and isinstance(node.iter.value, ast.Name) and node.iter.value.id == "probe"]
    controls = [node for node in nodes if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "control" for target in node.targets)
                and isinstance(node.value, ast.Call)]
    assert len(candidates) == len(controls) == 1
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(candidates + controls), type_ignores=[]))
    exec(compile(tree, str(campaign.__file__), "exec"), namespace)
    calls = namespace["calls"]
    assert len(calls) == 4
    assert {settings["seed"] for settings, _ in calls} == {config.CALIBRATION_SEED}
    assert config.CALIBRATION_SEED != config.PILOT_SEED
    assert [settings["fd_strength"] for settings, _ in calls] == [.03, .3, 3., 0.]
    for settings, dispatched in calls:
        assert dispatched["seed"] == settings["seed"]
        assert dispatched["strength"] == settings["fd_strength"]
        assert settings["initial_asset_id"] == "common-initializer"
        assert settings["reference_id"] == "common-reference-bank"


def test_resume_admission_costs_the_missing_fd_method_not_the_pair_average(tmp_path):
    namespace = _pair_scope(tmp_path, elapsed=10000., max_hours=5., completed_control=True)
    namespace["MAIN_SEEDS"] = (config.MAIN_SEEDS[0],)
    # Only 8000 s remain. Control is sealed; FD needs 9000 s before reserve.
    # The pair average (5000 s) must not admit this incomplete paired scope.
    with pytest.raises(TimeoutError, match="pair|ceiling|budget"):
        _admit_pairs(namespace)
    assert not namespace["calls"]


def test_insufficient_budget_admits_no_part_of_a_new_pair(tmp_path):
    namespace = _pair_scope(tmp_path, elapsed=15000., max_hours=5.)
    with pytest.raises(TimeoutError):
        _admit_pairs(namespace)
    assert not namespace["calls"]


def test_forecast_uses_actual_simulator_cadence_without_reassigning_training_time_to_evaluation():
    n, evaluations, intervals_per_episode = 128, 12, 5
    period = config.simulation().control_interval_s
    fit_seconds = 64.
    evaluation_seconds_per_episode = 30.
    summary = {
        "acquired_count": n,
        "timings": {"fit_seconds": fit_seconds, "fisher_seconds": 0.,
                    "scoring_seconds": float(n + evaluations * intervals_per_episode),
                    "evaluation_seconds": evaluations * evaluation_seconds_per_episode},
        "evaluations": [{"rows": [{"intervals": [{}] * intervals_per_episode}
                                   for _ in range(evaluations)]}],
        # Synthetic nonoverlapping timings: one second per training score;
        # held-out physics/scoring is already inside each 30-second evaluation.
        "active_seconds": n * period + fit_seconds + n
                          + evaluations * evaluation_seconds_per_episode,
    }
    expected = fit_seconds / n * 2048 + 2048 + 2048 * period + 156 * evaluation_seconds_per_episode
    assert _closure("projected")(summary) == pytest.approx(expected)


def test_progress_charges_failed_attempts_carries_resume_base_and_preserves_closeout(tmp_path):
    snapshots = []
    clock = SimpleNamespace(now=1000.)
    stop = SimpleNamespace(requested=False)
    stop.check = lambda: stop.requested
    progress = _closure("progress", root=tmp_path, active=None, status="test",
        last_save=0., last_report=0., last_storage=0., forecast={}, max_hours=28.,
        elapsed=lambda: 1234. + clock.now - 1000.,
        time=SimpleNamespace(monotonic=lambda: clock.now), stop=stop,
        parent=SimpleNamespace(checkpoint=lambda state, **kw: snapshots.append((copy.deepcopy(state), kw))),
        check_storage=lambda *args, **kwargs: None, refresh=lambda: None)
    progress(force=True)
    state, display = snapshots[-1]
    assert state["charged_seconds"] == 1294.
    assert display["progress"]["charged_seconds"] == 1234.
    clock.now += 17.
    progress("failed attempt committed", force=True, final=True)
    assert snapshots[-1][0]["charged_seconds"] == 1251.
    stop.requested = True
    with pytest.raises(InterruptedError):
        progress(force=True)
    stop.requested = False
    clock.now = 1000. + 28 * 3600 - config.CLOSEOUT_SECONDS - 1234.
    with pytest.raises(TimeoutError):
        progress(force=True)


@pytest.mark.parametrize("dependency", [
    "src/seeding.py",
    "numerical_experiment/picar_ball_width_probe.py",
    "numerical_experiment/picar_reference_reward_probe.py",
])
def test_source_freeze_includes_live_seed_reward_prompt_parser_and_image_processing(dependency):
    sources = config.scientific_sources()
    assert dependency in sources, f"live scientific dependency omitted from source freeze: {dependency}"
    assert sources[dependency] == sha256_file(config.PROJECT / dependency)


@pytest.mark.parametrize("changed", ["source", "theory"])
def test_resume_refuses_changed_science_but_uses_sealed_history_not_live_notebook(tmp_path, monkeypatch, changed):
    project = tmp_path / "project"
    project.mkdir()
    theory = project / "mathematical_overview.ipynb"
    theory.write_text("theory snapshot")
    notebook = project / "live.ipynb"
    notebook.write_text("historical report")
    source, _ = _source(tmp_path)
    references = _complete(tmp_path / "references", {"kind": "unit-reference"})
    root = tmp_path / "new-campaign"
    root.mkdir()
    monkeypatch.setattr(campaign, "PROJECT", project)
    monkeypatch.setattr(campaign, "NOTEBOOK", notebook)
    sources = {"unit_science.py": "a" * 64}
    monkeypatch.setattr(campaign, "scientific_sources", lambda: dict(sources))
    identity = campaign.run_identity(root, source, references, None, 28.)
    (root / "history.ipynb").write_bytes(notebook.read_bytes())
    notebook.write_text("new live progress is not a scientific identity change")
    assert campaign.run_identity(root, source, references, None, 28.) == identity
    path = root / "orchestration"
    with RunStore(path, identity, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"charged_seconds": 1234.})
    if changed == "source":
        sources["unit_science.py"] = "b" * 64
    else:
        theory.write_text("changed theory")
    revised = campaign.run_identity(root, source, references, None, 28.)
    assert revised != identity
    before = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    with pytest.raises(RunCollisionError):
        RunStore(path, revised, resume=True, checkpoint_policy="journal-v1")
    assert before == {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
