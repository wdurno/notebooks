"""Selection/ledger tests; dispatch tests seal JSON-only computational stubs."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from mnist_experiment.akaike_score_gate import development as dev
from mnist_experiment.akaike_score_gate.artifacts import UnitStore, file_hash
from mnist_experiment.akaike_score_gate.config import Study, canonical_hash


def study(smoke=True, **changes):
    name = "smoke" if smoke else "default"
    original = Study.from_path(f"mnist_experiment/akaike_score_gate/configs/{name}.json")
    return Study({**original.mapping(), **changes})


def source_rows(replicas=4, five_nll=.51, five_accuracy=.9):
    return [{"index": index, "family": family, "width": width,
             "source_nll": {5: five_nll, 6: .5, 0: .45}[width],
             "source_accuracy": {5: five_accuracy, 6: .905, 0: .93}[width]}
            for index in range(replicas) for family in dev.FAMILIES for width in dev.SOURCE_WIDTHS]


def retention_rows(family="rotation", replicas=2):
    return [{"index": index, "environment": f"{family}_{env}", "family": family,
             "depth": depth, "candidate": candidate["name"], "nll_auc": 1.,
             "initial_retention_nll": .5 + .1 * index,
             "retention_nll": .51 + .1 * index, "fit_calls": 10, "unresolved_fits": 0}
            for index in range(replicas) for env in range(2) for depth in dev.PILOT_DEPTHS
            for candidate in dev.retention_candidates(family)]


def test_default_ledger_counts_pairing_depth_balance_and_named_case_seeds():
    config = study(False)
    ledger = dev.build_ledger(config)
    assert len(ledger["sources"]) == 24
    assert len(ledger["pilots"]) == 120
    assert len(ledger["drivers"]) == 112
    assert len(ledger["cases"]) == 48
    assert sum(case["historical"] for case in ledger["cases"]) == 8
    assert Counter(case["source_depth"] for case in ledger["cases"]) == {1: 16, 4: 16, 6: 16}
    counts = Counter(case["frontier"] for case in ledger["cases"])
    assert max(counts.values()) - min(counts.values()) == 1
    assert set(counts) == set(range(1, 6))
    history = [case for case in ledger["cases"] if case["historical"]]
    assert [case["ordinal"] for case in history] == [0, 7, 13, 20, 27, 34, 40, 47]
    assert set(Counter(case["environment"]["name"] for case in history).values()) == {2}
    assert len({case["audit_initialization_seed"] for case in ledger["cases"]}) == 48
    assert all(case["audit_initialization_seed"] == config.seed("audit_initialization", case["ordinal"])
               for case in ledger["cases"])
    assert canonical_hash(ledger) == canonical_hash(dev.build_ledger(config))
    keys = {(row["index"], row["environment"]["name"], row["depth"]) for row in ledger["drivers"]}
    assert len(keys) == 112
    assert all((case["replica_index"], case["environment"]["name"], case["source_depth"]) in keys
               for case in ledger["cases"])
    for case in ledger["cases"]:
        env = case["environment"]
        steps = 120 if env["family"] == "rotation" else 10
        assert 0 <= case["step"] < steps - 1


def test_smoke_counts_are_explicit_not_outcome_dependent():
    ledger = dev.build_ledger(study())
    assert len(ledger["sources"]) == 6
    assert len(ledger["pilots"]) == 60
    assert len(ledger["drivers"]) == 28
    assert len(ledger["cases"]) == 4
    assert sum(case["historical"] for case in ledger["cases"]) == 1
    assert all(row["environment"]["name"] != "mixture_m1" for row in ledger["drivers"])
    assert dev.audit_positions(120) == (0, 59, 118)
    assert dev.audit_positions(10) == (0, 4, 8)
    assert dev.audit_positions(2) == (0,)
    assert dev.audit_positions(1) == (0,)
    with pytest.raises(ValueError):
        dev.audit_positions(0)


def test_retention_grid_is_exactly_five_one_axis_conditions():
    for family in dev.FAMILIES:
        candidates = dev.retention_candidates(family)
        original = candidates[0]
        assert len(candidates) == 5
        assert len({(row["omega"], row["ridge_ratio"]) for row in candidates}) == 5
        assert {row["omega"] for row in candidates} == {
            original["omega"] / 2, original["omega"], original["omega"] * 2}
        for row in candidates[1:]:
            assert (row["omega"] != original["omega"]) != (row["ridge_ratio"] != original["ridge_ratio"])
    assert dev.retention_candidates("rotation")[0] == {"name": "default", "omega": 39., "ridge_ratio": .1}
    assert dev.retention_candidates("mixture")[0] == {"name": "default", "omega": 19., "ridge_ratio": 0.}


def test_architecture_thresholds_are_source_only_and_old_width_is_diagnostic():
    assert dev.select_architecture(source_rows(), 4)["width"] == 5
    assert dev.select_architecture(source_rows(five_nll=.52, five_accuracy=.895), 4)["width"] == 5
    assert dev.select_architecture(source_rows(five_nll=.521), 4)["width"] == 6
    assert dev.select_architecture(source_rows(five_accuracy=.894), 4)["width"] == 6
    rows = source_rows()
    for row in rows:
        row["current_nll"] = -100 if row["width"] == 6 else 100
        row["final_test_nll"] = -100 if row["width"] == 6 else 100
    result = dev.select_architecture(rows, 4)
    assert result["width"] == 5
    assert result["capacity_limitation"]
    assert result["means"]["5"]["source_nll"] == pytest.approx(.51)


def test_source_selection_requires_complete_finite_paired_rows():
    rows = source_rows()
    with pytest.raises(ValueError, match="every paired"):
        dev.select_architecture(rows[:-1], 4)
    with pytest.raises(ValueError, match="every paired"):
        dev.select_architecture(rows + [rows[0]], 4)
    rows[0]["source_nll"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        dev.select_architecture(rows, 4)


def test_retention_margin_and_default_tie_rule_keep_failed_finite_fits():
    rows = retention_rows()
    for row in rows:
        if row["candidate"] == "omega_half":
            row["nll_auc"] = .996
            row["unresolved_fits"] = 10
    result = dev.select_retention(rows, "rotation")
    assert result["selected"]["name"] == "default"
    assert not result["constraint_failure"]
    half = next(row for row in result["candidates"] if row["name"] == "omega_half")
    assert half["unresolved_fits"] == 120
    for row in rows:
        if row["candidate"] == "omega_half":
            row["nll_auc"] = .99
    assert dev.select_retention(rows, "rotation")["selected"]["name"] == "omega_half"
    for row in rows:
        if row["candidate"] == "omega_half":
            row["retention_nll"] = row["initial_retention_nll"] + .021
    assert dev.select_retention(rows, "rotation")["selected"]["name"] == "default"


def test_retention_fallback_is_current_plus_retention_without_dropping_rows():
    rows = retention_rows("mixture")
    for row in rows:
        row["retention_nll"] = row["initial_retention_nll"] + .2
        if row["candidate"] == "omega_half":
            row["nll_auc"] = .99
            row["retention_nll"] += .1
        if row["candidate"] == "omega_double":
            row["nll_auc"] = 1.05
            row["retention_nll"] -= .1
    result = dev.select_retention(rows, "mixture")
    assert result["constraint_failure"]
    assert result["selected"]["name"] == "omega_double"
    assert all(row["trajectories"] == 12 for row in result["candidates"])
    with pytest.raises(ValueError, match="complete paired"):
        dev.select_retention(rows[:-1], "mixture")
    with pytest.raises(ValueError, match="complete paired"):
        dev.select_retention([row for row in rows if row["depth"] != 6], "mixture")


def test_optimizer_health_threshold_ceiling_and_residual_contract():
    config = study(False)
    assert dev.optimizer_ceilings(config) == (50, 100, 200)
    assert dev.optimizer_ceilings(study()) == (3,)
    assert not dev.optimizer_health([{"fit_calls": 100, "unresolved_fits": 5}])["raise_ceiling"]
    assert dev.optimizer_health([{"fit_calls": 100, "unresolved_fits": 6}])["raise_ceiling"]
    assert dev.optimizer_health([{"fit_calls": 19, "unresolved_fits": 0},
                                  {"fit_calls": 1, "unresolved_fits": 1}])["unresolved_fraction"] == .05
    for limit, evaluations in ((50, 75), (100, 150), (200, 300)):
        assert dev.solver_settings(config, limit) == {"max_iter": limit, "max_eval": evaluations,
                                                     "residual_tolerance": 1e-5}
    with pytest.raises(ValueError):
        dev.optimizer_health([{"fit_calls": 2, "unresolved_fits": 3}])


@pytest.fixture
def stub_dispatch(tmp_path, monkeypatch):
    """No dataset, model, optimizer, Fisher, or actual trajectory is executed."""
    config = study(source_epoch_ceiling=2)
    store = UnitStore(tmp_path, config)
    store.sources = {}
    calls = Counter()
    source_requests = []

    def seal(unit, required, summary, resume):
        session = store.begin(unit, required, resume=resume)
        if session is None:
            return store.paths(unit)[0]
        calls[unit["kind"]] += 1
        for name in required:
            session.write_json(name, summary if name == "summary.json" else {"test_fixture": True})
        return store.finish(session, required)

    def source(store, phase, index, family, width, *, source_only=False, extended_epochs=False, resume=True):
        assert phase == "phase1"
        source_requests.append((index, family, width, source_only, extended_epochs))
        unit = store.unit(phase, "source" if source_only else "assets", index, environment=family,
                          detail={"width": width, "extended_epochs": extended_epochs})
        summary = {"width": width, "source_nll": .5 + (width == 5) * .01,
                   "source_accuracy": .9, "convergence": {"settled": False},
                   "selected_epoch": 1, "completed_epochs": 2 if extended_epochs else 1,
                   "evaluation_split": "development_train"}
        return seal(unit, dev.SOURCE_REQUIRED if source_only else dev.ASSET_REQUIRED, summary, resume)

    def trajectory(store, phase, index, env, policy, asset_path, *, width, retention, profile,
                   budget_tier, snapshot_steps, resume):
        assert phase == "phase1" and profile is None and budget_tier == "unlimited"
        assert retention["residual_tolerance"] == config.tolerance_grad
        unit = store.unit(phase, "trajectory", index, environment=env["name"], policy=policy["name"],
            detail={"policy": policy, "width": width, "retention": retention,
                    "asset_hash": file_hash(asset_path / "config.json"), "snapshot_steps": list(snapshot_steps)})
        default = dev.retention_candidates(env["family"])[0]
        nll = .9 if retention["omega"] == default["omega"] / 2 else 1.
        summary = {"nll_auc": nll, "retention_nll": .51, "initial_retention_nll": .5,
                   "fit_calls": int(policy["depth"] > 0), "unresolved_fits": 0, "online_seconds": .001}
        return seal(unit, dev.TRAJECTORY_REQUIRED, summary, resume)

    def audit(store, index, case, asset_path, driver_path, *, resume=True, phase="phase1"):
        driver = json.loads((driver_path / "config.json").read_text())
        assert driver["index"] == case["replica_index"]
        assert driver["detail"]["policy"]["depth"] == case["source_depth"]
        assert case["step"] in driver["detail"]["snapshot_steps"]
        unit = store.unit(phase, "depth_audit" if phase == "phase1" else "fisher_audit", index,
                          environment=case["environment"]["name"], detail={"case": case})
        return seal(unit, ("summary.json", "audit_metrics.json", "diagnostics.pt"), {"case": case}, resume)

    monkeypatch.setattr(dev, "ensure_assets", source)
    monkeypatch.setattr(dev, "ensure_trajectory", trajectory)
    monkeypatch.setattr(dev, "ensure_depth_audit", audit)
    monkeypatch.setattr(dev, "ensure_audit", lambda *a, **kw: audit(*a, **kw, phase="phase2"))
    return store, calls, source_requests


def test_json_only_dispatch_freezes_common_epoch_protocol_and_reuses_on_resume(stub_dispatch):
    store, calls, source_requests = stub_dispatch
    refreshes = []
    result = dev.run_phase1(store, resume=False, refresh=lambda: refreshes.append(True))
    assert result["width"] == 5 and result["extended_epochs"] is True
    assert calls["source"] == 12 and calls["assets"] == 2
    assert calls["trajectory"] == 92
    assert calls["depth_audit"] == 4
    assert len(result["drivers"]) == len(result["default_drivers"]) == 28
    assert all(isinstance(driver["path"], str) and isinstance(driver["asset_path"], str)
               for driver in result["drivers"])
    assert result["retention"]["rotation"]["omega"] == 19.5
    assert result["retention"]["mixture"]["omega"] == 9.5
    assert {width for _, _, width, source_only, extended in source_requests if source_only and extended} == {0, 5, 6}
    assert len(refreshes) > 10
    before = calls.copy(), list(source_requests)
    assert dev.run_phase1(store) == result
    assert (calls, source_requests) == before
    with pytest.raises(RuntimeError, match="collision|resume"):
        dev.run_phase1(store, resume=False)
    paths = dev.run_phase2(store, result)
    assert len(paths) == calls["fisher_audit"] == 4
    assert all(isinstance(path, Path) and (path / "COMPLETED").is_file() for path in paths)
    assert dev.run_phase2(store, result) == paths
    assert calls["fisher_audit"] == 4
    modified = copy.deepcopy(result)
    modified["width"] = 6
    with pytest.raises(RuntimeError, match="exact frozen"):
        dev.run_phase2(store, modified)


def test_frozen_selection_detects_parent_corruption_and_changed_ledgers(stub_dispatch):
    store, _, _ = stub_dispatch
    result = dev.run_phase1(store)
    parent = next(reference for reference in result["parents"] if "/source__" in reference["path"])
    path = Path(parent["path"]) / "summary.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt|changed"):
        dev.run_phase1(store)


def test_root_selection_cannot_silently_change_a_sealed_decision(stub_dispatch):
    store, _, _ = stub_dispatch
    result = dev.run_phase1(store)
    result["extended_epochs"] = False
    (store.root / "phase1_selection.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="differs from its sealed"):
        dev.run_phase1(store)
