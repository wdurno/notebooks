"""Scalar gates, inference, immutable reports and shared-allowance accounting."""

from copy import deepcopy
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from mnist_experiment.combined_memory.adaptive_f1 import analysis as a, calibration as c
from mnist_experiment.combined_memory.adaptive_f1.config import Policy, STRENGTH, PROCESS_SCALE
from mnist_experiment.combined_memory.adaptive_f1.runner import archive_diagonal
from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_stm_capacity import analysis as old


def row(policy, replica=0, *, namespace="fresh", ridge=1., gap=0., endpoint_gap=None):
    N = 4
    recipe = Recipe("fd_replay_ewc", 32, 2, ridge=ridge, strength=STRENGTH)
    values = [16, 12, 9, 7, 5] if policy=="R" else [16, 10, 7, 4, 2]
    metrics = [dict(n=n, observations=4*n*(replica+1), novel_mse=float(values[n]*(1+.1*replica)),
        base_mse=4. if n==0 else (2.01+gap)**2,
        learning_seconds=float(n*(2 if policy=="R" else 2.5)), drift_strength=n/N) for n in range(N+1)]
    if endpoint_gap is not None:
        metrics[-1]["base_mse"] = (2.01+endpoint_gap)**2
    partition = "development" if namespace=="development" else "confirmation"
    return dict(old.summarize_trajectory(metrics, N=N), policy=policy, replica=replica,
        protocol=old.PROTOCOL, namespace=namespace, representation="low_rank_diagonal",
        family=partition+"--grid", partition=partition, context_key="grid", output_dim=54,
        target=N, stm_config="head_r1", arm="head_r1--fd_replay_ewc", method="fd_replay_ewc",
        recipe=recipe.mapping(), recipe_key=recipe.key, capacity=32, replay_count=2, valid=True,
        metrics=metrics, plan_hash=f"plan-{replica}", pairing_id=f"plan-{replica}",
        chart_hash="same", factor_seeds=dict(init=replica, probes=replica+100),
        retired_audit=[], process_scale=PROCESS_SCALE, active_seconds=10., peak_vram_bytes=0)


def cohort(n=4, **kwargs):
    return [row(p, i, **kwargs) for i in range(n) for p in ("R", "K")]


def test_two_arm_inference_has_correct_sign_pairing_and_savings():
    rows = cohort()
    result = a.analyze(rows, draws=40, seed=31)
    d = result["contrast"]
    assert d["acquisition_difference"]["ci"][1] < 0
    assert d["favorable_replicas"] == 4 and d["retention_equivalent"]
    point = result["efficiency"]["point"]
    assert point["saved_fraction"] == pytest.approx(1-8/3/4)
    assert point["learning_time_ratio"] == pytest.approx((8/3*2.5)/8)
    assert point["S"] == pytest.approx(point["saved_fraction"]/point["learning_time_ratio"])
    samples = np.array([a.replica_draw_indices(4, 31, i) for i in range(40)])
    differences = [np.mean([m["novel_mse"] for m in rows[2*i+1]["metrics"][1:]])-
                   np.mean([m["novel_mse"] for m in rows[2*i]["metrics"][1:]]) for i in range(4)]
    assert d["acquisition_difference"]["ci"] == pytest.approx(np.quantile(np.array(differences)[samples].mean(1), [.025, .975]))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["plan", "initial", "exposure", "duplicate", "cohort", "representation", "ridge"])
def test_bad_or_unpaired_inputs_are_rejected(mutation):
    rows = cohort()
    if mutation=="plan": rows[1]["plan_hash"] = rows[1]["pairing_id"] = "other"
    if mutation=="initial": rows[1]["metrics"][0]["base_mse"] += .1
    if mutation=="exposure": rows[1]["metrics"][-1]["observations"] += 1
    if mutation=="duplicate": rows.append(deepcopy(rows[0]))
    if mutation=="cohort": rows[0]["namespace"] = "development"
    if mutation=="representation": rows[1]["representation"] = "diagonal"
    if mutation=="ridge": rows[0]["recipe"]["ridge"] = 2.
    with pytest.raises(ValueError): a.analyze(rows, draws=10)


def test_bootstrap_checkpoints_are_reused_and_tamper_rejected():
    chunks = {}
    result = a.analyze(cohort(), draws=110, load_bootstrap=chunks.get,
        save_bootstrap=lambda i, v: chunks.__setitem__(i, v))
    assert a.analyze(cohort(), draws=110, load_bootstrap=chunks.get) == result
    chunks[0]["values"][0]["indices"][0] = 999
    with pytest.raises(ValueError): a.analyze(cohort(), draws=110, load_bootstrap=chunks.get)


def test_unsupported_inverse_draws_are_not_removed():
    rows = cohort()
    for r in rows:
        r["metrics"][0]["novel_mse"] = .01
        r.update({k: v for k, v in old.summarize_trajectory(r["metrics"], N=4).items() if k!="protocol"})
    result = a.analyze(rows, draws=10)
    assert result["efficiency"]["bootstrap"]["invalid"] == 10
    assert result["efficiency"]["bootstrap"]["ci"] is None


def test_development_gate_needs_both_metrics_and_all_four_members():
    rows = cohort(namespace="development")
    assert c.selection(rows)["passed"]
    assert c.selection(rows[:-1]) is None
    poor = [row(p, r, namespace="development", endpoint_gap=.03 if p=="K" else 0.)
            for r in range(4) for p in ("R", "K")]
    assert c.selection(poor) is None
    assert c.selection(poor, require_match=False)["maximum_violation"] > 0
    with pytest.raises(ValueError): c.selection(cohort())


def test_proposal_direction_uses_retention_not_acquisition_winner():
    protected = [row(p, r, namespace="development", gap=-.03 if p=="K" else 0.)
                 for r in range(4) for p in ("R", "K")]
    assert c.propose(protected)["ridge"] == .1
    forgetful = [row(p, r, namespace="development", gap=.03 if p=="K" else 0.)
                 for r in range(4) for p in ("R", "K")]
    assert c.propose(forgetful)["ridge"] == 10.


def test_archive_diagonal_is_exact_without_dense_materialization():
    factor = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    residual = torch.tensor([.3, .7], dtype=torch.float64)
    assert torch.equal(archive_diagonal(dict(factor=factor, residual_diagonal=residual)),
                       (factor@factor.T+torch.diag(residual)).diag())
    with pytest.raises(FloatingPointError):
        archive_diagonal(dict(factor=factor*float("nan"), residual_diagonal=residual))


def test_policy_contract():
    assert Policy("R").representation == Policy("K").representation == "low_rank_diagonal"
    assert Policy("K").adaptive and not Policy("R").adaptive
    with pytest.raises(ValueError): Policy("R", 2.)
    with pytest.raises(ValueError): Policy("K", process_scale=1e-4)


def test_report_preserves_historical_cells_and_generations(tmp_path):
    from mnist_experiment.combined_memory.adaptive_f1 import report
    path = tmp_path/"results.ipynb"
    original = nbformat.v4.new_markdown_cell("Historical result", id="old-result")
    nbformat.write(nbformat.v4.new_notebook(cells=[original]), path)
    root = tmp_path/"root"
    store = type("ReportRoot", (), {"root": root})()
    progress = dict(status="running", stage="calibration", development_completed=0,
        fresh_blocks_completed=0, charged_hours=4.4, forecast_hours=3.)
    snapshot = dict(protocol=report.PROTOCOL, progress=progress, development={})
    first = report.publish(snapshot, path, store)
    saved = (root/"report/generations"/first/"results.ipynb").read_bytes()
    snapshot["progress"]["status"] = "calibration_infeasible"
    second = report.publish(snapshot, path, store)
    assert first != second and (root/"report/generations"/first/"results.ipynb").read_bytes() == saved
    book = nbformat.read(path, as_version=4)
    assert [c for c in book.cells if not report.owned(c)] == [original]
    assert sum(c.id=="plan18-phase9e-progress" for c in book.cells) == 1
    assert report.publish(snapshot, path, store) == second


def test_shared_allowance_includes_original_and_new_roots(tmp_path, monkeypatch):
    from mnist_experiment.combined_memory.adaptive_f1 import runtime
    monkeypatch.setattr(runtime, "FAMILY", tmp_path)
    monkeypatch.setattr(runtime.previous, "peer_seconds", lambda root: 15000.)
    for name, seconds in (("production", 40.), ("smoke", 20.)):
        path = tmp_path/name
        path.mkdir()
        (path/"identity.json").write_text(json.dumps(dict(identity=dict(protocol=runtime.PROTOCOL))))
        monkeypatch.setattr(runtime.family_runtime, "_charged_seconds", lambda p: {"production":40., "smoke":20.}[p.name])
    assert runtime.peer_seconds(tmp_path/"production") == 15020.


@pytest.mark.parametrize("unsupported", [False, True])
def test_report_always_shows_compute_even_without_sample_equivalence(tmp_path, unsupported):
    from mnist_experiment.combined_memory.adaptive_f1 import report
    rows = cohort(8)
    if unsupported:
        for r in rows:
            r["metrics"][0]["novel_mse"] = .01
            r.update({k:v for k,v in old.summarize_trajectory(r["metrics"], N=4).items() if k!="protocol"})
    result = a.analyze(rows, draws=20)
    snapshot = dict(protocol=report.PROTOCOL, analysis=result,
        progress=dict(status="completed", stage="closeout", development_completed=4,
            fresh_blocks_completed=8, charged_hours=8., forecast_hours=0.))
    _, cells = report.cells(snapshot, tmp_path, {})
    text = "\n".join(c.source for c in cells)
    assert "Equal-budget online compute ratio" in text
    assert ("Sample equivalence is unavailable" in text) == unsupported
    if not unsupported:
        assert "Nominal 95% CI" in text


def test_final_repair_uses_first_pair_bracket_instead_of_repeating_completed_ridge():
    from mnist_experiment.combined_memory.adaptive_f1_scheduler.run import repair_proposal
    rows = [row(p, r, namespace="development", gap=-.03 if p=="K" else 0.)
            for r in range(4) for p in ("R", "K")]
    rows += [row("K", r, namespace="development", ridge=.1, gap=.04) for r in (0, 1)]
    proposal = repair_proposal(rows)
    assert .1 < proposal["ridge"] < 1.
    assert proposal["validation_members"] == list(range(4))
    assert not c.selection(rows)


def test_final_repair_does_not_revalidate_an_already_complete_first_pair_winner():
    from mnist_experiment.combined_memory.adaptive_f1_scheduler.run import repair_proposal
    rows = [row(p, r, namespace="development", gap=.04 if p=="K" and r>=2 else 0.)
            for r in range(4) for p in ("R", "K")]
    rows += [row("K", r, namespace="development", ridge=.1, gap=-.04) for r in (0, 1)]
    assert c.selection(rows, members=(0, 1))["passed"]
    assert c.selection(rows) is None
    assert repair_proposal(rows)["ridge"] != 1.


def test_versioned_store_adopts_partial_state_and_protects_parent_units(tmp_path):
    from mnist_experiment.combined_memory.adaptive_f1_scheduler.run import (
        OriginalStore, InheritedStore, patch_sources,
    )
    parent, child = tmp_path/"parent", tmp_path/"child"
    identity = dict(source_hash="frozen-numerics", role="smoke")
    with OriginalStore(parent, identity) as store:
        store.begin("sealed", dict(kind="fixture"))
        store.write_json("sealed", "result.json", dict(value=3))
        store.complete("sealed")
        store.begin("pending", dict(kind="grid_registration_trajectory"))
        store.save_checkpoint("pending", dict(next_segment=3, n=1, q_ref=.01))
    sealed = (parent/"units/sealed/COMPLETED").read_bytes()
    revision = dict(sources=patch_sources())
    with InheritedStore(OriginalStore(child, identity), parent, revision) as store:
        assert store.is_complete("sealed")
        assert store.load_checkpoint("pending")==dict(next_segment=3, n=1, q_ref=.01)
        assert store.unit_path("sealed")==parent/"units/sealed"
        with pytest.raises(Exception, match="cannot mutate"):
            store.write_json("sealed", "result.json", dict(value=4))
        store.write_json("pending", "result.json", dict(value=5))
        store.complete("pending")
        assert {r["unit_id"] for r in store.iter_units(completed_only=True)}=={"sealed", "pending"}
    assert (parent/"units/sealed/COMPLETED").read_bytes()==sealed
