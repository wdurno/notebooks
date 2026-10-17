"""Tiny Phase 3 diagnostic tests, without experiment trajectories or downloads."""

import json
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.akaike_score_gate import calibration_audit as ca
from mnist_experiment.akaike_score_gate import fisher_audit as fa
from mnist_experiment.akaike_score_gate.artifacts import StopExecution
from test_fisher_audit import _tiny_parents


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_observed_covariance_centers_within_arrivals_not_drifting_means():
    gradients = torch.tensor([[0.], [2.], [100.], [102.]], dtype=torch.float64)
    weights = torch.tensor([.1, .1, .4, .4], dtype=torch.float64)
    result = ca.observed_covariance(gradients, weights, [2, 2])
    assert result["mean_gradient"].item() == pytest.approx(81.)
    assert result["covariance"].item() == pytest.approx(.68)
    assert result["arrival_mean_gradients"].flatten().tolist() == [1., 101.]
    pooled = gradients.var(unbiased=True) * weights.square().sum()
    assert pooled > result["covariance"].item() * 1000
    nonuniform = ca.observed_covariance(gradients, torch.tensor([.1, .2, .3, .4]), [2, 2])
    assert nonuniform["covariance"].item() == pytest.approx(.6)


def test_positive_weight_singleton_covariance_is_unavailable_not_zero():
    gradients = torch.tensor([[0., 1.], [20., 5.], [21., 6.]])
    result = ca.observed_covariance(gradients, torch.tensor([.2, .4, .4]), [1, 2])
    assert result["covariance"] is None
    assert result["missing_covariance_arrivals"] == [0]
    assert result["status"] == "singleton_strata_unresolved"
    zero_weight = ca.observed_covariance(gradients, torch.tensor([0., .5, .5]), [1, 2])
    assert zero_weight["covariance"] is not None
    with pytest.raises(ValueError):
        ca.observed_covariance(gradients, torch.ones(3), [1, 2])
    with pytest.raises(ValueError):
        ca.observed_covariance(gradients, torch.tensor([.2, .4, .4]), [1, 1])


def test_complete_history_covariance_and_trace_uncertainty():
    values = torch.tensor([[50., -1.], [50., 1.], [50., 0.], [50., 2.]], dtype=torch.float64)
    result = ca.replicated_covariance(values)
    torch.testing.assert_close(result["covariance"], torch.cov(values.mT))
    shifted = ca.replicated_covariance(values + torch.tensor([100., -20.]))
    torch.testing.assert_close(shifted["covariance"], result["covariance"])
    assert result["covariance"][0, 0] == 0
    estimates = torch.tensor([torch.cov(values[torch.arange(len(values)) != i].mT).trace() for i in range(len(values))])
    expected = ((len(values) - 1) / len(values) * (estimates - estimates.mean()).square().sum()).sqrt()
    assert result["trace_jackknife_standard_error"] == pytest.approx(float(expected))
    assert ca.replicated_covariance(values[:2])["trace_jackknife_standard_error"] is None


def test_linear_response_restricts_before_solve_and_does_not_create_information():
    h = torch.tensor([[2., 1., .8], [1., 2., .5], [.8, .5, 2.]], dtype=torch.float64)
    v = torch.diag(torch.tensor([.3, .5, .7], dtype=torch.float64))
    result = ca.response_covariance(h, h * 0, v, torch.tensor([0, 1]))
    inverse = torch.linalg.inv(h[:2, :2])
    torch.testing.assert_close(result["covariance"], inverse @ v[:2, :2] @ inverse)
    assert not torch.allclose(result["covariance"], (torch.linalg.inv(h) @ v @ torch.linalg.inv(h))[:2, :2])
    singular = ca.response_covariance(torch.diag(torch.tensor([1., 0.])), torch.zeros(2, 2),
                                       torch.eye(2), torch.arange(2))
    assert singular["supported_subspace_only"]
    assert singular["covariance"][1, 1] == 0


def test_history_sampler_preserves_positions_counts_weights_and_actual_labels():
    inputs = torch.arange(6, dtype=torch.float64)[:, None]
    targets = torch.tensor([0, 1, 2, 3, 9, 9])
    snapshot = {"batch_positions": [0., 1.], "batch_sizes": [2, 3],
                "weights": torch.tensor([.1, .1, .2, .3, .3]), "position": 1.}
    first = ca.sample_history(inputs, targets, snapshot, family="mixture", seed=9)
    second = ca.sample_history(inputs, targets, snapshot, family="mixture", seed=9)
    torch.testing.assert_close(first["inputs"], second["inputs"])
    assert (first["targets"][:2] != 9).all() and (first["targets"][2:] == 9).all()
    assert first["batch_sizes"] == [2, 3] and first["batch_positions"] == [0., 1.]
    torch.testing.assert_close(first["weights"], snapshot["weights"])
    ids = torch.cat([part["pool_indices"] for part in first["arrival_draws"]])
    torch.testing.assert_close(first["targets"], targets[ids])
    assert first["arrival_draws"][0]["seed"] != first["arrival_draws"][1]["seed"]


def test_history_rotations_preserve_arrival_schedule(monkeypatch):
    monkeypatch.setattr(fa, "rotate_mnist_batch", lambda x, angle, config: x + angle)
    snapshot = {"batch_positions": [0., 10., 30.], "batch_sizes": [2, 2, 2],
                "weights": torch.ones(6) / 6, "position": 30.}
    result = ca.sample_history(torch.zeros(10, 1), torch.arange(10), snapshot, family="rotation", seed=8)
    assert result["inputs"].flatten().tolist() == [0., 0., 10., 10., 30., 30.]
    ids = torch.cat([part["pool_indices"] for part in result["arrival_draws"]])
    torch.testing.assert_close(result["targets"], torch.arange(10)[ids])


def tiny(tmp_path, monkeypatch, *, phase="phase3"):
    result = _tiny_parents(tmp_path, monkeypatch, phase=phase, adaptive=True)
    monkeypatch.setattr(ca, "active_indices", fa.active_indices)
    return result


def test_calibration_parent_api_artifacts_fixed_starts_and_complete_resume(tmp_path, monkeypatch):
    store, case, assets, driver, calls = tiny(tmp_path, monkeypatch)
    path = ca.ensure_calibration_audit(store, 0, case, assets, driver)
    summary = json.loads((path / "summary.json").read_text())
    diagnostics = torch.load(path / "diagnostics.pt", weights_only=False)
    assert summary["phase"] == "phase3" and summary["source_driver_depth"] is None
    assert summary["history_repeats"] == 2 and summary["all_history_fits_unresolved"]
    assert summary["ess"] == pytest.approx(4.)
    assert summary["model_score_bank_inputs"] == 4
    assert len(calls["fit"]) == 1 + 2 + 2
    assert len(calls["fisher"]) == 4  # Main and newest score banks, never one per history.
    assert all(torch.equal(call[1], torch.zeros(6, dtype=torch.float64)) for call in calls["fit"])
    assert all(torch.equal(call[2], torch.eye(6, dtype=torch.float64)) for call in calls["fit"])
    assert diagnostics["histories"][0]["seed"] != diagnostics["histories"][1]["seed"]
    assert diagnostics["calibration"]["between_arrival_mean_drift_not_pooled"]
    rows = json.loads((path / "audit_metrics.json").read_text())
    assert all(row["forecast_gain"] is None for row in rows if "forecast_gain" in row)
    hashes = {name: fa.file_hash(path / name) for name in ca.REQUIRED}
    counts = len(calls["fit"]), len(calls["fisher"])
    assert ca.ensure_calibration_audit(store, 0, case, assets, driver) == path
    assert hashes == {name: fa.file_hash(path / name) for name in ca.REQUIRED}
    assert counts == (len(calls["fit"]), len(calls["fisher"]))


def test_shadow_api_does_not_run_repeated_history_fits_and_can_use_depth_five(tmp_path, monkeypatch):
    store, case, assets, driver, calls = tiny(tmp_path, monkeypatch)
    case["frontier"] = 5
    path = ca.ensure_shadow_audit(store, 0, case, assets, driver)
    summary = json.loads((path / "summary.json").read_text())
    data = torch.load(path / "diagnostics.pt", weights_only=False)
    assert summary["history_repeats"] == 0 and "histories" not in data
    assert set(data["shadows"]["expansions"]) == {1}
    assert len(calls["fit"]) == 2
    assert data["shadows"]["newest_batch"]["episode_anchor_and_precision_unchanged"]


@pytest.mark.parametrize("phase", ["phase3", "phase4"])
def test_shadow_phase_controls_artifacts_summary_namespace_and_resume(tmp_path, monkeypatch, phase):
    store, case, assets, driver, calls = tiny(tmp_path, monkeypatch, phase=phase)
    options = {} if phase == "phase3" else {"phase": phase}
    path = ca.ensure_shadow_audit(store, 0, case, assets, driver, **options)
    summary = json.loads((path / "summary.json").read_text())
    config = json.loads((path / "config.json").read_text())
    assert path.parent == store.root / phase
    assert summary["phase"] == config["phase"] == phase
    assert summary["namespace"].startswith(f"{phase}:shadow_audit:")
    counts = len(calls["fit"]), len(calls["fisher"])
    hashes = {name: fa.file_hash(path / name) for name in ca.REQUIRED}
    assert ca.ensure_shadow_audit(store, 0, case, assets, driver, phase=phase) == path
    assert counts == (len(calls["fit"]), len(calls["fisher"]))
    assert hashes == {name: fa.file_hash(path / name) for name in ca.REQUIRED}


@pytest.mark.parametrize("gate_rank", [None, 0, 4])
def test_base_and_newest_use_gate_rank_without_changing_archive_precision(tmp_path, monkeypatch, gate_rank):
    store, case, assets, driver, _ = tiny(tmp_path, monkeypatch)
    _, _, ctx = ca._prepare(store, 0, case, assets, driver, resume=True,
                            kind="shadow_audit", phase="phase4")
    if gate_rank is not None:
        ctx.policy["gate_rank"] = gate_rank
    expected = ctx.rank if gate_rank is None else gate_rank
    archive_rank, precision = ctx.rank, ctx.precision.clone()
    represented_ranks = []
    represent = ca.represent

    def record_rank(matrix, *, kind, rank):
        represented_ranks.append(rank)
        return represent(matrix, kind=kind, rank=rank)

    monkeypatch.setattr(ca, "represent", record_rank)
    base = ca._base(ctx)
    ca._newest(ctx, base)
    assert represented_ranks == [expected, expected]
    assert ctx.rank == archive_rank == ctx.policy["rank"]
    torch.testing.assert_close(ctx.precision, precision)


def test_depth_opportunity_rejects_phase3_source_and_phase_namespaces_differ(tmp_path, monkeypatch):
    store, case, assets, driver, _ = tiny(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Phase 1"):
        fa.ensure_depth_audit(store, 0, case, assets, driver)
    _, _, calibration = ca._prepare(store, 0, case, assets, driver, resume=True, kind="calibration_audit")
    _, _, shadow = ca._prepare(store, 0, case, assets, driver, resume=True, kind="shadow_audit")
    assert calibration.seed("observed_history:0") != shadow.seed("observed_history:0")
    assert calibration.namespace.startswith("phase3:calibration_audit:")


@pytest.mark.parametrize("boundary", ["sample", "fit"])
def test_history_interruption_resumes_without_resampling_or_refitting(tmp_path, monkeypatch, boundary):
    store, case, assets, driver, calls = tiny(tmp_path / "resumed", monkeypatch)
    original_sampler, original_fit = ca.sample_history, fa.fit_restricted
    samples = []

    def sample(*args, **kwargs):
        result = original_sampler(*args, **kwargs)
        samples.append(kwargs["seed"])
        if boundary == "sample" and len(samples) == 1:
            store.limit.stop_requested = True
        return result

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        if boundary == "fit" and len(calls["fit"]) == 2:
            store.limit.stop_requested = True
        return result

    monkeypatch.setattr(ca, "sample_history", sample)
    monkeypatch.setattr(fa, "fit_restricted", fit)
    with pytest.raises(StopExecution):
        ca.ensure_calibration_audit(store, 0, case, assets, driver)
    store.limit.stop_requested = False
    path = ca.ensure_calibration_audit(store, 0, case, assets, driver)
    assert len(samples) == 2 and len(set(samples)) == 2
    assert len(calls["fit"]) == 5
    rows = json.loads((path / "audit_metrics.json").read_text())
    monkeypatch.setattr(ca, "sample_history", original_sampler)
    other, case, assets, driver, other_calls = tiny(tmp_path / "uninterrupted", monkeypatch)
    full = ca.ensure_calibration_audit(other, 0, case, assets, driver)
    assert rows == json.loads((full / "audit_metrics.json").read_text())
    assert len(other_calls["fit"]) == len(calls["fit"])


def test_newest_objective_stationarity_is_not_inherited_from_full_episode(monkeypatch):
    monkeypatch.setattr(ca, "active_indices", lambda layout, depth: (torch.arange(depth), {}))
    context = SimpleNamespace(layout=None, device=torch.device("cpu"), precision=torch.eye(3, dtype=torch.float64),
                              anchor=torch.zeros(3, dtype=torch.float64), policy={"c": 1.}, fit_policy={},
                              study=SimpleNamespace(tolerance_grad=1e-5, solve_floor_relative=1e-8))
    fitted = {"theta": torch.zeros(3), "fit": {"converged": True}}
    h = torch.eye(3, dtype=torch.float64)
    full = ca._gate(context, fitted, torch.tensor([0., -1., 0.]), h, h / 20, 1, 1, namespace="full")
    newest = ca._gate(context, fitted, torch.tensor([1., -1., 0.]), h, h / 2, 1, 1, namespace="newest")
    assert full["status"] in {"open", "closed"} and full["gain"] is not None
    assert newest["status"] == "restricted_fit_unresolved" and newest["gain"] is None
    assert newest["open"] is None
