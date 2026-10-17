"""Small deterministic audit checks; no MNIST downloads or experiment runs."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.parameters import ParameterLayout
from mnist_experiment.akaike_score_gate import fisher_audit as audit
from mnist_experiment.akaike_score_gate.artifacts import StopExecution, UnitStore
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.fisher import represent


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def study():
    return Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")


def test_matrix_metrics_and_subspace_are_finite_reference_comparisons():
    matrix = torch.diag(torch.tensor([1., 3., 6.], dtype=torch.float64))
    result = audit.matrix_metrics(matrix, matrix, subspace_rank=2)
    assert result["relative_frobenius_error"] == 0
    assert result["relative_diagonal_error"] == 0
    assert result["resolved_rank"] == 3
    assert result["leading_subspace_overlap"] == pytest.approx(1.)
    assert result["condition_estimate"] == pytest.approx(6.)
    assert result["reference_is_finite"]
    zero = torch.zeros_like(matrix)
    assert audit.matrix_metrics(zero, zero)["relative_frobenius_error"] == 0
    assert audit.matrix_metrics(matrix, zero)["relative_frobenius_error"] is None
    assert audit.matrix_metrics(zero, matrix)["leading_subspace_overlap"] is None
    skew = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    assert audit.matrix_metrics(matrix + skew, matrix)["relative_frobenius_error"] == 0


def test_reference_disagreement_uses_mean_norm_and_handles_zeros():
    a = torch.tensor([1., 0.])
    b = torch.tensor([3., 0.])
    assert audit.symmetric_relative_difference(a, b) == 1.
    assert audit.symmetric_relative_difference(a * 0, b * 0) == 0.
    assert audit.symmetric_relative_difference(a * 0, b) == 2.
    with pytest.raises(ValueError):
        audit.symmetric_relative_difference(a, torch.ones(3))


def _fake_gate(value, *, opens=False, status=None):
    return {"conditional_solution": torch.tensor([value], dtype=torch.float64),
            "open": opens, "status": status or ("open" if opens else "closed")}


def test_reference_thresholds_and_unresolved_gate_not_reported_as_agreement():
    eye = torch.eye(2, dtype=torch.float64)
    assert not audit.reference_disagreement(eye, eye * 1.04)["expand"]
    assert audit.reference_disagreement(eye, eye * 1.06)["reasons"] == ["matrix_disagreement"]
    result = audit.reference_disagreement(eye, eye, _fake_gate(1.), _fake_gate(1.2), residual_norm=1.)
    assert result["reasons"] == ["conditional_solve_disagreement"]
    assert not audit.reference_disagreement(eye, eye, _fake_gate(1.), _fake_gate(1.2), residual_norm=0.)["expand"]
    assert audit.reference_disagreement(eye, eye, _fake_gate(1.), _fake_gate(1., opens=True))["reasons"] == ["reference_gate_disagreement"]
    result = audit.reference_disagreement(eye, eye, _fake_gate(1., status="restricted_fit_unresolved"), _fake_gate(1.))
    assert not result["gate_comparison_resolved"]
    assert not result["gate_disagreement"]


def test_reference_weights_are_per_observation_mass_not_equal_batch_weights():
    positions, mass = audit.weighted_history([0., .1], [1, 3], torch.tensor([.1, .3, .3, .3]))
    assert mass.tolist() == pytest.approx([.1, .9])
    assert float(positions @ mass) == pytest.approx(.09)
    for sizes, weights in (([1, 2], torch.ones(4)), ([1., 3.], torch.ones(4)), ([1, 3], torch.zeros(4))):
        with pytest.raises(ValueError):
            audit.weighted_history([0., .1], sizes, weights)


def test_mixture_reference_matches_weighted_schedule_not_realized_buffer_labels():
    x = torch.arange(10, dtype=torch.float64)[:, None]
    y = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 9, 9])
    kwargs = dict(family="mixture", batch_positions=[0., 1.], batch_sizes=[1, 3],
                  weights=torch.tensor([.2, .1, .3, .4]), sample_size=4000, seed=23)
    sampled, meta = audit.sample_reference_inputs(x, y, **kwargs)
    again, meta_again = audit.sample_reference_inputs(x, y, **kwargs)
    assert torch.equal(sampled, again)
    assert torch.equal(meta["pool_indices"], meta_again["pool_indices"])
    assert meta["weighted_prevalence"] == pytest.approx(.8)
    drawn_y = y[meta["pool_indices"]]
    assert float((drawn_y == 9).double().mean()) == pytest.approx(.8, abs=.025)
    assert float((drawn_y[drawn_y != 9] == 1).double().mean()) == pytest.approx(5 / 8, abs=.065)
    _, current = audit.sample_reference_inputs(x, y, **kwargs, instantaneous=0.)
    assert current["weighted_prevalence"] == 0.
    assert not (y[current["pool_indices"]] == 9).any()
    with pytest.raises(ValueError, match="stratum"):
        audit.sample_reference_inputs(x[:8], y[:8], **kwargs)


def test_rotation_reference_mixes_past_angles_by_weight(monkeypatch):
    calls = []

    def rotate(x, angle, config):
        calls.append(angle)
        return x + angle

    monkeypatch.setattr(audit, "rotate_mnist_batch", rotate)
    x = torch.zeros(20, 1, 2, 2)
    y = torch.arange(20) % 10
    sample, meta = audit.sample_reference_inputs(
        x, y, family="rotation", batch_positions=[10., 30.], batch_sizes=[1, 2],
        weights=torch.tensor([.1, .2, .7]), sample_size=1000, seed=8)
    assert set(calls) == {10., 30.}
    assert meta["arrival_weight_mass"].tolist() == pytest.approx([.1, .9])
    assert float((sample[:, 0, 0, 0] == 30).double().mean()) == pytest.approx(.9, abs=.035)


def test_finite_replication_bias_variance_identity():
    matrices = torch.stack((torch.eye(2), torch.eye(2) * 2, torch.eye(2) * 3))
    result = audit.bias_variance(matrices, torch.eye(2) * 1.5)
    assert result["squared_bias_to_finite_reference"] == pytest.approx(.5)
    assert result["variance_population_denominator"] == pytest.approx(4 / 3)
    assert result["mean_squared_error_to_finite_reference"] == pytest.approx(.5 + 4 / 3)
    assert result["variance_unbiased"] == pytest.approx(2.)
    with pytest.raises(ValueError):
        audit.bias_variance(matrices[:1], matrices[0])


def test_orientation_control_preserves_spectrum_diagonal_and_psd():
    matrix = torch.ones(4, 4, dtype=torch.float64) + torch.diag(torch.arange(1, 5, dtype=torch.float64))
    changed = audit.orientation_control(matrix, seed=0)
    assert not torch.equal(changed, matrix)
    torch.testing.assert_close(changed.diagonal(), matrix.diagonal())
    torch.testing.assert_close(torch.linalg.eigvalsh(changed), torch.linalg.eigvalsh(matrix))
    assert torch.linalg.eigvalsh(changed).min() > 0


def test_cpu_checkpoint_detaches_and_clones_state_dict_before_shadow_reset():
    model = nn.Linear(2, 2)
    copied = audit._cpu({"model": model.state_dict()})
    before = copied["model"]["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.)
    torch.testing.assert_close(copied["model"]["weight"], before)


def test_historical_replay_uses_actual_contributions_eta_and_common_initial():
    initial = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float64)
    first = torch.tensor([[4., -.5], [-.5, 1.]], dtype=torch.float64)
    second = torch.tensor([[1., .2], [.2, 3.]], dtype=torch.float64)
    history = [{"eta": .25, "dense_new_fisher": first}, {"eta": .5, "dense_new_fisher": second}]
    expected = .5 * (.75 * initial + .25 * first) + .5 * second
    torch.testing.assert_close(audit.replay_archive(initial, history), expected)
    torch.testing.assert_close(audit.replay_archive(initial, history, kind="diag"), torch.diag(expected.diagonal()))
    expected_lr = represent(.5 * represent(.75 * represent(initial, rank=1) + .25 * first, rank=1) + .5 * second, rank=1)
    torch.testing.assert_close(audit.replay_archive(initial, history, kind="lr", rank=1), expected_lr)
    high = audit.replay_archive(initial, history, contributions=[first * 2, second * 3])
    torch.testing.assert_close(high, .5 * (.75 * initial + .25 * first * 2) + .5 * second * 3)
    with pytest.raises(ValueError):
        audit.replay_archive(initial, history, contributions=[first])


def test_identical_operator_woodbury_numerical_check():
    diagonal = torch.tensor([.2, .5, .3], dtype=torch.float64)
    factors = torch.tensor([[1., 2.], [3., -1.], [-.5, .8]], dtype=torch.float64)
    result = audit.identical_operator_solve(diagonal, factors, torch.tensor([1., 3., -2.]))
    torch.testing.assert_close(result["dense_solution"], result["woodbury_solution"], rtol=1e-12, atol=1e-12)
    assert result["woodbury_relative_residual"] < 1e-12
    assert "numerical_only" in result["comparison"]
    with pytest.raises(ValueError):
        audit.identical_operator_solve(diagonal * 0, factors, torch.ones(3))


def test_observed_hvp_matches_gradient_finite_difference_and_restores_no_parameters():
    torch.manual_seed(4)
    model = nn.Linear(2, 3).double()
    layout = ParameterLayout.from_module(model)
    x = torch.tensor([[.3, .2], [1., -.2], [-.1, .9]], dtype=torch.float64)
    y, weights = torch.tensor([0, 2, 1]), torch.tensor([.1, .2, .7], dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    direction = torch.linspace(-.1, .1, len(theta), dtype=torch.float64)
    product = audit.observed_hvp(model, layout, x, y, weights, direction)
    eps = 1e-5
    layout.copy_vector_to_module(model, theta + eps * direction)
    plus = audit.loss_gradient(model, layout, x, y, weights)
    layout.copy_vector_to_module(model, theta - eps * direction)
    minus = audit.loss_gradient(model, layout, x, y, weights)
    layout.copy_vector_to_module(model, theta)
    torch.testing.assert_close(product, (plus - minus) / (2 * eps), rtol=1e-7, atol=1e-10)
    gradient = audit.loss_gradient(model, layout, x, y, weights)
    layout.copy_vector_to_module(model, theta - .001 * gradient)
    assert audit._loss(model, x, y, weights) < nn.functional.cross_entropy(nn.functional.linear(x, theta[:6].reshape(3, 2), theta[6:]), y, reduction="none") @ weights


def test_unresolved_restricted_fit_does_not_emit_gain_or_open_gate():
    h = torch.eye(2, dtype=torch.float64)
    g = torch.tensor([0., -2.], dtype=torch.float64)
    study = SimpleNamespace(tolerance_grad=1e-5, solve_floor_relative=1e-8)
    gate = audit._gate(g, g, h, h * 0, 10., torch.tensor([0]), torch.tensor([1]), study, converged=False)
    assert gate["gain"] is None and gate["risk_reduction"] is None
    assert not gate["open"]
    assert gate["status"] == "restricted_fit_unresolved"


def test_paired_gain_se_stratifies_mixture_and_keeps_pairing():
    panel = {"ids": torch.arange(4), "targets": torch.tensor([0, 1, 9, 9]),
             "losses": torch.tensor([2., 4., 6., 8.])}
    after = {**panel, "losses": torch.ones(4)}
    result = audit.paired_panel_gain(panel, after, family="mixture", position=.25)
    assert result["gain"] == pytest.approx(.75 * 2 + .25 * 6)
    assert result["standard_error"] == pytest.approx((.75**2 + .25**2)**.5)
    with pytest.raises(ValueError, match="identities"):
        audit.paired_panel_gain(panel, {**after, "ids": torch.arange(4).flip(0)}, family="rotation", position=0.)


def test_exact_snapshot_selection_never_uses_nearest_step():
    first, last = {"step": 0}, {"step": 4}
    assert audit.select_snapshot([first, last], 4) is last
    with pytest.raises(ValueError, match="exactly one"):
        audit.select_snapshot([first, last], 3)
    with pytest.raises(ValueError, match="exactly one"):
        audit.select_snapshot([last, last], 4)


def test_sealed_checkpoint_survives_limit_and_detects_corruption(tmp_path):
    store = UnitStore(tmp_path, study())
    unit = store.unit("phase2", "test_checkpoints", 0)
    session = store.begin(unit, ("summary.json",), resume=True)
    calls = []

    def compute():
        calls.append(1)
        store.limit.stop_requested = True
        return torch.eye(2)

    cache = audit._Checkpoints(session, store.limit)
    with pytest.raises(StopExecution):
        cache.get("chunk", compute)
    store.limit.stop_requested = False
    restored = audit._Checkpoints(session, store.limit)
    torch.testing.assert_close(restored.get("chunk", compute), torch.eye(2))
    assert len(calls) == 1
    path = session.path / restored.index["chunk"]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt sealed"):
        restored.get("chunk", compute)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(6, dtype=torch.float64))

    def forward(self, inputs):
        return torch.cat((inputs[:, :1] * self.theta[None, :], inputs.new_zeros(len(inputs), 4)), dim=1)


def _tiny_parents(tmp_path, monkeypatch, *, phase="phase1", adaptive=False):
    config = study().mapping()
    config.update(dtype="float64", reference_samples=4, reference_max_samples=8,
                  fisher_chunk_size=2, score_samples=4)
    store = UnitStore(tmp_path, Study(config))
    model, layout = TinyModel(), ParameterLayout.from_module(TinyModel())
    x, y = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float64), torch.tensor([0, 1, 0, 1])
    case = {"environment": {"name": "rotation_linear", "family": "rotation", "schedule": "linear", "samples": 2},
            "step": 3, "frontier": 1, "source_depth": 4, "historical": False}
    assets = {"initial_fisher": torch.eye(6, dtype=torch.float64), "initial_model": copy.deepcopy(model.state_dict()),
              "parameter_layout": layout.metadata(), "reference_inputs": x, "reference_targets": y,
              "evaluation_inputs": x, "evaluation_targets": y,
              "streams": {"rotation_linear": {"positions": [0., 0., 0., 0., 0.], "rotation": {
                  "knots_degrees": [0., 30., 0., 30.], "transitions_per_arrow": 1,
                  "interpolation": "bilinear", "expand": False, "fill": 0.}}}}
    theta = layout.flatten_module(model, detach=True)
    snapshot = {"step": 3, "model": copy.deepcopy(model.state_dict()), "theta": theta, "anchor": theta,
                "archive": assets["initial_fisher"], "precision": assets["initial_fisher"],
                "inputs": x, "targets": y, "weights": torch.ones(4, dtype=torch.float64) / 4,
                "batch_positions": [0., 0.], "batch_sizes": [2, 2], "position": 0., "history": []}
    unit = store.unit(phase, "assets", 0)
    session = store.begin(unit, ("assets.pt", "summary.json"), resume=True)
    session.write_torch("assets.pt", assets)
    session.write_json("summary.json", {"width": 5})
    asset_path = store.finish(session, ("assets.pt", "summary.json"))
    detail = {"policy": {"ewc": "lr", "rank": 2, "gate": "lr", "score_samples": 4, "c": 1.},
              "retention": {"omega": 1., "ridge_ratio": 0., "max_iter": 1, "residual_tolerance": 1e-6}}
    if adaptive:
        detail["policy"]["depth"] = None
        case["source_depth"] = None
    unit = store.unit(phase, "trajectory", 0, environment="rotation_linear", detail=detail)
    session = store.begin(unit, ("snapshots.pt", "summary.json"), resume=True)
    session.write_torch("snapshots.pt", [snapshot])
    session.write_json("summary.json", {"width": 5})
    driver_path = store.finish(session, ("snapshots.pt", "summary.json"))

    def build(*args, **kwargs):
        result = TinyModel().to(device=kwargs.get("device", "cpu"), dtype=kwargs.get("dtype", torch.float64))
        return result, ParameterLayout.from_module(result)

    def active(layout, depth):
        return torch.arange(6 - depth, 6), {"candidate_indices": [5 - depth] if depth < 6 else []}

    calls = {"fit": [], "fisher": []}

    def fit(model, layout, inputs, targets, weights, anchor, precision, indices, **kwargs):
        calls["fit"].append((indices.clone(), layout.flatten_module(model, detach=True), precision.clone(), kwargs))
        return {"converged": False, "restricted_residual_max": 1.,
                "data_loss_after": float(audit._loss(model, inputs, targets, weights).detach())}

    def fisher(model, layout, inputs, weights=None, **kwargs):
        calls["fisher"].append(kwargs["seed"])
        return torch.eye(6, dtype=torch.float64), {"seed": kwargs["seed"]}

    monkeypatch.setattr(audit, "build_model", build)
    monkeypatch.setattr(audit, "active_indices", active)
    monkeypatch.setattr(audit, "layer_indices", lambda layout: {str(i): torch.tensor([i]) for i in range(6)})
    monkeypatch.setattr(audit, "fit_restricted", fit)
    monkeypatch.setattr(audit, "model_fisher", fisher)
    return store, case, asset_path, driver_path, calls


def test_tiny_audit_artifacts_and_completed_resume_do_no_scientific_work(tmp_path, monkeypatch):
    store, case, asset_path, driver_path, calls = _tiny_parents(tmp_path, monkeypatch)
    result = audit.ensure_audit(store, 0, case, asset_path, driver_path)
    summary = json.loads((result / "summary.json").read_text())
    rows = json.loads((result / "audit_metrics.json").read_text())
    assert summary["all_restricted_fits_unresolved"]
    assert len(rows) == 2 * 7 + 6
    assert all(row["gate_gain"] is None and row["gate_open"] is None for row in rows)
    # EWC rows pair their integration draws, but never borrow fitted matrices.
    assert max(calls["fisher"].count(seed) for seed in calls["fisher"]) == 2
    assert len(calls["fit"]) == 6
    assert all(torch.equal(item[1], torch.zeros(6, dtype=torch.float64)) for item in calls["fit"])
    assert all(item[3]["max_iter"] == 1 and item[3]["residual_tolerance"] == 1e-6 for item in calls["fit"])
    counts = (len(calls["fit"]), len(calls["fisher"]))
    digests = {name: audit.file_hash(result / name) for name in audit.REQUIRED}
    assert audit.ensure_audit(store, 0, case, asset_path, driver_path) == result
    assert counts == (len(calls["fit"]), len(calls["fisher"]))
    assert digests == {name: audit.file_hash(result / name) for name in audit.REQUIRED}


def test_depth_audit_keeps_all_unresolved_cases_and_splits_holdouts(tmp_path, monkeypatch):
    store, case, asset_path, driver_path, calls = _tiny_parents(tmp_path, monkeypatch)
    result = audit.ensure_depth_audit(store, 0, case, asset_path, driver_path)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["selected_depth"] is None
    assert len(calls["fit"]) == 6 and not calls["fisher"]
    data = torch.load(result / "diagnostics.pt", weights_only=False)
    selected = data["baseline"]["selection"]["current"]["ids"]
    scored = data["baseline"]["scoring"]["current"]["ids"]
    assert not set(selected.tolist()) & set(scored.tolist())


@pytest.mark.parametrize("stop_at", [1, 7])
def test_interrupted_score_chunk_resume_does_not_repeat_work_or_rows(tmp_path, monkeypatch, stop_at):
    store, case, assets, driver, calls = _tiny_parents(tmp_path / "resumed", monkeypatch)
    fisher = audit.model_fisher

    def interrupted(*args, **kwargs):
        result = fisher(*args, **kwargs)
        if len(calls["fisher"]) == stop_at:
            store.limit.stop_requested = True
        return result

    monkeypatch.setattr(audit, "model_fisher", interrupted)
    with pytest.raises(StopExecution):
        audit.ensure_audit(store, 0, case, assets, driver)
    assert len(calls["fisher"]) == stop_at
    store.limit.stop_requested = False
    monkeypatch.setattr(audit, "model_fisher", fisher)
    completed = audit.ensure_audit(store, 0, case, assets, driver)
    assert len(calls["fit"]) == 6
    assert max(calls["fisher"].count(seed) for seed in calls["fisher"]) == 2
    rows = json.loads((completed / "audit_metrics.json").read_text())
    keys = [(r["category"], r["representation"], r["repeat"], r.get("ewc")) for r in rows]
    assert len(keys) == len(set(keys))
    other, case, assets, driver, other_calls = _tiny_parents(tmp_path / "uninterrupted", monkeypatch)
    full = audit.ensure_audit(other, 0, case, assets, driver)
    assert rows == json.loads((full / "audit_metrics.json").read_text())
    assert len(calls["fisher"]) == len(other_calls["fisher"])


def test_reference_expands_once_to_ceiling_and_retains_unresolved_state():
    requests = []

    def estimate(key, state, count, **kwargs):
        requests.append((key, count))
        return {"matrix": torch.eye(2) * (1. if "half0" in key else 2.)}

    def gate(*args, **kwargs):
        return {**_fake_gate(1.), "conditional_residual": torch.ones(1)}

    context = SimpleNamespace(study=SimpleNamespace(reference_samples=4, reference_max_samples=8),
                              estimate=estimate, gate=gate)
    result = audit._Audit.reference(context, "ref", {}, None, None, None, None, None, converged=True)
    assert result["input_count"] == 8 and result["unresolved"]
    assert len(requests) == 4 and all(count == 2 for _, count in requests)
    assert len(result["checks"]) == 2
    torch.testing.assert_close(result["matrix"], torch.eye(2) * 1.5)


def test_historical_case_profiles_all_five_frontiers(tmp_path, monkeypatch):
    store, case, assets, driver, calls = _tiny_parents(tmp_path, monkeypatch)
    case["historical"] = True
    path = audit.ensure_audit(store, 0, case, assets, driver)
    data = torch.load(path / "diagnostics.pt", weights_only=False)
    assert set(data["frontiers"]) == set(range(1, 6))
    assert len(calls["fit"]) == 5 * 6
    assert data["history"]["consolidation_count"] == 0
    torch.testing.assert_close(data["history"]["reference"]["matrix"], torch.eye(6, dtype=torch.float64))


def test_depth_fits_restart_common_state_after_each_shadow(tmp_path, monkeypatch):
    store, case, assets, driver, calls = _tiny_parents(tmp_path, monkeypatch)
    fit = audit.fit_restricted

    def moving_fit(model, layout, *args, **kwargs):
        result = fit(model, layout, *args, **kwargs)
        with torch.no_grad():
            model.theta.add_(.25)
        return result

    monkeypatch.setattr(audit, "fit_restricted", moving_fit)
    audit.ensure_depth_audit(store, 0, case, assets, driver)
    assert all(torch.equal(item[1], torch.zeros(6, dtype=torch.float64)) for item in calls["fit"])


def test_high_sample_history_uses_each_saved_state_and_buffer():
    initial = torch.eye(2, dtype=torch.float64)
    first = {"model": {"past": 0}, "inputs": torch.tensor([1.]), "weights": torch.ones(1),
             "eta": .25, "dense_new_fisher": initial * 2, "archive_before": initial}
    second = {"model": {"past": 1}, "inputs": torch.tensor([5.]), "weights": torch.ones(1),
              "eta": .5, "dense_new_fisher": initial * 4, "archive_before": initial * 1.25}
    history = [first, second]
    calls = []

    def estimate(key, state, count, *, snapshot):
        calls.append((state, snapshot))
        return {"matrix": snapshot["dense_new_fisher"] * 1.1}

    context = SimpleNamespace(
        snapshot={"history": history, "model": {"current": True},
                  "archive": audit.replay_archive(initial, history, kind="diag")},
        assets={"initial_fisher": initial}, rank=1, policy={"ewc": "diag"},
        case={"historical": True}, study=SimpleNamespace(reference_samples=4, reference_max_samples=8),
        estimate=estimate)
    result = audit._history(context)
    assert [state["past"] for state, _ in calls] == [0, 0, 1, 1]
    assert all(state is snapshot["model"] for state, snapshot in calls)
    expected = audit.replay_archive(initial, history, contributions=[initial * 2.2, initial * 4.4])
    torch.testing.assert_close(result["reference"]["matrix"], expected)
