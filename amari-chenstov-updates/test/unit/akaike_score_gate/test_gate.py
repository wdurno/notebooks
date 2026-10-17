import pytest
import torch

from mnist_experiment.akaike_score_gate.gate import (
    block_preview, direct_block_preview, evaluate_gate, likelihood_gain, optimism,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("score,opens", [(.9, False), (1., False), (1.1, True)])
def test_classical_s_greater_than_2q_and_equality_stays_small(score, opens):
    n = 2
    H = torch.eye(2, dtype=torch.float64)
    g = tensor([0., score])
    result = evaluate_gate(g, g, H, torch.zeros_like(H), H / n, [0], [1])
    S = n * score**2
    assert result["gain"] == pytest.approx(S / (2 * n))
    assert result["delta_optimism"] == pytest.approx(1 / n)
    assert result["d_eff_active"] == 1
    assert result["d_eff_expanded"] == 2
    assert result["open"] is opens
    assert result["open"] == (S > 2)


def test_schur_matches_direct_joint_with_global_coordinates_and_nonzero_active_residual():
    generator = torch.Generator().manual_seed(31)
    X = torch.randn(5, 5, generator=generator, dtype=torch.float64)
    K = X @ X.mT + torch.eye(5, dtype=torch.float64)
    r = tensor([1., 2., 3., 4., 5.])
    A, B = [4, 1], [3, 0]
    result = block_preview(r, K, A, B)
    direct = direct_block_preview(r, K, A, B)
    torch.testing.assert_close(result["displacement"], direct)
    joined = torch.tensor(A + B)
    expected = -torch.linalg.solve(K[joined[:, None], joined[None, :]], r[joined])
    torch.testing.assert_close(result["displacement"][joined], expected)
    assert result["displacement"][2] == 0
    assert not torch.allclose(expected, (-torch.linalg.solve(K, r))[joined])
    assert result["solve_diagnostics"]["relative_residual"] < 1e-12


def test_diagonal_current_curvature_keeps_cross_penalty_and_forecasts_data_gain():
    H = torch.diag(tensor([1., 2.]))
    P = tensor([[3., 1.], [1., 2.]])
    offset = tensor([1., 0.])
    g = tensor([-3., -2.])
    r = g + P @ offset
    result = evaluate_gate(g, r, H, P, H / 20, [0], [1])
    assert result["schur"].item() == pytest.approx(4 - 1 / 4)
    torch.testing.assert_close(result["displacement"], -torch.linalg.solve(H + P, r))
    retained_gain = float(-r @ result["displacement"] - .5 * result["displacement"] @ (H + P) @ result["displacement"])
    assert result["gain"] != pytest.approx(retained_gain)
    assert result["gain"] == pytest.approx(likelihood_gain(g, H, result["displacement"]))
    diagonalized = evaluate_gate(g, r, H, torch.diag(P.diagonal()), H / 20, [0], [1])
    assert not torch.allclose(result["displacement"], diagonalized["displacement"])


def test_active_only_forecast_is_subtracted_and_unresolved_fit_does_not_open():
    H = torch.eye(2, dtype=torch.float64)
    g = tensor([10., 1.])
    result = evaluate_gate(g, g, H, torch.zeros_like(H), H / 100, [0], [1])
    assert result["gain_expanded"] == 50.5
    assert result["gain_active_only"] == 50
    assert result["gain"] == .5
    assert not result["open"]
    assert result["status"] == "restricted_fit_unresolved"


def test_shortened_preview_evaluates_actual_displacement():
    H = torch.eye(2, dtype=torch.float64)
    g = tensor([0., 2.])
    result = evaluate_gate(g, g, H, H, H / 30, [0], [1], alpha=.25)
    torch.testing.assert_close(result["displacement"], tensor([0., -.25]))
    assert result["gain"] == pytest.approx(.5 - .5 * .25**2)
    assert result["delta_optimism"] == pytest.approx(1 / 60)


def test_mean_sum_units_scale_gain_and_optimism_together():
    H = tensor([[2., .5], [.5, 1.]])
    P = tensor([[1., .2], [.2, .4]])
    g, r = tensor([-.1, -2.]), tensor([0., -1.8])
    mean = evaluate_gate(g, r, H, P, H / 10, [0], [1])
    summed = evaluate_gate(10*g, 10*r, 10*H, 10*P, 10*H, [0], [1])
    torch.testing.assert_close(mean["displacement"], summed["displacement"])
    assert summed["gain"] == pytest.approx(10 * mean["gain"])
    assert summed["delta_optimism"] == pytest.approx(10 * mean["delta_optimism"])
    assert summed["d_eff_expanded"] == pytest.approx(mean["d_eff_expanded"])
    assert summed["open"] == mean["open"]


def test_trace_optimism_and_effective_dimension_with_correlated_penalty():
    H = tensor([[2., .5], [.5, 1.]])
    P = tensor([[1., -.2], [-.2, .4]])
    V = tensor([[.2, .02], [.02, .1]])
    result = optimism(H, P, V)
    response = torch.linalg.inv(H + P)
    assert result["optimism"] == pytest.approx(float(torch.trace(response @ V)))
    assert result["d_eff"] == pytest.approx(float(torch.trace(response @ H)))
    # Replicated local quadratic fits, not synthetic auxiliary LFU scores.
    rng = torch.Generator().manual_seed(17)
    xi = torch.randn(30000, 2, generator=rng, dtype=torch.float64) @ torch.linalg.cholesky(V).mT
    estimation_error = -xi @ response.mT
    realized_optimism = -(xi * estimation_error).sum(1).mean()
    assert float(realized_optimism) == pytest.approx(result["optimism"], rel=.025)


def test_supported_zero_information_and_floor_do_not_create_dimension():
    zero = torch.zeros(2, 2, dtype=torch.float64)
    result = evaluate_gate(tensor([0., 0.]), tensor([0., 0.]), zero, zero, zero, [0], [1])
    assert not result["open"] and result["status"] == "no_information"
    assert result["d_eff_expanded"] == 0
    assert torch.count_nonzero(result["displacement"]) == 0
    H = torch.diag(tensor([1., 1e-12]))
    for floor in (1e-9, 1e-8, 1e-7):
        trace = optimism(H, zero, H / 10, floor_relative=floor)
        assert trace["d_eff"] == pytest.approx(2.)
        assert trace["optimism"] == pytest.approx(.2)
    singular = block_preview(tensor([0., 1.]), torch.diag(tensor([1., 0.])), [0], [1])
    assert singular["status"] == "solve_unresolved"


def test_tenfold_floor_perturbations_leave_well_resolved_gate_unchanged():
    H = tensor([[2., .4], [.4, 1.]])
    g = tensor([0., 1.])
    results = [evaluate_gate(g, g, H, .1*H, H / 30, [0], [1], floor_relative=floor)
               for floor in (1e-9, 1e-8, 1e-7)]
    assert all(result["open"] for result in results)
    for result in results[1:]:
        torch.testing.assert_close(result["displacement"], results[0]["displacement"])


def test_invalid_negative_optimism_is_not_a_free_expansion():
    H = torch.eye(2, dtype=torch.float64)
    result = evaluate_gate(tensor([0., 1.]), tensor([0., 1.]), H, torch.zeros_like(H),
                           torch.diag(tensor([1., -1.])), [0], [1])
    assert result["status"] == "negative_delta_optimism"
    assert not result["open"]


def test_empty_active_and_candidate_sets_and_overlap_validation():
    H = torch.eye(2, dtype=torch.float64)
    result = evaluate_gate(tensor([1., 2.]), tensor([1., 2.]), H, H, H / 10, [], [1])
    assert result["d_eff_active"] == 0
    assert result["displacement"][0] == 0
    empty = evaluate_gate(tensor([0., 0.]), tensor([0., 0.]), H, H, H, [0], [])
    assert empty["status"] == "no_candidate"
    with pytest.raises(ValueError, match="disjoint"):
        block_preview(tensor([0., 0.]), H, [0], [0])
