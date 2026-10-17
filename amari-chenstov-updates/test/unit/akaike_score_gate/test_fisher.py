import pytest
import torch
from torch import nn

from mnist_experiment.akaike_score_gate.fisher import (
    model_fisher, represent, representation_diagnostics,
)
from mnist_experiment.akaike_score_gate.model import active_indices, build_model
from src.parameters import ParameterLayout


class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-1., 1., 10, dtype=torch.float64))

    def forward(self, inputs):
        return self.bias.expand(inputs.shape[0], -1)


def linear_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(21)
        model = nn.Linear(2, 10, dtype=torch.float64)
    return model, ParameterLayout.from_module(model)


def test_class_integrated_fisher_matches_logit_jacobian_covariance():
    model, layout = linear_fixture()
    inputs = torch.tensor([[.2, -.5], [1., .7]], dtype=torch.float64)
    weights = torch.tensor([.8, .2], dtype=torch.float64)
    actual, diagnostics = model_fisher(model, layout, inputs, weights, exact_labels=True,
                                       chunk_size=1, return_score_bank=True)
    expected = torch.zeros_like(actual)
    for x, a in zip(inputs, weights):
        p = model(x).detach().softmax(-1)
        jacobian = torch.cat((torch.kron(torch.eye(10, dtype=torch.float64), x[None, :]),
                              torch.eye(10, dtype=torch.float64)), dim=1)
        expected += a * jacobian.mT @ (torch.diag(p) - p[:, None] * p[None, :]) @ jacobian
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    vector = torch.linspace(-1., 1., layout.total_numel, dtype=torch.float64)
    torch.testing.assert_close(actual @ vector, expected @ vector)
    assert diagnostics["score_count"] == 20
    assert diagnostics["input_draw_count"] == 2
    assert diagnostics["evidence_ess"] == pytest.approx(1 / (.8**2 + .2**2))
    bank = diagnostics["score_bank"]
    torch.testing.assert_close(actual, bank["scores"].mT @ (bank["weights"][:, None] * bank["scores"]))


def test_score_bank_sign_is_log_likelihood_not_loss_gradient():
    model = BiasModel()
    layout = ParameterLayout.from_module(model)
    _, diagnostics = model_fisher(model, layout, torch.ones(1, 1, dtype=torch.float64),
                                  exact_labels=True, return_score_bank=True)
    scores = diagnostics["score_bank"]["scores"]
    p = model.bias.detach().softmax(-1)
    torch.testing.assert_close(scores, torch.eye(10, dtype=torch.float64) - p[None, :])
    loss_gradient = torch.autograd.grad(-model.bias.log_softmax(-1)[3], model.bias)[0]
    torch.testing.assert_close(scores[3], -loss_gradient)
    assert model.bias.grad is None


def test_sampled_labels_converge_to_exact_fisher_and_are_not_observed_labels():
    model = BiasModel()
    layout = ParameterLayout.from_module(model)
    x = torch.ones(2, 1, dtype=torch.float64)
    exact, _ = model_fisher(model, layout, x, exact_labels=True)
    estimate, diag = model_fisher(model, layout, x, sample_size=16000, seed=12,
                                 chunk_size=500, return_score_bank=True)
    assert torch.linalg.matrix_norm(estimate - exact) / torch.linalg.matrix_norm(exact) < .035
    assert diag["score_bank"]["labels"].unique().numel() == 10
    assert diag["evidence_ess"] == 2
    assert diag["score_count"] == 16000
    assert diag["seeds"]["inputs"] != diag["seeds"]["labels"]


def test_weighted_inputs_and_label_rng_are_deterministic_across_chunking():
    model, layout = linear_fixture()
    x = torch.tensor([[1., 2.], [-2., .5], [3., -1.]], dtype=torch.float64)
    weights = torch.tensor([0., 1., 0.], dtype=torch.float64)
    state = torch.random.get_rng_state().clone()
    first, d1 = model_fisher(model, layout, x, weights, sample_size=19, seed=42,
                             chunk_size=3, return_score_bank=True)
    second, d2 = model_fisher(model, layout, x, weights, sample_size=19, seed=42,
                              chunk_size=8, return_score_bank=True)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(first, second)
    assert torch.equal(d1["score_bank"]["labels"], d2["score_bank"]["labels"])
    torch.testing.assert_close(d1["score_bank"]["scores"], d2["score_bank"]["scores"])
    assert d1["input_indices"].tolist() == [1] * 19
    assert d1["evidence_ess"] == 1


def test_weighted_monte_carlo_targets_same_input_average_as_exact_integration():
    model, layout = linear_fixture()
    x = torch.tensor([[1., 2.], [-2., .5]], dtype=torch.float64)
    weights = torch.tensor([.2, .8], dtype=torch.float64)
    exact, _ = model_fisher(model, layout, x, weights, exact_labels=True)
    sampled, _ = model_fisher(model, layout, x, weights, sample_size=16000,
                              seed=110, chunk_size=500)
    assert torch.linalg.matrix_norm(sampled - exact) / torch.linalg.matrix_norm(exact) < .05


def test_callback_partial_sums_and_coordinate_restriction():
    model, layout = linear_fixture()
    x = torch.tensor([[1., 2.], [-2., .5], [3., -1.]], dtype=torch.float64)
    chunks = []
    full, _ = model_fisher(model, layout, x, exact_labels=True, chunk_size=2)
    indices = torch.tensor([20, 1, 0, 29])
    restricted, diagnostics = model_fisher(model, layout, x, exact_labels=True, chunk_size=2,
                                           indices=indices, chunk_callback=chunks.append)
    torch.testing.assert_close(restricted, full[indices[:, None], indices[None, :]])
    torch.testing.assert_close(restricted, sum(chunk["weighted_outer_sum"] for chunk in chunks))
    assert [(chunk["start"], chunk["stop"]) for chunk in chunks] == [(0, 2), (2, 3)]
    assert sum(chunk["weight_sum"] for chunk in chunks) == pytest.approx(1.)
    assert diagnostics["score_cost_scope"] == "full_network"
    assert diagnostics["computed_score_dimension"] == 30
    assert diagnostics["returned_score_dimension"] == 4
    assert "score_bank" not in diagnostics
    raw = sum(chunk["weighted_outer_sum"] for chunk in chunks)
    assert diagnostics["min_eigenvalue_before_psd"] == pytest.approx(
        float(torch.linalg.eigvalsh((raw + raw.mT) / 2).min())
    )
    assert diagnostics["psd_projection_frobenius"] >= 0


def test_cnn_full_fisher_and_suffix_are_the_same_likelihood_chart():
    model, layout = build_model(3, dtype=torch.float64)
    x = torch.linspace(-1., 1., 28*28, dtype=torch.float64).reshape(1, 1, 28, 28)
    full, _ = model_fisher(model, layout, x, exact_labels=True, chunk_size=1)
    indices, _ = active_indices(layout, 1)
    restricted, diag = model_fisher(model, layout, x, exact_labels=True, indices=indices)
    assert full.shape == (501, 501)
    assert full.dtype == torch.float64
    torch.testing.assert_close(restricted, full[indices[:, None], indices[None, :]], atol=1e-12, rtol=1e-10)
    assert diag["computed_score_dimension"] == 501
    assert torch.linalg.eigvalsh(full).min() > -1e-12


@pytest.mark.parametrize("kind,rank", [("dense", 8), ("diag", 8), ("lr", 0), ("lr", 2),
                                     ("lr", 20), ("layer_block", 8)])
def test_same_sample_representations_preserve_diagonal_trace_and_psd(kind, rank):
    rng = torch.Generator().manual_seed(9)
    scores = torch.randn(9, 6, generator=rng, dtype=torch.float64)
    matrix = scores.mT @ scores / 9
    result = represent(matrix, kind, rank, layer_indices={"first": [0, 1], "second": [2, 3, 4, 5]})
    assert result.dtype == torch.float64
    torch.testing.assert_close(result.diagonal(), matrix.diagonal())
    torch.testing.assert_close(result.trace(), matrix.trace())
    assert torch.linalg.eigvalsh(result).min() > -1e-12
    if kind == "lr" and rank >= matrix.shape[0]:
        torch.testing.assert_close(result, matrix)
    if kind == "lr" and rank == 0:
        torch.testing.assert_close(result, torch.diag(matrix.diagonal()))
    if kind == "layer_block":
        assert torch.count_nonzero(result[:2, 2:]) == 0
        torch.testing.assert_close(result[:2, :2], matrix[:2, :2])
    diagnostics = representation_diagnostics(matrix, result, rank=rank)
    assert diagnostics["diagonal_max_error"] < 1e-12
    assert diagnostics["residual_diagonal_clamp_mass"] < 1e-12


def test_invalid_representation_and_partition_are_rejected():
    with pytest.raises(ValueError, match="PSD"):
        represent(torch.diag(torch.tensor([1., -1.])))
    with pytest.raises(ValueError, match="partition"):
        represent(torch.eye(3), "layer_block", layer_indices=[[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="partition"):
        represent(torch.eye(3), "layer_block", layer_indices=[[0, 1]])


def test_invalid_fisher_weights_fail_before_sampling():
    model = BiasModel()
    with pytest.raises(ValueError, match="weights"):
        model_fisher(model, ParameterLayout.from_module(model), torch.ones(2, 1, dtype=torch.float64),
                     torch.zeros(2))


@pytest.mark.parametrize("exact_labels", [False, True])
def test_interrupted_reference_resumes_raw_partial_sum_without_rescoring(exact_labels):
    model, layout = linear_fixture()
    x = torch.tensor([[1., 2.], [-2., .5], [3., -1.], [.1, .3], [2., 1.]], dtype=torch.float64)
    a = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
    kwargs = {"sample_size": 11, "seed": 24, "exact_labels": exact_labels}
    expected, expected_diag = model_fisher(model, layout, x, a, **kwargs, chunk_size=2,
                                          return_score_bank=True)
    sealed = []

    class Interrupted(Exception):
        pass

    def seal_then_interrupt(chunk):
        sealed.append(chunk)
        raise Interrupted

    with pytest.raises(Interrupted):
        model_fisher(model, layout, x, a, **kwargs, chunk_size=2,
                     chunk_callback=seal_then_interrupt)
    state = sealed[0]["resume_state"]
    saved_sum = state["weighted_outer_sum"].clone()
    assert state["next_input"] == 2
    remaining = []
    actual, diagnostics = model_fisher(model, layout, x, a, **kwargs, chunk_size=3,
                                       resume_state=state, chunk_callback=remaining.append)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state["weighted_outer_sum"], saved_sum)
    assert remaining[0]["start"] == 2
    assert diagnostics["resumed_input_draw_count"] == 2
    count = 5 if exact_labels else 11
    assert diagnostics["computed_input_draw_count"] == count - 2
    assert diagnostics["computed_score_count"] == (count - 2) * (10 if exact_labels else 1)
    assert diagnostics["score_count"] == expected_diag["score_count"]
    chunks = sealed + remaining
    torch.testing.assert_close(torch.cat([part["scores"] for part in chunks]),
                               expected_diag["score_bank"]["scores"])
    assert torch.equal(torch.cat([part["labels"] for part in chunks]),
                       expected_diag["score_bank"]["labels"])
    torch.testing.assert_close(sum(part["weighted_outer_sum"] for part in chunks), expected)
    called = []
    completed, completed_diag = model_fisher(model, layout, x, a, **kwargs,
                                             resume_state=remaining[-1]["resume_state"],
                                             chunk_callback=called.append)
    torch.testing.assert_close(completed, expected)
    assert completed_diag["computed_score_count"] == 0
    assert called == []


@pytest.mark.parametrize("change", ["model", "inputs", "weights", "seed", "count", "indices", "labels"])
def test_resume_rejects_changed_sampling_contract(change):
    model, layout = linear_fixture()
    x = torch.ones(3, 2, dtype=torch.float64)
    a = torch.ones(3, dtype=torch.float64)
    chunks = []
    model_fisher(model, layout, x, a, sample_size=7, seed=12, chunk_size=2,
                 chunk_callback=chunks.append)
    kwargs = {"sample_size": 7, "seed": 12}
    if change == "model":
        with torch.no_grad():
            model.bias.add_(.1)
    elif change == "inputs":
        x[0, 0] += .1
    elif change == "weights":
        a[0] += .1
    elif change == "seed":
        kwargs["seed"] = 13
    elif change == "count":
        kwargs["sample_size"] = 8
    elif change == "indices":
        kwargs["indices"] = [0, 1]
    else:
        kwargs["exact_labels"] = True
    with pytest.raises(ValueError, match="contract"):
        model_fisher(model, layout, x, a, **kwargs, resume_state=chunks[0]["resume_state"])


def test_resume_rejects_incomplete_or_incompatible_state_and_partial_score_banks():
    model, layout = linear_fixture()
    x = torch.ones(3, 2, dtype=torch.float64)
    chunks = []
    model_fisher(model, layout, x, sample_size=5, chunk_size=2, chunk_callback=chunks.append)
    state = chunks[0]["resume_state"]
    with pytest.raises(ValueError, match="schema"):
        model_fisher(model, layout, x, sample_size=5, resume_state={})
    with pytest.raises(ValueError, match="float64"):
        model_fisher(model, layout, x, sample_size=5,
                     resume_state={**state, "weighted_outer_sum": torch.zeros(30, 30)})
    with pytest.raises(ValueError, match="next_input"):
        model_fisher(model, layout, x, sample_size=5, resume_state={**state, "next_input": 6})
    with pytest.raises(ValueError, match="sealed callback"):
        model_fisher(model, layout, x, sample_size=5, resume_state=state, return_score_bank=True)
