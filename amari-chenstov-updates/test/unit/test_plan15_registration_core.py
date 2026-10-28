"""Small CPU contracts for Phase 10 physical-pixel Gaussian registration."""

import copy
import dataclasses
import io
import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.func import jacrev

from mnist_experiment.fisher_memory_registration import learner
from mnist_experiment.fisher_memory_registration.curvature import (
    SIGMA2, fd_values, gaussian_nll, jvp_values, reference_fisher,
)
from mnist_experiment.fisher_memory_registration.model import (
    GAUSSIAN_SCALE_PIXELS, build_model, flat_prediction, memory_indices,
)
from src.parameters import ParameterLayout
from src.representations import representation_from_artifact
from src.seeding import derive_component_seed


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def case(output_dim=6, *, moved=True, dtype=torch.float64):
    model, layout = build_model(17, output_dim=output_dim, dtype=dtype, factor_seed=57)
    theta = layout.flatten_module(model, detach=True)
    generator = torch.Generator().manual_seed(91)
    if moved:
        for spec in layout.specs:
            if spec.name.endswith("lora_B"):
                theta[spec.start:spec.stop] = .15 * torch.randn(
                    spec.numel, generator=generator, dtype=dtype)
    x = torch.randn(3, 2, 28, 28, generator=generator, dtype=dtype)
    ltm, stm = memory_indices(layout)
    probes = (2 * torch.randint(2, (2, len(stm)), generator=generator) - 1).to(dtype)
    return model, layout, theta.requires_grad_(), x, ltm, stm, probes


@pytest.fixture(params=(1, 6))
def fit_inputs(request):
    model, layout, theta, x, ltm, _, _ = case(request.param)
    y = torch.linspace(-2., 3., 3 * request.param, dtype=x.dtype).reshape(3, request.param)
    return dict(model=model, layout=layout, theta=theta.detach(), new_x=x[:1], new_y=y[:1],
                replay_x=x[1:], replay_y=y[1:], ages=torch.tensor([.25, 1.], dtype=x.dtype),
                fisher=.03 * torch.eye(len(ltm), dtype=x.dtype),
                anchor=theta[ltm].detach().clone(), seed=321)


def small_state(capacity=2, rank=2):
    return learner.initial_state(torch.tensor([1., 2., 3.], dtype=torch.float64),
        torch.tensor([0, 2]), torch.tensor([[2., .1], [.1, 1.]], dtype=torch.float64),
        capacity=capacity, rank=rank, seed=7)


def dense(artifact):
    return representation_from_artifact(artifact).to_dense()


@pytest.mark.parametrize("output_dim,channels,counts", [
    (1, (4, 6), (323, 153, 476)), (6, (4, 6), (448, 188, 636)),
    (1, (6, 8), (587, 205, 792)), (6, (6, 8), (752, 248, 1000)),
])
def test_model_layout_rng_pairing_and_physical_output(output_dim, channels, counts):
    rng = torch.get_rng_state().clone()
    model, layout = build_model(11, output_dim=output_dim, channels=channels,
                                factor_seed=42, dtype=torch.float64)
    twin, _ = build_model(11, output_dim=output_dim, channels=channels,
                          factor_seed=42, dtype=torch.float64)
    other, _ = build_model(11, output_dim=output_dim, channels=channels,
                           factor_seed=43, dtype=torch.float64)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.output_dim == output_dim and model.channels == channels
    ltm, stm = memory_indices(layout)
    assert (len(ltm), len(stm), layout.total_numel) == counts
    assert torch.equal(torch.cat((ltm, stm)).sort().values, torch.arange(counts[-1]))
    assert model.features[0].rank == model.features[3].rank == 2
    assert model.classifier.rank == (1 if output_dim == 1 else 2)
    theta = layout.flatten_module(model)
    assert torch.equal(theta, layout.flatten_module(twin))
    other_theta = layout.flatten_module(other)
    assert torch.equal(theta[ltm], other_theta[ltm])
    assert not torch.equal(theta[stm], other_theta[stm])
    x = torch.ones(2, 2, 28, 28, dtype=torch.float64)
    prediction = flat_prediction(model, layout, theta, x)
    assert prediction.shape == (2, output_dim)
    torch.testing.assert_close(prediction, other(x), rtol=0, atol=0)
    torch.testing.assert_close(prediction, model(x, stm_scale=0), rtol=0, atol=0)
    changed = theta.detach().clone()
    bias = next(spec for spec in layout.specs if spec.name == "classifier.bias")
    changed[bias.start:bias.stop] += 100
    torch.testing.assert_close(flat_prediction(model, layout, changed, x) - prediction,
                               torch.full_like(prediction, 100))
    assert all(value.requires_grad for value in model.parameters())
    with pytest.raises(ValueError, match="2, 28, 28"):
        model(x[:, :1])


def test_registration_chart_rejects_legacy_shapes_and_damaged_offsets():
    from mnist_experiment.fisher_memory_regression.model import build_model as scalar_model

    with pytest.raises(ValueError, match="registration"):
        memory_indices(scalar_model(1)[1])
    _, layout = build_model(1)
    broken = dataclasses.replace(layout.specs[0], stop=layout.specs[0].stop + 1)
    with pytest.raises(ValueError, match="offsets"):
        memory_indices(dataclasses.replace(layout, specs=(broken,) + layout.specs[1:]))
    with pytest.raises(ValueError, match="registration"):
        memory_indices(dataclasses.replace(layout, specs=tuple(reversed(layout.specs))))
    with pytest.raises(ValueError, match="count"):
        memory_indices(dataclasses.replace(layout, total_numel=layout.total_numel + 1))
    for kwargs in ({"output_dim": 2}, {"output_dim": True}, {"channels": (4, 0)},
                   {"channels": (4,)}, {"seed": True}, {"factor_seed": -1},
                   {"dtype": torch.float16}):
        with pytest.raises(ValueError):
            build_model(**({"seed": 1} | kwargs))


@pytest.mark.parametrize("output_dim", (1, 6))
def test_initial_adapter_gradients_and_ltm_only_pretraining(output_dim):
    model, layout, theta, x, ltm, stm, _ = case(output_dim, moved=False)
    labels = torch.full((len(x), output_dim), 2., dtype=x.dtype)
    gradient = torch.autograd.grad(gaussian_nll(flat_prediction(model, layout, theta, x), labels).mean(), theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    for spec in layout.specs:
        if spec.name.endswith("lora_A"):
            assert theta[spec.start:spec.stop].count_nonzero() == spec.numel
            assert gradient[spec.start:spec.stop].count_nonzero() == 0
        elif spec.name.endswith("lora_B"):
            assert theta[spec.start:spec.stop].count_nonzero() == 0
            assert gradient[spec.start:spec.stop].norm() > 0
    loss = gaussian_nll(flat_prediction(model, layout, theta, x, stm_scale=0), labels).mean()
    gradient = torch.autograd.grad(loss, theta)[0]
    assert gradient[stm].count_nonzero() == 0 and gradient[ltm].norm() > 0


@pytest.mark.parametrize("output_dim", (1, 6))
def test_gaussian_nll_sums_coordinates_negative_score_and_pixel_normalization(output_dim):
    model, layout, theta, x, _, _, _ = case(output_dim)
    y = torch.linspace(-2., 4., len(x) * output_dim, dtype=x.dtype).reshape(len(x), output_dim)
    prediction = flat_prediction(model, layout, theta, x)
    normal = Normal(prediction, 2.)
    nll = gaussian_nll(prediction, y)
    assert nll.shape == (len(x),) and nll.dtype == torch.float64
    torch.testing.assert_close(nll, -normal.log_prob(y).sum(-1))
    normalized = -Normal(prediction / 2, 1).log_prob(y / 2).sum(-1)
    torch.testing.assert_close(nll, normalized + output_dim * math.log(2))
    gradient = torch.autograd.grad(nll.sum(), theta, retain_graph=True)[0]
    score = torch.autograd.grad(normal.log_prob(y).sum(), theta, retain_graph=True)[0]
    torch.testing.assert_close(gradient, -score)
    torch.testing.assert_close(gradient, torch.autograd.grad(normalized.sum(), theta)[0])
    custom = gaussian_nll(prediction.detach(), y, scale=3.)
    torch.testing.assert_close(custom, -Normal(prediction.detach(), 3.).log_prob(y).sum(-1))
    with pytest.raises(ValueError, match="matching"):
        gaussian_nll(prediction[:, 0], y[:, 0])
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(prediction, y, scale=0)


@pytest.mark.parametrize("output_dim", (1, 6))
def test_fisher_is_sum_of_per_input_output_jacobians_not_residual_outer_products(output_dim):
    model, layout, theta, x, ltm, _, _ = case(output_dim)
    jacobian = jacrev(lambda p: flat_prediction(model, layout, p, x))(theta)
    rows = jacobian.reshape(-1, len(theta))
    expected = rows.T @ rows / (len(x) * SIGMA2)
    fisher = reference_fisher(model, layout, theta, x, chunk_size=2)
    assert fisher.dtype == torch.float64 and not fisher.requires_grad
    torch.testing.assert_close(fisher, expected, rtol=1e-12, atol=1e-14)
    block = reference_fisher(model, layout, theta, x, indices=ltm, chunk_size=1)
    torch.testing.assert_close(block, fisher[ltm][:, ltm], rtol=1e-12, atol=1e-14)
    wrong = jacobian.mean(0).T @ jacobian.mean(0) / SIGMA2
    assert not torch.allclose(fisher, wrong, atol=1e-12)
    y = flat_prediction(model, layout, theta, x).detach()
    observed = jacrev(lambda p: gaussian_nll(flat_prediction(model, layout, p, x), y))(theta)
    assert observed.count_nonzero() == 0 and fisher.norm() > 0
    for sample in jacobian:
        assert torch.linalg.matrix_rank(sample.T @ sample / SIGMA2) <= output_dim


class LinearPixels(nn.Module):
    def __init__(self, output_dim=6, duplicate=False):
        super().__init__()
        self.output_dim, self.duplicate = output_dim, duplicate
        width = 1 if duplicate else output_dim
        self.weight = nn.Parameter(torch.ones(width, 2, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(width, dtype=torch.float64))

    def forward(self, x, stm_scale=1.):
        means = x[:, :, 0, 0] @ self.weight.T + self.bias
        return means.expand(-1, self.output_dim) if self.duplicate else means


def test_linear_analytic_fisher_model_label_integration_and_monte_carlo():
    model = LinearPixels()
    layout = ParameterLayout.from_module(model)
    theta = layout.flatten_module(model, detach=True)
    x = torch.zeros(2, 2, 28, 28, dtype=theta.dtype)
    features = torch.tensor([[1., 2.], [-2., .5]], dtype=theta.dtype)
    x[:, :, 0, 0] = features
    jacobian = torch.zeros(2, 6, 18, dtype=theta.dtype)
    for coordinate in range(6):
        jacobian[:, coordinate, 2 * coordinate:2 * coordinate + 2] = features
        jacobian[:, coordinate, 12 + coordinate] = 1
    expected = torch.einsum("ndp,ndq->pq", jacobian, jacobian) / (len(x) * SIGMA2)
    fisher = reference_fisher(model, layout, theta, x, chunk_size=1)
    torch.testing.assert_close(fisher, expected, rtol=0, atol=0)
    means = flat_prediction(model, layout, theta, x).detach()
    # Symmetric coordinate-axis targets integrate the Gaussian's first two
    # moments exactly, including zero cross-coordinate residual covariance.
    integrated = torch.zeros_like(fisher)
    for direction in torch.eye(6, dtype=theta.dtype):
        for sign in (-1, 1):
            y = means + sign * 2 * math.sqrt(6) * direction
            gradients = jacrev(lambda p: gaussian_nll(flat_prediction(model, layout, p, x), y))(theta)
            integrated += gradients.T @ gradients / (12 * len(x))
    torch.testing.assert_close(integrated, fisher)
    residuals = 2 * torch.randn(16_384, 2, 6, dtype=theta.dtype,
                                generator=torch.Generator().manual_seed(55))
    scores = torch.einsum("ndp,mnd->mnp", jacobian, residuals) / SIGMA2
    sampled = torch.einsum("mnp,mnq->pq", scores, scores) / (len(residuals) * len(x))
    assert (sampled - fisher).norm() / fisher.norm() < .04


def test_duplicated_outputs_increase_precision_without_creating_new_directions():
    x = torch.zeros(3, 2, 28, 28, dtype=torch.float64)
    x[:, :, 0, 0] = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    matrices = []
    for output_dim in (1, 6):
        model = LinearPixels(output_dim, duplicate=True)
        layout = ParameterLayout.from_module(model)
        matrices.append(reference_fisher(model, layout, layout.flatten_module(model), x))
    torch.testing.assert_close(matrices[1], 6 * matrices[0])
    assert torch.linalg.matrix_rank(matrices[0]) == torch.linalg.matrix_rank(matrices[1]) == 3


@pytest.mark.parametrize("output_dim", (1, 6))
def test_fd_matches_symmetric_pixel_kl_jvp_and_both_block_gradients(output_dim):
    model, layout, theta, x, ltm, stm, probes = case(output_dim)
    step = 1e-4
    actual = fd_values(model, layout, theta, x, probes, step)
    exact = jvp_values(model, layout, theta, x, probes)
    assert actual.shape == (2, len(x)) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, exact, rtol=1e-6, atol=1e-10)
    embedded = theta.new_zeros(len(probes), len(theta))
    embedded[:, stm] = probes
    # Normal's general KL formula subtracts nearly equal unit-scale terms.
    # Use a wider stencil for that audit; keep the tighter FD/JVP test above.
    kl_step = .01
    kl_values = fd_values(model, layout, theta, x, probes, kl_step)
    for k, direction in enumerate(embedded):
        plus = Normal(flat_prediction(model, layout, theta + kl_step / 2 * direction, x), 2)
        minus = Normal(flat_prediction(model, layout, theta - kl_step / 2 * direction, x), 2)
        expected = (kl_divergence(plus, minus) + kl_divergence(minus, plus)).sum(-1) / kl_step**2
        torch.testing.assert_close(kl_values[k], expected, rtol=1e-6, atol=1e-10)
        normalized_plus, normalized_minus = Normal(plus.loc / 2, 1), Normal(minus.loc / 2, 1)
        normalized = (kl_divergence(normalized_plus, normalized_minus)
                      + kl_divergence(normalized_minus, normalized_plus)).sum(-1) / kl_step**2
        torch.testing.assert_close(kl_values[k], normalized, rtol=1e-6, atol=1e-10)
    grad_fd = torch.autograd.grad(actual.sum(), theta)[0]
    grad_exact = torch.autograd.grad(exact.sum(), theta)[0]
    torch.testing.assert_close(grad_fd, grad_exact, rtol=2e-5, atol=1e-8)
    assert grad_fd[ltm].norm() > 0 and grad_fd[stm].norm() > 0
    torch.testing.assert_close(fd_values(model, layout, theta, x, embedded, step),
                               fd_values(model, layout, theta, x, probes, step))
    embedded[:, ltm[0]] = 1
    with pytest.raises(ValueError, match="zero in LTM"):
        fd_values(model, layout, theta, x, embedded, step)


def test_fd_branches_pass_gradcheck_and_probes_are_detached():
    model, layout, theta, x, _, _, probes = case()
    probes.requires_grad_()
    selected = torch.tensor([spec.start for spec in layout.specs[::3]])

    def function(values):
        parameter = theta.detach().index_copy(0, selected, values)
        return fd_values(model, layout, parameter, x[:1], probes, .01)

    assert torch.autograd.gradcheck(function, (theta.detach()[selected].requires_grad_(),),
                                    eps=1e-5, atol=1e-6, rtol=1e-4)
    value = fd_values(model, layout, theta, x, probes, .01).sum()
    assert torch.autograd.grad(value, probes, allow_unused=True)[0] is None


def test_curvature_input_checks_and_cancellation():
    model, layout, theta, x, _, _, probes = case()
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(learner.StopRequested):
        reference_fisher(model, layout, theta, x, chunk_size=1, check_stop=stop)
    assert calls == 3
    for indices in (torch.tensor([1, 1]), torch.tensor([-1]), torch.tensor([1.])):
        with pytest.raises(ValueError, match="unique integer"):
            reference_fisher(model, layout, theta, x, indices=indices)
    with pytest.raises(ValueError, match="positive integer"):
        reference_fisher(model, layout, theta, x, chunk_size=True)
    with pytest.raises(ValueError, match="2, 28, 28"):
        jvp_values(model, layout, theta, x[:, :1], probes)
    with pytest.raises(ValueError, match="positive"):
        fd_values(model, layout, theta, x, probes, 0)
    with pytest.raises(ValueError, match="coordinates"):
        fd_values(model, layout, theta, x, probes[:, :1], .01)


def test_fifo_once_nominal_four_clock_replay_independence_and_roundtrip():
    state = small_state(capacity=4)
    initial = copy.deepcopy(state)
    live, stats = learner.commit_arrival(state, state["theta"] + .1,
                                         [0, 1, 2, 3], [10, 11, 12, 13], pi=.025)
    assert stats["eta"] == 0 and live["archive_updates"] == 0
    assert_same(live["fisher_L"], state["fisher_L"])
    assert_same(live["anchor_L"], state["anchor_L"])
    before, rng = copy.deepcopy(live), torch.get_rng_state().clone()
    selected = learner.replay_indices(live, 1, 71)
    all_rows = learner.replay_indices(live, 4, 71)
    assert torch.equal(selected, all_rows[:1]) and all_rows.unique().numel() == 4
    assert torch.equal(rng, torch.get_rng_state())
    staged = learner.stage_arrival(live, [4, 5], [14, 15])
    assert [r["exposure_id"] for r in staged["evicted"]] == [0, 1]
    torch.testing.assert_close(staged["ages"], torch.full((4,), .25, dtype=torch.float64))
    accepted, fresh = live["theta"] + .2, torch.eye(2, dtype=torch.float64)
    result, stats = learner.commit_arrival(live, accepted, [4, 5], [14, 15], fresh, pi=.025, rank=2)
    eta = 1 - .975**.5
    assert stats["eta"] == pytest.approx(eta)
    assert result["consolidated_ids"] == [0, 1] and result["archive_updates"] == 1
    assert [r["exposure_id"] for r in result["replay"]] == [2, 3, 4, 5]
    torch.testing.assert_close(dense(result["fisher_L"]),
                               (1 - eta) * dense(live["fisher_L"]) + eta * fresh)
    torch.testing.assert_close(result["anchor_L"], accepted[[0, 2]])
    assert_same(state, initial)
    assert_same(live, before)
    with pytest.raises(ValueError, match="already"):
        learner.stage_arrival(result, [0], [10])
    stream = io.BytesIO()
    torch.save(result, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    learner.validate_state(restored)
    assert_same(result, restored)


def test_zero_capacity_immediate_retirement_and_pixel_schema_rejects_angle_state():
    from mnist_experiment.fisher_memory_regression import learner as angle_learner

    state = small_state(capacity=0)
    staged = learner.stage_arrival(state, ["a"], [12])
    assert not staged["replay"] and len(staged["evicted"]) == 1
    assert staged["ages"].numel() == learner.replay_indices(state, 0, 13).numel() == 0
    result, stats = learner.commit_arrival(state, state["theta"] + .1, ["a"], [12],
                                          torch.eye(2), pi=.025, rank=2)
    assert stats["eta"] == pytest.approx(1 - .975**.25)
    assert result["replay"] == [] and result["consolidated_ids"] == ["a"]
    assert result["likelihood"] == "gaussian_fixed_scale_2_pixels"
    with pytest.raises(ValueError, match="nominal value 4"):
        learner.commit_arrival(state, state["theta"], ["a"], [12],
                               torch.eye(2), pi=.025, arrival_size=1)
    with pytest.raises(ValueError, match="schema or likelihood"):
        angle_learner.validate_state(result)
    legacy = dict(state, likelihood=angle_learner.LIKELIHOOD)
    with pytest.raises(ValueError, match="schema or likelihood"):
        learner.validate_state(legacy)


@pytest.mark.parametrize("rank", (0, 8, 16, 32))
def test_custom_archive_rank_compression_and_commits_preserve_diagonal_trace(rank):
    generator = torch.Generator().manual_seed(51)
    factor = torch.randn(36, 36, generator=generator, dtype=torch.float64)
    source = factor @ factor.T
    archive, info = learner.compress_fisher(source, rank=rank, seed=19)
    assert info["compression"]["requested_rank"] == rank
    assert representation_from_artifact(archive).rank == rank
    torch.testing.assert_close(dense(archive).diagonal(), source.diagonal())
    assert info["represented_trace"] == pytest.approx(float(source.trace()))
    state = learner.initial_state(torch.zeros(37, dtype=torch.float64), torch.arange(36),
                                  source, capacity=0, rank=rank)
    result, _ = learner.commit_arrival(state, state["theta"], [0], [1], source, pi=.025, rank=rank)
    assert result["archive_diagnostics"]["compression"]["requested_rank"] == rank
    assert representation_from_artifact(result["fisher_L"]).rank == rank
    torch.testing.assert_close(dense(result["fisher_L"]).diagonal(), source.diagonal())
    with pytest.raises(ValueError, match="indefinite"):
        learner.compress_fisher(torch.tensor([[1., 2.], [2., 1.]]), rank=rank)


def test_invalid_and_interrupted_commits_leave_state_unchanged():
    state = small_state(capacity=1)
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="eviction requires"):
        learner.commit_arrival(state, state["theta"], [0, 1], [10, 11], pi=.1)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(learner.StopRequested):
        learner.commit_arrival(state, state["theta"], [0, 1], [10, 11],
                               torch.eye(2), pi=.1, check_stop=stop)
    assert_same(state, original)
    with pytest.raises(ValueError, match="inconsistent"):
        learner.validate_state(dict(state, consolidated_ids=[0]))


def test_objective_normalizes_observations_probes_and_stm_not_coordinates(fit_inputs):
    args = fit_inputs
    cfg = learner.FitSettings(lambda_f=2, num_probes=2, fd_step=.01, chunk_size=1)
    objective, metadata = learner.prepare_objective(**args, condition="fd_replay_ewc", settings=cfg)
    total, parts = objective(args["theta"])
    new = gaussian_nll(flat_prediction(args["model"], args["layout"], args["theta"], args["new_x"]),
                       args["new_y"]).mean()
    replay = gaussian_nll(flat_prediction(args["model"], args["layout"], args["theta"], args["replay_x"]),
                          args["replay_y"]).mean()
    torch.testing.assert_close(parts["data_loss"], (new + replay) / 2)
    _, stm = memory_indices(args["layout"])
    probe_seed = derive_component_seed(args["seed"], learner.PROBE_SEED_COMPONENT)
    assert metadata["probe_seed"] == probe_seed
    assert metadata["sigma2"] == SIGMA2 == 4
    assert metadata["gaussian_scale_pixels"] == GAUSSIAN_SCALE_PIXELS == 2
    probes = (2 * torch.randint(2, (2, len(stm)), generator=torch.Generator().manual_seed(probe_seed)) - 1).double()
    values = fd_values(args["model"], args["layout"], args["theta"], args["replay_x"], probes, .01)
    expected = (args["ages"] * values.mean(0) / len(stm)).mean()
    torch.testing.assert_close(parts["auxiliary"], expected)
    torch.testing.assert_close(total, parts["data_loss"] + parts["ewc"] + parts["ridge"] + 2 * expected)
    duplicated = {**args, "replay_x": args["replay_x"].repeat(2, 1, 1, 1),
                  "replay_y": args["replay_y"].repeat(2, 1), "ages": args["ages"].repeat(2)}
    larger, _ = learner.prepare_objective(**duplicated, condition="fd_replay_ewc", settings=cfg)
    torch.testing.assert_close(larger(args["theta"])[0], total)
    control, _ = learner.prepare_objective(**args, condition="replay_ewc", settings=cfg)
    assert control(args["theta"])[1]["weighted_auxiliary"] == 0
    empty = {**args, "replay_x": args["replay_x"][:0], "replay_y": args["replay_y"][:0],
             "ages": args["ages"][:0]}
    boundary, _ = learner.prepare_objective(**empty, condition="ewc_only", settings=cfg)
    torch.testing.assert_close(boundary(args["theta"])[1]["data_loss"], new)
    with pytest.raises(ValueError, match="cannot receive replay"):
        learner.prepare_objective(**args, condition="ewc_only", settings=cfg)


def test_objective_freezes_all_inputs_and_replay_fd_gradients_reach_both_blocks(fit_inputs):
    args = fit_inputs
    rng = torch.get_rng_state().clone()
    objective, _ = learner.prepare_objective(**args, condition="fd_replay_ewc", settings={
        "lambda_f": 3., "num_probes": 2, "fd_step": .01})
    candidate = args["theta"].clone().requires_grad_()
    original, parts = objective(candidate)
    ltm, stm = memory_indices(args["layout"])
    for name in ("new_nll", "replay_nll", "auxiliary"):
        gradient = torch.autograd.grad(parts[name], candidate, retain_graph=True)[0]
        assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    gradient = torch.autograd.grad(original, candidate)[0]
    for key in ("theta", "new_x", "new_y", "replay_x", "replay_y", "ages", "fisher", "anchor"):
        args[key].add_(1)
    again = objective(candidate)[0]
    torch.testing.assert_close(original, again, rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.autograd.grad(again, candidate)[0], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_objective_validates_pair_target_and_age_shapes(fit_inputs):
    args = fit_inputs
    for replacements in ({"new_y": args["new_y"].squeeze(0)},
                         {"replay_x": args["replay_x"][:, :1]},
                         {"ages": args["ages"][:, None]},
                         {"new_y": torch.full_like(args["new_y"], float("nan"))}):
        with pytest.raises(ValueError):
            learner.prepare_objective(**(args | replacements), condition="fd_replay_ewc", settings={})


@pytest.mark.parametrize("condition", sorted(learner.CONDITIONS))
def test_bounded_fit_deterministic_observer_and_no_mutation(fit_inputs, condition):
    args = fit_inputs
    if condition == "ewc_only":
        args = {**args, "replay_x": args["replay_x"][:0], "replay_y": args["replay_y"][:0],
                "ages": args["ages"][:0]}
    before = {key: value.detach().clone() for key, value in args.items() if isinstance(value, torch.Tensor)}
    model_before, rng, observed = copy.deepcopy(args["model"].state_dict()), torch.get_rng_state().clone(), []

    def observer(accepted, objective, metadata, diagnostics):
        observed.append(float(objective(accepted)[0]))
        assert observed[-1] == pytest.approx(diagnostics["final_objective"])
        assert metadata["output_dim"] == args["model"].output_dim
        accepted.add_(20)

    cfg = learner.FitSettings(lambda_f=.2, num_probes=1, max_iter=2, max_eval=5, fd_step=.01)
    accepted, info = learner.fit_step(**args, condition=condition, settings=cfg, objective_observer=observer)
    twin, other = learner.fit_step(**args, condition=condition, settings=cfg)
    torch.testing.assert_close(accepted, twin, rtol=0, atol=0)
    assert len(observed) == 1 and info["final_objective"] == other["final_objective"]
    assert not accepted.requires_grad and not info["solver_failed"]
    assert info["closure_calls"] <= 5 and info["objective_evaluations"] <= 5
    assert info["final_objective"] <= info["initial_objective"]
    assert info["new_presentations"] == info["objective_evaluations"] * len(args["new_y"])
    assert info["replay_presentations"] == info["objective_evaluations"] * len(args["replay_y"])
    expected_fd = 2 * cfg.num_probes * info["replay_presentations"] if condition == "fd_replay_ewc" else 0
    assert info["fd_branch_presentations"] == expected_fd
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert_same(value, args[key])
    assert_same(model_before, args["model"].state_dict())


def test_interrupted_fit_and_closure_limit_do_not_accept_partial_steps(fit_inputs):
    original = fit_inputs["theta"].clone()
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 7

    cfg = learner.FitSettings(lambda_f=1, num_probes=1, max_iter=3, max_eval=5)
    with pytest.raises(learner.StopRequested):
        learner.fit_step(**fit_inputs, condition="fd_replay_ewc", settings=cfg, check_stop=stop)
    assert_same(original, fit_inputs["theta"])
    accepted, info = learner.fit_step(**fit_inputs, condition="replay_ewc",
                                      settings=dataclasses.replace(cfg, max_eval=1))
    assert_same(original, accepted)
    assert info["solver_reason"] == "max_evaluations"
