"""Small CPU contracts for Phase 12 charts and inherited Gaussian learning."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.func import functional_call

from mnist_experiment.fisher_memory.learner import FitSettings, StopRequested
from mnist_experiment.fisher_memory_grid_registration import (
    curvature as grid_curvature, learner as grid_learner, model as grid_model,
    numerical as grid_numerical,
)
from mnist_experiment.fisher_memory_grid_registration.runner import Runner
from mnist_experiment.fisher_memory_stm_capacity import core, model
from src.parameters import ParameterLayoutError
from src.seeding import derive_component_seed


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def archive():
    # Exercise the real dense objective without an expensive Fisher estimate.
    return torch.diag(torch.linspace(.5, 1.5, 3268, dtype=torch.float64))


def case(name="all_r2", *, moved=True):
    net, layout = model.build_model(17, factor_seed=57, dtype=torch.float64, stm_config=name)
    theta = layout.flatten_module(net, detach=True)
    generator = torch.Generator().manual_seed(91)
    if moved:
        for spec in layout.specs:
            if spec.name.endswith("lora_B"):
                theta[spec.start:spec.stop] = .1 * torch.randn(
                    spec.numel, generator=generator, dtype=theta.dtype)
    x = torch.randn(2, 2, 84, 84, generator=generator, dtype=theta.dtype)
    ltm, stm = model.memory_indices(layout)
    probes = (2 * torch.randint(2, (2, len(stm)), generator=generator) - 1).double()
    return net, layout, theta.requires_grad_(), x, ltm, stm, probes


def objective_args(net, layout, theta, x, ltm, archive, settings=None):
    y = torch.linspace(-1., 2., 108, dtype=theta.dtype).reshape(2, 54)
    cfg = settings or FitSettings(lambda_f=1.7, num_probes=2, fd_step=.01,
                                 ridge_ratio=.3, chunk_size=1)
    return (net, layout, theta, x[:1], y[:1], x[1:], y[1:],
            theta.new_tensor([.25]), archive, theta[ltm].detach().clone(),
            "fd_replay_ewc", cfg, 111)


def aliases():
    targets = [(grid_model, "build_model"), (grid_model, "memory_indices"),
               (grid_model, "flat_prediction")]
    targets.extend((module, name) for module in (grid_curvature, grid_learner, grid_numerical)
                   for name in ("memory_indices", "flat_prediction"))
    return {(module, name): getattr(module, name) for module, name in targets}


@pytest.mark.parametrize("name,counts", [("head_r1", (3268, 108)), ("all_r1", (3268, 172)),
                                         ("all_r2", (3268, 344)), ("all_r4", (3268, 688))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_counts_active_layers_prediction_and_rng(name, counts, dtype):
    before = torch.random.get_rng_state().clone()
    with model.use_chart(name):
        net, layout = model.build_model(11, factor_seed=42, dtype=dtype)
    twin, twin_layout = model.build_model(11, factor_seed=42, dtype=dtype, stm_config=name)
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.CONFIGURATIONS == ("head_r1", "all_r1", "all_r2", "all_r4")
    assert model.COUNTS[name] == counts
    assert net.stm_config == name
    ltm, stm = model.memory_indices(layout)
    assert ltm.dtype == stm.dtype == torch.int64 and ltm.device.type == "cpu"
    assert (len(ltm), len(stm), layout.total_numel) == (*counts, sum(counts))
    assert torch.equal(torch.cat((ltm, stm)).sort().values, torch.arange(sum(counts)))
    assert layout.metadata() == twin_layout.metadata()
    assert torch.equal(layout.flatten_module(net), twin_layout.flatten_module(twin))
    assert all(parameter.requires_grad for parameter in net.parameters())
    assert net.classifier.weight.shape == (54, 54)
    for prefix, rank in zip(model.AFFINE_NAMES, model.RANKS[name]):
        layer = net.get_submodule(prefix)
        if rank:
            assert layer.rank == rank and layer.lora_alpha / layer.rank == 1
            assert layer.lora_B.count_nonzero() == 0
            assert layer.lora_A.count_nonzero() == layer.lora_A.numel()
        else:
            assert type(layer) is nn.Conv2d
            assert not hasattr(layer, "lora_A") and not hasattr(layer, "lora_B")
    x = torch.ones(1, 2, 84, 84, dtype=dtype)
    theta = layout.flatten_module(net)
    prediction = model.flat_prediction(net, layout, theta, x)
    assert prediction.shape == (1, 54)
    torch.testing.assert_close(prediction, net(x, stm_scale=0), rtol=0, atol=0)
    with pytest.raises(ValueError, match="84, 84"):
        net(x[:, :, :28, :28])


@pytest.mark.parametrize("factor_seed", [None, 57])
def test_coupled_factor_rows_metadata_and_bitwise_legacy_initialization(factor_seed):
    legacy, legacy_layout = grid_model.build_model(13, factor_seed=factor_seed, dtype=torch.float64)
    reference = dict(legacy.named_parameters())
    charts = {}
    for name in model.CONFIGURATIONS:
        net, layout = model.build_model(13, factor_seed=factor_seed, dtype=torch.float64, stm_config=name)
        charts[name] = net
        values = dict(net.named_parameters())
        for key, value in values.items():
            if not key.endswith(("lora_A", "lora_B")):
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
        for prefix, rank in zip(model.AFFINE_NAMES, net.ranks):
            if not rank:
                assert net.factor_mapping["layers"][prefix] == dict(rank=0, rows=[])
                continue
            layer = net.get_submodule(prefix)
            torch.testing.assert_close(layer.lora_A[:2], reference[prefix + ".lora_A"][:rank],
                                       rtol=0, atol=0)
            for entry in net.factor_mapping["layers"][prefix]["rows"][2:]:
                component = entry["seed_component"]
                seed = derive_component_seed(net.factor_seed, component)
                assert seed == net.factor_seeds[component]
                generator = torch.Generator().manual_seed(seed)
                # Factors are drawn in the legacy initialization dtype, then cast.
                expected = torch.empty(1, layer.lora_A.shape[1], dtype=torch.float32)
                nn.init.kaiming_uniform_(expected, a=math.sqrt(5), generator=generator)
                torch.testing.assert_close(layer.lora_A[entry["row"]], expected[0].double(),
                                           rtol=0, atol=0)
        assert net.factor_seeds[model.FACTOR_SEED_COMPONENT] == net.factor_seed
        json.dumps(dict(mapping=net.factor_mapping, seeds=net.factor_seeds))
        if name == "all_r2":
            assert layout.metadata() == legacy_layout.metadata()
            assert torch.equal(layout.flatten_module(net), legacy_layout.flatten_module(legacy))
    torch.testing.assert_close(charts["head_r1"].classifier.lora_A,
                               charts["all_r1"].classifier.lora_A, rtol=0, atol=0)
    extra_seeds = list(charts["all_r4"].factor_seeds.values())
    assert len(extra_seeds) == len(set(extra_seeds)) == 7
    other, _ = model.build_model(13, factor_seed=58, dtype=torch.float64, stm_config="all_r4")
    for prefix in model.AFFINE_NAMES:
        assert not torch.equal(other.get_submodule(prefix).lora_A,
                               charts["all_r4"].get_submodule(prefix).lora_A)


def test_identical_initial_predictions_and_complete_ltm_jacobians():
    legacy, legacy_layout = grid_model.build_model(17, factor_seed=57, dtype=torch.float64)
    legacy_theta = legacy_layout.flatten_module(legacy, detach=True)
    legacy_ltm, _ = grid_model.memory_indices(legacy_layout)
    x = torch.randn(1, 2, 84, 84, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    baseline = grid_model.flat_prediction(legacy, legacy_layout, legacy_theta, x)
    expected = torch.cat(list(grid_curvature.output_jacobians(
        legacy, legacy_layout, legacy_theta, x, indices=legacy_ltm,
        chunk_size=1, output_chunk_size=6)), dim=1)
    assert expected.shape == (1, 54, 3268)
    assert expected.norm() > 0
    for name in model.CONFIGURATIONS:
        net, layout = model.build_model(17, factor_seed=57, dtype=torch.float64, stm_config=name)
        theta = layout.flatten_module(net, detach=True)
        ltm, _ = model.memory_indices(layout)
        torch.testing.assert_close(model.flat_prediction(net, layout, theta, x), baseline, rtol=0, atol=0)
        rows = torch.cat(list(core.output_jacobians(net, layout, theta, x, indices=ltm,
                                                   chunk_size=1, output_chunk_size=6)), dim=1)
        torch.testing.assert_close(rows, expected, rtol=0, atol=0)
        selected = torch.tensor([0, 73, 76, 290, 3214, 3267])
        fisher = core.reference_fisher(net, layout, theta, x, indices=ltm[selected],
                                       chunk_size=1, output_chunk_size=6)
        jacobian = expected[0, :, selected]
        torch.testing.assert_close(fisher, jacobian.T @ jacobian / 4, rtol=1e-14, atol=1e-14)
        assert not fisher.requires_grad and fisher.dtype == torch.float64
        assert torch.equal(fisher, fisher.T)


def test_rejects_unsupported_mixed_reordered_and_incompatible_charts():
    layouts = [model.build_model(1, stm_config=name)[1] for name in model.CONFIGURATIONS]
    for index, layout in enumerate(layouts):
        model.memory_indices(layout)
        for other in layouts[:index]:
            with pytest.raises(ParameterLayoutError, match="metadata"):
                layout.assert_metadata(other.metadata())
        with pytest.raises(ValueError, match="chart/order"):
            model.memory_indices(replace(layout, specs=tuple(reversed(layout.specs))))
        broken = replace(layout.specs[0], stop=layout.specs[0].stop + 1)
        with pytest.raises(ValueError, match="offsets"):
            model.memory_indices(replace(layout, specs=(broken,) + layout.specs[1:]))
        with pytest.raises(ValueError, match="count"):
            model.memory_indices(replace(layout, total_numel=layout.total_numel + 1))
    layout = layouts[2]
    broken = replace(layout.specs[2], shape=(1, 18))
    with pytest.raises(ValueError, match="chart/order"):
        model.memory_indices(replace(layout, specs=layout.specs[:2] + (broken,) + layout.specs[3:]))
    with pytest.raises(ValueError, match="chart/order"):
        model.memory_indices(grid_model.build_model(1, channels=(4, 8))[1])
    for kwargs in ({"stm_config": "all_r3"}, {"stm_config": []}, {"channels": (4, 8)},
                   {"output_dim": 6}, {"seed": True}, {"factor_seed": -1},
                   {"factor_seed": True}, {"dtype": torch.float16}):
        with pytest.raises(ValueError):
            model.build_model(**({"seed": 1} | kwargs))
    with pytest.raises(ValueError, match="configuration"):
        with model.use_chart("unknown"):
            pass


@pytest.mark.parametrize("name", model.CONFIGURATIONS)
def test_initial_factor_gradient_masks_and_zero_stm_pretraining(name):
    net, layout, theta, x, ltm, stm, _ = case(name, moved=False)
    prediction = model.flat_prediction(net, layout, theta, x[:1])
    gradient = torch.autograd.grad(core.gaussian_nll(prediction, torch.ones_like(prediction)).mean(), theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    for spec in layout.specs:
        block = gradient[spec.start:spec.stop]
        if spec.name.endswith("lora_A"):
            assert block.count_nonzero() == 0
        else:
            assert block.norm() > 0
    prediction = model.flat_prediction(net, layout, theta, x[:1], stm_scale=0)
    gradient = torch.autograd.grad(core.gaussian_nll(prediction, torch.ones_like(prediction)).mean(), theta)[0]
    assert gradient[stm].count_nonzero() == 0 and gradient[ltm].norm() > 0


@pytest.mark.parametrize("name", model.CONFIGURATIONS)
def test_fd_jvp_values_full_gradients_gaussian_scale_and_probe_coordinates(name):
    net, layout, theta, x, ltm, stm, probes = case(name)
    x = x[:1]
    before = aliases()
    approximate = core.fd_values(net, layout, theta, x, probes, 1e-4)
    exact = core.jvp_values(net, layout, theta, x, probes)
    assert aliases() == before
    assert approximate.shape == (2, 1) and approximate.dtype == torch.float64
    torch.testing.assert_close(approximate, exact, rtol=2e-6, atol=1e-10)
    gradient = torch.autograd.grad(approximate.mean() / len(stm), theta)[0]
    expected_gradient = torch.autograd.grad(exact.mean() / len(stm), theta)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-5, atol=1e-10)
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    full = theta.new_zeros(len(probes), len(theta)).index_copy(1, stm, probes)
    actual = core.fd_values(net, layout, theta, x, probes, .01)
    torch.testing.assert_close(actual, core.fd_values(net, layout, theta, x, full, .01), rtol=0, atol=0)
    for k, direction in enumerate(full):
        plus = model.flat_prediction(net, layout, theta + .005 * direction, x)
        minus = model.flat_prediction(net, layout, theta - .005 * direction, x)
        torch.testing.assert_close(actual[k], ((plus - minus) / .01).square().sum(-1) / 4,
                                   rtol=1e-12, atol=1e-14)
    full[:, ltm[0]] = 1
    with pytest.raises(ValueError, match="zero in LTM"):
        core.fd_values(net, layout, theta, x, full, .01)
    for step in (0, -1, True, float("nan")):
        with pytest.raises(ValueError, match="step"):
            core.fd_values(net, layout, theta, x, probes, step)
    learnable_probes = probes.clone().requires_grad_()
    value = core.fd_values(net, layout, theta, x, learnable_probes, .01).mean()
    assert torch.autograd.grad(value, learnable_probes, allow_unused=True)[0] is None


def test_head_only_bilinear_identity_cancellation_and_replay_head_gradient():
    net, layout, theta, x, _, stm, probes = case("head_r1")
    x = x[:1]
    named = layout.unflatten_named(theta)
    features = {name.removeprefix("features."): value for name, value in named.items()
                if name.startswith("features.")}
    hidden = functional_call(net.features, features, (x,), strict=True).flatten(1)
    embedded = theta.new_zeros(len(probes), len(theta)).index_copy(1, stm, probes)
    tangents = []
    for direction in embedded:
        values = layout.unflatten_named(direction)
        product = (values["classifier.lora_B"] @ named["classifier.lora_A"]
                   + named["classifier.lora_B"] @ values["classifier.lora_A"])
        tangents.append(hidden @ product.T)
    analytic = torch.stack(tangents).square().sum(-1) / 4
    analytic_gradient = torch.autograd.grad(analytic.mean() / len(stm), theta)[0]
    exact = core.jvp_values(net, layout, theta, x, probes)
    torch.testing.assert_close(exact, analytic, rtol=1e-14, atol=1e-14)
    for step in (.005, .01, .02):
        approximate = core.fd_values(net, layout, theta, x, probes, step)
        torch.testing.assert_close(approximate, analytic.detach(), rtol=1e-11, atol=1e-12)
        gradient = torch.autograd.grad(approximate.mean() / len(stm), theta)[0]
        torch.testing.assert_close(gradient, analytic_gradient, rtol=1e-9, atol=1e-12)
        for spec in layout.specs:
            block = gradient[spec.start:spec.stop]
            if spec.name in ("classifier.weight", "classifier.bias"):
                assert block.count_nonzero() == 0
            else:
                assert block.norm() > 0
    prediction = model.flat_prediction(net, layout, theta, x)
    replay_gradient = torch.autograd.grad(core.gaussian_nll(prediction, prediction.detach() + 1).mean(), theta)[0]
    for spec in layout.specs:
        if spec.name in ("classifier.weight", "classifier.bias"):
            assert replay_gradient[spec.start:spec.stop].norm() > 0


def test_legacy_all_r2_objective_and_gradient_parity(archive):
    net, layout, theta, x, ltm, _, _ = case()
    legacy, old_layout = grid_model.build_model(17, factor_seed=57, dtype=torch.float64)
    generator = torch.Generator().manual_seed(136)
    candidate = (theta.detach() + .001 * torch.randn(theta.shape, generator=generator,
                                                     dtype=theta.dtype)).requires_grad_()
    old_objective, old_metadata = grid_learner.prepare_objective(
        *objective_args(legacy, old_layout, theta, x, ltm, archive))
    old_value, old_parts = old_objective(candidate)
    old_gradient = torch.autograd.grad(old_value, candidate)[0]
    old_parts = {key: value.detach() for key, value in old_parts.items()}
    del old_objective
    before = aliases()
    objective, metadata = core.prepare_objective(*objective_args(net, layout, theta, x, ltm, archive))
    assert aliases() == before
    # Returned closures must work after the preparation bindings have exited.
    value, parts = objective(candidate)
    gradient = torch.autograd.grad(value, candidate)[0]
    assert aliases() == before and metadata == old_metadata
    torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
    for key, component in parts.items():
        torch.testing.assert_close(component, old_parts[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", model.CONFIGURATIONS)
def test_objective_keeps_ltm_ridge_scale_age_probe_and_stm_normalization(name, archive):
    net, layout, theta, x, ltm, stm, _ = case(name)
    args = objective_args(net, layout, theta, x, ltm, archive)
    cfg, seed = args[-2:]
    before = aliases()
    with model.use_chart("all_r4"):
        objective, metadata = core.prepare_objective(*args)
    assert aliases() == before
    candidate = (theta.detach() + .001).requires_grad_()
    value, parts = objective(candidate)
    assert metadata["stm_dimension"] == len(stm)
    kappa = cfg.ridge_ratio * float(archive.trace()) / 3268
    assert metadata["kappa"] == pytest.approx(kappa)
    assert metadata["fd_perturbation_norm"] == pytest.approx(cfg.fd_step * math.sqrt(len(stm)))
    assert metadata["probe_seed"] == derive_component_seed(seed, grid_learner.PROBE_SEED_COMPONENT)
    generator = torch.Generator().manual_seed(metadata["probe_seed"])
    probes = (2 * torch.randint(2, (2, len(stm)), generator=generator) - 1).to(theta)
    auxiliary = .25 * core.fd_values(net, layout, candidate, x[1:], probes, cfg.fd_step).mean() / len(stm)
    new = core.gaussian_nll(model.flat_prediction(net, layout, candidate, x[:1]), args[4]).mean()
    replay = core.gaussian_nll(model.flat_prediction(net, layout, candidate, x[1:]), args[6]).mean()
    dl, ds = candidate[ltm] - args[9], candidate[stm] - theta[stm].detach()
    omega = (1 - cfg.pi) / cfg.pi
    ewc = omega / 2 * (dl @ archive @ dl)
    ridge = omega * kappa / 2 * (dl.square().sum() + ds.square().sum())
    expected = .5 * (new + replay) + ewc + ridge + cfg.lambda_f * auxiliary
    for key, component in dict(auxiliary=auxiliary, new_nll=new, replay_nll=replay,
                               ewc=ewc, ridge=ridge).items():
        torch.testing.assert_close(parts[key], component, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(value, expected, rtol=1e-14, atol=1e-14)
    gradient = torch.autograd.grad(value, candidate)[0]
    expected_gradient = torch.autograd.grad(expected, candidate)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-13, atol=1e-13)
    assert aliases() == before


def test_tiny_two_update_legacy_accepted_fit_parity(archive):
    net, layout, theta, x, ltm, _, _ = case()
    legacy, old_layout = grid_model.build_model(17, factor_seed=57, dtype=torch.float64)
    cfg = FitSettings(lambda_f=1.7, num_probes=2, fd_step=.01, max_iter=1,
                      max_eval=4, chunk_size=1, learning_rate=.1)
    new_theta, old_theta = theta.detach().clone(), theta.detach().clone()
    original = layout.flatten_module(net, detach=True)
    for _ in range(2):
        old_theta, old_diagnostics = grid_learner.fit_step(
            *objective_args(legacy, old_layout, old_theta, x, ltm, archive, cfg))
        new_theta, diagnostics = core.fit_step(
            *objective_args(net, layout, new_theta, x, ltm, archive, cfg))
        assert diagnostics["accepted_steps"] > 0 and diagnostics["objective_decrease"] > 0
        torch.testing.assert_close(new_theta, old_theta, rtol=0, atol=0)
        for key, value in diagnostics.items():
            if not key.endswith("seconds"):
                assert value == old_diagnostics[key]
    assert torch.equal(layout.flatten_module(net, detach=True), original)


def test_nested_bindings_chart_selection_failure_restoration_and_runner_model(monkeypatch):
    sentinel = lambda layout: None
    monkeypatch.setattr(grid_numerical, "memory_indices", sentinel)
    before = aliases()
    assert model.CHART.get() == "all_r2"
    runner = object.__new__(Runner)
    runner.channels = (4, 6)
    runner.cfg = SimpleNamespace(seed_for=lambda *args: 17, device="cpu", dtype="float64")
    with model.use_chart("head_r1"), core.bindings():
        outer = aliases()
        for module in (grid_model, grid_curvature, grid_learner, grid_numerical):
            assert module.memory_indices is model.memory_indices
        for name in model.CONFIGURATIONS:
            with model.use_chart(name):
                net, layout = runner._model("calibration--grid", 0)
                assert net.stm_config == name
                assert tuple(len(block) for block in model.memory_indices(layout)) == model.COUNTS[name]
        with pytest.raises(RuntimeError, match="nested failure"):
            with model.use_chart("all_r4"), core.bindings():
                assert model.CHART.get() == "all_r4"
                explicit, _ = grid_model.build_model(1, stm_config="all_r1")
                assert explicit.stm_config == "all_r1" and model.CHART.get() == "all_r4"
                raise RuntimeError("nested failure")
        assert aliases() == outer and model.CHART.get() == "head_r1"
    assert aliases() == before and model.CHART.get() == "all_r2"
    with pytest.raises(RuntimeError, match="outer failure"):
        with core.bindings(), model.use_chart("all_r1"):
            raise RuntimeError("outer failure")
    assert aliases() == before and model.CHART.get() == "all_r2"


def test_lazy_jacobian_and_failed_direct_calls_restore_bindings():
    net, layout, theta, x, ltm, _, _ = case("head_r1")
    before = aliases()
    iterator = core.output_jacobians(net, layout, theta, x[:1], indices=ltm[:2],
                                     chunk_size=1, output_chunk_size=2)
    assert aliases() == before
    assert next(iterator).shape == (1, 2, 2)
    assert aliases() == before
    iterator.close()
    assert aliases() == before
    with pytest.raises(StopRequested):
        core.reference_fisher(net, layout, theta, x[:1], indices=ltm[:2], check_stop=lambda: True)
    assert aliases() == before
    with pytest.raises(StopRequested):
        next(core.output_jacobians(net, layout, theta, x[:1], indices=ltm[:2], check_stop=lambda: True))
    assert aliases() == before


def test_inherited_head_only_numerical_calibration_uses_bound_alias():
    net, layout, theta, x, _, _, _ = case("head_r1")
    y = torch.ones(2, 54, dtype=torch.float64)
    snapshot = dict(theta=theta.detach(), new_x=x[:1], new_y=y[:1], replay_x=x[1:],
                    replay_y=y[1:], ages=torch.tensor([.25]), seed=115, n=4, step=1,
                    layout=layout.metadata())
    cfg = SimpleNamespace(fd_step=.01, numerical_tolerance=1e-7, probes=2, dtype="float64")
    before = aliases()
    with model.use_chart("head_r1"), core.bindings():
        result = grid_numerical.calibrate_fd(cfg, [snapshot], net, layout)
    assert aliases() == before
    assert result["stm_dimension"] == 108 and result["selected_step"] == .01
    assert result["production_verified"] and result["lambda0"] > 0
    assert all(row["passed"] for row in result["numerical_rows"])
