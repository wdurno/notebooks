"""Small output-level cancellation and adapted-path precision regressions."""

import math

import pytest
import torch

from src.continual import picar_math as pm


def design(batch, bins, dtype=torch.float32):
    return pm.LikelihoodDesign(torch.arange(bins, dtype=dtype),
                               torch.zeros(batch, dtype=dtype), 5., torch.eye(7, dtype=dtype))


def output(logits):
    return {"return_logits": logits, "action_mean": logits.new_zeros(logits.shape[0], 7)}


def test_close_fp32_categorical_directed_kl_is_positive_and_matches_float64():
    generator = torch.Generator().manual_seed(9)
    left = torch.randn(256, 5, generator=generator) * 2
    right = left + 1e-4 * torch.randn(256, 5, generator=generator)
    actual = pm.directed_kl(output(left), output(right), design(256, 5))
    logp, logq = left.double().log_softmax(-1), right.double().log_softmax(-1)
    reference = (logp.exp() * (logp - logq)).sum(-1)
    assert actual.dtype == torch.float32
    assert bool((actual >= 0).all())
    torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=1e-15)


def test_sub_ulp_scale_symmetric_kl_uses_precise_output_arithmetic():
    generator = torch.Generator().manual_seed(9)
    left = torch.randn(64, 5, generator=generator) * 2
    right = left + 1e-7 * torch.randn(64, 5, generator=generator)
    actual = pm.symmetric_kl(output(left), output(right), design(64, 5))
    logp, logq = left.double().log_softmax(-1), right.double().log_softmax(-1)
    reference = ((logp.exp() - logq.exp()) * (logp - logq)).sum(-1)
    assert bool((actual >= 0).all())
    torch.testing.assert_close(actual.double(), reference, rtol=2e-6, atol=1e-25)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kl_shift_invariance_extreme_logits_and_both_endpoint_gradients(dtype):
    left = torch.tensor([[0.25, 0.5, -0.25], [1000., -1000., 0.]], dtype=dtype, requires_grad=True)
    shifted = left.detach() + 16
    fixed = design(2, 3, dtype)
    torch.testing.assert_close(pm.directed_kl(output(left), output(shifted), fixed),
                               torch.zeros(2, dtype=dtype), rtol=0, atol=1e-25)
    right = torch.tensor([[0.26, 0.49, -0.24], [-1000., 1000., 0.]],
                         dtype=dtype, requires_grad=True)
    actual = pm.directed_kl(output(left), output(right), fixed)
    actual_grads = torch.autograd.grad(actual.sum(), (left, right))
    logp, logq = left.double().log_softmax(-1), right.double().log_softmax(-1)
    reference = (logp.exp() * (logp - logq)).sum(-1)
    reference_grads = torch.autograd.grad(reference.sum(), (left, right))
    torch.testing.assert_close(actual.double(), reference, rtol=1e-6, atol=1e-14)
    for a, b in zip(actual_grads, reference_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("round_hidden", [False, True])
def test_nonzero_both_bank_gradients_do_not_certify_fd_convergence(round_hidden):
    # Place the adapted activation on a BF16 rounding boundary near one.
    params = {"ltm": torch.tensor(1., requires_grad=True),
              "stm": torch.tensor(math.sqrt(1 + 1 / 256), requires_grad=True)}
    fixed = design(1, 2)

    def predict(p):
        hidden = p["ltm"] * p["stm"].square()
        if round_hidden:
            hidden = hidden.to(torch.bfloat16).float()
        return {"return_logits": torch.zeros(1, 2),
                "action_mean": hidden * torch.tensor([[1., 0., 0., 0., 0., 0., 0.]])}

    penalties = []
    for epsilon in (0.0005, 0.001, 0.002):
        penalty = pm.stm_fd_penalty(predict, params, fixed, ["stm"],
                                    ({"stm": torch.tensor(1.)},), torch.ones(1), epsilon)
        gradients = torch.autograd.grad(penalty, tuple(params.values()))
        assert all(torch.isfinite(g) and g.abs() > 0 for g in gradients)
        penalties.append(float(penalty.detach()))
    ratios = [penalties[i] / penalties[i + 1] for i in range(2)]
    if round_hidden:
        assert ratios == pytest.approx([4., 4.], rel=1e-6)
    else:
        assert ratios == pytest.approx([1., 1.], rel=0.002)
