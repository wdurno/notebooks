"""Conditional working-model identities and fixed-reference differentiation."""

import math
import torch
from torch.func import jacrev

from numerical_experiment.picar_head_awac.model import Heads, Coordinates
from numerical_experiment.picar_head_awac.likelihood import Design, data_loss, joint_kl, fisher_factor, reference_design


def fixture():
    heads = Heads(hidden=3, width=2, seed=91).double()
    coordinates = Coordinates(heads)
    vector = coordinates.vector()
    feature = torch.tensor([[.2, -.1, .7]], dtype=torch.float64)
    action = torch.tensor([[.1]*7], dtype=torch.float64)
    design = Design(torch.tensor([2.], dtype=torch.float64), torch.tensor([.3], dtype=torch.float64), 4., 2.)
    generator = torch.Generator().manual_seed(122)
    draws = torch.randn((1, 32, 7), generator=generator, dtype=torch.float64)
    return coordinates, vector, feature, action, design, draws


def test_normalizers_and_twin_identity():
    coordinates, vector, h, z, d, _ = fixture()
    output = coordinates.predict(vector, h, z)
    parts = data_loss(output, z, d)
    actor = -torch.distributions.MultivariateNormal(output["mean"], torch.eye(7, dtype=z.dtype)*2).log_prob(z)
    q1, q2 = output["q1"], output["q2"]
    q = (q1+q2)/2
    critic = -torch.distributions.Normal(q, math.sqrt(2)).log_prob(d.targets)
    torch.testing.assert_close(parts["nll"], actor+critic)
    averaged = ((q1-d.targets).square()+(q2-d.targets).square())/(4*d.critic_variance)+.5*math.log(2*math.pi*d.critic_variance)
    torch.testing.assert_close(parts["critic"]+parts["disagreement"], averaged)


def test_kl_zero_and_fisher_direction_agree():
    c, vector, h, z, d, draws = fixture()
    torch.testing.assert_close(joint_kl(c, vector, vector, h, d, draws), torch.zeros(1, dtype=h.dtype))
    factor = fisher_factor(c, vector, h[0], d.weights[0], d.action_variance, d.critic_variance, draws[0])
    direction = torch.linspace(-.1, .1, vector.numel(), dtype=vector.dtype)
    epsilon = 1e-5
    actual = joint_kl(c, vector, vector+epsilon*direction, h, d, draws)*2/epsilon**2
    expected = (factor.T@direction).square().sum()
    torch.testing.assert_close(actual[0], expected, rtol=2e-4, atol=1e-8)


def test_nll_gradient_negative_score_and_joint_sampler_derivative():
    c, vector, h, z, d, draws = fixture()
    t = vector.clone().requires_grad_()
    score = torch.autograd.grad(-data_loss(c.predict(t,h,z),z,d)["nll"].sum(),t)[0]
    t = vector.clone().requires_grad_()
    negative_score = torch.autograd.grad(data_loss(c.predict(t,h,z),z,d)["nll"].sum(),t)[0]
    torch.testing.assert_close(score, -negative_score)
    target = vector+torch.linspace(-.01,.01,len(vector),dtype=vector.dtype)
    grad = jacrev(lambda v: joint_kl(c,v,target,h,d,draws).sum())(vector)
    direction = torch.sin(torch.arange(len(vector),dtype=vector.dtype))
    eps = 1e-5
    finite = (joint_kl(c,vector+eps*direction,target,h,d,draws)-joint_kl(c,vector-eps*direction,target,h,d,draws))/(2*eps)
    torch.testing.assert_close(grad@direction, finite[0], rtol=1e-5, atol=1e-8)


def test_targets_and_weights_detached_terminal_uses_observed_reward():
    c, vector, h, z, _, draws = fixture()
    reference = vector.requires_grad_()
    design, diagnostics = reference_design(c,reference,h,z,h,torch.tensor([2.],dtype=h.dtype),
        torch.zeros(1,dtype=h.dtype),torch.tensor([5.],dtype=h.dtype),draws,method="td_replay",
        temperature=3.,action_variance=4.,critic_variance=2.)
    assert not design.targets.requires_grad and not design.weights.requires_grad
    torch.testing.assert_close(design.targets, torch.tensor([2.],dtype=h.dtype))
    frozen = design.weights.clone()
    c.predict(reference+1,h,z)
    torch.testing.assert_close(design.weights, frozen)


def test_exact_bellman_mle_example():
    p = torch.tensor([[.6,.3],[.2,.7]],dtype=torch.float64)
    b = torch.eye(2,dtype=p.dtype)-.8*p
    reward = torch.tensor([.3,-.2],dtype=p.dtype)
    q = torch.zeros_like(reward)
    for _ in range(150):
        q = reward+.8*p@q
    torch.testing.assert_close(q, torch.linalg.solve(b,reward))
    d = torch.diag(torch.tensor([40.,60.],dtype=p.dtype))/.7
    inv = torch.linalg.inv(b)
    torch.testing.assert_close(inv@torch.linalg.inv(d)@inv.T,torch.linalg.inv(b.T@d@b))
