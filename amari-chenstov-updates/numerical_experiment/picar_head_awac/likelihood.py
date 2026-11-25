"""One normalized conditional working model; references are detached designs."""

from dataclasses import dataclass
import math

import torch
from torch.func import jacrev


@dataclass(frozen=True)
class Design:
    weights: torch.Tensor
    targets: torch.Tensor
    action_variance: float
    critic_variance: float

    def validate(self):
        if self.weights.requires_grad or self.targets.requires_grad:
            raise ValueError("conditional weights and targets must be frozen")
        if (not torch.isfinite(self.weights).all() or not (self.weights > 0).all()
                or not torch.isfinite(self.targets).all()
                or not self.action_variance > 0 or not self.critic_variance > 0):
            raise ValueError("invalid conditional likelihood design")


def data_loss(output, actions, design):
    design.validate()
    k = actions.shape[-1]
    actor = .5 * (design.weights * (output["mean"] - actions).square().sum(-1)
                  / design.action_variance
                  + k * (math.log(2 * math.pi * design.action_variance) - design.weights.log()))
    q = (output["q1"] + output["q2"]) / 2
    critic = .5 * ((q - design.targets).square() / design.critic_variance
                   + math.log(2 * math.pi * design.critic_variance))
    disagreement = (output["q1"] - output["q2"]).square() / (8 * design.critic_variance)
    return {"nll": actor + critic, "actor": actor, "critic": critic,
            "disagreement": disagreement}


def joint_kl(coordinates, source, target, features, design, draws):
    design.validate()
    zeros = features.new_zeros((len(features), 7))
    left = coordinates.predict(source, features, zeros)["mean"]
    right = coordinates.predict(target, features, zeros)["mean"]
    actor = .5 * design.weights * (left - right).square().sum(-1) / design.action_variance
    actions = left[:, None, :] + (design.action_variance / design.weights).sqrt()[:, None, None] * draws
    h = features[:, None, :].expand(-1, actions.shape[1], -1)
    p = coordinates.predict(source, h, actions)
    q = coordinates.predict(target, h, actions)
    delta = (p["q1"] + p["q2"] - q["q1"] - q["q2"]) / 2
    return actor + delta.square().mean(-1) / (2 * design.critic_variance)


def fisher_factor(coordinates, vector, feature, weight, action_variance, critic_variance, draws):
    """Analytically integrate the reward residual; integrate actions by draws.

    Fixed model labels are differentiated at fixed actions, not through the
    action sampler. All columns are contributions of this one model Fisher.
    """
    feature = feature.reshape(1, -1)
    zero = feature.new_zeros((1, 7))
    weight = torch.as_tensor(weight, device=feature.device, dtype=feature.dtype).detach()
    mean = coordinates.predict(vector, feature, zero)["mean"][0]
    actor_j = jacrev(lambda t: coordinates.predict(t, feature, zero)["mean"][0])(vector)
    actor_factor = actor_j.T * (weight / action_variance).sqrt()
    actions = (mean.detach()[None, :] + (action_variance / weight).sqrt() * draws).detach()
    def q_values(t):
        output = coordinates.predict(t, feature.expand(len(actions), -1), actions)
        return (output["q1"] + output["q2"]) / 2
    critic_j = jacrev(q_values)(vector)
    return torch.cat((actor_factor, critic_j.T / math.sqrt(critic_variance * len(actions))), 1).detach()


@torch.no_grad()
def reference_design(coordinates, reference, features, actions, next_features, rewards,
                     discounts, returns, draws, *, method, temperature,
                     action_variance, critic_variance, log_weight_bound=5.):
    reference = reference.detach()
    def value(h):
        mean = coordinates.predict(reference, h, torch.zeros_like(actions))["mean"]
        candidate_actions = mean[:, None, :] + math.sqrt(action_variance) * draws
        output = coordinates.predict(reference, h[:, None, :].expand(-1, draws.shape[1], -1),
                                     candidate_actions)
        return torch.minimum(output["q1"], output["q2"]).mean(-1)
    current = coordinates.predict(reference, features, actions)
    advantage = torch.minimum(current["q1"], current["q2"]) - value(features)
    log_weights = advantage / temperature
    weights = log_weights.clamp(-log_weight_bound, log_weight_bound).exp()
    target = returns if method == "mc_replay" else rewards + discounts * value(next_features)
    return Design(weights.detach(), target.detach(), action_variance, critic_variance), {
        "weight_bound_fraction": float((log_weights.abs() > log_weight_bound).float().mean()),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "target_min": float(target.min()), "target_max": float(target.max())}
