"""Plan 19 likelihood and memory kernels; see NUMERICAL_CONTRACT.md (v1).

Scores are gradients of log likelihood, hence the negative of NLL gradients.
All model-label expectations use this joint model, not the exploration policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

CONTRACT_VERSION = 1
ACTION_DIM = 7
Params = Mapping[str, Tensor]
Predict = Callable[[Params], Mapping[str, Tensor]]


def _finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite tensor")


def _accumulate(value: Tensor) -> Tensor:
    _finite(value, "numeric input")
    if not value.is_floating_point():
        raise ValueError("expected a floating point tensor")
    return value.to(torch.float64 if value.dtype == torch.float64 else torch.float32)


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _pi(value: float) -> float:
    value = _scalar(value, "pi", positive=True)
    if value > 1:
        raise ValueError("pi must be in (0, 1]")
    return value


@dataclass(frozen=True)
class LikelihoodDesign:
    """Detached per-history bin/weight/covariance snapshot, never a parameter."""

    bin_values: Tensor
    baseline: Tensor
    temperature: float | Tensor
    sigma_action: Tensor
    log_weight_cap: float | None = None
    log_weights: Tensor = field(init=False)
    weights: Tensor = field(init=False)
    cholesky: Tensor = field(init=False)
    cap_hit_rate: Tensor = field(init=False)

    def __post_init__(self) -> None:
        baseline = _accumulate(self.baseline.detach()).clone()
        if baseline.ndim != 1 or baseline.numel() == 0:
            raise ValueError("baseline must have nonempty shape [B]")
        batch = baseline.numel()

        def fixed(value: Tensor | float) -> Tensor:
            result = torch.as_tensor(value, device=baseline.device, dtype=baseline.dtype)
            _finite(result, "conditioning design")
            return result.detach().clone()

        bins = fixed(self.bin_values)
        if bins.ndim == 1:
            bins = bins.expand(batch, -1).clone()
        if (bins.ndim != 2 or bins.shape[0] != batch or bins.shape[1] < 2
                or not bool((bins[:, 1:] > bins[:, :-1]).all())):
            raise ValueError("bin_values must be increasing [K] or [B,K], K >= 2")
        temperature = fixed(self.temperature)
        if temperature.ndim == 0:
            temperature = temperature.expand(batch).clone()
        if temperature.shape != baseline.shape or not bool((temperature > 0).all()):
            raise ValueError("temperature must be positive scalar or [B]")
        sigma = fixed(self.sigma_action)
        if sigma.shape == (ACTION_DIM, ACTION_DIM):
            sigma = sigma.expand(batch, -1, -1).clone()
        if sigma.shape != (batch, ACTION_DIM, ACTION_DIM):
            raise ValueError("sigma_action must have shape [7,7] or [B,7,7]")
        if not torch.allclose(sigma, sigma.mT, rtol=1e-6, atol=0):
            raise ValueError("sigma_action must be symmetric")
        sigma = (sigma + sigma.mT) * 0.5
        chol, info = torch.linalg.cholesky_ex(sigma)
        if bool((info != 0).any()):
            raise ValueError("sigma_action must be positive definite")
        log_weights = (bins - baseline[:, None]) / temperature[:, None]
        _finite(log_weights, "log weights")
        hits = torch.zeros_like(log_weights, dtype=torch.bool)
        if self.log_weight_cap is not None:
            cap = float(self.log_weight_cap)
            if not math.isfinite(cap):
                raise ValueError("log_weight_cap must be finite")
            hits = log_weights > cap
            log_weights = log_weights.clamp_max(cap)
        weights = log_weights.exp()
        if not bool((torch.isfinite(weights) & (weights > 0)).all()):
            raise ValueError("advantage weights overflowed or underflowed; revise frozen design")
        for name, value in (("baseline", baseline), ("bin_values", bins),
                            ("temperature", temperature), ("sigma_action", sigma),
                            ("cholesky", chol), ("log_weights", log_weights),
                            ("weights", weights),
                            ("cap_hit_rate", hits.to(baseline.dtype).mean())):
            object.__setattr__(self, name, value)


def functional_predict(
    model: nn.Module,
    params: Params,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    buffers: Params | None = None,
) -> Mapping[str, Tensor]:
    """Evaluate explicit coordinates; omitted module parameters remain fixed.

    The caller freezes omitted parameters and supplies deterministic context.
    Cloned buffers prevent ordinary in-place running-stat updates to the module.
    """
    unknown = set(params) - set(dict(model.named_parameters()))
    if unknown:
        raise ValueError(f"unknown model parameter names: {sorted(unknown)}")
    buffer_values = dict(model.named_buffers())
    if buffers is not None:
        unknown_buffers = set(buffers) - buffer_values.keys()
        if unknown_buffers:
            raise ValueError(f"unknown model buffer names: {sorted(unknown_buffers)}")
        buffer_values.update(buffers)
    buffer_values = {name: value.detach().clone() for name, value in buffer_values.items()}
    return functional_call(model, (dict(params), buffer_values), args, dict(kwargs or {}))


def _outputs(output: Mapping[str, Tensor], design: LikelihoodDesign) -> tuple[Tensor, Tensor]:
    if not isinstance(output, Mapping) or not {"return_logits", "action_mean"} <= output.keys():
        raise ValueError("model must return return_logits and action_mean tensors")
    logits, mean = output["return_logits"], output["action_mean"]
    for value in (logits, mean):
        _finite(value, "model output")
        if not value.is_floating_point() or value.device != design.baseline.device:
            raise ValueError("model outputs must be floating point on the design device")
    if logits.shape != design.weights.shape or mean.shape != (logits.shape[0], ACTION_DIM):
        raise ValueError("model outputs must have shapes [B,K] and [B,7] matching design")
    return logits.to(design.baseline.dtype), mean.to(design.baseline.dtype)


def _targets(labels: Tensor, actions: Tensor, design: LikelihoodDesign) -> tuple[Tensor, Tensor]:
    batch, bins = design.weights.shape
    if (labels.shape != (batch,) or labels.dtype != torch.int64
            or labels.device != design.baseline.device
            or not bool(((labels >= 0) & (labels < bins)).all())):
        raise ValueError("labels must be int64 [B] in [0,K), on the design device")
    _finite(actions, "actions")
    if (actions.shape != (batch, ACTION_DIM) or not actions.is_floating_point()
            or actions.device != design.baseline.device):
        raise ValueError("actions must be floating point [B,7] on the design device")
    return labels.detach(), actions.detach().to(design.baseline.dtype)


def _whiten(value: Tensor, design: LikelihoodDesign) -> Tensor:
    return torch.linalg.solve_triangular(design.cholesky, value.unsqueeze(-1), upper=False).squeeze(-1)


def joint_nll(output: Mapping[str, Tensor], labels: Tensor, actions: Tensor,
              design: LikelihoodDesign) -> Tensor:
    """Full normalized NLL per record, with recorded/synthetic targets detached."""
    logits, mean = _outputs(output, design)
    labels, actions = _targets(labels, actions, design)
    index = labels[:, None]
    categorical = -logits.log_softmax(-1).gather(1, index).squeeze(1)
    weight = design.weights.gather(1, index).squeeze(1)
    log_weight = design.log_weights.gather(1, index).squeeze(1)
    logdet_half = design.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    result = (categorical + 0.5 * weight * _whiten(actions - mean, design).square().sum(-1)
              + 0.5 * ACTION_DIM * (math.log(2 * math.pi) - log_weight) + logdet_half)
    _finite(result, "joint NLL")
    return result


def directed_kl(source: Mapping[str, Tensor], target: Mapping[str, Tensor],
                design: LikelihoodDesign) -> Tensor:
    """Exact KL(source || target), with source model probabilities attached.

    Only the small categorical output arithmetic uses float64. This prevents
    near-equal FP32 distributions from producing cancellation-dominated KLs;
    it cannot repair reduced-precision discontinuities inside the predictor.
    """
    source_logits, source_mean = _outputs(source, design)
    target_logits, target_mean = _outputs(target, design)
    logp, logq = source_logits.double().log_softmax(-1), target_logits.double().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    difference = logp - logq
    # Sum(p log(p/q) + q - p) equals KL. expm1 avoids first-order cancellation
    # near equality; the bounded branch also prevents unselected exp overflow.
    small = difference.clamp(-0.5, 0.5)
    categorical = torch.where(difference.abs() <= 0.5,
        p * (torch.expm1(-small) + small), p * difference + q - p).sum(-1)
    result = (categorical
              + 0.5 * (p * design.weights).sum(-1)
              * _whiten(source_mean - target_mean, design).square().sum(-1))
    result = result.to(design.baseline.dtype)
    _finite(result, "directed KL")
    return result


def symmetric_kl(left: Mapping[str, Tensor], right: Mapping[str, Tensor],
                 design: LikelihoodDesign) -> Tensor:
    """Sum of the two directed KLs, not their mean."""
    left_logits, left_mean = _outputs(left, design)
    right_logits, right_mean = _outputs(right, design)
    logp, logq = left_logits.double().log_softmax(-1), right_logits.double().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    # Jeffreys form avoids summing two cancellation-prone directed expressions.
    result = (((p - q) * (logp - logq)).sum(-1)
              + 0.5 * ((p + q) * design.weights).sum(-1)
              * _whiten(left_mean - right_mean, design).square().sum(-1))
    result = result.to(design.baseline.dtype)
    _finite(result, "symmetric KL")
    return result


@torch.no_grad()
def sample_model_labels(output: Mapping[str, Tensor], design: LikelihoodDesign,
                        *, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    if generator is None:
        raise ValueError("model label sampling requires an explicit generator")
    logits, mean = _outputs(output, design)
    labels = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(1)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    scale = design.weights.gather(1, labels[:, None]).sqrt()
    actions = mean + (design.cholesky @ noise.unsqueeze(-1)).squeeze(-1) / scale
    _finite(actions, "sampled actions")
    return labels.detach(), actions.detach()


def _snapshot(params: Params) -> dict[str, Tensor]:
    if not params:
        raise ValueError("provide at least one bank parameter")
    result = {}
    for name, value in params.items():
        _accumulate(value)
        if value.numel() == 0:
            raise ValueError("parameter tensors must be nonempty")
        result[name] = value.detach().clone().requires_grad_(True)
    return result


def _gradients(scalar: Tensor, params: Params) -> dict[str, Tensor]:
    values = tuple(params.values())
    grads = (torch.autograd.grad(scalar, values, retain_graph=True, allow_unused=True)
             if scalar.requires_grad else (None,) * len(values))
    return {name: (torch.zeros_like(_accumulate(value)) if grad is None
                   else _accumulate(grad.detach()))
            for (name, value), grad in zip(params.items(), grads)}


@torch.enable_grad()
def per_record_scores(predict: Predict, params: Params, labels: Tensor,
                      actions: Tensor, design: LikelihoodDesign) -> dict[str, Tensor]:
    """Diagnostic scores [B,*shape]; never a gradient of the batch mean."""
    snapshot = _snapshot(params)
    nll = joint_nll(predict(snapshot), labels, actions, design)
    scores: dict[str, list[Tensor]] = {name: [] for name in snapshot}
    for loss in nll:
        for name, gradient in _gradients(loss, snapshot).items():
            scores[name].append(-gradient)
    return {name: torch.stack(records) for name, records in scores.items()}


@torch.enable_grad()
def model_fisher_diagonal(predict: Predict, params: Params, design: LikelihoodDesign,
                          *, method: str = "exact", draws: int = 1,
                          generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """Stream squared per-record scores/output derivatives in O(parameter count) space.

    Exact mode sums categorical and whitened-mean Jacobian squares. Sampled mode
    draws detached labels and averages recordwise squares, including shared-head
    cross terms in each draw, whose model expectation is zero.
    """
    if method not in {"exact", "sampled"}:
        raise ValueError("method must be 'exact' or 'sampled'")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if method == "sampled" and generator is None:
        raise ValueError("sampled Fisher requires an explicit generator")
    snapshot = _snapshot(params)
    output = predict(snapshot)
    logits, mean = _outputs(output, design)
    result = {name: torch.zeros_like(_accumulate(value)) for name, value in snapshot.items()}
    batch = logits.shape[0]

    def add(scalar: Tensor, weight: Tensor | float) -> None:
        for name, gradient in _gradients(scalar, snapshot).items():
            result[name].add_(gradient.square() * weight)

    if method == "exact":
        logp = logits.log_softmax(-1)
        probabilities = logp.detach().exp()
        expected_weight = (probabilities * design.weights).sum(-1)
        whitened = _whiten(mean, design)
        for record in range(batch):
            for category in range(logits.shape[1]):
                add(logp[record, category], probabilities[record, category] / batch)
            for coordinate in range(ACTION_DIM):
                add(whitened[record, coordinate], expected_weight[record] / batch)
    else:
        for _ in range(draws):
            labels, actions = sample_model_labels(output, design, generator=generator)
            for loss in joint_nll(output, labels, actions, design):
                add(loss, 1.0 / (batch * draws))
    for value in result.values():
        _finite(value, "Fisher diagonal")
    return result


def _bank(params: Params, names: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(names)
    if not names or len(set(names)) != len(names) or not set(names) <= params.keys():
        raise ValueError(f"{label} names must be nonempty, unique, and present in params")
    for name in names:
        _accumulate(params[name])
        if params[name].numel() == 0:
            raise ValueError(f"{label} tensors must be nonempty")
    return names


def _matching(values: Params, params: Params, names: Sequence[str], label: str,
              *, nonnegative: bool = False) -> None:
    if set(values) != set(names):
        raise ValueError(f"{label} must contain exactly its bank's names")
    for name in names:
        value = values[name]
        _finite(value, label)
        if (value.shape != params[name].shape or value.device != params[name].device
                or not value.is_floating_point()):
            raise ValueError(f"{label} shape/device must match parameters")
        if nonnegative and bool((value < 0).any()):
            raise ValueError(f"{label} must be nonnegative")


def rademacher_probes(params: Params, stm_names: Sequence[str], count: int,
                      *, generator: torch.Generator) -> tuple[dict[str, Tensor], ...]:
    names = _bank(params, stm_names, "STM")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or generator is None:
        raise ValueError("probe count must be positive; an explicit generator is required")
    return tuple({name: (2 * torch.randint(2, params[name].shape,
                                          device=params[name].device, generator=generator) - 1
                         ).to(params[name].dtype)
                  for name in names} for _ in range(count))


def replay_ages(acquired_count: int, arrival_counts: Tensor, capacity: int) -> Tensor:
    if (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
            or isinstance(acquired_count, bool) or not isinstance(acquired_count, int)
            or acquired_count < 0):
        raise ValueError("capacity must be positive and acquired_count nonnegative integers")
    _finite(arrival_counts, "arrival counts")
    if (arrival_counts.ndim != 1 or arrival_counts.numel() == 0
            or bool(((arrival_counts < 0) | (arrival_counts > acquired_count)).any())
            or bool((arrival_counts != arrival_counts.floor()).any())):
        raise ValueError("arrival counts must be integral [B], between zero and acquired_count")
    return ((acquired_count - arrival_counts.detach().to(torch.float64)) / capacity).clamp(0, 1)


def stm_fd_penalty(predict: Predict, params: Params, design: LikelihoodDesign,
                   stm_names: Sequence[str], probes: Sequence[Params], ages: Tensor,
                   epsilon: float) -> Tensor:
    names = _bank(params, stm_names, "STM")
    epsilon = _scalar(epsilon, "epsilon", positive=True)
    epsilon_squared = epsilon * epsilon
    if not math.isfinite(epsilon_squared) or epsilon_squared == 0:
        raise ValueError("epsilon squared must be finite and nonzero")
    _finite(ages, "ages")
    if (ages.shape != design.baseline.shape or ages.device != design.baseline.device
            or bool(((ages < 0) | (ages > 1)).any()) or len(probes) == 0):
        raise ValueError("ages must be [B] in [0,1]; at least one probe is required")
    age_weights = ages.detach().to(design.baseline.dtype).square()
    dimension = sum(params[name].numel() for name in names)
    penalties = []
    for probe in probes:
        _matching(probe, params, names, "probe")
        plus, minus = dict(params), dict(params)
        for name in names:
            shift = (0.5 * epsilon) * probe[name].detach().to(params[name].dtype)
            plus[name], minus[name] = params[name] + shift, params[name] - shift
        penalties.append((age_weights * symmetric_kl(predict(plus), predict(minus), design)).mean()
                         / epsilon_squared / dimension)
    result = torch.stack(penalties).mean()
    _finite(result, "STM FD penalty")
    return result


def regularized_objective(
    params: Params, *, ltm_names: Sequence[str], stm_names: Sequence[str],
    ltm_anchor: Params, fisher_diagonal: Params, stm_anchor: Params,
    new_nll: Tensor | None = None, replay_nll: Tensor | None = None,
    pi: float = 0.025, ridge_coefficient: float = 0.0, numerical_floor: float = 0.0,
    fd_penalty: Tensor | None = None, fd_strength: float = 0.0,
) -> dict[str, Tensor]:
    ltm, stm = _bank(params, ltm_names, "LTM"), _bank(params, stm_names, "STM")
    if set(ltm) & set(stm):
        raise ValueError("LTM and STM must be disjoint")
    _matching(ltm_anchor, params, ltm, "LTM anchor")
    _matching(stm_anchor, params, stm, "STM preupdate anchor")
    _matching(fisher_diagonal, params, ltm, "Fisher diagonal", nonnegative=True)
    pi = _pi(pi)
    ridge_coefficient = _scalar(ridge_coefficient, "ridge coefficient")
    numerical_floor = _scalar(numerical_floor, "numerical floor")
    fd_strength = _scalar(fd_strength, "FD strength")
    means = []
    for batch in (new_nll, replay_nll):
        if batch is not None:
            if batch.ndim != 1 or batch.numel() == 0:
                raise ValueError("NLL batches must be nonempty [B], or None")
            means.append(_accumulate(batch).mean())
    if not means:
        raise ValueError("at least one data batch is required")
    data = torch.stack(means).mean()
    zero = data.new_zeros(())
    omega = (1 - pi) / pi
    diagonal = {name: _accumulate(fisher_diagonal[name].detach()) for name in ltm}
    displacement = {name: _accumulate(params[name]) - _accumulate(ltm_anchor[name].detach())
                    for name in ltm}
    fisher_scale = sum((value.sum() for value in diagonal.values()), zero) / sum(
        params[name].numel() for name in ltm)
    kappa = ridge_coefficient * fisher_scale.clamp_min(numerical_floor)
    ewc = 0.5 * omega * sum(((diagonal[name] * displacement[name].square()).sum()
                             for name in ltm), zero)
    distances = sum((value.square().sum() for value in displacement.values()), zero)
    distances = distances + sum(((_accumulate(params[name]) - _accumulate(
        stm_anchor[name].detach())).square().sum() for name in stm), zero)
    ridge = 0.5 * omega * kappa * distances
    if fd_penalty is None:
        if fd_strength > 0 and replay_nll is not None:
            raise ValueError("positive FD strength on replay requires an FD penalty")
        fd = zero
    else:
        if fd_penalty.ndim != 0:
            raise ValueError("FD penalty must be scalar")
        fd = fd_strength * _accumulate(fd_penalty)
    total = data + ewc + ridge + fd
    _finite(total, "regularized objective")
    return dict(total=total, data=data, ewc=ewc, ridge=ridge, fd=fd,
                fisher_scale=fisher_scale, kappa=kappa)


@torch.no_grad()
def eviction_ema(diagonal: Params, anchor: Params, accepted: Params,
                 evicted_fisher: Params | None, *, retired_count: int,
                 pi: float = 0.025, arrival_unit: float = 1.0,
                 ) -> tuple[dict[str, Tensor], dict[str, Tensor], float]:
    names = _bank(accepted, tuple(accepted), "accepted LTM")
    _matching(diagonal, accepted, names, "archive diagonal", nonnegative=True)
    _matching(anchor, accepted, names, "archive anchor")
    pi, arrival_unit = _pi(pi), _scalar(arrival_unit, "arrival unit", positive=True)
    if isinstance(retired_count, bool) or not isinstance(retired_count, int) or retired_count < 0:
        raise ValueError("retired_count must be a nonnegative integer")
    if retired_count == 0:
        if evicted_fisher is not None:
            raise ValueError("zero retirements must not supply an eviction Fisher")
        return ({name: value.detach().clone() for name, value in diagonal.items()},
                {name: value.detach().clone() for name, value in anchor.items()}, 0.0)
    if evicted_fisher is None:
        raise ValueError("positive retirements require an eviction Fisher")
    _matching(evicted_fisher, accepted, names, "eviction Fisher", nonnegative=True)
    eta = 1.0 if pi == 1 else -math.expm1(retired_count / arrival_unit * math.log1p(-pi))
    updated = {name: (1 - eta) * _accumulate(diagonal[name])
               + eta * _accumulate(evicted_fisher[name]) for name in names}
    for value in updated.values():
        _finite(value, "updated archive diagonal")
    return updated, {name: accepted[name].detach().clone() for name in names}, eta


def time_discounts(elapsed: Tensor, gamma0: float = 0.99,
                   reference_duration: float = 0.5) -> Tensor:
    elapsed = _accumulate(elapsed.detach())
    gamma0 = _scalar(gamma0, "gamma0", positive=True)
    reference_duration = _scalar(reference_duration, "reference duration", positive=True)
    if gamma0 > 1 or elapsed.numel() == 0 or bool((elapsed <= 0).any()):
        raise ValueError("gamma0 must be <= 1 and elapsed times must be positive")
    return torch.exp(elapsed / reference_duration * math.log(gamma0))


def shaped_rewards(elapsed: Tensor, potential_start: Tensor, potential_end: Tensor,
                    terminal_bonus: Tensor, terminated: Tensor, gamma0: float = 0.99,
                    reference_duration: float = 0.5) -> Tensor:
    """Plan 19 fixed rewards for equal, nonempty [T] recorded input tensors.

    Rewards are gamma * Phi_end - Phi_start - .02 * dt/dt0 + terminal_bonus.
    True task terminals force Phi_end=0; infrastructure interruptions must not
    be marked terminal. Potentials are finite in [0,1], including values masked
    at terminals. Bonuses are finite and zero on nonterminal records. Inputs
    are detached; output uses elapsed's accumulation dtype and device.
    """
    elapsed = _accumulate(elapsed.detach())
    if elapsed.ndim != 1 or elapsed.numel() == 0:
        raise ValueError("elapsed must be nonempty [T]")
    discounts = time_discounts(elapsed, gamma0, reference_duration)
    if (terminated.shape != elapsed.shape or terminated.dtype != torch.bool
            or terminated.device != elapsed.device):
        raise ValueError("terminated must be boolean [T] on the elapsed device")
    fixed = []
    for name, value in (("potential_start", potential_start), ("potential_end", potential_end),
                        ("terminal_bonus", terminal_bonus)):
        value = _accumulate(value.detach())
        if value.shape != elapsed.shape or value.device != elapsed.device:
            raise ValueError(f"{name} must have elapsed's [T] shape and device")
        if name != "terminal_bonus" and bool(((value < 0) | (value > 1)).any()):
            raise ValueError("potentials must be in [0,1]")
        fixed.append(value.to(elapsed.dtype))
    start, end, bonus = fixed
    if bool((bonus[~terminated] != 0).any()):
        raise ValueError("terminal_bonus must be zero on nonterminal records")
    end = torch.where(terminated, torch.zeros_like(end), end)
    result = discounts * end - start - 0.02 * elapsed / reference_duration + bonus
    _finite(result, "shaped rewards")
    return result


def discounted_returns(rewards: Tensor, discounts: Tensor, terminated: Tensor,
                       truncated: Tensor | None = None) -> Tensor:
    """Complete observed returns; interruption is not a zero-valued terminal."""
    rewards, discounts = _accumulate(rewards.detach()), _accumulate(discounts.detach())
    if rewards.ndim != 1 or rewards.numel() == 0 or discounts.shape != rewards.shape:
        raise ValueError("rewards and discounts must be equal nonempty [T] tensors")
    if discounts.device != rewards.device or bool(((discounts < 0) | (discounts > 1)).any()):
        raise ValueError("discounts must be on the rewards device and in [0,1]")
    if (terminated.shape != rewards.shape or terminated.dtype != torch.bool
            or terminated.device != rewards.device):
        raise ValueError("terminated must be boolean [T] on the rewards device")
    if truncated is not None:
        if (truncated.shape != rewards.shape or truncated.dtype != torch.bool
                or truncated.device != rewards.device):
            raise ValueError("truncated must be boolean [T] on the rewards device")
        if bool(truncated.any()):
            raise ValueError("infrastructure truncations have incomplete returns")
    if not bool(terminated[-1]):
        raise ValueError("final record must terminate; incomplete returns cannot be labeled")
    result = torch.empty_like(rewards)
    running = rewards.new_zeros(())
    for index in range(rewards.numel() - 1, -1, -1):
        running = rewards[index] + (0 if bool(terminated[index]) else discounts[index] * running)
        result[index] = running
    _finite(result, "discounted returns")
    return result


def _edges(edges: Tensor) -> Tensor:
    edges = _accumulate(edges.detach())
    if (edges.ndim not in (1, 2) or edges.shape[-1] < 3 or edges.numel() == 0
            or not bool((edges[..., 1:] > edges[..., :-1]).all())):
        raise ValueError("edges must be strictly increasing [K+1] or [B,K+1], K >= 2")
    return edges


def return_bin_values(edges: Tensor) -> Tensor:
    edges = _edges(edges)
    return edges[..., :-1] * 0.5 + edges[..., 1:] * 0.5


def bin_returns(returns: Tensor, edges: Tensor) -> Tensor:
    returns, edges = _accumulate(returns.detach()), _edges(edges)
    if returns.ndim != 1 or returns.numel() == 0 or returns.device != edges.device:
        raise ValueError("returns must be nonempty [B] on the edges device")
    if edges.ndim == 1:
        edges = edges.expand(returns.numel(), -1)
    if edges.shape[0] != returns.numel():
        raise ValueError("per-item edges must match the return batch")
    if bool(((returns < edges[:, 0]) | (returns > edges[:, -1])).any()):
        raise ValueError("return outside declared bin range; no silent clipping")
    indices = torch.searchsorted(edges.contiguous(), returns[:, None].contiguous(), right=True) - 1
    return indices.squeeze(1).clamp_max(edges.shape[1] - 2)


def teacher_finite_logits(probabilities: Tensor, probability_floor: float) -> tuple[Tensor, Tensor]:
    """Floor BEFORE execution; return executed probabilities and reference logits."""
    probabilities = _accumulate(probabilities.detach())
    floor = _scalar(probability_floor, "probability floor", positive=True)
    if (probabilities.ndim < 1 or probabilities.shape[-1] != ACTION_DIM + 1
            or probabilities.numel() == 0 or floor >= 1 / (ACTION_DIM + 1)
            or bool((probabilities < 0).any())
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)),
                                  rtol=1e-5, atol=1e-7)):
        raise ValueError("teacher probabilities must be normalized [...,8]; floor must be in (0,1/8)")
    executed = probabilities.clamp_min(floor)
    executed = executed / executed.sum(-1, keepdim=True)
    logits = executed[..., :ACTION_DIM].log() - executed[..., ACTION_DIM:].log()
    _finite(logits, "teacher logits")
    return executed, logits


def budget_averaged_failure(samples: Tensor, success: Tensor) -> Tensor:
    samples, success = _accumulate(samples.detach()), _accumulate(success.detach())
    if (samples.ndim != 1 or samples.numel() < 2 or success.shape != samples.shape
            or samples.device != success.device or samples[0] != 0
            or not bool((samples[1:] > samples[:-1]).all())
            or bool(((success < 0) | (success > 1)).any())):
        raise ValueError("need increasing checkpoints from zero and matching success rates in [0,1]")
    return torch.trapezoid(1 - success, samples) / samples[-1]


def efficiency_metrics(control_samples: float, fd_samples: float,
                        control_seconds: float, fd_seconds: float) -> dict[str, float]:
    """Cost of caller-supported matched crossings, without fitting/extrapolation."""
    control_samples = _scalar(control_samples, "control samples", positive=True)
    fd_samples = _scalar(fd_samples, "FD samples")
    control_seconds = _scalar(control_seconds, "control seconds", positive=True)
    fd_seconds = _scalar(fd_seconds, "FD seconds", positive=True)
    f = 1 - fd_samples / control_samples
    c = fd_seconds / control_seconds
    if not math.isfinite(c) or c <= 0:
        raise ValueError("compute ratio overflowed or underflowed")
    result = dict(f=f, c=c, S=f / c)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("efficiency ratios overflowed or underflowed")
    return result
