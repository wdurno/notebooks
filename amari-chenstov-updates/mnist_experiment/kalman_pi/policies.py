"""Causal diagonal common-anchor observers and fixed-batch risk actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .observations import score_moments


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    value: float = 0.

    @property
    def condition(self):
        if self.kind in ("fixed", "covariance"):
            return self.kind
        return ("ema_h" if self.kind == "ema" else
                "kalman_mean_eta" if self.kind == "kalman_mean" else "kalman_eta") + f"{self.value:g}"

    @property
    def movement(self):
        return self.kind in ("ema", "kalman", "kalman_mean")

    @classmethod
    def parse(cls, condition):
        if condition in ("fixed", "covariance"):
            return cls(condition)
        for prefix, kind in (("kalman_mean_eta", "kalman_mean"), ("kalman_eta", "kalman"), ("ema_h", "ema")):
            if condition.startswith(prefix):
                value = float(condition[len(prefix):])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("positive finite observer setting required")
                return cls(kind, value)
        raise ValueError(f"unknown policy: {condition}")


def concentration(q, pi, batch_size):
    if q < 0 or not 0 < pi < 1 or batch_size < 1:
        raise ValueError("invalid weight concentration update")
    return (1 - pi)**2 * q + pi**2 / batch_size


def covariance_action(q, batch_size):
    if not math.isfinite(q) or q < 0 or batch_size < 1:
        raise ValueError("finite nonnegative concentration and positive batch size required")
    return q / (q + 1 / batch_size)


def symmetric_psd(matrix):
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    diagonal = matrix.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    bound = (diagonal[:, 0] * diagonal[:, 1]).sqrt()
    cross = matrix[:, 0, 1].clamp(-bound, bound)
    repaired = (matrix[:, 0, 0] < 0) | (matrix[:, 1, 1] < 0) | (matrix[:, 0, 1].abs() > bound)
    result = torch.stack((diagonal[:, 0], cross, cross, diagonal[:, 1]), -1).reshape(-1, 2, 2)
    return result, float(repaired.double().mean())


class Controller:
    """The choose API deliberately accepts no batch, prevalence, or segment."""

    def __init__(self, spec, cfg, fixed_c, state=None):
        self.spec, self.cfg, self.fixed_c = spec, cfg, fixed_c
        self.state = state if state is not None else {"assimilations": 0, "refreshes": 0}

    def initialize(self, anchor, diagonal, variance):
        if not self.spec.movement:
            return
        anchor, diagonal, variance = anchor.double(), diagonal.double(), variance.double()
        if not (torch.isfinite(variance).all() and (variance >= 0).all()):
            raise ValueError("finite nonnegative calibration variance required")
        floor = max(self.cfg.observer_floor_absolute,
                    self.cfg.observer_floor_relative * float(diagonal.mean()))
        mapping = diagonal.clamp_min(floor)
        k0 = variance / mapping.square()
        covariance = torch.zeros((len(anchor), 2, 2), dtype=torch.float64, device=anchor.device)
        covariance[:, 0, 0] = k0 / self.cfg.initialization_size
        covariance[:, 1, 1] = k0 / self.cfg.initialization_size
        self.state.update(anchor=anchor.clone(), anchor_diagonal=diagonal.clone(), mapping=mapping,
                          mean=torch.zeros((len(anchor), 2), dtype=torch.float64, device=anchor.device),
                          covariance=covariance, raw_variance=variance.clone(), k0=k0,
                          floor=floor, floor_fraction=float((diagonal < floor).double().mean()),
                          noise_updates=0, mapping_error_ratio=0., covariance_repairs=0.)

    def noise(self):
        return self.state["raw_variance"] / self.state["mapping"].square()

    def process_covariance(self):
        k0 = self.state["k0"]
        coefficient = self.spec.value if self.spec.kind.startswith("kalman") else 0.
        shape = k0.new_tensor([[1 / 3, 1 / 2], [1 / 2, 1.]])
        return coefficient * k0[:, None, None] * shape

    def choose(self, q, risk_diagonal, batch_size, step):
        cov = covariance_action(q, batch_size)
        mean_energy = uncertainty = 0.
        variance_scale = None
        fallback = ""
        raw = self.fixed_c if self.spec.kind == "fixed" else cov
        if self.spec.movement:
            metric = risk_diagonal.double()
            variance_scale = float((metric * self.noise()).sum())
            mean_energy = float((metric * self.state["mean"][:, 1].square()).sum())
            if self.spec.kind.startswith("kalman"):
                movement_variance = self.state["covariance"][:, 1, 1] + self.process_covariance()[:, 0, 0]
                uncertainty = float((metric * movement_variance).sum())
            movement = mean_energy + (uncertainty if self.spec.kind == "kalman" else 0.)
            if not all(math.isfinite(v) and v >= 0 for v in (movement, variance_scale)) or variance_scale <= 1e-30:
                fallback = "unsupported_movement_or_noise"
            else:
                # Scale both risk terms before division; a huge finite signal
                # must saturate the upper bound, not produce inf/inf -> NaN.
                scale = max(movement, variance_scale)
                signal = movement / scale + q * (variance_scale / scale)
                raw = signal / (signal + (variance_scale / scale) / batch_size)
        cold = self.spec.kind != "fixed" and step < self.cfg.cold_start
        applied = self.fixed_c if cold or self.spec.kind == "fixed" else min(self.cfg.pi_max, max(self.cfg.pi_min, raw))
        result = {"raw_pi": raw, "applied_pi": applied, "cold_start": cold,
                "motion_energy": mean_energy, "uncertainty_energy": uncertainty,
                "noise_scale": variance_scale,
                "rho": batch_size * (mean_energy + (uncertainty if self.spec.kind == "kalman" else 0.)) / variance_scale
                if variance_scale is not None and variance_scale > 1e-30 and not fallback else None,
                "boundary_low": not cold and self.spec.kind != "fixed" and raw < self.cfg.pi_min,
                "boundary_high": not cold and self.spec.kind != "fixed" and raw > self.cfg.pi_max,
                "fallback": fallback, "mapping_error_ratio": self.state.get("mapping_error_ratio", 0.),
                "observer_floor_fraction": self.state.get("floor_fraction", 0.)}
        for name in ("motion_energy", "uncertainty_energy", "noise_scale", "rho", "mapping_error_ratio"):
            if result[name] is not None and not math.isfinite(result[name]):
                result[name] = None
        return result

    def assimilate(self, scores, step):
        if not self.spec.movement:
            return {}
        if step != self.state["assimilations"]:
            raise ValueError("observer batch was duplicated or skipped")
        summary, variance = score_moments(scores)
        measurement = summary / self.state["mapping"]
        noise = self.noise() / len(scores)
        noise_floor = max(1e-30, float(noise.mean()) * 1e-12)
        noise = noise.clamp_min(noise_floor)
        state, prior = self.state["mean"], self.state["covariance"]
        transition = state.new_tensor([[1., 1.], [0., 1.]])
        predicted = state @ transition.T
        predicted_covariance = transition @ prior @ transition.T + self.process_covariance()
        predictive_variance = predicted_covariance[:, 0, 0] + noise
        residual = measurement - predicted[:, 0]
        if self.spec.kind == "ema":
            alpha = 1 - 2**(-1 / self.spec.value)
            gain = state.new_tensor([alpha, alpha**2]).expand(len(state), 2)
        else:
            gain = predicted_covariance[:, :, 0] / predictive_variance[:, None]
        updated = predicted + gain * residual[:, None]
        identity_minus_gain = torch.eye(2, dtype=torch.float64, device=state.device).expand(len(state), 2, 2).clone()
        identity_minus_gain[:, :, 0] -= gain
        # Joseph form preserves PSD without confusing state covariance and Fisher.
        posterior = identity_minus_gain @ predicted_covariance @ identity_minus_gain.transpose(-1, -2)
        posterior += noise[:, None, None] * gain[:, :, None] * gain[:, None, :]
        posterior, repairs = symmetric_psd(posterior)
        if not torch.isfinite(updated).all() or not torch.isfinite(posterior).all():
            raise FloatingPointError("nonfinite observer update")
        self.state["mean"], self.state["covariance"] = updated, posterior
        self.state["assimilations"] += 1
        self.state["covariance_repairs"] += repairs
        if variance is not None:
            weight = 1 - 2**(-1 / self.cfg.noise_half_life)
            self.state["raw_variance"] = (1 - weight) * self.state["raw_variance"] + weight * variance
            self.state["noise_updates"] += 1
        return {"innovation_log_score": float((-.5 * (math.log(2 * math.pi) + predictive_variance.log() +
                                                       residual.square() / predictive_variance)).mean()),
                "innovation_ratio": float((residual.square() / predictive_variance).mean()),
                "measurement_variance_fallback": variance is None,
                "covariance_repair_fraction": repairs}

    def reanchor(self, anchor, diagonal, old_scores, new_scores, risk_diagonal):
        if not self.spec.movement:
            return {}
        anchor, diagonal = anchor.double(), diagonal.double()
        old_mapping = self.state["mapping"]
        old_noise = self.noise()
        floor = max(self.cfg.observer_floor_absolute,
                    self.cfg.observer_floor_relative * float(diagonal.mean()))
        mapping = diagonal.clamp_min(floor)
        shift = self.state["anchor"] - anchor
        # Translation leaves physical location, velocity and covariance unchanged.
        self.state["mean"][:, 0] += shift
        paired_error = new_scores / mapping - old_scores / old_mapping - shift
        mean_error, error_variance = score_moments(paired_error)
        metric = risk_diagonal.double()
        numerator = float((metric * mean_error.square()).sum())
        denominator = float((metric * error_variance / len(new_scores)).sum()) if error_variance is not None else 0.
        ratio = numerator / max(denominator, 1e-30)
        _, variance = score_moments(new_scores)
        self.state.update(anchor=anchor.clone(), anchor_diagonal=diagonal.clone(), mapping=mapping,
                          floor=floor, floor_fraction=float((diagonal < floor).double().mean()),
                          raw_variance=variance.clone() if variance is not None else old_noise * mapping.square(),
                          noise_updates=0, mapping_error_ratio=ratio,
                          refreshes=self.state["refreshes"] + 1)
        return {"mapping_error_ratio": ratio, "mapping_error_energy": numerator,
                "mapping_error_noise": denominator, "anchor_refresh": True,
                "measurement_variance_fallback": variance is None}
