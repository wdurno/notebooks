"""Predictable velocity and conditional-discrepancy recommendations.

The inherited observer consumes observed-label scores s=-g. Its covariance
is a state-estimation covariance, never a second likelihood Fisher.
"""

from __future__ import annotations

import math

import torch

from mnist_experiment.kalman_pi.policies import (
    Controller as BaseController, PolicySpec as BaseSpec, covariance_action,
)


class PolicySpec:
    def __init__(self, kind, value=None):
        if kind not in ("fixed", "covariance", "velocity", "discrepancy"):
            raise ValueError("unknown Phase 7 policy")
        if kind in ("velocity", "discrepancy"):
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError("Kalman process scale must be finite and positive")
        elif value is not None:
            raise ValueError("fixed/covariance have no process scale")
        self.kind, self.value = kind, value

    @property
    def movement(self):
        return self.kind in ("velocity", "discrepancy")

    @property
    def condition(self):
        return f"{self.kind}_eta{self.value:g}" if self.movement else self.kind

    @classmethod
    def parse(cls, text):
        if "_eta" in text:
            kind, value = text.split("_eta", 1)
            return cls(kind, float(value))
        return cls(text)


class Controller(BaseController):
    def __init__(self, spec, cfg, fixed_c, state=None):
        self.phase7_spec = spec
        inherited = BaseSpec("kalman", spec.value) if spec.movement else BaseSpec(spec.kind)
        super().__init__(inherited, cfg, fixed_c, state)

    def choose(self, q, risk_diagonal, batch_size, step, theta):
        if not self.phase7_spec.movement:
            result = super().choose(q, risk_diagonal, batch_size, step)
            result.update(discrepancy_energy=0., level_uncertainty_energy=0.)
            return result
        metric = risk_diagonal.double()
        theta = theta.double()
        if theta.shape != metric.shape or not torch.isfinite(theta).all():
            raise ValueError("current learner must match the observer parameter chart")
        mean, covariance = self.state["mean"], self.state["covariance"]
        process = self.process_covariance()
        displacement = self.state["anchor"] + mean[:, 0] + mean[:, 1] - theta
        # Predict the level from the full 2x2 covariance, retaining its cross term.
        level_variance = (covariance[:, 0, 0] + 2 * covariance[:, 0, 1]
                          + covariance[:, 1, 1] + process[:, 0, 0]).clamp_min(0)
        discrepancy = float((metric * displacement.square()).sum())
        level_uncertainty = float((metric * level_variance).sum())
        velocity = float((metric * mean[:, 1].square()).sum())
        velocity_uncertainty = float((metric * (covariance[:, 1, 1] + process[:, 0, 0])).sum())
        noise = float((metric * self.noise()).sum())
        covariance_pi = covariance_action(q, batch_size)

        def recommendation(kind):
            a = (discrepancy + level_uncertainty if kind == "discrepancy"
                 else velocity + velocity_uncertainty + q * noise)
            if not all(math.isfinite(x) and x >= 0 for x in (a, noise)) or noise <= 1e-30:
                return covariance_pi, "unsupported_movement_or_noise"
            scale = max(a, noise)
            raw = (a / scale) / (a / scale + (noise / scale) / batch_size)
            return raw, ""

        raw, fallback = recommendation(self.phase7_spec.kind)
        cold = step < self.cfg.cold_start
        applied = self.fixed_c if cold else min(self.cfg.pi_max, max(self.cfg.pi_min, raw))
        result = dict(raw_pi=raw, applied_pi=applied, cold_start=cold,
            motion_energy=velocity, uncertainty_energy=velocity_uncertainty,
            discrepancy_energy=discrepancy, level_uncertainty_energy=level_uncertainty,
            noise_scale=noise, boundary_low=not cold and raw < self.cfg.pi_min,
            boundary_high=not cold and raw > self.cfg.pi_max, fallback=fallback,
            observer_floor_fraction=self.state.get("floor_fraction", 0.),
            mapping_error_ratio=self.state.get("mapping_error_ratio", 0.),
            shadow_velocity_pi=recommendation("velocity")[0],
            shadow_discrepancy_pi=recommendation("discrepancy")[0],
            rho=batch_size * (discrepancy + level_uncertainty if self.phase7_spec.kind == "discrepancy"
                              else velocity + velocity_uncertainty) / noise if noise > 1e-30 else None)
        for key, value in list(result.items()):
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
        return result
