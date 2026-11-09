"""Observe the actual inherited Kalman gains without changing its recursion."""

from __future__ import annotations

import time

import torch

from mnist_experiment.kalman_pi.policies import Controller as OriginalController


class Controller(OriginalController):
    def assimilate(self, scores, step):
        if not self.spec.movement:
            return super().assimilate(scores, step)
        started = time.perf_counter()
        noise = self.noise() / len(scores)
        floor = max(1e-30, float(noise.mean()) * 1e-12)
        noise = noise.clamp_min(floor)
        transition = self.state["mean"].new_tensor([[1., 1.], [0., 1.]])
        predicted = transition @ self.state["covariance"] @ transition.T + self.process_covariance()
        variance = predicted[:, 0, 0] + noise
        gain = predicted[:, :, 0] / variance[:, None]
        diagnostics = {}
        for i, name in enumerate(("level", "velocity")):
            values = gain[:, i]
            diagnostics[f"kalman_{name}_gain_mean"] = float(values.mean())
            for quantile in (.1, .9):
                diagnostics[f"kalman_{name}_gain_p{round(100*quantile)}"] = float(
                    torch.quantile(values, quantile))
        diagnostics.update(kalman_process_scale=self.spec.value,
            kalman_measurement_variance_mean=float(noise.mean()),
            kalman_measurement_variance_floor=floor,
            gain_logging_seconds=time.perf_counter()-started)
        result = super().assimilate(scores, step)
        return {**result, **diagnostics}
