"""Parameter-free, separately versioned remedies for saturated student inputs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from numerical_experiment.picar.models import PiCarStudent


MODEL_VERSION = "phase2-student-rms-v1"
MODES = ("feature_rms", "feature_and_projection_rms")


class NormalizedStudent(PiCarStudent):
    """Reuse a freshly initialized student's registered modules, without copying.

    The original student and wrapper share tensors; use only the wrapper after
    construction. Encoder features remain raw for teacher/evaluator roles and
    caches. Controls keep their original SI units. RMS is per timestep/channel
    vector, with gradients through both denominators and no learned scale.
    """

    def __init__(self, base, mode="feature_rms", eps=1e-6):
        if not isinstance(base, PiCarStudent) or isinstance(base, NormalizedStudent):
            raise TypeError("wrap a fresh, unnormalized PiCarStudent")
        if mode not in MODES:
            raise ValueError(f"normalization mode must be one of {MODES}")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS epsilon must be finite and positive")
        if base.distillation_receipt is not None or base.encoder.publication != 0:
            raise ValueError("normalization requires fresh initialization, not a trained student")
        # Do not call PiCarStudent.__init__: that would allocate and randomize
        # replacement heads, and could duplicate the already loaded Qwen.
        nn.Module.__init__(self)
        for name, module in base.named_children():
            self.add_module(name, module)
        self.hidden_size, self.return_bins = base.hidden_size, base.return_bins
        self.distillation_receipt = None
        self.mode, self.eps = mode, float(eps)
        self.provenance = {"version": MODEL_VERSION, "mode": mode, "eps": self.eps,
            "parameter_free": True, "feature_axis": -1, "controls_normalized": False}
        self.train(base.training)

    def _run(self, batch, features, state=None):
        b, t = features.shape[:2]
        reference = self.projection.ltm.weight
        features = features.to(reference.device, reference.dtype)
        features = F.rms_norm(features, (features.shape[-1],), eps=self.eps)
        control = torch.cat((batch["executed_actions"], batch["elapsed"]), -1).to(features)
        projected = self.projection(torch.cat((features, control), -1))
        if self.mode == "feature_and_projection_rms":
            projected = F.rms_norm(projected, (projected.shape[-1],), eps=self.eps)
        x = projected.tanh()
        h, c = (features.new_zeros(b, self.hidden_size), features.new_zeros(b, self.hidden_size)) \
            if state is None else (state.h.to(features), state.c.to(features))
        for j in range(0 if state is None else t - 1, t):
            h, c = self.recurrent(x[:, j], (h, c))
        return {"return_logits": self.return_head(h), "action_mean": self.action_head(h)}, (h, c)

    def get_extra_state(self):
        return {**super().get_extra_state(), "normalization": dict(self.provenance)}

    def set_extra_state(self, state):
        if state.get("normalization") != self.provenance:
            raise ValueError("student normalization identity differs")
        super().set_extra_state(state)
