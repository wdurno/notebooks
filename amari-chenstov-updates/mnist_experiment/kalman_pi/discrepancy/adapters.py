"""Bounded CPU rendering cache behind the accepted-arrival learner interface."""

from __future__ import annotations

import torch

from .data import panel_for_role, render_panel


def stream_view(assets, name):
    stream = assets["streams"][name]
    names = list(stream["segments"])
    if len(names) != len(stream["ids"]):
        raise ValueError("stream segment counts differ from arrival count")
    return {"segments": names, "hash": stream["hash"]}


class BatchReader:
    """Cache fixed panels and one current panel; never retain all checkpoints."""

    def __init__(self, cfg, assets, raw_data, check=lambda: None):
        self.cfg, self.assets, self.raw_data, self.check = cfg, assets, raw_data, check
        self.cache = {}
        self._current_plan = None
        self._current_role = None
        self._current_cache = None

    def plan(self, role):
        if role.startswith("current:"):
            if role != self._current_role:
                self._current_plan = panel_for_role(self.assets, role)
                self._current_role = role
            return self._current_plan
        return panel_for_role(self.assets, role)

    def _render(self, plan, indices):
        if not len(indices):
            return render_panel(self.raw_data, plan, indices)
        xs, ys = [], []
        for chunk in indices.split(256):
            self.check()
            x, y = render_panel(self.raw_data, plan, chunk)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def __call__(self, role, positions=None):
        plan = self.plan(role)
        indices = (torch.arange(len(plan["ids"])) if positions is None
                   else torch.as_tensor(positions, dtype=torch.long, device="cpu").reshape(-1))
        key = "initialization" if role in ("init", "initial", "initialization") else role
        dynamic = role.startswith("current:")
        if dynamic:
            # Exact endpoint aliases share the same render and prediction identity.
            for name in ("old", "shifted", "shifted25", "shifted45"):
                if plan["hash"] == self.assets["panels"][name]["hash"]:
                    key = name
                    self._current_cache = None
                    break
            else:
                key = "current"
        if key in self.cache:
            all_x, all_y = self.cache[key]
            x, y = all_x[indices], all_y[indices]
        elif key == "current":
            if self._current_cache is None or self._current_cache[0] != plan["hash"]:
                rendered = self._render(plan, torch.arange(len(plan["ids"])))
                self._current_cache = plan["hash"], rendered
            all_x, all_y = self._current_cache[1]
            x, y = all_x[indices], all_y[indices]
        else:
            x, y = self._render(plan, indices)
            if positions is None and key in ("old", "shifted", "shifted25", "shifted45",
                                              "initialization", "validation"):
                self.cache[key] = x, y
        dtype = getattr(torch, self.cfg.dtype)
        result = {"inputs": x.to(device=self.cfg.device, dtype=dtype),
                  "targets": y.to(device=self.cfg.device, dtype=dtype),
                  "digits": plan["digits"][indices].to(self.cfg.device),
                  "ids": plan["ids"][indices]}
        if "weights" in plan:
            result["weights"] = plan["weights"][indices].to(device=self.cfg.device, dtype=torch.float64)
        return result

    def preload_initialization(self):
        self("initialization", None)
        self("validation", None)

    def preload_evaluation(self, roles=("old", "shifted")):
        for role in roles:
            self(role, None)
