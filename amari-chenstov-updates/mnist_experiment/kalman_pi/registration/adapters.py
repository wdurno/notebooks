"""Render sealed plans through the narrow learner batch interface."""

from __future__ import annotations

import torch

from .data import render_panel


def stream_view(assets, name):
    stream = assets["streams"][name]
    names = list(stream["segments"])
    if len(names) != len(stream["ids"]):
        raise ValueError("stream segment counts differ from arrival count")
    return {"segments": names, "hash": stream["hash"]}


class BatchReader:
    """Inputs/targets only flow into the model after the caller freezes actions."""

    def __init__(self, cfg, assets, raw_data, check=lambda: None):
        self.cfg, self.assets, self.raw_data, self.check = cfg, assets, raw_data, check
        self.cache = {}

    def plan(self, role):
        if role in ("main", "stationary"):
            stream = self.assets["streams"][role]
            return {"ids": stream["ids"].reshape(-1), "transforms": stream["transforms"].reshape(-1, 6),
                    "digits": stream["digits"].reshape(-1), "test": False}
        return self.assets["panels"]["init" if role == "initialization" else role]

    def __call__(self, role, positions=None):
        plan = self.plan(role)
        indices = torch.arange(len(plan["ids"])) if positions is None else positions.long().cpu().reshape(-1)
        if role in self.cache:
            all_x, all_y = self.cache[role]
            x, y = all_x[indices], all_y[indices]
        else:
            xs, ys = [], []
            for chunk in indices.split(256):
                self.check()
                x, y = render_panel(self.raw_data, plan, chunk)
                xs.append(x)
                ys.append(y)
            x, y = torch.cat(xs), torch.cat(ys)
            if positions is None and role in ("old", "shifted", "initialization", "validation"):
                self.cache[role] = x, y
        dtype = getattr(torch, self.cfg.dtype)
        return {"inputs": x.to(device=self.cfg.device, dtype=dtype),
                "targets": y.to(device=self.cfg.device, dtype=dtype),
                "digits": plan["digits"][indices].to(self.cfg.device),
                "ids": plan["ids"][indices]}

    def preload_initialization(self):
        self("initialization", None)
        self("validation", None)
