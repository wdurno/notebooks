"""Bounded evaluation probes, isolated from replay, transport, and filtering."""

from __future__ import annotations

import copy
import time

import torch

from .curvature import fd_values
from .model import flat_prediction


def eviction_diagnostics(model, layout, theta, reader, previous, evicted, novel_flags,
                         step, *, cfg, replay_x, probes, check=lambda: None):
    """At most eight remembered IDs/predictions; their pixels never reenter learning."""
    records = copy.deepcopy(previous)
    counts = {name: sum(r["domain"] == name for r in records) for name in ("novel", "old")}
    added, probes_due = [], []
    if step > cfg.hold_steps:
        for index in evicted:
            domain = "novel" if bool(novel_flags[index]) else "old"
            if counts[domain] < 4:
                counts[domain] += 1
                added.append(dict(index=index, domain=domain, evicted_step=step, checks=[]))
    with torch.no_grad():
        if added:
            check()
            x, y = reader.stream([r["index"] for r in added])
            prediction = flat_prediction(model, layout, theta, x)
            for record, pred, target in zip(added, prediction, y):
                record.update(reference_prediction=pred.detach().cpu(),
                    eviction_mse=float((pred.double()-target.double()).reshape(27,2).square().sum(-1).mean()))
                records.append(record)
        for record in records:
            delay = step-record["evicted_step"]
            if delay in (32,128) and delay not in {r["delay"] for r in record["checks"]}:
                probes_due.append((record, delay))
        if probes_due:
            check()
            x, y = reader.stream([r["index"] for r, delay in probes_due])
            prediction = flat_prediction(model, layout, theta, x)
            muted = flat_prediction(model, layout, theta, x, stm_scale=0.)
            for (record, delay), pred, target, ablated in zip(probes_due, prediction, y, muted):
                record["checks"].append(dict(delay=delay,
                    mse=float((pred.double()-target.double()).reshape(27,2).square().sum(-1).mean()),
                    reference_kl=float((pred.double()-record["reference_prediction"].to(pred.device).double()).square().sum()/8),
                    stm_ablation_kl=float((pred.double()-ablated.double()).square().sum()/8)))
        sensitivity = None
        if step % cfg.eval_every == 0 and len(replay_x):
            check()
            values=fd_values(model,layout,theta.double(),replay_x.double(),probes.double(),cfg.fd_step)
            sensitivity=float(values.mean()/probes.shape[1])
    summary=dict(stm_sensitivity=sensitivity,
                 eviction_mse=sum(r["eviction_mse"] for r in added)/len(added) if added else None,
                 evicted_probe_count=len(records))
    return records,summary
