"""Production-shaped timing only, using cached acceptance initialization.

Call benchmark(runner) after acceptance; this never starts pretraining. These
single solves are not pilots: theta/Fisher quality, strengths, convergence and
method order are not representative of a complete production trajectory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import torch

from .config import CONDITIONS, Treatment, content_hash
from .curvature import reference_fisher
from .initialize import fit_settings
from .learner import age_weights, fit_step
from .model import effective_residuals, memory_indices


def _timed(device, operation):
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    synchronize()
    started = time.perf_counter()
    value = operation()
    synchronize()
    return value, time.perf_counter() - started


def benchmark(runner) -> dict:
    """Persist a resumable kind=benchmark unit; perform no scientific selection."""
    cfg, store, source = runner.cfg, runner.store, runner.source
    device, dtype = torch.device(cfg.device), getattr(torch, cfg.dtype)
    asset_id = "assets-acceptance-rotation-0000"
    seeds = {name: cfg.seed_for("benchmark", 0, name) for name in ("training_inputs", "evaluation_inputs", "fit_probes")}
    contract = dict(schema_version=1, kind="benchmark", asset_id=asset_id, seeds=seeds,
                    replay_capacity=32, arrival_size=4, max_iter=50, max_eval=75, probes=2,
                    evaluation_per_class=100, condition_order=list(CONDITIONS), strength=.1,
                    device=cfg.device, dtype=cfg.dtype, config_hash=cfg.scientific_hash,
                    source_hash=store.identity.get("source_hash"),
                    benchmark_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    contract["provenance_hash"] = content_hash(contract)
    unit = "benchmark-production-shapes-" + contract["provenance_hash"][:16]
    if store.unit_path(unit).exists() and store.is_complete(unit):
        return json.loads((store.unit_path(unit) / "result.json").read_text())
    if not store.unit_path(asset_id).exists() or not store.is_complete(asset_id):
        raise ValueError("benchmark requires completed acceptance rotation assets; no pretraining is permitted")
    store.begin(unit, contract)
    store.check()
    (_, shared), asset_seconds = _timed(device, lambda: runner._assets("acceptance", "rotation_linear", 0))
    model, layout = runner._model(shared)
    ltm, stm = memory_indices(layout)
    if (len(ltm), len(stm)) != (487, 176):
        raise ValueError("benchmark requires the production 487-LTM/176-STM parameter chart")
    plan = shared["assets"]["plan"]
    available = torch.ones(len(source.train_labels), dtype=torch.bool)
    available[plan["initialization_ids"] + plan["reference_ids"] + plan["stream_ids"]] = False
    candidates = torch.where(available)[0]
    if len(candidates) < 36:
        raise ValueError("benchmark requires 36 unused cached training images")
    generator = torch.Generator().manual_seed(seeds["training_inputs"])
    train_ids = candidates[torch.randperm(len(candidates), generator=generator)[:36]].tolist()
    generator = torch.Generator().manual_seed(seeds["evaluation_inputs"])
    evaluation_ids = []
    for label in range(10):
        candidates = torch.where(source.test_labels == label)[0]
        if len(candidates) < 100:
            raise ValueError("benchmark requires 100 cached test images per class")
        evaluation_ids.extend(candidates[torch.randperm(len(candidates), generator=generator)[:100]].tolist())
    x, y = source.images(train_ids).to(device=device, dtype=dtype), source.train_labels[train_ids].to(device)
    evaluation = source.images(evaluation_ids, test=True)
    ages = age_weights(torch.arange(4, 33, 4).repeat_interleave(4), 36, 32, device=device, dtype=dtype)
    settings = dataclasses.replace(fit_settings(cfg, "rotation_linear", Treatment.default(cfg, "replay_ewc")),
                                   solver="lbfgs", max_iter=50, max_eval=75, num_probes=2, chunk_size=16)
    state = store.load_checkpoint(unit)
    if state is None:
        theta = shared["learner"]["theta"].to(device=device, dtype=dtype).clone()
        warm_seconds, warmed = 0., False
        if not any(bool(value.count_nonzero()) for value in effective_residuals(model, layout, theta).values()):
            (theta, _), warm_seconds = _timed(device, lambda: fit_step(
                model, layout, theta, x[32:], y[32:], x[:32], y[:32], ages,
                shared["learner"]["fisher_L"], shared["learner"]["anchor_L"], "replay_ewc",
                dataclasses.replace(settings, max_iter=5, max_eval=8), seeds["fit_probes"], store.check))
            warmed = True
        active_stm = any(bool(value.count_nonzero()) for value in effective_residuals(model, layout, theta).values())
        if not active_stm:
            raise ValueError("bounded warm-up left STM residual zero; timing would not exercise the intended workload")
        result = dict(contract, methods={}, training_ids=train_ids, evaluation_ids=evaluation_ids,
                      shared_asset_load_seconds=asset_seconds, warmup_seconds=warm_seconds, warmed=warmed,
                      shared_initialization=dict(samples=len(plan["initialization_ids"]),
                                                 reference_samples=len(plan["reference_ids"]), low_fidelity=True),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      caveats=["Timing only; acceptance theta/Fisher, not predictive evidence or a trajectory forecast.",
                               "One solve per method; fixed order, no timing uncertainty or production convergence guarantee.",
                               "Strength 0.1 is uncalibrated; compute-matched replay uses the same 50/75 kernel budget.",
                               "Initial Fisher is reused; EWC anchor is the shared warmed LTM start.",
                               "Excludes downloads/pretraining, archive compression, I/O, rotations and cohort diagnostics."])
        state = dict(theta=theta.detach().cpu(), result=result)
        store.save_checkpoint(unit, state)
    theta, result = state["theta"].to(device=device, dtype=dtype), state["result"]
    for condition in CONDITIONS:
        store.check()
        if condition in result["methods"]:
            continue
        local = dataclasses.replace(settings, lambda_f=.1 if condition in ("jvp", "finite_difference", "directional") else 0.,
                                    lambda_d=.1 if condition == "distillation" else 0.)
        (_, diagnostics), seconds = _timed(device, lambda: fit_step(
            model, layout, theta.clone(), x[32:], y[32:], x[:32], y[:32], ages,
            shared["learner"]["fisher_L"], theta[ltm.to(device)].clone(), condition, local, seeds["fit_probes"], store.check))
        result["methods"][condition] = dict(seconds=seconds, diagnostics=diagnostics)
        store.save_checkpoint(unit, state)
    for name, operation in (
        ("eviction_fisher_4_seconds", lambda: reference_fisher(model, layout, theta, x[:4], indices=ltm, chunk_size=4)),
        ("evaluation_1000_seconds", lambda: runner._logits(model, layout, theta, evaluation)),
    ):
        store.check()
        if name not in result:
            _, result[name] = _timed(device, operation)
            store.save_checkpoint(unit, state)
    store.write_json(unit, "result.json", result)
    store.complete(unit)
    return result
