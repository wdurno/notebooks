"""Plan 19 resource/readiness probes, with immutable resumable units.

This entry point stops at Phase 1. Probe updates use fixed diagnostic labels;
they measure the cost of the declared learner and are never learning evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .config import PicarConfig, canonical_json, load_config
from .runtime import StopController, check_storage, source_fingerprint
from .storage import RunStore


VERSION = "phase1-v1"
PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Recipe:
    top_layers: int = 1
    hidden_size: int = 64
    ltm_rank: int = 4
    stm_rank: int = 1
    context_length: int = 8
    batch: int = 1
    fd_probes: int = 1
    fisher_draws: int = 1
    image_width: int = 160
    image_height: int = 120
    max_pixels: int = 160 * 120
    max_tokens: int = 2048
    allocator: str = "expandable_segments:True"
    frozen_prefix_cache: bool = False

    @property
    def key(self):
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()[:12]


def resolved(recipe: Recipe) -> PicarConfig:
    base = load_config(PROJECT / "numerical_experiment/picar/configs/plan19.json")
    return replace(base, version=VERSION,
        image_width=recipe.image_width, image_height=recipe.image_height,
        model=replace(base.model, top_layers=recipe.top_layers, hidden_size=recipe.hidden_size,
            ltm_rank=recipe.ltm_rank, stm_rank=recipe.stm_rank, context_length=recipe.context_length),
        learning=replace(base.learning, replay_batch=recipe.batch,
            fd_probes=recipe.fd_probes, fisher_draws=recipe.fisher_draws))


def probe_sources() -> dict:
    names = {"models", "teacher", "learner", "config", "runtime", "storage", "schemas",
             "phase1", "picar_math", "simulator", "protocol", "actions", "geometry", "drift"}
    return {path: checksum for path, checksum in source_fingerprint().items() if Path(path).stem in names}


def _identity(recipe, model_path, kind):
    return {"config": resolved(recipe).to_dict(), "phase1": {"kind": kind, "recipe": asdict(recipe)},
        "component_seeds": resolved(recipe).seeds(0), "source_sha256": probe_sources(),
        "model_source": str(model_path.resolve()) if model_path else resolved(recipe).model.model_id}


def _histories(recipe: Recipe):
    import numpy as np
    import torch
    from src.robotics.protocol import Control
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig

    images, actions, elapsed = [], [], []
    with PyBulletBackend(SimulatorConfig(width=recipe.image_width, height=recipe.image_height,
            control_interval_s=.5, realtime=False), run_id="phase1-probe") as sim:
        observation = sim.reset(seed=190101)
        for _ in range(recipe.context_length):
            image = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(recipe.image_height, recipe.image_width, 3)
            images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255)
            actions.append(tuple(asdict(observation.executed_controls).values()))
            elapsed.append(.5)
            observation = sim.step(Control(drive_mps=.12)).observation
    history = {"images": torch.stack(images)[None],
        "executed_actions": torch.tensor(actions)[None],
        "elapsed": torch.tensor(elapsed)[None, :, None],
        "contexts": [[{"text": "Go around the blue obstacle, approach the red ball, and stop.", "issued_at": 0.}]]}
    # All profile inputs are declared synthetic histories; no accuracy inference
    # uses the cloned batch. Distinct elapsed clocks avoid whole-input caching.
    return history


def _timed(function, *, repeats=3):
    import torch
    samples, last = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
            "samples": samples}, last


def run_probe(output: Path, recipe: Recipe, *, model_path=None, offline=False,
              resume=False, repeats=3, min_headroom_bytes=1_500_000_000):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", recipe.allocator)
    import torch
    from src.continual import picar_math as pm
    from src.robotics.actions import ACTION_NAMES
    from .learner import LearningBatch, MemoryLearner
    from .models import load_qwen_model
    from .teacher import FrozenQwenEvaluator, FrozenQwenTeacher

    identity = _identity(recipe, model_path, "phase1_resource_probe")
    with RunStore(output, identity, resume=resume) as store:
        if store.completed:
            return json.loads((output / "summary.json").read_text())
        started = time.monotonic()
        summary = {"schema_version": 1, "phase": 1, "smoke_only": True,
            "kind": "phase1_resource_probe", "recipe": asdict(recipe), "status": "running",
            "timings": {}, "memory": {}, "reason": None, "checks": {}}
        store.checkpoint({"stage": "load"}, progress={"phase": 1, "status": "loading", "counts": {"step": 0}})
        model = learner = None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for real-model profiling")
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            free, total = torch.cuda.mem_get_info()
            outside = total - free
            if free < 8 * 1024**3:
                raise RuntimeError("less than 8 GiB CUDA memory free before model load")
            torch.cuda.reset_peak_memory_stats()
            kwargs = {"seed": 191101, "top_layers": recipe.top_layers,
                "hidden_size": recipe.hidden_size, "ltm_rank": recipe.ltm_rank,
                "stm_rank": recipe.stm_rank, "context_length": recipe.context_length,
                "return_bins": 21, "max_tokens": recipe.max_tokens,
                "min_pixels": 56 * 56, "max_pixels": recipe.max_pixels,
                "local_files_only": offline or model_path is not None}
            if model_path is not None:
                kwargs["model_path"] = model_path
            t = time.monotonic()
            model = load_qwen_model(**kwargs).eval()
            if recipe.frozen_prefix_cache:
                if not hasattr(model.encoder, "enable_prefix_cache"):
                    raise RuntimeError("frozen prefix cache is unavailable in this implementation")
                model.encoder.enable_prefix_cache = True
            torch.cuda.synchronize()
            summary["timings"]["load_seconds"] = time.monotonic() - t
            summary["model_provenance"] = model.encoder.provenance
            summary["inventory"] = model.trainable_inventory()
            base = resolved(recipe)
            learner = MemoryLearner(model, base.learning, base.seeds(0))
            history = _histories(recipe)
            history_one = {**history}
            def make_batch(count, offset=0):
                histories = [{**history, "elapsed": history["elapsed"] + (i + offset) / 1024}
                             for i in range(count)]
                return LearningBatch(histories, torch.zeros(count), torch.full((count,), 10),
                    torch.full((count, 7), .1), torch.arange(count))
            replay, fresh, eviction = make_batch(recipe.batch), make_batch(1, 128), make_batch(1, 256)
            def infer():
                model.encoder.clear_caches()
                return learner.choose_action(history_one, explore=False)
            timing, action = _timed(infer, repeats=repeats)
            summary["timings"]["inference_seconds"] = timing["median"]
            summary.setdefault("timing_samples", {})["inference"] = timing
            summary["checks"]["finite_action"] = bool(torch.isfinite(action["logits"]).all())
            summary["cuda_allocator"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            store.checkpoint({"stage": "learner_probes"}, progress={"phase": 1, "status": "profiling", "counts": {"step": 1}})
            def nll_backward():
                model.encoder.clear_caches()
                learner.optimizer.zero_grad(set_to_none=True)
                loss = learner.nll(learner.params, replay).mean()
                loss.backward()
                return float(loss.detach())
            timing, _ = _timed(nll_backward, repeats=repeats)
            summary["timings"]["nll_backward_seconds"] = timing["median"]
            summary["timing_samples"]["nll_backward"] = timing
            probes = pm.rademacher_probes(learner.params, learner.banks["stm"], recipe.fd_probes,
                                          generator=learner.generators["fd_probes"])
            def fd_backward():
                model.encoder.clear_caches()
                learner.optimizer.zero_grad(set_to_none=True)
                loss = pm.stm_fd_penalty(lambda p: learner.predict(p, replay), learner.params,
                    learner.design(replay), learner.banks["stm"], probes,
                    torch.ones(recipe.batch, device=learner.device), base.learning.fd_step)
                loss.backward()
                return float(loss.detach())
            timing, fd_value = _timed(fd_backward, repeats=repeats)
            summary["timings"]["fd_backward_seconds"] = timing["median"]
            summary["timing_samples"]["fd_backward"] = timing
            summary["checks"]["fd_value"] = fd_value
            summary["checks"]["fd_bank_gradients"] = {bank: sum(float(learner.params[n].grad.detach().double().square().sum())
                for n in names if learner.params[n].grad is not None) ** .5 for bank, names in learner.banks.items()}
            # Fisher is measured per evicted observation. A larger delayed-label
            # retirement must stream these contributions at fixed accepted theta.
            def retire():
                model.encoder.clear_caches()
                return learner.retire(eviction)
            timing, _ = _timed(retire, repeats=repeats)
            summary["timings"]["fisher_seconds"] = timing["median"]
            summary["timing_samples"]["fisher"] = timing
            baseline = learner.state_dict()
            for condition, strength in (("control", 0.), ("fd", base.learning.fd_strength)):
                learner.load_state_dict(baseline)
                learner.config = replace(base.learning, fd_strength=strength)
                def fit():
                    model.encoder.clear_caches()
                    return learner.fit(new=fresh, replay=replay, acquired_count=128)
                timing, values = _timed(fit, repeats=repeats)
                summary["timings"][f"fit_{condition}_seconds"] = timing["median"]
                summary["timing_samples"][f"fit_{condition}"] = timing
                summary["checks"][f"{condition}_last_losses"] = values
            teacher = FrozenQwenTeacher(model.encoder, primitive_names=ACTION_NAMES)
            evaluator = FrozenQwenEvaluator(model.encoder)
            for name, callback in (("teacher", lambda: teacher.distribution(history, probability_floor=.001)),
                                   ("evaluator", lambda: evaluator.evaluate(history))):
                def role_call():
                    model.encoder.clear_caches()
                    return callback()
                timing, _ = _timed(role_call, repeats=repeats)
                summary["timings"][f"{name}_seconds"] = timing["median"]
                summary["timing_samples"][name] = timing
            payload = learner.state_dict()
            buffer = io.BytesIO()
            t = time.monotonic()
            torch.save(payload, buffer)
            summary["checkpoint"] = {"learner_state_bytes": buffer.tell(),
                                     "serialize_seconds": time.monotonic() - t}
            # Include fsync and checksum, not just an in-memory serialization.
            t = time.monotonic()
            store.checkpoint({"stage": "checkpoint_cost", "learner": payload},
                progress={"phase": 1, "status": "profile_complete", "counts": {"step": 2}})
            summary["checkpoint"]["transaction_seconds"] = time.monotonic() - t
            summary["frozen_prefix_cache"] = model.encoder.prefix_cache_stats
            del payload, buffer
            peak = torch.cuda.max_memory_reserved()
            summary["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": peak, "total_bytes": total,
                "initial_free_bytes": free, "outside_torch_initial_bytes": outside,
                "headroom_bytes": total - outside - peak,
                "required_headroom_bytes": min_headroom_bytes}
            summary["status"] = "passed" if summary["memory"]["headroom_bytes"] >= min_headroom_bytes else "inadmissible"
            if summary["status"] != "passed":
                summary["reason"] = "insufficient reserved GPU headroom"
            if any(not math.isfinite(value) for value in summary["timings"].values()):
                raise FloatingPointError("nonfinite profile timing")
        except torch.cuda.OutOfMemoryError as exc:
            summary.update(status="oom", reason=str(exc))
        except Exception as exc:
            summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        summary["active_seconds"] = time.monotonic() - started
        summary["source_changed_during_probe"] = probe_sources() != identity["source_sha256"]
        if summary["source_changed_during_probe"]:
            summary.update(status="failed", reason="profiling dependencies changed during probe")
        if torch.cuda.is_available():
            summary.setdefault("memory", {}).setdefault("peak_reserved_bytes", torch.cuda.max_memory_reserved())
        store.checkpoint({"stage": "done", "status": summary["status"]},
                         progress={"phase": 1, "status": summary["status"], "counts": {"step": 3}})
        store.complete(summary)
        return summary


def _summary(path):
    return json.loads((path / "summary.json").read_text())


def launch_probe(root, recipe, *, model_path, offline, resume, repeats=3, timeout_s=600):
    identity_hash = hashlib.sha256(canonical_json(_identity(recipe, model_path, "phase1_resource_probe")).encode()).hexdigest()[:8]
    path = root / "units" / f"probe-{recipe.key}-{identity_hash}"
    if (path / "COMPLETED").exists():
        with RunStore(path, _identity(recipe, model_path, "phase1_resource_probe"), resume=True):
            return _summary(path)
    root.joinpath("logs").mkdir(exist_ok=True)
    log = root / "logs" / f"probe-{recipe.key}-{time.time_ns()}.log"
    command = [sys.executable, "-u", "-m", "numerical_experiment.picar.phase1", "probe",
        "--output", str(path), "--recipe", canonical_json(asdict(recipe)), "--repeats", str(repeats)]
    if model_path:
        command += ["--model-path", str(model_path)]
    if offline:
        command.append("--offline")
    if resume and path.exists():
        command.append("--resume")
    with log.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    if result.returncode or not (path / "COMPLETED").exists():
        raise RuntimeError(f"probe subprocess failed; inspect {log}")
    return _summary(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--recipe", required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--model-path", type=Path)
    probe.add_argument("--offline", action="store_true")
    probe.add_argument("--resume", action="store_true")
    probe.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_probe(args.output, Recipe(**json.loads(args.recipe)), model_path=args.model_path,
        offline=args.offline, resume=args.resume, repeats=args.repeats)
    print(canonical_json({k: result[k] for k in ("status", "recipe", "timings", "memory", "reason", "active_seconds")}))


if __name__ == "__main__":
    main()
