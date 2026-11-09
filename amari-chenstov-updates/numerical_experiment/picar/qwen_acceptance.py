"""Explicit bounded real-model CUDA acceptance, never a learning experiment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from src.artifacts import _atomic_write_json
from .config import PicarConfig
from .runtime import runtime_metadata, source_fingerprint, utc_now


def acceptance_sources():
    names = {"models", "learner", "config", "runtime", "qwen_acceptance", "picar_math",
             "simulator", "geometry", "drift", "protocol", "actions"}
    return {path: checksum for path, checksum in source_fingerprint().items()
            if Path(path).stem in names}


def run_acceptance(output: Path, *, model_path: Path | None = None, offline: bool = False) -> dict:
    import torch

    from src.continual import picar_math as pm
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig
    from .learner import LearningBatch, MemoryLearner
    from .models import load_qwen_model

    if output.exists():
        raise FileExistsError("use a new acceptance directory; completed attempts are immutable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not substitute CPU/tiny acceptance for Qwen acceptance")
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError("real-model smoke requires at least 8 GiB free; wait for the GPU")
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"schema_version": 1, "kind": "qwen_acceptance", "phase": 0,
                "status": "running", "started_at": utc_now(), "environment": runtime_metadata(),
                "source_sha256": acceptance_sources(), "smoke_only": True}
    _atomic_write_json(output / "manifest.json", manifest)
    backend = None
    try:
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        kwargs = {"top_layers": 1, "ltm_rank": 4, "stm_rank": 1, "hidden_size": 16,
                  "context_length": 2, "return_bins": 5, "max_tokens": 1024,
                  "min_pixels": 56 * 56, "max_pixels": 112 * 112,
                  "local_files_only": offline or model_path is not None, "seed": 19007}
        if model_path is not None:
            kwargs["model_path"] = model_path
        model = load_qwen_model(**kwargs)
        model.eval()
        loaded_s = time.monotonic() - started
        backend = PyBulletBackend(SimulatorConfig(width=64, height=48, realtime=False), run_id="qwen-acceptance")
        observation = backend.reset(seed=19007)
        image = torch.frombuffer(bytearray(observation.rgb), dtype=torch.uint8).reshape(48, 64, 3)
        batch = {"images": image.permute(2, 0, 1).float()[None, None] / 255,
                 "executed_actions": torch.zeros(1, 1, 4), "elapsed": torch.full((1, 1, 1), .1),
                 "contexts": [[{"text": "Go around the obstacle, approach the red ball, and stop.", "issued_at": 0.}]]}
        cfg = PicarConfig()
        learner = MemoryLearner(model, replace(cfg.learning, fd_strength=.1), cfg.seeds(0))
        history = LearningBatch([batch], torch.zeros(1), torch.tensor([2]),
                                torch.full((1, 7), .1), torch.tensor([0]))
        torch.cuda.synchronize()
        t = time.monotonic()
        action = learner.choose_action(batch, explore=False)
        torch.cuda.synchronize()
        inference_s = time.monotonic() - t
        design = learner.design(history)
        probes = pm.rademacher_probes(learner.params, learner.banks["stm"], 1,
                                      generator=learner.generators["fd_probes"])
        fd_values = []
        bank_gradients = {}
        previous_gradient = None
        for epsilon in (.005, .01, .02):
            penalty = pm.stm_fd_penalty(lambda p: learner.predict(p, history), learner.params,
                design, learner.banks["stm"], probes, torch.ones(1, device="cuda"), epsilon)
            if not bool(torch.isfinite(penalty)):
                raise FloatingPointError("nonfinite Qwen FD")
            gradients = torch.autograd.grad(penalty, tuple(learner.params.values()), allow_unused=True)
            bank_gradients = {bank: sum(float(g.detach().double().square().sum())
                for (name, _), g in zip(learner.params.items(), gradients)
                if name in learner.banks[bank] and g is not None) ** .5 for bank in ("ltm", "stm")}
            vector = torch.cat([g.detach().float().flatten() for g in gradients if g is not None])
            cosine = None if previous_gradient is None else float(torch.nn.functional.cosine_similarity(
                vector[None], previous_gradient[None])[0])
            previous_gradient = vector
            fd_values.append({"epsilon": epsilon, "penalty": float(penalty.detach()),
                              "gradient_norms": bank_gradients, "total_gradient_norm": float(vector.norm()),
                              "gradient_cosine_to_previous": cosine})
        if not all(value > 0 for value in bank_gradients.values()):
            raise AssertionError("FD does not reach both declared trainable banks")
        penalties = [row["penalty"] for row in fd_values]
        fd_ratio = max(penalties) / min(penalties) if min(penalties) > 0 else None
        norms = [row["total_gradient_norm"] for row in fd_values]
        gradient_ratio = max(norms) / min(norms) if min(norms) > 0 else None
        fd_convergence = (fd_ratio is not None and fd_ratio <= 1.25 and
            gradient_ratio is not None and gradient_ratio <= 1.5) and all(
            row["gradient_cosine_to_previous"] >= .95 for row in fd_values[1:])
        metrics = learner.fit(new=None, replay=history, acquired_count=4)
        fisher = learner.retire(history)
        if not all(bool(torch.isfinite(d).all()) and bool((d >= 0).all()) for d in learner.diagonal.values()):
            raise AssertionError("invalid eviction Fisher diagonal")
        if any(p.requires_grad for name, p in model.encoder.qwen.named_parameters()
               if ".banks.ltm." not in name and ".banks.stm." not in name):
            raise AssertionError("frozen backbone parameter became trainable")
        torch.cuda.synchronize()
        summary = {"schema_version": 1, "phase": 0, "kind": "qwen_acceptance", "smoke_only": True,
            "model_provenance": model.encoder.provenance,
            "load_seconds": loaded_s, "inference_seconds": inference_s,
            "fd_half_base_double": fd_values, "fit": metrics, "retirement": fisher,
            "fd_convergence_ratio": fd_ratio, "fd_convergence_pass": fd_convergence,
            "fd_gradient_norm_ratio": gradient_ratio,
            "fisher_trace": sum(float(d.sum()) for d in learner.diagonal.values()),
            "trainable_parameters": {bank: sum(learner.params[n].numel() for n in names)
                                     for bank, names in learner.banks.items()},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_total_bytes": total, "initial_free_bytes": free,
            "action_probabilities": action["probabilities"].tolist(),
            "active_seconds": time.monotonic() - started,
            "source_changed_during_run": manifest["source_sha256"] != acceptance_sources()}
        _atomic_write_json(output / "summary.json", summary)
        if summary["source_changed_during_run"]:
            raise RuntimeError("source changed during acceptance; retain this attempt and rerun once stable")
        if not fd_convergence:
            raise RuntimeError("FD step/gradient consistency gate failed; retain diagnostics, do not start research")
        manifest.update(status="completed", completed_at=utc_now())
        _atomic_write_json(output / "manifest.json", manifest)
        _atomic_write_json(output / "COMPLETED", {"schema_version": 1})
        return summary
    except BaseException as exc:
        manifest.update(status="failed", completed_at=utc_now(), error=f"{type(exc).__name__}: {exc}",
                        active_seconds=time.monotonic() - started)
        _atomic_write_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    summary = run_acceptance(args.output, model_path=args.model_path, offline=args.offline)
    print(json.dumps({k: v for k, v in summary.items() if k != "model_provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
