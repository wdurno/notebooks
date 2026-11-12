"""CPU-only projection diagnostic from trusted, completed Phase 2 artifacts.

No model is constructed and no training, feature extraction or downloads occur.
Run with python -m numerical_experiment.picar_feature_health --help.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import combinations
import json
import math
import os
from pathlib import Path
import time

from .picar.report import read_completed_unit
from .picar.runtime import sha256_file
from .picar.storage import JOURNAL_CHECKPOINT_POLICY, RunStore


VERSION = "picar-feature-health-v1"
CONDITIONS = ("forward", "left", "right", "stop")
NORMALIZATIONS = ("none", "feature_rms", "feature_and_projection_rms")
DEFINITION = {
    "input": "x = cat(last encoded features, executed_actions[0,-1], elapsed[0,-1])",
    "preactivation": "z = linear(x, projection.ltm.weight + projection.stm.B @ projection.stm.A, projection.ltm.bias)",
    "stm_scale": 1.,
    "activation": "h = tanh(z)",
    "saturation": "mean(abs(h) > 0.99), pooled over archived instruction rows and projection coordinates",
    "normalization": "feature_rms: f *= rsqrt(mean(f^2)+eps); feature_and_projection_rms additionally z *= rsqrt(mean(z^2)+eps), per row",
    "normalization_scope": "Explicit reconstruction setting; match the source model before interpreting as its actual activation",
    "derivative": "1 - tanh(z)^2 after configured normalization, in saved projection dtype; torch.median; local tanh slope, not the full Jacobian",
    "deltas": "All six instruction pairs per arrival: L2 and Linf raw feature differences and Linf archived logit differences",
    "scope": "Last-frame projection only, not a recurrent rollout Jacobian or a causal intervention",
}


@contextmanager
def _completed(path, kind):
    unit = read_completed_unit(path)
    allowed = (kind,) if isinstance(kind, str) else tuple(kind)
    if unit["summary"].get("kind") not in allowed:
        raise ValueError(f"expected one of {allowed}: {path}")
    policy = unit["stored_config"].get("checkpoint_policy")
    if policy is not None and policy != JOURNAL_CHECKPOINT_POLICY:
        raise ValueError("unsupported source checkpoint policy")
    with RunStore(path, unit["stored_config"], resume=True,
                  checkpoint_policy="legacy" if policy is None else "journal-v1") as store:
        if not store.completed:
            raise ValueError("source unit must be completed")
        yield unit, store


def _source(unit, store):
    path = Path(unit["path"]).resolve()
    return {"path": str(path), "kind": unit["summary"]["kind"], "completion_sha256": sha256_file(path / "COMPLETED"),
            "summary_sha256": sha256_file(path / "summary.json"),
            "config_sha256": sha256_file(path / "config.json"),
            "current_manifest_sha256": store.manifest["current"]["manifest"]["sha256"]}


def load_inputs(initial_asset, language_unit):
    """Verify both RunStores before trusted CPU unpickling of the initial state."""
    with _completed(initial_asset, ("phase2_initial_asset", "phase2_bootstrap")) as (initial, source):
        with _completed(language_unit, "phase2_language") as (language, language_store):
            sources = {"initial_asset": _source(initial, source), "language_unit": _source(language, language_store)}
            state = source.load_checkpoint(trusted=True)
            if state is None:
                raise ValueError("initial asset has no checkpoint")
            return state, language["summary"], sources


def analyze_projection(state, language, *, normalization="none", eps=1e-6):
    """Reconstruct the specified frozen projection using CPU tensors only."""
    import torch
    from torch.nn import functional as F
    from .picar.models import batch_digest

    if normalization not in NORMALIZATIONS or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid normalization or epsilon")
    parameters = state["learner"]["parameters"]
    names = ("projection.ltm.weight", "projection.ltm.bias", "projection.stm.A", "projection.stm.B")
    tensors = [parameters[name].detach().cpu() for name in names]
    weight, bias, a, b = tensors
    if (weight.ndim != 2 or bias.shape != weight.shape[:1] or a.ndim != 2 or b.ndim != 2
            or a.shape[1] != weight.shape[1] or b.shape != (weight.shape[0], a.shape[0])
            or not weight.is_floating_point() or any(t.dtype != weight.dtype for t in tensors)
            or any(not bool(torch.isfinite(t).all()) for t in tensors)):
        raise ValueError("incompatible/nonfinite saved projection")
    scale = parameters.get("projection.stm.scale", 1.)
    if float(scale) != 1.:
        raise ValueError("this diagnostic requires the specified STM scale 1")
    if "model_version" in language and language["model_version"] != state["learner"]["version"]:
        raise ValueError("language/model version differs from initial asset")
    eligible = {item["arrival"]: item for item in state["eligible"]}
    if not eligible or len(eligible) != len(state["eligible"]):
        raise ValueError("initial eligible arrivals must be nonempty and unique")
    digests = {arrival: batch_digest(item["history"]) for arrival, item in eligible.items()}
    rows = language.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > 512:
        raise ValueError("expected a nonempty bounded language panel")
    grouped, inputs, provenance = {}, [], []
    with torch.no_grad():
        effective_weight = weight + b @ a
        for row in rows:
            arrival, condition = row["arrival"], row["condition"]
            if arrival not in eligible or condition not in CONDITIONS:
                raise ValueError("unknown language arrival/instruction")
            if row.get("source_history_digest") != digests[arrival]:
                raise ValueError("language source history digest differs from initial asset")
            group = grouped.setdefault(arrival, {})
            if condition in group:
                raise ValueError("duplicate language instruction for an arrival")
            features = torch.as_tensor(row["features"], dtype=weight.dtype, device="cpu")
            logits = torch.as_tensor(row["logits"], dtype=weight.dtype, device="cpu")
            history = eligible[arrival]["history"]
            action, elapsed = (history[k].detach().cpu().to(weight.dtype)[0, -1]
                               for k in ("executed_actions", "elapsed"))
            if (features.ndim != 1 or action.shape != (4,) or elapsed.shape != (1,)
                    or features.numel() + 5 != weight.shape[1] or logits.ndim != 1 or not logits.numel()
                    or any(not bool(torch.isfinite(t).all()) for t in (features, logits, action, elapsed))):
                raise ValueError("incompatible/nonfinite feature/control/logit row")
            group[condition] = (features, logits)
            projected_features = features if normalization == "none" else F.rms_norm(features, (features.numel(),), eps=eps)
            inputs.append(torch.cat((projected_features, action, elapsed)))
            provenance.append({"arrival": arrival, "condition": condition, "source_history_digest": digests[arrival]})
        if set(grouped) != set(eligible) or any(set(g) != set(CONDITIONS) for g in grouped.values()):
            raise ValueError("language panel must cover all four instructions for every eligible arrival")
        z = F.linear(torch.stack(inputs), effective_weight, bias)
        raw_projection_abs_median = float(z.abs().median())
        if normalization == "feature_and_projection_rms":
            z = F.rms_norm(z, (z.shape[-1],), eps=eps)
        if not bool(torch.isfinite(z).all()):
            raise ValueError("nonfinite reconstructed preactivation")
        h = torch.tanh(z)
        derivative = 1 - h.square()
        deltas = []
        for arrival, group in sorted(grouped.items()):
            for first, second in combinations(CONDITIONS, 2):
                f1, l1 = group[first]
                f2, l2 = group[second]
                if l1.shape != l2.shape:
                    raise ValueError("logit dimensions differ across instructions")
                deltas.append({"arrival": arrival, "instruction_a": first, "instruction_b": second,
                    "source_history_digest": digests[arrival], "feature_l2": float((f1 - f2).norm()),
                    "feature_linf": float((f1 - f2).abs().max()),
                    "logit_linf": float((l1 - l2).abs().max())})

        def histogram(values):
            counts, edges = torch.histogram(values.flatten().double(), bins=32)
            return {"edges": edges.tolist(), "counts": counts.to(torch.int64).tolist()}

        arrival_maxima = [max(d["feature_linf"] for d in deltas if d["arrival"] == arrival) for arrival in grouped]
        forward_maxima = [max(d["feature_linf"] for d in deltas
                             if d["arrival"] == arrival and d["instruction_a"] == "forward") for arrival in grouped]
        return {"device": "cpu", "dtype": str(weight.dtype), "normalization": normalization, "eps": eps,
            "language_rows": len(rows), "raw_projection_abs_median": raw_projection_abs_median,
            "eligible_arrivals": len(eligible), "projection_coordinates": z.numel(),
            "abs_preactivation_median": float(z.abs().median()),
            "saturated_fraction": float((h.abs() > .99).double().mean()),
            "derivative_median": float(derivative.median()), "derivative_max": float(derivative.max()),
            "feature_delta_l2_range": [min(d["feature_l2"] for d in deltas), max(d["feature_l2"] for d in deltas)],
            "feature_delta_linf_per_arrival_max_range": [min(arrival_maxima), max(arrival_maxima)],
            "feature_delta_linf_from_forward_max_range": [min(forward_maxima), max(forward_maxima)],
            "logit_delta_linf_max": max(d["logit_linf"] for d in deltas),
            "instruction_deltas": deltas, "original_digests": provenance,
            "distributions": {"preactivation": histogram(z), "derivative": histogram(derivative)}}


def run(initial_asset, language_unit, output, *, resume=False, normalization="none", eps=1e-6):
    if normalization not in NORMALIZATIONS or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid normalization or epsilon")
    started = time.monotonic()
    state, language, sources = load_inputs(initial_asset, language_unit)
    own = Path(__file__).resolve()
    dependencies = (own, own.parent / "picar/storage.py", own.parent / "picar/models.py")
    dtype = str(state["learner"]["parameters"]["projection.ltm.weight"].dtype)
    config = {"schema_version": 1, "version": VERSION, "phase": 2, "device": "cpu", "dtype": dtype, "seeds": {},
              "randomness": "none; deterministic archived-tensor diagnostic", "sources": sources,
              "source_sha256": {str(p.relative_to(own.parents[1])): sha256_file(p) for p in dependencies},
              "mathematical_definition": DEFINITION, "normalization": normalization, "eps": eps}
    with RunStore(output, config, resume=resume, metadata={"device": "cpu", "dtype": dtype, "cuda_used": False}) as store:
        if store.completed:
            return read_completed_unit(output)["summary"]
        summary = {"schema_version": 1, "kind": "phase2_feature_health", "phase": 2,
            **analyze_projection(state, language, normalization=normalization, eps=eps), "sources": sources,
            "source_sha256": config["source_sha256"],
            "mathematical_definition": DEFINITION, "active_seconds": time.monotonic() - started,
            "interpretation": "Projection saturation is a hypothesis, not causal proof",
            "next_check": "Any common-model feature-normalization rescue reinitializes weights and Fisher from original demonstrations"}
        store.checkpoint({"diagnostic": VERSION, "sources": sources}, progress={"phase": 2, "status": "CPU feature health complete"})
        store.complete(summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-asset", type=Path, required=True, help="Trusted local completed phase2_initial_asset or phase2_bootstrap")
    parser.add_argument("--language-unit", type=Path, required=True, help="Completed phase2_language")
    parser.add_argument("--output", type=Path, required=True, help="New immutable diagnostic unit under cache/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--normalization", choices=NORMALIZATIONS, default="none")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    result = run(args.initial_asset, args.language_unit, args.output, resume=args.resume,
                 normalization=args.normalization, eps=args.eps)
    print(json.dumps({k: result[k] for k in ("abs_preactivation_median", "saturated_fraction", "derivative_median", "derivative_max")}, indent=2))


if __name__ == "__main__":
    main()
