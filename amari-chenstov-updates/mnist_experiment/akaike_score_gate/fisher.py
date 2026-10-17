"""Model-label likelihood Fisher sampling and small dense diagnostic realizations.

Observed labels deliberately are not an argument. Scores are ``s = -grad NLL``;
sampling more scores improves integration, not the amount of training evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

from src.derivatives import per_sample_derivatives
from src.mnist_model import mnist_nll
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed


FISHER_RESUME_SCHEMA_VERSION = 1


def _sampling_fingerprint(model, layout, inputs, weights, indices, count, seed, exact_labels):
    """Bind partial sums to a fixed model, input measure, and score stream."""
    digest = hashlib.sha256()
    contract = {
        "schema": FISHER_RESUME_SCHEMA_VERSION, "layout": layout.metadata(),
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "training": model.training, "count": count, "seed": seed,
        "exact_labels": exact_labels, "indices": indices.cpu().tolist(),
    }
    digest.update(json.dumps(contract, sort_keys=True).encode("ascii"))
    tensors = list(model.state_dict().items()) + [("inputs", inputs), ("weights", weights)]
    for name, value in tensors:
        digest.update(json.dumps((name, list(value.shape), str(value.dtype))).encode("ascii"))
        flat = value.detach().reshape(-1)
        # Hash large input buffers in bounded CPU pieces, not one extra copy.
        for piece in flat.split(262144):
            digest.update(piece.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _matrix(matrix: Tensor) -> Tensor:
    result = torch.as_tensor(matrix).detach().to(dtype=torch.float64)
    if (result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]
            or not torch.isfinite(result).all()):
        raise ValueError("matrix must be finite, square, and nonempty")
    return (result + result.mT) / 2


def _indices(indices, dimension: int, device: torch.device) -> Tensor:
    if indices is None:
        return torch.arange(dimension, device=device)
    value = torch.as_tensor(indices, device=device)
    if value.ndim != 1 or value.numel() == 0 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be a nonempty integer vector")
    if value.min() < 0 or value.max() >= dimension or value.unique().numel() != value.numel():
        raise ValueError("indices must be unique global coordinates in range")
    return value.to(torch.int64)


def represent(matrix: Tensor, kind: str = "lr", rank: int = 8, layer_indices=None) -> Tensor:
    """Return a float64 dense realization of a same-sample PSD representation.

    ``lr`` uses exact leading eigenpairs plus a nonnegative residual diagonal.
    This deliberately small-dimension diagnostic backend is not a scalable
    Lanczos implementation or a claim about compressed storage/solve cost.
    Materially indefinite input is rejected; numerical PSD repair is allowed.
    """
    source = _matrix(matrix)
    aliases = {"diagonal": "diag", "low_rank_diagonal": "lr",
               "dense_same_samples": "dense"}
    kind = aliases.get(kind, kind)
    if kind not in ("diag", "lr", "dense", "layer_block"):
        raise ValueError(f"unknown Fisher representation: {kind}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    values, vectors = torch.linalg.eigh(source)
    scale = max(float(values.abs().max()), torch.finfo(torch.float64).tiny)
    tolerance = 100 * source.shape[0] * torch.finfo(torch.float64).eps * scale
    if float(values.min()) < -tolerance:
        raise ValueError("representation requires a PSD source Fisher")
    if kind == "diag":
        return torch.diag(source.diagonal().clamp_min(0))
    if kind == "dense":
        if values.min() >= 0:
            return source.clone()
        result = (vectors * values.clamp_min(0)) @ vectors.mT
    elif kind == "lr":
        keep = min(rank, source.shape[0])
        if keep == 0:
            return torch.diag(source.diagonal().clamp_min(0))
        factors = vectors[:, -keep:] * values[-keep:].clamp_min(0).sqrt()
        low_rank = factors @ factors.mT
        residual = (source.diagonal() - low_rank.diagonal()).clamp_min(0)
        result = low_rank + torch.diag(residual)
    else:
        if layer_indices is None:
            raise ValueError("layer_block requires a complete coordinate partition")
        groups = layer_indices.values() if isinstance(layer_indices, Mapping) else layer_indices
        groups = [_indices(group, source.shape[0], source.device) for group in groups]
        all_indices = torch.cat(groups) if groups else source.new_empty(0, dtype=torch.int64)
        if (all_indices.numel() != source.shape[0]
                or all_indices.unique().numel() != source.shape[0]):
            raise ValueError("layer_indices must partition all coordinates exactly once")
        result = torch.zeros_like(source)
        for group in groups:
            result[group[:, None], group[None, :]] = source[group[:, None], group[None, :]]
    return (result + result.mT) / 2


def representation_diagnostics(source: Tensor, represented: Tensor, *, rank: int = 8) -> dict:
    """Audit trace/diagonal preservation and the LR residual clamp mass."""
    source, represented = _matrix(source), _matrix(represented)
    if represented.shape != source.shape:
        raise ValueError("representation dimension differs from its source")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    values, vectors = torch.linalg.eigh(source)
    keep = min(rank, source.shape[0])
    low_diagonal = torch.zeros_like(source.diagonal())
    if keep:
        low_diagonal = (vectors[:, -keep:].square() * values[-keep:].clamp_min(0)).sum(1)
    return {
        "source_min_eigenvalue": float(values.min()),
        "represented_min_eigenvalue": float(torch.linalg.eigvalsh(represented).min()),
        "diagonal_max_error": float((source.diagonal() - represented.diagonal()).abs().max()),
        "trace_error": float(represented.trace() - source.trace()),
        "residual_diagonal_clamp_mass": float((low_diagonal - source.diagonal()).clamp_min(0).sum()),
    }


def model_fisher(
    model: nn.Module, layout: ParameterLayout, inputs: Tensor, weights: Tensor | None = None,
    *, sample_size: int = 64, seed: int = 0, exact_labels: bool = False,
    chunk_size: int = 32, indices=None, return_score_bank: bool = False,
    chunk_callback: Callable[[dict], None] | None = None,
    resume_state: Mapping | None = None,
) -> tuple[Tensor, dict]:
    """Estimate the likelihood Fisher at this model and weighted input measure.

    Sampled mode draws inputs with replacement using normalized ``weights``
    and independent model labels. Exact mode integrates all ten labels over
    *every* provided input; ``sample_size`` is ignored in that diagnostic mode.
    Both modes target the same weighted Fisher. Named input/label RNG streams
    and pre-generated uniforms make the sampling independent of chunk size.

    ``indices`` selects a principal block *after full-coordinate score AD*.
    Full-network score cost is explicitly reported, even for a small suffix.
    Optional score banks contain scores, their outer-product weights, labels,
    and source input indices. Without a bank, score memory is chunk-bounded.
    The callback gets each chunk's weighted outer sum (not a normalized chunk
    mean), input range, and score rows; these can be sealed as partial artifacts.

    Seal ``chunk["resume_state"]`` atomically, then pass it back as
    ``resume_state`` after interruption. It contains ``next_input``, the raw
    cumulative ``weighted_outer_sum`` (never a projected/renormalized Fisher),
    and a fingerprint of the fixed model, layout, inputs, weights, and random
    stream. A changed chunk size is allowed; changed sampling inputs are not.
    Sample/label RNG draws are regenerated cheaply; completed scores are not.
    A resumed call cannot return a complete in-memory score bank: assemble it
    from previously sealed and new callback chunks instead. Artifact checksum
    verification, source/config hashes, and durable writes belong to the caller.
    """
    layout.validate_module(model)
    for name, value in (("sample_size", sample_size), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if inputs.ndim < 2 or inputs.shape[0] == 0 or not torch.isfinite(inputs).all():
        raise ValueError("inputs must be a finite nonempty batch")
    first = next(model.parameters())
    if inputs.device != first.device or inputs.dtype != first.dtype:
        raise ValueError("inputs must share the model device and dtype")
    n = inputs.shape[0]
    a = torch.ones(n, dtype=torch.float64) if weights is None else torch.as_tensor(weights).detach().double().cpu()
    if (a.shape != (n,) or not torch.isfinite(a).all() or (a < 0).any()
            or not torch.isfinite(a.sum()) or a.sum() <= 0):
        raise ValueError("weights must be finite, nonnegative, and have positive total mass")
    a = a / a.sum()
    selected = _indices(indices, layout.total_numel, inputs.device)
    seeds = {name: derive_component_seed(seed, "model_fisher_" + name)
             for name in ("inputs", "labels")}
    input_rng = torch.Generator(device="cpu").manual_seed(seeds["inputs"])
    label_rng = torch.Generator(device="cpu").manual_seed(seeds["labels"])
    draws = torch.arange(n) if exact_labels else torch.multinomial(
        a, sample_size, replacement=True, generator=input_rng
    )
    count = draws.numel()
    uniforms = None if exact_labels else torch.rand(count, generator=label_rng, dtype=torch.float64)
    total = torch.zeros((selected.numel(), selected.numel()), device=inputs.device, dtype=torch.float64)
    fingerprint = None
    if chunk_callback is not None or resume_state is not None:
        fingerprint = _sampling_fingerprint(model, layout, inputs, a, selected, count, seed, exact_labels)
    resume_start = 0
    if resume_state is not None:
        if not isinstance(resume_state, Mapping):
            raise ValueError("resume_state must be a callback-state mapping")
        if resume_state.get("schema_version") != FISHER_RESUME_SCHEMA_VERSION:
            raise ValueError("incompatible Fisher resume schema")
        if resume_state.get("sampling_fingerprint") != fingerprint:
            raise ValueError("Fisher resume state does not match the model/input/sampling contract")
        resume_start = resume_state.get("next_input")
        if (isinstance(resume_start, bool) or not isinstance(resume_start, int)
                or not 0 <= resume_start <= count):
            raise ValueError("resume next_input must be an integer within the input draw range")
        raw = resume_state.get("weighted_outer_sum")
        if (not isinstance(raw, Tensor) or raw.dtype != torch.float64
                or raw.shape != total.shape or not torch.isfinite(raw).all()
                or not torch.allclose(raw, raw.mT, atol=1e-12, rtol=1e-10)):
            raise ValueError("resume weighted_outer_sum must be a finite symmetric float64 matrix")
        if resume_start == 0 and torch.count_nonzero(raw):
            raise ValueError("a zero-position resume must have a zero accumulated sum")
        if return_score_bank and resume_start:
            raise ValueError("resumed score banks must be assembled from sealed callback chunks")
        total = raw.detach().to(device=inputs.device).clone()
    score_parts, weight_parts, input_parts, label_parts = [], [], [], []
    score_count = resume_start * (10 if exact_labels else 1)
    for start in range(resume_start, count, chunk_size):
        stop = min(count, start + chunk_size)
        ids = draws[start:stop]
        batch = inputs.index_select(0, ids.to(inputs.device))
        with torch.no_grad():
            logits = model(batch)
            if logits.shape != (stop - start, 10) or not torch.isfinite(logits).all():
                raise ValueError("model must return ten finite logits per input")
            probabilities = logits.to(torch.float64).softmax(-1)
        if exact_labels:
            targets = torch.arange(10, device=inputs.device).repeat(stop - start)
            batch = batch.repeat_interleave(10, dim=0)
            row_weights = (a[ids].to(inputs.device)[:, None] * probabilities).reshape(-1)
            row_ids = ids.repeat_interleave(10)
        else:
            cumulative = probabilities.cpu().cumsum(-1)
            targets = (uniforms[start:stop, None] > cumulative).sum(-1).clamp_max(9).to(inputs.device)
            row_weights = total.new_full((stop - start,), 1 / count)
            row_ids = ids
        gradients = per_sample_derivatives(model, batch, targets, mnist_nll, layout,
                                           strategy="vmap").gradients
        scores = -gradients.index_select(1, selected).to(torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError("model generated nonfinite scores")
        partial = scores.mT @ (row_weights[:, None] * scores)
        total += partial
        score_count += scores.shape[0]
        if return_score_bank:
            score_parts.append(scores)
            weight_parts.append(row_weights)
            input_parts.append(row_ids)
            label_parts.append(targets.cpu())
        if chunk_callback is not None:
            chunk_callback({"start": start, "stop": stop, "input_indices": ids.clone(),
                            "weighted_outer_sum": partial.detach().clone(),
                            "weight_sum": float(row_weights.sum()),
                            "scores": scores.detach().clone(), "score_weights": row_weights.clone(),
                            "score_input_indices": row_ids.clone(), "labels": targets.cpu().clone(),
                            "resume_state": {
                                "schema_version": FISHER_RESUME_SCHEMA_VERSION,
                                "sampling_fingerprint": fingerprint, "next_input": stop,
                                "weighted_outer_sum": total.detach().cpu().clone(),
                            }})
    total = (total + total.mT) / 2
    values, vectors = torch.linalg.eigh(total)
    negative = values.clamp_max(0)
    projected = total if values.min() >= 0 else (vectors * values.clamp_min(0)) @ vectors.mT
    projected = (projected + projected.mT) / 2
    diagnostics = {
        "estimator": "exact_model_labels" if exact_labels else "sampled_model_labels",
        "seed": seed, "seeds": seeds, "input_count": n, "input_draw_count": count,
        "unique_input_draws": draws.unique().numel(), "score_count": score_count,
        "resumed_input_draw_count": resume_start,
        "computed_input_draw_count": count - resume_start,
        "computed_score_count": (count - resume_start) * (10 if exact_labels else 1),
        "sampling_fingerprint": fingerprint,
        "input_indices": draws.clone(), "input_weights": a.clone(),
        "evidence_ess": float(1 / a.square().sum()),
        "sampling_with_replacement": not exact_labels, "exact_labels": exact_labels,
        "chunk_size": chunk_size, "indices": selected.cpu(),
        "score_cost_scope": "full_network", "computed_score_dimension": layout.total_numel,
        "returned_score_dimension": selected.numel(),
        "min_eigenvalue_before_psd": float(values.min()),
        "negative_eigenvalue_count": int((values < 0).sum()),
        "negative_spectral_mass": float(-negative.sum()),
        "psd_projection_frobenius": float(torch.linalg.vector_norm(negative)),
    }
    if return_score_bank:
        diagnostics["score_bank"] = {
            "scores": torch.cat(score_parts), "weights": torch.cat(weight_parts),
            "input_indices": torch.cat(input_parts), "labels": torch.cat(label_parts),
        }
    return projected, diagnostics
