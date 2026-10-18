"""Pure, JSON-friendly Plan 15 metrics; no model inference or training.

Inputs may be NumPy arrays, sequences, or detached PyTorch tensors. Undefined
statistics are None, never NaN. Weights describe evaluation populations, not
additional independent observations. Uncertainty is clustered by replica.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

METRIC_SCHEMA_VERSION = 1


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _labels_weights(labels: Any, weights: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _array(labels)
    if labels.shape != (n,) or not np.all(np.isfinite(labels)):
        raise ValueError("labels must be a finite vector aligned with predictions")
    if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels > 9)):
        raise ValueError("labels must be integer digits in [0, 9]")
    weight = np.ones(n) if weights is None else _array(weights)
    if weight.shape != (n,) or not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("weights must be finite, nonnegative, and aligned")
    if n and weight.sum() <= 0:
        raise ValueError("weights must have positive total mass")
    return labels.astype(np.int64), weight


def _log_probabilities(logits: Any) -> np.ndarray:
    logits = _array(logits)
    if logits.ndim != 2 or logits.shape[1] != 10 or not np.all(np.isfinite(logits)):
        raise ValueError("logits must have finite shape (observations, 10)")
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _prevalence(p_ref: float) -> float:
    if isinstance(p_ref, bool) or not np.isfinite(p_ref) or not 0 < p_ref < 1:
        raise ValueError("p_ref must be strictly between zero and one")
    return float(p_ref)


def _mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    mass = float(weights.sum())
    return float(np.dot(values, weights) / mass) if mass else None


def predictive_metrics(
    logits: Any, labels: Any, weights: Any = None, *, p_ref: float = 0.1,
    conditional_weights: Any = None,
) -> dict[str, Any]:
    """NLL/accuracy and argmax digit-9 diagnostics at fixed evaluation prevalence.

    Class-conditional rates use weights normalized within each stratum. A
    missing stratum makes its rate undefined. No positive predictions gives
    precision zero with a flag, even when the positive stratum is absent.
    Optional conditional_weights preserve stratified evaluation support at a
    zero-prevalence endpoint without altering population-weighted NLL/accuracy.
    """
    p_ref = _prevalence(p_ref)
    logp = _log_probabilities(logits)
    labels, weight = _labels_weights(labels, weights, len(logp))
    conditional = weight if conditional_weights is None else _labels_weights(labels, conditional_weights, len(logp))[1]
    positive = labels == 9
    predicted = logp.argmax(axis=1) == 9
    losses = -logp[np.arange(len(labels)), labels]
    correct = logp.argmax(axis=1) == labels
    recall = _mean(predicted[positive], conditional[positive])
    fpr = _mean(predicted[~positive], conditional[~positive])
    no_predictions = not bool(np.any(predicted & (conditional > 0)))
    precision = None
    if no_predictions:
        precision = 0.0
    elif recall is not None and fpr is not None:
        denominator = p_ref * recall + (1 - p_ref) * fpr
        precision = float(p_ref * recall / denominator) if denominator else 0.0
    return {
        "n": len(labels),
        "weight_sum": float(weight.sum()),
        "nll": _mean(losses, weight),
        "accuracy": _mean(correct, weight),
        "nine_count": int(positive.sum()),
        "non_nine_count": int((~positive).sum()),
        "nine_nll": _mean(losses[positive], conditional[positive]),
        "non_nine_nll": _mean(losses[~positive], conditional[~positive]),
        "nine_recall": recall,
        "nine_fpr": fpr,
        "nine_precision": precision,
        "nine_precision_empirical": _mean(positive[predicted], weight[predicted])
        if weight[predicted].sum() > 0 else 0.0,
        "false_positives_per_1000": None if fpr is None else 1000 * fpr,
        "p_ref": p_ref,
        "no_positive_labels": not bool(positive.any()),
        "no_positive_support": recall is None,
        "no_negative_support": fpr is None,
        "no_predicted_positives": no_predictions,
        "recall_missing_reason": "no positive evaluation weight" if recall is None else None,
        "empty": not len(labels),
    }


def standardized_pr_curve(
    probabilities: Any, labels: Any, weights: Any = None, *, p_ref: float = 0.1
) -> dict[str, Any]:
    """Threshold-swept 9 PR and non-interpolated AP at fixed prevalence.

    Accept p(9) vectors or normalized ten-class probabilities. Ties enter as a
    group. Threshold None is the no-predictions point, assigned precision zero
    by the Plan 15 convention; its zero recall increment does not affect AP.
    """
    p_ref = _prevalence(p_ref)
    probabilities = _array(probabilities)
    if probabilities.ndim == 2:
        if probabilities.shape[1] != 10 or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("class probabilities must have ten columns and sum to one")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probabilities must be finite in [0, 1]")
        scores = probabilities[:, 9]
    elif probabilities.ndim == 1:
        scores = probabilities
    else:
        raise ValueError("probabilities must be a p(9) vector or ten-class matrix")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite in [0, 1]")
    labels, weight = _labels_weights(labels, weights, len(scores))
    positive = labels == 9
    pos_mass, neg_mass = weight[positive].sum(), weight[~positive].sum()
    missing = "no positive evaluation weight" if not pos_mass else (
        "no negative evaluation weight" if not neg_mass else None
    )
    if missing:
        return {"thresholds": [], "recall": [], "fpr": [], "precision": [],
                "average_precision": None, "missing_reason": missing, "p_ref": p_ref}
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1]
    # Cumulative and grouped sums can round differently by a few ulps.
    tpr = np.clip(np.r_[0.0, np.cumsum(weight[order] * positive[order])[ends] / pos_mass], 0, 1)
    fpr = np.clip(np.r_[0.0, np.cumsum(weight[order] * ~positive[order])[ends] / neg_mass], 0, 1)
    numerator = p_ref * tpr
    denominator = numerator + (1 - p_ref) * fpr
    precision = np.divide(numerator, denominator, out=np.zeros_like(tpr), where=denominator > 0)
    return {
        "thresholds": [None, *sorted_scores[ends].tolist()],
        "recall": tpr.tolist(), "fpr": fpr.tolist(), "precision": precision.tolist(),
        "average_precision": float(np.clip(np.dot(np.diff(tpr), precision[1:]), 0, 1)),
        "missing_reason": None, "p_ref": p_ref,
    }


def _curve(exposures: Any, values: Any) -> tuple[np.ndarray, np.ndarray]:
    x, y = _array(exposures), _array(values)
    if x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("curve requires aligned finite vectors")
    if np.any(np.diff(x) <= 0):
        raise ValueError("exposures must be strictly increasing; aggregate duplicates explicitly")
    return x, y


def normalized_auc(exposures: Any, values: Any) -> float | None:
    """Trapezoidal AUC divided by observed exposure span, never extrapolated."""
    x, y = _curve(exposures, values)
    if len(x) < 2:
        return None
    return float(np.dot(np.diff(x), (y[1:] + y[:-1]) * 0.5) / (x[-1] - x[0]))


def cohort_retention_auc(
    exposures: Any,
    values: Any,
    *,
    eviction_exposure: float,
    end_exposure: float,
    capacity: float,
    horizon: float | None = None,
) -> dict[str, Any]:
    """Retention from last-member eviction, with explicit right censoring.

    The declared horizon is min(2C, end - eviction), or a supplied paired
    horizon. Curves must bracket eviction. Endpoint interpolation is linear
    on the declared evaluation grid; no missing future is extrapolated.
    A partial AUC is descriptive, not eligible for a complete paired endpoint.
    """
    x, y = _curve(exposures, values)
    if not all(np.isfinite(v) for v in (eviction_exposure, end_exposure, capacity)) or capacity <= 0:
        raise ValueError("eviction, end, and positive capacity must be finite")
    requested = 2 * capacity if horizon is None else float(horizon)
    if not np.isfinite(requested) or requested < 0:
        raise ValueError("horizon must be finite and nonnegative")
    declared = max(0.0, min(requested, end_exposure - eviction_exposure))
    result = {"auc": None, "eviction_exposure": float(eviction_exposure),
              "requested_horizon": float(requested), "declared_horizon": float(declared),
              "observed_horizon": 0.0, "right_censored": declared < requested,
              "complete_for_declared_horizon": False, "missing_reason": None}
    if declared <= 0:
        result["missing_reason"] = "no post-eviction follow-up"
        return result
    if len(x) < 2 or x[0] > eviction_exposure or x[-1] <= eviction_exposure:
        result["missing_reason"] = "evaluation grid does not bracket eviction and follow-up"
        result["right_censored"] = True
        return result
    stop = min(eviction_exposure + declared, x[-1])
    interior = (x > eviction_exposure) & (x < stop)
    grid = np.r_[eviction_exposure, x[interior], stop]
    metric = np.interp(grid, x, y)
    observed = stop - eviction_exposure
    result.update(auc=normalized_auc(grid, metric), observed_horizon=float(observed),
                  right_censored=observed < requested,
                  complete_for_declared_horizon=bool(np.isclose(observed, declared)))
    return result


def cohort_summary(cohorts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-cohort within-replica average; partial follow-up stays descriptive."""
    usable = [c for c in cohorts if c.get("auc") is not None and c.get("complete_for_declared_horizon")]
    return {"mean_auc": float(np.mean([c["auc"] for c in usable])) if usable else None,
            "cohorts": len(cohorts), "usable_cohorts": len(usable),
            "right_censored_cohorts": sum(bool(c.get("right_censored")) for c in cohorts),
            "missing_cohorts": len(cohorts) - len(usable)}


def deterministic_seed(seed: int, component: str) -> int:
    """Stable named seed, independent of Python's process-randomized hash."""
    payload = f"plan15:{int(seed)}:{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def paired_summary(
    control: Mapping[Any, float | None],
    treatment: Mapping[Any, float | None],
    *,
    seed: int = 0,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Control-minus-treatment effects, resampling whole matched replicas.

    Call separately for every environment and protocol cohort. Missing and
    unmatched replicas are disclosed. One pair has no uncertainty interval;
    intervals from fewer than eight pairs are explicitly flagged as fragile.
    """
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    keys = sorted(control.keys() & treatment.keys(), key=str)
    pairs = []
    for key in keys:
        a, b = control[key], treatment[key]
        if a is not None and b is not None:
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError("paired summaries require finite scalar values or None")
            pairs.append((key, float(a - b)))
    differences = np.array([d for _, d in pairs])
    interval = None
    if len(pairs) > 1:
        rng = np.random.default_rng(deterministic_seed(seed, "paired_bootstrap"))
        bootstrap = []
        for start in range(0, resamples, 256):
            indices = rng.integers(0, len(pairs), size=(min(256, resamples - start), len(pairs)))
            bootstrap.extend(differences[indices].mean(axis=1).tolist())
        tail = (1 - confidence) / 2
        interval = np.quantile(bootstrap, [tail, 1 - tail]).tolist()
    return {"n_pairs": len(pairs), "mean_gain": float(differences.mean()) if len(pairs) else None,
            "median_gain": float(np.median(differences)) if len(pairs) else None,
            "favorable_replicas": int((differences > 0).sum()), "ci": interval,
            "confidence": confidence, "seed": int(seed), "resamples": int(resamples),
            "paired_values": [{"replica": k, "gain": d} for k, d in pairs],
            "unmatched_control": len(control.keys() - treatment.keys()),
            "unmatched_treatment": len(treatment.keys() - control.keys()),
            "missing_pairs": len(keys) - len(pairs), "small_sample": len(pairs) < 8,
            "sign": "positive favors treatment"}


def dependence_metrics(
    base_logits: Any, intervened_logits: Any, labels: Any, weights: Any = None
) -> dict[str, Any]:
    """Loss change and KL(base || intervention) from supplied predictions only.

    The caller defines attenuation or earlier-LTM/later-STM pairing, and must
    keep labels, inputs and weights identical. Positive loss change is harm.
    """
    base = _log_probabilities(base_logits)
    changed = _log_probabilities(intervened_logits)
    if base.shape != changed.shape:
        raise ValueError("base and intervened predictions must have matching shapes")
    labels, weight = _labels_weights(labels, weights, len(base))
    before = -base[np.arange(len(labels)), labels]
    after = -changed[np.arange(len(labels)), labels]
    kl = (np.exp(base) * (base - changed)).sum(axis=1)
    return {"base_nll": _mean(before, weight), "intervened_nll": _mean(after, weight),
            "nll_difference": _mean(after - before, weight),
            "predictive_kl": _mean(np.maximum(kl, 0), weight),
            "base_accuracy": _mean(base.argmax(axis=1) == labels, weight),
            "intervened_accuracy": _mean(changed.argmax(axis=1) == labels, weight), "n": len(labels)}


def future_vulnerability(
    earlier_logits: Any, earlier_ltm_later_stm_logits: Any, labels: Any, weights: Any = None
) -> dict[str, Any]:
    """Explicitly named wrapper for the frozen-earlier-LTM counterfactual."""
    result = dependence_metrics(earlier_logits, earlier_ltm_later_stm_logits, labels, weights)
    result["vulnerability_nll"] = result["nll_difference"]
    return result
