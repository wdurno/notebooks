"""Data-only Fix 4 predictive NLLs, independent of fitting and reward labeling.

For completed labels, p(k, z | history, baseline) is categorical in k and
Gaussian in the seven executed logits z, with mean action_mean and covariance
action_variance * I / exp((bin_values[k] - baseline) / temperature). The action
variance is NOT the exploration variance. No penalty or weight cap is used.
Audit arithmetic is CPU float64 on the saved source values; comparison with
the learner's float32 likelihood therefore requires float32 tolerance.

Current-policy capture belongs at the action-producing forward pass. A caller
can offer choose_action_with_prediction(history, explore=...) returning the
ordinary action dict plus a detached CPU copy of both output heads from that
SAME forward, together with the pre-action baseline and likelihood design.
After complete-episode labeling, pass those saved outputs to score_prediction.
Do not rerun a later model, reconstruct logits from a mean return, or substitute
the chosen action for a missing action_mean. For deterministic actions z=mean,
the Gaussian density still includes its full normalizer; it is not imitation
accuracy. This module neither changes choose_action nor samples actions.

Reference evaluation reconstructs predictions from full, sealed raw histories,
not replay-eligibility-trimmed histories. Callers must serialize learner access
and keep reference/evaluation records out of fitting and baseline updates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
import math
from numbers import Integral

import torch

from numerical_experiment.picar.learner import LearningBatch
from src.continual import picar_math as pm


SCHEMA_VERSION = 1


def _numeric(value, name: str) -> torch.Tensor:
    try:
        raw = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if raw.dtype == torch.bool or raw.is_complex():
            raise ValueError(f"{name} must be real numeric values, not boolean or complex")
        # Convert Python/JSON values directly, avoiding an intermediate FP32 round.
        result = (raw.to(device="cpu", dtype=torch.float64) if isinstance(value, torch.Tensor)
                  else torch.as_tensor(value, device="cpu", dtype=torch.float64))
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name: str, *, positive: bool = False) -> float:
    result = _numeric(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    number = float(result)
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _integer(value, name: str) -> int:
    if isinstance(value, torch.Tensor) and value.ndim == 0 and value.dtype != torch.bool:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not 0 <= value <= torch.iinfo(torch.int64).max:
        raise ValueError(f"{name} must be a nonnegative int64")
    return value


def _vector(value, name: str, *, size: int | None = None,
            singleton_batch: bool = False) -> torch.Tensor:
    result = _numeric(value, name)
    if singleton_batch and result.ndim == 2 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 1 or (size is not None and result.numel() != size):
        expected = f"[{size}]" if size is not None else "[K]"
        raise ValueError(f"{name} must have shape {expected}" +
                         (" or a singleton batch of that shape" if singleton_batch else ""))
    return result


@torch.no_grad()
def score_prediction(prediction: Mapping, *, label, action, baseline, bin_values,
                     temperature, action_variance) -> dict:
    """Score ONE saved prediction; return detached, JSON-friendly audit values.

    Heads accept [K]/[7] or [1,K]/[1,7]; action must be the executed [7] logits,
    never eight softmax probabilities or physical controls. label is the
    completed return-bin index, not a raw/coded continuous return. baseline
    is the saved pre-action baseline, never the current predicted mean return.
    bin_values are increasing bin midpoints in the same units as baseline.

    The top-level return_logits/action_mean fields can be passed back as
    prediction, with the saved label/action/design fields, for recomputation.
    Negative Gaussian NLLs are valid densities and are not clipped to zero.
    """
    if not isinstance(prediction, Mapping) or not {"return_logits", "action_mean"} <= prediction.keys():
        raise ValueError("prediction must contain return_logits and action_mean")
    bins = _vector(bin_values, "bin_values")
    logits = _vector(prediction["return_logits"], "return_logits", size=bins.numel(), singleton_batch=True)
    mean = _vector(prediction["action_mean"], "action_mean", size=pm.ACTION_DIM, singleton_batch=True)
    action = _vector(action, "action", size=pm.ACTION_DIM)
    label = _integer(label, "label")
    if label >= bins.numel():
        raise ValueError("label must be in [0,K)")
    baseline = _scalar(baseline, "baseline")
    temperature = _scalar(temperature, "temperature", positive=True)
    variance = _scalar(action_variance, "action_variance", positive=True)
    design = pm.LikelihoodDesign(bins, torch.tensor([baseline], dtype=torch.float64),
                                 temperature, torch.eye(pm.ACTION_DIM, dtype=torch.float64) * variance)
    log_weight = float(design.log_weights[0, label])
    weight = float(design.weights[0, label])
    return_nll = float(-logits.log_softmax(-1)[label])
    action_quadratic = .5 * weight * float(((action - mean) / math.sqrt(variance)).square().sum())
    action_log_normalizer = .5 * pm.ACTION_DIM * (math.log(2 * math.pi) + math.log(variance) - log_weight)
    action_nll = action_quadratic + action_log_normalizer
    joint_nll = return_nll + action_nll
    scalars = {"return_nll": return_nll, "action_nll": action_nll, "joint_nll": joint_nll,
               "action_quadratic": action_quadratic, "action_log_normalizer": action_log_normalizer}
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError("predictive NLL components must be finite")
    return {"schema_version": SCHEMA_VERSION, **scalars, "weight": weight, "log_weight": log_weight,
            "label": label, "bin_value": float(bins[label]), "baseline": baseline,
            "bin_values": bins.tolist(), "temperature": temperature, "action_variance": variance,
            "action": action.tolist(), "return_logits": logits.tolist(), "action_mean": mean.tolist()}


def _record_identity(record: Mapping) -> tuple[str, int]:
    required = {"history", "bin", "baseline", "logits", "arrival"}
    if not isinstance(record, Mapping) or not required <= record.keys():
        raise ValueError("reference record requires history, bin, baseline, logits and arrival")
    arrival = _integer(record["arrival"], "arrival")
    episode_id = record.get("episode_id")
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise ValueError("episode_id must be a nonempty string when supplied")
    record_id = record.get("record_id", json.dumps([episode_id, arrival], separators=(",", ":")))
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a nonempty string")
    return record_id, arrival


@contextmanager
def _evaluation(learner):
    if getattr(learner, "_cooperate", None) is not None:
        raise RuntimeError("predictive evaluation requires a quiescent learner")
    modes = [(module, module.training) for module in learner.model.modules()]
    try:
        learner.model.eval()
        with torch.no_grad():
            yield
    finally:
        # Preserve mixed submodule modes without a recursive train() override.
        for module, training in modes:
            module.training = training


def _evaluate_record(learner, record: Mapping) -> dict:
    record_id, arrival = _record_identity(record)
    history = record["history"]
    if (not isinstance(history, Mapping) or not isinstance(history.get("images"), torch.Tensor)
            or history["images"].ndim < 2 or history["images"].shape[0] != 1):
        raise ValueError("reference history must have an images tensor with batch dimension one")
    baseline = _scalar(record["baseline"], "baseline")
    label = _integer(record["bin"], "bin")
    action = _vector(record["logits"], "logits", size=pm.ACTION_DIM)
    # Unlike the training _batch helper, do not filter a sealed causal history.
    batch = LearningBatch([history], torch.tensor([baseline], dtype=torch.float64),
                          torch.tensor([label]), action.unsqueeze(0), torch.tensor([arrival]))
    prediction = learner.predict(learner.params, batch)
    row = score_prediction(prediction, label=label, action=action, baseline=baseline,
                           bin_values=pm.return_bin_values(learner.edges),
                           temperature=learner.config.temperature,
                           action_variance=learner.config.action_variance)
    row.update(record_id=record_id, arrival=arrival, aggregation_weight=1.)
    if "episode_id" in record:
        row["episode_id"] = record["episode_id"]
    if "environment" in record:
        if not isinstance(record["environment"], str) or not record["environment"]:
            raise ValueError("environment must be a nonempty string when supplied")
        row["environment"] = record["environment"]
    return row


def evaluate_record(learner, record: Mapping) -> dict:
    """Evaluate one fixed-reference record without learning, sampling or RNG use.

    Uses learner.predict(learner.params, LearningBatch) in eval/no_grad mode,
    restoring every module's original mode even on failure. The learner must
    implement deterministic, nonmutating prediction as MemoryLearner does.
    No baseline, replay, counters, optimizer, gradients or generators are updated.
    Save each returned row before advancing the campaign's scoring cursor.

    Explicit record_id is preferred; otherwise identity is JSON [episode_id,
    arrival], with null for an absent episode_id. Never use the scoring index
    as identity, since it changes across resume/reordering.
    """
    with _evaluation(learner):
        return _evaluate_record(learner, record)


def evaluate_records(learner, records: Sequence[Mapping], *, start_cursor: int = 0) -> list[dict]:
    """Return per-record audit rows in bank order, optionally from start_cursor.

    The bank must be nonempty with unique stable IDs. A cursor already at the
    end returns an empty list. summarize_scores is separate so saved prefix
    rows can be combined explicitly after resume. Per-episode/paired inference
    and transactional cursor persistence are caller-owned.
    """
    if len(records) == 0:
        raise ValueError("reference records must be nonempty")
    cursor = _integer(start_cursor, "start_cursor")
    if cursor > len(records):
        raise ValueError("start_cursor exceeds reference record count")
    ids = [_record_identity(record)[0] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("reference record IDs must be unique")
    with _evaluation(learner):
        return [_evaluate_record(learner, record) for record in records[cursor:]]


def summarize_scores(rows: Sequence[Mapping]) -> dict:
    """Equal-per-record means, NOT equal-episode or advantage-weighted means.

    Each aggregation_weight must be 1; row['weight'] is Gaussian precision,
    not an aggregation weight. Empty rows have None means, not zero losses.
    """
    names = ("return_nll", "action_nll", "joint_nll", "weight")
    values = {name: [] for name in names}
    for row in rows:
        if not isinstance(row, Mapping) or not {*names, "aggregation_weight"} <= row.keys():
            raise ValueError("summary requires scored rows with explicit aggregation_weight")
        if _scalar(row["aggregation_weight"], "aggregation_weight") != 1:
            raise ValueError("summary requires equal per-record aggregation_weight of 1")
        for name in names:
            values[name].append(_scalar(row[name], name, positive=name == "weight"))
    count = len(rows)
    summary = {"weighting": "equal_per_record", "record_count": count, "total_weight": float(count)}
    for name in names:
        summary["mean_" + name] = math.fsum(value / count for value in values[name]) if count else None
    return summary
