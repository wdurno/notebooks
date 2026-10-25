"""Small-artifact Phase 8 inference; no learner, model, or checkpoint imports.

Records are summaries with ``metrics`` and a shared-per-replica ``pairing_id``.
The frozen manifest supplies ``recipes`` (recipe dictionaries), ``replicas``,
``N``, ``n_F``, and optionally ``q_dev``, ``fit``, and ``fit_range``. Recipes
have unique ``recipe_key`` and ``method`` fields. A record can instead contain
``summary`` and ``metadata`` dictionaries, as in an artifact reader.

All MSEs are in squared degrees. Only independent whole replicas are resampled;
the same draw indices are used for every arm, metric, and checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from . import PROTOCOL
SCHEMA_VERSION = 1
FD = "fd_replay_ewc"
REPLAY = "replay_ewc"
CheckStop = Callable[[], Any] | None


def _check(check_stop: CheckStop) -> None:
    if check_stop is not None and check_stop():
        raise InterruptedError("Phase 8 analysis interrupted")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(float(value)) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str, minimum: float = 0.) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def flatten_record(record: Mapping[str, Any]) -> dict:
    """Accept flat snapshots or the runner's metadata/summary/metrics wrapper."""
    result = {**record.get("metadata", {}), **record.get("summary", {}),
              **{k: v for k, v in record.items() if k not in ("metadata", "summary")}}
    for key, value in result.get("recipe", {}).items():
        result.setdefault(key, value)
    return result


def summarize_trajectory(
    metrics: Iterable[Mapping[str, Any]], *, method: str | None = None,
    recipe_key: str | None = None, replica: Any = None, N: int | None = None,
    frozen_novel_mse: float | None = None, frozen_base_mse: float | None = None,
    pairing_id: str | None = None, capacity: int | None = None,
    replay_count: int | None = None, valid: bool = True,
) -> dict:
    """Compute A(N) from exactly n=1,...,N, never an arrival-weighted AUC.

    Checkpoints must be unique and ordered, with cumulative observations and
    learning_seconds. n=0 is optional only if both frozen MSEs are explicit.
    This function validates structure even for a numerically invalid trajectory;
    it never drops a bad checkpoint to manufacture a complete summary.
    """
    rows = [dict(row) for row in metrics]
    if not rows:
        raise ValueError("trajectory has no metrics")
    counts = [_integer(row["n"], "n") for row in rows]
    N = _integer(N if N is not None else counts[-1], "N", 1)
    start = 0 if counts[0] == 0 else 1
    if counts != list(range(start, N + 1)):
        raise ValueError("missing, duplicate, unordered, or out-of-budget novel counts")
    previous_observations, previous_seconds = -1, -1.
    for n, row in zip(counts, rows):
        row["n"] = n
        row["observations"] = _integer(row["observations"], "observations")
        row["learning_seconds"] = _number(row["learning_seconds"], "learning_seconds")
        for key in ("novel_mse", "base_mse"):
            row[key] = _number(row[key], key)
        if (row["observations"] < n or row["observations"] <= previous_observations
                or row["learning_seconds"] < previous_seconds):
            raise ValueError("nonmonotone exposure/work counters")
        if n == 0 and (row["observations"] != 0 or row["learning_seconds"] != 0):
            raise ValueError("initial checkpoint must precede online exposure/work")
        previous_observations = row["observations"]
        previous_seconds = row["learning_seconds"]
    frozen = {}
    for domain, supplied in (("novel", frozen_novel_mse), ("base", frozen_base_mse)):
        if start == 0:
            observed = rows[0][f"{domain}_mse"]
            if supplied is not None and not math.isclose(float(supplied), observed, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("frozen risk disagrees with initialization checkpoint")
            supplied = observed
        if supplied is None:
            raise ValueError("initial checkpoint or explicit frozen risks required")
        frozen[f"frozen_{domain}_mse"] = _number(supplied, f"frozen_{domain}_mse")
    online = rows[1:] if start == 0 else rows
    result = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, N=N,
                  novel_auc=float(np.mean([row["novel_mse"] for row in online])),
                  base_auc=float(np.mean([row["base_mse"] for row in online])),
                  final_novel_mse=rows[-1]["novel_mse"], final_base_mse=rows[-1]["base_mse"],
                  learning_seconds=rows[-1]["learning_seconds"],
                  observations=rows[-1]["observations"], valid=bool(valid), **frozen)
    for key, value in dict(method=method, recipe_key=recipe_key, replica=replica,
                           pairing_id=pairing_id, capacity=capacity, replay_count=replay_count).items():
        if value is not None:
            result[key] = value
    return result


def replica_draw_indices(n_replicas: int, seed: int, draw_index: int) -> np.ndarray:
    """Independently seeded draw identity, stable for resumable chunk scheduling."""
    n = _integer(n_replicas, "n_replicas", 2)
    seed = _integer(seed, "seed")
    index = _integer(draw_index, "draw_index")
    return np.random.default_rng(np.random.SeedSequence([seed, index])).integers(n, size=n)


def _interval(values: np.ndarray, means: np.ndarray, *, positive: bool = True) -> dict:
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci=[float(low), float(high)], n_pairs=len(values),
                favorable_replicas=int(np.sum(values > 0 if positive else values < 0)),
                standard_error=float(np.std(values, ddof=1) / math.sqrt(len(values))))


def paired_interval(values, seed: int, draws: int = 2000, *, check_stop: CheckStop = None) -> dict:
    """Nominal percentile interval on mean paired differences; positive is favorable."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired_interval requires at least two finite replica differences")
    draws = _integer(draws, "draws", 2)
    means = np.empty(draws)
    for index in range(draws):
        _check(check_stop)
        means[index] = values[replica_draw_indices(len(values), seed, index)].mean()
    return {**_interval(values, means), "draws": draws, "seed": int(seed),
            "resampling_unit": "paired_whole_replica", "confidence_level": .95}


def manifest_recipes(manifest: Mapping[str, Any]) -> list[dict]:
    recipes = manifest["recipes"]
    if isinstance(recipes, Mapping):
        recipes = [dict(value, recipe_key=value.get("recipe_key", key))
                   for key, value in recipes.items()]
    recipes = [dict(recipe) for recipe in recipes]
    keys = [recipe.get("recipe_key") for recipe in recipes]
    methods = [recipe.get("method") for recipe in recipes]
    if (not recipes or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys) or len(set(methods)) != len(methods)
            or FD not in methods or REPLAY not in methods):
        raise ValueError("manifest needs unique recipes/methods including FD and replay")
    return recipes


def _cohort(records, manifest, check_stop):
    if manifest.get("sealed", True) is not True:
        raise ValueError("confirmation manifest is not sealed")
    if manifest.get("protocol") != PROTOCOL or manifest.get("schema_version", 1) != 1:
        raise ValueError("incompatible confirmation manifest")
    recipes = manifest_recipes(manifest)
    replicas = list(manifest["replicas"])
    if len(replicas) < 2 or len(set(replicas)) != len(replicas):
        raise ValueError("confirmation needs at least two distinct planned replicas")
    N, n_fd = _integer(manifest["N"], "N", 2), _integer(manifest["n_F"], "n_F", 1)
    if n_fd >= N:
        raise ValueError("n_F must be smaller than N")
    planned = {(recipe["recipe_key"], replica) for recipe in recipes for replica in replicas}
    indexed = {}
    for source in records:
        _check(check_stop)
        record = flatten_record(source)
        key = record["recipe_key"], record["replica"]
        if key in indexed or key not in planned:
            raise ValueError("duplicate or unplanned confirmation record")
        if record.get("valid") is not True or record.get("status", "completed") not in ("completed", "complete"):
            raise ValueError("invalid/incomplete planned trajectory; exclusion is not allowed")
        if record.get("partition", "confirmation") != "confirmation":
            raise ValueError("development records cannot enter confirmation inference")
        if record.get("protocol") != PROTOCOL or record.get("schema_version", 1) != 1:
            raise ValueError("incompatible trajectory schema")
        recipe = next(recipe for recipe in recipes if recipe["recipe_key"] == key[0])
        for name, value in recipe.items():
            if name not in record or record[name] != value:
                raise ValueError(f"record disagrees with manifest recipe field {name}")
        if not record.get("pairing_id"):
            raise ValueError("pairing_id binding shared initialization/stream/panels is required")
        summary = summarize_trajectory(
            record["metrics"], N=N, frozen_novel_mse=record.get("frozen_novel_mse"),
            frozen_base_mse=record.get("frozen_base_mse"))
        for name in ("novel_auc", "base_auc", "frozen_novel_mse", "frozen_base_mse",
                     "observations", "learning_seconds"):
            if name in record and not math.isclose(float(record[name]), summary[name], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"summary disagrees with metrics: {name}")
        record.update(summary)
        if record["metrics"][0]["n"] != 0:
            record["metrics"] = [dict(n=0, observations=0, learning_seconds=0.,
                                      novel_mse=summary["frozen_novel_mse"],
                                      base_mse=summary["frozen_base_mse"]), *record["metrics"]]
        indexed[key] = record
    if set(indexed) != planned:
        raise ValueError("incomplete planned confirmation cohort; final inference is pending")
    pairing_ids = []
    for replica in replicas:
        arms = [indexed[recipe["recipe_key"], replica] for recipe in recipes]
        reference = arms[0]
        pairing_ids.append(reference["pairing_id"])
        for arm in arms[1:]:
            if arm["pairing_id"] != reference["pairing_id"]:
                raise ValueError("mismatched pairing provenance")
            for name in ("frozen_novel_mse", "frozen_base_mse"):
                if not math.isclose(arm[name], reference[name], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("mismatched shared frozen initialization")
            if [row["observations"] for row in arm["metrics"]] != [row["observations"] for row in reference["metrics"]]:
                raise ValueError("mismatched paired data exposures")
    if len(set(pairing_ids)) != len(pairing_ids):
        raise ValueError("distinct replicas cannot reuse a pairing_id")
    return recipes, replicas, N, n_fd, indexed


def analyze_confirmation(records, manifest: Mapping[str, Any], seed: int, draws: int = 2000,
                         *, check_stop: CheckStop = None) -> dict:
    """Analyze the entire frozen cohort, or raise instead of reporting partial CIs.

    ``learning_seconds_per_observation_limit`` and ``base_retention_guard_degrees``
    specify the frozen resource and retention guards. ``fit=False`` is the default;
    ``fit=True`` requires a registered ``fit_range=[lo, hi]`` and ``q_dev``.
    The caller persists this small result as an immutable analysis artifact.
    """
    _check(check_stop)
    recipes, replicas, N, n_fd, indexed = _cohort(records, manifest, check_stop)
    draws = _integer(draws, "draws", 2)
    guard = _number(manifest.get("base_retention_guard_degrees", 1.), "retention guard")
    limit = manifest.get("learning_seconds_per_observation_limit")
    limit = None if limit is None else _number(limit, "learning-time limit")
    by_method = {recipe["method"]: recipe["recipe_key"] for recipe in recipes}
    matrices, summaries = {}, {}
    for recipe in recipes:
        key = recipe["recipe_key"]
        arms = [indexed[key, replica] for replica in replicas]
        matrices[key] = {domain: np.asarray([[row[f"{domain}_mse"] for row in arm["metrics"]]
                                             for arm in arms], dtype=float)
                         for domain in ("novel", "base")}
        summaries[key] = {name: np.asarray([arm[name] for arm in arms], dtype=float)
                          for name in ("novel_auc", "base_auc", "frozen_novel_mse", "frozen_base_mse",
                                       "learning_seconds", "observations")}
    fd, replay = by_method[FD], by_method[REPLAY]
    statistics = {"risk_gain": summaries[replay]["novel_auc"] - summaries[fd]["novel_auc"],
                  "direct_contrast": matrices[fd]["novel"][:, n_fd] - matrices[replay]["novel"][:, N]}
    for key, summary in summaries.items():
        statistics[f"frozen:{key}"] = summary["frozen_novel_mse"] - summary["novel_auc"]
        statistics[f"retention:{key}"] = np.sqrt(summary["base_auc"]) - np.sqrt(summary["frozen_base_mse"])
    sampled = {name: np.empty(draws) for name in statistics}
    bands = {key: {domain: np.empty((draws, N + 1)) for domain in ("novel", "base")}
             for key in matrices}
    for draw in range(draws):
        _check(check_stop)
        indices = replica_draw_indices(len(replicas), seed, draw)
        for name, values in statistics.items():
            sampled[name][draw] = values[indices].mean()
        for key in matrices:
            for domain in ("novel", "base"):
                bands[key][domain][draw] = matrices[key][domain][indices].mean(axis=0)
    risk = _interval(statistics["risk_gain"], sampled["risk_gain"])
    direct = _interval(statistics["direct_contrast"], sampled["direct_contrast"], positive=False)
    direct.update(n_F=n_fd, n_R=N, supports_better_at_smaller_budget=direct["ci"][1] < 0,
                  interpretation="FD(n_F) minus replay(N); negative favors FD; not first-passage savings")
    curves, frozen_gain, retention, resources = {}, {}, {}, {}
    for recipe in recipes:
        key = recipe["recipe_key"]
        arms = [indexed[key, replica] for replica in replicas]
        summary = summaries[key]
        curves[key] = dict(method=recipe["method"], n=list(range(N + 1)), n_pairs=len(replicas))
        for domain in ("novel", "base"):
            mean = matrices[key][domain].mean(axis=0)
            low, high = np.quantile(bands[key][domain], [.025, .975], axis=0)
            curves[key][domain] = dict(mean_mse=mean.tolist(), ci_mse=[low.tolist(), high.tolist()],
                                       mean_rmse=np.sqrt(mean).tolist(),
                                       ci_rmse=[np.sqrt(low).tolist(), np.sqrt(high).tolist()])
        curves[key]["observations"] = np.mean([[row["observations"] for row in arm["metrics"]]
                                               for arm in arms], axis=0).tolist()
        curves[key]["learning_seconds"] = np.mean([[row["learning_seconds"] for row in arm["metrics"]]
                                                   for arm in arms], axis=0).tolist()
        frozen_gain[key] = _interval(statistics[f"frozen:{key}"], sampled[f"frozen:{key}"])
        ret = _interval(statistics[f"retention:{key}"], sampled[f"retention:{key}"], positive=False)
        ret.update(guard_degrees=guard, meets_guard=ret["ci"][1] <= guard,
                   mean_final_base_mse=float(matrices[key]["base"][:, -1].mean()),
                   mean_base_auc=float(summary["base_auc"].mean()))
        retention[key] = ret
        cost = float(summary["learning_seconds"].sum() / summary["observations"].sum())
        resources[key] = dict(method=recipe["method"], mean_learning_seconds=float(summary["learning_seconds"].mean()),
                              mean_observations=float(summary["observations"].mean()),
                              mean_seconds_per_observation=cost, seconds_per_observation_limit=limit,
                              resource_feasible=None if limit is None else cost <= limit)
        for name in ("peak_vram_bytes", "peak_host_bytes", "update_latency_p95_seconds"):
            if all(name in arm for arm in arms):
                resources[key][name] = max(_number(arm[name], name) for arm in arms)
        ceiling = manifest.get("memory_ceiling_bytes")
        if ceiling is not None and "peak_vram_bytes" in resources[key]:
            resources[key]["memory_ceiling_bytes"] = ceiling
            resources[key]["memory_feasible"] = resources[key]["peak_vram_bytes"] <= ceiling
            if limit is not None:
                resources[key]["resource_feasible"] = bool(
                    resources[key]["resource_feasible"] and resources[key]["memory_feasible"])
        for budget in (n_fd, N):
            resources[key][f"at_{budget}"] = {name: curves[key][name][budget]
                                             for name in ("observations", "learning_seconds")}
    fits = {"registered": False}
    if manifest.get("fit", False):
        lo, hi = manifest["fit_range"]
        lo, hi = _integer(lo, "fit lower bound", 1), _integer(hi, "fit upper bound", 2)
        if not 1 <= lo < hi <= N:
            raise ValueError("registered fit range is outside observed budgets")
        target = _number(manifest["q_dev"], "q_dev")
        fits = dict(registered=True, range=[lo, hi], q_dev=target, methods={})
        for key in (fd, replay):
            observed = np.asarray(curves[key]["novel"]["mean_mse"])
            x = np.arange(lo, hi + 1)
            slope, intercept = np.polyfit(x, observed[lo:hi + 1], 1)
            predicted = intercept + slope * x
            crossing = float((target - intercept) / slope) if slope < 0 else None
            issues = []
            if slope >= 0:
                issues.append("nondecreasing_fit")
            if observed[0] <= target:
                issues.append("target_already_met_at_initialization")
            if crossing is not None and not lo <= crossing <= hi:
                issues.append("crossing_outside_registered_range")
            if np.any(predicted < 0):
                issues.append("negative_fitted_mse")
            fits["methods"][key] = dict(slope=float(slope), intercept=float(intercept),
                                         fitted_mse=predicted.tolist(), n=x.tolist(),
                                         residual_rmse=float(np.sqrt(np.mean((predicted - observed[lo:hi + 1]) ** 2))),
                                         crossing=None if issues else crossing, issues=issues,
                                         interpretation="registered linear approximation; visual adequacy still required")
    return dict(schema_version=1, protocol=PROTOCOL, status="completed",
                cohort=dict(recipes=deepcopy(recipes), replicas=deepcopy(replicas), N=N, n_F=n_fd),
                bootstrap=dict(draws=draws, seed=int(seed), resampling_unit="paired_whole_replica",
                               confidence_level=.95, curve_bands="pointwise_not_simultaneous"),
                units=dict(risk="squared_degrees", retention="degrees", cost="seconds"),
                risk_gain=risk, frozen_gain=frozen_gain, direct_contrast=direct,
                retention=retention, resources=resources, curves=curves, linear_fits=fits,
                q_dev=manifest.get("q_dev"),
                assumptions="Independent paired replicas and adequate squared-error moments; nominal intervals "
                            "conditional on benchmark panels and frozen recipes, not familywise inference.")
