"""Validated two-policy scalar inference and whole-pair bootstrap."""

import math
import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import flatten_record, replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import Recipe, content_hash
from mnist_experiment.fisher_memory_stm_capacity import analysis as legacy
from mnist_experiment.combined_memory.reproduction import analysis as prior
from mnist_experiment.combined_memory.replicated_combination.analysis import _efficiency
from mnist_experiment.combined_memory.replicated_combination.calibration import summarize_metrics
from . import PROTOCOL
from .config import POLICIES, STRENGTH, PROCESS_SCALE, BOOTSTRAP_SEED

TOLERANCE, CEILING = .01, .5


def validate_record(raw):
    row = flatten_record(raw)
    policy, N = row.get("policy"), row.get("target")
    if (row.get("protocol") not in (legacy.PROTOCOL, PROTOCOL) or row.get("schema_version") != 1
            or policy not in POLICIES or row.get("namespace") not in ("development", "fresh", "acceptance")
            or type(row.get("replica")) is not int or row["replica"] < 0
            or type(N) is not int or N < 2 or row.get("representation") != "low_rank_diagonal"
            or row.get("valid") is not True or row.get("status", "completed") not in ("completed", "complete")
            or row.get("stm_config") != "head_r1" or row.get("output_dim") != 54
            or row.get("family") != row.get("partition", "") + "--grid"
            or row.get("partition") not in ("development", "confirmation")
            or row.get("pairing_id") != row.get("plan_hash") or not row.get("plan_hash")):
        raise ValueError("complete R/K trajectory with paired provenance required")
    recipe = Recipe.from_mapping(row["recipe"])
    if (recipe.key != row["recipe_key"] or recipe.method != "fd_replay_ewc"
            or row.get("method") != recipe.method or recipe.strength != STRENGTH
            or row.get("capacity") != recipe.capacity or row.get("replay_count") != recipe.replay_count
            or recipe.ridge > 1000 or (policy == "R" and recipe.ridge != 1.)
            or (policy == "K" and row.get("process_scale") != PROCESS_SCALE)
            or (N == 100 and (recipe.capacity != 32 or recipe.replay_count != 2
                or recipe.fd_step != .01 or recipe.probes != 2 or recipe.max_iter != 100 or recipe.max_eval != 150))):
        raise ValueError("R/K recipe differs from the approved protocol")
    summarize_metrics(row["metrics"], N=N)
    legacy.validate_records([{**row, "protocol": legacy.PROTOCOL}], N, row["partition"])
    return row


def paired(records, *, namespace="fresh", partial=False):
    indexed = {}
    for raw in records:
        row = validate_record(raw)
        if row["namespace"] != namespace:
            raise ValueError("cohorts cannot mix development and fresh records")
        key = row["replica"], row["policy"]
        if key in indexed:
            raise ValueError("duplicate policy/replica")
        indexed[key] = row
    members = sorted({r for r, p in indexed})
    complete = [r for r in members if all((r, p) in indexed for p in POLICIES)]
    excluded = [r for r in members if r not in complete]
    if (excluded and not partial) or len(complete) < 2:
        raise ValueError("at least two complete R/K pairs required")
    for r in complete:
        a, b = (indexed[r, p] for p in POLICIES)
        if a["plan_hash"] != b["plan_hash"] or a["target"] != b["target"]:
            raise ValueError("unpaired R/K data plan or horizon")
        for field in ("factor_seeds", "initialization_hash", "holdout_hash", "stream_hash", "chart_hash", "layout_hash"):
            if (field in a or field in b) and a.get(field) != b.get(field):
                raise ValueError("unpaired R/K identity: " + field)
        for field in ("base_mse", "novel_mse"):
            if not math.isclose(a["metrics"][0][field], b["metrics"][0][field], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("unpaired initial predictions")
        if [p["observations"] for p in a["metrics"]] != [p["observations"] for p in b["metrics"]]:
            raise ValueError("unpaired exposure clocks")
        for field in legacy.grid.DRIFT_FIELDS:
            if any(field in m for m in a["metrics"]+b["metrics"]):
                if (not all(field in m for m in a["metrics"]+b["metrics"])
                        or any(x[field] != y[field] for x, y in zip(a["metrics"], b["metrics"]))):
                    raise ValueError("unpaired drift values")
    groups = {p: [indexed[r, p] for r in complete] for p in POLICIES}
    if len({row["plan_hash"] for row in groups["R"]}) != len(complete):
        raise ValueError("distinct replicas must use distinct plans")
    for p, rows in groups.items():
        if len({row["recipe_key"] for row in rows}) != 1 or len({row["target"] for row in rows}) != 1:
            raise ValueError("one frozen recipe and horizon required per condition")
    return groups, complete, excluded


def analyze(records, *, draws=2000, seed=BOOTSTRAP_SEED, partial=False,
            check=lambda: None, load_bootstrap=None, save_bootstrap=None):
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("valid draw count and named seed required")
    groups, replicas, excluded = paired(records, partial=partial)
    N = groups["R"][0]["target"]
    fields = prior.FIELDS
    arrays = {p: {f: np.asarray([[m[f] for m in row["metrics"]] for row in rows], dtype=float)
                  for f in fields} for p, rows in groups.items()}
    samples = np.asarray([replica_draw_indices(len(replicas), seed, i) for i in range(draws)])
    identity = content_hash(dict(protocol=PROTOCOL,
        records=[r for p in POLICIES for r in groups[p]], draws=draws, seed=seed))
    inverses = []
    for start in range(0, draws, 100):
        check()
        end = min(draws, start+100)
        chunk = load_bootstrap(start) if load_bootstrap else None
        if chunk is None:
            chunk = dict(input_hash=identity, start=start, end=end, values=[dict(draw=i,
                indices=samples[i].tolist(), match=prior._match(arrays["K"], arrays["R"], samples[i]))
                for i in range(start, end)])
            chunk["payload_hash"] = content_hash(chunk)
            if save_bootstrap:
                save_bootstrap(start, chunk)
        if (chunk["input_hash"] != identity or chunk["start"] != start or chunk["end"] != end
                or chunk["payload_hash"] != content_hash({k: v for k, v in chunk.items() if k != "payload_hash"})
                or [v["draw"] for v in chunk["values"]] != list(range(start, end))
                or any(v["indices"] != samples[v["draw"]].tolist() for v in chunk["values"])):
            raise ValueError("bootstrap checkpoint changed/corrupt")
        inverses.extend(v["match"] for v in chunk["values"])
    interval = lambda values: prior._interval(values, samples)
    result = dict(protocol=PROTOCOL, schema_version=1, namespace="fresh", N=N, replicas=replicas,
        is_final=not partial and not excluded and N == 100 and replicas == list(range(8)),
        reference_policy="R", excluded_blocks=excluded, input_hash=identity,
        curves={}, summaries={}, bootstrap=dict(seed=seed, draws=draws,
            resampling_unit="whole_R_K_replica", confidence_level=.95),
        units=dict(risk="square_pixels_per_landmark", retention="RMS_landmark_pixels"))
    vectors = {}
    for p, a in arrays.items():
        query, old = a["novel_mse"], a["base_mse"]
        exposure = a["observations"]
        vectors[p] = dict(acquisition=query[:, 1:].mean(1), endpoint_mse=query[:, -1],
            retention=np.sqrt(old[:, 1:].mean(1))-np.sqrt(old[:, 0]),
            endpoint_retention=np.sqrt(old[:, -1])-np.sqrt(old[:, 0]),
            all_arrival_risk=((query[:, 1:]+query[:, :-1])*.5*np.diff(exposure, axis=1)).sum(1)/exposure[:, -1])
        summary = {f: interval(v) for f, v in vectors[p].items()}
        for f in ("retention", "endpoint_retention"):
            summary[f]["ceiling_supported"] = summary[f]["upper_one_sided_95"] <= CEILING
        summary.update(learning_seconds=interval(a["learning_seconds"][:, -1]),
            per_replica={f: v.tolist() for f, v in vectors[p].items()})
        result["summaries"][p] = summary
        curve = dict(n=list(range(N+1)), observations=exposure.mean(0).tolist())
        for domain, values in (("query", query), ("base", old)):
            curve[domain] = dict(mean_mse=values.mean(0).tolist(),
                ci_mse=np.quantile(values[samples].mean(1), [.025, .975], axis=0).tolist(),
                frozen_mean_mse=float(values[:, 0].mean()))
        result["curves"][p] = curve
    a, b = vectors["K"], vectors["R"]
    contrast = dict(first="K", second="R", favorable_replicas=int(np.sum(a["acquisition"] < b["acquisition"])))
    for f in a:
        contrast[f+"_difference"] = interval(a[f]-b[f])
    for f in ("retention", "endpoint_retention"):
        d = contrast[f+"_difference"]
        contrast[f+"_noninferiority_supported"] = d["upper_one_sided_95"] <= TOLERANCE
        contrast[f+"_equivalent"] = d["ci"][0] >= -TOLERANCE and d["ci"][1] <= TOLERANCE
    contrast["retention_equivalent"] &= contrast["endpoint_retention_equivalent"]
    result["contrast"] = contrast
    result["efficiency"] = _efficiency(arrays["K"], arrays["R"], samples, inverses,
        result["summaries"]["K"], result["summaries"]["R"], contrast, False)
    return result
