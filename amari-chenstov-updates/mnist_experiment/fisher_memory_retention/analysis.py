"""Paired whole-trajectory inference at the prospectively fixed retention target."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import (
    summarize_trajectory, replica_draw_indices, flatten_record,
)
from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe

from . import PROTOCOL


def analyze(records, manifest, *, seed, draws=2000, check=lambda: None):
    if manifest.get("protocol") != PROTOCOL or manifest.get("sealed") is not True:
        raise ValueError("incompatible or unsealed confirmation manifest")
    replicas = manifest["replicas"]
    if len(replicas)<2 or len(set(replicas))!=len(replicas):
        raise ValueError("confirmation requires distinct paired replicas")
    recipes = [Recipe.from_mapping(r) for r in manifest["recipes"]]
    if {r.method for r in recipes}!=set(METHODS) or len(recipes)!=3:
        raise ValueError("exactly three frozen methods are required")
    expected = {(r.key,replica) for r in recipes for replica in replicas}
    index = {}
    for raw in records:
        check()
        row = flatten_record(raw)
        key=(row["recipe_key"],row["replica"])
        if key not in expected or key in index:
            raise ValueError("duplicate or unplanned confirmation record")
        if row.get("partition")!="confirmation" or row.get("protocol")!=PROTOCOL or not row.get("valid"):
            raise ValueError("development or invalid trajectory cannot enter confirmation")
        frozen = next(r for r in recipes if r.key==key[0])
        if Recipe.from_mapping(row["recipe"])!=frozen:
            raise ValueError("confirmation recipe changed")
        summary=summarize_trajectory(row["metrics"],N=manifest["N"])
        for name in ("novel_auc","base_auc","frozen_novel_mse","frozen_base_mse","observations","learning_seconds"):
            if not math.isclose(summary[name],row[name],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError(f"trajectory summary differs from predictions: {name}")
        index[key]=row
    if set(index)!=expected:
        raise ValueError("incomplete frozen cohort; final inference is pending")
    pairing=[]
    for replica in replicas:
        arms=[index[r.key,replica] for r in recipes]
        pairing.append(arms[0].get("pairing_id"))
        if not pairing[-1] or any(a.get("pairing_id")!=pairing[-1] for a in arms):
            raise ValueError("mismatched pairing provenance")
        for arm in arms[1:]:
            if any(not math.isclose(arm[k],arms[0][k],rel_tol=1e-10,abs_tol=1e-10)
                   for k in ("frozen_base_mse","frozen_novel_mse")):
                raise ValueError("mismatched initialization")
            if [m["observations"] for m in arm["metrics"]]!=[m["observations"] for m in arms[0]["metrics"]]:
                raise ValueError("mismatched data exposures")
    if len(set(pairing))!=len(pairing):
        raise ValueError("replicas cannot share a pairing identity")
    if type(draws) is not int or draws<2:
        raise ValueError("at least two bootstrap draws required")
    selections=np.array([replica_draw_indices(len(replicas),seed,d) for d in range(draws)])

    def interval(values):
        values=np.asarray(values,dtype=float)
        if not np.isfinite(values).all(): raise ValueError("nonfinite statistic")
        means=np.empty(draws)
        for start in range(0,draws,100):
            check()
            take=selections[start:start+100]
            means[start:start+len(take)]=values[take].mean(axis=1)
        return dict(mean=float(values.mean()),ci=np.quantile(means,[.025,.975]).tolist(),
            upper_one_sided_95=float(np.quantile(means,.95)),n_pairs=len(values),
            favorable_replicas=int((values>0).sum()),
            standard_error=float(values.std(ddof=1)/math.sqrt(len(values))))

    result=dict(schema_version=1,protocol=PROTOCOL,status="completed",manifest_hash=manifest["manifest_hash"],
        replicas=replicas,N=manifest["N"],delta=manifest["delta"],epsilon=manifest["epsilon"],
        methods={},comparisons={},bootstrap=dict(draws=draws,seed=seed,
            resampling_unit="paired_whole_replica",confidence_level=.95,
            curve_bands="pointwise_not_simultaneous"),
        assumptions="Independent paired replicas with adequate squared-error moments; nominal intervals "
                    "conditional on the reused benchmark, panels and frozen recipes, not familywise inference.")
    vectors={}
    for recipe in recipes:
        arms=[index[recipe.key,r] for r in replicas]
        acquisition=np.array([a["novel_auc"] for a in arms])
        retention=np.array([math.sqrt(a["base_auc"])-math.sqrt(a["frozen_base_mse"]) for a in arms])
        frozen=np.array([a["frozen_novel_mse"] for a in arms])
        vectors[recipe.method]=(acquisition,retention)
        h=interval(retention)
        h.update(point_ceiling_feasible=h["mean"]<=manifest["delta"],
                 ceiling_supported=h["upper_one_sided_95"]<=manifest["delta"])
        seconds=sum(a["learning_seconds"] for a in arms)/sum(a["observations"] for a in arms)
        vram=max(a["peak_vram_bytes"] for a in arms)
        costs=dict(mean_learning_seconds=float(np.mean([a["learning_seconds"] for a in arms])),
            seconds_per_arrival=seconds,peak_vram_bytes=vram,
            mean_evaluation_seconds=float(np.mean([a["evaluation_seconds"] for a in arms])),
            mean_fit_seconds=float(np.mean([a["fit_seconds"] for a in arms])),
            mean_eviction_seconds=float(np.mean([a["eviction_seconds"] for a in arms])),
            resource_feasible=seconds<=manifest["learning_time_limit"] and vram<=manifest["memory_ceiling_bytes"])
        curves={}
        for domain in ("novel","base"):
            matrix=np.array([[m[f"{domain}_mse"] for m in a["metrics"]] for a in arms])
            boot=np.empty((draws,matrix.shape[1]))
            for start in range(0,draws,100):
                check(); take=selections[start:start+100]
                boot[start:start+len(take)]=matrix[take].mean(axis=1)
            curves[domain]=dict(mean_mse=matrix.mean(axis=0).tolist(),
                ci_mse=np.quantile(boot,[.025,.975],axis=0).tolist(),
                mean_rmse=np.sqrt(matrix.mean(axis=0)).tolist())
        curves["n"]=list(range(manifest["N"]+1))
        for name in ("observations","learning_seconds"):
            curves[name]=np.mean([[m[name] for m in a["metrics"]] for a in arms],axis=0).tolist()
        retired=[item["squared_error"] for a in arms for item in a.get("retired_audit",[])]
        result["methods"][recipe.method]=dict(recipe=recipe.mapping(),recipe_key=recipe.key,
            acquisition=interval(acquisition),retention=h,frozen_gain=interval(frozen-acquisition),
            costs=costs,curves=curves,retirement_audit=dict(count=len(retired),
                mean_squared_error=float(np.mean(retired)) if retired else None,
                interpretation="Descriptive fit on first retired novel training examples; not independent retention."))
    for a,b in combinations(METHODS,2):
        # Positive acquisition difference favors the first named method.
        gain=interval(vectors[b][0]-vectors[a][0])
        harm=interval(vectors[a][1]-vectors[b][1])
        equivalent=harm["ci"][0]>=-manifest["epsilon"] and harm["ci"][1]<=manifest["epsilon"]
        result["comparisons"][f"{a}_vs_{b}"]=dict(first=a,second=b,acquisition_gain=gain,
            retention_difference=harm,retention_equivalence_supported=equivalent,
            primary=(a=="fd_replay_ewc" and b=="replay_ewc"))
    return result
