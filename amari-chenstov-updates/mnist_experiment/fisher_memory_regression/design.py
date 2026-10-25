"""Deterministic development decisions; confirmation outcomes are not inputs."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import PROTOCOL
from .analysis import summarize_trajectory
from .config import Config, Recipe


def _finite(value, name, *, minimum=0.):
    if isinstance(value, (str, bool)):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def _development_records(records, *, target=None, require_valid=False):
    """Validate complete evidence without dropping numerical failures or replicas.

    Invalid-but-finite trajectories remain visible to aggregate(), which marks
    their entire policy invalid. Registration requires valid selected evidence.
    Pairing checks compare every policy sharing a replica, not just its integer ID.
    """
    seen, families, targets, paired, plans = set(), set(), set(), {}, {}
    for row in records:
        try:
            if row.get("schema_version") != 1 or row.get("protocol", PROTOCOL) != PROTOCOL:
                raise ValueError("incompatible development record schema/protocol")
            family = row["family"]
            if (row["partition"] != "development" or not isinstance(family, str)
                    or not family or family == "confirmation"):
                raise ValueError("design decisions require development-only evidence")
            families.add(family)
            event_target, replica = row["target"], row["replica"]
            if type(event_target) is not int or event_target < 1 or type(replica) is not int or replica < 0:
                raise ValueError("target and replica must be valid integer identities")
            if target is not None and event_target != target:
                raise ValueError("incomplete development trajectory for registered sample budget")
            if row.get("N", event_target) != event_target:
                raise ValueError("incompatible development target/N")
            targets.add(event_target)
            policy = Recipe(**row["recipe"])
            if row["recipe_key"] != policy.key:
                raise ValueError("development recipe key disagrees with its configuration")
            for name in ("method", "capacity", "replay_count"):
                if row.get(name) != getattr(policy, name):
                    raise ValueError(f"development recipe disagrees with {name}")
            key = row["recipe_key"], replica
            if key in seen:
                raise ValueError("duplicate replica in development policy estimate")
            seen.add(key)
            if type(row["valid"]) is not bool or (require_valid and not row["valid"]):
                raise ValueError("invalid selected development trajectory")
            plan = row["plan_hash"]
            if not isinstance(plan, str) or not plan or row.get("pairing_id", plan) != plan:
                raise ValueError("invalid development pairing provenance")
            if plan in plans and plans[plan] != replica:
                raise ValueError("distinct development replicas reuse the same data plan")
            plans[plan] = replica
            metrics = row["metrics"]
            if not metrics or metrics[0]["n"] != 0:
                raise ValueError("development evidence must include initialization")
            summary = summarize_trajectory(metrics, N=event_target,
                frozen_novel_mse=row["frozen_novel_mse"], frozen_base_mse=row["frozen_base_mse"])
            for name in ("novel_auc", "base_auc", "observations", "learning_seconds"):
                if not math.isclose(_finite(row[name], name), summary[name], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"development summary disagrees with complete metrics: {name}")
            _finite(row["active_seconds"], "active_seconds")
            _finite(row.get("peak_vram_bytes", 0), "peak_vram_bytes")
            reference = paired.setdefault(replica, row)
            if reference["plan_hash"] != plan:
                raise ValueError("unpaired development data plans")
            for name in ("frozen_novel_mse", "frozen_base_mse"):
                if not math.isclose(row[name], reference[name], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("unpaired development initialization/panels")
            if [item["observations"] for item in metrics] != [item["observations"] for item in reference["metrics"]]:
                raise ValueError("unpaired development observation counts")
        except (KeyError, TypeError, StopIteration) as error:
            raise ValueError(f"incomplete or malformed development evidence: {error}") from error
    if len(families) > 1 or len(targets) > 1:
        raise ValueError("incompatible development families or trajectory targets")


def aggregate(records: list[dict]) -> list[dict]:
    _development_records(records)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["recipe_key"]].append(row)
    result = []
    for key, rows in grouped.items():
        if len({row["replica"] for row in rows}) != len(rows):
            raise ValueError("duplicate replica in development policy estimate")
        recipe = rows[0]["recipe"]
        if any(row["recipe"] != recipe or row["target"] != rows[0]["target"] for row in rows):
            raise ValueError("incompatible development trajectories")
        observations = sum(row["observations"] for row in rows)
        result.append(dict(recipe_key=key, recipe=recipe, method=recipe["method"],
            capacity=recipe["capacity"], replay_count=recipe["replay_count"],
            replicas=sorted(row["replica"] for row in rows), count=len(rows),
            novel_auc=float(np.mean([row["novel_auc"] for row in rows])),
            base_harm=float(np.mean([math.sqrt(row["base_auc"]) - math.sqrt(row["frozen_base_mse"])
                                     for row in rows])),
            frozen_novel_mse=float(np.mean([row["frozen_novel_mse"] for row in rows])),
            seconds_per_arrival=sum(row["learning_seconds"] for row in rows) / observations,
            mean_active_seconds=float(np.mean([row["active_seconds"] for row in rows])),
            valid=all(row["valid"] for row in rows),
            max_vram_bytes=max(row.get("peak_vram_bytes", 0) for row in rows)))
    return sorted(result, key=lambda row: row["recipe_key"])


def rank_candidates(rows: list[dict], cfg: Config, *, envelope=math.inf, memory_ceiling=math.inf):
    for bound in (envelope, memory_ceiling):
        if math.isnan(bound) or bound < 0:
            raise ValueError("invalid resource envelope")
    for row in rows:
        if row["valid"]:
            for name in ("seconds_per_arrival", "novel_auc", "frozen_novel_mse"):
                _finite(row[name], name)
            _finite(row["base_harm"], "base_harm", minimum=-math.inf)
            _finite(row.get("max_vram_bytes", 0), "max_vram_bytes")
    eligible = [row for row in rows if row["valid"] and row["seconds_per_arrival"] <= envelope
                and row.get("max_vram_bytes", 0) <= memory_ceiling]
    if not eligible:
        raise ValueError("no numerically valid resource-feasible development candidate")
    admissible = [row for row in eligible if row["base_harm"] <= cfg.retention_margin_degrees]
    warning = None
    if not admissible:
        smallest = min(row["base_harm"] for row in eligible)
        admissible = [row for row in eligible if row["base_harm"] == smallest]
        warning = "No development policy met the base-retention tolerance; least violation carried forward."
    best = min(row["novel_auc"] for row in admissible)
    frozen = float(np.mean([row["frozen_novel_mse"] for row in admissible]))
    tolerance = cfg.acquisition_tie_fraction * frozen
    near = [row for row in admissible if row["novel_auc"] <= best + tolerance]
    order = lambda row: (row["seconds_per_arrival"], row["capacity"], row["replay_count"],
                         row["recipe"]["strength"], row["recipe"]["ridge"], row["recipe_key"])
    ranked = sorted(near, key=order) + sorted(
        [row for row in admissible if row not in near], key=lambda row: (row["novel_auc"], *order(row)))
    return ranked, warning


def choose(rows: list[dict], cfg: Config, **kwargs) -> dict:
    ranked, warning = rank_candidates(rows, cfg, **kwargs)
    return dict(selected=ranked[0], warning=warning,
                practical_tie_fraction=cfg.acquisition_tie_fraction,
                ranking=[row["recipe_key"] for row in ranked])


def grid(cfg: Config, calibration: dict, *, seconds_available: float) -> dict:
    """Reduce breadth from timing before looking at allocation outcomes."""
    cells = [(c, b) for c in cfg.capacities for b in cfg.replay_counts if b <= c]
    costs = {method: _finite(calibration["mean_active_seconds"][method], f"{method} timing")
             for method in ("fd_replay_ewc", "replay_ewc", "ewc_only")}
    per_pair = costs["fd_replay_ewc"] + costs["replay_ewc"]
    initialization = _finite(calibration["mean_initialization_seconds"], "initialization timing")
    # Shared assets are created once per fresh replica, not once per policy/cell.
    fixed_cost = 1.5 * cfg.allocation_replicas * (costs["ewc_only"] + initialization)
    cell_cost = max(1., 1.5 * cfg.allocation_replicas * per_pair)
    allowance = max(0., _finite(seconds_available, "seconds_available")
                    - cfg.confirmation_reserve_seconds - cfg.analysis_reserve_seconds)
    affordable = max(0, math.floor((allowance - fixed_cost) / cell_cost))
    if affordable < 1 or not cells:
        raise ValueError("no allocation cell fits after initialization and reserved work")
    if affordable < len(cells):
        reference = (cfg.reference_capacity, cfg.reference_replay)
        ordered = []
        for cell in (reference, (min(cfg.capacities), min(cfg.replay_counts)),
                     (max(cfg.capacities), min(cfg.replay_counts)),
                     (max(cfg.capacities), max(cfg.replay_counts)), *cells):
            if cell in cells and cell not in ordered:
                ordered.append(cell)
        cells = ordered[:affordable]
    if not cells:
        raise ValueError("no allocation cell fits the declared grid")
    return dict(cells=[list(cell) for cell in cells], replicas=list(range(cfg.allocation_replicas)),
                timing_reduced=len(cells) < sum(b <= c for c in cfg.capacities for b in cfg.replay_counts),
                predicted_seconds=fixed_cost + len(cells) * cell_cost,
                fit_review="pending", fit_registered=False)


def selected_recipes(cfg: Config, calibration: dict, selections: dict) -> list[Recipe]:
    recipes = []
    for method in ("fd_replay_ewc", "replay_ewc"):
        selected = selections[method]["selected"]
        policy = Recipe(**selected["recipe"])
        if policy.method != method or selected["recipe_key"] != policy.key:
            raise ValueError("selection does not match its declared method/recipe")
        recipes.append(policy)
    return recipes + [Recipe.from_config(cfg, "ewc_only", ridge=calibration["ridge"], policy=calibration["policy"])]


def efficiency_registration(records: list[dict], cfg: Config, recipes: list[Recipe]) -> dict:
    selected = [recipe for recipe in recipes if recipe.method != "ewc_only"]
    if len(selected) != 2 or {recipe.method for recipe in selected} != {"fd_replay_ewc", "replay_ewc"}:
        raise ValueError("sample-budget registration requires paired FD and replay recipes")
    _development_records(records, target=cfg.novel_count)
    groups = {recipe.method: [row for row in records if row["recipe_key"] == recipe.key] for recipe in selected}
    _development_records([row for group in groups.values() for row in group],
                         target=cfg.novel_count, require_valid=True)
    replica_sets = [{row["replica"] for row in group} for group in groups.values()]
    common = set.intersection(*replica_sets)
    if not common:
        raise ValueError("sample-budget registration requires paired development replicas")
    groups = {method: sorted((row for row in rows if row["replica"] in common), key=lambda row: row["replica"])
              for method, rows in groups.items()}

    def risk(method, n):
        return float(np.mean([row["metrics"][n]["novel_mse"] for row in groups[method]]))
    target = risk("replay_ewc", cfg.novel_count)
    initial = risk("fd_replay_ewc", 0)
    eligible = [n for n in sorted(cfg.smaller_budgets)
                if risk("fd_replay_ewc", n) <= target and risk("fd_replay_ewc", n) < initial]
    return dict(n_F=min(eligible) if eligible else max(cfg.smaller_budgets), n_R=cfg.novel_count,
                development_supported=bool(eligible), q_dev=target,
                development_replicas=sorted(common), frozen_initial_mse=initial,
                harm_margin=0., fit_registered=False,
                fit_reason="No preregistered development visual-fit approval; direct budget comparison remains primary.")


def confirmation_manifest(cfg: Config, calibration: dict, selections: dict, records: list[dict],
                          seconds_available: float) -> dict:
    recipes = selected_recipes(cfg, calibration, selections)
    _development_records(records, target=cfg.novel_count)
    selected_keys = {recipe.key for recipe in recipes}
    _development_records([row for row in records if row["recipe_key"] in selected_keys],
                         target=cfg.novel_count, require_valid=True)
    estimates = []
    for recipe in recipes:
        rows = [row for row in records if row["recipe_key"] == recipe.key]
        if rows:
            estimates.append(max(row["active_seconds"] for row in rows))
        else:
            estimates.append(_finite(calibration["mean_active_seconds"][recipe.method], "trajectory timing"))
    initialization = _finite(calibration["mean_initialization_seconds"], "initialization timing")
    per_replica = 1.5 * (sum(estimates) + initialization)
    count = min(cfg.confirmation_replicas, max(0, math.floor(
        (_finite(seconds_available, "seconds_available") - cfg.analysis_reserve_seconds) / max(per_replica, 1.))))
    if count < 2:
        raise ValueError("fewer than two complete fresh paired replicas fit the remaining allocation")
    return dict(schema_version=1, protocol=cfg.protocol, target=cfg.novel_count, N=cfg.novel_count,
                recipes=[dict(recipe.mapping(), recipe_key=recipe.key) for recipe in recipes], replicas=list(range(count)),
                precision_warning=count < cfg.preferred_min_confirmation,
                estimated_seconds=per_replica * count, retention_margin_degrees=cfg.retention_margin_degrees,
                base_retention_guard_degrees=cfg.retention_margin_degrees,
                envelope_seconds_per_arrival=calibration["envelope_seconds_per_arrival"],
                learning_seconds_per_observation_limit=calibration["envelope_seconds_per_arrival"],
                memory_ceiling_bytes=calibration["memory_ceiling_bytes"],
                **efficiency_registration(records, cfg, recipes))
