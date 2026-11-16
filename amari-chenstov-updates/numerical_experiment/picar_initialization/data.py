"""Stationary episode collection and disjoint, nested initialization data."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
import hashlib
import math

from numerical_experiment.picar.fix3_student import _Run
from numerical_experiment.picar_phase2.student import _coverage
from .config import SPLIT_SEED


def validation_episode(episode_id, seed=SPLIT_SEED):
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def namespace_data(data, namespace, *, arrival_offset=0):
    """Make IDs unique across chunks without changing recorded histories/labels."""
    records = []
    for row in data["records"]:
        item = dict(row)
        item["source_episode_id"] = row["episode_id"]
        item["episode_id"] = f"{namespace}/{row['episode_id']}"
        item["source_arrival"] = row["arrival"]
        item["arrival"] = arrival_offset + row["arrival"]
        item["record_id"] = f"{item['episode_id']}/{row['arrival']}"
        item["learned"] = False
        records.append(item)
    episodes = copy.deepcopy(data["episodes"])
    for episode in episodes:
        episode["episode_id"] = f"{namespace}/{episode['episode_id']}"
    return {**data, "records": records, "episodes": episodes}


def split_records(records, *, fixed_validation_ids=None):
    """Validation episodes never become training, even beyond the fixed panel."""
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate initialization record IDs")
    train = [r for r in records if not validation_episode(r["episode_id"])]
    validation = [r for r in records if validation_episode(r["episode_id"])]
    if fixed_validation_ids is not None:
        wanted = set(fixed_validation_ids)
        validation = [r for r in validation if r["record_id"] in wanted]
        if {r["record_id"] for r in validation} != wanted:
            raise ValueError("fixed validation panel is incomplete or crosses training")
    if not train or not validation:
        raise ValueError("whole-episode split needs both training and validation")
    train_episodes = {r["episode_id"] for r in train}
    if train_episodes & {r["episode_id"] for r in validation}:
        raise ValueError("training and validation episodes overlap")
    return train, validation


def prefix_episodes(records, target):
    """Stop only at an episode boundary; smaller rungs remain true prefixes."""
    result = []
    for row in records:
        if len(result) >= target and row["episode_id"] != result[-1]["episode_id"]:
            break
        result.append(row)
    return result


def coverage(records, episodes):
    counts = Counter(int(row["executed_probabilities"].argmax()) for row in records)
    returns = [row["return"] for row in records]
    intervals = [row for episode in episodes for row in episode["intervals"]]
    geometry = []
    for episode in episodes:
        layout = episode.get("layout", {})
        if "start" in layout and "target" in layout and "yaw" in layout:
            dx, dy = [b - a for a, b in zip(layout["start"], layout["target"])]
            bearing = math.atan2(dy, dx) - layout["yaw"]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            geometry.append({"episode_id": episode["episode_id"],
                "initial_distance_m": math.hypot(dx, dy),
                "initial_bearing_degrees": math.degrees(bearing),
                "direct_path_blocked": layout.get("curriculum", {}).get("direct_path_blocked"),
                "feasible_routes": layout.get("curriculum", {}).get("feasible_routes", [])})
    return {"records": len(records), "episodes": len(episodes),
        "dominant_actions": {str(i): counts[i] for i in range(8)},
        "distinct_dominant_actions": len(counts),
        "return_min": min(returns) if returns else None,
        "return_max": max(returns) if returns else None,
        "return_bin_counts": dict(Counter(str(row["bin"]) for row in records)),
        "outcomes": dict(Counter(episode["end_reason"] for episode in episodes)),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "episode_geometry": geometry,
        "geometry_is_reader_only": True,
        "controls": _coverage(intervals),
        "conditioning": "initial_A_fixed_distribution; dependent_within_episode",
        "degenerate_teacher_action_coverage": len(counts) < 2}


class CollectionRun(_Run):
    """Use proven real-time acquisition/recovery, never its historical fitting."""

    def __init__(self, model, config, sim_config, store, stopped, *, seed, max_transitions):
        sim_config = replace(sim_config, curriculum_exposure_override=0., domain_override="A")
        super().__init__(model, replace(config, seed=seed), sim_config, store, stopped,
            mode="bootstrap", seed=seed, max_transitions=max_transitions)

    def run(self):
        if self.s["stage"] == "collect":
            self.collect()
        if self.s["stage"] not in {"distill", "collection_done"}:
            raise ValueError("collector cannot enter an optimization/Fisher stage")
        if "incomplete_episode" in self.s:
            raise ValueError("unfinished episode is not completed initialization data")
        if self.learner.update_count or self.learner.retired_count:
            raise ValueError("stationary collection must not fit or archive data")
        self.s["stage"] = "collection_done"
        self.save()
        return {"records": len(self.s["eligible"]), "acquired_count": self.s["acquired_count"],
            "episodes": self.s["episodes"], "timings": self.s["timings"],
            "coverage": coverage(self.s["eligible"], self.s["episodes"]),
            "stationary_A": True, "fitting_updates": 0, "initial_pi_used": False}


def evaluate_initial(model, config, initial, store, stopped, *, evaluations, asset_id):
    """Reuse real-time frozen-policy panels without admitting learning episodes."""
    from numerical_experiment.picar_phase3.student import DevelopmentRun
    from .config import DEVELOPMENT_SEEDS, DEFAULT_RECIPE, simulation
    runner = DevelopmentRun(model, config, simulation(DEFAULT_RECIPE), store, stopped,
        mode="pilot", seed=DEVELOPMENT_SEEDS[0], max_transitions=640,
        initial=initial, evaluations=evaluations, initial_asset_id=asset_id)
    if runner.s["stage"] == "initial_evaluation":
        runner.evaluate("initial")
    if runner.s["acquired_count"] or runner.learner.update_count != initial["learner"]["update_count"]:
        raise ValueError("initial capability evaluation must not learn or acquire training data")
    runner.s["stage"] = "done"
    runner.save()
    return {"evaluations": runner.s["evaluations"], "videos": runner.s["videos"],
        "timings": runner.s["timings"], "training_acquisitions": 0, "learning": False}
