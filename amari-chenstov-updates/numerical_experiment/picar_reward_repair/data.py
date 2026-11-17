"""Read-only recovery of the exact stationary histories and post-action frames."""

from __future__ import annotations

import copy
from pathlib import Path

from numerical_experiment.picar.models import batch_digest
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_phase2.campaign import load_completed_state, read_json
from numerical_experiment.picar_initialization.data import namespace_data


def inspect_source(root):
    root = Path(root).resolve()
    unit = read_completed_unit(root / "orchestration")
    if unit["summary"].get("kind") != "initialization_campaign":
        raise ValueError("expected the completed stationary initialization repair")
    project = Path(__file__).resolve().parents[2]
    for name, expected in unit["config"].get("source_sha256", {}).items():
        path = (project / name).resolve()
        if not path.is_relative_to(project) or sha256_file(path) != expected:
            raise ValueError("historical numerical source changed: " + name)
    summary = unit["summary"]
    paths = {"fit": Path(summary["selected_fit"]), "asset": Path(summary["initial_asset"])}
    for path in paths.values():
        if not path.is_relative_to(root):
            raise ValueError("initializer artifact is outside its source campaign")
        read_completed_unit(path)
    return {"root": str(root), **{k: str(v) for k, v in paths.items()},
        "completion_sha256": sha256_file(root / "orchestration/COMPLETED"),
        "fit_completion_sha256": sha256_file(paths["fit"] / "COMPLETED"),
        "asset_completion_sha256": sha256_file(paths["asset"] / "COMPLETED")}


def restore_stationary(source):
    from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations

    root = Path(source["root"])
    selected = load_completed_state(source["fit"])
    if selected.get("stage") != "fit_done" or not selected.get("distilled_learner"):
        raise ValueError("completed source must contain the common post-KL checkpoint")
    wanted = {split: {r["record_id"]: r for r in selected["identity"]["data"][split]}
              for split in ("train", "validation")}
    if {k: len(v) for k, v in wanted.items()} != {"train": 515, "validation": 65}:
        raise ValueError("the declared stationary train/validation identities differ")
    parent = read_json(root / "orchestration/config.json")
    original_root = Path(parent["teacher_source"])
    original = namespace_data(restore_demonstrations(original_root), "original")
    chunks = [("original", original_root, original)]
    acquired = original["acquired_count"]
    paths = []
    for marker in (root / "units").glob("collection-*/COMPLETED"):
        read_completed_unit(marker.parent)
        settings = read_json(marker.parent / "config.json")["settings"]
        paths.append((settings["index"], marker.parent))
    for index, path in sorted(paths):
        saved = load_completed_state(path)
        chunk = namespace_data({"records": saved["eligible"], "episodes": saved["episodes"],
            "acquired_count": saved["acquired_count"]}, f"collection-{index}", arrival_offset=acquired)
        chunks.append((f"collection-{index}", path, chunk))
        acquired += chunk["acquired_count"]
    records, episode_sources = {}, {}
    for namespace, path, data in chunks:
        for record in data["records"]:
            records[record["record_id"]] = record
        for episode in data["episodes"]:
            episode_sources[episode["episode_id"]] = {"episode": episode, "frames_root": str(path / "frames"),
                "source_unit": str(path), "namespace": namespace}
    splits = {}
    for split, identities in wanted.items():
        splits[split] = []
        for record_id, reference in sorted(identities.items()):
            if record_id not in records:
                raise ValueError("missing declared stationary record: " + record_id)
            record = records[record_id]
            if (record["arrival"] != reference["arrival"] or record["episode_id"] != reference["episode_id"]
                    or batch_digest(record["history"]) != reference["history_sha256"]):
                raise ValueError("stationary history identity differs")
            splits[split].append(record)
    train_episodes = {r["episode_id"] for r in splits["train"]}
    validation_episodes = {r["episode_id"] for r in splits["validation"]}
    if train_episodes & validation_episodes:
        raise ValueError("stationary whole-episode split overlaps")
    selected_episodes = [episode_sources[e] for e in sorted(train_episodes | validation_episodes)]
    for item in selected_episodes:
        episode = item["episode"]
        if (not episode["complete"] or not episode["label_complete"] or episode.get("truncated")
                or episode.get("evaluation") or episode.get("reset_exposure") != 0.):
            raise ValueError("stationary labels require completed initial-A training episodes")
        if not episode["intervals"] or not episode["intervals"][-1]["terminated"]:
            raise ValueError("stationary episode has no terminal observation")
    return {**splits, "episodes": selected_episodes, "acquired_count": selected["acquired_count"],
        "distilled_learner": copy.deepcopy(selected["distilled_learner"]),
        "distill_receipt": copy.deepcopy(selected["distill_receipt"]),
        "shared_kl_selection": copy.deepcopy(selected["selection"]["kl"]),
        "source_fit_identity": selected["identity"]}


def image_jobs(data):
    jobs = {}
    for item in data["episodes"]:
        for row in item["episode"]["intervals"]:
            ref = row["frame_ref"]
            jobs.setdefault(ref, {"frame_ref": ref, "frames_root": item["frames_root"]})
    return [jobs[key] for key in sorted(jobs)]


def manifest(data, source):
    return {"source": source, "train_count": len(data["train"]), "validation_count": len(data["validation"]),
        "train_ids": [r["record_id"] for r in data["train"]],
        "validation_ids": [r["record_id"] for r in data["validation"]],
        "episodes": len(data["episodes"]), "unique_images": len(image_jobs(data)),
        "acquired_count": data["acquired_count"], "new_observations": 0,
        "shared_distilled_checkpoint_recovered": True, "stationary_A": True,
        "whole_episode_split": True, "initial_pi_used": False}


def relabel(data, scores, condition, edges):
    import torch
    from .math import episode_labels

    targets, episode_rows = {}, []
    for item in data["episodes"]:
        episode = item["episode"]
        rows = episode["intervals"]
        bonuses = episode["terminal_bonuses"]
        if len(rows) != len(bonuses):
            raise ValueError("source terminal bonuses do not align with recorded intervals")
        values = episode_labels(
            torch.tensor([r["elapsed_s"] for r in rows], dtype=torch.float64),
            torch.tensor([scores[r["frame_ref"]]["score"] for r in rows], dtype=torch.float64),
            torch.tensor(bonuses, dtype=torch.float64),
            torch.tensor([r["terminated"] for r in rows]), condition, edges.double().cpu())
        for index, row in enumerate(rows):
            key = (episode["episode_id"], row["arrival"])
            targets[key] = {"return": float(values["coded_returns"][index]),
                "raw_return": float(values["raw_returns"][index]), "bin": int(values["bins"][index])}
        episode_rows.append({"episode_id": episode["episode_id"], "end_reason": episode["end_reason"],
            "raw_return": float(values["raw_returns"][0]), "coded_return": float(values["coded_returns"][0]),
            "intervals": len(rows)})
    result = {}
    for split in ("train", "validation"):
        result[split] = [{**r, **targets[(r["episode_id"], r["source_arrival"])], "learned": False}
                         for r in data[split]]
    baseline = sum(r["return"] for r in result["train"]) / len(result["train"])
    for record in result["train"] + result["validation"]:
        record["baseline"] = baseline
    return {**result, "baseline_coded": baseline, "episode_rows": episode_rows,
        "coded_return_range": [min(r["return"] for r in result["train"]),
                               max(r["return"] for r in result["train"])],
        "initial_pi_used": False, "baseline_source": "mean_of_training_coded_returns_only"}
