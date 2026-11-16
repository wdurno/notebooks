"""Read-only restoration of declared recorded sequences and validation pools."""

import copy
from pathlib import Path

from numerical_experiment.picar.config import config_from_dict
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar_phase2.campaign import load_completed_state, read_json
from .config import SOURCE_PILOT


def inspect_source(source):
    root = Path(source).resolve()
    parent = read_completed_unit(root / "orchestration")
    if parent["summary"].get("kind") != "initialization_campaign":
        raise ValueError("solver repair requires the completed Fix 1 campaign")
    for name, expected in parent["stored_config"]["source_sha256"].items():
        if sha256_file(Path(__file__).resolve().parents[2] / name) != expected:
            raise ValueError("historical scientific source changed: " + name)
    asset = Path(parent["summary"]["initial_asset"])
    fit = Path(parent["summary"]["selected_fit"])
    if not asset.is_relative_to(root) or not fit.is_relative_to(root):
        raise ValueError("initial source paths are outside source campaign")
    for path in (asset, fit, root / "units" / SOURCE_PILOT):
        read_completed_unit(path)
    return {"root": str(root), "asset": str(asset), "fit": str(fit),
        "pilot": str(root / "units" / SOURCE_PILOT),
        "asset_id": sha256_file(asset / "COMPLETED"),
        "completion": sha256_file(root / "orchestration/COMPLETED")}


def initial_panels(source):
    for marker in sorted((Path(source["root"]) / "units").glob("evaluation-*/COMPLETED")):
        row = read_completed_unit(marker.parent)["summary"]
        if (row.get("checkpoint") == "accepted_nll" and len(row.get("evaluations", [])) == 2
                and all(e.get("initial_asset_id") == source["asset_id"] for e in row["evaluations"])):
            return copy.deepcopy(row["evaluations"])
    raise ValueError("completed initial A/B panel for the primary asset is missing")


def _records(path, summary, config):
    import torch
    from src.robotics.actions import Calibration
    from numerical_experiment.picar.storage import FrameStore
    from numerical_experiment.picar_phase2_rescue.data import _history
    from src.continual import picar_math as pm

    frames = FrameStore(Path(path) / "frames", readonly=True)
    edges = torch.linspace(config.learning.return_min, config.learning.return_max,
        config.model.return_bins + 1, dtype=torch.float32).double()
    records = []
    for episode in summary["episodes"]:
        if not episode.get("label_complete") or episode.get("truncated") or episode.get("evaluation"):
            continue
        if len(episode["intervals"]) != len(episode["returns"]):
            raise ValueError("recorded intervals and returns do not align")
        bins = pm.bin_returns(torch.tensor(episode["returns"], dtype=torch.float64), edges)
        for row, label, value in zip(episode["intervals"], bins, episode["returns"]):
            if row["logits"] is None:
                continue
            records.append({"episode_id": str(path) + "/" + episode["episode_id"],
                "arrival": row["arrival"], "history": _history(row, frames, config, Calibration()),
                "baseline": row["baseline"], "logits": torch.tensor(row["logits"], dtype=torch.float32),
                "bin": int(label), "return": value, "learned": False,
                "record_id": str(path) + "/" + str(row["arrival"]),
                "source_exposure": episode.get("reset_exposure", 0.)})
    return records


def restore(source):
    """Initial/Fisher records never enter this screen's replay or fitting."""
    from numerical_experiment.picar_initialization.data import namespace_data
    from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations

    initial = load_completed_state(source["asset"])
    if initial["eligible"] or initial["learner"]["retired_count"] != 515:
        raise ValueError("expected primary equal-weight initialization with empty replay")
    config = config_from_dict(initial["identity"]["config"])
    parent_config = read_json(Path(source["root"]) / "orchestration/config.json")
    fitted = load_completed_state(source["fit"])
    wanted = {r["record_id"] for r in fitted["identity"]["data"]["validation"]}
    original = namespace_data(restore_demonstrations(parent_config["teacher_source"]), "original")
    pool = list(original["records"])
    acquired = original["acquired_count"]
    chunks = []
    for marker in (Path(source["root"]) / "units").glob("collection-*/COMPLETED"):
        chunk_config = read_json(marker.parent / "config.json")
        chunks.append((chunk_config["settings"]["index"], marker.parent))
    for index, path in sorted(chunks):
        chunk = load_completed_state(path)
        restored = namespace_data({"records": chunk["eligible"], "episodes": chunk["episodes"],
            "acquired_count": chunk["acquired_count"]}, f"collection-{index}", arrival_offset=acquired)
        pool.extend(restored["records"])
        acquired += chunk["acquired_count"]
    old = [r for r in pool if r["record_id"] in wanted]
    if len(old) != len(wanted) or len(old) != 65:
        raise ValueError("fixed initial validation pool could not be restored")
    old = _spaced(old, 8)
    summary = read_completed_unit(source["pilot"])["summary"]
    records = _records(source["pilot"], summary, config)
    ids = list(dict.fromkeys(r["episode_id"] for r in records))
    if len(ids) < 12:
        raise ValueError("recorded source needs disjoint episode segments")
    validation_ids = set(ids[len(ids) // 2 - 2:len(ids) // 2])
    live = _spaced([r for r in records if r["episode_id"] in validation_ids], 8)
    train = [r for r in records if r["episode_id"] not in validation_ids]
    early = train[:64]
    late = [r for r in train if r["arrival"] >= records[len(records) // 2]["arrival"]][:64]
    if len(early) != 64 or len(late) != 64 or {r["record_id"] for r in early} & {r["record_id"] for r in late}:
        raise ValueError("recorded segments are incomplete or overlap")
    if {r["episode_id"] for r in early + late} & validation_ids:
        raise ValueError("recorded validation episode leaked into a training segment")
    return initial, {"early": early, "late": late}, {"old": old, "live": live}


def _spaced(records, count):
    if len(records) <= count:
        return records
    return [records[round(i * (len(records) - 1) / (count - 1))] for i in range(count)]


def manifest(source, segments, validation):
    return {"source": source, "segments": {name: {"record_ids": [r["record_id"] for r in rows],
        "episode_ids": sorted({r["episode_id"] for r in rows}), "count": len(rows),
        "source_exposure_range": [min(r["source_exposure"] for r in rows), max(r["source_exposure"] for r in rows)],
        "start_asset": source["asset_id"], "start_state": "common_pre_stream_initialization_not_historical_late_checkpoint"}
        for name, rows in segments.items()}, "validation": {name: {"record_ids": [r.get("record_id", str(r["arrival"])) for r in rows],
        "episode_ids": sorted({r["episode_id"] for r in rows}), "count": len(rows)}
        for name, rows in validation.items()}, "new_observations": 0, "teacher_calls": 0,
        "purpose": "conditional_recorded_sequence_diagnostic_not_counterfactual_rollout"}
