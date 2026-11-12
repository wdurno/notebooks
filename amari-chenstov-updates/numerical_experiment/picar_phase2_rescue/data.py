"""Restore original Fix3 A demonstrations for a NEW initialization, never replay.

Only sealed JSON and content-addressed frames are read. Old parameters, Adam
state, Fisher and anchors are deliberately neither loaded nor returned.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import torch

from numerical_experiment.picar.config import canonical_json, config_from_dict
from numerical_experiment.picar.models import batch_digest, validate_batch
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import FrameStore, IntegrityError, _read_json, _verify_file
from numerical_experiment.picar.teacher import prepare_teacher_distribution
from src.continual.picar_math import bin_returns
from src.robotics.actions import Calibration
from src.robotics.protocol import Control
from src.seeding import derive_seed_map


DATA_VERSION = "phase2-original-fix3-demonstrations-v1"
PROBABILITY_FLOOR = .001


def _sealed_summary(source):
    marker = _read_json(source / "COMPLETED")
    manifest = _read_json(source / "manifest.json")
    if (manifest["status"] != "completed"
            or sha256_file(source / "manifest.json") != marker["manifest_sha256"]):
        raise IntegrityError("bootstrap completion seal differs")
    _verify_file(source / "summary.json", manifest["final_files"]["summary.json"])
    resolved = _read_json(source / "config.json")
    if hashlib.sha256(canonical_json(resolved).encode()).hexdigest() != manifest["config_hash"]:
        raise IntegrityError("bootstrap resolved configuration hash differs")
    summary = _read_json(source / "summary.json")
    if summary["schema_version"] != 1 or summary["kind"] not in {"fix3_bootstrap", "phase1_fix3_bootstrap"}:
        raise ValueError("expected a completed Fix3 bootstrap summary, schema 1")
    return summary, resolved


def _history(row, frames, config, calibration):
    refs = row["context_refs"]
    if not refs or len(refs) != len(row["context_actions"]) or len(refs) != len(row["context_elapsed"]) \
            or len(refs) != len(row["context_decision_times"]):
        raise ValueError("recorded context lengths differ")
    images = []
    for ref in refs:
        rgb = frames.get(ref)
        if rgb.shape != (config.image_height, config.image_width, 3):
            raise ValueError("recorded frame dimensions differ from source configuration")
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)
    for action in row["context_actions"]:
        calibration.validate(Control(*action))
    history = {"images": torch.stack(images)[None],
        "executed_actions": torch.tensor(row["context_actions"], dtype=torch.float32)[None],
        "elapsed": torch.tensor(row["context_elapsed"], dtype=torch.float32)[None, :, None],
        "start_time": torch.tensor([row["context_start_time"]], dtype=torch.float32),
        "decision_time": torch.tensor([row["context_decision_times"]], dtype=torch.float32),
        "episode_deadline": torch.tensor([row["episode_deadline"]], dtype=torch.float32),
        "contexts": [copy.deepcopy(row["contexts"])],
        "status_events": [copy.deepcopy(row["context_status_events"])]}
    validate_batch(history, config.model.context_length)
    if batch_digest(history) != row["history_digest"]:
        raise IntegrityError("reconstructed history digest differs")
    return history


def _target(row, calibration):
    diagnostic = row.get("teacher_diagnostic")
    actions = diagnostic.get("actions") if isinstance(diagnostic, dict) else None
    if not isinstance(actions, list) or len(actions) != 1 or type(actions[0]) is not int or not 0 <= actions[0] < 8:
        raise ValueError("one recorded hard teacher action in [0,7] is required")
    onehot = torch.nn.functional.one_hot(torch.tensor(actions), 8).float()
    target = prepare_teacher_distribution(onehot, probability_floor=PROBABILITY_FLOOR,
                                          target_kind="hard_engineered")
    probabilities = torch.tensor(row["probabilities"], dtype=torch.float32)
    logits = torch.tensor(row["logits"], dtype=torch.float32)
    for value, expected in ((probabilities, target.executed_probabilities[0]),
                            (logits, target.reference_logits[0])):
        if value.shape != expected.shape or not torch.allclose(value, expected, rtol=1e-6, atol=1e-7):
            raise IntegrityError("recorded executed teacher probabilities/logits differ")
    expected = calibration.map_distribution(probabilities)
    # Terminal observations show the simulator's automatic safe stop, not a
    # different teacher distribution. Camera angles are retained by neutral().
    if row["terminated"]:
        expected = calibration.neutral(expected)
    actual = Control(**row["controls"])
    calibration.validate(actual)
    if any(not math.isclose(value, asdict(actual)[key], rel_tol=1e-6, abs_tol=1e-7)
           for key, value in asdict(expected).items()):
        raise IntegrityError("recorded executed control differs from teacher mapping/terminal stop")
    return target, probabilities, logits


def restore_demonstrations(source):
    """Return records/acquired_count/episodes/provenance; all records are unfitted.

    Arrival indices and full causal histories are preserved, including gaps for
    fallback intervals. The caller must use a fresh model and build new Fisher,
    anchors and Adam state. This is not permission to replay retired data online.
    """
    source = Path(source).resolve()
    summary, resolved = _sealed_summary(source)
    config = config_from_dict(resolved["config"])
    learning = config.learning
    if config.task_version != "ball-contact-v1":
        raise ValueError("restoration requires the original ball-contact-v1 task")
    if (config.model.return_bins, learning.return_min, learning.return_max, learning.teacher_floor,
            learning.reward_version) != (161, -6., 12., PROBABILITY_FLOOR, "dense-visual-centering-v1"):
        raise ValueError("restoration requires unchanged Fix3 return bins, reward and teacher floor")
    seeds = config.seeds(0)
    if summary["seed"] != config.seed or resolved["component_seeds"] != seeds:
        raise IntegrityError("bootstrap seed provenance differs")
    if summary.get("incomplete_episode"):
        raise ValueError("incomplete bootstrap episodes are not demonstration data")
    frames = FrameStore(source / "frames", readonly=True)
    calibration = Calibration()
    # Match the original learner's float32 edges promoted to double for labels.
    edges = torch.linspace(-6., 12., 162, dtype=torch.float32).double()
    records, frame_refs, captured = [], set(), 0
    for i, episode in enumerate(summary["episodes"]):
        expected_seed = derive_seed_map(seeds["layouts"], (f"episode_{i}",))[f"episode_{i}"]
        if episode["seed"] != expected_seed or episode["episode_id"] != f"bootstrap:{i}":
            raise IntegrityError("bootstrap episode seed/identity differs")
        if not episode["complete"] or not episode["label_complete"] or episode["truncated"] or episode["evaluation"]:
            raise ValueError("demonstrations require fully labeled, completed training episodes")
        if (episode.get("reset_exposure") != 0.
                or episode.get("layout", {}).get("curriculum", {}).get("curriculum_exposure_intervals") != 0.):
            raise ValueError("demonstrations must retain their original easy-A exposure")
        rows = episode["intervals"]
        if not rows or len(rows) != len(episode["returns"]) or not rows[-1]["terminated"]:
            raise ValueError("episode returns/terminal interval alignment differs")
        returns = torch.tensor(episode["returns"], dtype=torch.float64)
        bins = bin_returns(returns, edges)
        for j, row in enumerate(rows):
            if (row["arrival"] != captured or row["index"] != j or row["truncated"]
                    or row["terminated"] != (j == len(rows) - 1)):
                raise IntegrityError("captured arrival/index/terminal continuity differs")
            captured += 1
            frames.verify(row["frame_ref"])
            frame_refs.add(row["frame_ref"])
            if row["probabilities"] is None:
                if row.get("logits") is not None or row.get("teacher_diagnostic") is not None:
                    raise IntegrityError("fallback interval contains teacher evidence without probabilities")
                actual = Control(**row["controls"])
                calibration.validate(actual)
                if actual.drive_mps != 0. or actual.steering_rad != 0.:
                    raise IntegrityError("fallback interval must execute safe neutral")
                continue
            target, probabilities, logits = _target(row, calibration)
            history = _history(row, frames, config, calibration)
            frame_refs.update(row["context_refs"])
            baseline = float(row["baseline"])
            if not math.isfinite(baseline) or baseline != 0.:
                raise ValueError("Fix3 bootstrap uses the recorded fixed zero baseline")
            records.append({"episode_id": episode["episode_id"], "arrival": row["arrival"],
                "history": history, "baseline": baseline, "bin": int(bins[j]),
                "logits": logits, "return": float(returns[j]), "learned": False,
                "executed_probabilities": probabilities[None],
                "original_probabilities": target.original_probabilities,
                "target_kind": target.target_kind, "probability_floor": target.probability_floor,
                "source_history_digest": row["history_digest"], "source_seed": episode["seed"],
                "source_interval_index": j, "teacher_diagnostic": copy.deepcopy(row["teacher_diagnostic"])})
    digests = [r["source_history_digest"] for r in records]
    receipt = summary["distill_receipt"]
    if (not records or captured != summary["acquired_count"] or receipt["batches"] != len(records)
            or receipt["history_digests"] != digests
            or summary["distinct_bootstrap_samples"] != len(set(digests))):
        raise IntegrityError("restored dataset counts/history order differ from original bootstrap receipt")
    provenance = {"version": DATA_VERSION, "source": str(source), "source_seed": summary["seed"],
        "component_seeds": seeds, "model_initialization_seed": resolved["model_initialization_seed"],
        "completion_sha256": sha256_file(source / "COMPLETED"),
        "summary_sha256": sha256_file(source / "summary.json"),
        "config_sha256": sha256_file(source / "config.json"), "frame_refs": sorted(frame_refs),
        "record_count": len(records), "acquired_count": captured, "probability_floor": PROBABILITY_FLOOR,
        "return_bins": 161, "return_range": [-6., 12.], "source_distill_receipt": copy.deepcopy(receipt),
        "purpose": "new_initialization_from_original_A_demonstrations_not_online_archive_replay",
        "old_fisher_reused": False, "old_anchors_reused": False, "old_optimizer_reused": False,
        "rewards_recomputed": False, "teacher_queries": 0}
    provenance["dataset_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in provenance.items()
                                                                 if k != "source"}).encode()).hexdigest()
    return {"records": records, "acquired_count": captured, "episodes": copy.deepcopy(summary["episodes"]),
            "provenance": provenance}
