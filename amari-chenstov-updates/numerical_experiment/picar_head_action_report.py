"""Offline CPU recovery of saved Fix 5 steering; never train or run episodes.

Only the final replay window survives checkpoint pruning. Publish a separate
immutable scalar report; the notebook never opens Torch checkpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import astuple
import hashlib
import json
import math
from pathlib import Path
import re

from numerical_experiment.picar_head_awac import report
from src.robotics.actions import Calibration


VERSION = "plan19-fix5-turn-bouts-v2"
TURN_DEGREES = 5.
MOVING_MPS = .04


def digest(value):
    return hashlib.sha256(value).hexdigest()


def default_output(root):
    root = Path(root)
    return root.with_name(root.name + "-action-diagnostics-v2")


def control_summary(rows):
    """Count same-sign steering bouts, not repeated commands or measured yaw."""
    bouts, previous, angle_time, moving_time = 0, 0, 0., 0.
    for time, elapsed, controls in sorted(rows):
        if len(controls) != 4 or not all(math.isfinite(x) for x in (time, elapsed, *controls)) or elapsed <= 0:
            raise ValueError("invalid executed-control interval")
        steering, speed = controls[2:]
        moving = abs(speed) > MOVING_MPS
        direction = (1 if steering > 0 else -1) if moving and abs(math.degrees(steering)) > TURN_DEGREES else 0
        bouts += bool(direction and direction != previous)
        previous = direction
        if moving:
            moving_time += elapsed
            angle_time += elapsed * abs(math.degrees(steering))
    return {"turn_bouts": bouts, "moving_seconds": moving_time,
            "mean_abs_steering_deg": angle_time / moving_time if moving_time else None}


def episode_rows(summary, state):
    histories = defaultdict(dict)
    commands = defaultdict(dict)
    for record in state["replay"]:
        arrival = record["arrival"]
        if arrival in commands[record["episode_id"]]:
            raise ValueError("duplicate policy-command arrival")
        commands[record["episode_id"]][arrival] = record["action"].tolist()
        for key in ("history", "next_history"):
            history = record[key]
            times = history["decision_time"][0].tolist()
            elapsed = history["elapsed"][0, :, 0].tolist()
            controls = history["executed_actions"][0].tolist()
            if not len(times) == len(elapsed) == len(controls):
                raise ValueError("executed history lengths differ")
            for at, dt, control in zip(times, elapsed, controls):
                if not math.isfinite(at) or not math.isfinite(dt) or dt < 0:
                    raise ValueError("invalid executed history time")
                if dt == 0:
                    continue  # Initial observation, not an executed interval.
                value = (at, dt, control)
                previous = histories[record["episode_id"]].get(at)
                if previous is not None and previous != value:
                    raise ValueError("conflicting duplicate executed control")
                histories[record["episode_id"]][at] = value
    rows = []
    for episode in summary["episodes"]:
        controls = list(histories[episode["episode_id"]].values())
        expected = len(episode["raw_stage_rewards"])
        if len(controls) > expected:
            raise ValueError("recovered more executed intervals than episode contains")
        observed_complete = expected > 0 and len(controls) == expected
        policy = commands[episode["episode_id"]]
        expected_arrivals = list(range(episode["n"] + 1, episode["n"] + expected))
        complete = observed_complete and sorted(policy) == expected_arrivals
        # The first interval is the documented neutral startup, not a policy
        # command. Require every later command, rather than guess missing ones.
        recovered = len(policy) + int(observed_complete)
        row = {"episode_id": episode["episode_id"], "n": episode["n"],
               "sigma_m": episode["layout"]["curriculum"]["sigma_m"],
               "success": episode["success"], "expected_intervals": expected,
               "recovered_intervals": recovered, "complete_history": complete}
        if complete:
            calibration = Calibration(**episode["simulator_config"]["calibration"])
            sequence = [(0, 1., astuple(calibration.neutral()))]
            for arrival in expected_arrivals:
                logits = [*policy[arrival], 0.]
                if len(logits) != 8 or not all(math.isfinite(x) for x in logits):
                    raise ValueError("policy command requires seven finite reference logits")
                weights = [math.exp(value - max(logits)) for value in logits]
                control = calibration.map_distribution([value / sum(weights) for value in weights])
                sequence.append((arrival - episode["n"], 1., astuple(control)))
            diagnostic = control_summary(sequence)
            diagnostic["moving_commands"] = int(diagnostic.pop("moving_seconds"))
            row.update(diagnostic)
        else:
            row.update({"turn_bouts": None, "mean_abs_steering_deg": None, "moving_commands": None})
        rows.append(row)
    return rows


def source_files(unit):
    pointer = report.shared._json(unit / "CURRENT.json", limit=16_384)
    commit = pointer["commit_id"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{32}", commit):
        raise ValueError("unsafe checkpoint commit identity")
    transaction = unit / "transactions" / commit
    manifest = transaction / "manifest.json"
    if manifest.is_symlink() or digest(manifest.read_bytes()) != pointer["manifest"]["sha256"]:
        raise ValueError("checkpoint manifest checksum mismatch")
    metadata = report.shared._json(manifest, limit=1_000_000)
    state = transaction / "state.pt"
    seal = metadata["recovery_state"]
    if (state.is_symlink() or not state.is_file() or state.stat().st_size != seal["bytes"]
            or seal["bytes"] > 128 * 1024 * 1024):
        raise ValueError("checkpoint size or path mismatch")
    return state, {"unit": unit.name, "commit_id": commit, "state_sha256": seal["sha256"],
                   "summary_sha256": digest((unit / "summary.json").read_bytes())}


def analyze(root, output=None, *, resume=False):
    root = Path(root)
    output = default_output(root) if output is None else Path(output)
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("action analysis must be separate from the scientific run")
    if output.is_symlink() or any((p / "COMPLETED").exists() for p in output.resolve().parents):
        raise ValueError("action analysis destination is immutable")
    snapshot = report.load_report(root)
    selected = report._plot_units(snapshot)
    files = [source_files(root / "units" / unit["name"]) for unit in selected]
    identity = {"version": VERSION, "schema_version": 1, "turn_degrees": TURN_DEGREES,
                "moving_mps": MOVING_MPS, "source_sha256": digest(Path(__file__).read_bytes()),
                "sources": [identity for _, identity in files]}
    destination = output / "summary.json"
    if (output / "COMPLETED").is_file():
        previous = report.shared._json(destination, limit=report.MAX_JSON_BYTES)
        if resume and previous["identity"] == identity:
            return destination
        raise FileExistsError("completed action analysis is immutable; choose a new output")
    import io
    import torch

    units = []
    for unit, (path, _) in zip(selected, files):
        data = path.read_bytes()
        if digest(data) != identity["sources"][len(units)]["state_sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        summary = unit["summary"]
        if state.get("stage") != "done" or state["n"] != summary["n"]:
            raise ValueError("checkpoint is not the sealed completed trajectory")
        units.append({"name": unit["name"], "episodes": episode_rows(summary, state)})
    output.mkdir(parents=True, exist_ok=True)
    from numerical_experiment.picar.storage import _atomic_bytes

    payload = (json.dumps({"identity": identity, "units": units}, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(destination, payload)
    _atomic_bytes(output / "COMPLETED", (digest(payload) + "\n").encode())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(analyze(args.root, args.output, resume=args.resume))


if __name__ == "__main__":
    main()
