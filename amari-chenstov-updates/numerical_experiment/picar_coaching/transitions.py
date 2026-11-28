"""Executed-control telemetry and reward-preserving decision transitions."""

from dataclasses import asdict
import math

import torch

from src.robotics.actions import Calibration


def execution_segments(snapshot, start, end):
    """Read actual command/expiry intervals from the saved physics event log."""
    dt = snapshot["config"]["physics_dt_s"]
    calibration = Calibration(**snapshot["config"]["calibration"])
    control = calibration.neutral()
    at = snapshot["clock_start_ticks"] * dt
    deadline = at
    command_at = at
    segments = []

    def append(lo, hi, value, source):
        lo, hi = max(start, lo), min(end, hi)
        if hi <= lo + 1e-10:
            return
        entry = {"start_time": lo, "end_time": hi, "executed": list(asdict(value).values()), "source": source,
                 "command_at_s": command_at if source == "accepted_command" else None}
        if (segments and segments[-1]["executed"] == entry["executed"]
                and segments[-1]["source"] == source and segments[-1]["command_at_s"] == entry["command_at_s"]
                and abs(segments[-1]["end_time"] - lo) < 1e-8):
            segments[-1]["end_time"] = hi
        else:
            segments.append(entry)

    from src.robotics.protocol import Control
    for event in snapshot["events"]:
        if event["kind"] == "command":
            control = Control(**event["controls"])
            deadline = event["deadline_s"]
            command_at = at
        elif event["kind"] == "stop":
            control = calibration.neutral(control)
            deadline = at
        elif event["kind"] == "advance":
            stop = at + event["physics_steps"] * dt
            expiry = math.ceil((deadline - 1e-10) / dt) * dt
            active_end = min(stop, max(at, expiry))
            append(at, active_end, control, "accepted_command")
            if active_end < stop - 1e-10:
                control = calibration.neutral(control)
                append(active_end, stop, control, "expiry_neutral")
            at = stop
        else:
            raise ValueError("unrecognized physical event")
        if at >= end - 1e-10:
            break
    duration = sum(s["end_time"] - s["start_time"] for s in segments)
    if not math.isclose(duration, end - start, abs_tol=2e-6):
        raise ValueError("physical execution journal does not cover the interval")
    return segments


def decision_returns(rewards, discounts, terminals, policy_mask):
    """Collapse fallback gaps into the preceding actual policy decision.

    No synthetic action is created. A decision's TD target includes all
    discounted rewards until the next policy decision, including a terminal
    reward paid during imposed neutral motion. Acquisitions retain their IDs.
    """
    rewards, discounts = torch.as_tensor(rewards), torch.as_tensor(discounts)
    terminals, policy_mask = torch.as_tensor(terminals), torch.as_tensor(policy_mask)
    if not (rewards.ndim == 1 and rewards.shape == discounts.shape == terminals.shape == policy_mask.shape):
        raise ValueError("decision transition inputs must be aligned vectors")
    if not bool(torch.isfinite(rewards).all() and torch.isfinite(discounts).all()):
        raise ValueError("nonfinite decision transition")
    indices = torch.nonzero(policy_mask, as_tuple=False).flatten().tolist()
    result = []
    for position, first in enumerate(indices):
        last = indices[position + 1] - 1 if position + 1 < len(indices) else len(rewards) - 1
        value, weight = rewards.new_zeros(()), rewards.new_ones(())
        for j in range(first, last + 1):
            value = value + weight * rewards[j]
            weight = weight * discounts[j] * (~terminals[j])
        result.append({"first": first, "last": last, "reward": value, "discount": weight})
    return result
