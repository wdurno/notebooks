"""Measured-cost planning; no model loading or statistical efficacy estimates."""

from __future__ import annotations

import math


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def forecast(probe: dict, *, pairs=6, transitions=2400, control_period_s=.5,
             elapsed_hours=0., bootstrap_transitions=600, sleep_every=400,
             sleep_steps=2, checkpoint_commits_per_transition=3.05,
             capture_width=448, capture_height=336) -> dict:
    """Sequential GPU work versus simultaneous real-time physics, with no overlap
    assumed among inference/update/reward/Fisher kernels on the one backbone.

    Physics progresses independently; its duration is a lower bound, not an
    additional GPU charge. Full learning cadence is unresolved until Phase 3,
    so the baseline uses one accepted optimizer event per new observation.
    """
    if type(pairs) is not int or pairs < 1 or type(transitions) is not int or transitions < 1:
        raise ValueError("positive integer paired-replica and transition budgets required")
    times = probe["timings"]
    actor = _positive(times["inference_seconds"], "inference")
    control = _positive(times["fit_control_seconds"], "control update")
    fd = _positive(times["fit_fd_seconds"], "FD update")
    fisher = _positive(times["fisher_seconds"], "Fisher")
    evaluator = _positive(times["evaluator_seconds"], "evaluator")
    teacher = _positive(times["teacher_seconds"], "teacher")
    period = _positive(control_period_s, "control period")
    checkpoint = probe.get("checkpoint", {})
    commit = _positive(checkpoint.get("transaction_seconds", .01), "checkpoint")
    io_per_new = commit * checkpoint_commits_per_transition
    # Each potential endpoint is scored once, plus a small episode-start term.
    reward = evaluator * (1 + 1 / 40)
    sleeps = max(0, (transitions - 1) // sleep_every) if sleep_every else 0
    online = {}
    for name, update in (("control", control), ("fd", fd)):
        gpu = transitions * (actor + update + fisher + reward)
        io = transitions * io_per_new
        sleep = sleeps * sleep_steps * update
        physical = transitions * period
        # Sleep is separately stationary; replay-only sleep uses update timing
        # as a lower-bound proxy until the admitted larger batch is measured.
        online[name] = max(gpu + io, physical) + sleep
    # Seven checkpoints: initialization, five intermediate, and one final.
    # 4 A+4 B at initialization/intermediate; 12 A+12 B at the endpoint.
    evaluation_episodes_per_arm = 6 * 8 + 24
    evaluation_action_count = math.ceil(20 / period)
    evaluation_episode_seconds = max(20., actor * evaluation_action_count)
    evaluation_per_arm = evaluation_episodes_per_arm * evaluation_episode_seconds
    fresh = pairs * (online["control"] + online["fd"] + 2 * evaluation_per_arm)
    # Bootstrap is collected once and shared, not charged once per arm.
    bootstrap_collection = bootstrap_transitions * (teacher + reward)
    bootstrap_fit = bootstrap_transitions * control
    bootstrap_fisher = max(0, bootstrap_transitions - 128) * fisher
    readiness_panels = 40 * evaluation_episode_seconds
    bootstrap = bootstrap_collection + bootstrap_fit + bootstrap_fisher + readiness_panels
    # Eight initial development trajectories: two seeds and two recipes/arm;
    # allow up to four extra trajectories for bounded repairs.
    dev_length = 600
    mean_step = (online["control"] + online["fd"]) / (2 * transitions)
    development = [8 * dev_length * mean_step + 8 * 16 * evaluation_episode_seconds,
                   12 * dev_length * mean_step + 12 * 24 * evaluation_episode_seconds]
    bands = {
        "phase2": [bootstrap * .85 / 3600, bootstrap * 1.35 / 3600],
        "phase3": [development[0] / 3600, development[1] / 3600],
        "phase4": [fresh * .9 / 3600, fresh * 1.25 / 3600],
        "phase5": [.25, .75],
    }
    total = [elapsed_hours + sum(b[i] for b in bands.values()) for i in (0, 1)]
    state_bytes = checkpoint.get("learner_state_bytes", 0)
    new_total = pairs * 2 * transitions
    # Separate raw-retention estimates from the much larger all-state archive.
    # Retain the camera capture, not only the learner's downsampled tensor.
    frame_bytes = capture_width * capture_height * 3 + 128
    raw_bytes = (new_total + bootstrap_transitions) * (frame_bytes + 6000)
    growing_record_bytes = transitions / 2 * 7000
    full_checkpoint_bytes = new_total * checkpoint_commits_per_transition * (state_bytes + 1_000_000 + growing_record_bytes)
    return {"schema_version": 1, "phase": 1, "smoke_only": True, "kind": "phase1_forecast",
        "selected_recipe": probe["recipe"], "pairs": pairs, "transitions_per_arm": transitions,
        "control_period_s": period, "phase_hours": bands, "total_hours_range": total,
        "elapsed_hours": elapsed_hours, "budgetfit": total[1] <= 20,
        "online_seconds_per_arm": online, "heldout_seconds_per_arm": evaluation_per_arm,
        "operational_seconds_per_new": {"control": online["control"] / transitions,
                                        "fd": online["fd"] / transitions},
        "storage": {"raw_and_record_bytes": raw_bytes, "retain_all_checkpoint_bytes": full_checkpoint_bytes,
                    "learner_state_bytes": state_bytes, "capture_width": capture_width,
                    "capture_height": capture_height, "raw_frame_bytes": frame_bytes},
        "qualified_assumptions": [
            "One optimizer event per newly labeled transition; each FIFO eviction contributes one sampled Fisher.",
            "Replay and FD use the same admitted batch/context; Fisher timing is per retired transition.",
            "GPU kernels serialize; physics overlaps but cannot lower the one-GPU workload sum.",
            "Sleep uses two admitted-size update events every 400 transitions; a larger sleep batch needs a new timing.",
            "Evaluation is 72 real-time episodes per arm, with common A/B panels and seven checkpoints.",
            "Shared bootstrap: 600 transitions, one data pass each for distillation and joint fitting; rescue work adds cost.",
            "Development assumes 8--12 trajectories of 600 new transitions, with small separate A/B panels.",
            "Timing bands are engineering forecast ranges, not statistical confidence intervals.",
            "Retaining every full transaction state requires archive/compaction engineering beyond raw frame storage."],
        "blockers": []}
