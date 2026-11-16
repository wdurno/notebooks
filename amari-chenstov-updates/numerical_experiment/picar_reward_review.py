"""Lightweight review of sealed training rewards; never score test images."""

from __future__ import annotations

from bisect import bisect_right
import math
from statistics import mean

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.phase2_report import _note, _table
from numerical_experiment.picar_initialization import report as figures


METHODS = {"replay_ewc": ("Replay/EWC", "#1875ad", "-"),
           "stm_fd": ("FD", "#d16a27", "--")}
GRID = tuple(i / 4 for i in range(81))
INTRO = (
    "#### Training Reward Review (2026-11-16)\n\n"
    "**Training rewards only.** Frozen A/B evaluations saved no Qwen reward scores. "
    "No images were rescored and no training or episodes were rerun for this review. "
    "Early/middle/late are thirds of the 640-interval training budget, assigned by "
    "episode reset exposure; they are not initial/mid/final frozen-model tests. "
    "Clear/blocked means the saved straight-to-target path test at reset, not A/B: "
    "both types can occur during the continuously changing curriculum.\n\n"
    "Curves average episodes within each replica, then average the available replicas "
    "equally. Terminal reward stays in the cumulative curve after an episode ends; "
    "ended episodes are not silently dropped. Counts are shown because some groups "
    "are small. These are descriptive training summaries, not paired causal effects "
    "or confidence intervals."
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def cumulative_curve(elapsed, rewards, grid=GRID):
    """Credit each saved reward at its interval end; hold after termination."""
    if not elapsed or len(elapsed) != len(rewards):
        raise shared.ArtifactError("training reward durations and labels must align")
    times, totals, time, total = [0.], [0.], 0., 0.
    for dt, reward in zip(elapsed, rewards):
        if not _finite(dt) or dt <= 0 or not _finite(reward):
            raise shared.ArtifactError("training reward values must be finite with positive durations")
        time += dt
        total += reward
        times.append(time)
        totals.append(total)
    if any(not _finite(t) or t < 0 for t in grid):
        raise shared.ArtifactError("reward plot times must be finite and nonnegative")
    return [totals[bisect_right(times, t + 1e-10) - 1] for t in grid]


def _replica_mean(rows, key):
    replicas = {}
    for row in rows:
        replicas.setdefault(row["seed"], []).append(row[key])
    if not replicas:
        return None
    values = [list(map(mean, zip(*curves))) for curves in replicas.values()]
    return list(map(mean, zip(*values)))


def training_rows(snapshot):
    from numerical_experiment.picar_solver_repair import report as solver

    rows = []
    for unit in solver._pilot_units(snapshot):
        summary = unit["summary"]
        seed, _, _, condition = solver._pilot_identity(summary)
        budget = summary.get("settings", {}).get("transitions", 640)
        if condition not in METHODS or not _finite(budget) or budget <= 0:
            raise shared.ArtifactError("reward review needs the recorded condition and training budget")
        for episode in summary.get("episodes", []):
            if not episode.get("label_complete") or episode.get("truncated") or episode.get("evaluation"):
                continue
            intervals = episode.get("intervals", [])
            rewards = episode.get("stage_rewards", [])
            scores = episode.get("visual_scores", [])
            bonuses = episode.get("terminal_bonuses", [])
            if not intervals or not (len(intervals) == len(rewards) == len(scores) == len(bonuses)):
                raise shared.ArtifactError("completed training reward components are incomplete")
            exposure = episode.get("reset_exposure")
            blocked = episode.get("layout", {}).get("curriculum", {}).get("direct_path_blocked")
            if not _finite(exposure) or not 0 <= exposure <= budget or type(blocked) is not bool:
                raise shared.ArtifactError("training reset exposure and saved path class are required")
            elapsed = [r["elapsed_s"] for r in intervals]
            curve = cumulative_curve(elapsed, rewards)
            for dt, score, bonus, actual in zip(elapsed, scores, bonuses, rewards):
                if not isinstance(score, dict) or not all(_finite(score.get(k)) for k in
                        ("score", "proximity", "centering_bonus")) or not _finite(bonus):
                    raise shared.ArtifactError("finite saved visual reward components are required")
                expected = .01 * dt * score["score"] - .04 * dt + bonus
                if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
                    raise shared.ArtifactError("saved reward does not match the dense-visual-v1 scale")
            rows.append({"seed": seed, "condition": condition, "blocked": blocked,
                "third": min(2, int(3 * exposure / budget)), "curve": curve,
                "exposure": exposure, "budget": budget, "elapsed": elapsed,
                "scores": scores, "rewards": rewards, "bonuses": bonuses,
                "end_reason": episode.get("end_reason")})
    return rows


def _reward_figure(rows):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.2, 5.9), layout="constrained")
    axes = figure.subplots(2, 3, sharex=True, sharey=True)
    handles = {}
    for i, blocked in enumerate((False, True)):
        for third in range(3):
            axis, counts = axes[i, third], []
            for condition, (name, color, style) in METHODS.items():
                group = [r for r in rows if r["condition"] == condition
                         and r["blocked"] == blocked and r["third"] == third]
                values = _replica_mean(group, "curve")
                counts.append(f"{name}: {len(group)} ep / {len({r['seed'] for r in group})} reps")
                if values is not None:
                    handles[name], = axis.plot(GRID, values, color=color, linestyle=style,
                                               linewidth=1.8, label=name)
            axis.set_title(("Clear path" if not blocked else "Blocked path") + " | "
                           + ("Early", "Middle", "Late")[third], fontsize=10, pad=27)
            axis.text(.5, 1.015, " | ".join(counts), transform=axis.transAxes,
                      fontsize=7, va="bottom", ha="center")
            if not axis.lines:
                axis.text(.5, .5, "No episodes in this group", transform=axis.transAxes,
                          ha="center", va="center", fontsize=9, color="#666666")
            axis.grid(alpha=.2)
            axis.set_xlim(0, 20)
            axis.tick_params(labelleft=True, labelbottom=True)
            if i == 1:
                axis.set_xlabel("Episode time (seconds)")
            if third == 0:
                axis.set_ylabel("Mean cumulative reward\n(undiscounted)")
    figure.legend(list(handles.values()), list(handles), loc="outside lower center", ncols=2,
                  frameon=False, fontsize=9)
    return figures._figure(figure, "Training cumulative rewards by recorded path class and exposure third. "
                           "Reward is credited at interval end and retained after termination. "
                           "Missing groups have no curve; counts show episode and replica support.")


def _binned_components(rows, condition, bins=8):
    buckets = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        for j, (score, dt, reward, bonus) in enumerate(zip(
                row["scores"], row["elapsed"], row["rewards"], row["bonuses"])):
            position = (row["exposure"] + j) / row["budget"]
            index = min(bins - 1, int(bins * position))
            buckets.setdefault((row["seed"], index), []).append((score, dt, reward - bonus))
    result = []
    for index in range(bins):
        replicas = [samples for (seed, bucket), samples in buckets.items() if bucket == index]
        visible = [[s for s, _, _ in samples if s.get("visible")] for samples in replicas]
        proximity = [mean(s["proximity"] for s in samples) for samples in visible if samples]
        centering = [mean(s["centering_bonus"] for s in samples) for samples in visible if samples]
        rates = [sum(reward for _, _, reward in samples) / sum(dt for _, dt, _ in samples)
                 for samples in replicas]
        result.append({"progress": (index + .5) / bins,
            "proximity": mean(proximity) if proximity else None,
            "centering": mean(centering) if centering else None,
            "dense_rate": mean(rates) if rates else None})
    return result


def _component_figure(rows):
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(11.2, 3.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for condition, (name, color, style) in METHODS.items():
        values = _binned_components(rows, condition)
        x = [v["progress"] for v in values]
        axes[0].plot(x, [v["proximity"] for v in values], color=color, linestyle=style,
                     label=name + " proximity")
        axes[0].plot(x, [v["centering"] for v in values], color=color,
                     linestyle=":" if condition == "replay_ewc" else "-.",
                     label=name + " centering bonus")
        axes[1].plot(x, [v["dense_rate"] for v in values], color=color, linestyle=style, label=name)
    axes[0].set(title="Qwen scores when the ball is visible", ylabel="Mean score component")
    axes[0].set_ylim(bottom=0)
    axes[1].set(title="Dense reward before terminal bonuses", ylabel="Mean reward per second")
    axes[1].axhline(0, color="#777777", linewidth=.8, linestyle=":")
    for axis in axes:
        axis.set(xlabel="Incoming training intervals / 640", xlim=(0, 1))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
    figures._finish(axes)
    return figures._figure(figure, "Eight fixed exposure bins, equally weighted available replicas. "
                           "Score components condition on ball visibility; dense reward rate includes all "
                           "observations and divides summed nonterminal reward by summed physical duration.")


def summary_rows(rows):
    result = []
    for condition, (name, _, _) in METHODS.items():
        group = [r for r in rows if r["condition"] == condition]
        if not group:
            continue
        ratios = []
        for seed in sorted({r["seed"] for r in group}):
            visible = [s for r in group if r["seed"] == seed for s in r["scores"] if s.get("visible")]
            if visible:
                ratios.append(mean(s["proximity"] == 1 for s in visible))
        result.append({"Method": name, "Goals / episodes": f"{sum(r['end_reason']=='success' for r in group)}/{len(group)}",
            "Collisions / episodes": f"{sum(r['end_reason']=='collision' for r in group)}/{len(group)}",
            "Visible-ball proximity = 1/10": f"{mean(ratios):.1%}" if ratios else "unavailable"})
    return result


def behavior_note(snapshot):
    from numerical_experiment.picar_solver_repair import report as solver

    decisions = []
    for unit in solver._pilot_units(snapshot):
        for panel in unit["summary"]["evaluations"]:
            decisions.extend(r for e in panel["rows"] for r in e.get("intervals", [])
                             if isinstance(r.get("probabilities"), list))
    if not decisions:
        return ""
    forward = sum(max(range(8), key=lambda i: r["probabilities"][i]) == 2 for r in decisions)
    large = sum(abs(r["controls"]["steering_rad"]) > math.radians(5) for r in decisions)
    return (f"In the saved greedy evaluation decisions, forward had the largest action probability "
            f"{forward/len(decisions):.1%} of the time; {large}/{len(decisions)} commands exceeded 5 degrees "
            "of absolute steering. Reused initial panels are included, not treated as independent replicas. "
            "This is consistent with an inherited straight-driving policy, not proof that steering never "
            "occurred during exploratory training. A useful next hypothesis is insufficient coherent, "
            "rewarded detours: independent action noise need not produce successful multi-step turns. "
            "Probability-weighted left/right actions can also cancel steering; that is a mechanism to test, "
            "not an established cause here.")


def outputs(snapshot):
    rows = training_rows(snapshot)
    if not rows:
        return [_note("Complete, scored training episodes are unavailable; test rewards were not recorded.")]
    result = [_reward_figure(rows), _component_figure(rows), _table(summary_rows(rows)),
        _note("Reward per saved interval is r = 0.01 * score * elapsed_seconds - 0.04 * elapsed_seconds + terminal_bonus. "
              "Bonuses are +10 for success, -5 for collision and -1 for timeout. A far, centered ball scoring 1.5 "
              "therefore earns -0.025 per second before the terminal bonus. The learner uses full completed-episode "
              "discounted returns, not one-step Bellman/TD targets. Curves above show undiscounted earned reward, not return-to-go. "
              "Flat proximity scores support a weak shaping hypothesis, but do not by themselves prove distance miscalibration. "
              "Larger collision penalties are not an evident first fix: collisions already dominate these totals. "
              "A higher cumulative training curve can reflect later collisions or timeouts rather than reaching the ball; "
              "it is not evidence of hard-task mastery.")]
    behavior = behavior_note(snapshot)
    if behavior:
        result.append(_note(behavior))
    return result


def display_review(snapshot):
    from IPython.display import display
    for output in outputs(snapshot):
        display(output)
