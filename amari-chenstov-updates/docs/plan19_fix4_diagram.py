"""Render the draft Fix 4 illustration, not an executable experiment schedule."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


BOUNDARIES = (0, 128, 768, 1024, 1664, 2048)
LABELS = ("A hold", "Blocking ramp", "B initial hold", "Texture + steering", "Final B hold")
DIAGRAM = Path(__file__).parent / "figures" / "plan19_phase3_fix4_transition.png"


def shift(observations, start, duration):
    r = min(1., max(0., (observations - start) / duration))
    return r ** 3 * (10 + r * (-15 + 6 * r))


def design_values(observations):
    blocking = shift(observations, 128, 640)
    appearance = shift(observations, 1024, 640)
    return math.exp(blocking * math.log(.08)), 1 - .35 * appearance, appearance


def render(output=DIAGRAM):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.5, 5.6), layout="constrained")
    axes = figure.subplots(3, 1, sharex=True)
    x = sorted(set(range(0, 2049, 4)) | set(BOUNDARIES))
    values = [design_values(n) for n in x]
    shades = ("#f1f4f4", "#e7f2f1", "#f1f4f4", "#f8edf1", "#f1f4f4")
    for index, (axis, color, label) in enumerate(zip(
            axes, ("#147d92", "#a84470", "#657244"),
            ("Barrier spread (m)", "Steering gain", "Floor appearance"))):
        for left, right, shade in zip(BOUNDARIES, BOUNDARIES[1:], shades):
            axis.axvspan(left, right, color=shade, linewidth=0)
        for boundary in BOUNDARIES[1:-1]:
            axis.axvline(boundary, color="#b9c3c3", linewidth=.7)
        axis.plot(x, [row[index] for row in values], color=color, linewidth=2.4)
        axis.set(ylabel=label, xlim=(0, 2048))
        axis.grid(axis="y", color="#ccd2d2", alpha=.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].set(ylim=(.02, 1.12), yticks=(.08, 1.), yticklabels=("0.08: B", "1.00: A"))
    axes[1].set(ylim=(.60, 1.05), yticks=(.65, 1.), yticklabels=("0.65: B", "1.00: A"))
    axes[2].set(ylim=(-.10, 1.10), yticks=(0., 1.), yticklabels=("A", "B"))
    for left, right, label in zip(BOUNDARIES, BOUNDARIES[1:], LABELS):
        axes[0].text((left + right) / 2, 1.035, label, fontsize=8.5, ha="center",
                     transform=axes[0].get_xaxis_transform())
    axes[2].set(xticks=BOUNDARIES, xlabel="New training intervals (provisional stage lengths)")
    figure.suptitle("One persistent learner: A -> B initial -> B final", fontsize=15)
    figure.supxlabel("Draft only. Stationary A initialization precedes interval 0; learning continues in every stage.",
                    fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=125)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIAGRAM)
    print(render(parser.parse_args().output))
