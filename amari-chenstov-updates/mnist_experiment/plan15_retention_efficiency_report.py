"""Concise static Phase 8.1E presentation; never runs models or repairs inputs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plan15-retention-efficiency-mpl"))

from .fisher_memory_regression.report import _figure, _image, _style, COLORS, LABELS, MAX_NOTEBOOK_BYTES

TAG = "plan15_phase81e"
PREFIX = "plan15-phase81e-"


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def figure(result):
    fig = _figure(11, 4.5)
    axes = fig.subplots(1, 2)
    means, point = result["means"], result["point"]
    N = len(means["replay_ewc"]["novel_mse"]) - 1
    for axis in axes:
        for method in ("fd_replay_ewc", "replay_ewc"):
            axis.plot(range(N + 1), means[method]["novel_mse"], color=COLORS[method],
                      label=LABELS[method], linewidth=1.7)
        q = point["target_mse"]
        axis.axhline(q, color="#555", linestyle="--", linewidth=1, label="Replay mean at 100 images")
        axis.scatter([N], [q], color=COLORS["replay_ewc"], s=32, zorder=4)
        if point["status"] == "valid":
            budget = point["matching_budget"]
            axis.axvline(budget, color=COLORS["fd_replay_ewc"], linestyle=":", linewidth=1,
                         label=f"FD matching budget: {budget:.1f}")
            axis.scatter([budget], [q], color=COLORS["fd_replay_ewc"], s=40, zorder=5)
        axis.set(xlabel="Unique novel training images", ylabel="New-domain test MSE (degrees squared)")
        _style(axis)
    axes[0].set_title("Observed mean learning curves")
    axes[0].legend(fontsize=7)
    start = int(.75 * N)
    axes[1].set(title="Local sample-equivalence match", xlim=(start, N + .5))
    nearby = np.concatenate([np.array(v["novel_mse"])[start:] for v in means.values()])
    padding = max(1., .1 * float(np.ptp(nearby)))
    axes[1].set_ylim(float(nearby.min()) - padding, float(nearby.max()) + padding)
    if point["status"] == "valid" and start <= point["matching_budget"] <= N:
        axes[1].annotate("", xy=(N, point["target_mse"]),
            xytext=(point["matching_budget"], point["target_mse"]),
            arrowprops=dict(arrowstyle="<->", color="#333", linewidth=1))
        axes[1].text(.98, .98, f"{point['saved_novel']:.1f} fewer\nnovel images",
                      transform=axes[1].transAxes, ha="right", va="top", fontsize=10)
    return fig


def build_cells(result, href):
    def cell(name, text):
        return nbformat.v4.new_markdown_cell(text, id=PREFIX + name, metadata={TAG: True})
    intro = ("### Exploratory Sample Equivalence (Phase 8.1E)\n\n"
        "**Question:** how many novel images does the selected FD policy need to match replay/EWC after 100? "
        r"The target is observed mean held-out MSE $q=\overline R^R_{\mathrm{replay}}(100)$, not a fitted endpoint. "
        "Join adjacent FD mean checkpoints with straight segments, then use its last entry below the target "
        "that persists through the observed endpoint. Earlier crossings remain recorded. There is no global "
        "trend fit, smoothing, extrapolation or new training.\n\n"
        "This is an explicitly post hoc, mean-learning-curve comparison on the same 12 paired replicas. "
        "Fractional counts mean interpolation, not fractional training examples or average individual "
        "first-passage times. Changing the novel count also changes background-stream exposure: this is "
        "protocol-level equivalence, not an intervention varying only novel labels. "
        "The original budget-averaged confirmation remains unchanged.")
    fig = _image("phase81e-match", figure(result),
        "The same observed mean curves at full range and a last-quarter-budget zoom. Dashed horizontal: "
        "replay's observed endpoint target; dotted vertical and dot: the in-range FD match, when available. "
        "These are cohort means, not per-replica crossings; connecting segments do not guarantee behavior between observations.")
    fig["id"] = PREFIX + "figure"
    fig["metadata"] = {TAG: True}
    point, boot = result["point"], result["bootstrap"]
    if point["status"] != "valid":
        conclusion = f"**No in-range sample-saving estimate:** {point['status'].replace('_', ' ')}. "
    else:
        ci = boot["ci"]
        interval = (f"Nominal paired-bootstrap 95% interval: **[{ci['saved_novel'][0]:.1f}, "
                    f"{ci['saved_novel'][1]:.1f}] novel images saved**. " if ci else
                    f"An ordinary 95% inversion interval is unavailable: **{boot['invalid']}/{boot['draws']} draws "
                    "have unresolved crossings**. No unsuccessful draw was discarded. ")
        if boot["multiple_crossing_draws"]:
            interval += (f"**Crossing sensitivity:** {100*boot['multiple_crossing_draws']/boot['draws']:.1f}% of draws "
                         "enter the target region more than once; the inverse-risk estimate and its interval remain exploratory. ")
        conclusion = (f"**Preliminary estimate:** FD matches replay's 100-image mean error with approximately "
            f"**{point['matching_budget']:.1f} novel images**: **{point['saved_novel']:.1f} fewer "
            f"({point['saved_percent']:.1f}%)**. " + interval + "\n\n"
            "| Policy | Novel images | New-domain MSE | Learning seconds | Old-domain RMSE |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            f"| FD | {point['matching_budget']:.1f} | {point['target_mse']:.2f} | "
            f"{point['fd_learning_seconds']:.2f} | {point['fd_old_rmse']:.3f} |\n"
            f"| Replay/EWC | 100 | {point['target_mse']:.2f} | {point['replay_learning_seconds']:.2f} | "
            f"{point['replay_old_rmse']:.3f} |\n\n"
            f"**Compute at comparable prediction:** FD still costs **{point['learning_time_ratio']:.2f} times as much** "
            f"({point['same_budget_time_ratio']:.2f} times at equal 100-image budgets). ")
        if ci:
            conclusion += (f"Nominal 95% interval for that matched-budget time ratio: "
                           f"[{ci['learning_time_ratio'][0]:.2f}, {ci['learning_time_ratio'][1]:.2f}]. ")
        conclusion += (f"The respective mean total arrivals are {point['fd_observations']:.1f} and "
                       f"{point['replay_observations']:.1f}, including the prescribed background stream.\n\n"
                       f"**Retention context:** FD-minus-replay old-domain endpoint RMSE is "
                       f"{point['old_rmse_difference']:.3f} degrees")
        if ci:
            conclusion += f" [nominal 95% interval {ci['old_rmse_difference'][0]:.3f}, {ci['old_rmse_difference'][1]:.3f}]"
        conclusion += (". This is not H and does not extend its original retention-equivalence certificate "
                       "to unequal stopping budgets.\n\n")
    conclusion += (f"Bootstrap: {boot['draws']} whole-paired-replica draws, with the target and match recomputed together; "
        f"{boot['multiple_crossing_draws']} draws have multiple downward entries. "
        f"The point mean curve has {len(point['downward_crossings'])} downward "
        + ("entry" if len(point['downward_crossings']) == 1 else "entries") + "; all are retained in the artifact. "
        "The last-entry rule requires persistence only through the observed 100-image endpoint. "
        "This is an exploratory sample-equivalence estimate, not guaranteed future observations saved or "
        "an equal-compute win. It counts online novel images after calibration, not total labels.\n\n"
        + result["assumptions"] + f" [Sealed analysis and bootstrap diagnostics]({href}).")
    return [cell("intro", intro), fig, cell("conclusion", conclusion)]


def publish(result, store, notebook):
    notebook = Path(notebook)
    previous = json.loads(notebook.read_bytes())
    if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
        raise ValueError("cannot update frozen Phase 8 archive")
    cells = [c for c in previous["cells"] if not owned(c)]
    ends = [i for i, c in enumerate(cells) if c.get("id") == "plan15-retention-conclusion"]
    if len(ends) != 1:
        raise ValueError("exactly one completed Phase 8.1 conclusion is required")
    href = os.path.relpath(store.unit_path("analysis") / "result.json", notebook.parent)
    new = build_cells(result, href)
    book = deepcopy(previous)
    offset = ends[0] + 1
    book["cells"] = cells[:offset] + new + cells[offset:]
    book["metadata"][TAG] = dict(protocol=result["protocol"], parent_manifest_hash=result["parent_manifest_hash"],
        analysis_seal=json.loads((store.unit_path("analysis") / "COMPLETED").read_bytes())["seal_hash"], artifact_only=True,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    data = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(data) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    with store.reserve(len(data)):
        store.check()
        fd, tmp = tempfile.mkstemp(prefix=".tmp-phase81e-", dir=notebook.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            store.check()
            os.replace(tmp, notebook)
        finally:
            Path(tmp).unlink(missing_ok=True)
