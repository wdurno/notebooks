"""Render three concise Phase 7E cells from sealed analysis artifacts only."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import tempfile

import nbformat

from .fisher_memory.artifacts import ArtifactError, iter_units
from .fisher_memory.config import content_hash
from .fisher_memory.regularization_report import TAG as PHASE7_TAG, _read
from .fisher_memory.report import MAX_NOTEBOOK_BYTES, _plot_cell, reserved_publisher
from .plan15_efficiency import ANALYSIS, CONFIG, PLAN, PROTOCOL, REVIEW, dependency_snapshot

TAG = "plan15_phase7e"
LABELS = dict(replay_ewc="Replay/EWC", finite_difference="FD KL")
COLORS = dict(replay_ewc="#227c76", finite_difference="#b85747")


def load(root):
    root = Path(root).resolve()
    units = {unit["unit_id"]: unit for unit in iter_units(root)}
    if any(name not in units or units[name]["status"] != "completed" for name in (PLAN, ANALYSIS)):
        raise ArtifactError("Phase 7E notebook requires the sealed plan and completed bootstrap")
    identity = _read(root / "identity.json")["identity"]
    plan = _read(units[PLAN]["path"] / "plan.json")
    result = _read(units[ANALYSIS]["path"] / "result.json")
    draws = _read(units[ANALYSIS]["path"] / "draws.json")
    if (identity.get("protocol") != PROTOCOL or plan["config"] != CONFIG
            or identity.get("config_hash") != content_hash(CONFIG)
            or result.get("schema_version") != 1 or result.get("protocol") != PROTOCOL
            or result.get("status") != "completed" or result.get("config_hash") != content_hash(CONFIG)
            or result.get("dependencies_hash") != content_hash(plan["dependencies"])
            or identity.get("dependencies_hash") != result["dependencies_hash"]
            or result["established"] != plan["inputs"]["established"]
            or len(draws) != 2000 or result["bootstrap"]["draws"] != 2000):
        raise ArtifactError("incompatible Phase 7E schema, configuration, or dependency lineage")
    if dependency_snapshot(plan["inputs"]["roots"]) != plan["dependencies"]:
        raise ArtifactError("Phase 7E scientific dependencies no longer match their completion seals")
    review = None
    if REVIEW in units:
        if units[REVIEW]["status"] != "completed":
            raise ArtifactError("Phase 7E visual review is incomplete")
        review = _read(units[REVIEW]["path"] / "review.json")
        if review.get("schema_version") != 1 or review.get("assessment") not in ("approximate", "poor"):
            raise ArtifactError("invalid Phase 7E visual fit review")
    return plan, result, review


def conclusion(result):
    established = result["established"]
    primary = established["primary"]
    arms = {arm["condition"]: arm for arm in established["arm_summaries"] if arm["stream"] == "mixed"}
    fd = arms["finite_difference"]
    frozen = established["frozen_gain"][fd["treatment_id"]]["mixed"]
    retention = established["base_deterioration"][fd["treatment_id"]]["mixed"]
    ratio = fd["online_seconds"] / arms["replay_ewc"]["online_seconds"]
    return nbformat.v4.new_markdown_cell(
        "### Conclusion: Efficiency\n\n"
        "**Established budget-averaged advantage.** Under this mixed-stream protocol and 32-image replay budget, "
        f"FD KL improves rotated-domain predictive NLL over calibrated replay/EWC by **{primary['mean_gain']:.4f} nats** "
        f"(original paired 95% interval **[{primary['ci'][0]:.4f}, {primary['ci'][1]:.4f}]**); "
        f"{primary['favorable_replicas']}/{primary['n_pairs']} fresh replicas favor FD. "
        "This establishes better budget-averaged prediction over the first 100 new rotated observations "
        "in the tested regime, not a percentage of samples saved.\n\n"
        r"For an independent uniform budget $K\in\{1,\ldots,100\}$, "
        r"$\mathbb E_K[R_m^R(K)]=A_m^R(100)=\frac1{100}\sum_{n=1}^{100}R_m^R(n)$. "
        "In words: stop at a uniformly chosen label budget and FD has lower expected test NLL. "
        "This identity needs no monotonic learning curve or i.i.d. training stream.\n\n"
        f"FD also beats freezing by {frozen['mean_gain']:.4f} nats "
        f"(original nominal 95% interval [{frozen['ci'][0]:.4f}, {frozen['ci'][1]:.4f}]). "
        f"It still worsens unrotated NLL from initialization by {retention['mean_gain']:.4f} "
        f"[{retention['ci'][0]:.4f}, {retention['ci'][1]:.4f}] and costs **{ratio:.2f} times** online compute. "
        "Both selected ridge multipliers are at the strongest tested grid boundary, not bracketed optima. "
        "Equal data are not equal compute; these results do not identify STM-to-LTM transfer.")


def diagnostic(result):
    from matplotlib import pyplot as plt

    point = result["point"]
    figure, axis = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    for method in LABELS:
        axis.plot(range(101), result["means"][method], color=COLORS[method], alpha=.65,
                  linewidth=1.5, label=f"{LABELS[method]}: observed mean")
        axis.plot(range(1, 101), point["fits"][method]["fitted"], color=COLORS[method],
                  linestyle="--", linewidth=2.2, label=f"{LABELS[method]}: linear fit")
    axis.scatter([0], [point["initial"]], color="#555555", marker="D", s=28,
                 label="Shared initialization (not fitted)", zorder=5)
    axis.axhline(point["target"], color="#666666", linestyle=":", linewidth=1.4,
                 label=f"Target: fitted replay at 100 ({point['target']:.3f})")
    if point["inversion_valid"] and point["in_range"]:
        axis.plot([point["crossing"], point["crossing"]], [0, point["target"]], color="#69548b",
                  linestyle=":", linewidth=1.1)
        axis.scatter([point["crossing"]], [point["target"]], color="#69548b", s=42, zorder=6,
                     label=f"FD model crossing ({point['crossing']:.1f} images)")
    all_values = [value for curve in result["means"].values() for value in curve]
    all_values += [value for fit in point["fits"].values() for value in fit["fitted"]]
    low, high = min(all_values), max(all_values)
    axis.set_ylim(max(0, low - .07 * (high - low)), high + .08 * (high - low))
    axis.set(xlim=(-1, 101), xlabel="Actual new rotated observations", ylabel="Rotated-test NLL (nats)",
             title="Mixed training: Rotated test NLL")
    axis.grid(alpha=.16)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    return _plot_cell(figure, "Phase 7E fit diagnostic: twelve paired replicas; OLS uses events 1-100 only. "
                      "Test images are fixed held-out panels, never replay or training data. "
                      "Several new images in one batch can share the same post-batch prediction.")


def preliminary(root, notebook, result, review):
    point, bootstrap = result["point"], result["bootstrap"]
    relative = os.path.relpath(Path(root) / "units" / ANALYSIS / "result.json", Path(notebook).parent)
    paragraphs = ["**Exploratory linear-model sample equivalence (post hoc).** "
                  "The target is replay's fitted NLL at 100 new images, not a target selected by searching. "]
    if "target_already_met_without_adaptation" in point["issues"]:
        paragraphs.append("The frozen initialization already meets this target, so no new-learning "
                          "samples-saved claim is supported.")
    elif not point["inversion_valid"]:
        paragraphs.append("The point fit fails inversion checks; no samples-saved interpretation is supported. "
                          f"Recorded issues: {', '.join(point['issues'])}.")
    elif not point["in_range"]:
        paragraphs.append("The fitted FD crossing lies outside 1-100 images. It is range-limited, "
                          "not an observed saving; no extrapolation or clipping is used.")
    elif review is None:
        paragraphs.append("The numerical analysis is sealed; visual fit review is pending. "
                          "A samples-saved headline is withheld until that check.")
    elif review["assessment"] == "poor":
        paragraphs.append("The linear fit is visibly inadequate, so a persuasive samples-saved claim is withheld. "
                          + review["note"])
    else:
        paragraphs.append(f"FD's fitted mean reaches that target at approximately **{point['crossing']:.1f} "
                          f"new rotated images**, versus replay's 100: a model-implied difference of "
                          f"**{point['saved']:.1f} images ({point['saved_percent']:.1f}%)**. "
                          + review["note"])
    if bootstrap["ci"] is None:
        initial_failures = bootstrap["issues"].get("target_already_met_without_adaptation", 0)
        reason = (f"{initial_failures} draws already meet the fitted replay target at initialization. "
                  if initial_failures else "")
        paragraphs.append(f"An ordinary 95% inversion interval is **unavailable**: "
                          f"{bootstrap['invalid']}/{bootstrap['draws']} whole-paired-replica draws fail "
                          "the slope/loss/initial-reference checks. " + reason + "Failed draws were not discarded.")
    else:
        ci = bootstrap["ci"]
        paragraphs.append(f"The nominal paired-bootstrap 95% interval is "
                          f"[{ci['crossing'][0]:.1f}, {ci['crossing'][1]:.1f}] FD images; "
                          f"the model-implied saving interval is [{ci['saved'][0]:.1f}, {ci['saved'][1]:.1f}] images. "
                          + ("The interval is range-limited, not an in-range savings claim."
                             if not bootstrap["interval_in_range"] else ""))
    paragraphs.append(f"Bootstrap checks: {bootstrap['invalid']}/{bootstrap['draws']} inversion-invalid; "
                      f"{bootstrap['outside_range']}/{bootstrap['draws']} finite crossings outside 1-100 "
                      "(including draws failing other guards).")
    paragraphs.append("This is the inverse of a fitted mean curve, not mean first-passage time, "
                      "a guarantee of future samples saved, or proof of monotonic risk. "
                      "Uncertainty assumes independent paired replicas, adequate log-loss moments and the "
                      "fixed holdout population; it omits fit misspecification and post hoc selection. "
                      "Counts exclude shared pretraining and unrotated background data. "
                      f"[Sealed fits, original inference and bootstrap checks]({relative}).")
    return nbformat.v4.new_markdown_cell("\n\n".join(paragraphs))


def build_notebook(notebook, root):
    plan, result, review = load(root)
    book = nbformat.read(notebook, as_version=4)
    book.cells = [cell for cell in book.cells if TAG not in cell.metadata.get("tags", [])]
    positions = [i for i, cell in enumerate(book.cells) if PHASE7_TAG in cell.metadata.get("tags", [])]
    if not positions:
        raise ArtifactError("Phase 7E must follow an existing Phase 7 notebook section")
    cells = [conclusion(result), diagnostic(result), preliminary(root, notebook, result, review)]
    for cell in cells:
        cell.metadata["tags"] = [TAG]
        cell.metadata["phase7_roots"] = plan["inputs"]["roots"]
        cell.metadata["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    book.cells[positions[-1] + 1:positions[-1] + 1] = cells
    nbformat.validate(book)
    if len(nbformat.writes(book).encode()) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("Phase 7E notebook exceeds bounded publication size")
    return book


def publish(store, notebook):
    content = nbformat.writes(build_notebook(notebook, store.root)).encode()
    reserved_publisher(store)(store.root / "reports/fisher_memory_results.ipynb", content)
    with store.reserve(len(content)):
        descriptor, temporary = tempfile.mkstemp(prefix=".plan15-report-efficiency-", dir=Path(notebook).parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            store.check()
            os.replace(temporary, notebook)
            directory = os.open(Path(notebook).parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
