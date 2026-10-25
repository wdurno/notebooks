"""Concise Phase 8 notebook cells from a small supplied snapshot, never checkpoints.

The publisher preserves each historical cell's JSON object, including list-form
sources and outputs. Only metadata-tagged Phase 8 cells or ``plan15-phase8`` IDs
are replaced. Rendering and inference happen outside the notebook; its new cells
are static Markdown and PNG attachments, with no executable experiment code.
"""

from __future__ import annotations

import base64
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np

from .analysis import FD, PROTOCOL, REPLAY, flatten_record

TAG = "plan15_phase8"
PREFIX = "plan15-phase8"
MAX_NOTEBOOK_BYTES = 24_000_000
LABELS = {FD: "FD KL + replay/EWC", REPLAY: "Replay/EWC", "ewc_only": "EWC only"}
COLORS = {FD: "#ad503d", REPLAY: "#168477", "ewc_only": "#74668f"}


def _check(callback):
    if callback is not None and callback():
        raise InterruptedError("Phase 8 report interrupted")


def _markdown(name, text):
    return nbformat.v4.new_markdown_cell(text, id=f"{PREFIX}-{name}", metadata={TAG: True})


def _figure(width=9., height=4.):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _image(name, figure, caption):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    cell = _markdown(name, f"![{caption}](attachment:{name}.png)\n\n{caption}")
    cell["attachments"] = {f"{name}.png": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}}
    figure.clear()
    return cell


def _style(axis):
    axis.grid(alpha=.15)
    axis.spines[["top", "right"]].set_visible(False)


def _display(value, digits=3):
    if value is None:
        return "pending"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "/").replace("\n", " ")[:240]


def _interval_text(effect):
    return f"{effect['mean']:.3g} [{effect['ci'][0]:.3g}, {effect['ci'][1]:.3g}]"


def _brief(value, limit=160):
    """Short prose for structured status records, never a Python dict dump."""
    if isinstance(value, dict):
        value = "; ".join(f"{key.replace('_', ' ')}: {_brief(item, 60)}" for key, item in list(value.items())[:3])
    elif isinstance(value, (list, tuple)):
        value = "; ".join(_brief(item, 60) for item in value[:3])
    return _display(value)[:limit]


def _acceptance(snapshot):
    """Render compact test/smoke evidence, including acceptance.py's sealed result."""
    acceptance = snapshot.get("acceptance")
    if acceptance is None:
        return None
    tests = acceptance.get("tests", [])
    if isinstance(tests, dict):
        tests = [dict(name=name, result=result) for name, result in tests.items()]
    checks = list(tests)
    for smoke in acceptance.get("smokes", []):
        checks.append(dict(name=f"{smoke['device'].upper()} recovery smoke",
                           result=f"Complete; {smoke['units']} units, {smoke['novel_count']} novel observations",
                           seconds=smoke["runtime_seconds"]))
    rows = []
    for item in checks[:6]:
        result = item.get("result", item.get("status", "pending"))
        if "tests" in item:
            result = (f"{item['tests']} tests; {item.get('failures', 0)} failures, "
                      f"{item.get('errors', 0)} errors, {item.get('skipped', 0)} skipped")
        seconds = item.get("seconds", item.get("elapsed_seconds", item.get("process_seconds")))
        timing = "" if seconds is None else f" ({_display(seconds)} s)"
        name = item.get("name", Path(item["path"]).stem if item.get("path") else "Check")
        rows.append(f"| {_brief(name, 80)} | {_brief(result)}{timing} |")
    if not rows:
        rows = [f"| Acceptance suite | {_brief(acceptance.get('status', 'pending'))} |"]
    text = "### Implementation Acceptance\n\n| Check | Result |\n| --- | --- |\n" + "\n".join(rows)
    if acceptance.get("summary"):
        text += "\n\n" + _brief(acceptance["summary"], 240)
    if "total_tests" in acceptance:
        text += f"\n\n**{acceptance['total_tests']} tests recorded.**"
    if "charged_evidence_seconds" in acceptance:
        text += f" Charged acceptance compute: {_display(acceptance['charged_evidence_seconds'])} seconds."
    if acceptance.get("production_started") is False:
        text += " Production has not started."
    if len(checks) > 6:
        text += f" Additional checks: {len(checks) - 6}; detailed results remain in acceptance artifacts."
    text += "\n\nSoftware and recovery checks only. Smoke outcomes do not support scientific efficacy or sample-efficiency claims."
    return _markdown("acceptance", text)


def _changes(changes):
    paragraphs = []
    for change in changes[-3:]:
        if not isinstance(change, dict):
            paragraphs.append(_brief(change, 240))
            continue
        heading = ": ".join(_brief(change[key], 100) for key in ("stage", "change") if change.get(key))
        text = f"**{heading or 'Correction'}.**"
        if change.get("reason"):
            text += " " + _brief(change["reason"]) + "."
        before, after = change.get("before"), change.get("after")
        if isinstance(before, dict) and isinstance(after, dict):
            modified = [key for key in after if before.get(key) != after[key]]
            if modified:
                text += " " + "; ".join(f"{key.replace('_', ' ')} {_brief(before.get(key), 30)} -> "
                                         f"{_brief(after[key], 30)}" for key in modified[:3]) + "."
        if change.get("validity"):
            text += " " + _brief(change["validity"]) + "."
        if change.get("validation"):
            text += " Validation: " + _brief(change["validation"]) + "."
        paragraphs.append(text)
    return _markdown("changes", "**Significant corrections**\n\n" + "\n\n".join(paragraphs)
                     + "\n\nFull provenance remains in the study's decision artifacts.")


def _records(snapshot):
    confirmation = snapshot.get("confirmation") or {}
    return [flatten_record(row) for row in confirmation.get("rows", snapshot.get("records", []))
            if flatten_record(row).get("partition", "confirmation") == "confirmation"]


def _validated_analysis(snapshot):
    analysis = snapshot.get("analysis")
    if not analysis:
        return None
    if (analysis.get("schema_version") != 1 or analysis.get("protocol") != PROTOCOL
            or analysis.get("status") != "completed"):
        raise ValueError("report requires completed compatible precomputed analysis")
    cohort = analysis["cohort"]
    expected = {(recipe["recipe_key"], replica) for recipe in cohort["recipes"]
                for replica in cohort["replicas"]}
    if len(cohort["replicas"]) < 2 or not expected:
        raise ValueError("analysis has no complete paired cohort")
    confirmation = snapshot.get("confirmation")
    if confirmation is not None:
        if "replicas" in confirmation and confirmation["replicas"] != cohort["replicas"]:
            raise ValueError("analysis and confirmation replica ledgers disagree")
        if "recipes" in confirmation and confirmation["recipes"] != cohort["recipes"]:
            raise ValueError("analysis and confirmation recipes disagree")
    if confirmation is not None or "records" in snapshot:
        rows = _records(snapshot)
        keys = [(row["recipe_key"], row["replica"]) for row in rows]
        if (len(keys) != len(expected) or set(keys) != expected
                or any(row.get("valid") is not True for row in rows)):
            raise ValueError("final inference cannot be attached to an incomplete/invalid cohort")
    return analysis


def _density():
    figure = _figure(height=3.3)
    axis = figure.subplots()
    x = np.linspace(-90, 90, 1801)
    magnitude = np.abs(x)
    novel = np.zeros_like(x)
    mask = magnitude > 0
    shape = .25
    normalization = .5 * (1 + math.erf(math.log(90 / 40) / shape / math.sqrt(2)))
    novel[mask] = np.exp(-.5 * (np.log(magnitude[mask] / 40) / shape) ** 2) / (
        2 * magnitude[mask] * shape * math.sqrt(2 * math.pi) * normalization)
    axis.plot(x, novel, color=COLORS[FD], label="Novel: signed truncated lognormal", linewidth=2)
    axis.fill_between(x, novel, color=COLORS[FD], alpha=.15)
    axis.plot([-90, -10, -10, 10, 10, 90], [0, 0, .05, .05, 0, 0],
              color=COLORS[REPLAY], label="Base: Uniform[-10, 10]", linewidth=2)
    axis.set(xlabel="Applied rotation (degrees)", ylabel="Density (per degree)", xlim=(-90, 90),
             title="Angle populations: near-upright background and larger novel rotations")
    axis.legend(fontsize=8, loc="upper left")
    _style(axis)
    return _image("density", figure, "The drip increases the novel-component probability while retaining "
                  "near-upright arrivals. Rejection above 90 degrees removes about 0.059% of magnitude mass; "
                  "there is no clipping or endpoint atom.")


def _progress(snapshot):
    progress = snapshot.get("progress", {})
    storage = progress.get("storage", {}) or {}
    used = progress.get("active_seconds", 0) / 3600
    allocation = progress.get("allocation_seconds", 72000) / 3600
    forecast = progress.get("forecast_seconds")
    forecast = "pending" if forecast is None else f"{forecast / 3600:.2f} h"
    if isinstance(storage, dict):
        size = storage.get("used_bytes", storage.get("bytes_used"))
        external = storage.get("external_bytes", storage.get("bytes_external"))
        reserved = storage.get("reserved_bytes", storage.get("bytes_reserved"))
        total = None if size is None or external is None else size + external
        gb = lambda value: "pending" if value is None else f"{value / 1e9:.3g} GB"
        storage_text = (f"Shared storage: {gb(total)} used / {gb(storage.get('max_bytes', 500_000_000_000))} cap "
                        f"(this study {gb(size)} + external {gb(external)}); {gb(reserved)} reserved")
    else:
        storage_text = _display(storage)
    pairs = (f"| {progress['paired_completed']}/{progress['paired_planned']} complete confirmation pairs "
             if "paired_completed" in progress else "")
    return _markdown("progress", "**Phase 8 progress** "
                     f"| {_display(snapshot.get('stage', '8A'))}: {_display(snapshot.get('status', 'pending'))} "
                     f"| {progress.get('completed', 0)}/{progress.get('planned', '?')} planned units "
                     + pairs +
                     f"| {used:.2f}/{allocation:.1f} active h | remaining forecast: {forecast}\n\n"
                     f"{storage_text}. Heartbeat: {_display(snapshot.get('heartbeat', progress.get('heartbeat')))}; "
                     f"revision: {_display(snapshot.get('revision'))}. "
                     + ("**Smoke acceptance only; no scientific evidence.**" if snapshot.get("smoke") else
                        "Development is exploratory; confirmation intervals require the entire frozen cohort."))


def _allocation(snapshot):
    rows = snapshot.get("allocation") or []
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    if not rows:
        return None
    rows = [flatten_record(row) for row in rows]
    rows = [row for row in rows if row["method"] in (FD, REPLAY)]
    if not rows:
        return None
    capacities = sorted(set(snapshot.get("allocation_grid", {}).get("capacities", [])) |
                        {row["capacity"] for row in rows})
    counts = sorted(set(snapshot.get("allocation_grid", {}).get("replay_counts", [])) |
                    {row["replay_count"] for row in rows})
    groups = {}
    seen = set()
    for row in rows:
        identity = row["method"], row["capacity"], row["replay_count"], row["replica"]
        if identity in seen:
            raise ValueError("duplicate allocation replica in supplied snapshot")
        seen.add(identity)
        groups.setdefault(identity[:3], []).append(row)
    matrices = {method: np.full((len(counts), len(capacities)), np.nan) for method in (FD, REPLAY)}
    difference = np.full_like(matrices[FD], np.nan)
    paired_counts = {}
    for i, b in enumerate(counts):
        for j, c in enumerate(capacities):
            for method in (FD, REPLAY):
                cell = groups.get((method, c, b), [])
                if c >= b and cell and all(row.get("valid") is True for row in cell):
                    values = [float(row["novel_auc"]) for row in cell]
                    if not all(math.isfinite(value) and value >= 0 for value in values):
                        raise ValueError("invalid allocation MSE")
                    matrices[method][i, j] = np.mean(values)
            fd = {row["replica"]: row for row in groups.get((FD, c, b), [])}
            replay = {row["replica"]: row for row in groups.get((REPLAY, c, b), [])}
            common = set(fd) & set(replay)
            if c >= b and common and all(fd[r].get("valid") and replay[r].get("valid") for r in common):
                for replica in common:
                    for field in ("pairing_id", "N", "frozen_novel_mse", "frozen_base_mse", "observations"):
                        if field in fd[replica] or field in replay[replica]:
                            if fd[replica].get(field) != replay[replica].get(field):
                                raise ValueError(f"unpaired allocation comparison: {field}")
                difference[i, j] = np.mean([replay[r]["novel_auc"] - fd[r]["novel_auc"] for r in common])
                paired_counts[i, j] = len(common)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
    if high == low:
        high = low + 1.
    extent = float(np.max(np.abs(difference[np.isfinite(difference)]))) if np.any(np.isfinite(difference)) else 1.
    extent = max(extent, 1e-12)
    figure = _figure(width=11., height=4.)
    axes = figure.subplots(1, 3)
    calibration = snapshot.get("calibration") or {}
    guard = calibration.get("base_retention_guard_degrees", calibration.get("retention_margin_degrees", 1.))
    limit = calibration.get("learning_seconds_per_observation_limit")
    from matplotlib.patches import Rectangle
    for axis, method in zip(axes, (FD, REPLAY, "difference")):
        matrix = difference if method == "difference" else matrices[method]
        plot = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                           cmap="RdBu" if method == "difference" else "viridis_r",
                           vmin=-extent if method == "difference" else low,
                           vmax=extent if method == "difference" else high)
        for i, b in enumerate(counts):
            for j, c in enumerate(capacities):
                cell = (groups.get((FD, c, b), []) + groups.get((REPLAY, c, b), [])
                        if method == "difference" else groups.get((method, c, b), []))
                if b > c:
                    color, hatch, label = "#d2d2d2", "///", "b > C"
                elif cell and not all(row.get("valid") is True for row in cell):
                    color, hatch, label = "#f3b7ad", "xx", "invalid"
                elif not np.isfinite(matrix[i, j]):
                    color, hatch, label = "#f2f2f2", None, "unpaired" if method == "difference" else "pending"
                else:
                    color, hatch = None, None
                    count = paired_counts[i, j] if method == "difference" else len(cell)
                    label = f"{matrix[i, j]:.1f}\nn={count}"
                    if method != "difference" and cell and all("base_auc" in row and "frozen_base_mse" in row for row in cell):
                        retention = np.mean([math.sqrt(row["base_auc"]) - math.sqrt(row["frozen_base_mse"]) for row in cell])
                        costly = limit is not None and np.mean([row["learning_seconds"] / row["observations"] for row in cell]) > limit
                        if retention > guard or costly:
                            label += " !"
                if color:
                    axis.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=color, hatch=hatch, edgecolor="#aaa", linewidth=.4))
                axis.text(j, i, label, ha="center", va="center", fontsize=8,
                          bbox=None if color else dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
        axis.set(xticks=range(len(capacities)), xticklabels=capacities, yticks=range(len(counts)),
                 yticklabels=counts, xlabel="Capacity C", ylabel="Replay count b",
                 title="Paired replay - FD" if method == "difference" else LABELS[method])
        figure.colorbar(plot, ax=axis, shrink=.8, label="Budget-averaged MSE (degrees squared)")
    return _image("allocation", figure, "Development only. Risk maps share a color scale; positive paired "
                  "differences favor FD. n is the replica count (paired intersection in the difference map); "
                  "unequal sets are not pooled as paired evidence. ! marks a retention/cost violation. "
                  "Missing and invalid cells remain distinct. Visual review: "
                  + _display(snapshot.get("allocation_review", "pending; numerical selection can proceed")))


def _curves(snapshot, analysis):
    if analysis:
        return analysis["curves"], True
    confirmation = snapshot.get("confirmation") or {}
    recipes = confirmation.get("recipes", [])
    if isinstance(recipes, dict):
        recipes = [dict(value, recipe_key=value.get("recipe_key", key)) for key, value in recipes.items()]
    keys = {recipe["recipe_key"] for recipe in recipes}
    if not keys:
        return {}, False
    rows = [row for row in _records(snapshot) if row.get("valid") is True and row.get("metrics")]
    groups = {key: {} for key in keys}
    for row in rows:
        if row["recipe_key"] not in groups:
            raise ValueError("unplanned confirmation recipe in report")
        group = groups[row["recipe_key"]]
        if row["replica"] in group:
            raise ValueError("duplicate confirmation record")
        group[row["replica"]] = row
    common = set.intersection(*(set(group) for group in groups.values()))
    for replica in common:
        paired = [group[replica] for group in groups.values()]
        for arm in paired[1:]:
            for field in ("pairing_id", "frozen_novel_mse", "frozen_base_mse"):
                if field in arm or field in paired[0]:
                    if arm.get(field) != paired[0].get(field):
                        raise ValueError(f"unpaired partial curves: {field}")
            if [row["observations"] for row in arm["metrics"]] != [row["observations"] for row in paired[0]["metrics"]]:
                raise ValueError("unpaired partial curve exposures")
    curves = {}
    for recipe in recipes:
        arms = [groups[recipe["recipe_key"]][replica] for replica in sorted(common, key=str)]
        if not arms:
            continue
        ns = [row["n"] for row in arms[0]["metrics"]]
        if ns != list(range(ns[-1] + 1)) or any([row["n"] for row in arm["metrics"]] != ns for arm in arms):
            raise ValueError("partial curve requires complete exact-budget checkpoints")
        curve = dict(method=recipe["method"], n=ns, n_pairs=len(arms))
        for domain in ("novel", "base"):
            mean = np.mean([[row[f"{domain}_mse"] for row in arm["metrics"]] for arm in arms], axis=0)
            if not np.all(np.isfinite(mean)) or np.any(mean < 0):
                raise ValueError("invalid partial predictive risk")
            curve[domain] = dict(mean_mse=mean.tolist(), mean_rmse=np.sqrt(mean).tolist())
        for name in ("observations", "learning_seconds"):
            curve[name] = np.mean([[row[name] for row in arm["metrics"]] for arm in arms], axis=0).tolist()
        curves[recipe["recipe_key"]] = curve
    if len({tuple(curve["n"]) for curve in curves.values()}) > 1:
        raise ValueError("paired report curves use different novel budgets")
    return curves, False


def _draw_curves(axis, curves, domain, *, bands=True):
    for curve in curves.values():
        method = curve["method"]
        axis.plot(curve["n"], curve[domain]["mean_rmse"], label=LABELS.get(method, method),
                  color=COLORS.get(method), linewidth=1.8)
        if bands and "ci_rmse" in curve[domain]:
            low, high = curve[domain]["ci_rmse"]
            axis.fill_between(curve["n"], low, high, color=COLORS.get(method), alpha=.16)
    first = next(iter(curves.values()))
    axis.axhline(first[domain]["mean_rmse"][0], color="#555", linestyle="--", label="Frozen pretrained model")
    axis.set(xlabel="Unique novel observations n", ylabel="Held-out RMSE (degrees)")
    _style(axis)


def _fresh(curves, final):
    figure = _figure(width=10., height=4.)
    axes = figure.subplots(1, 2)
    for axis, domain, title in zip(axes, ("novel", "base"), ("New-domain learning", "Near-upright retention")):
        _draw_curves(axis, curves, domain)
        axis.set_title(title)
        axis.legend(fontsize=7)
    count = next(iter(curves.values()))["n_pairs"]
    return _image("fresh", figure, f"{count} complete paired replicas; "
                  + ("pointwise nominal 95% whole-replica bootstrap bands, not simultaneous bands. " if final else
                     "descriptive interim means only; final inference pending. ")
                  + "RMSE is the square root of mean MSE, not mean replica RMSE. Fixed held-out panels never "
                    "enter fitting, replay, FD or Fisher estimation. The two domains use different labeled y scales.")


def _retirement_summary(snapshot):
    """Mean error and actual contributor counts at each observed retirement age.

    The runner audits only its fixed first-eight-novel cohort. Observed ages are
    not interpolated/binned, and missing/unretired exposures are not zero-filled.
    These repeated training-cohort predictions have no inferential error bars.
    """
    grouped, members, identities, retirement_times = defaultdict(list), defaultdict(set), set(), {}
    labels = {}
    for row in _records(snapshot):
        audit = row.get("retired_audit") or []
        if not audit:
            continue
        key, replica = row["recipe_key"], row["replica"]
        labels[key] = dict(method=row["method"], capacity=row.get("capacity", row.get("recipe", {}).get("capacity")))
        checkpoints = {item["n"]: item["observations"] for item in row["metrics"]}
        for item in audit:
            for field in ("n", "exposure", "observations", "retirement_observations", "age_since_retirement"):
                if type(item[field]) is not int or item[field] < 0:
                    raise ValueError("retirement audit requires nonnegative integer exposure clocks")
            age, exposure = item["age_since_retirement"], item["exposure"]
            if (item["observations"] != checkpoints.get(item["n"])
                    or item["observations"] - item["retirement_observations"] != age
                    or item["retirement_observations"] <= exposure):
                raise ValueError("retirement audit clocks disagree with recorded checkpoints")
            identity = key, replica, exposure, age
            if identity in identities:
                raise ValueError("duplicate retired-cohort prediction at an observed age")
            identities.add(identity)
            member = key, replica, exposure
            retired = retirement_times.setdefault(member, item["retirement_observations"])
            if retired != item["retirement_observations"]:
                raise ValueError("one exposure has conflicting retirement times")
            error = float(item["squared_error"])
            if not math.isfinite(error) or error < 0:
                raise ValueError("retired-cohort squared error must be finite and nonnegative")
            members[key, replica].add(exposure)
            if len(members[key, replica]) > 8:
                raise ValueError("retirement diagnostic exceeds the fixed first-eight-novel cohort")
            grouped[key, age].append(error)
    result = {}
    for key, label in labels.items():
        ages = sorted(age for group, age in grouped if group == key)
        result[key] = dict(**label, age=ages,
                          mean_mse=[float(np.mean(grouped[key, age])) for age in ages],
                          counts=[len(grouped[key, age]) for age in ages],
                          unique_observations=sum(len(values) for (group, _), values in members.items() if group == key))
    return result


def _effects(analysis, snapshot):
    audits = _retirement_summary(snapshot)
    figure = _figure(width=12.5 if audits else 10., height=4.1 if audits else 3.5)
    if audits:
        grid = figure.add_gridspec(2, 3, height_ratios=[3, 1])
        left, right = figure.add_subplot(grid[:, 0]), figure.add_subplot(grid[:, 1])
        audit_axis = figure.add_subplot(grid[0, 2])
        count_axis = figure.add_subplot(grid[1, 2], sharex=audit_axis)
        for curve in audits.values():
            method, c = curve["method"], curve["capacity"]
            label = {FD: "FD KL", REPLAY: "Replay/EWC", "ewc_only": "EWC only"}.get(method, method)
            audit_axis.plot(curve["age"], curve["mean_mse"], color=COLORS.get(method),
                            linewidth=1.2, marker=".", markersize=2, label=f"{label} (C={c})")
            count_axis.plot(curve["age"], curve["counts"], color=COLORS.get(method), linewidth=1.2)
        audit_axis.set(title="Retired training cohort", ylabel="Mean squared error\n(degrees squared)")
        audit_axis.tick_params(labelbottom=False)
        audit_axis.legend(fontsize=7)
        count_axis.set(xlabel="Arrivals since actual retirement", ylabel="Audit\ncount", ylim=(0, None))
        for axis in (audit_axis, count_axis):
            _style(axis)
    else:
        left, right = figure.subplots(1, 2)
    if not analysis:
        for axis, title in ((left, "New-data efficiency"), (right, "Held-out retention constraint")):
            axis.set_title(title)
            axis.text(.5, .5, "Acceptance data only" if snapshot.get("smoke") else "Paired inference pending",
                      transform=axis.transAxes, ha="center", va="center", fontsize=10)
            axis.set_axis_off()
        return _image("effects", figure, _retirement_caption(audits) + " No scientific inference from smoke or partial cohorts.")
    recipes = analysis["cohort"]["recipes"]
    fd = next(row["recipe_key"] for row in recipes if row["method"] == FD)
    effects = [analysis["risk_gain"], analysis["frozen_gain"][fd]]
    for index, effect in enumerate(effects):
        mean, (low, high) = effect["mean"], effect["ci"]
        left.plot([low, high], [index, index], color=COLORS[FD], linewidth=2)
        left.scatter([mean], [index], color=COLORS[FD], zorder=3)
    left.axvline(0, color="#555", linewidth=.8)
    left.set(yticks=[0, 1], yticklabels=["Replay - FD", "Frozen - FD"],
             xlabel="Mean budget-averaged MSE gain\n(degrees squared; positive favors FD)",
             title="New-data efficiency", ylim=(-.6, 1.6))
    for index, recipe in enumerate(recipes):
        effect = analysis["retention"][recipe["recipe_key"]]
        right.plot(effect["ci"], [index, index], color=COLORS.get(recipe["method"]))
        right.scatter([effect["mean"]], [index], color=COLORS.get(recipe["method"]))
    guard = next(iter(analysis["retention"].values()))["guard_degrees"]
    right.axvline(guard, color="#555", linestyle="--", label=f"Guard: {guard:g} degree(s)")
    right.axvline(0, color="#aaa", linewidth=.8)
    right.set(yticks=range(len(recipes)), yticklabels=[LABELS.get(row["method"], row["method"]) for row in recipes],
              xlabel="Mean paired increase in sqrt(base A(N))\nversus frozen (degrees)", title="Held-out retention constraint")
    right.legend(fontsize=8)
    for axis in (left, right):
        _style(axis)
    return _image("effects", figure, "Nominal paired 95% intervals resample complete trajectories, not timepoints. "
                  "Acquisition averages predictive MSE over n=1,...,N; initialization is excluded. "
                  "Retention is mean per-replica sqrt(A_base) minus frozen RMSE. "
                  + _retirement_caption(audits))


def _retirement_caption(audits):
    if not audits:
        return "Retired training-cohort audits are pending, not zero error."
    counts = "; ".join(f"{LABELS.get(row['method'], row['method'])}: {row['unique_observations']} unique retired images"
                       for row in audits.values())
    return ("Retired panel: up to the first eight novel arrivals per replica, measured only after actual retirement. "
            "Each point averages supplied audit predictions at that exact observed age; the lower panel shows "
            "contributing prediction counts, not independent sample sizes. " + counts + ". "
            "Different capacities imply different retirement times and available cohorts: this is descriptive "
            "training-data retention, not a paired causal contrast or held-out generalization.")


def _equivalence(curves, analysis, snapshot):
    confirmation = snapshot.get("confirmation") or {}
    choices = analysis["cohort"] if analysis else confirmation
    n_fd, N = choices.get("n_F"), choices.get("N")
    q = analysis.get("q_dev") if analysis else confirmation.get("q_dev")
    if n_fd is None or N is None:
        return None
    figure = _figure(height=4.4)
    axis = figure.subplots()
    selected = {key: value for key, value in curves.items() if value["method"] in (FD, REPLAY)}
    if len(selected) != 2:
        return None
    _draw_curves(axis, selected, "novel")
    if q is not None:
        axis.axhline(math.sqrt(q), color="#97721c", linestyle=":", label="Fixed development target")
    budgets = []
    for key, curve in selected.items():
        budget = n_fd if curve["method"] == FD else N
        if budget not in curve["n"]:
            raise ValueError("prespecified sample budget has no measured checkpoint")
        position = curve["n"].index(budget)
        axis.axvline(budget, color=COLORS[curve["method"]], linestyle=":", alpha=.6)
        axis.scatter([budget], [curve["novel"]["mean_rmse"][position]], s=45,
                     color=COLORS[curve["method"]], zorder=5)
        budgets.append(f"{LABELS[curve['method']]} at n={budget}: "
                       f"{curve['observations'][position]:.1f} total arrivals, "
                       f"{curve['learning_seconds'][position]:.1f} learning seconds")
        fit = (analysis or {}).get("linear_fits", {}).get("methods", {}).get(key)
        if fit and min(fit["fitted_mse"]) >= 0:
            axis.plot(fit["n"], np.sqrt(fit["fitted_mse"]), color=COLORS[curve["method"]],
                      linestyle="--", alpha=.8, label=f"{LABELS[curve['method']]}: registered linear MSE fit")
    axis.set_title("Sample-equivalence diagnostic: directly observed comparison budgets")
    axis.legend(fontsize=8)
    caption = "; ".join(budgets) + ". Counts exclude shared pretraining but retain accompanying background exposure. "
    caption += "This is not a causal label-only effect or a first-passage samples-saved estimate."
    if analysis:
        caption += " FD(n_F) - replay(N) MSE: " + _interval_text(analysis["direct_contrast"]) + "."
    else:
        caption += " Interim descriptive comparison; final paired interval pending."
    return _image("equivalence", figure, caption)


def _conclusion(analysis):
    if not analysis:
        return _markdown("conclusion", "### Conclusion\n\n**Pending.** Partial or development results are descriptive; "
                         "no final inference is published until every planned confirmation trajectory and its analysis complete.")
    recipes = analysis["cohort"]["recipes"]
    fd = next(row["recipe_key"] for row in recipes if row["method"] == FD)
    gain, frozen = analysis["risk_gain"], analysis["frozen_gain"][fd]
    direct, retention, resources = analysis["direct_contrast"], analysis["retention"][fd], analysis["resources"][fd]
    budget_claim = "supports FD advantage" if gain["ci"][0] > 0 else "FD advantage not established"
    frozen_claim = "supports useful acquisition" if frozen["ci"][0] > 0 else "improvement over freezing not established"
    direct_claim = "supports better prediction at fewer observations" if direct["ci"][1] < 0 else "not established; nonsignificance is not equivalence"
    resource_claim = {True: "within measured mean-cost envelope", False: "exceeds mean-cost envelope", None: "resource envelope pending"}[resources["resource_feasible"]]
    body = "### Conclusion\n\n| Question | Mean [nominal 95% interval] | Interpretation |\n| --- | --- | --- |\n"
    body += f"| Budget-averaged replay - FD MSE | {_interval_text(gain)} | {budget_claim} |\n"
    body += f"| Frozen - FD MSE | {_interval_text(frozen)} | {frozen_claim} |\n"
    body += f"| FD({direct['n_F']}) - replay({direct['n_R']}) MSE | {_interval_text(direct)} | {direct_claim} |\n"
    body += f"| FD base RMSE increase (degrees) | {_interval_text(retention)} | " + (
        "retention guard supported" if retention["meets_guard"] else "retention guard not established") + " |\n"
    body += f"\nFD: {resources['mean_seconds_per_observation']:.3g} learning seconds per arrival; {resource_claim}. "
    if "peak_vram_bytes" in resources:
        body += f"Peak VRAM {resources['peak_vram_bytes'] / 1e9:.3g} GB. "
    body += (f"{gain['favorable_replicas']}/{gain['n_pairs']} paired replicas favor FD on budget-averaged new-domain risk. "
             "Practical success requires acquisition, retention and resource feasibility together. "
             "Input supply is assumed overlap-able; offloading engineering and large-network timing extrapolation are excluded.\n\n"
             + analysis["assumptions"])
    return _markdown("conclusion", body)


def build_cells(snapshot: dict, *, check_stop=None) -> list:
    """Return a progress cell followed by the chronological, at-most-five-figure section."""
    if snapshot.get("schema_version") != 1 or snapshot.get("protocol") != PROTOCOL:
        raise ValueError("incompatible Phase 8 snapshot schema/protocol")
    _check(check_stop)
    analysis = None if snapshot.get("smoke") else _validated_analysis(snapshot)
    cells = [_progress(snapshot), _markdown("intro", "## Phase 8: Regression, Allocation And Sample Efficiency\n\n"
             "**Question:** under limited new data and compute, how should replay capacity and replay work be "
             "allocated, and can FD KL improve learning while retaining near-upright prediction?\n\n"
             r"$C$ is logical FIFO replay capacity; $b$ is the fixed resident subset per solve, not total closure passes. "
             r"$n$ counts unique novel arrivals. $R^D(n)$ is predictive MSE on a fixed held-out image/angle panel; "
             r"$A^D(N)=N^{-1}\sum_{n=1}^{N}R^D(n)$ averages over novel-sample budgets, excluding initialization. "
             "Every n checkpoint follows an accepted update with exactly n novel observations. "
             "Gaussian NLL uses a fixed 45-degree scale and contains no regularizers. "
             "New-domain learning and near-upright retention are evaluated separately. "
             "Neither evaluation panel enters training, replay, or Fisher estimation; development selects policies, "
             "and separate confirmation panels never tune them.\n\n"
             + ("**Phase 8A smoke/pending scaffold; not scientific results.**" if snapshot.get("smoke") else
                "**Status:** " + _display(snapshot.get("status", "pending"))))]
    cells.append(_density())
    _check(check_stop)
    acceptance = _acceptance(snapshot)
    if acceptance:
        cells.append(acceptance)
    calibration = snapshot.get("calibration") or {}
    if calibration:
        rows = [("Initialization", calibration.get("initialization_status")),
                ("Solver/FD numerical checks", calibration.get("numerical_status")),
                ("Ridge multiplier c", calibration.get("ridge", calibration.get("c"))),
                ("FD strength / step", f"{_display(calibration.get('strength'))} / {_display(calibration.get('fd_step'))}"),
                ("Measured FD/replay learning-time ratio", calibration.get("cost_ratio")),
                ("Seconds/arrival allowance", calibration.get("learning_seconds_per_observation_limit"))]
        text = "### Calibration\n\n| Check | Development result |\n| --- | --- |\n"
        text += "\n".join(f"| {name} | {_display(value)} |" for name, value in rows)
    else:
        text = "### Calibration\n\nPending: regression initialization, numerical adequacy, regularization and measured cost. The categorical 2x ratio is not assumed."
    cells.append(_markdown("calibration", text))
    allocation = _allocation(snapshot)
    cells.append(allocation or _markdown("allocation-pending", "### Allocation\n\nDevelopment heatmaps pending; missing cells are not failed policies."))
    _check(check_stop)
    curves, final = _curves(snapshot, analysis)
    if curves:
        cells.append(_fresh(curves, final))
    else:
        cells.append(_markdown("confirmation-pending", "### Fresh Confirmation\n\nPending complete paired trajectories. Development does not contribute to confirmation intervals."))
    _check(check_stop)
    if analysis or any(row.get("retired_audit") for row in _records(snapshot)):
        cells.append(_effects(analysis, snapshot))
    if curves:
        equivalence = _equivalence(curves, analysis, snapshot)
        if equivalence:
            cells.append(equivalence)
    changes = snapshot.get("changes") or []
    if changes:
        cells.append(_changes(changes))
    cells.append(_markdown("conclusion", "### Conclusion\n\n**Acceptance only.** Smoke trajectories verify execution "
                           "and recovery, not scientific effectiveness. Development and fresh confirmation remain pending.")
                 if snapshot.get("smoke") else _conclusion(analysis))
    _check(check_stop)
    return cells


def _owned(cell):
    metadata = cell.get("metadata", {})
    return TAG in metadata or TAG in metadata.get("tags", []) or cell.get("id", "").startswith(PREFIX)


def _read_previous(path):
    path = Path(path)
    if path.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("existing notebook exceeds bounded publication size")
    with path.open("rb") as handle:
        content = handle.read(MAX_NOTEBOOK_BYTES + 1)
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ValueError("existing notebook exceeds bounded publication size")
    return json.loads(content)


def build_notebook(snapshot: dict, previous=None, *, check_stop=None) -> dict:
    """Keep all non-Phase-8 cells unchanged and ordered; insert progress near top."""
    if isinstance(previous, (str, Path)):
        previous = _read_previous(previous)
    book = deepcopy(previous) if previous is not None else dict(nbformat.v4.new_notebook())
    if book.get("nbformat") != 4 or not isinstance(book.get("cells"), list):
        raise ValueError("expected a version-4 notebook")
    book["nbformat_minor"] = max(5, book.get("nbformat_minor", 0))
    historical = [cell for cell in book["cells"] if not _owned(cell)]
    new = build_cells(snapshot, check_stop=check_stop)
    for cell in new:
        nbformat.validate(cell, ref="markdown_cell", version=4, version_minor=5)
    book["cells"] = historical[:1] + [new[0]] + historical[1:] + new[1:]
    if len(json.dumps(book, ensure_ascii=False, allow_nan=False).encode()) > MAX_NOTEBOOK_BYTES:
        raise ValueError("notebook exceeds bounded publication size")
    return book


def publish(snapshot: dict, notebook: Path, store=None) -> Path:
    """Atomic bounded publication; a supplied Store accounts for temporary bytes.

    Uses the legacy reserved publisher inside Store.root/reports. External
    exports reserve the same temporary bytes while retaining the historical
    JSON objects verbatim. No snapshot or archive discovery occurs here.
    """
    notebook = Path(notebook)
    check = getattr(store, "check", None)
    _check(check)
    previous = _read_previous(notebook) if notebook.exists() else None
    book = build_notebook(snapshot, previous, check_stop=check)
    content = json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ValueError("notebook exceeds bounded publication size")
    if store is not None and notebook.resolve().is_relative_to((Path(store.root) / "reports").resolve()):
        from ..fisher_memory.report import reserved_publisher
        reserved_publisher(store)(notebook, content)
        return notebook
    with store.reserve(len(content)) if store is not None else nullcontext():
        _check(check)
        notebook.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-plan15-phase8-", dir=notebook.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for offset in range(0, len(content), 1024 * 1024):
                    _check(check)
                    handle.write(content[offset:offset + 1024 * 1024])
                handle.flush()
                os.fsync(handle.fileno())
            _check(check)
            os.replace(temporary, notebook)
            directory = os.open(notebook.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return notebook
