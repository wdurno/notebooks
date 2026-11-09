"""Bounded, read-only scientific reporting with immutable atomic generations.

The public notebook is a presentation of scalar exports, not an execution
entry point. Missing exports in sealed trajectories are errors. An absent
store is a legitimate pending report. No checkpoint tensors are deserialized.
"""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan18-matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import nbformat
import numpy as np

from . import PROTOCOL
from .analysis import ARMS, LABELS, _interval, paired_draws, setting, summarize

REPORT_SCHEMA = 1
# Up to 78 production paths export about 1.6 MiB of scalar diagnostics each.
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_GENERATION_BYTES = 24 * 1024 * 1024
COLORS = dict(zip(ARMS, ("#53616b", "#21866c", "#377ab5", "#b07924", "#b84f65")))
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "plan18_results.ipynb"


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _cfg_mapping(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "mapping"):
        return cfg.mapping()
    return vars(cfg)


def _snapshot(root, cfg, check):
    seals, consumed = {}, 0

    def read(path, required=False):
        nonlocal consumed
        check()
        if not path.exists() and not required:
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"report input must be a regular file: {path.name}")
        size = path.stat().st_size
        if consumed + size > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        data = path.read_bytes()
        consumed += len(data)
        if consumed > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        seals[path.relative_to(root).as_posix()] = _hash(data)
        return json.loads(data)

    identity = read(root / "identity.json")
    if identity is not None:
        scientific = identity.get("identity", {})
        if (scientific.get("protocol", PROTOCOL) != PROTOCOL
                or scientific.get("metric_schema_version", 1) != 1
                or scientific.get("artifact_schema_version", 1) != 1):
            raise ValueError("incompatible Plan 18 protocol or scientific artifact schema")
    status = read(root / "status.json") or {}
    selection = read(root / "selection.json") or status.get("selection")
    config = read(root / "config.json")
    if cfg is None:
        config = config or (identity or {}).get("identity", {}).get("config", {})
        cfg = config
    records = []
    if identity is not None:
        from mnist_experiment.fisher_memory.artifacts import iter_units

        for unit in iter_units(root, check=check):
            check()
            path = unit["path"]
            read(path / "manifest.json", required=True)
            if unit["metadata"].get("kind") != "trajectory":
                continue
            record = {k: unit[k] for k in ("unit_id", "metadata", "status")}
            if unit["status"] == "completed":
                read(path / "COMPLETED", required=True)
                record["summary"] = read(path / "summary.json", required=True)
                record["metrics"] = read(path / "metrics.json", required=True)
                record["diagnostics"] = read(path / "diagnostics.json", required=True)
                if (not isinstance(record["summary"], dict) or not isinstance(record["metrics"], list)
                        or not isinstance(record["diagnostics"], list)):
                    raise ValueError("incompatible completed trajectory scalar export schema")
            records.append(record)
    elif (root / "units").exists() and any((root / "units").iterdir()):
        raise ValueError("scientific units exist without a valid Store identity")
    readiness = read(root / "readiness.json") or {}
    numerical = read(root / "numerics.json") or readiness.get("numerical_policy") or status.get("numerical_policy") or {}
    notes = read(root / "execution_notes.json") or []
    if not isinstance(notes, list) or any(not isinstance(n, (str, dict)) for n in notes):
        raise ValueError("execution_notes.json must be an append-only list of text or objects")
    resolved = _cfg_mapping(cfg)
    sources = {name: _hash(Path(__file__).with_name(name).read_bytes())
               for name in ("analysis.py", "report.py")}
    inputs = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, files=seals,
                  config=resolved, implementation=sources)
    return dict(records=records, cfg=cfg, status=status, selection=selection,
                readiness=readiness, numerical_policy=numerical, execution_notes=notes,
                inputs=inputs, source_digest=_hash(_bytes(inputs)))


def _figure(rows=1, columns=2):
    figure = Figure(figsize=(11, 3.25 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(rows, columns, squeeze=False)).ravel()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.16)
        axis.tick_params(labelsize=8)
    return figure, axes


def _pending(axis, title):
    axis.set_title(title, fontsize=11)
    axis.text(.5, .5, "Pending scalar artifacts", ha="center", va="center", transform=axis.transAxes,
              color="#687078", fontsize=10)


def _legend(axis):
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=7, loc="best")


def _flat(record):
    return {**record.get("summary", {}), **record, **record.get("metadata", {})}


def _paired_records(snapshot, result, cohort="fresh"):
    replicas = result["cohorts"][cohort]["replicas"]
    return [_flat(r) for r in snapshot["records"] if r.get("status") == "completed"
            and r["metadata"]["cohort"] == cohort and r["metadata"]["replica"] in replicas]


def _add_incumbent_contrast(snapshot, result):
    """Supplement the frozen analysis with B-A using its original paired draws."""
    fresh = result["cohorts"]["fresh"]
    replicas = fresh["replicas"]
    if not replicas:
        return
    reference = {r["replica"]: r for r in _paired_records(snapshot, result)
                 if r["recipe"]["arm"] == "A"}
    signs = [reference[r].get("global_sign", -1 if r % 2 else 1) for r in replicas]
    bootstrap = fresh["bootstrap"]
    samples = (paired_draws(replicas, bootstrap["draws"], bootstrap["seed"], signs)
               if bootstrap["draws"] else None)
    effects = {}
    for field in ("novel_auc", "Havg", "H100", "current_whole_rms"):
        values = [np.asarray(fresh["methods"][arm][field].get(
            "per_replica", [np.nan] * len(replicas)), dtype=float) for arm in ("B", "A")]
        effects[field] = _interval(values[0] - values[1], samples)
    fresh["comparisons"] = {"B-A": effects, **fresh["comparisons"]}
    result["comparisons"] = fresh["comparisons"]
    result["report_additions"] = {"B-A": "Post-run supplemental contrast; original paired bootstrap reused."}


def _task(snapshot, result):
    cfg = snapshot["cfg"]
    figure, axes = _figure(2, 2)
    hold, ramp = setting(cfg, "hold_steps", 160), setting(cfg, "ramp_steps", 304)
    m = setting(cfg, "samples_per_step", 4)
    steps = np.arange(2 * (hold + ramp) + 1)
    h = np.zeros_like(steps, dtype=float)
    out = (steps > hold) & (steps <= hold + ramp)
    back = (steps > hold + ramp) & (steps <= hold + 2 * ramp)
    h[out] = (1 - np.cos(np.pi * (steps[out] - hold) / ramp)) / 2
    h[back] = (1 + np.cos(np.pi * (steps[back] - hold - ramp) / ramp)) / 2
    axes[0].plot(m * steps, h, color="#414950", label="Declared regime envelope")
    axes[0].set(xlabel="All incoming grids", ylabel="Rotation severity h", title="Quiet / outbound / return / quiet")
    expected = np.cumsum(h * m * setting(cfg, "novel_probability", .125))
    axes[1].plot(m * steps, expected, color="#414950", linestyle="--", label="Expected novel grids")
    seen = set()
    for record in snapshot["records"]:
        row = _flat(record)
        if row.get("status") != "completed" or row["cohort"] == "stationary":
            continue
        key = row["cohort"], row["replica"]
        if key in seen:
            continue
        seen.add(key)
        metrics = row["metrics"]
        axes[1].plot([r["incoming_grids"] for r in metrics], [r["novel_count"] for r in metrics],
                     alpha=.6, label=f"{key[0]} r{key[1]}")
    axes[1].axhline(setting(cfg, "analysis_novel_count", 100), color="#b84f65", linewidth=1, label="Analysis budget")
    axes[1].set(xlabel="All incoming grids (9 digit pairs/grid)", ylabel="Unique novel grids", title="Expected and observed exposure")
    angle = np.linspace(-70, 70, 501)
    sigma = setting(cfg, "angle_std", 6.)
    for location, name, color in ((0., "Nominal", "#53616b"), (22.5, "Novel h=.5", "#21866c"),
                                  (45., "Novel h=1, positive cell", "#b84f65")):
        density = np.exp(-.5 * ((angle - location) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        mass = .5 * (math.erf((70 - location) / (sigma * math.sqrt(2)))
                     - math.erf((-70 - location) / (sigma * math.sqrt(2))))
        axes[2].plot(angle, density / mass, label=name, color=color)
    axes[2].set(xlabel="Cell rotation (degrees)", ylabel="Density / degree", title="Fixed truncation support [-70, 70]")
    # Rendered grids are audit exports; reporting never invokes the renderer.
    grid = snapshot["readiness"].get("rendered_grid")
    if grid is not None:
        values = np.asarray(grid, dtype=float)
        if values.ndim not in (2, 3) or values.size > 240 * 240 * 4 or not np.isfinite(values).all():
            raise ValueError("invalid bounded readiness rendered_grid export")
        axes[3].imshow(values, cmap="gray")
        axes[3].set_title("Exported rendering audit", fontsize=11)
        axes[3].set_axis_off()
    else:
        _pending(axes[3], "Rendered-grid readiness audit")
    for axis in axes[:3]:
        _legend(axis)
    return figure


def _development(snapshot, result):
    figure, axes = _figure()
    recipes = result["selection"]
    for arm in ARMS:
        group = [r for r in result["development"] if r["recipe"]["arm"] == arm]
        for axis, field in zip(axes, ("Havg", "H100")):
            points = [r for r in group if r.get(field) is not None and r.get("novel_auc") is not None]
            if points:
                axis.scatter([r[field] for r in points], [r["novel_auc"] for r in points], s=25,
                             color=COLORS[arm], alpha=.55, label=arm + " all attempts")
                chosen = [r for r in points if r["recipe"] == recipes.get(arm)]
                if chosen:
                    axis.scatter([r[field] for r in chosen], [r["novel_auc"] for r in chosen], s=80,
                                 facecolors="none", edgecolors=COLORS[arm], linewidths=1.7, marker="s")
            axis.axvline(setting(snapshot["cfg"], "retention_ceiling", .5), color="#777", linestyle=":")
            axis.set(xlabel=f"{field} deterioration (pixels)", ylabel="Novel A(100) (squared pixels / landmark)",
                     title="All development attempts; outlined = selected")
    for axis in axes:
        if not axis.collections:
            _pending(axis, "Development calibration pending")
        _legend(axis)
    return figure


def _predictive(snapshot, result):
    figure, axes = _figure(2, 3)
    fresh = result["cohorts"]["fresh"]
    records = _paired_records(snapshot, result)
    for arm, method in fresh["methods"].items():
        curves = method["curves"]
        for axis, field, title, root in zip(axes[:4],
                ("novel_mse", "old_mse", "novel_accuracy", "old_accuracy"),
                ("Fixed novel panel", "Fixed old panel", "Novel one-pixel accuracy", "Old one-pixel accuracy"),
                (True, True, False, False)):
            item = curves[field]
            values = np.asarray(item["mean"], dtype=float)
            transform = np.sqrt if root else lambda x: x
            axis.plot(curves["n"], transform(values), color=COLORS[arm], label=arm)
            if item["ci"] is not None:
                lower, upper = transform(np.asarray(item["ci"], dtype=float))
                axis.fill_between(curves["n"], lower, upper, color=COLORS[arm], alpha=.10)
            axis.set(xlabel="Unique novel grids seen", title=title,
                     ylabel="RMS landmark error (pixels)" if root else "Fraction within one pixel")
        group = [r for r in records if r["recipe"]["arm"] == arm]
        regular = [[r for r in row["metrics"] if r["kind"] in ("initial", "regular")] for row in group]
        if regular:
            clock = [r["incoming_grids"] for r in regular[0]]
            matrix = np.asarray([[r["current_mse"] for r in rows] for rows in regular])
            axes[4].plot(clock, np.sqrt(matrix.mean(axis=0)), color=COLORS[arm], label=arm)
            for row in regular:
                axes[5].plot([p["incoming_grids"] for p in row], [p["current_accuracy"] for p in row],
                             color=COLORS[arm], alpha=.3, linewidth=1)
            if arm == "A" and all(r.get("frozen_current_mse") is not None for rows in regular for r in rows):
                frozen = np.asarray([[r["frozen_current_mse"] for r in rows] for rows in regular])
                axes[4].plot(clock, np.sqrt(frozen.mean(axis=0)), color="#222", linestyle="--", label="Frozen, changing law")
            if arm == "A":
                for axis, field in zip(axes[:2], ("frozen_novel_mse", "frozen_old_mse")):
                    axis.axhline(math.sqrt(np.mean([r[field] for r in group])), color="#222", linestyle="--", label="Frozen initialization")
    axes[4].set(xlabel="All incoming grids", ylabel="RMS landmark error (pixels)", title="Current law: quiet / ramp / return")
    axes[5].set(xlabel="All incoming grids", ylabel="Fraction within one pixel", title="Current accuracy: individual replicas")
    if not fresh["methods"]:
        for axis, title in zip(axes, ("Fixed novel panel", "Fixed old panel", "Novel accuracy", "Old accuracy", "Current law", "Current accuracy")):
            _pending(axis, title)
    for axis in axes:
        _legend(axis)
    return figure


def _health(snapshot, result):
    figure, axes = _figure(2, 2)
    fields = (("applied_pi", "raw_pi"), ("motion_energy", "uncertainty_energy", "noise_scale"),
              ("negative_mass", "projection_norm"), ("stm_sensitivity", "eviction_mse", "stm_move"))
    for arm in ARMS:
        rows = [r for r in _paired_records(snapshot, result) if r["recipe"]["arm"] == arm]
        for axis, names in zip(axes, fields):
            for index, name in enumerate(names):
                by_batch = {}
                for row in rows:
                    for d in row.get("diagnostics", []):
                        value = d.get(name)
                        batch = d.get("incoming_grids", d.get("batch", 0) * setting(snapshot["cfg"], "samples_per_step", 4))
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            by_batch.setdefault(batch, []).append(value)
                if by_batch:
                    clock = sorted(by_batch)
                    axis.plot(clock, [np.mean(by_batch[b]) for b in clock], color=COLORS[arm],
                              linestyle=("-", "--", ":")[index], label=arm + " " + name, linewidth=1)
    for axis, title, ylabel in zip(axes,
            (r"Raw and applied $\pi$", "Movement / uncertainty / noise", "Before-projection archive health", "STM / eviction diagnostics (in-sample)"),
            (r"$\pi$", "Recorded energy or scale", "Recorded mass or norm", "Recorded proxy value")):
        axis.set(xlabel="All incoming grids", ylabel=ylabel, title=title)
        if not axis.lines:
            _pending(axis, title)
        _legend(axis)
    return figure


def _efficiency(snapshot, result):
    figure, axes = _figure(1, 3)
    items = result["efficiency"]
    for i, arm in enumerate(ARMS):
        item = items.get(f"{arm}-A")
        if not item:
            continue
        p = item["point"]
        if p["n"] is not None:
            axes[0].scatter(p["n"], i, color=COLORS[arm], label=arm)
        for name, offset, marker in (("equal100_time_ratio", -.1, "o"), ("matched_time_ratio", .1, "s")):
            if p.get(name) is not None:
                axes[1].scatter(p[name], i + offset, color=COLORS[arm], marker=marker,
                                label=name.replace("_", " ") if arm == "A" else None)
    axes[0].axvline(setting(snapshot["cfg"], "analysis_novel_count", 100), color="#777", linestyle=":")
    axes[1].axvline(1, color="#777", linestyle=":")
    for axis in axes[:2]:
        axis.set(yticks=range(5), yticklabels=ARMS)
    axes[0].set(xlabel="Novel grids to common A target", title="Persistent target crossing")
    axes[1].set(xlabel="Standalone learning time / A at 100", title="Equal-data / matched compute")
    stationary = _paired_records(snapshot, result, "stationary")
    for arm in ARMS:
        for row in stationary:
            if row["recipe"]["arm"] == arm:
                regular = [m for m in row["metrics"] if m["kind"] in ("initial", "regular")]
                axes[2].plot([m["incoming_grids"] for m in regular],
                    np.sqrt([m["current_mse"] for m in regular]), color=COLORS[arm], alpha=.65,
                    label=f"{arm} r{row['replica']}")
    axes[2].set(xlabel="All incoming grids", ylabel="RMS landmark error (pixels)", title="Stationary: paired trajectories")
    if not items:
        _pending(axes[0], "Common-target crossings pending")
        _pending(axes[1], "Compute ratios pending")
    if not stationary:
        _pending(axes[2], "Stationary diagnostic pending")
    for axis in axes:
        _legend(axis)
    return figure


def _fmt(value):
    return "pending" if value is None else f"{value:.3g}" if isinstance(value, (int, float)) else str(value).replace("|", "/").replace("\n", " ")


def _effect(item):
    text = _fmt(item.get("mean"))
    if item.get("ci") is not None:
        text += f" [{_fmt(item['ci'][0])}, {_fmt(item['ci'][1])}]"
    return text


def _table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "".join(
        "| " + " | ".join(_fmt(v) for v in row) + " |\n" for row in rows)


def _progress(snapshot, result):
    status = snapshot["status"]
    counts = Counter(_flat(r).get("cohort") for r in snapshot["records"] if r["status"] == "completed")
    current = status.get("current") or {}
    resources = status.get("resources") or {}
    completed = status.get("complete_counts", status.get("completed", status.get("complete", dict(counts))))
    total = status.get("total_paths") or "pending allocation"
    blocks = ", ".join(f"{c}: {r['n_pairs']}/{r['planned_pairs']}" for c, r in result["cohorts"].items())
    forecast = status.get("forecast")
    if forecast is None and status.get("forecast_remaining_seconds") is not None:
        forecast = f"{status['forecast_remaining_seconds'] / 3600:.2f} active h"
    if isinstance(forecast, dict):
        seconds = forecast.get("remaining_seconds", forecast.get("seconds"))
        forecast = "pending" if seconds is None else f"{seconds / 3600:.2f} active h"
    if forecast is None:
        forecast = "pending"
    headroom = resources.get("filesystem_free_bytes", resources.get("free_bytes"))
    phase = status.get("phase", "0: implementation; execution pending")
    if status.get("status") == "completed":
        phase = "Complete"
    active = status.get("active_seconds")
    charged = "pending ledger" if active is None else f"{active / 3600:.3f}"
    dev_limit = ("smoke grid" if setting(snapshot.get("cfg", {}), "smoke", False)
                 else "up to 34 planned, plus at most four refinement paths")
    return (f"# Plan 18: Combined Memory\n\n**Progress: {phase}**  \n"
        f"Revision: {status.get('revision', 'implementation')} | completed paths by role: {completed}. "
        f"Planned fresh + stationary paths: {total} (development excluded).  \n"
        f"Complete paired blocks: {blocks}. Development paths: {counts['development']} ({dev_limit}); tuning evidence only.  \n"
        f"Active unit/batch: {current.get('unit', 'none')} / {current.get('batch', 'pending')}. "
        f"Heartbeat: {status.get('heartbeat', status.get('updated_at', 'pending'))}.  \n"
        f"Charged active hours: {charged}; forecast remainder: {forecast}; "
        f"disk free: {'pending' if headroom is None else f'{headroom / 1e9:.2f} GB'}.\n\n"
        "Partial blocks remain resumable progress and are excluded from paired inference. "
        "No production results are implied by a pending report. " + _small_cohort_note(result))


def _small_cohort_note(result):
    count = result.get("cohorts", {}).get("fresh", {}).get("n_pairs", 0)
    if count < 4:
        return ("With fewer than four complete fresh pairs, displayed intervals are descriptive progress only; "
                "one replica per sign stratum gives a degenerate stratified bootstrap. ")
    return ""


def _readiness_text(snapshot):
    cfg = snapshot["cfg"]
    epochs = setting(cfg, "init_epochs", 45)
    if setting(cfg, "smoke", False):
        initialization = f"Smoke initialization uses {epochs} epoch(s), not the production fit budget."
    elif epochs == 45:
        initialization = ("The common initialization budget is 45 epochs, selecting the lowest nominal-validation "
            "loss at epochs 15, 30, or 45; the optional 60-epoch extension is not selected.")
    else:
        initialization = f"The configured common initialization budget is {epochs} epochs."
    chosen = []
    for initial in snapshot["readiness"].get("initializations", []):
        history = [row for row in initial.get("history", []) if row.get("validation_nll") is not None]
        if history:
            chosen.append(str(min(history, key=lambda row: row["validation_nll"])["epoch"]))
    history_text = ("Selected development checkpoint epochs: " + ", ".join(chosen) + "." if chosen else
                    "Development initialization histories are pending.")
    policy = snapshot.get("numerical_policy", {})
    if policy:
        numerical = (f"Numerical policy: FD arithmetic {_fmt(policy.get('fd_dtype'))}; "
            f"at most {_fmt(policy.get('max_iter'))} solver iterations / {_fmt(policy.get('max_eval'))} closure calls.")
    else:
        numerical = "Numerical-policy audit is pending."
    links = snapshot.get("artifact_hrefs", {})
    audit = links.get("readiness.json", links.get("numerics.json"))
    if audit:
        numerical += f" [Full readiness/numerical audit](<{audit}>)."
    text = "**Readiness:** " + initialization + " " + history_text + " " + numerical
    notes = snapshot.get("execution_notes", [])
    if notes:
        summaries = []
        for note in notes[-3:]:
            if isinstance(note, dict):
                note = next((note[k] for k in ("summary", "note", "description", "repair", "reason", "message")
                             if isinstance(note.get(k), str)), "Revision recorded; see the execution log.")
            summaries.append(" ".join(note.split())[:240])
        text += "\n\n**Execution notes:** " + "; ".join(summaries) + "."
        if len(notes) > 3:
            text += f" {len(notes) - 3} earlier entries remain in the full log."
        if links.get("execution_notes.json"):
            text += f" [Full append-only log](<{links['execution_notes.json']}>)."
    return text


def _settings(snapshot, result):
    rows = []
    B = [r for r in result["development"] if r["recipe"] == result["selection"].get("B") and r.get("Havg") is not None]
    for arm in ARMS:
        recipe = result["selection"].get(arm, {})
        dev = [r for r in result["development"] if r["recipe"] == recipe and r.get("Havg") is not None]
        matched = "pending"
        if len(B) == len(dev) == 2:
            mismatch = max(abs(np.mean([r[k] for r in dev]) - np.mean([r[k] for r in B])) for k in ("Havg", "H100"))
            matched = "reference" if arm == "B" else "matched" if mismatch <= setting(snapshot["cfg"], "retention_tolerance", .1) else "unmatched"
        rows.append([arm + " " + LABELS[arm], "3268 / 108", recipe.get("ridge"), recipe.get("strength"), recipe.get("process_scale"), matched])
    return ("## Frozen Settings\n\n" + _table(("Arm", "LTM / STM coordinates", "Ridge", "FD strength", "KF scale", "Development match"), rows)
        + "\nRidge is the dimensionless ratio $c$ in $\\kappa=c\\,\\mathrm{mean}(d)$. "
        + f"Every arm has the same {setting(snapshot.get('cfg', {}), 'capacity', 32)}-grid replay buffer "
          f"and {setting(snapshot.get('cfg', {}), 'replay_count', 2)} replay draws per arrival; "
          "A omits only the FD penalty, transport and adaptive weight.\n")


def _conclusions(result):
    effects = result["comparisons"]
    rows = [[name, _effect(v["novel_auc"]), _effect(v["Havg"]), _effect(v["H100"]), _effect(v["current_whole_rms"])]
            for name, v in effects.items()]
    if not rows:
        rows = [["Pending complete fresh paired blocks", "pending", "pending", "pending", "pending"]]
    text = "## Fresh Paired Effects\n\n" + _table(("Contrast (first minus second)", "Novel risk (px squared)", "Havg (px)", "H100 (px)", "Current RMS (px)"), rows)
    text += ("\nNovel risk is test MSE averaged over novel-sample budgets 1 through 100. "
        "$H_{\\mathrm{avg}}$ is $\\sqrt{\\mathrm{mean}(\\mathrm{old\\ MSE})}$ minus initial old RMS; "
        "$H_{100}$ is old RMS at 100 novel grids minus initial old RMS. "
        "These deteriorations are calculated within each replica before pairing. ")
    text += "Negative effects favor the first arm; interaction is E-C-D+B. Intervals are nominal 95%, not familywise. "
    if "B-A" in effects:
        text += ("B-A compares the STM FD penalty against Replay/EWC; this supplemental row was added "
                 "after the run using the same paired bootstrap draws. ")
    text += _small_cohort_note(result)
    text += "Stationary replicas remain descriptive paired diagnostics, with no novel-sample S.\n\n## Sample Equivalence\n\n"
    rows = []
    missing = "not estimable" if result.get("status") == "completed" else None
    for name, item in result["efficiency"].items():
        p = item["point"]
        inverse = item.get("ci") or {}
        interval = lambda key: _effect(dict(mean=p.get(key), ci=inverse.get(key)))
        qualification = ("weak target; " if p.get("weak_target") else "") + (
            "retention supported" if item["retention_supported"] else "retention descriptive")
        if p["status"] != "valid":
            qualification += "; " + p["status"].replace("_", " ")
        display_interval = lambda key: interval(key) if p.get(key) is not None else missing
        rows.append([name + (" secondary target" if name == "E-B" else " common A target"),
            p.get("n") if p.get("n") is not None else missing,
            display_interval("saved_percent"), p.get("equal100_time_ratio"),
            p.get("matched_time_ratio") if p.get("matched_time_ratio") is not None else missing,
            display_interval("S"), _effect(item["full_budget_old_gap"]),
            item.get("matched_old_gap_upper_one_sided_95") if item.get("matched_old_gap_upper_one_sided_95") is not None else missing,
            qualification, item["bootstrap_failure_frequency"]])
    if not rows:
        rows = [["Pending", *([None] * 7), "pending", None]]
    text += _table(("Arm-reference", "Novel n", "Saved %", "Equal-100 compute", "Matched compute", "S", "Full old gap (px)", "Matched old upper (px)", "Retention", "Failed inverse draws"), rows)
    text += ("\nS is the saved novel-sample fraction divided by matched-quality compute ratio. "
        "A has zero saving by definition; E-B uses a separate target. A target already met at frozen initialization "
        "or before novel exposure is rejected. Reference gains below 5% are weak targets. Any unsupported bootstrap "
        "crossing withholds inverse intervals, not full-budget effects. Retention support requires the one-sided "
        "95% old-gap upper bounds at both budgets below .10 pixels.\n\n## Conclusion\n\n")
    if not effects:
        text += "Pending: no complete fresh paired block supports a statement about gains beyond B or A, retention, or observer transfer."
    else:
        def direction(name):
            value = effects[name]["novel_auc"]
            if value["mean"] is None:
                return "an unresolved novel-risk comparison"
            return "lower observed novel risk" if value["mean"] is not None and value["mean"] < 0 else "no observed novel-risk improvement"
        text += (f"The combined policy shows {direction('E-B')} beyond the STM incumbent and {direction('E-A')} "
                 "beyond Replay/EWC in completed blocks. These are calibrated-policy comparisons; retention and cost "
                 "qualifications are shown above. Observer usefulness must be assessed from D-B and E-C behavior, "
                 "not noise proxies alone. ")
        text += ("The frozen allocation is complete; retention and compute qualifications still apply."
                 if result.get("status") == "completed" else
                 "The frozen allocation must finish before closeout claims.")
        if result.get("status") == "completed":
            text += (" At 100 novel grids, the combined arm's old-error difference versus B is "
                     f"{_effect(effects['E-B']['H100'])} pixels: assess retention separately from new-data efficiency.")
            incumbent = result["efficiency"].get("B-A")
            if incumbent and incumbent["point"]["status"] == "valid":
                point = incumbent["point"]
                text += (f"\n\nSTM-only B has an observed {_fmt(point['saved_percent'])}% sample saving at "
                         f"{_fmt(point['matched_time_ratio'])} times matched-quality compute "
                         f"(S={_fmt(point['S'])}).")
                if incumbent["bootstrap_failure_frequency"]:
                    text += (f" {_fmt(100 * incumbent['bootstrap_failure_frequency'])}% of bootstrap draws lack a supported "
                             "crossing, so ordinary sample-saving intervals are withheld.")
                if not incumbent["retention_supported"]:
                    text += " Fresh retention support is not established; this is an exploratory point estimate, not a demonstrated efficiency gain."
    text += " Replay-aware error correlation and exact optimality of the combined learner remain unproved."
    return text


def _notebook(snapshot, result, figures):
    cells = [nbformat.v4.new_markdown_cell(_progress(snapshot, result)),
        nbformat.v4.new_markdown_cell("## Question\n\nDo age-weighted STM sensitivity, paired Fisher transport, and predictable "
            "velocity-Kalman adaptation remain useful together after development-only retention calibration? "
            "The likelihood is fixed Gaussian; each outcome is independent-test squared error per 2D landmark, "
            "RMS pixels, or fraction within one pixel. Initialization, tuning, evaluation, and study overhead "
            "are separate from standalone learning compute. [Execution design](plan18.md).")]
    captions = [
        ("Task And Readiness", "The declared production stream has 152 expected novel grids; the analysis budget is 100. "
         "Incoming grids, novel grids, and nine constituent digit pairs per grid are distinct counts. Audit images are exported artifacts only."),
        ("Development", "All attempted development recipes remain visible, including unselected outcomes; outlines identify frozen recipes. "
         "Replica-1 screening and two-replica finalists are tuning evidence, not confirmation."),
        ("Fresh Prediction", "Novel and old curves use actual novel-count checkpoints; current-law curves use all incoming grids. "
         "Frozen fixed-panel references are horizontal; the current-law frozen reference varies with the law. "
         "Bands are pointwise nominal paired-bootstrap 95% intervals. Unsupported curve support remains missing."),
        ("Mechanism And Health", r"Raw/applied $\pi$, movement/noise, and pre-projection diagnostics are proxies, "
         "not proof of information transfer. `noise_scale` is the one-observation noise term; batch noise uses "
         f"`noise_scale / {setting(snapshot['cfg'], 'samples_per_step', 4)}`. "
         "The movement term adds `motion_energy` and `uncertainty_energy`. "
         "Eviction diagnostics are in-sample, not independent-test retention."),
        ("Efficiency And Stationarity", "Fewer novel grids to the same independent-test error, at the reported retention "
         "and compute cost, under this stream. Risks and cumulative compute are interpolated only between observed "
         "post-batch states; production resolution is at most four novel grids, never extrapolation.")]
    for index, ((title, caption), (_, png)) in enumerate(zip(captions, figures)):
        if index == 2:
            cells.append(nbformat.v4.new_markdown_cell(_settings(snapshot, result)))
        name = f"figure-{index + 1}.png"
        cell = nbformat.v4.new_markdown_cell(f"## {title}\n\n![{title}](attachment:{name})\n\n{caption}")
        cell["attachments"] = {name: {"image/png": base64.b64encode(png).decode("ascii")}}
        cells.append(cell)
        if index == 0:
            cells.append(nbformat.v4.new_markdown_cell(_readiness_text(snapshot)))
    cells.append(nbformat.v4.new_markdown_cell(_conclusions(result)))
    book = nbformat.v4.new_notebook(cells=cells, metadata={"plan18_report": {
        "schema": REPORT_SCHEMA, "protocol": PROTOCOL, "source_digest": snapshot["source_digest"]}})
    nbformat.validate(book)
    return nbformat.writes(book).encode()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def _lease(directory, check):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".publish.lock").open("a+b") as handle:
        # Do not wait behind another report with a noninterruptible flock.
        check()
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _verify_generation(path, digest, check):
    marker = json.loads((path / "COMPLETED").read_bytes())
    if marker.get("source_digest") != digest or marker.get("schema") != REPORT_SCHEMA:
        raise ValueError("incompatible immutable report generation")
    files = {p.relative_to(path).as_posix(): p for p in path.rglob("*") if p.is_file() and p.name != "COMPLETED"}
    if set(files) != set(marker["files"]):
        raise ValueError("immutable report file set changed")
    for name, file in files.items():
        check()
        if file.is_symlink() or _hash(file.read_bytes()) != marker["files"][name]:
            raise ValueError("immutable report generation changed")
    return json.loads((path / "generation.json").read_bytes())


def build_report(root, cfg=None, notebook_path=None, check=lambda: None, force=False):
    """Publish changed scalar inputs, returning immutable generation metadata.

    Output is ``root/report/<generation>/`` plus an atomic ``latest.json`` and
    the public notebook (default ``mnist_experiment/plan18_results.ipynb``).
    No identity, scientific unit, scheduler status, or checkpoint is written.
    ``force`` creates another generation; it never mutates an existing one.
    """
    root = Path(root)
    notebook = DEFAULT_NOTEBOOK if notebook_path is None else Path(notebook_path)
    report = root / "report"
    check()
    with _lease(report, check):
        snapshot = _snapshot(root, cfg, check)
        snapshot["artifact_hrefs"] = {name: os.path.relpath(root / name, notebook.parent)
            for name in ("readiness.json", "numerics.json", "execution_notes.json")
            if name in snapshot["inputs"]["files"]}
        snapshot["inputs"]["artifact_hrefs"] = snapshot["artifact_hrefs"]
        snapshot["source_digest"] = _hash(_bytes(snapshot["inputs"]))
        digest = snapshot["source_digest"]
        generation = digest[:24] + ("-" + uuid.uuid4().hex[:10] if force else "")
        latest = report / "latest.json"
        if latest.exists() and not force:
            pointer = json.loads(latest.read_bytes())
            if pointer.get("source_digest") == digest:
                candidate = pointer.get("generation", "")
                if not candidate or Path(candidate).name != candidate:
                    raise ValueError("invalid report generation pointer")
                generation = candidate
        destination = report / generation
        if destination.exists():
            metadata = _verify_generation(destination, digest, check)
            data = (destination / "results.ipynb").read_bytes()
            check()
            if not notebook.exists() or notebook.read_bytes() != data:
                _atomic(notebook, data)
            pointer = dict(schema=REPORT_SCHEMA, generation=generation, source_digest=digest)
            if not (report / "latest.json").exists() or (report / "latest.json").read_bytes() != _bytes(pointer):
                _atomic(report / "latest.json", _bytes(pointer))
            return {**metadata, "changed": False}
        result = summarize(snapshot["records"], snapshot["cfg"], snapshot["selection"], check=check)
        _add_incumbent_contrast(snapshot, result)
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=report))
        try:
            files, figures = {}, []
            for name, render in (("task", _task), ("development", _development), ("prediction", _predictive),
                                 ("health", _health), ("efficiency", _efficiency)):
                check()
                figure = render(snapshot, result)
                for axis in figure.axes:
                    axis.title.set_fontsize(10)
                    axis.xaxis.label.set_fontsize(9)
                    axis.yaxis.label.set_fontsize(9)
                output = io.BytesIO()
                figure.savefig(output, format="png", dpi=110, metadata={"Software": "Plan18 artifact report"})
                figure.clear()
                data = output.getvalue()
                figures.append((name, data))
                files[f"figures/{name}.png"] = data
            data = _notebook(snapshot, result, figures)
            metadata = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, generation=generation,
                source_digest=digest, created_at=datetime.now(timezone.utc).isoformat(),
                status=result["status"], completed_pairs={c: r["n_pairs"] for c, r in result["cohorts"].items()},
                notebook="results.ipynb", figure_count=5)
            files.update({"results.ipynb": data, "analysis.json": _bytes(result),
                          "inputs.json": _bytes(snapshot["inputs"]), "generation.json": _bytes(metadata)})
            if sum(map(len, files.values())) > MAX_GENERATION_BYTES:
                raise ValueError("bounded report generation allocation exceeded")
            for name, content in files.items():
                check()
                _atomic(temporary / name, content)
            marker = dict(schema=REPORT_SCHEMA, source_digest=digest,
                          files={name: _hash(content) for name, content in files.items()})
            _atomic(temporary / "COMPLETED", _bytes(marker))
            check()
            os.rename(temporary, destination)
            _sync_directory(report)
            _atomic(notebook, data)
            _atomic(report / "latest.json", _bytes(dict(schema=REPORT_SCHEMA, generation=generation, source_digest=digest)))
            return {**metadata, "changed": True}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
