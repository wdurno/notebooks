"""Artifact-only Phase 5E appendix to a pinned completed primary report.

Only compact scalar snapshots, sealed report generations, and reporting source
files are read. No learner, data, config, torch, or family-lock imports belong here.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
import numpy as np

from mnist_experiment.diagonal_lfu.report import (
    _atomic, _curve, _effect, _figure, _finish, _hash, _json_bytes,
    _load, _number, _table, _text,
)
from mnist_experiment.kalman_pi.report import _finite_json
from mnist_experiment.kalman_pi.retention import report as primary_report

from . import analysis

KIND = "plan17-zero-ridge-snapshot"
REPORT_KIND = "plan17-zero-ridge-report"
MARKER = "plan17_zero_ridge_report"
ARMS = ("fixed_ridge", "fixed_zero", "kalman_zero")
NAMES = {"fixed_ridge": "Fixed / r=.1", "fixed_zero": "Fixed / r=0", "kalman_zero": "Kalman / r=0"}
COLORS = {"fixed_ridge": "#3976a4", "fixed_zero": "#278574", "kalman_zero": "#b34c65"}
FIGURES = (("10_zero_ridge_effects", "Three Paired Contrasts"),
           ("11_zero_ridge_accuracy", "Fresh Accuracy By Length Cohort"))
CONTRASTS = (("kalman_zero", "fixed_ridge"), ("fixed_zero", "fixed_ridge"), ("kalman_zero", "fixed_zero"))


def _markdown(name, source):
    return nbformat.v4.new_markdown_cell(source, id="plan17-zero-ridge-" + name)


def _validated(root):
    snapshot, data = _load(root / "report_snapshot.json")
    _finite_json(snapshot)
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 1 or snapshot.get("kind") != KIND:
        raise ValueError("requires schema 1 kind plan17-zero-ridge-snapshot")
    for name in ("study_id", "status", "phase"):
        if not isinstance(snapshot.get(name), str) or not snapshot[name]:
            raise ValueError(f"snapshot.{name} must be a nonempty string")
    for name in ("config", "progress", "selection", "analysis"):
        if not isinstance(snapshot.get(name), dict):
            raise ValueError(f"snapshot.{name} must be an object")
    if not isinstance(snapshot.get("trajectories"), list):
        raise ValueError("snapshot.trajectories must be a list")
    if type(snapshot["config"].get("smoke", False)) is not bool:
        raise ValueError("config.smoke must be boolean")
    if snapshot["progress"].get("study_id", snapshot["study_id"]) != snapshot["study_id"]:
        raise ValueError("snapshot/progress study_id mismatch")
    seed = snapshot["selection"].get("bootstrap_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("selection.bootstrap_seed must be a persisted nonnegative integer")
    result = analysis.summarize(snapshot["trajectories"], snapshot["selection"],
                                bootstrap_draws=snapshot["config"].get("bootstrap_draws", 2000), seed=seed)
    if (type(result.get("schema")) is not int or result["schema"] != 1
            or result.get("kind") != "plan17-zero-ridge-analysis"):
        raise ValueError("incompatible supplemental analysis schema/kind")
    _finite_json(result)
    if snapshot["analysis"] and snapshot["analysis"] != result:
        raise ValueError("snapshot.analysis is corrupt or incompatible with trajectories/selection")
    return snapshot, result, data


def _comparisons(result):
    return result["comparisons"]


def _complete_blocks(result):
    return len(result["eligible_replicas"])


def _status(snapshot, result):
    complete = (snapshot["status"] in ("complete", "completed") and not result["counts"]["incomplete"]
                and _complete_blocks(result) == snapshot["selection"]["main_replicas"])
    return "Completed supplement" if complete else "Partial / incomplete supplement"


def _historical(snapshot):
    if snapshot["config"].get("smoke", False):
        if snapshot.get("historical_report_dir") is not None:
            raise ValueError("smoke must not use a production historical report")
        return None, None, {}
    source = snapshot.get("historical_report_dir")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("historical_report_dir must pin an absolute completed primary report generation")
    directory = Path(source).resolve()
    manifest = primary_report._verify_generation(directory)
    book, data = primary_report._read_book(directory / "report.ipynb")
    if snapshot.get("historical_notebook_sha256") != _hash(data):
        raise ValueError("pinned primary notebook hash mismatch")
    meta = book.metadata.get(primary_report.MARKER, {})
    sealed, sealed_bytes = _load(directory / "report_snapshot.json")
    if (manifest.get("kind") != "plan17-retention-report" or manifest.get("smoke") is not False
            or manifest.get("report_dir") != str(directory) or meta.get("schema") != 1
            or meta.get("smoke") is not False or meta.get("status") not in ("complete", "completed")
            or meta.get("generation") != directory.name or meta.get("study_id") != manifest.get("study_id")
            or meta.get("snapshot_sha256") != _hash(sealed_bytes) or manifest.get("snapshot_sha256") != _hash(sealed_bytes)
            or sealed.get("kind") != "plan17-retention-snapshot" or sealed.get("status") not in ("complete", "completed")
            or MARKER in book.metadata):
        raise ValueError("history must be a sealed completed primary retention report, not original, smoke, or supplemental")
    cells = {c.id: c for c in book.cells}
    required = {"plan17-conclusion", "plan17-retention-conclusion", "plan17-retention-progress",
                *("plan17-" + name for name, _ in primary_report.HISTORICAL_FIGURES),
                *("plan17-retention-" + name for name, _ in primary_report.FIGURES)}
    if not required <= cells.keys() or cells["plan17-retention-conclusion"].cell_type != "markdown":
        raise ValueError("primary report lacks original/Phase 5 evidence")
    # All inherited outputs stay embedded; no older renderer or checkpoint is used.
    for cell in book.cells:
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png is not None:
                try:
                    valid = base64.b64decode(png, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
                except (ValueError, TypeError) as exc:
                    raise ValueError("corrupt inherited image output") from exc
                if not valid:
                    raise ValueError("corrupt inherited image output")
    paths = [directory / "manifest.json", directory / "COMPLETED",
             *(directory / name for name in manifest["files_sha256"])]
    return book, directory, {str(p): _hash(p.read_bytes()) for p in paths}


def _progress(snapshot, result):
    p = snapshot["progress"]
    current = p.get("current") or {}
    target = snapshot["selection"]["main_replicas"]
    awaiting = len(result["awaiting_full_block_units"])
    checkpoint = p.get("last_checkpoint") or p.get("last_accepted_checkpoint") or current.get("last_checkpoint") or current.get("last_accepted_checkpoint")
    storage = p.get("storage") or {}
    used = storage.get("used_bytes")
    if used is not None:
        used += storage.get("external_bytes", 0) + storage.get("reserved_bytes", 0)
    cap = storage.get("limit_bytes", storage.get("max_bytes"))
    free = storage.get("free_bytes", storage.get("filesystem_free_bytes"))
    forecast = p.get("forecast_seconds")
    rows = [("Phase / status", snapshot["phase"] + " / " + _status(snapshot, result)),
            ("Changing three-arm blocks (complete / planned)", f"{_complete_blocks(result)} / {target}"),
            ("Trajectories (complete / planned); sealed prefixes", f"{result['counts']['complete']} / {p.get('planned_trajectories', 3 * target)}; {result['counts']['incomplete']} incomplete"),
            ("Completed arms awaiting full block", awaiting),
            ("Active unit / accepted checkpoint", f"{p.get('active_unit') or current.get('active_unit') or 'none'} / {checkpoint or 'pending'}"),
            ("Accepted step / total", f"{current.get('step', p.get('step', 'pending'))} / {current.get('total_steps', p.get('total_steps', 'pending'))}"),
            ("Heartbeat (snapshot freshness)", p.get("heartbeat", "pending")),
            ("Cumulative family active / allocated hours", f"{_number(p.get('active_hours', p.get('active_seconds', 0) / 3600))} / {_number(p.get('allocation_hours', p.get('allocation_seconds', 72000) / 3600))}"),
            ("Remaining forecast hours", _number(forecast / 3600) if forecast is not None else "pending"),
            ("Storage used / ceiling / free (GB)", " / ".join(_number(v / 1e9) if v is not None else "pending" for v in (used, cap, free)))]
    return _markdown("progress", "## Current Phase 5E Progress\n\n" + _table(("Item", "Recorded state"), rows) +
                     "\n\nSealed-artifact counts, not a liveness check. The preserved progress panels below are historical Phases 0--4 and primary Phase 5 snapshots; "
                     "this panel is the current supplement. Partial three-arm blocks never contribute paired evidence.")


def _effects(snapshot, result, destination):
    comparisons = { (c["condition"], c["control"]): c for c in _comparisons(result) }
    with _figure(rows=1, cols=2) as (fig, axes):
        for ax, metric, title in zip(axes.flat, ("nine_recall", "old_accuracy"), ("9 acquisition", "0--8 retention")):
            ax.axvline(0, color="#888888", linewidth=.8)
            for i, key in enumerate(CONTRASTS):
                c = comparisons.get(key)
                if c is None or not c["n_pairs"]:
                    continue
                estimate = c["observation_mean_pp"][metric]
                color = COLORS[key[0]]
                if estimate.get("ci95") is not None:
                    ax.plot(estimate["ci95"], [i, i], color=color)
                ax.plot([estimate["mean"]], [i], "o", color=color)
            ax.set_yticks(range(3), [NAMES[a] + " - " + NAMES[b] for a, b in CONTRASTS], fontsize=7)
            ax.set(title=title, xlabel="Whole-path paired accuracy difference (pp)")
            if not _complete_blocks(result):
                ax.text(.5, .85, "Pending complete fresh three-arm blocks", ha="center", transform=ax.transAxes)
        _finish(axes)
        fig.savefig(destination, dpi=135)


def _accuracy(snapshot, result, destination):
    by_id = {t["unit_id"]: t for t in snapshot["trajectories"]}
    arms = defaultdict(set)
    for c in _comparisons(result):
        if c["n_pairs"]:
            arms[c["control"]].update(c["control_units"])
            arms[c["condition"]].update(c["treatment_units"])
    with _figure(rows=1, cols=2) as (fig, axes):
        exposure = {}
        for arm in ARMS:
            cohorts = defaultdict(list)
            for unit in sorted(arms[arm]):
                t = by_id[unit]
                cohorts[(t.get("schedule_factor", 1), tuple(r["arrivals"] for r in t["metrics"]))].append(t)
            for (factor, grid), cohort in sorted(cohorts.items()):
                label = f"{NAMES[arm]}; L={factor:g}, N={grid[-1]:g}"
                for ax, metric in zip(axes.flat, ("nine_recall", "old_accuracy")):
                    _curve(ax, cohort, metric, label, color=COLORS[arm], linestyle="-" if factor == 1 else "--")
                    if arm == "fixed_ridge":
                        baseline = 100 * float(np.mean([t["metrics"][0][metric] for t in cohort]))
                        ax.plot([0, grid[-1]], [baseline, baseline], ":", color="#666666", label=f"Initial; L={factor:g}, N={grid[-1]:g}")
                if arm == "fixed_ridge":
                    means = []
                    for name in ("expected_nines", "observed_nines"):
                        values = [t["metrics"][-1].get(name) for t in cohort]
                        means.append(_number(float(np.mean(values)), 1) if all(v is not None for v in values) else "pending")
                    exposure[(factor, grid)] = f"L={factor:g}, N={grid[-1]:g}: online 9 exp/obs=" + "/".join(means)
        xmax = max((line.get_xdata()[-1] for ax in axes.flat for line in ax.lines), default=1)
        for ax, title in zip(axes.flat, ("Changing: 9 recall", "Changing: 0--8 macroaccuracy")):
            ax.set(xlabel="Online digits observed", ylabel="Accuracy (%)", xlim=(0, xmax), ylim=(0, 100))
            ax.set_title(title, pad=30)
            ax.text(0, 1.02, "\n".join(exposure.values()), fontsize=6, va="bottom", transform=ax.transAxes)
        _finish(axes)
        if axes[0, 0].get_legend() is not None:
            axes[0, 0].get_legend().remove()
        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            axes[0, 1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, .5), fontsize=6)
        fig.savefig(destination, dpi=135)


def _results(result):
    by_key = {(c["condition"], c["control"]): c for c in _comparisons(result)}
    rows = []
    for key in CONTRASTS:
        c = by_key.get(key)
        metrics = c["observation_mean_pp"] if c else {}
        costs = c.get("costs", {}) if c else {}
        rows.append((NAMES[key[0]] + " - " + NAMES[key[1]], c["n_pairs"] if c else 0,
                     _effect(metrics.get("nine_recall"), signed=True), _effect(metrics.get("old_accuracy"), signed=True),
                     _effect(costs.get("learning_seconds", {}).get("ratio")),
                     _effect(costs.get("wall_seconds", {}).get("ratio"))))
    return _markdown("results", "### Supplemental Paired Results\n\n" +
                     _table(("Contrast", "Complete pairs", "9 gap (pp), 95% CI", "Old gap (pp), 95% CI", "Learning time ratio, 95% CI", "Online wall ratio, 95% CI"), rows) +
                     "\n\nOnly complete fresh three-arm blocks contribute. Means use trapezoidal averaging on each replica's actual incoming-digit clock, including its shared initial checkpoint, "
                     "then equal weight per full replica. A missing interval or cost is pending, not zero. Learning time excludes evaluation/checkpoint I/O; online wall time includes it. "
                     "Initialization and shared observer calibration are separate. Ratios are means of paired path ratios, not ratios of group means; full costs remain linked below.")


def _interpretation(result):
    lines = []
    for c in _comparisons(result):
        if not c["n_pairs"]:
            continue
        nine, old = [c["observation_mean_pp"][m] for m in ("nine_recall", "old_accuracy")]
        if all(v.get("ci95") is not None and v["ci95"][0] > 0 for v in (nine, old)):
            reading = "both accuracy intervals favor the first arm"
        elif old.get("ci95") is not None and old["ci95"][0] > 0 and nine.get("ci95") is not None and nine["ci95"][0] <= 0 <= nine["ci95"][1]:
            reading = "the old-accuracy interval favors the first arm; a 9-recall gain is not established, and equivalence is not established"
        elif nine["mean"] > 0 and old["mean"] > 0:
            reading = "both point estimates favor the first arm; the joint gain remains uncertain"
        elif nine["mean"] * old["mean"] < 0:
            reading = "an acquisition/retention trade-off remains"
        else:
            reading = "the development lead does not replicate as a joint gain in these estimates"
        lines.append(f"{NAMES[c['condition']]} - {NAMES[c['control']]}: {reading}.")
        contrary = []
        for segment, values in c.get("segments", {}).items():
            for metric, estimate in values.get("observation_mean_pp", {}).items():
                whole = c["observation_mean_pp"].get(metric, {}).get("mean")
                if whole is not None and estimate.get("mean") is not None and whole * estimate["mean"] < 0:
                    contrary.append(f"{segment}: {metric} {_effect(estimate, signed=True)} pp")
        if contrary:
            lines.append("Contrary regime effects: " + "; ".join(contrary) + ".")
    return "\n\n".join(lines) if lines else "Pending: no complete fresh three-arm block; no replication conclusion is available."


def _section(snapshot, result, destination, notebook):
    smoke = snapshot["config"].get("smoke", False)
    selection = snapshot["selection"]
    banner = "**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.**" if smoke else "**" + _status(snapshot, result) + ".**"
    cells = [_markdown("section", "## Phase 5E: Zero-Ridge Follow-Up\n\n" + banner +
                       "\n\nZero-ridge development points improved acquisition and retention relative to fixed pi=.025, r=.1, but changed both policy and ridge. "
                       "This independent supplement asks whether that lead replicates, and whether removing ridge or changing policy accounts for it. "
                       "It does not change primary selection, matching tolerance, or primary confirmation.\n\n"
                       "Three frozen arms: fixed pi=.025/r=.1; fixed pi=.025/r=0; Kalman eta=.0001/r=0. "
                       "Zero ridge removes only the extra isotropic term, not diagonal EWC. "
                       f"{selection['main_replicas']} new changing replicas ({selection['fresh_trajectory_target']} paths), with predrawn length factors "
                       + " and ".join(f"{factor:g}" for factor in selection["schedule_factors"]) + ", "
                       "use separate named initialization/stream/holdout draws within the same data-role partitions; no development or primary-confirmation pooling. "
                       f"Persisted bootstrap seed: `{snapshot['selection']['bootstrap_seed']}`; draws: {result['bootstrap']['draws']}. "
                       "Shared whole-replica resampling across all three contrasts and both outcomes gives nominal 95% intervals; checkpoints are not independent replicas. "
                       "No retention matching or further selection is performed here.")]
    captions = ("Positive differences favor the first arm. These are both full-path accuracy outcomes, not a combined score or an equivalence test.",
                "Complete three-arm blocks only; length/evaluation-grid cohorts remain distinct. Initial references are context; online 9 exposure excludes initialization. "
                "Bands are descriptive pointwise mean +/-1.96 SE, not paired-effect intervals; single-replica cohorts have no band.")
    for (name, title), caption in zip(FIGURES, captions):
        cells.append(_markdown(name + "-heading", "### " + title + "\n\n" + caption))
        relative = os.path.relpath(destination / (name + ".png"), notebook.parent)
        cells.append(nbformat.v4.new_code_cell("from IPython.display import Image, display\n" + f"display(Image(filename={relative!r}))",
                    id="plan17-zero-ridge-" + name, execution_count=len(cells), outputs=[]))
        if name == "10_zero_ridge_effects":
            cells.append(_results(result))
    reading = "Smoke acceptance only; these tiny artifacts support no production efficacy or runtime conclusion." if smoke else _interpretation(result)
    by_key = {(c["condition"], c["control"]): c for c in _comparisons(result)}
    fixed = by_key.get(("fixed_zero", "fixed_ridge"))
    kalman = by_key.get(("kalman_zero", "fixed_zero"))
    if not smoke and fixed and kalman and fixed["n_pairs"] and kalman["n_pairs"]:
        if all(v.get("ci95") is not None and v["ci95"][0] > 0 for v in fixed["observation_mean_pp"].values()):
            reading += ("\n\n**What the controls establish:** removing the extra isotropic ridge already improves both accuracies under fixed pi. "
                        "The acquisition lift against the original reference therefore cannot, by itself, be credited to adaptive pi. "
                        "The common-zero-ridge contrast is the separate evidence about the policy package.")
    cells.append(_markdown("conclusion", "### Phase 5E Interpretation\n\n" + reading +
                           "\n\nFixed/0 minus fixed/.1 isolates the ridge contrast under the fixed policy. Kalman/0 minus fixed/0 compares policy packages at zero ridge; "
                           "Kalman/0 minus fixed/.1 combines both changes. This three-cell design does not estimate a full policy-by-ridge interaction, an optimal fixed pi, "
                           "an isolated observer effect, or actual samples saved. Any acquisition/retention trade-off remains visible. "
                           "Non-significance is not equality; exploratory intervals follow development-led selection, multiple contrasts and repeated MNIST use. "
                           "Fresh outcomes never select later settings. The cumulative 20-hour family cap is unchanged; shared initialization/calibration is not double-charged. "
                           "Repeated checkpoint I/O remains a tiny-model timing limitation."))
    cells.append(_markdown("artifacts", "[Sealed supplemental snapshot](" + os.path.relpath(destination / "report_snapshot.json", notebook.parent) +
                           ") | [All regime diagnostics, measured costs and paired analysis](" + os.path.relpath(destination / "analysis.json", notebook.parent) +
                           "). All PNG outputs are embedded. Opening this report never trains, downloads, runs inference, repairs artifacts or acquires the experiment family lock."))
    return cells


def _sources():
    here = Path(__file__).resolve()
    kalman = here.parents[2]
    files = (here, Path(analysis.__file__).resolve(), here.with_name("config.py"),
             kalman / "config.py", kalman / "retention/config.py",
             kalman.parent / "diagonal_lfu/config.py", here.parents[4] / "src/seeding.py")
    return {**primary_report._source_hashes(), **{str(p): _hash(p.read_bytes()) for p in files}}


def _generation(identity):
    return "generation-" + _hash(_json_bytes(identity))[:24]


def _verify_generation(directory, expected=None):
    manifest, data = _load(directory / "manifest.json")
    if manifest.get("schema") != 1 or manifest.get("kind") != REPORT_KIND or expected is not None and manifest != expected:
        raise ValueError("supplement report generation identity collision")
    if (directory / "COMPLETED").read_text().strip() != _hash(data):
        raise ValueError("corrupt supplement report completion seal")
    files = manifest.get("files_sha256", {})
    required = {"report.ipynb", "report_snapshot.json", "analysis.json", *(n + ".png" for n, _ in FIGURES)}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("supplement report generation missing artifact hashes")
    for name, digest in files.items():
        if Path(name).name != name or _hash((directory / name).read_bytes()) != digest:
            raise ValueError("corrupt supplement report artifact: " + name)
    return manifest


def _previous(root, snapshot, history):
    latest = root / "report/latest.json"
    if not latest.exists():
        return None
    manifest, _ = _load(latest)
    directory = Path(manifest.get("report_dir", "")).resolve()
    if directory.parent != root / "report":
        raise ValueError("latest generation escapes supplemental root")
    _verify_generation(directory, expected=manifest)
    if manifest.get("root") != str(root) or manifest.get("study_id") != snapshot["study_id"] or manifest.get("smoke") != snapshot["config"].get("smoke", False):
        raise ValueError("previous report ownership mismatch")
    old, _ = _load(directory / "report_snapshot.json")
    if old["selection"] != snapshot["selection"]:
        raise ValueError("frozen supplemental selection changed")
    if manifest.get("historical_sha256") != history:
        raise ValueError("pinned primary report changed")
    old_units = {t["unit_id"]: t for t in old["trajectories"] if t["completed"]}
    new_units = {t["unit_id"]: t for t in snapshot["trajectories"] if t["completed"]}
    if any(new_units.get(unit) != t for unit, t in old_units.items()):
        raise ValueError("snapshot regresses or changes completed supplemental evidence")
    return manifest


def _target_bytes(notebook, root, snapshot, historical, previous):
    if not notebook.exists():
        return None
    book, data = primary_report._read_book(notebook)
    if book.metadata.get("plan17_registration_report"):
        raise ValueError("target includes Phase 6; publish with the registration reporter or use a separate notebook")
    marker = book.metadata.get(MARKER)
    if marker and (marker.get("root") != str(root) or marker.get("study_id") != snapshot["study_id"]):
        raise ValueError("target report belongs to another supplement owner")
    candidates = []
    if historical is not None:
        candidates.append((historical / "report.ipynb").read_bytes())
    if previous is not None:
        candidates.append((Path(previous["report_dir"]) / "report.ipynb").read_bytes())
    if data not in candidates:
        raise ValueError("refusing to replace newer, modified or unrelated notebook content")
    return data


def build_report(root: Path, notebook: Path | None = None) -> dict:
    """Publish a source-hashed immutable appendix without touching primary state."""
    root = Path(root).resolve()
    snapshot, result, snapshot_bytes = _validated(root)
    smoke = snapshot["config"].get("smoke", False)
    notebook = Path(notebook).resolve() if notebook is not None else root / "supplement_smoke.ipynb" if smoke else Path(__file__).resolve().parents[3] / "plan17_results.ipynb"
    if notebook.suffix != ".ipynb" or notebook.is_relative_to(root / "report"):
        raise ValueError("output must be an .ipynb outside immutable report generations")
    if smoke and not notebook.is_relative_to(root):
        raise ValueError("smoke notebooks must remain inside their supplemental root")
    original, historical, history = _historical(snapshot)
    if historical is not None:
        primary_root = historical.parent.parent
        writes_primary = notebook.is_relative_to(primary_root) or root.is_relative_to(primary_root)
        if writes_primary:
            raise ValueError("supplement must not write inside the primary root")
    previous = _previous(root, snapshot, history)
    expected_target = _target_bytes(notebook, root, snapshot, historical, previous)
    identity = {"root": str(root), "study_id": snapshot["study_id"], "smoke": smoke,
                "notebook": str(notebook), "snapshot_sha256": _hash(snapshot_bytes),
                "sources_sha256": _sources(), "historical_sha256": history}
    destination = root / "report" / _generation(identity)
    staging = None
    try:
        if destination.exists():
            manifest = _verify_generation(destination)
            if any(manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("supplement report generation source/identity collision")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".building-", dir=destination.parent))
            for (name, _), renderer in zip(FIGURES, (_effects, _accuracy)):
                renderer(snapshot, result, staging / (name + ".png"))
            section = _section(snapshot, result, destination, notebook)
            for cell in section:
                if cell.cell_type == "code":
                    png = (staging / (cell.id.removeprefix("plan17-zero-ridge-") + ".png")).read_bytes()
                    cell.outputs = [nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode(), "text/plain": cell.id})]
            if smoke:
                book = nbformat.v4.new_notebook(cells=[_markdown("title", "# Phase 5E Smoke Acceptance Only\n\n**SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE.**")])
            else:
                book = deepcopy(original)
            book.cells.insert(1, _progress(snapshot, result))
            book.cells.extend(section)
            book.metadata[MARKER] = {"schema": 1, "root": str(root), "study_id": snapshot["study_id"], "smoke": smoke,
                                     "generation": destination.name, "snapshot_sha256": identity["snapshot_sha256"],
                                     "status": _status(snapshot, result), "historical_report_dir": str(historical) if historical else None}
            nbformat.validate(book)
            _atomic(staging / "report.ipynb", nbformat.writes(book).encode())
            _atomic(staging / "report_snapshot.json", snapshot_bytes)
            _atomic(staging / "analysis.json", _json_bytes(result))
            manifest = {"schema": 1, "kind": REPORT_KIND, **identity, "report_dir": str(destination),
                        "manifest": str(root / "report/latest.json"), "status": _status(snapshot, result), "counts": result["counts"],
                        "files_sha256": {p.name: _hash(p.read_bytes()) for p in sorted(staging.iterdir())}}
            _atomic(staging / "manifest.json", _json_bytes(manifest))
            _atomic(staging / "COMPLETED", (_hash(_json_bytes(manifest)) + "\n").encode())
            for path in staging.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
                _verify_generation(destination, expected=manifest)
        _verify_generation(destination, expected=manifest)
        # Recheck live pointers after rendering so stale work cannot replace newer content.
        if (root / "report_snapshot.json").read_bytes() != snapshot_bytes or _target_bytes(notebook, root, snapshot, historical, previous) != expected_target:
            raise ValueError("snapshot or target changed during report generation")
        primary_report._publish(notebook, root / "report/latest.json", (destination / "report.ipynb").read_bytes(), manifest)
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish artifact-only Phase 5E zero-ridge supplement")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.notebook), sort_keys=True))


if __name__ == "__main__":
    main()
