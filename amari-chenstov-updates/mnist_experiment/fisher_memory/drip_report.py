"""Compact, artifact-only Phase 5B notebook section."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import nbformat

from .artifacts import ArtifactError, iter_units
from .report import _plot_cell, _table, order_notebook_chronologically


TAG = "plan15_phase5b"
PROTOCOL = "plan15-drip-development-v1"


def _tag(cell):
    cell.metadata["tags"] = [TAG]
    return cell


def _read(root: Path):
    units = []
    for record in iter_units(root):
        metadata = record["metadata"]
        if metadata.get("kind") != "drip_trajectory":
            continue
        if metadata.get("protocol") != PROTOCOL:
            raise ArtifactError("incompatible drip trajectory protocol")
        path = record["path"]
        if record["status"] == "completed":
            rows = json.loads((path / "metrics.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
        else:
            progress = path / "progress.json"
            partial = json.loads(progress.read_text()) if progress.exists() else None
            if partial is not None and partial.get("schema_version") != 1:
                raise ArtifactError("incompatible partial drip metrics")
            rows = partial["rows"] if partial is not None else []
            summary = None
        if any("paired_rotated_event" not in row or "rotated_nll" not in row
               or "base_nll" not in row for row in rows):
            raise ArtifactError("incompatible drip metric row")
        units.append(dict(metadata=metadata, rows=rows, summary=summary,
                          completed=record["status"] == "completed"))
    status_path = root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return units, status


def _mean_curves(units, metric, selected=None):
    grouped = defaultdict(lambda: defaultdict(list))
    for unit in units:
        meta = unit["metadata"]
        if (selected is not None and meta["condition"] == "finite_difference"
                and meta["treatment"]["strength"] != selected):
            continue
        label = f"{meta['stream']} / {meta['condition']}"
        if meta["condition"] == "finite_difference":
            label += f" ({meta['treatment']['strength']:.4g})"
        for row in unit["rows"]:
            grouped[label][row["paired_rotated_event"]].append(row[metric])
    return {label: (sorted(points), [sum(points[n]) / len(points[n]) for n in sorted(points)])
            for label, points in grouped.items()}


def _plot(units, metric, title, selected=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    curves = _mean_curves(units, metric, selected)
    if not curves:
        return None
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for label, (x, y) in sorted(curves.items()):
        axis.plot(x, y, label=label, linewidth=1.7)
    axis.set(title=title, xlabel="Rotated images observed (paired event index for base-only)",
             ylabel="NLL", xlim=(0, 100))
    axis.grid(alpha=.2)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    cell = _tag(_plot_cell(figure, title))
    plt.close(figure)
    return cell


def _pilot_readout(complete, pilot, ledger):
    selected = pilot.get("selected_strength")
    if selected is None:
        return None
    table = next(row for row in pilot["table"] if row["strength"] == selected)
    source = (
        "### Development readout\n\n"
        f"The sealed pilot selected $\\lambda_F={selected:.4g}$: mean mixed-stream "
        f"30-degree acquisition gain **{table['mean_gain']:+.3f} NLL-AUC**, "
        f"mean 0-degree difference **{table['mean_harm']:+.3f} NLL-AUC** "
        "(positive means retention harm). This coefficient was selected on these "
        "same four replicas, so these are exploratory estimates, not a significance test."
    )
    strengths = pilot.get("grid_with_boundary", [])
    if strengths and selected in (min(strengths), max(strengths)):
        source += (" The selection remains at the tested grid boundary after the one "
                   "predeclared extension; the optimum is not bracketed.")
    index = {(u["metadata"]["replica"], u["metadata"]["stream"],
              u["metadata"]["condition"], u["metadata"]["treatment"]["strength"]): u["summary"]
             for u in complete}
    replicas = range(ledger["screen_start"], ledger["screen_start"] + ledger["screen_replicas"])
    paired = []
    for replica in replicas:
        keys = [(replica, stream, condition, selected if condition == "finite_difference" else 0.)
                for stream in ("mixed", "base") for condition in ("replay_ewc", "finite_difference")]
        if all(key in index for key in keys):
            mixed_control, mixed_fd, base_control, base_fd = (index[key] for key in keys)
            paired.append(dict(replica=replica, mixed_gain=mixed_control["rotated_auc"] - mixed_fd["rotated_auc"],
                               base_gain=base_control["rotated_auc"] - base_fd["rotated_auc"],
                               values=[mixed_control, mixed_fd, base_control, base_fd]))
    if len(paired) < ledger["screen_replicas"]:
        return source + f"\n\nBase-only paired comparison: {len(paired)}/{ledger['screen_replicas']} complete."
    labels = ["Mixed replay/EWC", "Mixed FD KL", "Base-only replay/EWC", "Base-only FD KL"]
    rows = []
    for position, label in enumerate(labels):
        values = [pair["values"][position] for pair in paired]
        rows.append([label, f"{sum(v['rotated_auc'] for v in values) / len(values):.3f}",
                     f"{sum(v['base_auc'] for v in values) / len(values):.3f}",
                     f"{sum(v['active_seconds'] for v in values) / len(values) / 60:.1f}"])
    mixed = sum(pair["mixed_gain"] for pair in paired) / len(paired)
    base = sum(pair["base_gain"] for pair in paired) / len(paired)
    interaction = mixed - base
    favorable = sum(pair["mixed_gain"] > pair["base_gain"] for pair in paired)
    source += ("\n\n" + _table(["Condition", "30-degree NLL-AUC", "0-degree NLL-AUC",
                                "Online min"], rows)
               + f"\n\nFD gain in mixed stream: **{mixed:+.3f}**; FD gain with no rotated "
                 f"training images: **{base:+.3f}**. The mixed-specific difference is "
                 f"**{interaction:+.3f} NLL-AUC** ({favorable}/{len(paired)} replicas positive). "
                 "The substantial base-only gain means most FD-versus-replay lift is not "
                 "unique to rare-domain exposure. This contrast does not prove STM-to-LTM "
                 "information transfer. Phase 5C has not started.")
    return source


def section(root: Path):
    units, status = _read(root)
    complete = [unit for unit in units if unit["completed"]]
    current = status.get("current_job") or {}
    pilot = status.get("pilot") or {}
    ledger = json.loads((root / "units" / "drip-development-ledger-v1" / "result.json").read_text())
    expected = (2 * ledger["diagnostic_replicas"]
                + (1 + len(ledger["grid"])) * ledger["screen_replicas"]
                + 2 * ledger["screen_replicas"])
    boundary = {unit["metadata"]["treatment"]["strength"] for unit in units
                if unit["metadata"]["condition"] == "finite_difference"
                and unit["metadata"]["stream"] == "mixed"
                and unit["metadata"]["treatment"]["strength"] not in ledger["grid"]}
    expected += len(boundary) * ledger["screen_replicas"]
    selected = pilot.get("selected_strength")
    selected_label = f"{selected:.4g}" if selected is not None else "pending"
    progress = (
        "## Phase 5B: Rare rotated-domain drip\n\n"
        "**Development pilot only.** The primary acquisition score is the average "
        "30-degree test NLL after each of the first 100 rotated training images; lower is better. "
        "Base-only controls train on the same source images without rotation and use their "
        "partner's rotated-event index only as a reporting clock.\n\n"
        f"**Progress:** {len(complete)}/{expected} planned trajectories"
        f"; stage "
        f"`{status.get('stage', 'not started')}`; active compute "
        f"{status.get('active_seconds', 0) / 3600:.2f} h; "
        f"current {current or 'none'}.\n\n"
        f"**Pilot selection:** {pilot.get('classification', 'pending')}; "
        f"selected coefficient {selected_label}. "
        "Pilot outcomes do not enter the fresh Phase 5C significance test."
    )
    cells = [_tag(nbformat.v4.new_markdown_cell(progress))]
    readout = _pilot_readout(complete, pilot, ledger)
    if readout is not None:
        cells.append(_tag(nbformat.v4.new_markdown_cell(readout)))
    rows = []
    for unit in sorted(complete, key=lambda u: (
            u["metadata"]["replica"], u["metadata"]["stream"],
            u["metadata"]["treatment"]["strength"])):
        meta, summary = unit["metadata"], unit["summary"]
        rows.append([meta["replica"], meta["stream"], meta["condition"],
                     meta["treatment"]["strength"],
                     round(summary["rotated_auc"], 4), round(summary["base_auc"], 4),
                     round(summary["final_rotated_accuracy"], 3),
                     round(summary["final_base_accuracy"], 3),
                     round(summary["active_seconds"] / 60, 1)])
    cells.append(_tag(nbformat.v4.new_markdown_cell(
        "### Completed trajectories\n\n" + (
            _table(["Replica", "Stream", "Condition", "KL strength", "30-degree AUC",
                    "0-degree AUC", "30-degree acc", "0-degree acc", "Online min"], rows[-8:]) if rows else
            "No sealed trajectory yet; the current accepted checkpoint is shown in the curves below.")
        + ("\n\nLatest eight shown; complete metrics remain in immutable units." if len(rows) > 8 else ""))))
    for metric, title in (("rotated_nll", "30-degree acquisition"),
                          ("base_nll", "0-degree retention")):
        cell = _plot(units, metric, title, pilot.get("selected_strength"))
        if cell is not None:
            cells.append(cell)
    return cells


def build_notebook(base_notebook: Path, root: Path):
    notebook = nbformat.read(base_notebook, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if TAG not in cell.metadata.get("tags", [])]
    notebook.cells.extend(section(root))
    order_notebook_chronologically(notebook)
    nbformat.validate(notebook)
    return notebook
