"""Small artifact-only progress notebook for the Plan 15 method extension."""

from __future__ import annotations

import base64
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np

from .artifacts import ArtifactError, iter_units
from .report import MAX_NOTEBOOK_BYTES


LABELS = {"rotation_linear": "Linear rotation", "rotation_sigmoid": "Sigmoid rotation",
          "digit9_mixture": "Digit-9 mixture", "jvp": "JVP", "directional": "Directional KL"}


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def collect(root: Path | str):
    root = Path(root)
    records = list(iter_units(root, completed_only=False))
    ledgers = [r for r in records if r["unit_id"] == "extension-ledger-v1"]
    if len(ledgers) != 1 or ledgers[0]["status"] != "completed":
        raise ArtifactError("extension ledger is absent or incomplete")
    ledger = json.loads((ledgers[0]["path"] / "result.json").read_text())
    status = json.loads((root / "status.json").read_text())
    expected = {}
    for job in ledger["jobs"]:
        from .config import Treatment
        unit = (f"trajectory-confirmation-{job['environment']}-{job['replica']:04d}-"
                f"{Treatment(**job['treatment']).key}")
        expected[unit] = job
    observed = {}
    for record in records:
        if record["metadata"].get("kind") != "trajectory":
            continue
        unit = record["unit_id"]
        if unit not in expected:
            raise ArtifactError(f"unplanned extension trajectory: {unit}")
        if record["status"] != "completed":
            continue
        job = expected[unit]
        meta = record["metadata"]
        if (meta["environment"] != job["environment"] or meta["replica"] != job["replica"]
                or meta["config"] != job["treatment"]):
            raise ArtifactError(f"extension trajectory metadata mismatch: {unit}")
        summary = json.loads((record["path"] / "summary.json").read_text())
        if summary["unit_id"] != unit:
            raise ArtifactError(f"extension summary identity mismatch: {unit}")
        observed[unit] = summary
    return dict(ledger=ledger, status=status, expected=expected, observed=observed)


def build_notebook(root: Path | str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = collect(root)
    ledger, status, observed = data["ledger"], data["status"], data["observed"]
    scope = ledger["scope"]
    rows = []
    groups = defaultdict(list)
    for unit, summary in observed.items():
        job = data["expected"][unit]
        control = job["control"]
        groups[(job["environment"], job["method"])].append(dict(
            replica=job["replica"], current=control["current_auc"] - summary["current_auc"],
            historical=(control["historical_auc"] - summary["historical_auc"]
                        if control["historical_auc"] is not None and summary["historical_auc"] is not None
                        else None),
            seconds=summary["active_seconds"], nonfinite=summary["nonfinite"],
            solver_failures=summary["solver_failures"]))
    for environment in scope["environments"]:
        for method in scope["methods"]:
            sample = groups[(environment, method)]
            historical = [r["historical"] for r in sample if r["historical"] is not None]
            rows.append([LABELS[environment], LABELS[method], f"{len(sample)}/{scope['replicas']}",
                         f"{np.mean([r['current'] for r in sample]):+.4f}" if sample else "-",
                         f"{len(historical)}/{scope['replicas']}",
                         f"{np.mean(historical):+.4f}" if historical else "-",
                         f"{np.mean([r['seconds'] for r in sample]):.0f}" if sample else "-",
                         sum(r["nonfinite"] or r["solver_failures"] > 0 for r in sample)])
    cells = [nbformat.v4.new_markdown_cell(
        "# Plan 15: Curvature-method extension\n\n"
        f"**Progress:** {len(observed)}/{len(ledger['jobs'])} trajectories; "
        f"status **{status['status']}**; cumulative active time "
        f"**{status.get('active_seconds', 0) / 3600:.2f} h**. "
        f"Current job: {status.get('current_job') or 'none'}.\n\n"
        "This is a separately versioned, post-selection extension of the completed Plan 15 study. "
        "All contrasts use the same replica's sealed replay/EWC control from that study. "
        "The table is descriptive while trajectories accumulate; do not stop or extend based on a running p-value."),
        nbformat.v4.new_markdown_cell(
            "## Paired predictive outcomes\n\n"
            "Gain is replay/EWC minus method NLL AUC, so positive is better. "
            "Current measures new-data prediction; historical measures post-eviction holdout prediction. "
            "Each method uses the same 487 LTM and 176 STM trainable coordinates as its control.\n\n"
            + _table(["Environment", "Method", "Replicas", "Current gain", "Historical n", "Historical gain",
                      "Online s/replica", "Unhealthy"], rows))]
    if observed:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, layout="constrained")
        pairs = [(environment, method) for environment in scope["environments"] for method in scope["methods"]]
        labels = [f"{LABELS[e]}\n{LABELS[m]}" for e, m in pairs]
        colors = {"jvp": "#277da1", "directional": "#e07a5f"}
        for axis, metric, title in zip(axes, ("current", "historical"),
                                       ("Current holdout", "Post-eviction holdout"), strict=True):
            axis.axhline(0, color="#555555", linewidth=.8)
            for x, pair in enumerate(pairs):
                values = sorted((row for row in groups[pair] if row[metric] is not None),
                                key=lambda row: row["replica"])
                if not values:
                    continue
                y = [row[metric] for row in values]
                axis.scatter([x] * len(y), y, color=colors[pair[1]], alpha=.5, s=18)
                axis.scatter([x], [np.mean(y)], marker="D", color=colors[pair[1]], s=47,
                             edgecolor="#222222", linewidth=.5, zorder=3)
            axis.set_title(title)
            axis.set_xticks(range(len(pairs)), labels, rotation=32, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Paired NLL AUC gain")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=125)
        plt.close(fig)
        output = nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode()})
        cells.append(nbformat.v4.new_code_cell(source="", outputs=[output], execution_count=None))
    cells.append(nbformat.v4.new_markdown_cell(
        "## Interpretation\n\n"
        "A smaller NLL does not by itself establish STM-to-LTM transfer: the frozen-initial and "
        "STM-muted diagnostics from the original notebook remain important. Pilot-calibrated strengths "
        "are frozen here; this extension is secondary evidence, not a replacement for the original "
        "finite-difference confirmation. A partial run is not a negative result."))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "plan15_extension": {"schema_version": 1, "artifact_only": True,
                             "refreshed_at": datetime.now(timezone.utc).isoformat()}})
    nbformat.validate(notebook)
    return notebook


def refresh(root: Path | str, *, publish):
    root = Path(root)
    content = nbformat.writes(build_notebook(root)).encode("utf-8")
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("extension notebook exceeds bounded publication size")
    target = root / "reports" / "fisher_memory_extension_results.ipynb"
    publish(target, content)
    return target
