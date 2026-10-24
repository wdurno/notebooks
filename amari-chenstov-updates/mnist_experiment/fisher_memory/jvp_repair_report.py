"""Artifact-only, compact live notebook for the Plan 15 JVP repair."""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat

from .artifacts import ArtifactError, iter_units
from .config import Treatment
from .jvp_repair import CONFIRM_FAMILY, LEDGER_UNIT, PILOT_FAMILY, SELECTION_UNIT, unit_id
from .metrics import paired_summary
from .report import MAX_NOTEBOOK_BYTES


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def collect(root: Path | str) -> dict:
    root = Path(root)
    records = list(iter_units(root, completed_only=False))
    by_id = {record["unit_id"]: record for record in records}
    ledger_record = by_id.get(LEDGER_UNIT)
    if ledger_record is None or ledger_record["status"] != "completed":
        raise ArtifactError("JVP repair ledger is missing or incomplete")
    ledger = json.loads((ledger_record["path"] / "result.json").read_text())
    if ledger.get("protocol") != "plan15-jvp-repair-v1" or ledger.get("schema_version") != 1:
        raise ArtifactError("incompatible JVP repair ledger")
    selection_record = by_id.get(SELECTION_UNIT)
    selection = None
    if selection_record is not None:
        if selection_record["status"] != "completed":
            raise ArtifactError("JVP repair selection is incomplete")
        selection = json.loads((selection_record["path"] / "result.json").read_text())
        if selection.get("protocol") != ledger["protocol"]:
            raise ArtifactError("incompatible JVP repair selection")
    expected = {}
    for family, count, treatments in (
        (PILOT_FAMILY, ledger["pilot_replicas"], [ledger["baseline"], *ledger["grid"]]),
        (CONFIRM_FAMILY, ledger["confirm_replicas"],
         [ledger["baseline"], *selection["confirm"]] if selection else []),
    ):
        for replica in range(count):
            for config in treatments:
                treatment = Treatment(**config)
                expected[unit_id(family, replica, treatment)] = (family, replica, config)
    summaries = {}
    diagnostics = {}
    for record in records:
        if record["metadata"].get("kind") != "trajectory":
            continue
        unit = record["unit_id"]
        if unit not in expected:
            raise ArtifactError(f"unplanned JVP repair trajectory: {unit}")
        if record["status"] != "completed":
            continue
        family, replica, config = expected[unit]
        meta = record["metadata"]
        if (meta["family"] != family or meta["replica"] != replica
                or meta["environment"] != ledger["environment"] or meta["config"] != config):
            raise ArtifactError(f"JVP repair trajectory identity mismatch: {unit}")
        summary = json.loads((record["path"] / "summary.json").read_text())
        if summary["unit_id"] != unit:
            raise ArtifactError(f"JVP repair summary identity mismatch: {unit}")
        summaries[unit] = summary
        metrics_path = record["path"] / "metrics.json"
        if not metrics_path.exists():
            raise ArtifactError(f"JVP repair metrics missing: {unit}")
        curvature = [row for row in json.loads(metrics_path.read_text())
                     if row.get("split") == "curvature"]
        latest = max(curvature, key=lambda row: row["exposure"]) if curvature else {}
        diagnostics[unit] = dict(exact_trace_stm=latest.get("exact_trace_stm"),
                                 under_solved_fraction=summary.get("under_solved_fraction"),
                                 solver_failures=summary.get("solver_failures"))
    status = json.loads((root / "status.json").read_text())
    return dict(ledger=ledger, selection=selection, status=status, expected=expected,
                summaries=summaries, diagnostics=diagnostics)


def _comparison(data: dict, family: str, treatment: Treatment) -> dict:
    ledger, summaries = data["ledger"], data["summaries"]
    count = ledger["pilot_replicas"] if family == PILOT_FAMILY else ledger["confirm_replicas"]
    baseline = Treatment(**ledger["baseline"])
    controls = {r: summaries[unit_id(family, r, baseline)] for r in range(count)
                if unit_id(family, r, baseline) in summaries}
    treated = {r: summaries[unit_id(family, r, treatment)] for r in range(count)
               if unit_id(family, r, treatment) in summaries}
    current = paired_summary({r: row["current_auc"] for r, row in controls.items()},
                             {r: row["current_auc"] for r, row in treated.items()})
    historical = paired_summary({r: row["historical_auc"] for r, row in controls.items()},
                                {r: row["historical_auc"] for r, row in treated.items()})
    audit = [data["diagnostics"][unit_id(family, r, treatment)] for r in treated]
    return dict(current=current, historical=historical, treated=treated, controls=controls,
                planned=count, audit=audit)


def _gain(value: dict, *, interval: bool = False) -> str:
    if value["mean_gain"] is None:
        return "-"
    result = f"{value['mean_gain']:+.4f}"
    if interval and value["ci"] is not None:
        result += f" [{value['ci'][0]:+.4f}, {value['ci'][1]:+.4f}]"
    return result


def _mean_field(rows: list[dict], field: str, scale: float = 1.) -> str:
    values = [row[field] for row in rows if row.get(field) is not None]
    return f"{sum(values) * scale / len(values):.1f}" if values else "-"


def build_notebook(root: Path | str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = collect(root)
    ledger, status, selection = data["ledger"], data["status"], data["selection"]
    pilot = [(Treatment(**item), _comparison(data, PILOT_FAMILY, Treatment(**item)))
             for item in ledger["grid"]]
    pilot_complete = sum(unit_id(PILOT_FAMILY, r, Treatment(**config)) in data["summaries"]
                         for r in range(ledger["pilot_replicas"])
                         for config in [ledger["baseline"], *ledger["grid"]])
    confirmation = ([(Treatment(**item), _comparison(data, CONFIRM_FAMILY, Treatment(**item)))
                     for item in selection["confirm"]] if selection else [])
    confirmation_complete = sum(unit_id(CONFIRM_FAMILY, r, Treatment(**config)) in data["summaries"]
                                for r in range(ledger["confirm_replicas"])
                                for config in [ledger["baseline"], *selection["confirm"]]) if selection else 0
    pilot_rows = [[f"{t.strength:.4g}", f"{v['current']['n_pairs']}/{v['planned']}",
                   _gain(v["current"]), _gain(v["historical"]),
                   f"{sum(s['active_seconds'] for s in v['treated'].values()) / len(v['treated']):.0f}"
                   if v["treated"] else "-"] for t, v in pilot]
    cells = [nbformat.v4.new_markdown_cell(
        "# Plan 15: Sigmoid JVP Repair\n\n"
        f"**Progress:** pilot {pilot_complete}/{ledger['pilot_replicas'] * (1 + len(ledger['grid']))} "
        f"trajectories; confirmation {confirmation_complete}/"
        f"{ledger['confirm_replicas'] * (1 + len(selection['confirm'])) if selection else 'pending selection'}; "
        f"status **{status['status']}**; active time **{status.get('active_seconds', 0) / 3600:.2f} h**. "
        f"Current job: {status.get('current_job') or 'none'}.\n\n"
        "New development and confirmation seed families are disjoint. Each JVP trajectory is paired "
        "with a same-seed replay/EWC control. The finite-difference-selected coefficient is always "
        "included in confirmation, even when the pilot picks another strength. Earlier Plan 15 "
        "confirmation runs are context, not pooled replicates."),
        nbformat.v4.new_markdown_cell(
            "## Development strength response\n\n"
            "Gain is paired replay/EWC minus JVP NLL AUC; positive favors JVP. "
            "The three-replica grid is for selection, not significance testing.\n\n"
            + _table(["JVP strength", "Pairs", "Current gain", "Post-eviction gain", "Online s/trajectory"],
                     pilot_rows))]
    if any(value["current"]["n_pairs"] for _, value in pilot):
        fig, ax = plt.subplots(figsize=(7.8, 3.2), layout="constrained")
        colors = {"current": "#227c76", "historical": "#b85747"}
        for endpoint, label in (("current", "Current holdout"), ("historical", "Post-eviction holdout")):
            points = [(t.strength, v[endpoint]["mean_gain"]) for t, v in pilot
                      if v[endpoint]["mean_gain"] is not None]
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                        color=colors[endpoint], label=label)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.axvline(ledger["matched_strength"], color="#888888", linestyle=":", linewidth=1,
                   label="Matched FD strength")
        if selection:
            ax.axvline(selection["selected"]["strength"], color="#333333", linestyle="--",
                       linewidth=1, label="Pilot selection")
        ax.set_xscale("log")
        ax.set_xlabel("JVP strength")
        ax.set_ylabel("Mean paired NLL-AUC gain")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=120)
        plt.close(fig)
        cells.append(nbformat.v4.new_code_cell(source="", outputs=[nbformat.v4.new_output(
            "display_data", data={"image/png": base64.b64encode(stream.getvalue()).decode()})],
            execution_count=None))
    if selection:
        confirm_rows = []
        for treatment, values in confirmation:
            current, historical = values["current"], values["historical"]
            confirm_rows.append([f"{treatment.strength:.4g}",
                                 "Pilot choice" if treatment.key == Treatment(**selection["selected"]).key
                                 else "Matched coefficient", f"{current['n_pairs']}/{values['planned']}",
                                 _gain(current, interval=True), _gain(historical, interval=True),
                                 f"{current['favorable_replicas']}; {historical['favorable_replicas']}",
                                 _mean_field(values["audit"], "exact_trace_stm"),
                                 _mean_field(values["audit"], "under_solved_fraction", 100.),
                                 _mean_field(list(values["treated"].values()), "active_seconds")])
        cells.append(nbformat.v4.new_markdown_cell(
            "## Fresh confirmation\n\n"
            f"Selection: **{selection['classification']}**; chosen JVP strength "
            f"**{selection['selected']['strength']:.4g}**. "
            "The pilot-selected arm is primary; the fixed matched-coefficient arm is secondary. "
            "Intervals are nominal paired-bootstrap intervals, and partial cohorts are descriptive. "
            "STM trace is the latest independent exact Fisher trace; solver percentage is the "
            "fraction of under-solved updates; time is online seconds per trajectory.\n\n"
            + _table(["JVP strength", "Role", "Pairs", "Current gain [95%]",
                      "Post-eviction gain [95%]", "Favorable current; historical",
                      "STM trace", "Under-solved %", "Online s"], confirm_rows)))
    else:
        cells.append(nbformat.v4.new_markdown_cell(
            "## Fresh confirmation\n\nPending completion of the frozen development grid."))
    cells.append(nbformat.v4.new_markdown_cell(
        "## Interpretation\n\n"
        "This repair asks whether the prior sigmoid JVP result was a strength-selection failure. "
        "It does not establish STM-to-LTM transfer. Rotation solves may remain under-optimized; "
        "trace and solver diagnostics are retained in each immutable trajectory artifact. "
        "Do not select a favorable endpoint or extend a cohort because an interval is close to zero."))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "plan15_jvp_repair": {"schema_version": 1, "artifact_only": True,
                              "refreshed_at": datetime.now(timezone.utc).isoformat()}})
    nbformat.validate(notebook)
    return notebook


def refresh(root: Path | str, *, publish):
    content = nbformat.writes(build_notebook(root)).encode("utf-8")
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("JVP repair notebook exceeds bounded publication size")
    target = Path(root) / "reports" / "fisher_memory_jvp_repair_results.ipynb"
    publish(target, content)
    return target
