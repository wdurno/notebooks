"""Compact Phase 1 reporting from validated scalar snapshots, never model code.

Call append_phase1(book, snapshot) after report._populate_outputs. It inserts
pre-rendered, independently re-executable cells immediately before Development.
Only load_report's complete_units are consumed; no artifact paths are opened.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


_RECIPE = ("top_layers", "hidden_size", "ltm_rank", "stm_rank", "context_length",
           "batch", "fd_probes", "fisher_draws", "image_width", "image_height", "max_pixels", "max_tokens",
           "allocator", "frozen_prefix_cache")
_TIMINGS = (("load_seconds", "Load s"), ("inference_seconds", "Infer s"),
            ("nll_backward_seconds", "NLL backward s"), ("fd_backward_seconds", "FD backward s"),
            ("fit_control_seconds", "Control fit s"), ("fit_fd_seconds", "FD fit s"),
            ("fisher_seconds", "Fisher s"), ("teacher_seconds", "Teacher s"),
            ("evaluator_seconds", "Evaluator s"))
_LIMIT = 6
_TAG = "plan19_phase1"


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _text(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (Mapping, list, tuple)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    elif type(value) is bool:
        return "true" if value else "false"
    elif type(value) is float:
        return f"{value:.3g}" if math.isfinite(value) else "unavailable"
    value = str(value)
    return value if len(value) <= 240 else value[:237] + "..."


def _units(snapshot, kind):
    # No live summaries, partial metrics, or secondary filesystem reads.
    return sorted((u for u in snapshot.get("complete_units", ())
                   if u.get("summary", {}).get("phase") == 1
                   and u["summary"].get("kind") == kind),
                  key=lambda u: (str(u.get("manifest", {}).get("completed_at", "")), u.get("path", "")))


def _recipe(unit):
    summary = unit["summary"]
    return _mapping(summary.get("recipe") or
                    _mapping(unit.get("stored_config", {}).get("phase1")).get("recipe"))


def _recipe_key(recipe):
    return tuple(_mapping(recipe).get(key) for key in _RECIPE)


def _forecast(snapshot):
    units = _units(snapshot, "phase1_forecast")
    return units[-1]["summary"] if units else {}


def _selected(unit, selected):
    return bool(_mapping(selected)) and _recipe_key(_recipe(unit)) == _recipe_key(selected)


def _candidates(snapshot):
    units = _units(snapshot, "phase1_resource_probe")
    selected = _forecast(snapshot).get("selected_recipe")
    # Keep the recorded selection, status diversity, then recent probes. Never
    # rank candidates by an invented competence or missing-as-zero cost score.
    chosen = [u for u in reversed(units) if _selected(u, selected)][:1]
    for status in ("oom", "inadmissible", "failed", "passed"):
        match = next((u for u in reversed(units) if u["summary"].get("status") == status), None)
        if match is not None and match not in chosen and len(chosen) < _LIMIT:
            chosen.append(match)
    for unit in reversed(units):
        if unit not in chosen and len(chosen) < _LIMIT:
            chosen.append(unit)
    return chosen, len(units)


def _label(unit):
    recipe = _recipe(unit)
    return (f"L={recipe.get('top_layers', '?')} H={recipe.get('hidden_size', '?')} "
            f"C={recipe.get('context_length', '?')} B={recipe.get('batch', '?')}")


def _status(unit):
    status = unit["summary"].get("status", "unavailable")
    if _recipe(unit).get("top_layers") == 0:
        return f"{status}; top-0 diagnostic only"
    if status == "passed":
        return "passed; checks failed" if _failed_checks(unit) else "passed; resource only"
    return status


def _checks(unit):
    summary = unit["summary"]
    return list(_leaves(summary.get("metricschecks", summary.get("metrics_checks", summary.get("checks", {})))))


def _failed_checks(unit):
    return [key for key, value in _checks(unit) if value is False]


def _range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = map(_number, value)
        if lo is not None and hi is not None and lo <= hi:
            return lo, hi
    return None


def _forecast_rows(snapshot):
    summary = _forecast(snapshot)
    hours = _mapping(summary.get("phase_hours"))
    rows = []
    for phase, name in ((2, "Bootstrap"), (3, "Development"), (4, "Fresh pairs"), (5, "Closeout")):
        bounds = _range(hours.get(f"phase{phase}", hours.get(str(phase))))
        rows.append({"Stage": f"{phase}: {name}",
                     "Low h": bounds[0] if bounds else None, "High h": bounds[1] if bounds else None})
    bounds = _range(summary.get("total_hours_range"))
    rows.append({"Stage": "Reported total", "Low h": bounds[0] if bounds else None,
                 "High h": bounds[1] if bounds else None})
    return rows


def _scope_rows(snapshot):
    scopes = {}
    for unit in _units(snapshot, "phase1_forecast"):
        value = unit["summary"]
        scopes[(value.get("pairs"), value.get("transitions_per_arm"))] = value
    rows = []
    for (pairs, transitions), value in scopes.items():
        if pairs is None or transitions is None:
            continue
        bounds = _range(value.get("total_hours_range"))
        rows.append({"Paired replicas": pairs, "New transitions / arm": transitions,
                     "Total active hours": f"{bounds[0]:.1f}--{bounds[1]:.1f}" if bounds else None,
                     "Within 20 h": value.get("budgetfit")})
    return rows


def phase1_conclusion(snapshot):
    campaigns = _units(snapshot, "phase1_campaign")
    if not campaigns:
        return None
    campaign = campaigns[-1]["summary"]
    recipe = campaign["selected_recipe"]
    scopes = _scope_rows(snapshot)
    cost = "; ".join(f"{r['Paired replicas']} pairs x {r['New transitions / arm']} new transitions/arm: "
                     f"{r['Total active hours']} h" for r in scopes)
    blockers = campaign.get("blockers", [])
    return ("## Conclusion\n\nPhase 1 resource calibration is complete; later learning phases have not started. "
            f"The resource shortlist uses {recipe['top_layers']} adapted transformer block(s), "
            f"history {recipe['context_length']}, and online replay batch {recipe['batch']}. "
            "This is a memory/timing shortlist, not demonstrated controller competence.\n\n"
            f"**Whole-study forecast:** {cost}. These are conditional engineering ranges, not confidence intervals.\n\n"
            + ("**Resolve before learning:** " + " ".join(blockers) if blockers else
               "The small diagnostic gates passed; full bootstrap and student readiness remain Phase 2 work.")
            + "\n\nNo sample-efficiency result is claimed. Physical calibration and a bounded full-study "
              "checkpoint/archive policy are still required before strong deployment/resource claims.")


def plot_resources(snapshot):
    """Measured peaks and separate operation timings; None is never a zero bar."""
    units, _ = _candidates(snapshot)
    if not units:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    memory_axis, time_axis = axes
    selected = _forecast(snapshot).get("selected_recipe")
    labels = [f"{i + 1}: {_label(u)}" + (" *" if _selected(u, selected) else "")
              for i, u in enumerate(units)]
    memory_labels = set()
    for index, unit in enumerate(units):
        memory = _mapping(unit["summary"].get("memory"))
        for offset, key, color in ((-.14, "peak_allocated_bytes", "#26705a"),
                                   (.14, "peak_reserved_bytes", "#ae4854")):
            value = _number(memory.get(key))
            if value is not None:
                label = key.split("_")[1]
                memory_axis.barh(index + offset, value / 1024**3, height=.25,
                                 color=color, label=label if label not in memory_labels else None)
                memory_labels.add(label)
        total = _number(memory.get("total_bytes"))
        headroom = _number(memory.get("headroom_bytes"))
        if total is not None:
            memory_axis.plot(total / 1024**3, index, "|", color="black", markersize=16,
                             label="device total" if "device total" not in memory_labels else None)
            memory_labels.add("device total")
        if all(_number(memory.get(k)) is None for k in ("peak_allocated_bytes", "peak_reserved_bytes")):
            memory_axis.text(.02, index, "VRAM unavailable", va="center", fontsize=8,
                             transform=memory_axis.get_yaxis_transform())
        if headroom is not None:
            memory_axis.annotate(f"headroom {headroom / 1024**3:.2g} GiB", (1, index + .38),
                                 xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=7)
    timings = (("inference_seconds", "Inference", "#555555"),
               ("fit_control_seconds", "Control fit", "#26705a"),
               ("fit_fd_seconds", "FD fit", "#ae4854"),
               ("fisher_seconds", "Fisher", "#986b20"))
    for offset, (key, label, color) in enumerate(timings):
        measured = [(i, _number(_mapping(u["summary"].get("timings")).get(key)))
                    for i, u in enumerate(units)]
        measured = [(i, value) for i, value in measured if value is not None]
        if measured:
            time_axis.scatter([value for _, value in measured],
                              [i + (offset - 1.5) * .12 for i, _ in measured],
                              s=22, label=label, color=color, zorder=3)
    for index, unit in enumerate(units):
        if not any(_number(_mapping(unit["summary"].get("timings")).get(key)) is not None
                   for key, _, _ in timings):
            time_axis.text(.02, index, "plotted timings unavailable", va="center", fontsize=8,
                           transform=time_axis.get_yaxis_transform())
    for axis in axes:
        axis.set_yticks(range(len(units)), labels if axis is memory_axis else
                        [str(i + 1) for i in range(len(units))], fontsize=8)
        axis.set_ylim(len(units) - .5, -.5)
        axis.set_xlim(left=0)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="x", alpha=.2)
        handles, names = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, names, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2)
    memory_axis.set_xlabel("Measured VRAM (GiB); failed probes may be partial")
    time_axis.set_xlabel("Seconds per recorded operation (not additive)")
    return figure


def plot_forecast(snapshot):
    """Artifact planning ranges, not measured runtimes or confidence intervals."""
    rows = _forecast_rows(snapshot)
    if not any(row["Low h"] is not None for row in rows):
        return None
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 2.7), constrained_layout=True)
    for i, row in enumerate(rows):
        if row["Low h"] is None:
            axis.text(.02, i, "unavailable", va="center", fontsize=8, transform=axis.get_yaxis_transform())
        else:
            axis.plot([row["Low h"], row["High h"]], [i, i], "|-", color="#26705a", linewidth=4)
    axis.axvline(20, color="#ae4854", linestyle="--", linewidth=1, label="20 h overall cap")
    axis.set_yticks(range(len(rows)), [row["Stage"] for row in rows], fontsize=9)
    axis.set(xlabel="Forecast active hours (planning ranges, not confidence intervals)",
             xlim=(0, None), ylim=(len(rows) - .5, -.5))
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=.2)
    return figure


def _leaves(value, prefix=""):
    if isinstance(value, Mapping):
        for key, child in sorted(value.items()):
            yield from _leaves(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        # Per-episode records and inventories belong in the archive, not the
        # narrative. Scalar lists (limitations, blockers) can be shown compactly.
        if not any(isinstance(child, (Mapping, list)) for child in value):
            yield prefix, value
    else:
        yield prefix, value


def _readiness_rows(snapshot):
    rows = []
    for kind, title in (("phase1_scene_audit", "Scene"), ("phase1_roles", "Roles")):
        units = _units(snapshot, kind)
        if not units:
            rows.append({"Audit": title, "Check / metric": "status", "Value": "not recorded"})
            continue
        summary = units[-1]["summary"]
        if kind == "phase1_roles" and "teacher_A_success_rate" in summary:
            episodes = summary.get("teacher_episodes", [])
            for key, value in (
                ("Teacher successes / rollouts", f"{sum(e['success'] for e in episodes)}/{len(episodes)} (A and B)"),
                ("Teacher A success rate", summary["teacher_A_success_rate"]),
                ("Completion score ranks success above far failure", summary.get("reward_success_above_far")),
                ("Control period (s)", summary.get("recommended_control_period_s")),
                ("Non-startup fallback fraction", summary.get("nonstartup_fallback_fraction")),
                ("Physical calibration", "not measured")):
                rows.append({"Audit": title, "Check / metric": key, "Value": _text(value)})
            continue
        selected = [(key, value) for key, value in _leaves(summary)
                    if any(word in key.lower() for word in
                           ("status", "pass", "success", "competence", "fallback", "collision", "layout", "count", "episode",
                            "language", "deadline", "physical", "calibrat", "reward", "geometry", "reason",
                            "feasible", "limitation", "diagnostic", "accelerated", "hardware"))
                    and not key.endswith(("_path", "_paths"))]
        selected.sort(key=lambda item: (
            not any(word in item[0].lower() for word in
                    ("success", "competence", "fallback", "language", "count", "episode", "limitation")),
            item[0].count("."), item[0]))
        rows.append({"Audit": title, "Check / metric": "source (latest completed)",
                     "Value": f"{Path(units[-1]['path']).name}; {len(units)} archived unit(s)"})
        for key, value in selected[:10]:
            rows.append({"Audit": title, "Check / metric": key.replace("_", " "), "Value": _text(value)})
        if not selected:
            rows.append({"Audit": title, "Check / metric": "status", "Value": "no recognized checks recorded"})
        if len(selected) > 10:
            rows.append({"Audit": title, "Check / metric": "additional checks", "Value": f"{len(selected) - 10} in archived summary"})
    return rows


def phase1_outputs(snapshot, *, notebook_dir=None):
    """Build bounded nbformat outputs shared by pre-rendering and cell execution."""
    import nbformat
    import pandas as pd

    outputs = []

    def note(message):
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=message + "\n"))

    def table(rows, *, details=False):
        frame = pd.DataFrame([{k: _text(v) for k, v in row.items()} for row in rows])
        markup = frame.to_html(index=False, escape=True)
        if details:
            markup = '<details><summary>Numerical probe checks</summary>' + markup + '</details>'
        outputs.append(nbformat.v4.new_output("display_data", data={
            "text/plain": frame.to_string(index=False), "text/html": markup}))

    def plot(figure, description):
        if figure is not None:
            import matplotlib.pyplot as plt
            try:
                data = io.BytesIO()
                figure.savefig(data, format="png", dpi=120)
                outputs.append(nbformat.v4.new_output("display_data", data={
                    "image/png": base64.b64encode(data.getvalue()).decode(), "text/plain": description}))
            finally:
                plt.close(figure)

    forecast = _forecast(snapshot)
    selected = forecast.get("selected_recipe")
    if _mapping(selected):
        groups = (("Model", ("top_layers", "hidden_size", "ltm_rank", "stm_rank")),
                  ("Online batch", ("context_length", "batch", "fd_probes", "fisher_draws")),
                  ("Sensor / tokens", ("image_width", "image_height", "max_pixels", "max_tokens")))
        table([{"Selected recipe": title, "Settings": "; ".join(
            f"{key.replace('_', ' ')}={_text(selected.get(key))}" for key in keys)} for title, keys in groups])
    else:
        note("Selected recipe: not recorded. No candidate is automatically admitted.")
    units, total = _candidates(snapshot)
    if units:
        rows = []
        for i, unit in enumerate(units):
            summary = unit["summary"]
            recipe = _recipe(unit)
            label = (_label(unit) + f"; ranks={recipe.get('ltm_rank', '?')}/{recipe.get('stm_rank', '?')}"
                     f"; FD/F={recipe.get('fd_probes', '?')}/{recipe.get('fisher_draws', '?')}")
            row = {"Probe (* selected)": str(i + 1) + (" *" if _selected(unit, selected) else ""), "Recipe": label,
                   "Status": _status(unit)}
            row.update({label: _number(_mapping(summary.get("timings")).get(key)) for key, label in _TIMINGS
                        if key in ("inference_seconds", "fit_control_seconds", "fit_fd_seconds", "fisher_seconds")})
            rows.append(row)
        table(rows)
        detail = next((u for u in units if _selected(u, selected)), units[0])
        table([{"Additional timings (probe)": units.index(detail) + 1,
                **{label: _number(_mapping(detail["summary"].get("timings")).get(key)) for key, label in _TIMINGS
                   if key not in ("inference_seconds", "fit_control_seconds", "fit_fd_seconds", "fisher_seconds")}}])
        note(f"Showing {len(units)} of {total} completed probes; all trials and exact recipes remain in archived units. "
             "Timings are individual operations, not a summed update cost; unavailable is not zero. "
             "A recorded selection (*) is not competence or admission certification.")
        plot(plot_resources(snapshot), "Measured candidate VRAM and operation timings; selection marked *.")
    else:
        note("Resource probes pending: no completed Phase 1 probe artifacts.")
    if forecast:
        source = _units(snapshot, "phase1_forecast")[-1]
        note(f"Forecast source (latest completed): {Path(source['path']).name}.")
        if _scope_rows(snapshot):
            table(_scope_rows(snapshot))
        table(_forecast_rows(snapshot))
        note(f"Recorded budget fit: {_text(forecast.get('budgetfit', forecast.get('budget_fit')))}. "
             "Total scope is as recorded, not inferred by adding phase estimates or concurrent probe charges.")
        note("Forecast includes shared bootstrap, development, 72 evaluation episodes per arm, "
             "sleep, and checkpoint I/O. One optimizer event and one eviction Fisher per new transition "
             "are assumed; larger sleep batches and skill rescue work need remeasurement. "
             "Ranges are engineering allowances, not confidence intervals.")
        if forecast.get("qualifiedassumptions"):
            note(f"Qualified assumptions: {_text(forecast['qualifiedassumptions'])}")
        note(f"Blockers: {_text(forecast.get('blockers'))}")
        plot(plot_forecast(snapshot), "Artifact-only forecast ranges; the 20 h cap is overall, not per phase.")
        storage_rows = []
        for source in _units(snapshot, "phase1_forecast"):
            value = source["summary"]
            storage = value.get("storage", {})
            if storage:
                storage_rows.append({"Fresh scope": f"{value['pairs']} pairs x {value['transitions_per_arm']}",
                    "Raw training frames / records (GB)": storage["raw_and_record_bytes"] / 1e9,
                    "Retain every full checkpoint (GB)": storage["retain_all_checkpoint_bytes"] / 1e9})
        if storage_rows:
            table(storage_rows)
            note("Storage is a fresh-training projection, not a total allocation: development, evaluation video, "
                 "and failed attempts add bytes. Keeping every transaction snapshot is not an admitted policy; "
                 "bounded recovery checkpoints plus an append-only event journal/archive are needed before learning.")
    else:
        note("Overall forecast pending: no completed forecast artifact. The 20 h target is not a measured runtime promise.")
    table(_readiness_rows(snapshot))
    checks = [{"Probe": i + 1, "Recorded numerical checks": "; ".join(f"{key}={_text(value)}" for key, value in _checks(u))
               or "unavailable"} for i, u in enumerate(units)]
    if checks:
        table(checks, details=True)
    rejected = [u for u in units if u["summary"].get("status") != "passed"
                or _recipe(u).get("top_layers") == 0 or _failed_checks(u)]
    if rejected:
        table([{"Probe": units.index(u) + 1, "Blocker / limitation": _status(u),
                "Recorded reason": u["summary"].get("reason") or "; ".join(_failed_checks(u)) or None} for u in rejected])
    matching = [u for u in _units(snapshot, "phase1_resource_probe") if _selected(u, selected)]
    if selected and not any(u["summary"].get("status") == "passed" and not _failed_checks(u)
                            and _number(_recipe(u).get("top_layers")) not in (None, 0)
                            for u in matching):
        note("Selection not supported by a matching passed top-L>0 probe; resource admission remains unresolved.")
    root = snapshot.get("root")
    if root:
        relative = os.path.relpath(root, notebook_dir or Path(__file__).parent)
        outputs.append(nbformat.v4.new_output("display_data", data={
            "text/html": '<details><summary>Archived scalar artifacts and exact recipes</summary>'
                         f'<p><a href="{quote(relative)}">{html.escape(str(root))}</a></p></details>',
            "text/plain": f"Archived artifacts: {root}"}))
    note("Readiness numbers are recorded diagnostics, not held-out student competence or sample savings. "
         "Missing role, language, fallback, or physical checks remain unverified. "
         "Top-0 is diagnostic-only. Strong sim-to-real claims require physical calibration. "
         "Forecast accounting must include evaluation, sleep, failed attempts, reporting and I/O, charged once; "
         "a budget extension requires approval.")
    return outputs


def display_phase1(snapshot):
    """Read-only notebook refresh using the same outputs as pre-rendering."""
    from IPython.display import display

    for output in phase1_outputs(snapshot, notebook_dir=Path.cwd()):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_phase1(book, snapshot, *, notebook_dir=None):
    """Insert/replace the Phase 1 section in place, return book, and never write files.

    Integration: book = append_phase1(_populate_outputs(book, snapshot), snapshot).
    No make_notebook hook is necessary. Existing cells and outputs are preserved.
    Pass notebook_dir for reports saved outside the default notebook directory.
    """
    import nbformat

    book.cells = [cell for cell in book.cells if not cell.metadata.get(_TAG)]
    cells = [nbformat.v4.new_markdown_cell(
        "## Phase 1: Feasibility and Forecast\n\n"
        "Can a controllable, language-conditioned task fit the online resource budget? "
        "Phase 1 measures this before learning curves. Resource probes, scene/role audits, "
        "and qualified forecasts are smoke-only diagnostics, not scientific learning results. "
        "Only completed scalar artifacts are used; missing measurements remain unavailable."),
        nbformat.v4.new_code_cell(
            "from numerical_experiment.picar.phase1_report import display_phase1\n"
            "display_phase1(snapshot)", outputs=phase1_outputs(snapshot, notebook_dir=notebook_dir))]
    for i, cell in enumerate(cells):
        cell.metadata[_TAG] = True
        cell.id = hashlib.sha256(f"{_TAG}:{i}:{cell.source}".encode()).hexdigest()[:12]
    index = next((i for i, cell in enumerate(book.cells)
                  if cell.cell_type == "markdown" and cell.source.startswith("## Development")), len(book.cells))
    book.cells[index:index] = cells
    return book
