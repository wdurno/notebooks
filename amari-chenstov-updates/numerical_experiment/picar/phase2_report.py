"""Artifact-only Phase 2 notebook reporting; no model or checkpoint imports.

Scientific summaries come only from the parent's validated complete_units.
Only orchestration progress/settings are read directly from the cache.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from datetime import datetime
import html
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


PENDING = "pending / not recorded"
_PREFIX = "phase2_"
_NOTEBOOK_DIR = Path(__file__).parent
_MOTIVATION = (
    "### Phase 2: Initial Skill and Common Student Readiness\n\n"
    "Before comparing FD, establish useful initial easy-A skill and common student trainability on B. "
    "Frozen Qwen need not solve the hard task. The baseline reuses the compatible Fix 3 bootstrap; "
    "its original counts and compute are excluded from new Phase 2 charges. Easy-A uses all 20 fixed "
    "layouts (192501--192520); hard-B uses the first 10. Clear and blocked paths are reported together. "
    "Two baseline student pilots (192101/192102) are bounded at 640 intervals each. The prior-study estimate is "
    "conservatively about 2.25 h; Phase 2 has a 5 h cap and ends at a check-in, not automatic Phase 3. "
    "Only sealed units supply results; live progress is operational information.\n\n"
    "A later language audit exposed a saturated visual-to-LSTM projection. Separately versioned, "
    "parameter-free RMS rescues reuse the recorded demonstrations but rebuild their one-time "
    "initialization and Fisher. They test conditioning with FD disabled, not STM efficacy. "
    "All Phase 2 versions share the same five-hour cap."
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value):
    if value is None:
        return PENDING
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _compact(value):
    text = _text(value)
    return text if len(text) <= 360 else text[:357] + "..."


def _kind(unit):
    return unit["summary"]["kind"][len(_PREFIX):]


def _phase2_path(path):
    return any(p.name == "phase2" or p.name.startswith(("phase2-", "phase2_"))
               for p in (path, *path.parents))


def _campaign(path, root):
    for parent in (path, *path.parents):
        if parent.name.startswith(("readiness-", "readiness_")):
            return parent
        if parent.name == "phase2" or parent.name.startswith(("phase2-", "phase2_")):
            return parent
        if parent == root:
            break
    return root


def _order(item):
    path, data = item
    times = []
    values = [data["progress"].get("updated_at")]
    values.extend(_mapping(u.get("manifest")).get("completed_at") for u in data["units"])
    for value in values:
        if _number(value):
            times.append(float(value))
        elif isinstance(value, str):
            try:
                times.append(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
            except ValueError:
                pass
    return max(times, default=0.), str(path)


def _campaigns(snapshot):
    root = Path(snapshot.get("root", _NOTEBOOK_DIR)).expanduser().resolve()
    campaigns = {}

    def group(path):
        campaign = _campaign(Path(path).resolve(), root)
        return campaigns.setdefault(campaign, {"units": [], "progress": {}})

    for unit in snapshot.get("complete_units", []):
        if str(_mapping(unit.get("summary")).get("kind", "")).startswith(_PREFIX):
            group(unit["path"])["units"].append(unit)
    for live in snapshot.get("live", []):
        path = Path(live["path"]).resolve()
        if _phase2_path(path) or any(p.name.startswith(("readiness-", "readiness_"))
                                     for p in (path, *path.parents)):
            group(path)["progress"].update(live)
    if snapshot.get("root") and root.is_dir():
        from .report import _json, _version

        for path in sorted(root.rglob("orchestration/progress.json")):
            if not (_phase2_path(path) or _campaign(path, root).name.startswith(("readiness-", "readiness_"))):
                continue
            progress = _json(path)
            if "schema_version" in progress:
                _version(progress["schema_version"], 1, "Phase 2 progress schema")
            group(path.parent)["progress"].update(progress)
    for data in campaigns.values():
        data["units"].sort(key=lambda u: (str(_mapping(u.get("manifest")).get("completed_at", "")), u["path"]))
    return campaigns


def _table(rows):
    import nbformat
    import pandas as pd

    keys = dict.fromkeys(key for row in rows for key in row)
    frame = pd.DataFrame([{key: _compact(row.get(key)) for key in keys} for row in rows])
    return nbformat.v4.new_output("display_data", data={
        "text/plain": frame.to_string(index=False), "text/html": frame.to_html(index=False, escape=True)})


def _note(text):
    import nbformat

    return nbformat.v4.new_output("stream", name="stdout", text=text + "\n")


def _count(value):
    if type(value) is int and value >= 0:
        return value
    return len(value) if isinstance(value, list) else None


def _scalars(value, keys):
    value = _mapping(value)
    return {key: value[key] for key in keys if key in value
            and (type(value[key]) in (str, bool) or _number(value[key]))}


def _settings(value):
    keys = ("transitions", "ridge", "learning_rate", "lr", "exploration_std", "teacher_floor",
            "bootstrap_transitions", "fd_strength", "phase2_cap_hours", "max_active_hours")
    settings = _scalars(value, keys)
    hyperparams = _scalars(_mapping(value).get("hyperparams"), keys)
    if hyperparams:
        settings["hyperparams"] = hyperparams
    return settings or None


def _asset_rows(units):
    keys = ("source", "source_marker_sha256", "reused", "acquired_count", "distinct_bootstrap_samples",
            "retired_count", "eligible_count", "initial_evidence_disjoint", "original_compute_excluded")
    return [{"Asset": _kind(u), **_scalars(u["summary"], keys)} for u in units]


def _language_rows(rows):
    # The archived probability order is src.robotics.actions.ACTION_NAMES.
    actions = ("drive-left", "drive-right", "drive-forward", "drive-backward",
               "look-left", "look-right", "look-up", "look-forward")
    result = []
    for condition in ("forward", "left", "right", "stop"):
        group = [row for row in rows if row.get("condition") == condition]
        probabilities = [row["probabilities"] for row in group
            if isinstance(row.get("probabilities"), list) and len(row["probabilities"]) == 8
            and all(_number(p) and 0 <= p <= 1 for p in row["probabilities"])
            and math.isclose(sum(row["probabilities"]), 1., abs_tol=1e-5)]
        counts = Counter(actions[max(range(8), key=p.__getitem__)] for p in probabilities)
        result.append({"Instruction": condition, "Rows": len(group), "Valid probability rows": len(probabilities),
            "Mean forward P": sum(p[2] for p in probabilities) / len(probabilities) if probabilities else None,
            "Mean left P": sum(p[0] for p in probabilities) / len(probabilities) if probabilities else None,
            "Mean right P": sum(p[1] for p in probabilities) / len(probabilities) if probabilities else None,
            "Argmax counts": "; ".join(f"{action}: {n}" for action, n in counts.most_common()) or None})
    return result


def _messages(value):
    if isinstance(value, str):
        return _compact(value)
    if isinstance(value, list):
        return _compact("; ".join(v for v in value if isinstance(v, str))) or None
    return None


def _forecast_rows(summary):
    fields = _scalars(summary, ("seconds_per_captured_interval", "phase2_active_seconds", "prior_study_seconds",
        "study_charged_seconds", "remaining_to_20h_seconds", "trainability_observed", "fd_unmeasured",
        "next_phase_authorized"))
    fields.update(_scalars(summary.get("totals"), ("prior_study_hours", "phase2_cap_hours", "total_hours")))
    definitions = _mapping(summary.get("cost_definitions"))
    optional = {"cost_definitions": "; ".join(f"{k}: {v}" for k, v in definitions.items() if isinstance(v, str)) or None,
                **{k: _messages(summary.get(k)) for k in ("readiness_conclusions", "blockers", "next_checkin")}}
    fields.update({k: v for k, v in optional.items() if v is not None})
    return [{"Field": key, "Recorded": value} for key, value in list(fields.items())[:8]]


def _rate(rows, field="success"):
    observed = []
    for row in rows:
        value = row.get(field)
        if field == "collision" and type(value) is not bool:
            reason = row.get("end_reason")
            if reason in ("collision", "obstacle_collision", "wall_collision", "obstacle_contact", "wall_contact"):
                value = True
            elif reason in ("success", "target_contact", "timeout", "time_limit", "deadline"):
                value = False
        if type(value) is bool:
            observed.append(value)
    if not observed:
        return PENDING
    unknown = f"; {len(rows) - len(observed)} unknown" if len(rows) != len(observed) else ""
    return f"{sum(observed)}/{len(observed)} ({sum(observed) / len(observed):.1%}){unknown}"


def _evaluations(summary, kind):
    grouped = defaultdict(list)
    if kind == "teacher_readiness":
        grouped[("frozen", "easy")].extend(_rows(summary.get("rows")))
    for evaluation in _rows(summary.get("evaluations")):
        grouped[(_text(evaluation.get("checkpoint")), _text(evaluation.get("difficulty")))].extend(
            _rows(evaluation.get("rows")))
    if kind in ("initial_readiness", "pilot"):
        for checkpoint in (("initial", "final") if kind == "pilot" else ("initial",)):
            for difficulty in ("easy", "hard"):
                grouped.setdefault((checkpoint, difficulty), [])
    return grouped


def _pairing(before, after):
    ids = []
    for rows in (before, after):
        seeds = [r.get("seed") for r in rows]
        if not seeds or any(type(seed) not in (int, str) for seed in seeds):
            return False, "pairing unverified: missing seed IDs"
        if len(set(seeds)) != len(seeds):
            return False, "pairing unverified: duplicate seed IDs"
        ids.append({r["seed"]: r for r in rows})
    if ids[0].keys() != ids[1].keys():
        return False, "seed IDs differ"
    if any(isinstance(ids[0][seed].get("layout"), dict) and isinstance(ids[1][seed].get("layout"), dict)
           and ids[0][seed]["layout"] != ids[1][seed]["layout"] for seed in ids[0]):
        return False, "layout metadata differ despite matching seed IDs"
    return True, f"matched seed IDs (n={len(ids[0])})"


def _approach(rows):
    values = defaultdict(list)
    for row in rows:
        geometry = _mapping(row.get("terminal_geometry") or row.get("terminal_diagnostics"))
        for key, value in geometry.items():
            if key == "physical_seconds":
                continue
            if _number(value):
                values[key].append(value)
    return "; ".join(f"{k}={sum(v) / len(v):.4g} (n={len(v)}/{len(rows)})"
                     for k, v in sorted(values.items())[:4]) or PENDING


def _label(summary, kind):
    settings = _mapping(summary.get("settings"))
    return _text(summary.get("controller", "frozen Qwen" if kind == "teacher_readiness" else
                             settings.get("name", "student")))


def _blocked(row):
    layout = _mapping(row.get("layout"))
    return layout.get("direct_path_blocked", _mapping(layout.get("curriculum")).get("direct_path_blocked"))


def _coverage(rows, difficulty):
    count = 20 if str(difficulty).lower().startswith("easy") else 10 if str(difficulty).lower().startswith("hard") else None
    if count is None:
        return PENDING
    seeds = [r.get("seed") for r in rows]
    if not seeds or any(type(s) is not int for s in seeds):
        return f"{len(rows)}/{count} rows; seed coverage unverified"
    expected, observed = set(range(192501, 192501 + count)), set(seeds)
    return (f"{len(observed & expected)}/{count} expected seed IDs; "
            f"{len(observed - expected)} unexpected; {len(seeds) - len(observed)} duplicates")


def _outcome_rows(unit):
    summary, kind = unit["summary"], _kind(unit)
    grouped = _evaluations(summary, kind)
    result = []
    for (checkpoint, difficulty), rows in grouped.items():
        easy = str(difficulty).lower().startswith("easy")
        clear = [r for r in rows if _blocked(r) is False]
        blocked = [r for r in rows if _blocked(r) is True]
        physical = [r["physical_seconds"] for r in rows if _number(r.get("physical_seconds"))]
        pairing = _pairing(grouped.get(("initial", difficulty), []), grouped.get(("final", difficulty), []))[1]
        result.append({"Controller / recipe": _label(summary, kind), "Student seed": summary.get("seed"),
            "Checkpoint": checkpoint, "Difficulty": difficulty, "Rows": len(rows), "Success": _rate(rows),
            "Panel coverage": _coverage(rows, difficulty),
            "Collisions": _rate(rows, "collision"), "Easy clear success": _rate(clear) if easy else "n/a",
            "Easy blocked success": _rate(blocked) if easy else "n/a",
            "Easy path unknown": len(rows) - len(clear) - len(blocked) if easy else "n/a",
            "Physical seconds (mean / n)": f"{sum(physical) / len(physical):.4g} / {len(physical)}" if physical else None,
            "Terminal geometry (mean)": _approach(rows),
            "Before/after identity": pairing if kind == "pilot" else "single checkpoint"})
    return result


def _outcome_table(rows):
    compact = []
    has_geometry = any(r["Terminal geometry (mean)"] != PENDING for r in rows)
    for row in rows:
        easy = row["Easy path unknown"] != "n/a"
        controller = row["Controller / recipe"]
        if row["Student seed"] is not None:
            controller += " / " + _text(row["Student seed"])
        compact.append({"Controller / seed": controller,
            "Checkpoint / level": f"{row['Checkpoint']} / {row['Difficulty']}",
            "Success": row["Success"], "Collisions": row["Collisions"],
            "Easy clear; blocked success": (f"{row['Easy clear success']}; {row['Easy blocked success']}; "
                f"{row['Easy path unknown']} unknown paths") if easy else "n/a",
            "Coverage / pairing": f"{row['Panel coverage']}; {row['Before/after identity']}",
            "Physical seconds (mean / n)": row["Physical seconds (mean / n)"]})
        if has_geometry:
            compact[-1]["Terminal geometry (mean)"] = row["Terminal geometry (mean)"]
    return _table(compact)


def _plot(pilots):
    import nbformat
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(9, 3.5), layout="constrained")
    axis = figure.subplots()
    labels, descriptions = [], []
    for unit in pilots:
        summary = unit["summary"]
        grouped = _evaluations(summary, "pilot")
        difficulties = list(dict.fromkeys(d for _, d in grouped)) or ["easy", "hard"]
        for difficulty in difficulties:
            x = len(labels)
            before, after = (grouped.get((c, difficulty), []) for c in ("initial", "final"))
            paired, identity = _pairing(before, after)
            label = f"{_label(summary, 'pilot')} / {summary.get('seed', '?')}\n{difficulty}"
            labels.append(label)
            descriptions.append(f"{label.replace(chr(10), ' ')}: {identity}; {_rate(before)} to {_rate(after)}")
            complete = before and after and all(type(r.get("success")) is bool for r in before + after)
            if paired and complete:
                rates = [sum(r["success"] for r in rows) / len(rows) for rows in (before, after)]
                axis.plot([x - .13, x + .13], rates, color="#777777", linewidth=1)
                for offset, rate, rows, color in zip((-.13, .13), rates, (before, after), ("#347aa3", "#31845e")):
                    axis.scatter(x + offset, rate, color=color, s=30)
                    axis.text(x + offset, rate + .04, f"{sum(r['success'] for r in rows)}/{len(rows)}",
                              ha="center", fontsize=8)
            else:
                axis.text(x, .08, "unverified" if not paired else "outcomes pending", ha="center", fontsize=8, rotation=90)
    axis.set(xticks=range(len(labels)), xticklabels=labels, ylim=(-.02, 1.16),
             ylabel="Fixed-panel success", title="Initial (blue) to final (green), verified seed pairs only")
    axis.tick_params(axis="x", labelsize=8)
    axis.yaxis.set_major_formatter(PercentFormatter(1.))
    axis.grid(axis="y", alpha=.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    figure.clear()
    return nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode(),
        "text/plain": "Paired student success. " + "; ".join(descriptions)})


def _videos(unit, campaign, notebook_dir):
    import nbformat

    summary = unit["summary"]
    links, captions = [], []
    for video in _rows(summary.get("videos")):
        controller = video.get("controller", video.get("actor", summary.get("controller", summary.get("actor"))))
        label = f"controller={_text(controller)}; " + "; ".join(
            f"{key}={_text(video.get(key, summary.get(key)))}" for key in ("checkpoint", "difficulty"))
        reference = video.get("path")
        path = None
        if isinstance(reference, str) and reference:
            relative = Path(reference)
            candidates = [relative] if relative.is_absolute() else [Path(unit["path"]) / relative, campaign / relative]
            path = next((p for p in candidates if p.resolve().is_relative_to(campaign.resolve()) and p.is_file()), None)
        captions.append(label + (": " + PENDING if path is None else ""))
        links.append(html.escape(label) + ": " + PENDING if path is None else
                     f'<a href="{quote(os.path.relpath(path, notebook_dir))}">{html.escape(label)}</a>')
    return nbformat.v4.new_output("display_data", data={
        "text/html": "<p>Recorded controller clips (not the old scripted-controller demos):<br>"
                     + "<br>".join(links or [PENDING]) + "</p>",
        "text/plain": "Recorded controller clips, distinct from old scripted demos: " + "; ".join(captions or [PENDING])})


def _status(campaign, data):
    if data["progress"].get("stopped") or (campaign / "STOP").exists():
        return "stopped"
    final = next((u["summary"] for u in reversed(data["units"]) if _kind(u) == "campaign"), None)
    return final.get("status", "completed") if final is not None else data["progress"].get("status", PENDING)


def _feature_health(summary, label):
    import nbformat
    from matplotlib.figure import Figure

    row = {"Health unit": label, "Normalization": summary.get("normalization", "none"),
        "Median |z|": summary.get("abs_preactivation_median"),
        "|tanh(z)| > .99": summary.get("saturated_fraction"),
        "Median derivative": summary.get("derivative_median"), "Max derivative": summary.get("derivative_max")}
    if summary.get("eps") is not None:
        row["epsilon"] = summary["eps"]
    outputs = [_table([row])]
    distributions = _mapping(summary.get("distributions"))
    if not distributions:
        return outputs + [_note("Projection-health distributions: " + PENDING)]
    figure = Figure(figsize=(7, 2.5), layout="constrained")
    for axis, key, label in zip(figure.subplots(1, 2), ("preactivation", "derivative"), ("Preactivation z", "1 - tanh(z)^2")):
        histogram = _mapping(distributions.get(key))
        edges, counts = histogram.get("edges"), histogram.get("counts")
        if edges is None and counts is None:
            axis.text(.5, .5, PENDING, ha="center", transform=axis.transAxes, fontsize=8)
        else:
            if (not isinstance(edges, list) or not isinstance(counts, list) or not counts
                    or len(edges) != len(counts) + 1 or not all(_number(v) for v in edges + counts)
                    or any(v < 0 for v in counts) or any(a >= b for a, b in zip(edges, edges[1:]))):
                raise ValueError("incompatible feature-health histogram")
            axis.stairs(counts, edges, fill=True, color="#347aa3" if key == "preactivation" else "#31845e")
        axis.set(xlabel=label, ylabel="Projection coordinates")
        axis.grid(axis="y", alpha=.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    figure.clear()
    outputs.append(nbformat.v4.new_output("display_data", data={
        "image/png": base64.b64encode(buffer.getvalue()).decode(),
        "text/plain": "Archived last-frame projection health: local tanh slope, not a full Jacobian; saturation is a hypothesis, not causal proof."}))
    return outputs


def phase2_outputs(snapshot, *, notebook_dir=None):
    """Produce small tables/figures from validated scalar artifacts only."""
    campaigns = _campaigns(snapshot)
    if not campaigns:
        return [_note("Phase 2 pending: no sealed units or orchestration progress.")]
    ordered = sorted(campaigns.items(), key=_order)
    outputs = []
    for campaign, data in ordered:
        units, progress = data["units"], data["progress"]
        final = next((u["summary"] for u in reversed(units) if _kind(u) == "campaign"), {})
        outputs.append(_note(f"Phase 2 campaign: {campaign.name}; revisions are not pooled."))
        outputs.append(_table([{"Status": _status(campaign, data), "Sealed units": len(units),
            "Progress": progress.get("counts"), "Active unit": progress.get("active_unit"),
            "Active seconds": final.get("total_active_seconds", final.get("phase2_active_seconds",
                progress.get("active_seconds", progress.get("total_active_seconds")))),
            "Remaining estimate (s)": progress.get("remaining_estimate_seconds"), "Settings": _settings(progress.get("settings"))}]))
        for unit in units:
            if _kind(unit) == "feature_health":
                outputs.extend(_feature_health(unit["summary"], Path(unit["path"]).name))
        if campaign != ordered[-1][0]:
            continue
        by_kind = defaultdict(list)
        for unit in units:
            by_kind[_kind(unit)].append(unit)
        assets = _asset_rows([u for kind in ("initial_asset", "bootstrap") for u in by_kind[kind]])
        outputs.append(_table(assets) if assets else _note("Initial asset / bootstrap provenance: " + PENDING))
        for unit in by_kind["reference_baselines"]:
            fields = ("model", "seed", "easy_initial", "easy_final", "hard_initial", "hard_final",
                      "source", "new_trajectory", "charged_again")
            references = [_scalars(row, fields) for row in _rows(unit["summary"].get("rows"))]
            outputs.append(_note("Historical baselines shown for comparison: not new runs and not pooled with rescue trajectories."))
            outputs.append(_table(references) if references else _note("Historical baseline outcomes: " + PENDING))
        outcomes = [row for kind in ("teacher_readiness", "initial_readiness", "pilot")
                    for u in by_kind[kind] for row in _outcome_rows(u)]
        outputs.append(_outcome_table(outcomes) if outcomes else _note("Teacher / initial student / pilot outcomes: " + PENDING))
        pilots = by_kind["pilot"]
        recipes, costs = [], []
        for unit in pilots:
            summary = unit["summary"]
            settings = _mapping(summary.get("settings"))
            recipes.append({"Student seed": summary.get("seed"), "Recipe": settings.get("name"),
                "Settings": _settings(settings),
                "Episodes": _count(summary.get("episodes")), "New acquired": summary.get("acquired_count"),
                "Initialization": "receipt archived" if _mapping(summary.get("distill_receipt")) else None})
            costs.append({"Student seed": summary.get("seed"), "Recipe": settings.get("name"),
                "Online updates": summary.get("online_updates"), "Sleep updates": summary.get("sleep_updates"),
                "Fallback counts": _scalars(summary.get("fallback_counts"), _mapping(summary.get("fallback_counts"))) or None,
                "Timings (s)": _scalars(summary.get("timings"), ("active_seconds", "evaluation_seconds", "fit_seconds",
                    "online_seconds", "fisher_seconds", "scoring_seconds", "checkpoint_seconds", "sleep_seconds")) or None})
        outputs.extend([_table(recipes), _table(costs), _plot(pilots)] if pilots else [_note("Student candidates / costs / paired plot: " + PENDING)])
        outputs.append(_note("Success/collision counts are descriptive fixed-layout observations, not independent training "
            "trajectories or significance tests. Missing outcomes are unknown, not failures; terminal geometry is optional. "
            "The 70% teacher / 60% student / within-15-point easy-A guides are engineering guides, not significance thresholds."))
        for kind in ("teacher_readiness", "initial_readiness", "pilot"):
            for unit in by_kind[kind]:
                outputs.append(_videos(unit, campaign, notebook_dir or _NOTEBOOK_DIR))
        if by_kind["language"]:
            outputs.append(_note("Language responsiveness: four instruction means, not proof of instruction following; raw histories remain archived."))
            for unit in by_kind["language"]:
                outputs.append(_table(_language_rows(_rows(unit["summary"].get("rows")))))
        else:
            outputs.append(_note("Language responsiveness: " + PENDING))
        for unit in by_kind["forecast"]:
            forecast = unit["summary"]
            scopes = [_scalars(row, ("scope", "pairs", "observations_per_arm", "transitions_per_arm",
                "fd_cost_multiplier_scenario", "training_processing_hours", "evaluation_hours_at_pilot_panel_size",
                "total_hours", "not_total_study_forecast")) for row in _rows(forecast.get("scope_rows"))]
            outputs.append(_table(scopes) if scopes else _note("Fresh-run scope forecast: " + PENDING))
            costs = _forecast_rows(forecast)
            outputs.append(_table(costs) if costs else _note("Forecast costs / readiness: " + PENDING))
        if not by_kind["forecast"]:
            outputs.append(_note("Forecast / FD overhead / readiness conclusions: " + PENDING))
        extras = []
        for unit in units:
            kind, summary = _kind(unit), unit["summary"]
            if kind in {"initial_asset", "bootstrap", "reference_baselines", "teacher_readiness", "initial_readiness", "pilot", "language", "feature_health", "forecast"}:
                continue
            fields = ("status", "trainability_observed", "next_phase_authorized", "total_active_seconds",
                      "passed", "test_result", "archive_sha256")
            extras.append({"Unit": kind, "Recorded": _scalars(summary, fields) or "sealed; details archived"})
        if extras:
            outputs.append(_table(extras))
        outputs.append(_note("Forecasts are conditional operational estimates, not FD or sample-efficiency measurements. "
            "FD multipliers are scenarios, not measurements; later development and closeout cost extra. "
            "Readiness completion does not authorize Phase 3 or physical operation; stop for the resource/task check-in."))
    return outputs


def display_phase2(snapshot, *, notebook_dir=None):
    from IPython.display import display

    for output in phase2_outputs(snapshot, notebook_dir=notebook_dir or Path.cwd()):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_phase2(book, snapshot, notebook_dir=None):
    """Insert/replace a concise section after Fix 3 and before Development."""
    import nbformat

    book.cells = [c for c in book.cells if not c.metadata.get("plan19_phase2")]
    if not _campaigns(snapshot):
        return book
    directory = str(Path(notebook_dir).resolve()) if notebook_dir is not None else None
    code = "from numerical_experiment.picar.phase2_report import display_phase2\n"
    code += f"display_phase2(snapshot, notebook_dir={directory!r})"
    section = [nbformat.v4.new_markdown_cell(_MOTIVATION), nbformat.v4.new_code_cell(
        code, outputs=phase2_outputs(snapshot, notebook_dir=notebook_dir))]
    for i, cell in enumerate(section):
        cell.id = f"plan19-phase2-{i}"
        cell.metadata["plan19_phase2"] = True
    index = next((i for i, c in enumerate(book.cells) if c.cell_type == "markdown"
                  and c.source.startswith(("## Development", "## Conclusion"))), len(book.cells))
    book.cells[index:index] = section
    return book


def conclusion(snapshot):
    """Summarize only the latest campaign, never promote completion to a pass."""
    campaigns = _campaigns(snapshot)
    if not campaigns:
        return None
    campaign, data = max(campaigns.items(), key=_order)
    units = data["units"]
    completed = any(_kind(u) == "campaign" for u in units)
    state = "complete as an artifact campaign" if completed and _status(campaign, data) != "stopped" else "pending or stopped"
    lines = ["| Controller / seed | Level | Before | After | Pairing |", "|---|---|---|---|---|"]
    assessment, fd = PENDING, PENDING
    interpretation = []
    resources, closeout_seconds = None, 0.

    def add(values):
        lines.append("| " + " | ".join(_compact(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |")

    for unit in units:
        kind = _kind(unit)
        summary = unit["summary"]
        if kind == "teacher_readiness":
            add(("Frozen Qwen", "easy", _rate(_rows(summary.get("rows"))), "n/a", "single checkpoint"))
        elif kind == "initial_readiness":
            for (_, difficulty), rows in _evaluations(summary, kind).items():
                add(("Initial student / " + _label(summary, kind), difficulty, _rate(rows), "n/a", _coverage(rows, difficulty)))
        elif kind == "pilot":
            grouped = _evaluations(summary, kind)
            for difficulty in dict.fromkeys(d for _, d in grouped):
                before, after = (grouped.get((c, difficulty), []) for c in ("initial", "final"))
                add((f"{_label(summary, kind)} / {summary.get('seed', '?')}", difficulty,
                     _rate(before), _rate(after), _pairing(before, after)[1]))
        elif kind == "forecast":
            assessment = _messages(summary.get("readiness_conclusions")) or PENDING
            if assessment == PENDING and type(summary.get("trainability_observed")) is bool:
                assessment = "common-student trainability observed: " + ("yes" if summary["trainability_observed"] else "no")
            fd = _text(summary.get("fd_unmeasured"))
            if summary.get("trainability_observed") is False:
                interpretation.append("**Readiness:** No completed candidate met the operational hard-panel "
                    "improvement guide. Common-controller learning remains unresolved; a larger FD comparison "
                    "is not yet justified by these results.")
            resources = [summary.get(key) for key in ("phase2_active_seconds", "study_charged_seconds",
                                                     "remaining_to_20h_seconds")]
        elif kind == "closeout":
            resources = [summary.get(key) for key in ("phase2_active_seconds", "study_charged_seconds",
                                                     "remaining_to_20h_seconds")]
            closeout_seconds = summary.get("closeout_reserve_seconds", 0.)
        elif kind == "feature_health" and summary.get("normalization") == "feature_and_projection_rms":
            if summary.get("saturated_fraction") == 0 and _number(summary.get("derivative_median")) \
                    and summary["derivative_median"] > 0:
                interpretation.append("**Conditioning:** Projection RMS produced unsaturated initialization "
                    "probes. This is a local sensitivity check, not evidence of correct instruction following "
                    "or navigation learning.")
    if resources is not None and all(_number(value) and value >= 0 for value in resources):
        allowance = (f" Includes {closeout_seconds/60:.0f} min conservative closeout allowance."
                     if _number(closeout_seconds) and closeout_seconds > 0 else "")
        interpretation.append(f"**Resources:** Phase 2 charged {resources[0]/3600:.2f} h; "
            f"the conservative study charge is {resources[1]/3600:.2f} h, leaving "
            f"{resources[2]/3600:.2f} h of the 20 h target." + allowance
            + (" FD overhead remains unmeasured." if fd == "True" else ""))
    table = "\n".join(lines) + "\n\n" if len(lines) > 2 else ""
    return (f"## Conclusion\n\nPhase 2 (`{campaign.name}`) is {state} (status: {_status(campaign, data)}); "
            f"completion alone is not a readiness pass (recorded assessment: {assessment}; FD overhead unmeasured: {fd}).\n\n"
            + table + ("\n\n".join(interpretation) + "\n\n" if interpretation else "")
            + "**Next check-in:** review A/B skill, language, provenance and cost with missing evidence unverified; "
            "these descriptive panels are not independent training trajectories, frozen Qwen need not solve the hard task, "
            "and there is no automatic Phase 3 or FD efficacy claim.")
