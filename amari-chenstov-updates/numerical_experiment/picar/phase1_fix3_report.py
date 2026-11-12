"""Read-only Fix 3 notebook cells from sealed JSON/CSV and optional RGB media.

Main integration: append_fix3(book, snapshot), conclusion(snapshot).
Units use the existing RunStore completion schema and kinds phase1_fix3_*,
not live summaries. The kind suffix selects geometry, rewards, teacher,
bootstrap or pilot. Campaign progress/settings come from
fix3*/orchestration/progress.json. Missing optional evidence stays pending.
Geometry uses lateral_histogram_edges_m and row lateral_density_per_m, at
rows 0/3/-1. Legacy density rows may be long (y, exposure, density) or wide.
Alpha is read from geometry rows or alpha_schedule rows, never inferred from
sigma. Media paths are local to the campaign; sha256 frame refs use frames/.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


PENDING = "pending / not recorded"
_PREFIX = "phase1_fix3_"
_NOTEBOOK_DIR = Path(__file__).parent
_REQUIRED = ("geometry", "rewards", "teacher", "bootstrap", "pilot")


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _text(value):
    if value is None:
        return PENDING
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _compact(value):
    if isinstance(value, (dict, list, tuple)):
        return f"{len(value)} archived entries"
    text = _text(value)
    return text if len(text) <= 360 else text[:357] + "..."


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _kind(unit):
    kind = unit["summary"]["kind"][len(_PREFIX):]
    return "geometry" if kind == "geometry_audit" else kind


def _count(value):
    if type(value) is int and value >= 0:
        return value
    return len(value) if isinstance(value, (list, tuple)) else None


def _teacher_rows(summaries):
    grouped = defaultdict(list)
    for row in summaries:
        grouped[_compact(row.get("variant"))].append(row)
    result = []
    for variant, rows in sorted(grouped.items()):
        counts, actions, unknown_actions = [], set(), False
        for row in rows:
            commands = row.get("commands")
            count = _count(commands)
            counts.append(count)
            recorded = _mapping(row.get("action_counts"))
            if recorded:
                actions.update(str(k) for k, n in recorded.items() if _number(n) and n > 0)
            elif isinstance(commands, list):
                for command in commands:
                    action = command.get("action") if isinstance(command, dict) else command
                    if type(action) in (str, int):
                        actions.add(str(action))
                    else:
                        unknown_actions = True
            else:
                unknown_actions = True
        reasons = Counter(str(r["end_reason"]) for r in rows if isinstance(r.get("end_reason"), str))
        result.append({"Variant": variant, "Episodes": len(rows), "Success count / rate": _rate(rows, "success"),
                       "Commands total": sum(counts) if all(n is not None for n in counts) else None,
                       "Unique actions": f"{len(actions)} known; coverage pending" if unknown_actions else len(actions),
                       "End reasons": "; ".join(f"{k}: {n}" for k, n in reasons.most_common(3)) or None})
    return result


def _bootstrap_row(summary):
    distillation = _mapping(summary.get("distillation") or summary.get("distill_receipt"))
    def kl(key, alias):
        value = distillation.get(key, distillation.get(alias))
        if value is None and key == "final_kl":
            value = distillation.get("last_preupdate_kl")
        return value if _number(value) else None
    return {"Transitions": _count(summary.get("transitions")),
            "Distillation initial KL": kl("initial_kl", "kl_initial"),
            "Distillation final KL": kl("final_kl", "kl_final")}


def _settings_row(units, progress, geometry, final):
    sources = [_mapping(progress.get("settings")), _mapping(geometry.get("simulator_settings")),
               _mapping(geometry.get("settings"))]
    sources.extend(_mapping(u.get("stored_config", {}).get("config")) for u in reversed(units))
    def setting(key, section=None):
        for source in sources:
            value = _mapping(source.get(section)).get(key) if section else source.get(key)
            if value is not None:
                return value
        return None
    ramp = setting("exposure_ramp")
    if isinstance(ramp, (list, tuple)) and len(ramp) == 2 and all(_number(x) for x in ramp):
        ramp = f"{ramp[0]}--{ramp[1]} intervals"
    elif not isinstance(ramp, str):
        total = setting("curriculum_total_intervals")
        ramp = f"{total} total intervals" if _number(total) else None
    return {"Context": setting("context_length", "model") or setting("context_length"),
            "Replay capacity": setting("replay_capacity", "learning") or setting("replay_capacity"),
            "Return bins": final.get("return_bins", setting("return_bins", "model") or setting("return_bins")),
            "Curriculum ramp": ramp}


def _blockers(final, progress):
    value = final.get("blockers", progress.get("blockers"))
    if isinstance(value, list):
        messages = [v for v in value if isinstance(v, str)]
        return "; ".join(_compact(v) for v in messages[:3]) + (
            f"; {len(messages) - 3} more archived" if len(messages) > 3 else "") if messages else "none recorded"
    return _compact(value)


def _campaign(path, root):
    for parent in (path, *path.parents):
        if parent.name.startswith("fix3"):
            return parent
        if parent == root:
            break
    return root


def _progress(path):
    from .report import _json, _version

    value = _json(path)
    if "schema_version" in value:
        _version(value["schema_version"], 1, "Fix 3 progress schema")
    return value


def _load(root):
    from .report import _json, read_completed_unit

    campaigns = {}

    def group(path):
        campaign = _campaign(path, root)
        return campaigns.setdefault(campaign, {"units": [], "progress": {}})

    for path in sorted(root.rglob("orchestration/progress.json")):
        if _campaign(path.parent, root).name.startswith("fix3"):
            group(path.parent)["progress"] = _progress(path)
    for marker in sorted(root.rglob("COMPLETED")):
        summary_path = marker.parent / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _json(summary_path)
        if str(summary.get("kind", "")).startswith(_PREFIX):
            group(marker.parent)["units"].append(read_completed_unit(marker.parent))
    for value in campaigns.values():
        value["units"].sort(key=lambda u: (str(u["manifest"].get("completed_at", "")), u["path"]))
    return campaigns


def _snapshot_campaigns(snapshot):
    """Use the parent's validated units, never live scientific summaries."""
    root = Path(snapshot.get("root", "/")).expanduser().resolve()
    campaigns = {}

    def group(path):
        campaign = _campaign(Path(path).resolve(), root)
        return campaigns.setdefault(campaign, {"units": [], "progress": {}})

    for unit in snapshot.get("complete_units", []):
        if str(unit.get("summary", {}).get("kind", "")).startswith(_PREFIX):
            group(unit["path"])["units"].append(unit)
    for live in snapshot.get("live", []):
        if _campaign(Path(live["path"]).resolve(), root).name.startswith("fix3"):
            group(live["path"])["progress"].update(live)
    if snapshot.get("root"):
        for path in sorted(root.rglob("orchestration/progress.json")):
            if _campaign(path.parent, root).name.startswith("fix3"):
                group(path.parent)["progress"].update(_progress(path))
    for data in campaigns.values():
        data["units"].sort(key=lambda u: (str(u.get("manifest", {}).get("completed_at", "")), u["path"]))
    return campaigns


def _table(rows):
    import nbformat
    import pandas as pd

    frame = pd.DataFrame([{key: _compact(value) for key, value in row.items()} for row in rows])
    return nbformat.v4.new_output("display_data", data={
        "text/plain": frame.to_string(index=False), "text/html": frame.to_html(index=False, escape=True)})


def _note(text):
    import nbformat

    return nbformat.v4.new_output("stream", name="stdout", text=text + "\n")


def _figure(figure, description):
    import nbformat

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    figure.clear()
    return nbformat.v4.new_output("display_data", data={
        "image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": description})


def _geometry(summary):
    from matplotlib.figure import Figure

    rows, densities = _rows(summary.get("rows")), _rows(summary.get("densities"))
    figure = Figure(figsize=(11, 3.1), layout="constrained")
    density, blockage, alpha = figure.subplots(1, 3)
    curves = defaultdict(list)
    edges = summary.get("lateral_histogram_edges_m", [])
    if isinstance(edges, list) and len(edges) >= 2 and all(_number(x) for x in edges):
        for index in dict.fromkeys((0, min(3, len(rows) - 1), len(rows) - 1)) if rows else ():
            row = rows[index]
            values = row.get("lateral_density_per_m")
            if isinstance(values, list) and len(values) == len(edges) - 1 and all(_number(x) for x in values):
                density.stairs(values, edges, label=f"n={_text(row.get('exposure'))}; sigma={_text(row.get('sigma'))}")
    for row in densities:
        if not _number(row.get("y")):
            continue
        if _number(row.get("density")):
            curves[_text(row.get("exposure", row.get("stage")))].append((row["y"], row["density"]))
        else:
            for key, value in row.items():
                if (key in ("early", "middle", "late", "easy", "hard") or "density" in key) and _number(value):
                    curves[key].append((row["y"], value))
    for label, points in curves.items():
        points.sort()
        density.plot(*zip(*points), label=label)
    for field, label in (("clear_fraction", "Clear"), ("blocked_fraction", "Blocked")):
        points = sorted((r["exposure"], r[field]) for r in rows
                        if _number(r.get("exposure")) and _number(r.get(field)))
        if points:
            blockage.plot(*zip(*points), "o-", label=label)
    alpha_rows = _rows(summary.get("alpha_schedule")) or rows
    points = sorted((r["exposure"], r["alpha"]) for r in alpha_rows
                    if _number(r.get("exposure")) and _number(r.get("alpha")))
    if points:
        alpha.plot(*zip(*points), "o-", label="Texture/gain alpha")
    for axis, title, xlabel, ylabel in (
        (density, "Barrier placement density", "Lateral y (m)", "Density"),
        (blockage, "Accepted-layout geometry", "Captured training intervals", "Fraction"),
        (alpha, "Separate texture/gain schedule", "Captured training intervals", "Alpha"),
    ):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
        if axis.lines or axis.patches:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, PENDING, ha="center", transform=axis.transAxes, fontsize=8)
    blockage.set_ylim(-.02, 1.02)
    return _figure(figure, "Placement density; clear/blocked fractions on accepted layouts; separate texture/gain alpha.")


def _local_path(reference, unit, campaign):
    if not isinstance(reference, str) or not reference:
        return None
    relative = Path(reference)
    candidates = [relative] if relative.is_absolute() else [Path(unit["path"]) / relative, campaign / relative]
    for candidate in candidates:
        if candidate.resolve().is_relative_to(campaign.resolve()) and candidate.is_file():
            return candidate
    return None


def _frame(row, unit, campaign, units):
    reference = row.get("frame_ref")
    if isinstance(reference, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", reference):
        paths = [Path(u["path"]) / "frames" / (reference[7:] + ".npy") for u in [unit, *units]]
        path = next((p for p in paths if p.is_file() and p.resolve().is_relative_to(campaign.resolve())), None)
        if path is not None and (path.stat().st_size > 32_000_000
                                 or hashlib.sha256(path.read_bytes()).hexdigest() != reference[7:]):
            return None
    else:
        path = _local_path(reference, unit, campaign)
    if path is None or path.stat().st_size > 32_000_000:
        return None
    import matplotlib.image as mpimg
    import numpy as np

    try:
        if path.suffix == ".npy":
            rgb = np.load(path, allow_pickle=False)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                return None
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            rgb = mpimg.imread(path)
        else:
            return None
        buffer = io.BytesIO()
        mpimg.imsave(buffer, rgb, format="png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    except (OSError, ValueError, SyntaxError, EOFError):
        return None


def _rewards(unit, campaign, units):
    import nbformat

    rows = _rows(unit["summary"].get("rows"))
    if not rows:
        return [_note("Reward contact sheet: " + PENDING)]
    wanted = ("middle-center", "middle-left", "middle-edge", "pan-centered", "close", "absent")
    selected = [next((r for r in rows if r.get("state") == name), None) for name in wanted]
    selected = [r for r in selected if r is not None] or rows[:6]
    columns = ("split", "state", "seed", "proximity", "centering", "visible", "pan_rad", "centering_bonus", "score")
    images = []
    for row in selected:
        frame = _frame(row, unit, campaign, units)
        caption = "; ".join(f"{key}={_text(row.get(key))}" for key in columns)
        media = f'<img src="{frame}" style="max-width:220px" alt="Recorded reward panel">' if frame else PENDING
        images.append(f'<td style="vertical-align:top;max-width:230px">{media}<br>{html.escape(caption)}</td>')
    grid = "".join("<tr>" + "".join(images[i:i+3]) + "</tr>" for i in range(0, len(images), 3))
    sheet = nbformat.v4.new_output("display_data", data={
        "text/html": "<table>" + grid + "</table>",
        "text/plain": "Reward contact sheet: " + "; ".join(
            "; ".join(f"{key}={_text(row.get(key))}" for key in columns) for row in selected)
            + ". Unavailable frames remain pending."})
    return [sheet]


def _rate(rows, field):
    observed = []
    for row in rows:
        value = row.get(field)
        if field == "collision" and type(value) is not bool and row.get("end_reason"):
            reason = row["end_reason"]
            if reason in ("collision", "obstacle_collision", "wall_collision", "obstacle_contact", "wall_contact"):
                value = True
            elif reason in ("success", "target_contact", "timeout", "time_limit"):
                value = False
        if type(value) is bool:
            observed.append(value)
    if not observed:
        return PENDING
    missing = f"; {len(rows) - len(observed)} unknown" if len(observed) != len(rows) else ""
    return f"{sum(observed)}/{len(observed)} ({sum(observed) / len(observed):.1%}){missing}"


def _evaluations(summary):
    grouped = defaultdict(list)
    for evaluation in _rows(summary.get("evaluations")):
        grouped[(evaluation.get("checkpoint"), evaluation.get("difficulty"))].extend(_rows(evaluation.get("rows")))
    def pairing(difficulty):
        identities = []
        for checkpoint in ("initial", "final"):
            rows = grouped[(checkpoint, difficulty)]
            ids = [r.get("layout_id", r.get("panel_id")) for r in rows]
            if not ids or any(value is None for value in ids):
                return "pairing unverified"
            identities.append(sorted(map(str, ids)))
        return "matched layout IDs" if identities[0] == identities[1] else "layout IDs differ"

    return [{"Seed": summary.get("seed"), "Checkpoint": checkpoint, "Difficulty": difficulty,
             "Episodes": len(grouped[(checkpoint, difficulty)]),
             "Success count / rate": _rate(grouped[(checkpoint, difficulty)], "success"),
             "Collision count / rate": _rate(grouped[(checkpoint, difficulty)], "collision"),
             "Panel identity": pairing(difficulty)}
            for difficulty in ("easy", "middle", "hard") for checkpoint in ("initial", "final")]


def _pending_evaluations(units):
    for unit in units:
        if _kind(unit) == "pilot":
            for row in _evaluations(unit["summary"]):
                if any(row[key] == PENDING or "unknown" in row[key] for key in (
                        "Success count / rate", "Collision count / rate")):
                    return True
    return False


def _videos(summary, unit, campaign, notebook_dir):
    import nbformat

    videos = _rows(summary.get("videos"))
    lines = []
    for checkpoint in ("initial", "final"):
        matching = [v for v in videos if v.get("checkpoint") == checkpoint]
        if not matching:
            lines.append(f"{checkpoint}: {PENDING}")
        for video in matching:
            label = "; ".join(f"{key}={_text(video.get(key))}" for key in ("checkpoint", "layout_id", "view"))
            path = _local_path(video.get("path"), unit, campaign)
            if path is None:
                lines.append(html.escape(label) + ": " + PENDING)
            else:
                href = quote(os.path.relpath(path, notebook_dir))
                lines.append(f'<a href="{href}">{html.escape(label)}</a>')
    return nbformat.v4.new_output("display_data", data={
        "text/html": "<p>Before/after clips (common-layout and synchronization identity must be recorded):<br>"
                     + "<br>".join(lines) + "</p>",
        "text/plain": "Before/after clips: " + "; ".join(re.sub("<[^>]*>", "", line) for line in lines)})


def outputs(root: Path):
    """Render completed evidence only; do not write, repair, or load checkpoints."""
    return _outputs(_load(Path(root).expanduser().resolve()))


def fix3_outputs(snapshot, *, notebook_dir=None):
    """Render the parent's validated artifact snapshot without loading models."""
    return _outputs(_snapshot_campaigns(snapshot), notebook_dir=notebook_dir)


def _outputs(campaigns, *, notebook_dir=None):
    result = []
    if not campaigns:
        return [_note("Phase 1 Fix 3 pending: no orchestration progress or completed unit summaries.")]
    for campaign, data in sorted(campaigns.items()):
        units, progress = data["units"], data["progress"]
        final = next((u["summary"] for u in reversed(units) if _kind(u) == "campaign"), {})
        counts = _mapping(progress.get("counts"))
        pending = progress.get("pending_units", counts.get("pending"))
        if pending is None and all(type(counts.get(k)) is int for k in ("step", "target")):
            pending = max(0, counts["target"] - counts["step"])
        geometry_unit = next((u for u in reversed(units) if _kind(u) == "geometry"), {})
        result.append(_note(f"Campaign: {campaign.name} (revisions are not pooled)."))
        result.append(_table([{
            "Status": "stopped" if progress.get("stopped") or (campaign / "STOP").exists() else
                progress.get("status", final.get("status")),
            "Completed units (sealed)": len(units), "Pending units": _count(pending),
            "Active compute (s)": final.get("total_active_seconds", progress.get("active_seconds",
                progress.get("total_active_seconds"))),
            "Cumulative repair (s)": progress.get("cumulative_repair_seconds", final.get("cumulative_repair_seconds")),
            "Remaining estimate (s)": progress.get("remaining_estimate_seconds", progress.get("remaining_seconds")),
        }]))
        result.append(_table([_settings_row(units, progress, geometry_unit.get("summary", {}), final)]))
        by_kind = defaultdict(list)
        for unit in units:
            by_kind[_kind(unit)].append(unit)
        geometry = by_kind["geometry"]
        result.append(_note("Geometry density/blockage and separate alpha schedule"))
        if geometry:
            summary = geometry[-1]["summary"]
            result.append(_geometry(summary))
            rows = _rows(summary.get("rows"))
            if len(rows) <= 3:
                result.append(_table([{key: row.get(key) for key in (
                    "exposure", "sigma", "clear_fraction", "blocked_fraction")}
                    for row in rows]) if rows else _note("Geometry rows: " + PENDING))
            result.append(_note("Blockage describes accepted layouts, not task success. Returning texture/gain alpha to A "
                                "does not restore the initial easy placement distribution."))
        else:
            result.append(_note("Geometry audit: " + PENDING))
        result.append(_note("Reward components and contact sheet (pan in radians)"))
        result.append(_table([{"Split": u["summary"].get("split"),
            "Images": len(_rows(u["summary"].get("rows"))),
            "Centering order fraction": u["summary"].get("centering_order_fraction"),
            "Approach order fraction": u["summary"].get("approach_order_fraction"),
            "Absent ball scores zero": u["summary"].get("absent_zero")}
            for u in by_kind["rewards"]]) if by_kind["rewards"] else _note("Rewards: " + PENDING))
        if by_kind["rewards"]:
            result.extend(_rewards(by_kind["rewards"][-1], campaign, units))
        if not by_kind["rewards"]:
            result.append(_note("Rewards: " + PENDING))
        teacher = [u["summary"] for u in by_kind["teacher"]]
        result.append(_note("Frozen teacher results (not student learning)"))
        result.append(_table(_teacher_rows(teacher)) if teacher else _note("Teacher: " + PENDING))
        bootstrap = [u["summary"] for u in by_kind["bootstrap"]]
        result.append(_note("Easy bootstrap and one-time student distillation"))
        result.append(_table([_bootstrap_row(row) for row in bootstrap]) if bootstrap else _note("Bootstrap: " + PENDING))
        result.append(_note("Student pilot: fixed-panel pre/post evaluation, separately by difficulty and seed"))
        for unit in by_kind["pilot"]:
            summary = unit["summary"]
            result.append(_table([{"Seed": summary.get("seed"), "Acquired transitions": _count(summary.get("acquired_count")),
                                   "Training episodes": _count(summary.get("episodes"))}]))
            result.append(_table(_evaluations(summary)))
            result.append(_videos(summary, unit, campaign, notebook_dir or _NOTEBOOK_DIR))
        if not by_kind["pilot"]:
            result.append(_note("Student pilot / fixed-panel evaluations / before-after clips: " + PENDING))
        result.append(_note("Fixed panels must retain identical layouts across checkpoints; missing layout IDs leave pairing "
                            "unverified. Training success mixes learning with changing difficulty and is not an evaluation rate."))
        missing = [kind for kind in _REQUIRED if not by_kind[kind]]
        if _pending_evaluations(units):
            missing.append("complete fixed-panel outcomes")
        result.append(_note("Readiness: " + ("pending " + ", ".join(missing) if missing else
                            "diagnostic units recorded; completion alone does not establish student learning")
                            + ". Unresolved issues: " + _blockers(final, progress)
                            + ". No STM efficacy or physical-operation claim; stop at the resource/task check-in."))
    return result


def display_fix3(snapshot):
    from IPython.display import display

    rendered = fix3_outputs(snapshot, notebook_dir=Path.cwd()) if isinstance(snapshot, dict) else outputs(snapshot)
    for output in rendered:
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def cells(root: Path) -> list:
    """Return independently refreshable nbformat cells for the parent report."""
    import nbformat

    root = Path(root).expanduser().resolve()
    result = [nbformat.v4.new_markdown_cell(_MOTIVATION),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nfrom numerical_experiment.picar.phase1_fix3_report import display_fix3\n"
            f"display_fix3(Path({str(root)!r}))", outputs=outputs(root))]
    return _tag(result)


_MOTIVATION = (
        "### Phase 1 Fix 3: Centering Rewards and Gradual Navigation\n\n"
        "Fix 2 yielded useful straight approaches but no obstacle-task successes; recognition did not imply reliable "
        "spatial guidance. Fix 3 adds a small pan-discounted centering bonus and a broad-to-concentrated barrier "
        "curriculum. The teacher and evaluator remain frozen: easy demonstrations support one-time student "
        "distillation, then the student learns from its own experience. These bounded diagnostics are not an STM "
        "comparison. Only sealed completed units enter results; live progress is operational information.")


def _tag(result):
    for index, cell in enumerate(result):
        cell.id = f"plan19-fix3-{index}"
        cell.metadata["plan19_fix3"] = True
    return result


def append_fix3(book, snapshot, *, notebook_dir=None):
    """Insert/replace the Fix 3 section before Development; never write a file."""
    import nbformat

    if not _snapshot_campaigns(snapshot):
        return book
    section = _tag([nbformat.v4.new_markdown_cell(_MOTIVATION), nbformat.v4.new_code_cell(
        "from numerical_experiment.picar.phase1_fix3_report import display_fix3\ndisplay_fix3(snapshot)",
        outputs=fix3_outputs(snapshot, notebook_dir=notebook_dir))])
    book.cells = [cell for cell in book.cells if not cell.metadata.get("plan19_fix3")]
    index = next((i for i, cell in enumerate(book.cells) if cell.cell_type == "markdown"
                  and cell.source.startswith(("## Development", "## Conclusion"))), len(book.cells))
    book.cells[index:index] = section
    return book


def conclusion(snapshot):
    """Latest campaign only, with no inference of readiness from completion."""
    campaigns = _snapshot_campaigns(snapshot)
    if not campaigns:
        return None
    campaign, data = max(campaigns.items(), key=lambda item: (
        max([str(item[1]["progress"].get("updated_at", ""))]
            + [str(u.get("manifest", {}).get("completed_at", "")) for u in item[1]["units"]]), str(item[0])))
    units, progress = data["units"], data["progress"]
    kinds = {_kind(u) for u in units}
    missing = [kind for kind in _REQUIRED if kind not in kinds]
    if _pending_evaluations(units):
        missing.append("complete fixed-panel outcomes")
    campaign_summaries = [u["summary"] for u in units if u["summary"]["kind"] == _PREFIX + "campaign"]
    final = campaign_summaries[-1] if campaign_summaries else {}
    status = "stopped" if progress.get("stopped") or (campaign / "STOP").exists() else final.get("status", progress.get("status"))
    pending = ", ".join(missing) if missing else "learning interpretation and resource acceptance"
    evidence = []
    teachers = _teacher_rows([u["summary"] for u in units if _kind(u) == "teacher"])
    if teachers:
        evidence.append("Frozen teacher success: " + "; ".join(
            f"{r['Variant']} {r['Success count / rate']}" for r in teachers) + ".")
    for unit in units:
        if _kind(unit) == "bootstrap":
            row = _bootstrap_row(unit["summary"])
            evidence.append(f"Easy bootstrap: {_text(row['Transitions'])} transitions; distillation KL "
                            f"{_text(row['Distillation initial KL'])} to {_text(row['Distillation final KL'])}.")
        if _kind(unit) == "pilot":
            rows = _evaluations(unit["summary"])
            rates = []
            for index, difficulty in enumerate(("easy", "middle", "hard")):
                before, after = rows[2 * index:2 * index + 2]
                if before["Episodes"] or after["Episodes"]:
                    rates.append(f"{difficulty} success {before['Success count / rate']} to {after['Success count / rate']}, "
                                 f"collisions {before['Collision count / rate']} to {after['Collision count / rate']}")
                else:
                    rates.append(f"{difficulty} pending")
            evidence.append(f"Student seed {_text(unit['summary'].get('seed'))}, initial to final: " + "; ".join(rates) + ".")
    return (f"## Conclusion\n\nPhase 1 Fix 3 (`{campaign.name}`): {_text(status)}. "
            f"{len(units)} sealed diagnostic units; pending: {pending}.\n\n"
            + "\n\n".join(evidence) + "\n\n"
            f"Unresolved issues: {_blockers(final, progress)}. "
            "These are bounded diagnostic rates, not proof of learning; missing outcomes remain unverified. "
            "No STM efficacy or sample-efficiency result is claimed. Full learning and physical operation "
            "remain outside this repair; stop at the resource/task check-in.")
