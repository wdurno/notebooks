"""Bounded Phase 6 scalar reports appended to sealed, corrected Phase 5 history.

Only completed scalar exports enter analysis. This module never loads a model,
checkpoint, dataset, or learner and never repairs scientific artifacts.
"""

from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import nbformat
import numpy as np

from .. import analysis
from .. import report as original
from ..broad_drift import report as broad

# Archive the implementations imported by this interpreter, not later disk edits.
_SOURCE_BYTES = {
    "report_source.py": Path(__file__).read_bytes(),
    "broad_report_source.py": broad._SOURCE_BYTES["report_source.py"],
    "original_report_source.py": broad._SOURCE_BYTES["original_report_source.py"],
    "analysis_source.py": broad._SOURCE_BYTES["analysis_source.py"],
}
PROTOCOL = "plan18-retention-match-v1"
REPORT_SCHEMA = 1
DEFAULT_ROOT = broad.DEFAULT_ROOT.with_name("retention_match_v1")
HISTORY_ROOT = broad.DEFAULT_ROOT
DEFAULT_NOTEBOOK = original.DEFAULT_NOTEBOOK
MAX_JSON_BYTES = original.MAX_JSON_BYTES
MAX_GENERATION_BYTES = original.MAX_GENERATION_BYTES
ARMS, COLORS = analysis.ARMS, original.COLORS
setting = analysis.setting
CONTRASTS = ("B-A", "C-B", "D-B", "E-B", "E-A", "interaction")


def _snapshot(root, cfg, check):
    # Late import keeps reporter import independent of orchestration and lets the
    # scalar reader enforce local completion and historical-reference checks.
    from . import runtime

    seals, consumed = {}, 0

    def read(name):
        nonlocal consumed
        check()
        path = root / name
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"report input must be a regular file: {name}")
        if consumed + path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        with path.open("rb") as handle:
            data = handle.read(MAX_JSON_BYTES - consumed + 1)
        consumed += len(data)
        if consumed > MAX_JSON_BYTES:
            raise ValueError("bounded report JSON input allocation exceeded")
        value = json.loads(data)
        original._bytes(value)  # Reject nonfinite JSON, including exponent overflow.
        seals[name] = original._hash(data)
        return value

    identity = read("identity.json")
    scientific = (identity or {}).get("identity", {})
    if identity is not None and (
        identity.get("schema") != 1
        or identity.get("identity_hash") != original._hash(original._bytes(scientific))
        or scientific.get("protocol") != PROTOCOL
        or scientific.get("metric_schema_version", 1) != 1
        or scientific.get("artifact_schema_version", 1) != 1
    ):
        raise ValueError("incompatible Phase 6 identity/protocol/schema")
    stored = scientific.get("config", {})
    cfg = cfg if cfg is not None else stored or dict(protocol=PROTOCOL, analysis_novel_count=1216,
                                                   main_replicas=4, stationary_replicas=2)
    resolved = original._cfg_mapping(cfg)
    if resolved.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 6 configuration protocol")
    if stored and original._bytes(stored) != original._bytes(resolved):
        raise ValueError("report configuration differs from sealed configuration")
    N = setting(cfg, "analysis_novel_count", 1216)
    if type(N) is not int or N < 1 or (not setting(cfg, "smoke", False) and N != 1216):
        raise ValueError("Phase 6 requires the 1216 outbound-grid production budget")
    if not setting(cfg, "smoke", False) and setting(cfg, "retention_tolerance", .1) != .1:
        raise ValueError("Phase 6 retention tolerance must remain .10 pixels")
    status = read("status.json") or {}
    if status.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 6 status protocol")
    development = read("development.json") or {}
    calibration_completed = read("calibration_completed.json") or {}
    refinement_decisions = []
    for path in sorted((root / "units").glob("decision-refinement-*")):
        if (path / "COMPLETED").exists():
            decision, digest = runtime.checked_scalar(root, path.name, "decision.json", check)
            refinement_decisions.append(decision)
            seals[f"units/{path.name}/decision.json"] = digest
    selection = read("selection.json")
    decision_digest = None
    if selection:
        decision, decision_digest = runtime.checked_scalar(root, "decision-selection", "decision.json", check)
        if decision != selection:
            raise ValueError("selection differs from the sealed calibration decision")
    records, provenance = runtime.read_exported_records(root, check=check)
    scalar_bytes = original._bytes(dict(records=records, provenance=provenance))
    if consumed + len(scalar_bytes) > MAX_JSON_BYTES:
        raise ValueError("bounded report JSON input allocation exceeded")
    inputs = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, files=seals, config=resolved,
        scalar_exports=provenance, scalar_digest=original._hash(scalar_bytes), selection_decision=decision_digest,
        implementation={name: original._hash(data) for name, data in _SOURCE_BYTES.items()})
    return dict(records=records, cfg=cfg, status=status, development=development,
                selection=selection, calibration_completed=calibration_completed, inputs=inputs,
                refinement_decisions=refinement_decisions)


def _verified_generation(path, digest, check):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("invalid immutable report directory")
    size = 0
    for file in path.rglob("*"):
        check()
        if file.is_symlink():
            raise ValueError("symlink in immutable report generation")
        if file.is_file():
            size += file.stat().st_size
            if size > MAX_GENERATION_BYTES:
                raise ValueError("bounded report generation allocation exceeded")
    return original._verify_generation(path, digest, check)


def _history(root, smoke, check):
    latest = root / "report/latest.json"
    if not latest.exists() and not latest.is_symlink() and smoke:
        return nbformat.v4.new_notebook(), None
    if latest.is_symlink() or not latest.is_file() or latest.stat().st_size > 4096:
        raise ValueError("sealed corrected Phase 5 history is required")
    pointer_bytes = latest.read_bytes()
    pointer = json.loads(pointer_bytes)
    name = pointer.get("generation", "")
    if (pointer.get("schema") != REPORT_SCHEMA or not name or Path(name).name != name
            or name in (".", "..")):
        raise ValueError("invalid historical report generation pointer")
    path = latest.parent / name
    metadata = _verified_generation(path, pointer["source_digest"], check)
    if metadata.get("protocol") != broad.PROTOCOL or metadata.get("status") != "completed":
        raise ValueError("history must be the completed Phase 5 report")
    data = (path / "results.ipynb").read_bytes()
    book = nbformat.reads(data.decode(), as_version=4)
    nbformat.validate(book)
    conclusion = book.cells[-1].source.lower() if book.cells else ""
    if ("phase 5 conclusion" not in conclusion or "calibration failed" not in conclusion
            or any(c.source.startswith("## Phase 6:") for c in book.cells)):
        raise ValueError("history requires the corrected Phase 5 calibration failure conclusion")
    return book, dict(generation=name, source_digest=pointer["source_digest"],
        pointer_sha256=original._hash(pointer_bytes), notebook_sha256=original._hash(data))


def _calibration(snapshot):
    selection = snapshot.get("selection") or {}
    status = selection.get("calibration_status", snapshot.get("status", {}).get("calibration_status", "pending"))
    matching = (selection.get("matching") or snapshot.get("development", {}).get("current_check")
                or snapshot.get("status", {}).get("calibration_check") or {})
    passed = (status == "passed" and selection.get("frozen") is True and matching.get("passed") is True
              and selection.get("members") == [0, 1, 2, 3]
              and set(analysis.selected_recipes(selection)) == set(ARMS))
    return dict(status=status, passed=passed, matching=matching,
                confirmation_authorized=passed)


def _summarize(snapshot, check=lambda: None):
    # Reuse the complete inherited S and reference-own-crossing calculations.
    result = broad._summarize(snapshot, check)
    result["protocol"] = PROTOCOL
    result["calibration"] = _calibration(snapshot)
    if result["methods"] and not result["calibration"]["passed"]:
        raise ValueError("fresh inference requires a passing four-member calibration decision")
    if "B-A" in result["comparisons"]:
        result["report_additions"]["B-A"] = "Prespecified Phase 6 contrast; inherited paired draws."
    fresh = result["cohorts"]["fresh"]
    replicas = fresh["replicas"]
    result["retention_equivalence"] = {}
    if not replicas:
        return result
    reference = {r["replica"]: r for r in original._paired_records(snapshot, result)
                 if r["recipe"]["arm"] == "A"}
    signs = [reference[r].get("global_sign", -1 if r % 2 else 1) for r in replicas]
    interval_ready = all(signs.count(sign) >= 2 for sign in (-1, 1))
    result["interval_status"] = ("nominal_percentile_95" if interval_ready
                                 else "descriptive_insufficient_sign_replicates")
    if not interval_ready:
        # Singleton sign strata resample to themselves and cannot estimate variance.
        def descriptive(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("ci", "upper_one_sided_95", "matched_old_gap_upper_one_sided_95",
                               "bootstrap_failure_frequency"):
                        value[key] = None
                    elif key in ("retention_supported", "sample_saving_supported"):
                        value[key] = False
                    elif key == "inverse_interval_status":
                        value[key] = result["interval_status"]
                    elif key == "status" and item == "nominal_percentile_95":
                        value[key] = result["interval_status"]
                    else:
                        descriptive(item)
            elif isinstance(value, list):
                for item in value:
                    descriptive(item)
        descriptive(fresh)
        for name in ("methods", "comparisons", "efficiency"):
            descriptive(result[name])
        fresh["bootstrap"].update(draws=0, intervals=result["interval_status"])
    bootstrap = fresh["bootstrap"]
    samples = (analysis.paired_draws(replicas, bootstrap["draws"], bootstrap["seed"],
        signs)
        if bootstrap["draws"] else None)
    for right_index, right in enumerate(ARMS):
        for left in ARMS[right_index + 1:]:
            name = f"{left}-{right}"
            stats = {}
            for field in ("Havg", "H100"):
                check()
                values = [np.asarray(result["methods"][a][field]["per_replica"]) for a in (left, right)]
                item = analysis._interval(values[0] - values[1], samples)
                ci = item["ci"]
                item["equivalent"] = bool(ci is not None and ci[0] >= -.1 and ci[1] <= .1)
                stats[field] = item
            result["retention_equivalence"][name] = dict(
                **stats, equivalent=all(item["equivalent"] for item in stats.values()),
                prespecified=name in CONTRASTS, tolerance_pixels=.1)
    return result


def _display_recipes(snapshot, result):
    if result["selection"]:
        return result["selection"]
    # Before freeze show the declared starting tuple, never select an unmatched
    # recipe from its outcomes. The scheduler's current check is separate.
    strength = setting(snapshot["cfg"], "fd_strength", .5691316767656495)
    return {arm: dict(arm=arm, ridge=ridge, strength=0. if arm == "A" else strength,
                     process_scale=1e-5 if arm in "DE" else 1e-4)
            for arm, ridge in zip(ARMS, (10., 10., 10., 1., .1))}


def _selected_development(snapshot, result):
    recipes = _display_recipes(snapshot, result)
    return [original._flat(r) for r in snapshot["records"]
        if r.get("status") == "completed" and r["metadata"]["cohort"] == "development"
        and r["metadata"]["recipe"] == recipes.get(r["metadata"]["recipe"]["arm"])]


def _development(snapshot, result):
    figure, axes = original._figure(1, 2)
    rows = [r for r in result["development"] if r["status"] in ("complete", "completed")]
    recipes = _display_recipes(snapshot, result)
    N = result["analysis_outbound_count"]
    markers = {0: "o", 1: "s", 2: "^", 3: "v"}
    for axis, field, label in zip(axes, ("Havg", "H100"), ("Havg", f"H{N}")):
        for row in rows:
            if row.get(field) is None or row.get("novel_auc") is None:
                continue
            arm, member = row["recipe"]["arm"], row["replica"]
            selected = row["recipe"] == recipes.get(arm)
            full = sum(r["recipe"] == recipes.get(arm) for r in rows) == 4
            labelled = arm in axis.get_legend_handles_labels()[1]
            axis.scatter(row[field], row["novel_auc"], color=COLORS[arm],
                marker=markers.get(member, "."), s=25, alpha=.85 if selected else .2,
                label=arm if selected and not full and not labelled else None)
        means = []
        for arm in ARMS:
            selected = [r for r in rows if r["recipe"] == recipes.get(arm)]
            if {r["replica"] for r in selected} == {0, 1, 2, 3} and len(selected) == 4:
                x, y = (float(np.mean([r[k] for r in selected])) for k in (field, "novel_auc"))
                means.append(x)
                axis.scatter(x, y, marker="X", s=70, color=COLORS[arm], label=arm + " pooled four")
        if len(means) == 5:
            axis.axvspan(min(means), min(means) + .1, color="#21866c", alpha=.08)
            axis.axvline(max(means), color="#666666", linestyle=":", linewidth=.8)
        if not rows:
            original._pending(axis, "Development calibration")
        axis.set(title=f"Development: {label} versus acquisition",
                 xlabel=f"{label} deterioration (px)", ylabel=f"Rotated A({N}) MSE (px squared)")
        original._legend(axis)
    return figure


def _prediction(snapshot, result):
    figure, axes = original._figure(2, 2)
    N = result["analysis_outbound_count"]
    for arm, method in result["methods"].items():
        curves = method["curves"]
        for axis, field in zip(axes[:2], ("novel_mse", "old_mse")):
            item = curves[field]
            axis.plot(curves["n"], item["mean"], color=COLORS[arm], label=arm)
            for row in item["per_replica"]:
                axis.plot(curves["n"], row, color=COLORS[arm], alpha=.15, linewidth=.7)
            if item["ci"] is not None:
                axis.fill_between(curves["n"], *item["ci"], color=COLORS[arm], alpha=.1)
            axis.scatter([N] * len(item["per_replica"]),
                         [r[-1] for r in item["per_replica"]], color=COLORS[arm], s=12)
    references = [r for r in original._paired_records(snapshot, result) if r["recipe"]["arm"] == "A"]
    if references:
        for axis, field in zip(axes[:2], ("frozen_novel_mse", "frozen_old_mse")):
            axis.axhline(float(np.mean([r[field] for r in references])), color="#222222",
                         linestyle="--", linewidth=.8, label="Frozen initialization")
    development = _selected_development(snapshot, result)
    for arm in ARMS:
        group = [r for r in development if r["recipe"]["arm"] == arm]
        curves = [analysis.novel_curve(r["metrics"], snapshot["cfg"]) for r in group]
        for row, curve in zip(group, curves):
            axes[2].plot(curve["n"], np.sqrt(curve["old_mse"]) - np.sqrt(row["frozen_old_mse"]),
                color=COLORS[arm], linestyle="--" if row["replica"] < 2 else ":", alpha=.4, linewidth=.8)
        if len(group) == 4 and {r["replica"] for r in group} == {0, 1, 2, 3}:
            axes[2].plot(curves[0]["n"], np.mean([np.sqrt(c["old_mse"]) - np.sqrt(r["frozen_old_mse"])
                for r, c in zip(group, curves)], axis=0), color=COLORS[arm], label=arm + " pooled four")
    for name, item in result["efficiency"].items():
        if name == "A-A":
            continue
        point = item["point"]
        if point.get("S") is not None:
            axes[3].scatter(point["matched_time_ratio"], point["saved_fraction"],
                color=COLORS[point["arm"]], marker="s" if name == "E-B" else "o", label=name,
                facecolors=None if item["retention_supported"] else "none")
    for axis, (title, xlabel, ylabel) in zip(axes, (
        ("Fresh fixed rotated panel", "Outbound grids", "MSE (px squared)"),
        ("Fresh fixed old panel", "Outbound grids", "MSE (px squared)"),
        ("Development old trajectories", "Outbound grids", "RMS deterioration (px)"),
        ("Common A target; E-B secondary", "Matched compute multiplier", "Saved fraction"))):
        if not axis.lines and not axis.collections:
            original._pending(axis, title)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        original._legend(axis)
    return figure


def _budget_closed(snapshot):
    state = snapshot["status"].get("status") or snapshot.get("calibration_completed", {}).get("status")
    return state == "calibration_budget_limited"


def _progress(snapshot, result):
    status, gate = snapshot["status"], result["calibration"]
    counts = Counter(original._flat(r)["cohort"] for r in snapshot["records"] if r.get("status") == "completed")
    imported = sum(original._flat(r)["replica"] in (0, 1) for r in snapshot["records"]
        if r.get("status") == "completed" and original._flat(r)["cohort"] == "development"
        and original._flat(r).get("imported"))
    # Historical members may also have new refinements; only the reader's
    # explicit imported flag counts a path as reused historical computation.
    expected_dev = status.get("expected_development_paths",
        max(20, imported + status.get("expected_development_new", 10), counts["development"]))
    key = lambda member, recipe: (member, json.dumps(recipe, sort_keys=True))
    baseline = {key(member, recipe) for member in range(4)
                for recipe in _display_recipes(snapshot, result).values()}
    baseline.update(key(r["metadata"]["replica"], r["metadata"]["recipe"])
                    for r in snapshot["records"] if r["metadata"].get("imported"))
    admitted = {key(member, recipe) for decision in snapshot.get("refinement_decisions", [])
                for recipe in decision["recipes"] for member in range(4)}
    expected_dev = max(expected_dev, imported + status.get("expected_development_new", 10)
                       + len(admitted - baseline))
    expected_fresh = 5 * result["cohorts"]["fresh"]["planned_pairs"]
    expected_stationary = 5 * result["cohorts"]["stationary"]["planned_pairs"]
    active = status.get("active_seconds")
    cap = status.get("max_active_seconds", 20 * 3600) / 3600
    remaining = status.get("remaining_seconds", None if active is None else max(0., cap * 3600 - active))
    forecast = status.get("forecast_remaining_seconds", (status.get("forecast") or {}).get("remaining_seconds"))
    current = status.get("current") or {}
    fmt = original._fmt
    closure = (" **Confirmation is closed due to budget; no confirmation work is queued.**"
               if _budget_closed(snapshot) else "")
    return ("# Plan 18 Progress\n\n"
        f"**Phase 6: {status.get('phase', '6A implementation; execution pending')}** | {status.get('status', 'preparing')}.  \n"
        f"Completed/expected paths: development {counts['development']}/{expected_dev} "
        f"(imported development {imported}; additional candidates may extend this allocation), "
        f"fresh {counts['fresh']}/{expected_fresh}, stationary {counts['stationary']}/{expected_stationary}.  \n"
        f"Calibration: **{gate['status']}**; passing four-member gate: {'yes' if gate['passed'] else 'no'}; "
        f"confirmation authorized by calibration: **{'yes' if gate['confirmation_authorized'] else 'no'}** "
        f"(resource reservation remains a separate requirement).{closure}  \n"
        f"Cumulative charged active hours: {fmt(None if active is None else active / 3600)} / {fmt(cap)} h; "
        f"remaining ceiling: {fmt(None if remaining is None else remaining / 3600)} h; "
        f"remaining forecast: {fmt(None if forecast is None else forecast / 3600)} h.  \n"
        f"Current unit/batch: {current.get('unit', 'none')} / {current.get('batch', 'none')}/{current.get('total', 'none')}.\n\n"
        "Only completed five-arm fresh blocks enter inference; incomplete work is progress only. "
        "Interim means and sample-equivalence points are descriptive until both rotation-sign strata "
        "have at least two fresh replicas; confidence intervals and supported-saving claims are withheld before then. "
        "Phase 0--5 sealed history, including its calibration-failure conclusion, is preserved below. "
        "The historically informed Phase 6 repair follows that conclusion.")


def _settings(snapshot, result):
    rows = []
    selected = _selected_development(snapshot, result)
    recipes = _display_recipes(snapshot, result)
    for arm in ARMS:
        recipe = recipes.get(arm, {})
        group = [r for r in selected if r["recipe"]["arm"] == arm]

        def means(field):
            values = []
            for members in ((0, 1), (2, 3), (0, 1, 2, 3)):
                part = [r for r in group if r["replica"] in members]
                values.append(original._fmt(float(np.mean([r[field] for r in part])))
                    if len(part) == len(members) and {r["replica"] for r in part} == set(members) else "pending")
            return " / ".join(values)

        rows.append([arm + " " + analysis.LABELS[arm], recipe.get("ridge"), recipe.get("strength"),
            "fixed .025" if arm in "ABC" else "velocity Kalman", recipe.get("process_scale") if arm in "DE" else "unused",
            means("Havg"), means("H100")])
    N = result["analysis_outbound_count"]
    matching = result["calibration"]["matching"]
    ranges = "; ".join(f"{'H' + str(N) if name == 'H100' else name}={original._fmt(value)} px"
                       for name, value in matching.get("ranges", {}).items()) or "pending"
    selected_curves = [analysis.novel_curve(r["metrics"], snapshot["cfg"]) for r in selected]
    by_arm = {arm: [c for r, c in zip(selected, selected_curves) if r["recipe"]["arm"] == arm] for arm in ARMS}
    separation = None
    if all(len(by_arm[arm]) == 4 for arm in ARMS):
        mean_rms = np.asarray([np.sqrt([c["old_mse"] for c in by_arm[arm]]).mean(axis=0) for arm in ARMS])
        separation = float(np.ptp(mean_rms, axis=0).max())
    return ("### Phase 6 Settings And Calibration\n\n" + original._table(
        ("Method", "Ridge c", "FD strength", "pi policy", "Process multiplier", "Havg hist/new/pool (px)",
         f"H{N} hist/new/pool (px)"), rows)
        + ("\nFrozen selected tuple. " if result["selection"] else
           "\nDeclared starting tuple for display only; no selection is frozen. ")
        + "Historical members 0/1, new members 2/3, and pooled four are development evidence only. "
        "Roots are taken within each member before averaging. Both all-arm pooled ranges must be <= .10 px; "
        "closeness to A alone is insufficient. The .50 px preference is descriptive, not a selection ceiling. "
        f"Scheduler's current range checks (unrounded for the gate, possibly a different tuple before freeze): {ranges}. "
        f"Maximum mean old-RMS separation for the displayed tuple over outbound acquisition: {original._fmt(separation)} px. "
        "Matching summaries need not imply pointwise matching or low absolute forgetting. "
        "All methods retain FIFO32/replay2, 3,268 LTM and 108 STM coordinates; ridge is c times the mean archive diagonal. "
        "Process multiplier, Kalman gain, and applied pi are different quantities. "
        + " ".join(f"[{name}]({href})" for name, href in snapshot.get("artifact_hrefs", {}).items()))


def _effects(result):
    ordered = {name: result["comparisons"][name] for name in CONTRASTS if name in result["comparisons"]}
    return broad._effects({**result, "comparisons": ordered}).replace("Phase 5 Paired Effects", "Phase 6 Paired Effects")


def _retention(result):
    rows = []
    for name, item in result["retention_equivalence"].items():
        rows.append([f"**{name}**" if item["prespecified"] else name,
            original._effect(item["Havg"]), "yes" if item["Havg"]["equivalent"] else "unresolved",
            original._effect(item["H100"]), "yes" if item["H100"]["equivalent"] else "unresolved"])
    return ("### Fresh Retention Equivalence\n\n" + original._table(
        ("Contrast", "Havg difference (px)", "Equivalent", f"H{result['analysis_outbound_count']} difference (px)", "Equivalent"),
        rows or [["Pending complete fresh blocks", None, "unresolved", None, "unresolved"]])
        + "\nAll ten pairwise contrasts use the same whole-replica, sign-stratified nominal two-sided 95% draws. "
        "Intervals remain unavailable until each sign has at least two fresh replicas; singleton strata "
        "cannot estimate sampling variation. "
        "Equivalence requires the entire interval inside [-.10, .10] px for each statistic; "
        "nonsignificance is insufficient. These are not simultaneous all-arm guarantees. "
        "Bold contrasts were prespecified. One-sided noninferiority for sample savings is a different check.")


def _conclusion(snapshot, result):
    gate = result["calibration"]
    if not gate["passed"]:
        return ("### Phase 6 Conclusion\n\n"
            f"**Calibration: {gate['status']}; no passing freeze is established.** "
            "Fresh confirmation is not authorized. Development candidates describe the calibration frontier, "
            "not fresh method efficacy; no conclusion about STM, LFU, or adaptive pi follows. "
            "A failed or budget-limited calibration is a comparison-validity result, not a method failure.")
    retained = result["retention_equivalence"]
    equivalence = (f"{sum(v['equivalent'] for v in retained.values())}/10 contrasts support both nominal "
                   "two-sided retention checks." if retained else "Fresh retention equivalence is unresolved.")
    outcomes = []
    for name in ("B-A", "E-B", "E-A"):
        if name in result["comparisons"]:
            outcomes.append(f"{name} rotated-risk difference: {original._effect(result['comparisons'][name]['novel_auc'])} px squared.")
    supported = [name for name, item in result["efficiency"].items()
                 if name != "A-A" and item["sample_saving_supported"]]
    budget_closed = _budget_closed(snapshot)
    if budget_closed:
        detail = snapshot.get("calibration_completed", {}).get("detail", "")
        closeout = ("**Confirmation is closed due to budget.** " + (str(detail) + " " if detail else "")
            + "No further confirmation work is queued; a sufficient resource reservation is required before execution.")
    else:
        closeout = ("The planned paired allocation is complete." if result["status"] == "completed" else
                    "Allocation remains incomplete; this is a rolling report, not scientific closeout.")
    no_outcomes = ("Fresh new-data performance was not assessed in complete paired blocks." if budget_closed else
                   "New-data performance awaits complete fresh paired blocks.")
    timing_note = ""
    if result["status"] == "completed":
        reference = result.get("reference_clock_diagnostics", {}).get("A", {})
        crossings = reference.get("downward_crossings", [])
        combined = result["efficiency"].get("E-A", {}).get("point", {})
        if reference.get("multiple_crossings") and crossings and combined.get("n") is not None:
            timing_note = ("\n\n**Efficiency qualification:** E reaches sustained target quality at "
                f"{original._fmt(combined['n'])} outbound grids; A's own sustained crossing is "
                f"{original._fmt(reference['n'])}, although A first crosses at {original._fmt(crossings[0])} "
                "and then rises above the target again. The large endpoint-budget saving therefore describes "
                "earlier *sustained* quality on a nonmonotone path, not twice-as-fast initial learning. "
                "Its wide interval must accompany the point estimate.")
            stm = result["efficiency"].get("B-A", {}).get("point", {})
            batch_size = setting(snapshot["cfg"], "samples_per_step", 4)
            if stm.get("n") is not None and abs(stm["n"]-reference["n"]) <= batch_size:
                timing_note += (" B's crossing is within one incoming batch of A's own crossing; "
                    "its small allocated-budget saving does not establish an additional timing advantage.")
    return ("### Phase 6 Conclusion\n\n**Development calibration passed.** " + equivalence + "\n\n"
        + (" ".join(outcomes) or no_outcomes)
        + "\n\nSample savings with positive supported intervals and full/matched-budget retention: "
        + (", ".join(supported) if supported else "not established")
        + ". Compute multipliers and S above determine whether any saving offsets its learning cost. "
        "Calibration is not defined by a favorable STM outcome. Matched substantial forgetting is not low absolute error; "
        "neither gains nor null outcomes establish adaptive optimality or identify a mechanism. "
        + closeout + timing_note)


def _notebook(snapshot, result, history, figures):
    history.cells.insert(0, nbformat.v4.new_markdown_cell(_progress(snapshot, result)))
    resource_note = ("\n\n**Compute authorization:** after the four-member calibration passed, "
        "the user-approved cumulative ceiling increased from 20 to 21.5 active hours. "
        "The separately archived continuation revalidated the calibration seal and kept the methods, "
        ".10 px tolerance and 30-trajectory confirmation allocation unchanged."
        if snapshot["status"].get("max_active_seconds", 72000) > 72000 else "")
    cells = [nbformat.v4.new_markdown_cell(
        "## Phase 6: Retention-Matched Recomparison\n\n"
        "**Phase 5 calibration failed:** the STM reference's stronger ridge confounded the intended "
        "five-policy comparison at comparable retention. This historically informed repair asks which method "
        "learns new observations most efficiently at the Replay/EWC control's old-data deterioration. "
        "It does not presume an STM benefit. The whole-stream rotation law and 1,216-grid production "
        "outbound budget are unchanged; development and fresh evidence remain separate. [Design](plan18.md#phase-6-retention-matched-recomparison).\n\n"
        "**Interim uncertainty:** a reporting repair before fresh execution withholds intervals and "
        "supported-saving claims until both sign strata contain two fresh replicas. Singleton strata "
        "otherwise resample only themselves and can give misleading zero-width intervals. "
        "Point estimates and the planned four-replica analysis retain their original definitions.\n\n"
        "**Development proposal review:** the starting tuple missed the average-retention range by .0051 px. "
        "The automatic proposer omitted historical ridge responses incomplete on four members. "
        "A paired historical average-retention bracket, with both endpoint responses already in band, "
        "therefore supplied a recorded geometric-midpoint proposal. It is an additional permitted recipe, "
        "not an interpolated result; all four executions and the original .10 px gate remain required. "
        "The prior admitted probe and all completed artifacts are retained; the reviewed decision archives its source."
        + resource_note),
        nbformat.v4.new_markdown_cell(_settings(snapshot, result))]
    captions = (
        ("Development Retention And Acquisition", "Circles/squares: historical members 0/1; up/down triangles: new members 2/3; "
         "X: displayed pooled-four means (frozen selection, or the declared starting tuple before freeze). "
         "Faint points are other attempted recipes. The shaded .10 px band starts at "
         "the smallest displayed mean; the dotted line marks the largest. Each retention metric must pass separately."),
        ("Fresh Prediction And Sample Efficiency", "Fresh curves show unregularized holdout MSE, replica paths/endpoint points, "
         "and nominal pointwise 95% bands when both sign strata have two replicas. Earlier means are descriptive. "
         "Dashed black is frozen initialization; n=0 is after initial quiet training. "
         "The acquisition clock ends before return training. Development old-error "
         "paths are dashed for historical members, dotted for new members, solid for pooled means. "
         "Hollow efficiency points lack full/matched-budget retention support; square is the secondary E-B target."),
        ("Pi, Gain And Motion Diagnostics", "Changing-stream diagnostics average the complete fresh paired replicas. "
         "Pi is the applied adaptation weight, not the Kalman gain or process-noise multiplier. "
         "Gain shading shows coordinate p10/p90 ranges averaged across replicas, not confidence intervals. "
         "One-grid noise is V; batch noise is V/m. Movement includes trend energy plus uncertainty, "
         "weighted by the current archive. LFU corrections are recorded before PSD projection. "
         "Stationary paths are descriptive and have no sample-equivalence score."))
    for (name, png), (title, caption) in zip(figures, captions):
        attachment = f"phase6-{name}.png"
        cell = nbformat.v4.new_markdown_cell(f"### {title}\n\n![{title}](attachment:{attachment})\n\n{caption}")
        cell["attachments"] = {attachment: {"image/png": base64.b64encode(png).decode("ascii")}}
        cells.append(cell)
    cells.extend(nbformat.v4.new_markdown_cell(text) for text in (
        _effects(result), _retention(result), broad._efficiency(result).replace("Phase 5 Sample Equivalence", "Phase 6 Sample Equivalence")
        + "\n\n**Estimate and interval:** savings are calculated from the mean trajectories, not a median of replica savings. "
        "Each paired bootstrap draw resamples whole trajectories within rotation-sign strata and recalculates "
        "the control's endpoint target and the method's last downward crossing that stays below it through the budget. "
        "The interval uses the 2.5th and 97.5th percentiles. Switching between early and late crossings can produce "
        "a strongly asymmetric interval; four fresh replicas provide limited precision.",
        _conclusion(snapshot, result)))
    history.cells.extend(cells)
    history.metadata["plan18_phase6_report"] = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL,
        source_digest=snapshot["source_digest"], history=snapshot["inputs"]["history"])
    nbformat.validate(history)
    return nbformat.writes(history).encode()


def build_report(root=DEFAULT_ROOT, cfg=None, notebook_path=None, check=lambda: None, force=False, *, history_root=None):
    """Atomically publish one immutable scalar generation under a report lease.

    Production requires the sealed corrected Phase 5 conclusion. Smoke output
    is local to the root even when an explicit public notebook is supplied.
    """
    root = Path(root)
    report = root / "report"
    with original._lease(report, check):
        snapshot = _snapshot(root, cfg, check)
        smoke = setting(snapshot["cfg"], "smoke", False)
        notebook = root / "smoke_results.ipynb" if smoke else Path(notebook_path or DEFAULT_NOTEBOOK)
        history, seal = _history(Path(history_root or HISTORY_ROOT), smoke, check)
        snapshot["inputs"]["history"] = seal
        snapshot["artifact_hrefs"] = {name: os.path.relpath(root / name, notebook.parent)
            for name in ("development.json", "selection.json", "status.json", "calibration_completed.json")
            if name in snapshot["inputs"]["files"]}
        snapshot["inputs"]["artifact_hrefs"] = snapshot["artifact_hrefs"]
        digest = original._hash(original._bytes(snapshot["inputs"]))
        snapshot["source_digest"] = digest
        generation = digest[:24] + ("-" + uuid.uuid4().hex[:10] if force else "")
        latest = report / "latest.json"
        if latest.is_symlink():
            raise ValueError("invalid report generation pointer")
        if latest.exists() and not force:
            if latest.stat().st_size > 4096:
                raise ValueError("invalid report generation pointer")
            pointer = json.loads(latest.read_bytes())
            if pointer.get("source_digest") == digest:
                generation = pointer.get("generation", "")
                if (pointer.get("schema") != REPORT_SCHEMA or not generation
                        or Path(generation).name != generation or generation in (".", "..")):
                    raise ValueError("invalid report generation pointer")
        destination = report / generation
        if destination.exists() or destination.is_symlink():
            metadata = _verified_generation(destination, digest, check)
            if metadata.get("protocol") != PROTOCOL:
                raise ValueError("incompatible Phase 6 report protocol")
            data = (destination / "results.ipynb").read_bytes()
            changed = False
        else:
            result = _summarize(snapshot, check)
            temporary = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=report))
            try:
                files, figures = dict(_SOURCE_BYTES), []
                for name, render in (("development", _development), ("prediction", _prediction),
                                     ("diagnostics", broad._diagnostics)):
                    check()
                    figure = render(snapshot, result)
                    try:
                        for axis in figure.axes:
                            axis.title.set_fontsize(9)
                            axis.xaxis.label.set_fontsize(8)
                            axis.yaxis.label.set_fontsize(8)
                        output = io.BytesIO()
                        figure.savefig(output, format="png", dpi=100,
                            metadata={"Software": "Plan18 Phase6 scalar report"})
                        png = output.getvalue()
                    finally:
                        figure.clear()
                    figures.append((name, png))
                    files[f"figures/{name}.png"] = png
                data = _notebook(snapshot, result, history, figures)
                metadata = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, generation=generation,
                    source_digest=digest, created_at=datetime.now(timezone.utc).isoformat(), status=result["status"],
                    calibration_status=result["calibration"]["status"],
                    completed_pairs={c: r["n_pairs"] for c, r in result["cohorts"].items()},
                    notebook="results.ipynb", figure_count=len(figures))
                files.update({"results.ipynb": data, "analysis.json": original._bytes(result),
                    "inputs.json": original._bytes(snapshot["inputs"]), "generation.json": original._bytes(metadata)})
                if sum(map(len, files.values())) > MAX_GENERATION_BYTES:
                    raise ValueError("bounded report generation allocation exceeded")
                for name, content in files.items():
                    check()
                    original._atomic(temporary / name, content)
                original._atomic(temporary / "COMPLETED", original._bytes(dict(schema=REPORT_SCHEMA,
                    source_digest=digest, files={name: original._hash(content) for name, content in files.items()})))
                check()
                os.rename(temporary, destination)
                original._sync_directory(report)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            changed = True
        check()
        if not notebook.exists() or notebook.read_bytes() != data:
            original._atomic(notebook, data)
        pointer_bytes = original._bytes(dict(schema=REPORT_SCHEMA, generation=generation, source_digest=digest))
        if not latest.exists() or latest.read_bytes() != pointer_bytes:
            original._atomic(latest, pointer_bytes)
        return {**metadata, "changed": changed}
