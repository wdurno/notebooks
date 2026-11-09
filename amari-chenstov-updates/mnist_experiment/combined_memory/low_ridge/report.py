"""Scalar-only Phase 7 reports appended to immutable Phase 0--6 history.

B/ridge-one defines development retention; calibrated A defines primary sample
equivalence. Reporting never opens datasets, models, or checkpoints.
"""

from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import io
import json
import math
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
from ..retention_match import report as phase6

# Capture implementations imported by this interpreter, not later disk edits.
_SOURCE_BYTES = {
    "report_source.py": Path(__file__).read_bytes(),
    "phase6_report_source.py": phase6._SOURCE_BYTES["report_source.py"],
    **{name: data for name, data in phase6._SOURCE_BYTES.items() if name != "report_source.py"},
}
PROTOCOL = "plan18-low-ridge-v1"
REPORT_SCHEMA = phase6.REPORT_SCHEMA
DEFAULT_ROOT = phase6.DEFAULT_ROOT.with_name("low_ridge_v1")
HISTORY_ROOT = phase6.DEFAULT_ROOT
DEFAULT_NOTEBOOK = original.DEFAULT_NOTEBOOK
MAX_JSON_BYTES = original.MAX_JSON_BYTES
MAX_GENERATION_BYTES = original.MAX_GENERATION_BYTES
ARMS, COLORS = analysis.ARMS, original.COLORS
CONTRASTS = phase6.CONTRASTS
setting = analysis.setting


def _snapshot(root, cfg, check):
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
        original._bytes(value)
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
        raise ValueError("incompatible Phase 7 identity/protocol/schema")
    stored = scientific.get("config", {})
    cfg = cfg if cfg is not None else stored or dict(protocol=PROTOCOL, analysis_novel_count=1216,
                                                   main_replicas=8, stationary_replicas=2)
    resolved = original._cfg_mapping(cfg)
    if resolved.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 7 configuration protocol")
    if stored and original._bytes(stored) != original._bytes(resolved):
        raise ValueError("report configuration differs from sealed configuration")
    N = setting(cfg, "analysis_novel_count", 1216)
    if type(N) is not int or N < 1 or (not setting(cfg, "smoke", False) and N != 1216):
        raise ValueError("Phase 7 requires the 1216 outbound-grid production budget")
    if not setting(cfg, "smoke", False) and (
        setting(cfg, "retention_tolerance", .1) != .1
        or setting(cfg, "main_replicas", 8) != 8
        or setting(cfg, "stationary_replicas", 2) != 2
    ):
        raise ValueError("Phase 7 requires .10 px tolerance and eight fresh plus two stationary replicas")
    status = read("status.json") or {}
    if status.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("incompatible Phase 7 status protocol")
    development = read("development.json") or {}
    completed = read("calibration_completed.json") or {}
    decisions = []
    for path in sorted((root / "units").glob("decision-refinement-*")):
        if (path / "COMPLETED").exists():
            decision, digest = runtime.checked_scalar(root, path.name, "decision.json", check)
            decisions.append(decision)
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
    return dict(records=records, cfg=cfg, status=status, development=development, selection=selection,
                calibration_completed=completed, refinement_decisions=decisions, inputs=inputs)


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
        raise ValueError("sealed completed Phase 6 history is required")
    pointer_bytes = latest.read_bytes()
    pointer = json.loads(pointer_bytes)
    name, digest = pointer.get("generation", ""), pointer.get("source_digest")
    if (pointer.get("schema") != REPORT_SCHEMA or not isinstance(name, str) or not name
            or Path(name).name != name or name in (".", "..") or not isinstance(digest, str)):
        raise ValueError("invalid historical report generation pointer")
    path = latest.parent / name
    metadata = _verified_generation(path, digest, check)
    if metadata.get("protocol") != phase6.PROTOCOL or metadata.get("status") != "completed":
        raise ValueError("history must be the completed Phase 6 report")
    data = (path / "results.ipynb").read_bytes()
    book = nbformat.reads(data.decode(), as_version=4)
    nbformat.validate(book)
    if (book.metadata.get("plan18_phase6_report", {}).get("protocol") != phase6.PROTOCOL
            or not any(c.source.startswith("## Phase 6:") for c in book.cells)
            or any(c.source.startswith("## Phase 7:") for c in book.cells)):
        raise ValueError("history must contain Phase 6, without an appended Phase 7")
    # Phase 6 is valid history; do not impose Phase 5's invalidity qualification.
    return book, dict(generation=name, source_digest=digest,
        pointer_sha256=original._hash(pointer_bytes), notebook_sha256=original._hash(data))


def _calibration(snapshot):
    gate = phase6._calibration(snapshot)
    selection = snapshot.get("selection") or {}
    recipes = analysis.selected_recipes(selection)
    reference = recipes.get("B", {})
    ranges = gate["matching"].get("ranges", {})
    tolerance = setting(snapshot["cfg"], "retention_tolerance", .1)
    fixed = (selection.get("reference_arm") == "B" and selection.get("reference_ridge") == 1
        and selection.get("efficiency_reference_arm") == "A" and reference.get("ridge") == 1
        and reference.get("strength") == setting(snapshot["cfg"], "fd_strength", .5691316767656495))
    matched = all(isinstance(ranges.get(k), (int, float)) and math.isfinite(ranges[k])
                  and 0 <= ranges[k] <= tolerance + 1e-8 for k in ("Havg", "H100"))
    gate.update(passed=bool(gate["passed"] and fixed and matched), reference_arm="B", reference_ridge=1,
                sample_equivalence_reference="A")
    gate["confirmation_authorized"] = gate["passed"]
    return gate


def _display_recipes(snapshot, result):
    if result["selection"]:
        return result["selection"]
    strength = setting(snapshot["cfg"], "fd_strength", .5691316767656495)
    return {arm: dict(arm=arm, ridge=ridge, strength=0. if arm == "A" else strength,
                     process_scale=1e-5 if arm in "DE" else 1e-4)
            for arm, ridge in zip(ARMS, (1., 1., 1., .1, .1))}


def _selected_development(snapshot, result):
    recipes = _display_recipes(snapshot, result)
    return [original._flat(r) for r in snapshot["records"] if r.get("status") == "completed"
            and r["metadata"]["cohort"] == "development"
            and r["metadata"]["recipe"] == recipes.get(r["metadata"]["recipe"]["arm"])]


def _diagnostic_scalars(snapshot, result):
    """Keep saved s/kappa/moves; derive odds only from the saved applied pi."""
    records = original._paired_records(snapshot, result)
    cohort = "fresh" if records else "development"
    records = records or _selected_development(snapshot, result)
    methods = {}
    for arm in ARMS:
        values = []
        for row in records:
            if row["recipe"]["arm"] != arm:
                continue
            for diagnostic in row.get("diagnostics", []):
                pi = diagnostic.get("applied_pi")
                omega = (1 - pi) / pi if isinstance(pi, (int, float)) and 0 < pi <= 1 else None
                kappa = diagnostic.get("kappa")
                item = dict(replica=row["replica"], incoming_grids=diagnostic.get("incoming_grids"),
                    c=row["recipe"]["ridge"], s=diagnostic.get("s"), kappa=kappa, omega=omega,
                    omega_kappa=omega * kappa if omega is not None and isinstance(kappa, (int, float)) else None,
                    ltm_move=diagnostic.get("ltm_move"), stm_move=diagnostic.get("stm_move"))
                values.append(item)
        series = {}
        for field in ("c", "s", "kappa", "omega", "omega_kappa", "ltm_move", "stm_move"):
            by_clock, by_replica = {}, {}
            for row in values:
                value, clock = row[field], row["incoming_grids"]
                if isinstance(value, (int, float)) and math.isfinite(value) and clock is not None:
                    by_clock.setdefault(clock, []).append(value)
                    by_replica.setdefault(row["replica"], []).append(value)
            x = sorted(by_clock)
            series[field] = dict(incoming_grids=x, mean=[float(np.mean(by_clock[t])) for t in x],
                replica_means={str(r): float(np.mean(v)) for r, v in sorted(by_replica.items())},
                pooled_mean=float(np.mean([np.mean(v) for v in by_replica.values()])) if by_replica else None)
        methods[arm] = series
    return dict(cohort=cohort, methods=methods,
                provenance="Saved scalar diagnostics s, kappa, applied_pi, ltm_move, stm_move; c from recipe; omega=(1-pi)/pi.")


def _bootstrap_efficiency(snapshot, result, check):
    fresh = result["cohorts"]["fresh"]
    bootstrap = fresh.get("bootstrap", {})
    draws = bootstrap.get("draws", 0)
    if not draws:
        return {}
    replicas = fresh["replicas"]
    records = {(r["recipe"]["arm"], r["replica"]): r for r in original._paired_records(snapshot, result)}
    signs = [records["A", r].get("global_sign", -1 if r % 2 else 1) for r in replicas]
    samples = analysis.paired_draws(replicas, draws, bootstrap["seed"], signs)
    arrays = {}
    for arm in ("A", "B", "E"):
        rows = [records[arm, r] for r in replicas]
        curves = [analysis.novel_curve(row["metrics"], snapshot["cfg"]) for row in rows]
        arrays[arm] = {field: np.asarray([c[field] for c in curves], dtype=float)
                      for field in ("old_mse", "novel_mse", "learning_seconds", "incoming_grids")}
        arrays[arm]["frozen_novel_mse"] = np.asarray([r["frozen_novel_mse"] for r in rows])
    result_draws = {}
    for arm in ("B", "E"):
        outcomes = []
        for index, sample in enumerate(samples):
            if index % 25 == 0:
                check()
            # One draw indexes every arm, time, risk and retention curve together.
            outcomes.append(analysis._equivalence(arrays, arm, "A", sample))
        failures = Counter(p["status"] for p in outcomes if p["status"] != "valid")
        valid = not failures and result["efficiency"][f"{arm}-A"]["point"]["status"] == "valid"
        saved = [p.get("saved_fraction") for p in outcomes]
        result_draws[f"{arm}-A"] = dict(draws=draws, seed=bootstrap["seed"], replicas=replicas, signs=signs,
            status="estimable" if valid else "withheld_unsupported_crossings",
            saved_fraction=saved, target_mse=[p["target_mse"] for p in outcomes],
            n=[p["n"] for p in outcomes], statuses=[p["status"] for p in outcomes],
            first_crossing=[next(iter(p.get("downward_crossings", [])), None) for p in outcomes],
            failures=dict(failures), failure_frequency=sum(failures.values()) / draws,
            median=float(np.median(saved)) if valid else None,
            point_from_mean_curves=result["efficiency"][f"{arm}-A"]["point"]["saved_fraction"])
    return result_draws


def _summarize(snapshot, check=lambda: None):
    gate = _calibration(snapshot)
    result = phase6._summarize(snapshot, check)
    result.update(protocol=PROTOCOL, calibration=gate,
                  reference_roles=dict(development_retention="B", primary_sample_equivalence="A", secondary="E-B"))
    if result["methods"] and not gate["passed"]:
        raise ValueError("fresh inference requires a passing four-member B/ridge-1 calibration decision")
    if not result["methods"]:
        result["interval_status"] = "descriptive_insufficient_sign_replicates"
    terminal = snapshot["status"].get("status") or snapshot.get("calibration_completed", {}).get("status")
    if terminal in ("calibration_infeasible", "calibration_budget_limited"):
        result["status"] = terminal
    if "B-A" in result["comparisons"]:
        result["report_additions"]["B-A"] = "Primary prespecified Phase 7 contrast; unchanged paired draws."
    result["scalar_diagnostics"] = _diagnostic_scalars(snapshot, result)
    result["bootstrap_saved_fractions"] = _bootstrap_efficiency(snapshot, result, check)
    result["crossing_diagnostics"] = {}
    for name, item in result["efficiency"].items():
        point = item["point"]
        own = result["reference_clock_diagnostics"].get("A", {}) if name == "A-A" else point
        crossings = own.get("downward_crossings", [])
        result["crossing_diagnostics"][name] = dict(target_mse=point["target_mse"],
            first_crossing=next(iter(crossings), None), persistent_crossing=own.get("n"),
            allocated_n=result["analysis_outbound_count"], status=point["status"])
    return result


def _development(snapshot, result):
    figure, axes = original._figure(2, 2)
    rows = [r for r in result["development"] if r["status"] in ("completed", "complete")]
    recipes = _display_recipes(snapshot, result)
    N = result["analysis_outbound_count"]
    for axis, field, label in zip(axes[:2], ("Havg", "H100"), ("Havg", f"H{N}")):
        means = []
        for arm in ARMS:
            candidates = [r for r in rows if r["recipe"]["arm"] == arm and r.get(field) is not None]
            for row in candidates:
                chosen = row["recipe"] == recipes[arm]
                axis.scatter(row[field], row["novel_auc"], color=COLORS[arm],
                    marker=("o", "s", "^", "v")[row["replica"] % 4], alpha=.8 if chosen else .18,
                    s=22, label=arm if chosen and arm not in axis.get_legend_handles_labels()[1] else None)
            selected = [r for r in candidates if r["recipe"] == recipes[arm]]
            if len(selected) == 4 and {r["replica"] for r in selected} == {0, 1, 2, 3}:
                value = float(np.mean([r[field] for r in selected]))
                means.append(value)
                axis.scatter(value, np.mean([r["novel_auc"] for r in selected]), color=COLORS[arm], marker="X", s=65)
        if len(means) == 5:
            axis.axvspan(min(means), min(means) + .1, color="#21866c", alpha=.08)
            axis.axvline(max(means), color="#555555", linestyle=":", linewidth=.8)
        axis.set(title=f"Development: {label} versus acquisition", xlabel=f"{label} deterioration (px; lower better)",
                 ylabel=f"Rotated A({N}) MSE (px squared)")
    selected = _selected_development(snapshot, result)
    for arm in ARMS:
        group = [r for r in selected if r["recipe"]["arm"] == arm]
        curves = [analysis.novel_curve(r["metrics"], snapshot["cfg"]) for r in group]
        for row, curve in zip(group, curves):
            axes[2].plot(curve["n"], np.sqrt(curve["old_mse"]), color=COLORS[arm], alpha=.2, linewidth=.7)
            if arm == "B":
                axes[3].plot(curve["n"], curve["novel_mse"], color=COLORS[arm], alpha=.4, linewidth=.8,
                             label=f"B1 member {row['replica']}")
        if len(group) == 4 and {r["replica"] for r in group} == {0, 1, 2, 3}:
            axes[2].plot(curves[0]["n"], np.mean([np.sqrt(c["old_mse"]) for c in curves], axis=0),
                         color=COLORS[arm], label=arm + " pooled four")
            if arm == "B":
                axes[3].plot(curves[0]["n"], np.mean([c["novel_mse"] for c in curves], axis=0),
                             color=COLORS[arm], linewidth=2, label="B1 pooled four")
        if arm == "B":
            historical = [original._flat(r) for r in snapshot["records"] if r.get("status") == "completed"
                and r["metadata"]["cohort"] == "development" and r["metadata"]["recipe"] == {**recipes["B"], "ridge": 10.}
                and r["metadata"]["replica"] in {v["replica"] for v in group}]
            old_curves = [analysis.novel_curve(r["metrics"], snapshot["cfg"]) for r in historical]
            for axis, field in ((axes[2], "old_mse"), (axes[3], "novel_mse")):
                for row, curve in zip(historical, old_curves):
                    y = np.sqrt(curve[field]) if field == "old_mse" else curve[field]
                    axis.plot(curve["n"], y, color="#333333", linestyle="--", alpha=.3, linewidth=.7,
                              label=f"B10 member {row['replica']}" if axis is axes[3] else None)
    for axis, title, ylabel in ((axes[2], "Development old paths; B10 dashed", "Old RMS (px; lower better)"),
                              (axes[3], "Paired development ridge response: B1 / B10", "Rotated MSE (px squared)")):
        axis.set(title=title, xlabel="Outbound grids", ylabel=ylabel)
    for axis in axes:
        if not axis.lines and not axis.collections:
            original._pending(axis, axis.get_title())
        original._legend(axis)
    return figure


def _prediction(snapshot, result):
    figure = broad._prediction(snapshot, result)
    for arm, method in result["methods"].items():
        for axis, field in zip(figure.axes[:2], ("novel_mse", "old_mse")):
            for values in method["curves"][field]["per_replica"]:
                axis.plot(method["curves"]["n"], np.sqrt(np.asarray(values, dtype=float)),
                          color=COLORS[arm], alpha=.15, linewidth=.6)
    axis = figure.axes[4]
    axis.clear()
    for name, item in result["efficiency"].items():
        point = item["point"]
        if name != "A-A" and point.get("S") is not None:
            axis.scatter(point["matched_time_ratio"], point["saved_fraction"], edgecolors=COLORS[point["arm"]],
                facecolors=COLORS[point["arm"]] if item["retention_supported"] else "none",
                marker="s" if name == "E-B" else "o", label=name)
    if not axis.collections:
        original._pending(axis, "Sample equivalence")
    axis.set(title="Calibrated A target; E-B secondary", xlabel="Matched compute multiplier (lower better)",
             ylabel="Saved fraction (higher better)")
    original._legend(axis)
    return figure


def _ridge_motion(snapshot, result):
    figure, axes = original._figure(2, 3)
    fields = ("c", "s", "kappa", "omega", "omega_kappa", "ltm_move")
    for arm, methods in result["scalar_diagnostics"]["methods"].items():
        for axis, field in zip(axes, fields):
            item = methods[field]
            if item["incoming_grids"]:
                axis.plot(item["incoming_grids"], item["mean"], color=COLORS[arm], label=arm + (" LTM" if field == "ltm_move" else ""))
        item = methods["stm_move"]
        if item["incoming_grids"]:
            axes[5].plot(item["incoming_grids"], item["mean"], color=COLORS[arm], linestyle="--", label=arm + " STM")
    for axis, title in zip(axes, ("Ridge ratio c", "Saved archive mean s", "Saved kappa", "Applied odds omega", "Effective isotropic omega*kappa", "Accepted LTM / STM displacement")):
        if not axis.lines:
            original._pending(axis, title)
        axis.set(title=title, xlabel="All incoming grids", ylabel="Euclidean norm" if axis is axes[5] else title)
        broad._shade(axis, snapshot["cfg"])
        original._legend(axis)
    return figure


def _bootstrap_histogram(snapshot, result):
    figure, axes = original._figure(1, 2)
    for axis, name in zip(axes, ("B-A", "E-A")):
        item = result["bootstrap_saved_fractions"].get(name, {})
        if item.get("status") == "estimable":
            axis.hist(item["saved_fraction"], bins=20, color=COLORS[name[0]], alpha=.65)
            for bound in result["efficiency"][name]["ci"]["saved_fraction"]:
                axis.axvline(bound, color="#555555", linestyle=":", linewidth=1)
            axis.axvline(item["point_from_mean_curves"], color="#222222", label="Mean-curve point")
            axis.axvline(item["median"], color="#555555", linestyle="--", label="Bootstrap median")
        else:
            axis.text(.5, .5, "Withheld: unsupported crossings" if item else "Awaiting >=2 fresh replicas per sign",
                      ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axis.set(title=name + " whole-pair bootstrap", xlabel="Saved fraction (higher better)", ylabel="Draws")
        original._legend(axis)
    return figure


def _progress(snapshot, result):
    status, gate = snapshot["status"], result["calibration"]
    complete = [original._flat(r) for r in snapshot["records"] if r.get("status") == "completed"]
    counts = Counter(r["cohort"] for r in complete)
    imported = sum(r["cohort"] == "development" and bool(r.get("imported") or r.get("origin_root")) for r in complete)
    expected = max(20, counts["development"], status.get("expected_development_paths", 20))
    blocks = "; ".join(f"{name} {c['n_pairs']}/{c['planned_pairs']} paired blocks" for name, c in result["cohorts"].items())
    additional = status.get("additional_active_seconds")
    baseline, active = status.get("baseline_seconds"), status.get("active_seconds")
    if additional is None and baseline is not None and active is not None:
        additional = max(0., active - baseline)
    cap = status.get("incremental_allowance_seconds", 72000)
    fmt = lambda seconds: original._fmt(None if seconds is None else seconds / 3600)
    current = status.get("current") or {}
    proposal = status.get("proposal_round", snapshot.get("development", {}).get("round", len(snapshot.get("refinement_decisions", []))))
    return ("# Plan 18 Progress\n\n"
        f"**Phase 7: {status.get('phase', '7A implementation')}** | {status.get('status', 'preparing')}.  \n"
        f"Development {counts['development']}/{expected} paths: imported {imported}, new {counts['development'] - imported}; "
        f"active proposal round {proposal}. {blocks}.  \n"
        f"Calibration: **{gate['status']}**; passing four-member B/ridge-1 gate: {'yes' if gate['passed'] else 'no'}. "
        "Resource reservation is separately required for confirmation.  \n"
        f"Additional charged active hours: {fmt(additional)} / {fmt(cap)} h; "
        f"family-cumulative: {fmt(active)} h; launch baseline: {fmt(baseline)} h; "
        f"remaining forecast: {fmt(status.get('forecast_remaining_seconds', (status.get('forecast') or {}).get('remaining_seconds')))} h.  \n"
        f"Current unit/batch: {current.get('unit', 'none')} / {current.get('batch', 'none')}/{current.get('total', 'none')}.\n\n"
        "Only complete five-arm fresh blocks enter inference. Intervals and supported savings are withheld "
        "until each rotation sign has at least two fresh replicas. Phase 0--6 cells and qualifications remain intact below. "
        "B defines development retention; calibrated A defines primary sample equivalence.")


def _settings(snapshot, result):
    recipes = _display_recipes(snapshot, result)
    selected = _selected_development(snapshot, result)
    rows = []
    for arm in ARMS:
        recipe = recipes[arm]
        group = [r for r in selected if r["recipe"]["arm"] == arm]
        def pooled(field):
            return (original._fmt(np.mean([r[field] for r in group])) + " (4/4)"
                    if len(group) == 4 and {r["replica"] for r in group} == {0, 1, 2, 3}
                    else f"pending ({len(group)}/4)")
        diagnostics = result["scalar_diagnostics"]["methods"][arm]
        effective = " / ".join(original._fmt(diagnostics[k]["pooled_mean"]) for k in ("s", "kappa", "omega", "omega_kappa"))
        rows.append([arm + (" retention reference" if arm == "B" else " sample reference" if arm == "A" else " " + analysis.LABELS[arm]),
            recipe["ridge"], recipe["strength"], "fixed .025" if arm in "ABC" else "velocity Kalman",
            pooled("Havg"), pooled("H100"), effective])
    N = result["analysis_outbound_count"]
    ranges = result["calibration"]["matching"].get("ranges", {})
    separation = None
    curves = {arm: [analysis.novel_curve(r["metrics"], snapshot["cfg"])["old_mse"] for r in selected if r["recipe"]["arm"] == arm] for arm in ARMS}
    if all(len(c) == 4 for c in curves.values()):
        separation = float(np.ptp([np.sqrt(c).mean(axis=0) for c in curves.values()], axis=0).max())
    reference = [r for r in selected if r["recipe"]["arm"] == "B"]
    runtime = original._fmt(np.sum([r["learning_seconds"] for r in reference])) if reference else "pending"
    ridge_gaps = []
    for row in reference:
        historical = [original._flat(r) for r in snapshot["records"] if r.get("status") == "completed"
            and r["metadata"]["cohort"] == "development" and r["metadata"]["replica"] == row["replica"]
            and r["metadata"]["recipe"] == {**recipes["B"], "ridge": 10.}]
        if len(historical) == 1:
            ridge_gaps.append([row[k] - historical[0][k] for k in ("Havg", "H100", "novel_auc")])
    ridge_note = (" / ".join(original._fmt(v) for v in np.mean(ridge_gaps, axis=0)) if ridge_gaps else "pending")
    return ("### Phase 7 Settings And Development\n\n" + original._table(
        ("Policy / role", "Ridge c", "FD strength", "pi", "Havg (px)", f"H{N} (px)", "Mean s / kappa / omega / omega*kappa"), rows)
        + ("\nFrozen selection. " if result["selection"] else "\nDeclared starting recipes only; no selection is frozen. ")
        + "B is fixed at ridge 1; calibrated A remains the sample-equivalence reference. "
        "Four reused development members (0--3), never historical confirmation, define calibration. "
        "Roots are taken within each member before pooling. Both all-arm ranges must be <= .10 px; "
        "proximity to B alone and endpoint matching alone are insufficient. There is no absolute .50 px ceiling. "
        f"Current Havg / H{N} ranges: {original._fmt(ranges.get('Havg'))} / {original._fmt(ranges.get('H100'))} px; "
        f"maximum mean pointwise old-RMS separation: {original._fmt(separation)} px. "
        f"B1 learning time on {len(reference)}/4 development members: {runtime} s. "
        f"Paired B1-B10 development gaps (Havg / H{N} / acquisition MSE), {len(ridge_gaps)}/4 members: {ridge_note}. "
        "This is a development ridge-response diagnostic, not fresh efficacy evidence. "
        f"Effective-scale summaries use {result['scalar_diagnostics']['cohort']} scalar diagnostics, equal member weight; missing means pending. "
        "FIFO32/replay2 and 3,268 LTM + 108 STM coordinates are unchanged; D/E process multiplier is 1e-5, not a gain or pi. "
        + " ".join(f"[{name}]({href})" for name, href in snapshot.get("artifact_hrefs", {}).items()))


def _effects(result):
    return phase6._effects(result).replace("Phase 6 Paired Effects", "Phase 7 Paired Effects")


def _retention(result):
    return phase6._retention(result)


def _efficiency(result):
    crossings = [[name, item["target_mse"],
                  None if item["first_crossing"] is None else f"{item['first_crossing']:.1f}",
                  None if item["persistent_crossing"] is None else f"{item['persistent_crossing']:.1f}"]
                 for name, item in result["crossing_diagnostics"].items()]
    return (broad._efficiency(result).replace("Phase 5 Sample Equivalence", "Phase 7 Sample Equivalence")
        + "\n\n**S: fraction of samples saved divided by the compute multiplier at the same new-data test quality.** "
        "Points come from mean trajectories, not the median of bootstrap savings. Every draw recomputes the "
        "reference endpoint target and the method's sustained last downward crossing from whole paired scalar curves. "
        "Failures are retained, never converted to zero or discarded; any failure withholds the histogram and inverse interval. "
        "Phase 6/7 use different calibrated A targets, so their savings are not directly comparable effect sizes.\n\n"
        + original._table(("Target", "Target MSE", "First crossing (grids)", "Persistent last crossing (grids)"),
                          crossings or [["Pending", None, None, None]])
        + "\nA-A shows A's own persistent crossing; its allocated-budget saving is zero by definition. "
        "E-B is a separate secondary target, never a row in the common-A ranking.")


def _conclusion(snapshot, result):
    gate = result["calibration"]
    if not gate["passed"]:
        return ("### Phase 7 Conclusion\n\n"
            f"**Calibration: {gate['status']}; no passing freeze is established.** "
            "Fresh confirmation is not authorized. Development evidence describes the low-ridge calibration frontier, "
            "not STM efficacy. An infeasible or budget-limited comparison is not evidence of method failure.")
    retained = sum(v["equivalent"] for v in result["retention_equivalence"].values())
    effects = " ".join(f"{name} acquisition-risk difference: {original._effect(result['comparisons'][name]['novel_auc'])} px squared."
                       for name in ("B-A", "E-B", "E-A") if name in result["comparisons"])
    supported = [name for name, item in result["efficiency"].items() if name != "A-A" and item["sample_saving_supported"]]
    complete = result["status"] == "completed"
    primary = result["efficiency"].get("B-A", {})
    point = primary.get("point", {})
    primary_note = ""
    if point.get("status") == "valid":
        bounds = (primary.get("ci") or {}).get("saved_percent")
        interval = f" (nominal 95% interval {bounds[0]:.3f}--{bounds[1]:.3f}%)" if bounds else " (inverse interval unavailable)"
        primary_note = (f"\n\n**Primary STM comparison:** B's mean curve reaches A's endpoint target at "
            f"{point['n']:.1f} of {result['analysis_outbound_count']} allocated grids: "
            f"{point['saved_percent']:.3f}% endpoint-budget saving{interval}, "
            f"with {point['matched_time_ratio']:.2f}x matched learning compute and S={point['S']:.5f}. ")
        risk_bounds = result["comparisons"].get("B-A", {}).get("novel_auc", {}).get("ci")
        if risk_bounds and risk_bounds[0] <= 0 <= risk_bounds[1]:
            primary_note += "The average acquisition-risk interval includes zero. "
        clock = result["crossing_diagnostics"].get("A-A", {})
        if clock.get("persistent_crossing") is not None:
            primary_note += (f"A's own sustained crossing is {clock['persistent_crossing']:.1f} grids. "
                "Endpoint-budget saving alone is not additional timing advantage if A already reaches its target equally early. ")
    return ("### Phase 7 Conclusion\n\n**Four-member low-ridge calibration passed.** "
        f"Fresh two-sided retention equivalence: {retained}/10 contrasts support both statistics. "
        + (effects or "Acquisition performance awaits complete fresh paired blocks.")
        + primary_note
        + "\n\nPositive endpoint-budget savings satisfying the defined full/matched-budget retention checks: "
        + (", ".join(supported) if supported else "not established")
        + ". The reported matched compute multipliers and S determine whether savings offset learning cost. "
        "These one-sided retention checks allow better old-data retention; they do not establish two-sided equivalence. "
        "First and sustained crossings can differ on nonmonotone curves; endpoint-budget savings are not necessarily "
        "faster first learning. Calibration does not assume an STM benefit. Different calibrated ridges limit "
        "mechanistic interaction claims; neither favorable nor null results establish universally optimal ridge or pi. "
        "Phase 6 remains valid at its different operating point. "
        + ("The fixed eight-changing/two-stationary allocation is complete; integrity closeout is separately recorded."
           if complete else "Allocation is incomplete; this is a rolling report, not scientific closeout."))


def _notebook(snapshot, result, history, figures):
    history.cells.insert(0, nbformat.v4.new_markdown_cell(_progress(snapshot, result)))
    cells = [nbformat.v4.new_markdown_cell(
        "## Phase 7: Low-Ridge STM With Recalibrated Retention\n\n"
        "Ridge may suppress the freedom through which STM helps. This historically informed probe fixes B at ridge 1 "
        "and recalibrates A/C/D/E to its retention; replacing B alone would violate the earlier matching tolerance. "
        "It does not invalidate Phase 6 or presume recovered STM benefit. The whole-stream law, held-out panels, "
        "outbound-grid clock and frozen learner are unchanged. Eight new changing and two new stationary replicas "
        "are separate from four reused development members and from all historical confirmation. "
        "[Design](plan18.md#phase-7-low-ridge-stm-with-recalibrated-retention)."),
        nbformat.v4.new_markdown_cell(_settings(snapshot, result))]
    captions = {
        "development": ("Development Retention And Acquisition", "Individual markers identify members 0--3; X is the pooled four. "
            "Faint points retain other candidates. Each .10 px band applies separately. Thin old paths and individual B1 new paths "
            "retain member variation; black dashed B10 paths are paired development diagnostics. Summary matching need not be pointwise matching."),
        "prediction": ("Fresh Prediction And Sample Efficiency", "Held-out unregularized prediction: lower MSE/RMS is better; "
            "higher one-pixel accuracy is better. Thin paths/endpoint points are replicas; bands are nominal pointwise 95% when >=2/sign. "
            "Dashed frozen initialization is distinct from pre-novel n=0. Current-law paths use all arrivals, not the acquisition clock. "
            "Hollow efficiency points lack retention support; E-B squares use a separate target."),
        "diagnostics": ("Pi, Gain And Motion Diagnostics", "Applied/raw pi, level/velocity gains, one-grid V versus batch V/m, "
            "motion plus uncertainty and pre-PSD LFU health come from saved scalar diagnostics. Gain p10/p90 are coordinate quantiles, "
            "not confidence intervals. Complete fresh paired means are used when available; otherwise development diagnostics are descriptive. "
            "Stationary paths have no acquisition clock or S."),
        "ridge_motion": ("Effective Ridge And STM/LTM Movement", "Saved archive scale s and kappa are not reconstructed from a model. "
            "Odds omega=(1-pi)/pi use the recorded applied pi; omega*kappa is effective isotropic curvature. "
            "Solid/dashed displacement norms are actual accepted LTM/STM moves in the fixed chart. Missing scalar fields remain missing."),
        "bootstrap": ("Saved-Fraction Bootstrap", "Whole paired trajectories are resampled within global-sign strata, "
            "with the A endpoint target, crossings, compute and retention recomputed together. Dotted lines are nominal 95% bounds; "
            "solid is the mean-curve point, dashed the bootstrap median. Early/late crossing mixtures may be asymmetric. "
            "Histograms are withheld if any crossing is unsupported or either sign has fewer than two replicas."),
    }
    for name, png in figures:
        title, caption = captions[name]
        attachment = f"phase7-{name}.png"
        cell = nbformat.v4.new_markdown_cell(f"### {title}\n\n![{title}](attachment:{attachment})\n\n{caption}")
        cell["attachments"] = {attachment: {"image/png": base64.b64encode(png).decode("ascii")}}
        cells.append(cell)
    cells.extend(nbformat.v4.new_markdown_cell(text) for text in (
        _effects(result), _retention(result), _efficiency(result), _conclusion(snapshot, result)))
    history.cells.extend(cells)
    history.metadata["plan18_phase7_report"] = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL,
        source_digest=snapshot["source_digest"], history=snapshot["inputs"]["history"])
    nbformat.validate(history)
    return nbformat.writes(history).encode()


def build_report(root=DEFAULT_ROOT, cfg=None, notebook_path=None, check=lambda: None, force=False, *, history_root=None):
    """Publish a bounded immutable scalar generation under a single-writer lease."""
    root = Path(root)
    report = root / "report"
    with original._lease(report, check):
        snapshot = _snapshot(root, cfg, check)
        smoke = setting(snapshot["cfg"], "smoke", False)
        notebook = root / "smoke_results.ipynb" if smoke else Path(notebook_path or DEFAULT_NOTEBOOK)
        history, seal = _history(Path(history_root or HISTORY_ROOT), smoke, check)
        snapshot["inputs"]["history"] = seal
        snapshot["artifact_hrefs"] = {name: os.path.relpath(root / name, notebook.parent)
            for name in ("identity.json", "development.json", "selection.json", "status.json", "calibration_completed.json")
            if name in snapshot["inputs"]["files"]}
        snapshot["artifact_hrefs"]["exhaustive scalar analysis"] = os.path.relpath(report / "latest.json", notebook.parent)
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
                if (pointer.get("schema") != REPORT_SCHEMA or not isinstance(generation, str) or not generation
                        or Path(generation).name != generation or generation in (".", "..")):
                    raise ValueError("invalid report generation pointer")
        destination = report / generation
        if destination.exists() or destination.is_symlink():
            metadata = _verified_generation(destination, digest, check)
            if metadata.get("protocol") != PROTOCOL:
                raise ValueError("incompatible Phase 7 report protocol")
            data = (destination / "results.ipynb").read_bytes()
            changed = False
        else:
            result = _summarize(snapshot, check)
            temporary = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=report))
            try:
                files, figures = dict(_SOURCE_BYTES), []
                for name, render in (("development", _development), ("prediction", _prediction),
                                     ("diagnostics", broad._diagnostics), ("ridge_motion", _ridge_motion),
                                     ("bootstrap", _bootstrap_histogram)):
                    check()
                    figure = render(snapshot, result)
                    try:
                        for axis in figure.axes:
                            axis.title.set_fontsize(9)
                            axis.xaxis.label.set_fontsize(8)
                            axis.yaxis.label.set_fontsize(8)
                        output = io.BytesIO()
                        figure.savefig(output, format="png", dpi=100, metadata={"Software": "Plan18 Phase7 scalar report"})
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
