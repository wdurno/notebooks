"""Artifact-only Phase 9 reporting.

Public API: ``publish(snapshot, notebook=DEFAULT_NOTEBOOK, store=None)`` returns
generation metadata. A store supplies ``root`` and optional ``reserve(bytes)``.
Only JSON scalars are accepted; this module never loads model checkpoints.

Snapshots use protocol ``plan18-replicated-combination-v1`` and schema_version
1. Progress, settings, calibration, fidelity, changes and diagnostics can arrive
before analysis. Analysis accepts complete paired R/A/B/C/D/E curves keyed by
policy, each with n (or observations) and query/base mean_mse and optional ci_mse.
Summaries and signed contrasts follow the analyzer's native keys. Its
efficiency[policy] uses A and efficiency_vs_B[policy] uses B; nested
referenceA/referenceB exports are also accepted. Unavailable fields remain
unresolved. An acquisition
gain is explicitly negated when converted to a lower-is-better difference.
Development candidates and diagnostic trajectories are optional scalar exports.

``build_cells`` is pure presentation. ``load_analysis``, ``load_snapshot`` and
``display_figures`` let notebook cells rerender the sealed scalar exports. Report
generations include the exact renderer source, inputs, figures and checksums.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import nbformat

PROTOCOL = "plan18-replicated-combination-v1"
ANALYSIS_PROTOCOL = "plan18-replicated-combination-analysis-v1"
REPORT_SCHEMA = 1
TAG, PREFIX = "plan18-phase9", "plan18-phase9-"
REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "cache/mnist_experiment/plan18/replicated_combination_v1"
DEFAULT_NOTEBOOK = REPO / "mnist_experiment/plan18_results.ipynb"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 64 * 1024 * 1024
MAX_GENERATION_BYTES = 160 * 1024 * 1024
POLICIES = ("R", "A", "B", "C", "D", "E")
CONTRASTS = ("B-R", "C-B", "D-B", "E-B", "B-A", "E-A", "E-R")
LABELS = dict(R="R: F1 original", A="A: Diagonal Replay/EWC", B="B: Diagonal F1",
              C="C: F1 + LFU", D="D: F1 + adaptive pi", E="E: Combined")
COLORS = dict(R="#444444", A="#2878a3", B="#be5078", C="#32936f", D="#967344", E="#7c55a0")
_SOURCE_BYTES = Path(__file__).read_bytes()


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _json_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("scalar report mappings require string keys")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("report accepts finite JSON scalars, not model/tensor artifacts")


def _bytes(value):
    return (json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _read(path, limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"invalid or oversized report artifact: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("bounded artifact size exceeded")
    return data


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _number(value, digits=3):
    return f"{value:.{digits}g}" if _finite(value) else "unavailable"


def _text(value):
    return html.escape(str(value), quote=False).replace("|", "/").replace("\n", " ")


def _first(item, *keys):
    return next((item[k] for k in keys if k in item), None)


def _effect(value, scale=1., interval=True):
    item = _mapping(value)
    point = _first(item, "mean", "point", "estimate") if item else value
    result = _number(point * scale if _finite(point) else None)
    ci = _first(item, "ci", "ci95", "confidence_interval")
    if interval and _finite(point) and isinstance(ci, list) and len(ci) == 2 and all(_finite(v) for v in ci):
        if ci[0] > ci[1]:
            raise ValueError("reversed scalar confidence interval")
        result += " [" + ", ".join(_number(v * scale) for v in ci) + "]"
    return result


def _table(headers, rows):
    return "\n".join("| " + " | ".join(_text(v) for v in row) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def owned(item):
    return item.get("metadata", {}).get(TAG) is True or item.get("id", "").startswith(PREFIX)


def cell(name, source):
    return nbformat.v4.new_markdown_cell(source, id=PREFIX + name, metadata={TAG: True})


def _curve(raw, domain):
    item = _mapping(raw)
    metric = _mapping(item.get(domain))
    y = metric.get("mean_mse")
    if y is None:
        return None
    x = _first(item, "n", "observations")
    if (not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y)
            or any(not _finite(v) or v < 0 for v in x + y)
            or any(b <= a for a, b in zip(x, x[1:]))):
        raise ValueError(f"invalid {domain} curve or observation clock")
    ci = metric.get("ci_mse")
    if ci is not None:
        if (not isinstance(ci, list) or len(ci) != 2 or any(not isinstance(v, list) or len(v) != len(y) for v in ci)
                or any(not _finite(v) or v < 0 for bound in ci for v in bound)
                or any(lo > hi for lo, hi in zip(*ci))):
            raise ValueError("invalid curve confidence interval")
    return x, y, ci


def _validate_analysis(value):
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("scalar analysis must be a mapping or None")
    data = _json_value(value)
    if data.get("protocol", PROTOCOL) not in (PROTOCOL, ANALYSIS_PROTOCOL) or data.get("schema_version", 1) != 1:
        raise ValueError("incompatible Phase 9 analysis schema/protocol")
    if data.get("namespace", "fresh") != "fresh" or data.get("pairing_complete") is False:
        raise ValueError("analysis must contain fresh complete paired blocks")
    replicas = data.get("replicas", [])
    if (not isinstance(replicas, list) or any(type(r) not in (str, int) for r in replicas)
            or len(set(replicas)) != len(replicas) or len(replicas) > 8):
        raise ValueError("invalid paired replica identifiers")
    curves, summaries = data.get("curves", {}), data.get("summaries", {})
    if not isinstance(curves, Mapping) or not isinstance(summaries, Mapping):
        raise ValueError("analysis curves/summaries must be mappings")
    if curves and (set(curves) != set(POLICIES) or not replicas):
        raise ValueError("curves require complete six-policy paired blocks")
    if curves and set(summaries) != set(POLICIES):
        raise ValueError("paired curves require complete policy summaries")
    if summaries and set(summaries) != set(POLICIES):
        raise ValueError("summaries require complete six-policy paired blocks")
    for policy, raw in curves.items():
        if any(_curve(raw, domain) is None for domain in ("query", "base")):
            raise ValueError(f"incomplete paired query/base curves for {policy}")
    if curves:
        clock = _curve(curves["R"], "query")[0]
        if any(_curve(raw, "query")[0] != clock for raw in curves.values()):
            raise ValueError("paired curves must share the same exposure clock")
        exposures = [raw.get("observations") for raw in curves.values()]
        if any(v is not None for v in exposures) and any(v != exposures[0] for v in exposures):
            raise ValueError("paired curves must share total exposure summaries")
    if any(not isinstance(value, Mapping) for value in summaries.values()):
        raise ValueError("policy summaries must be scalar mappings")
    if data.get("is_final") is True and (len(replicas) != 8 or not curves or not summaries or data.get("provisional") is True):
        raise ValueError("final analysis requires eight complete paired blocks")
    return data


def _validate_snapshot(value):
    if not isinstance(value, Mapping):
        raise ValueError("report snapshot must be a mapping")
    snapshot = _json_value(value)
    if snapshot.get("protocol", PROTOCOL) != PROTOCOL or snapshot.get("schema_version", 1) != 1:
        raise ValueError("incompatible Phase 9 snapshot schema/protocol")
    _validate_analysis(snapshot.get("analysis"))
    return snapshot


def load_analysis(path, expected_sha256=None):
    """Read only a bounded scalar analysis export, with optional digest binding."""
    data = _read(Path(path), MAX_JSON_BYTES)
    if expected_sha256 is not None and _hash(data) != expected_sha256:
        raise ValueError("scalar analysis checksum mismatch")
    return _validate_analysis(json.loads(data))


def load_snapshot(path, expected_sha256=None):
    """Read only bounded JSON needed for development and diagnostic figures."""
    data = _read(Path(path), MAX_JSON_BYTES)
    if expected_sha256 is not None and _hash(data) != expected_sha256:
        raise ValueError("scalar snapshot checksum mismatch")
    return _validate_snapshot(json.loads(data))


def _status(value):
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    item = _mapping(value)
    return str(item.get("status", "pending") if item else value or "pending")


def _hours(progress, name):
    value = progress.get(name + "_hours")
    seconds = progress.get(name + "_seconds")
    return value if _finite(value) else seconds / 3600 if _finite(seconds) else None


def _count(counts, keys, target):
    raw = _first(counts, *keys)
    item = _mapping(raw)
    done = _first(item, "completed", "done") if item else raw
    total = _first(item, "planned", "total", "expected") if item else None
    return f"{_number(done, 4)}/{_number(total if total is not None else target, 4)}"


def _progress(snapshot):
    p = _mapping(snapshot.get("progress"))
    counts = dict(_mapping(p.get("counts")))
    for key in ("completed_paths", "fresh_blocks", "fresh_blocks_completed", "fresh_paths", "fresh_completed"):
        if key in p:
            counts.setdefault(key, p[key])
    current = p.get("current")
    if isinstance(current, Mapping):
        current = ", ".join(f"{k}={v}" for k, v in current.items() if not isinstance(v, (dict, list)))
    terminal = p.get("status") in ("completed", "calibration_infeasible", "calibration_budget_limited")
    forecast = 0. if terminal else _hours(p, "forecast")
    return cell("progress", "### Phase 9 Progress\n\n"
        f"**{_text(p.get('stage', '9A implementation'))}: {_text(p.get('status', 'pending'))}**. "
        + (f"Current: {_text(current)}. " if current else "")
        + f"Calibration: **{_text(_status(snapshot.get('calibration', snapshot.get('calibration_manifest'))))}**.  \n"
        f"Fresh paths {_count(counts, ('fresh_paths', 'fresh_completed', 'completed_paths', 'fresh', 'paths'), 48)}; "
        f"paired blocks {_count(counts, ('fresh_blocks', 'fresh_blocks_completed', 'paired_blocks'), 8)}.  \n"
        + (f"Development paths {_text(p['development_completed'])}; " if 'development_completed' in p else "")
        + (f"candidate recipes {_text(p['candidates'])}.  \n" if 'candidates' in p else "")
        + f"Elapsed {_number(_hours(p, 'elapsed'))} h; charged {_number(_hours(p, 'charged'))} h; "
        f"remaining allowance {_number(_hours(p, 'remaining'))} h; forecast {_number(forecast)} h. "
        "Only complete six-condition blocks enter fresh summaries; interim estimates are provisional.")


def _settings(snapshot):
    raw = snapshot.get("settings") or {}
    if isinstance(raw, list):
        settings = {str(_first(_mapping(v), "policy", "id", "arm")): _mapping(v) for v in raw}
    else:
        settings = _mapping(raw)
    rows = []
    for policy in POLICIES:
        item = _mapping(settings.get(policy))
        ridge = _first(item, "ridge", "ridge_ratio", "c")
        rows.append([LABELS[policy], "rank 64 + diagonal" if policy == "R" else "diagonal",
            "off" if policy == "A" else "3.34056", "on" if policy in ("C", "E") else "off",
            "Velocity Kalman" if policy in ("D", "E") else ".025",
            _number(ridge) if ridge is not None else "1 (fixed)" if policy in ("R", "B") else "calibration pending"])
    scales = [(policy, _mapping(settings.get(policy)).get("process_scale")) for policy in ("D", "E")]
    scales = [(policy, value) for policy, value in scales if _finite(value)]
    selected_scale = _mapping(_mapping(snapshot.get("calibration")).get("process_scale")).get("process_scale")
    selected = _finite(selected_scale)
    if selected:
        scales = [(policy, selected_scale) for policy in ("D", "E")]
    return cell("settings", "### Settings\n\n" + _table(
        ("Policy", "EWC", "FD strength", "LFU", "pi", "Ridge c"), rows)
        + ("\n\n" + ("Selected" if selected else "Configured") + " Kalman process-noise scales: "
           + ", ".join(p + "=" + _number(v) for p, v in scales) + "." if scales else "")
        + "\n\nAll policies have 3,268 LTM and 108 head-rank-one STM parameters, FIFO32/replay2, "
        "and 100 designated novel grids; each grid contains nine constituent digit pairs. "
        "Replay/EWC retains the same trainable adapter capacity. "
        "R is the original low-rank F1 policy; B is its diagonal-only version. "
        "LFU is applied only to diagonal LTM archives. The STM penalty is recomputed at current parameters."
        "\n\n**B-R holds both recipes fixed and measures representation. A-E are retention-calibrated "
        "to fixed B using development data. R is outside that matching requirement.**")


def _calibration(snapshot):
    return _mapping(snapshot.get("calibration", snapshot.get("calibration_manifest")))


def _summary(item, name):
    item = _mapping(item)
    keys = dict(acquisition=("acquisition", "acquisition_risk", "novel_risk", "average_risk"),
                retention_avg=("retention_avg", "Havg", "H_avg", "retention"),
                retention_endpoint=("retention_endpoint", "H100", "H_100", "endpoint_retention"),
                frozen=("frozen_mse", "frozen_new_mse"),
                new_endpoint=("new_endpoint", "query_endpoint", "endpoint_mse"),
                old_endpoint=("old_endpoint", "base_endpoint"))
    value = _first(item, *keys[name])
    if name.startswith("retention_") and (value is None or isinstance(value, Mapping) and "mean" not in value):
        value = _first(_mapping(item.get("retention")),
                       *(('avg', 'average') if name == "retention_avg" else ('endpoint', '100')))
    return value


def _mean(value):
    return _first(_mapping(value), "mean", "point", "estimate") if isinstance(value, Mapping) else value


def _candidates(snapshot):
    calibration = _calibration(snapshot)
    development = _mapping(snapshot.get("development", calibration.get("development")))
    raw = development.get("candidates", calibration.get("candidates", []))
    result = []
    if isinstance(raw, Mapping):
        for policy, entries in raw.items():
            if isinstance(entries, list):
                result.extend(dict(_mapping(v), policy=policy) for v in entries)
            elif isinstance(entries, Mapping):
                result.append(dict(entries, policy=policy))
    elif isinstance(raw, list):
        result = [dict(v) for v in raw if isinstance(v, Mapping)]
    return result


def development_curves(records, ridges):
    """Export descriptive selected-policy means from verified scalar records."""
    if set(ridges) != set(POLICIES) or any(r.get("namespace") != "development" for r in records):
        raise ValueError("six-policy development records and explicit ridges required")
    selected = [r for r in records if r["recipe"]["ridge"] == ridges[r["policy"]]]
    grouped = {(r["policy"], r["replica"]): r for r in selected}
    expected = {(p, r) for p in POLICIES for r in range(4)}
    if set(grouped) != expected or len(selected) != len(expected):
        raise ValueError("selected development curves need four complete six-policy blocks")
    clock = [v["n"] for v in selected[0]["metrics"]]
    if len(clock) < 3 or clock != list(range(len(clock))):
        raise ValueError("complete development novel-count clock required")
    for replica in range(4):
        reference = grouped["R", replica]
        for policy in POLICIES:
            row = grouped[policy, replica]
            if (any(row[k] != reference[k] for k in ("plan_hash", "pairing_id"))
                    or [v["n"] for v in row["metrics"]] != clock
                    or [v["observations"] for v in row["metrics"]] != [v["observations"] for v in reference["metrics"]]):
                raise ValueError("unpaired development trajectory")
            for domain in ("base_mse", "novel_mse"):
                if (row["metrics"][0][domain] != reference["metrics"][0][domain]
                        or any(not _finite(v[domain]) or v[domain] < 0 for v in row["metrics"])):
                    raise ValueError("invalid development risk or frozen initialization")
    curves = {}
    for policy in POLICIES:
        metrics = [grouped[policy, r]["metrics"] for r in range(4)]
        mean = lambda key: [math.fsum(row[i][key] / 4 for row in metrics) for i in clock]
        curves[policy] = dict(n=clock, observations=mean("observations"),
            query=dict(mean_mse=mean("novel_mse")), base=dict(mean_mse=mean("base_mse")))
    return dict(replicas=list(range(4)), curves=curves)


def _calibration_cell(snapshot):
    calibration = _calibration(snapshot)
    selected = _mapping(_first(calibration, "selected", "selected_recipes", "tuple", "policies"))
    terminal = _status(calibration) in ("calibration_infeasible", "calibration_budget_limited")
    ranges = _mapping(_mapping(_mapping(calibration.get("selection")).get("matching")).get("ranges"))
    range_note = ("\n\nMeasured pooled A-E ranges: average **" + _number(ranges.get("Havg"), 4)
                  + " px**; endpoint **" + _number(ranges.get("H100"), 4) + " px** (limit .01 px each)."
                  if ranges else "")
    rows = []
    for policy in POLICIES:
        item = _mapping(selected.get(policy))
        ridge = _first(item, "ridge", "ridge_ratio", "c")
        rows.append([policy, _number(1. if ridge is None and policy in ("B", "R") else ridge),
                     _effect(_summary(item, "retention_avg")), _effect(_summary(item, "retention_endpoint"))])
    return cell("calibration", "### Development Calibration\n\n"
        f"Selection: **{_text(_status(calibration))}**. "
        "The pooled four-replica A-E range must be at most .01 px for both average and endpoint "
        "old RMS deterioration; the separate absolute ceiling is .50 px. B and R keep c=1.\n\n"
        + _table(("Policy", "Proposed c" if terminal else "Selected c", "Average deterioration (px)",
                  "Endpoint deterioration (px)"), rows)
        + range_note
        + "\n\nA failed or budget-limited calibration is reported as such; it does not establish matched retention. "
        "Fresh equivalence requires two-sided intervals inside +/- .01 px. A nonsignificant difference alone is insufficient. "
        "All proposals, coverage, seed lists, Kalman scale selection and range checks remain in the linked scalar snapshot.")


def _fidelity(snapshot):
    fidelity = _mapping(snapshot.get("fidelity"))
    changes = snapshot.get("changes", [])
    if not isinstance(changes, list):
        changes = [changes] if changes else []
    notes = []
    for value in changes[:5]:
        item = _mapping(value)
        notes.append(_text(_first(item, "summary", "description", "message") if item else value))
    return cell("fidelity", "### Readiness And Recorded Changes\n\n"
        f"Fidelity and numerical acceptance: **{_text(_status(fidelity))}**. "
        "Same-seed acceptance and development paths are excluded from fresh inference. "
        + ("Recorded changes: " + "; ".join(notes) + ". " if notes else "No material changes recorded in this snapshot. ")
        + "Detailed identities, audit checks and changes are retained in the snapshot. "
        "A favorable efficacy result is not an acceptance requirement.")


def _contrast(item, name):
    item = _mapping(item)
    keys = dict(acquisition=("acquisition_difference", "acquisition_risk_difference", "novel_risk_difference", "acquisition"),
                endpoint=("endpoint_risk_difference", "endpoint_difference"),
                retention_avg=("retention_avg_difference", "Havg_difference", "retention_avg", "retention_difference"),
                retention_endpoint=("retention_endpoint_difference", "H100_difference", "retention_endpoint", "endpoint_retention_difference"))
    value = _first(item, *keys[name])
    if value is None and name == "acquisition" and item.get("acquisition_gain") is not None:
        gain = item["acquisition_gain"]
        if _finite(gain):
            return -gain
        gain = _mapping(gain)
        value = dict(gain)
        if _finite(gain.get("mean")):
            value["mean"] = -gain["mean"]
        if isinstance(gain.get("ci"), list) and len(gain["ci"]) == 2:
            value["ci"] = [-gain["ci"][1], -gain["ci"][0]]
    return value


def _effects(result):
    contrasts = _mapping(result.get("contrasts"))
    rows = []
    for name in CONTRASTS:
        item = _mapping(contrasts.get(name))
        acquisition = _contrast(item, "acquisition")
        favorable = item.get("favorable_replicas", _mapping(acquisition).get("favorable_replicas"))
        rows.append([name, _effect(acquisition), _effect(_contrast(item, "endpoint")),
            _effect(_contrast(item, "retention_avg")), _effect(_contrast(item, "retention_endpoint")),
            f"{favorable}/{len(result.get('replicas', []))}" if favorable is not None else "unavailable"])
    interaction = result.get("interaction", contrasts.get("interaction"))
    value = _contrast(interaction, "acquisition")
    rows.append(["E-C-D+B", _effect(interaction if value is None else value), "", "", "", ""])
    return cell("effects", "### Fresh Paired Effects\n\n" + _table(
        ("Difference", "Mean novel risk [95% CI]", "Endpoint risk", "Old average (px)", "Old endpoint (px)", "Favorable"), rows)
        + "\n\nDifferences subtract the second policy; negative risk differences favor the first. "
        "Novel risk averages held-out error over novel counts 1-100, in px^2 per landmark (54 coordinate errors / 27). "
        "Retention takes roots within each replica before averaging: average deterioration roots its budget-average old MSE, "
        "while endpoint deterioration roots its final old MSE; both subtract that replica's frozen old RMS. "
        "A small improvement over frozen initialization is possible as further nominal examples arrive. "
        "The interaction E-C-D+B concerns calibrated policies. Intervals use whole six-condition replica blocks; "
        "they are nominal paired 95% intervals, not familywise significance guarantees.")


def _efficiency_entries(result):
    raw = _mapping(result.get("efficiency"))
    entries = []
    # Native analyzer uses efficiency[policy] against A and efficiency_vs_B.
    for reference, mapping in (("A", raw), ("B", _mapping(result.get("efficiency_vs_B")))):
        for policy, item in mapping.items():
            if policy in POLICIES and isinstance(item, Mapping) and any(k in item for k in ("point", "status", "bootstrap")):
                entries.append((policy, reference, item))
    for reference, aliases in (("A", ("referenceA", "reference_A", "A")), ("B", ("referenceB", "reference_B", "B"))):
        group = _first(raw, *aliases)
        if not isinstance(group, Mapping):
            continue
        methods = _mapping(_first(group, "methods", "policies", "estimates") or group)
        for policy, item in methods.items():
            if policy in POLICIES and isinstance(item, Mapping) and not any(p == policy and r == reference for p, r, _ in entries):
                entries.append((policy, reference, item))
    for name, item in raw.items():
        if name in tuple(p + "-" + r for p in POLICIES for r in ("A", "B")) and isinstance(item, Mapping):
            policy, reference = name.split("-")
            if not any(p == policy and r == reference for p, r, _ in entries):
                entries.append((policy, reference, item))
    return entries


def _inverse(item):
    point, bootstrap = _mapping(item.get("point")), _mapping(item.get("bootstrap"))
    valid = point.get("status") == "valid"
    intervals = valid and bootstrap.get("invalid") == 0 and _finite(bootstrap.get("draws")) and bootstrap["draws"] > 0
    return point, bootstrap, _mapping(bootstrap.get("ci")) if intervals else {}, intervals


def _efficiency(result):
    rows = []
    for policy, reference, item in _efficiency_entries(result):
        point, bootstrap, ci, intervals = _inverse(item)
        valid = point.get("status") == "valid"
        def estimate(*keys, scale=1.):
            key = next((k for k in keys if k in point), keys[0])
            return _effect(dict(mean=point.get(key) if valid else None, ci=ci.get(key)), scale)
        full, matched = item.get("full_budget_retention_supported"), item.get("matched_budget_retention_supported")
        support = lambda v: "supported" if v is True else "not established" if v is False else "unavailable"
        matched_label = support(matched) if intervals else "not estimable"
        inverse_status = point.get("status", item.get("status", "unavailable"))
        if valid and not intervals:
            inverse_status += f"; CI withheld ({bootstrap.get('invalid', '?')}/{bootstrap.get('draws', '?')} invalid)"
        if point.get("useful_target") is False:
            inverse_status += "; weak reference target (<5% improvement)"
        equal = item.get("equal_budget_learning_time_ratio")
        rows.append([f"{policy}:{reference}", estimate("saved_fraction", scale=100),
            estimate("learning_time_ratio", "matched_time_ratio", "compute_ratio"), estimate("S"),
            _effect(dict(mean=equal, ci=item.get("equal_budget_learning_time_ratio_ci"))) if _finite(equal) else _effect(equal),
            support(full) + " / " + matched_label, inverse_status])
    return cell("efficiency", "### Sample And Compute Efficiency\n\n" + _table(
        ("Method:target", "Saved %", "Matched compute", "S", "Equal-budget compute", "Retention full / matched", "Inverse support"),
        rows or [["Replay/EWC A (primary); STM B (secondary)", "unavailable", "unavailable", "unavailable", "unavailable", "unavailable", "pending scalar analysis"]])
        + "\n\n**S is the fraction of samples saved divided by the compute ratio needed to reach the same predictive quality.** "
        "Saved % says how much less novel data reaches the reference endpoint; matched compute says what that costs. "
        "A is the primary Replay/EWC target; B is the secondary STM target. These differ from Phase 8's R2 target. "
        "Crossings use unsmoothed mean risk, the last sustained crossing, interpolation and no extrapolation. "
        "Unsupported targets are unavailable, not zero savings. Any invalid inverse bootstrap draw withholds ordinary inverse CIs. "
        "Retention support requires .01 px one-sided noninferiority at full and matched budgets and the .50 px absolute ceiling; "
        "this differs from two-sided equivalence. Online costs include all nominal arrivals, replay, FD, LFU, observer and eviction. "
        "Shared preparation and evaluation are disclosed separately. Savings are conditional on the full stream and initialization.")


def _diagnostics(snapshot):
    raw = snapshot.get("diagnostics", _mapping(snapshot.get("analysis")).get("diagnostics"))
    if isinstance(raw, list):
        grouped = {}
        for row in raw:
            item = _mapping(row)
            policy = item.get("policy")
            if policy in POLICIES:
                grouped.setdefault(policy, []).append(item)
        raw = grouped
    result = {}
    for policy, value in _mapping(raw).items():
        if policy not in POLICIES:
            continue
        if isinstance(value, list):
            rows = [v for v in value if isinstance(v, Mapping)]
            # Row exports must already be aggregated; this renderer never pools replicas.
            common = set.intersection(*(set(v) for v in rows)) if rows else set()
            result[policy] = {k: [v[k] for v in rows] for k in common
                              if k != "policy" and all(_finite(v[k]) for v in rows)}
        elif isinstance(value, Mapping):
            result[policy] = value
    return result


def _series(item, keys):
    raw = _first(item, *keys)
    metric = _mapping(raw)
    y = metric.get("mean", metric.get("values")) if metric else raw
    if y is None:
        return None
    x = _first(item, "observations", "n", "groups")
    if (not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y)
            or any(not _finite(v) for v in x + y) or any(b <= a for a, b in zip(x, x[1:]))):
        raise ValueError("invalid diagnostic scalar series or clock")
    return x, y


DIAGNOSTIC_FIELDS = (
    (("pi", "applied_pi", "mean_pi"), 0, "pi"),
    (("gain", "kalman_gain", "mean_gain", "kalman_velocity_gain_mean"), 0, "Kalman velocity gain"),
    (("motion_energy", "motion"), 1, "Motion energy"),
    (("noise_scale", "noise"), 1, "Noise scale"),
    (("lfu_relative_correction", "relative_correction", "correction_norm"), 2, "LFU correction"),
    (("projection_fraction", "lfu_projection_fraction", "clipped_fraction", "negative_fraction"), 3, "Projection fraction"),
)


def _has_diagnostics(snapshot):
    return any(_series(item, keys) is not None for item in _diagnostics(snapshot).values()
               for keys, _, _ in DIAGNOSTIC_FIELDS)


def _figure(snapshot, view):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    result = _mapping(snapshot.get("analysis"))
    figure = Figure(figsize=(12, 7.2 if view == "diagnostics" else 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    if view in ("fresh", "development_paths"):
        axes = figure.subplots(1, 2)
        curves = _mapping((result if view == "fresh" else _mapping(snapshot.get("development"))).get("curves"))
        for axis, domain in zip(axes, ("query", "base")):
            for policy in POLICIES:
                curve = _curve(curves.get(policy), domain)
                if curve is None:
                    continue
                x, y, ci = curve
                root = domain == "base"
                transform = (lambda values: [math.sqrt(v) for v in values]) if root else (lambda values: values)
                axis.plot(x, transform(y), label=LABELS[policy], color=COLORS[policy],
                          linewidth=2.2 if policy in ("R", "B") else 1.6, linestyle="--" if policy == "R" else "-")
                if ci:
                    axis.fill_between(x, *map(transform, ci), color=COLORS[policy], alpha=.06)
            # n=0 is used only when supplied, rather than assuming the first saved state is frozen.
            reference = _curve(curves.get("R"), domain)
            if reference and reference[0][0] == 0:
                frozen = reference[1][0]
                axis.axhline(math.sqrt(frozen) if domain == "base" else frozen, color="black",
                             linestyle=":", linewidth=1.3, label="Frozen initialization")
            clock = "Designated novel grids" if "n" in _mapping(curves.get("R")) else "Total integrated grids"
            title = "New-data prediction" if domain == "query" else "Old-data retention"
            axis.set(title=("Development: " if view == "development_paths" else "") + title, xlabel=clock,
                     ylabel="Mean held-out risk (px^2 / landmark)" if domain == "query" else "Root of mean old MSE (px)")
    elif view == "development":
        axes = figure.subplots(1, 2)
        candidates = _candidates(snapshot)
        for axis, key, title in zip(axes, ("retention_avg", "retention_endpoint"), ("Average retention", "Endpoint retention")):
            baseline = [v for v in candidates if v.get("policy") == "B" and v.get("selected") is True]
            baseline = baseline or [v for v in candidates if v.get("policy") == "B"]
            target = _mean(_summary(baseline[-1], key)) if baseline else None
            if _finite(target):
                axis.axvspan(target - .01, target + .01, color=COLORS["B"], alpha=.1, label="B +/- .01 px guide")
            for policy in POLICIES:
                points = [(v, _mean(_summary(v, key)), _mean(_summary(v, "acquisition")))
                          for v in candidates if v.get("policy") == policy]
                points = [(v, x, y) for v, x, y in points if _finite(x) and _finite(y)]
                if points:
                    axis.scatter([p[1] for p in points], [p[2] for p in points], label=LABELS[policy],
                                 color=COLORS[policy], marker="s" if policy == "R" else "o", s=35)
                    for item, x, y in points:
                        if item.get("selected") is True:
                            axis.scatter([x], [y], facecolors="none", edgecolors="black", s=110, linewidths=1.)
            axis.set(title=title, xlabel="Old RMS deterioration (px)", ylabel="Mean novel risk (px^2 / landmark)")
    elif view == "diagnostics":
        axes = figure.subplots(2, 2).ravel()
        diagnostics = _diagnostics(snapshot)
        for policy in POLICIES:
            item = _mapping(diagnostics.get(policy))
            for keys, index, label in DIAGNOSTIC_FIELDS:
                values = _series(item, keys)
                if values:
                    axes[index].plot(*values, color=COLORS[policy], linestyle="--" if label in ("Kalman velocity gain", "Noise scale") else "-",
                                     label=policy + ": " + label)
        for axis, title, ylabel in zip(axes,
                (r"$\pi$ and Kalman gain", "Motion and noise", "LFU correction", "PSD projection"),
                ("Applied scalar weight", "Supplied energy / noise scale", "Recorded correction diagnostic", "Clipped-coordinate fraction")):
            axis.set(title=title, xlabel="Incoming grids / supplied observer clock", ylabel=ylabel)
    else:
        raise ValueError("unknown Phase 9 figure view")
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.title.set_fontsize(12)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, loc="best")
        else:
            axis.text(.5, .5, "Scalar artifacts pending", ha="center", va="center", transform=axis.transAxes)
    return figure


def display_figures(snapshot, view="fresh"):
    """Display a figure using scalar exports only; no computation of inference."""
    from IPython.display import display

    figure = _figure(_validate_snapshot(snapshot), view)
    try:
        display(figure)
    finally:
        figure.clear()


def _plot_cell(snapshot, view, snapshot_href):
    figure = _figure(snapshot, view)
    try:
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=120, metadata={"Software": "Plan18 Phase9 scalar report v1"})
        png = output.getvalue()
    finally:
        figure.clear()
    source = ("from pathlib import Path\n"
        "from mnist_experiment.combined_memory.replicated_combination.report import load_snapshot, display_figures\n"
        f"_paths = [Path({str(snapshot_href)!r}), Path('snapshot.json')]\n"
        "_artifact = next((p for p in _paths if p.is_file()), _paths[0])\n"
        f"_snapshot = load_snapshot(_artifact, expected_sha256={_hash(_bytes(snapshot))!r})\n"
        f"display_figures(_snapshot, view={view!r})")
    return nbformat.v4.new_code_cell(source, id=PREFIX + view + "-plot", metadata={TAG: True}, execution_count=1,
        outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode("ascii"),
            "text/plain": "Phase 9 scalar " + view + " figure"})])


def _outcome(snapshot, result):
    status = _mapping(snapshot.get("progress")).get("status")
    if status in ("calibration_infeasible", "calibration_budget_limited"):
        reason = ("The bounded development search did not produce a retention-matched tuple."
                  if status == "calibration_infeasible" else
                  "The remaining allowance could not reserve the fixed confirmation cohort.")
        return ("**Calibration closed without fresh confirmation.** " + reason
                + " Development results and every tested recipe remain available above and in immutable artifacts. "
                "No fresh efficacy, interaction or sample-saving conclusion is supported; this is not an "
                "efficacy rejection of diagonal EWC, LFU or adaptive pi.")
    complete = _mapping(snapshot.get("progress")).get("status") == "completed" and result.get("is_final") is True
    if not complete:
        return ("Inference remains provisional until the fixed eight paired replicas and integrity closeout are complete. "
                "Calibration failure or missing results are visible above; no efficacy claim is inferred from progress.")
    contrasts = _mapping(result.get("contrasts"))
    observations = []
    for name, label in (("B-R", "Diagonal representation"), ("C-B", "LFU increment"),
                        ("D-B", "Adaptive pi increment"), ("E-B", "Combined increment")):
        estimate = _contrast(contrasts.get(name), "acquisition")
        ci = _mapping(estimate).get("ci")
        direction = "uncertain"
        if isinstance(ci, list) and len(ci) == 2 and all(_finite(v) for v in ci):
            direction = "lower novel risk" if ci[1] < 0 else "higher novel risk" if ci[0] > 0 else "interval includes zero"
        observations.append(f"{label} ({name}): **{_effect(estimate)}**, {direction}")
    return "; ".join(observations) + (
        ". Read each effect with its retention and compute qualification. A-E calibration does not guarantee fresh equivalence; "
        "B-R and E-R also compare different Fisher representations. Unsupported sample inversions remain exploratory. "
        "These observations do not establish STM-to-LTM information transfer or exact optimality of the pi rule.")


def build_cells(snapshot, analysis_href="analysis.json", snapshot_href=None):
    """Build at most three figure cells; leave caller and historical cells untouched."""
    snapshot = _validate_snapshot(snapshot)
    result = _validate_analysis(snapshot.get("analysis"))
    snapshot_href = snapshot_href or str(Path(analysis_href).with_name("snapshot.json"))
    cells = [_progress(snapshot), cell("intro", "## Phase 9: Combining Methods In The Replicated STM Regime\n\n"
        "Phase 8 reproduced a useful STM policy. Can diagonal LFUs and velocity-Kalman pi improve its "
        "sample efficiency at comparable old-data retention and acceptable compute? "
        "The six-policy study first measures the representation change, then the increments within diagonal EWC. "
        "[Design](plan18.md#phase-9-combining-methods-in-the-replicated-stm-regime)."), _settings(snapshot),
        _fidelity(snapshot), _calibration_cell(snapshot)]
    if _candidates(snapshot):
        cells.append(_plot_cell(snapshot, "development", snapshot_href))
    if not result and _mapping(snapshot.get("development")).get("curves"):
        cells.extend([cell("development-paths", "### Development Trajectories\n\n"
            "Means over the four paired development replicas for the final proposed tuple. "
            "These policies were selected using these data: the curves are descriptive, not fresh inference, "
            "and have no confidence intervals. Fixed held-out panels are evaluated after each accepted update; "
            "the x axis counts novel training grids. Old curves root pooled mean MSE; tabled retention roots "
            "within each replica. Dotted black is frozen initialization."),
            _plot_cell(snapshot, "development_paths", snapshot_href)])
    if result:
        final = result.get("is_final") is True and _mapping(snapshot.get("progress")).get("status") == "completed"
        exposures = _mapping(_mapping(result.get("curves")).get("R")).get("observations", [])
        exposure_note = (f"Mean total incoming grids per trajectory: {_number(exposures[-1], 5)}. "
                         if isinstance(exposures, list) and exposures else "")
        cells.append(cell("fresh", "### Fresh Learning And Retention\n\n"
            + ("**Final fixed cohort.** " if final else "**Provisional paired summaries.** ")
            + f"{len(result.get('replicas', []))}/8 complete six-policy blocks. "
            + exposure_note
            +
            "Each point evaluates fixed official-test panels after its accepted training segment. "
            "The x axis counts training exposure, never holdout observations. "
            "Old curves show the root of pooled mean MSE; tabled retention first roots within each replica. "
            "R and B identify the fixed representation comparison; dotted black is frozen initialization."))
        if result.get("curves"):
            cells.append(_plot_cell(snapshot, "fresh", snapshot_href))
        cells.extend([_effects(result), _efficiency(result)])
    else:
        terminal = _mapping(snapshot.get("progress")).get("status") in (
            "calibration_infeasible", "calibration_budget_limited")
        cells.append(cell("pending", ("### Fresh Confirmation Not Run\n\n"
            "Calibration closed without authorizing the fresh cohort; the planned 48 paths were not run. "
            if terminal else "### Analysis Pending\n\n"
            "Readiness and calibration can be inspected while programming or running. "
            "There is no fresh inference until complete paired blocks are exported. ")
            + "This notebook only reads scalar artifacts and renders plots; it never trains, downloads data, "
            "loads checkpoints, reconstructs Fisher matrices, bootstraps or repairs a run."))
    if _has_diagnostics(snapshot):
        cells.extend([cell("diagnostics", "### Pi, Gain, Motion And LFU Diagnostics\n\n"
            "These recorded scalar trajectories show whether the additions were active. "
            "Kalman gain differs from pi. Noise is the one-grid scale $V$; batch noise is $V/4$. "
            "Mean-velocity energy is shown separately from the filter uncertainty that also enters pi. "
            "LFU shows the mean segment $\\|\\Delta\\|/\\|d\\|$; PSD clipping is the unprojected "
            "negative-coordinate fraction. These are health diagnostics, not proof of predictive benefit."),
            _plot_cell(snapshot, "diagnostics", snapshot_href)])
    cells.extend([cell("artifacts", "[Scalar analysis](<" + str(analysis_href).replace(">", "%3E") + ">) | "
        "[Snapshot, calibration and detailed audits](<" + str(snapshot_href).replace(">", "%3E") + ">). "
        "Per-replica arrays, all ten A-E retention comparisons and unsupported inverse draws remain in artifacts."),
        cell("conclusion", "### Conclusion\n\n" + _outcome(snapshot, result)
             + ("\n\nCloseout: " + _text(snapshot["closeout"]) if isinstance(snapshot.get("closeout"), str) else ""))])
    return cells


def _history(previous):
    history = [c for c in previous["cells"] if not owned(c)]
    position = next((sum(not owned(c) for c in previous["cells"][:i])
                     for i, c in enumerate(previous["cells"])
                     if owned(c) and c.get("id") != PREFIX + "progress"), len(history))
    book = deepcopy(previous)
    book["cells"] = history
    return book, position


def _notebook(history, position, cells):
    book = deepcopy(history)
    section = book["cells"][:position] + cells[1:] + book["cells"][position:]
    book["cells"] = section[:1] + [cells[0]] + section[1:]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))
    data = (json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(data) > MAX_NOTEBOOK_BYTES:
        raise ValueError("bounded notebook size exceeded")
    return data


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("report destination must not be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lease(report):
    if report.is_symlink() or (report / ".publish.lock").is_symlink():
        raise ValueError("report directory/lock must not be a symlink")
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".publish.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another report publisher holds the lease") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _verify_generation(path, digest):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("invalid immutable report generation")
    seal = json.loads(_read(path / "COMPLETED", MAX_JSON_BYTES))
    files = _mapping(seal.get("files"))
    if seal.get("schema") != REPORT_SCHEMA or seal.get("source_digest") != digest or not files:
        raise ValueError("incompatible immutable report seal")
    actual, size = set(), 0
    for file in path.rglob("*"):
        if file.is_symlink():
            raise ValueError("symlink in immutable report generation")
        if file.is_file():
            size += file.stat().st_size
            actual.add(file.relative_to(path).as_posix())
    if size > MAX_GENERATION_BYTES or actual != set(files) | {"COMPLETED"}:
        raise ValueError("immutable report generation size/file set mismatch")
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or _hash(_read(path / name, MAX_GENERATION_BYTES)) != expected:
            raise ValueError("immutable report generation checksum mismatch")
    metadata = json.loads(_read(path / "generation.json", MAX_JSON_BYTES))
    if metadata.get("protocol") != PROTOCOL or metadata.get("source_digest") != digest:
        raise ValueError("incompatible immutable report metadata")
    return metadata


def publish(snapshot, notebook=DEFAULT_NOTEBOOK, store=None):
    """Seal immutable reports before atomic publication under a single lease.

    Earlier cells, outputs, attachments and notebook metadata survive as exact
    JSON values. Same inputs reuse and verify their generation. Failed writes
    can be retried; completed generations are never repaired or overwritten.
    """
    snapshot = _validate_snapshot(snapshot)
    analysis = _validate_analysis(snapshot.get("analysis"))
    snapshot_data, analysis_data = _bytes(snapshot), _bytes(analysis)
    if max(len(snapshot_data), len(analysis_data)) > MAX_JSON_BYTES:
        raise ValueError("bounded scalar report size exceeded")
    notebook = Path(notebook)
    root = Path(store.root) if store is not None else DEFAULT_ROOT
    if root.is_symlink():
        raise ValueError("report root must not be a symlink")
    report = root / "report"
    with _lease(report):
        if notebook.exists() or notebook.is_symlink():
            previous = json.loads(_read(notebook, MAX_NOTEBOOK_BYTES))
        else:
            previous = dict(nbformat.v4.new_notebook())
        if any(_mapping(v).get("role") == "frozen_archive" for v in _mapping(previous.get("metadata")).values()):
            raise ValueError("cannot publish into a frozen historical notebook")
        nbformat.validate(nbformat.from_dict(deepcopy(previous)))
        history, position = _history(previous)
        inputs = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, snapshot_sha256=_hash(snapshot_data),
            history_sha256=_hash(_bytes(history)), position=position, source_sha256=_hash(_SOURCE_BYTES),
            notebook_parent=str(notebook.absolute().parent))
        digest = _hash(_bytes(inputs))
        generations = report / "generations"
        if generations.is_symlink() or (report / "latest.json").is_symlink():
            raise ValueError("report generations/pointer must not be a symlink")
        generations.mkdir(exist_ok=True)
        destination = generations / digest[:24]
        changed = not (destination.exists() or destination.is_symlink())
        if not changed:
            metadata = _verify_generation(destination, digest)
            data = _read(destination / "results.ipynb", MAX_NOTEBOOK_BYTES)
        else:
            cells = build_cells(snapshot, os.path.relpath(destination / "analysis.json", notebook.parent),
                                os.path.relpath(destination / "snapshot.json", notebook.parent))
            data = _notebook(history, position, cells)
            metadata = dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, generation=destination.name,
                            source_digest=digest, status=_mapping(snapshot.get("progress")).get("status", "pending"))
            files = {"snapshot.json": snapshot_data, "analysis.json": analysis_data, "results.ipynb": data,
                     "inputs.json": _bytes(inputs), "generation.json": _bytes(metadata), "report_source.py": _SOURCE_BYTES}
            for item in cells:
                for output in item.get("outputs", []):
                    encoded = output.get("data", {}).get("image/png")
                    if encoded:
                        files["figures/" + item["id"].removeprefix(PREFIX) + ".png"] = base64.b64decode(encoded)
            files["COMPLETED"] = _bytes(dict(schema=REPORT_SCHEMA, source_digest=digest,
                files={name: _hash(content) for name, content in files.items()}))
            size = sum(map(len, files.values()))
            if size > MAX_GENERATION_BYTES:
                raise ValueError("bounded report generation size exceeded")
            reserve = store.reserve(size) if store is not None and hasattr(store, "reserve") else nullcontext()
            with reserve:
                temporary = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=generations))
                try:
                    for name, content in files.items():
                        _atomic(temporary / name, content)
                    os.rename(temporary, destination)
                    _sync(generations)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
        pointer = _bytes(dict(schema=REPORT_SCHEMA, protocol=PROTOCOL, generation=destination.name,
            path=f"generations/{destination.name}", source_digest=digest))
        updates = [(notebook, data), (report / "latest.json", pointer)]
        updates = [(path, content) for path, content in updates
                   if not path.exists() or _read(path, MAX_NOTEBOOK_BYTES) != content]
        reserve = store.reserve(sum(len(content) for _, content in updates)) if updates and store is not None and hasattr(store, "reserve") else nullcontext()
        with reserve:
            for path, content in updates:
                _atomic(path, content)
        return {**metadata, "changed": changed, "path": str(destination)}
