"""Read-only Phase 7 presentation, without training or statistical inference.

Trajectory payloads use DripRunner schema 1. Decision/audit results also declare
schema_version=1. Confirmation ledgers freeze replicas, event_target, and full
treatments; explicit trajectory_ids/jobs manifests are also accepted. Only a
sealed results decision with complete/inference_ready true and every ledger
trajectory sealed can supply intervals or tests. The full analysis stays in its
linked JSON; only six core comparisons are presented here.

Relative status.development_root paths are relative to the current Store.
Policy identities remain separate, including superseded numerical audits.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import nbformat

from .artifacts import ArtifactError, iter_units
from .config import Treatment
from .report import MAX_JSON_BYTES, MAX_NOTEBOOK_BYTES, _plot_cell, _table


TAG = "plan15_phase7"
PROGRESS_TAG = "plan15_phase7_progress"
DEVELOPMENT = "regularization_development"
CONFIRMATION = "regularization_confirmation"
DOMAINS = ("rotated", "base")
DOMAIN_LABELS = {"rotated": "Rotated", "base": "Unrotated"}
STREAM_LABELS = {"mixed": "Mixed training", "base": "Unrotated-only control"}
CONDITIONS = ("replay_ewc", "finite_difference")
COLORS = {"replay_ewc": "#227c76", "finite_difference": "#b85747"}


def _read(path: Path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number

    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ValueError("JSON exceeds report read bound")
        return json.loads(content, parse_constant=reject, parse_float=finite_float)
    except (OSError, ValueError, UnicodeError) as error:
        raise ArtifactError(f"cannot read Phase 7 artifact {path}: {error}") from error


def _number(value, label, *, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ArtifactError(f"invalid finite Phase 7 number: {label}")
    return value


def _schema(value, label):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ArtifactError(f"incompatible Phase 7 {label} schema")


def _same(actual, expected, label):
    if not math.isclose(_number(actual, label), expected, rel_tol=1e-7, abs_tol=1e-8):
        raise ArtifactError(f"Phase 7 {label} disagrees with reconstructed metrics")


def _recipe(meta):
    treatment = meta.get("treatment")
    if not isinstance(treatment, dict) or treatment.get("condition") not in CONDITIONS:
        raise ArtifactError("invalid Phase 7 treatment")
    condition = treatment["condition"]
    if meta.get("condition", condition) != condition:
        raise ArtifactError("Phase 7 treatment/condition mismatch")
    ridge = _number(treatment.get("ridge_multiplier"), "ridge multiplier", minimum=0)
    strength = _number(treatment.get("strength"), "FD strength", minimum=0)
    if (condition == "replay_ewc" and strength != 0) or (condition == "finite_difference" and strength <= 0):
        raise ArtifactError("Phase 7 strength/condition mismatch")
    return condition, ridge, strength


def _policy(meta):
    return str(meta.get("policy_id", meta.get("policy_iterations",
               meta.get("treatment", {}).get("inner_steps", meta.get("protocol", "unspecified")))))


def _identity(meta):
    if type(meta.get("replica")) is not int or meta["replica"] < 0 or meta.get("stream") not in ("mixed", "base"):
        raise ArtifactError("invalid Phase 7 replica/stream")
    return meta["replica"], meta["stream"], _recipe(meta), _policy(meta)


def _trajectory(record):
    meta, path, name = record["metadata"], record["path"], record["unit_id"]
    _schema(meta, "trajectory metadata")
    _identity(meta)
    if meta.get("phase") != "7" or meta.get("family") not in (DEVELOPMENT, CONFIRMATION):
        raise ArtifactError(f"incompatible Phase 7 trajectory: {name}")
    unit = dict(record, rows=[], fit=[], summary=None)
    if record["status"] != "completed":
        return unit
    if _read(path / "metadata.json") != meta:
        raise ArtifactError(f"Phase 7 metadata disagrees with manifest: {name}")
    rows, fit, summary = (_read(path / filename) for filename in ("metrics.json", "fit.json", "summary.json"))
    _schema(summary, "summary")
    target = meta.get("event_target")
    if type(target) is not int or target < 1:
        raise ArtifactError(f"invalid Phase 7 event target: {name}")
    for key, expected in dict(unit_id=name, replica=meta["replica"], stream=meta["stream"],
                              condition=_recipe(meta)[0], strength=_recipe(meta)[2], event_target=target).items():
        if summary.get(key) != expected:
            raise ArtifactError(f"Phase 7 summary {key} mismatch: {name}")
    if (not isinstance(rows, list) or len(rows) != target + 1
            or any(not isinstance(row, dict) or type(row.get("paired_rotated_event")) is not int for row in rows)
            or [row.get("paired_rotated_event") for row in rows] != list(range(target + 1))):
        raise ArtifactError(f"incomplete or unordered Phase 7 event grid: {name}")
    for row in rows:
        for domain in DOMAINS:
            _number(row.get(f"{domain}_nll"), f"{name}/{domain} NLL", minimum=0)
            _number(row.get(f"{domain}_accuracy"), f"{name}/{domain} accuracy", minimum=0, maximum=1)
        actual = row.get("actual_rotated_events")
        if type(actual) is not int or actual != (row["paired_rotated_event"] if meta["stream"] == "mixed" else 0):
            raise ArtifactError(f"Phase 7 actual exposure mismatch: {name}")
    for domain in DOMAINS:
        _same(summary.get(f"{domain}_auc"), statistics.fmean(row[f"{domain}_nll"] for row in rows[1:]),
              f"{name}/{domain} AUC")
        for metric in ("nll", "accuracy"):
            _same(summary.get(f"final_{domain}_{metric}"), rows[-1][f"{domain}_{metric}"],
                  f"{name}/final {domain} {metric}")
    _number(summary.get("active_seconds"), f"{name}/online seconds", minimum=0)
    if not isinstance(fit, list) or any(not isinstance(row, dict) for row in fit):
        raise ArtifactError(f"invalid Phase 7 fit records: {name}")
    unit.update(rows=rows, fit=fit, summary=summary)
    return unit


def _audit(record):
    meta = record["metadata"]
    _identity(meta)
    if (not isinstance(meta.get("parent_id"), str) or type(meta.get("event")) is not int
            or meta["event"] < 1 or type(meta.get("policy_iterations")) is not int):
        raise ArtifactError("invalid Phase 7 audit identity")
    value = _read(record["path"] / "result.json")
    _schema(value, "audit")
    ordinary = _number(value.get("ordinary_objective"), "ordinary objective")
    extended = _number(value.get("extended_objective"), "extended objective")
    _same(value.get("objective_gain"), ordinary - extended, "audit objective gain")
    _same(value.get("relative_objective_gain"), (ordinary - extended) / max(1., abs(ordinary)),
          "audit relative objective gain")
    _number(value.get("seconds"), "audit seconds", minimum=0)
    for field in ("original_gradient_inf", "extended_gradient_inf", "ltm_motion", "stm_motion"):
        _number(value.get(field), field, minimum=0)
    for field in ("parent_id", "parent_state_hash", "step"):
        if field in value and field in meta and value[field] != meta[field]:
            raise ArtifactError(f"Phase 7 audit {field} mismatch")
    for state in ("ordinary", "extended"):
        scores = value.get(state)
        if not isinstance(scores, dict):
            raise ArtifactError("missing Phase 7 audit holdouts")
        for domain in DOMAINS:
            metric = scores.get(domain)
            if not isinstance(metric, dict):
                raise ArtifactError("missing Phase 7 audit domain")
            _number(metric.get("nll"), "audit holdout NLL", minimum=0)
            _number(metric.get("accuracy"), "audit holdout accuracy", minimum=0, maximum=1)
    tv = value.get("prediction_tv")
    if not isinstance(tv, dict):
        raise ArtifactError("missing Phase 7 prediction TV")
    for domain in DOMAINS:
        _number(tv.get(domain), "prediction TV", minimum=0, maximum=1)
    return dict(record, result=value)


def _load(root):
    trajectories, audits, decisions, pending = [], [], {}, []
    for record in iter_units(root, completed_only=False):
        meta, kind = record["metadata"], record["metadata"].get("kind")
        if kind not in ("drip_trajectory", "regularization_audit", "regularization_decision"):
            continue
        if kind == "drip_trajectory" and meta.get("phase") != "7":
            continue
        if record["status"] != "completed":
            pending.append(dict(unit_id=record["unit_id"], kind=kind))
        if kind == "drip_trajectory":
            trajectories.append(_trajectory(record))
        elif record["status"] == "completed":
            if kind == "regularization_audit":
                audits.append(_audit(record))
            else:
                decision = meta.get("decision")
                if (not isinstance(decision, str) or record["unit_id"] != f"regularization-{decision}-v1"
                        or decision in decisions):
                    raise ArtifactError("invalid or duplicate Phase 7 decision identity")
                value = _read(record["path"] / "result.json")
                _schema(value, f"{decision} decision")
                decisions[decision] = dict(record, result=value)
    parents = {unit["unit_id"]: unit for unit in trajectories}
    seen = set()
    for branch in audits:
        meta = branch["metadata"]
        parent = parents.get(meta["parent_id"])
        if (parent is None or meta["event"] > parent["metadata"].get("event_target", 0)
                or any(meta[field] != parent["metadata"][field] for field in ("replica", "stream", "treatment"))):
            raise ArtifactError("Phase 7 audit does not match its parent trajectory")
        key = meta["parent_id"], meta["event"]
        if key in seen:
            raise ArtifactError("duplicate Phase 7 audit snapshot")
        seen.add(key)
    return dict(root=Path(root), trajectories=trajectories, audits=audits, decisions=decisions, pending=pending)


def _validate_pairing(trajectories):
    seen, initial, assets = set(), {}, {}
    for unit in trajectories:
        meta = unit["metadata"]
        key = (meta["family"], *_identity(meta))
        if key in seen:
            raise ArtifactError("duplicate Phase 7 trajectory recipe/replica/policy")
        seen.add(key)
        if unit["status"] != "completed":
            continue
        key = meta["family"], meta["replica"]
        first = unit["rows"][0]
        if key in initial:
            for domain in DOMAINS:
                for metric in ("nll", "accuracy"):
                    field = f"{domain}_{metric}"
                    _same(first[field], initial[key][field], f"paired initial {field}")
        else:
            initial[key] = first
        for field in ("asset_id", "plan_hash"):
            if field in meta:
                asset_key = (*key, field)
                if asset_key in assets and assets[asset_key] != meta[field]:
                    raise ArtifactError(f"Phase 7 unpaired {field}")
                assets[asset_key] = meta[field]


def collect(root: Path) -> dict:
    """Validate scalar artifacts in the current Store and its development source."""
    root = Path(root).resolve()
    status = _read(root / "status.json")
    if not isinstance(status, dict) or status.get("stage") not in ("7A", "7B", "7C", "7D", "diagnostic"):
        raise ArtifactError("invalid Phase 7 status/stage")
    _number(status.get("active_seconds"), "active seconds", minimum=0)
    if "aggregate_active_seconds" in status:
        _number(status["aggregate_active_seconds"], "aggregate active seconds", minimum=0)
    for field in ("completed", "expected"):
        if status["stage"] == "diagnostic" and status.get(field) is None:
            continue
        if type(status.get(field)) is not int or status[field] < 0:
            raise ArtifactError(f"invalid Phase 7 progress {field}")
    data = _load(root)
    data.update(status=status, development=None)
    source = status.get("development_root")
    confirmation = data["decisions"].get("confirmation-ledger")
    if not source and confirmation:
        lineage = confirmation["result"].get("source", {})
        if not isinstance(lineage, dict):
            raise ArtifactError("invalid Phase 7 confirmation source lineage")
        source = lineage.get("root")
    if source:
        if not isinstance(source, str):
            raise ArtifactError("invalid Phase 7 development root")
        path = Path(source)
        path = (path if path.is_absolute() else root / path).resolve()
        if path == root:
            raise ArtifactError("confirmation development root points to itself")
        data["development"] = _load(path)
        if any(unit["metadata"]["family"] != DEVELOPMENT for unit in data["development"]["trajectories"]):
            raise ArtifactError("confirmation trajectories in Phase 7 development source")
    _validate_pairing(data["trajectories"] + (data["development"]["trajectories"] if data["development"] else []))
    if confirmation:
        expected = _confirmation_plan(confirmation["result"])
        for unit in data["trajectories"]:
            if unit["metadata"]["family"] != CONFIRMATION:
                continue
            name = unit["unit_id"]
            if (name not in expected or (expected[name] is not None and
                    any(unit["metadata"].get(key) != value for key, value in expected[name].items()))):
                raise ArtifactError("Phase 7 confirmation trajectory differs from frozen ledger")
    return data


def _confirmation_plan(plan):
    """Reconstruct only prospective identities, never an analysis or a fit."""
    if "treatments" not in plan:
        names = plan.get("trajectory_ids", plan.get("jobs"))
        if isinstance(names, list):
            names = [item.get("unit_id") if isinstance(item, dict) else item for item in names]
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names))):
            raise ArtifactError("Phase 7 confirmation ledger requires unique trajectory_ids/jobs")
        return dict.fromkeys(names)
    replicas, target, mappings = plan.get("replicas"), plan.get("event_target"), plan["treatments"]
    if (type(replicas) is not int or replicas < 1 or type(target) is not int or target < 1
            or not isinstance(mappings, list) or len(mappings) not in (2, 3)):
        raise ArtifactError("invalid Phase 7 confirmation recipe ledger")
    try:
        choices = [Treatment(**value) for value in mappings]
    except (TypeError, ValueError) as error:
        raise ArtifactError("invalid Phase 7 confirmation recipe") from error
    if len({choice.key for choice in choices}) != len(choices):
        raise ArtifactError("duplicate Phase 7 confirmation recipe")
    for choice in choices:
        _recipe(dict(treatment=choice.mapping()))
    return {f"trajectory-{CONFIRMATION}-{replica:04d}-{stream}-{choice.key}":
            dict(replica=replica, stream=stream, treatment=choice.mapping(), event_target=target)
            for replica in range(replicas) for stream in ("mixed", "base") for choice in choices}


def _final_results(data):
    record = data["decisions"].get("results")
    if record is None:
        return None
    result = record["result"]
    if result.get("complete") is not True or result.get("inference_ready") is not True:
        return None
    ledger = data["decisions"].get("confirmation-ledger")
    if ledger is None:
        raise ArtifactError("final Phase 7 inference requires a sealed confirmation ledger")
    plan = ledger["result"]
    expected = _confirmation_plan(plan)
    confirmation = {unit["unit_id"]: unit for unit in data["trajectories"]
                    if unit["metadata"]["family"] == CONFIRMATION}
    if (data["status"]["stage"] != "7D" or set(confirmation) != set(expected)
            or any(unit["status"] != "completed" for unit in confirmation.values())):
        raise ArtifactError("cannot publish Phase 7 final inference from an incomplete confirmation cohort")
    for name, wanted in expected.items():
        if wanted is not None and any(confirmation[name]["metadata"].get(key) != value for key, value in wanted.items()):
            raise ArtifactError("Phase 7 confirmation trajectory differs from frozen ledger")
    for field in ("completed", "expected"):
        if field in result and result[field] != len(expected):
            raise ArtifactError("Phase 7 sealed analysis cohort count disagrees with ledger")
    if plan.get("smoke") is True or result.get("smoke") is True:
        raise ArtifactError("Phase 7 smoke results cannot release inference")
    return result


def _groups(units, family, stream):
    groups = defaultdict(list)
    for unit in units:
        meta = unit["metadata"]
        if unit["status"] == "completed" and meta["family"] == family and meta["stream"] == stream:
            groups[(*_recipe(meta), _policy(meta))].append(unit)
    return sorted(groups.items())


def _label(key, *, include_policy=True):
    condition, ridge, strength, policy = key
    method = "FD" if condition == "finite_difference" else "Replay/EWC"
    return (f"{method} c={ridge:g}" + (f", strength={strength:g}" if strength else "")
            + (f"; policy {policy}" if include_policy else ""))


def _mean(values):
    return statistics.fmean(values)


def _delta(unit, domain):
    return unit["summary"][f"{domain}_auc"] - unit["rows"][0][f"{domain}_nll"]


def _empty(text):
    return nbformat.v4.new_markdown_cell(text)


def _audit_plot(audits):
    if not audits:
        return _empty("### Numerical audit\n\nNo sealed audit branches yet; incomplete branches are progress only.")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for axis, domain in zip(axes, DOMAINS):
        groups = defaultdict(list)
        for audit in audits:
            meta = audit["metadata"]
            groups[(_recipe(meta)[0], meta["stream"], meta["policy_iterations"])].append(audit)
        for (condition, stream, policy), values in sorted(groups.items()):
            method = "FD" if condition == "finite_difference" else "Replay/EWC"
            axis.scatter([v["result"]["relative_objective_gain"] for v in values],
                         [v["result"]["extended"][domain]["nll"] - v["result"]["ordinary"][domain]["nll"]
                          for v in values], marker="o" if stream == "mixed" else "^",
                         facecolors=COLORS[condition] if policy == min(a["metadata"]["policy_iterations"] for a in audits) else "none",
                         edgecolors=COLORS[condition],
                         label=f"{method}; {STREAM_LABELS[stream].lower()}; {policy} iterations")
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.axvline(.001, color="#777777", linestyle=":", linewidth=.8)
        axis.set(title=f"{DOMAIN_LABELS[domain]} development holdout", xlabel="Extra objective reduction / max(1, |J|)",
                 ylabel="Extended minus ordinary held-out NLL")
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=6, frameon=False)
    return _plot_cell(figure, "All sealed prespecified branches. Hollow markers identify the common-ceiling repair; "
                      "unrotated-only control triangles use paired-event snapshots. The live learner commits only ordinary updates.")


def _development_plot(units, *, policy_iterations=None, lambda0=160.89, shortlisted=None):
    def iterations(unit):
        meta = unit["metadata"]
        return meta.get("policy_iterations", meta["treatment"].get("inner_steps"))

    if policy_iterations is None:
        policies = [iterations(unit) for unit in units if unit["metadata"]["family"] == DEVELOPMENT
                    and unit["status"] == "completed" and iterations(unit) is not None]
        policy_iterations = max(policies) if policies else None
    units = [unit for unit in units if iterations(unit) == policy_iterations]
    if not any(_groups(units, DEVELOPMENT, stream) for stream in ("mixed", "base")):
        return _empty("### Development response\n\nNo sealed development trajectories yet.")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained")
    ridges = sorted({_recipe(unit["metadata"])[1] for unit in units if unit["metadata"]["family"] == DEVELOPMENT})
    markers = {ridge: ("o", "s", "^", "D")[i % 4] for i, ridge in enumerate(ridges)}
    for row, stream in enumerate(("mixed", "base")):
        axis, retune = axes[row]
        fd = defaultdict(list)
        timings = defaultdict(list)
        for key, group in _groups(units, DEVELOPMENT, stream):
            x, y = (_mean(_delta(unit, domain) for unit in group) for domain in ("base", "rotated"))
            if key[0] == "replay_ewc" or math.isclose(key[2], lambda0, rel_tol=1e-12):
                axis.scatter([x], [y], color=COLORS[key[0]], marker=markers[key[1]],
                             s=60 if key[0] == "finite_difference" else 30)
                timings[key[0]].extend(unit["summary"]["active_seconds"] / 60 for unit in group)
            if key[0] == "finite_difference" and (shortlisted is None or key[1] in shortlisted):
                fd[key[1], key[3]].append((key[2], y))
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.axvline(0, color="#777777", linewidth=.7)
        axis.scatter([0], [0], marker="+", color="black", label="Frozen initial model")
        for condition, times in sorted(timings.items()):
            method = "FD" if condition == "finite_difference" else "Replay/EWC"
            axis.plot([], [], color=COLORS[condition], linewidth=2,
                      label=f"{method}; grid mean {_mean(times):.1f} min")
        for ridge in ridges:
            axis.plot([], [], marker=markers[ridge], linestyle="none", color="#555555", label=f"c={ridge:g}")
        clock = ("Mixed training: actual novel count" if stream == "mixed" else
                 "Unrotated-only control: paired-event clock\n(zero novel exposures)")
        axis.set(title=clock, xlabel="Unrotated retention NLL\nminus own initial NLL",
                 ylabel="Rotated acquisition NLL\nminus own initial NLL")
        axis.legend(fontsize=7, frameon=False, ncol=2)
        for (ridge, _), points in sorted(fd.items()):
            points.sort()
            retune.plot([x for x, _ in points], [y for _, y in points], marker="o",
                        label=f"c={ridge:g}")
        if fd:
            retune.set_xscale("log")
            retune.legend(fontsize=7, frameon=False)
        retune.axhline(0, color="#777777", linewidth=.7)
        retune.set(title=f"FD coefficient response\n{STREAM_LABELS[stream]}",
                   xlabel="FD strength", ylabel="Rotated acquisition NLL\nminus own initial NLL")
        for item in (axis, retune):
            item.grid(alpha=.15)
    return _plot_cell(figure, f"Development policy: {policy_iterations or 'not recorded'} iterations; baseline FD strength {lambda0:g}. "
                      "The ridge grid shows replay/EWC and baseline-strength FD only; retuning is separate at right. "
                      "Superseded policies remain in the numerical audit only. Subtract each replica's own initial "
                      "held-out score before averaging; lower/left is better. Sealed-cohort means are descriptive; "
                      "replica counts and online time remain in the compact diagnostics.")


def _confirmation_plot(units, result):
    groups = {stream: _groups(units, CONFIRMATION, stream) for stream in ("mixed", "base")}
    if not any(groups.values()):
        return _empty("### Confirmation curves\n\nNo sealed confirmation trajectories yet. "
                      "No partial checkpoint is treated as a completed trajectory.")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(3, 4, figsize=(16, 11), layout="constrained")
    for row, stream in enumerate(("mixed", "base")):
        replay_index = 0
        for group_index, (key, group) in enumerate(groups[stream]):
            targets = {len(unit["rows"]) for unit in group}
            if len(targets) != 1:
                raise ArtifactError("Phase 7 confirmation arms have incompatible event targets")
            x = [item["paired_rotated_event"] for item in group[0]["rows"]]
            label = _label(key, include_policy=False) + f"; n={len(group)}"
            frozen_label = "Frozen initial model" if group_index == 0 else "_nolegend_"
            linestyle = "--" if key[0] == "replay_ewc" and replay_index else "-"
            if key[0] == "replay_ewc":
                replay_index += 1
            for col, field in enumerate(("rotated_nll", "base_nll", "rotated_accuracy", "base_accuracy")):
                curve = [_mean(unit["rows"][i][field] for unit in group) for i in range(len(x))]
                axes[row, col].plot(x, curve, label=label, color=COLORS[key[0]], linestyle=linestyle)
                axes[row, col].axhline(curve[0], color="#888888", linestyle=":", linewidth=.9,
                                      label=frozen_label)
            running, total = [], 0.
            for i in range(1, len(x)):
                total += _mean(v["rows"][i]["rotated_nll"] for v in group)
                running.append(total / i)
            axes[2, row].plot(x[1:], running, label=label, color=COLORS[key[0]], linestyle=linestyle)
            axes[2, row].axhline(_mean(v["rows"][0]["rotated_nll"] for v in group),
                                color="#888888", linestyle=":", linewidth=.9, label=frozen_label)
        for col, title in enumerate(("Rotated test NLL", "Unrotated test NLL", "Rotated accuracy", "Unrotated accuracy")):
            axes[row, col].set(title=f"{STREAM_LABELS[stream]}\n{title}",
                               ylabel="Test NLL (nats)" if col < 2 else "Test accuracy (%)",
                               xlabel="Actual novel observations" if stream == "mixed" else "Paired-event pseudo-clock\n(zero novel exposures)")
            if col >= 2:
                axes[row, col].yaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=1))
        if groups[stream]:
            axes[row, 0].legend(fontsize=7, frameon=False)
            axes[2, row].legend(fontsize=6, frameon=False)
        axes[2, row].set(title=f"Running A(n): {'mixed training' if stream == 'mixed' else 'unrotated-only control'}",
                         ylabel="Mean rotated-test NLL (nats)\nthrough event n",
                         xlabel="Actual novel observations" if stream == "mixed" else "Paired-event pseudo-clock")
    gain_axis, time_axis = axes[2, 2:]
    primary = result.get("primary", {}) if result else {}
    pairs = primary.get("paired_values", []) if isinstance(primary, dict) else []
    if pairs and all(isinstance(item, dict) and isinstance(item.get("gain"), (int, float)) for item in pairs):
        gain_axis.scatter([item["gain"] for item in pairs], range(len(pairs)), color="#227c76")
    ci = primary.get("ci") if isinstance(primary, dict) else None
    mean = primary.get("mean_gain") if isinstance(primary, dict) else None
    if isinstance(ci, list) and len(ci) == 2 and mean is not None:
        gain_axis.plot(ci, [-1, -1], color="black", linewidth=2)
        gain_axis.scatter([mean], [-1], marker="D", color="black")
    if not result:
        gain_axis.text(.5, .5, "Inference withheld\nAwait sealed complete-cohort results", ha="center", va="center",
                       transform=gain_axis.transAxes)
    gain_axis.axvline(0, color="#777777", linewidth=.7)
    gain_axis.set(title="Paired gains (sealed 95% CI)", ylabel="Replica (diamond: mean)",
                  xlabel="Calibrated replay minus FD\nacquisition NLL (nats)")
    times = [(f"{STREAM_LABELS[stream]}\n{'FD' if key[0] == 'finite_difference' else 'Replay/EWC'} c={key[1]:g}; n={len(group)}",
              _mean(unit["summary"]["active_seconds"] for unit in group) / 60)
             for stream in ("mixed", "base") for key, group in groups[stream]]
    time_axis.barh(range(len(times)), [value for _, value in times], color="#777777")
    time_axis.set_yticks(range(len(times)), [label for label, _ in times], fontsize=6)
    time_axis.set(title="Online time (audits excluded)", xlabel="Minutes per trajectory",
                  ylabel="Training stream and method")
    for axis in axes.flat:
        axis.grid(alpha=.15)
        axis.title.set_fontsize(10)
        axis.xaxis.label.set_fontsize(9)
        axis.yaxis.label.set_fontsize(9)
    policies = sorted({_policy(unit["metadata"]) for unit in units if unit["metadata"]["family"] == CONFIRMATION})
    return _plot_cell(figure, f"Common confirmation policy: {', '.join(policies)} iterations. "
                      "Sealed trajectories only; partial-cohort curves are descriptive without uncertainty bands. "
                      "Each arm uses its own sealed cohort (n in legend); counts may differ and cohorts need not be paired. "
                      "Gray dotted lines show the frozen initial model on the same holdout panel, "
                      "averaged over each arm's sealed cohort (these means can differ until the cohort is complete). "
                      "Initial evaluation is excluded from running A(n); shared post-batch predictions count once per novel event. "
                      "Unrotated-only controls have zero novel exposures. NLL and running A(n) are in nats; "
                      "accuracy is a percentage. Panel limits are set independently, not shared across plots. "
                      "Equal data is not equal compute.")


def _link(record, filename="result.json"):
    path = (record["path"] / filename).resolve().as_posix()
    return f"[{record['unit_id']}]({path})"


def _decisions(data):
    rows = []
    sources = [data["development"], data] if data["development"] else [data]
    for source in sources:
        for name, record in sorted(source["decisions"].items()):
            if name not in ("policy", "shortlist", "selection") and not name.startswith("extension-"):
                continue
            value = record["result"]
            if name == "policy":
                policy = value.get("policy", {})
                summary = f"{policy.get('inner_steps', '?')} iterations / {policy.get('max_evaluations', '?')} closures"
                if value.get("repaired"):
                    summary += "; common-ceiling repair"
                if value.get("optimization_limited"):
                    summary += "; optimization-limited"
            elif name.startswith("extension-"):
                summary = f"Ridge {value.get('ridge', '?')}: " + (
                    f"FD strength {value['strength']:g}" if value.get("strength") is not None else "no extension")
            else:
                replay, fd = value.get("replay", {}), value.get("fd", {})
                summary = f"Replay ridge {replay.get('ridge_multiplier', '?')}; FD ridge {fd.get('ridge_multiplier', '?')}"
                if name == "selection":
                    summary += f", strength {fd.get('strength', '?')}"
                if value.get("status"):
                    summary += f"; {value['status'].replace('_', ' ')}"
                if value.get("warnings"):
                    summary += "; retention warning (see artifact)"
            rows.append([name, summary, _link(record)])
    development = data["development"] or data
    decisions = development["decisions"]
    grid = decisions.get("ledger", {}).get("result", {}).get("ridge_grid", [])
    choice = decisions.get("selection", decisions.get("shortlist", {})).get("result", {})
    edges = []
    if grid:
        for key, label in (("replay", "Replay"), ("fd", "FD")):
            ridge = choice.get(key, {}).get("ridge_multiplier")
            if ridge in (min(grid), max(grid)):
                side = "upper" if ridge == max(grid) else "lower"
                edges.append(f"{label}: $c={ridge:g}$ ({side} edge)")
    boundary = ("\n\n**Ridge boundary:** " + "; ".join(edges)
                + ". These are best tested settings; the ridge optimum is not bracketed beyond that edge."
                if edges else "")
    return _empty("### Frozen decisions\n\n" + (_table(["Decision", "State", "Artifact"], rows) if rows else
                  "No sealed policy/shortlist/selection decisions yet.") +
                  boundary +
                  "\n\nDevelopment candidates and their full metrics remain in the linked immutable units below; "
                  "development outcomes never become confirmation replicas.")


def _effect_row(label, effect, *, lower_better=False):
    mean, ci = effect.get("mean_gain"), effect.get("ci")
    estimate = f"{mean:+.4f}" if isinstance(mean, (int, float)) else "not recorded"
    interval = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if isinstance(ci, list) and len(ci) == 2 else "not recorded"
    favorable = effect.get("favorable_replicas")
    if lower_better:
        # The analysis's generic favorable_replicas counts positive values, including harms.
        pairs = effect.get("paired_values")
        favorable = sum(item["gain"] < 0 for item in pairs) if isinstance(pairs, list) else None
    count = f"{favorable}/{effect.get('n_pairs', '?')}" if favorable is not None else "not recorded"
    return [label, estimate, interval, count]


def _inference(data, result):
    if result is None:
        return _empty("### Confirmation inference\n\n**Descriptive progress only.** Intervals, tests, "
                      "and confirmatory conclusions are withheld until complete-cohort analysis is sealed. "
                      "No bootstrap, significance test, or model fitting runs during reporting.")
    record = data["decisions"]["results"]
    plan = data["decisions"]["confirmation-ledger"]["result"]
    rows = []
    if isinstance(result.get("primary"), dict):
        rows.append(_effect_row("FD gain over calibrated replay (primary)", result["primary"]))
    if result.get("matched_ridge_is_primary") is False and isinstance(result.get("matched_ridge"), dict):
        rows.append(_effect_row("FD gain at the same ridge (secondary)", result["matched_ridge"]))
    if isinstance(result.get("retention_harm"), dict):
        rows.append(_effect_row("Mixed-stream unrotated retention harm: FD minus replay", result["retention_harm"], lower_better=True))
    recipes = plan.get("treatments", [])
    deterioration = None
    fd_frozen = None
    for name, condition, field in (("FD", "finite_difference", "treatment"), ("Replay", "replay_ewc", "baseline")):
        mapping = plan.get(field) or next((item for item in recipes if item.get("condition") == condition), None)
        if mapping is None:
            continue
        key = Treatment(**mapping).key
        frozen = result.get("frozen_gain", {}).get(key, {}).get("mixed")
        if isinstance(frozen, dict):
            rows.append(_effect_row(f"{name} acquisition gain over freezing (mixed)", frozen))
        if name == "FD":
            fd_frozen = frozen
            deterioration = result.get("base_deterioration", {}).get(key, {}).get("mixed")
    if isinstance(deterioration, dict):
        rows.append(_effect_row("FD unrotated deterioration from initialization (mixed)", deterioration, lower_better=True))
    p_value = result.get("primary", {}).get("p_value")
    test = f"Primary one-sided sign-flip p = {p_value:.6g}; " if isinstance(p_value, (int, float)) else ""
    reading = []
    primary_ci = result.get("primary", {}).get("ci")
    if isinstance(primary_ci, list) and len(primary_ci) == 2:
        if primary_ci[0] > 0:
            reading.append("The primary interval favors FD over calibrated replay in this tested regime.")
        elif primary_ci[1] < 0:
            reading.append("The primary interval favors calibrated replay over FD in this tested regime.")
        else:
            reading.append("The primary interval does not resolve an acquisition advantage for either recipe.")
    frozen_ci = fd_frozen.get("ci") if isinstance(fd_frozen, dict) else None
    if isinstance(frozen_ci, list) and len(frozen_ci) == 2:
        if frozen_ci[0] > 0:
            reading.append("The nominal frozen-reference interval also favors FD over no adaptation.")
        elif frozen_ci[1] < 0:
            reading.append("FD still acquires worse predictions on average than no adaptation by this KPI.")
        else:
            reading.append("FD's acquisition improvement over no adaptation remains uncertain.")
    deterioration_ci = deterioration.get("ci") if isinstance(deterioration, dict) else None
    if isinstance(deterioration_ci, list) and len(deterioration_ci) == 2 and deterioration_ci[0] > 0:
        reading.append("FD's unrotated-domain NLL still worsens from initialization: this is a trade-off, not forgetting-free adaptation.")
    conclusion = "\n\n**Reading:** " + " ".join(reading) if reading else ""
    return _empty("### Confirmation inference\n\nBetter than calibrated replay is distinct from better than learning nothing. "
                  "Positive gains and negative unrotated retention harms are favorable. "
                  "Deterioration compares trajectory-average unrotated-test NLL with its initial value on the "
                  "same holdout panel; positive means worse. It is not an endpoint-only difference.\n\n" +
                  _table(["Comparison", "Mean", "95% interval", "Favorable replicas"], rows[:6]) +
                  conclusion + f"\n\n{test}sign-flip inference assumes exchangeable signs; secondary intervals are nominal, "
                  "not proof of noninferiority or memory transfer. " + _link(record) + " contains the full sealed analysis. "
                  "Fresh stochastic replicas reuse the MNIST benchmark; no inference is recomputed here.")


def _diagnostics(data, units, audits):
    groups = defaultdict(list)
    for unit in units:
        if unit["status"] == "completed":
            meta = unit["metadata"]
            groups[(meta["family"], meta["stream"], *_recipe(meta), _policy(meta))].append(unit)

    def fit_mean(group, field):
        values = []
        artifact_name = {
            "new_nll": "final_new_nll", "replay_nll": "final_replay_nll",
            "s_t": "fisher_scale", "omega_kappa": "effective_ridge_weight",
            "ltm_motion": "ltm_displacement_norm", "stm_motion": "stm_displacement_norm",
        }.get(field, field)
        for unit in group:
            observations = [_number(row.get(field, row.get(artifact_name)), field)
                            for row in unit["fit"] if row.get(field, row.get(artifact_name)) is not None]
            if observations:
                values.append(_mean(observations))
        return f"{_mean(values):.3g}" if values else "-"

    def summary_mean(group, field):
        values = [unit["summary"].get(field) for unit in group]
        if any(value is None for value in values):
            return "-"
        return f"{_mean(_number(value, field, minimum=0) for value in values):.1f}"

    rows = []
    for key, group in sorted(groups.items())[:16]:
        family, stream, *recipe = key
        def fit_fields(fields):
            return " / ".join(fit_mean(group, field) for field in fields)
        failures = sum(unit["summary"].get("solver_failures", 0) for unit in group)
        under_solved = sum(unit["summary"].get("under_solved", 0) for unit in group)
        rows.append([("Confirmation" if family == CONFIRMATION else "Development") + f", {STREAM_LABELS[stream].lower()}",
                     _label(recipe), len(group), fit_fields(("new_nll", "replay_nll")),
                     fit_fields(("s_t", "kappa", "omega_kappa")), fit_fields(("ltm_motion", "stm_motion")),
                     f"{failures} / {under_solved}",
                     " / ".join(summary_mean(group, field) for field in (
                         "fit_seconds", "eviction_seconds", "evaluation_seconds", "active_seconds"))])
    pending = data["pending"] + (data["development"]["pending"] if data["development"] else [])
    text = ("<details>\n<summary>Diagnostics, candidate table, and incomplete progress</summary>\n\n"
            "Held-out NLL/accuracy exclude all regularizers. Fit new_nll/replay_nll are post-fit training "
            "losses, not holdout scores. Means weight trajectories equally within each stream/recipe/policy "
            "and keep development separate from confirmation. Timing columns are mean seconds per trajectory; "
            "online totals also include checkpoint and loop overhead. Aggregate study time above additionally "
            "includes initialization, reporting, audits, and acceptance. Missing diagnostics are not reconstructed.\n\n")
    text += _table(["Cohort / stream", "Recipe / policy", "Sealed n", "Post-fit new_nll / replay_nll",
                    "Ridge s_t / kappa / omega_kappa", "LTM / STM motion", "Failures / under-solved",
                    "Fit / eviction / evaluation / online s"], rows)
    text += f"\n\nShowing {len(rows)} of {len(groups)} cohort summaries (maximum 16). "
    text += f"\n\nAudit branches: {len(audits)} sealed; separately measured audit time: "
    text += f"{sum(a['result']['seconds'] for a in audits):.2f} s.\n\n"
    text += f"**Incomplete units (progress only, excluded from curves and summaries): {len(pending)}.**\n\n"
    sources = [data["development"], data] if data["development"] else [data]
    for source in sources:
        text += f"[Complete trajectory/audit JSON directory]({(source['root'] / 'units').resolve().as_posix()}). "
        for name in ("shortlist", "selection", "audit-initial", "audit-repaired"):
            if name in source["decisions"]:
                text += _link(source["decisions"][name]) + " "
    text += f"\n\n[Live progress and change log]({(data['root'] / 'status.json').resolve().as_posix()})."
    return _empty(text + "\n\n</details>")


def build_notebook(base_notebook: Path, root: Path):
    """Replace only this module's tagged cells, appending one bounded section."""
    data = collect(root)
    result = _final_results(data)
    status = data["status"]
    units = [*data["development"]["trajectories"], *data["trajectories"]] if data["development"] else data["trajectories"]
    audits = [*data["development"]["audits"], *data["audits"]] if data["development"] else data["audits"]
    aggregate = status.get("aggregate_active_seconds", status["active_seconds"])
    development = data["development"] or data
    decisions = development["decisions"]
    policy = decisions.get("policy", {}).get("result", {}).get("policy", {}).get("inner_steps")
    lambda0 = decisions.get("ledger", {}).get("result", {}).get("lambda0", 160.89)
    shortlist = decisions.get("shortlist", {}).get("result", {}).get("ridges")
    forecast = status.get("forecast_hours") or decisions.get("timing", {}).get("result", {}).get("remaining_hours")
    progress = ("<a id=\"plan15-phase7\"></a>\n\n## Phase 7: Regularization and useful acquisition\n\n"
                "**Motivation:** Phases 5B and 5C raised an overfitting concern: fitting tiny incoming/replay "
                "batches coexisted with deteriorating held-out predictions. In 5C, FD beat replay but its mean "
                "rotated acquisition NLL still exceeded the frozen initial model's. Phase 7 therefore checks "
                "numerical solve adequacy, then tunes the existing ridge and FD penalties to test whether "
                "better regularization turns relative protection into useful learning. Overfitting is a "
                "working explanation, not a mechanism established by the earlier curves.\n\n"
                "Outcome-informed follow-up; earlier phases are unchanged. "
                f"**Progress:** {status.get('completed', 0)}/{status.get('expected') if status.get('expected') is not None else 'pending'} "
                f"scheduled jobs; stage **{status['stage']}**; status **{status.get('status', 'unknown')}**; "
                f"aggregate active **{aggregate / 3600:.2f} h** (current root {status['active_seconds'] / 3600:.2f} h). "
                f"Current job: `{status.get('current_job') or 'none'}`. "
                f"Forecast recorded after 7A (hours remaining at that point): "
                f"{json.dumps(forecast, ensure_ascii=True)}.\n\n"
                f"Verified Phase 7 artifacts: {sum(u['status'] == 'completed' for u in units)} sealed trajectories, "
                f"{len(audits)} sealed audit branches. Scheduled-job counters are live progress, not evidence of completion.\n\n"
                "We ask whether calibrating the existing regularizers improves prediction from scarce new observations, "
                "relative to both calibrated replay and the frozen initial model. The ridge multiplier is $c$, with "
                "$\\kappa_t=c\\,\\operatorname{tr}(\\widehat F_{LL,t})/487$, where 487 is the LTM coordinate count; "
                "the actual quadratic coefficient is "
                "$39\\kappa_t/2$. The FD penalty strength is $\\lambda_F$. Both methods keep "
                "$\\pi=0.025$, the 32-image FIFO capacity, and the model fixed. Held-out predictive NLL excludes "
                "all these penalties. All tuning uses development replicas before fresh confirmation.")
    cells = [_empty(progress), _audit_plot(audits),
             _development_plot(units, policy_iterations=policy, lambda0=lambda0, shortlisted=shortlist), _decisions(data),
             _empty("### Fresh confirmation\n\nHeld-out predictive NLL and accuracy use the same fixed class-weighted "
                    "panels within a replica. **Gray dotted lines show the frozen initial model's performance** "
                    "on those same panels, with no continual-learning updates.\n\n"
                    "- **Mixed training:** mostly unrotated digits plus rare $30^\\circ$-rotated digits. "
                    "The horizontal coordinate counts actual new rotated images, ending at 100.\n"
                    "- **Unrotated-only control:** the matched training images stay unrotated. "
                    "It has zero novel training exposures; its horizontal coordinate follows the matching "
                    "mixed stream's event times, not rotated images seen by this control.\n\n"
                    "**Both Running $A(n)$ panels evaluate the fixed rotated-digit test panel**, regardless of "
                    "training stream. Each is the cumulative mean of rotated-test NLL at events 1 through $n$, "
                    "excluding initialization, not the loss on that stream's training data. Lower is better. "
                    "NLL panels use nats; accuracy panels use percentages. The paired-gain panel lists replicas "
                    "on its vertical axis, and the compute panel lists training streams and methods. Axis limits "
                    "are set independently for each panel. "
                    "Partial results remain descriptive; each arm's sealed cohort count is shown in its legend "
                    "and may differ, so partial curves need not be paired."),
             _confirmation_plot(units, result), _inference(data, result), _diagnostics(data, units, audits)]
    for cell in cells:
        cell.metadata["tags"] = [TAG]
    notebook = nbformat.read(base_notebook, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not {TAG, PROGRESS_TAG}.intersection(cell.metadata.get("tags", []))]
    current = str(status.get("current_job") or "none").replace("\n", " ").replace("`", "'")[:96]
    heartbeat = str(status.get("heartbeat") or "not recorded")[:40]
    summary = _empty(
        f"**Phase 7 execution:** `{status['stage']}` / `{str(status.get('status', 'unknown'))[:32]}` | "
        f"total active **{aggregate / 3600:.2f} h** | **{status.get('completed', 0)}/"
        f"{status.get('expected') if status.get('expected') is not None else 'pending'}** jobs  \n"
        f"Current: `{current}` | heartbeat: `{heartbeat}` | [Phase 7 details](#plan15-phase7)")
    summary.metadata["tags"] = [PROGRESS_TAG]
    title_end = next((index + 1 for index, cell in enumerate(notebook.cells)
                      if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# ")), 0)
    notebook.cells.insert(title_end, summary)
    notebook.cells.extend(cells)
    nbformat.validate(notebook)
    if len(nbformat.writes(notebook).encode("utf-8")) > MAX_NOTEBOOK_BYTES:
        raise ArtifactError("Phase 7 notebook exceeds bounded publication size")
    return notebook
