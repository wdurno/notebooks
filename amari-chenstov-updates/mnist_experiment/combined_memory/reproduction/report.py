"""Artifact-only Phase 8 presentation and immutable report publication.

``publish(snapshot, notebook, store=None)`` accepts progress (stage, status,
counts, charged_hours, forecast_hours), fidelity, settings and optional analysis.
Analysis is precomputed by ``analysis.analyze``, never by this module. Its
``curves[context][policy]``, summaries, contrasts, efficiency and blur_interaction
are adapted without inference. Presentation also accepts
``blurred={arms, comparisons, efficiency}`` and
``blur_contrast={contexts, comparisons, interaction}``; absent fields stay
unresolved. Arm keys may be R2/R1/F1 or their historical policy names. Curves
use n (blurred) or incoming_grids (context comparison), and novel/old mappings
with mean_mse arrays. Legacy base/mean_mse and novel_mse/old_mse are supported.

The Store supplies root and optionally reserve(bytes). Without it, publication
uses DEFAULT_ROOT. Only this phase's explicitly tagged cells are replaceable.
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

from . import PROTOCOL

REPORT_SCHEMA = 1
ANALYSIS_PROTOCOL = "plan18-stm-reproduction-analysis-v1"
TAG, PREFIX = "plan18_phase8", "plan18-phase8-"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "cache/mnist_experiment/plan18/stm_reproduction_v1"
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[2] / "plan18_results.ipynb"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 64 * 1024 * 1024
MAX_GENERATION_BYTES = 160 * 1024 * 1024
ARMS = ("R2", "R1", "F1")
POLICIES = dict(R2="all_r2--replay_ewc", R1="head_r1--replay_ewc", F1="head_r1--fd_replay_ewc")
LABELS = dict(R2="R2: rank-two Replay/EWC", R1="R1: head Replay/EWC", F1="F1: head FD policy")
COLORS = dict(R2="#59636e", R1="#2676aa", F1="#b13d70")
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
    raise ValueError("report accepts finite JSON scalars only, not model/tensor artifacts")


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
        raise ValueError("bounded report artifact size exceeded")
    return data


def _text(value):
    return html.escape(str(value), quote=False).replace("|", "/").replace("\n", " ")


def _number(value, digits=3):
    return f"{value:.{digits}g}" if type(value) in (int, float) and math.isfinite(value) else "unresolved"


def _effect(value, scale=1., interval=True):
    item = _mapping(value)
    point = item.get("mean", item.get("point", value if not item else None))
    result = _number(point * scale if type(point) in (int, float) else None)
    ci = item.get("ci")
    if interval and isinstance(ci, (list, tuple)) and len(ci) == 2:
        result += " [" + ", ".join(_number(v * scale) for v in ci) + "]"
    return result


def _table(headers, rows):
    return "\n".join("| " + " | ".join(_text(v) for v in row) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def owned(cell):
    return cell.get("metadata", {}).get(TAG) is True or cell.get("id", "").startswith(PREFIX)


def cell(name, text):
    return nbformat.v4.new_markdown_cell(text, id=PREFIX + name, metadata={TAG: True})


def _blurred(result):
    if "blur" in _mapping(result.get("curves")):
        context = _native_context(result, "blur")
        context["comparisons"] = {}
        for name, raw in _mapping(result.get("contrasts")).items():
            item = _mapping(raw)
            gain = _mapping(item.get("acquisition_gain"))
            difference = dict(gain)
            if type(gain.get("mean")) in (int, float):
                difference["mean"] = -gain["mean"]
            if isinstance(gain.get("ci"), list) and len(gain["ci"]) == 2:
                difference["ci"] = [-gain["ci"][1], -gain["ci"][0]]
            context["comparisons"][name] = dict(novel_auc=difference,
                endpoint_risk=item.get("endpoint_risk_difference"), retention=item.get("retention_difference"))
        context["efficiency"] = {"F1-R2": _mapping(result.get("efficiency")).get("F1", {})}
        return context
    return _mapping(result.get("blurred", result.get("fresh", result)))


def _contrast(result):
    if "blur_interaction" in result:
        raw = _mapping(result.get("blur_interaction"))
        endpoints = _mapping(raw.get("endpoint_differences"))
        return dict(status=raw.get("status"), interaction=raw.get("interaction"), comparisons={
            "blurred": dict(average_risk_difference=raw.get("delta_blur"), endpoint_difference=endpoints.get("blur")),
            "no_blur": dict(average_risk_difference=raw.get("delta_no_blur"), endpoint_difference=endpoints.get("no_blur"))},
            contexts={"blurred": _native_context(result, "blur"), "no_blur": _native_context(result, "no_blur")})
    return _mapping(result.get("blur_contrast", result.get("blur", {})))


def _native_context(result, name):
    curves = _mapping(_mapping(result.get("curves")).get(name))
    summaries = _mapping(_mapping(result.get("summaries")).get(name))
    arms = {}
    for policy, raw in curves.items():
        curve = _mapping(raw)
        arms[policy] = dict(curves={**curve, "incoming_grids": curve.get("observations"),
                                   "novel": curve.get("query"), "old": curve.get("base")})
    reference = "R2" if "R2" in arms else "R1"
    base = _mapping(_mapping(curves.get(reference)).get("base")).get("mean_mse", [])
    # The native analyzer validates n=0 as frozen initialization, not a trained state.
    frozen = dict(novel_mse=_mapping(_mapping(summaries.get(reference)).get("frozen_mse")).get("mean"),
                  old_mse=base[0] if base else None)
    return dict(arms=arms, replicas=result.get("replicas"), frozen=frozen, summaries=summaries)


def _arms(context):
    arms = _mapping(context.get("arms", context.get("methods")))
    return {arm: _mapping(arms.get(arm, arms.get(POLICIES[arm]))) for arm in ARMS
            if arm in arms or POLICIES[arm] in arms}


def _curve(item, domain, clock):
    curves = _mapping(item.get("curves"))
    metric = _mapping(curves.get(domain, curves.get("base") if domain == "old" else None))
    x = curves.get(clock)
    y = metric.get("mean_mse", curves.get(domain + "_mse"))
    if x is None and y is None:
        return None
    if y is None:
        return None
    if (not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in x + y)
            or any(b <= a for a, b in zip(x, x[1:]))):
        raise ValueError(f"invalid scalar {domain} curve or {clock} clock")
    return x, y


def _contexts(result):
    contrast = _contrast(result)
    contexts = _mapping(contrast.get("contexts", contrast))
    return {"blurred": _mapping(contexts.get("blurred", contexts.get("blur"))),
            "no_blur": _mapping(contexts.get("no_blur", contexts.get("unblurred")))}


def _validate_analysis(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("scalar analysis must be a mapping or None")
    result = _json_value(value)
    if result.get("schema_version", 1) != 1 or result.get("protocol", PROTOCOL) not in (PROTOCOL, ANALYSIS_PROTOCOL):
        raise ValueError("incompatible Phase 8 scalar analysis schema/protocol")
    if result.get("namespace", "fresh") != "fresh":
        raise ValueError("historical analysis cannot be presented as fresh inference")
    for name, policies in (("blur", set(ARMS)), ("no_blur", {"R1", "F1"})):
        curves = _mapping(_mapping(result.get("curves")).get(name))
        if curves and set(curves) != policies:
            raise ValueError("native scalar curves require complete paired policy blocks")
    contexts = [(_blurred(result), "n"), *[(c, "incoming_grids") for c in _contexts(result).values()]]
    for context, clock in contexts:
        if context.get("pairing_complete") is False:
            raise ValueError("analysis must contain complete paired blocks, not unmatched paths")
        replicas = context.get("replicas")
        if replicas is not None and (not isinstance(replicas, list)
                or any(type(r) not in (str, int) for r in replicas) or len(set(replicas)) != len(replicas)):
            raise ValueError("invalid paired replica identifiers")
        for item in _arms(context).values():
            for domain in ("novel", "old"):
                _curve(item, domain, clock)
    return result


def load_analysis(path, expected_sha256=None):
    """Read bounded scalar JSON; never load checkpoints, repair, or bootstrap."""
    data = _read(Path(path), MAX_JSON_BYTES)
    if expected_sha256 is not None and _hash(data) != expected_sha256:
        raise ValueError("scalar analysis checksum mismatch")
    return _validate_analysis(json.loads(data))


def _status(value):
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    item = _mapping(value)
    if item.get("smoke_only") is True:
        return _text(item.get("status", "pending")) + " (smoke only; not production acceptance)"
    return _text(item.get("status", _status(item["passed"]) if "passed" in item else "pending") if item else value or "pending")


def _count(counts, keys, target):
    value = next((counts[k] for k in keys if k in counts), None)
    item = _mapping(value)
    done = item.get("completed", item.get("done")) if item else value
    total = item.get("planned", item.get("total", target)) if item else target
    return f"{_number(done, 4) if done is not None else '?'}/{_number(total, 4)}"


def _progress(snapshot):
    p = _mapping(snapshot.get("progress"))
    counts = _mapping(p.get("counts"))
    forecast = 0. if p.get("status") == "completed" else p.get("forecast_hours")
    counts = dict(counts)
    for key, field in (("fresh_blocks", "fresh_blocks_completed"), ("blurred", "blurred_completed"),
                       ("no_blur", "no_blur_completed")):
        if field in p:
            counts.setdefault(key, p[field])
    return cell("progress", "### Phase 8 Progress\n\n"
        f"**{_text(p.get('stage', '8A'))}: {_text(p.get('status', 'pending'))}** | "
        f"Fidelity: **{_status(snapshot.get('fidelity'))}**.  \n"
        f"Fresh paired blocks {_count(counts, ('fresh_blocks', 'paired_blocks', 'blocks'), 8)}; "
        f"8B blurred trajectories {_count(counts, ('blurred', 'blur', 'blurred_trajectories', 'fresh'), 24)}; "
        f"8C no-blur trajectories {_count(counts, ('no_blur', 'no_blur_trajectories'), 16)}.  \n"
        f"Charged compute: {_number(p.get('charged_hours'))} h; "
        f"remaining forecast: {_number(forecast)} h. "
        "Only complete paired blocks enter summaries; partial results are provisional.")


def _fidelity(snapshot):
    fidelity = _mapping(snapshot.get("fidelity"))
    checks = fidelity.get("checks", fidelity.get("same_seed_paths", {}))
    if "checks" not in fidelity and fidelity.get("same_seed_paths"):
        checks = [dict(name="Historical source identity", status="passed",
                       details=f"{fidelity.get('source_files', '?')} frozen files")]
        for item in fidelity["same_seed_paths"]:
            unit = item.get("historical_unit", "Historical path")
            policy = "R2" if unit.startswith("all_r2-") else "F1" if "fd_replay_ewc" in unit else "R1"
            comparisons = item.get("checks", [])
            maximum = max((c.get("max_absolute", 0.) for c in comparisons), default=0.)
            checks.append(dict(name=f"{policy} same-seed trajectory", status=_status(item),
                               details=f"{len(comparisons)} comparisons; max absolute discrepancy {_number(maximum)}"))
    rows = []
    if isinstance(checks, Mapping):
        checks = [dict(_mapping(value), name=name) if isinstance(value, Mapping)
                  else dict(name=name, status=_status(value)) for name, value in checks.items()]
    if isinstance(checks, list):
        for check in checks:
            item = _mapping(check)
            href = item.get("href", item.get("audit_href"))
            link = f"[Audit](<{str(href).replace('>', '%3E')}>)" if href else item.get("details", "pending")
            rows.append([item.get("name", item.get("check", "Audit")), _status(item), link])
    if fidelity.get("statistics"):
        rows.append(["Historical scalar statistics", _status(fidelity["statistics"]),
                     f"{_mapping(fidelity['statistics']).get('checks', '?')} scalar checks; see sealed snapshot"])
    return cell("fidelity", "### Historical Fidelity\n\n" + _table(
        ("Check", "Result", "Details"), rows or [["Historical replay and scalar audit", _status(fidelity), "pending"]])
        + "\n\nHistorical confirmation and same-seed reruns are acceptance evidence, never fresh replicas. "
        "Numerical fidelity is required before fresh execution; a favorable efficacy result is not.")


def _settings(snapshot):
    rows = []
    settings = snapshot.get("settings") or []
    if isinstance(settings, Mapping):
        settings = [dict(name=k, value=v) for k, v in settings.items()]
    for setting in settings:
        item = _mapping(setting)
        if item:
            rows.append([item.get("name", item.get("label", item.get("arm", item.get("policy", "Setting")))),
                         item.get("value", item.get("description", " / ".join(
                             f"{k}: {_text(v)}" for k, v in item.items() if k not in ("name", "label", "arm"))))])
        elif isinstance(setting, (list, tuple)) and len(setting) == 2:
            rows.append(setting)
        else:
            rows.append(["Setting", setting])
    return cell("settings", "### Frozen Settings\n\n" + _table(("Policy", "STM ranks / ridge / FD strength"), [
        ["R2: historical headline reference", "2,2,2 / 3.16228 / 0"],
        ["R1: same-chart control", "0,0,1 / 3.16228 / 0"],
        ["F1: historical treatment", "0,0,1 / 1 / 3.34056"]])
        + ("\n\n" + _table(("Setting", "Recorded value"), rows) if rows else "")
        + "\n\nFixed pi=.025; FIFO32/replay2; rank-64-plus-diagonal archive; 100 designated novel grids. "
        "The no-blur control keeps the paired stopping schedule, but those arrivals are not novel. "
        "Recipes are frozen policies with different ridges, not an isolated FD-term intervention.")


def _figure(result, view):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    contrast = view == "blur_contrast"
    figure = Figure(figsize=(12, 7.2 if contrast else 3.9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2 if contrast else 1, 2, squeeze=False)
    groups = _contexts(result) if contrast else {"blurred": _blurred(result)}
    for row, (context, data) in enumerate(groups.items()):
        for axis, domain in zip(axes[row], ("novel", "old")):
            for arm, item in _arms(data).items():
                if contrast and arm == "R2":
                    continue
                curve = _curve(item, domain, "incoming_grids" if contrast else "n")
                if curve:
                    x, y = curve
                    axis.plot(x, [math.sqrt(v) for v in y] if domain == "old" else y,
                              color=COLORS[arm], label=LABELS[arm], linewidth=2,
                              linestyle="--" if arm == "R1" else "-")
            frozen = _mapping(data.get("frozen")).get(domain + "_mse")
            if type(frozen) in (int, float) and frozen >= 0:
                axis.axhline(math.sqrt(frozen) if domain == "old" else frozen,
                             color="#242424", linestyle=":", label="Frozen initialization")
            task = "No-blur stationary control" if context == "no_blur" else "Blurred task"
            axis.set(title=task + (": held-out learning" if domain == "novel" else ": old-data retention"),
                     xlabel="Total integrated grids" if contrast else "Designated novel grids",
                     ylabel="Held-out risk (px^2 / landmark)" if domain == "novel" else "Root mean old risk (px)")
            axis.grid(alpha=.18)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=10)
            axis.title.set_fontsize(12)
            if axis.lines:
                axis.legend(fontsize=9, loc="best")
            else:
                axis.text(.5, .5, "Complete paired scalar curves pending", ha="center", va="center", transform=axis.transAxes)
    return figure


def display_figures(result, view="blurred"):
    """Render only already-aggregated scalar curves, including on notebook rerun."""
    if view not in ("blurred", "blur_contrast"):
        raise ValueError("unknown Phase 8 figure view")
    from IPython.display import display

    figure = _figure(_validate_analysis(result), view)
    try:
        display(figure)
    finally:
        figure.clear()


def _plot_cell(result, view, artifact_href):
    figure = _figure(result, view)
    try:
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=120, metadata={"Software": "Plan18 Phase8 scalar report"})
        png = output.getvalue()
    finally:
        figure.clear()
    source = ("from pathlib import Path\n"
        "from mnist_experiment.combined_memory.reproduction.report import load_analysis, display_figures\n"
        f"_paths = [Path({str(artifact_href)!r}), Path('analysis.json')]\n"
        "_artifact = next((p for p in _paths if p.is_file()), _paths[0])\n"
        f"_analysis = load_analysis(_artifact, expected_sha256={_hash(_bytes(result))!r})\n"
        f"display_figures(_analysis, view={view!r})")
    return nbformat.v4.new_code_cell(source, id=PREFIX + view + "-plot", metadata={TAG: True}, execution_count=1,
        outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode("ascii"),
            "text/plain": "Phase 8 paired scalar trajectories"})])


def _comparisons(context):
    return _mapping(context.get("comparisons", context.get("effects", context.get("contrasts"))))


def _effects(result):
    context = _blurred(result)
    comparisons = _comparisons(context)
    rows = []
    for name in ("F1-R2", "F1-R1"):
        item = _mapping(comparisons.get(name))
        average = item.get("novel_auc", item.get("average_risk_difference"))
        favorable = _mapping(average).get("favorable_replicas", item.get("favorable_replicas"))
        rows.append([name, _effect(average), _effect(item.get("endpoint_risk", item.get("endpoint_difference"))),
                     _effect(item.get("old_rms_deterioration", item.get("retention"))),
                     _number(favorable, 4) + "/" + str(len(context.get("replicas") or []) or "?")])
    return cell("effects", "### Fresh Blurred-Task Paired Effects\n\n" + _table(
        ("F1 minus reference", "Average risk [95% CI]", "Endpoint risk", "Old RMS difference (px)", "Favorable replicas"), rows)
        + "\n\nNegative risk differences favor F1. Prediction risk is unregularized held-out squared error / 27 landmarks, "
        "not / 54 coordinates. Retention roots within each replica before averaging. "
        "Intervals are nominal whole-replica paired summaries, not familywise-adjusted claims. "
        "Evicted-training-cohort error is a separate diagnostic in the scalar analysis, not a holdout outcome.")


def _efficiency(result):
    context = _blurred(result)
    efficiencies = _mapping(context.get("efficiency"))
    rows = []
    diagnostic = ""
    for name in ("F1-R2", "F1-R1"):
        if name == "F1-R1" and name not in efficiencies:
            continue
        item = _mapping(efficiencies.get(name, _arms(context).get("F1", {}).get("efficiency") if name == "F1-R2" else None))
        point, bootstrap = _mapping(item.get("point")), _mapping(item.get("bootstrap"))
        ci = _mapping(bootstrap.get("ci", item.get("ci")))
        valid = point.get("status") == "valid"
        intervals = valid and bootstrap.get("invalid") == 0
        gain = point.get("target_mse_gain_fraction")
        if name == "F1-R2" and type(gain) in (int, float):
            diagnostic = ("\n\n**Reference improvement from initialization:** "
                + _number(100 * gain) + "% (5% minimum diagnostic).")

        def estimate(keys, scale=1.):
            key = next((k for k in keys if k in point), keys[0])
            return _effect(dict(mean=point.get(key) if valid else None, ci=ci.get(key) if intervals else None), scale)

        full = item.get("full_budget_retention_supported")
        matched = item.get("matched_budget_retention_supported")
        full_label = "supported" if full is True else "not established" if full is False else "unresolved"
        matched_label = ("not estimable" if not intervals else "supported" if matched is True
                         else "not established" if matched is False else "unresolved")
        rows.append([name + (" headline" if name == "F1-R2" else " secondary target"),
            estimate(("saved_fraction",), 100), estimate(("learning_time_ratio", "matched_time_ratio")),
            estimate(("S",)), _effect(dict(mean=item.get("equal_budget_learning_time_ratio"),
                ci=item.get("equal_budget_learning_time_ratio_ci"))),
            full_label + " / " + matched_label,
            str(bootstrap.get("invalid", "?")) + "/" + str(bootstrap.get("draws", "?")) + ("; CI withheld" if not intervals else "")])
    return cell("efficiency", "### Historical-Definition Sample Efficiency\n\n" + _table(
        ("Target", "Saved %", "Matched compute ratio", "S", "Equal-budget time ratio", "Retention full / matched", "Invalid draws"), rows)
        + diagnostic
        + "\n\n**S is sample fraction saved divided by the compute ratio needed to reach the same predictive quality.** "
        "It is not percent saved: historical 41.4% savings and S near .605 are different quantities. "
        "The headline uses R2's endpoint at N=100 and F1's last sustained downward crossing of unsmoothed mean risk. "
        "R2 uses its allocated budget by definition. R1-target equivalence is secondary. "
        "Any unsupported inverse draw withholds ordinary inverse intervals; unresolved targets are not zero savings. "
        "The 5% improvement diagnostic, .50 px ceiling and .10 px one-sided retention tolerance apply at full and matched budgets, "
        "not as two-sided equivalence. Online time includes background arrivals, replay, FD, retirement and compression; "
        "pretraining, audits and reporting are charged separately. Savings are conditional on initialization, background arrivals, "
        "drift history and the endpoint target, not a causal estimate of deleting labels.")


def _blur_effects(result):
    contrast = _contrast(result)
    comparisons = _comparisons(contrast)
    rows = []
    for key, label in (("blurred", "Blurred: F1-R1"), ("no_blur", "No blur: F1-R1")):
        item = _mapping(comparisons.get(key))
        rows.append([label, _effect(item.get("average_risk_difference", item.get("Q_difference"))),
                     _effect(item.get("endpoint_difference"))])
    rows.append(["Policy-by-context interaction (blurred minus no blur)", _effect(contrast.get("interaction")), "See context endpoints above"])
    return cell("blur-effects", "### Policy By Blur Context\n\n" + _table(
        ("Paired contrast", "All-arrival average risk [95% CI]", "Endpoint risk difference"), rows)
        + "\n\nAll-arrival averages use the fixed trapezoidal rule divided by each replica's shared total-grid horizon. "
        "Complete four-path blocks are resampled jointly. Negative interaction favors F1 relatively more under blur. "
        "Both training and evaluation appearance change; this is blur-context dependence, not a pure training-blur causal effect. "
        "The no-blur condition is a stationary nominal-law control: no novel-sample savings or S is calculated. "
        "Do not infer interaction from one significant and one nonsignificant within-context result.")


def _outcome(result):
    if not result or not result.get("is_final"):
        return "Fresh statistical replication and blur dependence remain provisional until the fixed cohort is complete. "
    item = _mapping(_mapping(result.get("efficiency")).get("F1"))
    point, bootstrap = _mapping(item.get("point")), _mapping(item.get("bootstrap"))
    ci = _mapping(bootstrap.get("ci")) if bootstrap.get("invalid") == 0 else {}
    if point.get("status") == "valid":
        def value(key, scale=1.):
            return _effect(dict(mean=point.get(key), ci=ci.get(key)), scale)
        text = ("On fresh replicas, F1's point sample-saving estimate against R2 is **"
            + value("saved_fraction", 100) + "%**, at **" + value("learning_time_ratio")
            + "x** matched compute, with **S=" + value("S") + "**. ")
        if bootstrap.get("invalid"):
            text += f"The inverse CI is withheld: {bootstrap['invalid']}/{bootstrap.get('draws', '?')} draws are unsupported. "
        if not point.get("useful_target"):
            text += "The reference target does not pass the minimum 5% improvement diagnostic. "
    else:
        reason = {"target_not_reached_at_observed_endpoint": "F1 did not reach the reference quality within the observed budget",
                  "target_already_met_at_initialization": "the target was already met before online learning",
                  "crossing_below_first_post_observation_checkpoint": "the crossing precedes the first saved post-observation checkpoint"}
        text = "Sample savings are not estimable: " + reason.get(point.get("status"), "the inverse is unresolved") + ". "
    if item.get("full_budget_retention_supported") and item.get("matched_budget_retention_supported"):
        text += "The full- and matched-budget retention checks pass. "
    else:
        text += "Retention qualification is not established for the sample-saving claim. "
    risk = _mapping(_comparisons(_blurred(result)).get("F1-R2")).get("novel_auc")
    text += "Average new-domain risk difference (F1-R2): " + _effect(risk) + " px^2 per landmark. "
    interaction = _mapping(_contrast(result).get("interaction"))
    bounds = interaction.get("ci")
    text += "The blur-context interaction is " + _effect(interaction) + " px^2 per landmark. "
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        if bounds[1] < 0:
            text += "F1's relative risk effect is more favorable in the blurred context. "
        elif bounds[0] > 0:
            text += "F1's relative risk effect is more favorable in the no-blur context. "
        else:
            text += "This interval includes zero, so blur dependence remains unresolved. "
    return text + "Intervals are nominal paired 95% summaries. "


def build_cells(snapshot, analysis_href="analysis.json"):
    """Build only Phase 8 cells from supplied JSON; do not mutate the snapshot."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("report snapshot must be a mapping")
    result = _validate_analysis(snapshot.get("analysis"))
    out = [_progress(snapshot), cell("intro", "## Phase 8: Reproduction And Blur Dependence\n\n"
        "Can the historical Plan 15 computation be reproduced, does its frozen policy help on fresh replicas, "
        "and does its relative benefit depend on blur? These are separate questions. "
        "Blur is a plausible explanation for weak transfer to Plan 18, not an established cause. "
        "[Design](plan18.md#phase-8-reproduction-and-blur-dependence)."), _fidelity(snapshot), _settings(snapshot)]
    if result:
        final = result.get("is_final") and _mapping(snapshot.get("progress")).get("status") == "completed"
        out.append(cell("provisional", ("**Final: the fixed eight-replica allocation and integrity closeout are complete.** " if final
            else "**Provisional unless the fixed eight-replica allocation and integrity closeout are complete.** ")
            + "Only complete three-policy blurred blocks and complete four-path context blocks enter their respective summaries. "
            "Unfinished paths are pending, not silently dropped; smoke results are not efficacy evidence."))
        out.append(cell("learning", "### Fresh Learning And Retention\n\n"
        "Blurred task: R2, R1 and F1; dotted black is the supplied frozen reference. "
            "Old curves root aggregated MSE; they are not the replica-rooted retention statistics in the table. "
            "Native analyzer n=0 is validated frozen initialization; missing reference scalars remain unavailable."))
        out.append(_plot_cell(result, "blurred", analysis_href))
        out.extend([_effects(result), _efficiency(result), cell("blur-learning", "### Paired Blur / No-Blur Learning And Retention\n\n"
            "R1 and F1 use paired total-grid clocks and corresponding held-out panels. "
            "The stationary no-blur observations are not designated as novel."),
            _plot_cell(result, "blur_contrast", analysis_href), _blur_effects(result)])
    else:
        out.append(cell("pending", "### Analysis Pending\n\n"
            "Progress and fidelity are available before fresh analysis. No inference or efficiency claim is made. "
            "This notebook only loads validated scalar artifacts and renders lightweight plots; "
            "it does not train, download data, reconstruct Fisher matrices, bootstrap or repair runs."))
    if analysis_href:
        out.append(cell("artifacts", "[Scalar analysis](<" + str(analysis_href).replace(">", "%3E")
            + ">) | [Snapshot and detailed audits](<" + str(Path(analysis_href).with_name("snapshot.json")).replace(">", "%3E")
            + ">). Per-replica values, inverse failures and numerical logs remain in artifacts."))
    out.append(cell("conclusion", "### Conclusion And Closeout\n\n"
        f"Computational reproduction: **{_status(snapshot.get('fidelity'))}**. "
        + (_outcome(result) if result and _mapping(snapshot.get("progress")).get("status") == "completed"
           else "Fresh statistical replication and blur dependence remain provisional. " if result
           else "Fresh statistical replication and blur dependence remain pending. ")
        + "Different historical ridges limit mechanistic claims: neither this contrast nor parameter counts establish "
        "an isolated FD effect, STM-to-LTM transfer or the cause of Plan 18's whole performance gap. "
        "Integrity/pairing audit and restart notes: " + _text(snapshot.get("closeout",
            "complete; the driver verified the fixed paired allocation and immutable outputs"
            if _mapping(snapshot.get("progress")).get("status") == "completed"
            else "pending; no completed closeout is claimed"))))
    return out


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
    actual = set()
    size = 0
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
    """Seal a generation, then atomically publish the notebook and latest pointer.

    Earlier cells (including source lists, outputs and metadata) are preserved
    as JSON values. Existing generations are verified, never repaired or edited.
    Interrupted publication can resume by reusing the completed generation.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("report snapshot must be a mapping")
    snapshot = _json_value(snapshot)
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
        if previous.get("metadata", {}).get("plan15_phase8_presentation", {}).get("role") == "frozen_archive":
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
            cells = build_cells(snapshot, os.path.relpath(destination / "analysis.json", notebook.parent))
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
