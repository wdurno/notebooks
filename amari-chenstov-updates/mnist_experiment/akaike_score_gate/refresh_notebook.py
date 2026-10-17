"""Render the progressive Plan 14 lab notebook from sealed scalar artifacts only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan14-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import analysis
from .analysis import PHASE_LABELS, PHASE_ORDER


DEFAULT_ROOT = Path("cache/mnist_experiment/akaike_score_gate/default_float64")
DEFAULT_NOTEBOOK = Path("mnist_experiment/akaike_score_gate_results.ipynb")
TARGETS = (.5, .75, 1.)
MISSING = "not recorded"
RAW_DETAILS_LIMIT = 12_000
FIGURE_CACHE_MAX_BYTES = 64 * 1024 * 1024
FIGURE_CACHE_MAX_ENTRIES = 14
_FIGURE_CACHE: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
GLOSSARY = r"""## Reader's Glossary And Gate

| Quantity | Meaning |
| --- | --- |
| NLL | Mean negative log likelihood, in nats per observation; lower is better |
| $g=-s$ | Loss gradient is the negative likelihood score |
| $\mathcal I(\theta)$ | The unique unregularized likelihood Fisher field |
| $\Omega_e$, $\mu_e$, $P_e$ | Historical curvature, anchor, and actual EWC precision |
| $H$, $V$, $K=H+P_e$ | Current likelihood curvature, mean-gradient noise covariance, retained curvature |
| ESS $n_{\rm eff}$ | Actual weighted observed-data effective sample size, not model-score draws |
| $d_{\rm eff}$ | Optimism-based effective dimension, not raw parameter count |
| $\widehat G_B$, $\Delta O_B$ | Forecast data-NLL gain and additional fitting optimism |
| Pointwise 95% CI | Seeded percentile bootstrap across replicas at one scheduled evaluation; not a simultaneous band |
| Paired effect | Treatment minus comparator within the same replica and design cell; negative NLL effects favor treatment |

All gate quantities use **mean-observation units**. During each evidence episode,
the EWC anchor and precision remain fixed across depth comparisons:

$$g=\nabla_\theta L=-s,\qquad
\mathcal I(\theta)=\mathbb E_\theta[ss^T],\qquad
Q_t=\sum_i a_i L_i+\tfrac12(\theta-\mu_e)^TP_e(\theta-\mu_e).$$

The environmental prevalence moves $\theta^\star$; it is not a second argument
of the Fisher. The original learning process chooses displacements; the
auxiliary process maintains curvature. Plan 14 explicitly omits LFU in its
default conditions, not because the LFU residual vanishes.

Schur profiling accounts for refitting the already-active coordinates $A$:
$$K_{B\mid A}=K_{BB}-K_{BA}K_{AA}^{-1}K_{AB},\qquad
r_{B\mid A}=r_B-K_{BA}K_{AA}^{-1}r_A.$$

$$n_{\rm eff}=1/\sum_i a_i^2,\quad
O_C=\operatorname{tr}(K_{CC}^{-1}V_{CC})\simeq d_{{\rm eff},C}/n_{\rm eff},\quad
d_{{\rm eff},C}=\operatorname{tr}[(H_{CC}+P_{CC})^{-1}H_{CC}].$$

$$\widehat G_B=-g^T\delta-\tfrac12\delta^TH\delta,\qquad
\Delta G\equiv\widehat G_B>c\Delta O_B.$$

Equality retains the smaller model. Statistical rejection, inadequate
optimization, and budget denial are distinct outcomes. The shortcut $V\simeq
H/n_{\rm eff}$ requires calibration; numerical damping is not statistical ridge.
For a converged regular restricted MLE, $P=0$, equal weights, $V=F/n$,
summed score $U=-ng$, and $q=|B|$:
$$\widehat G_B=S_B/(2n),\quad\Delta O_B=q/n,\quad
c=1:\ \widehat G_B>\Delta O_B\ \Longleftrightarrow\ S_B>2q.$$

### Exposure And Uncertainty

The treatment origin is **0 new observations**. Mixtures share an **eight-digit
warm-up at $p=0$**, outside treatment exposure and adaptation cost. The specified
scales are **4.4 expected 9s / 80 treatment digits (development-only)**,
**9.9 / 180**, and **100.1 / 1,820**. Smoke configurations may have other counts;
figure labels always use recorded counts. Linear and sigmoid rotations, data
scales, evidence protocols, and compute tiers are not pooled. Strict-budget
panels are development-only. Progress along a changing task is not a pure
sample-size effect.

NLL-AUC is trapezoidal area divided by the exposure span, including a recorded
origin evaluation. Early endpoints use the first complete arrival at or beyond
25% and 50% of treatment exposure and state its actual count. Missing origins
are not extrapolated. Mean effects use replica pairing, not time points as
independent observations. Bootstrap seed and sample count are printed below;
one replica has no estimable CI. Incomplete replicas are counted, never repaired.

Targets {.5, .75, 1.0} nat use **fixed-task NLL**, not current-task NLL: the
$p_{\rm ref}=.1$ mixture panel or fixed 30-degree rotation panel. Attainment
requires **two consecutive evaluations**, including cost through the second.
An already-satisfied baseline is flagged, not assigned invented zero cost.
Unreached outcomes are right-censored at observed follow-up; no success-only
resource mean is reported. Later loss of an attained target is retained.

Recorded precision is averaged only where its defined flag is true; the
defined fraction is reported alongside it. Undefined precision is not zero.
Standardized precision is $.1\,\mathrm{TPR}/(.1\,\mathrm{TPR}+.9\,\mathrm{FPR})$;
threshold-swept curves, when recorded, are diagnostic, not threshold selection.
Source preparation, common warm-up, online-policy time, charged work, research
shadows, references, evaluation, and reporting costs remain separate.
"""


def _display(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.5g}" if np.isfinite(value) else MISSING
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No sealed measurements available yet._"
    def escaped(value: Any) -> str:
        return _display(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(escaped, row)) + " |" for row in rows),
    ])


def _details(title: str, body: str) -> str:
    return f"<details>\n<summary>{html.escape(title)}</summary>\n\n{body}\n\n</details>"


def _raw_details(title: str, value: Any, *, artifact_link: str) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True)
    if len(payload) > RAW_DETAILS_LIMIT:
        return _details(title, "Large nested record; full data: " + artifact_link)
    return _details(title, "<pre>" + html.escape(payload) + "</pre>\n\n" + artifact_link)


def _compact_scalars(value: dict[str, Any], *, exclude: tuple[str, ...] = (), limit: int = 10) -> str:
    fields = [(name, item) for name, item in value.items()
              if name not in exclude and not name.endswith(("hash", "hashes"))
              and (item is None or isinstance(item, (bool, int, float, str)))
              and (not isinstance(item, str) or len(item) <= 180)]
    # Status flags precede descriptive scalar fields; full values remain linked.
    fields.sort(key=lambda pair: (not isinstance(pair[1], bool), pair[0]))
    return "; ".join(f"{html.escape(name)}={html.escape(_display(item))}" for name, item in fields[:limit]) or MISSING


def _artifact_link(report: dict[str, Any], unit: dict[str, Any], name: str) -> str:
    path = Path(unit["path"]) / name
    target = os.path.relpath(path, report["notebook_dir"]) if report.get("notebook_dir") else str(path)
    return f"[{name}]({quote(target, safe='/._-')})"


def _summary_links(report: dict[str, Any], units: list[dict[str, Any]]) -> str:
    return table(["Kind", "Environment", "Index", "Full summary"], [
        [unit["config"].get("kind"), unit["config"].get("environment"), unit["config"].get("index"),
         _artifact_link(report, unit, "summary.json")] for unit in units])


def _summary_overview(summary: dict[str, Any]) -> str:
    parts = [_compact_scalars(summary, exclude=("phase", "kind", "environment", "family", "replica_index", "selected_id"))]
    for name in ("selected", "solver", "case"):
        if isinstance(summary.get(name), dict):
            parts.append(f"{name}: {_compact_scalars(summary[name], limit=6)}")
    retention = summary.get("retention")
    if isinstance(retention, dict):
        for family in ("rotation", "mixture"):
            if isinstance(retention.get(family), dict):
                parts.append(f"{family} retention: {_compact_scalars(retention[family], limit=5)}")
    return "; ".join(parts)


def _interval(result: dict[str, Any]) -> str:
    if result["mean"] is None:
        return f"{MISSING} (0/{result['n_total']})"
    ci = "CI unavailable" if result["ci95_low"] is None else (
        f"95% CI [{result['ci95_low']:.4g}, {result['ci95_high']:.4g}]")
    return f"{result['mean']:.4g}; {ci}; n={result['n']}/{result['n_total']}"


def _progress_phase(report: dict[str, Any], phase: str) -> dict[str, Any]:
    progress = report["progress"] or {}
    phases = progress.get("phases", progress.get("phase_progress", {}))
    if isinstance(phases, dict):
        value = phases.get(phase, {})
        return value if isinstance(value, dict) else {"status": value}
    if isinstance(phases, list):
        return next((row for row in phases if isinstance(row, dict) and row.get("phase") == phase), {})
    return {}


def status_markdown(report: dict[str, Any]) -> str:
    counts = []
    for row in report["phase_progress"]:
        phase = row["phase"]
        recorded = _progress_phase(report, phase)
        counts.append([phase, recorded.get("planned", recorded.get("planned_units")),
                       row["completed"], row["incomplete"], recorded.get("failed"),
                       recorded.get("pending"), recorded.get("status", "awaiting evidence")])
    progress = report["progress"]
    recorded = [] if progress is None else [[key, value] for key, value in sorted(progress.items())
                                           if key not in {"phases", "phase_progress"} and not isinstance(value, (dict, list))]
    hashes = Counter(unit["config"].get("study_hash", MISSING) for unit in report["units"])
    return ("## Study Status\n\n"
            "**Implementation/reporting is available; research/data phases are not declared complete "
            "by this notebook.** Only validated sealed units count as evidence; planned work and "
            "cohort completeness require the recorded execution overview.\n\n"
            f"Validated sealed units: **{len(report['units'])}**. Incomplete units: "
            f"**{report['incomplete']}** (including `.incomplete` staging directories). "
            "Incomplete contents are not read. Mutable progress is an overview, not sealed evidence.\n\n"
            + table(["Phase", "Planned", "Sealed", "Incomplete", "Failed", "Pending", "Recorded status"], counts)
            + "\n\n### Execution Overview\n\n" + table(["Recorded field", "Value"], recorded)
            + "\n\n" + _raw_details("Full execution overview", progress,
                artifact_link=_artifact_link(report, {"path": report["root"]}, "progress.json"))
            + "\n\n" + _raw_details("Frozen study configuration and hashes", {"study": report["study"], "sealed_study_hashes": dict(hashes)},
                artifact_link=_artifact_link(report, {"path": report["root"]}, "study.json")
                + "; per-unit hashes remain in the linked phase manifests below.")
            + f"\n\nScalar bootstrap: seed={analysis.BOOTSTRAP_SEED}, draws={analysis.BOOTSTRAP_SAMPLES}.\n\n"
            "Development and final cohorts are shown separately. Missing timing forecasts, "
            "active compute, paired counts, or selected hashes remain unreported until recorded.")


def group_label(units: list[dict[str, Any]]) -> str:
    metadata = analysis.unit_metadata(units[0])
    parts = [str(metadata.get("environment", MISSING))]
    if metadata.get("schedule") is not None:
        parts.append(str(metadata["schedule"]))
    mixture = metadata.get("family") == "mixture" or "mixture" in str(metadata.get("environment", ""))
    expected = metadata.get("expected_nines") if mixture else None
    if expected is not None:
        parts.append(f"{_display(expected)} expected 9s / {_display(metadata.get('total_observations'))} treatment digits")
        realized = [unit["metrics"][-1].get("realized_nines") for unit in units if unit.get("metrics")]
        valid = [value for value in realized if analysis._finite(value) is not None]
        parts.append(f"realized 9s range {min(valid):g}-{max(valid):g}" if valid else "realized 9s not recorded")
        parts.append("common warm-up: 8 digits, excluded")
    if metadata.get("budget_tier") is not None:
        parts.append(str(metadata["budget_tier"]))
    if expected == 4.4 or "strict" in str(metadata.get("budget_tier", "")):
        parts.append("DEVELOPMENT ONLY")
    variants = {analysis.unit_metadata(unit)["protocol_variant"] for unit in units}
    parts.append(f"protocol variant {metadata['protocol_variant']}" if len(variants) == 1 else
                 f"{len(variants)} separately identified policy protocols")
    settings = [f"{name}={_display(metadata[name])}" for name in ("c", "half_life", "cadence")
                if name in metadata and len(variants) == 1]
    if settings:
        parts.append(", ".join(settings))
    return " | ".join(parts)


def _policy_groups(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for unit in units:
        result[str(analysis.unit_metadata(unit).get("policy"))].append(unit)
    return dict(sorted(result.items()))


def _summary_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [analysis.unit_metadata(unit) for unit in units]


def _comparators(report: dict[str, Any], environment: str) -> set[str]:
    names = {"no_update", "full_space", "depth6_diag", "depth6_lr"}
    declared = report["study"].get("principal_comparator")
    if isinstance(declared, str):
        names.add(declared)
    for unit in report["units"]:
        if unit["config"].get("kind") == "decision":
            selection = unit["summary"].get("comparators", {}).get(environment, {})
            if isinstance(selection, dict) and isinstance(selection.get("fixed"), str):
                names.add(selection["fixed"])
    return names


def full_trajectory_reason(report: dict[str, Any], units: list[dict[str, Any]]) -> str | None:
    """A predeclared display filter, never a performance-based variant selection."""
    metadata = analysis.unit_metadata(units[0])
    phase = metadata["phase"]
    if phase not in {"phase1", "phase3"}:
        return "full phase comparison"
    study = report["study"]
    episodes = metadata.get("episode_arrivals")
    if episodes is not None and episodes != study.get("episode_arrivals", 32):
        return None
    candidate = {name: metadata.get(name) for name in ("c", "half_life", "cadence")}
    default = {"c": 1., "half_life": 8., "cadence": 1}
    decisions = report.get("decisions", {})
    selected = decisions.get("phase3_calibration", {}).get("selected")
    if candidate != default and candidate != selected:
        return None
    if phase == "phase3":
        return "default c=1, h=8, cadence=1" if candidate == default else "frozen Phase 3 selected principal"
    family = metadata.get("family")
    if family is None:
        family = "rotation" if str(metadata.get("environment", "")).startswith("rotation") else "mixture"
    retention = metadata.get("retention", {})
    frozen = decisions.get("phase1_selection", {}).get("retention", {}).get(family)
    if frozen is not None and retention == frozen:
        return "frozen Phase 1 selected retention"
    omega, ridge = (39., .1) if family == "rotation" else (19., 0.)
    if retention.get("omega") != omega or retention.get("ridge_ratio") != ridge:
        return None
    solver = decisions.get("optimizer_selection", {}).get("solver", {})
    max_iter = solver.get("max_iter", study.get("max_iter"))
    if max_iter is not None and retention.get("max_iter") != max_iter:
        return None
    return "default retention with common solver"


def _plot_groups(report: dict[str, Any], phase: str, *, include_development_variants: bool) -> list[list[dict[str, Any]]]:
    groups = [members for members in analysis.trajectory_groups(report, phase)
              if include_development_variants or full_trajectory_reason(report, members)]
    if phase in {"phase4", "phase5"}:
        buckets = defaultdict(list)
        for members in groups:
            buckets[analysis._comparison_key(members[0])].append(members)
        combined = []
        for parts in buckets.values():
            unique = {unit["path"]: unit for members in parts for unit in members}
            identities = [(analysis.unit_metadata(unit).get("policy"), analysis.unit_metadata(unit).get("replica_index"))
                          for unit in unique.values()]
            # Different named policies may share a panel, never a replica mean.
            combined.extend(parts if len(set(identities)) != len(identities) else [list(unique.values())])
        groups = combined
    return [members for members in groups if not (
        all(analysis._fixed_policy(unit) for unit in members)
        and any({unit["path"] for unit in members} < {unit["path"] for unit in other} for other in groups))]


def _early_rows(unit: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return analysis.early_exposure_summary(unit["metrics"], origin=0,
                                               total_observations=analysis.unit_metadata(unit).get("total_observations"))
    except ValueError:
        return []


def _targets(unit: dict[str, Any], target: float) -> dict[str, Any]:
    try:
        return analysis.target_attainment(unit["metrics"], target, origin=0)
    except ValueError:
        return analysis.target_attainment([], target, origin=0)


def phase_markdown(report: dict[str, Any], phase: str) -> str:
    units = [unit for unit in report["units"] if unit["config"]["phase"] == phase]
    counts = Counter((unit["config"].get("kind"), unit["config"].get("environment")) for unit in units)
    parts = ["### Sealed Evidence", table(["Kind", "Environment", "Sealed units"],
             [[kind, environment, count] for (kind, environment), count in sorted(counts.items(), key=str)]),
             _details(f"Full unit inventory ({len(units)} units)", table(
        ["Kind", "Environment", "Replica/index", "Policy", "Configuration hash", "Full configuration and manifest"],
        [[unit["config"].get("kind"), unit["config"].get("environment"),
          unit["config"].get("index"), unit["config"].get("policy"),
          unit["manifest"].get("config_hash"),
          _artifact_link(report, unit, "config.json") + " / " + _artifact_link(report, unit, "manifest.json")]
         for unit in units])),
             _details("Full summaries and scientific provenance", _summary_links(report, units))]
    non_trajectories = [unit for unit in units if unit["config"].get("kind") != "trajectory"]
    if non_trajectories:
        parts.extend(["### Source, Initialization, Contracts, And Audit Summaries",
                      table(["Kind", "Environment", "Index", "Recorded scalar summary"], [
                          [unit["config"].get("kind"), unit["config"].get("environment"),
                           unit["config"].get("index"), _summary_overview(unit["summary"])]
                          for unit in non_trajectories]),
                      _details("Full non-trajectory summaries, parents, hashes, and decisions", _summary_links(report, non_trajectories))])
    audit_rows = [(unit, row) for unit in units for row in unit["audit_metrics"] or []]
    if audit_rows:
        groups = defaultdict(list)
        for unit, row in audit_rows:
            key = (unit["config"].get("environment"), row.get("category"), row.get("representation"), row.get("ewc"), row.get("gate"))
            groups[key].append(row)
        compact = []
        for key, rows in sorted(groups.items(), key=str):
            means = []
            for field in ("gate_gain", "realized_current_gain", "decision.conditional_solve_relative_error", "gate_regret"):
                values = [value for row in rows if (value := analysis._finite(_nested(row, field))) is not None]
                means.append(f"{np.mean(values):.4g} (n={len(values)}/{len(rows)})" if values else MISSING)
            statuses = Counter(row.get("gate_status", MISSING) for row in rows)
            compact.append([*key, len(rows), *means, "; ".join(f"{name}: {count}" for name, count in sorted(statuses.items(), key=str))])
        parts.extend(["### Local Audit Overview", table(
            ["Environment", "Category", "Representation", "EWC", "Gate", "Rows", "Mean forecast gain", "Mean realized gain", "Mean solve error", "Mean regret", "Statuses"], compact),
            "Descriptive audit means only: states, frontiers, and score-bank repeats are not "
            "independent trajectory replicas. Finite references are not truth.",
            _details(f"Raw audit rows ({len(audit_rows)} recorded rows; full scalar artifacts)", table(
                ["Environment", "Case/index", "Rows", "Full audit data"],
                [[unit["config"].get("environment"), unit["config"].get("index"), len(unit["audit_metrics"]),
                  _artifact_link(report, unit, "audit_metrics.json")] for unit in units if unit["audit_metrics"]]))])
    for members in analysis.trajectory_groups(report, phase):
        parts.append("### " + group_label(members))
        metadata = analysis.unit_metadata(members[0])
        parts.append(table(["Stable protocol variant", "Scientific settings"], [
            [metadata["protocol_variant"], metadata["evidence_protocol"]]]))
        if len({analysis.unit_metadata(unit)["protocol_variant"] for unit in members}) > 1:
            parts.append("Compatible fixed-depth controls are reused: their canonical c=1/cadence=1 "
                         "are inactive gate settings. Evidence horizon, retention, episode, architecture, "
                         "sample/compute budgets, and replica identity still match this candidate.")
        policies = _policy_groups(members)
        rows = []
        for policy, replicas in policies.items():
            stats = [analysis.mean_interval([unit["summary"].get(field) for unit in replicas])
                     for field in ("nll_auc", "retention_nll", "online_seconds", "charged_seconds")]
            rows.append([policy, len(replicas), *map(_interval, stats)])
        parts.append(table(["Policy", "Replicas", "NLL-AUC", "Retention NLL", "Online seconds", "Charged seconds"], rows))
        if full_trajectory_reason(report, members) is None:
            parts.append("Full development-variant plots are omitted by the predeclared display rule, "
                         "not by performance. This variant remains in the summary and calibration overview. "
                         "Its sealed scalar artifacts remain available below; "
                         "`phase_figures(report, phase, include_development_variants=True)` renders all variants.")
            parts.append(_details("Development-variant artifact inventory", table(["Policy", "Replica", "Sealed artifact directory"], [
                [analysis.unit_metadata(unit).get("policy"), analysis.unit_metadata(unit).get("replica_index"),
                 _artifact_link(report, unit, "metrics.json")] for unit in members])))
            continue
        early = []
        for policy, replicas in policies.items():
            for fraction in (.25, .5, 1.):
                prefixes = [next((row for row in _early_rows(unit) if row["fraction"] == fraction), {})
                            for unit in replicas]
                counts = sorted({row["observations"] for row in prefixes if row.get("observations") is not None})
                early.append([policy, fraction, counts or None,
                              _interval(analysis.mean_interval([row.get("current_nll") for row in prefixes])),
                              _interval(analysis.mean_interval([row.get("nll_auc") for row in prefixes]))])
        parts.extend(["#### Fixed Early-Exposure Endpoints", table(
            ["Policy", "Exposure fraction", "Actual treatment counts", "Current NLL", "Normalized prefix AUC"], early)])
        comparisons = []
        controls = [policy for policy in policies if policy in _comparators(report, metadata.get("environment"))]
        for control in controls:
            for treatment in policies:
                if treatment == control:
                    continue
                # The panel already isolates active variants and admits only
                # controls with compatible gate-independent scientific settings.
                fields = tuple(field for field in analysis.GROUP_FIELDS if field != "protocol_variant")
                for effect in analysis.paired_effects(_summary_rows(members), treatment, control, "nll_auc", group_fields=fields):
                    comparisons.append([treatment, control, _interval(effect), effect["unpaired_treatment"], effect["unpaired_control"]])
        parts.extend(["#### Paired NLL-AUC Effects (Treatment Minus Control)", table(
            ["Treatment", "Comparator", "Replica effect", "Unpaired treatment", "Unpaired comparator"], comparisons),
            "Only recorded principal comparators, no-update, or full-space depth-6 controls are used; "
            "no comparator is chosen using these outcomes."])
        targets = []
        for policy, replicas in policies.items():
            for target in TARGETS:
                stats = analysis.target_summary([_targets(unit, target) for unit in replicas])
                targets.append([policy, target, f"{stats['attained']}/{stats['replicas']}",
                                stats["censored"], stats["baseline_satisfied"], stats["baseline_unknown"],
                                stats["later_loss"], stats["missing_metrics"]])
        parts.extend(["#### Fixed-Task Target Attainment", table(
            ["Policy", "Target NLL", "Attained/all", "Censored", "Baseline satisfied", "Baseline unknown", "Later loss", "Missing metrics"], targets)])
        pr_rows = []
        for field in ("nine_precision", "standardized_nine_precision"):
            for row in analysis.pointwise_summary(members, field):
                pr_rows.append([row["policy"], row["step"], field, _interval(row), row["defined_fraction"]])
        if any(row.get("nine_precision_defined") is not None for unit in members for row in unit["metrics"]):
            last_steps = {policy: max(row[1] for row in pr_rows if row[0] == policy) for policy in policies}
            headers = ["Policy", "Step", "Metric", "Mean among defined replicas", "Defined/all fraction"]
            parts.extend(["#### Final Precision And Defined Fractions", table(headers, [
                row for row in pr_rows if row[1] == last_steps[row[0]]]),
                _details("Precision and defined fractions at every evaluation", table(headers, pr_rows))])
        statuses = Counter((str(analysis.unit_metadata(unit).get("policy")), row.get("gate_status", MISSING))
                           for unit in members for row in unit["metrics"])
        parts.extend(["#### Gate And Optimization Status", table(["Policy", "Recorded status", "Evaluations"],
                     [[policy, status, count] for (policy, status), count in sorted(statuses.items(), key=str)])])
        coverage = []
        for field in (*analysis.METRIC_FIELDS, "nine_precision_defined", "gate_status", "prequential_nll",
                      "peak_memory_bytes", "evidence_age", "fit_residual"):
            available = sum(row.get(field) is not None for unit in members for row in unit["metrics"])
            total = sum(len(unit["metrics"]) for unit in members)
            coverage.append([field, available, total - available])
        parts.append(_details("Measurement coverage (no zero filling)", table(["Field", "Recorded rows", "Missing rows"], coverage)))
        cost_fields = sorted({key for unit in members for key in unit["summary"]
                              if "seconds" in key or "memory" in key or "cost" in key or "bytes" in key})
        parts.extend(["#### Recorded Cost And Memory Breakdown", table(["Policy", *cost_fields], [
            [policy, *(_interval(analysis.mean_interval([unit["summary"].get(key) for unit in replicas]))
                        for key in cost_fields)] for policy, replicas in policies.items()])])
        if not any(unit["pr_curves"] for unit in members):
            parts.append("Threshold-swept PR curves: not recorded as sealed scalar `pr_curves.json`; "
                         "argmax operating points above are not substituted for a curve.")
        else:
            parts.append("Final threshold-swept PR uses a common recall grid, interpolating each replica "
                         "separately within its recorded defined recall support. At tied recall, the "
                         "last point in the descending-threshold sweep is retained. Missing replicas "
                         "remain in the denominator. Bands are "
                         "pointwise replica CIs for precision. No extrapolation, precision envelope, "
                         "AP recomputation, or threshold selection is performed. Exact tensor curves "
                         "are checksum-verified but never deserialized.")
    if not analysis.trajectory_groups(report, phase):
        parts.append("No sealed trajectories yet. Source/init and validation evidence above remains available.")
    diagnostics = {"phase1": "Active-parameter layouts, optimizer health, and local depth-opportunity audits",
                   "phase2": "Cached Fisher spectra/errors, layer-block comparisons, conditional solve error, EWC energy, and EWC-by-gate attribution",
                   "phase3": "Optimism calibration, evidence ages, greedy-frontier and blocked-lookahead diagnostics",
                   "phase4": "Matched-data/compute development comparisons, bounded repairs, and prequential corroboration",
                   "phase5": "Frozen fresh-cohort comparisons and retention-qualified predictive gains",
                   "phase6": "Final integrity, unresolved claims, and theory-to-evidence closeout"}
    if phase in diagnostics:
        parts.append(diagnostics[phase] + ": only recorded summaries/audit rows are shown; absent diagnostics remain unresolved.")
    return "\n\n".join(parts)


def _color(policy: str) -> Any:
    names = [f"depth{depth}_{ewc}" for ewc in ("diag", "lr") for depth in range(1, 7)]
    names += [f"asg_{ewc}_{gate}" for ewc in ("diag", "lr") for gate in ("diag", "lr")]
    names += ["asg_uncalibrated", "full_space"]
    if policy == "no_update":
        return "#555555"
    index = names.index(policy) if policy in names else int(hashlib.sha256(str(policy).encode()).hexdigest()[:8], 16) % 20
    return plt.get_cmap("tab20")(index)


def _nested(row: dict[str, Any], field: str) -> Any:
    value = row
    for name in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def _managed_figure(name: str, figure: Any) -> Iterator[tuple[str, Any]]:
    try:
        yield name, figure
    finally:
        plt.close(figure)


def _source_figures(units: list[dict[str, Any]], phase: str) -> Iterator[tuple[str, Any]]:
    grouped = defaultdict(list)
    for unit in units:
        if unit["config"]["phase"] == phase and unit["config"].get("kind") != "trajectory" and unit["metrics"]:
            if any(row.get("epoch") is not None for row in unit["metrics"]):
                grouped[unit["config"].get("environment", MISSING)].append(unit)
    for environment, members in sorted(grouped.items()):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, field in zip(axes, ("validation_nll", "validation_accuracy")):
            for unit in members:
                points = [(row["epoch"], row[field]) for row in unit["metrics"]
                          if row.get("epoch") is not None and analysis._finite(row.get(field)) is not None]
                if points:
                    axis.plot(*zip(*points), label=f"index {unit['config']['index']}, width {unit['summary'].get('width', MISSING)}")
            axis.set(xlabel="Source epoch", ylabel=field.replace("_", " "))
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
            else:
                axis.text(.5, .5, MISSING, transform=axis.transAxes, ha="center")
        _finish_figure(figure, f"{phase}: {environment}", "Source fitting only; individual recorded histories, not treatment exposure")
        yield from _managed_figure(f"source-{environment}", figure)


def _curve(axis: Any, units: list[dict[str, Any]], field: str, *, x_field: str = "observations",
           early: bool = False) -> None:
    rows = analysis.pointwise_summary(units, field, x_field=x_field)
    policies = sorted({row["policy"] for row in rows}, key=str)
    plotted = False
    for policy in policies:
        points = [row for row in rows if row["policy"] == policy and row["x"] is not None]
        if not points or not any(row["mean"] is not None for row in points):
            continue
        x = [row["x"] for row in points]
        y = [np.nan if row["mean"] is None else row["mean"] for row in points]
        axis.plot(x, y, label=f"{policy} (n={min(row['n'] for row in points)}-{max(row['n'] for row in points)})",
                  color=_color(policy), linewidth=1.6)
        low = [np.nan if row["ci95_low"] is None else row["ci95_low"] for row in points]
        high = [np.nan if row["ci95_high"] is None else row["ci95_high"] for row in points]
        axis.fill_between(x, low, high, color=_color(policy), alpha=.15)
        if early:
            axis.set_xlim(0, max(x) * .5 if max(x) > 0 else 1)
        plotted = True
    if not plotted:
        axis.text(.5, .5, MISSING, ha="center", va="center", transform=axis.transAxes)
    axis.set_title(field.replace("_", " ") + (" (early 50%)" if early else ""), fontsize=10)
    xlabel = {"observations": "Treatment observations (warm-up excluded)",
              "expected_nines": "Cumulative expected 9s (not realized)",
              "online_seconds": "Mean online seconds at scheduled evaluation"}.get(x_field, x_field)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.grid(alpha=.2)
    if plotted:
        axis.legend(fontsize=6, loc="best")


def _finish_figure(figure: Any, title: str, subtitle: str) -> None:
    import textwrap
    figure.suptitle("\n".join(textwrap.wrap(title, width=120)) + "\n" + subtitle, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .88))


def _audit_figures(units: list[dict[str, Any]], phase: str) -> Iterator[tuple[str, Any]]:
    grouped = defaultdict(list)
    for unit in units:
        if unit["config"]["phase"] == phase and unit["audit_metrics"] is not None:
            grouped[unit["config"].get("environment", MISSING)].extend(unit["audit_metrics"])
    pairs = (("gate_gain", "realized_current_gain"),
             ("preview.fisher_gain", "preview.actual_weighted_nll_gain"),
             ("predicted_gain", "actual_gain"), ("forecast_gain", "realized_gain"),
             ("predicted_gain", "next_holdout_gain"), ("predicted_optimism", "observed_optimism"),
             ("predicted_gain", "realized_gain"))
    for environment, rows in sorted(grouped.items()):
        available = [(x, y) for x, y in pairs if any(analysis._finite(_nested(row, x)) is not None
                     and analysis._finite(_nested(row, y)) is not None for row in rows)]
        if available:
            figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5), squeeze=False)
            for axis, (x, y) in zip(axes.flat, available):
                for representation in sorted({row.get("representation", "recorded") for row in rows}):
                    points = [(_nested(row, x), _nested(row, y)) for row in rows
                              if row.get("representation", "recorded") == representation
                              and analysis._finite(_nested(row, x)) is not None
                              and analysis._finite(_nested(row, y)) is not None]
                    if points:
                        axis.scatter(*zip(*points), s=18, alpha=.7, label=representation, color=_color(representation))
                bounds = [min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1])]
                axis.plot(bounds, bounds, color="gray", linestyle=":")
                axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
                axis.grid(alpha=.2)
                axis.legend(fontsize=7)
            _finish_figure(figure, f"{phase}: {environment}", "Forecast versus actual; audit states are not independent replicas")
            yield from _managed_figure(f"audit-{environment}", figure)
        spectra = [row for row in rows if isinstance(_nested(row, "finite_reference.eigenvalues"), list)]
        if spectra:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            by_representation = defaultdict(list)
            for row in spectra:
                by_representation[row.get("representation", "recorded")].append(row)
            for representation, members in sorted(by_representation.items()):
                lengths = {len(_nested(row, "finite_reference.eigenvalues")) for row in members}
                if len(lengths) == 1:
                    values = np.asarray([_nested(row, "finite_reference.eigenvalues") for row in members])
                    axes[0].plot(np.mean(values, axis=0), label=representation, color=_color(representation))
                for row in members:
                    x = _nested(row, "finite_reference.relative_frobenius_error")
                    y = row.get("ewc_realized_directional_energy_error")
                    if analysis._finite(x) is not None and analysis._finite(y) is not None:
                        axes[1].scatter(x, y, color=_color(representation), s=14, alpha=.6)
            axes[0].set(xlabel="Ordered eigenvalue index", ylabel="Mean recorded eigenvalue (no PSD repair)")
            axes[0].set_yscale("symlog", linthresh=1e-8)
            axes[1].set(xlabel="Finite-reference relative Frobenius error", ylabel="Realized EWC directional energy error")
            for axis in axes:
                axis.grid(alpha=.2)
            if axes[0].get_legend_handles_labels()[0]:
                axes[0].legend(fontsize=7)
            _finish_figure(figure, f"{phase}: {environment}", "Cached spectra and decision-relevant errors; descriptive audit averages, no replica CI")
            yield from _managed_figure(f"geometry-{environment}", figure)
        factorial = [row for row in rows if row.get("category") == "ewc_gate_attribution"]
        if factorial:
            figure, axes = plt.subplots(1, 2, figsize=(11, 5))
            ewcs = sorted({row["ewc"] for row in factorial})
            gates = sorted({row["gate"] for row in factorial})
            for axis, field in zip(axes, ("gate_regret", "realized_current_gain")):
                values = np.full((len(ewcs), len(gates)), np.nan)
                for i, ewc in enumerate(ewcs):
                    for j, gate in enumerate(gates):
                        cell = [row[field] for row in factorial if row["ewc"] == ewc and row["gate"] == gate
                                and analysis._finite(row.get(field)) is not None]
                        if cell:
                            values[i, j] = np.mean(cell)
                            axis.text(j, i, f"{values[i, j]:.3g}\nn={len(cell)}", ha="center", va="center", fontsize=9)
                mesh = axis.imshow(values, cmap="coolwarm", aspect="auto")
                figure.colorbar(mesh, ax=axis, shrink=.8)
                axis.set_xticks(range(len(gates)), gates)
                axis.set_yticks(range(len(ewcs)), ewcs)
                axis.set(xlabel="Gate representation", ylabel="EWC representation", title=field.replace("_", " "))
            _finish_figure(figure, f"{phase}: {environment}", "Two-factor attribution; descriptive audit means, not independent-replica uncertainty")
            yield from _managed_figure(f"attribution-{environment}", figure)


def _variant_overview_figures(report: dict[str, Any], phase: str) -> Iterator[tuple[str, Any]]:
    if phase not in {"phase1", "phase3"}:
        return
    by_environment = defaultdict(list)
    for members in analysis.trajectory_groups(report, phase):
        metadata = analysis.unit_metadata(members[0])
        key = (metadata.get("environment"), metadata.get("schedule"), metadata.get("budget_tier"))
        by_environment[key].append(members)
    for index, (key, groups) in enumerate(sorted(by_environment.items(), key=str)):
        figure, axes = plt.subplots(1, 2, figsize=(13, 6))
        labels = []
        for members in groups:
            metadata = analysis.unit_metadata(members[0])
            retention = metadata.get("retention", {})
            labels.append(f"{metadata['protocol_variant'][:6]}\nc={_display(metadata.get('c'))}, h={_display(metadata.get('half_life'))}"
                          f"\ncad={_display(metadata.get('cadence'))}, omega={_display(retention.get('omega'))}")
        for axis, field in zip(axes, ("nll_auc", "retention_nll")):
            seen = set()
            for ordinal, members in enumerate(groups):
                for policy, replicas in _policy_groups(members).items():
                    stats = analysis.mean_interval([unit["summary"].get(field) for unit in replicas])
                    if stats["mean"] is None:
                        continue
                    axis.scatter(ordinal, stats["mean"], color=_color(policy), s=20,
                                 label=policy if policy not in seen else None)
                    if stats["ci95_low"] is not None:
                        axis.vlines(ordinal, stats["ci95_low"], stats["ci95_high"], color=_color(policy), linewidth=1)
                    seen.add(policy)
            axis.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=6)
            axis.set_ylabel(field.replace("_", " "))
            axis.grid(alpha=.2)
            if seen:
                axis.legend(fontsize=6, ncol=2)
        _finish_figure(figure, f"{phase}: {' | '.join(str(value) for value in key if value is not None)}",
                       "All development variants; separate replica means and 95% CIs; no variant pooling")
        yield from _managed_figure(f"development-variants-{index:03d}", figure)


def phase_figures(report: dict[str, Any], phase: str, *,
                  include_development_variants: bool = False) -> Iterator[tuple[str, Any]]:
    """Yield one live figure at a time; advancing/closing the iterator closes it.

    Development defaults and frozen selections receive trajectory figures;
    all variants still receive tables and compact overview plots. The opt-in
    renders every development protocol from its recorded scalars, without fits.
    """
    yield from _source_figures(report["units"], phase)
    yield from _audit_figures(report["units"], phase)
    yield from _variant_overview_figures(report, phase)
    for group_index, units in enumerate(_plot_groups(report, phase, include_development_variants=include_development_variants)):
        label = group_label(units)
        prefix = f"group-{group_index:03d}"
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        for axis, field, early in zip(axes.flat,
                ("current_nll", "current_nll", "retention_nll", "current_accuracy", "retention_accuracy", "prequential_nll"),
                (False, True, False, False, False, False)):
            _curve(axis, units, field, early=early)
        _finish_figure(figure, label, "Learning curves; means and pointwise 95% replica bootstrap CIs")
        yield from _managed_figure(prefix + "-learning", figure)
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, field in zip(axes.flat, ("selected_depth", "n_eff", "effective_dimension_ratio", "active_parameters")):
            _curve(axis, units, field, x_field="online_seconds" if field == "active_parameters" else "observations")
        _finish_figure(figure, label, "Depth and information; mean and pointwise 95% replica CIs")
        yield from _managed_figure(prefix + "-capacity", figure)
        if phase in {"phase1", "phase3"}:
            continue
        metadata = analysis.unit_metadata(units[0])
        if metadata.get("family") == "mixture" or "mixture" in str(metadata.get("environment", "")):
            figure, axes = plt.subplots(2, 3, figsize=(15, 9))
            for axis, field, x in zip(axes.flat,
                    ("standardized_nine_precision", "nine_recall", "nine_false_positive_rate", "standardized_nine_precision", "expected_nines", "realized_nines"),
                    ("observations", "observations", "observations", "expected_nines", "observations", "observations")):
                _curve(axis, units, field, x_field=x)
            _finish_figure(figure, label, "Rare-class efficiency; undefined precision stays missing; defined fractions in table")
            yield from _managed_figure(prefix + "-precision-recall", figure)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, cost in zip(axes, ("online_seconds", "charged_seconds")):
            for policy, members in _policy_groups(units).items():
                x = analysis.mean_interval([unit["summary"].get(cost) for unit in members])
                y = analysis.mean_interval([unit["summary"].get("nll_auc") for unit in members])
                if x["mean"] is not None and y["mean"] is not None:
                    axis.scatter(x["mean"], y["mean"], color=_color(policy), label=policy)
                    if y["ci95_low"] is not None:
                        axis.vlines(x["mean"], y["ci95_low"], y["ci95_high"], color=_color(policy))
                    if x["ci95_low"] is not None:
                        axis.hlines(y["mean"], x["ci95_low"], x["ci95_high"], color=_color(policy))
            axis.set(xlabel=cost.replace("_", " "), ylabel="Mean normalized current NLL-AUC")
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6)
            else:
                axis.text(.5, .5, MISSING, transform=axis.transAxes, ha="center")
        _finish_figure(figure, label, "Compute-quality comparison; 95% replica CIs; research-only costs excluded from online time")
        yield from _managed_figure(prefix + "-compute-quality", figure)
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        for column, target in enumerate(TARGETS):
            for axis, resource in zip(axes[:, column], ("observations", "online_seconds")):
                for policy_index, (policy, members) in enumerate(_policy_groups(units).items()):
                    offsets = np.linspace(-.12, .12, len(members)) if len(members) > 1 else [0.]
                    for unit, offset in zip(members, offsets):
                        outcome = _targets(unit, target)
                        if outcome[resource] is not None:
                            marker = "o" if outcome["reached"] else ("|" if outcome["status"] == "missing_metrics" else "x")
                            axis.scatter(outcome[resource], policy_index + offset, color=_color(policy),
                                         marker=marker, s=22,
                                         facecolors="none" if outcome["reached"] else None)
                axis.set_yticks(range(len(_policy_groups(units))), list(_policy_groups(units)), fontsize=7)
                axis.set_title(f"Fixed NLL <= {target:g}, two consecutive evaluations", fontsize=9)
                axis.set_xlabel(resource.replace("_", " "))
                limits = [row[resource] for unit in units for row in unit["metrics"]
                          if analysis._finite(row.get(resource)) is not None]
                axis.set_xlim(0, max(limits) * 1.04 if limits and max(limits) > 0 else 1)
                axis.grid(alpha=.2)
        _finish_figure(figure, label, "Resource to target: circles attained, x right-censored, vertical ticks missing metrics; all replicas retained")
        yield from _managed_figure(prefix + "-targets", figure)
        if any(unit["pr_curves"] for unit in units):
            figure, axis = plt.subplots(figsize=(7, 5))
            rows = analysis.pr_curve_summary(units)
            for policy in _policy_groups(units):
                points = [row for row in rows if str(row["policy"]) == policy and row["mean"] is not None]
                if points:
                    x = [row["recall"] for row in points]
                    axis.plot(x, [row["mean"] for row in points], color=_color(policy),
                              label=f"{policy} (n={min(row['n'] for row in points)}-{max(row['n'] for row in points)})")
                    axis.fill_between(x, [np.nan if row["ci95_low"] is None else row["ci95_low"] for row in points],
                                      [np.nan if row["ci95_high"] is None else row["ci95_high"] for row in points],
                                      color=_color(policy), alpha=.15)
                operating = [unit["metrics"][-1] for unit in _policy_groups(units)[policy] if unit["metrics"]]
                defined = [row for row in operating if analysis.metric_value(row, "nine_precision") is not None
                           and analysis.metric_value(row, "nine_recall") is not None]
                if defined:
                    axis.scatter(np.mean([row["nine_recall"] for row in defined]),
                                 np.mean([row["nine_precision"] for row in defined]),
                                 marker="D", s=28, color=_color(policy), label=f"{policy} final argmax")
            axis.set(xlabel="Recall", ylabel="Precision standardized to p_ref=.1", xlim=(0, 1), ylim=(0, 1))
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
            _finish_figure(figure, label, "Final threshold sweep; pointwise replica CI; interpolated scalar plot samples only")
            yield from _managed_figure(prefix + "-pr-sweep", figure)


def _recorded_interval(value: Any) -> str:
    if not isinstance(value, dict) or value.get("mean") is None:
        return MISSING
    low, high = value.get("ci95_low"), value.get("ci95_high")
    interval = "CI unavailable" if low is None or high is None else f"95% CI [{_display(low)}, {_display(high)}]"
    return f"{_display(value['mean'])}; {interval}; n={_display(value.get('n'))}"


def _sealed_report_rows(unit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Read an additional scalar closeout table without widening the analysis API."""
    path = Path(unit["path"]) / name
    if not (path.parent / "COMPLETED").is_file():
        raise analysis.ArtifactError(f"Unsealed closeout artifact: {path}")
    integrity = analysis._object(path.parent / "integrity.json")
    entry = integrity.get("files", integrity).get(name)
    if entry is None and not path.exists():
        return []
    digest = entry.get("sha256") if isinstance(entry, dict) else entry
    if digest is None or analysis._hash_file(path) != digest:
        raise analysis.ArtifactError(f"Corrupt or uncovered closeout artifact: {path}")
    rows = analysis._read_json(path)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise analysis.ArtifactError(f"Expected closeout scalar row list: {path}")
    return rows


def closeout_markdown(report: dict[str, Any]) -> str:
    parts = ["## Conclusions And Unresolved Claims",
             "No favorable schedule, successful-replica subset, or hindsight depth is selected here. "
             "The tables below reproduce sealed closeout estimates and their recorded uncertainty."]
    closeout = [unit for unit in report["units"] if unit["config"]["phase"] == "phase6"
                and unit["config"].get("kind") == "closeout"]
    if not closeout:
        return "\n\n".join([*parts, "Final integrity review and evidence-qualified conclusions are pending sealed Phase 6 units."])
    for unit in closeout:
        summary = unit["summary"]
        conclusions = summary.get("conclusions", [])
        if not isinstance(conclusions, list) or any(not isinstance(row, dict) for row in conclusions):
            raise analysis.ArtifactError(f"Invalid closeout conclusions: {unit['path']}")
        parts.append(_compact_scalars(summary, exclude=("phase", "schema_version", "effect_convention", "inference", "success_target")))
        parts.extend(["### Frozen Final Comparisons", table(
            ["Condition", "Policy", "Replicas", "NLL-AUC", "Early 25% NLL-AUC", "Retention change", "Online seconds", "Retention admissible"],
            [[row.get("environment"), row.get("policy"), row.get("replicas"),
              *(_recorded_interval(row.get(field)) for field in ("nll_auc", "early_25_nll_auc", "retention_change", "online_seconds")),
              row.get("retention_admissible_mean")] for row in conclusions])])
        effects = _sealed_report_rows(unit, "paired_effects.json")
        primary = [row for row in effects if row.get("metric") == "nll_auc"]
        parts.extend(["### Paired Primary Contrasts", table(
            ["Condition", "Treatment", "Comparator", "NLL-AUC effect", "Unpaired treatment", "Unpaired comparator"],
            [[row.get("environment"), row.get("treatment"), row.get("control"), _recorded_interval(row),
              row.get("unpaired_treatment"), row.get("unpaired_control")] for row in primary]),
            str(summary.get("effect_convention", "Treatment minus control; negative NLL differences favor treatment.")),
            str(summary.get("inference", "Uncertainty is as recorded in the sealed closeout artifacts."))])
        if not primary:
            parts.append("Paired primary contrasts are not recorded; none are reconstructed or inferred here.")
        targets = [[row.get("environment"), row.get("policy"), target, stats.get("attained"),
                    stats.get("replicas"), stats.get("censored"), stats.get("baseline_satisfied"),
                    stats.get("later_loss"), stats.get("missing_metrics")]
                   for row in conclusions for target, stats in row.get("quality_targets", {}).items()]
        if targets:
            parts.extend(["### Target Attainment Including Censoring", table(
                ["Condition", "Policy", "Target", "Attained", "All", "Censored", "Baseline satisfied", "Later loss", "Missing"], targets)])
        limitations = summary.get("limitations", [])
        if isinstance(limitations, list):
            parts.extend(["### Recorded Limitations", "\n".join(f"- {html.escape(item)}" for item in limitations if isinstance(item, str))])
        links = [_artifact_link(report, unit, name) for name in (
            "summary.json", "paired_effects.json", "factorial_effects.json", "metrics.json", "theory_note.md")
                 if (Path(unit["path"]) / name).is_file()]
        parts.append(_details("Full closeout evidence, secondary contrasts, and provenance", "\n\n".join(links)))
    return "\n\n".join(parts)


def lab_log_markdown(report: dict[str, Any]) -> str:
    log = report["lab_log"]
    parts = ["## Lab Log"]
    if log:
        rows = []
        for index, entry in enumerate(log):
            if isinstance(entry, dict):
                rows.append([index + 1, entry.get("time", entry.get("timestamp")),
                             html.escape(str(entry.get("message", MISSING))),
                             _compact_scalars(entry, exclude=("time", "timestamp", "message"))])
            else:
                rows.append([index + 1, None, html.escape(str(entry)), MISSING])
        parts.extend([table(["Entry", "Timestamp", "Decision, failure, or repair", "Scalar flags"], rows),
                      _raw_details("Full lab-log details (including nested decisions and repair records)", log,
                                   artifact_link=_artifact_link(report, {"path": report["root"]}, "lab_log.json"))])
    else:
        parts.append("No recorded decisions yet.")
    parts.append(closeout_markdown(report))
    return "\n\n".join(parts)


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str, outputs: list[dict[str, Any]], *, phase: str | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "metadata": {} if phase is None else {"plan14_phase": phase},
            "source": source, "outputs": outputs, "execution_count": None}


def _markdown_output(source: str) -> dict[str, Any]:
    return {"output_type": "display_data", "metadata": {},
            "data": {"text/markdown": source, "text/plain": source}}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _phase_figure_key(report: dict[str, Any], phase: str) -> str:
    units = [(unit["path"], unit["manifest"],
              analysis._hash_file(Path(unit["path"]) / "integrity.json"))
             for unit in report["units"] if unit["config"]["phase"] == phase]
    payload = {"units": sorted(units, key=lambda row: row[0]), "study": report["study"],
               "decisions": report.get("decisions", {}), "targets": TARGETS,
               "bootstrap": [analysis.BOOTSTRAP_SEED, analysis.BOOTSTRAP_SAMPLES],
               "render_sources": [analysis._hash_file(Path(path).resolve())
                                  for path in (__file__, analysis.__file__)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _exports_current(exports: dict[Path, tuple[int, ...]]) -> bool:
    try:
        return all(analysis._fingerprint(path) == fingerprint for path, fingerprint in exports.items())
    except OSError:
        return False


def _phase_figure_outputs(report: dict[str, Any], phase: str, export_root: Path, *,
                          force_fresh: bool = False) -> list[dict[str, Any]]:
    """Reuse only derived images, after refresh has validated every sealed unit."""
    export_root = export_root.resolve()
    slot = (str(Path(report["root"]).resolve()), phase, str(export_root))
    key = _phase_figure_key(report, phase)
    cached = _FIGURE_CACHE.get(slot)
    if not force_fresh and cached and cached["key"] == key and _exports_current(cached["exports"]):
        _FIGURE_CACHE.move_to_end(slot)
        return cached["outputs"]
    _FIGURE_CACHE.pop(slot, None)
    outputs = []
    exports = {}
    figures = phase_figures(report, phase)
    try:
        for name, figure in figures:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight", metadata={"Software": "Plan14 artifact report"})
            png = buffer.getvalue()
            png_path = export_root / f"{phase}-{name}.png"
            _atomic_write(png_path, png)
            exports[png_path] = analysis._fingerprint(png_path)
            svg = io.BytesIO()
            with matplotlib.rc_context({"svg.hashsalt": "plan14"}):
                figure.savefig(svg, format="svg", bbox_inches="tight", metadata={"Date": None})
            svg_path = export_root / f"{phase}-{name}.svg"
            _atomic_write(svg_path, svg.getvalue())
            exports[svg_path] = analysis._fingerprint(svg_path)
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png).decode("ascii")}})
    finally:
        figures.close()
    # Retain one generation per slot, with bounded serialized output and no figures.
    size = len(json.dumps(outputs)) + sum(len(str(path)) + 128 for path in exports) + 256
    if size <= FIGURE_CACHE_MAX_BYTES and FIGURE_CACHE_MAX_ENTRIES > 0:
        while _FIGURE_CACHE and (len(_FIGURE_CACHE) >= FIGURE_CACHE_MAX_ENTRIES
                or sum(entry["bytes"] for entry in _FIGURE_CACHE.values()) + size > FIGURE_CACHE_MAX_BYTES):
            _FIGURE_CACHE.popitem(last=False)
        _FIGURE_CACHE[slot] = {"key": key, "outputs": outputs, "exports": exports, "bytes": size}
    return outputs


def refresh(root: Path, notebook_path: Path, *, force_fresh: bool = False) -> Path:
    """Validate first, pre-render via Agg, and atomically replace only the report.

    Figure exports are derived report artifacts, not immutable scientific units.
    Unchanged phases reuse bounded in-process outputs; force_fresh bypasses both caches.
    A corrupt unit aborts refresh before the previous notebook is touched.
    """
    report = analysis.load_report(root, force_fresh=force_fresh)
    root, notebook_path = Path(root).resolve(), Path(notebook_path).resolve()
    report["notebook_dir"] = str(notebook_path.parent)
    if any((parent / "COMPLETED").is_file() for parent in notebook_path.parents):
        raise analysis.ArtifactError(f"Report destination is inside an immutable completed unit: {notebook_path}")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        root_expression = f"REPO_ROOT / {str(root.relative_to(repo_root))!r}"
    except ValueError:
        root_expression = f"Path({str(root)!r})"
    setup = ("from pathlib import Path\nimport sys\n"
             f"candidates = [Path.cwd(), *Path.cwd().parents, Path({str(repo_root)!r})]\n"
             "REPO_ROOT = next(path for path in candidates if (path / 'mnist_experiment/akaike_score_gate/analysis.py').is_file())\n"
             "if str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n"
             "from IPython.display import Markdown, display\n"
             "from mnist_experiment.akaike_score_gate import analysis, refresh_notebook as report_view\n"
             "import matplotlib.pyplot as plt\n"
             f"ROOT = {root_expression}\nreport = analysis.load_report(ROOT)\n"
             f"report['notebook_dir'] = {str(notebook_path.parent)!r}")
    cells = [_markdown_cell("# Plan 14: Akaike Score Gating\n\nProgressive artifact-only research lab notebook. "
                            "Sealed evidence only; no training, downloads, resumption, or repair."),
             _code_cell(setup, []),
             _code_cell("display(Markdown(report_view.status_markdown(report)))",
                        [_markdown_output(status_markdown(report))]), _markdown_cell(GLOSSARY)]
    export_root = root / "report" / "figures"
    for phase in PHASE_ORDER:
        cells.append(_markdown_cell("## " + PHASE_LABELS[phase]))
        cells.append(_code_cell(f"display(Markdown(report_view.phase_markdown(report, {phase!r})))",
                                [_markdown_output(phase_markdown(report, phase))], phase=phase))
        outputs = _phase_figure_outputs(report, phase, export_root, force_fresh=force_fresh)
        cells.append(_code_cell(f"figures = report_view.phase_figures(report, {phase!r})\n"
                                "for name, figure in figures:\n    display(figure)\n    plt.close(figure)",
                                outputs, phase=phase))
    cells.append(_code_cell("display(Markdown(report_view.lab_log_markdown(report)))",
                            [_markdown_output(lab_log_markdown(report))]))
    count = 0
    for index, cell in enumerate(cells):
        cell["id"] = f"plan14-{index:03d}"
        if cell["cell_type"] == "code":
            count += 1
            cell["execution_count"] = count
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "plan14_report_schema_version": 1}}
    _atomic_write(notebook_path, (json.dumps(notebook, indent=1, allow_nan=False) + "\n").encode())
    return notebook_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--force-fresh", action="store_true", help="Bypass SHA256 and figure caches (final integrity audit)")
    args = parser.parse_args(argv)
    print(refresh(args.output_root, args.notebook, force_fresh=args.force_fresh))


if __name__ == "__main__":
    main()
