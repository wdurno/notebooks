"""Artifact-only Phase 5C reporting; inference requires the full frozen cohort.

``collect(root)`` returns ledger, status, units, completed, and expected.
``analyze(units, ledger)`` is deterministic and returns JSON-safe results; its
``primary`` is the mixed-stream control-minus-FD acquisition contrast.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import nbformat

from .artifacts import ArtifactError, iter_units
from .config import Treatment
from .drip_runner import trajectory_id
from .metrics import paired_summary
from .report import _plot_cell, _table
from src.seeding import derive_component_seed


PROTOCOL = "plan15-drip-confirmation-v1"
FAMILY = "drip_confirmation"
LEDGER = "drip-confirmation-ledger-v1"
RESULTS = "drip-confirmation-results-v1"
TAG = "plan15_phase5c"
ARMS = (("mixed", "replay_ewc"), ("mixed", "finite_difference"),
        ("base", "replay_ewc"), ("base", "finite_difference"))
CAVEAT = ("The sign-flip test assumes exchangeable signs under the null; it is not "
          "distribution-free for a mean-zero null. Sample skewness is descriptive, "
          "and twelve replicas cannot establish symmetry. Bootstrap intervals are "
          "nominal replica-level 95% intervals; secondary contrasts are descriptive. "
          "A nonsignificant difference does not establish noninferiority or memory transfer.")


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ArtifactError(f"cannot read confirmation artifact: {path}") from error


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactError(f"{label}: expected finite number")
    return float(value)


def _planned(ledger):
    if (not isinstance(ledger, dict) or ledger.get("schema_version") != 1
            or ledger.get("protocol") != PROTOCOL or ledger.get("family") != FAMILY):
        raise ArtifactError("incompatible confirmation ledger schema/protocol/family")
    replicas, target = ledger.get("replicas"), ledger.get("event_target")
    if (type(replicas) is not int or type(target) is not int
            or (replicas, target) not in ((12, 100), (2, 2))):
        raise ArtifactError("confirmation ledger requires 12 x 100 production or 2 x 2 smoke")
    try:
        baseline, treatment = (Treatment(**ledger[key]) for key in ("baseline", "treatment"))
    except (TypeError, ValueError, KeyError) as error:
        raise ArtifactError("invalid confirmation treatment mapping") from error
    selected = _number(ledger.get("selected_strength"), "selected strength")
    if (baseline.condition != "replay_ewc" or baseline.strength != 0
            or treatment.condition != "finite_difference" or selected <= 0
            or treatment.strength != selected
            or (target == 100 and not math.isclose(selected, 160.89, rel_tol=1e-12))):
        raise ArtifactError("confirmation treatments differ from frozen FD selection")
    if (type(ledger.get("bootstrap_samples")) is not int or ledger["bootstrap_samples"] < 1
            or type(ledger.get("inference_seed")) is not int
            or not 0 <= ledger["inference_seed"] < 2**63
            or ledger.get("alpha") != .05):
        raise ArtifactError("invalid frozen confirmation inference settings")
    return {trajectory_id(FAMILY, replica, stream, choice):
            dict(replica=replica, stream=stream, condition=choice.condition,
                 treatment=choice.mapping(), treatment_id=choice.key)
            for replica in range(replicas) for stream in ("mixed", "base")
            for choice in (baseline, treatment)}


def _validate_identity(unit, expected, ledger):
    meta = unit.get("metadata")
    if not isinstance(meta, dict):
        raise ArtifactError("confirmation trajectory metadata must be an object")
    try:
        name = trajectory_id(FAMILY, meta["replica"], meta["stream"],
                             Treatment(**meta["treatment"]))
    except (KeyError, ValueError, TypeError) as error:
        raise ArtifactError("invalid confirmation trajectory identity") from error
    if (name not in expected or unit.get("unit_id", name) != name
            or unit.get("id", name) != name or type(meta.get("replica")) is not int
            or meta.get("schema_version") != 1 or meta.get("kind") != "drip_trajectory"
            or meta.get("phase") != "5C" or meta.get("protocol") != PROTOCOL
            or meta.get("family") != FAMILY or meta.get("event_target") != ledger["event_target"]
            or any(meta.get(key) != value for key, value in expected[name].items())):
        raise ArtifactError(f"unplanned or incompatible confirmation trajectory: {name}")
    return name


def _validate_payload(unit, ledger, name):
    summary, rows = unit.get("summary"), unit.get("rows")
    meta, target = unit["metadata"], ledger["event_target"]
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise ArtifactError(f"incompatible confirmation summary: {name}")
    for key, value in dict(unit_id=name, replica=meta["replica"], stream=meta["stream"],
                           condition=meta["condition"], strength=meta["treatment"]["strength"],
                           event_target=target).items():
        if summary.get(key) != value:
            raise ArtifactError(f"confirmation summary {key} mismatch: {name}")
    if (not isinstance(rows, list) or len(rows) != target + 1
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("paired_rotated_event") for row in rows] != list(range(target + 1))):
        raise ArtifactError(f"confirmation event grid incomplete or duplicated: {name}")
    for row in rows:
        for key in ("rotated_nll", "base_nll", "rotated_accuracy", "base_accuracy"):
            value = _number(row.get(key), f"{name}/{key}")
            if value < 0 or (key.endswith("accuracy") and value > 1):
                raise ArtifactError(f"invalid confirmation predictive metric: {name}/{key}")
        actual = row.get("actual_rotated_events")
        if actual != (row["paired_rotated_event"] if meta["stream"] == "mixed" else 0):
            raise ArtifactError(f"confirmation actual event count mismatch: {name}")
    for domain in ("rotated", "base"):
        average = statistics.fmean(row[f"{domain}_nll"] for row in rows[1:])
        if not math.isclose(_number(summary.get(f"{domain}_auc"), name), average,
                            rel_tol=1e-7, abs_tol=1e-8):
            raise ArtifactError(f"confirmation AUC disagrees with event metrics: {name}")


def collect(root: Path | str) -> dict:
    """Load verified artifacts only; incomplete trajectories never supply curves."""
    root = Path(root)
    records = list(iter_units(root))
    ledger_record = next((row for row in records if row["unit_id"] == LEDGER), None)
    if ledger_record is None or ledger_record["status"] != "completed":
        raise ArtifactError("confirmation ledger is missing or incomplete")
    ledger = _read(ledger_record["path"] / "result.json")
    expected = _planned(ledger)
    units = []
    for record in records:
        if record["metadata"].get("kind") != "drip_trajectory":
            continue
        unit = dict(unit_id=record["unit_id"], metadata=record["metadata"],
                    completed=record["status"] == "completed", rows=[], summary=None, fit=[])
        name = _validate_identity(unit, expected, ledger)
        if unit["completed"]:
            path = record["path"]
            if _read(path / "metadata.json") != unit["metadata"]:
                raise ArtifactError(f"confirmation metadata disagrees with manifest: {name}")
            unit.update(rows=_read(path / "metrics.json"), summary=_read(path / "summary.json"),
                        fit=_read(path / "fit.json"))
            if not isinstance(unit["fit"], list):
                raise ArtifactError(f"incompatible confirmation fit records: {name}")
            _validate_payload(unit, ledger, name)
        units.append(unit)
    status = _read(root / "status.json") if (root / "status.json").exists() else {}
    if not isinstance(status, dict):
        raise ArtifactError("confirmation status must be an object")
    return dict(ledger=ledger, status=status, units=units, expected=len(expected),
                completed=sum(unit["completed"] for unit in units))


def _exact_sign_flip(differences):
    observed = math.fsum(differences)
    # Enumerate all 2^n sign assignments, retaining ties in the upper tail.
    tolerance = 1e-12 * max(math.fsum(abs(value) for value in differences), 1e-15)
    count = sum(math.fsum(sign * value for sign, value in zip(signs, differences))
                >= observed - tolerance
                for signs in itertools.product((-1, 1), repeat=len(differences)))
    return count / (2 ** len(differences))


def _effect(values, inferential, seed, resamples):
    differences = list(values.values())
    mean = statistics.fmean(differences) if differences else None
    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.
    variance = statistics.fmean((value - mean) ** 2 for value in differences) if differences else 0.
    skew = (statistics.fmean((value - mean) ** 3 for value in differences) / variance ** 1.5
            if variance else None)
    result = dict(n_pairs=len(values), mean_gain=mean, ci=None,
                  paired_values=[dict(replica=replica, gain=gain) for replica, gain in values.items()],
                  favorable_replicas=sum(value > 0 for value in differences),
                  standardized_effect=mean / deviation if deviation else None,
                  sample_skewness=skew, asymmetry_warning=bool(skew is not None and abs(skew) > 1),
                  p_value=None)
    if inferential:
        paired = paired_summary(values, dict.fromkeys(values, 0.), seed=seed, resamples=resamples)
        result["ci"] = paired["ci"]
    result.update(seed=seed, resamples=resamples)
    return result


def analyze(units: list[dict], ledger: dict) -> dict:
    """Analyze only sealed, complete four-arm replicas from this frozen family.

    No confidence interval or p-value is released before the whole production
    cohort is sealed. Complete smoke cohorts remain acceptance evidence only.
    """
    expected = _planned(ledger)
    index, seen = {}, set()
    for unit in units:
        name = _validate_identity(unit, expected, ledger)
        if name in seen:
            raise ArtifactError(f"duplicate confirmation trajectory: {name}")
        seen.add(name)
        if unit.get("completed"):
            _validate_payload(unit, ledger, name)
            meta = unit["metadata"]
            index[(meta["replica"], meta["stream"], meta["condition"])] = unit
    paired = [replica for replica in range(ledger["replicas"])
              if all((replica, *arm) in index for arm in ARMS)]
    complete = len(index) == len(expected)
    smoke = ledger["event_target"] == 2
    inference_ready = complete and not smoke
    differences = {key: {} for key in (
        "mixed_gain", "base_only_gain", "interaction", "base_retention_harm")}
    for replica in paired:
        mixed_control, mixed_fd, base_control, base_fd = (
            index[(replica, *arm)]["summary"] for arm in ARMS)
        gain = mixed_control["rotated_auc"] - mixed_fd["rotated_auc"]
        base_gain = base_control["rotated_auc"] - base_fd["rotated_auc"]
        for key, value in dict(mixed_gain=gain, base_only_gain=base_gain,
                               interaction=gain - base_gain,
                               base_retention_harm=mixed_fd["base_auc"] - mixed_control["base_auc"]).items():
            differences[key][replica] = value
    comparisons = {key: _effect(values, inference_ready,
                               derive_component_seed(ledger["inference_seed"], f"bootstrap:{key}"),
                               ledger["bootstrap_samples"])
                   for key, values in differences.items()}
    primary = comparisons["mixed_gain"]
    if inference_ready:
        primary["p_value"] = _exact_sign_flip(list(differences["mixed_gain"].values()))
    arm_summaries = []
    for arm in ARMS:
        cohort = [index[(replica, *arm)] for replica in paired]
        records = [unit["summary"] for unit in cohort]
        row = dict(stream=arm[0], condition=arm[1], n_pairs=len(paired))
        for key in ("rotated_auc", "base_auc", "active_seconds", "closure_calls",
                    "final_rotated_nll", "final_base_nll",
                    "final_rotated_accuracy", "final_base_accuracy", "exposure"):
            values = [_number(record[key], key) for record in records if record.get(key) is not None]
            row[key] = statistics.fmean(values) if values else None
        prequential = []
        for unit in cohort:
            count, total, previous = 0, 0., 0
            for fit in unit.get("fit", []):
                event = fit.get("paired_rotated_events", previous)
                mass = event - previous
                previous = event
                if mass and fit.get("rotated_prequential_nll") is not None:
                    total += mass * _number(fit["rotated_prequential_nll"], "prequential NLL")
                    count += mass
            if count:
                prequential.append(total / count)
        row["rotated_prequential_nll"] = statistics.fmean(prequential) if prequential else None
        row["per_class_events"] = {
            str(digit): statistics.fmean(record["per_class_events"][str(digit)] for record in records)
            for digit in range(10) if records and all(
                str(digit) in record.get("per_class_events", {}) for record in records)}
        arm_summaries.append(row)
    sealed = [unit["summary"] for unit in index.values()]
    health = dict(sealed_trajectories=len(sealed),
                  solver_failures=sum(row.get("solver_failures", 0) for row in sealed),
                  nonfinite_trajectories=sum(bool(row.get("nonfinite")) for row in sealed),
                  under_solved_updates=sum(row.get("under_solved", 0) for row in sealed))
    health["status"] = ("failures_recorded" if health["solver_failures"] or health["nonfinite_trajectories"]
                        else "no_recorded_solver_failures")
    return dict(schema_version=1, protocol=PROTOCOL, family=FAMILY, result_id=RESULTS,
                complete=complete, smoke=smoke, inference_ready=inference_ready,
                completed=len(index), expected=len(expected), n_pairs=len(paired),
                planned_replicas=ledger["replicas"], paired_replicas=paired,
                event_target=ledger["event_target"], comparisons=comparisons, primary=primary,
                arm_summaries=arm_summaries, inference_caveat=CAVEAT, alpha=ledger["alpha"],
                source_lineage=ledger.get("source", {}), health=health,
                selected_strength=ledger["selected_strength"],
                bootstrap_samples=ledger["bootstrap_samples"], inference_seed=ledger["inference_seed"])


def _tag(cell):
    cell.metadata["tags"] = [TAG]
    return cell


def _format(value, signed=False):
    return "pending" if value is None else format(value, "+.4f" if signed else ".4f")


def _predictive_plot(units, paired, target, accuracy=False):
    if not paired:
        return None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    field = "accuracy" if accuracy else "nll"
    for axis, domain, title in zip(axes, ("rotated", "base"),
                                   ("New domain: 30 degrees", "Base retention: 0 degrees")):
        for stream, condition in ARMS:
            cohort = [unit for unit in units if unit["completed"]
                      and unit["metadata"]["replica"] in paired
                      and unit["metadata"]["stream"] == stream
                      and unit["metadata"]["condition"] == condition]
            curve = [statistics.fmean(unit["rows"][event][f"{domain}_{field}"] for unit in cohort)
                     for event in range(target + 1)]
            if accuracy:
                curve = [100 * value for value in curve]
            label = f"{'Mixed' if stream == 'mixed' else 'Base-only'} / " + (
                "FD KL" if condition == "finite_difference" else "replay/EWC")
            axis.plot(range(target + 1), curve, label=label,
                      color="#147d73" if condition == "finite_difference" else "#a43b68",
                      linestyle="-" if stream == "mixed" else "--", linewidth=1.7)
        axis.set(title=title, xlim=(0, target), xlabel="Paired rotated-event index",
                 ylabel="Accuracy (%)" if accuracy else "Test NLL")
        if accuracy:
            axis.set_ylim(0, 100)
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    caption = (f"Phase 5C {'accuracy' if accuracy else 'NLL'}: {len(paired)} complete four-arm replicas. "
               "Mixed-stream index counts actual rotated images; base-only uses its paired clock "
               "and observes zero rotated training images. Curves have a fixed replica cohort.")
    return _tag(_plot_cell(figure, caption))


def section(root: Path):
    data = collect(root)
    ledger, status = data["ledger"], data["status"]
    result = analyze(data["units"], ledger)
    target = ledger["event_target"]
    headline = ("## Phase 5C: Fresh Rare-Domain Confirmation\n\n"
                f"**Progress: {result['completed']}/{result['expected']} sealed trajectories; "
                f"paired n = {result['n_pairs']}/{ledger['replicas']}.** "
                f"Status: {status.get('status', 'pending')}; "
                f"active compute {status.get('active_seconds', 0) / 3600:.2f} h. "
                f"Current job: {status.get('current_job') or 'none'}.\n\n"
                f"The acquisition score is **average new-domain test NLL after each of the first "
                f"{target} rotated training images**; lower is better. Event zero is excluded. "
                "Multiple rotated images in one batch share its post-update evaluation. "
                "All four conditions share a 32-image FIFO replay buffer and eviction-only EWC; "
                f"the selected FD coefficient is {ledger['treatment']['strength']:.4g}. "
                "Base-only conditions train on unrotated versions of the same images: their "
                "event index is a paired reporting clock, with zero actual rotated exposures. "
                "Fresh replicas and a separate test-image pool exclude development evidence.\n\n"
                "Curves and descriptive comparisons use sealed, complete four-arm replicas. "
                "Final inference is released once the entire production cohort is sealed.")
    if result["smoke"]:
        headline += " **SMOKE ACCEPTANCE ONLY: no scientific inference.**"
    health = result["health"]
    headline += (f"\n\n**Solver health:** {health['solver_failures']} failures; "
                 f"{health['nonfinite_trajectories']} trajectories flagged nonfinite; "
                 f"{health['under_solved_updates']} under-solved updates. "
                 "Under-solved updates are diagnostics, not solver failures; no sealed trajectory is excluded.")
    changes = status.get("changes", [])
    if changes:
        headline += "\n\n**Recorded changes:** " + json.dumps(changes[-3:], ensure_ascii=True)
    cells = [_tag(nbformat.v4.new_markdown_cell(headline))]
    rows = []
    for key, label in (("mixed_gain", "Mixed acquisition gain"),
                       ("base_only_gain", "Base-only gain on rotated holdout"),
                       ("interaction", "Mixed gain minus base-only gain"),
                       ("base_retention_harm", "Mixed base-retention harm")):
        effect = result["comparisons"][key]
        interval = (f"[{effect['ci'][0]:+.4f}, {effect['ci'][1]:+.4f}]"
                    if effect["ci"] is not None else "pending full cohort")
        rows.append([label, _format(effect["mean_gain"], True), interval])
    readout = ("### Paired effects\n\n" + _table(["Contrast", "Mean NLL-AUC", "95% CI"], rows)
               + "\n\nPositive acquisition gains favor FD; positive base-retention harm favors "
                 "replay. The retention tolerance is 0.01 NLL, a descriptive guardrail.")
    primary = result["primary"]
    if result["inference_ready"]:
        readout += (f"\n\n**Primary one-sided sign-flip p = {primary['p_value']:.6g}** "
                    f"(all {2 ** result['n_pairs']} sign assignments enumerated; "
                    f"alpha = {ledger['alpha']}). "
                    f"Favorable replicas: {primary['favorable_replicas']}/{result['n_pairs']}. "
                    + ("**Asymmetry warning:** absolute sample skewness exceeds one. "
                       if primary["asymmetry_warning"] else ""))
    else:
        readout += "\n\n**Interim descriptive results only; no p-value or uncertainty interval.**"
    readout += "\n\n" + CAVEAT + " Individual gains and detailed effect diagnostics remain in result JSON."
    cells.append(_tag(nbformat.v4.new_markdown_cell(readout)))
    for accuracy in (False, True):
        cell = _predictive_plot(data["units"], result["paired_replicas"], target, accuracy)
        if cell is not None:
            cells.append(cell)
    if result["n_pairs"]:
        rows = []
        for arm in result["arm_summaries"]:
            label = f"{arm['stream']} / {arm['condition']}"
            rows.append([label, _format(arm["rotated_prequential_nll"]),
                         _format(arm["final_rotated_accuracy"]), _format(arm["final_base_accuracy"]),
                         _format(arm["active_seconds"] / 60 if arm["active_seconds"] is not None else None),
                         _format(arm["closure_calls"])])
        counts = [unit["summary"].get("per_class_events", {}) for unit in data["units"]
                  if unit["completed"] and unit["metadata"]["replica"] in result["paired_replicas"]
                  and unit["metadata"]["stream"] == "mixed"
                  and unit["metadata"]["condition"] == "replay_ewc"]
        counts = {str(digit): statistics.fmean(row[str(digit)] for row in counts)
                  for digit in range(10) if counts and all(str(digit) in row for row in counts)}
        cells.append(_tag(nbformat.v4.new_markdown_cell(
            "### Prediction and compute audit\n\n" + _table(
                ["Condition", "Rotated prequential NLL", "Final rotated acc (0-1)",
                 "Final base acc (0-1)", "Online min", "Closures"], rows)
            + "\n\nPrequential NLL predicts incoming rotated images before their updates; "
              "base-only predictions of those images are audit-only. Means use the same paired "
              "replicas. Per-digit rotated training counts (mean): " + json.dumps(counts)
            + ". Online time includes fitting, evaluation and eviction work, excluding initialization.")))
    return cells


def build_notebook(base_notebook: Path, root: Path):
    """Preserve all earlier sections, replace only this phase's tagged cells."""
    notebook = nbformat.read(base_notebook, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if TAG not in cell.metadata.get("tags", [])]
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and ("Phase 5B" in cell.source
                or "plan15_phase5b" in cell.metadata.get("tags", [])):
            cell.source = cell.source.replace("Phase 5C has not started.", "").rstrip()
    notebook.cells[1:1] = section(Path(root))
    nbformat.validate(notebook)
    return notebook
