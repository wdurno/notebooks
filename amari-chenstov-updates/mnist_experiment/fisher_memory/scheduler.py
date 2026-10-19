"""Autonomous, restartable Plan 15 phases 1--4, with sealed selection decisions.

Integration: Scheduler(runner).run() uses runner.cfg, runner.store,
runner.run_trajectory(family, phase, env, replica, treatment), and
runner.calibrate(family, env, replica, baseline_treatment). Explicit callbacks
can instead be supplied to the constructor for tests or alternate runners.

Trajectory summaries use current_auc, historical_auc (None when unavailable),
active_seconds, fit_seconds, and nonfinite. Optional acquisition_ok,
retention_ok, solver_ok, nine_precision/nine_recall, and mechanism fields are
retained. Missing health flags mean unresolved, never success. Historical AUC
is independent historical-holdout post-eviction AUC, not trained-cohort AUC.

Calibration returns {"methods": {name: {...}}} for jvp, finite_difference,
directional, and distillation. A valid method supplies valid=True,
unit_gradient_ratio (>0, at coefficient one), gradient_ratio_ltm,
gradient_ratio_stm, probes, and fd_step. Invalid methods supply valid=False
and reason. Detailed numerical checks and timings can accompany these fields.
probes/fd_step default to the configured values. Calibration must use the
shared young, pre-eviction, and post-eviction checkpoints, not test outcomes.

All scheduler units have metadata.kind scheduler_decision or scheduler_receipt;
report loaders must distinguish them from trajectory units. Scheduler receipts
contain only small JSON summaries, never models, score banks, or images.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .artifacts import ArtifactError
from .config import Config, METHODS, Treatment, content_hash
from .control import ResourceLimit, StopRequested, utc_now
from .metrics import paired_summary


SCHEDULER_SCHEMA = 1
DEVELOPMENT_FAMILIES = ("policy_development", "pilot_screen", "pilot_validation")
CONFIRMATION_FAMILY = "confirmation"
MECHANISM_FIELDS = ("mean_stm_removal_harm", "mean_ltm_only_nll", "mean_future_stm_swap_harm")


class SchedulerContractError(ValueError):
    """A callback omitted information needed for an interpretable comparison."""


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_summary(value: Mapping) -> dict:
    result = dict(value)
    if not isinstance(result.get("nonfinite"), bool):
        raise SchedulerContractError("trajectory summary requires boolean nonfinite")
    for field in ("active_seconds", "fit_seconds"):
        if not _finite(result.get(field)) or result[field] < 0:
            raise SchedulerContractError(f"trajectory summary requires nonnegative {field}")
    for field in ("current_auc", "historical_auc"):
        if field not in result:
            raise SchedulerContractError(f"trajectory summary requires explicit {field}, including missingness")
        metric = result.get(field)
        if metric is None and (field == "historical_auc" or result["nonfinite"]):
            continue
        if not _finite(metric):
            raise SchedulerContractError(f"invalid trajectory summary {field}")
    for field in ("acquisition_ok", "retention_ok", "solver_ok"):
        if result.get(field) is not None and not isinstance(result[field], bool):
            raise SchedulerContractError(f"{field} must be boolean or unresolved")
    for field in ("nine_precision", "nine_recall"):
        value = result.get(field)
        if value is not None and (not _finite(value) or not 0 <= value <= 1):
            raise SchedulerContractError(f"invalid {field}")
    mechanism = result.get("mechanism")
    if mechanism is not None:
        if not isinstance(mechanism, Mapping):
            raise SchedulerContractError("mechanism must be a scalar diagnostic mapping")
        for field in MECHANISM_FIELDS:
            if mechanism.get(field) is not None and not _finite(mechanism[field]):
                raise SchedulerContractError(f"invalid mechanism {field}")
    # No nonfinite JSON sentinel may be hidden in an optional diagnostic.
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SchedulerContractError("summary must be finite JSON-compatible data") from error
    return result


def validate_calibration(value: Mapping, cfg: Config) -> dict:
    result = dict(value)
    methods = result.get("methods")
    if not isinstance(methods, Mapping):
        raise SchedulerContractError("calibration requires a methods mapping")
    resolved = {}
    for name in (*METHODS, "distillation"):
        item = dict(methods.get(name, {}))
        if not isinstance(item.get("valid"), bool):
            raise SchedulerContractError(f"calibration missing validity for {name}")
        if item["valid"]:
            if not _finite(item.get("unit_gradient_ratio")) or item["unit_gradient_ratio"] <= 0:
                raise SchedulerContractError(f"{name} needs a positive nondegenerate unit gradient ratio")
            for block in ("ltm", "stm"):
                field = f"gradient_ratio_{block}"
                if item.get(field) is not None and (not _finite(item[field]) or item[field] < 0):
                    raise SchedulerContractError(f"invalid {name} {field}")
            probes = item.setdefault("probes", cfg.probes)
            h = item.setdefault("fd_step", cfg.fd_step)
            if isinstance(probes, bool) or probes not in (1, 2, 4, 8):
                raise SchedulerContractError("probe calibration must use K in {1,2,4,8}")
            if not _finite(h) or h <= 0:
                raise SchedulerContractError("finite-difference step must be positive")
        elif not isinstance(item.get("reason"), str) or not item["reason"]:
            raise SchedulerContractError(f"invalid {name} must disclose its reason")
        for field in ("relative_error", "gradient_error", "seconds"):
            if item.get(field) is not None and (not _finite(item[field]) or item[field] < 0):
                raise SchedulerContractError(f"invalid {name} {field}")
        resolved[name] = item
    result["methods"] = resolved
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SchedulerContractError("calibration must be finite JSON-compatible data") from error
    return result


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _se(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def select_candidate(baseline: Sequence[dict], candidates: Mapping[str, Sequence[dict]],
                     *, retention_margin: float) -> dict:
    """Predeclared NLL selection, one-paired-SE cost tie, no significance test.

    Complete paired replicas are required per candidate. A missing historical
    endpoint is not retention feasibility. Infeasible but numerically valid
    candidates remain eligible only as explicitly labeled diagnostic fallbacks.
    PR and mechanism diagnostics are preserved, not collapsed into a new score.
    """
    if not baseline or not _finite(retention_margin) or retention_margin < 0:
        raise SchedulerContractError("selection needs paired controls and a nonnegative margin")
    table = []
    for key, rows in candidates.items():
        if len(rows) != len(baseline):
            raise SchedulerContractError("candidate/control replica counts differ")
        valid = all(not row["nonfinite"] and _finite(row.get("current_auc"))
                    and row.get("solver_ok") is not False for row in (*baseline, *rows))
        if not valid:
            table.append({"key": key, "valid": False, "reason": "nonfinite_or_unresolved_solver"})
            continue
        gains = [control["current_auc"] - row["current_auc"]
                 for control, row in zip(baseline, rows, strict=True)]
        history = all(_finite(row.get("historical_auc")) for row in (*baseline, *rows))
        harms = ([row["historical_auc"] - control["historical_auc"]
                  for control, row in zip(baseline, rows, strict=True)] if history else None)
        harm = _mean(harms) if harms is not None else None
        table.append({"key": key, "valid": True, "current_gain": _mean(gains),
                      "current_gain_se": _se(gains), "paired_current_gains": gains,
                      "current_auc": _mean([r["current_auc"] for r in rows]),
                      "historical_auc": _mean([r["historical_auc"] for r in rows]) if history else None,
                      "retention_harm": harm, "retention_feasible": harm is not None and harm <= retention_margin + 1e-12,
                      "active_seconds": _mean([r["active_seconds"] for r in rows]),
                      "fit_seconds": _mean([r["fit_seconds"] for r in rows]),
                      "nine_precision": _optional_mean(rows, "nine_precision"),
                      "nine_recall": _optional_mean(rows, "nine_recall"),
                      "mechanism": [r.get("mechanism") for r in rows]})
    valid = [row for row in table if row["valid"]]
    feasible = [row for row in valid if row["retention_feasible"]]
    if not valid:
        return {"selected": None, "classification": "no_valid_candidate", "table": table,
                "pareto_keys": [], "retention_margin": retention_margin}
    if feasible:
        best = max(feasible, key=lambda row: (row["current_gain"], -row["active_seconds"], row["key"]))
        ties = []
        for row in feasible:
            differences = [a - b for a, b in zip(best["paired_current_gains"], row["paired_current_gains"], strict=True)]
            uncertainty = _se(differences)
            if _mean(differences) <= (0. if uncertainty is None else uncertainty) + 1e-12:
                ties.append(row)
        winner = min(ties, key=lambda row: (row["active_seconds"], -row["current_gain"], row["key"]))
        classification = "favorable_development_candidate" if winner["current_gain"] > 0 else "diagnostic_no_acquisition_gain"
    else:
        known = [row for row in valid if row["retention_harm"] is not None]
        pool = known or valid
        winner = min(pool, key=lambda row: (
            row["retention_harm"] if known else 0., -row["current_gain"], row["active_seconds"], row["key"]))
        classification = "diagnostic_retention_margin_failed" if known else "diagnostic_retention_unavailable"
    pareto = []
    for row in valid:
        if row["historical_auc"] is None:
            continue
        def dominates(other):
            if other is row or other["historical_auc"] is None:
                return False
            dimensions = ("current_auc", "historical_auc", "active_seconds")
            return (all(other[k] <= row[k] for k in dimensions)
                    and any(other[k] < row[k] for k in dimensions))
        if not any(dominates(other) for other in valid):
            pareto.append(row["key"])
    return {"selected": winner["key"], "classification": classification, "table": table,
            "pareto_keys": pareto, "retention_margin": retention_margin,
            "tie_rule": "one paired standard error; then lower active wall time",
            "primary_sign": "positive current gain favors treatment; positive retention harm is worse"}


def _optional_mean(rows, field):
    values = [row.get(field) for row in rows]
    return _mean(values) if all(_finite(value) for value in values) else None


class Scheduler:
    def __init__(self, runner=None, *, cfg: Config | None = None, store=None,
                 run_trajectory: Callable | None = None, calibrate: Callable | None = None,
                 refresh: Callable | None = None, closeout: Callable | None = None,
                 check_stop: Callable | None = None, progress=None,
                 max_units: int | None = None):
        self.cfg = cfg if cfg is not None else runner.cfg
        self.store = store if store is not None else runner.store
        self.run_trajectory = run_trajectory or runner.run_trajectory
        self.calibrate = calibrate or runner.calibrate
        self.refresh = refresh
        self.closeout = closeout
        self.check_stop = check_stop
        self.progress = progress
        if max_units is not None and (isinstance(max_units, bool) or not isinstance(max_units, int) or max_units < 0):
            raise ValueError("max_units must be a nonnegative integer")
        self.max_units, self.completed_this_invocation = max_units, 0
        self.cfg.validate()

    def _status(self, **fields):
        if self.progress is not None:
            self.progress.state.update(fields)
            self.progress(force=False)

    def _check(self):
        self.store.check()
        if self.check_stop is not None and self.check_stop():
            raise StopRequested("scheduler stopped at a work-unit boundary")

    def _audit(self):
        """Bounded Store checks include checkpoint dependencies, not just receipts."""
        started = time.perf_counter()
        for _ in self.store.iter_units(completed_only=True):
            self._check()
        if self.progress is not None:
            self._status(validation_seconds=self.progress.state.get("validation_seconds", 0.)
                         + time.perf_counter() - started)

    def _cached(self, kind: str, name: str, request: dict, build: Callable) -> dict:
        self._check()
        request = {"schema_version": SCHEDULER_SCHEMA, "scheduler": "plan15-staged-v1",
                   "config_hash": self.cfg.scientific_hash, "name": name, **request}
        unit_id = f"scheduler-{name}-{content_hash(request)[:20]}"
        path = self.store.unit_path(unit_id)
        if self.store.is_complete(unit_id):
            envelope = json.loads((path / "result.json").read_text())
            if envelope.get("request") != request:
                raise ArtifactError("scheduler request changed under a completed decision")
            return envelope["result"]
        if (name == "trajectory" and self.max_units is not None
                and self.completed_this_invocation >= self.max_units):
            raise ResourceLimit("scheduler trajectory-unit allocation exhausted; resume to continue")
        status = {key: request[key] for key in ("phase", "environment", "replica") if key in request}
        if "treatment" in request:
            status["condition"] = request["treatment"]["condition"]
        self._status(**status, status="running")
        self.store.begin(unit_id, {"kind": kind, **request})
        result = build()
        self._check()
        self.store.write_json(unit_id, "result.json", {"request": request, "result": result})
        self.store.complete(unit_id)
        if name == "trajectory":
            self.completed_this_invocation += 1
        self._status(scheduler_completed_this_invocation=self.completed_this_invocation)
        if self.refresh is not None:
            self.refresh()
        return result

    def _decision(self, name, request, build):
        return self._cached("scheduler_decision", name, request, build)

    def _trajectory(self, family, phase, env, replica, treatment):
        def build():
            before = self.store.storage_status()
            started = time.perf_counter()
            summary = validate_summary(self.run_trajectory(family, phase, env, replica, treatment))
            after = self.store.storage_status()
            summary["scheduler_callback_seconds"] = time.perf_counter() - started
            summary["scheduler_added_bytes"] = max(0, after["used_bytes"] + after.get("external_bytes", 0)
                                                    - before["used_bytes"] - before.get("external_bytes", 0))
            return summary
        return self._cached("scheduler_receipt", "trajectory", {
            "family": family, "phase": phase, "environment": env, "replica": replica,
            "treatment": treatment.mapping()}, build)

    def _calibration(self, env, replica, treatment):
        return self._cached("scheduler_receipt", "calibration", {
            "family": DEVELOPMENT_FAMILIES[0], "phase": "phase1", "environment": env,
            "replica": replica, "treatment": treatment.mapping()},
            lambda: validate_calibration(self.calibrate(DEVELOPMENT_FAMILIES[0], env, replica, treatment), self.cfg))

    def _rows(self, family, phase, env, treatment, count):
        return [self._trajectory(family, phase, env, replica, treatment) for replica in range(count)]

    def phase1(self) -> dict:
        return self._decision("phase1", {"phase": "phase1"}, lambda: {
            "environments": {env: self._decision("policy", {"phase": "phase1", "environment": env},
                                                  lambda env=env: self._policy(env))
                             for env in self.cfg.environments},
            "confirmation_replicas": self.cfg.confirmation_replicas,
            "replica_count_rule": "configured before confirmation; never resized from effect estimates"})

    def _policy(self, env):
        baseline = Treatment.default(self.cfg, "replay_ewc")
        family = DEVELOPMENT_FAMILIES[0]
        trials = {}
        def trial(treatment):
            rows = self._rows(family, "phase1", env, treatment, self.cfg.phase1_replicas)
            trials[treatment.key] = {"treatment": treatment.mapping(), "rows": rows}
            return rows
        default_rows = trial(baseline)
        solver_repair = any(r.get("solver_ok") is False
                            or (not self.cfg.smoke and r.get("under_solved_fraction", 0.) > .5)
                            for r in default_rows)
        if solver_repair and baseline.inner_steps < 100:
            if self.progress is not None:
                change = {"date": utc_now(), "revision": "Sol numerical repair v4",
                          "phase": "phase1", "environment": env,
                          "problem": "Common learner failed or most updates retained gradients above 1e-3 at the solve limit.",
                          "change": "Common ceiling: 100 iterations / at least 150 closures. Rounding-sized rejected increases retain the accepted iterate. Directional mathematical checks use a bounded decreasing-radius ladder; the training radius is unchanged.",
                          "affected_results": f"{env} common-policy development; previous run roots remain separate.",
                          "validation": "Development-only diagnostic; common ceilings and dtype-aware stopping apply to all treatments. Rejected increase and tolerance are recorded.",
                          "next": "Re-evaluate the common policy before predictive screening."}
                changes = self.progress.state.get("changes", [])
                recorded = any(c.get("problem") == change["problem"] and c.get("environment") == env for c in changes)
                self._status(changes=changes if recorded else [*changes, change])
            baseline = dataclasses.replace(baseline, inner_steps=100,
                                            max_evaluations=max(150, baseline.max_evaluations),
                                            label="common-solver-ceiling-repair")
            default_rows = trial(baseline)
        health = lambda rows: all(not r["nonfinite"] and r.get("solver_ok") is not False
                                  and r.get("acquisition_ok") is True and r.get("retention_ok") is True for r in rows)
        known_failure = any(r["nonfinite"] or any(r.get(k) is False for k in
                            ("solver_ok", "acquisition_ok", "retention_ok")) for r in default_rows)
        if known_failure:
            for replay_weight in (.25, .5):
                for ridge in (.01, .1):
                    candidate = dataclasses.replace(baseline, replay_weight=replay_weight,
                                                     ridge_multiplier=ridge)
                    if candidate.key not in trials:
                        trial(candidate)
            # Keep the original short solve as an audit, not as a policy contender after repair.
            eligible = {key: item for key, item in trials.items()
                        if item["treatment"]["inner_steps"] == baseline.inner_steps
                        and item["treatment"]["max_evaluations"] == baseline.max_evaluations}
            healthy = {key: item["rows"] for key, item in eligible.items() if health(item["rows"])}
            if healthy:
                # A broken default is not a valid paired reference. Every item
                # here already passed both predeclared acquisition/retention checks.
                selected = min(healthy, key=lambda key: (_mean([r["current_auc"] for r in healthy[key]]),
                                                        _mean([r["active_seconds"] for r in healthy[key]]), key))
                selection = {"selected": selected, "classification": "healthy_common_policy",
                             "rule": "among policies passing acquisition and retention checks, minimum current AUC then cost"}
            else:
                valid = {key: item["rows"] for key, item in eligible.items()
                         if all(not r["nonfinite"] and r.get("solver_ok") is not False for r in item["rows"])}
                reference = next(iter(valid.values()), default_rows)
                selection = select_candidate(reference, valid, retention_margin=self.cfg.retention_margin)
            if selection["selected"] is not None:
                baseline = Treatment(**trials[selection["selected"]]["treatment"])
        else:
            selection = {"selected": baseline.key, "classification": "default_healthy" if health(default_rows) else "default_health_unresolved"}
        audits = [self._calibration(env, replica, baseline) for replica in range(self.cfg.phase1_replicas)]
        methods = {}
        for method in (*METHODS, "distillation"):
            items = [audit["methods"][method] for audit in audits]
            measured = {}
            for field in ("relative_error", "gradient_error", "seconds"):
                available = [item[field] for item in items if item.get(field) is not None]
                measured[field] = ((sum(available) if field == "seconds" else max(available))
                                   if available else None)
            invalid = [item.get("reason", "numerical failure") for item in items if not item["valid"]]
            if invalid:
                methods[method] = {"valid": False, "reason": "; ".join(invalid), **measured}
                continue
            steps = {item["fd_step"] for item in items}
            if method == "finite_difference" and len(steps) != 1:
                methods[method] = {"valid": False, "reason": "no common numerically validated finite-difference step", **measured}
                continue
            ratio = statistics.median(item["unit_gradient_ratio"] for item in items)
            methods[method] = {"valid": True, "unit_gradient_ratio": ratio,
                               "gradient_ratio_ltm": _optional_mean(items, "gradient_ratio_ltm"),
                               "gradient_ratio_stm": _optional_mean(items, "gradient_ratio_stm"),
                               "strengths": [r / ratio for r in self.cfg.strength_ratios],
                               "target_ratios": list(self.cfg.strength_ratios),
                               "probes": max(item["probes"] for item in items),
                               "fd_step": items[0]["fd_step"], **measured}
        selected_rows = trials[baseline.key]["rows"]
        return {"baseline": baseline.mapping(), "policy_selection": selection,
                "baseline_health": "healthy" if health(selected_rows) else "unresolved_or_failed",
                "baseline_numerically_usable": all(not r["nonfinite"] and r.get("solver_ok") is not False for r in selected_rows),
                "solver_ceiling_repair": solver_repair, "trials": trials,
                "calibration": audits, "methods": methods,
                "mean_baseline_active_seconds": _mean([r["active_seconds"] for r in selected_rows]),
                "mean_baseline_fit_seconds": _mean([r["fit_seconds"] for r in selected_rows])}

    def phase2(self, phase1: dict) -> dict:
        return self._decision("phase2", {"phase": "phase2", "phase1_hash": content_hash(phase1)}, lambda: {
            "environments": {env: self._decision("selection", {"phase": "phase2", "environment": env,
                                 "policy_hash": content_hash(phase1["environments"][env])},
                                 lambda env=env: self._pilot(env, phase1["environments"][env]))
                             for env in self.cfg.environments}})

    def _pilot(self, env, policy):
        baseline = Treatment(**policy["baseline"])
        if not policy["baseline_numerically_usable"]:
            return {"baseline": baseline.mapping(), "screen": {},
                    "primary_selection": {"selected": None, "classification": "common_learner_unavailable", "pareto_keys": []},
                    "winner": None, "distillation": None, "compute_match": None,
                    "validation": {"baseline": []}, "shortlist": {},
                    "validity": "common_learner_unavailable", "baseline_health": policy["baseline_health"]}
        screen_family, validation_family = DEVELOPMENT_FAMILIES[1:]
        controls = self._rows(screen_family, "phase2_screen", env, baseline, self.cfg.pilot_screen_replicas)
        screened, shortlisted = {}, {}
        for method in (*METHODS, "distillation"):
            numerical = policy["methods"][method]
            if not numerical["valid"]:
                screened[method] = {"unavailable": numerical["reason"]}
                continue
            treatments = [dataclasses.replace(baseline, condition=method, strength=strength,
                                               probes=numerical["probes"], fd_step=numerical["fd_step"], label="")
                          for strength in numerical["strengths"]]
            rows = {t.key: self._rows(screen_family, "phase2_screen", env, t, self.cfg.pilot_screen_replicas)
                    for t in treatments}
            first = select_candidate(controls, rows, retention_margin=self.cfg.retention_margin)
            extension = {"direction": None, "reason": "interior optimum or no valid candidate"}
            if first["selected"] is not None:
                selected = next(t for t in treatments if t.key == first["selected"])
                valid_rows = [r for r in first["table"] if r["valid"]]
                all_suppressed = all(r["current_gain"] < 0 for r in valid_rows)
                low, high = min(t.strength for t in treatments), max(t.strength for t in treatments)
                direction = "down" if all_suppressed or selected.strength == low else "up" if selected.strength == high else None
                if direction:
                    extra = dataclasses.replace(selected, strength=low / 10 if direction == "down" else high * 10)
                    extension = {"direction": direction, "reason": "all strengths suppress acquisition" if all_suppressed else "selected strength at bracket edge",
                                 "treatment": extra.mapping()}
            # Freeze the extension decision before its trajectory results exist.
            extension = self._decision("bracket", {"phase": "phase2_screen", "environment": env,
                                       "method": method, "screen_hash": content_hash(first)}, lambda: extension)
            if extension["direction"]:
                extra = Treatment(**extension["treatment"])
                treatments.append(extra)
                rows[extra.key] = self._rows(screen_family, "phase2_screen", env, extra, self.cfg.pilot_screen_replicas)
            choice = select_candidate(controls, rows, retention_margin=self.cfg.retention_margin)
            screened[method] = {"selection": choice, "extension": extension,
                                "treatments": [t.mapping() for t in treatments]}
            if choice["selected"]:
                shortlisted[method] = next(t for t in treatments if t.key == choice["selected"])
        # Validation is a new family; coefficients/numerical settings are frozen.
        validation_controls = self._rows(validation_family, "phase2_validate", env, baseline,
                                          self.cfg.pilot_validate_replicas)
        validated = {method: self._rows(validation_family, "phase2_validate", env, treatment,
                                        self.cfg.pilot_validate_replicas)
                     for method, treatment in shortlisted.items()}
        primary = select_candidate(validation_controls,
                                    {shortlisted[m].key: rows for m, rows in validated.items() if m in METHODS},
                                    retention_margin=self.cfg.retention_margin)
        winner = next((t for t in shortlisted.values() if t.key == primary["selected"]), None)
        distillation = shortlisted.get("distillation")
        if distillation is not None and any(r["nonfinite"] or r.get("solver_ok") is False for r in validated["distillation"]):
            distillation = None
        match = None
        if winner is not None:
            winner_rows = validated[winner.condition]
            ratios = [row["fit_seconds"] / control["fit_seconds"]
                      for row, control in zip(winner_rows, validation_controls, strict=True)
                      if control["fit_seconds"] > 0]
            multiplier = max(1., max(ratios, default=1.))
            matched = dataclasses.replace(baseline, condition="replay_compute_matched",
                                          inner_steps=math.ceil(baseline.inner_steps * multiplier),
                                          max_evaluations=math.ceil(baseline.max_evaluations * multiplier), label="")
            match = {"treatment": matched.mapping(), "fit_time_multiplier": multiplier,
                     "fit_seconds_ceiling": max(r["fit_seconds"] for r in winner_rows),
                     "timing_supported": len(ratios) == len(validation_controls),
                     "rule": "maximum paired development fit-time ratio, converted to fixed optimizer ceilings"}
        return {"baseline": baseline.mapping(), "screen": screened, "primary_selection": primary,
                "winner": None if winner is None else winner.mapping(),
                "distillation": None if distillation is None else distillation.mapping(),
                "compute_match": match, "validation": {"baseline": validation_controls, **validated},
                "shortlist": {name: t.mapping() for name, t in shortlisted.items()},
                "validity": "complete_design" if winner is not None and distillation is not None else "numerically_unavailable_condition",
                "baseline_health": policy["baseline_health"]}

    def freeze_confirmation(self, phase1: dict, phase2: dict) -> dict:
        def build():
            environments = {}
            for env in self.cfg.environments:
                selected = phase2["environments"][env]
                treatments = [] if selected["validity"] == "common_learner_unavailable" else [selected["baseline"]]
                for field in ("winner", "distillation"):
                    if selected[field] is not None:
                        treatments.append(selected[field])
                if selected["compute_match"] is not None:
                    treatments.append(selected["compute_match"]["treatment"])
                environments[env] = {"treatments": treatments,
                                     "classification": selected["primary_selection"]["classification"],
                                     "validity": selected["validity"],
                                     "compute_match": selected["compute_match"]}
            jobs = [{"family": CONFIRMATION_FAMILY, "phase": "phase3", "environment": env,
                     "replica": replica, "treatment": treatment}
                    for replica in range(phase1["confirmation_replicas"])
                    for env in self.cfg.environments for treatment in environments[env]["treatments"]]
            if not jobs:
                raise SchedulerContractError("all common learners are numerically unavailable; repair before launching confirmation")
            total_seconds, added_bytes = 0., 0
            for env in self.cfg.environments:
                validation = phase2["environments"][env]["validation"]
                controls = validation["baseline"]
                if not controls:
                    continue
                startup = max(max(0., r.get("scheduler_callback_seconds", 0.) - r["active_seconds"]) for r in controls)
                total_seconds += startup * phase1["confirmation_replicas"]
                for treatment in environments[env]["treatments"]:
                    name = treatment["condition"]
                    if name == "replay_compute_matched":
                        multiplier = environments[env]["compute_match"]["fit_time_multiplier"]
                        seconds = _mean([r["active_seconds"] + (multiplier - 1) * r["fit_seconds"] for r in controls])
                        sample = controls
                    else:
                        sample = validation["baseline" if name == "replay_ewc" else name]
                        seconds = _mean([r["active_seconds"] for r in sample])
                    total_seconds += seconds * phase1["confirmation_replicas"]
                    added_bytes += max(r.get("scheduler_added_bytes", 0) for r in sample) * phase1["confirmation_replicas"]
            storage = self.store.storage_status()
            # Headroom covers atomic replacement/report overlap; admission, not
            # this development extrapolation, enforces the hard byte ceiling.
            forecast_bytes = storage["used_bytes"] + storage.get("external_bytes", 0) + 2 * added_bytes
            development_units = sum(len(item["trials"]) * self.cfg.phase1_replicas
                                    for item in phase1["environments"].values())
            for selected in phase2["environments"].values():
                if selected["validity"] == "common_learner_unavailable":
                    continue
                development_units += self.cfg.pilot_screen_replicas * (
                    1 + sum(len(v.get("treatments", [])) for v in selected["screen"].values()))
                development_units += self.cfg.pilot_validate_replicas * (1 + len(selected["shortlist"]))
            return {"schema_version": 1, "family": CONFIRMATION_FAMILY,
                    "replicas": phase1["confirmation_replicas"], "environments": environments,
                    "jobs": jobs, "planned_units": len(jobs),
                    "planned_trajectory_units_all_phases": development_units + len(jobs),
                    "confirmation_complete": False, "release_inference": False,
                    "forecast": {"remaining_seconds": 1.2 * total_seconds,
                                 "forecast_bytes": forecast_bytes,
                                 "basis": "development timing and byte increments, 20% time overhead and 2x new-byte headroom; frozen before confirmation",
                                 "storage_timing_measured": added_bytes > 0},
                    "stopping_rule": "fixed cohort completion; no outcome-dependent stopping or extension",
                    "retention_margin": self.cfg.retention_margin,
                    "evaluation_protocol_hash": self.cfg.scientific_hash}
        return self._decision("confirmation-ledger", {"phase": "phase3", "phase1_hash": content_hash(phase1),
                              "phase2_hash": content_hash(phase2)}, build)

    def phase3(self, ledger: dict) -> list[dict]:
        results = []
        for job in ledger["jobs"]:
            self._check()
            treatment = Treatment(**job["treatment"])
            summary = self._trajectory(job["family"], job["phase"], job["environment"], job["replica"], treatment)
            results.append({**job, "summary": summary})
            self._status(confirmation_completed_units=len(results),
                         remaining_seconds=ledger["forecast"]["remaining_seconds"] *
                         (ledger["planned_units"] - len(results)) / ledger["planned_units"])
        return results

    def phase4(self, ledger: dict, results: list[dict]) -> dict:
        if len(results) != ledger["planned_units"]:
            raise SchedulerContractError("closeout requires every predeclared confirmation job")
        if [{key: value for key, value in item.items() if key != "summary"} for item in results] != ledger["jobs"]:
            raise SchedulerContractError("confirmation results do not match the frozen job ledger")
        def build():
            conclusions = {}
            for env in self.cfg.environments:
                self._check()
                rows = [row for row in results if row["environment"] == env]
                control = {row["replica"]: row["summary"] for row in rows if row["treatment"]["condition"] == "replay_ewc"}
                comparisons = {}
                for treatment in ledger["environments"][env]["treatments"]:
                    name = treatment["condition"]
                    if name == "replay_ewc":
                        continue
                    treated = {row["replica"]: row["summary"] for row in rows if row["treatment"]["condition"] == name}
                    endpoints = {}
                    for field in ("current_auc", "historical_auc"):
                        a = {r: value.get(field) if not value["nonfinite"] and value.get("solver_ok") is not False else None for r, value in control.items()}
                        b = {r: value.get(field) if not value["nonfinite"] and value.get("solver_ok") is not False else None for r, value in treated.items()}
                        endpoints[field] = paired_summary(a, b, seed=self.cfg.seed_for("confirmation", 0, f"bootstrap:{env}:{name}:{field}"),
                                                          resamples=self.cfg.bootstrap_samples)
                    mechanisms = {}
                    for field in MECHANISM_FIELDS:
                        self._check()
                        def diagnostic(values):
                            return {r: (value.get("mechanism") or {}).get(field)
                                    if not value["nonfinite"] and value.get("solver_ok") is not False else None
                                    for r, value in values.items()}
                        comparison = paired_summary(
                            diagnostic(control), diagnostic(treated),
                            seed=self.cfg.seed_for("confirmation", 0, f"bootstrap:{env}:{name}:{field}"),
                            resamples=self.cfg.bootstrap_samples)
                        comparison["sign"] = "control minus treatment: positive means lower diagnostic, not proof of consolidation"
                        mechanisms[field] = comparison
                    endpoints["mechanism"] = mechanisms
                    current_gain = endpoints["current_auc"]["mean_gain"]
                    historical_gain = endpoints["historical_auc"]["mean_gain"]
                    if current_gain is None or historical_gain is None:
                        practical = "predictive balance unresolved"
                    elif current_gain > 0 and historical_gain >= 0:
                        practical = "favorable mean acquisition and retention balance"
                    elif current_gain > 0:
                        practical = "mean acquisition gain with a retention cost"
                    elif historical_gain > 0:
                        practical = "mean retention gain without an acquisition gain"
                    else:
                        practical = "no favorable mean predictive balance"
                    endpoints["practical_interpretation"] = practical
                    endpoints["unresolved"] = {
                        "incomplete_predictive_pairs": any(endpoints[field]["n_pairs"] < ledger["replicas"]
                                                           for field in ("current_auc", "historical_auc")),
                        "incomplete_mechanism_pairs": any(value["n_pairs"] < ledger["replicas"] for value in mechanisms.values()),
                        "acquisition_interval_includes_no_gain": endpoints["current_auc"]["ci"] is None
                            or endpoints["current_auc"]["ci"][0] <= 0,
                        "under_solved_units": sum(_finite(value.get("under_solved_fraction"))
                                                  and value["under_solved_fraction"] > .5
                                                  for value in (*control.values(), *treated.values())),
                    }
                    endpoints["mechanism_caution"] = (
                        "Paired historical-cohort interventions are descriptive. Lower STM-removal harm can reflect forgetting; "
                        "lower future-swap harm can reflect less movement and is counterfactual. Check LTM-only prediction, "
                        "full-model acquisition/retention, and update magnitudes together; consolidation is not inferred automatically.")
                    comparisons[name] = endpoints
                conclusions[env] = {"comparisons": comparisons,
                                     "design_validity": ledger["environments"][env]["validity"],
                                     "failed_units": sum(row["summary"]["nonfinite"] or row["summary"].get("solver_ok") is False for row in rows),
                                     "retention_unavailable_units": sum(row["summary"]["historical_auc"] is None for row in rows),
                                     "realized_nines": [{"replica": row["replica"], "count": row["summary"].get("realized_nines")}
                                                        for row in rows if row["treatment"]["condition"] == "replay_ewc"]}
                descriptions = {}
                for treatment in ledger["environments"][env]["treatments"]:
                    name = treatment["condition"]
                    sample = [row["summary"] for row in rows if row["treatment"]["condition"] == name]
                    descriptions[name] = {
                        "mean_active_seconds": _optional_mean(sample, "active_seconds"),
                        "mean_fit_seconds": _optional_mean(sample, "fit_seconds"),
                        "mean_nine_precision": _optional_mean(sample, "nine_precision"),
                        "mean_nine_recall": _optional_mean(sample, "nine_recall"),
                        "right_censored_cohorts": sum(bool(c.get("right_censored")) for row in sample
                                                      for c in row.get("historical_cohorts", [])),
                        "mechanism": [row.get("mechanism") for row in sample],
                    }
                conclusions[env]["descriptive"] = descriptions
            sentences = [f"Completed the predeclared {ledger['replicas']}-replica cohort for every numerically available condition."]
            for env, outcome in conclusions.items():
                primary = next((name for name in METHODS if name in outcome["comparisons"]), None)
                if primary is None:
                    sentences.append(f"{env}: no numerically valid primary curvature comparison.")
                    continue
                endpoints = outcome["comparisons"][primary]
                def display(value):
                    return "unavailable" if value is None else f"{value:.4g}"
                sentences.append(f"{env}, {primary}: mean current NLL-AUC gain {display(endpoints['current_auc']['mean_gain'])}; "
                                 f"post-eviction historical gain {display(endpoints['historical_auc']['mean_gain'])}; "
                                 f"{endpoints['practical_interpretation']}.")
                concerns = []
                if outcome["failed_units"]:
                    concerns.append(f"{outcome['failed_units']} failed units")
                if endpoints["unresolved"]["incomplete_predictive_pairs"]:
                    concerns.append("incomplete predictive pairs")
                if endpoints["unresolved"]["incomplete_mechanism_pairs"]:
                    concerns.append("incomplete mechanism comparisons")
                if endpoints["unresolved"]["under_solved_units"]:
                    concerns.append("persistent under-solving warnings")
                if concerns:
                    sentences.append("Unresolved: " + ", ".join(concerns) + ".")
            sentences.append("Positive gains favor treatment. Intervals and missing pairs are reported separately; these means alone do not establish significance or retention noninferiority. Functional dependence and new learning must support any consolidation interpretation.")
            summary = {"schema_version": 1, "phase": "phase4", "confirmation_complete": True,
                       "release_inference": True, "completed_units": len(results),
                       "planned_units": ledger["planned_units"], "replicas": ledger["replicas"],
                       "environments": conclusions, "optional_extensions_started": False,
                       "conclusion": " ".join(sentences),
                       "interpretation": "Nominal paired evidence per environment; missing endpoints remain unavailable. No automatic noninferiority or familywise claim."}
            self.store.reconcile()
            self._audit()
            if self.closeout is not None:
                self.closeout(summary)
            return summary
        return self._decision("closeout", {"phase": "phase4", "ledger_hash": content_hash(ledger),
                              "results_hash": content_hash(results)}, build)

    def run(self, *, through_phase: int = 4) -> dict:
        if isinstance(through_phase, bool) or through_phase not in (1, 2, 3, 4):
            raise ValueError("through_phase must be 1, 2, 3, or 4")
        self._audit()
        first = self.phase1()
        calibration_rows = [{"phase": "phase1", "environment": env, "method": method,
                             "relative_error": value.get("relative_error"),
                             "gradient_error": value.get("gradient_error"),
                             "seconds": value.get("seconds"),
                             "decision": "numerically valid; predictive pilot pending" if value["valid"] else value["reason"]}
                            for env in self.cfg.environments
                            for method, value in first["environments"][env]["methods"].items()]
        self._status(phase="phase1", calibration=calibration_rows,
                     common_policy={env: first["environments"][env]["baseline"] for env in self.cfg.environments},
                     baseline_health={env: first["environments"][env]["baseline_health"] for env in self.cfg.environments})
        if through_phase == 1:
            return first
        second = self.phase2(first)
        selections = {env: {"winner": selected["winner"],
                            "classification": selected["primary_selection"]["classification"],
                            "pareto_keys": selected["primary_selection"]["pareto_keys"],
                            "distillation": selected["distillation"], "compute_match": selected["compute_match"]}
                      for env, selected in second["environments"].items()}
        for row in calibration_rows:
            selected = second["environments"][row["environment"]]
            if row["method"] in selected["validation"]:
                row["seconds"] = _mean([r["active_seconds"] for r in selected["validation"][row["method"]]])
            if selected["winner"] is not None and row["method"] == selected["winner"]["condition"]:
                row["decision"] = selected["primary_selection"]["classification"]
            elif row["method"] in selected["shortlist"]:
                row["decision"] = "validated secondary control" if row["method"] == "distillation" else "valid alternative; available for later extension"
        self._status(phase="phase2", calibration=calibration_rows, selection=selections)
        if through_phase == 2:
            return second
        ledger = self.freeze_confirmation(first, second)
        self._status(phase="phase3", confirmation_planned_units=ledger["planned_units"],
                     planned_units=ledger["planned_trajectory_units_all_phases"],
                     confirmation_replicas=ledger["replicas"], confirmation_ledger_hash=content_hash(ledger),
                     confirmation_complete=False, release_inference=False, **ledger["forecast"])
        results = self.phase3(ledger)
        if through_phase == 3:
            return {"ledger": ledger, "results": results}
        result = self.phase4(ledger, results)
        self._status(phase="phase4", status="completed", confirmation_complete=True,
                     release_inference=True, confirmation_completed_units=len(results), remaining_seconds=0.,
                     conclusion=result["conclusion"])
        if self.refresh is not None:
            self.refresh()
        return result


def run_study(runner, *, progress=None, max_units: int | None = None) -> dict:
    """CLI bridge. max_units counts newly completed trajectories, not decisions.

    Reports are updated via the supplied progress object. Stop/resource
    exceptions intentionally propagate to the CLI's graceful-stop handler.
    """
    scheduler = Scheduler(runner, progress=progress, max_units=max_units)
    return scheduler.run()
