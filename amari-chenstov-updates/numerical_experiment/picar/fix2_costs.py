"""Resource forecasts from sealed Fix 2 operation timings, never efficacy claims."""

from __future__ import annotations

import statistics

from .phase1_forecast import forecast


def resource_forecast(pipelines, teachers, checkpoint, *, elapsed_hours):
    by_condition = {row["condition"]: row for row in pipelines}
    values = [t["seconds"] for e in teachers for t in e["commands"]]
    if not values:
        return {"status": "unavailable", "reason": "No valid teacher commands were executed."}
    control, fd = by_condition["replay_ewc"], by_condition["stm_fd"]
    actor = [r["operation_medians"]["inference_seconds"] for r in pipelines]
    fisher = [r["operation_medians"]["fisher_seconds"] for r in pipelines]
    evaluator = [r["operation_medians"]["evaluator_seconds"] for r in pipelines]
    fit = [r["operation_medians"]["fit_seconds"] for r in pipelines]
    if any(x is None or x <= 0 for x in [*actor, *fisher, *evaluator, *fit]):
        return {"status": "unavailable", "reason": "The pipeline did not measure every required operation."}
    probe = {"recipe": {"top_layers": 1, "context_length": 4, "batch": 1, "return_bins": control["return_bins"]},
        "checkpoint": checkpoint, "timings": {"inference_seconds": max(actor),
            "fit_control_seconds": control["operation_medians"]["fit_seconds"],
            "fit_fd_seconds": fd["operation_medians"]["fit_seconds"],
            "fisher_seconds": max(fisher), "evaluator_seconds": max(evaluator),
            "teacher_seconds": statistics.median(values)}}
    rows = []
    for pairs, transitions in ((6, 2400), (4, 1200)):
        result = forecast(probe, pairs=pairs, transitions=transitions, control_period_s=1.,
                          elapsed_hours=elapsed_hours, capture_width=320, capture_height=240)
        rows.append({"pairs": pairs, "transitions_per_arm": transitions,
            "phase_hours": result["phase_hours"], "total_hours_range": result["total_hours_range"],
            "fits_20h": result["budgetfit"], "raw_artifact_bytes": result["storage"]["raw_and_record_bytes"]})
    return {"status": "conditional", "operations": probe["timings"], "checkpoint": checkpoint,
        "scopes": rows, "assumptions": [
            "One accepted optimizer event and one eventual Fisher admission per new observation; not a selected cadence.",
            "Teacher generation, reward generation and complete-return delay now appear in measured costs.",
            "Timing histories shrink at the eligibility boundary; actual future history/batch changes require remeasurement.",
            "The prior evaluation/development episode allowances are retained as conditional planning assumptions.",
            "Journal retains recovery states plus evidence; the old retain-every-state terabyte forecast is not the policy.",
            "The earlier forecaster's extra 1/40 reward-call allowance is conservatively retained although dense stage rewards need no initial endpoint.",
            "Skill, retention calibration, labels, speech cost if enabled and budget approval remain separate admission requirements."]}
