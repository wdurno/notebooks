"""Lightweight descriptive readiness and conditional resource forecasts."""

from collections import Counter


def panel_summary(evaluations):
    result = []
    for panel in evaluations:
        for stratum in ("all", "clear", "blocked"):
            rows = [r for r in panel["rows"] if stratum == "all" or
                r.get("layout", {}).get("curriculum", {}).get("direct_path_blocked") == (stratum == "blocked")]
            result.append({"checkpoint": panel.get("checkpoint"), "difficulty": panel["difficulty"],
                "stratum": stratum, "count": len(rows),
                "success_count": sum(r["success"] is True for r in rows),
                "success_rate": sum(r["success"] is True for r in rows)/len(rows) if rows else None,
                "end_reasons": dict(Counter(r["end_reason"] for r in rows))})
    return result


def hard_gain(summary):
    panels = {p["checkpoint"]: p["rows"] for p in summary["evaluations"] if p["difficulty"] == "hard"}
    before, after = panels["initial"], panels["final"]
    if [r["seed"] for r in before] != [r["seed"] for r in after] or not before:
        raise ValueError("readiness comparison requires the same nonempty layout panel")
    return (sum(r["success"] for r in after)-sum(r["success"] for r in before))/len(before)


def observed_trainability(pilots):
    return any(hard_gain(p) >= .1-1e-12 for p in pilots)


def forecast(pilots, *, phase2_seconds, prior_study_seconds):
    count = sum(p["acquired_count"] for p in pilots)
    processing = sum(p["timings"]["active_seconds"]-p["timings"]["evaluation_seconds"] for p in pilots)
    seconds_per = processing/count if count else None
    evaluation = sum(p["timings"]["evaluation_seconds"] for p in pilots)/len(pilots) if pilots else None
    rows = []
    if seconds_per is not None:
        for pairs, observations in ((4, 1200), (6, 2400)):
            control_hours = pairs*observations*seconds_per/3600
            for fd_multiplier in (1., 2.):
                rows.append({"pairs": pairs, "observations_per_arm": observations,
                    "fd_cost_multiplier_scenario": fd_multiplier,
                    "training_processing_hours": control_hours*(1+fd_multiplier),
                    "evaluation_hours_at_pilot_panel_size": 2*pairs*evaluation/3600,
                    "not_total_study_forecast": True})
    return {"schema_version": 1, "seconds_per_captured_interval": seconds_per,
        "fd_unmeasured": True, "scope_rows": rows,
        "phase2_active_seconds": phase2_seconds, "prior_study_seconds": prior_study_seconds,
        "study_charged_seconds": phase2_seconds+prior_study_seconds,
        "remaining_to_20h_seconds": max(0., 72000-phase2_seconds-prior_study_seconds),
        "trainability_observed": observed_trainability(pilots),
        "limitations": ["FD multipliers are scenarios, not measurements.",
            "Later development, additional evaluation checkpoints and final analysis are extra.",
            "Pilot layouts and bootstrap are shared; this is not confirmatory efficacy inference."],
        "next_phase_authorized": False}
