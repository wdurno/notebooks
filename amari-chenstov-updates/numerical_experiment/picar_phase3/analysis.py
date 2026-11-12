"""Descriptive development selection, never confirmation or significance tests."""

from collections import defaultdict
import math


def candidate(summary):
    curves = {}
    for difficulty in ("easy", "hard"):
        panels = [p for p in summary["evaluations"] if p["difficulty"] == difficulty]
        by_checkpoint = {p["checkpoint"]:p for p in panels}
        if not {"initial", "final"}.issubset(by_checkpoint):
            raise ValueError("development endpoints are incomplete")
        ids = None
        curve = []
        for panel in panels:
            rows = panel["rows"]
            current = [r["seed"] for r in rows]
            if not current or len(current) != len(set(current)):
                raise ValueError("a nonempty unique development panel is required")
            if ids is None:
                ids = set(current)
            if set(current) != ids or any(type(r["success"]) is not bool for r in rows):
                raise ValueError("development panels must use the same complete seed IDs")
            count = panel.get("acquired_count")
            if count is None:
                if panel["checkpoint"] == "initial":
                    count = 0
                elif panel["checkpoint"] == "final":
                    count = summary["acquired_count"]
                else:
                    raise ValueError("midpoint exposure is missing; do not invent its location")
            if type(count) is not int or count < 0:
                raise ValueError("invalid measured exposure")
            curve.append((count, sum(r["success"] for r in rows)/len(rows)))
        curve.sort()
        if curve[0][0] != 0 or curve[-1][0] != summary["acquired_count"] or curve[-1][0] <= 0:
            raise ValueError("development curve does not span the measured trajectory")
        if any(b[0] <= a[0] for a,b in zip(curve,curve[1:])):
            raise ValueError("duplicate or decreasing development exposure")
        average = sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(curve,curve[1:]))/curve[-1][0]
        curves[difficulty] = {"initial":curve[0][1],"final":curve[-1][1],"average":average,
                              "checkpoints":len(curve),"panel_size":len(ids)}
    easy, hard = curves["easy"],curves["hard"]
    return {"name":summary["settings"]["name"],"seed":summary["seed"],
        "fd_strength":summary["settings"]["fd_strength"],"ridge":summary["settings"]["ridge"],
        "easy_initial":easy["initial"],"easy_final":easy["final"],
        "hard_initial":hard["initial"],"hard_final":hard["final"],
        "old_gap_average":easy["initial"]-easy["average"],
        "old_gap_endpoint":easy["initial"]-easy["final"],
        "new_gain_average":hard["average"]-hard["initial"],
        "new_gain_endpoint":hard["final"]-hard["initial"],
        "checkpoints":min(easy["checkpoints"],hard["checkpoints"]),
        "panel_resolution":1/easy["panel_size"],"source":summary["unit_path"]}


def select_control(pilots):
    rows = [candidate(p) for p in pilots if p["settings"]["fd_strength"] == 0]
    if not rows:
        raise ValueError("no completed common-control candidate")
    retained = [r for r in rows if r["old_gap_endpoint"] <= .1+1e-12]
    selected = max(retained or rows, key=lambda r:(r["new_gain_endpoint"],r["new_gain_average"],
        -max(0.,r["old_gap_endpoint"]),-max(0.,r["old_gap_average"]),-abs(r["ridge"]-.1)))
    return selected, rows


def selection(pilots, *, control_name=None):
    control, rows = select_control(pilots)
    if control_name is not None:
        control = next(r for r in rows if r["name"] == control_name)
    controls = {r["seed"]:r for r in rows if r["name"] == control["name"]}
    comparisons = []
    for pilot in pilots:
        if pilot["settings"]["fd_strength"] <= 0:
            continue
        fd = candidate(pilot)
        reference = controls.get(fd["seed"])
        if reference is None:
            continue
        endpoint = fd["old_gap_endpoint"]-reference["old_gap_endpoint"]
        average = fd["old_gap_average"]-reference["old_gap_average"]
        comparisons.append({"name":fd["name"],"seed":fd["seed"],
            "retention_average_difference":average,"retention_endpoint_difference":endpoint,
            "new_endpoint_difference":fd["new_gain_endpoint"]-reference["new_gain_endpoint"],
            "retention_matched":abs(endpoint)<=.05+1e-12 and abs(average)<=.05+1e-12,
            "fd_source":fd["source"],"control_source":reference["source"]})
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["name"]].append(row)
    matched = [name for name, group in grouped.items() if all(r["retention_matched"] for r in group)]
    pool = matched or list(grouped)
    fd_name = max(pool,key=lambda name:(len(grouped[name])>=2,
        sum(r["new_endpoint_difference"] for r in grouped[name])/len(grouped[name]))) if pool else None
    learning = (sum(r["new_gain_endpoint"] for r in controls.values())/len(controls) >= .1-1e-12
        and sum(r["old_gap_endpoint"] for r in controls.values())/len(controls) <= .1+1e-12)
    enough_seeds = len(controls)>=2 and fd_name is not None and len(grouped[fd_name])>=2
    ready = learning and bool(matched) and enough_seeds
    return {"selected_control":control["name"],"selected_fd":fd_name,
        "control_learning_observed":learning,"retention_matched":fd_name in matched if fd_name else None,
        "selected_control_rows":list(controls.values()),"comparisons":comparisons,
        "candidate_rows":[candidate(p) for p in pilots],"development_replica_count":len({p["seed"] for p in pilots}),
        "ready_to_propose_confirmation":ready,"next_phase_authorized":False,
        "reason":("A retention-matched development pair is available; fresh execution still requires review."
            if ready else "Common learning, matched retention, or two-seed replication remains unresolved; preserve the frontier."),
        "retention_tolerance":.05,"control_retention_limit":.1,
        "retention_equivalence_proved":False,"sample_savings_available":False}


def forecast(pilots, decision, *, phase3_seconds, prior_study_seconds):
    controls = [p for p in pilots if p["settings"]["name"]==decision["selected_control"]]
    treatments = [p for p in pilots if p["settings"]["name"]==decision["selected_fd"]]
    def seconds_per(rows):
        count=sum(p["acquired_count"] for p in rows)
        return sum(p["timings"]["active_seconds"]-p["timings"]["evaluation_seconds"] for p in rows)/count if count else None
    control,fd=seconds_per(controls),seconds_per(treatments)
    rows=[]
    for pairs,intervals in ((4,1200),(6,2400)):
        process = None if control is None or fd is None else pairs*intervals*(control+fd)/3600
        evaluation = sum(p["timings"]["evaluation_seconds"] for p in controls+treatments)/max(1,len(controls+treatments))
        rows.append({"pairs":pairs,"observations_per_arm":intervals,
            "training_processing_hours":process,"evaluation_hours_at_pilot_panel_size":2*pairs*evaluation/3600,
            "not_total_study_forecast":True})
    return {"scope_rows":rows,"phase3_active_seconds":phase3_seconds,
        "prior_study_seconds":prior_study_seconds,"study_charged_seconds":prior_study_seconds+phase3_seconds,
        "remaining_to_20h_seconds":max(0.,72000-prior_study_seconds-phase3_seconds),
        "control_seconds_per_captured_interval":control,"fd_seconds_per_captured_interval":fd,
        "fd_processing_multiplier":None if fd is None or control is None else fd/control,
        "fd_unmeasured":fd is None,"next_phase_authorized":False,
        "limitations":["Later development, richer confirmation panels and closeout are additional.",
            "Development trajectories share initialization and layouts; these are not fresh efficacy results."]}
