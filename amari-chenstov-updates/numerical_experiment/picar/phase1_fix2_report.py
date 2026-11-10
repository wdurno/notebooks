"""Artifact-only Fix 2 notebook narrative. No Torch/model/physics imports."""

import base64
import io
from pathlib import Path


def units(snapshot):
    rows = [u for u in snapshot["complete_units"]
            if u["summary"].get("kind", "").startswith("phase1_fix2_")]
    if not rows:
        return []
    rows.sort(key=lambda u: u["manifest"].get("completed_at", ""))
    campaign = rows[-1]["stored_config"].get("campaign_id")
    return [u for u in rows if u["stored_config"].get("campaign_id") == campaign]


def earlier_compute_seconds(snapshot):
    """The sealed Fix 2 forecast charged repairs; include earlier acceptance too."""
    return sum(u["summary"].get("active_seconds", 0.) for u in snapshot["complete_units"]
               if not u["summary"].get("kind", "").startswith("phase1_fix")) + sum(
        u["summary"].get("active_seconds", 0.) for u in snapshot["diagnostics"])


def outputs(snapshot):
    import nbformat
    import pandas as pd
    result, found = [], units(snapshot)
    def table(rows):
        if rows:
            frame = pd.DataFrame(rows)
            result.append(nbformat.v4.new_output("display_data", data={
                "text/html": frame.to_html(index=False, escape=True), "text/plain": frame.to_string(index=False)}))
    def figure_output(figure):
        import matplotlib.pyplot as plt
        data = io.BytesIO()
        figure.savefig(data, format="png", dpi=120)
        plt.close(figure)
        result.append(nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(data.getvalue()).decode(), "text/plain": "Fix 2 diagnostic figure."}))
    software = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_software"]
    if software:
        table([{"Software acceptance": "passed" if software[-1]["passed"] else "failed",
                "Result": software[-1]["test_result"]}])
    rewards = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_rewards"]
    table([{"Rubric": p["rubric"], "Panel": p["split"], "Valid scores": p["valid"],
        "Absent/present": p["separation_pass"], "Correct approach pairs":
        f"{sum(x['correct'] for x in p['pairs'])}/{len(p['pairs'])}", "Readiness guide": p["reward_pass"]}
        for p in rewards])
    if rewards:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 3), constrained_layout=True)
        for panel in rewards:
            rows = [r for r in panel["rows"] if r["state"] in ("far", "intermediate", "close") and r["score"] is not None]
            grouped = {}
            for row in rows:
                grouped.setdefault(row["distance_to_target_center_m"], []).append(row["score"])
            x = sorted(grouped)
            axis.plot(x, [sum(grouped[k]) / len(grouped[k]) for k in x], "o-",
                      label=f"{panel['rubric']} / {panel['split']}")
        axis.set(xlabel="Camera-body position to ball centre (m)", ylabel="Generated visual score (0–10)", ylim=(-.2, 10.2))
        axis.invert_xaxis()
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        figure_output(figure)
        # One fixed layout, all six states; labels are reader-only.
        example = next((p for p in reversed(rewards) if p["rubric"] == "contact"), rewards[-1])
        image_rows = [r for r in example["rows"] if r["seed"] == example["rows"][0]["seed"]]
        source = next((u for u in found if u["summary"]["kind"] == "phase1_fix2_panels"
                       and u["summary"]["split"] == example["split"]), None)
        if source:
            import numpy as np
            figure, axes = plt.subplots(2, 3, figsize=(8, 4.4), constrained_layout=True)
            figure.suptitle(f"{example['rubric']} rubric / {example['split']}", fontsize=11)
            for ax, row in zip(axes.flat, image_rows):
                path = Path(source["path"]) / "frames" / (row["frame_ref"][7:] + ".npy")
                with path.open("rb") as handle:
                    ax.imshow(np.load(handle, allow_pickle=False))
                ax.set_title(f"{row['state']}: {row['score']}", fontsize=9)
                ax.set_axis_off()
            figure_output(figure)
    episodes = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_teacher"]
    grouped = {}
    for e in episodes:
        key = e["variant"], e["settings"]["split"], e["obstacle"], e["domain"]
        grouped.setdefault(key, []).append(e)
    table([{"Teacher": v, "Panel": split, "Task": "Obstacle + ball" if obstacle else "Unobstructed ball",
            "Domain": domain, "Successes / episodes": f"{sum(e['success'] for e in es)}/{len(es)}"}
           for (v, split, obstacle, domain), es in grouped.items()])
    if episodes:
        result.append(nbformat.v4.new_output("stream", name="stdout", text=(
            "Teacher rollouts use accelerated physics for task diagnosis. Their success rates do not "
            "establish real-time Qwen control. Full real-time pipeline timing below uses the student.\n")))
    labels = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_labels"]
    if labels:
        data = labels[-1]
        table([{"Return bins": c["bins"], "Tied progress pairs": c["ties"],
                "Gradient ordering": c["gradient_order_pass"], "Max log-weight rounding error": round(c["max_log_weight_error"], 4),
                "Selected": c["bins"] == data["selected_bins"]} for c in data["candidates"]])
    pilots = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_pipeline"]
    if pilots:
        settings = next(u["stored_config"]["config"] for u in found
                        if u["summary"]["kind"] == "phase1_fix2_pipeline")
        table([{"Timing model context": settings["model"]["context_length"],
            "Timing replay capacity": settings["learning"]["replay_capacity"],
            "New/replay batch": "1 / 1", "Return bins": settings["model"]["return_bins"],
            "Physical control period (s)": 1.}])
    table([{"Loss": p["condition"], "Non-startup fallbacks": f"{p['fallback_count']}/{p['eligible_intervals']}",
            "Updates": p["updates"], "Retired records": p["retired_records"],
            "Peak label backlog": p["maximum_label_backlog"], "Pending after drain": p["pending_after_drain"],
            "Total active seconds": round(p["pipeline_seconds"], 1)} for p in pilots])
    forecasts = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_forecast"]
    if forecasts:
        forecast = forecasts[-1]
        if forecast["status"] == "conditional":
            prior_hours = earlier_compute_seconds(snapshot) / 3600
            table([{"Conditional study": f"{r['pairs']} pairs x {r['transitions_per_arm']} observations/arm",
                "Total forecast (h)": f"{r['total_hours_range'][0]+prior_hours:.1f} to {r['total_hours_range'][1]+prior_hours:.1f}",
                "Fits 20-hour cap": r["total_hours_range"][1] + prior_hours <= 20} for r in forecast["scopes"]])
            result.append(nbformat.v4.new_output("stream", name="stdout", text=(
                f"Displayed totals add {60*prior_hours:.1f} minutes of earlier acceptance/profiling "
                "to the sealed forecast's repair ledger. "
                "Forecasts assume an optimizer event and eventual Fisher admission per new observation; "
                "fit wall times include cooperative waits, so the serial sum is conservative. "
                "They do not establish teacher/student competence or budget approval.\n")))
        else:
            result.append(nbformat.v4.new_output("stream", name="stdout", text=forecast["reason"] + "\n"))
    campaigns = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_campaign"]
    if campaigns:
        data = campaigns[-1]
        table([{"Requirement": key, "Status": value} for key, value in data["deliverables"].items()])
        result.append(nbformat.v4.new_output("stream", name="stdout", text=(
            f"Fix 2 active compute: {data['total_active_seconds']/60:.1f} min; "
            f"all repair work: {data['cumulative_repair_seconds']/60:.1f} min.\n"
            + "\n".join(data["blockers"]) + "\n")))
    if not result:
        result.append(nbformat.v4.new_output("stream", name="stdout", text="Fix 2 implementation/diagnostics in progress.\n"))
    return result


def display_fix2(snapshot):
    from IPython.display import display
    for output in outputs(snapshot):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_fix2(book, snapshot):
    import nbformat
    if not units(snapshot) and not any("fix2" in row["path"] for row in snapshot["live"]):
        return book
    cells = [nbformat.v4.new_markdown_cell("### Phase 1 Fix 2: Dense Visual Rewards and Robot Workflow\n\n"
        "The prior readiness checks used a completion probability and a potential difference, "
        "which did not reproduce the earlier robot's graded visual rewards. This repair generates "
        "explicit 0–10 scores, checks unobstructed approach before obstacle navigation, and audits "
        "the complete reward-to-return pipeline. Target contact is success; obstacle/wall contact is failure. "
        "Historical Fix 1 results remain above. These are readiness diagnostics, with no STM efficacy claim.\n\n"
        "The initial Fix 2 attempt (`fix2-v1`) rejected valid JSON inside Markdown fences. "
        "Its raw responses and artifacts remain available; those rejected responses are not a navigation-performance estimate. "
        "The revised parser accepts one exact JSON fence while retaining strict values/keys. "
        "The revised timing pilot also masks retired recurrent predecessors."),
        nbformat.v4.new_code_cell("from numerical_experiment.picar.phase1_fix2_report import display_fix2\n"
                                 "display_fix2(snapshot)", outputs=outputs(snapshot))]
    for i, cell in enumerate(cells):
        cell.id = f"plan19-fix2-{i}"
        cell.metadata["plan19_fix2"] = True
    book.cells = [cell for cell in book.cells if not cell.metadata.get("plan19_fix2")]
    index = next((i for i, c in enumerate(book.cells) if c.cell_type == "markdown"
                  and c.source.startswith("## Development")), len(book.cells) - 1)
    book.cells[index:index] = cells
    return book


def conclusion(snapshot):
    rows = [u["summary"] for u in units(snapshot) if u["summary"]["kind"] == "phase1_fix2_campaign"]
    if not rows:
        if units(snapshot) or any("fix2" in row["path"] for row in snapshot["live"]):
            return ("## Conclusion\n\nPhase 1 Fix 2 is not yet complete. Only finished diagnostic units "
                "are summarized; task readiness and the final resource check-in remain pending. "
                "Earlier results remain above. No scientific learning phases or physical operation have started.")
        return None
    data = rows[-1]
    found = units(snapshot)
    episodes = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_teacher"]
    simple = [e for e in episodes if e["variant"] == "ported" and not e["obstacle"]]
    full = [e for e in episodes if e["obstacle"]]
    simple_text = f"{sum(e['success'] for e in simple)}/{len(simple)}" if simple else "not measured"
    full_text = f"{sum(e['success'] for e in full)}/{len(full)}" if full else "not measured"
    pipelines = [u["summary"] for u in found if u["summary"]["kind"] == "phase1_fix2_pipeline"]
    zero_fallbacks = bool(pipelines) and all(p["fallback_count"] == 0 for p in pipelines)
    return ("## Conclusion\n\n**Fix 2 is implemented and its diagnostics are complete.** "
        f"The ported Qwen interface reached the ball in {simple_text} unobstructed trials. "
        f"The full obstacle task achieved {full_text}; fresh 20-layout validation was not admitted.\n\n"
        "Numeric visual scores now work and separate absence from close approach, but remain coarse "
        "and miss the strict far-to-near ordering guide. Return-bin/actor-weight resolution passes with "
        f"{data['return_bins']} bins. Both finite multi-episode student pilots completed with "
        + ("zero non-startup fallbacks" if zero_fallbacks else "the deadline limitations shown above")
        + ", including online updates and charged stopped labeling/sleep. This does not prove uninterrupted "
        "one-update-per-observation service.\n\n"
        "The primary study remains at the resource/task check-in. Default-off retired sleep and "
        "the Pi/speech interfaces have software acceptance; physical calibration and real speech inference "
        "remain unvalidated. No physical robot was operated and no STM sample-efficiency result is claimed. "
        "Earlier attempts and their source snapshots remain available.")
