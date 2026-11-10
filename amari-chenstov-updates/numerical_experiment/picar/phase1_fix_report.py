"""Concise, read-only Phase 1 Fix 1 notebook section. Never imports Torch."""

import base64
import io


def repair_units(snapshot):
    units = sorted((u for u in snapshot["complete_units"]
        if u["summary"].get("kind", "").startswith("phase1_fix1_")),
        key=lambda u: u["manifest"].get("completed_at", ""))
    if not units:
        return []
    latest = units[-1]["stored_config"]["source_sha256"]
    return [u for u in units if u["stored_config"]["source_sha256"] == latest]


def repair_outputs(snapshot):
    import nbformat
    import pandas as pd
    outputs = []
    units = repair_units(snapshot)
    def table(rows):
        if rows:
            frame = pd.DataFrame(rows)
            outputs.append(nbformat.v4.new_output("display_data", data={"text/plain": frame.to_string(index=False),
                "text/html": frame.to_html(index=False, escape=True)}))
    audits = [u["summary"] for u in units if u["summary"]["kind"] == "phase1_fix1_interface"]
    if audits:
        table([{"Scene": row["state"], "Native/split relative error": f"{row['relative_hidden_l2']:.2g}",
            "Parity": row["parity_pass"], "Mass on action digits": f"{row['allowed_answer_mass'][0]:.2g}",
            "Native description": row["native_description"][:180]} for row in audits[-1]["rows"]])
    panels = [u["summary"] for u in units if u["summary"]["kind"] == "phase1_fix1_panels"]
    table([{"Role interface": p["variant"], "Valid teacher outputs": p["teacher_output_valid"],
        "Instruction contrasts": p["language_contrast_pass"], "Visual reward ordering": p["reward_order_pass"]}
        for p in panels])
    episodes = [u["summary"] for u in units if u["summary"]["kind"] == "phase1_fix1_teacher"]
    grouped = {}
    for e in episodes:
        key = e["settings"]["variant"], e["settings"]["split"], e["domain"]
        grouped.setdefault(key, []).append(e)
    table([{"Teacher": v, "Panel": split, "Domain": domain, "Successes / episodes": f"{sum(e['success'] for e in es)}/{len(es)}"}
           for (v, split, domain), es in grouped.items()])
    controls = [u["summary"] for u in units if u["summary"]["kind"] == "phase1_fix1_control"]
    table([{"History": c["context_length"], "Period (s)": c["period_s"], "Cooperative": c["cooperative"],
        "Fallbacks / non-startup": f"{c['fallback_count']}/{c['eligible_intervals']}",
        "Fallback (%)": round(100 * c["fallback_fraction"], 1),
        "Update handoffs": c["model_access"]["cooperative_yields"], "Deadline guide": c["deadline_pass"]}
        for c in controls])
    if panels:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
        for p in panels:
            by_state = {}
            for row in p["rows"]:
                by_state.setdefault(row["state"], []).append(row["potential"])
            names = ("far_deadline_failure", "near_goal_not_success", "stopped_success", "obstacle_collision")
            axis.plot(range(4), [sum(by_state[n]) / len(by_state[n]) for n in names], "o-", label=p["variant"])
        axis.set_xticks(range(4), ["Far failure", "Near, incomplete", "Stopped success", "Collision"])
        axis.set(ylabel="Visual potential (A/B mean)", ylim=(0, 1))
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8, ncol=2)
        data = io.BytesIO()
        figure.savefig(data, format="png", dpi=120)
        plt.close(figure)
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(data.getvalue()).decode(),
            "text/plain": "A useful completion potential should favor stopped success over far failure and collision."}))
    campaigns = [u["summary"] for u in units if u["summary"]["kind"] == "phase1_fix1_campaign"]
    if campaigns:
        last = campaigns[-1]
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=(
            f"Repair compute: {last['total_active_seconds'] / 60:.1f} active minutes. "
            "Later learning phases remain unstarted.\n" + "\n".join(last["blockers"]) + "\n")))
    if not units:
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text="Repair diagnostics pending.\n"))
    for unit in units:
        if unit["summary"]["kind"] == "phase1_fix1_provenance":
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=(
                "Execution notes: " + " ".join(unit["summary"].get("operator_events", [])) + "\n")))
    return outputs


def display_repair(snapshot):
    from IPython.display import display
    for output in repair_outputs(snapshot):
        if output.output_type == "stream":
            print(output.text, end="")
        else:
            display(output.data, raw=True)


def append_repair(book, snapshot):
    import nbformat
    if not repair_units(snapshot) and not any("fix" in u["path"] for u in snapshot["live"]):
        return book
    cells = [nbformat.v4.new_markdown_cell("### Phase 1 Fix 1: Grounding and Online Readiness\n\n"
        "The original run found an unvalidated action/reward interface and updates that blocked control. "
        "This repair checks native-model parity, clearer motor semantics and visual history, then "
        "cooperative scheduling. Original results remain above. These are engineering diagnostics, "
        "not STM learning results; the newest repair revision is shown without pooling attempts."),
        nbformat.v4.new_code_cell("from numerical_experiment.picar.phase1_fix_report import display_repair\n"
            "display_repair(snapshot)", outputs=repair_outputs(snapshot))]
    for i, cell in enumerate(cells):
        cell.metadata["plan19_fix1"] = True
        cell.id = f"plan19-fix1-{i}"
    book.cells = [c for c in book.cells if not c.metadata.get("plan19_fix1")]
    index = next((i for i, c in enumerate(book.cells) if c.cell_type == "markdown"
                  and c.source.startswith("## Development")), len(book.cells))
    book.cells[index:index] = cells
    return book


def repair_conclusion(snapshot):
    campaigns = [u["summary"] for u in repair_units(snapshot) if u["summary"]["kind"] == "phase1_fix1_campaign"]
    if not campaigns:
        return None
    summary = campaigns[-1]
    teacher = (f"{summary['teacher_validation_successes']}/{summary['teacher_validation_count']}"
               if summary['teacher_validation_count'] else "not run because development readiness failed")
    return ("## Conclusion\n\nPhase 1 Fix 1 is recorded; no scientific learning phases have started. "
        f"Native/custom parity: {summary['parity_pass']}. Online deadline/controllability guide: "
        f"{summary['control_pass']}. Fresh teacher validation: {teacher}.\n\n"
        + ("**Outstanding:** " + " ".join(summary["blockers"]) if summary["blockers"] else
           "The bounded readiness checks pass; bootstrap/student competence remains separate Phase 2 work.")
        + "\n\nOriginal failed diagnostics remain available. No sample savings are inferred here; "
          "the original runtime forecasts are conditional and no budget extension is authorized.")
