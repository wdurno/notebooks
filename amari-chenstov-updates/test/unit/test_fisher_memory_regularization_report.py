"""Small sealed scalar fixtures; no MNIST, learner, or production compute."""

import copy
import json

import nbformat
import pytest

from mnist_experiment.fisher_memory import regularization_report as report
from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store
from mnist_experiment.fisher_memory.config import Treatment


def trajectory(replica=0, stream="mixed", condition="replay_ewc", *, family=report.DEVELOPMENT,
               policy=100, initial=3., ridge=.1):
    strength = 160.89 if condition == "finite_difference" else 0.
    name = f"trajectory-{family}-{replica}-{stream}-{condition}-{ridge}-{policy}"
    meta = dict(schema_version=1, kind="drip_trajectory", phase="7", family=family,
                replica=replica, stream=stream, condition=condition, event_target=2,
                policy_iterations=policy, treatment=dict(condition=condition, strength=strength,
                                                         ridge_multiplier=ridge))
    improvement = .5 if strength else .2
    rows = [dict(paired_rotated_event=event, actual_rotated_events=event if stream == "mixed" else 0,
                 rotated_nll=initial - improvement * event, base_nll=.2 + .1 * event,
                 rotated_accuracy=.5 + .1 * event, base_accuracy=.9 - .1 * event)
            for event in range(3)]
    summary = dict(schema_version=1, unit_id=name, replica=replica, stream=stream,
                   condition=condition, strength=strength, event_target=2,
                   rotated_auc=(rows[1]["rotated_nll"] + rows[2]["rotated_nll"]) / 2,
                   base_auc=.35, final_rotated_nll=rows[-1]["rotated_nll"],
                   final_base_nll=.4, final_rotated_accuracy=.7, final_base_accuracy=.7,
                   active_seconds=120. if strength else 60., solver_failures=0, under_solved=1)
    fit = [dict(new_nll=.1, replay_nll=.2, s_t=.02, kappa=.002,
                omega_kappa=.078, ltm_motion=.03, stm_motion=.04)]
    return dict(unit_id=name, metadata=meta, rows=rows, fit=fit, summary=summary)


def audit(parent, *, event=1, gain=.1):
    meta = parent["metadata"]
    name = f"audit-{parent['unit_id']}-{event}"
    metadata = dict(kind="regularization_audit", parent_id=parent["unit_id"], replica=meta["replica"],
                    stream=meta["stream"], treatment=meta["treatment"], event=event,
                    policy_iterations=meta["policy_iterations"])
    scores = dict(rotated=dict(nll=1., accuracy=.6), base=dict(nll=.3, accuracy=.8))
    extended = copy.deepcopy(scores)
    extended["rotated"]["nll"] += .05
    value = dict(schema_version=1, ordinary_objective=2., extended_objective=2. - gain,
                 objective_gain=gain, relative_objective_gain=gain / 2,
                 original_gradient_inf=.3, extended_gradient_inf=.02,
                 ltm_motion=.01, stm_motion=.02, seconds=3., ordinary=scores, extended=extended,
                 prediction_tv=dict(rotated=.02, base=.01))
    return dict(unit_id=name, metadata=metadata, result=value)


def decision(name, value):
    return dict(unit_id=f"regularization-{name}-v1",
                metadata=dict(kind="regularization_decision", decision=name),
                result=dict(schema_version=1, **value))


def seal(store, unit, *, completed=True):
    name, meta = unit["unit_id"], unit["metadata"]
    store.begin(name, meta)
    if "result" in unit:
        store.write_json(name, "result.json", unit["result"])
    else:
        for filename, value in (("metadata.json", meta), ("metrics.json", unit["rows"]),
                                ("fit.json", unit["fit"]), ("summary.json", unit["summary"])):
            store.write_json(name, filename, value)
    if completed:
        store.complete(name)


def make_root(tmp_path, units=(), *, stage="7A", name="run", decisions=(), incomplete=(), source=None):
    root = tmp_path / name
    with Store(root, {"test": "regularization-report"}) as store:
        for value in (*units, *decisions):
            seal(store, value)
        for value in incomplete:
            seal(store, value, completed=False)
        store.write_json(None, "status.json", dict(stage=stage, status="running", active_seconds=3600.,
                                                   completed=len(units), expected=len(units) + len(incomplete),
                                                   current_job=incomplete[0]["unit_id"] if incomplete else None,
                                                   changes=[], development_root=source, forecast_hours=1.5))
    return root


def base_notebook(tmp_path):
    cells = [nbformat.v4.new_markdown_cell("# Existing study"),
             nbformat.v4.new_markdown_cell("## Phase 5C: Existing conclusion",
                                          metadata={"tags": ["plan15_phase5c"]}),
             nbformat.v4.new_code_cell("previous = 1", execution_count=2, outputs=[
                 nbformat.v4.new_output("stream", name="stdout", text="Preserve me\n")]),
             nbformat.v4.new_markdown_cell("## Phase 6: Deferred questions")]
    path = tmp_path / "base.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    return path


def cohort():
    return [trajectory(replica, stream, condition, family=report.CONFIRMATION)
            for replica in range(2) for stream in ("mixed", "base")
            for condition in report.CONDITIONS]


def analysis():
    return dict(complete=True, inference_ready=True,
                primary=dict(mean_gain=.45, ci=[.3, .6], p_value=.25,
                             paired_values=[dict(replica=r, gain=.45) for r in range(2)]))


def skip_render(monkeypatch):
    def capture(figure, caption):
        from matplotlib import pyplot as plt
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Sealed scalar figure", outputs=[
            nbformat.v4.new_output("display_data", data={"text/plain": caption})])

    monkeypatch.setattr(report, "_plot_cell", capture)


def test_diagnostics_read_the_actual_learner_artifact_field_names(tmp_path):
    unit = trajectory()
    unit["fit"] = [dict(final_new_nll=.123, final_replay_nll=.456,
                        fisher_scale=.02, kappa=.002, effective_ridge_weight=.078,
                        ltm_displacement_norm=.03, stm_displacement_norm=.04)]
    data = report.collect(make_root(tmp_path, [unit]))
    cell = report._diagnostics(data, data["trajectories"], [])
    assert "0.123 / 0.456" in cell.source
    assert "0.02 / 0.002 / 0.078" in cell.source
    assert "0.03 / 0.04" in cell.source


def test_boundary_selection_is_described_as_best_tested_not_bracketed(tmp_path):
    records = [decision("ledger", dict(ridge_grid=[.03, .1])),
               decision("shortlist", dict(replay=dict(ridge_multiplier=.1),
                                           fd=dict(ridge_multiplier=.03)))]
    data = report.collect(make_root(tmp_path, decisions=records))
    text = report._decisions(data).source
    assert "upper edge" in text and "lower edge" in text
    assert "best tested" in text and "not bracketed" in text


def test_diagnostics_show_recorded_timing_breakdown_without_reconstruction(tmp_path):
    units = [trajectory(replica=r, family=report.CONFIRMATION) for r in range(2)]
    for unit, scale in zip(units, (1, 3)):
        unit["summary"].update(fit_seconds=30. * scale, eviction_seconds=12. * scale,
                               evaluation_seconds=6. * scale, active_seconds=60. * scale)
    data = report.collect(make_root(tmp_path, units))
    text = report._diagnostics(data, data["trajectories"], []).source
    assert "Fit / eviction / evaluation / online s" in text
    assert "60.0 / 24.0 / 12.0 / 120.0" in text
    units[1]["summary"].pop("eviction_seconds")
    missing = report.collect(make_root(tmp_path, units, name="missing"))
    text = report._diagnostics(missing, missing["trajectories"], []).source
    assert "60.0 / - / 12.0 / 120.0" in text


def test_chronology_replacement_is_append_only_and_preserves_outputs(tmp_path, monkeypatch):
    skip_render(monkeypatch)
    parent = trajectory()
    root = make_root(tmp_path, [parent, audit(parent)])
    base = base_notebook(tmp_path)
    original = nbformat.read(base, as_version=4)
    old = nbformat.v4.new_markdown_cell("Stale Phase 7", metadata={"tags": [report.TAG]})
    original.cells.insert(1, old)
    nbformat.write(original, base)
    before = base.read_bytes()
    first = report.build_notebook(base, root)
    assert base.read_bytes() == before
    expected = [cell for cell in original.cells if report.TAG not in cell.metadata.get("tags", [])]
    preserved = [cell for cell in first.cells if not {report.TAG, report.PROGRESS_TAG}.intersection(cell.metadata.get("tags", []))]
    assert preserved == expected
    assert report.PROGRESS_TAG in first.cells[1].metadata["tags"]
    assert all(report.TAG in cell.metadata.get("tags", []) for cell in first.cells[-8:])
    nbformat.write(first, base)
    second = report.build_notebook(base, root)
    assert len(second.cells) == len(first.cells) == len(expected) + 9
    assert [cell for cell in second.cells if not {report.TAG, report.PROGRESS_TAG}.intersection(cell.metadata.get("tags", []))] == expected
    summaries = [cell for cell in second.cells if report.PROGRESS_TAG in cell.metadata.get("tags", [])]
    assert len(summaries) == 1 and len(summaries[0].source) < 600
    assert "#plan15-phase7" in summaries[0].source
    assert "Stale Phase 7" not in nbformat.writes(second)
    nbformat.validate(second)


@pytest.mark.parametrize("mutation", ["auc", "endpoint", "event", "actual", "summary_schema", "metadata", "fit"])
def test_completed_trajectory_reconstruction_rejects_incompatible_payloads(tmp_path, mutation):
    value = trajectory()
    if mutation == "auc":
        value["summary"]["rotated_auc"] += .1
    elif mutation == "endpoint":
        value["summary"]["final_base_nll"] += .1
    elif mutation == "event":
        value["rows"][1]["paired_rotated_event"] = 2
    elif mutation == "actual":
        value["rows"][1]["actual_rotated_events"] = 0
    elif mutation == "summary_schema":
        value["summary"]["schema_version"] = 2
    elif mutation == "metadata":
        value["metadata"]["condition"] = "finite_difference"
    else:
        value["fit"] = {}
    root = make_root(tmp_path, [value])
    with pytest.raises(ArtifactError):
        report.collect(root)


@pytest.mark.parametrize("field", ["objective_gain", "relative_objective_gain"])
def test_audit_checks_frozen_objective_gain_reconstruction(tmp_path, field):
    parent = trajectory()
    branch = audit(parent)
    branch["result"][field] += .01
    root = make_root(tmp_path, [parent, branch])
    with pytest.raises(ArtifactError, match="reconstructed"):
        report.collect(root)


def test_collect_checks_seals_without_repairing_artifacts(tmp_path):
    value = trajectory()
    root = make_root(tmp_path, [value])
    snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report.collect(root)
    assert snapshot == {path: path.read_bytes() for path in snapshot}
    path = root / "units" / value["unit_id"] / "summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ArtifactError, match="mismatch"):
        report.collect(root)


def test_incomplete_results_cannot_leak_intervals_or_tests(tmp_path, monkeypatch):
    skip_render(monkeypatch)
    units = cohort()
    stale = decision("results", analysis())
    root = make_root(tmp_path, units[:-1], stage="7D", incomplete=[units[-1], stale],
                     decisions=[decision("confirmation-ledger", dict(trajectory_ids=[u["unit_id"] for u in units]))])
    notebook = report.build_notebook(base_notebook(tmp_path), root)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "Descriptive progress only" in text and "Incomplete units (progress only" in text
    assert "p_value" not in text and "0.25" not in text and "[0.3, 0.6]" not in text
    assert "7/8" not in text  # Scheduled counters are not inferred from a ledger.
    assert "7/9" in text


def test_sealed_results_without_inference_ready_are_descriptive(tmp_path):
    stale = decision("results", {**analysis(), "inference_ready": False})
    root = make_root(tmp_path, decisions=[stale], stage="7D")
    notebook = report.build_notebook(base_notebook(tmp_path), root)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "Descriptive progress only" in text
    assert "p_value" not in text


@pytest.mark.parametrize("missing", ["ledger", "trajectory", "seal"])
def test_final_inference_requires_sealed_ledger_and_its_entire_cohort(tmp_path, missing, monkeypatch):
    skip_render(monkeypatch)
    units = cohort()
    decisions = [decision("results", analysis())]
    if missing != "ledger":
        decisions.append(decision("confirmation-ledger", dict(trajectory_ids=[u["unit_id"] for u in units])))
    root = make_root(tmp_path, units[:-1], stage="7D", decisions=decisions,
                     incomplete=[units[-1]] if missing == "seal" else [])
    with pytest.raises(ArtifactError, match="sealed confirmation ledger|incomplete confirmation cohort"):
        report.build_notebook(base_notebook(tmp_path), root)


def test_compact_notebook_has_three_rendered_figures_and_only_sealed_inference(tmp_path, monkeypatch):
    development = [trajectory(stream=stream, condition=condition) for stream in ("mixed", "base")
                   for condition in report.CONDITIONS]
    source = make_root(tmp_path, [*development, audit(development[0]), audit(development[-1])], name="development")
    units = cohort()
    decisions = [decision("results", analysis()),
                 decision("confirmation-ledger", dict(trajectory_ids=[u["unit_id"] for u in units]))]
    root = make_root(tmp_path, units, stage="7D", name="confirmation", decisions=decisions,
                     source="../development")
    assert report.collect(root)["development"]["root"] == source
    from mnist_experiment.fisher_memory import learner, metrics

    def forbidden(*args, **kwargs):
        pytest.fail("report must not train, infer, or open a writer")

    monkeypatch.setattr(learner, "fit_step", forbidden)
    monkeypatch.setattr(metrics, "paired_summary", forbidden)
    monkeypatch.setattr(Store, "__init__", forbidden)
    notebook = report.build_notebook(base_notebook(tmp_path), root)
    cells = [cell for cell in notebook.cells if report.TAG in cell.metadata.get("tags", [])]
    assert len(cells) == 8
    plots = [cell for cell in cells if cell.cell_type == "code"]
    assert len(plots) == 3
    assert all(cell.outputs[0].data["image/png"] for cell in plots)
    assert all(cell.execution_count is None and "import " not in cell.source for cell in plots)
    text = "\n".join(cell.source for cell in cells)
    assert "one-sided sign-flip p = 0.25" in text
    assert "better than learning nothing" in text
    assert "zero novel training exposures" in text
    assert "Both Running $A(n)$ panels evaluate the fixed rotated-digit test panel" in text
    assert "matched training images stay unrotated" in text
    assert "Base-only" not in text and "base-domain" not in text
    assert "Axis limits are set independently for each panel" in text
    assert "**Motivation:** Phases 5B and 5C" in cells[0].source
    assert "overfitting concern" in cells[0].source
    assert "numerical solve adequacy" in cells[0].source
    assert "working explanation, not a mechanism established" in cells[0].source
    assert "new_nll" in text and "omega_kappa" in text
    nbformat.validate(notebook)


def test_development_subtracts_each_replica_initial_and_keeps_policies_separate(monkeypatch):
    units = []
    for replica, initial in enumerate((2., 20.)):
        value = trajectory(replica, initial=initial)
        value.update(status="completed")
        units.append(value)
    other = trajectory(policy=200)
    other.update(status="completed")
    units.append(other)

    def capture(figure, caption):
        from matplotlib import pyplot as plt
        axis = figure.axes[0]
        assert len(axis.collections) == 2  # Active policy and the frozen origin only.
        assert axis.collections[0].get_offsets()[0, 1] == pytest.approx(-.3)
        assert not axis.texts
        assert "c=0.1" in [text.get_text() for text in axis.get_legend().texts]
        assert "Unrotated-only control" in figure.axes[2].get_title()
        assert "paired-event clock" in figure.axes[2].get_title()
        assert "Unrotated retention NLL" in figure.axes[2].get_xlabel()
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Captured")

    monkeypatch.setattr(report, "_plot_cell", capture)
    report._development_plot(units, policy_iterations=100)


def test_audit_labels_distinguish_unrotated_domain_from_training_control(monkeypatch):
    branches = [audit(trajectory(stream=stream)) for stream in ("mixed", "base")]

    def capture(figure, caption):
        from matplotlib import pyplot as plt
        assert figure.axes[1].get_title() == "Unrotated development holdout"
        labels = [text.get_text() for text in figure.axes[0].get_legend().texts]
        assert any("unrotated-only control" in label for label in labels)
        assert all("base" not in label and "replay_ewc" not in label for label in labels)
        assert "unrotated-only control triangles" in caption
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Captured")

    monkeypatch.setattr(report, "_plot_cell", capture)
    report._audit_plot(branches)


def test_confirmation_running_average_excludes_initial_and_separates_pseudo_clock(monkeypatch):
    units = cohort()
    for value in units:
        value.update(status="completed")

    def capture(figure, caption):
        from matplotlib import pyplot as plt
        axes = figure.axes
        assert list(axes[0].lines[0].get_xdata()) == [0, 1, 2]
        assert "Actual novel" in axes[0].get_xlabel()
        assert "zero novel exposures" in axes[4].get_xlabel()
        assert list(axes[8].lines[0].get_xdata()) == [1, 2]
        assert list(axes[8].lines[0].get_ydata()) == pytest.approx([2.5, 2.25])
        assert list(axes[8].lines[2].get_ydata()) == pytest.approx([2.8, 2.7])
        for axis in (axes[0], axes[4], axes[8], axes[9]):
            labels = [label.get_text() for label in axis.get_legend().texts]
            assert labels.count("Frozen initial model") == 1
            reference = next(line for line in axis.lines if line.get_label() == "Frozen initial model")
            assert reference.get_linestyle() == ":"
            assert list(reference.get_ydata()) == pytest.approx([3., 3.])
        assert "mixed training" in axes[8].get_title()
        assert "unrotated-only control" in axes[9].get_title()
        assert "rotated-test NLL" in axes[9].get_ylabel()
        for i, axis in enumerate(axes):
            assert axis.get_ylabel()
            assert "Base" not in axis.get_title()
            assert "base:" not in " ".join(tick.get_text() for tick in axis.get_yticklabels())
            if i < 8:
                expected = "Test NLL (nats)" if i % 4 < 2 else "Test accuracy (%)"
                assert axis.get_ylabel() == expected
        assert "Unrotated test NLL" in axes[1].get_title()
        assert "Unrotated-only control" in axes[4].get_title()
        assert any("Unrotated-only control" in tick.get_text() for tick in axes[11].get_yticklabels())
        assert "pseudo-clock" in axes[9].get_xlabel()
        assert "Inference withheld" in axes[10].texts[0].get_text()
        assert [bar.get_width() for bar in axes[11].patches] == [2., 1., 2., 1.]
        assert all("policy" not in line.get_label() for line in axes[0].lines)
        assert "%" in axes[2].yaxis.get_major_formatter()(.6)
        assert "cohorts need not be paired" in caption
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Captured")

    monkeypatch.setattr(report, "_plot_cell", capture)
    report._confirmation_plot(units, None)


def test_initial_scheduler_status_renders_before_jobs_are_known(tmp_path):
    root = make_root(tmp_path)
    path = root / "status.json"
    status = json.loads(path.read_text())
    status.update(stage="diagnostic", expected=None, aggregate_active_seconds=10800.,
                  current_job="initialization", heartbeat="2026-10-24T01:00:00+00:00")
    path.write_text(json.dumps(status))
    notebook = report.build_notebook(base_notebook(tmp_path), root)
    phase = [cell for cell in notebook.cells if report.TAG in cell.metadata.get("tags", [])]
    assert len(phase) == 8
    assert "0/pending" in phase[0].source
    assert "aggregate active **3.00 h** (current root 1.00 h)" in phase[0].source
    assert "Descriptive progress only" in phase[-2].source
    front = [cell for cell in notebook.cells if report.PROGRESS_TAG in cell.metadata.get("tags", [])]
    assert len(front) == 1 and len(front[0].source) < 600
    assert "3.00 h" in front[0].source and "initialization" in front[0].source
    assert "2026-10-24T01:00:00+00:00" in front[0].source


def native_ledger_and_cohort():
    units = cohort()
    mappings = []
    for value in units:
        meta = value["metadata"]
        choice = Treatment(**meta["treatment"])
        meta["treatment"] = choice.mapping()
        meta["treatment_id"] = choice.key
        name = f"trajectory-{report.CONFIRMATION}-{meta['replica']:04d}-{meta['stream']}-{choice.key}"
        value["unit_id"] = value["summary"]["unit_id"] = name
        if choice.mapping() not in mappings:
            mappings.append(choice.mapping())
    return dict(replicas=2, event_target=2, treatments=mappings), units


@pytest.mark.parametrize("mutation", [None, "treatment", "missing", "duplicate", "smoke"])
def test_native_recipe_ledger_validates_exact_frozen_cohort(tmp_path, monkeypatch, mutation):
    skip_render(monkeypatch)
    plan, units = native_ledger_and_cohort()
    value = analysis()
    if mutation == "treatment":
        units[0]["metadata"]["treatment"]["ridge_multiplier"] = .3
    elif mutation == "missing":
        units.pop()
    elif mutation == "duplicate":
        plan["treatments"].append(copy.deepcopy(plan["treatments"][0]))
    elif mutation == "smoke":
        plan["smoke"] = True
    root = make_root(tmp_path, units, stage="7D", decisions=[decision("results", value),
                     decision("confirmation-ledger", plan)])
    if mutation:
        with pytest.raises(ArtifactError):
            report.build_notebook(base_notebook(tmp_path), root)
    else:
        notebook = report.build_notebook(base_notebook(tmp_path), root)
        assert "one-sided sign-flip p =" in "\n".join(cell.source for cell in notebook.cells)


def test_confirmation_development_root_falls_back_to_sealed_settings_source(tmp_path):
    development = make_root(tmp_path, name="development")
    plan, _ = native_ledger_and_cohort()
    plan["source"] = dict(root=str(development))
    root = make_root(tmp_path, name="confirmation", stage="7D", decisions=[
        decision("confirmation-ledger", plan)])
    assert report.collect(root)["development"]["root"] == development


@pytest.mark.parametrize("mutation", ["initial", "duplicate", "asset"])
def test_rejects_unpaired_or_duplicate_trajectories(tmp_path, mutation):
    units = [trajectory(), trajectory(condition="finite_difference")]
    if mutation == "initial":
        units[1] = trajectory(condition="finite_difference", initial=4.)
    elif mutation == "duplicate":
        other = copy.deepcopy(units[0])
        other["unit_id"] = other["summary"]["unit_id"] = "duplicate-name"
        units.append(other)
    else:
        units[0]["metadata"]["asset_id"] = "asset-A"
        units[1]["metadata"]["asset_id"] = "asset-B"
    root = make_root(tmp_path, units)
    with pytest.raises(ArtifactError, match="paired initial|duplicate|unpaired"):
        report.collect(root)


@pytest.mark.parametrize("mutation", ["parent", "replica", "event", "duplicate"])
def test_audit_snapshots_remain_bound_to_their_original_parent(tmp_path, mutation):
    parent = trajectory()
    branch = audit(parent)
    units = [parent, branch]
    if mutation == "parent":
        branch["metadata"]["parent_id"] = "absent-parent"
    elif mutation == "replica":
        branch["metadata"]["replica"] = 1
    elif mutation == "event":
        branch["metadata"]["event"] = 3
    else:
        other = copy.deepcopy(branch)
        other["unit_id"] = "duplicate-audit"
        units.append(other)
    root = make_root(tmp_path, units)
    with pytest.raises(ArtifactError, match="parent trajectory|duplicate Phase 7 audit"):
        report.collect(root)


def test_audit_can_complete_while_parent_trajectory_is_still_incomplete(tmp_path):
    parent = trajectory()
    root = make_root(tmp_path, [audit(parent)], incomplete=[parent])
    data = report.collect(root)
    assert len(data["audits"]) == 1
    assert data["trajectories"][0]["rows"] == []


def test_partial_confirmation_rejects_foreign_recipe_without_releasing_inference(tmp_path):
    plan, units = native_ledger_and_cohort()
    units[0]["metadata"]["treatment"]["ridge_multiplier"] = .3
    root = make_root(tmp_path, units[:1], stage="7D", decisions=[decision("confirmation-ledger", plan)])
    with pytest.raises(ArtifactError, match="frozen ledger"):
        report.collect(root)


@pytest.mark.parametrize("same_ridge, expected_rows", [(False, 6), (True, 5)])
def test_visible_inference_has_only_core_human_readable_comparisons(tmp_path, same_ridge, expected_rows):
    replay = Treatment("replay_ewc", ridge_multiplier=.1)
    fd = Treatment("finite_difference", strength=160.89, ridge_multiplier=.1 if same_ridge else .3)
    effect = dict(n_pairs=2, mean_gain=.25, ci=[.1, .4], favorable_replicas=1,
                  paired_values=[dict(replica=0, gain=.5), dict(replica=1, gain=0.)])
    harm = dict(effect, paired_values=[dict(replica=0, gain=-.1), dict(replica=1, gain=0.)])
    value = dict(primary=dict(effect, p_value=.25), matched_ridge=effect, matched_ridge_is_primary=same_ridge,
                 retention_harm=harm, frozen_gain={choice.key: dict(mixed=effect) for choice in (replay, fd)},
                 base_deterioration={fd.key: dict(mixed=harm)},
                 source_lineage=dict(large_hidden_payload="MUST_NOT_APPEAR" * 1000),
                 arm_summaries=[dict(large_hidden_payload="ALSO_HIDDEN" * 1000)])
    data = dict(decisions={"results": dict(unit_id="regularization-results-v1", path=tmp_path),
                           "confirmation-ledger": dict(result=dict(baseline=replay.mapping(), treatment=fd.mapping()))})
    cell = report._inference(data, value)
    table = [line for line in cell.source.splitlines() if line.startswith("| ")]
    assert len(table) == expected_rows + 2
    assert "MUST_NOT_APPEAR" not in cell.source and "ALSO_HIDDEN" not in cell.source
    assert "source_lineage" not in cell.source and "arm_summaries" not in cell.source
    assert "FD acquisition gain over freezing" in cell.source and "Replay acquisition gain over freezing" in cell.source
    assert "FD unrotated deterioration from initialization" in cell.source
    assert "trajectory-average unrotated-test NLL" in cell.source
    assert "not an endpoint-only difference" in cell.source
    assert "1/2" in cell.source and "[+0.1000, +0.4000]" in cell.source
    assert ("same ridge" in cell.source) is not same_ridge
    assert "regularization-results-v1" in cell.source
    assert "primary interval favors FD" in cell.source
    assert "nominal frozen-reference interval also favors FD" in cell.source
    assert "not forgetting-free adaptation" in cell.source

    value["primary"]["ci"] = [-.1, .2]
    value["frozen_gain"][fd.key]["mixed"]["ci"] = [-.3, -.1]
    uncertain = report._inference(data, value).source
    assert "does not resolve an acquisition advantage" in uncertain
    assert "worse predictions on average than no adaptation" in uncertain


def test_visible_decisions_exclude_ledgers_and_audit_snapshot_payloads(tmp_path):
    decisions = {}
    for name in ("policy", "shortlist", "extension-0_1", "selection", "ledger", "confirmation-ledger",
                 "audit-initial", "audit-repaired", "timing", "results"):
        decisions[name] = dict(unit_id=f"regularization-{name}-v1", path=tmp_path / name,
                               result=dict(policy=dict(inner_steps=100, max_evaluations=150),
                                           snapshots=["HIDDEN_SNAPSHOT"] * 100))
    cell = report._decisions(dict(development=None, decisions=decisions))
    table = [line for line in cell.source.splitlines() if line.startswith("| ")]
    assert len(table) == 6  # Header, separator, and four allowed decisions.
    assert "HIDDEN_SNAPSHOT" not in cell.source
    assert "audit-initial" not in cell.source and "confirmation-ledger" not in cell.source
    assert "100 iterations / 150 closures" in cell.source


def test_collapsed_diagnostics_aggregate_replicas_and_cap_summary_rows(tmp_path):
    units = []
    for ridge_index in range(20):
        for replica in range(3):
            value = trajectory(replica=replica, ridge=(ridge_index + 1) / 10)
            value.update(status="completed", path=tmp_path / value["unit_id"])
            units.append(value)
    data = dict(root=tmp_path, development=None, decisions={}, pending=[], status={})
    cell = report._diagnostics(data, units, [])
    table = [line for line in cell.source.splitlines() if line.startswith("| ")]
    assert len(table) == 18
    assert "Showing 16 of 20 cohort summaries" in cell.source
    assert all("| 3 |" in row for row in table[2:])
    assert "<details>" in cell.source and "</details>" in cell.source
    assert "Complete trajectory/audit JSON directory" in cell.source
    assert not any(unit["unit_id"] in cell.source for unit in units)


def test_development_ridge_grid_excludes_retuned_and_superseded_candidates(monkeypatch):
    values = [trajectory(condition=condition, policy=200) for condition in report.CONDITIONS]
    retuned = trajectory(condition="finite_difference", policy=200)
    retuned["metadata"]["treatment"]["strength"] = 482.67
    retuned["metadata"]["treatment_id"] = "retuned"
    values.extend([retuned, trajectory(policy=100)])
    for value in values:
        value.update(status="completed")

    def capture(figure, caption):
        from matplotlib import pyplot as plt
        left, right = figure.axes[:2]
        assert len(left.collections) == 3  # Two methods plus frozen origin.
        assert not left.texts
        assert list(right.lines[0].get_xdata()) == [160.89, 482.67]
        assert all("policy" not in text.get_text() for text in left.get_legend().texts)
        assert "200 iterations" in caption
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Captured")

    monkeypatch.setattr(report, "_plot_cell", capture)
    report._development_plot(values, policy_iterations=200)
