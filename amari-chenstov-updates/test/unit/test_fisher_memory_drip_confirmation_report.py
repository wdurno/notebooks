"""Confirmation inference, pairing, immutable-artifact and notebook checks."""

import copy
import json

import nbformat
import pytest

from mnist_experiment.fisher_memory import drip_confirmation_report as report
from mnist_experiment.fisher_memory import report as presentation
from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store
from mnist_experiment.fisher_memory.config import Treatment
from mnist_experiment.fisher_memory.drip_runner import trajectory_id
from mnist_experiment.fisher_memory.metrics import paired_summary
from src.seeding import derive_component_seed


def ledger(smoke=False):
    baseline = Treatment("replay_ewc", inner_steps=100, max_evaluations=150)
    selected = .269 if smoke else 160.89000000000001
    treatment = Treatment("finite_difference", selected, inner_steps=100, max_evaluations=150)
    return dict(schema_version=1, protocol=report.PROTOCOL, family=report.FAMILY,
                replicas=2 if smoke else 12, event_target=2 if smoke else 100,
                baseline=baseline.mapping(), treatment=treatment.mapping(), selected_strength=selected,
                bootstrap_samples=101, inference_seed=15, alpha=.05, source={"selection_seal": "pilot"})


def unit(plan, replica, stream, condition, rotated, base=.4):
    treatment = Treatment(**plan["baseline" if condition == "replay_ewc" else "treatment"])
    name = trajectory_id(report.FAMILY, replica, stream, treatment)
    meta = dict(schema_version=1, kind="drip_trajectory", phase="5C", protocol=report.PROTOCOL,
                family=report.FAMILY, replica=replica, stream=stream, condition=condition,
                treatment=treatment.mapping(), treatment_id=treatment.key, event_target=plan["event_target"])
    rows = [dict(paired_rotated_event=event, actual_rotated_events=event if stream == "mixed" else 0,
                 rotated_nll=rotated if event else 4., base_nll=base if event else .8,
                 rotated_accuracy=.6, base_accuracy=.9)
            for event in range(plan["event_target"] + 1)]
    summary = dict(schema_version=1, unit_id=name, replica=replica, stream=stream,
                   condition=condition, strength=treatment.strength, event_target=plan["event_target"],
                   rotated_auc=rotated, base_auc=base, final_rotated_nll=rotated, final_base_nll=base,
                   final_rotated_accuracy=.6, final_base_accuracy=.9, active_seconds=120.,
                   closure_calls=30, per_class_events={str(k): 10 for k in range(10)}, exposure=1080,
                   solver_failures=0, nonfinite=False, under_solved=5)
    return dict(unit_id=name, metadata=meta, rows=rows, summary=summary, completed=True,
                fit=[dict(paired_rotated_events=1, rotated_prequential_nll=1.),
                     dict(paired_rotated_events=plan["event_target"], rotated_prequential_nll=3.)])


def cohort(plan, gains=None):
    gains = gains or [1.] * plan["replicas"]
    units = []
    for replica, gain in enumerate(gains):
        units.extend([unit(plan, replica, "mixed", "replay_ewc", 3.),
                      unit(plan, replica, "mixed", "finite_difference", 3. - gain, .42),
                      unit(plan, replica, "base", "replay_ewc", 3.),
                      unit(plan, replica, "base", "finite_difference", 2.75)])
    return units


def seal(store, value):
    name = value["unit_id"]
    store.begin(name, value["metadata"])
    for filename, payload in (("metadata.json", value["metadata"]), ("metrics.json", value["rows"]),
                              ("summary.json", value["summary"]), ("fit.json", value["fit"])):
        store.write_json(name, filename, payload)
    store.complete(name)


def store_root(tmp_path, plan, units):
    root = tmp_path / "run"
    with Store(root, {"protocol": report.PROTOCOL}) as store:
        store.begin(report.LEDGER, {"kind": "drip_confirmation_ledger"})
        store.write_json(report.LEDGER, "result.json", plan)
        store.complete(report.LEDGER)
        for value in units:
            seal(store, value)
        store.write_json(None, "status.json", dict(status="running", stage="confirmation",
                                                   active_seconds=3600, current_job=None, changes=[]))
    return root


def test_final_effect_signs_exact_p_and_frozen_settings():
    plan = ledger()
    result = report.analyze(cohort(plan), plan)
    assert result["complete"] and result["inference_ready"]
    assert result["completed"] == result["expected"] == 48
    assert result["n_pairs"] == 12
    assert result["primary"]["mean_gain"] == 1.
    assert result["primary"]["p_value"] == 1 / 4096
    assert result["primary"]["ci"] == [1., 1.]
    assert result["primary"]["standardized_effect"] is None
    assert result["primary"]["favorable_replicas"] == 12
    assert result["comparisons"]["base_only_gain"]["mean_gain"] == .25
    assert result["comparisons"]["interaction"]["mean_gain"] == .75
    assert result["comparisons"]["base_retention_harm"]["mean_gain"] == pytest.approx(.02)
    assert all(value["p_value"] is None for key, value in result["comparisons"].items()
               if key != "mixed_gain")
    assert result["selected_strength"] == plan["selected_strength"]
    assert result["source_lineage"] == plan["source"]
    assert result["bootstrap_samples"] == plan["bootstrap_samples"]
    assert result["primary"]["seed"] == derive_component_seed(plan["inference_seed"], "bootstrap:mixed_gain")
    json.dumps(result, allow_nan=False)


def test_named_bootstrap_matches_shared_estimator_and_asymmetry_is_data_driven():
    plan = ledger()
    gains = [.1] * 11 + [2.]
    result = report.analyze(cohort(plan, gains), plan)
    primary = result["primary"]
    reference = paired_summary(dict(enumerate(gains)), dict.fromkeys(range(12), 0.),
                               seed=derive_component_seed(plan["inference_seed"], "bootstrap:mixed_gain"),
                               resamples=plan["bootstrap_samples"])
    assert primary["ci"] == pytest.approx(reference["ci"])
    assert primary["standardized_effect"] > 0
    assert primary["asymmetry_warning"] and primary["sample_skewness"] > 1
    assert "exchangeable signs" in result["inference_caveat"]
    assert result == report.analyze(list(reversed(cohort(plan, gains))), plan)


@pytest.mark.parametrize("differences, expected", [([1., 1.], .25), ([-1., -1.], 1.),
                                                   ([0., 0.], 1.), ([1., -1.], .75)])
def test_sign_flip_enumerates_ties(differences, expected):
    assert report._exact_sign_flip(differences) == expected


def test_no_inference_at_47_even_when_all_12_primary_pairs_are_sealed():
    plan = ledger()
    units = cohort(plan)
    units[-1].update(completed=False, rows=[], summary=None)
    result = report.analyze(units, plan)
    assert result["completed"] == 47 and result["n_pairs"] == 11
    assert not result["complete"] and not result["inference_ready"]
    assert all(value["ci"] is None and value["p_value"] is None
               for value in result["comparisons"].values())
    assert result["health"]["sealed_trajectories"] == 47


def test_smoke_accepts_own_selection_without_inference():
    plan = ledger(smoke=True)
    result = report.analyze(cohort(plan), plan)
    assert result["complete"] and result["smoke"] and not result["inference_ready"]
    assert result["primary"]["p_value"] is None and result["primary"]["ci"] is None


def test_health_retains_all_sealed_trajectories_and_weights_prequential_by_events():
    plan = ledger()
    units = cohort(plan)
    units[0]["summary"].update(solver_failures=2, nonfinite=True)
    result = report.analyze(units, plan)
    assert result["complete"] and result["n_pairs"] == 12
    assert result["health"] == dict(sealed_trajectories=48, solver_failures=2,
                                     nonfinite_trajectories=1, under_solved_updates=240,
                                     status="failures_recorded")
    assert result["arm_summaries"][0]["rotated_prequential_nll"] == pytest.approx(2.98)
    assert result["arm_summaries"][0]["per_class_events"]["9"] == 10


@pytest.mark.parametrize("mutation", ["pilot", "treatment", "duplicate", "missing_event", "auc", "schema"])
def test_rejects_foreign_pairs_wrong_treatment_and_incompatible_metrics(mutation):
    plan = ledger()
    units = cohort(plan)
    if mutation == "pilot":
        units[0]["metadata"]["family"] = "drip_development"
    elif mutation == "treatment":
        units[1]["metadata"]["treatment"]["strength"] = .269
    elif mutation == "duplicate":
        units.append(copy.deepcopy(units[0]))
    elif mutation == "missing_event":
        units[0]["rows"].pop()
    elif mutation == "auc":
        units[0]["summary"]["rotated_auc"] += .1
    else:
        units[0]["summary"]["schema_version"] = 2
    with pytest.raises(ArtifactError):
        report.analyze(units, plan)


def test_collect_verifies_seals_and_requires_complete_ledger(tmp_path):
    plan = ledger(smoke=True)
    units = cohort(plan)
    root = store_root(tmp_path, plan, units)
    data = report.collect(root)
    assert data["expected"] == data["completed"] == 8
    assert report.analyze(data["units"], data["ledger"])["complete"]
    path = root / "units" / units[0]["unit_id"] / "summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ArtifactError):
        report.collect(root)


def test_collect_rejects_ledger_schema(tmp_path):
    plan = ledger(smoke=True)
    plan["schema_version"] = 9
    root = store_root(tmp_path, plan, [])
    with pytest.raises(ArtifactError, match="ledger schema"):
        report.collect(root)


def test_notebook_preserves_prior_cells_replaces_own_section_and_clears_stale_text(tmp_path):
    plan = ledger(smoke=True)
    root = store_root(tmp_path, plan, cohort(plan))
    base = tmp_path / "base.ipynb"
    title = nbformat.v4.new_markdown_cell("# Prior work")
    prior = nbformat.v4.new_code_cell("saved = 1", execution_count=1, outputs=[
        nbformat.v4.new_output("stream", name="stdout", text="Preserved output")])
    pilot = nbformat.v4.new_markdown_cell("## Phase 5B\nPilot. Phase 5C has not started.",
                                          metadata={"tags": ["plan15_phase5b"]})
    nbformat.write(nbformat.v4.new_notebook(cells=[title, pilot, prior]), base)
    first = report.build_notebook(base, root)
    nbformat.write(first, base)
    second = report.build_notebook(base, root)
    nbformat.validate(second)
    tagged = [cell for cell in second.cells if report.TAG in cell.metadata.get("tags", [])]
    assert len(tagged) == 5
    assert len(first.cells) == len(second.cells)
    assert second.cells[0] == title and second.cells[-1] == prior
    assert second.cells[-2].source == "## Phase 5B\nPilot."
    assert "8/8" in second.cells[1].source and "paired n = 2/2" in second.cells[1].source
    assert "SMOKE ACCEPTANCE ONLY" in second.cells[1].source
    assert "0 actual rotated" not in second.cells[1].source  # Counts are explained in words.
    assert "zero actual rotated exposures" in second.cells[1].source
    assert all("p =" not in cell.source for cell in tagged)
    images = [cell for cell in tagged if cell.cell_type == "code"]
    assert len(images) == 2
    assert all(cell.outputs[0].data["image/png"] for cell in images)


def test_final_notebook_is_compact_and_only_primary_has_p_value(tmp_path, monkeypatch):
    plan = ledger()
    units = cohort(plan)
    monkeypatch.setattr(report, "collect", lambda root: dict(ledger=plan, status={}, units=units))
    base = tmp_path / "base.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Existing")]), base)
    notebook = report.build_notebook(base, tmp_path)
    assert len(notebook.cells) <= 7
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "48/48" in source and "paired n = 12/12" in source
    assert source.count("sign-flip p =") == 1
    assert "0.000244141" in source
    assert "mean / sample SD" not in source
    assert "Individual gains and detailed effect diagnostics remain in result JSON" in source


def test_visual_efficiency_uses_registered_score_pairing_and_actual_mixed_cost(monkeypatch):
    plan = ledger()
    units = cohort(plan, [.5] * 12)
    for value in units:
        value["summary"]["active_seconds"] = (360. if value["metadata"]["condition"] == "finite_difference"
                                                 else 120.) if value["metadata"]["stream"] == "mixed" else 99999.
        if value["metadata"]["stream"] == "mixed":
            first, second = (5., 1.) if value["metadata"]["condition"] == "replay_ewc" else (4., 1.)
            value["rows"][1]["rotated_nll"] = first
            value["rows"][2]["rotated_nll"] = second
    result = report.analyze(units, plan)

    def check_plot(figure, caption):
        from matplotlib import pyplot as plt
        learning, acquisition, effect, compute = figure.axes
        assert list(learning.lines[0].get_xdata()) == list(range(101))
        assert list(learning.lines[0].get_ydata()) == [4., 5., 1., *([3.] * 98)]
        assert list(learning.lines[1].get_ydata()) == [4., 4., 1., *([2.5] * 98)]
        assert list(acquisition.lines[0].get_xdata()) == list(range(1, 101))
        assert list(acquisition.lines[0].get_ydata()) == [5., *([3.] * 99)]
        assert list(acquisition.lines[1].get_ydata()) == [4., *([2.5] * 99)]
        for line, arm in zip(acquisition.lines[:2], result["arm_summaries"][:2]):
            assert line.get_ydata()[-1] == pytest.approx(arm["rotated_auc"])
        assert list(learning.lines[2].get_ydata()) == [4., 4.]
        assert list(acquisition.lines[2].get_ydata()) == [4., 4.]
        assert "A_m^R(n)" in acquisition.get_ylabel()
        assert "Held-out NLL after batch updates" == learning.get_title()
        assert list(effect.collections[0].get_offsets()[:, 0]) == [.5] * 12
        assert list(effect.collections[1].get_segments()[0][:, 0]) == [.5, .5]
        assert [bar.get_width() for bar in compute.patches] == [2., 6.]
        assert "3.00x" in compute.get_title()
        plt.close(figure)
        return nbformat.v4.new_code_cell("# rendered")

    monkeypatch.setattr(presentation, "_plot_cell", check_plot)
    presentation._phase5c_efficiency_plot(units, result)


def presentation_fixture(tmp_path, monkeypatch):
    plan = ledger()
    units = cohort(plan)
    monkeypatch.setattr(report, "collect", lambda root: dict(ledger=plan, status={}, units=units))
    result = report.analyze(units, plan)
    sealed_path = tmp_path / "units" / report.RESULTS / "result.json"
    sealed_path.parent.mkdir(parents=True)
    sealed_path.write_text(json.dumps(result))
    base = tmp_path / "base.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Prior study")]), base)
    original = report.build_notebook(base, tmp_path)
    original.cells.append(nbformat.v4.new_markdown_cell(
        "### Sample and compute efficiency\n\n| Legacy metric | Value |\n| --- | --- |\n| NLL | 0.3 |"))
    original.cells.append(nbformat.v4.new_code_cell("preserved = True"))
    nbformat.write(original, base)
    return base, sealed_path


def test_visual_presentation_is_idempotent_preserves_earlier_work_and_sealed_results(tmp_path, monkeypatch):
    base, sealed = presentation_fixture(tmp_path, monkeypatch)
    before = sealed.read_bytes()
    first = presentation.build_phase5c_presentation(base, tmp_path)
    nbformat.write(first, base)
    second = presentation.build_phase5c_presentation(base, tmp_path)
    tagged = [cell for cell in second.cells if report.TAG in cell.metadata.get("tags", [])]
    assert len(tagged) == 6
    assert len(first.cells) == len(second.cells)
    assert second.cells[0].source == "# Prior study"
    assert second.cells[2].source == "preserved = True"
    assert second.cells[3:] == tagged
    source = "\n".join(cell.source for cell in tagged)
    assert "## Phase 5C: Sample and compute efficiency" in source
    assert "images 1 through 100" in source
    assert r"A_m^R(n)=\frac{1}{n}\sum_{k=1}^{n}R_m^R(k)" in source
    assert "**2.000 for FD KL**" in source
    assert "**3.000 for replay/EWC**" in source
    assert "33.3% lower; a descriptive ratio of means" in source
    assert "same buffer" in source
    assert "Starting-point context" in source
    assert "not necessarily net improvement over initialization" in source
    assert "does not estimate a percentage of samples saved" in source
    assert "whole mixed stream" in source
    assert source.count("<details>") == 2
    assert "Evaluation protocol and comparable NLLs" in source
    assert "official MNIST test split" in source
    assert "identical test images, labels, rotations, and class weights" in source
    assert "not loss on the incoming batch or replay buffer" in source
    assert "**excludes EWC, Ridge, and KL penalties**" in source
    assert r"\ell_r^R(\theta)=-\sum_{c=0}^{9}\frac{q_c}{|H_{r,c}|}" in source
    assert r"F_r^R&=\ell_r^R(\theta_{r,0})" in source
    assert r"E_{r,m}^R&=\ell_r^R(\theta_{r,m,b_r(N)})" in source
    assert r"A_{r,m}^R(N)&=\frac{1}{N}\sum_{k=1}^{N}\ell_r^R(\theta_{r,m,b_r(k)})" in source
    assert "equal-weight means over 12 replicas" in source
    assert source.count("**Execution health:**") == 1
    assert "240 under-solved updates" in source
    assert "selected FD coefficient is 160.89" in source
    visible = "\n".join(cell.source for cell in tagged if not cell.source.startswith("<details>"))
    assert "| Contrast |" not in visible and "per-digit" not in visible.lower()
    assert "Earlier study: sample and compute audit" in second.cells[1].source
    assert "| NLL | 0.3 |" in second.cells[1].source
    assert "Phase 5C at the end" in second.cells[1].source
    assert before == sealed.read_bytes()
    assert len([cell for cell in tagged if cell.cell_type == "code"]) == 2


def test_chronological_order_preserves_phase_blocks_outputs_and_original_cells():
    def block(label, *, tagged):
        metadata = {"tags": [f"plan15_phase{label.lower()}"]} if tagged else {}
        return [nbformat.v4.new_markdown_cell(f"## Phase {label}: Results", metadata=metadata),
                nbformat.v4.new_code_cell(
                    f"# Plot for {label}", metadata=copy.deepcopy(metadata), execution_count=3,
                    outputs=[nbformat.v4.new_output("display_data", data={"text/plain": f"Plot {label}"})]),
                nbformat.v4.new_markdown_cell(f"Caption for {label}", metadata=copy.deepcopy(metadata))]

    original = [nbformat.v4.new_markdown_cell("# Original study"),
                nbformat.v4.new_markdown_cell("## Original conclusion")]
    blocks = {label: block(label, tagged=label not in ("5", "5A"))
              for label in ("5", "5A", "5B", "5C", "6", "10")}
    notebook = nbformat.v4.new_notebook(cells=[original[0], *blocks["5C"], *blocks["5B"],
                                              original[1], *blocks["10"], *blocks["6"],
                                              *blocks["5"], *blocks["5A"]])
    before = {cell.id: copy.deepcopy(cell) for cell in notebook.cells}
    presentation.order_notebook_chronologically(notebook)
    expected = [*original, *(cell for label in ("5", "5A", "5B", "5C", "6", "10")
                            for cell in blocks[label])]
    assert notebook.cells == expected
    assert {cell.id: cell for cell in notebook.cells} == before
    assert notebook.metadata["plan15_section_order"] == "oldest_to_newest"
    once = copy.deepcopy(notebook)
    presentation.order_notebook_chronologically(notebook)
    assert notebook == once
    nbformat.validate(notebook)


def test_visual_presentation_exports_standalone_png_with_reserved_publisher(tmp_path, monkeypatch):
    base, sealed = presentation_fixture(tmp_path, monkeypatch)
    before = sealed.read_bytes()
    published = {}
    target = presentation.refresh_phase5c(tmp_path, base, publish=lambda path, data: published.update({path: data}))
    assert target == tmp_path / "reports" / "fisher_memory_results.ipynb"
    assert published[tmp_path / "reports" / "phase5c_efficiency.png"].startswith(b"\x89PNG\r\n\x1a\n")
    nbformat.validate(nbformat.reads(published[target].decode(), as_version=4))
    assert before == sealed.read_bytes()


def test_visual_presentation_rejects_incomplete_or_disagreeing_analysis(tmp_path, monkeypatch):
    base, sealed = presentation_fixture(tmp_path, monkeypatch)
    data = json.loads(sealed.read_text())
    data["primary"]["mean_gain"] += 1
    sealed.write_text(json.dumps(data))
    with pytest.raises(presentation.ArtifactError, match="disagrees with sealed"):
        presentation.build_phase5c_presentation(base, tmp_path)
    plan = ledger()
    units = cohort(plan)[:-1]
    monkeypatch.setattr(report, "collect", lambda root: dict(ledger=plan, status={}, units=units))
    with pytest.raises(presentation.ArtifactError, match="complete production cohort"):
        presentation.build_phase5c_presentation(base, tmp_path)
