"""Phase 7E scalar-only mathematics, immutable recovery, and human-facing output."""

import json

import nbformat
import numpy as np
import pytest

from mnist_experiment import plan15_efficiency as analysis
from mnist_experiment import plan15_efficiency_report as report
from mnist_experiment.fisher_memory import regularization_report as phase7
from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store, iter_units
from mnist_experiment.fisher_memory.config import Treatment, content_hash
from mnist_experiment.fisher_memory.control import StopRequested


def curves():
    n = np.arange(101)
    values = {"replay_ewc": [], "finite_difference": []}
    for replica in range(12):
        for method, slope in (("replay_ewc", -.005), ("finite_difference", -.01)):
            curve = 2 + replica * .01 + slope * n
            curve[0] = 3 + replica * .01
            values[method].append(curve.tolist())
    return values


def established(values):
    arms = []
    for method in analysis.METHODS:
        for stream in ("mixed", "base"):
            arms.append(dict(condition=method, stream=stream, treatment_id=method,
                             rotated_auc=float(np.mean(np.asarray(values[method])[:, 1:])),
                             online_seconds=2 if method == "finite_difference" else 1))
    return dict(schema_version=1, complete=True, inference_ready=True, expected=48, completed=48, n_pairs=12,
                primary=dict(mean_gain=.2525, ci=[.2, .3], favorable_replicas=12, n_pairs=12, p_value=.000244),
                arm_summaries=arms,
                frozen_gain=dict(finite_difference=dict(mixed=dict(mean_gain=1.5, ci=[1.4, 1.6]))),
                base_deterioration=dict(finite_difference=dict(mixed=dict(mean_gain=.06, ci=[.05, .07]))))


def inputs():
    values = curves()
    return dict(curves=values, established=established(values), roots=[])


def identity(dependencies=None):
    return dict(protocol=analysis.PROTOCOL, config_hash=content_hash(analysis.CONFIG),
                dependencies_hash=content_hash(dependencies or {}))


def seals(root):
    return {unit["unit_id"]: unit["completion"]["seal_hash"] for unit in iter_units(root, completed_only=True, full=True)}


def test_exact_linear_fit_full_window_excludes_initialization():
    curve = 2 - .005 * np.arange(101)
    curve[0] = 100_000
    fit = analysis.fit_curve(curve)
    assert fit["intercept"] == pytest.approx(2)
    assert fit["slope"] == pytest.approx(-.005)
    assert fit["rmse"] < 1e-14
    changed = curve.copy()
    changed[1] += 1
    assert analysis.fit_curve(changed)["slope"] != pytest.approx(fit["slope"])


def test_exact_target_inversion():
    values = curves()
    point = analysis.match_curves(values["replay_ewc"][0], values["finite_difference"][0])
    assert point["target"] == pytest.approx(1.5)
    assert point["crossing"] == pytest.approx(50)
    assert point["saved"] == pytest.approx(50)
    assert point["saved_percent"] == pytest.approx(50)
    assert point["inversion_valid"] and point["in_range"]


@pytest.mark.parametrize("kind", ["flat", "positive", "near_zero", "negative_fit", "already_met", "outside"])
def test_inversion_guards_keep_raw_numerical_outputs(kind):
    values = curves()
    replay, fd = (np.asarray(values[method][0]) for method in analysis.METHODS)
    if kind == "flat":
        fd[1:] = 2
    elif kind == "positive":
        fd[1:] = 1 + .005 * np.arange(1, 101)
    elif kind == "near_zero":
        fd[1:] = 2 - 1e-14 * np.arange(1, 101)
    elif kind == "negative_fit":
        fd[1:] = 0
        fd[1] = 100
    elif kind == "already_met":
        replay[0] = fd[0] = 1
    else:
        fd[1:] = 1.2 - .001 * np.arange(1, 101)
    point = analysis.match_curves(replay, fd)
    if kind == "outside":
        assert point["inversion_valid"] and not point["in_range"]
        assert point["crossing"] < 0 and point["status"] == "range_limited"
    else:
        assert not point["inversion_valid"]
        assert point["issues"]
        if kind == "already_met":
            assert point["crossing"] == pytest.approx(50)  # Numerical output is not erased.


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_loss_fails_clearly(bad):
    curve = curves()["replay_ewc"][0]
    curve[5] = bad
    with pytest.raises(ArtifactError):
        analysis.fit_curve(curve)


def test_indices_are_deterministic_named_and_paired():
    indices = analysis.resample_indices()
    assert indices == analysis.resample_indices()
    assert indices != analysis.resample_indices(42)
    assert np.asarray(indices).shape == (2000, 12)
    assert analysis.CONFIG["seed_name"] == analysis.SEED_NAME


def test_bootstrap_reestimates_target_initial_and_both_fits():
    indices = [[0] * 12, [11] * 12] * 1000
    values = curves()
    for replica in range(12):
        values["replay_ewc"][replica][1:] = (2 + .01 * replica -
                (.005 + .0001 * replica) * np.arange(1, 101)).tolist()
    result, draws = analysis.analyze(values, indices)
    for number, replica in ((0, 0), (1, 11)):
        expected = analysis.match_curves(*(values[method][replica] for method in analysis.METHODS))
        for field in ("target", "initial", "crossing"):
            assert draws[number][field] == pytest.approx(expected[field])
        for method in analysis.METHODS:
            assert draws[number]["fits"][method]["slope"] == pytest.approx(expected["fits"][method]["slope"])
    assert draws[0]["target"] != draws[1]["target"]
    assert draws[0]["initial"] != draws[1]["initial"]
    assert result["bootstrap"]["ci"] is not None


def test_even_one_invalid_draw_makes_ordinary_interval_unavailable():
    values = curves()
    values["finite_difference"][0][1:] = [2.] * 100
    indices = [[0] * 12] + [[11] * 12] * 1999
    result, draws = analysis.analyze(values, indices)
    assert len(draws) == 2000 and result["bootstrap"]["invalid"] == 1
    assert result["bootstrap"]["ci"] is None
    assert result["bootstrap"]["interval_status"] == "unavailable_invalid_draws"


def test_initial_reference_guard_is_applied_in_each_draw():
    values = curves()
    values["replay_ewc"][0][0] = values["finite_difference"][0][0] = 1
    result, draws = analysis.analyze(values, [[0] * 12] + [[11] * 12] * 1999)
    assert result["bootstrap"]["invalid"] == 1
    assert "target_already_met_without_adaptation" in draws[0]["issues"]
    assert result["bootstrap"]["ci"] is None


def test_finite_outside_crossings_are_in_intervals_not_clipped_or_filtered():
    values = curves()
    for replica in range(12):
        values["finite_difference"][replica][1:] = (1.2 + replica * .01 - .001 * np.arange(1, 101)).tolist()
    result, draws = analysis.analyze(values, analysis.resample_indices())
    assert result["bootstrap"]["invalid"] == 0
    assert result["bootstrap"]["outside_range"] == 2000
    assert result["bootstrap"]["ci"]["crossing"][1] < 0
    assert result["bootstrap"]["interval_status"] == "range_limited"


def test_invalid_indices_and_unpaired_initialization_are_rejected():
    with pytest.raises(ArtifactError):
        analysis.analyze(curves(), [[12] * 12] * 2000)
    values = curves()
    values["finite_difference"][0][0] += 1
    with pytest.raises(ArtifactError):
        analysis.analyze(values, analysis.resample_indices())


def test_resume_skips_inference_preserves_seals_and_review_is_immutable(tmp_path, monkeypatch):
    root = tmp_path / "analysis"
    data = inputs()
    with Store(root, identity()) as store:
        first = analysis.execute(store, data, {})
    original = seals(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed bootstrap repeated")

    monkeypatch.setattr(analysis, "analyze", forbidden)
    with Store(root, identity(), resume=True) as store:
        second = analysis.execute(store, data, {}, fit_review="approximate", fit_note="Toy curves are exactly linear.")
    assert second == first
    assert all(seals(root)[name] == seal for name, seal in original.items())
    with Store(root, identity(), resume=True) as store:
        with pytest.raises(ArtifactError, match="review already sealed"):
            analysis.execute(store, data, {}, fit_review="poor", fit_note="Changed assessment.")


def test_interrupted_bootstrap_resumes_to_same_results(tmp_path, monkeypatch):
    data, dependencies = inputs(), {}
    root = tmp_path / "resumed"
    original_analyze = analysis.analyze

    def interrupted(values, indices, **kwargs):
        def stop(count):
            if count == 100:
                raise StopRequested("test interruption")
        return original_analyze(values, indices, progress=stop)

    monkeypatch.setattr(analysis, "analyze", interrupted)
    with Store(root, identity()) as store:
        with pytest.raises(StopRequested):
            analysis.execute(store, data, dependencies)
    prior = seals(root)
    assert set(prior) == {analysis.PLAN}
    monkeypatch.setattr(analysis, "analyze", original_analyze)
    with Store(root, identity(), resume=True) as store:
        resumed = analysis.execute(store, data, dependencies)
    with Store(tmp_path / "fresh", identity()) as store:
        fresh = analysis.execute(store, data, dependencies)
        fresh_draws = json.loads((store.unit_path(analysis.ANALYSIS) / "draws.json").read_text())
    for result in (resumed, fresh):
        for field in ("started_at", "completed_at"):
            result.pop(field)
    assert resumed == fresh
    assert json.loads((root / "units" / analysis.ANALYSIS / "draws.json").read_text()) == fresh_draws
    assert seals(root)[analysis.PLAN] == prior[analysis.PLAN]


def test_dependency_corruption_is_detected_without_repair(tmp_path):
    root = tmp_path / "dependency"
    with Store(root, {"toy": 1}) as store:
        store.begin("science", {"kind": "toy"})
        store.write_json("science", "result.json", {"nll": 1})
        store.complete("science")
    before = analysis.dependency_snapshot([root], full=True)
    assert before == analysis.dependency_snapshot([root], full=True)
    path = root / "units/science/result.json"
    path.write_text('{"nll": 2}\n')
    with pytest.raises(ArtifactError, match="checksum|size"):
        analysis.dependency_snapshot([root], full=True)
    assert path.read_text() == '{"nll": 2}\n'


def notebook(path):
    cells = [nbformat.v4.new_markdown_cell("# Study"),
             nbformat.v4.new_markdown_cell("Old phase", metadata={"tags": ["plan15_phase5c"]}),
             nbformat.v4.new_markdown_cell("Phase 7", metadata={"tags": [phase7.TAG]}),
             nbformat.v4.new_code_cell("# Prior plot", execution_count=1, outputs=[
                 nbformat.v4.new_output("stream", name="stdout", text="Unchanged output")],
                 metadata={"tags": [phase7.TAG]}),
             nbformat.v4.new_markdown_cell("Future phase")]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    return cells


def test_compact_chronological_render_does_not_train_fit_or_bootstrap(tmp_path, monkeypatch):
    root, path = tmp_path / "analysis", tmp_path / "results.ipynb"
    with Store(root, identity()) as store:
        analysis.execute(store, inputs(), {}, fit_review="approximate", fit_note="Exact toy linear fits.")
    old_cells = notebook(path)
    before = seals(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("render must not perform inference/training")

    from mnist_experiment.fisher_memory import learner, metrics
    monkeypatch.setattr(analysis, "analyze", forbidden)
    monkeypatch.setattr(analysis, "fit_curve", forbidden)
    monkeypatch.setattr(learner, "fit_step", forbidden)
    monkeypatch.setattr(metrics, "paired_summary", forbidden)
    book = report.build_notebook(path, root)
    cells = [cell for cell in book.cells if report.TAG in cell.metadata.get("tags", [])]
    assert len(cells) == 3 and sum(cell.cell_type == "code" for cell in cells) == 1
    assert [cell for cell in book.cells if report.TAG not in cell.metadata.get("tags", [])] == old_cells
    assert book.cells.index(cells[0]) == 4  # After Phase 7, before a future phase.
    text = "\n".join(cell.source for cell in cells)
    assert "0.2525 nats" in text and "[0.2000, 0.3000]" in text
    assert "50.0" in text and "post hoc" in text and "not mean first-passage" in text
    assert cells[1].outputs[0].data.get("image/png")
    nbformat.write(book, path)
    repeated = report.build_notebook(path, root)
    assert len(repeated.cells) == len(book.cells)
    assert seals(root) == before


def test_diagnostic_includes_legends_and_full_mean_and_fit_windows(monkeypatch):
    result, _ = analysis.analyze(curves(), analysis.resample_indices())
    captured = {}

    def inspect(figure, caption):
        from matplotlib import pyplot as plt
        axis = figure.axes[0]
        captured["figure_legends"] = len(figure.legends)
        captured["labels"] = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        captured["x"] = [list(line.get_xdata()) for line in axis.lines[:4]]
        plt.close(figure)
        return nbformat.v4.new_code_cell("# Fixture")

    monkeypatch.setattr(report, "_plot_cell", inspect)
    report.diagnostic(result)
    assert captured["figure_legends"] == 1
    assert len(captured["labels"]) == 7
    assert captured["x"][0] == captured["x"][2] == list(range(101))
    assert captured["x"][1] == captured["x"][3] == list(range(1, 101))


def test_incomplete_and_incompatible_render_fail_clearly(tmp_path):
    root = tmp_path / "incomplete"
    with Store(root, identity()) as store:
        store.begin(analysis.PLAN, {"kind": "plan"})
    with pytest.raises(ArtifactError, match="sealed plan"):
        report.load(root)


@pytest.mark.parametrize("case", ["initial", "poor", "invalid", "outside"])
def test_reporting_withholds_unsupported_savings(tmp_path, case):
    result, _ = analysis.analyze(curves(), analysis.resample_indices())
    review = dict(assessment="approximate", note="Toy fit.")
    if case == "initial":
        result["point"]["issues"] = ["target_already_met_without_adaptation"]
    elif case == "poor":
        review["assessment"] = "poor"
    elif case == "invalid":
        result["bootstrap"].update(ci=None, invalid=1)
    elif case == "outside":
        result["point"]["in_range"] = False
    cell = report.preliminary(tmp_path, tmp_path / "test.ipynb", result, review)
    assert {"initial": "already meets", "poor": "withheld", "invalid": "unavailable", "outside": "range-limited"}[case] in cell.source
    if case in ("initial", "poor", "outside"):
        assert "50.0 images (50.0%)" not in cell.source


def source_fixture(tmp_path, *, missing=False, wrong_angle=False):
    development, root = tmp_path / "dev", tmp_path / "confirmation"
    with Store(development, {"toy": "development"}):
        pass
    values, recipes = curves(), [Treatment(condition=method, ridge_multiplier=1,
                                           strength=160.89 if method == "finite_difference" else 0)
                                for method in analysis.METHODS]
    with Store(root, {"toy": "confirmation", "source_hash": content_hash(analysis.scientific_sources())}) as store:
        for replica in range(12):
            for stream in ("mixed", "base"):
                for treatment in recipes:
                    if missing and replica == 11 and stream == "base" and treatment.condition == "finite_difference":
                        continue
                    method = treatment.condition
                    name = f"trajectory-{phase7.CONFIRMATION}-{replica:04d}-{stream}-{treatment.key}"
                    meta = dict(schema_version=1, kind="drip_trajectory", phase="7", family=phase7.CONFIRMATION,
                                replica=replica, stream=stream, condition=method, event_target=100,
                                treatment=treatment.mapping(), treatment_id=treatment.key)
                    rows = [dict(paired_rotated_event=n, actual_rotated_events=n if stream == "mixed" else 0,
                                 rotated_nll=value, base_nll=.2, rotated_accuracy=.6, base_accuracy=.9)
                            for n, value in enumerate(values[method][replica])]
                    summary = dict(schema_version=1, unit_id=name, replica=replica, stream=stream,
                                   condition=method, strength=treatment.strength, event_target=100,
                                   rotated_auc=float(np.mean(values[method][replica][1:])), base_auc=.2,
                                   final_rotated_nll=rows[-1]["rotated_nll"], final_rotated_accuracy=.6,
                                   final_base_nll=.2, final_base_accuracy=.9, active_seconds=1)
                    store.begin(name, meta)
                    for filename, value in (("metadata.json", meta), ("metrics.json", rows),
                                            ("fit.json", []), ("summary.json", summary)):
                        store.write_json(name, filename, value)
                    store.complete(name)
        ledger = dict(schema_version=1, replicas=12, event_target=100, angle=45 if wrong_angle else 30,
                      treatments=[recipe.mapping() for recipe in recipes], source=dict(root=str(development)))
        for decision, value in (("confirmation-ledger", ledger), ("results", established(values))):
            unit = f"regularization-{decision}-v1"
            store.begin(unit, dict(kind="regularization_decision", decision=decision))
            store.write_json(unit, "result.json", value)
            store.complete(unit)
        store.write_json(None, "status.json", dict(stage="7D", status="completed", active_seconds=1,
                                                   completed=48, expected=48, development_root=str(development)))
    return root


@pytest.mark.parametrize("case", ["complete", "missing", "wrong_angle"])
def test_source_requires_all_48_native_confirmation_trajectories(tmp_path, case):
    root = source_fixture(tmp_path, missing=case == "missing", wrong_angle=case == "wrong_angle")
    before = seals(root)
    if case == "complete":
        loaded = analysis.load_source(root)
        assert loaded["curves"] == curves()
        assert loaded["established"]["primary"]["ci"] == [.2, .3]
    else:
        with pytest.raises(ArtifactError):
            analysis.load_source(root)
    assert seals(root) == before


def test_phase7_refresh_keeps_followup_after_its_plots(tmp_path, monkeypatch):
    # Render-only fixture: no inference is released by the older report.
    root = tmp_path / "phase7"
    with Store(root, {"toy": 1}) as store:
        store.write_json(None, "status.json", dict(stage="7A", status="running", active_seconds=1,
                                                   completed=0, expected=1))
    path = tmp_path / "old.ipynb"
    cells = [nbformat.v4.new_markdown_cell("# Study")]
    cells += [nbformat.v4.new_markdown_cell(f"Followup {n}", metadata={"tags": [report.TAG]}) for n in range(3)]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    refreshed = phase7.build_notebook(path, root)
    assert refreshed.cells[-3:] == cells[-3:]
    assert all(phase7.TAG in cell.metadata.get("tags", []) for cell in refreshed.cells[-11:-3])


def test_cli_stop_clear_resume_publication_and_completed_skip(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    root, path = tmp_path / "analysis", tmp_path / "results.ipynb"
    original_cells = notebook(path)
    before = seals(source)
    args = ["--root", str(root), "--source-root", str(source), "--budget-root", str(tmp_path),
            "--data-root", str(tmp_path / "no-dataset-needed"), "--notebook", str(path)]
    original_analyze = analysis.analyze

    def stop(values, indices, **kwargs):
        (root / "STOP").touch()
        kwargs["check"]()

    monkeypatch.setattr(analysis, "analyze", stop)
    assert analysis.main(args) == 3
    assert json.loads((root / "status.json").read_text())["status"] == "paused"
    assert nbformat.read(path, as_version=4).cells == original_cells
    plan_seal = seals(root)[analysis.PLAN]
    with pytest.raises(StopRequested):
        analysis.main([*args, "--resume"])
    monkeypatch.setattr(analysis, "analyze", original_analyze)
    assert analysis.main([*args, "--resume", "--clear-stop", "--fit-review", "approximate",
                          "--fit-note", "Toy curves are exactly linear."]) == 0
    assert not (root / "STOP").exists()
    assert seals(root)[analysis.PLAN] == plan_seal
    assert json.loads((root / "status.json").read_text())["status"] == "completed"

    def forbidden(*args, **kwargs):
        raise AssertionError("completed inference repeated")

    monkeypatch.setattr(analysis, "analyze", forbidden)
    prior = seals(root)
    assert analysis.main([*args, "--resume"]) == 0
    assert seals(root) == prior and seals(source) == before
    assert len([cell for cell in nbformat.read(path, as_version=4).cells
                if report.TAG in cell.metadata.get("tags", [])]) == 3
