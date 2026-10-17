"""Artifact-only rendering with fabricated scalar units, never MNIST training."""

import ast
import base64
import hashlib
from html.parser import HTMLParser
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mnist_experiment.akaike_score_gate import analysis, refresh_notebook as report_view
if __package__:
    from .test_analysis import make_root, reseal, seal_unit, write_json
else:
    from test_analysis import make_root, reseal, seal_unit, write_json


def test_phase_order_and_metadata():
    phases = list(reversed(report_view.PHASE_ORDER))
    assert sorted(phases, key=analysis.phase_sort_key) == [f"phase{i}" for i in range(7)]
    assert report_view._code_cell("pass", [], phase="phase3")["metadata"]["plan14_phase"] == "phase3"


def test_empty_report_is_progressive_reproducible_and_has_executable_cells(tmp_path):
    root = make_root(tmp_path / "root")
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    first = notebook.read_bytes()
    report_view.refresh(root, notebook)
    assert notebook.read_bytes() == first
    document = json.loads(first)
    assert document["nbformat"] == 4 and document["nbformat_minor"] == 5
    headings = [cell["source"] for cell in document["cells"] if cell["cell_type"] == "markdown"
                and cell["source"].startswith("## Phase")]
    assert headings == ["## " + label for label in report_view.PHASE_LABELS.values()]
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    text = first.decode()
    for term in ("eight-digit", "4.4", "9.9", "100.1", "S_B>2q", "two consecutive", "right-censored"):
        assert term in text


def test_source_only_progress_and_missing_optional_fields_render(tmp_path):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase1", kind="source", policy=None, environment="mixture",
              summary={"family": "mixture", "source_nll": .25, "source_accuracy": .92},
              metrics=[{"epoch": 1, "validation_nll": .25}])
    (root / "phase0" / ".incomplete" / "interrupted").mkdir(parents=True)
    report_view.refresh(root, tmp_path / "report.ipynb")
    report = analysis.load_report(root)
    assert "source_nll" in report_view.phase_markdown(report, "phase1")
    assert "No sealed trajectories yet" in report_view.phase_markdown(report, "phase1")
    assert "Incomplete units: **1**" in report_view.status_markdown(report)
    figures = list(report_view.phase_figures(report, "phase1"))
    try:
        assert len(figures) == 1 and figures[0][0] == "source-mixture"
    finally:
        for _, figure in figures:
            report_view.plt.close(figure)


def test_cli_defaults_to_float64_science_root(monkeypatch):
    calls = []

    def refresh(root, notebook_path, *, force_fresh=False):
        calls.append((root, notebook_path, force_fresh))
        return notebook_path

    monkeypatch.setattr(report_view, "refresh", refresh)
    report_view.main([])
    assert calls == [(Path("cache/mnist_experiment/akaike_score_gate/default_float64"),
                      report_view.DEFAULT_NOTEBOOK, False)]


def test_cli_does_not_require_study_or_execution_imports(tmp_path):
    root = make_root(tmp_path / "root")
    notebook = tmp_path / "report.ipynb"
    script = ("import sys\n"
              "from mnist_experiment.akaike_score_gate.refresh_notebook import main\n"
              "main(sys.argv[1:])\n"
              "assert 'torch' not in sys.modules\n"
              "assert 'mnist_experiment.akaike_score_gate.config' not in sys.modules\n"
              "assert 'mnist_experiment.akaike_score_gate.trajectory' not in sys.modules\n")
    result = subprocess.run([sys.executable, "-c", script, "--output-root", str(root),
                             "--notebook", str(notebook)], capture_output=True, text=True, check=True)
    assert str(notebook) in result.stdout and notebook.is_file()


def test_corruption_does_not_replace_previous_notebook_or_repair_unit(tmp_path):
    root = make_root(tmp_path / "root")
    unit = seal_unit(root)
    notebook = tmp_path / "report.ipynb"
    notebook.write_text("previous report", encoding="utf-8")
    write_json(unit / "summary.json", {"broken": True})
    before = {path.name: path.read_bytes() for path in unit.iterdir() if path.is_file()}
    with pytest.raises(analysis.ArtifactError, match="Corrupt artifact"):
        report_view.refresh(root, notebook)
    assert notebook.read_text() == "previous report"
    assert before == {path.name: path.read_bytes() for path in unit.iterdir() if path.is_file()}


def test_rendered_trajectory_has_embedded_outputs_and_shareable_figures(tmp_path):
    root = make_root(tmp_path / "root")
    for i in range(2):
        unit = seal_unit(root, index=i)
        write_json(unit / "pr_curves.json", [
            {"step": 2, "observations": 180, "threshold": .1, "precision": .2,
             "recall": .95, "precision_defined": True, "false_positive_rate": .4},
            {"step": 2, "observations": 180, "threshold": .9, "precision": .9,
             "recall": .25, "precision_defined": True, "false_positive_rate": .003},
            {"step": 2, "observations": 180, "threshold": None, "precision": None,
             "recall": 0., "precision_defined": False, "false_positive_rate": 0.},
        ])
        reseal(unit)
    snapshot = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    first = notebook.read_bytes()
    report_view.refresh(root, notebook)
    assert notebook.read_bytes() == first
    document = json.loads(notebook.read_text())
    images = [output["data"]["image/png"] for cell in document["cells"] for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    assert len(images) >= 6
    assert any((root / "report/figures").glob("*pr-sweep.png"))
    assert all(base64.b64decode(value).startswith(b"\x89PNG\r\n\x1a\n") for value in images)
    assert len(list((root / "report/figures").glob("*.png"))) == len(images)
    assert len(list((root / "report/figures").glob("*.svg"))) == len(images)
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in snapshot.items())
    assert report_view.plt.get_fignums() == []


def test_emitted_audit_gain_fields_render_without_matrix_artifacts(tmp_path):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase2", kind="fisher_audit", environment="rotation_sigmoid", policy=None,
              summary={"metric_schema_version": 1, "case": {"step": 3}},
              audit=[{"metric_schema_version": 1, "gate_gain": .2, "realized_current_gain": .1,
                      "delta_optimism": .01, "gate_status": "closed"}])
    figures = list(report_view.phase_figures(analysis.load_report(root), "phase2"))
    try:
        assert len(figures) == 1
        assert figures[0][1].axes[0].get_xlabel() == "gate gain"
    finally:
        for _, figure in figures:
            report_view.plt.close(figure)


def test_rotation_labels_do_not_claim_mixture_warmup(tmp_path):
    root = make_root(tmp_path / "root")
    seal_unit(root, environment="rotation_linear", summary={"family": "rotation", "expected_nines": 40})
    units = analysis.trajectory_groups(analysis.load_report(root))[0]
    assert "warm-up" not in report_view.group_label(units)
    assert "expected 9s" not in report_view.group_label(units)


def test_report_code_uses_lightweight_module_functions_not_embedded_training():
    source = Path(report_view.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and any(part in name for part in ("torch", "trajectory", "assets", "config")) for name in imports)


def test_refresh_refuses_notebook_destination_inside_sealed_unit(tmp_path):
    root = make_root(tmp_path / "root")
    unit = seal_unit(root)
    with pytest.raises(analysis.ArtifactError, match="immutable completed unit"):
        report_view.refresh(root, unit / "report.ipynb")
    assert not (unit / "report.ipynb").exists()


def test_development_figures_are_bounded_by_default_and_frozen_selected_protocol(tmp_path):
    root = make_root(tmp_path / "root")
    base = {"c": 1., "half_life": 8., "cadence": 1}
    candidates = [base, *({**base, "c": value} for value in (.5, 2., 4.)),
                  *({**base, "half_life": value} for value in (4., 16.)),
                  *({**base, "cadence": value} for value in (2, 4))]
    for i, candidate in enumerate(candidates):
        path = seal_unit(root, phase="phase3", policy="asg_lr_lr", detail={
            "policy": {**candidate, "gate": "lr"}, "episode_arrivals": 32})
        path.rename(path.with_name(f"candidate-{i}"))
    write_json(root / "decisions/phase3_calibration.json", {"schema_version": 1, "selected": candidates[2]})
    report = analysis.load_report(root)
    assert len(analysis.trajectory_groups(report, "phase3")) == 8
    selected = report_view._plot_groups(report, "phase3", include_development_variants=False)
    assert {analysis.unit_metadata(group[0])["c"] for group in selected} == {1., 2.}
    assert len(selected) == 2
    assert len(report_view._plot_groups(report, "phase3", include_development_variants=True)) == 8
    markdown = report_view.phase_markdown(report, "phase3")
    assert markdown.count("Full development-variant plots are omitted") == 6
    for group in analysis.trajectory_groups(report, "phase3"):
        assert analysis.unit_metadata(group[0])["protocol_variant"] in markdown


def test_phase1_retention_pilots_keep_all_summaries_but_only_default_and_selected_curves(tmp_path):
    root = make_root(tmp_path / "root")
    base = {"c": 1., "half_life": 8., "cadence": 1, "gate": None}
    settings = [{"omega": omega, "ridge_ratio": ridge} for omega, ridge in
                [(19., 0.), (9.5, 0.), (38., 0.), (19., .01), (19., .1)]]
    for index, retention in enumerate(settings):
        unit = seal_unit(root, phase="phase1", policy="depth1_lr", detail={"policy": base, "retention": retention})
        unit.rename(unit.with_name(f"retention-{index}"))
    seal_unit(root, phase="phase1", kind="phase1_selection", policy=None,
              summary={"retention": {"mixture": settings[2]}})
    report = analysis.load_report(root)
    groups = report_view._plot_groups(report, "phase1", include_development_variants=False)
    assert len(groups) == 2
    assert {analysis.unit_metadata(group[0])["retention"]["omega"] for group in groups} == {19., 38.}
    assert report_view.phase_markdown(report, "phase1").count("Full development-variant plots are omitted") == 3


def test_phase5_named_policies_share_panel_without_pooling_replica_means(tmp_path):
    root = make_root(tmp_path / "root")
    for name, gate, c in [("depth6_lr", None, 1.), ("asg_lr_lr", "lr", 2.), ("asg_uncalibrated", "lr", 1.)]:
        seal_unit(root, phase="phase5", policy=name, detail={"policy": {
            "c": c, "half_life": 8., "cadence": 1, "gate": gate}})
    report = analysis.load_report(root)
    groups = report_view._plot_groups(report, "phase5", include_development_variants=False)
    assert len(groups) == 1 and len(groups[0]) == 3
    values = analysis.pointwise_summary(groups[0], "current_nll")
    assert {row["policy"] for row in values} == {"depth6_lr", "asg_lr_lr", "asg_uncalibrated"}
    assert all(row["n"] == 1 for row in values)


def test_figure_iterator_is_lazy_and_holds_at_most_one_live_figure(tmp_path):
    root = make_root(tmp_path / "root")
    seal_unit(root)
    figures = report_view.phase_figures(analysis.load_report(root), "phase4")
    assert report_view.plt.get_fignums() == []
    try:
        next(figures)
        assert len(report_view.plt.get_fignums()) == 1
        next(figures)
        assert len(report_view.plt.get_fignums()) == 1
    finally:
        figures.close()
    assert report_view.plt.get_fignums() == []


def visible_text(markdown):
    class VisibleText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag == "details":
                self.depth += 1

        def handle_endtag(self, tag):
            if tag == "details":
                self.depth -= 1

        def handle_data(self, data):
            if self.depth == 0:
                self.parts.append(data)

    parser = VisibleText()
    parser.feed(markdown)
    assert parser.depth == 0
    return "".join(parser.parts)


def test_phase_readability_collapses_provenance_and_raw_audits_without_losing_access(tmp_path):
    root = make_root(tmp_path / "root")
    parents = [{"path": f"parent-{i}", "hash": "PROVENANCE_SENTINEL"} for i in range(1000)]
    source = seal_unit(root, phase="phase1", kind="source", environment="mixture", policy=None,
                       summary={"source_nll": .25, "source_accuracy": .92, "parents": parents,
                                "resolved_config": {"large_source_hash_inventory": parents}})
    seal_unit(root, phase="phase1", kind="decision", policy="selected_protocol",
              summary={"selected": {"c": 2., "half_life": 8., "cadence": 1}, "parents": parents})
    audit = seal_unit(root, phase="phase1", kind="depth_audit", environment="rotation_linear", policy=None,
                      summary={"row_count": 1024, "unresolved_restricted_rows": 2, "parents": parents},
                      audit=[{"step": index, "representation": "lr", "gate_gain": .2,
                              "realized_current_gain": .1, "gate_status": "closed", "raw_id": f"AUDIT_ROW_{index}"}
                             for index in range(1024)])
    report = analysis.load_report(root)
    text = report_view.phase_markdown(report, "phase1")
    visible = visible_text(text)
    assert "source_nll=0.25" in visible and "source_accuracy=0.92" in visible
    assert "unresolved_restricted_rows=2" in visible
    assert "selected: c=2" in visible and "half_life=8" in visible
    assert "PROVENANCE_SENTINEL" not in text and "AUDIT_ROW_1023" not in text
    assert report["units"][0]["manifest"]["config_hash"] not in visible
    assert "n=1024/1024" in visible and "not independent trajectory replicas" in visible
    assert len(visible.splitlines()) < 100
    assert "<summary>Raw audit rows (1024" in text
    assert str(source / "summary.json") in text and str(audit / "audit_metrics.json") in text
    assert json.loads((source / "summary.json").read_text())["parents"] == parents
    assert len(json.loads((audit / "audit_metrics.json").read_text())) == 1024


def test_large_phase2_audits_are_linked_not_embedded_even_when_collapsed(tmp_path):
    root = make_root(tmp_path / "root")
    unit = seal_unit(root, phase="phase2", kind="fisher_audit", policy=None,
                     summary={"row_count": 20_000, "failed": False},
                     audit=[{"gate_status": "closed", "gate_gain": .2, "realized_current_gain": .1,
                             "raw_id": f"LARGE_AUDIT_SENTINEL_{index}"} for index in range(20_000)])
    text = report_view.phase_markdown(analysis.load_report(root), "phase2")
    assert "n=20000/20000" in visible_text(text) and "closed: 20000" in visible_text(text)
    assert "LARGE_AUDIT_SENTINEL" not in text and len(text) < 12_000
    assert str(unit / "audit_metrics.json") in text


def test_large_nested_metadata_and_log_details_stay_out_of_notebook_dom(tmp_path):
    root = make_root(tmp_path / "root")
    parents = [{"path": f"parent-{index}", "hash": "LARGE_PARENT_SENTINEL"} for index in range(1000)]
    log = [{"timestamp": "2026-10-07T12:00:00Z", "message": "Frozen selection recorded",
            "failed": False, "detail": {"parents": parents}}]
    write_json(root / "lab_log.json", log)
    write_json(root / "progress.json", {"status": "running", "parents": parents})
    write_json(root / "study.json", {"schema_version": 1, "source_hashes": parents})
    report = analysis.load_report(root)
    text = report_view.status_markdown(report) + report_view.lab_log_markdown(report)
    assert "LARGE_PARENT_SENTINEL" not in text and len(text) < 12_000
    assert log[0]["timestamp"] in visible_text(text) and log[0]["message"] in visible_text(text)
    assert "failed=False" in visible_text(text)
    for name in ("study.json", "progress.json", "lab_log.json"):
        assert str(root / name) in text
    assert json.loads((root / "lab_log.json").read_text()) == log


def test_lab_log_shows_time_message_and_flags_but_escapes_collapsed_payload(tmp_path):
    root = make_root(tmp_path / "root")
    log = [{"time": "2026-10-07T12:00:00Z", "message": "Retention constraint failed; continuing frozen policy",
            "constraint_failure": True, "repaired": False, "phase": "phase4",
            "detail": {"raw_marker": "NESTED_LOG_SENTINEL", "untrusted_text": "</details><script>raw</script>"}}]
    write_json(root / "lab_log.json", log)
    text = report_view.lab_log_markdown(analysis.load_report(root))
    visible = visible_text(text)
    assert log[0]["time"] in visible and log[0]["message"] in visible
    assert "constraint_failure=True" in visible and "repaired=False" in visible
    assert "NESTED_LOG_SENTINEL" not in visible and "NESTED_LOG_SENTINEL" in text
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "Final integrity review" in visible


def test_closeout_shows_recorded_condition_intervals_and_primary_paired_contrasts(tmp_path, monkeypatch):
    root = make_root(tmp_path / "root")
    stats = lambda mean: {"mean": mean, "ci95_low": mean - .1, "ci95_high": mean + .1, "n": 32, "n_total": 32}
    summary = {"cohort_complete": True, "replicas_per_condition": 32,
               "effect_convention": "treatment minus control; negative NLL difference favors treatment",
               "inference": "Recorded paired replica 95% CIs, not simultaneous intervals",
               "conclusions": [{"environment": environment, "policy": "asg_lr_lr", "replicas": 32,
                                "nll_auc": stats(nll), "early_25_nll_auc": stats(nll + .1),
                                "retention_change": stats(.01), "online_seconds": stats(3.),
                                "retention_admissible_mean": True}
                               for environment, nll in [("rotation_linear", .6), ("rotation_sigmoid", .8)]],
               "parents": ["CLOSEOUT_PARENT_SENTINEL"], "limitations": ["Finite reference is not truth."]}
    unit = seal_unit(root, phase="phase6", kind="closeout", policy=None, summary=summary)
    effects = [{"environment": "rotation_linear", "treatment": "asg_lr_lr", "control": "depth1_lr",
                "metric": "nll_auc", "unpaired_treatment": 0, "unpaired_control": 0, **stats(-.1)},
               {"environment": "rotation_linear", "treatment": "SECONDARY_ONLY_SENTINEL", "control": "depth1_lr",
                "metric": "retention_nll", **stats(.02)}]
    write_json(unit / "paired_effects.json", effects)
    reseal(unit)
    report = analysis.load_report(root)
    def no_new_bootstrap(*args, **kwargs):
        raise AssertionError("Closeout must display its recorded inference, not recompute it")
    monkeypatch.setattr(analysis, "mean_interval", no_new_bootstrap)
    text = report_view.closeout_markdown(report)
    visible = visible_text(text)
    assert "rotation_linear" in visible and "rotation_sigmoid" in visible
    assert "0.6; 95% CI [0.5, 0.7]; n=32" in visible
    assert "0.8; 95% CI [0.7, 0.9]; n=32" in visible
    assert "-0.1; 95% CI [-0.2, 0]; n=32" in visible and "depth1_lr" in visible
    assert "CLOSEOUT_PARENT_SENTINEL" not in visible and "SECONDARY_ONLY_SENTINEL" not in visible
    assert "cohort_complete=True" in visible and "Finite reference is not truth." in visible
    assert str(unit / "summary.json") in text and str(unit / "paired_effects.json") in text
    assert json.loads((unit / "paired_effects.json").read_text()) == effects


def test_closeout_extra_tables_cannot_bypass_seal_validation(tmp_path):
    root = make_root(tmp_path / "root")
    unit = seal_unit(root, phase="phase6", kind="closeout", policy=None, summary={"conclusions": []})
    write_json(unit / "paired_effects.json", [])
    reseal(unit)
    report = analysis.load_report(root)
    write_json(unit / "paired_effects.json", [{"mean": 1234}])
    with pytest.raises(analysis.ArtifactError, match="closeout artifact"):
        report_view.closeout_markdown(report)


def test_artifact_links_are_relative_to_generated_notebook_and_keep_full_data(tmp_path):
    root = make_root(tmp_path / "root with spaces")
    unit = seal_unit(root, phase="phase1", kind="source", policy=None, summary={"source_nll": .2})
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    document = json.loads(notebook.read_text())
    setup = next(cell["source"] for cell in document["cells"] if cell["cell_type"] == "code")
    assert "report['notebook_dir']" in setup
    outputs = "\n".join(output["data"].get("text/markdown", "") for cell in document["cells"]
                        for output in cell.get("outputs", []))
    assert "root%20with%20spaces/phase1/" in outputs
    assert "<details>" in outputs and (unit / "summary.json").is_file()


@pytest.fixture
def cached_renderer(monkeypatch):
    calls = report_view.Counter()
    monkeypatch.setattr(report_view, "_FIGURE_CACHE", report_view.OrderedDict())

    # Cache tests isolate invalidation from Agg; the real-image test covers exports.
    class Figure:
        def savefig(self, buffer, *, format, **kwargs):
            buffer.write(f"cache fixture {format}".encode())

    def figures(report, phase):
        calls[phase] += 1
        if any(unit["config"]["phase"] == phase for unit in report["units"]):
            yield "fixture", Figure()

    monkeypatch.setattr(report_view, "phase_figures", figures)
    return calls


def test_unchanged_phases_reuse_outputs_but_always_load_and_force_fresh_renders(tmp_path, monkeypatch, cached_renderer):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase1", kind="source", policy=None)
    notebook = tmp_path / "report.ipynb"
    calls = []
    original = analysis.load_report
    def load(root, *, force_fresh=False):
        calls.append(force_fresh)
        return original(root, force_fresh=force_fresh)
    monkeypatch.setattr(analysis, "load_report", load)
    report_view.refresh(root, notebook)
    first = notebook.read_bytes()
    exports = {path: analysis._fingerprint(path) for path in (root / "report/figures").iterdir()}
    report_view.refresh(root, notebook)
    assert notebook.read_bytes() == first
    assert all(count == 1 for count in cached_renderer.values())
    assert all(analysis._fingerprint(path) == fingerprint for path, fingerprint in exports.items())
    report_view.refresh(root, notebook, force_fresh=True)
    assert calls == [False, False, True]
    assert all(count == 2 for count in cached_renderer.values())
    assert notebook.read_bytes() == first


def test_new_units_invalidate_only_their_phase_and_overviews_stay_fresh(tmp_path, cached_renderer):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase0", kind="contract", policy=None)
    seal_unit(root, phase="phase1", kind="source", policy=None)
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    added = seal_unit(root, phase="phase1", kind="source", policy=None, index=1)
    write_json(root / "progress.json", {"status": "fresh progress sentinel"})
    write_json(root / "lab_log.json", [{"message": "fresh lab log sentinel"}])
    report_view.refresh(root, notebook)
    assert cached_renderer["phase1"] == 2
    assert all(count == 1 for phase, count in cached_renderer.items() if phase != "phase1")
    assert "fresh progress sentinel" in notebook.read_text() and "fresh lab log sentinel" in notebook.read_text()
    (added / "COMPLETED").unlink()
    report_view.refresh(root, notebook)
    assert cached_renderer["phase1"] == 3
    assert all(count == 1 for phase, count in cached_renderer.items() if phase != "phase1")


@pytest.mark.parametrize("changed", ["selection", "study_sources", "targets", "renderer_source", "analysis_source", "bootstrap"])
def test_frozen_render_dependencies_invalidate_cached_phases(tmp_path, monkeypatch, cached_renderer, changed):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase1", kind="source", policy=None)
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    if changed == "selection":
        write_json(root / "decisions/phase3_calibration.json", {"schema_version": 1, "selected": {"c": 2}})
    elif changed == "study_sources":
        write_json(root / "study.json", {"schema_version": 1, "source_hashes": {"renderer.py": "changed"}})
    elif changed == "targets":
        monkeypatch.setattr(report_view, "TARGETS", (.4, .5, .8))
    elif changed == "bootstrap":
        monkeypatch.setattr(analysis, "BOOTSTRAP_SEED", 987)
    else:
        source = Path(report_view.__file__ if changed == "renderer_source" else analysis.__file__).resolve()
        original = analysis._hash_file
        def changed_hash(path, *, force_fresh=False):
            return "changed source hash" if path.resolve() == source else original(path, force_fresh=force_fresh)
        monkeypatch.setattr(analysis, "_hash_file", changed_hash)
    report_view.refresh(root, notebook)
    assert all(count == 2 for count in cached_renderer.values())


def test_figure_cache_never_masks_corrupt_scientific_artifacts(tmp_path, cached_renderer):
    root = make_root(tmp_path / "root")
    unit = seal_unit(root, phase="phase1", kind="source", policy=None)
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    first = notebook.read_bytes()
    write_json(unit / "summary.json", {"corrupt": True})
    with pytest.raises(analysis.ArtifactError, match="Corrupt artifact"):
        report_view.refresh(root, notebook)
    assert notebook.read_bytes() == first and all(count == 1 for count in cached_renderer.values())


@pytest.mark.parametrize("damage", ["missing_png", "changed_svg"])
def test_missing_or_changed_exports_cannot_be_skipped(tmp_path, cached_renderer, damage):
    root = make_root(tmp_path / "root")
    seal_unit(root, phase="phase1", kind="source", policy=None)
    notebook = tmp_path / "report.ipynb"
    report_view.refresh(root, notebook)
    suffix = "png" if damage == "missing_png" else "svg"
    path = root / "report/figures" / f"phase1-fixture.{suffix}"
    expected = path.read_bytes()
    if damage == "missing_png":
        path.unlink()
    else:
        path.write_bytes(b"broken export")
    report_view.refresh(root, notebook)
    assert path.read_bytes() == expected and cached_renderer["phase1"] == 2
    assert all(count == 1 for phase, count in cached_renderer.items() if phase != "phase1")


def test_export_destination_and_study_root_are_part_of_cache_identity(tmp_path, cached_renderer):
    roots = [make_root(tmp_path / name) for name in ("first", "second")]
    for root in roots:
        seal_unit(root, phase="phase1", kind="source", policy=None)
    reports = [analysis.load_report(root) for root in roots]
    for report, output in [(reports[0], "export-a"), (reports[0], "export-b"), (reports[1], "export-b")]:
        outputs = report_view._phase_figure_outputs(report, "phase1", tmp_path / output)
        assert outputs and (tmp_path / output / "phase1-fixture.png").is_file()
        assert (tmp_path / output / "phase1-fixture.svg").is_file()
    assert cached_renderer["phase1"] == 3


def test_cache_replacement_and_retention_are_bounded(tmp_path, monkeypatch, cached_renderer):
    root = make_root(tmp_path / "root")
    for phase in ("phase0", "phase1"):
        seal_unit(root, phase=phase, kind="contract", policy=None)
    report = analysis.load_report(root)
    destination = tmp_path / "exports"
    for phase in ("phase0", "phase1"):
        report_view._phase_figure_outputs(report, phase, destination)
    maximum = max(entry["bytes"] for entry in report_view._FIGURE_CACHE.values())
    report_view._FIGURE_CACHE.clear()
    monkeypatch.setattr(report_view, "FIGURE_CACHE_MAX_BYTES", maximum)
    for phase in ("phase0", "phase1", "phase0"):
        report_view._phase_figure_outputs(report, phase, destination)
        assert len(report_view._FIGURE_CACHE) == 1
        assert sum(entry["bytes"] for entry in report_view._FIGURE_CACHE.values()) <= maximum
    assert cached_renderer["phase0"] == 3
    monkeypatch.setattr(report_view, "FIGURE_CACHE_MAX_BYTES", 1)
    report_view._phase_figure_outputs(report, "phase0", destination, force_fresh=True)
    assert not report_view._FIGURE_CACHE
    assert report_view.plt.get_fignums() == []
