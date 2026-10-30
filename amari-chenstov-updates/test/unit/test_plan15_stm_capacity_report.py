"""Phase 12 notebook preservation and bounded artifact-only rendering."""

import base64
from contextlib import contextmanager
from copy import deepcopy
import io
import json

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_stm_capacity import analysis as a, report


def pending():
    return dict(protocol=a.PROTOCOL, progress=dict(status="running", stage="12A", active_seconds=5,
        mandatory_development=46, optional_development=0, paired_blocks=0, confirmation_blocks_target=8,
        completed_by_partition=dict(development=0, confirmation=0)),
        configurations=[dict(name=c, stm_parameters=d["stm_parameters"], ltm_parameters=d["ltm_parameters"])
                        for c, d in a.STM_CONFIGS.items()], development=[], selection={}, analysis=None, changes=[])


def analyzed():
    records = []
    for arm in a.ARMS:
        chart, method = arm.split("--")
        recipe = Recipe(method, 32, 2, strength=.5 if method == a.FD else 0)
        novel = [16, 11, 7, 3] if method == a.FD else [16, 14, 11, 8]
        for r in range(2):
            metrics = [dict(n=n, observations=(10+r)*n, learning_seconds=(2 if method == a.FD else 1)*n,
                novel_mse=risk*(1+.1*r), base_mse=4 if n == 0 else 4.1,
                novel_accuracy1=.3+.05*n, base_accuracy1=.4, novel_mae=2., base_mae=1.)
                for n, risk in enumerate(novel)]
            records.append(dict(a.summarize_trajectory(metrics, N=3), family="confirmation--grid", partition="confirmation",
                target=3, context_key="grid", output_dim=54, stm_config=chart, arm=arm, method=method,
                recipe=recipe.mapping(), recipe_key=recipe.key, capacity=32, replay_count=2,
                replica=r, plan_hash=f"p-{r}", pairing_id=f"p-{r}", metrics=metrics,
                active_seconds=10., peak_vram_bytes=0,
                retired_audit=[dict(n=2, exposure=4, squared_error=4.5+r), dict(n=3, exposure=4, squared_error=5+r)]))
    return a.analyze(records, N=3, draws=30, seed=17)


@pytest.fixture(scope="module")
def completed_analysis():
    return analyzed()


def complete(result):
    snapshot = pending()
    snapshot["analysis"] = deepcopy(result)
    for arm, item in result["arms"].items():
        row = dict(stm_config=item["stm_config"], method=item["method"], arm=arm,
            recipe=item["recipe"], novel_auc=item["acquisition"]["mean"], H=item["retention"]["mean"],
            seconds_per_arrival=item["costs"]["seconds_per_arrival"])
        snapshot["development"].append(row)
        snapshot["selection"][arm] = dict(selected=deepcopy(row), eligible=True, failures=[])
    return snapshot


def prose(cells):
    return "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in cells)


def test_pending_section_has_small_progress_and_honest_no_inference_text():
    snapshot = pending()
    snapshot["progress"].update(heartbeat="2026-10-29T12:34:56Z", paired_blocks=2, completed_pairs=8,
        planned_pairs=32, current="unit-x", last_accepted_checkpoint="n=75", forecast_seconds=3600,
        storage=dict(used_bytes=10_000_000_000, limit_bytes=500_000_000_000))
    cells = report.build_cells(snapshot)
    text = prose(cells)
    for phrase in ("Phase 12", "unverified", "12A", "2/8 eight-arm paired blocks", "8/32 within-size pairs",
                   "Mandatory development ?/46", "optional ?/0", "unit-x/n=75", "forecast remaining 1.00 h",
                   "490.00 GB headroom", "2026-10-29T12:34:56Z", "Fresh Confirmation Pending", "no inference",
                   "Common reference B", "larger lifetime development search"):
        assert phrase in text
    assert cells[0]["id"] == report.PREFIX+"progress"
    assert all(c["cell_type"] == "markdown" and c["metadata"][report.TAG] for c in cells)
    assert not any(c.get("attachments") for c in cells)


def test_settings_explains_fd_and_replay_in_a_separate_paragraph_below_the_table():
    source = report._settings(pending(), None)["source"]
    note = ("Both **FD** and **Replay** use a replay buffer, EWC, ridge, and trainable STM; "
            "**FD** also adds the finite-difference KL penalty.")
    assert "\n\n"+note+"\n\nAll base weights" in source
    assert source.index("| All r4 Replay |") < source.index(note)


def test_configuration_mappings_and_pending_selection_are_tolerated():
    snapshot = pending()
    first = report.build_cells(snapshot)
    snapshot["configurations"] = {row["name"]: row for row in snapshot["configurations"]}
    assert report.build_cells(snapshot) == first
    snapshot["selection"] = {a.INCUMBENT: dict(selected=None, eligible=None)}
    # Selection decisions may arrive before their selected recipe exists.
    assert "Settings" in prose(report.build_cells(snapshot))


def test_only_exact_phase12_tag_or_prefix_is_owned():
    assert report.owned(dict(id="plan15-phase12-intro"))
    assert report.owned(dict(id="arbitrary", metadata={report.TAG: True}))
    assert not report.owned(dict(id="plan15-phase120-original"))
    assert not report.owned(dict(id="plan15-phase11-original", metadata={report.TAG: False}))


def test_publish_preserves_every_historical_cell_output_and_metadata_and_is_idempotent(tmp_path):
    path = tmp_path/"results.ipynb"
    history = [nbformat.v4.new_markdown_cell("# Results", id="title"),
        nbformat.v4.new_code_cell("raise RuntimeError('never execute')", id="historic-output", execution_count=7,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="original output\n")]),
        nbformat.v4.new_markdown_cell("Phase 10 frozen conclusion", id="plan15-phase10-conclusion", metadata={"plan15_phase10": True}),
        nbformat.v4.new_markdown_cell("Phase 11 frozen conclusion", id="plan15-phase11-conclusion", metadata={"plan15_phase11": True})]
    history[1]["source"] = ["raise RuntimeError('never execute')\n"]
    book = dict(nbformat.v4.new_notebook(cells=history, metadata={"custom": {"nested": [1, 2]}}))
    path.write_text(json.dumps(book))
    before = deepcopy(history)
    report.publish(pending(), path)
    first = json.loads(path.read_bytes())
    assert [c for c in first["cells"] if not report.owned(c)] == before
    assert first["metadata"] == book["metadata"]
    assert first["cells"][1]["id"] == "plan15-phase12-progress"
    assert first["cells"].index(next(c for c in first["cells"] if c["id"] == "plan15-phase11-conclusion")) < first["cells"].index(
        next(c for c in first["cells"] if c["id"] == "plan15-phase12-intro"))
    assert sum(c.get("id") == "plan15-phase12-intro" for c in first["cells"]) == 1
    report.publish(pending(), path)
    assert json.loads(path.read_bytes()) == first
    nbformat.validate(nbformat.read(path, as_version=4))
    later = nbformat.v4.new_markdown_cell("Later user commentary", id="later-user")
    first["cells"].append(later)
    path.write_text(json.dumps(first))
    report.publish(pending(), path)
    assert json.loads(path.read_bytes())["cells"][-1] == later


def test_complete_report_has_four_nonblank_png_figures_three_tables_and_no_mutation(completed_analysis):
    snapshot = complete(completed_analysis)
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot, "supplement.json")
    figures = [c for c in cells if c.get("attachments")]
    assert len(figures) == 4
    assert sum("| ---" in c["source"] for c in cells) == 3
    for figure in figures:
        encoded = next(iter(figure["attachments"].values()))["image/png"]
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
        assert pixels.shape[0] > 200 and pixels.shape[1] > 600 and pixels.std() > 10
        assert "attachment:" in figure["source"]
    text = prose(cells)
    for phrase in ("108 / 3,268", "172 / 3,268", "344 / 3,268", "688 / 3,268", "Head only",
                   "3 novel grids = 27 novel digit pairs", "27 landmarks", "all_r2 Replay/EWC (B)", "**S [95% CI]**",
                   "Paired S minus incumbent", "Within-size replay minus FD", "Evicted-training-cohort",
                   "No smoothing, extrapolation", "one-sided", "not zero", "not a pure dimension", "STM-to-LTM",
                   "Partial results are descriptive", "Smoke/development-sized", "Machine-readable supplement"):
        assert phrase in text
    assert snapshot == before


def test_head_only_is_never_connected_in_capacity_plot(completed_analysis, monkeypatch):
    captured = {}

    def picture(name, figure, caption):
        captured[name] = figure
        return report.cell(name, caption)

    monkeypatch.setattr(report, "picture", picture)
    report._scores(completed_analysis)
    score, _ = captured["scores"].axes
    connected = [line for line in score.lines if len(line.get_xdata()) == 3]
    assert len(connected) == 2
    assert all(list(line.get_xdata()) == [172, 344, 688] for line in connected)


def test_failed_inverse_cannot_be_relabelled_as_valid_or_hide_full_budget_intervals(completed_analysis):
    result = deepcopy(completed_analysis)
    item = result["arms"][f"head_r1--{a.FD}"]["efficiency"]
    item["bootstrap"].update(invalid=1, ci=None)
    effect = result["incumbent_differences"][f"head_r1--{a.FD}"]["S_difference"]
    effect["ci"] = None
    text = report._efficiency_table(result)["source"]
    assert "CI withheld" in text and "1/30" in text
    assert "Supported / Unresolved" in text
    assert "Novel A" in report._predictive_table(result)["source"]


@pytest.mark.parametrize("progress", [dict(partial_confirmation=True, completed_blocks=8, planned_blocks=8),
    dict(completed_blocks=8, planned_blocks=10), dict(paired_blocks=2, confirmation_blocks_target=8)])
def test_partial_confirmation_never_announces_a_winner_even_with_positive_nominal_intervals(completed_analysis, progress):
    snapshot = complete(completed_analysis)
    snapshot["progress"] = progress
    snapshot["analysis"]["is_final"] = True
    arm = f"head_r1--{a.FD}"
    snapshot["analysis"]["incumbent_differences"][arm]["S_difference"] = dict(mean=1, ci=[.5, 1.5])
    snapshot["analysis"]["arms"][arm]["efficiency"]["retention_qualified"] = True
    before = deepcopy(snapshot)
    text = prose(report.build_cells(snapshot))
    assert "Partial confirmation, descriptive only" in text
    assert "provisional repeated-look summaries" in text
    assert "until every frozen confirmation block is complete" in text
    assert "Retention-qualified improvement in S over the incumbent:" not in text
    assert snapshot == before


def test_completed_frozen_cohort_can_report_a_qualified_paired_improvement(completed_analysis):
    snapshot = complete(completed_analysis)
    snapshot["progress"] = dict(completed_blocks=8, planned_blocks=8)
    snapshot["analysis"]["is_final"] = True
    arm = f"head_r1--{a.FD}"
    snapshot["analysis"]["incumbent_differences"][arm]["S_difference"] = dict(mean=1, ci=[.5, 1.5])
    snapshot["analysis"]["arms"][arm]["efficiency"]["retention_qualified"] = True
    text = prose(report.build_cells(snapshot))
    assert "Retention-qualified improvement in S over the incumbent: Head r1" in text
    assert "Partial confirmation, descriptive only" not in text


@pytest.mark.parametrize("changes", [dict(protocol="old"), dict(schema_version=2), dict(status="running"),
    dict(context_key="vector"), dict(output_dim=6), dict(arms={}), dict(replicas=[0]),
    dict(reference_arm=f"all_r1--{a.REPLAY}"), dict(incumbent_arm=f"all_r4--{a.FD}")])
def test_incompatible_analysis_fails_clearly(completed_analysis, changes):
    snapshot = pending()
    snapshot["analysis"] = {**completed_analysis, **changes}
    with pytest.raises(ValueError, match="compatible"):
        report.build_cells(snapshot)


def test_development_analysis_is_not_published_as_confirmation(completed_analysis):
    snapshot = pending()
    snapshot["analysis"] = {**completed_analysis, "partition": "development"}
    with pytest.raises(ValueError, match="development"):
        report.build_cells(snapshot)


def test_report_never_bootstraps_trains_or_loads_models(completed_analysis, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact-only report called inference")

    for name in ("analyze", "bootstrap_chunk", "replica_draw_indices"):
        monkeypatch.setattr(a, name, forbidden)
    report.publish(pending(), tmp_path/"pending.ipynb")
    report.publish(complete(completed_analysis), tmp_path/"completed.ipynb")


class Store:
    def __init__(self, root):
        self.root, self.reservations = root, []

    @contextmanager
    def reserve(self, size):
        self.reservations.append(size)
        yield


def test_publication_accounts_for_bytes_and_uses_immutable_content_addressed_supplement(tmp_path):
    store, path = Store(tmp_path/"cache"), tmp_path/"results.ipynb"
    snapshot = pending()
    report.publish(snapshot, path, store)
    reports = list((store.root/"reports").glob("phase12-supplement-*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_bytes()) == snapshot
    assert sum(store.reservations) == reports[0].stat().st_size+path.stat().st_size
    original = reports[0].read_bytes()
    report.publish(snapshot, path, store)
    assert len(store.reservations) == 3
    assert reports[0].read_bytes() == original
    snapshot["changes"] = ["A scientific correction; full details in the supplement"]
    report.publish(snapshot, path, store)
    assert len(list((store.root/"reports").glob("*.json"))) == 2
    assert "scientific correction" in path.read_text()


def test_frozen_archive_or_oversize_notebook_is_never_mutated(tmp_path, monkeypatch):
    path = tmp_path/"archive.ipynb"
    path.write_text(json.dumps(dict(nbformat.v4.new_notebook(
        metadata={"plan15_phase8_presentation": {"role": "frozen_archive"}}))))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), path)
    monkeypatch.setattr(report, "MAX_NOTEBOOK_BYTES", 2)
    with pytest.raises(ValueError, match="size"):
        report.publish(pending(), path)
    assert path.read_bytes() == original
