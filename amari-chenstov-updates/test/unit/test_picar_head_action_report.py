"""Post-hoc action recovery: complete-history coverage, counts and immutability."""

import hashlib
import json
import math

import pytest
import torch

from numerical_experiment import picar_head_action_report as actions
from numerical_experiment.picar_head_awac import report
from dataclasses import asdict
from src.robotics.actions import Calibration


def history(times, degrees, *, speeds=None, elapsed=None):
    speeds = [.2] * len(times) if speeds is None else speeds
    elapsed = [1.] * len(times) if elapsed is None else elapsed
    return {"decision_time": torch.tensor([times], dtype=torch.float32),
            "elapsed": torch.tensor([[ [dt] for dt in elapsed]], dtype=torch.float32),
            "executed_actions": torch.tensor([[[0., 0., math.radians(angle), speed]
                                                for angle, speed in zip(degrees, speeds)]], dtype=torch.float32)}


def episodes():
    rows = [{"episode_id": "full", "n": 751, "raw_stage_rewards": [0., 0., 0.],
             "layout": {"curriculum": {"sigma_m": .515}}, "success": False},
            {"episode_id": "partial", "n": 740, "raw_stage_rewards": [0., 0.],
             "layout": {"curriculum": {"sigma_m": .52}}, "success": False},
            {"episode_id": "missing", "n": 0, "raw_stage_rewards": [0.],
             "layout": {"curriculum": {"sigma_m": 1.}}, "success": False}]
    for row in rows:
        row["simulator_config"] = {"calibration": asdict(Calibration())}
    return rows


def state():
    return {"n": 1024, "stage": "done", "replay": [
        {"episode_id": "full", "arrival": 752, "action": torch.tensor([10., -10., -10., -10., -10., -10., -10.]),
         "history": history([0., 1.], [0., 8.], elapsed=[0., 1.]),
         "next_history": history([1., 2.], [8., 10.])},
        {"episode_id": "full", "arrival": 753, "action": torch.tensor([-10., 10., -10., -10., -10., -10., -10.]),
         "history": history([2.], [10.]), "next_history": history([3.], [0.], speeds=[0.])},
        {"episode_id": "partial", "arrival": 741, "action": torch.zeros(7),
         "history": history([0.], [0.], elapsed=[0.]),
         "next_history": history([2.], [8.])},
    ]}


def sealed_unit(root):
    unit = root / "units" / "trajectory"
    transaction = unit / "transactions" / ("a" * 32)
    transaction.mkdir(parents=True)
    payload = {"version": report.VERSION, "schema_version": 1, "kind": "head_awac_trajectory",
               "operation": "trajectory", "settings": {"phase": "confirmation", "method": "td_fd",
               "seed": 7, "ewc_rank": 0, "total": 1024}, "n": 1024, "active_seconds": 1., "episodes": episodes()}
    (unit / "summary.json").write_text(json.dumps(payload))
    torch.save(state(), transaction / "state.pt")
    data = (transaction / "state.pt").read_bytes()
    manifest = {"recovery_state": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    (transaction / "manifest.json").write_text(json.dumps(manifest))
    (unit / "CURRENT.json").write_text(json.dumps({"commit_id": "a" * 32, "manifest": {
        "sha256": hashlib.sha256((transaction / "manifest.json").read_bytes()).hexdigest()}}))
    (unit / "COMPLETED").touch()
    return unit


def test_bouts_require_movement_and_mean_angle_includes_subthreshold_steering():
    angles = [0., 7., 8., 0., -7., -8., 7., 7., 7.]
    speeds = [.2, .2, .2, .2, -.1, -.1, .2, 0., .2]
    rows = [(i + 1., 1., [0., 0., math.radians(angle), speed])
            for i, (angle, speed) in enumerate(zip(angles, speeds))]
    outcome = actions.control_summary(rows)
    assert outcome["turn_bouts"] == 4
    assert outcome["moving_seconds"] == 8.
    assert outcome["mean_abs_steering_deg"] == pytest.approx(44. / 8)
    gentle = actions.control_summary([(1., 1., [0., 0., math.radians(2.), .2])])
    assert gentle["turn_bouts"] == 0 and gentle["mean_abs_steering_deg"] == pytest.approx(2.)
    stopped = actions.control_summary([(1., 1., [0., 0., .4, 0.])])
    assert stopped["turn_bouts"] == 0 and stopped["mean_abs_steering_deg"] is None


def test_overlapping_histories_deduplicate_and_partial_episodes_are_not_zero_turns():
    rows = actions.episode_rows({"episodes": episodes()}, state())
    assert rows[0]["complete_history"] and rows[0]["recovered_intervals"] == 3
    assert rows[0]["turn_bouts"] == 2
    assert not rows[1]["complete_history"] and rows[1]["recovered_intervals"] == 1
    assert rows[1]["turn_bouts"] is None
    assert rows[2]["turn_bouts"] is None and rows[2]["recovered_intervals"] == 0


def test_conflicting_duplicate_controls_and_invalid_intervals_fail():
    value = state()
    value["replay"][1]["history"]["executed_actions"][0, 0, 2] = .4
    with pytest.raises(ValueError, match="conflicting"):
        actions.episode_rows({"episodes": episodes()}, value)
    with pytest.raises(ValueError, match="invalid"):
        actions.control_summary([(1., 1., [0., 0., float("nan"), .2])])


def test_offline_analysis_is_immutable_resumable_and_plot_uses_only_complete_histories(tmp_path, monkeypatch):
    import nbformat

    root = tmp_path / "run"
    sealed_unit(root)
    result = actions.analyze(root)
    before = result.read_bytes(), result.stat().st_mtime_ns
    snapshot = report.load_report(root)
    data = snapshot["action_diagnostics"]["units"][0]["episodes"]
    assert data[0]["turn_bouts"] == 2 and data[1]["turn_bouts"] is None
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("resume must not reread a completed tensor analysis"))
    assert actions.analyze(root, resume=True) == result
    assert (result.read_bytes(), result.stat().st_mtime_ns) == before
    with pytest.raises(FileExistsError, match="immutable"):
        actions.analyze(root)
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    outputs = report.action_outputs(snapshot)
    assert list(figures[0].axes[0].lines[0].get_ydata()) == [2.]
    assert list(figures[0].axes[0].lines[0].get_xdata()) == [751 / 1024]
    assert "1 / 3" in outputs[1]["data"]["text/plain"]
    assert "not zero turns" in report.action_notes(snapshot)
    base, output = tmp_path / "base.ipynb", tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), base)
    report.build_notebook(root, output, base)
    book = nbformat.read(output, as_version=4)
    assert any(cell.cell_type == "code" and "display_actions" in cell.source for cell in book.cells)
    assert len([cell for cell in book.cells if cell.cell_type == "markdown"
                and cell.source.startswith("#### Turning During")]) == 1
    rendered = []
    from IPython import display

    monkeypatch.setattr(display, "display", lambda data, **kwargs: rendered.append(data))
    cell = next(cell for cell in book.cells if cell.cell_type == "code" and "display_actions" in cell.source)
    exec(cell.source, {})
    assert len(rendered) == 2
    assert "Recovered Action Coverage" in rendered[-1]["text/plain"]


def test_action_plot_averages_replica_means_not_unequal_episode_counts(monkeypatch):
    import nbformat

    units, diagnoses = [], []
    for seed, bouts in ((1, [0]), (2, [2, 2, 2])):
        name = f"trajectory-{seed}"
        units.append({"name": name, "summary": {"operation": "trajectory", "settings": {
            "phase": "confirmation", "method": "td_fd", "total": 1024, "seed": seed}}})
        diagnoses.append({"name": name, "episodes": [
            {"complete_history": True, "n": 900, "sigma_m": .5, "turn_bouts": count,
             "mean_abs_steering_deg": float(count)} for count in bouts]})
    snapshot = {"units": units, "action_diagnostics": {"units": diagnoses}}
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    report.action_outputs(snapshot)
    assert list(figures[0].axes[0].lines[0].get_ydata()) == [1.]
    assert list(figures[0].axes[0].lines[0].get_xdata()) == [900 / 1024]


def test_checkpoint_checksums_and_scalar_report_seals_are_verified(tmp_path):
    root = tmp_path / "run"
    unit = sealed_unit(root)
    checkpoint = unit / "transactions" / ("a" * 32) / "state.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="checkpoint checksum"):
        actions.analyze(root)
    checkpoint.write_bytes(original)
    result = actions.analyze(root)
    result.write_text(result.read_text() + " ")
    with pytest.raises(report.ArtifactError, match="action-report checksum"):
        report.load_report(root)


def test_versioned_action_report_can_be_selected_without_overwriting_earlier_analysis(tmp_path, monkeypatch):
    import nbformat

    root = tmp_path / "run"
    sealed_unit(root)
    first = actions.analyze(root)
    original = first.read_bytes(), first.stat().st_mtime_ns
    alternate = tmp_path / "complete-cohort-actions"
    second = actions.analyze(root, output=alternate)
    assert first != second
    snapshot = report.load_report(root, action_report=alternate)
    assert len(snapshot["action_diagnostics"]["units"]) == 1
    base, output = tmp_path / "base.ipynb", tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), base)
    monkeypatch.setattr(report, "_figure", lambda figure, description:
                        nbformat.v4.new_output("display_data", data={"text/plain": description}))
    report.build_notebook(root, output, base, action_report=alternate)
    book = nbformat.read(output, as_version=4)
    code = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    assert "action_report=" in code and str(alternate) in code
    assert (first.read_bytes(), first.stat().st_mtime_ns) == original
    with pytest.raises(report.ArtifactError, match="explicit action report is not complete"):
        report.load_report(root, action_report=tmp_path / "absent")
