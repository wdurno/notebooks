"""Supplementary figures preserve the estimands and bind links to their cohort."""

from copy import deepcopy

import numpy as np

from mnist_experiment import plan15_regression_diagnostics as diagnostics
from mnist_experiment.fisher_memory_regression import report
from test_plan15_regression_analysis_report import snapshot, historical_notebook


def test_per_digit_figure_uses_root_mean_square_error_across_replicas():
    rows = [{"method": "replay_ewc", "metrics": [
        {"per_digit_mse": {domain: {str(k): 1. for k in range(10)} for domain in ("base", "novel")}},
        {"per_digit_mse": {domain: {str(k): value for k in range(10)} for domain in ("base", "novel")}}
    ]} for value in (1., 9.)]
    figure = diagnostics.per_digit(rows, rows)
    for axis in figure.axes:
        np.testing.assert_allclose(axis.lines[0].get_ydata(), np.sqrt(5.))
        np.testing.assert_allclose(axis.lines[1].get_ydata(), 1.)
    assert diagnostics.png(figure)


def test_renderer_examples_use_cached_pixels_and_declared_angles():
    import torch

    class Source:
        test_labels = torch.arange(10)

        def images(self, ids, *, test=False):
            assert ids == list(range(10))
            assert test
            result = torch.zeros((10, 1, 28, 28))
            result[:, :, 7:20, 12:15] = 1
            return result

    result = diagnostics.examples(Source(), {"evaluation_ids": {"base": list(range(10))}}, lambda: None)
    assert len(result.axes) == 30
    assert [result.axes[i * 10].get_ylabel() for i in range(3)] == ["-40 degrees", "+0 degrees", "+40 degrees"]
    assert diagnostics.png(result)


def test_supplement_link_survives_refresh_only_for_the_same_analysis():
    value = snapshot(complete=True)
    value["analysis"]["manifest_hash"] = "frozen-cohort"
    value["diagnostics"] = {"manifest_hash": "frozen-cohort", "href": "../cache/report.html"}
    previous = historical_notebook()
    book = report.build_notebook(value, previous)
    assert "Supplementary diagnostics" in "\n".join("".join(c["source"]) for c in book["cells"])
    refreshed = deepcopy(value)
    del refreshed["diagnostics"]
    again = report.build_notebook(refreshed, book)
    assert "Supplementary diagnostics" in "\n".join("".join(c["source"]) for c in again["cells"])
    refreshed["analysis"]["manifest_hash"] = "different-cohort"
    changed = report.build_notebook(refreshed, again)
    assert "Supplementary diagnostics" not in "\n".join("".join(c["source"]) for c in changed["cells"])
    assert [c for c in changed["cells"] if not report._owned(c)] == previous["cells"]


def test_supplement_layout_is_drawn_before_export(monkeypatch):
    figure = diagnostics.figure()
    figure.subplots().set(title="Visible labels", xlabel="Digit", ylabel="Error")
    events = []
    draw, save = figure.canvas.draw, figure.savefig

    def drawn():
        events.append("drawn")
        return draw()

    def exported(*args, **kwargs):
        assert events == ["drawn"]
        return save(*args, **kwargs)

    monkeypatch.setattr(figure.canvas, "draw", drawn)
    monkeypatch.setattr(figure, "savefig", exported)
    assert diagnostics.png(figure)
