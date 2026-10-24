"""Fast Plan 15 drip protocol checks; no training."""

import dataclasses
from types import SimpleNamespace

import nbformat
import pytest
import torch

from mnist_experiment.fisher_memory import drip
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import Config
from mnist_experiment.fisher_memory.data import class_weights, plan_data
from mnist_experiment.fisher_memory.drip_report import _pilot_readout, build_notebook
from mnist_experiment.fisher_memory.drip_runner import DripRunner
from mnist_experiment.fisher_memory.metrics import predictive_metrics


class Source:
    train_labels = torch.arange(10000) % 10
    test_labels = torch.arange(4000) % 10


def test_drip_plan_has_exact_unique_events_and_disjoint_panels():
    cfg = dataclasses.replace(Config(), init_size=100, reference_size=20, eval_per_class=5)
    plan = plan_data(Source(), cfg, drip.FAMILY, 1, "drip")
    assert plan == plan_data(Source(), cfg, drip.FAMILY, 1, "drip")
    assert plan["event_target"] == 100
    assert plan["event_batches"][0]["probability"] == 0
    assert plan["event_batches"][39]["probability"] == pytest.approx(.1)
    assert plan["event_batches"][-1]["cumulative_events"] == 100
    assert sum(sum(batch["rotated"]) for batch in plan["event_batches"]) == 100
    assert len(plan["stream_ids"]) == len(set(plan["stream_ids"]))
    assert not set(plan["stream_ids"]) & set(plan["initialization_ids"])
    assert not set(plan["stream_ids"]) & set(plan["reference_ids"])
    confirm = plan_data(Source(), cfg, "drip_confirmation", 1, "drip")
    assert not set(plan["evaluation_ids"]) & set(confirm["evaluation_ids"])


def test_panel_nll_matches_fixed_class_weighted_predictive_loss():
    labels = torch.tensor([c for c in range(10) for _ in range(1 + c % 3)])
    q = torch.arange(1, 11, dtype=torch.float64)
    q /= q.sum()
    logits = torch.arange(len(labels) * 10, dtype=torch.float64).reshape(-1, 10) / 100
    weights = class_weights(labels, q.tolist())
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = sum(float(q[c] * losses[labels == c].mean()) for c in range(10))
    assert float(weights.sum()) == pytest.approx(1.)
    assert predictive_metrics(logits, labels, weights)["nll"] == pytest.approx(expected)
    assert predictive_metrics(logits, labels, weights, p_ref=.01)["nll"] == pytest.approx(expected)
    assert predictive_metrics(logits, labels, weights, p_ref=.9)["nll"] == pytest.approx(expected)


def test_batch_evaluation_reuses_separate_panel_loss_at_crossed_event_counts():
    labels = torch.tensor([0, 9])
    logits = torch.zeros(2, 10, dtype=torch.float64)
    logits[0, 0], logits[1, 9] = 2., 4.
    weights = torch.tensor([.7, .3], dtype=torch.float64)

    def score(model, layout, theta, inputs):
        assert inputs is logits
        return inputs

    runner = SimpleNamespace(_logits=score)
    panels = {"rotated": (logits, labels, weights)}
    rows = []
    DripRunner._evaluate(runner, None, None, None, panels, 0, 0, 0, 0, "mixed", rows)
    DripRunner._evaluate(runner, None, None, None, panels, 3, 12, 0, 2, "mixed", rows)
    DripRunner._evaluate(runner, None, None, None, panels, 4, 16, 2, 2, "mixed", rows)
    expected = predictive_metrics(logits, labels, weights)["nll"]
    assert [row["paired_rotated_event"] for row in rows] == [0, 1, 2]
    assert [row["rotated_nll"] for row in rows] == pytest.approx([expected] * 3)
    assert [(row["step"], row["exposure"]) for row in rows] == [(0, 0), (3, 12), (3, 12)]


def test_selection_respects_retention_then_warns_on_fallback():
    rows = [dict(strength=.1, valid=True, mean_gain=.1, mean_harm=.02, online_seconds=2),
            dict(strength=1., valid=True, mean_gain=.05, mean_harm=.005, online_seconds=3)]
    chosen, classification = drip.select(rows, .01)
    assert chosen["strength"] == 1.
    assert classification == "retention_feasible"
    chosen, classification = drip.select(rows[:1], .01)
    assert chosen["strength"] == .1
    assert classification == "pilot_warning_acquisition_fallback"
    with pytest.raises(ValueError, match="numerically invalid"):
        drip.select([dict(valid=False)], .01)


def test_report_replaces_only_tagged_section(tmp_path):
    base = tmp_path / "base.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Existing"),
                                           nbformat.v4.new_markdown_cell("## Keep this")])
    nbformat.write(notebook, base)
    root = tmp_path / "run"
    with Store(root, {"test": True}) as store:
        store.begin(drip.LEDGER, {"kind": "drip_ledger"})
        store.write_json(drip.LEDGER, "result.json", drip._ledger(Config(smoke=True)))
        store.complete(drip.LEDGER)
        store.write_json(None, "status.json", dict(stage="diagnostic", active_seconds=0))
        first = build_notebook(base, root)
        nbformat.write(first, base)
        second = build_notebook(base, root)
    assert sum("Phase 5B" in cell.source for cell in second.cells) == 1
    assert second.cells[1].source == "## Keep this"
    assert "Development pilot only" in second.cells[2].source
    assert [cell.source for cell in first.cells] == [cell.source for cell in second.cells]


def test_pilot_readout_distinguishes_mixed_from_base_only_gain():
    units = []
    for replica in (3, 4):
        for stream, condition, rotated in (("mixed", "replay_ewc", 2.),
                                           ("mixed", "finite_difference", 1.5),
                                           ("base", "replay_ewc", 3.),
                                           ("base", "finite_difference", 2.6)):
            units.append(dict(metadata=dict(replica=replica, stream=stream,
                                            condition=condition,
                                            treatment=dict(strength=160.89 if condition == "finite_difference" else 0.)),
                              summary=dict(rotated_auc=rotated, base_auc=.4,
                                           active_seconds=300.)))
    pilot = dict(selected_strength=160.89, grid_with_boundary=[.1, 160.89],
                 table=[dict(strength=160.89, mean_gain=.5, mean_harm=0.)])
    result = _pilot_readout(units, pilot, dict(screen_start=3, screen_replicas=2))
    assert "mixed-specific difference" in result
    assert "+0.100 NLL-AUC" in result
    assert "not unique to rare-domain exposure" in result
