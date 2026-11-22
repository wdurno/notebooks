"""CPU-only predictive arithmetic and nonlearning reference evaluation."""

import copy
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar.learner import LearningBatch, predict_histories
from numerical_experiment.picar_curriculum_repair.predictive import (
    evaluate_record, evaluate_records, score_prediction, summarize_scores,
)
from src.continual import picar_math as pm


def inputs(dtype=torch.float64):
    return {"prediction": {"return_logits": torch.tensor([[.3, -.7, 1.1]], dtype=dtype),
                           "action_mean": torch.tensor([[.1, -.2, .3, -.4, .5, -.6, .7]], dtype=dtype)},
            "label": 2, "action": torch.tensor([-.3, .7, -.1, .4, .2, .1, -.6], dtype=dtype),
            "baseline": .125, "bin_values": torch.tensor([-2., .25, 3.], dtype=dtype),
            "temperature": 1.75, "action_variance": 2.25}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("label", [0, 1, 2])
def test_joint_decomposition_matches_frozen_likelihood(dtype, label):
    args = inputs(dtype)
    args["label"] = label
    row = score_prediction(**args)
    design = pm.LikelihoodDesign(args["bin_values"], torch.tensor([args["baseline"]], dtype=dtype),
                                 args["temperature"], torch.eye(7, dtype=dtype) * args["action_variance"])
    expected = pm.joint_nll(args["prediction"], torch.tensor([label]), args["action"][None], design)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    assert row["joint_nll"] == pytest.approx(float(expected[0]), rel=tolerance, abs=tolerance)
    assert row["joint_nll"] == row["return_nll"] + row["action_nll"]
    assert row["action_nll"] == row["action_quadratic"] + row["action_log_normalizer"]
    assert row["weight"] == pytest.approx(math.exp((float(args["bin_values"][label]) - .125) / 1.75))


@pytest.mark.parametrize("baseline,temperature,variance", [(-1.2, 2., .3), (2.5, .8, 4.), (0., 10., 1.)])
def test_manual_full_normalized_gaussian(baseline, temperature, variance):
    args = inputs()
    args.update(baseline=baseline, temperature=temperature, action_variance=variance)
    row = score_prediction(**args)
    logits = args["prediction"]["return_logits"][0].tolist()
    maximum = max(logits)
    categorical = maximum + math.log(sum(math.exp(x - maximum) for x in logits)) - logits[args["label"]]
    weight = math.exp((float(args["bin_values"][args["label"]]) - baseline) / temperature)
    conditional_variance = variance / weight
    squared_error = sum((a - b) ** 2 for a, b in zip(args["action"].tolist(), args["prediction"]["action_mean"][0].tolist()))
    gaussian = .5 * squared_error / conditional_variance + 3.5 * math.log(2 * math.pi * conditional_variance)
    assert row["return_nll"] == pytest.approx(categorical, abs=1e-14)
    assert row["action_nll"] == pytest.approx(gaussian, abs=1e-13)
    assert row["joint_nll"] == pytest.approx(categorical + gaussian, abs=1e-13)


def test_labels_and_saved_baselines_change_the_declared_conditional_model():
    args = inputs()
    first = score_prediction(**args)
    changed_label = score_prediction(**{**args, "label": 0})
    changed_baseline = score_prediction(**{**args, "baseline": 1.125})
    assert first["return_nll"] != changed_label["return_nll"]
    assert first["action_nll"] != changed_label["action_nll"]
    assert first["return_nll"] == changed_baseline["return_nll"]
    assert changed_baseline["weight"] / first["weight"] == pytest.approx(math.exp(-1 / args["temperature"]))
    assert first["action_nll"] != changed_baseline["action_nll"]


def test_deterministic_action_keeps_normalizers_and_negative_density_nll():
    args = inputs()
    args.update(action=args["prediction"]["action_mean"][0], action_variance=.001)
    row = score_prediction(**args)
    assert row["action_quadratic"] == 0.
    assert row["action_nll"] == row["action_log_normalizer"] < 0.
    assert row["joint_nll"] < 0.


def test_save_and_exact_recompute_from_float32_outputs_and_python_lists(tmp_path):
    args = inputs(torch.float32)
    for value in args["prediction"].values():
        value.requires_grad_()
    before = copy.deepcopy(args)
    row = score_prediction(**args)
    path = tmp_path / "predictive.json"
    path.write_text(json.dumps(row, allow_nan=False))
    saved = json.loads(path.read_text())
    recovered = score_prediction(saved, **{name: saved[name] for name in
        ("label", "action", "baseline", "bin_values", "temperature", "action_variance")})
    assert recovered == saved
    _assert_same(args, before)
    assert all(not isinstance(value, torch.Tensor) for value in row.values())
    # Matches the current-policy student integration, including tensor bin values.
    result = score_prediction({key: value[0].tolist() for key, value in args["prediction"].items()},
                              **{key: value.tolist() if key == "action" else value
                                 for key, value in args.items() if key != "prediction"})
    assert result == row


@pytest.mark.parametrize("field,value", [
    ("label", -1), ("label", 3), ("label", 1.), ("label", True), ("label", torch.tensor([1])),
    ("label", torch.tensor(1.)), ("action", [0.] * 8), ("action", [[0.] * 7]),
    ("action", [math.nan] * 7), ("action", [math.inf] * 7), ("action", [True] * 7),
    ("baseline", math.nan), ("baseline", math.inf), ("baseline", [0.]),
    ("temperature", 0.), ("temperature", -1.), ("temperature", math.inf), ("temperature", math.nan),
    ("temperature", [1.]), ("action_variance", 0.), ("action_variance", -1.),
    ("action_variance", math.nan), ("action_variance", math.inf), ("action_variance", [1.]),
    ("bin_values", [0.]), ("bin_values", []), ("bin_values", [1., 0., 2.]),
    ("bin_values", [0., 0., 1.]), ("bin_values", [0., 1., math.inf]),
    ("bin_values", [[-2., .25, 3.]]), ("baseline", 1e4), ("baseline", -1e4),
])
def test_invalid_targets_and_design_fail_without_clipping(field, value):
    args = inputs()
    args[field] = value
    with pytest.raises(ValueError):
        score_prediction(**args)


@pytest.mark.parametrize("prediction", [
    None, {}, {"return_logits": [0., 0., 0.]}, {"action_mean": [0.] * 7},
    {"return_logits": [0., 0.], "action_mean": [0.] * 7},
    {"return_logits": [[0., 0., 0.]] * 2, "action_mean": [0.] * 7},
    {"return_logits": [0., 0., 0.], "action_mean": [[0.] * 7] * 2},
    {"return_logits": [0., 0., 0.], "action_mean": [0.] * 8},
    {"return_logits": [0., math.nan, 0.], "action_mean": [0.] * 7},
    {"return_logits": [0., math.inf, 0.], "action_mean": [0.] * 7},
    {"return_logits": [0., 0., 0.], "action_mean": [math.inf] * 7},
    {"return_logits": [0., 0., 0.], "action_mean": [1j] * 7},
])
def test_missing_or_invalid_heads_are_errors_not_fabricated_predictions(prediction):
    with pytest.raises(ValueError):
        score_prediction(**{**inputs(), "prediction": prediction})


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([.3, -.7, 1.1]))
        self.mean = torch.nn.Parameter(torch.arange(7, dtype=torch.float32) / 10.)
        self.norm = torch.nn.BatchNorm1d(7)
        self.dropout = torch.nn.Dropout(.8)

    def forward(self, history):
        assert not self.training and not self.norm.training and not self.dropout.training
        assert not torch.is_grad_enabled()
        n = history["images"].shape[0]
        length = history["images"].shape[1]
        return {"return_logits": self.logits.expand(n, -1) * length,
                "action_mean": self.dropout(self.norm(self.mean.expand(n, -1)))}


class _Learner:
    def __init__(self):
        self.model = _Model()
        self.model.train()
        self.model.norm.eval()  # Mixed modes must be restored exactly.
        self.params = dict(self.model.named_parameters())
        self.params["mean"].grad = torch.full_like(self.params["mean"], 3.)
        self.edges = torch.tensor([-3., -1., 1., 3.])
        self.config = SimpleNamespace(temperature=1.3, action_variance=2.25)
        self.generators = {name: torch.Generator().manual_seed(index + 10)
                           for index, name in enumerate(("exploration", "fd_probes", "fisher_labels"))}
        self.version, self.retired_count, self.update_count = 4, 8, 16
        self.anchor = {"logits": self.model.logits.detach().clone()}
        self.optimizer_state = {"moment": torch.ones(7)}
        self._cooperate = None

    def predict(self, params, batch):
        assert isinstance(batch, LearningBatch)
        assert params is self.params
        return predict_histories(self.model, params, batch.histories)

    def choose_action(self, *args, **kwargs):
        raise AssertionError("predictive scoring must not call choose_action")


def records():
    result = []
    for index in range(3):
        length = index + 1
        history = {"images": torch.ones(1, length, 3, 2, 2),
                   "executed_actions": torch.zeros(1, length, 4),
                   "elapsed": torch.full((1, length, 1), .5),
                   "contexts": [[{"text": "Navigate.", "issued_at": 0.}]],
                   "decision_time": torch.arange(length, dtype=torch.float32).reshape(1, -1)}
        result.append({"history": history, "bin": index, "baseline": index / 4.,
                       "logits": torch.arange(7, dtype=torch.float32) / 8., "arrival": 10 + index,
                       "record_id": f"ref:{index}", "episode_id": "ep0" if index < 2 else "ep1",
                       "environment": "B_initial"})
    return result


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


def _state(learner):
    return copy.deepcopy({"model": learner.model.state_dict(),
        "grads": {key: value.grad for key, value in learner.params.items()},
        "modes": [m.training for m in learner.model.modules()],
        "generators": {key: g.get_state() for key, g in learner.generators.items()},
        "global_torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
        "counts": (learner.version, learner.retired_count, learner.update_count),
        "anchor": learner.anchor, "optimizer": learner.optimizer_state})


def test_raw_reference_evaluation_is_json_friendly_and_preserves_state_and_rng():
    learner, bank = _Learner(), records()
    before, original_bank = _state(learner), copy.deepcopy(bank)
    rows = evaluate_records(learner, bank)
    assert isinstance(rows, list) and len(rows) == 3
    _assert_same(_state(learner), before)
    _assert_same(bank, original_bank)
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows
    for record, row in zip(bank, rows):
        assert row["record_id"] == record["record_id"]
        assert row["episode_id"] == record["episode_id"] and row["environment"] == "B_initial"
        assert row["baseline"] == record["baseline"] and row["label"] == record["bin"]
        assert row["return_logits"] == pytest.approx((learner.model.logits * record["history"]["images"].shape[1]).tolist())
        assert row["action"] == record["logits"].tolist()
        assert row["aggregation_weight"] == 1.


def test_single_record_cursor_and_reordering_keep_identity_and_scores():
    learner, bank = _Learner(), records()
    rows = evaluate_records(learner, bank)
    assert evaluate_record(learner, bank[0]) == rows[0]
    assert evaluate_records(learner, [bank[0]])[0] == rows[0]
    assert evaluate_records(learner, bank, start_cursor=1) == rows[1:]
    assert evaluate_records(learner, bank, start_cursor=3) == []
    assert evaluate_records(learner, bank[::-1]) == rows[::-1]
    for record in bank:
        record.pop("record_id")
    implicit = evaluate_records(learner, bank)
    assert implicit[0]["record_id"] == '["ep0",10]'
    assert evaluate_records(learner, bank, start_cursor=1)[0]["record_id"] == implicit[1]["record_id"]


def test_summary_is_explicitly_record_weighted_not_episode_or_precision_weighted():
    rows = evaluate_records(_Learner(), records())
    summary = summarize_scores(rows)
    assert summary["weighting"] == "equal_per_record"
    assert summary["record_count"] == 3 and summary["total_weight"] == 3.
    for key in ("return_nll", "action_nll", "joint_nll", "weight"):
        assert summary["mean_" + key] == pytest.approx(sum(row[key] for row in rows) / 3)
    a, b, c = [row["joint_nll"] for row in rows]
    assert summary["mean_joint_nll"] != pytest.approx(((a + b) / 2 + c) / 2)
    assert summary["mean_joint_nll"] != pytest.approx(sum(r["weight"] * r["joint_nll"] for r in rows) / sum(r["weight"] for r in rows))
    assert summarize_scores([])["mean_joint_nll"] is None
    with pytest.raises(ValueError, match="aggregation_weight"):
        summarize_scores([{**rows[0], "aggregation_weight": rows[0]["weight"]}])


@pytest.mark.parametrize("field", ["history", "bin", "baseline", "logits", "arrival"])
def test_missing_record_data_errors(field):
    record = records()[0]
    record.pop(field)
    with pytest.raises(ValueError, match="reference record requires"):
        evaluate_records(_Learner(), [record])


@pytest.mark.parametrize("cursor", [-1, 4, True, 1.5])
def test_bad_cursor_errors(cursor):
    with pytest.raises(ValueError, match="start_cursor"):
        evaluate_records(_Learner(), records(), start_cursor=cursor)


def test_empty_bank_and_duplicate_identity_error():
    learner, bank = _Learner(), records()
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_records(learner, [])
    with pytest.raises(ValueError, match="unique"):
        evaluate_records(learner, [bank[0], bank[0]])


def test_missing_prediction_restores_modes_and_state(monkeypatch):
    learner = _Learner()
    before = _state(learner)
    monkeypatch.setattr(learner, "predict", lambda params, batch: {"action_mean": torch.zeros(1, 7)})
    with pytest.raises(ValueError, match="return_logits and action_mean"):
        evaluate_records(learner, records())
    _assert_same(_state(learner), before)


def test_scoring_rejects_active_cooperative_learning():
    learner = _Learner()
    learner._cooperate = lambda: None
    before = _state(learner)
    with pytest.raises(RuntimeError, match="quiescent"):
        evaluate_records(learner, records())
    _assert_same(_state(learner), before)
