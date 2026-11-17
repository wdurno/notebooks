"""Reward-compatible offline fitting never reuses the old Fisher or repeats KL."""

import copy
from dataclasses import replace

import pytest
import torch

from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar_initialization import numerics as nm
from numerical_experiment.picar_reward_repair.config import CONDITIONS, configuration
from numerical_experiment.picar_reward_repair.data import relabel
from numerical_experiment.picar_reward_repair.initialization import RewardInitialization, pristine_from_distilled
from test_picar_initialization_numerics import MemoryStore, record, equal


@pytest.fixture(autouse=True)
def small_policy(monkeypatch):
    monkeypatch.setattr(nm, "FIT_POLICY", {**nm.FIT_POLICY,
        "nll_max_passes": 2, "min_passes": 1, "block_items": 2, "diagnostic_histories": 1})


def setup():
    config = configuration(offline_fit=True)
    config = replace(config, learning=replace(config.learning, learning_rate=.01))
    model = build_tiny_model(31, hidden_size=2, width=4, depth=1, return_bins=3, context_length=4)
    pristine = MemoryLearner(model, config.learning, config.seeds(0)).state_dict()
    train = [record(0, "a", 1), record(1, "a", 1)]
    validation = [record(3, "v", 1)]
    return model, config, pristine, train, validation


def construct(values, store, *, stopped=lambda: None, fitted=None):
    model, config, pristine, train, validation = values
    return RewardInitialization(model, config, train, validation, store, stopped,
        pristine=pristine, acquired_count=5, fitted_state=fitted,
        distill_receipt={"stage": "one_time_multiepoch_recorded_teacher_kl", "source": "sealed_test"},
        shared_kl_selection={"epoch": 2, "completed_passes": 2})


def test_shared_KL_skipped_and_four_draw_fisher_built_at_new_fitted_anchor():
    values = setup()
    source = copy.deepcopy(values[2])
    fit = construct(values, MemoryStore())
    summary = fit.run_fit()
    assert summary["shared_KL_reused"]
    assert summary["counts"]["optimization_steps"]["kl"] == 0
    assert [r["stage"] for r in summary["curves"]] == ["nll"] * 3
    assert summary["initial_pi_used"] is summary["initial_penalties_used"] is False
    assert summary["counts"]["optimization_exposures"]["nll"] == 4
    equal(values[2], source)
    fisher = construct(values, MemoryStore(), fitted=copy.deepcopy(fit.s))
    result = fisher.run_fisher()
    assert result["archive"]["retired_count"] == 2
    assert result["archive"]["eligible_count"] == 0
    assert result["normalization"]["history_weight"] == .5
    assert result["normalization"]["pi"] is None


def test_initialization_halt_resume_is_exact_and_never_repeats_distillation():
    control = construct(setup(), MemoryStore())
    control.run_fit()
    interrupted_store = MemoryStore()
    def halt():
        raise InterruptedError("test stop after committed block")
    with pytest.raises(InterruptedError):
        construct(setup(), interrupted_store, stopped=halt).run_fit()
    resumed = construct(setup(), interrupted_store)
    resumed.run_fit()
    equal(control.s["learner"], resumed.s["learner"])
    equal(control.s["curves"], resumed.s["curves"])
    assert resumed.s["optimization_steps"]["kl"] == 0


def test_pristine_reset_copies_parameters_but_discards_old_archive_and_moments():
    _, _, state, _, _ = setup()
    state["retired_count"], state["update_count"] = 100, 200
    for value in state["diagonal"].values():
        value.fill_(17.)
    original = copy.deepcopy(state)
    clean = pristine_from_distilled({"distilled_learner": state})
    assert clean["retired_count"] == clean["update_count"] == 0
    assert clean["optimizer"]["state"] == {}
    assert all(not bool(v.count_nonzero()) for v in clean["diagonal"].values())
    for name in clean["anchor"]:
        equal(clean["anchor"][name], clean["parameters"][name])
    equal(state, original)


def test_relabel_includes_fallback_rewards_but_baseline_uses_training_only():
    episodes = []
    records = {}
    scores = {}
    for split, bonus, q in (("train", 10., 7.), ("validation", -5., 2.)):
        rows = [{"arrival": i, "frame_ref": f"{split}-{i}", "elapsed_s": .5, "terminated": i == 1} for i in range(2)]
        episodes.append({"episode": {"episode_id": split, "intervals": rows,
            "terminal_bonuses": [0., bonus], "end_reason": "success" if bonus > 0 else "collision"}})
        # First interval has no policy record but its reward still affects returns.
        records[split] = [{"record_id": split+"/1", "episode_id": split, "source_arrival": 1}]
        scores.update({r["frame_ref"]: {"score": q} for r in rows})
    data = {**records, "episodes": episodes}
    result = relabel(data, scores, CONDITIONS[1], torch.linspace(-6., 12., 162))
    expected = result["train"][0]["return"]
    assert result["baseline_coded"] == expected
    assert result["validation"][0]["baseline"] == expected
    assert result["validation"][0]["return"] != expected
    assert len(result["episode_rows"]) == 2
