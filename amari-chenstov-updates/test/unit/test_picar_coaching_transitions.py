from dataclasses import asdict

import pytest
import torch

from numerical_experiment.picar_coaching.transitions import execution_segments, decision_returns
from src.robotics.actions import Calibration


def test_terminal_bonus_and_guidance_survive_fallback_gaps_without_fake_actions():
    transitions = decision_returns(torch.tensor([0., .2, .2, 10.]), torch.tensor([.9] * 4),
                                   torch.tensor([False, False, False, True]),
                                   torch.tensor([False, True, False, False]))
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["first"] == 1 and transition["last"] == 3
    assert float(transition["reward"]) == pytest.approx(.2 + .9 * .2 + .9**2 * 10)
    assert float(transition["discount"]) == 0.
    assert not decision_returns(torch.ones(3), torch.ones(3) * .9,
                                torch.tensor([False, False, True]), torch.zeros(3, dtype=torch.bool))


def test_consecutive_policy_decisions_preserve_original_one_step_targets():
    transitions = decision_returns(torch.tensor([1., 2., 3.]), torch.tensor([.9, .8, .7]),
                                   torch.tensor([False, False, True]), torch.ones(3, dtype=torch.bool))
    assert [r["first"] for r in transitions] == [0, 1, 2]
    torch.testing.assert_close(torch.stack([r["reward"] for r in transitions]), torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(torch.stack([r["discount"] for r in transitions]), torch.tensor([.9, .8, 0.]))


def test_execution_journal_includes_backend_expiry_neutralization():
    calibration = Calibration()
    control = calibration.map_distribution([1., 0., 0., 0., 0., 0., 0., 0.])
    snapshot = {"config": {"physics_dt_s": .1, "calibration": asdict(calibration)},
                "clock_start_ticks": 0, "events": [
                    {"kind": "command", "controls": asdict(control), "deadline_s": .55},
                    {"kind": "advance", "physics_steps": 10, "training": False}]}
    segments = execution_segments(snapshot, 0., 1.)
    assert len(segments) == 2
    assert segments[0]["end_time"] == pytest.approx(.6)
    assert segments[0]["executed"][3] > 0
    assert segments[1]["executed"][2:] == [0., 0.]
    assert segments[1]["source"] == "expiry_neutral"
    assert sum(s["end_time"] - s["start_time"] for s in segments) == pytest.approx(1.)
