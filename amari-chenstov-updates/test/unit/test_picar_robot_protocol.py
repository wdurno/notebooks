from dataclasses import asdict, replace
import math
import subprocess
import sys

import pytest

from src.robotics import (
    ACTION_VECTORS, Calibration, Control, RobotCommand, action_distribution_to_vector,
)
from src.robotics.drift import drift_alpha, steering_gain
from src.robotics.geometry import (
    TerminalGeometry, in_success_region, target_geometry, terminal_event,
)


def test_golden_legacy_mapping_and_one_si_calibration():
    golden = ((0, 0, -1, 1), (0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 0, -1),
              (1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0))
    assert ACTION_VECTORS == golden
    calibration = Calibration()
    for i, vector in enumerate(golden):
        probabilities = [float(i == j) for j in range(8)]
        assert action_distribution_to_vector(probabilities) == vector
        control = calibration.map_distribution(probabilities)
        assert control.pan_rad == pytest.approx(vector[0] * math.radians(60))
        assert control.tilt_rad == pytest.approx(math.radians(-8 + vector[1] * 38))
        assert control.steering_rad == pytest.approx(vector[2] * math.radians(25))
        assert control.drive_mps == pytest.approx(vector[3] * (.2 if vector[3] >= 0 else .1))
        calibration.validate(control)


def test_mixtures_not_sampling_and_opposites_cancel_before_calibration():
    c = Calibration()
    assert c.map_distribution([0, 0, .5, .5, 0, 0, 0, 0]) == c.neutral()
    assert c.map_distribution([0, 0, 0, 0, .5, .5, 0, 0]) == c.neutral()
    assert c.map_distribution([.5, .5, 0, 0, 0, 0, 0, 0]).steering_rad == 0
    assert c.map_distribution([.5, .5, 0, 0, 0, 0, 0, 0]).drive_mps == .2
    assert c.map_distribution([0, 0, 0, 0, .5, 0, .5, 0]).drive_mps == 0
    assert c.neutral(Control(pan_rad=.3)).pan_rad == .3
    assert c.neutral().tilt_rad != 0
    # Steering without drive cannot trigger legacy automatic forward motion.
    c.validate(Control(steering_rad=.2, drive_mps=0))


@pytest.mark.parametrize("values", [[0] * 8, [1] * 8, [-.1, 1.1] + [0] * 6,
                                   [float("nan")] + [0] * 7, [1]])
def test_invalid_action_probabilities_fail(values):
    with pytest.raises(ValueError):
        action_distribution_to_vector(values)


def test_wire_schema_units_bounds_and_immutable_command():
    command = RobotCommand("run", "session", 0, "frame", Control(), 1., .5, 1.5)
    assert RobotCommand.from_dict(command.to_dict()) == command
    with pytest.raises(ValueError):
        RobotCommand.from_dict({k: v for k, v in asdict(command).items() if k != "schema_version"})
    for changed in ({"sequence_id": True}, {"duration_s": 0}, {"expires_at_s": math.inf},
                    {"expires_at_s": .1}, {"schema_version": 99}):
        with pytest.raises(ValueError):
            replace(command, **changed)
    with pytest.raises(ValueError):
        Calibration().validate(Control(drive_mps=.21))
    with pytest.raises(ValueError):
        Control(drive_mps=math.nan)


def test_float32_probability_roundoff_is_renormalized_not_rejected():
    import torch
    values = torch.tensor([.1, .2, .3, .4, 0, 0, 0, 0], dtype=torch.float32)
    total = sum(float(p) for p in values)
    assert abs(total - 1) > 1e-8
    result = action_distribution_to_vector(values)
    assert result[2] == pytest.approx((float(values[1]) - float(values[0])) / total, abs=1e-14)
    assert result[3] == pytest.approx(sum(float(values[i]) * ACTION_VECTORS[i][3]
                                         for i in range(8)) / total, abs=1e-14)
    Calibration().validate(Calibration().map_distribution(torch.softmax(torch.arange(8.).float(), dim=0)))
    with pytest.raises(ValueError):
        action_distribution_to_vector([1.00001] + [0] * 7)


def test_smooth_outward_return_schedule_and_continuous_first_two_derivatives():
    for fraction, expected in ((0, 0), (.1, 0), (.45, 1), (.5, 1), (.55, 1), (.9, 0), (1, 0)):
        assert drift_alpha(fraction * 1000, 1000) == pytest.approx(expected)
    assert steering_gain(1) == pytest.approx(.65)
    for x in range(101):
        assert drift_alpha(x * 10, 1000) == pytest.approx(drift_alpha(1000 - x * 10, 1000))
    eps = .001
    for boundary in (100., 450., 550., 900.):
        left, center, right = [drift_alpha(boundary + offset, 1000) for offset in (-eps, 0, eps)]
        assert abs((right - left) / (2 * eps)) < 1e-7
        assert abs((right - 2 * center + left) / eps ** 2) < 1e-6
    assert 0 < drift_alpha(100.5, 1000) < drift_alpha(101, 1000)


def test_precise_bumper_clearance_and_terminal_precedence():
    # Center .61 from target: .16 bumper offset + .10 radius + .35 clearance.
    g = target_geometry((2.29, 0), 0., 0., (2.9, 0), False)
    assert g.clearance_m == pytest.approx(.35)
    assert in_success_region(g)
    assert terminal_event(g, .999, 19.99, 20) is None
    assert terminal_event(g, 1., 20., 20.) == "success"
    assert terminal_event(replace(g, contact=True), 1., 20., 20.) == "collision"
    assert terminal_event(g, .5, 20., 20.) == "deadline"
    for changed in ({"speed_mps": .0201}, {"heading_error_rad": math.radians(15.01)},
                    {"clearance_m": .2499}, {"clearance_m": .4501}):
        assert not in_success_region(replace(g, **changed))


def test_pi_import_has_no_heavy_optional_dependencies():
    code = ("import sys; from demo.picar.service import RobotService; "
            "assert not {'torch','numpy','pybullet','transformers'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
