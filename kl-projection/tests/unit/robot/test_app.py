import inspect

from picar_kl.robot import app as robot_app


def test_robot_parser_accepts_no_camera_flag():
    args = robot_app.build_parser().parse_args(["--no-camera", "--camera-index", "1"])

    assert args.no_camera is True
    assert args.camera_index == 1


def test_controller_startup_does_not_import_legacy_camera_module():
    source = inspect.getsource(robot_app.LegacyPiCarController.__init__)

    assert "legacy_car_env" not in source
    assert "VideoCapture" not in source
