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



class FakeCapture:
    def __init__(self, *, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened and not self.released

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        if self.frames:
            frame = self.frames.pop(0)
            return frame is not None, frame
        return False, None

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, captures):
        self.captures = captures
        self.opened_indexes = []

    def VideoCapture(self, index):
        self.opened_indexes.append(index)
        return self.captures.get(index, FakeCapture(opened=False))


def test_robot_parser_defaults_camera_auto():
    args = robot_app.build_parser().parse_args([])

    assert args.camera_index == "auto"


def test_robot_parser_parses_explicit_camera_index():
    args = robot_app.build_parser().parse_args(["--camera-index", "1"])

    assert args.camera_index == 1


def test_auto_camera_detection_skips_open_but_unreadable_index():
    frame = [[[1, 2, 3]]]
    cv2 = FakeCV2(
        {
            0: FakeCapture(opened=True, frames=[None, None, None]),
            1: FakeCapture(opened=True, frames=[frame]),
        }
    )

    capture, camera_index = robot_app._open_first_readable_camera(cv2, max_index=2)

    assert camera_index == 1
    assert capture is cv2.captures[1]
    assert cv2.opened_indexes == [0, 1]
    assert cv2.captures[0].released is True


def test_auto_camera_detection_reports_failures():
    cv2 = FakeCV2({0: FakeCapture(opened=False), 1: FakeCapture(opened=True, frames=[None])})

    try:
        robot_app._open_first_readable_camera(cv2, max_index=2)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected camera detection failure")

    assert "0: open failed" in message
    assert "1: read failed" in message
