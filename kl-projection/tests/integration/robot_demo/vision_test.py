from __future__ import annotations

import os
import sys
import time

import pytest

from picar_kl.robot.client import PiCarClient, PiCarClientConfig


@pytest.mark.integration
@pytest.mark.robot
def test_manual_vision_stream(vision_seconds: float):
    host = os.environ.get("PICAR_V_HOST")
    if not host:
        pytest.skip("Set PICAR_V_HOST=<host:port> to run the manual vision integration test.")

    x_resize = _optional_int_env("PICAR_VISION_X_RESIZE", default=160)
    y_resize = _optional_int_env("PICAR_VISION_Y_RESIZE", default=120)
    interval_seconds = _float_env("PICAR_VISION_INTERVAL_SECONDS", default=0.2)
    show_window = os.environ.get("PICAR_SHOW_VISION_WINDOW", "0") == "1"

    frame_streamer = DesktopFrameStreamer("PiCar Vision Test") if show_window else NullFrameStreamer()
    client = PiCarClient(
        PiCarClientConfig(
            host=host,
            timeout_seconds=5.0,
            retries=3,
            retry_sleep_seconds=0.1,
            x_resize=x_resize,
            y_resize=y_resize,
        )
    )

    duration_label = "infinite" if vision_seconds < 0 else f"{vision_seconds:.1f} seconds"
    print("Manual vision test starting.", flush=True)
    print(f"host={host}", flush=True)
    print(f"resize={x_resize}x{y_resize}", flush=True)
    print(f"duration={duration_label}", flush=True)
    print("Press Ctrl-C in this terminal to end early.", flush=True)

    frame_count = 0
    interrupted = False
    started_at = time.monotonic()
    try:
        while True:
            if vision_seconds >= 0 and (time.monotonic() - started_at) >= vision_seconds:
                break
            image = client.capture_image()
            frame = image.image_rgb
            assert frame.ndim == 3
            assert frame.shape[2] == 3
            assert frame.dtype.name == "uint8"
            frame_count += 1
            frame_streamer.show(frame)
            print(
                "frame="
                f"{frame_count} shape={tuple(frame.shape)} "
                f"mean={float(frame.mean()):.2f} "
                f"ball=({image.ball_x},{image.ball_y},{image.ball_radius})",
                flush=True,
            )
            time.sleep(max(0.0, interval_seconds))
    except KeyboardInterrupt:
        interrupted = True
        print("Ctrl-C detected. Ending vision test early.", flush=True)
    finally:
        frame_streamer.close()

    assert frame_count > 0
    print(f"Vision test complete. frames={frame_count}, interrupted={interrupted}", flush=True)


class NullFrameStreamer:
    def show(self, image_rgb):
        del image_rgb

    def close(self):
        return None


class DesktopFrameStreamer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.cv2 = self._load_cv2()
        self._created = False

    def show(self, image_rgb):
        frame_bgr = self.cv2.cvtColor(image_rgb, self.cv2.COLOR_RGB2BGR)
        if not self._created:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self._created = True
        self.cv2.imshow(self.window_name, frame_bgr)
        self.cv2.waitKey(1)

    def close(self):
        if self._created:
            self.cv2.destroyWindow(self.window_name)
            self.cv2.waitKey(1)

    @staticmethod
    def _load_cv2():
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            pytest.skip("Desktop display unavailable; unset PICAR_SHOW_VISION_WINDOW or configure display.")
        try:
            import cv2
        except ImportError as exc:
            pytest.skip(f"opencv-python is required for desktop frame streaming: {exc}")
        return cv2


def _optional_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)
