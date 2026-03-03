from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .config import PiCarControlConfig


class PiCarControlClient:
    """Thin adapter around the local `picar-v-rl-env` client module."""

    def __init__(self, config: PiCarControlConfig):
        self.config = config
        self._car_client = _load_car_client_module(config.repo_root)

    def reset(self):
        if hasattr(self._car_client, "look_forward"):
            self._car_client.look_forward(self.config.host)
        return self.capture_image()

    def capture_image(self):
        image_rgb, _x, _y, _r = self._car_client.img(
            self.config.host,
            x_resize=self.config.x_resize,
            y_resize=self.config.y_resize,
        )
        return image_rgb

    def apply_vector(self, vector: dict[str, float]) -> Any:
        return self._car_client.apply_vector(
            self.config.host,
            pan=float(vector.get("pan", 0.0)),
            tilt=float(vector.get("tilt", 0.0)),
            turn=float(vector.get("turn", 0.0)),
            drive=float(vector.get("drive", 0.0)),
        )

    def look_forward(self) -> None:
        self._car_client.look_forward(self.config.host)
        return None


def _load_car_client_module(repo_root: Path):
    src_root = repo_root / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"Could not find picar-v-rl-env src directory at {src_root}")
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    return importlib.import_module("car_env.car_client")
