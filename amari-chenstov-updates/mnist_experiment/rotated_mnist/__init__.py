"""Detachable rotated-MNIST experiments for Plan 5."""

from .config import RotatedExperimentConfig, load_config
from .schedule import RotationSchedule, resolve_rotation_schedule

__all__ = [
    "RotatedExperimentConfig",
    "RotationSchedule",
    "load_config",
    "resolve_rotation_schedule",
]
