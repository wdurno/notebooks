"""Shared robot contract; importing it never loads Bullet, Torch, or hardware."""

from .actions import ACTION_NAMES, ACTION_VECTORS, Calibration, action_distribution_to_vector
from .protocol import (
    SCHEMA_VERSION, CommandReceipt, Control, RewardComponents, RobotBackend,
    RobotCommand, RobotObservation, StepResult,
)

__all__ = [
    "ACTION_NAMES", "ACTION_VECTORS", "Calibration", "action_distribution_to_vector",
    "SCHEMA_VERSION", "CommandReceipt", "Control", "RewardComponents", "RobotBackend",
    "RobotCommand", "RobotObservation", "StepResult",
]
