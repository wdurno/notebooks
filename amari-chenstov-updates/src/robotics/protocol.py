"""Backend-neutral robot wire contract. Times use the backend's advertised clock."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Control:
    """Absolute camera yaw/pitch and right-positive steering (radians), speed (m/s).

    Pan is left-positive; tilt is up-positive. Zero drive never implies motion,
    even with nonzero steering. Servo electrical neutral belongs to calibration.
    """

    pan_rad: float = 0.0
    tilt_rad: float = math.radians(-8)
    steering_rad: float = 0.0
    drive_mps: float = 0.0

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("controls must be finite numbers")


@dataclass(frozen=True)
class RobotCommand:
    run_id: str
    session_id: str
    sequence_id: int
    observation_id: str
    controls: Control
    issued_at_s: float
    duration_s: float
    expires_at_s: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported robot command schema")
        if not all(isinstance(v, str) and 0 < len(v) <= 160
                   for v in (self.run_id, self.session_id, self.observation_id)):
            raise ValueError("command identifiers must be nonempty bounded strings")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("sequence_id must be a nonnegative integer")
        if not isinstance(self.controls, Control):
            raise ValueError("controls must be Control")
        times = (self.issued_at_s, self.duration_s, self.expires_at_s)
        if any(isinstance(v, bool) or not math.isfinite(v) for v in times):
            raise ValueError("command times must be finite")
        if self.issued_at_s < 0 or self.duration_s <= 0 or self.expires_at_s <= self.issued_at_s:
            raise ValueError("invalid command timing")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RobotCommand:
        value = dict(value)
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing robot command schema")
        value["controls"] = Control(**value["controls"])
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandReceipt:
    run_id: str
    session_id: str
    sequence_id: int
    observation_id: str
    received_at_s: float
    executed_controls: Control
    accepted: bool
    safety_status: str
    execution_deadline_s: float
    capture_to_command_s: float | None = None
    duplicate: bool = False
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CommandReceipt:
        value = dict(value)
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported receipt schema")
        value["executed_controls"] = Control(**value["executed_controls"])
        return cls(**value)


@dataclass(frozen=True)
class RobotObservation:
    """Policy-visible data only. RGB is packed row-major uint8, not a world map."""

    run_id: str
    session_id: str
    observation_id: str
    capture_time_s: float
    wall_time_s: float
    physical_time_s: float
    simulator_time_s: float | None
    remaining_time_s: float | None
    width: int
    height: int
    rgb: bytes
    executed_controls: Control
    safety_status: str
    clock_domain: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or len(self.rgb) != self.width * self.height * 3:
            raise ValueError("RGB must contain width * height * 3 packed bytes")


@dataclass(frozen=True)
class RewardComponents:
    """Inputs for the numerical reward kernel, not a second return definition.

    Endpoint potentials are deliberately absent: they require frozen RGB-only
    scoring. step_cost_units is elapsed nominal acquisition intervals.
    """

    elapsed_s: float
    step_cost_units: float
    terminal_bonus: float
    terminal_potential_zero: bool


@dataclass(frozen=True)
class StepResult:
    observation: RobotObservation
    receipt: CommandReceipt | None
    reward_components: RewardComponents
    terminated: bool
    truncated: bool
    end_reason: str | None
    timing: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class RobotBackend(Protocol):
    def observe(self) -> RobotObservation: ...
    def command(self, command: RobotCommand) -> CommandReceipt: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
