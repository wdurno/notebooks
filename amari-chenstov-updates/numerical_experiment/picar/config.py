"""Strict configuration and pairing for Plan 19, without model imports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from src.seeding import derive_seed_map


SEED_COMPONENTS = (
    "initialization", "layouts", "exploration", "replay", "fisher_labels",
    "fd_probes", "evaluation", "teacher", "sensor", "bootstrap",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "tiny"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str | None = "66285546d2b821cf421d4f5eb2576359d3770cd3"
    top_layers: int = 1
    ltm_rank: int = 4
    stm_rank: int = 1
    hidden_size: int = 32
    context_length: int = 8
    return_bins: int = 21
    quantized: bool = True

    def validate(self) -> None:
        if self.kind not in {"tiny", "qwen"}:
            raise ValueError("model.kind must be tiny or qwen")
        for name in ("top_layers", "ltm_rank", "stm_rank", "hidden_size", "context_length"):
            _integer(getattr(self, name), f"model.{name}")
        _integer(self.return_bins, "model.return_bins", 2)


@dataclass(frozen=True)
class LearningConfig:
    replay_capacity: int = 8
    replay_batch: int = 2
    learning_rate: float = 0.001
    optimizer_steps: int = 1
    pi: float = 0.025
    ridge: float = 0.1
    ridge_floor: float = 1e-8
    fd_strength: float = 0.1
    fd_step: float = 0.01
    fd_probes: int = 1
    fisher_draws: int = 1
    temperature: float = 5.0
    action_variance: float = 1.0
    exploration_std: float = 0.1
    teacher_floor: float = 0.001
    return_min: float = -8.0
    return_max: float = 11.0
    gamma: float = 0.99
    nominal_period_s: float = 0.5
    sleep_every: int = 0
    sleep_steps: int = 2
    sleep_batch: int = 4
    retired_archive_replay: bool = False
    retired_replay_protocol: str = "off"
    retired_replay_fraction: float = 0.25
    reward_version: str = "potential-v1"
    visual_reward_budget: float = 2.0
    reward_horizon_s: float = 20.0
    step_cost: float = 0.02

    def validate(self) -> None:
        for name in ("replay_capacity", "replay_batch", "optimizer_steps", "fd_probes", "fisher_draws", "sleep_steps", "sleep_batch"):
            _integer(getattr(self, name), f"learning.{name}")
        _integer(self.sleep_every, "learning.sleep_every", 0)
        for name in ("learning_rate", "ridge_floor", "fd_step", "temperature", "action_variance", "nominal_period_s"):
            _positive(getattr(self, name), f"learning.{name}")
        for name in ("ridge", "fd_strength", "exploration_std"):
            _positive(getattr(self, name), f"learning.{name}", zero=True)
        if not 0 < self.pi < 1 or not 0 < self.gamma <= 1:
            raise ValueError("pi must be in (0,1) and gamma in (0,1]")
        if not 0 < self.teacher_floor < 1 / 8:
            raise ValueError("teacher_floor must be in (0,1/8)")
        if not math.isfinite(self.return_min + self.return_max) or self.return_min >= self.return_max:
            raise ValueError("return bounds must be finite and ordered")
        if self.replay_batch > self.replay_capacity:
            raise ValueError("replay_batch cannot exceed replay_capacity")
        if not isinstance(self.retired_archive_replay, bool):
            raise ValueError("retired_archive_replay must be boolean")
        if not math.isfinite(self.retired_replay_fraction) or not 0 < self.retired_replay_fraction <= 1:
            raise ValueError("retired replay fraction must be in (0,1]")
        expected = "supplemental-rehearsal-v1" if self.retired_archive_replay else "off"
        if self.retired_replay_protocol != expected:
            raise ValueError("retired archive replay requires a separately versioned scientific protocol")
        if self.reward_version not in {"potential-v1", "dense-visual-v1", "dense-visual-centering-v1"}:
            raise ValueError("unsupported reward version")
        _positive(self.visual_reward_budget, "learning.visual_reward_budget", zero=True)
        _positive(self.reward_horizon_s, "learning.reward_horizon_s")
        _positive(self.step_cost, "learning.step_cost", zero=True)
        if self.reward_version in {"dense-visual-v1", "dense-visual-centering-v1"}:
            if (self.visual_reward_budget, self.reward_horizon_s, self.step_cost, self.nominal_period_s) != (2., 20., .02, .5):
                raise ValueError("dense-visual-v1 fixes its reward scales; changes require a new version")
            lower = -5 - self.step_cost * self.reward_horizon_s / self.nominal_period_s
            upper = 10 + self.visual_reward_budget
            if self.return_min > lower or self.return_max < upper:
                raise ValueError("dense visual reward return range must enclose conservative bounds")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "float32"
    threads: int = 1
    wall_clock_limit_s: float = 72000.0
    local_quota_bytes: int = 100_000_000_000
    archive_quota_bytes: int = 1_000_000_000_000
    reserve_bytes: int = 20_000_000_000
    checkpoint_every: int = 1
    heartbeat_s: float = 900.0

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("runtime.device must be cpu or cuda")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("runtime.dtype must be float32 or float64")
        for name in ("threads", "local_quota_bytes", "archive_quota_bytes", "checkpoint_every"):
            _integer(getattr(self, name), f"runtime.{name}")
        _integer(self.reserve_bytes, "runtime.reserve_bytes", 0)
        _positive(self.wall_clock_limit_s, "runtime.wall_clock_limit_s")
        _positive(self.heartbeat_s, "runtime.heartbeat_s")


@dataclass(frozen=True)
class PicarConfig:
    schema_version: int = 1
    experiment: str = "plan19"
    version: str = "phase0-v1"
    mode: str = "smoke"
    seed: int = 190001
    replicas: int = 1
    transitions: int = 12
    bootstrap_transitions: int = 8
    episode_steps: int = 4
    simulation_period_s: float = 0.1
    task_version: str = "safe-stop-v1"
    image_width: int = 64
    image_height: int = 48
    conditions: tuple[str, ...] = ("replay_ewc", "stm_fd")
    model: ModelConfig = field(default_factory=ModelConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.schema_version != 1 or self.experiment != "plan19":
            raise ValueError("unsupported Plan 19 configuration schema")
        if not self.version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in self.version):
            raise ValueError("version must be a nonempty portable identifier")
        if self.mode not in {"smoke", "production"}:
            raise ValueError("mode must be smoke or production")
        for name in ("replicas", "transitions", "bootstrap_transitions", "episode_steps", "image_width", "image_height"):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", 0)
        _positive(self.simulation_period_s, "simulation_period_s")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be nonempty and unique")
        if not set(self.conditions).issubset({"replay_ewc", "stm_fd"}):
            raise ValueError("unrecognized condition")
        if self.mode == "production" and self.model.kind != "qwen":
            raise ValueError("production cannot silently use the tiny test model")
        if self.task_version not in {"safe-stop-v1", "ball-contact-v1"}:
            raise ValueError("unsupported task version")
        if self.mode == "production" and self.learning.retired_archive_replay:
            raise ValueError("primary production comparison keeps retired archive replay disabled")
        self.model.validate()
        self.learning.validate()
        self.runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def seeds(self, replica: int) -> dict[str, int]:
        _integer(replica, "replica", 0)
        return derive_seed_map(self.seed + replica, SEED_COMPONENTS)

    def for_condition(self, name: str) -> PicarConfig:
        if name not in self.conditions:
            raise ValueError("condition is not present in the declared comparison")
        strength = 0.0 if name == "replay_ewc" else self.learning.fd_strength
        return replace(self, learning=replace(self.learning, fd_strength=strength))


def _construct(cls: type, value: dict[str, Any]) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def config_from_dict(value: dict[str, Any]) -> PicarConfig:
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    data = dict(value)
    for key, cls in (("model", ModelConfig), ("learning", LearningConfig), ("runtime", RuntimeConfig)):
        data[key] = _construct(cls, data.get(key, {}))
    if "conditions" in data:
        if not isinstance(data["conditions"], (list, tuple)):
            raise ValueError("conditions must be an array")
        data["conditions"] = tuple(data["conditions"])
    config = _construct(PicarConfig, data)
    config.validate()
    return config


def load_config(path: str | Path) -> PicarConfig:
    return config_from_dict(json.loads(Path(path).read_text()))
