"""Resolved immutable settings for the five-condition Plan 18 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from src.seeding import derive_component_seed
from . import PROTOCOL, SCHEMA_VERSION


ARMS = ("A", "B", "C", "D", "E")
LABELS = dict(zip(ARMS, ("Replay/EWC", "STM incumbent", "STM + transport",
                        "STM + adaptation", "Combined")))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA_VERSION
    seed: int = 181106
    device: str = "cuda"
    dtype: str = "float32"
    smoke: bool = False
    threads: int = 2
    samples_per_step: int = 4
    hold_steps: int = 160
    ramp_steps: int = 304
    novel_probability: float = .125
    angle_max: float = 45.
    angle_std: float = 6.
    angle_limit: float = 70.
    init_size: int = 2300
    validation_size: int = 200
    reference_size: int = 256
    diagnostic_size: int = 128
    eval_size: int = 128
    init_epochs: int = 45
    max_init_epochs: int = 60
    init_batch: int = 32
    init_lr: float = .001
    capacity: int = 32
    replay_count: int = 2
    replay_weight: float = .5
    nominal_arrival: int = 4
    fixed_pi: float = .025
    fd_step: float = .01
    probes: int = 2
    fd_dtype: str = "float32"
    fisher_chunk: int = 2
    output_chunk: int = 6
    evaluation_chunk: int = 32
    max_iter: int = 100
    max_eval: int = 150
    history_size: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    cold_start: int = 16
    pi_min: float = .0001
    pi_max: float = .2
    noise_half_life: float = 32.
    observer_floor_absolute: float = 1e-12
    observer_floor_relative: float = 1e-6
    initialization_size: int = 2300
    analysis_novel_count: int = 100
    eval_every: int = 16
    bootstrap_draws: int = 2000
    main_replicas: int = 6
    stationary_replicas: int = 2
    ridge_candidates: tuple = (.1, 1., 10.)
    strength_ratios: tuple = (.1, 1., 10.)
    process_scales: tuple = (1e-5, 1e-4, 1e-3)
    retention_ceiling: float = .50
    retention_tolerance: float = .10
    max_bytes: int = 500_000_000_000
    min_free_bytes: int = 20_000_000_000

    @property
    def num_p_steps(self):
        return 2 * (self.hold_steps + self.ramp_steps)

    @property
    def stream_size(self):
        return self.num_p_steps * self.samples_per_step

    @property
    def scientific_hash(self):
        return content_hash(self.mapping())

    def mapping(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def seed_for(self, cohort, replica, component):
        if not isinstance(replica, int) or replica < 0:
            raise ValueError("replica index must be a nonnegative integer")
        return derive_component_seed(self.seed, f"{PROTOCOL}:{cohort}:{replica}:{component}")

    def validate(self):
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA_VERSION:
            raise ValueError("incompatible combined-memory protocol/schema")
        if self.device not in ("cpu", "cuda") or self.dtype not in ("float32", "float64"):
            raise ValueError("unsupported device or training precision")
        if self.fd_dtype not in ("float32", "float64"):
            raise ValueError("unsupported FD arithmetic precision")
        for name in ("seed", "threads", "samples_per_step", "hold_steps", "ramp_steps",
                     "init_size", "validation_size", "reference_size", "diagnostic_size",
                     "eval_size", "init_epochs", "max_init_epochs", "init_batch", "capacity",
                     "replay_count", "nominal_arrival", "probes", "fisher_chunk", "output_chunk",
                     "evaluation_chunk", "max_iter", "max_eval", "history_size", "cold_start",
                     "initialization_size", "analysis_novel_count", "eval_every", "bootstrap_draws",
                     "main_replicas", "stationary_replicas", "max_bytes", "min_free_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid positive setting: {item.name}")
        if not (0 < self.pi_min <= self.fixed_pi <= self.pi_max < 1
                and 0 < self.novel_probability < 1 and 0 < self.replay_weight < 1
                and self.replay_count <= self.capacity and self.max_eval >= self.max_iter
                and self.init_epochs <= self.max_init_epochs
                and self.initialization_size == self.init_size):
            raise ValueError("inconsistent dimensions, action, or solve settings")
        if 9 * (self.init_size + self.validation_size + self.reference_size
                + self.diagnostic_size + self.stream_size) > 60000:
            raise ValueError("disjoint training source allocation exceeds MNIST")
        if 3 * self.eval_size * 9 > 5000:
            raise ValueError("evaluation panels exceed their disjoint test half")
        if not self.smoke and (self.samples_per_step != 4 or self.nominal_arrival != 4
                or self.hold_steps != 160 or self.ramp_steps != 304
                or self.novel_probability != .125 or self.angle_max != 45.
                or self.angle_std != 6. or self.angle_limit != 70.
                or self.capacity != 32 or self.replay_count != 2
                or self.probes != 2 or self.fd_step != .01
                or self.analysis_novel_count != 100 or self.init_size != 2300
                or self.main_replicas not in (4, 6) or self.stationary_replicas != 2):
            raise ValueError("production scientific design differs from Plan 18")
        if self.max_bytes > 500_000_000_000 or self.min_free_bytes < 0:
            raise ValueError("invalid storage allocation")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        for key in ("ridge_candidates", "strength_ratios", "process_scales"):
            if key in values:
                values[key] = tuple(values[key])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    @classmethod
    def read(cls, path):
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def smoke_config(cls, device="cpu"):
        cfg = cls(smoke=True, device=device, dtype="float64", fd_dtype="float64",
            samples_per_step=2, hold_steps=2, ramp_steps=3, novel_probability=.7,
            init_size=8, initialization_size=8, validation_size=4, reference_size=4,
            diagnostic_size=4, eval_size=4, init_epochs=1, max_init_epochs=1,
            init_batch=4, capacity=4, replay_count=2, nominal_arrival=2,
            max_iter=2, max_eval=4, cold_start=2, analysis_novel_count=2,
            eval_every=2, bootstrap_draws=50, main_replicas=2,
            min_free_bytes=268_435_456)
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class Recipe:
    arm: str
    ridge: float = 1.
    strength: float = 0.
    process_scale: float = 1e-4

    def __post_init__(self):
        if self.arm not in ARMS or any(not math.isfinite(v) or v < 0
                                      for v in (self.ridge, self.strength, self.process_scale)):
            raise ValueError("invalid combined-memory recipe")
        if self.arm == "A" and self.strength != 0:
            raise ValueError("Replay/EWC has no FD penalty")
        if self.process_scale <= 0:
            raise ValueError("positive process scale required")

    @property
    def transport(self):
        return self.arm in ("C", "E")

    @property
    def adaptive(self):
        return self.arm in ("D", "E")

    @property
    def key(self):
        return self.arm + "-" + content_hash(asdict(self))[:14]

    def mapping(self):
        return asdict(self)
