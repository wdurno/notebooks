"""Immutable settings for registration, with a small recovery configuration."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from src.seeding import derive_component_seed
from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe, content_hash
from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Task:
    key: str
    output_dim: int

    def mapping(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, value):
        result = cls(**value)
        if (result.key, result.output_dim) != ("grid", 54):
            raise ValueError("invalid registration task")
        return result


TASKS = (Task("grid", 54),)


@dataclass(frozen=True)
class Config:
    protocol: str = PROTOCOL
    schema_version: int = 1
    smoke: bool = False
    seed: int = 2026102811
    device: str = "cuda"
    dtype: str = "float32"
    threads: int = 2
    output_dims: tuple[int, ...] = (54,)
    channels: tuple[int, int] = (4, 6)
    sigma_y: float = 2.
    init_size: int = 3_000
    init_validation_size: int = 200
    reference_size: int = 128
    reference_max_size: int = 256
    init_epochs: int = 15
    max_init_epochs: int = 45
    init_epoch_stages: tuple[int, ...] = (15, 30, 45)
    init_batch: int = 32
    init_learning_rate: float = .001
    eval_per_class: int = 1
    eval_size: int = 128
    evaluation_chunk: int = 32
    fisher_chunk: int = 2
    nominal_batch: int = 4
    ramp_groups: int = 40
    novel_probability: float = .1
    novel_count: int = 100
    screen_novel_count: int = 30
    calibration_replicas: int = 2
    finalist_extra_replicas: int = 2
    confirmation_replicas: int = 12
    preferred_min_confirmation: int = 8
    reference_capacity: int = 32
    reference_replay: int = 2
    capacities: tuple[int, ...] = (32,)
    replay_counts: tuple[int, ...] = (2,)
    ridge_grid: tuple[float, ...] = (.1, 1., math.sqrt(10), 10.)
    strength_ratios: tuple[float, ...] = (.1, 1., 10.)
    retention_margin: float = .5
    matching_tolerance: float = .1
    envelope_factor: float = 2.5
    refinement_settings: int = 6
    blur_max: float = 1.8
    diagnostic_base_size: int = 64
    diagnostic_novel_size: int = 32
    diagnostic_reserve_size: int = 128
    diagnostic_epochs: int = 5
    max_diagnostic_epochs: int = 10
    diagnostic_batch: int = 32
    diagnostic_learning_rate: float = .001
    archive_rank: int = 8
    pi: float = .025
    probes: int = 2
    fd_step: float = .01
    max_iter: int = 100
    max_eval: int = 150
    audit_multiplier: int = 2
    audit_objective_tolerance: float = .001
    numerical_tolerance: float = .05
    bootstrap_samples: int = 2_000
    analysis_reserve_seconds: float = 1_800.
    confirmation_reserve_seconds: float = 18_000.
    max_artifact_bytes: int = 500_000_000_000
    free_reserve_bytes: int = 2_000_000_000

    def mapping(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def scientific_hash(self):
        return content_hash(self.mapping())

    def seed_for(self, family, replica, component):
        stage = family.split("--", 1)[0]
        return derive_component_seed(self.seed, f"{PROTOCOL}:{stage}:{replica}:{component}")

    def validate(self):
        if self.protocol != PROTOCOL or self.schema_version != 1:
            raise ValueError("incompatible registration protocol/schema")
        if self.device not in ("cpu", "cuda") or self.dtype not in ("float32", "float64"):
            raise ValueError("unsupported device/dtype")
        if type(self.smoke) is not bool or self.output_dims != (54,) or self.sigma_y != 2.:
            raise ValueError("grid registration requires 54 outputs and the fixed pixel likelihood")
        if self.channels != (4, 6) or self.nominal_batch != 4:
            raise ValueError("change architecture only through the recorded capability decision")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and (not math.isfinite(v) or v <= 0):
                raise ValueError(f"invalid {f.name}")
        for name in ("seed", "threads", "init_size", "init_validation_size", "reference_size",
                     "reference_max_size", "init_epochs", "max_init_epochs", "init_batch",
                     "eval_per_class", "eval_size", "diagnostic_reserve_size", "evaluation_chunk", "fisher_chunk", "novel_count",
                     "calibration_replicas", "confirmation_replicas", "reference_capacity",
                     "reference_replay", "max_iter", "max_eval", "bootstrap_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"positive integer required: {name}")
        if not (0 < self.pi < 1 and 0 < self.novel_probability <= 1
                and 0 < self.matching_tolerance < self.retention_margin
                and 0 <= self.refinement_settings <= 6
                and self.reference_size <= self.reference_max_size
                and self.init_epochs in self.init_epoch_stages
                and self.init_epoch_stages[-1] == self.max_init_epochs
                and 0 <= self.finalist_extra_replicas <= 2
                and self.max_artifact_bytes <= 500_000_000_000):
            raise ValueError("invalid registration budgets or calibration limits")
        if not self.smoke and (self.novel_count != 100 or self.calibration_replicas != 2
                or self.confirmation_replicas not in (8, 10, 12)
                or self.reference_capacity != 32 or self.reference_replay != 2
                or self.novel_probability != .1 or self.ramp_groups != 40):
            raise ValueError("new production scope requires a versioned protocol")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        for name in ("output_dims", "channels", "init_epoch_stages", "capacities",
                     "replay_counts", "ridge_grid", "strength_ratios"):
            if name in values:
                values[name] = tuple(values[name])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    @classmethod
    def read(cls, path):
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def smoke_config(cls, device="cpu"):
        cfg = cls(smoke=True, device=device, dtype="float64", init_size=8,
            init_validation_size=4, reference_size=2, reference_max_size=4,
            init_epochs=1, max_init_epochs=1, init_epoch_stages=(1,), init_batch=4,
            eval_per_class=1, eval_size=2, fisher_chunk=1, novel_count=3, screen_novel_count=1, ramp_groups=2,
            novel_probability=.5, finalist_extra_replicas=0, confirmation_replicas=2,
            preferred_min_confirmation=2, reference_capacity=2, reference_replay=1,
            capacities=(2,), replay_counts=(1,), ridge_grid=(1.,), strength_ratios=(.1,),
            diagnostic_base_size=2, diagnostic_novel_size=1, diagnostic_reserve_size=2, diagnostic_epochs=1,
            max_diagnostic_epochs=1, diagnostic_batch=2, max_iter=2, max_eval=4,
            bootstrap_samples=50, refinement_settings=0, confirmation_reserve_seconds=30.,
            analysis_reserve_seconds=10., free_reserve_bytes=268_435_456)
        cfg.validate()
        return cfg
