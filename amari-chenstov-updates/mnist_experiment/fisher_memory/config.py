"""Versioned scientific settings and deterministic Plan 15 work identities."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
ENVIRONMENTS = ("rotation_linear", "rotation_sigmoid", "digit9_mixture")
METHODS = ("jvp", "finite_difference", "directional")
CONDITIONS = ("replay_ewc", *METHODS, "distillation", "replay_compute_matched")
MAX_BYTES = 500_000_000_000


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Config:
    schema_version: int = 1
    protocol: str = "plan15-v1"
    seed: int = 20261017
    smoke: bool = False
    device: str = "cuda"
    dtype: str = "float32"
    threads: int = 2
    environments: tuple[str, ...] = ENVIRONMENTS
    rank: int = 2
    replay_capacity: int = 32
    replay_weight: float = .5
    ridge_multiplier: float = .1
    rotation_steps: int = 120
    rotation_batch: int = 4
    mixture_steps: int = 100
    mixture_batch: int = 2
    mixture_max: float = .1
    rotation_pi: float = .025
    mixture_pi: float = .05
    init_size: int = 30_000
    init_epochs: int = 15
    init_batch: int = 256
    init_learning_rate: float = .01
    reference_size: int = 1024
    eval_per_class: int = 100
    evaluation_every: int = 4
    fisher_chunk: int = 16
    optimizer: str = "lbfgs"
    inner_steps: int = 50
    max_evaluations: int = 75
    learning_rate: float = 1.0
    probes: int = 2
    diagnostic_probes: int = 16
    diagnostic_inputs: int = 8
    fd_step: float = .01
    direction_radius: float = .05
    phase1_replicas: int = 1
    pilot_screen_replicas: int = 2
    pilot_validate_replicas: int = 2
    confirmation_replicas: int = 12
    strength_ratios: tuple[float, ...] = (.03, .1, .3)
    retention_margin: float = .01
    max_artifact_bytes: int = MAX_BYTES
    free_reserve_bytes: int = 2_000_000_000
    compute_allocation_seconds: float = 0.0
    bootstrap_samples: int = 2000

    def validate(self) -> None:
        if self.schema_version != 1 or self.protocol != "plan15-v1":
            raise ValueError("unsupported Plan 15 contract")
        if self.device not in ("cpu", "cuda") or self.dtype not in ("float32", "float64"):
            raise ValueError("unsupported runtime")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):
            raise ValueError("invalid environment")
        if len(set(self.environments)) != len(self.environments):
            raise ValueError("duplicate environment")
        for name in ("threads", "rank", "replay_capacity", "rotation_steps", "rotation_batch",
                     "mixture_steps", "mixture_batch", "init_size", "init_epochs", "init_batch",
                     "reference_size", "eval_per_class", "evaluation_every", "fisher_chunk",
                     "inner_steps", "max_evaluations", "probes", "diagnostic_probes", "diagnostic_inputs",
                     "phase1_replicas", "pilot_screen_replicas", "pilot_validate_replicas",
                     "confirmation_replicas", "bootstrap_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rank != 2:
            raise ValueError("rank is frozen at two; use a new protocol for a rank experiment")
        if self.rotation_steps % 3:
            raise ValueError("rotation steps must comprise three equal arrows")
        if not 0 < self.mixture_max <= .1 or not 0 < self.replay_weight < 1:
            raise ValueError("invalid prevalence or replay weight")
        if self.optimizer not in ("lbfgs", "adam"):
            raise ValueError("unsupported optimizer")
        for name in ("rotation_pi", "mixture_pi"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError("pi must lie strictly between zero and one")
        for name in ("ridge_multiplier", "init_learning_rate", "learning_rate", "fd_step",
                     "direction_radius", "retention_margin", "compute_allocation_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        for name in ("init_learning_rate", "learning_rate", "fd_step", "direction_radius"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.max_artifact_bytes <= MAX_BYTES or self.free_reserve_bytes < 0:
            raise ValueError("artifact cap cannot exceed 500 decimal GB")
        if not self.strength_ratios or any(not math.isfinite(r) or r <= 0 for r in self.strength_ratios):
            raise ValueError("invalid strength ratios")

    @classmethod
    def read(cls, path: Path | str) -> "Config":
        values = json.loads(Path(path).read_text())
        for key in ("environments", "strength_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def scientific_hash(self) -> str:
        value = self.mapping()
        for key in ("max_artifact_bytes", "free_reserve_bytes", "compute_allocation_seconds"):
            value.pop(key)
        return content_hash(value)

    def seed_for(self, family: str, replica: int, component: str) -> int:
        return derive_component_seed(self.seed, f"plan15:{family}:{replica}:{component}")

    def schedule(self, environment: str) -> list[float]:
        if environment == "digit9_mixture":
            return [self.mixture_max * j / self.mixture_steps for j in range(1, self.mixture_steps + 1)]
        if environment not in ENVIRONMENTS:
            raise ValueError("unknown environment")
        count = self.rotation_steps // 3
        knots = (0., 30., 0., 30.)
        values = []
        sigmoid = lambda x: 1 / (1 + math.exp(-8 * (x - .5)))
        low, high = sigmoid(0), sigmoid(1)
        for a, b in zip(knots[:-1], knots[1:]):
            for j in range(1, count + 1):
                u = j / count
                if environment == "rotation_sigmoid":
                    u = (sigmoid(u) - low) / (high - low)
                values.append(a + (b - a) * u)
        return values

    def batch_size(self, environment: str) -> int:
        return self.mixture_batch if environment == "digit9_mixture" else self.rotation_batch

    def expected_nines(self, environment: str) -> float | None:
        return self.mixture_batch * sum(self.schedule(environment)) if environment == "digit9_mixture" else None


@dataclasses.dataclass(frozen=True)
class Treatment:
    condition: str
    strength: float = 0.
    probes: int = 2
    fd_step: float = .01
    inner_steps: int = 50
    max_evaluations: int = 75
    replay_weight: float = .5
    ridge_multiplier: float = .1
    label: str = ""

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS or not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("invalid treatment")

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def key(self) -> str:
        return f"{self.condition}-{content_hash(self.mapping())[:12]}"

    @classmethod
    def default(cls, cfg: Config, condition: str, strength: float = 0., **overrides: Any) -> "Treatment":
        values = dict(condition=condition, strength=strength, probes=cfg.probes,
                      fd_step=cfg.fd_step, inner_steps=cfg.inner_steps,
                      max_evaluations=cfg.max_evaluations, replay_weight=cfg.replay_weight,
                      ridge_multiplier=cfg.ridge_multiplier)
        values.update(overrides)
        return cls(**values)


def scientific_sources() -> dict[str, str]:
    paths = [p for p in PACKAGE.rglob("*.py") if p.name not in (
        "report.py", "drip_report.py", "regularization_report.py", "control.py")]
    paths.extend(ROOT / name for name in (
        "src/parameters.py", "src/seeding.py", "src/mnist_data.py", "src/mnist_model.py",
        "src/lanczos_wrapper.py", "src/representations.py",
        "mnist_experiment/rotated_mnist/transform.py",
        "mnist_experiment/rotated_mnist/config.py",
        "mnist_experiment/rotated_mnist/plan12/gauge.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.exists()}


def provenance(cfg: Config) -> dict[str, Any]:
    import torch
    sources = scientific_sources()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return dict(schema_version=1, config=cfg.mapping(), config_hash=cfg.scientific_hash,
                source_hash=content_hash(sources), sources=sources, git_commit=git("rev-parse", "HEAD"),
                git_dirty=bool(git("status", "--porcelain")), torch=torch.__version__,
                python=platform.python_version(), cuda=torch.version.cuda, device=cfg.device,
                dependencies={name: importlib.metadata.version(name) for name in ("numpy", "torchvision", "nbformat", "matplotlib")},
                device_name=torch.cuda.get_device_name(0) if cfg.device == "cuda" and torch.cuda.is_available() else "CPU",
                dtype=cfg.dtype)
