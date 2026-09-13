"""Frozen Plan 11 study and per-trajectory configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed

from ..phase5_double_lap_config import RotatedDoubleLapConfig


SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
ANCHORS = (0.01, 0.025, 0.05)
GAINS = (0.0, 0.025, 0.05, 0.10, 0.20, 1.0)
SOURCE_FILES = (
    "mnist_experiment/rotated_mnist/plan11/config.py",
    "mnist_experiment/rotated_mnist/plan11/policy.py",
    "mnist_experiment/rotated_mnist/plan11/artifacts.py",
    "mnist_experiment/rotated_mnist/plan11/run.py",
    "mnist_experiment/rotated_mnist/plan11/orchestrate.py",
    "mnist_experiment/rotated_mnist/plan11/analysis.py",
    "mnist_experiment/rotated_mnist/run.py",
    "mnist_experiment/rotated_mnist/run_phase3.py",
    "mnist_experiment/rotated_mnist/run_phase5_double_lap.py",
    "mnist_experiment/rotated_mnist/run_phase8.py",
    "mnist_experiment/rotated_mnist/data.py",
    "mnist_experiment/rotated_mnist/phase4_metrics.py",
    "mnist_experiment/rotated_mnist/transform.py",
    "src/controller.py",
    "src/ewc.py",
    "src/hybrid.py",
    "src/initialization.py",
    "src/lanczos_wrapper.py",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Policy:
    anchor: float
    gain: float

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS or self.gain not in GAINS:
            raise ValueError("policy lies outside the frozen Plan 11 grid")

    @property
    def name(self) -> str:
        return f"c{self.anchor:.3f}_g{self.gain:.3f}"

    def mapping(self) -> dict[str, float]:
        return {"anchor": self.anchor, "gain": self.gain}


@dataclasses.dataclass(frozen=True)
class Study:
    protocol: RotatedDoubleLapConfig
    seed_root: int = 20260912
    development_replicas: int = 12
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.development_replicas != 12:
            raise ValueError("Plan 11 development requires 12 replicas")
        if self.protocol.rotation.transitions_per_arrow != (2 if self.smoke else 40):
            raise ValueError("Plan 11 rotation length is incompatible")
        if self.protocol.data.samples_per_step != 4:
            raise ValueError("Plan 11 requires m=4")
        if self.protocol.fisher.rank != 8:
            raise ValueError("Plan 11 requires rank 8")
        if self.protocol.controller.trend_half_life_degrees != 1.875:
            raise ValueError("Plan 11 trend half-life changed")

    @classmethod
    def from_path(cls, path: str | Path) -> "Study":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(value) != {"schema_version", "protocol_file", "seed_root", "development_replicas", "smoke"}:
            raise ValueError("Plan 11 study configuration has unexpected keys")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("incompatible Plan 11 configuration schema")
        protocol = RotatedDoubleLapConfig.from_mapping(
                json.loads((Path(__file__).parents[3] / value["protocol_file"]).read_text(encoding="utf-8"))
            )
        if value["smoke"]:
            protocol = dataclasses.replace(
                protocol,
                runtime=dataclasses.replace(protocol.runtime, device="cpu", num_workers=0),
            )
        return cls(
            protocol=protocol,
            seed_root=int(value["seed_root"]),
            development_replicas=int(value["development_replicas"]),
            smoke=bool(value["smoke"]),
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol": self.protocol.to_mapping(),
            "seed_root": self.seed_root,
            "development_replicas": self.development_replicas,
            "smoke": self.smoke,
            "anchors": list(ANCHORS),
            "gains": list(GAINS),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.mapping())

    def replica_seed(self, phase: str, index: int) -> int:
        return derive_component_seed(self.seed_root, f"plan11:{phase}:replica:{index:04d}")

    def protocol_for_replica(self, phase: str, index: int) -> RotatedDoubleLapConfig:
        return dataclasses.replace(
            self.protocol,
            replica_id=f"{phase}-replica-{index:04d}",
            replica_seed=self.replica_seed(phase, index),
        )
