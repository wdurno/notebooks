"""Frozen configuration and deterministic identities for Plan 13."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.phase5_double_lap_config import (
    RotatedDoubleLapConfig,
)


SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
ROTATION_SCHEDULES = ("linear", "sigmoid")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not result > 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclasses.dataclass(frozen=True)
class Plan13Study:
    protocol: RotatedDoubleLapConfig
    seed_root: int
    smoke: bool
    burn_in_candidates: tuple[int, ...]
    rank_candidates: tuple[int, ...]
    covariance_half_lives: tuple[float, ...]
    inner_steps: int
    learning_rate: float
    max_backtracks: int
    phase1_replicas: int
    phase2_replicas: int
    phase3_probe_replicas: int
    phase3_replicas: int
    phase4_replicas: int
    phase5_replicas: int

    def __post_init__(self) -> None:
        if not self.burn_in_candidates or tuple(sorted(set(self.burn_in_candidates))) != self.burn_in_candidates:
            raise ValueError("burn-in candidates must be sorted and unique")
        if not self.rank_candidates or tuple(sorted(set(self.rank_candidates))) != self.rank_candidates:
            raise ValueError("rank candidates must be sorted and unique")
        if any(value < 1 for value in self.burn_in_candidates):
            raise ValueError("burn-in candidates must be positive")
        if any(value < 1 or value > 487 for value in self.rank_candidates):
            raise ValueError("rank candidates must lie in [1, 487]")
        if any(rank > max(self.burn_in_candidates) for rank in self.rank_candidates):
            raise ValueError("rank cannot exceed the largest burn-in")
        if not self.covariance_half_lives or any(value <= 0 for value in self.covariance_half_lives):
            raise ValueError("covariance half-lives must be positive")
        for name in (
            "inner_steps",
            "max_backtracks",
            "phase1_replicas",
            "phase2_replicas",
            "phase3_probe_replicas",
            "phase3_replicas",
            "phase4_replicas",
            "phase5_replicas",
        ):
            _positive_int(getattr(self, name), name)
        _positive_float(self.learning_rate, "learning_rate")
        transitions = self.protocol.rotation.transitions_per_arrow * 3
        if max(self.burn_in_candidates) >= transitions:
            raise ValueError("burn-in must leave at least one scored transition")
        if tuple(self.protocol.schedule_kinds) != ROTATION_SCHEDULES:
            raise ValueError("Plan 13 requires separate linear and sigmoid schedules")
        if self.protocol.data.samples_per_step != 4:
            raise ValueError("rotation study requires four observations per step")
        if self.protocol.fisher.rank != 8:
            raise ValueError("rotation study requires a rank-eight Fisher archive")

    @classmethod
    def from_path(cls, path: str | Path) -> "Plan13Study":
        config_path = Path(path)
        value = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "protocol_file",
            "seed_root",
            "smoke",
            "burn_in_candidates",
            "rank_candidates",
            "covariance_half_lives",
            "inner_steps",
            "learning_rate",
            "max_backtracks",
            "phase1_replicas",
            "phase2_replicas",
            "phase3_probe_replicas",
            "phase3_replicas",
            "phase4_replicas",
            "phase5_replicas",
        }
        if set(value) != expected or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid Plan 13 study configuration")
        repo_root = Path(__file__).parents[2]
        protocol_path = repo_root / value["protocol_file"]
        protocol = RotatedDoubleLapConfig.from_mapping(
            json.loads(protocol_path.read_text(encoding="utf-8"))
        )
        if bool(value["smoke"]):
            protocol = dataclasses.replace(
                protocol,
                runtime=dataclasses.replace(
                    protocol.runtime,
                    device="cpu",
                    num_workers=0,
                ),
            )
        return cls(
            protocol=protocol,
            seed_root=int(value["seed_root"]),
            smoke=bool(value["smoke"]),
            burn_in_candidates=tuple(int(item) for item in value["burn_in_candidates"]),
            rank_candidates=tuple(int(item) for item in value["rank_candidates"]),
            covariance_half_lives=tuple(float(item) for item in value["covariance_half_lives"]),
            inner_steps=int(value["inner_steps"]),
            learning_rate=float(value["learning_rate"]),
            max_backtracks=int(value["max_backtracks"]),
            phase1_replicas=int(value["phase1_replicas"]),
            phase2_replicas=int(value["phase2_replicas"]),
            phase3_probe_replicas=int(value["phase3_probe_replicas"]),
            phase3_replicas=int(value["phase3_replicas"]),
            phase4_replicas=int(value["phase4_replicas"]),
            phase5_replicas=int(value["phase5_replicas"]),
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "protocol": self.protocol.to_mapping(),
            "seed_root": self.seed_root,
            "smoke": self.smoke,
            "burn_in_candidates": list(self.burn_in_candidates),
            "rank_candidates": list(self.rank_candidates),
            "covariance_half_lives": list(self.covariance_half_lives),
            "inner_steps": self.inner_steps,
            "learning_rate": self.learning_rate,
            "max_backtracks": self.max_backtracks,
            "phase1_replicas": self.phase1_replicas,
            "phase2_replicas": self.phase2_replicas,
            "phase3_probe_replicas": self.phase3_probe_replicas,
            "phase3_replicas": self.phase3_replicas,
            "phase4_replicas": self.phase4_replicas,
            "phase5_replicas": self.phase5_replicas,
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.mapping())

    def seed(self, component: str, index: int = 0) -> int:
        return derive_component_seed(
            self.seed_root,
            f"plan13:{component}:{index:05d}",
        )

    def protocol_for_replica(
        self,
        phase: str,
        index: int,
    ) -> RotatedDoubleLapConfig:
        return dataclasses.replace(
            self.protocol,
            replica_id=f"plan13-{phase}-replica-{index:04d}",
            replica_seed=self.seed(f"{phase}:replica", index),
        )
