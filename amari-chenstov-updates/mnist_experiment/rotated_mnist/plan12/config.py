"""Frozen configuration and deterministic identities for Plan 12."""

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
RIDGE_SCALE_RATIOS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SCHEDULES = ("linear", "sigmoid")
PI = 0.025


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Plan12Study:
    protocol: RotatedDoubleLapConfig
    seed_root: int
    smoke: bool
    anchors_per_schedule: int
    local_batches_per_anchor: int
    local_target_samples: int
    phase2_replicas: int
    phase3_checkpoints: int
    phase3_resamples: int
    phase4_replicas: int
    ridge_scale_ratios: tuple[float, ...] = RIDGE_SCALE_RATIOS

    def __post_init__(self) -> None:
        expected = {
            True: (1, 2, 16, 2, 2, 8, 2),
            False: (4, 32, 1024, 64, 8, 512, 64),
        }[self.smoke]
        actual = (
            self.anchors_per_schedule,
            self.local_batches_per_anchor,
            self.local_target_samples,
            self.phase2_replicas,
            self.phase3_checkpoints,
            self.phase3_resamples,
            self.phase4_replicas,
        )
        if actual != expected:
            raise ValueError(f"Plan 12 {'smoke' if self.smoke else 'default'} sizes must be {expected}")
        if self.protocol.data.samples_per_step != 4 or self.protocol.fisher.rank != 8:
            raise ValueError("Plan 12 requires four-observation batches and rank eight")
        expected_transitions = 2 if self.smoke else 40
        if self.protocol.rotation.transitions_per_arrow != expected_transitions:
            raise ValueError("Plan 12 protocol has the wrong schedule resolution")
        if self.ridge_scale_ratios != RIDGE_SCALE_RATIOS:
            raise ValueError("Plan 12 ridge grid changed")

    @classmethod
    def from_path(cls, path: str | Path) -> "Plan12Study":
        config_path = Path(path)
        value = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "protocol_file",
            "seed_root",
            "smoke",
            "anchors_per_schedule",
            "local_batches_per_anchor",
            "local_target_samples",
            "phase2_replicas",
            "phase3_checkpoints",
            "phase3_resamples",
            "phase4_replicas",
            "ridge_scale_ratios",
        }
        if set(value) != expected or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid Plan 12 study configuration")
        repo_root = Path(__file__).parents[3]
        protocol = RotatedDoubleLapConfig.from_mapping(
            json.loads((repo_root / value["protocol_file"]).read_text(encoding="utf-8"))
        )
        if bool(value["smoke"]):
            protocol = dataclasses.replace(
                protocol,
                runtime=dataclasses.replace(protocol.runtime, device="cpu", num_workers=0),
            )
        return cls(
            protocol=protocol,
            seed_root=int(value["seed_root"]),
            smoke=bool(value["smoke"]),
            anchors_per_schedule=int(value["anchors_per_schedule"]),
            local_batches_per_anchor=int(value["local_batches_per_anchor"]),
            local_target_samples=int(value["local_target_samples"]),
            phase2_replicas=int(value["phase2_replicas"]),
            phase3_checkpoints=int(value["phase3_checkpoints"]),
            phase3_resamples=int(value["phase3_resamples"]),
            phase4_replicas=int(value["phase4_replicas"]),
            ridge_scale_ratios=tuple(float(item) for item in value["ridge_scale_ratios"]),
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol": self.protocol.to_mapping(),
            "seed_root": self.seed_root,
            "smoke": self.smoke,
            "anchors_per_schedule": self.anchors_per_schedule,
            "local_batches_per_anchor": self.local_batches_per_anchor,
            "local_target_samples": self.local_target_samples,
            "phase2_replicas": self.phase2_replicas,
            "phase3_checkpoints": self.phase3_checkpoints,
            "phase3_resamples": self.phase3_resamples,
            "phase4_replicas": self.phase4_replicas,
            "ridge_scale_ratios": list(self.ridge_scale_ratios),
            "fixed_pi": PI,
            "schedules": list(SCHEDULES),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.mapping())

    def seed(self, component: str, index: int = 0) -> int:
        return derive_component_seed(self.seed_root, f"plan12:{component}:{index:05d}")

    def protocol_for_replica(self, phase: str, index: int) -> RotatedDoubleLapConfig:
        return dataclasses.replace(
            self.protocol,
            replica_id=f"plan12-{phase}-replica-{index:04d}",
            replica_seed=self.seed(f"{phase}:replica", index),
        )
