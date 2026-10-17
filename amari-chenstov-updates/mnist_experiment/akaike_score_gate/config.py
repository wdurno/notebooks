"""Resolved Plan 14 configurations, paired seeds, and policy identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _resolve(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError("cyclic configuration inheritance")
    value = json.loads(path.read_text())
    if "base" in value:
        base = _resolve(path.parent / value.pop("base"), (*seen, path))
        base.update(value)
        value = base
    return value


@dataclasses.dataclass(frozen=True)
class Study:
    raw: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "Study":
        return cls(_resolve(Path(path)))

    def __post_init__(self) -> None:
        expected = json.loads((Path(__file__).parent / "configs/default.json").read_text())
        if set(self.raw) != set(expected) or self.raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError("incompatible Plan 14 configuration keys/schema")
        for name, value in self.raw.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"nonfinite {name}")
            if isinstance(value, (float, int)) and not isinstance(value, bool) and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.raw["device"] not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.raw["dtype"] not in {"float32", "float64"} or self.raw["matrix_dtype"] != "float64":
            raise ValueError("Plan 14 requires float64 matrix accumulation")
        if len(self.raw["mixture_samples"]) != 2 or any(m <= 0 for m in self.raw["mixture_samples"]):
            raise ValueError("two positive principal mixture sample sizes required")
        if set(self.raw["phase_replicas"]) != {"phase1", "phase3", "phase4", "phase5"}:
            raise ValueError("incompatible replica ledger")
        if any(n < 1 for n in self.raw["phase_replicas"].values()):
            raise ValueError("replica counts must be positive")
        if sum(self.raw[n] for n in ("source_size", "source_validation_size", "online_pool_size",
                                    "reference_pool_size", "development_evaluation_size")) > 60000:
            raise ValueError("disjoint MNIST training partitions exceed 60000")
        if self.raw["reference_max_samples"] < self.raw["reference_samples"]:
            raise ValueError("reference ceiling below initial budget")

    def __getattr__(self, name: str) -> Any:
        raw = object.__getattribute__(self, "raw")
        if name not in raw:
            raise AttributeError(name)
        return raw[name]

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.raw)

    def mapping(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))

    def seed(self, component: str, index: int = 0) -> int:
        return derive_component_seed(self.seed_root, f"plan14:{component}:{index:05d}")

    def environments(self, *, scarcity: bool = False) -> list[dict[str, Any]]:
        result = [{"name": f"rotation_{s}", "family": "rotation", "schedule": s,
                   "samples": self.rotation_samples} for s in ("linear", "sigmoid")]
        for m in self.mixture_samples + ([self.scarcity_samples] if scarcity else []):
            result.append({"name": f"mixture_m{m}", "family": "mixture", "schedule": "linear",
                           "samples": m, "development_only": m == self.scarcity_samples})
        return result


def policies(c: float = 1.0, half_life: float = 8.0, cadence: int = 1,
             rank: int = 8, score_samples: int = 64) -> list[dict[str, Any]]:
    common = {"c": c, "half_life": half_life, "cadence": cadence,
              "rank": rank, "score_samples": score_samples}
    result = []
    for ewc in ("diag", "lr"):
        for depth in range(1, 7):
            result.append({**common, "name": f"depth{depth}_{ewc}", "depth": depth,
                           "ewc": ewc, "gate": None})
        for gate in ("diag", "lr"):
            result.append({**common, "name": f"asg_{ewc}_{gate}", "depth": None,
                           "ewc": ewc, "gate": gate})
    result.append({**common, "name": "asg_uncalibrated", "depth": None, "ewc": "lr",
                   "gate": "lr", "c": 1.0, **({"alias_of": "asg_lr_lr"} if c == 1 else {})})
    result.append({**common, "name": "no_update", "depth": 0, "ewc": "lr", "gate": None})
    return result


def canonical_policy(policy: dict[str, Any]) -> dict[str, Any]:
    result = dict(policy)
    alias = result.pop("alias_of", None)
    if alias is not None:
        if alias != "asg_lr_lr" or result["name"] != "asg_uncalibrated" or result["c"] != 1:
            raise ValueError("invalid policy alias")
        result["name"] = alias
    return result
