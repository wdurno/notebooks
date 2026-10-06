"""Rotation environment adapter for Plan 13."""

from __future__ import annotations

from pathlib import Path

from mnist_experiment.rotated_mnist.plan12.assets import (
    ASSET_REQUIRED,
    ensure_replica_assets,
    runtime,
)

from .artifacts import UnitStore


def ensure_rotation_assets(
    store: UnitStore,
    phase: str,
    index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    """Create Plan 13-owned assets using the validated Plan 12 builder."""
    return ensure_replica_assets(
        store,
        phase,
        index,
        data_root=data_root,
        resume=resume,
    )


__all__ = ["ASSET_REQUIRED", "ensure_rotation_assets", "runtime"]
