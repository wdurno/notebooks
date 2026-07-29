"""Stable, independently named seeds for paired experimental components."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

SEED_SCHEMA_VERSION = 1

DEFAULT_SEED_COMPONENTS = (
    "initialization",
    "data_partition",
    "initialization_data",
    "initialization_loader",
    "online_stream",
    "reference_stream",
    "evaluation_data",
    "stencil_probes",
    "lanczos",
)


def derive_component_seed(replica_seed: int, component: str) -> int:
    if (
        not isinstance(replica_seed, int)
        or isinstance(replica_seed, bool)
        or not 0 <= replica_seed < 2**63
    ):
        raise ValueError("replica_seed must be an integer in [0, 2**63)")
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a nonempty string")

    payload = f"{SEED_SCHEMA_VERSION}:{replica_seed}:{component}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def derive_seed_map(
    replica_seed: int,
    components: Iterable[str] = DEFAULT_SEED_COMPONENTS,
) -> dict[str, int]:
    component_names = tuple(components)
    if len(component_names) != len(set(component_names)):
        raise ValueError("seed component names must be unique")
    return {
        component: derive_component_seed(replica_seed, component)
        for component in component_names
    }
