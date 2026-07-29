import pytest

from src.seeding import (
    DEFAULT_SEED_COMPONENTS,
    derive_component_seed,
    derive_seed_map,
)


def test_component_seeds_are_stable_and_distinct() -> None:
    first = derive_seed_map(1729)
    second = derive_seed_map(1729)

    assert first == second
    assert tuple(first) == DEFAULT_SEED_COMPONENTS
    assert len(set(first.values())) == len(first)


def test_seed_changes_with_replica_and_component() -> None:
    assert derive_component_seed(1, "online_stream") != derive_component_seed(
        2, "online_stream"
    )
    assert derive_component_seed(1, "online_stream") != derive_component_seed(
        1, "reference_stream"
    )


def test_duplicate_seed_component_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        derive_seed_map(1, ("stream", "stream"))
