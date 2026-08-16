from pathlib import Path
from types import SimpleNamespace

import pytest

from mnist_experiment.run_controller import _validate_external_oracle_provenance
from mnist_experiment.run_coupled import _tensor_hash
from src.mnist_model import build_canonical_model


def _fixture(*, derived: bool):
    model, layout = build_canonical_model(17)
    initial_hash = _tensor_hash(layout.flatten_module(model, detach=True))
    source_bundle = "parent-bundle" if derived else "direct-bundle"
    metadata = {"bundle_id": "direct-bundle"}
    if derived:
        metadata["stream_derivation"] = {"parent_bundle_id": source_bundle}
    loaded = SimpleNamespace(model=model, layout=layout, metadata=metadata)
    oracle = SimpleNamespace(content_hash="oracle-hash")
    metrics = {
        "replica_bundle_id": source_bundle,
        "parameter_count": layout.total_numel,
        "oracle_path_hash": oracle.content_hash,
        "pairing": {"initial_parameter_hash": initial_hash},
    }
    return loaded, oracle, metrics


@pytest.mark.parametrize("derived", [False, True])
def test_external_oracle_validates_direct_or_parent_bundle(derived: bool) -> None:
    loaded, oracle, metrics = _fixture(derived=derived)

    result = _validate_external_oracle_provenance(
        Path("run/phase8_reference_optimum.pt"),
        metrics,
        loaded,
        oracle,
    )

    assert result["source_replica_bundle_id"] == metrics["replica_bundle_id"]
    assert result["parent_derivation_validated"] is derived


def test_external_oracle_rejects_different_initial_model() -> None:
    loaded, oracle, metrics = _fixture(derived=True)
    metrics["pairing"]["initial_parameter_hash"] = "wrong"

    with pytest.raises(RuntimeError, match="different initial model state"):
        _validate_external_oracle_provenance(
            Path("run/phase8_reference_optimum.pt"),
            metrics,
            loaded,
            oracle,
        )
