from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mnist_experiment.rotated_mnist.plan12.q01_probe import (
    MP_Q01_CONDITION,
    MP_Q01_RATIO,
    ExternalReference,
    normalized_higher_quantile,
    trajectory_detail,
)


def test_q01_quantile_uses_conservative_higher_order_statistic() -> None:
    maxima = torch.arange(1, 101, dtype=torch.float64)
    ratio = normalized_higher_quantile(maxima, 2.0, quantile=0.01)
    assert ratio == 1.0


@pytest.mark.parametrize("mean", [0.0, -1.0, float("nan")])
def test_q01_quantile_rejects_invalid_normalization(mean: float) -> None:
    with pytest.raises(ValueError, match="mean eigenvalue"):
        normalized_higher_quantile(torch.ones(8), mean, quantile=0.01)


def test_q01_trajectory_detail_freezes_phase4_pairing(tmp_path: Path) -> None:
    run = tmp_path / "phase4-assets"
    run.mkdir()
    (run / "integrity.json").write_text('{"assets.pt":"abc"}\n', encoding="utf-8")
    reference = ExternalReference(
        action="assets",
        index=1,
        schedule=None,
        condition=None,
        unit={"phase": "phase4", "index": 1},
        path=run,
        required=("assets.pt",),
    )
    detail = trajectory_detail(reference, tmp_path)
    assert MP_Q01_CONDITION.ridge_ratio == MP_Q01_RATIO
    assert detail["mp_quantile"] == 0.01
    assert detail["quantile_method"] == "higher"
    assert detail["protocol_phase"] == "phase4"
    assert detail["numerical_seed_phase"] == "phase4"
    assert detail["numerical_seed_condition"] == "spectral_selector"
    assert detail["external_asset"]["artifact_hashes"] == {"assets.pt": "abc"}
