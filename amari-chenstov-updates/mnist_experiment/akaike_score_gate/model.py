"""Plan 14's smooth six-group CNN in the existing Helmert source chart."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mnist_experiment.rotated_mnist.plan12.gauge import SumToZeroLinear
from src.parameters import ParameterLayout


PARAMETER_GROUPS = ("conv1", "conv2", "hidden1", "hidden2", "hidden3", "output")
DEPTH_NAMES = ("output_only", "output_hidden3", "output_hidden2", "dense_head",
               "conv2_dense_head", "full_network")


class AkaikeScoreGateCNN(nn.Module):
    """Six ordered affine groups; all three hidden dense maps have SiLU."""

    def __init__(self, width: int = 5) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width not in (5, 6):
            raise ValueError("Plan 14 width must be 5 or 6")
        self.width = width
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 6, 3, padding=1)
        self.hidden1 = nn.Linear(24, width)
        self.hidden2 = nn.Linear(width, width)
        self.hidden3 = nn.Linear(width, width)
        self.output = SumToZeroLinear(width)
        expected = {5: 501, 6: 559}[width]
        if sum(p.numel() for p in self.parameters()) != expected:
            raise RuntimeError("unexpected Plan 14 coordinate count")

    def forward(self, inputs: Tensor) -> Tensor:
        features = nn.functional.avg_pool2d(nn.functional.silu(self.conv1(inputs)), 2)
        features = nn.functional.adaptive_avg_pool2d(
            nn.functional.silu(self.conv2(features)), (2, 2)
        ).flatten(1)
        features = nn.functional.silu(self.hidden1(features))
        features = nn.functional.silu(self.hidden2(features))
        features = nn.functional.silu(self.hidden3(features))
        return self.output(features)


def build_model(
    seed: int, width: int = 5, *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[AkaikeScoreGateCNN, ParameterLayout]:
    """Build a paired CPU-seeded initialization without changing global RNG state."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("model dtype must be float32 or float64")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = AkaikeScoreGateCNN(width)
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def layer_indices(layout: ParameterLayout) -> dict[str, Tensor]:
    """Return the six groups' GLOBAL coordinates as CPU int64 tensors."""
    names = tuple(spec.name for spec in layout.specs)
    expected = tuple(f"{group}.{part}" for group in PARAMETER_GROUPS
                     for part in ("weight", "bias"))
    if names != expected:
        raise ValueError("layout is not the six-group Plan 14 parameter chart")
    return {
        group: torch.cat([torch.arange(spec.start, spec.stop, dtype=torch.int64)
                          for spec in layout.specs if spec.name.startswith(group + ".")])
        for group in PARAMETER_GROUPS
    }


def active_indices(layout: ParameterLayout, depth: int) -> tuple[Tensor, dict]:
    """Return a nested suffix and serializable depth/next-frontier metadata.

    Restrictions never toggle ``requires_grad``: the global layout and Fisher
    chart remain unchanged. Fitting can initially use ``functional_call``;
    this module does not maintain a potentially stale prefix-feature cache.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 6:
        raise ValueError("depth must be an integer from 1 through 6")
    groups = layer_indices(layout)
    active_groups = PARAMETER_GROUPS[-depth:]
    indices = torch.cat([groups[name] for name in active_groups])
    candidate = PARAMETER_GROUPS[-depth - 1] if depth < 6 else None
    metadata = {
        "depth": depth,
        "name": DEPTH_NAMES[depth - 1],
        "groups": list(active_groups),
        "parameter_names": [spec.name for spec in layout.specs
                            if spec.name.split(".")[0] in active_groups],
        "active_count": indices.numel(),
        "indices": indices.tolist(),
        "candidate_group": candidate,
        "candidate_indices": [] if candidate is None else groups[candidate].tolist(),
        "total_count": layout.total_numel,
    }
    return indices, metadata
