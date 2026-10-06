"""Named adaptation-geometry conditions for Plan 13."""

from __future__ import annotations

import dataclasses
from typing import Any

from .controller import InnovationControllerConfig


KINDS = {
    "no_update",
    "full_space",
    "bias_only",
    "head_only",
    "random_rank_matched",
    "static_projector",
    "static_subgd",
    "online_subgd",
    "adaptive_subgd",
}


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    covariance_half_life: float | None = None
    epsilon: float = 0.0
    controller: InnovationControllerConfig | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unsupported Plan 13 condition kind: {self.kind}")
        if self.epsilon < 0:
            raise ValueError("orthogonal floor must be nonnegative")
        if self.kind == "online_subgd" and self.covariance_half_life is None:
            raise ValueError("online SubGD requires a covariance half-life")
        if self.kind == "adaptive_subgd" and self.controller is None:
            raise ValueError("adaptive SubGD requires a controller")
        if self.kind != "adaptive_subgd" and self.controller is not None:
            raise ValueError("only adaptive SubGD accepts a controller")
        if self.kind != "adaptive_subgd" and self.epsilon != 0:
            raise ValueError("only adaptive SubGD accepts an orthogonal floor")

    def mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        return value


def phase2_conditions(half_lives: tuple[float, ...]) -> tuple[Condition, ...]:
    return (
        Condition("no_update", "no_update"),
        Condition("full_space", "full_space"),
        Condition("head_only", "head_only"),
        Condition("random_rank_matched", "random_rank_matched"),
        Condition("static_projector", "static_projector"),
        Condition("static_subgd", "static_subgd"),
        *tuple(
            Condition(
                f"online_subgd_h{half_life:g}",
                "online_subgd",
                covariance_half_life=half_life,
            )
            for half_life in half_lives
        ),
    )


def default_probe_conditions() -> tuple[Condition, ...]:
    configs = (
        InnovationControllerConfig(2.0, 2.0, 16.0, 2.0, 2.0),
        InnovationControllerConfig(4.0, 4.0, 32.0, 4.0, 4.0),
        InnovationControllerConfig(8.0, 8.0, 32.0, 4.0, 8.0),
    )
    return tuple(
        Condition(f"adaptive_probe_{index}", "adaptive_subgd", controller=config)
        for index, config in enumerate(configs, start=1)
    )


def adaptive_floor_conditions(
    controller: InnovationControllerConfig,
    *,
    prefix: str = "adaptive_floor",
) -> tuple[Condition, ...]:
    return tuple(
        Condition(
            f"{prefix}_{epsilon:g}",
            "adaptive_subgd",
            epsilon=epsilon,
            controller=controller,
        )
        for epsilon in (0.0, 0.01, 0.05, 0.10)
    )


def geometry_rate_conditions(
    controller: InnovationControllerConfig,
) -> tuple[Condition, ...]:
    """Axial beta-response candidates with the trust mapping held fixed."""

    variants = (
        dataclasses.replace(
            controller,
            beta_min_half_life=controller.beta_min_half_life * 2,
            beta_max_half_life=controller.beta_max_half_life * 2,
            beta_scale=max(controller.beta_scale / 2, 1e-6),
        ),
        controller,
        dataclasses.replace(
            controller,
            beta_min_half_life=max(controller.beta_min_half_life / 2, 1e-6),
            beta_max_half_life=max(controller.beta_max_half_life / 2, 1e-6),
            beta_scale=controller.beta_scale * 2,
        ),
    )
    return tuple(
        Condition(f"adaptive_geometry_{index}", "adaptive_subgd", controller=value)
        for index, value in enumerate(variants, start=1)
    )


def condition_from_mapping(value: dict[str, Any]) -> Condition:
    controller = value.get("controller")
    return Condition(
        name=str(value["name"]),
        kind=str(value["kind"]),
        covariance_half_life=(
            None
            if value.get("covariance_half_life") is None
            else float(value["covariance_half_life"])
        ),
        epsilon=float(value.get("epsilon", 0.0)),
        controller=(
            None
            if controller is None
            else InnovationControllerConfig(**controller)
        ),
    )
