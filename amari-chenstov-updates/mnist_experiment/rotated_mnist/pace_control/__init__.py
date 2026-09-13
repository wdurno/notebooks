"""Detachable fixed-composition pace-control experiments for Plan 10."""

from .theory import (
    centered_pace,
    fixed_batch_risk,
    fixed_pi_q_equilibrium,
    fixed_pi_q_at_step,
    fixed_pi_q_update,
    movement_energy_target,
    optimal_fixed_batch_pi,
    pace_roots,
    stationary_movement_target,
)

__all__ = [
    "centered_pace",
    "fixed_batch_risk",
    "fixed_pi_q_equilibrium",
    "fixed_pi_q_at_step",
    "fixed_pi_q_update",
    "movement_energy_target",
    "optimal_fixed_batch_pi",
    "pace_roots",
    "stationary_movement_target",
]
