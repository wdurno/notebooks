"""Disjoint replay-to-EWC archive state for Plan 3 hybrids."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .directional_ridge import DirectionalRidgeLFUState, DirectionalRidgeUpdate
from .fisher import LFUBatchEstimate
from .lanczos_wrapper import LanczosDiagnostics, approximate_low_rank_diagonal
from .replay import FifoReplayBuffer, ReplayEvent
from .representations import (
    LowRankDiagonalFisher,
    PSDProjectionDiagnostics,
    project_psd_frobenius,
    representation_from_artifact,
)


HYBRID_ARCHIVE_STATE_SCHEMA_VERSION = 1
INITIAL_ARCHIVE_SOURCE_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class HybridArchiveState:
    anchor: Tensor
    fisher: LowRankDiagonalFisher
    initial_anchor_observations: int
    initial_fisher_score_observations: int
    archived_online_events: int = 0
    consolidation_steps: int = 0

    def validate(self) -> None:
        if (
            self.anchor.ndim != 1
            or not self.anchor.is_floating_point()
            or not torch.isfinite(self.anchor).all()
            or self.fisher.shape[0] != self.anchor.numel()
            or self.fisher.device != self.anchor.device
        ):
            raise ValueError("archive anchor and Fisher are incompatible")
        for name in (
            "initial_anchor_observations",
            "initial_fisher_score_observations",
            "archived_online_events",
            "consolidation_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"archive {name} must be a nonnegative integer")

    @property
    def parameter_count(self) -> int:
        return self.anchor.numel()

    @property
    def rank(self) -> int:
        return self.fisher.rank

    def canonical_persistent_bytes(self, scalar_bytes: int = 4) -> int:
        if not isinstance(scalar_bytes, int) or scalar_bytes < 1:
            raise ValueError("scalar_bytes must be a positive integer")
        return self.parameter_count * (self.rank + 2) * scalar_bytes

    def measured_tensor_bytes(self) -> int:
        return self.anchor.numel() * self.anchor.element_size() + self.fisher.storage_bytes()

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": HYBRID_ARCHIVE_STATE_SCHEMA_VERSION,
            "anchor": self.anchor.detach().cpu(),
            "fisher": self.fisher.artifact_mapping(),
            "initial_anchor_observations": self.initial_anchor_observations,
            "initial_fisher_score_observations": (
                self.initial_fisher_score_observations
            ),
            "archived_online_events": self.archived_online_events,
            "consolidation_steps": self.consolidation_steps,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        device: torch.device | str,
        anchor_dtype: torch.dtype,
        fisher_dtype: torch.dtype,
    ) -> "HybridArchiveState":
        if value.get("schema_version") != HYBRID_ARCHIVE_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported hybrid archive state schema")
        fisher = representation_from_artifact(
            dict(value["fisher"]),
            device=device,
        )
        if not isinstance(fisher, LowRankDiagonalFisher):
            raise ValueError("hybrid archive Fisher must be low-rank plus diagonal")
        state = cls(
            anchor=value["anchor"].to(device=device, dtype=anchor_dtype),
            fisher=fisher.to(device=device, dtype=fisher_dtype),
            initial_anchor_observations=value["initial_anchor_observations"],
            initial_fisher_score_observations=(
                value["initial_fisher_score_observations"]
            ),
            archived_online_events=value["archived_online_events"],
            consolidation_steps=value["consolidation_steps"],
        )
        state.validate()
        return state


@dataclasses.dataclass(frozen=True)
class InitialArchiveSource:
    state: HybridArchiveState
    artifact_sha256: str
    condition: str
    checkpoint_step: int


def load_initial_archive_source(
    path: str | Path,
    expected_anchor: Tensor,
    *,
    expected_rank: int,
    initial_anchor_observations: int,
    initial_fisher_score_observations: int,
    device: torch.device | str,
    anchor_dtype: torch.dtype,
    fisher_dtype: torch.dtype,
) -> InitialArchiveSource:
    artifact_path = Path(path)
    value = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if (
        value.get("schema_version") == INITIAL_ARCHIVE_SOURCE_SCHEMA_VERSION
        and value.get("kind") == "plan3_initial_archive"
    ):
        state_mapping = value.get("archive_state")
        if not isinstance(state_mapping, Mapping):
            raise ValueError("standalone archive source is missing archive state")
        state = HybridArchiveState.from_mapping(
            state_mapping,
            device=device,
            anchor_dtype=anchor_dtype,
            fisher_dtype=fisher_dtype,
        )
        expected = expected_anchor.to(device=device, dtype=anchor_dtype)
        if not torch.equal(state.anchor, expected):
            raise ValueError(
                "standalone archive source anchor does not match paired initialization"
            )
        if state.rank != expected_rank:
            raise ValueError("standalone archive source has the wrong Fisher rank")
        if state.initial_anchor_observations != initial_anchor_observations:
            raise ValueError(
                "standalone archive source has the wrong initialization sample count"
            )
        if not (
            0 < state.initial_fisher_score_observations
            <= initial_fisher_score_observations
        ):
            raise ValueError(
                "standalone archive score count exceeds the configured Fisher budget"
            )
        return InitialArchiveSource(
            state=state,
            artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            condition="p0_adaptive_reference_fisher",
            checkpoint_step=0,
        )

    conditions = value.get("conditions")
    if value.get("schema_version") != 4 or not isinstance(conditions, Mapping):
        raise ValueError("archive source must be a schema-v4 controller checkpoint")
    if len(conditions) != 1:
        raise ValueError("archive source must contain exactly one Fisher condition")
    condition, checkpoints = next(iter(conditions.items()))
    checkpoint = checkpoints.get("0") if isinstance(checkpoints, Mapping) else None
    if not isinstance(checkpoint, Mapping):
        raise ValueError("archive source is missing checkpoint step zero")
    decision = checkpoint.get("controller_decision")
    if not isinstance(decision, Mapping) or float(decision.get("applied_pi")) != 0.05:
        raise ValueError("archive source must be the fixed-pi=.05 EWC condition")
    source_anchor = checkpoint["parameter"].to(
        device=device,
        dtype=anchor_dtype,
    )
    expected = expected_anchor.to(device=device, dtype=anchor_dtype)
    if not torch.equal(source_anchor, expected):
        raise ValueError("archive source anchor does not match paired initialization")
    fisher = representation_from_artifact(
        dict(checkpoint["representation"]),
        device=device,
    )
    if not isinstance(fisher, LowRankDiagonalFisher) or fisher.rank != expected_rank:
        raise ValueError("archive source has the wrong Fisher representation or rank")
    state = HybridArchiveState(
        anchor=source_anchor,
        fisher=fisher.to(device=device, dtype=fisher_dtype),
        initial_anchor_observations=initial_anchor_observations,
        initial_fisher_score_observations=initial_fisher_score_observations,
    )
    state.validate()
    return InitialArchiveSource(
        state=state,
        artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        condition=str(condition),
        checkpoint_step=0,
    )


@dataclasses.dataclass(frozen=True)
class HybridReplayTransition:
    replay: FifoReplayBuffer
    evicted: tuple[ReplayEvent, ...]
    archived_event_ids: frozenset[int]


def stage_hybrid_replay_transition(
    replay: FifoReplayBuffer,
    current_events: Iterable[ReplayEvent],
    archived_event_ids: Iterable[int],
) -> HybridReplayTransition:
    current = tuple(current_events)
    archive_before = frozenset(int(value) for value in archived_event_ids)
    replay_before = {event.event_id for event in replay.events}
    current_ids = {event.event_id for event in current}
    if len(current_ids) != len(current):
        raise ValueError("current events repeat an identity")
    if archive_before.intersection(replay_before | current_ids):
        raise ValueError("active and archived event identities overlap")
    if replay_before.intersection(current_ids):
        raise ValueError("current events already occur in replay")

    candidate = FifoReplayBuffer.from_mapping(replay.to_mapping())
    evicted = candidate.insert(current)
    evicted_ids = {event.event_id for event in evicted}
    if archive_before.intersection(evicted_ids):
        raise ValueError("an event would be archived more than once")
    archive_after = archive_before | evicted_ids
    replay_after = {event.event_id for event in candidate.events}
    if archive_after.intersection(replay_after):
        raise RuntimeError("staged replay and archive identities overlap")
    expected = archive_before | replay_before | current_ids
    if archive_after | replay_after != expected:
        raise RuntimeError("staged transition lost or duplicated an online event")
    return HybridReplayTransition(candidate, evicted, frozenset(archive_after))


@dataclasses.dataclass(frozen=True)
class ArchiveFisherUpdate:
    representation: LowRankDiagonalFisher
    blend_gain: float
    previous_trace: float
    fresh_trace: float
    candidate_trace: float
    lanczos: LanczosDiagnostics


@dataclasses.dataclass(frozen=True)
class ArchiveLFUFisherUpdate:
    representation: LowRankDiagonalFisher
    blend_gain: float
    correction_method: str
    previous_trace: float
    fresh_trace: float
    prediction_trace: float
    candidate_trace: float
    correction_fro: float
    prediction_fro: float
    candidate_fro: float
    projection_backend: str
    projection: PSDProjectionDiagnostics
    ridge: DirectionalRidgeUpdate
    lanczos: LanczosDiagnostics


def _project_psd_robust(matrix: Tensor):
    try:
        return project_psd_frobenius(matrix), matrix.device.type
    except RuntimeError as error:
        if "linalg.eigh" not in str(error) or "failed to converge" not in str(error):
            raise
        projected = project_psd_frobenius(
            matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        return dataclasses.replace(
            projected,
            projected=projected.projected.to(device=matrix.device, dtype=matrix.dtype),
            symmetrized=projected.symmetrized.to(
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ), "cpu_float64_fallback"


def update_archive_fisher_lfu(
    previous: LowRankDiagonalFisher,
    estimate: LFUBatchEstimate,
    direction: Tensor,
    ridge_state: DirectionalRidgeLFUState,
    *,
    correction_method: str,
    blend_gain: float,
    rank: int,
    lanczos_seed: int,
) -> ArchiveLFUFisherUpdate:
    """Apply one lagged LFU, project to PSD, and recompress the archive Fisher."""

    if correction_method not in {"ac_only", "full_lfu"}:
        raise ValueError("archive LFU correction must be ac_only or full_lfu")
    expected = previous.shape
    if (
        estimate.fisher.shape != expected
        or estimate.amari_chentsov.shape != expected
        or estimate.residual.shape != expected
        or direction.shape != (expected[0],)
    ):
        raise ValueError("archive LFU statistics do not match the Fisher shape")
    tensors = (
        estimate.fisher,
        estimate.amari_chentsov,
        estimate.residual,
        direction,
    )
    if any(
        tensor.device != previous.device
        or tensor.dtype != previous.dtype
        or not torch.isfinite(tensor).all()
        for tensor in tensors
    ):
        raise ValueError("archive LFU tensors must match the previous Fisher")
    if not 0.0 < float(blend_gain) <= 1.0:
        raise ValueError("archive Fisher blend gain must be in (0, 1]")

    ridge = ridge_state.update(
        direction,
        estimate.amari_chentsov,
        estimate.residual,
    )
    correction = (
        ridge.amari_chentsov if correction_method == "ac_only" else ridge.full
    )
    prediction = previous.to_dense() + correction
    candidate = (
        (1.0 - float(blend_gain)) * prediction
        + float(blend_gain) * estimate.fisher
    )
    candidate = (candidate + candidate.mT) / 2
    projected, projection_backend = _project_psd_robust(candidate)
    approximation = approximate_low_rank_diagonal(
        lambda vector: projected.projected @ vector,
        torch.diagonal(projected.projected),
        rank=rank,
        seed=lanczos_seed,
    )
    return ArchiveLFUFisherUpdate(
        representation=approximation.representation,
        blend_gain=float(blend_gain),
        correction_method=correction_method,
        previous_trace=float(previous.diagonal_vector().sum()),
        fresh_trace=float(torch.trace(estimate.fisher)),
        prediction_trace=float(torch.trace(prediction)),
        candidate_trace=float(torch.trace(candidate)),
        correction_fro=float(torch.linalg.matrix_norm(correction, ord="fro")),
        prediction_fro=float(torch.linalg.matrix_norm(prediction, ord="fro")),
        candidate_fro=float(torch.linalg.matrix_norm(candidate, ord="fro")),
        projection_backend=projection_backend,
        projection=projected.diagnostics,
        ridge=ridge,
        lanczos=approximation.diagnostics,
    )


def blend_archive_fisher(
    previous: LowRankDiagonalFisher,
    fresh_fisher: Tensor,
    *,
    blend_gain: float,
    rank: int,
    lanczos_seed: int,
) -> ArchiveFisherUpdate:
    if (
        fresh_fisher.shape != previous.shape
        or fresh_fisher.device != previous.device
        or fresh_fisher.dtype != previous.dtype
        or not torch.isfinite(fresh_fisher).all()
    ):
        raise ValueError("fresh Fisher must match the archive representation")
    if not 0.0 < float(blend_gain) <= 1.0:
        raise ValueError("archive Fisher blend gain must be in (0, 1]")
    symmetric_fresh = (fresh_fisher + fresh_fisher.mT) / 2
    diagonal = (
        (1.0 - float(blend_gain)) * previous.diagonal_vector()
        + float(blend_gain) * torch.diagonal(symmetric_fresh)
    )

    def operator(vector: Tensor) -> Tensor:
        return (
            (1.0 - float(blend_gain)) * previous.matvec(vector)
            + float(blend_gain) * (symmetric_fresh @ vector)
        )

    approximation = approximate_low_rank_diagonal(
        operator,
        diagonal,
        rank=rank,
        seed=lanczos_seed,
    )
    previous_trace = float(previous.diagonal_vector().sum())
    fresh_trace = float(torch.trace(symmetric_fresh))
    return ArchiveFisherUpdate(
        representation=approximation.representation,
        blend_gain=float(blend_gain),
        previous_trace=previous_trace,
        fresh_trace=fresh_trace,
        candidate_trace=(
            (1.0 - float(blend_gain)) * previous_trace
            + float(blend_gain) * fresh_trace
        ),
        lanczos=approximation.diagnostics,
    )
