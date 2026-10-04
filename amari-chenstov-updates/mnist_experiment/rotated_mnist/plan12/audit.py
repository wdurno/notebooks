"""Phase 0 artifact, gauge, dense-Fisher, and numerical-damping audit."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.mnist_data import load_mnist_datasets
from src.mnist_model import build_canonical_model
from src.representations import DenseFisher, LowRankDiagonalFisher, representation_from_artifact

from ..data import RotatedPartitions
from ..run_phase3 import _estimate_initial_fisher
from .artifacts import UnitStore, file_hash
from .assets import ensure_replica_assets, runtime
from .gauge import (
    build_gauge_fixed_model,
    chart_embedding,
    exact_gauge_basis,
    load_canonical_state,
)
from .ridge import RidgeFisher, leading_subspace


AUDIT_REQUIRED = ("audit.json", "spectra.pt")
VALIDATION_REQUIRED = ("validation.json", "validation.pt")


def _uniform_subset(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    indices = torch.linspace(0, len(paths) - 1, maximum).round().long().tolist()
    return [paths[index] for index in indices]


def _spectral_summary(eigenvalues: torch.Tensor) -> dict[str, float | None]:
    clipped = eigenvalues.clamp_min(0)
    trace = float(clipped.sum())
    probabilities = clipped / max(trace, torch.finfo(clipped.dtype).tiny)
    positive = probabilities > 0
    entropy = -float(torch.sum(probabilities[positive] * torch.log(probabilities[positive])))
    positive_values = clipped[clipped > max(float(clipped.max()) * 1e-12, 1e-15)]
    return {
        "trace": trace,
        "effective_rank": math.exp(entropy),
        "top8_trace_fraction": float(clipped[-8:].sum()) / max(trace, torch.finfo(clipped.dtype).tiny),
        "largest_eigenvalue": float(clipped[-1]),
        "smallest_resolved_eigenvalue": float(positive_values[0]) if positive_values.numel() else 0.0,
        "resolved_condition_ratio": (
            float(clipped[-1] / positive_values[0]) if positive_values.numel() else None
        ),
    }


def _damping_locations(repo_root: Path) -> list[dict[str, Any]]:
    paths = (
        repo_root / "src/controller.py",
        repo_root / "src/representations.py",
        repo_root / "src/lanczos_wrapper.py",
        repo_root / "mnist_experiment/rotated_mnist/plan11/policy.py",
    )
    rows = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if "damping" in lowered or "trace_epsilon" in lowered:
                rows.append(
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "line": number,
                        "text": line.strip(),
                        "classification": "numerical_or_controller_guard_not_ewc_ridge",
                    }
                )
    return rows


def run_existing_artifact_audit(
    store: UnitStore,
    roots: tuple[Path, ...],
    *,
    resume: bool,
    maximum_spectra: int = 48,
) -> Path:
    unit = store.unit("phase0", "existing_audit", 1, detail={"roots": [str(root) for root in roots], "maximum_spectra": maximum_spectra})
    session = store.begin(unit, AUDIT_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, AUDIT_REQUIRED)
        assert completed is not None
        return completed
    candidates = sorted(
        path.parent
        for root in roots
        if root.exists()
        for path in root.rglob("final_state.pt")
        if (path.parent / "COMPLETED").is_file()
    )
    selected = _uniform_subset(candidates, maximum_spectra)
    raw_model, raw_layout = build_canonical_model(0, dtype=torch.float64)
    del raw_model
    gauge = exact_gauge_basis(raw_layout)
    rows = []
    spectra = []
    invalid = []
    for path in selected:
        try:
            integrity_path = path / "integrity.json"
            if integrity_path.is_file():
                import json

                integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
                if "final_state.pt" in integrity and integrity["final_state.pt"] != file_hash(path / "final_state.pt"):
                    raise RuntimeError("final-state integrity hash differs")
            import json

            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            state = torch.load(path / "final_state.pt", map_location="cpu", weights_only=False)
            fisher = representation_from_artifact(state["fisher"])
            if not isinstance(fisher, LowRankDiagonalFisher):
                raise RuntimeError("historical Fisher is not rank-plus-diagonal")
            dense = fisher.to_dense().to(torch.float64)
            eigenvalues = torch.linalg.eigvalsh((dense + dense.mT) / 2)
            spectral = _spectral_summary(eigenvalues)
            residual = fisher.residual_diagonal.to(torch.float64)
            gauge_action = dense @ gauge
            policy = config.get("policy") or {}
            row = {
                "path": str(path),
                "phase": config.get("phase"),
                "schedule": config.get("schedule"),
                "anchor": policy.get("anchor"),
                "gain": policy.get("gain"),
                **spectral,
                "residual_zero_count": int((residual == 0).sum()),
                "residual_near_zero_count": int((residual <= residual.max() * 1e-12).sum()),
                "gauge_action_frobenius": float(torch.linalg.matrix_norm(gauge_action, ord="fro")),
                "gauge_quadratic_mean": float(torch.trace(gauge.mT @ dense @ gauge) / gauge.shape[1]),
            }
            rows.append(row)
            spectra.append(eigenvalues)
        except Exception as error:
            invalid.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    if not rows:
        raise RuntimeError("Plan 12 found no valid historical spectra")
    audit = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "valid_count": len(rows),
        "invalid": invalid,
        "selection_rule": "lexicographic_uniform_subset",
        "aggregate": {
            key: {
                "mean": statistics.fmean(float(row[key]) for row in rows if row[key] is not None),
                "median": statistics.median(float(row[key]) for row in rows if row[key] is not None),
                "minimum": min(float(row[key]) for row in rows if row[key] is not None),
                "maximum": max(float(row[key]) for row in rows if row[key] is not None),
            }
            for key in ("effective_rank", "top8_trace_fraction", "resolved_condition_ratio", "gauge_action_frobenius")
        },
        "rows": rows,
        "damping_locations": _damping_locations(store.repo_root),
        "unrecoverable_from_historical_artifacts": [
            "intermediate model vectors for most Plan 11 trajectories",
            "matched dense archive Fishers after initialization",
            "independent local pseudo-true targets",
            "per-sample score histories for conditional variance",
        ],
    }
    session.write_json("audit.json", audit)
    session.write_torch("spectra.pt", {"eigenvalues": torch.stack(spectra)})
    return store.finish(session, AUDIT_REQUIRED)


def run_gauge_dense_validation(
    store: UnitStore,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    asset_path = ensure_replica_assets(store, "phase0", 1, data_root=data_root, resume=True)
    unit = store.unit("phase0", "gauge_dense_validation", 1)
    session = store.begin(unit, VALIDATION_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, VALIDATION_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    device, training_dtype, matrix_dtype = runtime(store.study)
    config = store.study.protocol_for_replica("phase0", 1)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    raw, raw_layout = build_canonical_model(config.replica_seed, device=device, dtype=training_dtype)
    raw.load_state_dict(assets["raw_initial_state"])
    chart, chart_layout = build_gauge_fixed_model(config.replica_seed, device=device, dtype=training_dtype)
    load_canonical_state(chart, raw)
    embedding = chart_embedding(raw_layout, chart_layout, device=device, dtype=matrix_dtype)
    gauge = exact_gauge_basis(raw_layout, device=device, dtype=matrix_dtype)
    base_inputs = assets["base_inputs"][:32].to(device=device, dtype=training_dtype)
    with torch.no_grad():
        raw_logits = raw(base_inputs)
        chart_logits = chart(base_inputs)
    difference = raw_logits - chart_logits
    common_shift_error = float(torch.max(torch.abs(difference - difference.mean(dim=1, keepdim=True))))
    probability_error = float(torch.max(torch.abs(raw_logits.softmax(1) - chart_logits.softmax(1))))

    train_dataset, _ = load_mnist_datasets(data_root, download=False)
    partitions = RotatedPartitions.from_mapping(assets["partitions"])
    direct_chart, direct_metrics = _estimate_initial_fisher(
        chart,
        chart_layout,
        train_dataset,
        partitions.reference,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    transformed = assets["chart_initial_dense_fisher"].to(device=device, dtype=matrix_dtype)
    raw_dense = assets["raw_initial_dense_fisher"].to(device=device, dtype=matrix_dtype)
    raw_compressed = representation_from_artifact(assets["raw_initial_fisher"], device=device)
    chart_compressed = representation_from_artifact(assets["chart_initial_fisher"], device=device)
    transformed_error = torch.linalg.matrix_norm(direct_chart - transformed, ord="fro") / torch.linalg.matrix_norm(direct_chart, ord="fro")
    raw_compression_error = torch.linalg.matrix_norm(raw_compressed.to_dense() - raw_dense, ord="fro") / torch.linalg.matrix_norm(raw_dense, ord="fro")
    chart_compression_error = torch.linalg.matrix_norm(chart_compressed.to_dense() - transformed, ord="fro") / torch.linalg.matrix_norm(transformed, ord="fro")
    eigenvalues, basis = leading_subspace(transformed, 8)
    vector = torch.linspace(-1, 1, chart_layout.total_numel, device=device, dtype=matrix_dtype)
    kappa = float(torch.trace(transformed) / transformed.shape[0]) * 0.01
    isotropic = RidgeFisher(DenseFisher(transformed), kappa, "isotropic")
    tail = RidgeFisher(DenseFisher(transformed), kappa, "tail", basis)
    identity = torch.eye(transformed.shape[0], device=device, dtype=matrix_dtype)
    tail_projector = identity - basis @ basis.mT
    validation = {
        "raw_parameter_count": raw_layout.total_numel,
        "chart_parameter_count": chart_layout.total_numel,
        "removed_dimension": raw_layout.total_numel - chart_layout.total_numel,
        "embedding_orthogonality_error": float(torch.linalg.matrix_norm(embedding.mT @ embedding - identity, ord="fro")),
        "embedding_gauge_orthogonality_error": float(torch.linalg.matrix_norm(embedding.mT @ gauge, ord="fro")),
        "common_shift_error": common_shift_error,
        "probability_error": probability_error,
        "direct_transformed_fisher_relative_error": float(transformed_error),
        "raw_exact_gauge_action_frobenius": float(torch.linalg.matrix_norm(raw_dense @ gauge, ord="fro")),
        "raw_compressed_gauge_action_frobenius": float(torch.linalg.matrix_norm(raw_compressed.to_dense() @ gauge, ord="fro")),
        "raw_compression_relative_error": float(raw_compression_error),
        "chart_compression_relative_error": float(chart_compression_error),
        "isotropic_quadratic_error": float(torch.abs(isotropic.quadratic(vector) - vector @ (transformed + kappa * identity) @ vector)),
        "tail_quadratic_error": float(torch.abs(tail.quadratic(vector) - vector @ (transformed + kappa * tail_projector) @ vector)),
        "direct_fisher_metrics": direct_metrics,
        "top_eigenvalues": eigenvalues[:16].tolist(),
        "wall_time_seconds": time.perf_counter() - started,
    }
    tolerance = 5e-5 if training_dtype == torch.float32 else 1e-10
    if common_shift_error > tolerance or probability_error > tolerance or float(transformed_error) > tolerance:
        raise RuntimeError(f"Plan 12 gauge equivalence failed: {validation}")
    session.write_json("validation.json", validation)
    session.write_torch(
        "validation.pt",
        {
            "direct_chart_fisher": direct_chart.cpu(),
            "transformed_chart_fisher": transformed.cpu(),
            "embedding": embedding.cpu(),
            "gauge_basis": gauge.cpu(),
        },
    )
    return store.finish(session, VALIDATION_REQUIRED)
