"""Phase 3 empirical and deformed-MP ridge calibration units."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import UnitStore
from .assets import runtime
from .config import PI
from .local_response import REFERENCE_REQUIRED, run_anchor_reference
from .spectral import (
    effective_sample_size_ema,
    empirical_quantile,
    empirical_tail_maxima,
    fit_local_power_law,
    pit_value,
    power_law_quadrature,
    solve_deformed_mp_edge,
)


CALIBRATION_REQUIRED = ("calibration.pt", "summary.json")


def run_calibration(
    store: UnitStore,
    anchor_index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    reference_path = run_anchor_reference(store, anchor_index, data_root=data_root, resume=True)
    unit = store.unit("phase3", "calibration", anchor_index)
    session = store.begin(unit, CALIBRATION_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, CALIBRATION_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    device, _, matrix_dtype = runtime(store.study)
    reference = torch.load(reference_path / "reference.pt", map_location="cpu", weights_only=False)
    scores = reference["scores"].to(device=device, dtype=matrix_dtype)
    split = scores.shape[0] // 2
    if split < 8 or scores.shape[0] - split < 8:
        raise RuntimeError("MP calibration requires at least eight disjoint fit and calibration scores")
    fit_scores = scores[:split]
    calibration_scores = scores[split:]
    fisher = fit_scores.mT @ fit_scores / fit_scores.shape[0]
    fisher = (fisher + fisher.mT) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh((fisher + fisher.mT) / 2)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    resolved_rank = 8
    effective_size = effective_sample_size_ema(batch_size=4, gain=PI)
    sample_size = max(8, round(effective_size))
    tail_dimension = fisher.shape[0] - resolved_rank
    gamma = tail_dimension / effective_size
    status = "complete"
    warning = None
    theoretical: dict[str, Any] | None = None
    pseudo: dict[str, Any] | None = None
    full_maxima = torch.empty(0)
    pseudo_maxima = torch.empty(0)
    try:
        fit = fit_local_power_law(eigenvalues[:resolved_rank], start_rank=3, stop_rank=8)
        values, weights = power_law_quadrature(
            fit,
            first_rank=resolved_rank + 1,
            parameter_count=fisher.shape[0],
        )
        edge = solve_deformed_mp_edge(
            values,
            weights,
            gamma=gamma,
            effective_sample_size=effective_size,
        )
        full_maxima = empirical_tail_maxima(
            calibration_scores,
            eigenvectors[:, :resolved_rank],
            sample_size=sample_size,
            resamples=store.study.phase3_resamples,
            seed=store.study.seed("phase3:bootstrap", anchor_index),
        )
        empirical = empirical_quantile(full_maxima)
        theoretical = {
            "fit": fit.mapping(),
            "edge": edge.mapping(),
            "empirical_quantile_99": empirical,
            "empirical_to_theoretical_ratio": empirical / edge.upper_quantile,
            "outlier_check": "no_separate_spike_model; empirical maximum is the protective benchmark",
        }

        pseudo_rank = resolved_rank - 2
        pseudo_fit = fit_local_power_law(eigenvalues[:pseudo_rank], start_rank=2, stop_rank=pseudo_rank)
        pseudo_values, pseudo_weights = power_law_quadrature(
            pseudo_fit,
            first_rank=pseudo_rank + 1,
            parameter_count=fisher.shape[0],
        )
        pseudo_edge = solve_deformed_mp_edge(
            pseudo_values,
            pseudo_weights,
            gamma=(fisher.shape[0] - pseudo_rank) / effective_size,
            effective_sample_size=effective_size,
        )
        pseudo_maxima = empirical_tail_maxima(
            calibration_scores,
            eigenvectors[:, :pseudo_rank],
            sample_size=sample_size,
            resamples=store.study.phase3_resamples,
            seed=store.study.seed("phase3:pseudo-bootstrap", anchor_index),
        )
        observed = float(eigenvalues[pseudo_rank])
        pseudo = {
            "held_out_ranks": [pseudo_rank + 1, resolved_rank],
            "observed_maximum": observed,
            "fit": pseudo_fit.mapping(),
            "edge": pseudo_edge.mapping(),
            "empirical_pit": pit_value(pseudo_maxima, observed),
        }
        if fit.r_squared < 0.8 or not edge.regular:
            status = "likely_miscalibrated"
            warning = "poor power-law fit or nonregular deformed-MP edge"
    except Exception as error:
        status = "failed_theoretical_model"
        warning = f"{type(error).__name__}: {error}"

    mean_eigenvalue = float(torch.trace(fisher) / fisher.shape[0])
    heuristic = 0.01 * float(eigenvalues[0])
    selected = (
        theoretical["empirical_quantile_99"]
        if theoretical is not None
        else heuristic
    )
    summary = {
        "anchor_index": anchor_index,
        "status": status,
        "warning": warning,
        "parameter_count": fisher.shape[0],
        "resolved_rank": resolved_rank,
        "effective_sample_size": effective_size,
        "bootstrap_sample_size": sample_size,
        "spectral_fit_score_count": fit_scores.shape[0],
        "empirical_calibration_score_count": calibration_scores.shape[0],
        "sample_split": "first_half_spectral_fit_second_half_empirical_calibration",
        "gamma": gamma,
        "top_eigenvalues": eigenvalues[:16].tolist(),
        "mean_eigenvalue": mean_eigenvalue,
        "heuristic_kappa": heuristic,
        "theoretical": theoretical,
        "pseudo_tail": pseudo,
        "selected_empirical_kappa": selected,
        "selected_scale_ratio": selected / mean_eigenvalue,
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch(
        "calibration.pt",
        {"full_maxima": full_maxima, "pseudo_maxima": pseudo_maxima},
    )
    session.write_json("summary.json", summary)
    return store.finish(session, CALIBRATION_REQUIRED)
