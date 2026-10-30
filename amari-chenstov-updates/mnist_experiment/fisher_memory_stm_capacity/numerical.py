"""Fixed-training-state audits, separate from predictive selection and S timing."""

from dataclasses import replace
import math

import torch

from mnist_experiment.fisher_memory.learner import _bounded_solve
from mnist_experiment.fisher_memory_grid_registration.runner import settings
from mnist_experiment.fisher_memory_grid_registration.compression import compress_fisher as exact_compress
from mnist_experiment import plan15_grid_subspace as subspace
from src.representations import representation_from_artifact
from . import core, model
from .config import FD


def audit_policy(cfg, recipe, snapshots, fitted, layout, data, check=lambda: None):
    if not snapshots:
        raise ValueError("selected policy has no training-state snapshots")
    rows = []
    if recipe.method == FD:
        fd = core.calibrate_fd(replace(cfg,fd_step=recipe.fd_step),snapshots,fitted,layout,check)
        if fd["selected_step"] != recipe.fd_step:
            raise ValueError("selected-policy FD step needs a versioned correction")
    else:
        fd = None
    ltm, _ = model.memory_indices(layout)
    for snapshot in snapshots:
        check()
        theta = snapshot["theta"].to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
        arrays = [snapshot[k].to(theta) for k in ("new_x","new_y","replay_x","replay_y","ages")]
        objective, metadata = core.prepare_objective(fitted,layout,theta,*arrays,
            snapshot["fisher"],snapshot["anchor"],recipe.method,settings(cfg,recipe),snapshot["seed"],check)
        fit_cfg = settings(cfg,recipe)
        accepted, diagnostics = _bounded_solve(theta,objective,fit_cfg,check)
        expanded = replace(fit_cfg,max_iter=2*fit_cfg.max_iter,max_eval=2*fit_cfg.max_eval)
        reference, ref_diag = _bounded_solve(theta,objective,expanded,check)
        value, value_ref = diagnostics["final_objective"], ref_diag["final_objective"]
        constant = 54*math.log(cfg.sigma_y*math.sqrt(2*math.pi))
        gap = max(0.,value-value_ref)/max(abs(value-constant),1e-12)
        x = torch.cat((arrays[0],arrays[2]))
        with torch.no_grad():
            start_y = model.flat_prediction(fitted,layout,theta,x)
            accepted_y = model.flat_prediction(fitted,layout,accepted,x)
            reference_y = model.flat_prediction(fitted,layout,reference,x)
        change = float((reference_y-accepted_y).double().norm())
        update = float((accepted_y-start_y).double().norm())
        sensitivity = change/max(update,1e-12)
        solve_ok = (not diagnostics["solver_failed"] and not ref_diag["solver_failed"]
                    and gap <= cfg.audit_objective_tolerance and sensitivity <= cfg.numerical_tolerance)

        # Rebuild this retirement's source matrix, rather than auditing only an
        # already-compressed matrix. Its evictees follow the frozen FIFO clock.
        segment = data["plan"]["segments"][snapshot["step"]]
        evicted = list(range(max(0,segment["start"]-recipe.capacity),
                             max(0,segment["end"]-recipe.capacity)))
        matrix_error, compression_ok = None, True
        if evicted:
            fresh = core.reference_fisher(fitted,layout,accepted,
                data["stream_x"][evicted].to(theta),indices=ltm,chunk_size=cfg.fisher_chunk,check_stop=check)
            old = representation_from_artifact(snapshot["fisher"]).to_dense().to(fresh)
            eta = -math.expm1(len(evicted)/cfg.nominal_batch*math.log1p(-cfg.pi))
            source = (1-eta)*old+eta*fresh
            compressed, details = subspace.compress_fisher(source,cfg.archive_rank,snapshot["seed"],check,
                                                         "subspace_"+cfg.device)
            exact, _ = exact_compress(source,cfg.archive_rank,snapshot["seed"],check,"exact_"+cfg.device)
            a = representation_from_artifact(compressed).to_dense()
            b = representation_from_artifact(exact).to_dense().to(a)
            matrix_error = float((a-b).norm()/b.norm().clamp_min(1e-30))
            compression_ok = matrix_error <= .01
        rows.append(dict(n=snapshot["n"], seed=snapshot["seed"],
            relative_objective_gap=gap, prediction_update_sensitivity=sensitivity,
            bounded_solver_failed=diagnostics["solver_failed"], reference_solver_failed=ref_diag["solver_failed"],
            solve_passed=solve_ok, compression_relative_frobenius=matrix_error,
            compression_passed=compression_ok, retirement_count=len(evicted),
            passed=solve_ok and compression_ok))
    return dict(passed=all(r["passed"] for r in rows), rows=rows, fd=fd,
                population="fixed development training states; no holdout gradients")

