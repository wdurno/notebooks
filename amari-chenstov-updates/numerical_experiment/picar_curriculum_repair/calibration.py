"""Bounded numerical scaling and development-only retention selection."""

import copy
from dataclasses import replace
import math
import time

import torch

from numerical_experiment.picar.fix3_student import _batch
from numerical_experiment.picar_solver_repair.learner import SolverLearner
from src.continual import picar_math as pm
from src.seeding import derive_seed_map


def fd_scale(model, config, state):
    """Probe the actually visited timing histories, never a held-out reference.

    The learner state and FD probe RNG are separate from production learners.
    Normalized Rademacher probes and the existing symmetric FD formula are
    unchanged; only the multiplier is calibrated to a live-gradient scale.
    """
    learner = SolverLearner(model, config.learning, config.seeds(0), recipe="proximal_3")
    learner.load_state_dict(state["learner"])
    records = state["eligible"]
    if len(records) < 2:
        raise ValueError("FD calibration requires two eligible timing histories")
    batch = _batch(records[0], records)
    params = learner.params
    seed = derive_seed_map(198302, ("fd_resolution",))["fd_resolution"]
    generator = torch.Generator(device=learner.device).manual_seed(seed)
    probes = pm.rademacher_probes(params, learner.banks["stm"], 1, generator=generator)
    def vector(loss):
        gradients = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
            for p, g in zip(params.values(), gradients)])
    data_norm = float(vector(learner.nll(params, batch).mean()).norm())
    rows, previous = [], None
    for step in (.005, .01, .02):
        tick = time.monotonic()
        value = pm.stm_fd_penalty(lambda p: learner.predict(p, batch), params, learner.design(batch),
            learner.banks["stm"], probes, torch.ones(1, device=learner.device), step)
        gradient = vector(value)
        cosine = None if previous is None else float(torch.nn.functional.cosine_similarity(gradient, previous, dim=0))
        rows.append({"step": step, "value": float(value.detach()), "gradient_norm": float(gradient.norm()),
            "cosine_to_previous": cosine, "seconds": time.monotonic()-tick})
        previous = gradient
    stable = (all(math.isfinite(r["value"]) and r["value"] > 0 and math.isfinite(r["gradient_norm"])
                  and r["gradient_norm"] > 0 for r in rows) and math.isfinite(data_norm) and data_norm > 0
        and max(r["value"] for r in rows)/min(r["value"] for r in rows) <= 1.25
        and max(r["gradient_norm"] for r in rows)/min(r["gradient_norm"] for r in rows) <= 1.5
        and all(r["cosine_to_previous"] >= .95 for r in rows[1:]))
    if not stable:
        raise ValueError("FD resolution check failed; preserve diagnostics and repair before main learning: "+repr(rows))
    return {"rows": rows, "stable": stable, "step": .01, "data_gradient_norm": data_norm,
        "strengths": [ratio*data_norm/rows[1]["gradient_norm"] for ratio in (.01, .1, 1.)],
        "target_gradient_ratios": [.01, .1, 1.], "seed": seed,
        "probe_age": 1., "histories": [records[0]["arrival"]], "new_observations": 0}


def final_quality(summary, environment):
    refs = summary.get("reference_scores", [])
    final = max(refs, key=lambda p: p["target"]) if refs else {"rows": []}
    rows = [r for r in final["rows"] if r["environment"] == environment]
    panels = [p for p in summary.get("evaluations", []) if p["difficulty"] == environment]
    panel = max(panels, key=lambda p: p["target"]) if panels else {"rows": []}
    return {"nll": sum(r["joint_nll"] for r in rows)/len(rows) if rows else None,
        "success": sum(e["success"] for e in panel["rows"])/len(panel["rows"]) if panel["rows"] else None,
        "episodes": len(panel["rows"])}


def select(control, candidates):
    """Explicitly exploratory small-panel matching, not equivalence testing."""
    old = final_quality(control, "A")
    rows = []
    for candidate in candidates:
        for key in ("seed", "reference_id", "initial_asset_id"):
            if candidate["settings"].get(key) != control["settings"].get(key):
                raise ValueError("calibration candidate and control must share "+key)
        a, b = final_quality(candidate, "A"), final_quality(candidate, "B_initial")
        values = (a["nll"], b["nll"], a["success"], old["success"], old["nll"], b["success"])
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError("calibration requires scored fixed A/B-initial reference and task panels")
        success_tolerance = 1/max(1, a["episodes"])
        matched = a["success"] >= old["success"]-success_tolerance and a["nll"] <= old["nll"]+.25
        rows.append({"path": candidate["unit_path"], "strength": candidate["fd_strength"],
            "ridge": candidate["ridge"], "A": a, "B_initial": b, "retention_match_diagnostic": matched,
            "A_nll_gap": a["nll"]-old["nll"]})
    if not rows:
        raise ValueError("positive FD candidate is required")
    acceptable = [r for r in rows if r["retention_match_diagnostic"]]
    chosen = min(acceptable or rows, key=lambda r: (0 if acceptable else max(0., r["A_nll_gap"]),
        r["B_initial"]["nll"], r["strength"]))
    return {"rows": rows, "selected": chosen, "control_A": old,
        "selection_rule": "A success no more than one pilot episode lower, A reference NLL gap <=.25; then smallest B-initial reference NLL. If unmatched, smallest positive A NLL deterioration first.",
        "retention_equivalence": False, "development_only": True,
        "warning": None if acceptable else "A retention could not be matched in bounded calibration; interpret main outcomes as a trade-off."}
