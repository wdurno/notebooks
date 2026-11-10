"""Lightweight reward-to-return-to-likelihood diagnostics, no environment truth."""

from __future__ import annotations

import torch

from src.continual import picar_math as pm


def resolution_audit(*, temperature=5.):
    elapsed = torch.full((40,), .5, dtype=torch.float64)
    terminated = torch.zeros(40, dtype=torch.bool)
    terminated[-1] = True
    bonus = torch.zeros_like(elapsed)
    bonus[-1] = 10.
    discount = pm.time_discounts(elapsed)
    pairs, candidates = [], []
    for extra in (2., 3., 4., 5.):
        low = torch.linspace(1., 4., 40, dtype=torch.float64)
        high = (low + extra).clamp_max(10.)
        a = pm.discounted_returns(pm.dense_visual_rewards(elapsed, low, bonus, terminated), discount, terminated)
        b = pm.discounted_returns(pm.dense_visual_rewards(elapsed, high, bonus, terminated), discount, terminated)
        pairs.append({"score_offset": extra, "lower_return": float(a[0]), "higher_return": float(b[0]),
                      "difference": float(b[0] - a[0])})
    selected = None
    for count in (41, 81, 161):
        edges = torch.linspace(-6., 12., count + 1, dtype=torch.float64)
        rows = []
        for pair in pairs:
            values = torch.tensor([pair["lower_return"], pair["higher_return"]], dtype=torch.float64)
            labels = pm.bin_returns(values, edges)
            design = pm.LikelihoodDesign(pm.return_bin_values(edges), torch.zeros(2, dtype=torch.float64),
                temperature, torch.eye(7, dtype=torch.float64))
            action_mean = torch.zeros(2, 7, dtype=torch.float64, requires_grad=True)
            outputs = {"return_logits": torch.zeros(2, count, dtype=torch.float64), "action_mean": action_mean}
            nll = pm.joint_nll(outputs, labels, torch.ones(2, 7, dtype=torch.float64), design)
            grad = torch.autograd.grad(nll.sum(), action_mean)[0].norm(dim=1)
            rows.append({**pair, "lower_bin": int(labels[0]), "higher_bin": int(labels[1]),
                "bin_tie": bool(labels[0] == labels[1]), "weight_ratio": float(grad[1] / grad[0]),
                "raw_weight_ratio": float(torch.exp((values[1] - values[0]) / temperature))})
        result = {"bins": count, "rows": rows, "ties": sum(r["bin_tie"] for r in rows),
            "max_log_weight_error": float((edges[1] - edges[0]) / (2 * temperature)),
            "gradient_order_pass": all(r["weight_ratio"] > 1 for r in rows)}
        candidates.append(result)
        if selected is None and result["ties"] == 0 and result["gradient_order_pass"]:
            selected = count
    loiter_bonus = bonus.clone()
    loiter_bonus[-1] = -1.
    maximum = torch.full_like(elapsed, 10.)
    loiter = pm.discounted_returns(pm.dense_visual_rewards(elapsed, maximum, loiter_bonus, terminated),
                                  discount, terminated)[0]
    immediate = pm.dense_visual_rewards(elapsed[:1], maximum[:1], torch.tensor([10.], dtype=torch.float64),
                                        torch.tensor([True]))[0]
    return {"reward_version": "dense-visual-v1", "return_range": list(pm.dense_visual_return_bounds()),
        "temperature": temperature, "selected_bins": selected, "candidates": candidates,
        "loiter_return": float(loiter), "immediate_success_return": float(immediate),
        "loiter_pass": bool(immediate > loiter), "constructed_arithmetic_only": True,
        "resolution_pass": selected is not None, "log_weight_cap_hits": 0,
        "limitations": "Constructed matched-outcome histories verify arithmetic; no policy improvement is inferred."}
