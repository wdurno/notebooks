"""Seal a bounded paired-development bracket proposal for the next round.

The unchanged runner consumes this ordinary immutable decision on resume.
This review records its own source and never changes scientific source seals,
completed decisions, candidates, matching tolerance, or confirmation settings.
"""

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

from ...config import Recipe
from ..config import Config
from ..runtime import DEFAULT_ROOT, HISTORY_ROOT, DATA_ROOT, open_store, read_exported_records, read_json


def paired_bracket(records, *, arm, member, low, high, strength, reference_ridge=10., tolerance=.1):
    """Validate observed paired bracketing of both statistics; propose no outcome."""
    if arm not in "BCDE" or not 0 < low < high <= 1000:
        raise ValueError("a positive ordered B--E bracket is required")
    process = 1e-5 if arm in "DE" else 1e-4
    recipes = (Recipe(arm, low, strength, process), Recipe(arm, high, strength, process),
               Recipe("A", reference_ridge))
    rows = []
    for recipe in recipes:
        found = [r for r in records if r["status"] == "completed"
                 and r["metadata"]["cohort"] == "development"
                 and r["metadata"]["replica"] == member
                 and r["metadata"]["recipe"] == recipe.mapping()]
        if len(found) != 1:
            raise ValueError("exactly one paired completed development observation is required")
        rows.append(found[0])
    if len({r["metadata"]["pairing_asset_hash"] for r in rows}) != 1:
        raise ValueError("bracket endpoints and reference must share their pairing assets")
    fields = ("Havg", "H100")
    bracketed, in_band = [], []
    for key in fields:
        lo, hi, target = (r["summary"][key] for r in rows)
        if not all(math.isfinite(v) for v in (lo, hi, target)):
            raise ValueError("finite paired responses are required")
        if min(lo, hi) <= target <= max(lo, hi):
            bracketed.append(key)
        elif max(abs(lo-target),abs(hi-target)) <= tolerance:
            in_band.append(key)
        else:
            raise ValueError("each statistic must bracket the reference or already lie in its tolerance band")
    if not bracketed:
        raise ValueError("at least one actual observed bracket is required")
    ridge = math.sqrt(low * high)
    return dict(recipe=Recipe(arm, ridge, strength, process).mapping(), member=member,
        bracket=[low, high], reference_ridge=reference_ridge, tolerance=tolerance,
        bracketed_statistics=bracketed, already_in_band_statistics=in_band,
        paired_evidence=[dict(unit_id=r["unit_id"], recipe=r["metadata"]["recipe"],
            replica=member, pairing_asset_hash=r["metadata"]["pairing_asset_hash"],
            **{k:r["summary"][k] for k in fields}) for r in rows],
        reason="paired historical bracket of a binding statistic, with the other statistic bracketed or in band; "
               "includes a candidate incomplete on four members; "
               "geometric midpoint is a proposal only, requiring execution on all four members")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--arm", choices=list("BCDE"), required=True)
    parser.add_argument("--member", type=int, required=True)
    parser.add_argument("--low", type=float, required=True)
    parser.add_argument("--high", type=float, required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--round-index", type=int, required=True)
    args = parser.parse_args(argv)
    cfg = Config.read(args.config)
    if args.round_index not in (0, 1) or args.reference_index not in range(4):
        raise ValueError("review must belong to a declared bounded reference/round")
    store, budget = open_store(cfg, args.root, args.history_root, DATA_ROOT,
                               resume=True, clear_stop=True)
    try:
        if (store.root/"selection.json").exists() or any(
                u["metadata"].get("cohort") in ("fresh", "stationary")
                for u in store.iter_units(completed_only=False)):
            raise ValueError("development review must precede confirmation and freeze")
        records, _ = read_exported_records(store.root, budget.check)
        reference = (10., 1., .1, 100.)[args.reference_index]
        review = paired_bracket(records, arm=args.arm, member=args.member, low=args.low,
            high=args.high, strength=cfg.fd_strength, reference_ridge=reference,
            tolerance=cfg.retention_tolerance)
        counts = Counter()
        for unit in store.iter_units(completed_only=True):
            if unit["metadata"].get("name", "").startswith("refinement-"):
                old = read_json(unit["path"]/"decision.json")
                counts.update(r["arm"] for r in old["recipes"])
        if counts[args.arm] >= cfg.refinements_per_arm:
            raise ValueError("per-arm refinement allowance is exhausted")
        name = f"refinement-{args.reference_index}-{args.round_index}"
        uid = "decision-"+name
        if store.is_complete(uid):
            raise ValueError("review cannot replace a completed proposal")
        if args.round_index and not store.is_complete(
                f"decision-refinement-{args.reference_index}-{args.round_index-1}"):
            raise ValueError("prior admitted round must be sealed first")
        durations=[r["summary"]["online_wall_seconds"] for r in records
                   if r["metadata"]["recipe"]["arm"] == args.arm]
        needed=4*max(durations)*1.25+(.35*3600 if not cfg.smoke else 15)
        if needed>budget.remaining:
            raise ValueError("the complete reviewed round cannot fit with closeout reserve")
        source=Path(__file__).read_bytes()
        payload=dict(reference_ridge=reference, recipes=[review["recipe"]],
            prior_allowances={a:counts[a] for a in "BCDE"}, evidence=[review],
            reason=review["reason"], review_source_sha256=hashlib.sha256(source).hexdigest(),
            review_source=source.decode(), review_is_development_only=True)
        store.begin(uid, dict(kind="decision", name=name))
        store.write_json(uid,"decision.json",payload)
        store.complete(uid)
        print(f"Sealed {name}: {args.arm} ridge={review['recipe']['ridge']:.12g}; all-four execution required.",flush=True)
    finally:
        budget.close()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
