"""Versioned failure/serialization guard around the frozen Fix 2 campaign.

This wrapper changes no likelihood, gradient, metric, proposal or random draw.
Its separate source seal records candidate-level finite-output failures and
represents undefined scalar diagnostics as null plus explicit field metadata.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import RunStore


VERSION = "plan19-solver-supervisor-v1"


def sanitize_diagnostics(value):
    unavailable = []

    def visit(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            unavailable.append({"path": path, "value": "nan" if math.isnan(item)
                else "positive_infinity" if item > 0 else "negative_infinity"})
            return None
        if isinstance(item, dict):
            return {key: visit(child, f"{path}.{key}" if path else str(key)) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [visit(child, f"{path}[{i}]") for i, child in enumerate(item)]
        return item

    return visit(value, ""), unavailable


def guard_identity():
    return {"version": VERSION, "source_sha256": sha256_file(Path(__file__)),
        "purpose": "candidate_failure_isolation_and_finite_json_diagnostics",
        "changes_numerical_updates": False}


class GuardedRunStore(RunStore):
    def complete(self, summary=None):
        safe, unavailable = sanitize_diagnostics(dict(summary or {}))
        safe["supervisor"] = guard_identity()
        if unavailable:
            safe["unavailable_diagnostic_fields"] = unavailable
            safe["diagnostic_nonfinite_values_are_zero"] = False
        return super().complete(safe)


def guarded_recorded(base):
    class GuardedRecorded(base):
        def run(self):
            # An already committed numerical failure has no remaining fit work.
            if self.s.get("status") == "numerical_failure":
                return self.summary()
            try:
                return super().run()
            except (FloatingPointError, ValueError) as exc:
                if not isinstance(exc, FloatingPointError) and "finite" not in str(exc).lower():
                    raise
                committed = self.store.load_checkpoint(trusted=True)
                if committed is None:
                    raise
                self.s = copy.deepcopy(committed)
                self.s.update(status="numerical_failure", reason=str(exc),
                              failure_boundary="last_committed_event_block")
                self.save()
                return self.summary()
    return GuardedRecorded


def main():
    from numerical_experiment.picar_solver_repair import campaign, recorded
    from numerical_experiment.picar_solver_repair.config import DEFAULT_OUTPUT, VERSION as CAMPAIGN_VERSION

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args, _ = parser.parse_known_args()
    root = args.output.resolve()
    marker = root / "supervisor-guard-v1"
    settings = {"schema_version": 1, "version": CAMPAIGN_VERSION, "supervisor": guard_identity()}
    with GuardedRunStore(marker, settings, resume=True, checkpoint_policy="journal-v1") as store:
        if not store.completed:
            store._write_root("supervisor.py", Path(__file__).read_bytes())
            store.checkpoint({"stage": "done"})
            store.complete({"schema_version": 1, "kind": "solver_supervisor", "phase": 3,
                "version": CAMPAIGN_VERSION, "smoke_only": True, "development_only": True,
                "unit_path": str(marker), "scientific_sources_unchanged": True})
    original_store, original_recorded = campaign.RunStore, recorded.RecordedRun
    try:
        campaign.RunStore = GuardedRunStore
        recorded.RecordedRun = guarded_recorded(original_recorded)
        campaign.main()
    finally:
        campaign.RunStore, recorded.RecordedRun = original_store, original_recorded


if __name__ == "__main__":
    main()
