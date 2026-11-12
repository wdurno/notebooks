"""Resume Phase 9E without repeating trajectories or losing its final bracket.

The frozen numerical/observer source component keeps its original hash. This
wrapper is an additional explicitly hashed orchestration component in the new
store identity and provenance. Completed parent units are read-only references;
incomplete accepted checkpoints are adopted under the new envelope identity.
"""

import argparse
import fcntl
from pathlib import Path
import sys

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory_regression.runtime import read_json, file_digest
from mnist_experiment.fisher_memory_regression.config import content_hash
from mnist_experiment.combined_memory.reproduction.audit import load_checkpoint, compare_array
from mnist_experiment.combined_memory.adaptive_f1 import run as original
from mnist_experiment.combined_memory.adaptive_f1 import calibration as cal
from mnist_experiment.combined_memory.adaptive_f1.config import FAMILY, ROOT as PARENT, REPO
from mnist_experiment.combined_memory.adaptive_f1.runtime import sources

ROOT = FAMILY / "adaptive_f1_v2"
REVISION = "plan18-9e-calibration-repair-v2"
OriginalStore = artifacts.Store
OriginalStudy = original.Study
EXCLUDE = {"decision-proposal-repair", "decision-design", "decision-analysis", "decision-integrity"}


def patch_sources():
    return {str(p.relative_to(REPO)): file_digest(p) for p in Path(__file__).parent.glob("*.py")}


def repair_proposal(records):
    """Use every measured first-pair bracket; validate the proposal on all four.

    A candidate already complete on all four cannot consume the final repair
    slot. A previously probed but unfinished candidate may be completed instead.
    """
    rows = cal.candidates(records)
    done = {r["ridge"] for r in rows if r["policy"]=="K" and r["complete"]}
    first = cal.selection(records, members=(0, 1), require_match=False)
    if first["passed"] and first["ridge"] not in done:
        return dict(ridge=first["ridge"], reason="complete the passing first-pair candidate",
                    validation_members=list(range(4)))
    value = cal.propose(records, members=(0, 1))
    if value and value["ridge"] not in done:
        return dict(value, reason="final four-member repair from all measured first-pair brackets: "+value["reason"],
                    validation_members=list(range(4)))
    return None


class Study(OriginalStudy):
    _sealing_repair = False

    def fidelity(self):
        if not self.args.smoke or self.decision("fidelity"):
            return super().fidelity()
        # The tiny R reference may live in the parent, not this store's root.
        runner = self.runner("acceptance")
        runner.forced_pi = .025
        reference = self.job("acceptance", 0, original.Policy("R"))
        current = self.job("acceptance", 0, original.Policy("K"))
        runner.forced_pi = None
        checks = []
        for field in ("base_mse", "novel_mse", "base_nll", "novel_nll"):
            checks.append(compare_array([m[field] for m in reference["metrics"]],
                                        [m[field] for m in current["metrics"]], field))
        old = self.store.load_checkpoint(reference["physical_unit"])
        new = self.store.load_checkpoint(current["physical_unit"])
        for field in ("theta", "anchor_L", "ltm_indices"):
            checks.append(compare_array(old["learner"][field], new["learner"][field], field))
        for field in ("factor", "residual_diagonal"):
            checks.append(compare_array(old["learner"]["fisher_L"][field],
                                        new["learner"]["fisher_L"][field], field))
        if old["retirements"]!=new["retirements"] or old["learner"]["replay"]!=new["learner"]["replay"]:
            raise artifacts.ArtifactError("migrated forced-pi FIFO fidelity failed")
        self.seal("fidelity", dict(status="passed", smoke_only=True, checks=checks))

    def decision(self, name):
        old = super().decision(name)
        if (old is not None or name!="proposal-repair" or self._sealing_repair
                or self.stage!="9E.2 final bounded repair"):
            return old
        proposal = repair_proposal(self.development_records())
        reservation = self.reserve(extra=("K",)*4, validation=False)
        if proposal is None or not reservation["passed"]:
            proposal = dict(skipped="no distinct repair or resource reserve", reservation=reservation)
        self._sealing_repair = True
        try:
            return self.seal(name, proposal)
        finally:
            self._sealing_repair = False


class InheritedStore:
    """One new writer with immutable, checksum-verified parent references."""

    def __init__(self, base, parent, revision):
        self.base, self.parent, self.revision = base, Path(parent), revision
        self.checked = {}
        self.parent_identity = artifacts._identity(self.parent)
        if self.parent_identity["identity"]["source_hash"] != base.identity["source_hash"]:
            raise artifacts.ImmutableError("parent numerical component differs")
        for field in ("config_hash", "data_hash", "role", "authorization_sha256"):
            if self.parent_identity["identity"].get(field)!=base.identity.get(field):
                raise artifacts.ImmutableError("parent configuration differs: "+field)
        adopted = []
        for entry in artifacts.iter_units(self.parent, check=base.check):
            unit = entry["unit_id"]
            if entry["status"]=="completed" or self.excluded(unit):
                continue
            if entry["metadata"].get("kind")!="grid_registration_trajectory":
                raise artifacts.ArtifactError("unexpected unfinished parent operation")
            if not (base.unit_path(unit)/"manifest.json").exists():
                saved = load_checkpoint(self.parent, unit)
                if saved is None:
                    raise artifacts.ArtifactError("parent trajectory lacks accepted checkpoint")
                base.begin(unit, entry["metadata"])
                base.save_checkpoint(unit, saved)
                adopted.append(dict(unit=unit, parent_pointer=read_json(entry["path"]/"latest.json"),
                                    accepted_segment=saved["next_segment"], novel_count=saved["n"]))
        if not (base.root/"scheduler_migration.json").exists():
            base.write_json(None, "scheduler_migration.json", dict(revision=revision,
                parent=str(self.parent.relative_to(REPO)) if self.parent.is_relative_to(REPO) else str(self.parent),
                parent_identity=self.parent_identity["identity_hash"], adopted=adopted,
                note="Original numerical state, seeds, proposal allowance and cohort preserved"))

    @staticmethod
    def excluded(unit):
        return unit in EXCLUDE or unit.startswith("decision-bootstrap-")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def external(self, unit):
        if not unit or self.excluded(unit) or (self.base.unit_path(unit)/"manifest.json").exists():
            return False
        if not (self.parent/"units"/unit/"COMPLETED").exists():
            return False
        if unit not in self.checked:
            self.checked[unit] = artifacts._unit_record(self.parent, unit,
                self.parent_identity["identity_hash"], {}, self.base.check, True)
        return True

    def unit_path(self, unit):
        return self.parent/"units"/unit if self.external(unit) else self.base.unit_path(unit)

    def is_complete(self, unit, **kwargs):
        return self.external(unit) or self.base.is_complete(unit, **kwargs)

    def load_checkpoint(self, unit):
        return load_checkpoint(self.parent, unit) if self.external(unit) else self.base.load_checkpoint(unit)

    def begin(self, unit, metadata):
        if self.external(unit):
            raise artifacts.ImmutableError("cannot begin a referenced completed parent unit")
        return self.base.begin(unit, metadata)

    def save_checkpoint(self, unit, payload, **kwargs):
        if self.external(unit):
            raise artifacts.ImmutableError("cannot mutate a referenced parent checkpoint")
        return self.base.save_checkpoint(unit, payload, **kwargs)

    def write_json(self, unit, name, value, **kwargs):
        if self.external(unit):
            raise artifacts.ImmutableError("cannot mutate a referenced parent artifact")
        if unit is None and name=="provenance.json":
            value = dict(value, scheduler_revision=self.revision)
        return self.base.write_json(unit, name, value, **kwargs)

    def iter_units(self, **kwargs):
        names = {p.name for p in (self.base.root/"units").iterdir()}
        yield from self.base.iter_units(**kwargs)
        for entry in artifacts.iter_units(self.parent, completed_only=True, check=self.base.check):
            if entry["unit_id"] not in names and not self.excluded(entry["unit_id"]):
                yield entry

    def __enter__(self):
        self.base.__enter__()
        return self

    def __exit__(self, kind, value, trace):
        try:
            if patch_sources()!=self.revision["sources"]:
                raise artifacts.ImmutableError("scheduler source changed during execution")
            for unit in self.checked:
                artifacts._unit_record(self.parent, unit, self.parent_identity["identity_hash"], {}, lambda: None, True)
        finally:
            self.base.__exit__(kind, value, trace)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inherit-root", type=Path, default=PARENT)
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not any(v=="--root" or v.startswith("--root=") for v in remaining):
        remaining += ["--root", str(ROOT)]
    if any(stage in remaining for stage in ("status", "stop", "report")):
        return original.main(remaining)
    parent = args.inherit_root.resolve()
    handle = (parent/".run.lock").open("a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise artifacts.ArtifactError("stop the parent worker before migration") from error
    revision = dict(name=REVISION, sources=patch_sources(), parent_identity=artifacts._identity(parent)["identity_hash"])
    combined_sources = {**sources(), **revision["sources"]}
    revision["execution_source_hash"] = content_hash(combined_sources)

    def store_factory(root, identity, **kwargs):
        if Path(root).resolve()==parent:
            raise ValueError("scheduler revision needs a new immutable root")
        identity = dict(identity, scheduler_revision=revision)
        base = OriginalStore(root, identity, **kwargs)
        try:
            return InheritedStore(base, parent, revision)
        except BaseException:
            base.close()
            raise

    prior_store, prior_study = artifacts.Store, original.Study
    try:
        artifacts.Store, original.Study = store_factory, Study
        return original.main(remaining)
    finally:
        artifacts.Store, original.Study = prior_store, prior_study
        handle.close()


if __name__=="__main__":
    raise SystemExit(main())
