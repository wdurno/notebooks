"""Seal actual smoke/recovery/fidelity evidence before production calibration."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import utc_now
from mnist_experiment.fisher_memory_regression.config import content_hash
from mnist_experiment.fisher_memory_regression.runtime import read_json
from .config import FAMILY, ROOT, REPO
from .runtime import sources, atomic_json, charge_preparation


def xml_result(path, minimum):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag=="testsuite" else list(tree.findall("testsuite"))
    count = sum(int(s.get("tests", 0)) for s in suites)
    bad = sum(int(s.get(f, 0)) for s in suites for f in ("errors", "failures", "skipped"))
    seconds = sum(float(s.get("time", 0.)) for s in suites)
    if count < minimum or bad:
        raise ValueError("required test evidence did not pass: "+str(path))
    return dict(path=str(path), tests=count, seconds=seconds, sha256=artifacts._file_hash(path))


def smoke_result(root, source_hash):
    identity = artifacts._identity(root)
    if identity["identity"]["source_hash"] != source_hash or identity["identity"]["role"] != "smoke":
        raise ValueError("smoke sources/role differ")
    units = list(artifacts.iter_units(root, full=True))
    if any(u["status"]!="completed" for u in units) or read_json(root/"status.json")["status"]!="completed":
        raise ValueError("smoke did not finish")
    paths = [u for u in units if u["metadata"].get("kind")=="grid_registration_trajectory"]
    if len(paths)!=6:
        raise ValueError("six actual acceptance/fresh smoke paths required")
    for r in range(2):
        paired = [u for u in paths if u["metadata"].get("namespace")=="fresh" and u["metadata"].get("replica")==r]
        if {u["metadata"].get("policy") for u in paired}!={"R", "K"} or len({u["metadata"]["plan_hash"] for u in paired})!=1:
            raise ValueError("unpaired smoke")
    fidelity = read_json(root/"units/decision-fidelity/result.json")
    if fidelity["status"]!="passed":
        raise ValueError("smoke forced-pi fidelity did not pass")
    return dict(root=str(root), units=len(units), paths=len(paths), identity_hash=identity["identity_hash"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-root", type=Path, required=True)
    p.add_argument("--cuda-root", type=Path, required=True)
    p.add_argument("--recovery-xml", type=Path, default=FAMILY/"phase9e_recovery.xml")
    p.add_argument("--units-xml", type=Path, default=FAMILY/"phase9e_units.xml")
    p.add_argument("--production-root", type=Path, default=ROOT)
    args = p.parse_args(argv)
    code_hash = content_hash(sources())
    evidence = dict(status="passed", source_hash=code_hash,
        units=xml_result(args.units_xml, 144), recovery=xml_result(args.recovery_xml, 4),
        cpu=smoke_result(args.cpu_root, code_hash), cuda=smoke_result(args.cuda_root, code_hash))
    if artifacts._identity(args.production_root)["identity"]["source_hash"] != code_hash:
        raise ValueError("full-length fidelity source differs")
    fidelity_unit = artifacts._unit_record(args.production_root, "decision-fidelity",
        artifacts._identity(args.production_root)["identity_hash"], {}, lambda: None, verify=True)
    fidelity = read_json(fidelity_unit["path"]/"result.json")
    if fidelity["status"]!="passed" or fidelity.get("smoke_only", True):
        raise ValueError("full-length forced-pi fidelity required")
    evidence.update(full_fidelity=fidelity, sealed_at=utc_now())
    target = FAMILY/"phase9e_acceptance"/(code_hash+".json")
    if target.exists():
        old = read_json(target)
        if {k:v for k,v in old.items() if k!="sealed_at"}!={k:v for k,v in evidence.items() if k!="sealed_at"}:
            raise artifacts.ImmutableError("source-keyed acceptance evidence differs")
        evidence = old
    else:
        # Recovery roots live in pytest's /tmp; count their complete suite once.
        charge_preparation(evidence["units"]["seconds"]+evidence["recovery"]["seconds"],
                           "Phase 9E unit and actual interruption acceptance: "+code_hash)
        atomic_json(target, evidence)
    atomic_json(FAMILY/"phase9e_acceptance.json", evidence)
    print(json.dumps(dict(status="passed", source_hash=code_hash, acceptance=str(target))))


if __name__=="__main__":
    main()
