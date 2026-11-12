"""Seal source-matched CPU/CUDA and actual interruption acceptance evidence."""

import argparse
import fcntl
from pathlib import Path
import xml.etree.ElementTree as ET

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import utc_now
from mnist_experiment.fisher_memory_regression.runtime import read_json,file_digest
from mnist_experiment.fisher_memory_regression.config import content_hash
from .config import FAMILY,POLICIES
from .runtime import sources,atomic_json,charge_preparation
from . import PROTOCOL


def xml_evidence(path,minimum):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['tests']<minimum or any(counts[k] for k in ('failures','errors','skipped')):
        raise ValueError(f'incomplete or failed test acceptance: {path}: {counts}')
    return dict(path=str(path),sha256=file_digest(path),seconds=sum(float(s.get('time',0)) for s in suites),**counts)


def smoke_evidence(path,device,digest):
    identity=artifacts._identity(path)['identity']
    if identity.get('protocol')!=PROTOCOL or identity.get('role')!='smoke' or identity.get('source_hash')!=digest:
        raise ValueError('smoke scientific source identity does not match')
    if read_json(path/'resolved_config.json')['device']!=device:
        raise ValueError('wrong smoke execution device')
    complete=read_json(path/'study_completed.json')
    if complete['status']!='completed' or complete['trajectories']!=12:
        raise ValueError('complete two-block, six-policy smoke required')
    units=list(artifacts.iter_units(path,full=True))
    if any(u['status']!='completed' for u in units):
        raise ValueError('incomplete smoke unit')
    records=[u['metadata'] for u in units if u['metadata'].get('kind')=='grid_registration_trajectory']
    if any({r['policy'] for r in records if r['replica']==i}!=set(POLICIES) for i in (0,1)):
        raise ValueError('smoke paired condition set mismatch')
    return dict(root=str(path),device=device,units=len(units),source_hash=digest,
                completion_sha256=file_digest(path/'study_completed.json'),
                identity_hash=artifacts._identity(path)['identity_hash'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-root',type=Path,default=FAMILY/'phase9_smoke_cpu_v3')
    p.add_argument('--cuda-root',type=Path,default=FAMILY/'phase9_smoke_cuda_v1')
    p.add_argument('--unit-xml',type=Path,default=FAMILY/'phase9_unit_tests_frozen.xml')
    p.add_argument('--recovery-xml',type=Path,default=FAMILY/'phase9_recovery_tests_final.xml')
    args=p.parse_args(argv)
    with (FAMILY/'.phase9-acceptance.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        digest=content_hash(sources())
        unit=xml_evidence(args.unit_xml,123)
        recovery=xml_evidence(args.recovery_xml,4)
        cpu=smoke_evidence(args.cpu_root,'cpu',digest)
        cuda=smoke_evidence(args.cuda_root,'cuda',digest)
        description='Frozen recovery acceptance '+recovery['sha256']
        ledger=FAMILY/'phase9_preparation_compute.json'
        charged=read_json(ledger) if ledger.exists() else dict(entries=[])
        if not any(e['description']==description for e in charged['entries']):
            charge_preparation(recovery['seconds']+unit['seconds'],description)
        destination=FAMILY/'phase9_acceptance'/f'{digest}.json'
        if destination.exists():
            value=read_json(destination)
        else:
            payload=dict(protocol=PROTOCOL,status='passed',source_hash=digest,accepted_at=utc_now(),
                         unit=unit,recovery=recovery,cpu=cpu,cuda=cuda)
            value=dict(payload,sha256=content_hash(payload))
            atomic_json(destination,value)
        if value['sha256']!=content_hash({k:v for k,v in value.items() if k!='sha256'}):
            raise ValueError('invalid immutable acceptance evidence')
        atomic_json(FAMILY/'phase9_acceptance.json',value)
        print(f'Phase 9 acceptance passed: {digest}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
