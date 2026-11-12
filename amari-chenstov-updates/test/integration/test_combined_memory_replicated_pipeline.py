"""Actual signal and split-group recovery for the approved Phase 9 pipeline."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.combined_memory.replicated_combination.config import REPO,DATA

pytestmark=pytest.mark.skipif(not (DATA/'MNIST/raw/train-images-idx3-ubyte').exists(),reason='cached MNIST required')


def command(root):
    return [sys.executable,'-u','-m','mnist_experiment.combined_memory.replicated_combination.run',
            '--root',str(root),'--notebook',str(root/'results.ipynb'),'--smoke','--device','cpu',
            '--replicas','2','--resume']


def execute(root,extra=(),expected=0):
    result=subprocess.run(command(root)+list(extra),cwd=REPO,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,timeout=600)
    assert result.returncode==expected,result.stdout


def payload(root,policy):
    unit=next((root/'units').glob(f'fresh-{policy}--*0000-*'))
    pointer=json.loads((unit/'latest.json').read_text())['current']
    return torch.load(unit/pointer['file'],map_location='cpu',weights_only=False)['payload']


def seals(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'units').rglob('*') if p.is_file() and (p.parent/'COMPLETED').exists()}


def compare(root,direct):
    for policy in ('R','A','B','C','D','E'):
        a,b=payload(root,policy),payload(direct,policy)
        assert torch.equal(a['learner']['theta'],b['learner']['theta']),policy
        assert a['retirements']==b['retirements'] and a['learner']['replay']==b['learner']['replay']
        if policy=='R':
            for field in ('factor','residual_diagonal'):
                assert torch.equal(a['learner']['fisher_L'][field],b['learner']['fisher_L'][field])
        else:
            assert torch.equal(a['learner']['fisher_L']['values'],b['learner']['fisher_L']['values'])
            assert a['q_ref']==b['q_ref'] and a['q_archive']==b['q_archive']
            assert not a['pending_scores'] and a['group_pi']==b['group_pi']
            if policy in ('D','E'):
                assert a['controller']['assimilations']==b['controller']['assimilations']
                assert torch.equal(a['controller']['mean'],b['controller']['mean'])
                assert torch.equal(a['controller']['covariance'],b['controller']['covariance'])


def test_real_sigterm_and_split_group_resume_preserve_all_six_conditions(tmp_path):
    root=tmp_path/'signal'
    public=REPO/'mnist_experiment/plan18_results.ipynb'
    old_public=public.read_bytes()
    with (tmp_path/'signal.log').open('w+') as log:
        worker=subprocess.Popen(command(root),cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if list((root/'units').glob('fresh-R--*trajectory*/latest.json')):
                    break
                if worker.poll() is not None:
                    log.seek(0);pytest.fail(log.read())
                time.sleep(.03)
            else:
                pytest.fail('no recoverable signal checkpoint')
            os.killpg(worker.pid,signal.SIGTERM)
            worker.wait(timeout=60)
            log.seek(0)
            assert worker.returncode==2,log.read()
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid,signal.SIGKILL);worker.wait()
    before=seals(root)
    execute(root)
    assert all(seals(root).get(k)==v for k,v in before.items())
    direct=tmp_path/'direct'
    execute(direct)
    compare(root,direct)
    split=tmp_path/'split'
    execute(split,('--smoke-stop-at','split_group'),expected=2)
    pending=payload(split,'D')
    assert pending['pending_scores'] and pending['group_pi'] is not None
    execute(split)
    compare(split,direct)
    assert public.read_bytes()==old_public


@pytest.mark.parametrize('boundary',['pretrain','mid_solve','eviction'])
def test_interruptions_leave_valid_resumable_state(tmp_path,boundary):
    root=tmp_path/boundary
    execute(root,('--smoke-stop-at',boundary),expected=2)
    assert not (root/'study_completed.json').exists()
    execute(root)
    assert (root/'study_completed.json').exists()
