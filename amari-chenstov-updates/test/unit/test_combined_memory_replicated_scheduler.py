from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from mnist_experiment.combined_memory.replicated_combination.run import Study,main
from mnist_experiment.combined_memory.replicated_combination.config import HISTORY


def test_failed_retention_calibration_never_launches_fresh_paths():
    study=object.__new__(Study)
    study.args=SimpleNamespace(smoke=False,stage='all',replicas=8,device='cpu')
    study.history=json.loads((HISTORY/'resolved_config.json').read_text())
    study.store=SimpleNamespace(identity=dict(source_hash='fixture-source'))
    study.seal=lambda *args:None
    study.runner=lambda *args:SimpleNamespace(cfg=SimpleNamespace(novel_count=100))
    study.progress=lambda **kw:None
    study.fidelity=lambda:None
    study.calibrate=lambda:dict(status='calibration_infeasible')
    study.job=lambda *a,**kw:pytest.fail('unmatched calibration launched confirmation')
    study.execute()
    assert study.state=='calibration_infeasible'
    assert study.stage=='9B calibration closeout'


def test_smoke_cannot_write_production_notebook_or_default_root(tmp_path):
    with pytest.raises(ValueError,match='separate root and notebook'):
        main(['--smoke','--device','cpu','--replicas','2','--root',str(tmp_path)])


def test_production_cannot_silently_reduce_replica_count():
    with pytest.raises(ValueError,match='eight fresh paired'):
        main(['--replicas','2'])


def test_no_artificial_failure_injection_in_production():
    with pytest.raises(ValueError,match='smoke-only'):
        main(['--smoke-stop-at','eviction'])
