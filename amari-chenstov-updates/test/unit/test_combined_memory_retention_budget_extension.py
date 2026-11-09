"""The resource extension must not bypass the scientific calibration seal."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from mnist_experiment.combined_memory.retention_match import calibration
from mnist_experiment.combined_memory.retention_match.config import Config
from mnist_experiment.combined_memory.retention_match.tools import continue_budget as extension
from mnist_experiment.fisher_memory.artifacts import ArtifactError
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import ResourceLimit


def fixture():
    cfg=Config.smoke_config()
    records=[]
    recipes=calibration.initial_recipes(cfg.fd_strength)
    for a,r in recipes.items():
        for m in range(4):
            records.append(dict(status="completed",metadata=dict(kind="trajectory",cohort="development",
                replica=m,recipe=r.mapping()), summary=dict(Havg=2.,H100=4.,novel_auc=8.,learning_seconds=1.)))
    chosen=calibration.select_tuple(records,strength=cfg.fd_strength,tolerance=cfg.retention_tolerance)
    selected=dict(chosen,frozen=True,calibration_status="passed",source_identity="identity",
        main_replicas=cfg.main_replicas,stationary_replicas=cfg.stationary_replicas)
    return cfg,records,selected


@pytest.mark.parametrize("defect", [None,"failed","missing","different_evidence","allocation"])
def test_extension_requires_the_actual_passing_seal_and_identical_allocation(monkeypatch,defect):
    cfg,records,selected=fixture()
    if defect=="failed": selected["calibration_status"]="failed"
    if defect=="missing": records.pop()
    if defect=="different_evidence": records[-1]["summary"]["Havg"]=4.
    if defect=="allocation": selected["main_replicas"]+=2
    monkeypatch.setattr(extension.artifacts,"_identity",lambda root:dict(identity_hash="identity"))
    monkeypatch.setattr(extension,"checked_scalar",lambda *a:(selected,"digest"))
    monkeypatch.setattr(extension,"read_exported_records",lambda *a:(records,{}))
    if defect is None:
        before=deepcopy(selected)
        assert extension.require_calibrated("unused",cfg)==(selected,"digest")
        assert selected==before
    else:
        with pytest.raises((ArtifactError,ValueError)):
            extension.require_calibrated("unused",cfg)


def test_extension_budget_resumes_above_twenty_hours_but_stops_at_the_approved_ceiling(monkeypatch,tmp_path):
    control=SimpleNamespace(active_seconds=20.25*3600)
    store=SimpleNamespace(root=tmp_path,control=control,check=lambda:None)
    monkeypatch.setattr(extension.ExtendedBudget,"pulse",lambda self,**kw:None)
    budget=extension.ExtendedBudget(store,other_seconds=0.,allocation_seconds=21.5*3600)
    budget.pulse=lambda **kw:None
    budget.check()
    assert budget.remaining==pytest.approx(1.25*3600)
    control.active_seconds=21.5*3600
    with pytest.raises(ResourceLimit,match="21.5-hour"):
        budget.check()


@pytest.mark.parametrize("seconds", [20*3600,22*3600,float("nan")])
def test_extension_budget_cannot_silently_accept_a_different_ceiling(seconds):
    with pytest.raises(ValueError,match="explicitly approved"):
        extension.ExtendedBudget(None,allocation_seconds=seconds)


@pytest.mark.parametrize("hours,note", [(20.,"approved"),(22.,"approved"),(21.5,"")])
def test_an_unapproved_resource_ceiling_is_rejected_before_opening_any_store(hours,note):
    with pytest.raises(ValueError,match="explicit 21.5-hour"):
        extension.open_extended_store(Config.smoke_config(),"unused","unused",
            approved_hours=hours,approval_note=note)


def test_real_store_continuation_preserves_scientific_units_and_resumes_its_approval(monkeypatch,tmp_path):
    cfg,records,selected=fixture()
    family=tmp_path/"family"
    root=family/"study"
    identity=dict(protocol=cfg.protocol,allocation_family="plan18",config_hash=cfg.scientific_hash,
                  git_commit="test",git_dirty_at_creation=False)
    with Store(root,identity,max_bytes=cfg.max_bytes,free_reserve_bytes=0) as store:
        for n,record in enumerate(records):
            uid=f"path-development-{n:03}"
            store.begin(uid,record["metadata"])
            store.write_json(uid,"summary.json",record["summary"])
            store.write_json(uid,"metrics.json",[])
            store.write_json(uid,"diagnostics.json",{})
            store.complete(uid)
        selected["source_identity"]=store.identity_hash
        store.begin("decision-selection",dict(kind="decision",name="selection"))
        store.write_json("decision-selection","decision.json",selected)
        store.complete("decision-selection")
        store.write_json(None,"timing_lease.json",dict(running=False,charged_until=20.25*3600))
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/"units").rglob("*") if p.is_file()}
    monkeypatch.setattr(extension.original,"FAMILY",family)
    monkeypatch.setattr(extension.original,"setup",lambda cfg:None)
    monkeypatch.setattr(extension,"provenance",lambda *a:dict(identity))
    for _ in range(2):
        store,budget=extension.open_extended_store(cfg,root,"unused",approved_hours=21.5,
                                                 approval_note="Explicit test authorization")
        try:
            assert 20.25*3600 <= budget.used < 20.26*3600
            assert budget.remaining <= 1.25*3600
            assert store.is_complete("decision-resource-extension-21p5")
        finally:
            budget.close()
            store.close()
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
               for name,digest in before.items())
    approval=json.loads((root/"units/decision-resource-extension-21p5/decision.json").read_text())
    assert approval["approved_ceiling_hours"]==21.5
    assert approval["main_replicas"]==cfg.main_replicas
