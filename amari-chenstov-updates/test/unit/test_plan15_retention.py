"""Fast synthetic Phase 8.1 checks; no MNIST, model inference or GPU."""

from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from mnist_experiment.fisher_memory_retention import PROTOCOL
from mnist_experiment.fisher_memory_retention.config import Config
from mnist_experiment.fisher_memory_retention import design, analysis, report
from mnist_experiment.fisher_memory_regression.config import Recipe, METHODS
from mnist_experiment.fisher_memory_regression.analysis import summarize_trajectory
from mnist_experiment.fisher_memory_regression import report as historical_report


def row(method,*,ridge=10.,strength=None,h=1.2,a=100.,replica=0,partition="development",b=4):
    cfg=Config()
    recipe=Recipe.from_config(cfg,method,ridge=ridge,
        strength=(2. if method=="fd_replay_ewc" else 0.) if strength is None else strength,replay_count=b)
    metrics=[dict(n=n,observations=10*n,learning_seconds=.2*n,
        novel_mse=a if n else 200.,base_mse=(3+h)**2 if n else 9.) for n in range(101)]
    return dict(schema_version=1,protocol=PROTOCOL,family="development",metrics=metrics,
        pairing_id=f"plan-{replica}",plan_hash=f"plan-{replica}",
        recipe_key=recipe.key,recipe=recipe.mapping(),method=method,replica=replica,
        partition=partition,novel_auc=a,base_auc=(3+h)**2,frozen_base_mse=9.,
        frozen_novel_mse=200.,observations=1000,learning_seconds=20.,active_seconds=25.,
        valid=True,target=100,peak_vram_bytes=1000)


def rows_for(method,**kwargs):
    return [row(method,replica=r,**kwargs) for r in range(4)]


def test_config_grid_scope_and_seeds_are_separate_from_phase8():
    cfg=Config();cfg.validate()
    recipes=design.initial_grid(cfg)
    assert len(recipes)==36 and len({r.key for r in recipes})==36
    assert {m:sum(r.method==m for r in recipes) for m in METHODS}==dict(
        fd_replay_ewc=24,replay_ewc=8,ewc_only=4)
    assert cfg.seed_for("development",0,"stream")!=cfg.seed_for("confirmation",0,"stream")
    from mnist_experiment.fisher_memory_regression.config import Config as OldConfig
    assert cfg.seed_for("development",0,"stream")!=OldConfig().seed_for("development",0,"stream")
    assert Config.smoke_config().smoke
    Config.smoke_config().validate()
    with pytest.raises(ValueError): replace(cfg,matching_tolerance=2.).validate()
    with pytest.raises(ValueError): replace(cfg,refinement_settings=21).validate()


def test_table_uses_mean_of_replica_roots_and_complete_sets():
    cfg=Config();records=rows_for("replay_ewc")
    records[0]["base_auc"]=100.
    for m in records[0]["metrics"][1:]: m["base_mse"]=100.
    result=design.table(records,cfg)[0]
    expected=np.mean([math.sqrt(r["base_auc"])-3 for r in records])
    assert result["base_harm"]==pytest.approx(expected)
    assert expected!=pytest.approx(math.sqrt(np.mean([r["base_auc"] for r in records]))-3)
    assert design.table(records[:3],cfg)==[]
    assert design.table(records[:3],cfg,partial=True)[0]["count"]==3
    with pytest.raises(ValueError,match="confirmation"):
        design.table([{**r,"partition":"confirmation"} for r in records],cfg)
    with pytest.raises(ValueError): design.table(records+[records[0]],cfg)


def test_ceiling_and_matching_band_are_not_synonyms():
    cfg=Config()
    measured=design.table(rows_for("replay_ewc",h=.2)+rows_for("replay_ewc",h=1.2,ridge=3.),cfg)
    below=next(r for r in measured if r["recipe"]["ridge"]==10.)
    matched=next(r for r in measured if r["recipe"]["ridge"]==3.)
    assert below["ceiling_feasible"] and not below["in_band"]
    assert matched["in_band"]


def test_final_selection_preserves_joint_improvement_and_unmatched_controls():
    cfg=Config()
    records=[]
    for method in METHODS:
        records+=rows_for(method,h=1.2,a=100.,ridge=10.)
        records+=rows_for(method,h=.8,a=90.,ridge=30.)
    measured=design.table(records,cfg)
    result=design.select(measured,cfg)
    assert all(v["selected"]["recipe"]["ridge"]==30 for v in result.values())
    assert all("joint-improvement" in v["reason"] for v in result.values())
    for record in records:
        record["learning_seconds"]=1000.
        for m in record["metrics"]: m["learning_seconds"]=m["n"]*10.
    diagnostic=design.select(design.table(records,cfg),cfg)
    assert all("resource-infeasible" in v["reason"] for v in diagnostic.values())


def test_shortlist_is_deterministic_bounded_and_includes_best_below_band_alternative():
    cfg=Config();records=[]
    for method in METHODS:
        records+=rows_for(method,h=1.2,a=100.,ridge=10.)
        records+=rows_for(method,h=.8,a=90.,ridge=30.)
        records+=rows_for(method,h=2.,a=50.,ridge=1.)
    measured=design.table(records,cfg)
    choices=design.shortlist(measured,cfg)
    assert choices==design.shortlist(list(reversed(measured)),cfg)
    for values in choices.values(): assert [v["ridge"] for v in values]==[10.,30.]


def test_refinement_brackets_observed_curve_and_does_not_repeat_sealed_points():
    cfg=Config();records=[]
    for method in METHODS:
        records+=rows_for(method,ridge=3.,h=2.)
        records+=rows_for(method,ridge=30.,h=.5)
    measured=design.table(records,cfg)
    proposal=design.refinement_proposals(measured,cfg,[])[0]
    assert proposal["kind"]=="midpoint"
    assert proposal["recipe"]["ridge"]==pytest.approx(math.sqrt(90.))
    assert design.refinement_proposals(list(reversed(measured)),cfg,[])[0]==proposal
    new=Recipe.from_mapping(proposal["recipe"])
    more=rows_for(new.method,ridge=new.ridge,strength=new.strength,h=1.2)
    after=design.refinement_proposals(design.table(records+more,cfg),cfg,[proposal])
    assert new.key not in {r["recipe_key"] for r in after}


def cohort():
    cfg=Config.smoke_config();records=[];recipes=[]
    for method in METHODS:
        recipe=Recipe.from_config(cfg,method,replay_count=1,strength=.5 if method=="fd_replay_ewc" else 0.)
        recipes.append({**recipe.mapping(),"recipe_key":recipe.key})
        for replica in range(3):
            predictions=[100.,60.,40.,20.] if method=="fd_replay_ewc" else [100.,70.,50.,30.]
            metrics=[dict(n=n,observations=n*4,learning_seconds=float(n),
                novel_mse=v+replica,base_mse=9.+.2*n+replica) for n,v in enumerate(predictions)]
            s=summarize_trajectory(metrics,N=3)
            records.append(dict({**s,"protocol":PROTOCOL},partition="confirmation",recipe=recipe.mapping(),
                recipe_key=recipe.key,replica=replica,method=method,metrics=metrics,pairing_id=f"pair-{replica}",
                peak_vram_bytes=1000,evaluation_seconds=1.,fit_seconds=2.,eviction_seconds=1.))
    manifest=dict(protocol=PROTOCOL,sealed=True,recipes=recipes,replicas=[0,1,2],N=3,
        manifest_hash="cohort",delta=1.3342351463,epsilon=.2,learning_time_limit=1.,memory_ceiling_bytes=2000)
    return records,manifest


def test_analysis_uses_full_paired_cohort_and_separate_retention_claims():
    records,manifest=cohort()
    result=analysis.analyze(records,manifest,seed=12,draws=30)
    primary=result["comparisons"]["fd_replay_ewc_vs_replay_ewc"]
    assert primary["acquisition_gain"]["mean"]==pytest.approx(10.)
    assert primary["retention_equivalence_supported"]
    assert result["methods"]["fd_replay_ewc"]["retention"]["ceiling_supported"]
    assert result==analysis.analyze(records,manifest,seed=12,draws=30)
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError,match="incomplete"): analysis.analyze(records[:-1],manifest,seed=1,draws=10)
    with pytest.raises(ValueError,match="duplicate"): analysis.analyze(records+[records[0]],manifest,seed=1,draws=10)
    for mutation in (lambda r:r.update(partition="development"),lambda r:r.update(pairing_id="wrong"),
                     lambda r:r.update(valid=False),lambda r:r.update(novel_auc=999)):
        bad=deepcopy(records);mutation(bad[0])
        with pytest.raises(ValueError): analysis.analyze(bad,manifest,seed=1,draws=10)


def test_analysis_bootstrap_is_stop_aware_and_does_not_infer_equivalence_from_nonsignificance():
    records,manifest=cohort()
    records[0]["metrics"][-1]["base_mse"]+=8
    records[0]["base_auc"]=np.mean([m["base_mse"] for m in records[0]["metrics"]][1:])
    result=analysis.analyze(records,manifest,seed=1,draws=100)
    assert not result["comparisons"]["fd_replay_ewc_vs_replay_ewc"]["retention_equivalence_supported"]
    def stop(): raise InterruptedError("stop")
    with pytest.raises(InterruptedError): analysis.analyze(records,manifest,seed=1,draws=10,check=stop)


def test_report_preserves_archival_guard_and_historical_cells(tmp_path):
    from contextlib import nullcontext
    class Store:
        reserve=lambda self,n:nullcontext()
        check=lambda self:None
    cfg=Config()
    previous=dict(nbformat=4,nbformat_minor=5,metadata={historical_report.PRESENTATION:{"role":"historical_link"}},
        cells=[dict(cell_type="markdown",id="old",source=["# Old content\n"],metadata={}),
            dict(cell_type="markdown",id="plan15-retention-draft",source="Draft",metadata={})])
    path=tmp_path/"results.ipynb";path.write_text(json.dumps(previous))
    snapshot=dict(config=cfg.mapping(),progress=dict(status="running",stage="8.1B"),development=[])
    report.publish(snapshot,path,Store())
    actual=json.loads(path.read_bytes())
    assert actual["cells"][0]==previous["cells"][0]
    assert not any(c["id"]=="plan15-retention-draft" for c in actual["cells"])
    assert "not demonstrated retention equivalence" in "".join("".join(c["source"]) for c in actual["cells"])
    actual["metadata"][historical_report.PRESENTATION]["role"]="frozen_archive"
    path.write_text(json.dumps(actual));before=path.read_bytes()
    with pytest.raises(ValueError,match="frozen"): report.publish(snapshot,path,Store())
    assert path.read_bytes()==before


def test_final_report_explains_total_arrivals_and_comparison_direction(monkeypatch):
    records,manifest=cohort()
    result=analysis.analyze(records,manifest,seed=12,draws=30)
    figures=[]
    def capture(name,figure,caption):
        figures.append((name,figure,caption))
        return report.cell(name,caption)
    monkeypatch.setattr(report,"image",capture)
    cells=report.build_cells(dict(config=Config().mapping(),
        progress=dict(status="completed",stage="8.1D"),analysis=result))
    confirmation=next(caption for name,_,caption in figures if name=="confirmation")
    assert "12.0 mean total arriving images" in confirmation
    assert "excluding pretraining and calibration" in confirmation
    effects=next(figure for name,figure,_ in figures if name=="effects")
    assert all(" vs " in tick.get_text() for tick in effects.axes[0].get_yticklabels())
    assert "positive favors first" in effects.axes[0].get_title()
    assert any("Retention equivalence is supported" in cell["source"] for cell in cells)
    conclusion=next(cell["source"] for cell in cells if cell["id"].endswith("conclusion"))
    assert "Budget-averaged sample efficiency" in conclusion
    assert "not an estimate of actual samples saved" in conclusion
    assert "does not isolate the FD penalty" in conclusion
    assert "equal-compute advantage" in conclusion


def test_supplement_pairs_only_measured_equal_allocation_and_ridge():
    records,manifest=cohort()
    result=analysis.analyze(records,manifest,seed=12,draws=30)
    development=rows_for("fd_replay_ewc",a=90.)+rows_for("replay_ewc",a=100.)
    development+=rows_for("fd_replay_ewc",ridge=3.,a=50.)
    snapshot=dict(config=Config().mapping(),analysis=result)
    actual=report.diagnostic_html(snapshot,development)
    assert "10.000" in actual
    assert actual.count("<td>0.0000</td>")==1
    assert "without interpolation or significance claims" in actual
    assert "not fresh confirmation or matched-retention comparisons" in actual
    assert "not unique images" in actual
    assert "stream_ids/stream_novel seed aliases are not" in actual
    bad=deepcopy(development);bad[0]["pairing_id"]="wrong"
    with pytest.raises(ValueError,match="unpaired"):
        report.diagnostic_html(snapshot,bad)
