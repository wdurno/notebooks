"""Local inverse-risk mathematics and immutable CPU-only reporting/recovery."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json

import numpy as np
import pytest
import nbformat

from mnist_experiment import plan15_retention_efficiency as efficiency
from mnist_experiment import plan15_retention_efficiency_report as report
from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_retention import report as parent_report
from mnist_experiment.fisher_memory_retention.config import Config


def inputs(replicas=12):
    n = np.arange(101, dtype=float)
    data = {}
    for method in efficiency.METHODS:
        fd = method == "fd_replay_ewc"
        data[method] = {key: [] for key in efficiency.FIELDS}
        for r in range(replicas):
            data[method]["novel_mse"].append(300 + 12*r - (2 + .15*r if fd else 1 + .1*r)*n)
            data[method]["base_mse"].append(9 + .02*n*(1+r/10))
            data[method]["learning_seconds"].append((.4 if fd else .25)*n)
            data[method]["observations"].append((10+r)*n)
    return efficiency.validate_arrays(data)


def test_linear_recovery_and_fractional_cost_at_corresponding_budget():
    arrays = inputs(2)
    means = {m: {f: a.mean(axis=0) for f,a in fs.items()} for m,fs in arrays.items()}
    result = efficiency.match_means(means)
    expected = 100*1.05/2.075
    assert result["matching_budget"] == pytest.approx(expected)
    assert result["saved_novel"] == pytest.approx(100-expected)
    assert result["learning_time_ratio"] == pytest.approx(.4*expected/25)
    assert result["fd_observations"] == pytest.approx(10.5*expected)
    assert result["fd_old_rmse"] == pytest.approx(np.sqrt(9+.021*expected))


def test_late_reversal_uses_last_entry_not_first_dip():
    result = efficiency.crossing([10,8,4,7,4,3],5)
    assert result["downward_crossings"] == pytest.approx([1.75,3+2/3])
    assert result["matching_budget"] == pytest.approx(3+2/3)
    assert result["multiple_crossings"]


@pytest.mark.parametrize("curve,target,status",[
    ([3,4,5],3,"target_already_met_at_initialization"),
    ([10,4,8],5,"target_not_reached_at_observed_endpoint"),
    ([10,0,0],9,"crossing_below_first_post_observation_checkpoint"),
    ([10,8,5],5,"valid"),
    ([10,5,5,5],5,"valid"),
])
def test_crossing_guards_and_exact_plateaus(curve,target,status):
    result = efficiency.crossing(curve,target)
    assert result["status"] == status
    if status == "valid": assert 1 <= result["matching_budget"] <= len(curve)-1


@pytest.mark.parametrize("curve",[[10,float("nan"),0],[10,-1,0],[10,0]])
def test_invalid_predictive_curves_fail_clearly(curve):
    with pytest.raises(ArtifactError): efficiency.crossing(curve,5)


def test_paired_bootstrap_recomputes_target_and_crossing_together():
    arrays = inputs(3)
    first = efficiency.sampled_result(arrays,np.array([0,0,0]))
    last = efficiency.sampled_result(arrays,np.array([2,2,2]))
    assert first["target_mse"] != last["target_mse"]
    assert first["matching_budget"] != last["matching_budget"]
    assert first == efficiency.sampled_result(arrays,np.array([0,0,0]))


def test_failed_draw_is_not_dropped_from_interval():
    arrays = inputs(3)
    valid = efficiency.sampled_result(arrays)
    invalid = efficiency.crossing([10,4,8],5)
    result = efficiency.summarize(arrays,[valid,invalid])
    assert result["bootstrap"]["invalid"] == 1
    assert result["bootstrap"]["ci"] is None
    complete = efficiency.summarize(arrays,[valid,valid])
    assert complete["bootstrap"]["ci"]["saved_novel"] == [valid["saved_novel"]]*2


def test_report_exposes_unresolved_draws_and_crossing_sensitivity():
    arrays = inputs(3)
    valid = efficiency.sampled_result(arrays);valid["multiple_crossings"] = True
    result = efficiency.summarize(arrays,[valid,efficiency.crossing([10,4,8],5)])
    cells = report.build_cells(result,"analysis.json")
    conclusion = cells[-1]["source"]
    assert "1/2 draws" in conclusion
    assert "No unsuccessful draw was discarded" in conclusion
    assert "50.0% of draws" in conclusion
    assert "not H" in conclusion
    assert "not guaranteed future observations saved" in conclusion


def test_report_withholds_in_range_claim_if_point_is_unresolved():
    arrays = inputs(3)
    arrays["fd_replay_ewc"]["novel_mse"] = arrays["replay_ewc"]["novel_mse"]+np.arange(101)
    result = efficiency.summarize(arrays,[efficiency.sampled_result(arrays)])
    cells = report.build_cells(result,"analysis.json")
    assert "No in-range sample-saving estimate" in cells[-1]["source"]
    assert "| Policy |" not in cells[-1]["source"]


def test_validation_rejects_unpaired_exposures_or_initial_panels():
    arrays = inputs(3)
    bad = deepcopy(arrays);bad["replay_ewc"]["observations"][0,-1] += 1
    with pytest.raises(ArtifactError,match="unpaired stream"): efficiency.validate_arrays(bad)
    bad = deepcopy(arrays);bad["replay_ewc"]["novel_mse"][0,0] += 1
    with pytest.raises(ArtifactError,match="unpaired initial"): efficiency.validate_arrays(bad)


def file_hashes(root):
    return {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for u in (root/"units").iterdir() if (u/"COMPLETED").exists()
            for f in u.iterdir() if f.is_file()}


def test_bootstrap_blocks_resume_immutably_and_respond_to_stop(tmp_path):
    root = tmp_path/"analysis";arrays = inputs()
    identity = dict(protocol=efficiency.PROTOCOL)
    kwargs = dict(dependencies={"seals":{}},manifest={"manifest_hash":"test"},
                  established={"comparisons":{"fd_replay_ewc_vs_replay_ewc":{}}})
    with Store(root,identity) as store:
        with pytest.raises(StopRequested): efficiency.execute(store,arrays,**kwargs,max_units=2)
    before = file_hashes(root)
    assert "bootstrap-0000" in " ".join(before)
    def stop(): raise StopRequested("interrupt unfinished block")
    with Store(root,identity,resume=True) as store:
        with pytest.raises(StopRequested): efficiency.execute(store,arrays,**kwargs,check=stop)
    assert file_hashes(root) == before
    with Store(root,identity,resume=True) as store:
        result = efficiency.execute(store,arrays,**kwargs)
    assert before.items() <= file_hashes(root).items()
    complete = file_hashes(root)
    with Store(root,identity,resume=True) as store:
        assert efficiency.execute(store,arrays,**kwargs) == result
    assert file_hashes(root) == complete
    assert result["bootstrap"]["draws"] == 2000
    assert result["bootstrap"]["invalid"] == 0


def test_report_preserves_parent_and_later_cells_and_survives_parent_refresh(tmp_path,monkeypatch):
    arrays = inputs(3)
    result = efficiency.summarize(arrays,[efficiency.sampled_result(arrays)]*3)
    result["parent_manifest_hash"] = "test"
    class FakeStore:
        reserve = lambda self,n:nullcontext()
        check = lambda self:None
        unit_path = lambda self,unit:tmp_path/unit
    (tmp_path/"analysis").mkdir()
    (tmp_path/"analysis/COMPLETED").write_text(json.dumps(dict(seal_hash="seal")))
    old = [dict(nbformat.v4.new_markdown_cell("Historical",id="old")),
           dict(nbformat.v4.new_markdown_cell("Parent",id="plan15-retention-conclusion")),
           dict(nbformat.v4.new_markdown_cell("Later",id="later"))]
    book = dict(nbformat.v4.new_notebook());book["cells"] = old
    path = tmp_path/"book.ipynb";path.write_text(json.dumps(book))
    report.publish(result,FakeStore(),path)
    current = json.loads(path.read_text())
    assert [c for c in current["cells"] if not report.owned(c)] == old
    assert [c["id"] for c in current["cells"]][2:5] == [report.PREFIX+n for n in ("intro","figure","conclusion")]
    followup = [c for c in current["cells"] if report.owned(c)]
    monkeypatch.setattr(parent_report,"build_cells",lambda s:[
        parent_report.cell("progress","Done"),parent_report.cell("conclusion","Refreshed parent")])
    parent_report.publish(dict(config=Config().mapping()),path,FakeStore())
    refreshed = json.loads(path.read_text())
    assert refreshed["cells"][-3:] == followup
    assert refreshed["cells"][-4]["id"] == "plan15-retention-conclusion"
    refreshed["metadata"]["plan15_phase8_presentation"] = dict(role="frozen_archive")
    path.write_text(json.dumps(refreshed));before = path.read_bytes()
    with pytest.raises(ValueError,match="frozen"): report.publish(result,FakeStore(),path)
    assert path.read_bytes() == before
