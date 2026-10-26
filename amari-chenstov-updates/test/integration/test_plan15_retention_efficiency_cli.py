"""Artifact-only CLI acceptance: synthetic sealed predictions, no model or MNIST."""

import hashlib
import json

import nbformat
import pytest

from mnist_experiment import plan15_retention_efficiency as efficiency
from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError
from mnist_experiment.fisher_memory_regression.config import Recipe, content_hash
from mnist_experiment.fisher_memory_regression.analysis import summarize_trajectory
from mnist_experiment.fisher_memory_retention.config import Config
from mnist_experiment.fisher_memory_retention.analysis import analyze


def source(root, *, missing=False, unpaired=False):
    cfg = Config()
    recipes = [Recipe.from_config(cfg,m,strength=2. if m=="fd_replay_ewc" else 0.)
               for m in ("fd_replay_ewc","replay_ewc","ewc_only")]
    manifest = dict(protocol=efficiency.PARENT_PROTOCOL,sealed=True,schema_version=1,
        replicas=list(range(12)),N=100,recipes=[{**r.mapping(),"recipe_key":r.key} for r in recipes],
        delta=cfg.retention_margin_degrees,epsilon=cfg.matching_tolerance,
        learning_time_limit=cfg.learning_time_limit,memory_ceiling_bytes=cfg.memory_ceiling_bytes,
        bootstrap_seed=123)
    manifest["manifest_hash"] = content_hash(manifest)
    records = []
    identity = dict(protocol=efficiency.PARENT_PROTOCOL,role="production",allocation_family=efficiency.ALLOCATION_FAMILY)
    with Store(root,identity) as store:
        for recipe in recipes:
            for replica in range(12):
                fd = recipe.method == "fd_replay_ewc"
                metrics = [dict(n=n,novel_mse=300+12*replica-(2+.15*replica if fd else 1+.1*replica)*n,
                    base_mse=9+.02*n*(1+replica/10),learning_seconds=(.4 if fd else .25)*n,
                    observations=(10+replica)*n) for n in range(101)]
                record = dict(summarize_trajectory(metrics,N=100),schema_version=1,
                    protocol=efficiency.PARENT_PROTOCOL,method=recipe.method,recipe_key=recipe.key,
                    recipe=recipe.mapping(),replica=replica,partition="confirmation",family="confirmation",
                    pairing_id=f"pair-{replica}",plan_hash=f"pair-{replica}",metrics=metrics,target=100,
                    evaluation_seconds=1.,fit_seconds=20.,eviction_seconds=5.,peak_vram_bytes=1000)
                records.append(record)
                if missing and recipe.method=="ewc_only" and replica==11: continue
                unit = f"trajectory-{recipe.key}-{replica}"
                store.begin(unit,dict(kind="regression_trajectory",partition="confirmation"))
                stored = {k:v for k,v in record.items() if k!="metrics"}
                if unpaired and fd and replica==0: stored["pairing_id"] = "different"
                store.write_json(unit,"summary.json",stored)
                store.write_json(unit,"metrics.json",metrics)
                store.complete(unit)
        result = analyze(records,manifest,seed=123,draws=20)
        for name,value in (("confirmation-manifest",manifest),("analysis",result),("closeout",dict(done=True))):
            unit = "decision-"+name
            store.begin(unit,dict(kind="retention_decision"));store.write_json(unit,"result.json",value);store.complete(unit)
    return root


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for u in (root/"units").iterdir() if (u/"COMPLETED").exists() for p in u.iterdir() if p.is_file()}


def test_cpu_cli_resume_stop_file_and_publication_preserve_parent(tmp_path):
    parent = source(tmp_path/"source");original = hashes(parent)
    book = dict(nbformat.v4.new_notebook());book["cells"] = [dict(nbformat.v4.new_markdown_cell(
        "Original conclusion",id="plan15-retention-conclusion"))]
    notebook = tmp_path/"book.ipynb";notebook.write_text(json.dumps(book))
    root = tmp_path/"followup"
    args = ["--source-root",str(parent),"--root",str(root),"--budget-root",str(tmp_path),
            "--data-root",str(tmp_path/"no-dataset-needed"),"--notebook",str(notebook)]
    assert efficiency.main([*args,"--max-units","2"]) == 2
    before = hashes(root)
    (root/"STOP").touch()
    assert efficiency.main([*args,"--resume"]) == 2
    assert efficiency.main([*args,"--resume","--clear-stop"]) == 0
    assert before.items() <= hashes(root).items()
    complete = hashes(root)
    assert efficiency.main([*args,"--resume"]) == 0
    assert hashes(root) == complete
    assert hashes(parent) == original
    actual = json.loads(notebook.read_text())
    assert actual["cells"][0] == book["cells"][0]
    assert len(actual["cells"]) == 4
    assert json.loads((root/"status.json").read_text())["status"] == "completed"
    assert len(complete) > 20


@pytest.mark.parametrize("case",["missing","unpaired"])
def test_loader_requires_all_three_frozen_arms_and_pairing(tmp_path,case):
    root = source(tmp_path/"source",**{case:True})
    dependencies = efficiency.dependency_snapshot(root)
    with pytest.raises((ArtifactError,ValueError)):
        efficiency.load_source(root,dependencies)
