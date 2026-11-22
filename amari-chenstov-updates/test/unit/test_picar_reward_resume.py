"""Theory-provenance recovery must not relax numerical or input identities."""

import json
from pathlib import Path

import pytest

from numerical_experiment import picar_reward_resume as resume


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project, root = tmp_path / "project", tmp_path / "run"
    project.mkdir()
    (root / "orchestration").mkdir(parents=True)
    (project / resume.THEORY).write_bytes(b"new notebook")
    saved = {"source_sha256": {"kernel.py": "unchanged"},
             "theory_sha256": resume.digest(b"old notebook"), "model_source": "Qwen/test",
             "source_campaign": "source", "reference_campaign": "reference", "max_hours": 12.,
             "checkpoint_policy": {"name": "journal", "version": 1}}
    (root / "orchestration/config.json").write_text(json.dumps(saved))
    monkeypatch.setattr(resume.campaign, "PROJECT", project)
    monkeypatch.setattr(resume.campaign, "sources", lambda: dict(saved["source_sha256"]))
    monkeypatch.setattr(resume, "recover_theory", lambda *args: (b"old notebook", "historical_commit"))
    monkeypatch.setattr(resume, "git", lambda *args: b"git provenance")

    def identity(*args):
        result = {k: v for k, v in saved.items() if k != "checkpoint_policy"}
        result["theory_sha256"] = resume.digest((resume.campaign.PROJECT / resume.THEORY).read_bytes())
        return result

    monkeypatch.setattr(resume.campaign, "run_identity", identity)
    return root, project, saved


def test_view_redirects_only_theory_and_restores_project(setup):
    root, project, saved = setup
    _, snapshot, _ = resume.prepare(root)
    with resume.pinned_project(snapshot):
        assert resume.campaign.PROJECT / resume.THEORY == snapshot
        assert resume.campaign.PROJECT / "test/unit" == project / "test/unit"
        assert Path(resume.campaign.PROJECT) == project
        assert (project / "kernel.py").relative_to(resume.campaign.PROJECT) == Path("kernel.py")
    assert resume.campaign.PROJECT == project
    assert (project / resume.THEORY).read_bytes() == b"new notebook"


def test_receipt_preserves_both_theory_versions_and_snapshot(setup):
    root, _, saved = setup
    _, snapshot, path = resume.prepare(root)
    receipt = json.loads(path.read_text())
    assert snapshot.read_bytes() == b"old notebook"
    assert receipt["frozen_theory_sha256"] == saved["theory_sha256"]
    assert receipt["current_theory_sha256"] == resume.digest(b"new notebook")
    assert receipt["resolved_identity_matches"]
    assert not receipt["numerical_settings_changed"]
    before = snapshot.read_bytes()
    _, second, _ = resume.prepare(root)
    assert second == snapshot and snapshot.read_bytes() == before


def test_changed_numerical_sources_are_rejected(setup, monkeypatch):
    root, _, _ = setup
    monkeypatch.setattr(resume.campaign, "sources", lambda: {"kernel.py": "changed"})
    with pytest.raises(ValueError, match="scientific sources"):
        resume.prepare(root)


def test_other_identity_changes_are_rejected(setup, monkeypatch):
    root, _, saved = setup
    monkeypatch.setattr(resume.campaign, "run_identity", lambda *args: {**saved, "max_hours": 13.})
    with pytest.raises(ValueError, match="identity differs"):
        resume.prepare(root)


def test_corrupt_snapshot_is_rejected(setup):
    root, _, _ = setup
    _, snapshot, _ = resume.prepare(root)
    snapshot.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="snapshot is corrupt"):
        resume.prepare(root)


def test_run_restores_project_after_interruption_and_preserves_stop(setup, monkeypatch):
    root, project, saved = setup
    (root / "STOP").touch()

    def run(*args, **kwargs):
        assert not (root / "STOP").exists()
        assert resume.campaign.PROJECT / resume.THEORY != project / resume.THEORY
        assert kwargs["resume"] and kwargs["max_hours"] == saved["max_hours"]
        (root / "STOP").touch()
        raise InterruptedError("deliberate checkpoint stop")

    monkeypatch.setattr(resume.campaign, "run", run)
    with pytest.raises(InterruptedError):
        resume.run(root, clear_stop=True, offline=True)
    assert resume.campaign.PROJECT == project and (root / "STOP").exists()
    outcome = json.loads(next((root / "resume_provenance").glob("*.outcome.json")).read_text())
    assert outcome["exception_type"] == "InterruptedError"


def test_historical_recovery_requires_exact_content_hash(tmp_path, monkeypatch):
    (tmp_path / resume.THEORY).write_bytes(b"new notebook")

    def git(project, *args):
        if args[0] == "log":
            return b"new\nold\n"
        return b"old notebook" if args[1].startswith("old:") else b"new notebook"

    monkeypatch.setattr(resume, "git", git)
    content, origin = resume.recover_theory(tmp_path, resume.digest(b"old notebook"))
    assert content == b"old notebook" and origin == "old"
    with pytest.raises(ValueError, match="exact frozen"):
        resume.recover_theory(tmp_path, resume.digest(b"unavailable"))
