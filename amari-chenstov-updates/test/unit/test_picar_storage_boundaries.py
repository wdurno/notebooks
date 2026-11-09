import pytest

from numerical_experiment.picar.storage import FrameStore, RunCollisionError, RunStore


def complete(path):
    with RunStore(path, {"schema_version": 1}, reserve_bytes=0, local_quota_bytes=1000000) as run:
        run.checkpoint({"complete": True})
        run.complete()


def test_sealed_ancestor_rejects_new_runs_and_frame_writes(tmp_path):
    root = tmp_path / "sealed"
    complete(root)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(RunCollisionError, match="completed artifact"):
        RunStore(root / "new", {"schema_version": 1})
    with pytest.raises(RuntimeError, match="completed artifact"):
        FrameStore(root / "new-frames")
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_sealed_child_verification_is_allowed_with_sealed_parent(tmp_path):
    root = tmp_path / "parent"
    with RunStore(root, {"schema_version": 1}, reserve_bytes=0, local_quota_bytes=1000000) as parent:
        complete(root / "child")
        parent.checkpoint({"children_complete": True})
        parent.complete()
    with RunStore(root / "child", {"schema_version": 1}, resume=True,
                  reserve_bytes=0, local_quota_bytes=1000000) as child:
        assert child.completed
        child.verify_completed()
