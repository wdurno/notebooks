import threading

import pytest

from numerical_experiment.picar.runtime import (
    ArchiveUnavailable, BoundedWorkQueue, BudgetExceeded, ComputeLedger,
    ModelAccess, StopController, check_storage,
)


def test_budget_and_stop(tmp_path):
    ledger = ComputeLedger(1)
    with ledger.charge():
        pass
    assert ledger.charged_s > 0
    with pytest.raises(BudgetExceeded):
        ledger.admit(2)
    stop = StopController(tmp_path)
    assert not stop.check()
    (tmp_path / "STOP").touch()
    assert stop.check()


def test_queue_has_a_hard_bound():
    q = BoundedWorkQueue(1)
    assert q.offer("a")
    assert not q.offer("b")
    assert q.take() == "a"
    assert q.take() is None


def test_storage_admission(tmp_path):
    (tmp_path / "blob").write_bytes(b"0123456789")
    assert check_storage(tmp_path, quota_bytes=20, reserve_bytes=0)["used_bytes"] == 10
    with pytest.raises(BudgetExceeded):
        check_storage(tmp_path, quota_bytes=20, reserve_bytes=0, incoming_bytes=11)
    with pytest.raises(ArchiveUnavailable):
        check_storage(tmp_path / "unmounted", quota_bytes=20, reserve_bytes=0)


def test_model_publication_is_atomic():
    access = ModelAccess()
    with access.inference() as version:
        assert version == 0
    with access.update() as publish:
        publish()
        with pytest.raises(RuntimeError):
            publish()
    with access.inference() as version:
        assert version == 1
    with pytest.raises(ValueError):
        with access.update():
            raise ValueError("training failed before publication")
    with access.inference() as version:
        assert version == 1


def test_inference_waits_for_accepted_update():
    access = ModelAccess()
    entered = threading.Event()
    results = []
    def infer():
        entered.set()
        with access.inference() as version:
            results.append(version)
    with access.update() as publish:
        worker = threading.Thread(target=infer)
        worker.start()
        assert entered.wait(1)
        assert not results
        publish()
    worker.join(1)
    assert not worker.is_alive()
    assert results == [1]
