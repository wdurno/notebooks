import json
from types import SimpleNamespace

from mnist_experiment.kalman_pi.run import Scheduler


def test_resume_recovers_measured_initialization_cost_without_loading_models(tmp_path, monkeypatch):
    from mnist_experiment.kalman_pi import run
    initial = tmp_path / "units" / "initial-development-r000"
    initial.mkdir(parents=True)
    (initial / "summary.json").write_text(json.dumps({"wall_seconds": 8.25}))
    monkeypatch.setattr(run, "load_data", lambda _: {})
    store = SimpleNamespace(root=tmp_path, unit_path=lambda uid: tmp_path / "units" / uid,
                            iter_units=lambda **_: iter([{"unit_id": initial.name,
                                                        "metadata": {"kind": "initialization"}}]))
    scheduler = Scheduler(SimpleNamespace(), store, SimpleNamespace(), tmp_path)
    assert scheduler.initial_times == {initial.name: 8.25}
    assert scheduler.initial_cache == {}
