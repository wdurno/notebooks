from types import SimpleNamespace

import pytest

from mnist_experiment.diagonal_lfu.run import Scheduler
from mnist_experiment.fisher_memory.control import StopRequested


def test_common_action_is_selected_from_all_development_replicas():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cfg = SimpleNamespace(control_cs=(.01, .05), ridge_ratio=.1, dev_replicas=2)
    scheduler.selection = {}
    scheduler.pulse = lambda **kw: None
    calls = []
    def trajectory(cohort, replica, schedule, condition, c, ridge):
        calls.append((replica, condition, c))
        value = {(0, .01): .9, (0, .05): .1, (1, .01): 0., (1, .05): 1.}[replica, c]
        return {"c": c, "metrics": [{"arrivals": 0, "nine_recall": value, "old_accuracy": value},
                                     {"arrivals": 10, "nine_recall": value, "old_accuracy": value}],
                "summary": {"wall_seconds": 1.}}
    scheduler.trajectory = trajectory
    def selected():
        raise StopRequested("selection inspected")
    scheduler.save_selection = selected
    with pytest.raises(StopRequested, match="selection inspected"):
        scheduler.run()
    assert scheduler.selection["common_c"] == .05
    assert all(c["replicas"] == 2 for c in scheduler.selection["candidates"])
