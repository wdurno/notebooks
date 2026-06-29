import pytest

from picar_kl.latency import LatencyEvent, LatencyTimer


def test_latency_event_records_raw_wall_clock_bounds():
    event = LatencyEvent.from_bounds(
        "action_decision",
        started_at=10.0,
        ended_at=12.5,
        metadata={"step_index": 7},
    )

    assert event.duration_seconds == 2.5
    assert event.to_dict()["metadata"] == {"step_index": 7}
    assert LatencyEvent.from_dict(event.to_dict()) == event


def test_latency_event_rejects_negative_duration():
    with pytest.raises(ValueError, match="ended before"):
        LatencyEvent.from_bounds("bad", started_at=2.0, ended_at=1.0)


def test_latency_timer_records_event():
    with LatencyTimer("vlm_call") as timer:
        pass

    assert timer.event is not None
    assert timer.event.name == "vlm_call"
    assert timer.event.duration_seconds >= 0.0
