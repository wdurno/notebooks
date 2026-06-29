from picar_kl.actions import action_name_to_distribution
from picar_kl.latency import LatencyEvent
from picar_kl.records import ActionRecord, Phase1ObservationRecord


def test_legacy_action_record_reconstructs_one_hot_distribution():
    record = ActionRecord.from_legacy_dict(
        {
            "agentic_action_name": "look-left",
            "executed_action_vector": {"pan": 1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
            "generated_text": "Looking left.",
            "critic_value": -0.5,
        }
    )

    assert record.distribution == action_name_to_distribution("look-left")
    assert record.action_name == "look-left"
    assert record.generated_text == "Looking left."
    assert record.metadata["critic_value"] == -0.5


def test_new_observation_record_round_trips_latency_and_action():
    action = ActionRecord.from_distribution(
        action_name_to_distribution("drive-forward"),
        generated_text="Driving forward.",
        source="vlm",
    )
    latency = LatencyEvent.from_bounds("decision", started_at=1.0, ended_at=2.0)
    observation = Phase1ObservationRecord(
        run_uuid="run-1",
        run_dir=None,
        timestamp="2026-01-01T00:00:00+00:00",
        source="step",
        step_index=3,
        image_path=None,
        messages=[],
        user_texts=["find the red ball"],
        action=action,
        reward=1.0,
        latency_events=(latency,),
    )

    loaded = Phase1ObservationRecord.from_dict(observation.to_dict(), run_uuid="run-1")

    assert loaded.action == action
    assert loaded.latency_events == (latency,)
    assert loaded.user_texts == ["find the red ball"]


def test_legacy_observation_without_action_keeps_action_missing():
    observation = Phase1ObservationRecord.from_legacy_dict(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source": "reset",
            "step_index": 0,
            "image_path": "images/step_000000.npz",
            "messages": [],
            "user_texts": [],
            "action": None,
            "reward": None,
            "done": False,
        },
        run_uuid="run-1",
    )

    assert observation.action is None
    assert observation.latency_events == ()
