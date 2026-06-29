from picar_kl.actions import action_name_to_distribution
from picar_kl.vlm.control import build_phase1_messages, parse_action_response


def test_parse_action_response_prefers_embedded_json():
    decision = parse_action_response('noise {"action": "look-left", "say": "Checking left."} tail')

    assert decision.action_distribution == action_name_to_distribution("look-left")
    assert decision.generated_text == "Checking left."
    assert decision.metadata["parser"] == "json"


def test_parse_action_response_falls_back_to_action_text():
    decision = parse_action_response("Final answer: drive-forward")

    assert decision.action_distribution == action_name_to_distribution("drive-forward")
    assert decision.metadata["parser"] == "text"


def test_build_phase1_messages_contains_task_and_operator_text():
    messages = build_phase1_messages(
        task_prompt="find the red ball",
        step_index=3,
        user_texts=["turn around"],
        last_generated_text="Moving.",
    )

    assert messages[0]["role"] == "system"
    assert "find the red ball" in messages[0]["content"][0]["text"]
    assert "turn around" in messages[1]["content"][0]["text"]
    assert "step_index=3" in messages[1]["content"][0]["text"]
