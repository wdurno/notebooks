from picar_kl.context.messages import build_control_system_message, build_operator_status_message


def test_control_prompt_prioritizes_latest_operator_speech():
    text = build_control_system_message()["content"][0]["text"]

    assert "latest operator speech overrides" in text
    assert "obey it immediately" in text


def test_operator_status_message_adds_command_mapping_when_speech_present():
    message = build_operator_status_message(
        user_texts=["Red ball is to your left, you need to turn left."],
        step_index=8,
        last_reward=1.0,
        current_reward=0.0,
        reward_prompt_id="reward_prompt_1",
    )
    text = message["content"][0]["text"]

    assert "Operator command priority:" in text
    assert "`turn left`, `drive left`, or `go left` means choose `drive-left`." in text


def test_operator_status_message_omits_command_mapping_without_speech():
    message = build_operator_status_message(
        user_texts=[],
        step_index=0,
        last_reward=0.0,
        current_reward=1.0,
        reward_prompt_id="reward_prompt_1",
    )

    assert "Operator command priority:" not in message["content"][0]["text"]
