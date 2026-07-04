from picar_kl.context.messages import build_control_system_message, build_operator_status_message


def test_control_prompt_prioritizes_latest_operator_speech():
    text = build_control_system_message()["content"][0]["text"]

    assert "Authority order:" in text
    assert "Operator speech is ground truth" in text
    assert "previous `say` messages are tentative status reports" in text
    assert "the task is not complete" in text
    assert "obey it immediately" in text
    assert "answer that question directly" in text


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


def test_operator_question_message_requires_direct_spoken_answer():
    message = build_operator_status_message(
        user_texts=["Do you see the red ball?"],
        step_index=3,
        last_reward=1.0,
        current_reward=2.0,
        reward_prompt_id="reward_prompt_1",
    )
    text = message["content"][0]["text"]

    assert "answer the question directly in `say` before task narration" in text
    assert "begin `say` with yes, no, or I am not sure" in text
    assert "whether you see the red ball" in text


def test_goal_prompt_treats_operator_corrections_as_authoritative():
    from picar_kl.context.messages import build_goal_system_message

    text = build_goal_system_message(task_text="find the red ball")["content"][0]["text"]

    assert "Operator speech is authoritative" in text
    assert "Treat operator corrections as new observations" in text
    assert "task is not complete" in text
