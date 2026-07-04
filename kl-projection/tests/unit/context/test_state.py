from picar_kl.context import ContextConfig, EpisodeContext
from picar_kl.reward import RewardResult


class WordTokenizer:
    def count_prompt_tokens(self, messages):
        total = 0
        for message in messages:
            for item in message.get("content") or []:
                if item.get("type") == "text":
                    total += len(str(item.get("text") or "").split())
                elif item.get("type") == "image":
                    total += 1
        return total


def reward(value=1.0):
    return RewardResult(
        prompt_id="reward_prompt_1",
        raw_text='{"reward": 1}',
        reward=value,
        clipped_reward=value,
        metadata={"task_text": "find the red ball"},
    )


def test_episode_context_preserves_goal_and_bounds_history():
    context = EpisodeContext(config=ContextConfig(history_window=3, prompt_token_window=8000))
    tokenizer = WordTokenizer()

    for idx in range(4):
        context.add_observation(
            user_texts=[f"command {idx}"],
            reward_result=reward(float(idx)),
            step_index=idx,
            tokenizer=tokenizer,
        )
        context.add_assistant_action(action_name="look-forward", generated_text=f"reply {idx}")

    rendered = context.render(tokenizer=tokenizer)

    assert rendered.messages[0]["role"] == "system"
    assert rendered.messages[1]["content"][0]["text"].startswith("Persistent goals:")
    assert len(context.history) == 3
    assert any("reply 3" in item.get("text", "") for item in rendered.messages[-1]["content"])


def test_episode_context_applies_prompt_token_window_and_clips_latest():
    context = EpisodeContext(config=ContextConfig(history_window=10, prompt_token_window=260))
    tokenizer = WordTokenizer()

    context.add_observation(
        user_texts=["old words " * 20],
        reward_result=reward(1.0),
        step_index=0,
        tokenizer=tokenizer,
    )
    context.add_assistant_action(action_name="look-forward", generated_text="old reply " * 20)
    rendered = context.add_observation(
        user_texts=["new operator command " * 20],
        reward_result=reward(2.0),
        step_index=1,
        tokenizer=tokenizer,
    )

    assert rendered.metadata["prompt_truncated"] is True
    assert rendered.metadata["prompt_tokens"] <= 260
    rendered_text = "\n".join(
        str(item.get("text") or "")
        for message in rendered.messages
        for item in message.get("content") or []
        if item.get("type") == "text"
    )
    assert "truncated" in rendered_text
