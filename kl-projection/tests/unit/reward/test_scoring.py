import numpy as np

from picar_kl.reward import FrozenVLMRewardScorer, RewardConfig, parse_reward_text


def test_parse_reward_text_clips_and_falls_back_to_number():
    assert parse_reward_text('{"reward": 7.5}') == (7.5, 7.5)
    assert parse_reward_text("reward is 99", max_reward=10.0) == (99.0, 10.0)


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def generate_completion(self, **kwargs):
        self.calls.append(kwargs)
        return '{"reward": 8}', {"prompt_tokens": 12}


def test_reward_scorer_uses_shared_runtime_base_context():
    runtime = FakeRuntime()
    scorer = FrozenVLMRewardScorer(RewardConfig(prompt_id="reward_prompt_1"), runtime=runtime)

    result = scorer.score(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.clipped_reward == 8.0
    assert result.task_text == "find the red ball"
    assert runtime.calls[0]["use_base_model"] is True
    assert runtime.calls[0]["max_new_tokens"] == 16
