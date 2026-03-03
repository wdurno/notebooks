from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.rewarding import RewardPromptRegistry, parse_reward_text


def test_parse_reward_text_prefers_json_and_clips():
    reward, clipped = parse_reward_text('{"reward": 11.5}', min_reward=0.0, max_reward=10.0)

    assert reward == 11.5
    assert clipped == 10.0


def test_parse_reward_text_falls_back_to_numeric_text():
    reward, clipped = parse_reward_text("reward = 4.25", min_reward=0.0, max_reward=10.0)

    assert reward == 4.25
    assert clipped == 4.25


def test_default_reward_registry_exposes_prompt_one():
    registry = RewardPromptRegistry.default()

    assert registry.list_prompt_ids() == ["reward_prompt_1"]
    assert "red ball" in registry.get("reward_prompt_1").prompt_text
