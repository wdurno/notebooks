from pathlib import Path
import json
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.phase1_finalize import (
    Phase1FinalizeConfig,
    _resolve_fit_iters,
    _validate_config,
    _set_model_optimization_mode,
    build_parser,
    build_target_text,
    count_step_rows,
    iter_transitions_from_run,
    resolve_data_runs,
)


def _write_observation_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(images_dir / "step_000000.npz", image=np.zeros((2, 2, 3), dtype=np.uint8))
    np.savez_compressed(images_dir / "step_000001.npz", image=np.ones((2, 2, 3), dtype=np.uint8))
    np.savez_compressed(images_dir / "step_000002.npz", image=np.full((2, 2, 3), 2, dtype=np.uint8))

    rows = [
        {
            "source": "reset",
            "step_index": 0,
            "t": 0.0,
            "last_reward": 0.0,
            "done": False,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]}],
            "metadata": {"reward_prompt_id": "reward_prompt_1"},
            "image_path": "images/step_000000.npz",
        },
        {
            "source": "step",
            "step_index": 1,
            "t": 1.0,
            "last_reward": 0.5,
            "reward": 1.25,
            "done": False,
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "moving"}]}],
            "metadata": {"reward_prompt_id": "reward_prompt_1"},
            "image_path": "images/step_000001.npz",
            "action": {
                "agentic_action_name": "drive-forward",
                "agentic_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
                "actor_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.6},
                "executed_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.8},
                "critic_value": 1.0,
                "generated_text": "moving",
                "logp_beta_sum": -0.25,
            },
        },
        {
            "source": "step",
            "step_index": 2,
            "t": 1.0,
            "last_reward": 1.25,
            "reward": 2.0,
            "done": False,
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}],
            "metadata": {"reward_prompt_id": "reward_prompt_1"},
            "image_path": "images/step_000002.npz",
            "action": {
                "agentic_action_name": "look-forward",
                "agentic_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
                "actor_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
                "executed_action_vector": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
                "critic_value": 1.2,
                "generated_text": "done",
                "logp_beta_sum": -0.1,
            },
        },
    ]
    with (run_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_phase1_finalize_transition_reconstruction(tmp_path):
    run_dir = tmp_path / "data-run"
    _write_observation_run(run_dir)

    transitions = list(iter_transitions_from_run(run_dir))

    assert count_step_rows(run_dir) == 2
    assert len(transitions) == 2
    first = transitions[0]
    assert first.observation.step_index == 0
    assert first.next_observation.step_index == 1
    assert first.executed_action_vector["drive"] == 0.8
    assert first.reward == 1.25
    assert first.target_action_name == "drive-forward"
    assert first.target_text == '{"action": "drive-forward", "say": "moving"}'
    second = transitions[1]
    assert second.observation.step_index == 1
    assert second.next_observation.step_index == 2
    assert second.target_text == "done"


def test_phase1_finalize_resolve_data_runs_requires_observations(tmp_path):
    good = tmp_path / "good"
    _write_observation_run(good)
    missing = tmp_path / "missing"
    missing.mkdir(parents=True, exist_ok=True)

    resolved = resolve_data_runs([good])
    assert resolved == [good.resolve()]

    try:
        resolve_data_runs([missing])
    except FileNotFoundError as exc:
        assert "observations.jsonl" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing observations.jsonl")


def test_build_target_text_switches_at_t_equals_one():
    action = {"agentic_action_name": "look-left", "generated_text": "checking left"}

    at_agentic = build_target_text(action=action, observation_t=0.25)
    at_actor_only = build_target_text(action=action, observation_t=1.0)

    assert at_agentic == '{"action": "look-left", "say": "checking left"}'
    assert at_actor_only == "checking left"


def test_set_model_optimization_mode_prefers_explicit_hook():
    class DummyModel:
        def __init__(self):
            self.called = 0
            self.train_called = 0

        def set_optimization_mode(self):
            self.called += 1

        def train(self):
            self.train_called += 1

    model = DummyModel()
    _set_model_optimization_mode(model)

    assert model.called == 1
    assert model.train_called == 0


def test_phase1_finalize_parser_prompt_token_window_defaults():
    parser = build_parser()
    args = parser.parse_args(["--data-runs", "data/run-a"])

    assert args.prompt_token_window == 512
    assert args.fit_iters is None
    assert args.log_level == "INFO"


def test_phase1_finalize_validate_rejects_negative_prompt_token_window():
    config = Phase1FinalizeConfig(data_runs=[Path("/tmp/run")], prompt_token_window=-1)

    try:
        _validate_config(config)
    except ValueError as exc:
        assert "--prompt-token-window" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative prompt token window.")


def test_phase1_finalize_validate_rejects_non_positive_fit_iters():
    config = Phase1FinalizeConfig(data_runs=[Path("/tmp/run")], fit_iters=0)

    try:
        _validate_config(config)
    except ValueError as exc:
        assert "--fit-iters" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive fit_iters.")


def test_phase1_finalize_resolve_fit_iters_defaults_to_approx_one_pass():
    assert _resolve_fit_iters(replay_size=60, batch_size=4, fit_iters=None) == 15


def test_phase1_finalize_resolve_fit_iters_respects_explicit_override():
    assert _resolve_fit_iters(replay_size=60, batch_size=4, fit_iters=7) == 7
