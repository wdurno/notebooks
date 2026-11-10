import copy
import json

import pytest
import torch
from test_picar_models import miniature_qwen

from numerical_experiment.picar.dense_roles import parse_decision, parse_reward, prepare_conversation
from numerical_experiment.picar.learner import predict_histories
from numerical_experiment.picar.models import batch_digest, build_tiny_model, render_context, validate_batch


def history():
    return {"images": torch.full((1, 2, 3, 4, 4), .2),
        "executed_actions": torch.zeros(1, 2, 4), "elapsed": torch.ones(1, 2, 1),
        "contexts": [[{"text": "turn left", "issued_at": 0., "completed_at": 1.5},
                      {"text": "stop", "issued_at": 2.1}]],
        "episode_deadline": torch.tensor([20.]), "decision_time": torch.tensor([[1., 2.2]]),
        "status_events": [[{"text": "score=3", "available_at": 1.8, "kind": "visual_score"},
                           {"text": "future", "available_at": 3., "kind": "robot_status"}]]}


def test_countdown_decision_clock_and_status_are_causal_and_hashed():
    data = history()
    validate_batch(data, 4)
    first = json.loads(render_context(data, 0, 0))
    last = json.loads(render_context(data, 0, 1))
    assert first["schema"] == 2 and first["time_remaining_s"] == 19
    assert first["status_events"] == []
    assert first["instructions"][0]["status"] == "active"
    assert last["time_remaining_s"] == pytest.approx(17.8)
    assert [e["text"] for e in last["instructions"]] == ["turn left", "stop"]
    assert last["instructions"][0]["status"] == "completed"
    assert [e["text"] for e in last["status_events"]] == ["score=3"]
    other = copy.deepcopy(data)
    other["status_events"][0][-1]["text"] = "different future"
    assert batch_digest(data) == batch_digest(other)
    other["episode_deadline"] += 1
    assert batch_digest(data) != batch_digest(other)
    other = copy.deepcopy(data)
    other["decision_time"][0, 0] = .5
    with pytest.raises(ValueError, match="decision_time"):
        validate_batch(other, 4)


def test_grouped_prediction_retains_countdown_and_status_context():
    model = build_tiny_model(9, width=4, hidden_size=4, return_bins=3, context_length=4)
    data, other = history(), history()
    other["episode_deadline"] += 5
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    actual = predict_histories(model, params, [data, other])
    for key in actual:
        torch.testing.assert_close(actual[key], torch.cat([model(data)[key], model(other)[key]]),
                                   atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("text", ['{"reward":true}', '{"reward":"2"}', '{"reward":NaN}',
    '{"reward":11}', '{"reward":-1}', '[]', '{"reward":2,"extra":1}',
    '{"reward":2,"reward":3}', 'extra\n```json\n{"reward":3}\n```',
    '```json\n{"reward":3}\n```\nextra'])
def test_scores_are_numeric_strict_and_never_default_to_zero(text):
    with pytest.raises(ValueError):
        parse_reward(text)


def test_score_and_action_json_contract():
    assert parse_reward('{"reward":2.5}') == 2.5
    assert parse_reward('```json\n{"reward":3}\n```') == 3
    assert parse_decision('{"action":"drive-right","say":"Turning right."}') == (1, "Turning right.")
    assert parse_decision('{"action":"stop"}')[0] == 7
    assert parse_decision('```json\n{"action":"stop"}\n```')[0] == 7
    with pytest.raises(ValueError):
        parse_decision('{"action":"teleport"}')


def test_scoring_input_is_rgb_only_and_teacher_gets_causal_context():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return json.dumps(messages)
        def __call__(self, **kwargs):
            return {"input_ids": torch.ones(1, 3, dtype=torch.long)}
    class Encoder:
        context_length, max_context_chars, max_tokens = 4, 4096, 100
        processor = Processor()
    encoder, data = Encoder(), history()
    data["privileged_pose"] = "secret truth"
    prepare_conversation(encoder, data, "scoring rubric", visual_only=True)
    text = json.dumps(encoder.processor.messages)
    assert encoder.processor.messages[0]["role"] == "system"
    assert "turn left" not in text and "secret truth" not in text
    prepare_conversation(encoder, data, "control rubric")
    text = encoder.processor.messages[-1]["content"][-1]["text"]
    assert "time_remaining_s" in text and "score=3" in text
    assert "future" not in text and "secret truth" not in text
    active = text.split("CURRENT OPERATOR INSTRUCTIONS:")[1]
    assert "turn left" not in active and "stop" in active


def test_generated_scoring_uses_frozen_role_and_records_retry(monkeypatch, miniature_qwen):
    from numerical_experiment.picar import dense_roles
    from numerical_experiment.picar.models import QwenContextEncoder
    encoder = QwenContextEncoder(miniature_qwen, None, top_layers=1, ltm_rank=2)
    monkeypatch.setattr(dense_roles, "prepare_conversation", lambda *a, **k: {
        "attention_mask": torch.ones(1, 3, dtype=torch.long)})
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs["role"])
        return ['{"reward":NaN}'] if len(calls) == 1 else ['{"reward":4.5}']
    monkeypatch.setattr(dense_roles, "native_text", generate)
    scorer = dense_roles.GeneratedVisualReward(encoder)
    before = {k: v.detach().clone() for k, v in encoder.named_parameters()}
    assert scorer.scores(history()).tolist() == [4.5]
    assert calls == ["evaluator", "evaluator"]
    assert len(scorer.last_diagnostic["attempts"]) == 2
    for name, value in encoder.named_parameters():
        assert torch.equal(value, before[name]) and value.grad is None


def test_live_ledger_and_transcripts_use_the_same_replayable_history():
    import numpy as np
    from demo.picar.speech import InstructionLedger
    from numerical_experiment.picar.fix2_data import history_batch
    typed, speech = InstructionLedger(), InstructionLedger()
    typed.typed("turn", "Turn left", at=1.)
    speech.transcript("turn", "Turn left", utterance_start=.2, utterance_end=.5, available_at=1.)
    rows = [{"rgb": np.zeros((4, 4, 3), dtype=np.uint8), "elapsed": 0., "time": .7,
             "action": (0., 0., 0., 0.)}]
    earlier = history_batch(rows, instructions=speech, deadline=20., width=4, height=4)
    assert json.loads(render_context(earlier, 0, 0))["instructions"] == []
    rows[0]["decision_time"] = 1.1
    a = history_batch(rows, instructions=typed, deadline=20., width=4, height=4)
    b = history_batch(rows, instructions=speech, deadline=20., width=4, height=4)
    assert batch_digest(a) == batch_digest(b)
    speech.typed("finished", "Turn completed", at=1.2, kind="completed", target_id="turn")
    resumed = InstructionLedger.from_snapshot(speech.snapshot())
    rows[0]["decision_time"] = 1.3
    after = history_batch(rows, instructions=resumed, deadline=20., width=4, height=4)
    validate_batch(after, 4)
    visible = json.loads(render_context(after, 0, 0))
    assert visible["instructions"][0]["status"] == "completed"
    assert visible["status_events"][0]["text"] == "Turn completed"
