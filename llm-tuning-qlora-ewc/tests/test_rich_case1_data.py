import re
from collections import Counter

from src.data.generate import SPLITS, generate_rule_transform_v2
from src.run_case1 import metadata_slice_summary, prefixed_slice_fields


REQUIRED_METADATA = {
    "entity_id",
    "rule_id",
    "requires_reasoning",
    "validator",
    "relation_to_next_task",
    "frequency_bucket",
    "composition_type",
}
RELATIONS = {"shared", "direct_conflict", "near_conflict", "rare_rule", "heldout_composition"}


def test_rule_transform_v2_is_deterministic():
    first = generate_rule_transform_v2("task_a", seed=13, n_per_split=16)
    second = generate_rule_transform_v2("task_a", seed=13, n_per_split=16)
    assert first == second


def test_rule_transform_v2_split_counts_and_metadata():
    examples = generate_rule_transform_v2("task_a", seed=13, n_per_split=16)
    assert Counter(example.split for example in examples) == {split: 16 for split in SPLITS}

    for example in examples:
        assert REQUIRED_METADATA.issubset(example.metadata)
        assert example.metadata["relation_to_next_task"] in RELATIONS
        validator = example.metadata["validator"]
        assert validator["type"] == "regex"
        assert re.fullmatch(validator["pattern"], example.target)


def test_rule_transform_v2_eval_heldout_contains_all_relations():
    examples = generate_rule_transform_v2("task_a", seed=13, n_per_split=16)
    eval_relations = {
        example.metadata["relation_to_next_task"]
        for example in examples
        if example.split == "eval_heldout"
    }
    assert RELATIONS.issubset(eval_relations)


def test_metadata_slice_summary_fields():
    rows = [
        {"exact_match": 1.0, "token_f1": 1.0, "format_validity": 1.0, "metadata": {"relation_to_next_task": "shared"}},
        {"exact_match": 0.0, "token_f1": 0.0, "format_validity": 1.0, "metadata": {"relation_to_next_task": "shared"}},
        {
            "exact_match": 1.0,
            "token_f1": 1.0,
            "format_validity": 1.0,
            "metadata": {"relation_to_next_task": "near_conflict"},
        },
        {"exact_match": 1.0, "token_f1": 1.0, "format_validity": 1.0, "metadata": {}},
    ]
    summary = metadata_slice_summary(rows, "relation_to_next_task", ["shared", "near_conflict", "rare_rule"])
    assert summary["shared_n"] == 2
    assert summary["shared_em"] == 0.5
    assert summary["near_conflict_n"] == 1
    assert summary["near_conflict_em"] == 1.0
    assert summary["rare_rule_n"] == 0
    assert summary["rare_rule_em"] is None


def test_prefixed_slice_fields_before_after_and_counts():
    before = {"shared_n": 2, "shared_em": 1.0}
    after = {"shared_n": 2, "shared_em": 0.5}
    fields = prefixed_slice_fields("task_a", before, after)
    assert fields == {
        "task_a_shared_n": 2,
        "task_a_shared_before_em": 1.0,
        "task_a_shared_after_em": 0.5,
    }
