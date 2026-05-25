from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


SPLITS = ("ewc_init", "train", "eval_seen", "eval_heldout")


@dataclass(frozen=True)
class Example:
    id: str
    task_family: str
    task_id: str
    split: str
    prompt: str
    target: str
    answer_key: str
    metadata: dict


def _write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")


def generate_factory_qa(task_id: str, seed: int, n_per_split: int) -> list[Example]:
    rng = random.Random(seed)
    machines = [f"ZX-{idx:02d}" for idx in range(10, 90)]
    coolants = ["amber-4", "blue-7", "clear-2", "green-9"]
    components = ["flow regulator", "seal plate", "intake valve", "thermal relay"]
    errors = ["E17", "E42", "E66", "E91"]
    manual = {
        machine: {
            "coolant": rng.choice(coolants),
            "error_components": {error: rng.choice(components) for error in errors},
            "restart_threshold": rng.choice([10, 12, 15]),
        }
        for machine in machines
    }
    examples: list[Example] = []

    for split in SPLITS:
        for i in range(n_per_split):
            machine = rng.choice(machines)
            error = rng.choice(errors)
            pressure = rng.choice([8, 10, 12, 14, 18])
            coolant = manual[machine]["coolant"]
            component = manual[machine]["error_components"][error]
            threshold = manual[machine]["restart_threshold"]
            case = rng.randrange(3)

            if case == 0:
                prompt = f"Factory manual question: What coolant does {machine} use? Answer with only the coolant label."
                target = coolant
                rule_id = "coolant_lookup"
                requires_reasoning = False
            elif case == 1:
                prompt = f"Factory manual question: {machine} reports {error}. What component should be inspected first? Answer with only the component name."
                target = component
                rule_id = "error_component_lookup"
                requires_reasoning = False
            else:
                decision = "ALLOW_RESTART" if pressure <= threshold else "DENY_RESTART"
                prompt = (
                    f"Factory safety question: {machine} reports {error}. Restart is allowed only when pressure is at or below "
                    f"{threshold} PSI. Current pressure is {pressure} PSI. Answer ALLOW_RESTART or DENY_RESTART."
                )
                target = decision
                rule_id = "pressure_restart_decision"
                requires_reasoning = True

            examples.append(
                Example(
                    id=f"{task_id}_{split}_{i:06d}",
                    task_family="factory_qa",
                    task_id=task_id,
                    split=split,
                    prompt=prompt,
                    target=target,
                    answer_key=target,
                    metadata={
                        "entity_id": machine,
                        "rule_id": rule_id,
                        "requires_reasoning": requires_reasoning,
                    },
                )
            )
    return examples


def generate_rule_transform(task_id: str, seed: int, n_per_split: int, palette: tuple[str, str]) -> list[Example]:
    rng = random.Random(seed)
    prefixes = ["AB", "CD", "EF", "GH", "JK", "LM"]
    examples: list[Example] = []

    for split in SPLITS:
        for i in range(n_per_split):
            prefix = rng.choice(prefixes)
            number = rng.randrange(0, 100)
            color = palette[0] if prefix in {"AB", "EF", "JK"} else palette[1]
            code = f"{prefix}-{number:02d}"
            target = f"ROUTE_{color}_{number:02d}"
            prompt = f"Routing transform: Convert product code {code} to its route label. Answer with only the route label."
            examples.append(
                Example(
                    id=f"{task_id}_{split}_{i:06d}",
                    task_family="rule_transform",
                    task_id=task_id,
                    split=split,
                    prompt=prompt,
                    target=target,
                    answer_key=target,
                    metadata={
                        "entity_id": code,
                        "rule_id": f"route_{prefix}",
                        "requires_reasoning": False,
                        "validator": {"type": "regex", "pattern": r"ROUTE_[A-Z]+_[0-9]{2}"},
                    },
                )
            )
    return examples


Relation = Literal["shared", "direct_conflict", "near_conflict", "rare_rule", "heldout_composition"]


def _cycle_pick(values: list[str], index: int, rng: random.Random) -> str:
    offset = rng.randrange(len(values))
    return values[(index + offset) % len(values)]


def _relation_schedule(split: str, n_per_split: int) -> list[Relation]:
    if split == "eval_heldout":
        pattern: list[Relation] = ["shared", "direct_conflict", "near_conflict", "rare_rule", "heldout_composition"]
    elif split == "eval_seen":
        pattern = ["shared", "direct_conflict", "near_conflict", "rare_rule"]
    elif split == "train":
        pattern = [
            "shared",
            "shared",
            "shared",
            "direct_conflict",
            "direct_conflict",
            "near_conflict",
            "near_conflict",
            "rare_rule",
        ]
    else:
        pattern = [
            "shared",
            "shared",
            "shared",
            "direct_conflict",
            "direct_conflict",
            "near_conflict",
            "near_conflict",
            "rare_rule",
        ]
    return [pattern[index % len(pattern)] for index in range(n_per_split)]


def _rule_transform_v2_attrs(task_id: str, relation: Relation, index: int, rng: random.Random) -> dict[str, str]:
    number = f"{(17 * index + rng.randrange(100)) % 100:02d}"
    if relation == "shared":
        prefix = _cycle_pick(["CD", "GH"], index, rng)
        priority = _cycle_pick(["LOW", "MEDIUM"], index, rng)
        region = _cycle_pick(["NORTH", "SOUTH"], index, rng)
        mode = _cycle_pick(["STANDARD", "BULK"], index, rng)
        color = "GREEN"
        frequency_bucket = "common"
        composition_type = "seen"
    elif relation == "direct_conflict":
        prefix = "AB"
        priority = _cycle_pick(["LOW", "HIGH"], index, rng)
        region = _cycle_pick(["NORTH", "SOUTH"], index, rng)
        mode = "STANDARD"
        color = "RED" if task_id == "task_b" else "BLUE"
        frequency_bucket = "common"
        composition_type = "seen"
    elif relation == "near_conflict":
        prefix = "EF"
        priority = "HIGH" if task_id == "task_b" else "LOW"
        region = _cycle_pick(["NORTH", "SOUTH"], index, rng)
        mode = "STANDARD"
        color = "RED" if task_id == "task_b" else "BLUE"
        frequency_bucket = "medium"
        composition_type = "seen"
    elif relation == "rare_rule":
        prefix = "JK"
        priority = "MEDIUM"
        region = "EAST"
        mode = "LEGACY"
        color = "PURPLE"
        frequency_bucket = "rare"
        composition_type = "seen"
    else:
        prefix = "LM"
        priority = "HIGH"
        region = "WEST"
        mode = "NIGHT"
        color = "CYAN"
        frequency_bucket = "medium"
        composition_type = "heldout_triple"

    return {
        "prefix": prefix,
        "number": number,
        "priority": priority,
        "region": region,
        "mode": mode,
        "color": color,
        "frequency_bucket": frequency_bucket,
        "composition_type": composition_type,
    }


def generate_rule_transform_v2(task_id: str, seed: int, n_per_split: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []

    for split in SPLITS:
        for i, relation in enumerate(_relation_schedule(split, n_per_split)):
            attrs = _rule_transform_v2_attrs(task_id, relation, i, rng)
            code = f"{attrs['prefix']}-{attrs['number']}"
            target = f"ROUTE_{attrs['color']}_{attrs['number']}"
            prompt = (
                f"Routing transform v2: Convert product code {code} with priority {attrs['priority']}, "
                f"region {attrs['region']}, and mode {attrs['mode']} to its route label. "
                "Answer with only the route label."
            )
            examples.append(
                Example(
                    id=f"{task_id}_{split}_{i:06d}",
                    task_family="rule_transform_v2",
                    task_id=task_id,
                    split=split,
                    prompt=prompt,
                    target=target,
                    answer_key=target,
                    metadata={
                        "entity_id": code,
                        "rule_id": f"route_{attrs['prefix']}_{attrs['priority']}_{attrs['region']}_{attrs['mode']}",
                        "requires_reasoning": False,
                        "relation_to_next_task": relation,
                        "frequency_bucket": attrs["frequency_bucket"],
                        "composition_type": attrs["composition_type"],
                        "prefix": attrs["prefix"],
                        "priority": attrs["priority"],
                        "region": attrs["region"],
                        "mode": attrs["mode"],
                        "validator": {"type": "regex", "pattern": r"ROUTE_[A-Z]+_[0-9]{2}"},
                    },
                )
            )
    return examples


def generate_case1(output_dir: Path, seed: int, n_per_split: int, generator: str = "rule_transform") -> None:
    if generator == "rule_transform":
        task_a = generate_rule_transform("task_a", seed, n_per_split, ("BLUE", "GREEN"))
        task_b = generate_rule_transform("task_b", seed + 1, n_per_split, ("RED", "GREEN"))
    elif generator == "rule_transform_v2":
        task_a = generate_rule_transform_v2("task_a", seed, n_per_split)
        task_b = generate_rule_transform_v2("task_b", seed + 1, n_per_split)
    else:
        raise ValueError(f"Unknown Case 1 generator: {generator}")

    factory = generate_factory_qa("factory_a", seed + 2, n_per_split)

    task_a_filename = "case1_rich_task_a.jsonl" if generator == "rule_transform_v2" else "case1_task_a.jsonl"
    task_b_filename = "case1_rich_task_b.jsonl" if generator == "rule_transform_v2" else "case1_task_b.jsonl"
    _write_jsonl(output_dir / task_a_filename, task_a)
    _write_jsonl(output_dir / task_b_filename, task_b)
    _write_jsonl(output_dir / "factory_qa.jsonl", factory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-per-split", type=int, default=128)
    parser.add_argument("--generator", default="rule_transform", choices=("rule_transform", "rule_transform_v2"))
    args = parser.parse_args()
    generate_case1(args.output_dir, args.seed, args.n_per_split, args.generator)


if __name__ == "__main__":
    main()
