from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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


def generate_case1(output_dir: Path, seed: int, n_per_split: int) -> None:
    task_a = generate_rule_transform("task_a", seed, n_per_split, ("BLUE", "GREEN"))
    task_b = generate_rule_transform("task_b", seed + 1, n_per_split, ("RED", "GREEN"))
    factory = generate_factory_qa("factory_a", seed + 2, n_per_split)

    _write_jsonl(output_dir / "case1_task_a.jsonl", task_a)
    _write_jsonl(output_dir / "case1_task_b.jsonl", task_b)
    _write_jsonl(output_dir / "factory_qa.jsonl", factory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-per-split", type=int, default=128)
    args = parser.parse_args()
    generate_case1(args.output_dir, args.seed, args.n_per_split)


if __name__ == "__main__":
    main()
