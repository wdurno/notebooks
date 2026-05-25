from __future__ import annotations

import argparse
from pathlib import Path

from src.data.io import write_jsonl
from src.modeling import ModelConfig, load_base_model, load_tokenizer
from src.training import generate_answer


DEFAULT_PROMPTS = [
    "Answer with only the capital city: What is the capital of France?",
    "Answer with only the result: What is 7 plus 8?",
    "Answer with only the color: What color is a clear daytime sky usually described as?",
    "Answer with only yes or no: Is water made of hydrogen and oxygen?",
    "Answer with only the next number: 2, 4, 6, 8,",
    "Answer with only the animal: What animal says meow?",
    "Answer with only the month: Which month comes after March?",
    "Answer with only true or false: The Earth orbits the Sun.",
]


def build_original_retention_set(model, tokenizer, output_path: Path, repeats: int = 3) -> list[dict]:
    records = []
    for idx, prompt in enumerate(DEFAULT_PROMPTS):
        answers = [generate_answer(model, tokenizer, prompt, max_new_tokens=16) for _ in range(repeats)]
        if len(set(answers)) == 1 and answers[0]:
            records.append(
                {
                    "id": f"eval_original_{idx:06d}",
                    "task_family": "eval_original",
                    "task_id": "base_phi4_mini",
                    "split": "eval_original",
                    "prompt": prompt,
                    "target": answers[0],
                    "answer_key": answers[0],
                    "metadata": {"validator": {"type": "regex", "pattern": r".+"}, "source": "base_model_stable"},
                }
            )
    write_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval_original retention data from base model stable outputs.")
    parser.add_argument("--model-name", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--cache-dir", default="data/models")
    parser.add_argument("--output-path", type=Path, default=Path("data/generated/eval_original.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_base_model(ModelConfig(name=args.model_name, cache_dir=args.cache_dir, load_in_4bit=True))
    build_original_retention_set(model, tokenizer, args.output_path, repeats=args.repeats)


if __name__ == "__main__":
    main()
