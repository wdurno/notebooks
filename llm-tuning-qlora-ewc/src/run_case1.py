from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from src.data.generate import generate_case1
from src.data.io import filter_split, read_jsonl, write_jsonl
from src.ewc import estimate_ewc, make_ewc_penalty
from src.metrics import RetentionScores, average_retention, worst_task_retention
from src.modeling import LoraConfigValues, ModelConfig, load_qlora_model, load_tokenizer
from src.training import TrainingConfig, evaluate_examples, evaluate_examples_with_predictions, train_supervised


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_data(config: dict) -> None:
    data_cfg = config["data"]
    task_a_path = Path(data_cfg["task_a_path"])
    task_b_path = Path(data_cfg["task_b_path"])
    expected_count = int(data_cfg.get("n_per_split", 128)) * 4
    if task_a_path.exists() and task_b_path.exists():
        try:
            if len(read_jsonl(task_a_path)) >= expected_count and len(read_jsonl(task_b_path)) >= expected_count:
                return
        except Exception:
            pass
    generate_case1(
        output_dir=Path(data_cfg["generated_dir"]),
        seed=int(data_cfg.get("seed", 13)),
        n_per_split=int(data_cfg.get("n_per_split", 128)),
    )


def metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {"exact_match": 0.0, "token_f1": 0.0, "format_validity": 0.0}
    return {
        "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows),
        "token_f1": sum(float(row["token_f1"]) for row in rows) / len(rows),
        "format_validity": sum(float(row["format_validity"]) for row in rows) / len(rows),
    }


CONFLICTING_TASK_A_RULES = {"route_AB", "route_EF", "route_JK"}


def task_a_conflict_summary(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    conflicting = []
    nonconflicting = []
    for row in rows:
        metadata = row.get("metadata") or {}
        rule_id = metadata.get("rule_id") if isinstance(metadata, dict) else None
        if rule_id in CONFLICTING_TASK_A_RULES:
            conflicting.append(row)
        else:
            nonconflicting.append(row)

    conflicting_metrics = metric_summary(conflicting) if conflicting else None
    nonconflicting_metrics = metric_summary(nonconflicting) if nonconflicting else None
    return {
        "task_a_conflicting_n": len(conflicting),
        "task_a_nonconflicting_n": len(nonconflicting),
        "task_a_conflicting_em": None if conflicting_metrics is None else conflicting_metrics["exact_match"],
        "task_a_nonconflicting_em": None if nonconflicting_metrics is None else nonconflicting_metrics["exact_match"],
    }


def maybe_write_predictions(config: dict, name: str, rows: list[dict[str, object]], ewc_n0: int, ewc_rank: int, ewc_lambda: float) -> None:
    run_dir = Path(config["outputs"]["run_dir"])
    suffix = f"n0-{ewc_n0}_rank-{ewc_rank}_lambda-{ewc_lambda:g}"
    write_jsonl(run_dir / f"predictions_{name}_{suffix}.jsonl", rows)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def completed_keys(metrics_path: str | Path) -> set[tuple[int, int, float]]:
    path = Path(metrics_path)
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((int(record["ewc_n0"]), int(record["ewc_rank"]), float(record["ewc_lambda"])))
    return keys


def run_single(config: dict, ewc_n0: int, ewc_rank: int, ewc_lambda: float, limit_eval: int | None = None) -> dict:
    cleanup_cuda()
    ensure_data(config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_cfg = ModelConfig(**config["model"])
    train_cfg = TrainingConfig(
        max_seq_length=int(config["training"]["max_seq_length"]),
        per_device_batch_size=int(config["training"]["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        epochs=int(config["training"]["epochs"]),
    )
    lora_cfg = LoraConfigValues(
        rank=int(config["training"]["qlora_rank"]),
        alpha=int(config["training"]["qlora_alpha"]),
        dropout=float(config["training"]["qlora_dropout"]),
    )

    tokenizer = load_tokenizer(model_cfg.name, model_cfg.cache_dir)
    model = load_qlora_model(model_cfg, lora_cfg)

    task_a = read_jsonl(config["data"]["task_a_path"])
    task_b = read_jsonl(config["data"]["task_b_path"])
    task_a_train = filter_split(task_a, "train")
    task_a_ewc = filter_split(task_a, "ewc_init")
    task_a_eval = filter_split(task_a, "eval_heldout")
    task_b_train = filter_split(task_b, "train")
    task_b_eval = filter_split(task_b, "eval_heldout")
    if limit_eval is not None:
        task_a_eval = task_a_eval[:limit_eval]
        task_b_eval = task_b_eval[:limit_eval]

    run_start = time.time()
    task_a_train_metrics = train_supervised(model, tokenizer, task_a_train, train_cfg)
    task_a_before_rows = evaluate_examples_with_predictions(model, tokenizer, task_a_eval)
    task_a_before = metric_summary(task_a_before_rows)

    ewc_state = estimate_ewc(
        model=model,
        tokenizer=tokenizer,
        examples=task_a_ewc,
        ewc_n0=ewc_n0,
        ewc_rank=ewc_rank,
        max_seq_length=train_cfg.max_seq_length,
        min_off_diag=float(config["ewc"].get("min_off_diag", 1e-12)),
    )
    task_b_train_metrics = train_supervised(
        model,
        tokenizer,
        task_b_train,
        train_cfg,
        ewc_penalty=make_ewc_penalty(model, ewc_state),
        ewc_lambda=ewc_lambda,
    )
    task_b_after_rows = evaluate_examples_with_predictions(model, tokenizer, task_b_eval)
    task_b_after = metric_summary(task_b_after_rows)
    task_a_after_rows = evaluate_examples_with_predictions(model, tokenizer, task_a_eval)
    task_a_after = metric_summary(task_a_after_rows)
    task_a_before_conflict = task_a_conflict_summary(task_a_before_rows)
    task_a_after_conflict = task_a_conflict_summary(task_a_after_rows)
    maybe_write_predictions(config, "task_a_before", task_a_before_rows, ewc_n0, ewc_rank, ewc_lambda)
    maybe_write_predictions(config, "task_a_after", task_a_after_rows, ewc_n0, ewc_rank, ewc_lambda)
    maybe_write_predictions(config, "task_b_after", task_b_after_rows, ewc_n0, ewc_rank, ewc_lambda)

    original_metrics = None
    original_path = Path(config["data"]["original_eval_path"])
    if original_path.exists():
        original_eval = read_jsonl(original_path)
        if limit_eval is not None:
            original_eval = original_eval[:limit_eval]
        original_metrics = evaluate_examples(model, tokenizer, original_eval)

    retention = RetentionScores(task_a_before["exact_match"], task_a_after["exact_match"])
    retention_ratios = [retention.retention_ratio]
    if original_metrics is not None:
        retention_ratios.append(original_metrics["exact_match"])

    record = {
        "case": "case1",
        "train_n": len(task_b_train),
        "ewc_n0": ewc_state.ewc_n0,
        "er_buffer": 0,
        "qlora_rank": lora_cfg.rank,
        "ewc_rank": ewc_rank,
        "ewc_effective_rank": ewc_state.effective_rank,
        "ewc_lambda": ewc_lambda,
        "target_em": task_b_after["exact_match"],
        "target_token_f1": task_b_after["token_f1"],
        "target_format_validity": task_b_after["format_validity"],
        "task_a_before_em": task_a_before["exact_match"],
        "task_a_after_em": task_a_after["exact_match"],
        "task_a_conflicting_n": task_a_after_conflict["task_a_conflicting_n"],
        "task_a_nonconflicting_n": task_a_after_conflict["task_a_nonconflicting_n"],
        "task_a_conflicting_before_em": task_a_before_conflict["task_a_conflicting_em"],
        "task_a_conflicting_after_em": task_a_after_conflict["task_a_conflicting_em"],
        "task_a_nonconflicting_before_em": task_a_before_conflict["task_a_nonconflicting_em"],
        "task_a_nonconflicting_after_em": task_a_after_conflict["task_a_nonconflicting_em"],
        "forgetting_delta": retention.forgetting_delta,
        "retention_ratio": retention.retention_ratio,
        "average_retention": average_retention(retention_ratios),
        "worst_task_retention": worst_task_retention(retention_ratios),
        "eval_original_em": None if original_metrics is None else original_metrics["exact_match"],
        "peak_vram_mb": peak_vram_mb(),
        "fisher_wall_seconds": ewc_state.fisher_wall_seconds,
        "train_a_wall_seconds": task_a_train_metrics["train_wall_seconds"],
        "train_b_wall_seconds": task_b_train_metrics["train_wall_seconds"],
        "run_wall_seconds": time.time() - run_start,
    }
    append_jsonl(config["outputs"]["metrics_path"], record)
    del model
    cleanup_cuda()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Case 1 EWC n0 sweep.")
    parser.add_argument("--config", default="configs/case1.yaml")
    parser.add_argument("--ewc-n0", type=int)
    parser.add_argument("--ewc-rank", type=int)
    parser.add_argument("--ewc-lambda", type=float)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run only the first configured combination.")
    parser.add_argument("--resume", action="store_true", help="Skip combinations already present in the metrics JSONL.")
    args = parser.parse_args()
    config = load_config(args.config)
    done = completed_keys(config["outputs"]["metrics_path"]) if args.resume else set()

    n0_values = [args.ewc_n0] if args.ewc_n0 is not None else config["ewc"]["n0_values"]
    rank_values = [args.ewc_rank] if args.ewc_rank is not None else config["ewc"]["rank_values"]
    lambda_values = [args.ewc_lambda] if args.ewc_lambda is not None else config["ewc"]["lambda_values"]

    for ewc_n0 in n0_values:
        for ewc_rank in rank_values:
            for ewc_lambda in lambda_values:
                key = (int(ewc_n0), int(ewc_rank), float(ewc_lambda))
                if key in done:
                    print(json.dumps({"skipped_existing": True, "ewc_n0": key[0], "ewc_rank": key[1], "ewc_lambda": key[2]}))
                    continue
                record = run_single(config, int(ewc_n0), int(ewc_rank), float(ewc_lambda), args.limit_eval)
                print(json.dumps(record, sort_keys=True))
                if args.smoke:
                    return


if __name__ == "__main__":
    main()
