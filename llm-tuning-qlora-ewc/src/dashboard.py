from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.io import JsonlExample


COMBO_COLUMNS = ["ewc_n0", "ewc_rank", "ewc_lambda"]
LEGACY_CONFLICTING_RULES = {"route_AB", "route_EF", "route_JK"}
METADATA_COLUMNS = [
    "rule_id",
    "relation_to_next_task",
    "frequency_bucket",
    "composition_type",
    "legacy_conflict_group",
]
PROTECTED_RELATIONS = ["shared", "near_conflict", "rare_rule"]


def available_columns(df: pd.DataFrame, preferred_columns: Iterable[str]) -> list[str]:
    return [column for column in preferred_columns if column in df.columns]


def load_metrics(metrics_path: str | Path) -> pd.DataFrame:
    path = Path(metrics_path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.insert(0, "run_index", range(len(df)))
    combo_columns = available_columns(df, COMBO_COLUMNS)
    if len(combo_columns) == len(COMBO_COLUMNS):
        latest_indices = df.groupby(COMBO_COLUMNS, dropna=False)["run_index"].transform("max")
        combo_counts = df.groupby(COMBO_COLUMNS, dropna=False)["run_index"].transform("count")
        df["is_latest_for_combo"] = df["run_index"] == latest_indices
        df["combo_run_count"] = combo_counts
    else:
        df["is_latest_for_combo"] = True
        df["combo_run_count"] = 1
    return df


def latest_metrics_by_combo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "is_latest_for_combo" in df.columns:
        return df[df["is_latest_for_combo"]].copy().reset_index(drop=True)
    combo_columns = available_columns(df, COMBO_COLUMNS)
    if len(combo_columns) == len(COMBO_COLUMNS) and "run_index" in df.columns:
        return df.sort_values("run_index").groupby(COMBO_COLUMNS, dropna=False).tail(1).reset_index(drop=True)
    return df.copy().reset_index(drop=True)


def duplicate_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not set(COMBO_COLUMNS).issubset(df.columns):
        return pd.DataFrame()
    grouped = (
        df.groupby(COMBO_COLUMNS, dropna=False)["run_index"]
        .agg(run_count="count", first_run_index="min", latest_run_index="max")
        .reset_index()
    )
    return grouped[grouped["run_count"] > 1].reset_index(drop=True)


def prediction_path(
    run_dir: str | Path,
    split_name: str,
    ewc_n0: int,
    ewc_rank: int,
    ewc_lambda: float = 1.0,
) -> Path:
    return Path(run_dir) / f"predictions_{split_name}_n0-{int(ewc_n0)}_rank-{int(ewc_rank)}_lambda-{ewc_lambda:g}.jsonl"


def metadata_by_id(examples: Iterable[JsonlExample]) -> dict[str, dict]:
    return {example.id: example.metadata for example in examples}


def load_predictions(
    run_dir: str | Path,
    split_name: str,
    ewc_n0: int,
    ewc_rank: int,
    ewc_lambda: float = 1.0,
    metadata_by_id: dict[str, dict] | None = None,
) -> pd.DataFrame:
    path = prediction_path(run_dir, split_name, ewc_n0, ewc_rank, ewc_lambda)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return _attach_empty_metadata_columns(df)

    fallback = metadata_by_id or {}
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].map(lambda metadata: metadata if isinstance(metadata, dict) else {})
        missing_metadata = df["metadata"].map(lambda metadata: not metadata)
        if missing_metadata.any() and "id" in df.columns:
            df.loc[missing_metadata, "metadata"] = df.loc[missing_metadata, "id"].map(
                lambda row_id: fallback.get(row_id, {})
            )
    elif "id" in df.columns:
        df["metadata"] = df["id"].map(lambda row_id: fallback.get(row_id, {}))
    else:
        df["metadata"] = [{} for _ in range(len(df))]

    df["rule_id"] = df["metadata"].map(lambda metadata: metadata.get("rule_id"))
    df["relation_to_next_task"] = df["metadata"].map(lambda metadata: metadata.get("relation_to_next_task"))
    df["frequency_bucket"] = df["metadata"].map(lambda metadata: metadata.get("frequency_bucket"))
    df["composition_type"] = df["metadata"].map(lambda metadata: metadata.get("composition_type"))
    df["legacy_conflict_group"] = df.apply(_legacy_conflict_group, axis=1)
    return df


def slice_retention(pred_before: pd.DataFrame, pred_after: pd.DataFrame, metadata_key: str) -> pd.DataFrame:
    if metadata_key not in pred_before.columns or metadata_key not in pred_after.columns:
        return pd.DataFrame()
    if pred_before.empty or pred_after.empty:
        return pd.DataFrame()
    if pred_before[metadata_key].isna().all() or pred_after[metadata_key].isna().all():
        return pd.DataFrame()

    before = _group_exact_match(pred_before, metadata_key, "before")
    after = _group_exact_match(pred_after, metadata_key, "after")
    merged = before.merge(after, on=metadata_key, how="outer")
    merged["n"] = merged[["before_n", "after_n"]].max(axis=1).fillna(0).astype(int)
    merged["forgetting_delta"] = merged["before_em"] - merged["after_em"]
    return merged.sort_values(metadata_key).reset_index(drop=True)


def dataset_inventory(examples: Iterable[JsonlExample]) -> pd.DataFrame:
    rows = [
        {
            "split": example.split,
            "relation_to_next_task": example.metadata.get("relation_to_next_task"),
            "frequency_bucket": example.metadata.get("frequency_bucket"),
            "composition_type": example.metadata.get("composition_type"),
        }
        for example in examples
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def headline_scorecard(
    metric_row: pd.Series,
    task_a_before: pd.DataFrame | None = None,
    task_a_after: pd.DataFrame | None = None,
    task_b_after: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "Target learning",
            "score": _mean_for_slice(task_b_after, "composition_type", ["seen"])[0],
            "n": _mean_for_slice(task_b_after, "composition_type", ["seen"])[1],
            "readout": "Task B eval on seen rule compositions.",
        },
        {
            "metric": "Target heldout composition",
            "score": _mean_for_slice(task_b_after, "relation_to_next_task", ["heldout_composition"])[0],
            "n": _mean_for_slice(task_b_after, "relation_to_next_task", ["heldout_composition"])[1],
            "readout": "Task B eval on recombined attributes.",
        },
        {
            "metric": "Protected retention",
            "score": _mean_for_slice(task_a_after, "relation_to_next_task", PROTECTED_RELATIONS)[0],
            "n": _mean_for_slice(task_a_after, "relation_to_next_task", PROTECTED_RELATIONS)[1],
            "readout": "Task A retention on shared, near-conflict, and rare-rule probes.",
        },
        {
            "metric": "Near-conflict retention",
            "score": _mean_for_slice(task_a_after, "relation_to_next_task", ["near_conflict"])[0],
            "n": _mean_for_slice(task_a_after, "relation_to_next_task", ["near_conflict"])[1],
            "readout": "Task A cases where Task B changed a neighboring condition only.",
        },
        {
            "metric": "Expected overwrite",
            "score": _mean_for_slice(task_b_after, "relation_to_next_task", ["direct_conflict"])[0],
            "n": _mean_for_slice(task_b_after, "relation_to_next_task", ["direct_conflict"])[1],
            "readout": "Task B direct-conflict cases where latest behavior should win.",
        },
        {
            "metric": "Base retention",
            "score": _metric_value(metric_row, "eval_original_em"),
            "n": None,
            "readout": "Original-model retention set exact match.",
        },
        {
            "metric": "Runtime",
            "score": _metric_value(metric_row, "run_wall_seconds"),
            "n": None,
            "readout": "End-to-end run seconds.",
        },
        {
            "metric": "Peak VRAM",
            "score": _metric_value(metric_row, "peak_vram_mb"),
            "n": None,
            "readout": "Peak allocated GPU memory in MB.",
        },
    ]

    scorecard = pd.DataFrame(rows)
    if task_a_before is not None and task_a_after is not None:
        protected_before, _protected_before_n = _mean_for_slice(
            task_a_before, "relation_to_next_task", PROTECTED_RELATIONS
        )
        protected_after, _protected_after_n = _mean_for_slice(task_a_after, "relation_to_next_task", PROTECTED_RELATIONS)
        if protected_before is not None and protected_after is not None:
            scorecard.loc[scorecard["metric"] == "Protected retention", "delta"] = protected_before - protected_after
    return scorecard


def headline_grid_row(
    metric_row: pd.Series,
    task_a_before: pd.DataFrame | None = None,
    task_a_after: pd.DataFrame | None = None,
    task_b_after: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    target_seen_em, target_seen_n = _mean_for_slice(task_b_after, "composition_type", ["seen"])
    target_heldout_em, target_heldout_n = _mean_for_slice(
        task_b_after, "relation_to_next_task", ["heldout_composition"]
    )
    protected_before_em, _protected_before_n = _mean_for_slice(
        task_a_before, "relation_to_next_task", PROTECTED_RELATIONS
    )
    protected_after_em, protected_n = _mean_for_slice(task_a_after, "relation_to_next_task", PROTECTED_RELATIONS)
    near_conflict_em, near_conflict_n = _mean_for_slice(task_a_after, "relation_to_next_task", ["near_conflict"])
    expected_overwrite_em, overwrite_n = _mean_for_slice(
        task_b_after, "relation_to_next_task", ["direct_conflict"]
    )

    protected_drop = None
    if protected_before_em is not None and protected_after_em is not None:
        protected_drop = protected_before_em - protected_after_em

    return {
        "run_index": _metric_int(metric_row, "run_index"),
        "ewc_n0": _metric_int(metric_row, "ewc_n0"),
        "ewc_rank": _metric_int(metric_row, "ewc_rank"),
        "ewc_lambda": _metric_value(metric_row, "ewc_lambda"),
        "target_seen_em": target_seen_em,
        "target_seen_n": target_seen_n,
        "target_heldout_composition_em": target_heldout_em,
        "target_heldout_composition_n": target_heldout_n,
        "protected_retention_em": protected_after_em,
        "protected_retention_drop": protected_drop,
        "protected_retention_n": protected_n,
        "near_conflict_retention_em": near_conflict_em,
        "near_conflict_n": near_conflict_n,
        "expected_overwrite_em": expected_overwrite_em,
        "expected_overwrite_n": overwrite_n,
        "base_retention_em": _metric_value(metric_row, "eval_original_em"),
        "run_wall_seconds": _metric_value(metric_row, "run_wall_seconds"),
        "peak_vram_mb": _metric_value(metric_row, "peak_vram_mb"),
    }


def format_scorecard_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    numeric_formats = {
        "ewc_lambda": "{:.3g}",
        "target_seen_em": "{:.3f}",
        "target_heldout_composition_em": "{:.3f}",
        "protected_retention_em": "{:.3f}",
        "protected_retention_drop": "{:.3f}",
        "near_conflict_retention_em": "{:.3f}",
        "expected_overwrite_em": "{:.3f}",
        "base_retention_em": "{:.3f}",
        "run_wall_seconds": "{:.1f}",
        "peak_vram_mb": "{:.0f}",
    }
    count_formats = {
        "run_index": "{:.0f}",
        "ewc_n0": "{:.0f}",
        "ewc_rank": "{:.0f}",
        "target_seen_n": "{:.0f}",
        "target_heldout_composition_n": "{:.0f}",
        "protected_retention_n": "{:.0f}",
        "near_conflict_n": "{:.0f}",
        "expected_overwrite_n": "{:.0f}",
    }
    return df.style.format({**numeric_formats, **count_formats}, na_rep="")


def _group_exact_match(df: pd.DataFrame, metadata_key: str, prefix: str) -> pd.DataFrame:
    grouped = df.groupby(metadata_key, dropna=True)["exact_match"].agg(["mean", "count"]).reset_index()
    return grouped.rename(columns={"mean": f"{prefix}_em", "count": f"{prefix}_n"})


def _mean_for_slice(df: pd.DataFrame | None, column: str, values: Iterable[str]) -> tuple[float | None, int | None]:
    if df is None or df.empty or column not in df.columns or "exact_match" not in df.columns:
        return None, None
    allowed = set(values)
    sliced = df[df[column].isin(allowed)]
    if sliced.empty:
        return None, 0
    return float(sliced["exact_match"].mean()), int(len(sliced))


def _metric_value(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _metric_int(row: pd.Series, column: str) -> int | None:
    value = _metric_value(row, column)
    if value is None:
        return None
    return int(value)


def _legacy_conflict_group(row: pd.Series) -> str:
    if row.get("relation_to_next_task") == "direct_conflict":
        return "conflicting"
    rule_id = row.get("rule_id")
    if isinstance(rule_id, str) and any(rule_id == rule or rule_id.startswith(f"{rule}_") for rule in LEGACY_CONFLICTING_RULES):
        return "conflicting"
    return "nonconflicting"


def _attach_empty_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["metadata", *METADATA_COLUMNS]:
        df[column] = []
    return df
