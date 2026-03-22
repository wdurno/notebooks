from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SnapshotLoadResult:
    path: Path
    loaded_trainable_keys: list[str]
    has_ssr_state: bool


class SnapshotStore:
    """Persist tunable model state, SSR statistics, and replay metadata."""

    def __init__(self, run_dir: Path, *, max_keep: int = 3):
        self.run_dir = Path(run_dir)
        self.snapshots_dir = self.run_dir / "snapshots"
        self.run_meta_path = self.run_dir / "run_meta.json"
        self.max_keep = max(1, int(max_keep))
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        _atomic_write_json(self.run_meta_path, metadata)

    def save_snapshot(
        self,
        *,
        model: Any,
        replay_buffer: Any,
        step_index: int,
        t: float,
        reason: str,
        memorize_count: int | None = None,
    ) -> Path:
        reason_slug = _slug(reason)
        suffix = f"-mem-{int(memorize_count):04d}" if memorize_count is not None else ""
        filename = f"snapshot-step-{int(step_index):06d}-{reason_slug}{suffix}.pt"
        path = self.snapshots_dir / filename
        payload = {
            "snapshot_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step_index": int(step_index),
            "t": float(t),
            "reason": reason,
            "memorize_count": memorize_count,
            "tunable_state_dict": _collect_trainable_state(model),
            "ssr_state": _collect_ssr_state(model),
            "replay_metadata": _collect_replay_metadata(replay_buffer),
        }
        torch.save(payload, path)
        self._enforce_retention()
        return path

    def load_into_model(self, *, snapshot_path: Path, model: Any) -> SnapshotLoadResult:
        payload = torch.load(snapshot_path, map_location="cpu")
        trainable_state = payload.get("tunable_state_dict", {})
        model_state = model.state_dict()
        loaded_keys: list[str] = []
        for name, value in trainable_state.items():
            if name not in model_state:
                continue
            model_state[name] = value
            loaded_keys.append(name)
        model.load_state_dict(model_state, strict=False)

        ssr_state = payload.get("ssr_state")
        has_ssr_state = bool(ssr_state)
        if has_ssr_state and hasattr(model, "load_ssr_dict"):
            model.load_ssr_dict(ssr_state)
        return SnapshotLoadResult(
            path=Path(snapshot_path),
            loaded_trainable_keys=loaded_keys,
            has_ssr_state=has_ssr_state,
        )

    def _enforce_retention(self) -> None:
        snapshot_paths = sorted(self.snapshots_dir.glob("*.pt"), key=lambda path: path.name)
        overflow = len(snapshot_paths) - self.max_keep
        if overflow <= 0:
            return None
        for path in snapshot_paths[:overflow]:
            path.unlink(missing_ok=True)
        return None


def resolve_snapshot_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Snapshot path does not exist: {candidate}")
    nested_snapshots = candidate / "snapshots"
    search_dir = nested_snapshots if nested_snapshots.is_dir() else candidate
    matches = sorted(search_dir.glob("*.pt"), key=lambda item: item.name)
    if not matches:
        raise FileNotFoundError(f"No .pt snapshots found under: {search_dir}")
    return matches[-1]


def resolve_latest_snapshot_from_model_root(model_root: Path) -> Path:
    root = Path(model_root)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"Model root does not exist: {root}")

    candidate_paths: list[Path] = []
    try:
        candidate_paths.append(resolve_snapshot_path(root))
    except FileNotFoundError:
        pass

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            candidate_paths.append(resolve_snapshot_path(child))
        except FileNotFoundError:
            continue

    if not candidate_paths:
        raise FileNotFoundError(f"No snapshots found under model root: {root}")

    unique_paths = sorted({path.resolve() for path in candidate_paths}, key=lambda item: str(item))
    latest_path = max(unique_paths, key=_snapshot_sort_key)
    return latest_path


def _collect_trainable_state(model: Any) -> dict[str, torch.Tensor]:
    state = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state[name] = parameter.detach().cpu()
    return state


def _collect_ssr_state(model: Any) -> dict[str, Any] | None:
    if not hasattr(model, "ssr_dict"):
        return None
    raw_state = model.ssr_dict()
    if raw_state is None:
        return None
    state: dict[str, Any] = {}
    for key, value in raw_state.items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu()
        else:
            state[key] = value
    return state


def _collect_replay_metadata(replay_buffer: Any) -> dict[str, Any]:
    metadata = {
        "type": type(replay_buffer).__name__,
        "size": int(len(replay_buffer)),
    }
    for key in ("capacity", "n"):
        if hasattr(replay_buffer, key):
            metadata[key] = int(getattr(replay_buffer, key))
    return metadata


def _snapshot_sort_key(path: Path) -> tuple[datetime, float, str]:
    created_at = _read_snapshot_created_at(path)
    if created_at is None:
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return created_at, float(path.stat().st_mtime), path.name


def _read_snapshot_created_at(path: Path) -> datetime | None:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    value = payload.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _slug(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return normalized or "snapshot"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
