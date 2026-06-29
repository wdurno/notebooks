"""Load phase 1 observation runs from old and new layouts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from picar_kl.records import Phase1ObservationRecord


OBSERVATIONS_FILENAME = "observations.jsonl"
RUN_META_FILENAME = "run_meta.json"


def iter_phase1_run_dirs(root: Path) -> Iterator[Path]:
    """Yield directories containing phase 1 observation JSONL files."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if root.name != OBSERVATIONS_FILENAME:
            raise ValueError(f"Expected `{OBSERVATIONS_FILENAME}` file: {root}")
        yield root.parent
        return

    run_dirs = sorted(path.parent for path in root.rglob(OBSERVATIONS_FILENAME))
    yielded: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir in yielded:
            continue
        yielded.add(run_dir)
        yield run_dir


def read_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / RUN_META_FILENAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Run metadata must be a JSON object: {path}")
    return payload


def load_phase1_run(run_dir: Path) -> list[Phase1ObservationRecord]:
    run_dir = Path(run_dir)
    observations_path = run_dir / OBSERVATIONS_FILENAME
    if not observations_path.exists():
        raise FileNotFoundError(observations_path)

    metadata = read_run_metadata(run_dir)
    run_uuid = _run_uuid_from_metadata_or_dir(metadata, run_dir)
    records = []
    with observations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {observations_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Observation row must be a JSON object at {observations_path}:{line_number}")
            records.append(
                _record_from_payload(
                    payload,
                    run_uuid=run_uuid,
                    run_dir=run_dir,
                )
            )
    return records


def iter_phase1_records(root: Path) -> Iterator[Phase1ObservationRecord]:
    for run_dir in iter_phase1_run_dirs(root):
        yield from load_phase1_run(run_dir)


def load_record_image(record: Phase1ObservationRecord) -> Any:
    image_file = record.image_file
    if image_file is None:
        raise ValueError("Record has no image path")
    if not image_file.exists():
        raise FileNotFoundError(image_file)
    if image_file.suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to read compressed phase 1 images") from exc

        payload = np.load(image_file)
        if "image" not in payload:
            raise ValueError(f"Compressed image file lacks `image` array: {image_file}")
        return payload["image"]
    raise ValueError(f"Unsupported phase 1 image format: {image_file}")


def _record_from_payload(
    payload: dict[str, Any],
    *,
    run_uuid: str | None,
    run_dir: Path,
) -> Phase1ObservationRecord:
    action_payload = payload.get("action")
    is_new_record = bool(action_payload and "distribution" in action_payload)
    if is_new_record:
        return Phase1ObservationRecord.from_dict(payload, run_uuid=run_uuid, run_dir=run_dir)
    return Phase1ObservationRecord.from_legacy_dict(payload, run_uuid=run_uuid, run_dir=run_dir)


def _run_uuid_from_metadata_or_dir(metadata: dict[str, Any], run_dir: Path) -> str | None:
    for key in ("uuid", "run_uuid"):
        value = metadata.get(key)
        if value:
            return str(value)
    name = Path(run_dir).name
    if name:
        return name
    return None
