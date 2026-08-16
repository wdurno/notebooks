import json

from src.classification_backfill import (
    classification_backfill_path,
    load_classification_backfill,
    overlay_classification_backfill,
)


def test_classification_backfill_is_content_addressed_and_overlaid(tmp_path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    (source / "COMPLETED").touch()
    (source / "phase8_trajectories.pt").write_bytes(b"immutable trajectory")
    output_root = tmp_path / "backfills"
    destination = classification_backfill_path(source, output_root)
    destination.mkdir(parents=True)
    (destination / "COMPLETED").touch()
    artifact = {
        "schema_version": 1,
        "backfill_id": destination.name,
        "source_run_id": source.name,
        "source_trajectory_sha256": destination.name.rsplit("__", 1)[-1],
        "rows": [],
    }
    # The suffix is an identity hash, not the source-file hash; obtain the
    # expected source hash through the loader's path contract.
    import hashlib

    artifact["source_trajectory_sha256"] = hashlib.sha256(
        b"immutable trajectory"
    ).hexdigest()
    artifact["rows"] = [
        {
            "method": "rank8",
            "step": 0,
            "p": 0.5,
            "after_nine_ovr_accuracy": 0.8,
            "after_nine_precision": 0.75,
            "after_nine_recall": 0.9,
        }
    ]
    (destination / "classification_metrics.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )

    loaded = load_classification_backfill(source, output_root)
    rows = overlay_classification_backfill(
        [{"method": "rank8", "step": 0, "p": 0.5}],
        source,
        output_root,
    )

    assert loaded is not None
    assert rows[0]["after_nine_precision"] == 0.75
    assert rows[0]["classification_metric_source"] == destination.name
