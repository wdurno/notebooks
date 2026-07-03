"""Precompute Qwen visual-token encodings for phase 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from picar_kl.data.phase1 import iter_phase1_records, load_record_image
from picar_kl.models.visual import QwenVisualTokenEncoder
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.vlm.runtime import QwenRuntime, QwenRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute phase 2 Qwen visual-token encodings.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=None,
        help="Phase 1 data root. May be repeated. Defaults to artifacts/data/phase1.",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/data/phase2/encodings"))
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--vlm-manifest",
        type=Path,
        default=Path("artifacts/manifests/tracked/models/vlm_models.json"),
    )
    parser.add_argument("--vlm-model-name", default="qwen2.5-vl-3b")
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--output-dtype", default="float16")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_roots = args.data_root or [Path("artifacts/data/phase1")]
    runtime = QwenRuntime.from_config(
        QwenRuntimeConfig(
            model_name=args.vlm_model_name,
            model_root=args.model_root,
            manifest_path=args.vlm_manifest,
            allow_downloads=bool(args.allow_model_downloads),
        )
    )
    encoder = QwenVisualTokenEncoder(runtime, output_dtype=args.output_dtype)
    cache = VisualEncodingCache(
        EncodingCacheConfig(
            cache_root=args.cache_root,
            model_name=args.vlm_model_name,
            manifest_path=args.vlm_manifest,
            encoder_id=encoder.encoder_id,
            config_hash=f"output_dtype={args.output_dtype}",
        )
    )

    encoded = 0
    skipped = 0
    for data_root in data_roots:
        for record in iter_phase1_records(data_root):
            if args.limit is not None and encoded >= int(args.limit):
                print(f"[phase2-encodings] encoded={encoded} skipped={skipped} cache={args.cache_root}", flush=True)
                return 0
            if cache.has(record):
                skipped += 1
                continue
            image = load_record_image(record)
            encoding = encoder.encode_image(image)
            entry = cache.store(record, image_shape=tuple(int(item) for item in image.shape), encoding=encoding)
            encoded += 1
            print(
                "[phase2-encodings] "
                f"encoded step={record.step_index} run={record.run_uuid} shape={entry.encoding_shape} key={entry.cache_key}",
                flush=True,
            )
    print(f"[phase2-encodings] encoded={encoded} skipped={skipped} cache={args.cache_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
