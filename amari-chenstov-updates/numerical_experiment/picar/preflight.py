"""Explicit resource/asset preflight; never imported by the results notebook."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .runtime import runtime_metadata, sha256_file, source_fingerprint


def inspect_environment(*, cuda: bool = False) -> dict:
    result = runtime_metadata()
    result["source_sha256"] = source_fingerprint()
    result["working_disk"] = dict(zip(("total_bytes", "used_bytes", "free_bytes"), shutil.disk_usage(Path.cwd())))
    if cuda:
        import torch
        result["cuda"] = {
            "available": torch.cuda.is_available(), "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_free_total_bytes": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
        }
    return result


def inspect_model_assets(path: str | Path, *, hash_weights: bool = False) -> dict:
    """Verify local shard completeness before allowing a model loader to run."""
    path = Path(path)
    config_path = path / "config.json"
    model_config = json.loads(config_path.read_text())
    if model_config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Plan 19 expects the declared Qwen2.5-VL model family")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    elif (path / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise ValueError("model directory contains no safetensors weights")
    for name in shards:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("model index contains an unsafe shard reference")
        if not (path / name).is_file():
            raise ValueError(f"model shard is missing: {name}")
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")
    if any(not (path / name).is_file() for name in required):
        raise ValueError("model processor/tokenizer assets are incomplete")
    files = {}
    for name in (*required, *shards):
        source = path / name
        files[name] = {"bytes": source.stat().st_size}
        if hash_weights or name not in shards:
            files[name]["sha256"] = sha256_file(source)
    return {"model_type": "qwen2_5_vl", "files": files, "weights_hashed": hash_weights,
            "total_bytes": sum((path / name).stat().st_size for name in shards)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_environment(cuda=args.cuda)
    if args.model_path:
        result["model_assets"] = inspect_model_assets(args.model_path, hash_weights=args.hash_weights)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
