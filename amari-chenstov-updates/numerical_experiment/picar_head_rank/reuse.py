"""Read sealed source assets without replaying or modifying their journals."""

import hashlib
import io
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_file(path, seal):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != seal["bytes"]:
        raise ValueError("source artifact size/path mismatch: " + str(path))
    if digest(path) != seal["sha256"]:
        raise ValueError("source artifact checksum mismatch: " + str(path))


def receipt(path):
    path = Path(path)
    marker = json.loads((path / "COMPLETED").read_text())
    if digest(path / "manifest.json") != marker["manifest_sha256"]:
        raise ValueError("source completion seal mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("source is not complete")
    from numerical_experiment.picar.config import canonical_json
    config = json.loads((path / "config.json").read_text())
    if hashlib.sha256(canonical_json(config).encode()).hexdigest() != manifest["config_hash"]:
        raise ValueError("source resolved configuration changed")
    for name, seal in manifest["final_files"].items():
        if Path(name).name != name:
            raise ValueError("invalid final artifact name")
        verify_file(path / name, seal)
    current = json.loads((path / "CURRENT.json").read_text())
    if current != manifest["current"]:
        raise ValueError("source CURRENT differs from completion")
    commit = current["commit_id"]
    if len(commit) != 32 or any(x not in "0123456789abcdef" for x in commit):
        raise ValueError("invalid transaction identity")
    transaction = path / "transactions" / commit
    verify_file(transaction / "manifest.json", current["manifest"])
    transaction_manifest = json.loads((transaction / "manifest.json").read_text())
    state_seal = transaction_manifest["recovery_state"]
    return {"unit": str(path), "completion_sha256": digest(path / "COMPLETED"),
            "manifest_sha256": marker["manifest_sha256"], "commit_id": commit,
            "summary_sha256": digest(path / "summary.json"), "state": state_seal}


def load_state(path):
    import torch
    path = Path(path)
    identity = receipt(path)
    state = path / "transactions" / identity["commit_id"] / "state.pt"
    if identity["state"]["bytes"] > 128 * 1024 * 1024:
        raise ValueError("source checkpoint exceeds read bound")
    verify_file(state, identity["state"])
    return torch.load(io.BytesIO(state.read_bytes()), map_location="cpu", weights_only=True)


def source_index(root):
    root = Path(root)
    receipts = [receipt(root / "orchestration")]
    units = []
    for marker in sorted((root / "units").glob("*/COMPLETED")):
        receipts.append(receipt(marker.parent))
        units.append(json.loads((marker.parent / "summary.json").read_text()))
    return units, receipts


def one(units, operation, **settings):
    matches = [s for s in units if s["operation"] == operation
               and all(s["settings"].get(k) == v for k, v in settings.items())]
    if len(matches) != 1:
        raise ValueError(f"ambiguous source {operation}: {settings}")
    return matches[0]
