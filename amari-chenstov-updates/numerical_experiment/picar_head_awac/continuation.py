"""Verified orchestration-only continuation; never rewrite the source campaign."""

import copy
import json
import os
from pathlib import Path
import shutil

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import sha256_file
from numerical_experiment.picar.storage import RunStore


ORCHESTRATION_FILES = frozenset({
    "numerical_experiment/picar_head_awac/campaign.py",
    "numerical_experiment/picar_head_awac/config.py",
    "numerical_experiment/picar_head_awac/continuation.py",
})


def validate_identity(previous, current):
    """Only the named scheduler files and explicit lineage may differ."""
    ignored = {"sources", "numerical_revision", "continuation"}
    if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in current.items() if k not in ignored}:
        raise ValueError("continuation changes experimental identity")
    old_sources, new_sources = previous["sources"], current["sources"]
    names = (set(old_sources) | set(new_sources)) - ORCHESTRATION_FILES
    if any(old_sources.get(name) != new_sources.get(name) for name in names):
        raise ValueError("continuation changes numerical sources")


def _key(operation, settings):
    return canonical_json({"operation": operation, "settings": settings})


def _copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError("continuation asset collision: " + str(target))
        return
    # Only content-addressed frame blobs are shared inodes. Other assets copy.
    if source.parent.name == "frames" and source.suffix == ".npy":
        if sha256_file(source) != source.stem:
            raise ValueError("invalid content-addressed source frame")
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copyfile(source, target)


class Continuation:
    """Hold the old orchestration lock while importing its verified units."""

    def __init__(self, root, current_identity):
        self.root = Path(root).resolve()
        path = self.root / "orchestration"
        config = json.loads((path / "config.json").read_text())
        identity = {k: v for k, v in config.items() if k != "checkpoint_policy"}
        validate_identity(identity, current_identity)
        self.parent = RunStore(path, config, resume=True, checkpoint_policy="journal-v1")
        try:
            self.state = self.parent.load_checkpoint(trusted=True)
            if self.state is None or self.parent.completed:
                raise ValueError("continuation requires an unfinished campaign checkpoint")
            self.pointer_sha256 = sha256_file(path / "CURRENT.json")
            self.units = {}
            for unit_config in sorted((self.root / "units").glob("*/config.json")):
                config = json.loads(unit_config.read_text())
                unit_identity = {k: v for k, v in config.items()
                                 if k not in ("checkpoint_policy", "operation", "settings")}
                validate_identity(unit_identity, current_identity)
                key = _key(config["operation"], config["settings"])
                if key in self.units:
                    raise ValueError("ambiguous continuation unit")
                self.units[key] = unit_config.parent
        except BaseException:
            self.close()
            raise

    def close(self):
        self.parent.close()

    def initial_state(self):
        state = copy.deepcopy(self.state)
        state["completed"] = []
        state["continuation"] = {"source": str(self.root),
            "parent_pointer_sha256": self.pointer_sha256,
            "inherited_charged_seconds": state["charged_seconds"]}
        return state

    def restore(self, store, operation, settings):
        """Return old summary/prior charge after importing the exact checkpoint."""
        source = self.units.get(_key(operation, settings))
        if source is None or operation == "software":
            return None, 0.
        with RunStore(source, json.loads((source / "config.json").read_text()),
                      resume=True, checkpoint_policy="journal-v1") as old:
            state = old.load_checkpoint(trusted=True)
            if store._current is None:
                if state is None:
                    raise ValueError("source unit has no committed checkpoint")
                for folder in ("frames", "videos"):
                    for asset in sorted((source / folder).rglob("*")):
                        if asset.is_symlink():
                            raise ValueError("symlink continuation asset")
                        if asset.is_file():
                            _copy_file(asset, store.root / asset.relative_to(source))
                progress = json.loads((old._current["directory"] / "progress.json").read_text())
                store.checkpoint(state, progress=progress)
            receipt = {"source_unit": str(source),
                "source_pointer_sha256": sha256_file(source / "CURRENT.json"),
                "source_completed": old.completed}
            if old.completed:
                summary = json.loads((source / "summary.json").read_text())
                receipt["source_completion_sha256"] = sha256_file(source / "COMPLETED")
                return {**summary, "unit_path": str(store.root), "continuation": receipt}, 0.
            prior = self.state.get("unit_charges", {}).get(str(source), 0.)
            store._write_root("continuation.json", (canonical_json(receipt) + "\n").encode())
            return None, prior


def remaining_training_intervals(summaries, active=None):
    """Nominal acquisition work only; evaluation cost remains a forecast."""
    total = 11 * 1024 + 4 * 128
    for row in summaries:
        settings = row.get("settings", {})
        if row.get("operation") == "trajectory" and settings.get("phase") in {
                "resource_development", "development", "retention_calibration", "confirmation"}:
            total -= settings["total"]
    if active and active.get("stage") != "done":
        total -= min(active.get("n", 0), active.get("total", 1024))
    return max(0, total)
