"""Resumable work-unit orchestration for optional Plan 9 studies."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import RotatedArtifactError
from .artifacts import read_json, write_json


BUNDLE_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    completed_marker: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkUnit":
        expected = {"unit_id", "command", "dependencies", "completed_marker"}
        if set(value) != expected:
            raise RotatedArtifactError("Plan 9 work-unit fields are invalid")
        command = value["command"]
        dependencies = value["dependencies"]
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise RotatedArtifactError("Plan 9 work-unit command is invalid")
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise RotatedArtifactError("Plan 9 work-unit dependencies are invalid")
        result = cls(
            unit_id=str(value["unit_id"]),
            command=tuple(command),
            dependencies=tuple(dependencies),
            completed_marker=str(value["completed_marker"]),
        )
        if (
            not result.unit_id
            or not result.command
            or not result.completed_marker
            or Path(result.completed_marker).is_absolute()
            or ".." in Path(result.completed_marker).parts
        ):
            raise RotatedArtifactError("Plan 9 work-unit identity is invalid")
        return result

    def mapping(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "command": list(self.command),
            "dependencies": list(self.dependencies),
            "completed_marker": self.completed_marker,
        }


@dataclasses.dataclass(frozen=True)
class Bundle:
    bundle_id: str
    units: tuple[WorkUnit, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Bundle":
        if set(value) != {"schema_version", "bundle_id", "units"}:
            raise RotatedArtifactError("Plan 9 bundle fields are invalid")
        if value["schema_version"] != BUNDLE_SCHEMA_VERSION or not isinstance(
            value["units"], list
        ):
            raise RotatedArtifactError("Plan 9 bundle schema is invalid")
        result = cls(
            bundle_id=str(value["bundle_id"]),
            units=tuple(WorkUnit.from_mapping(item) for item in value["units"]),
        )
        identifiers = [unit.unit_id for unit in result.units]
        if not result.bundle_id or len(set(identifiers)) != len(identifiers):
            raise RotatedArtifactError("Plan 9 bundle identities are invalid")
        known: set[str] = set()
        for unit in result.units:
            if any(name not in known for name in unit.dependencies):
                raise RotatedArtifactError("Plan 9 dependencies must precede their unit")
            known.add(unit.unit_id)
        return result

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "units": [unit.mapping() for unit in self.units],
        }


def load_bundle(path: str | Path) -> Bundle:
    value = read_json(Path(path))
    if not isinstance(value, Mapping):
        raise RotatedArtifactError("Plan 9 bundle must be an object")
    return Bundle.from_mapping(value)


def unit_is_complete(repo_root: Path, unit: WorkUnit) -> bool:
    return (repo_root / unit.completed_marker).is_file()


def execute_bundle(
    bundle: Bundle,
    *,
    bundle_path: Path,
    repo_root: Path,
    resume: bool,
    max_wall_minutes: float | None,
    runner: Callable[[Sequence[str]], None] | None = None,
) -> dict[str, Any]:
    if max_wall_minutes is not None and (
        not math.isfinite(max_wall_minutes) or max_wall_minutes <= 0.0
    ):
        raise ValueError("max_wall_minutes must be finite and positive")
    invoke = runner or (
        lambda command: subprocess.run(command, cwd=repo_root, check=True)
    )
    stop_path = bundle_path.parent / "STOP_REQUESTED.json"
    started = time.monotonic()
    completed = {
        unit.unit_id for unit in bundle.units if unit_is_complete(repo_root, unit)
    }
    if completed and not resume:
        raise RotatedArtifactError("bundle has completed units; pass --resume")
    executed: list[str] = []
    skipped: list[str] = []
    for unit in bundle.units:
        if unit.unit_id in completed:
            skipped.append(unit.unit_id)
            continue
        if stop_path.exists():
            break
        if max_wall_minutes is not None and (
            time.monotonic() - started >= 60.0 * max_wall_minutes
        ):
            break
        if any(name not in completed for name in unit.dependencies):
            raise RotatedArtifactError(f"dependencies incomplete for {unit.unit_id}")
        command = [sys.executable if item == "{python}" else item for item in unit.command]
        print(f"Plan 9 unit started: {unit.unit_id}", flush=True)
        invoke(command)
        if not unit_is_complete(repo_root, unit):
            raise RotatedArtifactError(
                f"unit exited without its completion marker: {unit.unit_id}"
            )
        completed.add(unit.unit_id)
        executed.append(unit.unit_id)
        print(f"Plan 9 unit completed: {unit.unit_id}", flush=True)
    return {
        "bundle_id": bundle.bundle_id,
        "executed": executed,
        "skipped": skipped,
        "completed": [unit.unit_id for unit in bundle.units if unit.unit_id in completed],
        "remaining": [unit.unit_id for unit in bundle.units if unit.unit_id not in completed],
        "stop_requested": stop_path.exists(),
        "wall_limit_reached": (
            max_wall_minutes is not None
            and time.monotonic() - started >= 60.0 * max_wall_minutes
        ),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("run", "status", "request-stop", "clear-stop", "tmux-command"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", required=True, type=Path)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--max-wall-minutes", type=float)
        if name == "tmux-command":
            command.add_argument("--session", default="plan9")
            command.add_argument("--max-wall-minutes", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    repo_root = Path(__file__).parents[3]
    bundle_path = arguments.bundle
    if not bundle_path.is_absolute():
        bundle_path = repo_root / bundle_path
    bundle = load_bundle(bundle_path)
    stop_path = bundle_path.parent / "STOP_REQUESTED.json"
    if arguments.operation == "request-stop":
        write_json(stop_path, {"requested_at": time.time(), "bundle_id": bundle.bundle_id})
        print(f"stop requested for {bundle.bundle_id}")
        return
    if arguments.operation == "clear-stop":
        stop_path.unlink(missing_ok=True)
        print(f"stop request cleared for {bundle.bundle_id}")
        return
    if arguments.operation == "status":
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "stop_requested": stop_path.exists(),
                    "units": {
                        unit.unit_id: (
                            "completed" if unit_is_complete(repo_root, unit) else "pending"
                        )
                        for unit in bundle.units
                    },
                },
                indent=2,
            )
        )
        return
    relative_bundle = bundle_path.relative_to(repo_root)
    if arguments.operation == "tmux-command":
        command = [
            sys.executable,
            "-m",
            "mnist_experiment.rotated_mnist.markov_movement.command_center",
            "run",
            "--bundle",
            str(relative_bundle),
            "--resume",
        ]
        if arguments.max_wall_minutes is not None:
            command.extend(["--max-wall-minutes", str(arguments.max_wall_minutes)])
        rendered = " ".join(shlex.quote(item) for item in command)
        log = bundle_path.parent / "command_center.log"
        relative_log = log.relative_to(repo_root)
        print(
            "tmux new-session -d -s "
            f"{shlex.quote(arguments.session)} "
            f"{shlex.quote(rendered + ' 2>&1 | tee -a ' + shlex.quote(str(relative_log)))}"
        )
        return
    result = execute_bundle(
        bundle,
        bundle_path=bundle_path,
        repo_root=repo_root,
        resume=arguments.resume,
        max_wall_minutes=arguments.max_wall_minutes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
