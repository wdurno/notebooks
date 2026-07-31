"""Validate and execute the artifact-only results notebook without Jupyter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

BANNED_CODE_FRAGMENTS = (
    "mnist_experiment.run_",
    "src.derivatives",
    "src.fisher",
    "src.mnist_data",
    "src.mnist_model",
    "src.reference",
    "torch.load",
    "download=",
)


def _cell_source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def load_notebook(path: str | Path) -> Mapping[str, Any]:
    notebook_path = Path(path)
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read notebook {notebook_path}: {exc}") from exc
    if not isinstance(notebook, Mapping) or notebook.get("nbformat") != 4:
        raise RuntimeError("results notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("results notebook must contain cells")
    return notebook


def validate_notebook_source(notebook: Mapping[str, Any]) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        violations = [
            fragment for fragment in BANNED_CODE_FRAGMENTS if fragment in source
        ]
        if violations:
            raise RuntimeError(
                f"code cell {index} contains banned operations: {violations}"
            )


def execute_notebook_cells(
    notebook: Mapping[str, Any],
    *,
    notebook_path: Path,
) -> float:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    namespace: dict[str, Any] = {
        "__name__": "__results_notebook__",
        "__file__": str(notebook_path),
    }
    started = time.perf_counter()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        try:
            exec(
                compile(source, f"{notebook_path}:cell-{index}", "exec"),
                namespace,
            )
        except BaseException as exc:
            raise RuntimeError(f"notebook code cell {index} failed: {exc}") from exc
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).with_name("results.ipynb"),
    )
    parser.add_argument("--max-seconds", type=float, default=20.0)
    arguments = parser.parse_args()

    notebook = load_notebook(arguments.notebook)
    validate_notebook_source(notebook)
    elapsed = execute_notebook_cells(
        notebook,
        notebook_path=arguments.notebook.resolve(),
    )
    if elapsed > arguments.max_seconds:
        raise RuntimeError(
            f"notebook took {elapsed:.3f}s; limit is {arguments.max_seconds:.3f}s"
        )
    print(
        json.dumps(
            {
                "notebook": str(arguments.notebook),
                "code_cells": sum(
                    cell.get("cell_type") == "code"
                    for cell in notebook["cells"]
                ),
                "elapsed_seconds": elapsed,
                "status": "valid",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
