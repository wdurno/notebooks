"""Build the Plan 12 conclusion from validated immutable trajectory artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import _atomic_write, _read_json, _write_json
from .plan12.artifacts import UnitStore
from .plan12.config import Plan12Study


DEFAULT_CONFIG = Path("mnist_experiment/rotated_mnist/plan12/configs/default.json")
DEFAULT_PHASE4_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12")
DEFAULT_Q01_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12_mp_q01_v1")
DEFAULT_NOTEBOOK = Path("mnist_experiment/rotated_mnist/ridge_estimator_health.ipynb")
DEFAULT_REPORT = DEFAULT_Q01_ROOT / "reports/conclusion.json"
REPLICAS = 16
CONDITIONS = (
    "gauge_no_ridge",
    "isotropic_ridge",
    "tail_ridge",
    "mp_q01_isotropic",
    "spectral_selector",
)
LABELS = {
    "gauge_no_ridge": "No added ridge",
    "isotropic_ridge": r"Isotropic $0.1s_t$",
    "tail_ridge": r"Tail $0.1s_t$",
    "mp_q01_isotropic": r"MP $q_{.01}$",
    "spectral_selector": r"MP $q_{.99}$",
}


def _validated_trajectory_paths(
    store: UnitStore,
    ledger_path: Path,
) -> dict[tuple[int, str, str], Path]:
    ledger = _read_json(ledger_path)
    paths = {}
    for item in ledger["items"]:
        unit = item["unit"]
        if item["action"] != "trajectory":
            continue
        index = int(unit["index"])
        schedule = unit["schedule"]
        condition = unit["condition"]
        if index > REPLICAS or schedule not in {"linear", "sigmoid"}:
            continue
        path = store.completed(unit, tuple(item["required"]))
        if path is None:
            raise RuntimeError(f"missing completed trajectory: {unit}")
        paths[(index, schedule, condition)] = path
    return paths


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty conclusion metric")
    return sum(values) / len(values)


def build_conclusion(
    phase4_paths: dict[tuple[int, str, str], Path],
    q01_paths: dict[tuple[int, str, str], Path],
) -> dict[str, Any]:
    rows = []
    by_condition_schedule: dict[tuple[str, str], dict[str, float]] = {}
    for schedule in ("linear", "sigmoid"):
        for condition in CONDITIONS:
            values = []
            for index in range(1, REPLICAS + 1):
                key = (index, schedule, condition)
                path = q01_paths[key] if condition == "mp_q01_isotropic" else phase4_paths[key]
                metrics = _read_json(path / "metrics.json")
                initial, final = metrics[0], metrics[-1]
                trajectory = torch.load(
                    path / "trajectory.pt", map_location="cpu", weights_only=True
                )
                movement = float(
                    trajectory["displacements"].to(torch.float64).square().sum()
                )
                values.append(
                    {
                        "rot30_accuracy_learning_pp": 100.0
                        * (
                            float(final["panel_030_accuracy"])
                            - float(initial["panel_030_accuracy"])
                        ),
                        "rot30_nll_learning": float(initial["panel_030_nll"])
                        - float(final["panel_030_nll"]),
                        "upright_accuracy_change_pp": 100.0
                        * (
                            float(final["panel_000_accuracy"])
                            - float(initial["panel_000_accuracy"])
                        ),
                        "upright_nll_change": float(initial["panel_000_nll"])
                        - float(final["panel_000_nll"]),
                        "cumulative_squared_movement": movement,
                    }
                )
            aggregate = {
                key: _mean([value[key] for value in values]) for key in values[0]
            }
            by_condition_schedule[(condition, schedule)] = aggregate

    for condition in CONDITIONS:
        linear = by_condition_schedule[(condition, "linear")]
        sigmoid = by_condition_schedule[(condition, "sigmoid")]
        no_ridge_linear = by_condition_schedule[("gauge_no_ridge", "linear")]
        no_ridge_sigmoid = by_condition_schedule[("gauge_no_ridge", "sigmoid")]
        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "rot30_accuracy_learning_pp": {
                    "linear": linear["rot30_accuracy_learning_pp"],
                    "sigmoid": sigmoid["rot30_accuracy_learning_pp"],
                },
                "upright_accuracy_change_pp": {
                    "linear": linear["upright_accuracy_change_pp"],
                    "sigmoid": sigmoid["upright_accuracy_change_pp"],
                },
                "movement_percent_of_no_ridge": {
                    "linear": 100.0
                    * linear["cumulative_squared_movement"]
                    / no_ridge_linear["cumulative_squared_movement"],
                    "sigmoid": 100.0
                    * sigmoid["cumulative_squared_movement"]
                    / no_ridge_sigmoid["cumulative_squared_movement"],
                },
                "rot30_nll_learning": {
                    "linear": linear["rot30_nll_learning"],
                    "sigmoid": sigmoid["rot30_nll_learning"],
                },
                "upright_nll_change": {
                    "linear": linear["upright_nll_change"],
                    "sigmoid": sigmoid["upright_nll_change"],
                },
            }
        )

    isotropic = next(row for row in rows if row["condition"] == "isotropic_ridge")
    no_ridge = next(row for row in rows if row["condition"] == "gauge_no_ridge")
    retained_learning = {
        schedule: isotropic["rot30_accuracy_learning_pp"][schedule]
        / no_ridge["rot30_accuracy_learning_pp"][schedule]
        for schedule in ("linear", "sigmoid")
    }
    nll_forgetting_reduction = {
        schedule: 1.0
        - isotropic["upright_nll_change"][schedule]
        / no_ridge["upright_nll_change"][schedule]
        for schedule in ("linear", "sigmoid")
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replicas_per_schedule": REPLICAS,
        "rows": rows,
        "recommendation": {
            "condition": "isotropic_ridge",
            "ridge_rule": "kappa_t = 0.1 * s_t",
            "summary": (
                "Best observed balance: isotropic ridge with kappa_t = 0.1 s_t. "
                "It retains nearly all no-ridge rotated-accuracy learning while reducing "
                "upright probability-quality deterioration."
            ),
            "rotated_accuracy_learning_retained_fraction": retained_learning,
            "upright_nll_forgetting_reduction_fraction": nll_forgetting_reduction,
        },
        "limitations": (
            "The table uses the 16 replicas shared with the post hoc q=.01 probe. "
            "It ranks tested conditions rather than identifying an optimal ridge value; "
            "the literal no-update control remains absent."
        ),
    }


def _pair(value: dict[str, float], *, digits: int = 1, suffix: str = "") -> str:
    return (
        f"{value['linear']:.{digits}f}{suffix} / "
        f"{value['sigmoid']:.{digits}f}{suffix}"
    )


def conclusion_markdown(report: dict[str, Any]) -> str:
    table = [
        "| Condition | $30^\\circ$ accuracy learned, linear / sigmoid | "
        "Upright accuracy change, linear / sigmoid | Movement vs no ridge, linear / sigmoid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        movement = row["movement_percent_of_no_ridge"]
        movement_text = (
            _pair(movement, digits=0, suffix="%")
            if min(movement.values()) >= 1.0
            else _pair(movement, digits=2, suffix="%")
        )
        table.append(
            f"| {row['label']} | "
            f"{_pair(row['rot30_accuracy_learning_pp'], suffix=' pp')} | "
            f"{_pair(row['upright_accuracy_change_pp'], suffix=' pp')} | "
            f"{movement_text} |"
        )
    retained = report["recommendation"][
        "rotated_accuracy_learning_retained_fraction"
    ]
    reduction = report["recommendation"][
        "upright_nll_forgetting_reduction_fraction"
    ]
    return "\n".join(
        [
            "## Conclusion",
            "",
            "**Best balance:** `isotropic_ridge` with",
            "",
            "$$",
            "\\kappa_t=0.1s_t.",
            "$$",
            "",
            *table,
            "",
            "Positive $30^\\circ$ values measure learning from the common initial model. "
            "Negative upright values measure forgetting by the final rotated checkpoint.",
            "",
            "The isotropic ridge retained approximately "
            f"{100 * retained['linear']:.0f}% and {100 * retained['sigmoid']:.0f}% of "
            "no-ridge rotated-accuracy learning under linear and sigmoid schedules. "
            "It reduced upright NLL deterioration by approximately "
            f"{100 * reduction['linear']:.0f}% and {100 * reduction['sigmoid']:.0f}%, "
            "respectively. Tail ridge showed no reliable advantage, while both MP conditions "
            "favored preservation too heavily to be the preferred learning-retention compromise.",
            "",
            report["limitations"],
            "",
        ]
    )


def update_notebook(notebook_path: Path, report_path: Path, report: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if not cell.get("metadata", {}).get("plan12_conclusion")
    ]
    cell = {
        "cell_type": "markdown",
        "metadata": {
            "plan12_conclusion": True,
            "artifact_report": str(report_path.resolve()),
        },
        "source": conclusion_markdown(report).splitlines(keepends=True),
    }
    insertion = next(
        (
            index
            for index, existing in enumerate(notebook["cells"])
            if "## Interpretation guardrails" in "".join(existing.get("source", []))
        ),
        len(notebook["cells"]),
    )
    notebook["cells"].insert(insertion, cell)
    notebook.setdefault("metadata", {}).setdefault("plan12", {})[
        "conclusion_report"
    ] = str(report_path.resolve())
    _atomic_write(
        notebook_path,
        (json.dumps(notebook, indent=1) + "\n").encode("utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--q01-root", type=Path, default=DEFAULT_Q01_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    repo_root = Path(__file__).parents[2]
    study = Plan12Study.from_path(args.config)
    phase4_store = UnitStore(args.phase4_root, study, repo_root)
    q01_store = UnitStore(args.q01_root, study, repo_root)
    phase4_paths = _validated_trajectory_paths(
        phase4_store, args.phase4_root / "ledgers/phase4.json"
    )
    q01_paths = _validated_trajectory_paths(
        q01_store, args.q01_root / "ledgers/phase5_probe16.json"
    )
    report = build_conclusion(phase4_paths, q01_paths)
    _write_json(args.report, report)
    update_notebook(args.notebook, args.report, report)
    print(
        json.dumps(
            {
                "notebook": str(args.notebook),
                "report": str(args.report),
                "recommendation": report["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
