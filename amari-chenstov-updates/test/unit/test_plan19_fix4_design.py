"""Documentation checks only; no learner, simulation or artifact generation."""

from pathlib import Path

import nbformat
import pytest

from docs.plan19_fix4_diagram import BOUNDARIES, design_values


PROJECT = Path(__file__).resolve().parents[2]
PI = PROJECT / "numerical_experiment" / "picar"
IMAGE_LINK = "../../docs/figures/plan19_phase3_fix4_transition.png"


def test_documented_schedule_reaches_each_endpoint_and_never_returns_to_a():
    assert BOUNDARIES == (0, 128, 768, 1024, 1664, 2048)
    assert design_values(0) == pytest.approx((1., 1., 0.))
    assert design_values(128) == pytest.approx((1., 1., 0.))
    assert design_values(768) == pytest.approx((.08, 1., 0.))
    assert design_values(1024) == pytest.approx((.08, 1., 0.))
    assert design_values(1664) == pytest.approx((.08, .65, 1.))
    assert design_values(3000) == pytest.approx((.08, .65, 1.))
    rows = [design_values(n) for n in range(2049)]
    assert all(a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] for a, b in zip(rows, rows[1:]))


def test_plan_and_notebook_share_draft_diagram_and_clear_motivation():
    plan = (PI / "plan19.md").read_text()
    book = nbformat.read(PI / "plan19_results.ipynb", as_version=4)
    drafts = [c for c in book.cells if c.metadata.get("plan19_phase3_fix4_draft")]
    assert len(drafts) == 1
    source = drafts[0].source
    assert IMAGE_LINK in plan and IMAGE_LINK in source
    assert (PI / IMAGE_LINK).resolve().is_file()
    assert "**Motivation:**" in source
    assert "never settled" in source and "held-out" in source
    assert "not executed" in source and "FD" in source
    assert book.cells[-1] == drafts[0]
    assert book.cells[-2].source.startswith("## Phase 3 Fix 3 Conclusion")
