"""Presentation migration preserves evidence and cannot revive an old layout."""

from copy import deepcopy
import hashlib
import json

import nbformat
import pytest

from mnist_experiment.fisher_memory_regression import report


def original():
    def cell(name, text):
        return dict(cell_type="markdown", id=f"plan15-phase8-{name}",
                    source=text, metadata={report.TAG: True})

    return dict(nbformat=4, nbformat_minor=5, metadata={"keep": [1, 2],
                "plan15_phase8_diagnostics": {"href": "../cache/report.html"}}, cells=[
        dict(cell_type="markdown", id="original-title", source=["# Earlier work\n"], metadata={}),
        cell("progress", "Completed"),
        dict(cell_type="code", id="original-code", source=["print('unchanged')\n"], metadata={},
             execution_count=3, outputs=[dict(output_type="stream", name="stdout", text=["unchanged\n"])]),
        cell("intro", ["## Phase 8\n", "Original explanation"]),
        {**cell("figure", "![Unchanged](attachment:figure.png)"),
         "attachments": {"figure.png": {"image/png": "aW1hZ2U="}}},
        cell("conclusion", "Budget-averaged replay - FD MSE: unfavorable"),
    ])


def provenance():
    return dict(source_revision="revision", source_notebook_sha256="notebook-digest",
                scientific_source_hash="science-digest", run_root="cache/original", identity_hash="run-digest")


def split(book=None):
    return report.prepare_phase81_notebooks(original() if book is None else book,
        provenance=provenance(), archive_name="fisher_memory_phase8_initial_results.ipynb")


def test_split_preserves_historical_cells_outputs_and_phase8_attachments():
    before = original()
    untouched = deepcopy(before)
    main, archive = split(before)
    assert before == untouched
    owned = [c for c in before["cells"] if report._owned(c)]
    assert archive["cells"][1:] == owned
    old = [c for c in before["cells"] if not report._owned(c)]
    assert [c for c in main["cells"] if c["id"] in {x["id"] for x in old}] == old
    assert not any(report._owned(c) for c in main["cells"])
    assert main["cells"][1]["id"] == "plan15-retention-progress"
    assert [c["id"] for c in main["cells"][-2:]] == [
        "plan15-regression-initial-summary", "plan15-retention-draft"]
    digest = hashlib.sha256(json.dumps(owned, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False).encode()).hexdigest()
    for book, role in ((main, "historical_link"), (archive, "frozen_archive")):
        nbformat.validate(nbformat.from_dict(deepcopy(book)))
        meta = book["metadata"][report.PRESENTATION]
        assert meta["phase8_cells_sha256"] == digest
        assert meta["source_revision"] == provenance()["source_revision"]
        assert meta["role"] == role
        assert book["metadata"]["keep"] == [1, 2]
    assert archive["metadata"]["plan15_phase8_diagnostics"] == before["metadata"]["plan15_phase8_diagnostics"]
    assert "plan15_phase8_diagnostics" not in main["metadata"]


def test_archive_guard_rejects_refresh_without_touching_either_document():
    for book in split():
        previous = deepcopy(book)
        with pytest.raises(ValueError, match="report is archived"):
            report.build_notebook({}, book)
        with pytest.raises(ValueError, match="already archived"):
            split(book)
        assert book == previous


def test_publish_guard_preserves_file_bytes(tmp_path):
    for index, book in enumerate(split()):
        path = tmp_path / f"{index}.ipynb"
        content = json.dumps(book).encode()
        path.write_bytes(content)
        with pytest.raises(ValueError, match="report is archived"):
            report.publish({}, path)
        assert path.read_bytes() == content


@pytest.mark.parametrize("mutation", [
    lambda book: book.update(nbformat=3),
    lambda book: book["cells"].pop(),
    lambda book: book["cells"][-1].update(source="Pending confirmation"),
])
def test_incomplete_or_incompatible_notebook_cannot_be_archived(mutation):
    book = original()
    mutation(book)
    with pytest.raises(ValueError):
        split(book)


def test_archive_requires_recoverable_provenance_and_sibling_path():
    with pytest.raises(ValueError, match="provenance"):
        report.prepare_phase81_notebooks(original(), provenance={}, archive_name="archive.ipynb")
    with pytest.raises(ValueError, match="sibling"):
        report.prepare_phase81_notebooks(original(), provenance=provenance(), archive_name="elsewhere/archive.ipynb")


def test_new_section_is_pending_and_keeps_unfavorable_history_visible():
    main, archive = split()
    text = "\n".join("".join(c["source"]) for c in main["cells"])
    assert "no results yet" in text
    assert "higher budget-averaged novel-domain MSE" in text
    assert "1.3342351463" in text
    assert "0.20" in text
    assert "online training after calibration" in text
    assert "not a discarded experiment" in archive["cells"][0]["source"]
