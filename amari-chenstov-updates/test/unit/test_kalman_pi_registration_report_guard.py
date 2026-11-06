"""The old publisher must not remove the newly appended Phase 6 report."""

import nbformat
import pytest

from mnist_experiment.kalman_pi.retention.supplement.report import _target_bytes


def test_phase5_publisher_preserves_phase6_notebook(tmp_path):
    path = tmp_path / "results.ipynb"
    book = nbformat.v4.new_notebook()
    book.metadata["plan17_registration_report"] = {"schema": 1}
    nbformat.write(book, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="includes Phase 6"):
        _target_bytes(path, tmp_path, {"study_id": "prior"}, None, None)
    assert path.read_bytes() == before
