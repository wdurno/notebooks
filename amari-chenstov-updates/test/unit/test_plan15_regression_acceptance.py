"""Acceptance accounting must not disguise skipped or failed coverage."""

import pytest

from mnist_experiment.fisher_memory_regression import acceptance


def test_junit_requires_all_passed_and_charges_process_overhead(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="4" time="1.5" errors="0" failures="0" skipped="0"/></testsuites>')
    result = acceptance.test_evidence(path)
    assert result["tests"] == 4
    assert result["process_seconds"] == 3.5
    assert result["sha256"]


@pytest.mark.parametrize("field,value", [("tests", "0"), ("errors", "1"), ("failures", "1"),
                                         ("skipped", "1"), ("time", "nan"), ("time", "-1")])
def test_junit_bad_evidence_is_rejected(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    values = dict(tests="1", time="1", errors="0", failures="0", skipped="0")
    values[field] = value
    path = tmp_path / "tests.xml"
    path.write_text('<testsuite ' + " ".join(f'{key}="{item}"' for key, item in values.items()) + '/>')
    with pytest.raises(ValueError):
        acceptance.test_evidence(path)
