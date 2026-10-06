from mnist_experiment.continual_subgd.phase5r import evaluate_gate


def _endpoint(recall: float, p0_accuracy: float, balanced: float) -> dict[str, float]:
    return {
        "nine_ovr_recall": recall,
        "p0_accuracy": p0_accuracy,
        "nine_ovr_balanced_accuracy": balanced,
    }


def test_phase5r_gate_is_deterministic_and_requires_every_threshold() -> None:
    passing = [_endpoint(0.65, 0.70, 0.75), _endpoint(0.75, 0.80, 0.85)]
    observed, criteria, passed = evaluate_gate(passing)
    assert passed
    assert all(criteria.values())
    assert observed["mean_nine_ovr_recall"] == 0.70

    failing = [_endpoint(0.59, 0.90, 0.90), _endpoint(0.59, 0.90, 0.90)]
    _, criteria, passed = evaluate_gate(failing)
    assert not passed
    assert not criteria["mean_nine_ovr_recall"]
    assert criteria["mean_p0_accuracy"]
    assert criteria["mean_nine_ovr_balanced_accuracy"]
