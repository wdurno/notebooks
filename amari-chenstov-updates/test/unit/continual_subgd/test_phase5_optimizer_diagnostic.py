from mnist_experiment.continual_subgd.phase5_optimizer_diagnostic import CONDITIONS


def test_optimizer_diagnostic_is_complete_two_by_two_factorial() -> None:
    cells = {(condition.optimizer, condition.use_ewc) for condition in CONDITIONS}
    assert cells == {
        ("armijo", False),
        ("armijo", True),
        ("lbfgs", False),
        ("lbfgs", True),
    }
    assert len({condition.name for condition in CONDITIONS}) == 4
