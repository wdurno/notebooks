from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vision-seconds",
        action="store",
        default="30",
        help="Duration for vision integration test in seconds. Negative runs until Ctrl-C.",
    )


@pytest.fixture
def vision_seconds(pytestconfig: pytest.Config) -> float:
    raw_value = pytestconfig.getoption("vision_seconds")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise pytest.UsageError(
            f"Invalid --vision-seconds value {raw_value!r}; expected a number."
        ) from exc
