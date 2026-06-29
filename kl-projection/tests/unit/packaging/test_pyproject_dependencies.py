import tomllib
from pathlib import Path


def test_base_dependencies_are_robot_light():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    base_dependencies = set(pyproject["project"]["dependencies"])
    server_dependencies = set(pyproject["project"]["optional-dependencies"]["server"])

    assert "torch" not in base_dependencies
    assert "torch" in server_dependencies


def test_robot_and_phase1_entry_points_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]

    assert scripts["picar-kl-robot-server"] == "picar_kl.robot.app:main"
    assert scripts["picar-kl-phase1"] == "picar_kl.phase1.cli:main"
