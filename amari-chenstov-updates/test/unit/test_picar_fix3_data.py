from dataclasses import replace

from numerical_experiment.picar.fix3_data import render_panels
from numerical_experiment.picar.fix3_rollout import evaluation_configs
from numerical_experiment.picar.storage import RunStore
from src.robotics.curriculum import sample_layout


def test_fixed_panels_include_clear_and_blocked_geometry_without_outcome_selection():
    panels = evaluation_configs()
    easy, hard = panels[0], panels[-1]
    assert all(not sample_layout(s, 0., easy["config"])["curriculum"]["direct_path_blocked"] for s in easy["seeds"])
    assert all(sample_layout(s, 0., hard["config"])["curriculum"]["direct_path_blocked"] for s in hard["seeds"])
    assert easy["seeds"] == hard["seeds"]
    assert easy["config"].domain_override == "A"
    assert hard["config"].domain_override == "B"


def test_panel_generation_is_owned_and_resumable(tmp_path):
    with RunStore(tmp_path / "panels", {"test": "fix3"}, checkpoint_policy="journal-v1") as store:
        result = render_panels(store, split="development")
        assert len(result["rows"]) == 20
        first = store.frames.get(result["rows"][0]["frame_ref"])
        assert first.shape == (240, 320, 3)
        assert first.std() > 5
        assert render_panels(store, split="development") == result
        assert {r["state"] for r in result["rows"]} >= {"absent", "detour", "pan-centered"}
