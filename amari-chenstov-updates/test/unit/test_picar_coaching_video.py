"""CPU-only saved-media checks, with a fake snapshot renderer and tiny MP4s."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from numerical_experiment.picar_coaching import video


class Frames:
    def __init__(self):
        self.reads = []

    def get(self, reference):
        self.reads.append(reference)
        # Different camera and top-down pixels catch accidental synthetic camera use.
        rgb = np.zeros((24, 40, 3), dtype=np.uint8)
        rgb[:] = (230, 30, 20)
        rgb[0, 0] = (255, 255, 255)
        return rgb


class Renderer:
    restores = []

    def __init__(self, snapshot):
        self.episode_time_s = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def restore(self, snapshot):
        self.restores.append(snapshot["time"])
        self.episode_time_s = snapshot["time"]

    def render(self):
        raise AssertionError("must use original saved camera, never a newly rendered camera")

    def render_topdown(self):
        rgb = np.zeros((30, 30, 3), dtype=np.uint8)
        rgb[:] = (20, 220, 30)
        return rgb


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "new-unit"
    root.mkdir()
    return SimpleNamespace(root=root, completed=False, frames=Frames())


@pytest.fixture
def episode():
    return {"episode_id": "training:seed-7:episode-2", "seed": 7, "success": False, "n": 512,
            "difficulty_start": .5, "difficulty_end": .55,
            "frames": [{"time": t, "config": {}, "run_id": "saved"} for t in (1., 1.5)],
            "rows": [{"frame_ref": f"sha256:frame-{i}", "start_time": t - .5, "end_time": t, "elapsed": .5,
                      "action": [0., 0., -1., 1.], "requested_controls": {"steering_rad": -.2, "drive_mps": .1},
                      "executed_controls": {"steering_rad": -.1, "drive_mps": .1},
                      "reward_parts": {"task_raw": .4, "guidance_raw": 2. if i else 0., "total_encoded": .24 if i else .04},
                      "compliance": {"judgment": "adherent", "computed_at": 99.} if i else None,
                      "feature": "not part of compact media source"}
                     for i, t in enumerate((1., 1.5))],
            "coaching_timeline": [{"instruction_id": "cue-1", "text": "Turn left toward the ball.",
                                   "issued_at_s": .7, "available_at_s": .8, "status": "active", "reason": "clear left passage"}]}


@pytest.fixture(autouse=True)
def fake_renderer(monkeypatch):
    Renderer.restores = []
    monkeypatch.setattr(video, "_renderer_factory", Renderer)


def render(store, episode, **kwargs):
    return video.render_video(store, episode, condition="G", phase="development", checkpoint=512,
                              fps=2., panel_size=(160, 120), **kwargs)[0]


def test_sources_save_all_snapshots_actions_camera_refs_and_cue_strips(store, episode):
    original = copy.deepcopy(episode)
    source = video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)
    payload = json.loads((store.root / source["path"]).read_text())
    assert payload["frames"] == episode["frames"]
    assert [r["frame_ref"] for r in payload["rows"]] == [r["frame_ref"] for r in episode["rows"]]
    assert all("feature" not in r for r in payload["rows"])
    assert source["cue_strips"] == 1
    assert payload["cue_strips"][0]["frame_refs"] == [r["frame_ref"] for r in episode["rows"]]
    assert video.load_saved_episode(store, source)["rows"] == payload["rows"]
    assert episode == original
    assert not list(store.root.rglob("*.mp4"))


def test_source_is_deterministic_and_cannot_be_changed_in_place(store, episode):
    source = video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)
    assert source == video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)
    episode["rows"][0]["executed_controls"]["drive_mps"] = 0
    with pytest.raises(ValueError, match="differs"):
        video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)


def test_synchronized_original_camera_and_topdown_resume_validated_without_rerender(store, episode, monkeypatch):
    import imageio.v2 as imageio

    result = render(store, episode)
    assert Renderer.restores == [1., 1.5]
    assert store.frames.reads[-2:] == [r["frame_ref"] for r in episode["rows"]]
    assert result["frames"] == 2 and result["source"]["snapshots"] == 2
    with imageio.get_reader(store.root / result["path"]) as reader:
        frame = reader.get_data(0)
    assert frame.shape[:2] == tuple(result["shape"])
    assert frame[60, 80, 0] > 180 and frame[60, 80, 1] < 80
    assert frame[60, 240, 1] > 160 and frame[60, 240, 0] < 80
    monkeypatch.setattr(video, "_renderer_factory", lambda _: (_ for _ in ()).throw(AssertionError("resume rendered")))
    again = render(store, episode, selection="failure_gallery")
    assert again["path"] == result["path"] and again["sha256"] == result["sha256"]
    assert len(list(store.root.rglob("*.mp4"))) == 1
    assert not list(store.root.rglob(".fix6-encode-*"))


def test_completed_units_and_sealed_parents_cannot_receive_media(store, episode):
    store.completed = True
    with pytest.raises(RuntimeError, match="immutable"):
        render(store, episode)
    store.completed = False
    (store.root.parent / "COMPLETED").touch()
    with pytest.raises(RuntimeError, match="immutable"):
        render(store, episode)
    assert not list(store.root.rglob("*.mp4"))


def test_snapshot_clock_mismatch_rejected_no_final_media_or_temp(store, episode):
    episode["frames"][1]["time"] += .01
    with pytest.raises(ValueError, match="not synchronized"):
        render(store, episode)
    assert not list(store.root.rglob("*.mp4"))
    assert not list(store.root.rglob(".fix6-encode-*"))
    assert list(store.root.rglob("*.episode.json"))  # Original evidence retained for diagnosis.


@pytest.mark.parametrize("field,value", [("end_time", float("nan")), ("end_time", -1.),
                                         ("camera_time_s", 8.), ("topdown_time_s", 8.)])
def test_bad_or_unsynchronized_times_fail(store, episode, field, value):
    episode["rows"][0][field] = value
    with pytest.raises(ValueError):
        render(store, episode)


def test_nonincreasing_frame_times_and_missing_snapshot_rejected(store, episode):
    episode["rows"][1]["end_time"] = 1
    with pytest.raises(ValueError, match="increasing"):
        render(store, episode)
    episode["frames"].pop()
    with pytest.raises(ValueError, match="snapshot"):
        render(store, episode)


def test_no_future_guidance_or_retrospective_reward_backdated(episode):
    row = episode["rows"][1]
    before = video.subtitle_lines(episode, row, .5, condition="G", phase="development", checkpoint=512, selection="preselected")
    assert not any("Turn left" in line for line in before)
    assert not any("adherent" in line for line in before)
    after = video.subtitle_lines(episode, row, 1.5, condition="G", phase="development", checkpoint=512, selection="preselected")
    assert any("Exact delivered guidance: Turn left toward the ball." in line for line in after)
    assert any("RETROSPECTIVE Qwen judgment" in line and "adherent" in line for line in after)
    assert any("Requested:" in line and "-0.2" in line for line in after)
    assert any("Executed calibrated controls:" in line and "-0.1" in line for line in after)
    assert any("guidance_raw" in line and "task_raw" in line for line in after)


@pytest.mark.parametrize("metadata", [{"reward_mode": "task_only"}, {"adherence_enabled": False},
    {"protocol": {"reward_mode": "task_only", "adherence_enabled": False, "compliance_raw_magnitude": 0}}])
def test_task_only_saved_source_preserves_mode_cues_and_disables_judgment_labels(store, episode, metadata):
    episode.update(metadata)
    for row in episode["rows"]:
        row["reward_parts"]["guidance_raw"] = 0.
        row["reward_parts"]["total_encoded"] = .04
    original = copy.deepcopy(episode)
    source = video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)
    saved = video.load_saved_episode(store, source)
    assert saved["reward_mode"] == "task_only" and saved["adherence_enabled"] is False
    assert saved["compliance_raw_magnitude"] == 0
    assert saved["coaching_timeline"] == episode["coaching_timeline"]
    assert saved["rows"] == json.loads((store.root / source["path"]).read_bytes())["rows"]
    for at in (.5, 1.5):
        lines = video.subtitle_lines(saved, saved["rows"][1], at, condition="G", phase="development",
                                     checkpoint=512, selection="preselected")
        assert any("Adherence: disabled by design" in line for line in lines)
        assert not any("Qwen" in line or "adherent" in line for line in lines)
        if at == 1.5:
            assert "Exact delivered guidance: Turn left toward the ball." in lines
            assert "Geometric reason: clear left passage" in lines
            assert any("status=active" in line for line in lines)
            assert any("Requested:" in line for line in lines)
            assert any("Executed calibrated controls:" in line for line in lines)
            assert any('"task_raw": 0.4' in line and '"guidance_raw": 0.0' in line for line in lines)
        else:
            assert not any("Turn left" in line for line in lines)
            assert "Reward parts: pending at this physical time" in lines
    assert episode == original


def test_task_only_store_protocol_survives_encoding_and_regeneration_without_config(store, episode, monkeypatch):
    store.config = {"protocol": {"reward_mode": "task_only", "adherence_enabled": False, "compliance_raw_magnitude": 0}}
    for row in episode["rows"]:
        row["reward_parts"]["guidance_raw"] = 0.
    original = copy.deepcopy(episode)
    captured, subtitle = [], video.subtitle_lines

    def capture(*args, **kwargs):
        lines = subtitle(*args, **kwargs)
        captured.append(lines)
        return lines

    monkeypatch.setattr(video, "subtitle_lines", capture)
    result = render(store, episode)
    assert captured and all(any("disabled by design" in line for line in lines) for lines in captured)
    assert not any("Qwen" in line for lines in captured for line in lines)
    assert result["frames"] == 2 and video._validate_video(store.root / result["path"], result)
    saved = video.load_saved_episode(store, result["source"])
    del store.config
    assert saved["adherence_enabled"] is False and saved["reward_mode"] == "task_only"
    assert render(store, saved)["sha256"] == result["sha256"]
    assert episode == original


def test_legacy_absent_judgment_is_not_relabelled_disabled(episode):
    episode["rows"][0]["reward_parts"]["guidance_raw"] = 0.
    lines = video.subtitle_lines(episode, episode["rows"][0], .5, condition="U", phase="assessment",
                                 checkpoint=512, selection="preselected")
    assert "Qwen judgment / reward parts: pending at this physical time" in lines
    assert not any("disabled by design" in line for line in lines)


def test_completed_cue_cannot_be_marked_completed_before_its_end(episode):
    episode["coaching_timeline"][0].update(status="completed", ended_at_s=1.25)
    lines = video.subtitle_lines(episode, episode["rows"][0], 1., condition="G", phase="development", checkpoint=512,
                                 selection="preselected")
    assert any("status=active" in line for line in lines)
    lines = video.subtitle_lines(episode, episode["rows"][1], 1.5, condition="G", phase="development", checkpoint=512,
                                 selection="preselected")
    assert any("status=completed" in line for line in lines)


def test_instruction_ledger_completion_keeps_original_text_and_geometric_reason(episode):
    episode.pop("coaching_timeline")
    episode["coaching_events"] = [
        {"event_id": "cue-a", "kind": "instruction", "text": "Turn left.", "utterance_start": .7,
         "utterance_end": .7, "available_at": .8},
        {"event_id": "cue-a:completed", "target_id": "cue-a", "kind": "completed", "text": "Guidance completed: Turn left.",
         "utterance_start": 1.2, "utterance_end": 1.2, "available_at": 1.2}]
    episode["coaching_audits"] = {"cue-a": {"reason": "barrier blocks right"}}
    lines = video.subtitle_lines(episode, episode["rows"][0], 1., condition="G", phase="development", checkpoint=512,
                                 selection="preselected")
    assert any("status=active" in line for line in lines)
    lines = video.subtitle_lines(episode, episode["rows"][1], 1.5, condition="G", phase="development", checkpoint=512,
                                 selection="preselected")
    assert "Exact delivered guidance: Turn left." in lines
    assert any("status=completed" in line for line in lines)
    assert "Geometric reason: barrier blocks right" in lines


def test_episode_compliance_receipt_only_on_its_closing_row_and_stage_parts(episode):
    for i, row in enumerate(episode["rows"]):
        row.update(id=f"transition-{i}")
        row.pop("compliance")
        row.pop("reward_parts")
    episode.update(task_stage_rewards=[1., 2.], guidance_stage_rewards=[0., -2.], raw_stage_rewards=[1., 0.],
                   coded_stage_rewards=[.1, 0.], compliance={"window-1": {
                       "closing_row": "transition-1", "judgment": {"judgment": "violated", "reason": "did not turn"},
                       "receipt": {"raw": -2., "encoded": -.2, "transition_id": "transition-1"},
                       "evidence": {"end_time": 1.5}}})
    before = video.subtitle_lines(episode, episode["rows"][0], 1., condition="G", phase="development", checkpoint=512,
                                  selection="preselected")
    assert not any("violated" in line for line in before)
    after = video.subtitle_lines(episode, episode["rows"][1], 1.5, condition="G", phase="development", checkpoint=512,
                                 selection="preselected")
    assert any("RETROSPECTIVE Qwen" in line and "violated" in line and "window-1" in line for line in after)
    assert any('"guidance_raw": -2.0' in line and '"task_raw": 2.0' in line for line in after)


def test_letterboxing_preserves_all_edges_and_long_subtitles_are_not_cropped():
    rgb = np.zeros((30, 120, 3), dtype=np.uint8)
    rgb[:] = (200, 20, 10)
    rgb[:, :5] = rgb[:, -5:] = (255, 255, 255)
    long = "Exact delivered guidance: " + "a" * 500 + "\nEnd of exact instruction."
    frame, layout = video.compose_frame(rgb, rgb, [long], panel_size=(160, 120))
    assert frame.shape[0] == 120 + layout["band_height"]
    assert layout["required_height"] <= layout["band_height"]
    assert "".join(layout["lines"]) == long.replace("\n", "")
    assert np.all(frame[60, 2] == 255) and np.all(frame[60, 157] == 255)
    assert np.all(frame[5, 80] == [23, 25, 27])  # Letterbox, not stretched/cropped.
    with pytest.raises(ValueError, match="never be cropped"):
        video.compose_frame(rgb, rgb, [long], panel_size=(160, 120), band_height=20)


def test_subtitle_band_is_stable_across_short_and_long_timeline(store, episode):
    episode["coaching_timeline"].append({"text": "Steer right. " * 20, "available_at": 1.2,
                                        "issued_at": 1.2, "status": "new"})
    result = render(store, episode)
    assert video._validate_video(store.root / result["path"], result)


def test_corrupt_video_and_source_are_not_silently_skipped(store, episode):
    result = render(store, episode)
    (store.root / result["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        render(store, episode)
    source = result["source"]
    (store.root / source["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="missing or changed"):
        video.load_saved_episode(store, source)


def test_interrupted_encoding_restarts_at_same_deterministic_path(store, episode, monkeypatch):
    compose = video.compose_frame
    attempts = []

    def interrupted(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 2:
            raise InterruptedError("clean stop")
        return compose(*args, **kwargs)

    monkeypatch.setattr(video, "compose_frame", interrupted)
    with pytest.raises(InterruptedError, match="clean stop"):
        render(store, episode)
    assert not list(store.root.rglob("*.mp4")) and not list(store.root.rglob(".fix6-encode-*"))
    source_paths = list(store.root.rglob("*.episode.json"))
    monkeypatch.setattr(video, "compose_frame", compose)
    result = render(store, episode)
    assert Path(result["path"]).stem == source_paths[0].name.removesuffix(".episode.json")
    assert video._validate_video(store.root / result["path"], result)


def test_interrupted_receipt_publication_adopts_valid_video_without_rerender(store, episode, monkeypatch):
    publish = video._publish_bytes

    def interrupted(store, path, data):
        if str(path).endswith(".video.json"):
            raise InterruptedError("receipt interrupted")
        return publish(store, path, data)

    monkeypatch.setattr(video, "_publish_bytes", interrupted)
    with pytest.raises(InterruptedError):
        render(store, episode)
    assert len(list(store.root.rglob("*.mp4"))) == 1
    monkeypatch.setattr(video, "_publish_bytes", publish)
    monkeypatch.setattr(video, "_renderer_factory", lambda _: (_ for _ in ()).throw(AssertionError("must adopt")))
    result = render(store, episode)
    assert video._validate_video(store.root / result["path"], result)


def test_symlinked_output_directory_is_not_used(store, episode, tmp_path):
    outside = tmp_path / "old-unit"
    outside.mkdir()
    (store.root / "coaching_media").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        render(store, episode)
    assert list(outside.iterdir()) == []


def test_input_source_does_not_need_torch_or_training_imports(store, episode, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("numerical_experiment.picar_head_awac"):
            raise AssertionError("media assembly must not load learner")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    source = video.save_episode(store, episode, condition="G", phase="development", checkpoint=512)
    assert video.load_saved_episode(store, source)["episode_id"] == episode["episode_id"]
