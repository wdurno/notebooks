"""Regenerable Fix 6 media from original camera frames and saved simulator state.

This explicit CPU producer never selects actions, advances a live environment,
loads a learner checkpoint, or infers rewards. Top-down rendering reconstructs
only saved simulator states and is reader-only. Call ``save_episode`` for every
episode, not just those currently selected for a full video.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


VERSION = "plan19-phase3-fix6-video-v1"
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_FRAMES = 10_000
ROW_KEYS = {"id", "frame_ref", "elapsed", "action", "requested_controls", "requested_control", "generated_at",
            "executed_controls", "executed_control", "start_time", "end_time", "time_s",
            "camera_time_s", "topdown_time_s", "coaching_events", "coaching", "guidance",
            "reward_parts", "task_reward", "guidance_reward", "raw_reward", "encoded_reward",
            "compliance", "judgment", "timing", "fallback", "terminal", "truncated",
            "end_reason", "bonus", "arrival", "receipt", "instruction", "geometric_reason",
            "execution_segments", "instruction_ids_seen", "generated_at", "inference_finished_at_s",
            "observed_controls"}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("video source contains a non-JSON value")


def _bytes(value):
    return json.dumps(_plain(value), ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _safe(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    if not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("video paths must stay inside the new unit without symlinks")
    return path


def _writable(store):
    root = Path(store.root).absolute()
    if getattr(store, "completed", False) or any((p / "COMPLETED").exists() for p in (root, *root.parents)):
        raise RuntimeError("completed units are immutable; media requires a writable new unit")
    _safe(root, root)
    if callable(getattr(store, "_writable", None)):
        store._writable()
    return root


def _admit(store, incoming):
    if getattr(store, "_local_guard", None) is not None:
        store._local_guard.check(incoming)


def _refresh(store):
    if callable(getattr(store, "_refresh_local_usage", None)):
        store._refresh_local_usage()


def _publish_bytes(store, path, data):
    """Idempotent exact-content publication; never overwrite another identity."""
    root = _writable(store)
    path = _safe(root, path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("existing media/source differs; use a new unit")
        return
    _admit(store, len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".fix6-source-", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        _refresh(store)


def _identity(episode, condition, phase, checkpoint):
    if condition not in ("U", "G") or phase not in ("initialization", "development", "assessment", "readiness"):
        raise ValueError("video condition/phase must follow the Fix 6 contract")
    if not isinstance(episode.get("episode_id"), str) or not episode["episode_id"]:
        raise ValueError("stable episode_id required")
    return {"version": VERSION, "condition": condition, "phase": phase, "checkpoint": checkpoint,
            "episode_id": episode["episode_id"], "seed": episode.get("seed")}


def _task_only(episode):
    protocol = episode.get("protocol", {})
    return (episode.get("reward_mode", protocol.get("reward_mode")) == "task_only"
            or episode.get("adherence_enabled", protocol.get("adherence_enabled")) is False)


def _media_episode(store, episode):
    """Carry the resolved protocol into regenerable sources, not just overlays."""
    if _task_only(episode) or _task_only(getattr(store, "config", {})):
        return {**episode, "reward_mode": "task_only", "adherence_enabled": False, "compliance_raw_magnitude": 0}
    return episode


def _stem(identity):
    return "fix6-" + identity["condition"] + "-" + hashlib.sha256(_bytes(identity)).hexdigest()[:20]


def _event_time(event):
    for key in ("available_at_s", "available_at", "time_s", "timestamp_s", "issued_at_s", "issued_at"):
        if type(event.get(key)) in (int, float) and math.isfinite(event[key]):
            return float(event[key])
    return None


def _guidance_at(episode, row, at):
    """Resolve status events back to their exact original instruction text."""
    instructions, current = {}, None
    for event in _timeline(episode):
        available = _event_time(event)
        if available is None or available > at:
            continue
        kind = event.get("kind", event.get("event"))
        if kind in ("completed", "interrupted", "expired", "superseded", "cancelled"):
            target = event.get("target_id", event.get("instruction_id"))
            if target not in instructions:
                continue
            cue = instructions[target]
            status = kind
            if kind == "interrupted":
                status = next((s for s in ("expired", "superseded", "cancelled")
                               if event.get("text", "").startswith("Guidance " + s + ":")), "interrupted")
            cue.update(status=status, ended_at_s=available)
            current = target
        else:
            identity = event.get("instruction_id", event.get("event_id", str(len(instructions))))
            cue = dict(event)
            cue["instruction_id"] = identity
            cue["issued_at_s"] = event.get("issued_at_s", event.get("issued_at", event.get("utterance_end", available)))
            cue["available_at_s"] = available
            cue["status"] = event.get("status", "new" if abs(at - available) < 1e-8 else "active")
            ended = event.get("ended_at_s", event.get("completed_at_s", event.get("completed_at")))
            if ended is not None and at < ended and cue["status"] in ("completed", "expired", "superseded", "cancelled"):
                cue["status"] = "active"
            expires = cue.get("expires_at_s", cue.get("expires_at"))
            if expires is not None and at >= expires and cue["status"] in ("new", "issued", "active"):
                cue["status"] = "expired"
            instructions[identity], current = cue, identity
    cue = instructions.get(current, {})
    audits = episode.get("coaching_audits", episode.get("coach_audits", {}))
    audit = audits.get(current, {}) if isinstance(audits, dict) else {}
    return {**cue, "reason": cue.get("reason", cue.get("geometric_reason", audit.get("reason", row.get("geometric_reason"))))}


def _timeline(episode):
    timeline = list(episode.get("coaching_timeline", episode.get("coaching_events", [])))
    for row in episode.get("rows", []):
        timeline.extend(row.get("coaching_events", []))
    result, seen = [], set()
    for event in timeline:
        value = _plain(event)
        if not isinstance(value, dict):
            raise ValueError("coaching timeline must contain objects")
        key = _bytes(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return sorted(result, key=lambda e: _event_time(e) if _event_time(e) is not None else math.inf)


def _row_time(row):
    value = row.get("end_time", row.get("time_s"))
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("original rows require finite physical end_time/time_s")
    return float(value)


def _cue_strips(rows, events):
    strips = []
    times = [_row_time(r) for r in rows]
    for index, event in enumerate(events):
        kind = event.get("event", event.get("kind", event.get("status", "issued")))
        if kind not in ("issued", "new", "active", "instruction", "operator_instruction"):
            continue
        at = _event_time(event)
        if at is None or not times:
            continue
        nearest = min(range(len(times)), key=lambda i: abs(times[i] - at))
        chosen = list(range(max(0, nearest - 2), min(len(times), nearest + 3)))
        strips.append({"event_index": index, "instruction_id": event.get("instruction_id", event.get("event_id")), "cue_time_s": at,
                       "rows": chosen, "frame_refs": [rows[i]["frame_ref"] for i in chosen],
                       "times_s": [times[i] for i in chosen], "selection": "bounded original cue evidence"})
    return strips


def save_episode(store, episode, *, condition, phase, checkpoint):
    """Save original state/events and all compact actions, even without an MP4.

    Camera pixels remain in the original store's content-addressed FrameStore;
    their references, simulator snapshots and cue-strip references are durable
    JSON. This is sufficient for later outcome selection and regeneration.
    """
    root = _writable(store)
    episode = _media_episode(store, episode)
    identity = _identity(episode, condition, phase, checkpoint)
    rows, frames = episode.get("rows", []), episode.get("frames", [])
    if (not isinstance(rows, list) or not isinstance(frames, list) or len(rows) != len(frames)
            or max(len(rows), len(frames)) > MAX_FRAMES):
        raise ValueError("episode rows/snapshots exceed bounds")
    compact = [{k: _plain(v) for k, v in row.items() if k in ROW_KEYS} for row in rows]
    if any(not isinstance(row.get("frame_ref"), str) for row in compact):
        raise ValueError("every original row requires frame_ref")
    timeline = _timeline(episode)
    payload = {"schema_version": 1, "identity": identity, "frames": _plain(frames), "rows": compact,
               "coaching_timeline": timeline, "cue_strips": _cue_strips(compact, timeline),
               "episode": {k: _plain(episode[k]) for k in ("success", "n", "layout", "environment", "evaluation",
                            "difficulty", "difficulty_start", "difficulty_end", "task_return", "guidance_return",
                            "raw_return", "coded_return", "end_reason", "compliance", "coaching_windows", "coaching_audits",
                            "coach_audits", "initial_frame_ref", "initial_frame_time", "raw_stage_rewards", "task_stage_rewards",
                            "guidance_stage_rewards", "coded_stage_rewards", "protocol", "adherence_enabled",
                            "reward_mode", "compliance_raw_magnitude") if k in episode}}
    payload["episode"]["guidance_status"] = "coached" if timeline else "uncoached"
    stem = _stem(identity)
    for index, strip in enumerate(payload["cue_strips"]):
        strip["path"] = f"coaching_media/{stem}.cue-{index:04d}.png"
    data = _bytes(payload)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("episode source JSON exceeds size bound")
    path = root / "coaching_media" / (stem + ".episode.json")
    _publish_bytes(store, path, data)
    for strip in payload["cue_strips"]:
        _save_cue_strip(store, strip)
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(data).hexdigest(),
            "schema_version": 1, "camera_frames": len(compact), "snapshots": len(frames),
            "cue_strips": len(payload["cue_strips"])}


def load_saved_episode(store, source):
    """Recover a saved JSON media source, without learner checkpoint unpickling."""
    path = _safe(store.root, Path(store.root) / source["path"])
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES or _digest(path) != source["sha256"]:
        raise ValueError("saved episode source is missing or changed")
    value = json.loads(path.read_bytes())
    if value.get("schema_version") != 1 or value.get("identity", {}).get("version") != VERSION:
        raise ValueError("incompatible saved episode source")
    return {**value["episode"], **{k: value["identity"][k] for k in ("episode_id", "seed")},
            "frames": value["frames"], "rows": value["rows"], "coaching_timeline": value["coaching_timeline"]}


def _renderer_factory(snapshot):
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig

    return PyBulletBackend(SimulatorConfig.from_dict(snapshot["config"]), run_id=snapshot["run_id"])


def _format(value):
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value
    return json.dumps(_plain(value), ensure_ascii=True, sort_keys=True)


def _save_cue_strip(store, strip):
    import io
    from PIL import Image, ImageDraw, ImageOps

    width, height = 240, 180
    canvas = Image.new("RGB", (width * len(strip["frame_refs"]), height + 26), "#17191b")
    draw = ImageDraw.Draw(canvas)
    for index, (reference, at) in enumerate(zip(strip["frame_refs"], strip["times_s"])):
        camera = store.frames.get(reference)
        image = ImageOps.contain(Image.fromarray(camera).convert("RGB"), (width, height))
        canvas.paste(image, (index * width + (width - image.width) // 2, (height - image.height) // 2))
        draw.text((index * width + 5, height + 5), f"Original camera t={at:.3f}s", fill="white", font=_font())
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    _publish_bytes(store, Path(store.root) / strip["path"], buffer.getvalue())


def _labels_for_row(episode, row):
    index = next((i for i, r in enumerate(episode.get("rows", [])) if r is row), None)
    if index is None and row.get("id") is not None:
        index = next((i for i, r in enumerate(episode.get("rows", [])) if r.get("id") == row["id"]), None)
    judgments = []
    for window_id, value in episode.get("compliance", {}).items():
        if row.get("id") is not None and value.get("closing_row") == row["id"]:
            result = value.get("judgment", {})
            judgments.append({"window_id": window_id,
                              "judgment": result.get("judgment") if isinstance(result, dict) else result,
                              "reason": result.get("reason") if isinstance(result, dict) else None,
                              "raw_bonus": value.get("receipt", {}).get("raw"),
                              "evidence_end": value.get("evidence", {}).get("end_time")})
    compliance = judgments or row.get("compliance", row.get("judgment"))
    parts = row.get("reward_parts")
    if parts is None:
        parts = {}
        for name, key, fallback in (("task_raw", "task_stage_rewards", "task_reward"),
                                    ("guidance_raw", "guidance_stage_rewards", "guidance_reward"),
                                    ("total_raw", "raw_stage_rewards", "raw_reward"),
                                    ("total_encoded", "coded_stage_rewards", "encoded_reward")):
            values = episode.get(key, [])
            parts[name] = values[index] if index is not None and index < len(values) else row.get(fallback)
    return compliance, parts


def subtitle_lines(episode, row, at, *, condition, phase, checkpoint, selection):
    """As-of guidance plus explicitly retrospective closing-transition labels."""
    cue = _guidance_at(episode, row, at)
    issued, available, status = cue.get("issued_at_s"), cue.get("available_at_s"), cue.get("status", "inactive")
    text = cue.get("text", cue.get("instruction", row.get("instruction", "No corrective guidance")))
    reason = cue.get("reason")
    requested = row.get("requested_controls", row.get("requested_control"))
    if requested is None and row.get("action") is not None:
        requested = {"action_logits (not calibrated controls)": _plain(row["action"])}
    executed = row.get("executed_controls", row.get("executed_control", row.get("receipt", {}).get("executed_controls")))
    compliance, parts = _labels_for_row(episode, row)
    evaluated = _row_time(row)
    lines = [f"{condition} | {phase} | episode={episode['episode_id']} | learning checkpoint={checkpoint} | {selection}",
             f"Camera + top-down synchronized at physical t={at:.3f}s | top-down/overlays: reader-only",
             f"Guidance issued={_format(issued)}s | available={_format(available)}s | status={status}",
             "Exact delivered guidance: " + str(text), "Geometric reason: " + _format(reason),
             "Requested: " + _format(requested), "Executed calibrated controls: " + _format(executed)]
    expiry = sum(s["end_time"] - s["start_time"] for s in row.get("execution_segments", []) if s["source"] == "expiry_neutral")
    if expiry:
        lines.append(f"Backend command-expiry neutral time in this interval: {expiry:.3f}s")
    if _task_only(episode):
        lines.append("Adherence: disabled by design | identical U/G task-only reward; no compliance bonus")
        lines.append("RETROSPECTIVE reward parts: " + _format(parts) if at + 1e-8 >= evaluated
                     else "Reward parts: pending at this physical time")
    elif at + 1e-8 >= evaluated:
        lines += [f"RETROSPECTIVE Qwen judgment at evaluated transition t={evaluated:.3f}s: {_format(compliance)}",
                  "RETROSPECTIVE reward parts: " + _format(parts)]
    else:
        lines += ["Qwen judgment / reward parts: pending at this physical time"]
    return lines


def _font():
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _wrap_pixels(lines, font, width):
    """Wrap every character, including long IDs, without dropping any content."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    result = []
    for text in lines:
        for paragraph in text.split("\n"):
            current = ""
            for character in paragraph:
                if current and draw.textlength(current + character, font=font) > width:
                    result.append(current)
                    current = ""
                current += character
            result.append(current)
    return result


def compose_frame(camera, topdown, lines, *, panel_size=(480, 360), band_height=None):
    """Letterbox both complete images and wrap subtitles in an uncropped band."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    width, height = panel_size
    if any(type(v) is not int or v < 32 or v % 2 for v in panel_size):
        raise ValueError("panel dimensions must be even and >=32")
    font = _font()
    wrapped = _wrap_pixels(lines, font, 2 * width - 20)
    line_height = max(18, font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6)
    required = 16 + len(wrapped) * line_height
    if band_height is None:
        band_height = required + required % 2
    if band_height < required:
        raise ValueError("subtitle band too small; text must never be cropped")
    canvas = Image.new("RGB", (2 * width, height + band_height), "#17191b")
    for index, array in enumerate((camera, topdown)):
        array = np.asarray(array)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("camera/top-down must be original HWC uint8 RGB")
        image = ImageOps.contain(Image.fromarray(array), (width, height))
        canvas.paste(image, (index * width + (width - image.width) // 2, (height - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(wrapped):
        draw.text((10, height + 8 + index * line_height), line, fill="white", font=font)
    return np.asarray(canvas), {"lines": wrapped, "required_height": required, "band_height": band_height}


def _validate_video(path, expected):
    import imageio.v2 as imageio

    reader = None
    try:
        reader = imageio.get_reader(path)
        try:
            count = reader.count_frames()
            first = reader.get_data(0)
            last = reader.get_data(count - 1)
            metadata = reader.get_meta_data()
        finally:
            reader.close()
        return (count == expected["frames"] and list(first.shape[:2]) == expected["shape"]
                and first.shape == last.shape and abs(metadata["fps"] - expected["fps"]) < .01)
    except (OSError, ValueError, RuntimeError, IndexError, KeyError):
        return False


def render_video(store, episode, *, condition, phase, checkpoint, selection="preselected",
                 fps=5., panel_size=(480, 360), sync_tolerance_s=.001):
    """Return one deterministic, synchronized MP4 descriptor for this episode.

    Resume skips encoding only after source identity, file checksum, frame
    count, shape and actual decoding validate. An interrupted encode leaves no
    final file; a durable MP4 without its receipt is validated and adopted.
    Never call this from a results notebook.
    """
    import imageio.v2 as imageio

    root = _writable(store)
    episode = _media_episode(store, episode)
    if selection not in ("preselected", "outcome_gallery", "success_gallery", "failure_gallery"):
        raise ValueError("declare preselected or outcome gallery selection")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("positive finite fps required")
    if type(sync_tolerance_s) not in (int, float) or not math.isfinite(sync_tolerance_s) or sync_tolerance_s < 0:
        raise ValueError("nonnegative finite sync tolerance required")
    rows, snapshots = episode.get("rows", []), episode.get("frames", [])
    if not rows or len(rows) != len(snapshots):
        raise ValueError("one original camera row and simulator snapshot per synchronized frame required")
    times = [_row_time(row) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("physical frame times must be strictly increasing")
    for row, at in zip(rows, times):
        for key in ("camera_time_s", "topdown_time_s"):
            value = row.get(key, at)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value - at) > sync_tolerance_s:
                raise ValueError("camera/top-down frames are not synchronized")
    source = save_episode(store, episode, condition=condition, phase=phase, checkpoint=checkpoint)
    identity = _identity(episode, condition, phase, checkpoint)
    # Selection is metadata, not a second copy of an already recorded episode.
    encoding = {"source_sha256": source["sha256"], "fps": fps, "panel_size": list(panel_size),
                "sync_tolerance_s": sync_tolerance_s, "producer_sha256": _digest(__file__)}
    path = _safe(root, root / "coaching_media" / (_stem(identity) + ".mp4"))
    receipt = path.with_suffix(".video.json")
    if receipt.exists():
        previous = json.loads(receipt.read_bytes())
        if previous.get("encoding") != encoding or not path.is_file() or _digest(path) != previous.get("sha256"):
            raise ValueError("saved video/receipt identity differs or is corrupt")
        if not _validate_video(path, previous):
            raise ValueError("saved video failed decode validation")
        return [{**previous, "selection": selection}]
    lines = [subtitle_lines(episode, row, at, condition=condition, phase=phase, checkpoint=checkpoint,
                            selection="recorded episode") for row, at in zip(rows, times)]
    font = _font()
    line_height = max(18, font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6)
    band = max(16 + len(_wrap_pixels(text, font, 2 * panel_size[0] - 20)) * line_height for text in lines)
    band += band % 2
    # Zero-order hold preserves the physical clock; no invented motion frames.
    duration = times[-1] - times[0]
    encoded_count = max(1, math.ceil(duration * fps) + 1)
    samples, index = [], 0
    for tick in range(encoded_count):
        at = min(times[-1], times[0] + tick / fps)
        while index + 1 < len(times) and times[index + 1] <= at + 1e-9:
            index += 1
        samples.append(index)
    expected = {"frames": encoded_count, "fps": fps, "shape": [panel_size[1] + band, 2 * panel_size[0]]}
    result = {**identity, **expected, "schema_version": 1, "encoding": encoding,
              "path": str(path.relative_to(root)), "source": source, "selection": selection,
              "success": episode.get("success"), "difficulty_start": episode.get("difficulty_start", episode.get("difficulty")),
              "difficulty_end": episode.get("difficulty_end"), "topdown_reader_only": True,
              "original_camera_frames": True, "retrospective_labels": True,
              "physical_start_s": times[0], "physical_end_s": times[-1],
              "playback": "Original saved observations held until next physical timestamp; no motion interpolation."}
    if path.exists():
        if not _validate_video(path, expected):
            raise ValueError("unreceipted video failed validation; do not overwrite")
    else:
        _admit(store, encoded_count * expected["shape"][0] * expected["shape"][1] * 3)
        fd, filename = tempfile.mkstemp(prefix=".fix6-encode-", suffix=".mp4", dir=path.parent)
        os.close(fd)
        temporary = Path(filename)
        try:
            with _renderer_factory(snapshots[0]) as renderer:
                writer = imageio.get_writer(temporary, fps=fps, codec="libx264", macro_block_size=2)
                try:
                    last, canvas = None, None
                    for index in samples:
                        if index != last:
                            renderer.restore(snapshots[index])
                            if abs(renderer.episode_time_s - times[index]) > sync_tolerance_s:
                                raise ValueError("snapshot and original camera times are not synchronized")
                            camera = store.frames.get(rows[index]["frame_ref"])
                            topdown = renderer.render_topdown()
                            canvas, _ = compose_frame(camera, topdown, lines[index], panel_size=panel_size, band_height=band)
                            last = index
                        writer.append_data(canvas)
                finally:
                    # ImageIO's context manager skips close() on exceptions.
                    writer.close()
            if not _validate_video(temporary, expected):
                raise ValueError("new video failed decode validation")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
            _refresh(store)
    result["sha256"] = _digest(path)
    _publish_bytes(store, receipt, _bytes(result))
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return [result]


# Explicit alias makes the all-episode persistence requirement clear at call sites.
persist_episode = save_episode
