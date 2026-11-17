"""Small frozen-Qwen prompt probe, isolated from training and notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


PROMPT = """Judge only the apparent horizontal width of the red ball in this image.
Do not estimate physical distance. Ignore blue obstacles, floor, walls, and whether
the robot is driving correctly. Estimate the visible red ball's left-to-right span
as a percentage of the ENTIRE image width, not of the ball's bounding box.
If partly cropped or occluded, use the horizontal span of the visible red ball.
If no red ball is visible, width_percent and reward must both be 0.
Set reward = min(10, width_percent / 8).
80 percent width or more gives the maximum reward 10.
A small ball must get a low reward even if centered. A ball largely filling the
image width must get a high reward. Decimals are allowed.
Return only JSON with exactly two numeric keys named width_percent and reward.
Supply your own estimates for THIS image. Do not include Markdown or prose."""

EDGE_PROMPT = """Locate the visible red ball with a tight bounding box.
Use the ENTIRE image as the coordinate frame, including all background.
Coordinates are normalized: left/top image edge = 0, right/bottom edge = 1000.
Find the actual leftmost, topmost, rightmost, and bottommost visible red edges.
Be precise for small balls; do not round their size to a broad category.
If partly cropped or occluded, bound only the visible part of the red ball.
If no red ball is visible, use null instead of a bounding box.
Return only JSON with one key named bbox, whose value is
[left, top, right, bottom] or null. Do not include Markdown or prose.
Do not estimate distance or reward."""

PIXEL_PROMPT = """Locate the red ball and return its tight bounding box.
The image is {width} pixels wide and {height} pixels tall.
Use absolute pixel coordinates, NOT normalized coordinates or percentages.
Find the actual leftmost, topmost, rightmost, and bottommost visible red edges.
Include only the visible red ball, not its shadow, background, or obstacles.
Be precise even when the ball is small. Do not estimate distance or reward.
If no red ball is visible, return null for bbox_2d.
Return only JSON with the single key bbox_2d and value
[left, top, right, bottom] or null. No Markdown or prose."""

HORIZONTAL_PROMPT = """Measure the horizontal extent of the visible red ball.
The full image is {width} pixels wide and {height} pixels tall.
The left image edge is x=0 and the right image edge is x={width}.
Locate the red ball's leftmost and rightmost visible red pixels.
Return their x coordinates in pixels, not percentages and not a distance estimate.
Exclude background and shadow. If cropped, measure only the visible red part.
Return only JSON with the key span and value [left_x, right_x].
If no red ball is visible, return null for span. No Markdown or prose."""


def parse(text):
    from numerical_experiment.picar.dense_roles import _strict_json
    import math

    value = _strict_json(text)
    if set(value) != {"width_percent", "reward"}:
        raise ValueError("expected width_percent and reward")
    for key, limit in (("width_percent", 100), ("reward", 10)):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 <= value[key] <= limit:
            raise ValueError("invalid " + key)
    return value


def parse_edges(text, *, width=1000, height=1000, key="bbox", horizontal=False):
    from numerical_experiment.picar.dense_roles import _strict_json
    import math

    if any(type(x) is not int or x <= 0 for x in (width, height)):
        raise ValueError("positive integer image dimensions required")
    value = _strict_json(text)
    if set(value) != {key}:
        raise ValueError("expected " + key)
    box = value[key]
    if box is None:
        width = 0.
    else:
        limits = (width, width) if horizontal else (width, height, width, height)
        if not isinstance(box, list) or len(box) != len(limits):
            raise ValueError("wrong coordinate count")
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= limit
               for x, limit in zip(box, limits)):
            raise ValueError("coordinate outside declared image dimensions")
        right = box[1] if horizontal else box[2]
        if box[0] >= right or (not horizontal and box[1] >= box[3]):
            raise ValueError("bbox edges must be ordered and have positive area")
        width = 100. * (right - box[0]) / width
    # Qwen estimates geometry; software applies the unchanged reward rule.
    return {key: box, "width_percent": width, "reward": min(10., width / 8.)}


def run(args):
    import numpy as np
    import torch
    from matplotlib.figure import Figure
    from numerical_experiment.picar.dense_roles import prepare_conversation
    from numerical_experiment.picar.fix2_data import panel_history
    from numerical_experiment.picar.models import load_qwen_model
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.role_repair import native_text
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import FrameStore, RunStore

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA Qwen required for this probe")
    torch.set_num_threads(2)
    edges = args.variant != "direct"
    horizontal = args.variant == "horizontal"
    pixel_edges = args.variant in ("pixels", "horizontal")
    prompt = HORIZONTAL_PROMPT if horizontal else PIXEL_PROMPT if pixel_edges else EDGE_PROMPT if edges else PROMPT
    source = args.panels.resolve()
    unit = read_completed_unit(source)
    states = ("absent", "far", "intermediate", "close", "occluded", "too_close")
    panels = [next(p for p in unit["summary"]["panels"] if p["seed"] == 211 and p["state"] == state)
              for state in states]
    version = 5 if horizontal else 4 if pixel_edges else 3 if edges else 2
    settings = {"schema_version": 1, "kind": f"ball_width_prompt_probe_v{version}",
        "variant": args.variant, "prompt": prompt,
        "reward_source": "software_from_qwen_bbox" if edges else "native_qwen",
        "source": str(source), "source_completion_sha256": sha256_file(source / "COMPLETED"),
        "panels": panels, "source_sha256": {str(Path(__file__)): sha256_file(Path(__file__))},
        "model_override": str(args.model_path) if args.model_path else None,
        "seed": 191101, "role": "evaluator", "max_new_tokens": 96,
        "max_pixels": 76800, "selection": "one saved development seed, all six states, before outcomes",
        "not_training": True, "notebook_modified": False}
    with RunStore(args.output, settings) as store:
        model = load_qwen_model(**({"model_path": args.model_path} if args.model_path else {}),
            local_files_only=args.offline, seed=191101, top_layers=1, ltm_rank=4, stm_rank=1,
            hidden_size=64, context_length=4, return_bins=161, max_tokens=2048).eval()
        frames = FrameStore(source / "frames", readonly=True)
        figure = Figure(figsize=(11.4, 6.2), layout="constrained")
        axes = figure.subplots(2, 3).flat
        rows = []
        for panel, axis in zip(panels, axes):
            rgb = frames.get(panel["frame_ref"])
            batch = panel_history(panel, frames)
            tick = time.monotonic()
            effective_prompt = prompt
            if pixel_edges:
                effective_prompt = prompt.format(width=batch["images"].shape[-1], height=batch["images"].shape[-2])
            inputs = prepare_conversation(model.encoder, batch, effective_prompt, visual_only=True, max_pixels=76800)
            if pixel_edges:
                # Qwen 2.5 grounding uses pixels of the processor-resized image.
                grid = inputs["image_grid_thw"][0]
                patch = model.encoder.processor.image_processor.patch_size
                height, width = int(grid[1]) * patch, int(grid[2]) * patch
                effective_prompt = prompt.format(width=width, height=height)
                inputs = prepare_conversation(model.encoder, batch, effective_prompt, visual_only=True, max_pixels=76800)
            raw = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
            result = {"panel_id": panel["panel_id"], "state": panel["state"],
                "frame_ref": panel["frame_ref"], "raw_text": raw, "seconds": time.monotonic() - tick}
            if pixel_edges:
                result.update(prompt=effective_prompt, processed_width=width, processed_height=height)
            try:
                parsed = (parse_edges(raw, width=width, height=height, key="span", horizontal=True) if horizontal
                          else parse_edges(raw, width=width, height=height, key="bbox_2d") if pixel_edges
                          else parse_edges(raw) if edges else parse(raw))
                result.update(parsed)
                result["rule_reward_from_reported_width"] = min(10., result["width_percent"] / 8)
            except ValueError as exc:
                result["parse_error"] = str(exc)
            red = (rgb[..., 0].astype(float) > 100) & (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 1.5 * rgb[..., 2])
            x = np.nonzero(red)[1]
            result["reference_red_pixel_span_percent"] = 100 * (int(x.max()-x.min()+1) if len(x) else 0) / rgb.shape[1]
            rows.append(result)
            axis.imshow(rgb);axis.axis("off")
            title = {"absent": "No ball", "too_close": "Very close"}.get(panel["state"], panel["state"].capitalize())
            reward_label = "computed reward" if edges else "reward"
            judgment = (f"Qwen width: {result['width_percent']:.1f}%\n{reward_label.capitalize()}: {result['reward']:.2f} / 10"
                        if "reward" in result else "Invalid output\nSee saved raw response")
            axis.set_title(title + "\n" + judgment, fontsize=10)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "width prompt diagnostic",
                "counts": {"scored": len(rows), "target": len(panels)}})
            print(json.dumps(result), flush=True)
        figure.savefig(store.root / "contact-sheet.png", dpi=140)
        store.complete({"schema_version": 1, "kind": "ball_width_prompt_probe", "phase": 3,
            "smoke_only": True, "development_only": True, "prompt": prompt, "rows": rows,
            "variant": args.variant, "reward_source": settings["reward_source"],
            "model_provenance": model.encoder.provenance,
            "image_sha256": sha256_file(store.root / "contact-sheet.png"),
            "reference_note": "Red-pixel span is a reader-only rough check, never provided to Qwen."})
    print(args.output / "contact-sheet.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, default=Path(
        "cache/numerical_experiment/picar/phase1/fix2-v2/units/panels-28ca3c9f841f"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("direct", "edges", "pixels", "horizontal"), default="direct")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
