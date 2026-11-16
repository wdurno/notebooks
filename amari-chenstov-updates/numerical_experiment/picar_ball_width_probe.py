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
    source = args.panels.resolve()
    unit = read_completed_unit(source)
    states = ("absent", "far", "intermediate", "close", "occluded", "too_close")
    panels = [next(p for p in unit["summary"]["panels"] if p["seed"] == 211 and p["state"] == state)
              for state in states]
    settings = {"schema_version": 1, "kind": "ball_width_prompt_probe_v2", "prompt": PROMPT,
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
            inputs = prepare_conversation(model.encoder, batch, PROMPT, visual_only=True, max_pixels=76800)
            raw = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
            result = {"panel_id": panel["panel_id"], "state": panel["state"],
                "frame_ref": panel["frame_ref"], "raw_text": raw, "seconds": time.monotonic() - tick}
            try:
                result.update(parse(raw))
                result["rule_reward_from_reported_width"] = min(10., result["width_percent"] / 8)
            except ValueError as exc:
                result["parse_error"] = str(exc)
            red = (rgb[..., 0].astype(float) > 100) & (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 1.5 * rgb[..., 2])
            x = np.nonzero(red)[1]
            result["reference_red_pixel_span_percent"] = 100 * (int(x.max()-x.min()+1) if len(x) else 0) / rgb.shape[1]
            rows.append(result)
            axis.imshow(rgb);axis.axis("off")
            title = {"absent": "No ball", "too_close": "Very close"}.get(panel["state"], panel["state"].capitalize())
            judgment = (f"Qwen: {result['width_percent']:g}% width | reward {result['reward']:g} / 10"
                        if "reward" in result else "Qwen: invalid JSON (see raw output)")
            axis.set_title(title + "\n" + judgment, fontsize=10)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "width prompt diagnostic",
                "counts": {"scored": len(rows), "target": len(panels)}})
            print(json.dumps(result), flush=True)
        figure.savefig(store.root / "contact-sheet.png", dpi=140)
        store.complete({"schema_version": 1, "kind": "ball_width_prompt_probe", "phase": 3,
            "smoke_only": True, "development_only": True, "prompt": PROMPT, "rows": rows,
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
