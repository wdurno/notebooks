"""Twenty fixed-order Qwen width evaluations paired with a single-image control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from numerical_experiment.picar_ball_width_probe import PROMPT, parse
from numerical_experiment.picar_width_scenes import SCENES, render_contact_sheet, render_scene


def error_fields(width, reward, truth_width):
    ideal_reward = min(10., truth_width / 8.)
    width_based_reward = min(10., width / 8.)
    return {"width_error_pp": width - truth_width, "absolute_width_error_pp": abs(width - truth_width),
        "ideal_reward": ideal_reward, "reward_from_reported_width": width_based_reward,
        "reward_error": reward - ideal_reward, "absolute_reward_error": abs(reward - ideal_reward),
        "arithmetic_error": reward - width_based_reward}


def summarize(panels, rows):
    result = {}
    for method in ("direct", "four_examples"):
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(panels) or {r["panel_id"] for r in selected} != {p["panel_id"] for p in panels}:
            raise ValueError("cannot summarize an incomplete paired diagnostic")
        valid = [row for row in selected if "parse_error" not in row]
        visible = [row for row in valid if row["true_visible"]]
        invisible = [row for row in valid if not row["true_visible"]]
        def average(key, values):
            return mean(row[key] for row in values) if values else None
        result[method] = {"cases": len(selected), "valid": len(valid), "invalid": len(selected) - len(valid),
            "valid_visible": len(visible), "valid_invisible": len(invisible),
            "width_mae_pp_all": average("absolute_width_error_pp", valid),
            "width_mae_pp_visible": average("absolute_width_error_pp", visible),
            "width_bias_pp_visible": average("width_error_pp", visible),
            "zero_width_on_visible": sum(row["width_percent"] == 0 for row in visible),
            "positive_width_on_invisible": sum(row["width_percent"] > 0 for row in invisible),
            "reward_mae": average("absolute_reward_error", valid),
            "arithmetic_mae": mean(abs(row["arithmetic_error"]) for row in valid) if valid else None}
    paired = [(next(r for r in rows if r["method"] == "direct" and r["panel_id"] == panel["panel_id"]),
               next(r for r in rows if r["method"] == "four_examples" and r["panel_id"] == panel["panel_id"]))
              for panel in panels]
    for a, b in paired:
        if ("target_pixels_sha256" in a and "target_pixels_sha256" in b
                and a["target_pixels_sha256"] != b["target_pixels_sha256"]):
            raise ValueError("paired processed target images differ")
    paired = [(a, b) for a, b in paired if "parse_error" not in a and "parse_error" not in b]
    result["paired_width_errors"] = {"valid_pairs": len(paired),
        "examples_better": sum(b["absolute_width_error_pp"] < a["absolute_width_error_pp"] for a, b in paired),
        "examples_worse": sum(b["absolute_width_error_pp"] > a["absolute_width_error_pp"] for a, b in paired),
        "ties": sum(b["absolute_width_error_pp"] == a["absolute_width_error_pp"] for a, b in paired)}
    return result


def render_calibration(path, rows):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.4, 4.6), layout="constrained")
    axes = figure.subplots(1, 2)
    for method, color, marker, label in (("direct", "#ac4848", "x", "Single image"),
                                         ("four_examples", "#16786c", "o", "Four examples + target")):
        selected = [row for row in rows if row["method"] == method and "parse_error" not in row]
        axes[0].scatter([r["true_width_percent"] for r in selected], [r["width_percent"] for r in selected],
                        color=color, marker=marker, s=48, alpha=.8, label=label)
        axes[1].scatter([r["ideal_reward"] for r in selected], [r["reward"] for r in selected],
                        color=color, marker=marker, s=48, alpha=.8, label=label)
    for axis, limit in zip(axes, (100, 10)):
        axis.plot([0, limit], [0, limit], color="#777777", linestyle="--", linewidth=1., label="Ideal agreement")
        axis.set(xlim=(-.02 * limit, 1.02 * limit), ylim=(-.02 * limit, 1.02 * limit))
        axis.grid(alpha=.18)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set(title="Visible ball width", xlabel="Renderer-mask width (% of image)", ylabel="Qwen width (% of image)")
    axes[1].set(title="Returned reward", xlabel="Reward rule applied to mask width", ylabel="Raw Qwen reward")
    figure.savefig(path, dpi=160)


def source_config(args):
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.runtime import sha256_file

    unit = read_completed_unit(args.source_run)
    if unit["summary"].get("format") != "few-shot":
        raise ValueError("expected the original completed worked-example diagnostic")
    config = json.loads((args.source_run / "config.json").read_text())
    metadata = {"source_run": str(args.source_run.resolve()),
        "source_completion_sha256": sha256_file(args.source_run / "COMPLETED"),
        "references": config["references"], "reference_frames_root": config["source"],
        "source_sha256": {str(path): sha256_file(path) for path in (Path(__file__),
            Path("numerical_experiment/picar_width_scenes.py"), Path("numerical_experiment/picar_reference_reward_probe.py"),
            Path("numerical_experiment/picar_ball_width_probe.py"))}}
    return config, metadata


def make_panels(args):
    from dataclasses import asdict
    import numpy as np
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import RunStore

    _, metadata = source_config(args)
    settings = {**metadata, "schema_version": 1, "kind": "diverse_width_panels_v1", "phase": 3,
        "scenes": [asdict(scene) for scene in SCENES], "scene_layout_seeds": list(range(291001, 291021)),
        "width": 320, "height": 240, "truth": "visible object-ID mask, horizontal min/max inclusive span",
        "selection": "20 fixed cases before Qwen outputs; exactly 3 invisible and 17 visible", "not_training": True}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root)
            return
        panels = (store.load_checkpoint(trusted=True) or {}).get("panels", [])
        masks_root = store.root / "masks"
        masks_root.mkdir(exist_ok=True)
        for index, spec in enumerate(SCENES[len(panels):], start=len(panels)):
            rgb, mask, panel = render_scene(spec, index)
            panel["frame_ref"] = store.frames.put(rgb)
            if panel["frame_ref"] in {item["frame_ref"] for item in settings["references"] + panels}:
                raise ValueError("test image duplicates a reference or another test image")
            mask_path = masks_root / (panel["panel_id"] + ".npz")
            np.savez_compressed(mask_path, visible_target=mask)
            panel.update(mask_file=str(mask_path.relative_to(store.root)), mask_sha256=sha256_file(mask_path))
            panels.append(panel)
            store.checkpoint({"panels": panels}, progress={"phase": 3, "status": "fixed measurement panels",
                "counts": {"rendered": len(panels), "target": len(SCENES)}})
            print(json.dumps(panel), flush=True)
        if sum(not panel["truth"]["visible"] for panel in panels) != 3:
            raise ValueError("expected exactly three no-visible-ball images")
        render_contact_sheet(store.root / "panels.png", panels, store.frames)
        store.complete({"schema_version": 1, "kind": settings["kind"], "phase": 3, "smoke_only": True,
            "development_only": True, "panels": panels, "rgb_mask_equivalence_checked": True,
            "image_sha256": sha256_file(store.root / "panels.png"),
            "not_model_inputs": ["object-ID masks", "scene parameters", "visibility labels", "reference widths"]})
    print(args.output, flush=True)


def evaluate(args):
    import torch
    from numerical_experiment.picar.dense_roles import prepare_conversation
    from numerical_experiment.picar.fix2_data import panel_history
    from numerical_experiment.picar.models import load_qwen_model
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.role_repair import native_text
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import FrameStore, RunStore
    from numerical_experiment.picar_reference_reward_probe import prepare_references

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA Qwen required")
    torch.set_num_threads(2)
    config, metadata = source_config(args)
    data = read_completed_unit(args.panels_run)["summary"]
    panels = data["panels"]
    if len(panels) != 20 or sum(not p["truth"]["visible"] for p in panels) != 3:
        raise ValueError("expected the predeclared 20-image panel")
    frames = FrameStore(args.panels_run / "frames", readonly=True)
    reference_frames = FrameStore(Path(config["source"]) / "frames", readonly=True)
    references = [{**item, "rgb": reference_frames.get(item["frame_ref"])} for item in config["references"]]
    if {p["frame_ref"] for p in panels} & {r["frame_ref"] for r in references}:
        raise ValueError("test/reference overlap")
    for panel in panels:
        if sha256_file(args.panels_run / panel["mask_file"]) != panel["mask_sha256"]:
            raise ValueError("reference mask checksum mismatch")
    settings = {**metadata, "schema_version": 1, "kind": "diverse_width_evaluation_v1", "phase": 3,
        "panels_run": str(args.panels_run.resolve()),
        "panels_completion_sha256": sha256_file(args.panels_run / "COMPLETED"), "panels": panels,
        "prompt": PROMPT, "example_order": "ABCD", "example_answers": ["width_percent", "reward"],
        "model_override": str(args.model_path) if args.model_path else None,
        "model_seed": 191101, "role": "evaluator", "max_new_tokens": 96, "max_pixels_per_image": 76800,
        "not_training": True, "notebook_modified": False}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root)
            return
        rows = (store.load_checkpoint(trusted=True) or {}).get("rows", [])
        model = load_qwen_model(**({"model_path": args.model_path} if args.model_path else {}),
            local_files_only=args.offline, seed=191101, top_layers=1, ltm_rank=4, stm_rank=1,
            hidden_size=64, context_length=4, return_bins=161, max_tokens=2048).eval()
        jobs = [(panel, method) for panel in panels for method in ("direct", "four_examples")]
        last_target_pixels = None
        for panel, method in jobs[len(rows):]:
            rgb = frames.get(panel["frame_ref"])
            tick = time.monotonic()
            if method == "direct":
                inputs = prepare_conversation(model.encoder, panel_history(panel, frames), PROMPT,
                                              visual_only=True, max_pixels=76800)
                last_target_pixels = inputs["pixel_values"].clone()
            else:
                inputs = prepare_references(model.encoder, references, rgb, few_shot=True)
            image_count = int(inputs["image_grid_thw"].shape[0])
            if image_count != (1 if method == "direct" else 5):
                raise ValueError("unexpected image count")
            target_patch_count = int(inputs["image_grid_thw"][-1].prod())
            target_pixels = inputs["pixel_values"][-target_patch_count:]
            if method == "four_examples" and last_target_pixels is not None and not torch.equal(target_pixels, last_target_pixels):
                raise ValueError("paired conditions have different processed target images")
            raw = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
            row = {"panel_id": panel["panel_id"], "state": panel["state"], "method": method,
                "frame_ref": panel["frame_ref"], "raw_text": raw, "seconds": time.monotonic() - tick,
                "input_tokens": int(inputs["input_ids"].shape[1]), "image_count": image_count,
                "target_pixels_sha256": hashlib.sha256(target_pixels.contiguous().numpy().tobytes()).hexdigest(),
                "true_width_percent": panel["truth"]["width_percent"], "true_visible": panel["truth"]["visible"]}
            try:
                row.update(parse(raw))
                row.update(error_fields(row["width_percent"], row["reward"], row["true_width_percent"]))
            except ValueError as exc:
                row["parse_error"] = str(exc)
            rows.append(row)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "fixed-order width benchmark",
                             "counts": {"scored": len(rows), "target": len(jobs)}})
            print(json.dumps(row), flush=True)
        metrics = summarize(panels, rows)
        render_calibration(store.root / "calibration.png", rows)
        for method in ("direct", "four_examples"):
            render_contact_sheet(store.root / f"{method}.png", panels, frames,
                                 [row for row in rows if row["method"] == method])
        with (store.root / "judgments.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
        images = ("calibration.png", "direct.png", "four_examples.png")
        store.complete({"schema_version": 1, "kind": settings["kind"], "phase": 3,
            "smoke_only": True, "development_only": True, "rows": rows, "metrics": metrics,
            "prompt": PROMPT, "model_provenance": model.encoder.provenance,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "image_sha256": {name: sha256_file(store.root / name) for name in images},
            "judgments_sha256": sha256_file(store.root / "judgments.csv"),
            "limitations": "Predetermined coverage cases, not a random population sample or an RL learning trial."})
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("panels", "evaluate"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--panels-run", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "evaluate" and args.panels_run is None:
        parser.error("evaluate requires --panels-run")
    (make_panels if args.stage == "panels" else evaluate)(args)


if __name__ == "__main__":
    main()
