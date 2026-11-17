"""Visualize five-image worked examples under balanced reference permutations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from numerical_experiment.picar_ball_width_probe import PROMPT, parse
from numerical_experiment.picar_reference_reward_probe import SCORE_ONLY_PROMPT, prepare_references


ORDERS = ("ABCD", "DCBA", "BDAC", "CADB")


def ordered_references(references, order):
    mapping = {item["reference"]: item for item in references}
    if set(mapping) != set("ABCD") or len(references) != 4:
        raise ValueError("four distinct reference identities required")
    if len(order) != 4 or set(order) != set(mapping):
        raise ValueError("order must be a permutation of ABCD")
    return [mapping[identity] for identity in order]


def storyboard(root, references, target, target_rgb, rows):
    from matplotlib.figure import Figure

    label = {"far": "Far", "intermediate": "Medium"}[target["state"]]
    figure = Figure(figsize=(17.6, 10.2), layout="constrained")
    grid = figure.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1, 1.05])
    figure.suptitle(f"{label} target: same test image in every row\n"
                    "Four reference image/answer examples, then the target; left to right", fontsize=16)
    for index, order in enumerate(ORDERS):
        ordered = ordered_references(references, order)
        row = next(item for item in rows if item["panel_id"] == target["panel_id"] and item["order"] == order)
        for column in range(5):
            axis = figure.add_subplot(grid[index, column])
            axis.imshow(ordered[column]["rgb"] if column < 4 else target_rgb)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(column == 4)
                spine.set_color("#176b49")
                spine.set_linewidth(2.)
            if column < 4:
                item = ordered[column]
                axis.set_title(f"{column + 1}: Reference {item['reference']}\nKnown score {item['reward']:.2f}", fontsize=11)
            else:
                axis.set_title("5: TARGET\nNo answer supplied", fontsize=11, color="#176b49")
        axis = figure.add_subplot(grid[index, 5])
        axis.axis("off")
        if "reward" in row:
            text = f"Order: {order}\n\nQwen target score\n{row['reward']:.4f} / 10\n\n"
            text += (f"Qwen width\n{row['width_percent']:.4f}%" if "width_percent" in row
                     else "Score-only prompt\nNo width requested")
        else:
            text = f"Order: {order}\n\nInvalid response\nSee saved raw text"
        axis.text(.05, .5, text, va="center", fontsize=13)
    figure.savefig(root / f"{target['state']}-orders.png", dpi=150)


def run(args):
    import torch
    from numerical_experiment.picar.models import load_qwen_model
    from numerical_experiment.picar.dense_roles import parse_reward
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.role_repair import native_text
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import FrameStore, RunStore

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA Qwen required")
    torch.set_num_threads(2)
    previous = read_completed_unit(args.source_run)
    old_config = json.loads((args.source_run / "config.json").read_text())
    if previous["summary"].get("format") != "few-shot":
        raise ValueError("expected the completed four-worked-example diagnostic")
    frames_root = Path(old_config["source"])
    read_completed_unit(frames_root)
    frames = FrameStore(frames_root / "frames", readonly=True)
    references = [{**item, "rgb": frames.get(item["frame_ref"])} for item in old_config["references"]]
    targets = [next(item for item in old_config["targets"] if item["seed"] == 212 and item["state"] == state)
               for state in ("far", "intermediate")]
    reference_frames = {item["frame_ref"] for item in references}
    if reference_frames & {item["frame_ref"] for item in targets}:
        raise ValueError("reference/test image overlap")
    comparison = read_completed_unit(args.comparison_run) if args.comparison_run else None
    if comparison is not None and comparison["summary"]["kind"] != "reference_order_visual_probe_v1":
        raise ValueError("expected the completed original reference-order diagnostic")
    prompt = SCORE_ONLY_PROMPT if args.score_only else PROMPT
    settings = {"schema_version": 1, "kind": "reference_order_visual_probe_v2" if args.score_only else "reference_order_visual_probe_v1", "phase": 3,
        "source_run": str(args.source_run.resolve()),
        "source_completion_sha256": sha256_file(args.source_run / "COMPLETED"),
        "comparison_run": str(args.comparison_run.resolve()) if args.comparison_run else None,
        "comparison_completion_sha256": sha256_file(args.comparison_run / "COMPLETED") if args.comparison_run else None,
        "references": old_config["references"], "targets": targets, "orders": ORDERS,
        "target_position": 5, "prompt": prompt, "format": "few-shot", "score_only": args.score_only,
        "example_answer_keys": ["reward"] if args.score_only else ["width_percent", "reward"],
        "model_override": str(args.model_path) if args.model_path else None,
        "model_seed": 191101, "role": "evaluator", "max_new_tokens": 96, "max_pixels_per_image": 76800,
        "selection": "preselected seed 212 far and intermediate; four position-balanced reference orders",
        "source_sha256": {str(path): sha256_file(path) for path in (Path(__file__),
            Path("numerical_experiment/picar_reference_reward_probe.py"), Path("numerical_experiment/picar_ball_width_probe.py"))},
        "not_training": True, "notebook_modified": False}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root, flush=True)
            return
        rows = (store.load_checkpoint(trusted=True) or {}).get("rows", [])
        model = load_qwen_model(**({"model_path": args.model_path} if args.model_path else {}),
            local_files_only=args.offline, seed=191101, top_layers=1, ltm_rank=4, stm_rank=1,
            hidden_size=64, context_length=4, return_bins=161, max_tokens=2048).eval()
        jobs = [(target, order) for target in targets for order in ORDERS]
        for target, order in jobs[len(rows):]:
            ordered = ordered_references(references, order)
            inputs = prepare_references(model.encoder, ordered, frames.get(target["frame_ref"]),
                                        few_shot=True, score_only=args.score_only)
            if inputs["image_grid_thw"].shape[0] != 5:
                raise ValueError("expected four example images plus one target")
            tick = time.monotonic()
            raw = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
            row = {"panel_id": target["panel_id"], "state": target["state"], "order": order,
                "sequence_frame_refs": [item["frame_ref"] for item in ordered] + [target["frame_ref"]],
                "raw_text": raw, "seconds": time.monotonic() - tick,
                "input_tokens": int(inputs["input_ids"].shape[1]), "image_count": 5,
                "pixel_tensor_sha256": hashlib.sha256(inputs["pixel_values"].contiguous().numpy().tobytes()).hexdigest()}
            try:
                if args.score_only:
                    row["reward"] = parse_reward(raw)
                else:
                    row.update(parse(raw))
            except ValueError as exc:
                row["parse_error"] = str(exc)
            old = None
            if comparison is not None:
                old = next(item for item in comparison["summary"]["rows"]
                           if item["panel_id"] == target["panel_id"] and item["order"] == order)
                if old["sequence_frame_refs"] != row["sequence_frame_refs"]:
                    raise ValueError("comparison images/order differ")
            elif order in ("ABCD", "DCBA"):
                method = "few_shot" if order == "ABCD" else "few_shot_reversed"
                old = next(item for item in previous["summary"]["rows"]
                           if item["panel_id"] == target["panel_id"] and item["method"] == method)
            if old is not None:
                row["previous_reward"] = old.get("reward")
                row["previous_raw_match"] = raw == old["raw_text"]
                row["previous_score_match"] = row.get("reward") == old.get("reward")
            rows.append(row)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "reference permutation diagnostic",
                             "counts": {"scored": len(rows), "target": len(jobs)}})
            print(json.dumps(row), flush=True)
        for target in targets:
            storyboard(store.root, references, target, frames.get(target["frame_ref"]), rows)
        image_names = [f"{target['state']}-orders.png" for target in targets]
        store.complete({"schema_version": 1, "kind": settings["kind"], "phase": 3,
            "smoke_only": True, "development_only": True, "rows": rows, "prompt": prompt,
            "score_only": args.score_only,
            "model_provenance": model.encoder.provenance, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "image_sha256": {name: sha256_file(store.root / name) for name in image_names},
            "note": "Storyboards show images in actual input order; labels/outlines are presentation only. "
                    "Each example's answer moves with its image. Target is always last. "
                    + ("Only rewards are provided/requested; no width estimates or formula in model context."
                       if args.score_only else "Example answers contain width and score.")})
    print(args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--comparison-run", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
