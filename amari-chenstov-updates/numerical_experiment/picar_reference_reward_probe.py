"""Frozen-Qwen visual-reference diagnostic; no training or notebook updates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from numerical_experiment.picar_ball_width_probe import PROMPT, parse


REFERENCE_STATES = ("absent", "far", "intermediate", "too_close")
TEST_STATES = ("absent", "far", "intermediate", "close", "occluded", "too_close")
REFERENCE_PROMPT = """You will see four labeled reference images, followed by one TARGET image.
Judge only the apparent horizontal size of the visible red ball relative to the
ENTIRE image width. Do not estimate physical distance or navigation quality.
Choose the reference image whose visible red ball is most similar in relative
horizontal size to the ball in the TARGET image. Ignore background, floor,
obstacles, texture, and the ball's position within the image.
If no red ball is visible in the TARGET, choose reference A, including when an
obstacle fully hides it. If partly hidden or cropped, compare only the visible
red part. The references' identities stay the same even if their order changes.
Return only JSON with the single key reference and value A, B, C, or D.
Do not include Markdown, explanations, coordinates, or arithmetic."""

SCORE_ONLY_PROMPT = """The first four images are scored examples.
The final image is the image you must score.

Use the examples as a visual scoring scale.
Larger visible red balls should receive higher scores.
If no red ball is visible, give a score of 0.

Return only JSON with one numeric key, "reward",
containing your score from 0 to 10."""


def select_panels(panels):
    references = [next(p for p in panels if p["seed"] == 211 and p["state"] == state)
                  for state in REFERENCE_STATES]
    targets = [next(p for p in panels if p["seed"] == seed and p["state"] == state)
               for seed in (212, 213, 214) for state in TEST_STATES]
    if {p["seed"] for p in references} & {p["seed"] for p in targets}:
        raise ValueError("reference/test seeds overlap")
    if {p["frame_ref"] for p in references} & {p["frame_ref"] for p in targets}:
        raise ValueError("reference/test image contents overlap")
    if len({p["frame_ref"] for p in references + targets}) != len(references + targets):
        raise ValueError("duplicate images in diagnostic selection")
    return references, targets


def parse_reference(text):
    from numerical_experiment.picar.dense_roles import _strict_json

    value = _strict_json(text)
    if set(value) != {"reference"} or value["reference"] not in ("A", "B", "C", "D"):
        raise ValueError("expected exactly one reference identity A, B, C, or D")
    return value["reference"]


def pixel_width(rgb):
    import numpy as np

    red = ((rgb[..., 0].astype(float) > 100) & (rgb[..., 0] > 1.5 * rgb[..., 1])
           & (rgb[..., 0] > 1.5 * rgb[..., 2]))
    x = np.nonzero(red)[1]
    return 100. * (int(x.max() - x.min() + 1) if len(x) else 0) / rgb.shape[1]


def prepare_references(encoder, references, target_rgb, *, reverse=False, few_shot=False, score_only=False):
    import torch

    if score_only and not few_shot:
        raise ValueError("score-only requires scored worked examples")
    content, images = [], []
    ordered = list(reversed(references)) if reverse else references
    prompt = SCORE_ONLY_PROMPT if score_only else PROMPT if few_shot else REFERENCE_PROMPT
    messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    for item in ordered:
        if few_shot:
            answer = {"reward": round(item["reward"], 4)}
            if not score_only:
                answer = {"width_percent": round(item["width_percent"], 4), **answer}
            messages.extend([
                {"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": "Score the image and return JSON."}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer)}]}])
        else:
            content.extend([{"type": "text", "text": f"REFERENCE {item['reference']} (score {item['reward']:.3f}/10):"},
                            {"type": "image"}])
        images.append(torch.from_numpy(item["rgb"].copy()).permute(2, 0, 1).float() / 255)
    if few_shot:
        content = [{"type": "image"}, {"type": "text", "text": "Score the image and return JSON."}]
    else:
        content.extend([{"type": "text", "text": "TARGET image: choose the matching reference for THIS image only."},
                        {"type": "image"}])
    images.append(torch.from_numpy(target_rgb.copy()).permute(2, 0, 1).float() / 255)
    messages.append({"role": "user", "content": content})
    text = encoder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = dict(encoder.processor(text=[text], images=images, padding=True, return_tensors="pt",
                                    do_rescale=False, max_pixels=76800))
    if inputs["input_ids"].shape[1] > encoder.max_tokens:
        raise ValueError("reference prompt exceeds max_tokens; no silent truncation")
    return inputs


def figures(root, references, targets, frames, rows, method):
    from matplotlib.figure import Figure

    reference_figure = Figure(figsize=(12.8, 3.5), layout="constrained")
    for item, axis in zip(references, reference_figure.subplots(1, 4).flat):
        axis.imshow(item["rgb"])
        axis.axis("off")
        label = {"absent": "No visible ball", "far": "Far", "intermediate": "Medium", "too_close": "Very close"}[item["state"]]
        axis.set_title(f"{item['reference']}: {label}\nAnchor score {item['reward']:.2f} / 10", fontsize=10)
    reference_figure.savefig(root / "references.png", dpi=140)
    test_figure = Figure(figsize=(11.4, 7.2), layout="constrained")
    for target, axis in zip([p for p in targets if p["seed"] == 212], test_figure.subplots(2, 3).flat):
        axis.imshow(frames.get(target["frame_ref"]))
        axis.axis("off")
        row = next(r for r in rows if r["panel_id"] == target["panel_id"] and r["method"] == method)
        label = {"absent": "No ball", "too_close": "Very close"}.get(target["state"], target["state"].capitalize())
        if "reward" in row:
            result = (f"Choice {row['reference']}" if "reference" in row else f"Qwen width: {row['width_percent']:.1f}%")
            result += f" | score {row['reward']:.2f} / 10"
        else:
            result = "Invalid output"
        axis.set_title(f"{label} | test seed 212\n{result}", fontsize=10)
    test_figure.savefig(root / "test-examples.png", dpi=140)


def run(args):
    import torch
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
    reference_panels, targets = select_panels(unit["summary"]["panels"])
    frames = FrameStore(source / "frames", readonly=True)
    references = []
    for identity, panel in zip("ABCD", reference_panels):
        rgb = frames.get(panel["frame_ref"])
        width = pixel_width(rgb)
        references.append({**panel, "reference": identity, "rgb": rgb,
                           "width_percent": width, "reward": min(10., width / 8.)})
    reference_metadata = [{k: v for k, v in item.items() if k != "rgb"} for item in references]
    few_shot = args.format == "few-shot"
    method_name = "few_shot" if few_shot else "references"
    prompt = PROMPT if few_shot else REFERENCE_PROMPT
    jobs = [(method, target) for target in targets for method in ("direct", method_name)]
    jobs += [(method_name + "_reversed", target) for target in targets if target["state"] in ("far", "intermediate")]
    settings = {"schema_version": 1, "kind": "visual_reference_prompt_probe_v2" if few_shot else "visual_reference_prompt_probe_v1", "phase": 3,
        "source": str(source), "source_completion_sha256": sha256_file(source / "COMPLETED"),
        "references": reference_metadata, "targets": targets, "prompt": prompt, "format": args.format,
        "baseline_prompt": PROMPT, "model_override": str(args.model_path) if args.model_path else None,
        "model_seed": 191101, "role": "evaluator", "max_pixels_per_image": 76800,
        "max_new_tokens": {"direct": 96, method_name: 96 if few_shot else 32, method_name + "_reversed": 96 if few_shot else 32},
        "source_sha256": {str(path): sha256_file(path) for path in
            (Path(__file__), Path("numerical_experiment/picar_ball_width_probe.py"))},
        "selection": "four seed-211 references; all six states at seeds 212-214; reverse far/medium order check",
        "reward_rule": ("native Qwen width and reward after four worked examples" if few_shot else
                        "nearest reference only; anchor score min(10, reference red-pixel width percent / 8)"),
        "development_only": True, "not_training": True, "notebook_modified": False}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root, flush=True)
            return
        saved = store.load_checkpoint(trusted=True) or {}
        rows = saved.get("rows", [])
        model = load_qwen_model(**({"model_path": args.model_path} if args.model_path else {}),
            local_files_only=args.offline, seed=191101, top_layers=1, ltm_rank=4, stm_rank=1,
            hidden_size=64, context_length=4, return_bins=161, max_tokens=2048).eval()
        for method, target in jobs[len(rows):]:
            rgb = frames.get(target["frame_ref"])
            tick = time.monotonic()
            if method == "direct":
                inputs = prepare_conversation(model.encoder, panel_history(target, frames), PROMPT,
                                              visual_only=True, max_pixels=76800)
            else:
                inputs = prepare_references(model.encoder, references, rgb, reverse=method.endswith("reversed"), few_shot=few_shot)
            raw = native_text(model.encoder, inputs, role="evaluator",
                              max_new_tokens=settings["max_new_tokens"][method])[0]
            row = {"panel_id": target["panel_id"], "seed": target["seed"], "state": target["state"],
                "frame_ref": target["frame_ref"], "method": method, "raw_text": raw,
                "seconds": time.monotonic() - tick, "input_tokens": int(inputs["input_ids"].shape[1])}
            try:
                if method == "direct" or method.startswith("few_shot"):
                    row.update(parse(raw))
                    row["rule_reward_from_reported_width"] = min(10., row["width_percent"] / 8.)
                else:
                    identity = parse_reference(raw)
                    anchor = next(item for item in references if item["reference"] == identity)
                    row.update(reference=identity, reward=anchor["reward"])
            except ValueError as exc:
                row["parse_error"] = str(exc)
            # The target's pixel measurement is audit-only, not a Qwen input or reward override.
            row["pixel_width_percent"] = pixel_width(rgb)
            rows.append(row)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "visual reference prompt diagnostic",
                             "counts": {"scored": len(rows), "target": len(jobs)}})
            print(json.dumps(row), flush=True)
        figures(store.root, references, targets, frames, rows, method_name)
        with (store.root / "judgments.csv").open("w", newline="") as stream:
            columns = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        store.complete({"schema_version": 1, "kind": settings["kind"], "phase": 3,
            "development_only": True, "smoke_only": True, "rows": rows, "references": reference_metadata,
            "prompt": prompt, "baseline_prompt": PROMPT, "format": args.format, "model_provenance": model.encoder.provenance,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "image_sha256": {name: sha256_file(store.root / name) for name in ("references.png", "test-examples.png")},
            "judgments_sha256": sha256_file(store.root / "judgments.csv"),
            "limitations": ("Saved development panels, not a training result. " +
                            ("Worked examples may anchor or bias predictions." if few_shot else
                             "Four discrete scores, not continuous interpolation."))})
    print(args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, default=Path(
        "cache/numerical_experiment/picar/phase1/fix2-v2/units/panels-28ca3c9f841f"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--format", choices=("references", "few-shot"), default="references")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
