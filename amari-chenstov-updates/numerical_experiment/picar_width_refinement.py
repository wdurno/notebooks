"""Paired four/six-example width probe on a fresh, fixed diagnostic panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from numerical_experiment.picar_ball_width_probe import PROMPT, parse
from numerical_experiment.picar_width_benchmark import error_fields
from numerical_experiment.picar_width_scenes import Scene, render_contact_sheet, render_scene


METHODS = ("four_examples", "six_examples")
FRESH_SCENES = (
    Scene("No ball: rear view", 1.22, yaw_deg=173., visible=False),
    Scene("No ball: side view", 1.45, domain="B", yaw_deg=-94., visible=False),
    Scene("Ball fully hidden", 1.2, barrier="full", visible=False),
    Scene("Tiny, centered", 2.65, bearing_deg=-5.),
    Scene("Far, left", 1.95, bearing_deg=25.),
    Scene("Far, right", 1.85, bearing_deg=-18., domain="B"),
    Scene("Far, centered", 1.52, bearing_deg=2., domain="B"),
    Scene("Medium, left", 1.22, bearing_deg=18.),
    Scene("Medium, centered", 1.08, bearing_deg=-3.),
    Scene("Medium, right", .83, bearing_deg=-23., domain="B"),
    Scene("Medium, blue beside", 1.2, bearing_deg=10., domain="B", barrier="beside"),
    Scene("Medium, partly hidden", 1.15, barrier="partial_left"),
    Scene("Close, centered", .72, domain="B", tilt_deg=-3.),
    Scene("Close, left", .48, bearing_deg=23.),
    Scene("Close, right", .45, bearing_deg=-23., domain="B"),
    Scene("Close, blue beside", .59, bearing_deg=-8., barrier="beside"),
    Scene("Close, partly hidden", .56, domain="B", barrier="partial_right"),
    Scene("Large, centered", .33, domain="B"),
    Scene("Very close, centered", .18, domain="B"),
    Scene("Large, edge-cropped", .205, bearing_deg=32.),
)


def select_references(original, development):
    if len(original) != 4 or len({r["frame_ref"] for r in original}) != 4:
        raise ValueError("expected four distinct original references")
    extra = []
    for identity, panel_id in (("E", "C18"), ("F", "C17")):
        panel = next(p for p in development if p["panel_id"] == panel_id)
        width = panel["truth"]["width_percent"]
        extra.append({**panel, "reference": identity, "width_percent": width,
                      "reward": min(10., width / 8.), "source_group": "development"})
    if not 35 <= extra[0]["width_percent"] <= 45 or not extra[1]["scene"]["barrier"].startswith("partial"):
        raise ValueError("added references must cover middle width and partial visibility")
    if extra[1]["width_percent"] <= 0:
        raise ValueError("partially visible reference must have positive width")
    baseline = [{**r, "source_group": "original"} for r in original]
    refined = sorted(baseline + extra, key=lambda r: r["width_percent"])
    if len({r["frame_ref"] for r in refined}) != 6:
        raise ValueError("duplicate reference image")
    return baseline, refined


def check_disjoint(panels, references, development):
    targets = [p["frame_ref"] for p in panels]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate fresh target")
    if set(targets) & {p["frame_ref"] for p in references + development}:
        raise ValueError("fresh targets overlap reference/development images")


def summarize(panels, rows):
    result, paired = {}, {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        if len(selected) != len(panels) or {r["panel_id"] for r in selected} != {p["panel_id"] for p in panels}:
            raise ValueError("incomplete paired refinement")
        paired[method] = {r["panel_id"]: r for r in selected}
        valid = [r for r in selected if "parse_error" not in r]
        def average(key, values):
            return mean(r[key] for r in values) if values else None
        visible = [r for r in valid if r["true_visible"]]
        partial = [r for r in valid if r["partial_occlusion"]]
        middle = [r for r in valid if 30 <= r["true_width_percent"] <= 60]
        result[method] = {
            "cases": len(selected), "valid": len(valid), "invalid": len(selected) - len(valid),
            "width_mae_pp_all": average("absolute_width_error_pp", valid),
            "width_mae_pp_visible": average("absolute_width_error_pp", visible),
            "reward_mae": average("absolute_reward_error", valid),
            "partial_cases": len(partial), "partial_reward_mae": average("absolute_reward_error", partial),
            "partial_zero_rewards": sum(r["reward"] == 0 for r in partial),
            "middle_cases": len(middle), "middle_reward_mae": average("absolute_reward_error", middle),
            "zero_width_on_visible": sum(r["width_percent"] == 0 for r in visible),
            "positive_width_on_invisible": sum(r["width_percent"] > 0 for r in valid if not r["true_visible"]),
            "mean_seconds": average("seconds", selected), "mean_input_tokens": average("input_tokens", selected)}
    pairs = [(paired[METHODS[0]][p["panel_id"]], paired[METHODS[1]][p["panel_id"]]) for p in panels]
    if any(a["target_pixels_sha256"] != b["target_pixels_sha256"] for a, b in pairs):
        raise ValueError("paired target pixel mismatch")
    valid_pairs = [(a, b) for a, b in pairs if "parse_error" not in a and "parse_error" not in b]
    result["paired_rewards"] = {"valid_pairs": len(valid_pairs),
        "six_better": sum(b["absolute_reward_error"] < a["absolute_reward_error"] for a, b in valid_pairs),
        "six_worse": sum(b["absolute_reward_error"] > a["absolute_reward_error"] for a, b in valid_pairs),
        "ties": sum(b["absolute_reward_error"] == a["absolute_reward_error"] for a, b in valid_pairs)}
    return result


def sources(args):
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.runtime import sha256_file

    original_unit = read_completed_unit(args.source_run)
    if original_unit["summary"].get("format") != "few-shot":
        raise ValueError("expected original worked-example diagnostic")
    config = json.loads((args.source_run / "config.json").read_text())
    development = read_completed_unit(args.development_run)["summary"]["panels"]
    baseline, refined = select_references(config["references"], development)
    metadata = {"schema_version": 1, "phase": 3, "not_training": True, "notebook_modified": False,
        "source_run": str(args.source_run.resolve()), "development_run": str(args.development_run.resolve()),
        "source_completion_sha256": sha256_file(args.source_run / "COMPLETED"),
        "development_completion_sha256": sha256_file(args.development_run / "COMPLETED"),
        "reference_frames_root": config["source"], "references": refined,
        "source_sha256": {str(p): sha256_file(p) for p in (Path(__file__),
            Path("numerical_experiment/picar_width_scenes.py"), Path("numerical_experiment/picar_width_benchmark.py"),
            Path("numerical_experiment/picar_reference_reward_probe.py"), Path("numerical_experiment/picar_ball_width_probe.py"))}}
    return metadata, baseline, refined, development


def make_panels(args):
    import numpy as np
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import RunStore

    metadata, _, references, development = sources(args)
    settings = {**metadata, "kind": "width_refinement_panels_v1", "scenes": [asdict(s) for s in FRESH_SCENES],
        "layout_seeds": list(range(291101, 291121)), "selection": "20 fixed coverage cases before fresh Qwen outputs",
        "truth": "visible object-ID horizontal inclusive span; scoring audit only"}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root)
            return
        panels = (store.load_checkpoint(trusted=True) or {}).get("panels", [])
        (store.root / "masks").mkdir(exist_ok=True)
        for index, scene in enumerate(FRESH_SCENES[len(panels):], start=len(panels)):
            rgb, mask, panel = render_scene(scene, 100 + index)
            panel["panel_id"] = f"T{index + 1:02d}"
            panel["frame_ref"] = store.frames.put(rgb)
            check_disjoint(panels + [panel], references, development)
            mask_file = store.root / "masks" / (panel["panel_id"] + ".npz")
            np.savez_compressed(mask_file, visible_target=mask)
            panel.update(mask_file=str(mask_file.relative_to(store.root)), mask_sha256=sha256_file(mask_file))
            panels.append(panel)
            store.checkpoint({"panels": panels}, progress={"phase": 3, "status": "fresh refinement panels",
                "counts": {"rendered": len(panels), "target": len(FRESH_SCENES)}})
            print(json.dumps(panel), flush=True)
        render_contact_sheet(store.root / "panels.png", panels, store.frames)
        store.complete({"schema_version": 1, "phase": 3, "kind": settings["kind"], "development_only": True,
            "smoke_only": True, "panels": panels, "rgb_mask_equivalence_checked": True,
            "image_sha256": sha256_file(store.root / "panels.png")})


def render_figures(root, references, panels, frames, rows):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.4, 3.3), layout="constrained")
    for ref, axis in zip(references, figure.subplots(1, 6)):
        axis.imshow(ref["rgb"])
        axis.axis("off")
        axis.set_title(f"{ref['reference']}: {ref['width_percent']:.1f}% width\nReward {ref['reward']:.2f} / 10", fontsize=10)
    figure.savefig(root / "references.png", dpi=160)
    figure = Figure(figsize=(10.6, 4.5), layout="constrained")
    axes = figure.subplots(1, 2)
    for method, color, marker, label in ((METHODS[0], "#ac4848", "x", "Four examples (current)"),
                                        (METHODS[1], "#16786c", "o", "Six examples (refined)")):
        selected = [r for r in rows if r["method"] == method and "parse_error" not in r]
        for axis, truth, estimate in ((axes[0], "true_width_percent", "width_percent"),
                                      (axes[1], "ideal_reward", "reward")):
            axis.scatter([r[truth] for r in selected], [r[estimate] for r in selected],
                         color=color, marker=marker, s=48, alpha=.8, label=label)
    for axis, limit in zip(axes, (100, 10)):
        axis.plot([0, limit], [0, limit], "--", color="#777777", linewidth=1, label="Ideal agreement")
        axis.set(xlim=(-.02 * limit, 1.02 * limit), ylim=(-.02 * limit, 1.02 * limit))
        axis.grid(alpha=.18)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set(xlabel="Mask width (% of image)", ylabel="Qwen width (% of image)", title="Visible ball width")
    axes[1].set(xlabel="Reward rule applied to mask width", ylabel="Raw Qwen reward", title="Returned reward")
    figure.savefig(root / "calibration.png", dpi=160)
    for method in METHODS:
        render_contact_sheet(root / f"{method}.png", panels, frames, [r for r in rows if r["method"] == method])


def evaluate(args):
    import torch
    from numerical_experiment.picar.models import load_qwen_model
    from numerical_experiment.picar.report import read_completed_unit
    from numerical_experiment.picar.role_repair import native_text
    from numerical_experiment.picar.runtime import sha256_file
    from numerical_experiment.picar.storage import FrameStore, RunStore
    from numerical_experiment.picar_reference_reward_probe import prepare_references

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA Qwen required")
    torch.set_num_threads(2)
    metadata, baseline, refined, development = sources(args)
    panel_unit = read_completed_unit(args.panels_run)
    if panel_unit["summary"].get("kind") != "width_refinement_panels_v1":
        raise ValueError("expected fresh refinement panels")
    panels = panel_unit["summary"]["panels"]
    check_disjoint(panels, refined, development)
    if len(panels) != 20 or sum(not p["truth"]["visible"] for p in panels) != 3:
        raise ValueError("fresh diagnostic coverage mismatch")
    for panel in panels:
        if sha256_file(args.panels_run / panel["mask_file"]) != panel["mask_sha256"]:
            raise ValueError("fresh mask checksum mismatch")
    frames = FrameStore(args.panels_run / "frames", readonly=True)
    reference_frames = {
        "original": FrameStore(Path(metadata["reference_frames_root"]) / "frames", readonly=True),
        "development": FrameStore(args.development_run / "frames", readonly=True)}
    references = [{**r, "rgb": reference_frames[r["source_group"]].get(r["frame_ref"])} for r in refined]
    banks = {METHODS[0]: [next(r for r in references if r["reference"] == old["reference"]) for old in baseline],
             METHODS[1]: references}
    jobs = [(p, method) for p in panels for method in METHODS]
    settings = {**metadata, "kind": "width_refinement_evaluation_v1", "prompt": PROMPT,
        "panels_run": str(args.panels_run.resolve()), "panels_completion_sha256": sha256_file(args.panels_run / "COMPLETED"),
        "panels": panels, "methods": {m: [r["reference"] for r in bank] for m, bank in banks.items()},
        "model_override": str(args.model_path) if args.model_path else None, "model_seed": 191101,
        "max_new_tokens": 96, "max_tokens": 2048, "max_pixels_per_image": 76800,
        "role": "evaluator", "example_answers": ["width_percent", "reward"]}
    with RunStore(args.output, settings, resume=args.resume) as store:
        if store.completed:
            print("Already complete:", store.root)
            return
        rows = (store.load_checkpoint(trusted=True) or {}).get("rows", [])
        if [(r["panel_id"], r["method"]) for r in rows] != [(p["panel_id"], m) for p, m in jobs[:len(rows)]]:
            raise ValueError("checkpoint job order mismatch")
        model = load_qwen_model(**({"model_path": args.model_path} if args.model_path else {}),
            local_files_only=args.offline, seed=191101, top_layers=1, ltm_rank=4, stm_rank=1,
            hidden_size=64, context_length=4, return_bins=161, max_tokens=2048).eval()
        for panel, method in jobs[len(rows):]:
            tick = time.monotonic()
            inputs = prepare_references(model.encoder, banks[method], frames.get(panel["frame_ref"]), few_shot=True)
            if int(inputs["image_grid_thw"].shape[0]) != len(banks[method]) + 1:
                raise ValueError("unexpected image count")
            patches = int(inputs["image_grid_thw"][-1].prod())
            target_pixels = inputs["pixel_values"][-patches:]
            raw = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=96)[0]
            row = {"panel_id": panel["panel_id"], "method": method, "state": panel["state"],
                "frame_ref": panel["frame_ref"], "raw_text": raw, "seconds": time.monotonic() - tick,
                "input_tokens": int(inputs["input_ids"].shape[1]), "image_count": len(banks[method]) + 1,
                "target_pixels_sha256": hashlib.sha256(target_pixels.contiguous().numpy().tobytes()).hexdigest(),
                "true_width_percent": panel["truth"]["width_percent"], "true_visible": panel["truth"]["visible"],
                "partial_occlusion": panel["scene"]["barrier"].startswith("partial")}
            try:
                row.update(parse(raw))
                row.update(error_fields(row["width_percent"], row["reward"], row["true_width_percent"]))
            except ValueError as exc:
                row["parse_error"] = str(exc)
            rows.append(row)
            store.checkpoint({"rows": rows}, progress={"phase": 3, "status": "four versus six examples",
                "counts": {"scored": len(rows), "target": len(jobs)}})
            print(json.dumps(row), flush=True)
        metrics = summarize(panels, rows)
        render_figures(store.root, references, panels, frames, rows)
        with (store.root / "judgments.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for r in rows for key in r}))
            writer.writeheader()
            writer.writerows(rows)
        store.complete({"schema_version": 1, "phase": 3, "kind": settings["kind"], "smoke_only": True,
            "development_only": True, "rows": rows, "metrics": metrics, "model_provenance": model.encoder.provenance,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "image_sha256": {name: sha256_file(store.root / name) for name in
                ("references.png", "calibration.png", "four_examples.png", "six_examples.png")},
            "judgments_sha256": sha256_file(store.root / "judgments.csv"),
            "limitations": "Fixed purposive diagnostic panel; not population inference or evidence of learning."})
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("panels", "evaluate"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--panels-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "evaluate" and args.panels_run is None:
        parser.error("evaluate requires --panels-run")
    (make_panels if args.stage == "panels" else evaluate)(args)


if __name__ == "__main__":
    main()
