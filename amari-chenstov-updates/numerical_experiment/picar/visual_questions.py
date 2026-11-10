"""Independent frozen-Qwen questions on saved RGB; no simulator or training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import time

from .config import PicarConfig
from .runtime import StopController, sha256_file, source_fingerprint
from .storage import RunStore


QUESTIONS = (
    "What objects do you see?",
    "Where is the red ball?",
    "Where is the ball?",
    "Do you see a sphere?",
    "How far away is the sphere?",
    "How far away is the red sphere?",
)
SYSTEM = "You are a helpful assistant."


def read_rgb(path):
    import numpy as np
    from PIL import Image
    path = Path(path)
    if path.suffix == ".npy":
        rgb = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 8 or max(rgb.shape[:2]) > 2048):
        raise ValueError("expected bounded HWC uint8 RGB")
    return rgb


def question_inputs(encoder, rgb, question, *, max_pixels=76800):
    import torch
    # Each request has one image and one question, with no task/history/labels.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    text = encoder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    inputs = dict(encoder.processor(text=[text], images=[image], padding=True,
        return_tensors="pt", do_rescale=False, max_pixels=max_pixels))
    if inputs["input_ids"].shape[1] > encoder.max_tokens:
        raise ValueError("diagnostic question exceeds model token allowance")
    return inputs


def run(output, images, *, model_path=None, offline=False, resume=False, max_minutes=10.):
    if not images or len({label for label, path in images}) != len(images):
        raise ValueError("provide nonempty, uniquely labeled images")
    if not 0 < max_minutes <= 30:
        raise ValueError("diagnostic limit must be in (0,30] minutes")
    sources = source_fingerprint()
    base = PicarConfig()
    config = replace(base, version="visual-questions-v1", mode="production",
        image_width=320, image_height=240,
        model=replace(base.model, kind="qwen", hidden_size=64, context_length=4, return_bins=41),
        runtime=replace(base.runtime, device="cuda", threads=2))
    config.validate()
    identity = {"config": config.to_dict(),
        "kind": "visual_questions", "questions": QUESTIONS, "system_prompt": SYSTEM,
        "images": [{"label": label, "path": str(Path(path).resolve()),
                    "sha256": sha256_file(Path(path))} for label, path in images],
        "source_sha256": sources, "max_minutes": max_minutes,
        "model_source": str(Path(model_path).resolve()) if model_path else None,
        "max_pixels": 76800, "max_new_tokens": 128, "model_initialization_seed": 191101,
        "role": "evaluator", "decoding": "greedy", "fresh_conversation_per_question": True}
    output = Path(output)
    stop = StopController(output)
    stop.install()
    try:
        with RunStore(output, identity, resume=resume, checkpoint_policy="journal-v1") as store:
            if store.completed:
                return json.loads((output / "summary.json").read_text())
            saved = store.load_checkpoint(trusted=True) or {"rows": [], "active_seconds": 0.}
            rows, prior = saved["rows"], saved["active_seconds"]
            started = time.monotonic()
            def save(status):
                store.checkpoint({"rows": rows, "active_seconds": prior + time.monotonic() - started},
                    progress={"phase": 1, "status": status,
                              "counts": {"step": len(rows), "target": len(images) * len(QUESTIONS)}})
            try:
                import torch
                from .models import load_qwen_model
                from .role_repair import native_text
                if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                    raise RuntimeError("diagnostic needs CUDA with at least 8 GiB free")
                torch.set_num_threads(2)
                kwargs = {} if model_path is None else {"model_path": model_path}
                model = load_qwen_model(**kwargs, local_files_only=offline or model_path is not None,
                    seed=191101, top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64,
                    context_length=4, return_bins=41, min_pixels=56 * 56, max_pixels=19200).eval()
                pending = [(label, path, question) for label, path in images for question in QUESTIONS]
                for label, path, question in pending[len(rows):]:
                    if stop.check() or prior + time.monotonic() - started >= max_minutes * 60:
                        raise InterruptedError("visual questions checkpointed; use --resume")
                    rgb = read_rgb(path)
                    frame_ref = store.frames.put(rgb)
                    before = time.monotonic()
                    inputs = question_inputs(model.encoder, rgb, question)
                    answer = native_text(model.encoder, inputs, role="evaluator", max_new_tokens=128)[0]
                    torch.cuda.synchronize()
                    rows.append({"image": label, "frame_ref": frame_ref, "question": question,
                        "answer": answer, "seconds": time.monotonic() - before,
                        "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
                        "input_tokens": int(inputs["attention_mask"].sum()),
                        "captured_width_height": [rgb.shape[1], rgb.shape[0]]})
                    save("frozen visual questions")
                    print(json.dumps(rows[-1]), flush=True)
                if source_fingerprint() != sources:
                    raise RuntimeError("sources changed; preserve this diagnostic and version a new run")
                result_csv = io.StringIO(newline="")
                writer = csv.DictWriter(result_csv, fieldnames=("image", "question", "answer", "seconds"))
                writer.writeheader()
                writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
                (output / "responses.csv").write_text(result_csv.getvalue())
                summary = {"kind": "phase1_visual_questions", "phase": 1, "smoke_only": True,
                    "diagnostic_only": True, "training_performed": False, "physical_hardware_used": False,
                    "rows": rows, "model_provenance": model.encoder.provenance,
                    "device": torch.cuda.get_device_name(), "dtype": "BF16 frozen prefix; FP32 suffix/readout",
                    "active_seconds": prior + time.monotonic() - started,
                    "responses_csv_sha256": sha256_file(output / "responses.csv")}
                save("frozen visual questions complete")
                store.complete(summary)
                return summary
            finally:
                if not store.completed:
                    save("visual questions incomplete; explicit resume required")
    finally:
        stop.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=10.)
    args = parser.parse_args(argv)
    images = []
    for value in args.image:
        if "=" not in value or not value.split("=", 1)[0]:
            parser.error("--image requires LABEL=PATH")
        label, path = value.split("=", 1)
        images.append((label, Path(path)))
    if args.resume:
        (args.output / "STOP").unlink(missing_ok=True)
    result = run(args.output, images, model_path=args.model_path, offline=args.offline,
                 resume=args.resume, max_minutes=args.max_minutes)
    print(json.dumps({"complete": True, "answers": len(result["rows"]),
                      "active_seconds": result["active_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
