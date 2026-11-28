"""Assemble an outcome-selected example from original saved camera observations.

This offline CPU producer never runs a policy, simulator, evaluator or optimizer.
The source summary preserves each scored image's raw-pixel hash, even when
the corresponding episode was not selected for a paired camera/top-down MP4.
"""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from numerical_experiment.picar_head_rank.reuse import digest, receipt


VERSION = "plan19-saved-success-example-v1"


def episode_images(episode, frames):
    import numpy as np

    targets = [row["diagnostic"]["target"] for row in episode["scores"]]
    wanted = {target["sha256"] for target in targets}
    found = {}
    for path in sorted(Path(frames).glob("*.npy")):
        if path.is_symlink() or path.stat().st_size > 32_001_024:
            raise ValueError("unsafe or oversized stored image")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("stored image must be HWC uint8 RGB")
        key = hashlib.sha256(array.tobytes()).hexdigest()
        if key in wanted:
            if digest(path) != path.stem:
                raise ValueError("stored image content-address checksum mismatch")
            found[key] = (path, array)
            if wanted <= found.keys():
                break
    if not wanted <= found.keys():
        raise ValueError("original episode camera observations are unavailable")
    result = []
    for target in targets:
        path, array = found[target["sha256"]]
        if target["shape"] != list(array.shape) or target["dtype"] != "uint8":
            raise ValueError("scored image identity differs from saved image")
        result.append((path, array))
    return result


def run(source, output, episode_id, *, resume=False):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    from numerical_experiment.picar.storage import RunStore

    source, output = Path(source).resolve(), Path(output).resolve()
    seal = receipt(source)
    summary = json.loads((source / "summary.json").read_text())
    matches = [(panel, episode) for panel in summary["evaluations"]
               for episode in panel["episodes"] if episode["episode_id"] == episode_id]
    if len(matches) != 1:
        raise ValueError("episode must identify one saved evaluation")
    panel, episode = matches[0]
    if not (episode["success"] and panel["n"] == summary["settings"]["total"]
            and panel["environment"] == "moderate_A"):
        raise ValueError("select a successful moderate-A endpoint episode")
    identity = {"version": VERSION, "source_receipt": seal, "episode_id": episode_id,
                "producer_sha256": digest(__file__), "selection": "outcome-selected success",
                "learning": False, "new_evaluation": False}
    with RunStore(output, identity, resume=resume, checkpoint_policy="journal-v1") as store:
        if store.completed:
            return json.loads((output / "summary.json").read_text())
        images = episode_images(episode, source / "frames")
        name = "saved-camera-" + uuid.uuid4().hex + ".mp4"
        path = output / name
        with imageio.get_writer(path, fps=5, codec="libx264", macro_block_size=2) as writer:
            for i, (_, array) in enumerate(images):
                canvas = Image.new("RGB", (480, 412), "#14171a")
                canvas.paste(Image.fromarray(array).resize((480, 360)), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 366), "OUTCOME-SELECTED SUCCESS | moderate A | final checkpoint", fill="white")
                draw.text((8, 383), f"Saved camera observation {i+1}/{len(images)}; no top-down saved", fill="white")
                for _ in range(5):
                    writer.append_data(np.asarray(canvas))
        if receipt(source) != seal:
            raise ValueError("source changed while assembling video")
        result = {**identity, "schema_version": 1, "source_unit": str(source),
                  "method": summary["settings"]["method"], "seed": summary["settings"]["seed"],
                  "ewc_rank": summary["settings"]["ewc_rank"], "environment": panel["environment"],
                  "n": panel["n"], "actual_acquisitions": episode["n"], "success": True,
                  "raw_return": episode["raw_return"], "layout_seed": episode["seed"],
                  "direct_path_blocked": episode["layout"]["curriculum"]["direct_path_blocked"],
                  "video": name, "video_sha256": digest(path),
                  "camera_frames": [p.name for p, _ in images],
                  "playback": "One original scored camera observation per second; no motion interpolation.",
                  "not_representative": True, "topdown_available": False}
        store.checkpoint({"stage": "done"})
        store.complete(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.episode_id, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
