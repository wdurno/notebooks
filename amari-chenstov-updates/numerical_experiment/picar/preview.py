"""Generate a fixed-seed simulator/interface preview, not a capability test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import PicarConfig
from .runtime import sha256_file


def create_preview(output: Path):
    import numpy as np
    from PIL import Image
    from src.robotics.protocol import Control
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig
    from .report import assemble_video
    from .storage import RunStore

    config = PicarConfig()
    with RunStore(output, {"config": config.to_dict(), "purpose": "fixed_seed_render_acceptance"}) as store:
        with PyBulletBackend(SimulatorConfig(width=320, height=240, control_interval_s=.1,
                deadline_s=3., total_training_intervals=30, realtime=False), run_id="preview") as sim:
            observation = sim.reset(seed=190001)
            frames = []
            for index in range(31):
                ego, top = sim.render(), sim.render_topdown()
                if min(float(ego.std()), float(top.std())) < 5:
                    raise AssertionError("camera preview is nearly blank")
                ego_path, top_path = output / f"ego-{index:03d}.png", output / f"top-{index:03d}.png"
                Image.fromarray(ego).save(ego_path)
                Image.fromarray(top).save(top_path)
                frames.append({"time_s": sim.episode_time_s, "ego_path": str(ego_path),
                    "topdown_path": str(top_path), "instruction": "Fixed scripted motion: rendering test only",
                    "outcome": sim.diagnostics()["end_reason"], "drift_alpha": sim.alpha,
                    "executed_control": asdict(observation.executed_controls)})
                if sim.diagnostics()["end_reason"]:
                    break
                observation = sim.step(Control(drive_mps=.18, steering_rad=.12)).observation
            video = assemble_video(frames, output / "scene.mp4", seed=190001, fps=10.)
            video["sha256"] = sha256_file(output / "scene.mp4")
            store.checkpoint({"simulator": sim.snapshot()}, progress={"phase": 0, "status": "done"})
            store.complete({"phase": 0, "smoke_only": True, "kind": "visual_acceptance",
                "videos": [video], "camera_std": float(np.asarray(ego).std()),
                "topdown_std": float(np.asarray(top).std()), "scene_frames": len(frames)})
    return video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(create_preview(args.output))


if __name__ == "__main__":
    main()
