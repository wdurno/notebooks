"""Record a supplemental fixed-checkpoint PiCar evaluation, without learning.

This explicit GPU entry point is separate from the frozen Fix 3 campaign.
Its recording is a new evaluation, not recovered footage of an earlier trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from numerical_experiment.picar.runtime import StopController, sha256_file
from numerical_experiment.picar.storage import RunStore


def select_panel(state, summary, difficulty, layout_seed):
    panel = next(p for p in state["identity"]["evaluations"] if p["difficulty"] == difficulty)
    rows = next(p["rows"] for p in summary["evaluations"]
                if p["checkpoint"] == "final" and p["difficulty"] == difficulty)
    seed = panel["seeds"][0] if layout_seed is None else layout_seed
    if seed not in panel["seeds"]:
        raise ValueError("layout seed must belong to the original evaluation panel")
    row = next(r for r in rows if r["seed"] == seed)
    return panel, row


def assert_unchanged(before, after):
    """Check parameters, optimizer, Fisher, anchors, RNGs and update counters."""
    import torch
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor) and torch.equal(before.cpu(), after.cpu())
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_unchanged(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after)
        for first, second in zip(before, after):
            assert_unchanged(first, second)
    else:
        assert before == after


def render_video(runner, current, difficulty):
    from PIL import Image
    from numerical_experiment.picar.report import assemble_video
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig
    snapshots, rows = current["frames"], []
    config = SimulatorConfig.from_dict(snapshots[0]["config"])
    with PyBulletBackend(config, run_id=snapshots[0]["run_id"]) as renderer:
        for snapshot in snapshots:
            renderer.restore(snapshot)
            at = renderer.episode_time_s
            if rows and at <= rows[-1]["time_s"]:
                continue
            paths = []
            for rgb in (renderer.render(), renderer.render_topdown()):
                ref = runner.store.frames.put(rgb)
                path = runner.store.root / (ref[7:] + ".png")
                if not path.exists():
                    Image.fromarray(rgb).save(path)
                paths.append(str(path))
            rows.append({"time_s": at, "ego_path": paths[0], "topdown_path": paths[1],
                "drift_alpha": renderer.diagnostics()["alpha"],
                "instruction": f"SUPPLEMENTAL final student replay | {difficulty.upper()} | no learning",
                "outcome": renderer.diagnostics()["end_reason"] or "running"})
    path = runner.store.root / f"supplemental-final-{difficulty}-{current['seed']}-{time.time_ns()}.mp4"
    return [assemble_video(rows, path, seed=current["seed"], selection="failure_gallery", fps=5.)]


def run(source, output, *, difficulty="hard", layout_seed=None, model_path=None,
        offline=False, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_config = json.loads((source / "config.json").read_text())
    pointer = json.loads((source / "CURRENT.json").read_text())
    transaction = source / "transactions" / pointer["commit_id"]
    transaction_manifest = json.loads((transaction / "manifest.json").read_text())
    identity = {"version": "supplemental-evaluation-video-v1", "source": str(source),
        "source_completion_sha256": sha256_file(source / "COMPLETED"),
        "source_checkpoint_sha256": transaction_manifest["recovery_state"]["sha256"],
        "difficulty": difficulty, "layout_seed": layout_seed,
        "model_source": str(Path(model_path).resolve()) if model_path else None,
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "learning": False, "exploration": False, "reward_scoring": False,
        "selection": "failure_gallery", "not_original_trial_footage": True}
    started = time.monotonic()
    with RunStore(output, identity, resume=resume, checkpoint_policy="journal-v1") as store:
        if store.completed:
            return json.loads((output / "summary.json").read_text())
        with RunStore(source, source_config, resume=True, checkpoint_policy="journal-v1") as original:
            if not original.completed:
                raise ValueError("source must be a completed immutable pilot")
            state = original.load_checkpoint(trusted=True)
            summary = json.loads((source / "summary.json").read_text())
        if state["stage"] != "done":
            raise ValueError("source checkpoint must be final")
        panel, old_row = select_panel(state, summary, difficulty, layout_seed)
        from numerical_experiment.picar.config import config_from_dict
        from numerical_experiment.picar.fix3_student import _Run
        from numerical_experiment.picar.models import load_qwen_model
        from src.robotics.simulator import SimulatorConfig
        import torch
        torch.set_num_threads(2)
        config = config_from_dict(state["identity"]["config"])
        kwargs = {} if model_path is None else {"model_path": model_path}
        model = load_qwen_model(**kwargs, local_files_only=offline,
            seed=source_config["model_initialization_seed"],
            top_layers=config.model.top_layers, ltm_rank=config.model.ltm_rank,
            stm_rank=config.model.stm_rank, hidden_size=config.model.hidden_size,
            context_length=config.model.context_length, return_bins=config.model.return_bins,
            quantized=config.model.quantized, min_pixels=56*56, max_pixels=19200,
            max_tokens=2048).eval()
        model.encoder.enable_prefix_cache = True
        stop = StopController(output)
        stop.install()
        def stopped():
            if stop.check():
                raise InterruptedError("video evaluation checkpointed; use --resume")
        try:
            runner = _Run(model, config, SimulatorConfig.from_dict(panel["config"]),
                store, stopped, mode="pilot", seed=summary["seed"], max_transitions=320,
                initial=state)
            before = runner.learner.state_dict()
            runner.episode(runner.sim_config, seed=old_row["seed"],
                exposure=old_row["reset_exposure"], evaluation=True, video=True)
            assert_unchanged(before, runner.learner.state_dict())
            current = runner.s["current"]
            videos = render_video(runner, current, difficulty)
            for clip in videos:
                clip.update(selection="failure_gallery", supplemental=True,
                    controller="trained student, final checkpoint; no learning",
                    not_original_trial_footage=True)
            result = {"schema_version": 1, "kind": "supplemental_evaluation_video",
                "phase": 1, "smoke_only": True, "student_seed": summary["seed"],
                "layout_seed": old_row["seed"], "difficulty": difficulty,
                "end_reason": current["end_reason"], "success": current["success"],
                "physical_seconds": sum(x["elapsed_s"] for x in current["intervals"]),
                "original_end_reason": old_row["end_reason"],
                "original_physical_seconds": old_row["physical_seconds"],
                "learning_state_unchanged": True, "videos": videos,
                "active_seconds": time.monotonic() - started,
                "not_original_trial_footage": True,
                "limitation": "Same final checkpoint and layout, but fresh real-time scheduling."}
            runner.save()
            store.complete(result)
            return result
        finally:
            stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("easy", "middle", "hard"), default="hard")
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.source, args.output, difficulty=args.difficulty,
        layout_seed=args.layout_seed, model_path=args.model_path,
        offline=args.offline, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
