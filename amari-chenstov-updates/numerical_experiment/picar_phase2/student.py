"""Phase 2 exploratory readiness, reusing the frozen Fix3 learning/recovery path.

The caller owns budgets, curriculum, panels and initial-asset provenance. Pilot
initialization loads the completed bootstrap without repeating its recorded KL.
"""

from __future__ import annotations

import copy
import time

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.fix2_data import TASK
from numerical_experiment.picar.fix3_student import _Run, _batch
from numerical_experiment.picar.models import batch_digest
from src.robotics.simulator import PyBulletBackend


PHASE2_SCHEMA = 1


class _Phase2Store:
    """Adapt identity checking and labels locally, never patch the frozen runner."""

    def __init__(self, store, marker):
        self.store, self.marker = store, marker

    def __getattr__(self, name):
        return getattr(self.store, name)

    def load_checkpoint(self, **kwargs):
        saved = self.store.load_checkpoint(**kwargs)
        if saved is None:
            return None
        identity = dict(saved["identity"])
        if canonical_json(identity.pop("phase2", None)) != canonical_json(self.marker):
            raise ValueError("Phase 2 resume identity differs")
        return {**saved, "identity": identity}

    def checkpoint(self, state, *, metrics=(), progress=None, **kwargs):
        progress = {**(progress or {}), "phase": 2,
            "status": f"Phase 2 {state['identity']['mode']}: {state['stage']}"}
        return self.store.checkpoint(state, metrics=[{**r, "phase": 2} for r in metrics],
                                     progress=progress, **kwargs)


def _terminal_diagnostics(session):
    snapshot = (session or {}).get("worker", {}).get("backend")
    result = {"source": "unavailable_no_terminal_worker_snapshot", "clearance_m": None,
        "heading_error_rad": None, "target_contact": None, "unsafe_contact": None,
        "physical_seconds": None}
    if snapshot is not None:
        with PyBulletBackend.from_snapshot(snapshot) as renderer:
            diagnostic = renderer.diagnostics()
        result.update(source="terminal_worker_snapshot",
            clearance_m=diagnostic["geometry"]["clearance_m"],
            heading_error_rad=diagnostic["geometry"]["heading_error_rad"],
            target_contact=diagnostic["target_contact"], unsafe_contact=diagnostic["unsafe_contact"],
            physical_seconds=diagnostic["physical_time_s"])
    return result


def _coverage(intervals):
    probabilities = torch.tensor([r["probabilities"] for r in intervals
                                  if r["probabilities"] is not None], dtype=torch.float64).reshape(-1, 8)
    result = {"policy_intervals": len(probabilities),
        "dominant_action_counts": torch.bincount(probabilities.argmax(-1), minlength=8).tolist(),
        "mean_probabilities": probabilities.mean(0).tolist() if len(probabilities) else None}
    for key in ("drive_mps", "steering_rad"):
        values = [r["controls"][key] for r in intervals]
        result[key + "_range"] = [min(values), max(values)] if values else None
    return result


class ReadinessRun(_Run):
    def __init__(self, model, config, sim_config, store, stopped, *, mode="pilot", seed,
                 max_transitions, initial=None, teacher_variant="centering", evaluations=()):
        if mode not in {"pilot", "bootstrap"}:
            raise ValueError("Phase 2 mode must be pilot or bootstrap")
        panels = copy.deepcopy(list(evaluations))
        for panel in panels:
            panel["seeds"] = tuple(panel["seeds"])
            if (not panel["seeds"] or len(set(panel["seeds"])) != len(panel["seeds"])
                    or any(type(s) is not int or s < 0 for s in panel["seeds"])):
                raise ValueError("evaluation panels require unique nonnegative integer seeds")
        if len({p["difficulty"] for p in panels}) != len(panels):
            raise ValueError("evaluation difficulty labels must be unique")
        marker = {"schema_version": PHASE2_SCHEMA, "phase": 2, "mechanism": "exploratory",
            "stm_efficacy": False, "video_selection": "first_seed_each_panel",
            "evaluation_diagnostics": "terminal_snapshot_v1"}
        super().__init__(model, config, sim_config, _Phase2Store(store, marker), stopped,
            mode=mode, seed=seed, max_transitions=max_transitions, initial=initial,
            teacher_variant=teacher_variant, evaluations=panels)
        self.identity["phase2"] = marker
        self.s.update(identity=self.identity, phase=2, phase2=marker,
                      mechanism="exploratory", stm_efficacy=False)

    def evaluate(self, checkpoint):
        """Frozen student panels; resumable rows pair the same seeds before/after."""
        if checkpoint not in {"initial", "final"}:
            raise ValueError("checkpoint must be initial or final")
        s = self.s
        if s["pending"] or s["ready"] or (s["current"] and not s["current"]["evaluation"]):
            raise RuntimeError("evaluation requires a drained training episode boundary")
        for panel in self.evaluations:
            found = next((e for e in s["evaluations"] if e["checkpoint"] == checkpoint
                          and e["difficulty"] == panel["difficulty"]), None)
            if found is None:
                found = {"checkpoint": checkpoint, "difficulty": panel["difficulty"], "rows": []}
                s["evaluations"].append(found)
            for seed in panel["seeds"][len(found["rows"]):]:
                before = time.monotonic()
                video = seed == panel["seeds"][0]
                self.episode(panel["config"], seed=seed, exposure=0., evaluation=True, video=video)
                current = s["current"]
                if current["truncated"]:
                    raise RuntimeError("evaluation infrastructure truncation is not a task failure")
                terminal = _terminal_diagnostics(s["session"])
                clips = self.render_video(current, checkpoint, panel["difficulty"]) if video else []
                row = {k: v for k, v in current.items() if k not in {"frames", "label_complete"}}
                row.update(explore=False, learning=False, reward_scoring=False, actor="learned_student",
                    videos=clips, terminal_diagnostics=terminal, coverage=_coverage(current["intervals"]),
                    physical_seconds=sum(x["elapsed_s"] for x in current["intervals"]))
                found["rows"].append(row)
                s["videos"].extend(clips)
                s["session"], s["history"], s["current"], s["pending"] = None, None, None, []
                s["timings"]["evaluation_seconds"] += time.monotonic() - before
                self.save()
        s["stage"] = "collect" if checkpoint == "initial" else "done"
        self.save()

    def render_video(self, current, checkpoint, difficulty):
        from PIL import Image
        from numerical_experiment.picar.report import assemble_video

        snapshots = current["frames"]
        if not snapshots:
            return []
        label = f"learned student | {checkpoint} | {difficulty} | seed={current['seed']}"
        rows = []
        with PyBulletBackend.from_snapshot(snapshots[0]) as renderer:
            for snapshot in snapshots:
                renderer.restore(snapshot)
                timestamp = renderer.time_s
                if rows and timestamp <= rows[-1]["time_s"]:
                    continue
                paths = []
                for rgb in (renderer.render(), renderer.render_topdown()):
                    ref = self.store.frames.put(rgb)
                    path = self.store.root / (ref[7:] + ".png")
                    if not path.exists():
                        Image.fromarray(rgb).save(path)
                    paths.append(str(path))
                rows.append({"time_s": timestamp, "ego_time_s": timestamp, "topdown_time_s": timestamp,
                    "ego_path": paths[0], "topdown_path": paths[1], "instruction": TASK,
                    "outcome": f"{label} | {renderer.diagnostics()['end_reason'] or 'running'}"})
        path = self.store.root / f"phase2-student-{checkpoint}-{difficulty}-{current['seed']}-{time.time_ns()}.mp4"
        result = assemble_video(rows, path, seed=current["seed"], fps=5.)
        return [{**result, "checkpoint": checkpoint, "difficulty": difficulty, "label": label,
            "actor": "learned_student", "layout_id": current["seed"], "view": "ego+topdown",
            "views": "synchronized_reconstruction_of_recorded_physics_snapshots"}]

    def run(self):
        summary = super().run()
        return {**summary, "kind": f"phase2_{self.mode}", "phase": 2,
            "phase2": self.identity["phase2"], "mechanism": "exploratory", "stm_efficacy": False}


@torch.no_grad()
def language_readiness(learner, eligible=None):
    """Descriptive counterfactuals only; same retained pixels/actions/times, no fit.

    Accept a runner, or a learner with explicit eligible records. This changes
    inference caches, not parameters, optimizer, replay or RNGs.
    It tests responsiveness, not whether the instructed behavior is correct.
    """
    if eligible is None:
        learner, eligible = learner.learner, learner.s["eligible"]
    if not eligible:
        raise ValueError("language readiness requires stored eligible histories")
    rows = []
    for item in eligible:
        source = _batch(item, eligible).histories[0]
        digest = batch_digest(source)
        for name, instruction in (("forward", "Move forward."), ("left", "Turn left."),
                                  ("right", "Turn right."), ("stop", "Stop.")):
            history = copy.deepcopy(source)
            history["contexts"] = [[{"text": instruction, "issued_at": 0.}]]
            action = learner.choose_action(history, explore=False)
            features, _ = learner.model.encoder.inference_features(history, role="learner")
            rows.append({"arrival": item["arrival"], "source_history_digest": digest,
                "condition": name, "instruction": instruction, "logits": action["logits"].tolist(),
                "probabilities": action["probabilities"].tolist(),
                "features": features[0, -1].detach().cpu().tolist()})
    return {"phase": 2, "schema_version": PHASE2_SCHEMA, "mechanism": "exploratory",
            "learning": False, "explore": False, "model_version": learner.version, "rows": rows}
