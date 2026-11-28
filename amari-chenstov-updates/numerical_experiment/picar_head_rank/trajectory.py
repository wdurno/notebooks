"""Reporting instrumentation around the unchanged Fix 5 trajectory."""

from dataclasses import astuple
import math

from numerical_experiment.picar_head_awac.trajectory import Trajectory
from numerical_experiment.picar_head_action_report import control_summary
from src.robotics.actions import Calibration


class RankTrajectory(Trajectory):
    def save(self, session=None, history=None):
        if self.s.get("current") is None:
            snapshots = self.s.setdefault("archive_metrics", {})
            key = str(self.s["n"])
            snapshots[key] = {"n": self.s["n"], "fits": self.learner.version,
                              **self.learner.archive.diagnostics()}
        super().save(session, history)

    def finish_episode(self):
        episode = self.s["current"]
        if episode is not None and not episode["evaluation"]:
            calibration = Calibration(**episode["simulator_config"]["calibration"])
            sequence = []
            for i, row in enumerate(episode["rows"]):
                if row["action"] is None:
                    controls = calibration.neutral()
                else:
                    values = [*row["action"].tolist(), 0.]
                    weights = [math.exp(v - max(values)) for v in values]
                    controls = calibration.map_distribution([w / sum(weights) for w in weights])
                sequence.append((i, 1., astuple(controls)))
            summary = control_summary(sequence)
            summary["moving_commands"] = int(summary.pop("moving_seconds"))
            summary.update(episode_id=episode["episode_id"], n=episode["n"],
                           sigma_m=episode["layout"]["curriculum"]["sigma_m"],
                           success=episode["success"], expected_intervals=len(sequence),
                           complete_history=True)
            self.s.setdefault("action_metrics", {})[episode["episode_id"]] = summary
        super().finish_episode()

    def run(self):
        result = super().run()
        from .numerics import block_diagnostics
        result["allocation"] = block_diagnostics(self.learner.archive)
        result["action_metrics"] = list(self.s.get("action_metrics", {}).values())
        result["archive_metrics"] = list(self.s.get("archive_metrics", {}).values())
        return result
