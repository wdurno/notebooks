"""Operator work on the independent control clock, before inference submission."""

from numerical_experiment.picar.control import ControlSession


class CoachingSession(ControlSession):
    def __init__(self, *args, before_submit, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_submit = before_submit

    def _submit(self, observation):
        # ControlSession holds its cycle lock here. Recovery captures this
        # operator state and the inference job together, without pausing physics.
        observation = self.before_submit(self, observation)
        super()._submit(observation)
