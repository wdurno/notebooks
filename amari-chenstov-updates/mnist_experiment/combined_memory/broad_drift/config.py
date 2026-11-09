"""Phase 5 settings with the original numerical/production constraints intact.

``analysis_novel_count`` is the outbound-grid budget, not a rare-component
count. ``novel_probability`` survives only for compatibility with the original
source allocator; it has no effect on the resolved broad-drift law.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.seeding import derive_component_seed
from ..config import Config as OriginalConfig, content_hash
from . import PROTOCOL, SCHEMA_VERSION


@dataclass(frozen=True)
class Config(OriginalConfig):
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA_VERSION
    seed: int = 18110705
    analysis_novel_count: int | None = None

    def __post_init__(self):
        if self.analysis_novel_count is None:
            object.__setattr__(self, "analysis_novel_count", self.outbound_budget)

    @property
    def outbound_budget(self):
        return self.ramp_steps * self.samples_per_step

    def compatibility_projection(self):
        """Project only amended identity/budget fields for the old validator.

        Do not set ``smoke=True`` here: that would disable the inherited strict
        production checks on the unchanged scientific and numerical design.
        """
        original = OriginalConfig()
        values = self.mapping()
        values.update(protocol=original.protocol, schema_version=original.schema_version,
                      analysis_novel_count=original.analysis_novel_count)
        return OriginalConfig.from_mapping(values)

    def validate(self):
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA_VERSION:
            raise ValueError("incompatible broad-drift protocol/schema")
        if type(self.smoke) is not bool:
            raise ValueError("smoke must be a boolean")
        if (type(self.analysis_novel_count) is not int
                or self.analysis_novel_count != self.outbound_budget):
            raise ValueError("analysis_novel_count must equal the full outbound-grid budget")
        self.compatibility_projection()
        if self.seed >= 2**63:
            raise ValueError("seed must lie in [1, 2**63)")

    def seed_for(self, cohort, replica, component):
        if type(replica) is not int or replica < 0:
            raise ValueError("replica index must be a nonnegative integer")
        if not isinstance(cohort, str) or not cohort or not isinstance(component, str) or not component:
            raise ValueError("cohort and component must be nonempty strings")
        # This sole legacy call defines the study-wide development/test split.
        if (cohort, replica, component) == ("study-partition", 0, "test"):
            return OriginalConfig(seed=181106).seed_for(cohort, replica, component)
        return derive_component_seed(self.seed, f"{self.protocol}:{cohort}:{replica}:{component}")

    @classmethod
    def smoke_config(cls, device="cpu"):
        values = OriginalConfig.smoke_config(device).mapping()
        values.update(protocol=PROTOCOL, schema_version=SCHEMA_VERSION, seed=18110705,
                      novel_probability=.125,
                      analysis_novel_count=values["ramp_steps"] * values["samples_per_step"])
        return cls.from_mapping(values)
