"""Context identities around the unchanged historical capacity runner."""

from mnist_experiment.fisher_memory_stm_capacity.runner import Runner as LegacyRunner
from mnist_experiment.fisher_memory_regression.runtime import read_json


class NamespaceStore:
    def __init__(self, base, namespace, context):
        self.base, self.namespace, self.context = base, namespace, context

    def __getattr__(self, name):
        return getattr(self.base, name)

    def key(self, unit):
        # The no-blur branch reuses the exact same initial assets.
        context = self.context if "trajectory-" in unit else "shared"
        return f"{self.namespace}-{context}--{unit}"

    def unit_path(self, unit):
        return self.base.unit_path(self.key(unit))

    def is_complete(self, unit, **kwargs):
        return self.base.is_complete(self.key(unit), **kwargs)

    def begin(self, unit, metadata):
        return self.base.begin(self.key(unit), dict(metadata, namespace=self.namespace,
                                                   blur_context=self.context if "trajectory-" in unit else "shared"))

    def write_json(self, unit, name, value, **kwargs):
        return self.base.write_json(self.key(unit), name, value, **kwargs)

    def write_csv(self, unit, name, value, **kwargs):
        return self.base.write_csv(self.key(unit), name, value, **kwargs)

    def save_checkpoint(self, unit, state, **kwargs):
        return self.base.save_checkpoint(self.key(unit), state, **kwargs)

    def load_checkpoint(self, unit, **kwargs):
        return self.base.load_checkpoint(self.key(unit), **kwargs)

    def complete(self, unit, **kwargs):
        return self.base.complete(self.key(unit), **kwargs)

    def iter_units(self, **kwargs):
        prefix = f"{self.namespace}-shared--"
        for entry in self.base.iter_units(**kwargs):
            if entry["unit_id"].startswith(prefix):
                yield {**entry, "unit_id": entry["unit_id"][len(prefix):]}


class Runner(LegacyRunner):
    def run_chart(self, *args, **kwargs):
        result = super().run_chart(*args, **kwargs)
        return self.record(result["unit_id"])


def record(base, unit):
    path = base.unit_path(unit)
    metadata = read_json(path / "manifest.json")["metadata"]
    result = {**read_json(path / "summary.json"),
              "metrics": read_json(path / "metrics.json"),
              "retired_audit": read_json(path / "retirement.json")["audit"],
              "context": metadata["blur_context"], "namespace": metadata["namespace"],
              "physical_unit": unit}
    return result
