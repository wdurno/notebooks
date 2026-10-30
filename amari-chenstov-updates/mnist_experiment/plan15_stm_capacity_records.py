"""Phase 12 v2: consistently return sealed records across fresh runs and resume.

The frozen v1 engine enriches summary.json before sealing, but its fresh return
omits that metadata. Loading a completed unit returns the enriched record.
Normalize at this boundary; training, checkpoints and statistics are unchanged.
"""

from contextlib import contextmanager
from pathlib import Path
import sys

from mnist_experiment.fisher_memory_stm_capacity import run, acceptance
from mnist_experiment.fisher_memory_stm_capacity.config import ROOT
from mnist_experiment.fisher_memory_regression.runtime import file_digest

BaseRunner, BaseStudy = run.Runner, run.Study
base_sources, base_process_alive = run.sources, run.process_alive


class Runner(BaseRunner):
    def run_chart(self, *args, **kwargs):
        result = super().run_chart(*args, **kwargs)
        return self.record(result["unit_id"])


class Study(BaseStudy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        note = ("2026-10-29 bookkeeping correction: v2 returns the sealed record after both fresh "
                "and resumed trajectories, preserving identical chart metadata for analysis. "
                "The original code and pre-repair artifacts remain preserved; numerical learning is unchanged.")
        if note not in self.status["changes"]:
            self.status["changes"].append(note)


def sources():
    return dict(sorted({**base_sources(), str(Path(__file__).resolve().relative_to(ROOT)):
                        file_digest(Path(__file__).resolve())}.items()))


def process_alive(root, pid):
    if base_process_alive(root,pid):
        return True
    if type(pid) is not int or pid <= 0:
        return False
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if "mnist_experiment.plan15_stm_capacity_records" not in parts:
            return False
        requested = Path(parts[parts.index("--root")+1])
        if not requested.is_absolute():
            requested = Path(f"/proc/{pid}/cwd").resolve()/requested
        return requested.resolve() == Path(root).resolve()
    except (OSError,ValueError,IndexError,UnicodeError):
        return False


@contextmanager
def bindings():
    changes = [(run,"Runner",Runner),(run,"Study",Study),(run,"sources",sources),
               (run,"process_alive",process_alive),(acceptance,"sources",sources)]
    previous = [(module,name,getattr(module,name)) for module,name,_ in changes]
    try:
        for module,name,value in changes:
            setattr(module,name,value)
        yield
    finally:
        for module,name,value in reversed(previous):
            setattr(module,name,value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    with bindings():
        return acceptance.main(argv[1:]) if argv[:1] == ["accept"] else run.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
