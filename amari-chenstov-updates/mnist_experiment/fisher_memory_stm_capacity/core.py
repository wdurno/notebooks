"""Explicit process-local bindings to frozen grid curvature and learning.

Use ``with model.use_chart(name), bindings():`` around an inherited Runner or
learner call. Direct wrappers below manage their own bindings. This changes no
likelihood, Gaussian normalization, retention scale, or frozen source file.
Bindings mutate module globals and are not intended for concurrent threads.
"""

from contextlib import contextmanager
from functools import wraps

from mnist_experiment.fisher_memory_grid_registration import (
    curvature as grid_curvature, learner as grid_learner, model as grid_model,
    numerical as grid_numerical,
)

from . import model


CORE_SCHEMA_VERSION = 1
SIGMA2 = grid_curvature.SIGMA2
gaussian_nll = grid_curvature.gaussian_nll


@contextmanager
def bindings():
    """Temporarily bind all imported chart aliases; restore on nesting/failure."""
    replacements = [(grid_model, "build_model", model.build_model),
                    (grid_model, "memory_indices", model.memory_indices),
                    (grid_model, "flat_prediction", model.flat_prediction)]
    for module in (grid_curvature, grid_learner, grid_numerical):
        replacements.extend(((module, "memory_indices", model.memory_indices),
                             (module, "flat_prediction", model.flat_prediction)))
    previous = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    try:
        for module, name, value in replacements:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(previous):
            setattr(module, name, value)


def _bound(function):
    @wraps(function)
    def call(*args, **kwargs):
        with bindings():
            return function(*args, **kwargs)
    return call


fd_values = _bound(grid_curvature.fd_values)
jvp_values = _bound(grid_curvature.jvp_values)
reference_fisher = _bound(grid_curvature.reference_fisher)
fit_step = _bound(grid_learner.fit_step)
calibrate_fd = _bound(grid_numerical.calibrate_fd)


@wraps(grid_curvature.output_jacobians)
def output_jacobians(*args, **kwargs):
    """Bind each lazy evaluation without leaving globals patched across yield."""
    iterator = grid_curvature.output_jacobians(*args, **kwargs)
    try:
        while True:
            with bindings():
                rows = next(iterator, None)
            if rows is None:
                return
            yield rows
    finally:
        with bindings():
            iterator.close()


@wraps(grid_learner.prepare_objective)
def prepare_objective(*args, **kwargs):
    """Return a fixed objective that rebinds on every later closure evaluation."""
    with bindings():
        objective, metadata = grid_learner.prepare_objective(*args, **kwargs)
    return _bound(objective), metadata
