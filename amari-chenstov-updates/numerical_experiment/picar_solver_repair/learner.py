"""Versioned, transactional curvature-aware learner; historical code is reused.

Candidate tensors are functional leaves. Actor parameters, centers, Fisher and
the adaptive metric are published only after the entire event succeeds. A
checkpoint during a handoff stores the accepted boundary and a frozen event
for deterministic replay, not partially published inner iterates. The caller
still owns eligibility/retirement IDs and commits that ledger with this state.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import asdict
from typing import Mapping

import torch
from torch import Tensor

from numerical_experiment.picar.config import LearningConfig
from numerical_experiment.picar.learner import LearningBatch, MemoryLearner
from src.continual import picar_math as pm
from .numerics import BETA2, EPSILON, RECIPE_STEPS, accumulation_dtype, adaptive_metric, diagonal_step


def _clone(value, *, cpu=False):
    if isinstance(value, Tensor):
        return (value.detach().cpu() if cpu else value.detach()).clone()
    if isinstance(value, LearningBatch):
        return LearningBatch(**{name: _clone(item, cpu=cpu) for name, item in vars(value).items()})
    if isinstance(value, dict):
        return {name: _clone(item, cpu=cpu) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item, cpu=cpu) for item in value)
    return copy.deepcopy(value)


def _identity(value):
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, LearningBatch):
            add(vars(item))
        elif isinstance(item, dict):
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
            digest.update(b"\0")
    add(value)
    return digest.hexdigest()


def _norm(values):
    values = tuple(values)
    return float(torch.sqrt(sum(value.detach().double().square().sum() for value in values))) if values else 0.


def _scalars(values):
    return dict(zip(values, torch.stack([value.detach() for value in values.values()]).cpu().tolist()))


def _valid(values, *, nonnegative=False):
    return bool(torch.stack([(torch.isfinite(v) & (v >= 0) if nonnegative else torch.isfinite(v)).all()
                             for v in values]).all())


class _MetricOptimizer:
    """Small compatibility facade, not Adam. The learner owns event commits."""

    def __init__(self, params, learning_rate):
        self.param_groups = [{"lr": learning_rate}]
        self.second_moments = {name: torch.zeros_like(value, dtype=accumulation_dtype(value))
                               for name, value in params.items()}
        self.gradient_evaluations = 0

    def state_dict(self):
        return {"schema_version": 1, "param_groups": copy.deepcopy(self.param_groups),
                "second_moments": _clone(self.second_moments, cpu=True),
                "gradient_evaluations": self.gradient_evaluations,
                "beta2": BETA2, "epsilon": EPSILON}

    def load_state_dict(self, state):
        if (state["schema_version"] != 1 or state["beta2"] != BETA2 or state["epsilon"] != EPSILON
                or state["param_groups"] != self.param_groups):
            raise ValueError("optimizer checkpoint settings differ")
        count = state["gradient_evaluations"]
        if type(count) is not int or count < 0 or set(state["second_moments"]) != set(self.second_moments):
            raise ValueError("invalid optimizer checkpoint")
        moments = {}
        for name, reference in self.second_moments.items():
            value = state["second_moments"][name].to(reference).clone()
            if value.shape != reference.shape or not bool((torch.isfinite(value) & (value >= 0)).all()):
                raise ValueError("invalid optimizer second moment")
            moments[name] = value
        self.second_moments, self.gradient_evaluations = moments, count


class SolverLearner(MemoryLearner):
    """One accepted statistical event, with one or three live-gradient solves.

    ``steps`` may only equal the recipe's declared count. Additional replay
    passes must be declared separately by the caller, never silently inferred
    from the historical LearningConfig.optimizer_steps default.
    """

    def __init__(self, model, config: LearningConfig, seeds: Mapping[str, int], *,
                 recipe="proximal_1"):
        config.validate()
        if recipe not in RECIPE_STEPS:
            raise ValueError(f"unknown solver recipe: {recipe}")
        super().__init__(model, config, seeds)
        self.recipe = recipe
        self.optimizer = _MetricOptimizer(self.params, config.learning_rate)
        self.center = _clone(self.params)
        self.previous_center = _clone(self.params)
        self.center_version = 0
        self.event_count = 0
        self.timings = {"fit_seconds": 0., "fisher_seconds": 0., "failed_fit_seconds": 0.}
        self._event = None
        self._pending_event = None
        self._active = False

    @staticmethod
    def as_legacy(state):
        """Construction-only MemoryLearner view, with fresh, empty Adam state."""
        result = {key: _clone(state[key], cpu=True) for key in (
            "parameters", "diagonal", "anchor", "rng", "version", "retired_count", "update_count", "banks")}
        rate = state.get("solver_config", {}).get("learning", {}).get("learning_rate", .003)
        dummy = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([dummy], lr=rate).state_dict()
        optimizer["param_groups"][0]["params"] = list(range(len(state["parameters"])))
        result.update(schema_version=1, optimizer=optimizer)
        return result

    def legacy_state_dict(self):
        return self.as_legacy(self.state_dict())

    def _settings(self):
        return {"learning": asdict(self.config), "recipe": self.recipe,
                "beta2": BETA2, "epsilon": EPSILON,
                "learning_rate": self.optimizer.param_groups[0]["lr"]}

    def state_dict(self):
        # Metric updates are event-local, so this is always an accepted boundary.
        state = {"schema_version": 2, "solver_config": self._settings(),
                 "parameters": _clone(self.params, cpu=True),
                 "optimizer": self.optimizer.state_dict(),
                 "diagonal": _clone(self.diagonal, cpu=True), "anchor": _clone(self.anchor, cpu=True),
                 "center": _clone(self.center, cpu=True),
                 "previous_center": _clone(self.previous_center, cpu=True),
                 "rng": {name: gen.get_state().cpu().clone() for name, gen in self.generators.items()},
                 "version": self.version, "retired_count": self.retired_count,
                 "update_count": self.update_count, "banks": copy.deepcopy(self.banks),
                 "center_version": self.center_version, "event_count": self.event_count,
                 "timings": dict(self.timings),
                 "pending_event": _clone(self._event or self._pending_event, cpu=True)}
        return state

    def load_state_dict(self, state):
        if self._active:
            raise RuntimeError("cannot restore during an active solver event")
        schema = state.get("schema_version")
        if schema not in (1, 2) or state["banks"] != self.banks:
            raise ValueError("learner checkpoint schema/parameter banks differ")
        if schema == 2 and state["solver_config"] != self._settings():
            raise ValueError("solver resume configuration differs")

        def coordinates(key, names, *, nonnegative=False, accumulation=False):
            if set(state[key]) != set(names):
                raise ValueError(f"checkpoint {key} coordinates differ")
            result = {}
            for name in names:
                reference = self.params[name]
                dtype = accumulation_dtype(reference) if accumulation else reference.dtype
                value = state[key][name].to(device=self.device, dtype=dtype).clone()
                if (value.shape != reference.shape or not bool(torch.isfinite(value).all())
                        or (nonnegative and bool((value < 0).any()))):
                    raise ValueError(f"invalid checkpoint {key}")
                result[name] = value
            return result

        parameters = coordinates("parameters", self.params)
        diagonal = coordinates("diagonal", self.banks["ltm"], nonnegative=True, accumulation=True)
        if schema == 2:
            center = coordinates("center", self.params)
            previous = coordinates("previous_center", self.params)
            anchor = coordinates("anchor", self.banks["ltm"])
            if (any(not torch.equal(center[n], parameters[n]) for n in self.params)
                    or any(not torch.equal(anchor[n], center[n]) for n in anchor)):
                raise ValueError("checkpoint accepted center differs from parameters/anchor")
            optimizer = _MetricOptimizer(self.params, self.config.learning_rate)
            optimizer.param_groups = copy.deepcopy(self.optimizer.param_groups)
            optimizer.load_state_dict(state["optimizer"])
            counters = {name: state[name] for name in (
                "version", "retired_count", "update_count", "center_version", "event_count")}
            if counters["update_count"] != optimizer.gradient_evaluations:
                raise ValueError("checkpoint gradient count differs")
            timings = copy.deepcopy(state["timings"])
        else:
            # Legacy Adam moments and retirement-tied anchors have no meaning here.
            center, previous = _clone(parameters), _clone(parameters)
            optimizer = _MetricOptimizer(self.params, self.config.learning_rate)
            counters = dict(version=state["version"], retired_count=state["retired_count"],
                            update_count=0, center_version=0, event_count=0)
            timings = {"fit_seconds": 0., "fisher_seconds": 0., "failed_fit_seconds": 0.}
        if any(type(v) is not int or v < 0 for v in counters.values()):
            raise ValueError("invalid learner checkpoint counters")
        if set(timings) != set(self.timings) or any(not math.isfinite(v) or v < 0 for v in timings.values()):
            raise ValueError("invalid learner checkpoint timing")
        if set(state["rng"]) != set(self.generators):
            raise ValueError("checkpoint RNG names differ")
        rngs = {}
        for name in self.generators:
            generator = torch.Generator(device=self.device)
            generator.set_state(state["rng"][name].cpu())
            rngs[name] = generator
        pending = _clone(state.get("pending_event")) if schema == 2 else None
        if pending is not None and (pending["event_id"] != counters["event_count"]
                                    or pending["steps"] != RECIPE_STEPS[self.recipe]):
            raise ValueError("invalid checkpoint event cursor")
        with torch.no_grad():
            for name, value in self.params.items():
                value.copy_(parameters[name])
        self.diagonal, self.center, self.previous_center = diagonal, center, previous
        self.anchor = {name: self.center[name].clone() for name in self.banks["ltm"]}
        self.optimizer, self.generators, self.timings = optimizer, rngs, timings
        for name, value in counters.items():
            setattr(self, name, value)
        self._pending_event = pending
        self.model.encoder.publication = self.version
        self.model.encoder.clear_caches()

    def _live(self, params, new, replay, probes, ages):
        means = [self.nll(params, batch).to(accumulation_dtype(next(iter(params.values())))).mean()
                 for batch in (new, replay) if batch is not None]
        data = torch.stack(means).mean()
        fd = data.new_zeros(())
        if probes is not None:
            fd = self.config.fd_strength * pm.stm_fd_penalty(
                lambda p: self.predict(p, replay), params, self.design(replay),
                self.banks["stm"], probes, ages, self.config.fd_step)
        live = data + fd
        if not bool(torch.isfinite(live)):
            raise FloatingPointError("nonfinite live objective")
        return live, data, fd

    def _components(self, params, data, fd, center, previous, ewc, ridge, scale, kappa):
        zero = data.new_zeros(())
        anchor = previous if self.recipe == "lagged_1" else center
        penalty = .5 * sum(((ewc[n] * (params[n].to(ewc[n]) - anchor[n].to(ewc[n])).square()).sum()
                            for n in self.params), zero)
        ridge_loss = .5 * ridge * sum(((params[n].to(ridge) - center[n].to(ridge)).square().sum()
                                      for n in self.params), zero)
        return dict(total=data + fd + penalty + ridge_loss, data=data, fd=fd, ewc=penalty,
                    ridge=ridge_loss, fisher_scale=scale, kappa=kappa,
                    pi=data.new_tensor(self.config.pi), omega=data.new_tensor((1 - self.config.pi) / self.config.pi))

    def fit(self, *, new, replay, acquired_count, steps=None, cooperate=None,
            diagnostics=False, event_id=None):
        count = RECIPE_STEPS[self.recipe]
        if steps is not None and (type(steps) is not int or steps != count):
            raise ValueError("steps must equal the solver recipe's declared count")
        if new is None and replay is None:
            raise ValueError("empty optimizer event")
        if type(acquired_count) is not int or acquired_count < 0:
            raise ValueError("acquired_count must be nonnegative")
        if cooperate is not None and not callable(cooperate):
            raise TypeError("cooperate must be callable")
        if type(diagnostics) is not bool:
            raise TypeError("diagnostics must be boolean")
        if self._active:
            raise RuntimeError("nested solver event")
        if event_id is not None and event_id != self.event_count:
            raise ValueError("event ID must be the next accepted event count")
        started = time.monotonic()
        rollback = self.state_dict()
        original_pending = self._pending_event
        rng_fd = self.generators["fd_probes"].get_state().clone()
        self._active, self._cooperate = True, cooperate
        cuda_start = cuda_end = None
        try:
            # Resume replays the complete frozen event from its accepted boundary.
            identity = _identity((new, replay, acquired_count, count, diagnostics))
            if original_pending is not None:
                if identity != original_pending["identity"]:
                    raise ValueError("resumed frozen event differs")
                event = _clone(original_pending)
                new, replay = event["new"], event["replay"]
                probes = event["probes"]
                if probes is not None:
                    probes = tuple({n: p.to(self.params[n]) for n, p in probe.items()} for probe in probes)
                self.generators["fd_probes"].set_state(event["rng_fd_after"].cpu())
            else:
                new, replay = _clone(new), _clone(replay)
                probes = pm.rademacher_probes(self.params, self.banks["stm"], self.config.fd_probes,
                    generator=self.generators["fd_probes"]) if replay is not None and self.config.fd_strength > 0 else None
                event = {"event_id": self.event_count, "steps": count, "replay_from_inner": 0,
                         "completed_inner_steps": 0,
                         "identity": identity, "new": new, "replay": replay,
                         "probes": probes, "acquired_count": acquired_count, "diagnostics": diagnostics,
                         "rng_fd_after": self.generators["fd_probes"].get_state().clone()}
            self._event = event
            center, previous = _clone(self.center), _clone(self.previous_center)
            diagonal = _clone(self.diagonal)
            if not _valid(diagonal.values(), nonnegative=True):
                raise FloatingPointError("invalid frozen Fisher diagonal")
            dtype = accumulation_dtype(next(iter(self.params.values())))
            zero = torch.zeros((), device=self.device, dtype=dtype)
            scale = sum((v.to(dtype).sum() for v in diagonal.values()), zero) / sum(
                self.params[n].numel() for n in self.banks["ltm"])
            kappa = self.config.ridge * scale.clamp_min(self.config.ridge_floor)
            omega = (1 - self.config.pi) / self.config.pi
            ridge = omega * kappa
            ewc = {n: omega * diagonal[n].to(dtype) if n in diagonal else torch.zeros_like(p, dtype=dtype)
                   for n, p in self.params.items()}
            ages = None
            if probes is not None:
                ages = pm.replay_ages(acquired_count, replay.arrivals.to(self.device) + 1,
                                      self.config.replay_capacity)
                if replay.archived is not None:
                    ages = torch.where(replay.archived.to(self.device), torch.ones_like(ages), ages)
            candidate = {n: p.detach().clone().requires_grad_() for n, p in self.params.items()}
            moments = _clone(self.optimizer.second_moments)
            evaluations = self.optimizer.gradient_evaluations
            rows = []
            prior_step = {n: center[n].double() - previous[n].double() for n in center}
            if self.device.type == "cuda":
                cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            for j in range(count):
                inner_start = time.monotonic()
                live, data, fd = self._live(candidate, new, replay, probes, ages)
                gradients = torch.autograd.grad(live, tuple(candidate.values()), allow_unused=True)
                gradients = {n: torch.zeros_like(p) if g is None else g.detach()
                             for (n, p), g in zip(candidate.items(), gradients)}
                if not _valid(gradients.values()):
                    raise FloatingPointError("nonfinite live gradient")
                evaluations += 1
                metrics = self._components(candidate, data, fd, center, previous, ewc, ridge, scale, kappa)
                row = _scalars(metrics)
                proposal, no_ewc, forces, deltas, ratios = {}, {}, {}, {}, []
                for name, x in candidate.items():
                    metric, moments[name] = adaptive_metric(gradients[name], moments[name], evaluations,
                                                          self.optimizer.param_groups[0]["lr"], validate=False)
                    delta = diagonal_step(x.detach(), gradients[name], metric, center[name], ewc[name], ridge,
                                          recipe=self.recipe, previous_center=previous[name], validate=False)
                    free = diagonal_step(x.detach(), gradients[name], metric, center[name],
                                         torch.zeros_like(ewc[name]), ridge, recipe="proximal_1", validate=False)
                    deltas[name] = delta
                    proposal[name] = (x.detach().to(delta) + delta).to(x.dtype)
                    no_ewc[name] = (x.detach().to(free) + free).to(x.dtype)
                    force = ewc[name] * (x.detach().to(metric) - (
                        previous[name] if self.recipe == "lagged_1" else center[name]).to(metric))
                    forces[name] = force
                    ratios.append((ewc[name] / (metric + ridge)).reshape(-1))
                realized = {n: proposal[n].double() - candidate[n].detach().double() for n in candidate}
                difference = {n: proposal[n].double() - no_ewc[n].double() for n in proposal}
                ratio = torch.cat(ratios)
                previous_norm, step_norm = _norm(prior_step.values()), _norm(realized.values())
                cosine = (float(sum((realized[n] * prior_step[n]).sum() for n in realized))
                          / (previous_norm * step_norm)) if previous_norm * step_norm > 0 else 0.
                stiffness_max, stiffness_mean, stiffness_ge_one = torch.stack((
                    ratio.max(), ratio.mean(), (ratio >= 1).to(ratio.dtype).mean())).cpu().tolist()
                row.update(inner_step=j + 1, gradient_evaluation=evaluations,
                           live_gradient_norm=_norm(gradients.values()), ewc_force_norm=_norm(forces.values()),
                           ewc_proposal_difference_norm=_norm(difference.values()),
                           ewc_suppression_fraction=_norm(difference.values()) / max(_norm(
                               (no_ewc[n].double() - candidate[n].detach().double() for n in no_ewc)), 1e-300),
                           realized_step_norm=step_norm, step_cosine=cosine,
                           proposed_step_norm=_norm(deltas.values()),
                           rounding_error_norm=_norm(realized[n] - deltas[n].double() for n in realized),
                           stiffness_max=stiffness_max, stiffness_mean=stiffness_mean,
                           stiffness_ge_one_fraction=stiffness_ge_one,
                           unchanged_coordinates=int(sum((v == 0).sum() for v in realized.values())),
                           pre_step=1, inner_seconds=time.monotonic() - inner_start)
                if not _valid(proposal.values()) or not _valid(moments.values(), nonnegative=True):
                    raise FloatingPointError("nonfinite candidate parameter")
                rows.append(row)
                candidate = {n: value.detach().requires_grad_() for n, value in proposal.items()}
                prior_step = realized
                event["completed_inner_steps"] = j + 1
                if self._cooperate is not None:
                    self._cooperate()
            # Optional one-pass, post-step objective and full-gradient residual.
            algebraic = self._components(candidate, zero, zero, center, previous, ewc, ridge, scale, kappa)
            centered_ewc = .5 * sum((ewc[n] * (candidate[n].to(ewc[n]) - center[n].to(ewc[n])).square()).sum()
                                    for n in candidate)
            post = _scalars(dict(post_ewc=algebraic["ewc"], post_ridge=algebraic["ridge"],
                                 post_centered_ewc=centered_ewc))
            if diagnostics:
                _, data, fd = self._live(candidate, new, replay, probes, ages)
                components = self._components(candidate, data, fd, center, previous, ewc, ridge, scale, kappa)
                residual = torch.autograd.grad(components["total"], tuple(candidate.values()), allow_unused=True)
                post.update(_scalars({"post_" + n: v for n, v in components.items()}))
                residual_norm = _norm(g for g in residual if g is not None)
                centered_residual = []
                for (n, p), g in zip(candidate.items(), residual):
                    value = torch.zeros_like(p) if g is None else g
                    if self.recipe == "lagged_1":
                        value = value - ewc[n] * (center[n].to(ewc[n]) - previous[n].to(ewc[n]))
                    centered_residual.append(value)
                centered_norm = _norm(centered_residual)
                post.update(post_gradient_residual_norm=residual_norm,
                            post_scaled_gradient_residual=residual_norm / max(1., rows[-1]["live_gradient_norm"]),
                            post_centered_total=float((components["total"] - components["ewc"] + centered_ewc).detach()),
                            post_centered_gradient_residual_norm=centered_norm,
                            post_centered_scaled_gradient_residual=centered_norm / max(1., rows[-1]["live_gradient_norm"]))
            # No cooperative handoff can observe a partially copied publication.
            self._cooperate = None
            publication_start = time.monotonic()
            with torch.no_grad():
                for name, parameter in self.params.items():
                    parameter.copy_(candidate[name])
            self.previous_center, self.center = center, _clone(self.params)
            self.anchor = {n: self.center[n].clone() for n in self.banks["ltm"]}
            self.optimizer.second_moments, self.optimizer.gradient_evaluations = moments, evaluations
            self.update_count = evaluations
            self.version += 1
            self.center_version += 1
            self.event_count += 1
            self.model.mark_published()
            publication_seconds = time.monotonic() - publication_start
            if cuda_end is not None:
                cuda_end.record()
                # Timing completion is deferred: do not change the online scheduler
                # with a global synchronization. A caller may time isolated kernels.
                gpu_ms = cuda_start.elapsed_time(cuda_end) if cuda_end.query() else None
                peak = torch.cuda.max_memory_allocated(self.device)
            else:
                gpu_ms, peak = 0., 0
            elapsed = time.monotonic() - started
            self.timings["fit_seconds"] += elapsed
            self._event = self._pending_event = None
            result = {name: rows[-1][name] for name in metrics}
            result.update(post)
            result.update(inner_steps=rows, pre_step=1, event_id=self.event_count - 1,
                          center_version=self.center_version, inner_step_count=count,
                          gradient_evaluations=count, diagnostic_gradient_evaluations=int(diagnostics),
                          total_gradient_evaluations=count + int(diagnostics),
                          fd_evaluations=(count + int(diagnostics)) if probes is not None else 0,
                          optimizer_exposures=count * sum(len(b.histories) for b in (new, replay) if b is not None),
                          incoming_observations=len(new.histories) if new is not None else 0,
                          first_step_ewc_suppression=rows[0]["ewc_suppression_fraction"],
                          first_step_ewc_proposal_difference_norm=rows[0]["ewc_proposal_difference_norm"],
                          update_seconds=elapsed, publication_seconds=publication_seconds,
                          gpu_milliseconds=gpu_ms, gpu_timing_ready=int(gpu_ms is not None),
                          gpu_seconds=None if gpu_ms is None else gpu_ms / 1000.,
                          peak_memory_bytes=peak, peak_vram_bytes=peak,
                          peak_memory_is_process_high_water=1, learning_rate=self.optimizer.param_groups[0]["lr"],
                          beta2=BETA2, epsilon=EPSILON)
            return result
        except BaseException:
            actor_rng = self.generators["exploration"].get_state().clone()
            self._active, self._cooperate, self._event = False, None, None
            self.load_state_dict(rollback)
            if cooperate is not None:
                self.generators["exploration"].set_state(actor_rng)
            self.generators["fd_probes"].set_state(rng_fd)
            self.timings["failed_fit_seconds"] += time.monotonic() - started
            raise
        finally:
            self._active, self._cooperate, self._event = False, None, None

    def retire(self, batch):
        if self._active or self._pending_event is not None:
            raise RuntimeError("retirement cannot interrupt an unfinished event")
        rollback = self.state_dict()
        started = time.monotonic()
        try:
            ltm = {n: self.params[n] for n in self.banks["ltm"]}
            fisher = pm.model_fisher_diagonal(
                lambda p: self.predict(p, batch), ltm, self.design(batch), method="sampled",
                draws=self.config.fisher_draws, generator=self.generators["fisher_labels"])
            count = len(batch.histories)
            eta = -math.expm1(count * math.log1p(-self.config.pi))
            diagonal = {n: (1 - eta) * self.diagonal[n].to(fisher[n]) + eta * fisher[n] for n in ltm}
            if any(not bool((torch.isfinite(v) & (v >= 0)).all()) for v in diagonal.values()):
                raise FloatingPointError("invalid retired Fisher")
            self.diagonal = diagonal
            self.retired_count += count
            elapsed = time.monotonic() - started
            self.timings["fisher_seconds"] += elapsed
            return {"fisher_seconds": elapsed, "retirement_eta": eta,
                    "center_version": self.center_version, "retired_count": self.retired_count}
        except BaseException:
            self.load_state_dict(rollback)
            raise
