"""Transactional small-head fitting with fixed references and implicit EWC."""

import copy
from dataclasses import dataclass, asdict
import time

import torch

from src.seeding import derive_component_seed
from numerical_experiment.picar_solver_repair.numerics import adaptive_metric
from .model import Coordinates
from .likelihood import data_loss, joint_kl, fisher_factor, reference_design


@dataclass(frozen=True)
class Recipe:
    capacity: int = 128
    batch_size: int = 32
    rounds: int = 2
    steps: int = 3
    learning_rate: float = .001
    temperature: float = 3.
    action_variance: float = 16.
    critic_variance: float = 4.
    exploration_std: float = 4.
    target_rate: float = .05
    rank: int = 3
    ridge: float = .1
    pi: float = .025
    fd_strength: float = 0.
    fd_step: float = .01
    integration_draws: int = 4
    log_weight_bound: float = 5.


def batch(records, device):
    return {k: torch.stack([torch.as_tensor(r[k]) for r in records]).to(device=device, dtype=torch.float32)
            for k in ("feature", "action", "next_feature", "reward", "discount", "return")}


class Learner:
    def __init__(self, heads, recipe, seed, method):
        from .archive import Archive
        self.heads, self.coordinates, self.recipe = heads, Coordinates(heads), recipe
        self.device = next(heads.parameters()).device
        self.ltm = self.coordinates.indices["ltm"].to(self.device)
        self.stm = self.coordinates.indices["stm"].to(self.device)
        self.vector = self.coordinates.vector()
        self.target = self.vector.clone()
        self.moment = torch.zeros_like(self.vector)
        self.archive = Archive(torch.zeros(len(self.ltm), device=self.device), rank=recipe.rank)
        self.anchor = self.vector[self.ltm].clone()
        self.generators = {name: torch.Generator(device=self.device).manual_seed(
            derive_component_seed(seed, "head_awac:" + name))
            for name in ("exploration", "targets", "fisher", "fd")}
        self.method = method
        self.evaluations = self.version = self.retired = 0

    def state_dict(self):
        return {"schema": 1, "recipe": asdict(self.recipe), "method": self.method,
                "vector": self.vector.detach().cpu().clone(), "target": self.target.detach().cpu().clone(),
                "moment": self.moment.cpu().clone(), "anchor": self.anchor.cpu().clone(),
                "archive": self.archive.state_dict(), "evaluations": self.evaluations,
                "version": self.version, "retired": self.retired,
                "rng": {k: g.get_state().cpu() for k, g in self.generators.items()}}

    def load_state_dict(self, state):
        from .archive import Archive
        if state["recipe"] != asdict(self.recipe) or state["method"] != self.method:
            raise ValueError("learner resume identity differs")
        for key in ("vector", "target", "moment", "anchor"):
            setattr(self, key, state[key].to(self.device).clone())
        if not torch.isfinite(self.vector).all():
            raise ValueError("nonfinite checkpoint")
        self.archive = Archive.from_state_dict(state["archive"], device=self.device)
        self.evaluations, self.version, self.retired = state["evaluations"], state["version"], state["retired"]
        for key, state_rng in state["rng"].items():
            self.generators[key].set_state(state_rng)
        self.coordinates.publish(self.vector)

    def design(self, records, *, generator="targets"):
        data = batch(records, self.device)
        draws = torch.randn((len(records), self.recipe.integration_draws, 7),
                            generator=self.generators[generator], device=self.device)
        design, diagnostic = reference_design(self.coordinates, self.target,
            data["feature"], data["action"], data["next_feature"], data["reward"],
            data["discount"], data["return"], draws, method=self.method,
            temperature=self.recipe.temperature, action_variance=self.recipe.action_variance,
            critic_variance=self.recipe.critic_variance, log_weight_bound=self.recipe.log_weight_bound)
        return data, design, diagnostic

    def fit(self, records, acquired, check=lambda: None):
        rollback = copy.deepcopy(self.state_dict())
        tick = time.monotonic()
        try:
            cfg = self.recipe
            data, design, diagnostic = self.design(records)
            center = self.vector.clone()
            vector = center.clone().requires_grad_()
            moments = self.moment.clone()
            ages = torch.tensor([min(1., max(0., (acquired-r["arrival"]-1)/cfg.capacity))
                                 for r in records], device=self.device)
            draws = torch.randn((len(records), cfg.integration_draws, 7),
                                generator=self.generators["fd"], device=self.device)
            probe = torch.zeros_like(vector)
            probe[self.stm] = (2*torch.randint(2, (len(self.stm),), generator=self.generators["fd"],
                                             device=self.device)-1).to(vector.dtype)
            omega = (1-cfg.pi)/cfg.pi
            ridge = omega*cfg.ridge*self.archive.diagonal.mean().clamp_min(1e-12)
            rows = []
            for j in range(cfg.steps):
                output = self.coordinates.predict(vector, data["feature"], data["action"])
                parts = data_loss(output, data["action"], design)
                nll = parts["nll"].mean()
                live = nll + parts["disagreement"].mean()
                fd = vector.new_zeros(())
                if cfg.fd_strength > 0:
                    shift = cfg.fd_step * .5 * probe
                    kl = .5*(joint_kl(self.coordinates, vector+shift, vector-shift, data["feature"], design, draws)
                             + joint_kl(self.coordinates, vector-shift, vector+shift, data["feature"], design, draws))
                    fd = (ages.square()*kl).mean()/(cfg.fd_step**2*len(self.stm))
                    live = live + cfg.fd_strength*fd
                gradient = torch.autograd.grad(live, vector)[0].detach()
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite head gradient")
                metric, moments = adaptive_metric(gradient, moments, self.evaluations+j+1, cfg.learning_rate)
                offset = vector.detach()-center
                force = gradient + ridge*offset
                force[self.ltm] += omega*self.archive.matvec(vector.detach()[self.ltm]-self.anchor)
                displacement = -force/(metric+ridge)
                displacement[self.ltm] = -self.archive.solve(force[self.ltm], metric[self.ltm]+ridge, weight=omega)
                free_force = gradient[self.ltm]+ridge*offset[self.ltm]
                free = -free_force/(metric[self.ltm]+ridge)
                ewc = .5*omega*self.archive.quadratic(vector.detach()[self.ltm]-self.anchor)
                rows.append({"data_nll": float(nll.detach()), "fd": float((cfg.fd_strength*fd).detach()),
                    "ewc": float(ewc), "gradient_norm": float(gradient.norm()),
                    "ewc_proposal_difference": float((displacement[self.ltm]-free).norm()),
                    "step_norm": float(displacement.norm())})
                vector = (vector.detach()+displacement).requires_grad_()
                check()
            if not torch.isfinite(vector).all():
                raise FloatingPointError("nonfinite head proposal")
            self.vector, self.moment = vector.detach(), moments.detach()
            self.anchor = self.vector[self.ltm].clone()
            self.target.lerp_(self.vector, cfg.target_rate)
            self.coordinates.publish(self.vector)
            self.evaluations += cfg.steps
            self.version += 1
            return {**rows[0], **diagnostic, "inner": rows, "n": acquired,
                "update_seconds": time.monotonic()-tick, "replay_count": len(records),
                "age_mean": float(ages.mean()), "age_saturated": float((ages >= 1).float().mean()),
                "optimizer_exposures": len(records)*cfg.steps,
                "post_step_ewc": float(.5*omega*self.archive.quadratic(self.vector[self.ltm]-center[self.ltm]))}
        except BaseException:
            self.load_state_dict(rollback)
            raise

    def retire(self, record):
        cfg = self.recipe
        data, design, _ = self.design([record], generator="fisher")
        draws = torch.randn((cfg.integration_draws, 7), generator=self.generators["fisher"], device=self.device)
        factors = fisher_factor(self.coordinates, self.vector, data["feature"][0], design.weights[0],
                                cfg.action_variance, cfg.critic_variance, draws)[self.ltm]
        diagnostic = self.archive.update(factors, alpha=1-cfg.pi, beta=cfg.pi)
        self.retired += 1
        return diagnostic

    @torch.no_grad()
    def choose(self, feature, explore=True):
        h = feature.to(self.device).reshape(1, -1)
        output = self.coordinates.predict(self.vector, h, h.new_zeros((1, 7)))
        action = output["mean"][0]
        if explore:
            action = action+self.recipe.exploration_std*torch.randn((7,),
                generator=self.generators["exploration"], device=self.device)
        return action.detach().cpu()
