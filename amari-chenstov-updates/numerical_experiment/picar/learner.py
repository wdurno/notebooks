"""Model/optimizer integration of the independently tested numerical contract."""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.continual import picar_math as pm
from .config import LearningConfig


@dataclass
class LearningBatch:
    histories: list[dict[str, Any]]
    baselines: Tensor
    labels: Tensor
    actions: Tensor
    arrivals: Tensor

    def __post_init__(self) -> None:
        count = len(self.histories)
        if count < 1 or self.baselines.shape != (count,) or self.labels.shape != (count,):
            raise ValueError("learning records and target shapes disagree")
        if self.actions.shape != (count, 7) or self.arrivals.shape != (count,):
            raise ValueError("invalid action/arrival shapes")
        if any(history["images"].shape[0] != 1 for history in self.histories):
            raise ValueError("each raw history must have batch dimension one")


def predict_histories(model, params: Mapping[str, Tensor], histories: Sequence[dict]) -> dict[str, Tensor]:
    """Batch equal-length histories, avoiding synthetic recurrent padding."""
    groups = defaultdict(list)
    for index, history in enumerate(histories):
        groups[history["images"].shape[1]].append((index, history))
    outputs = {}
    for group in groups.values():
        rows = [history for _, history in group]
        batch = {key: torch.cat([row[key] for row in rows], dim=0)
                 for key in ("images", "executed_actions", "elapsed")}
        batch["contexts"] = [row["contexts"][0] for row in rows]
        if any("start_time" in row for row in rows):
            batch["start_time"] = torch.cat([
                row.get("start_time", row["elapsed"].new_zeros(1)) for row in rows
            ])
        prediction = pm.functional_predict(model, params, args=(batch,))
        for offset, (index, _) in enumerate(group):
            outputs[index] = {key: value[offset:offset + 1] for key, value in prediction.items()}
    return {key: torch.cat([outputs[i][key] for i in range(len(histories))])
            for key in ("return_logits", "action_mean")}


class MemoryLearner:
    """Atomic numerical events; the outer store commits their evidence ledger.

    Model tensors, optimizer, and all learner RNGs are rollback-able. The outer
    caller must serialize model access and commit retirement IDs with this state.
    """

    def __init__(self, model, config: LearningConfig, seeds: Mapping[str, int]):
        self.model = model
        self.config = config
        self.model.eval()
        self.params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.banks = model.parameter_banks()
        declared = self.banks["ltm"] + self.banks["stm"]
        if len(declared) != len(set(declared)) or set(declared) != set(self.params):
            raise ValueError("parameter banks do not partition all trainable coordinates")
        self.device = next(iter(self.params.values())).device
        self.dtype = next(iter(self.params.values())).dtype
        self.optimizer = torch.optim.Adam(self.params.values(), lr=config.learning_rate, weight_decay=0)
        self.diagonal = {name: torch.zeros_like(self.params[name]) for name in self.banks["ltm"]}
        self.anchor = {name: self.params[name].detach().clone() for name in self.banks["ltm"]}
        self.generators = {name: torch.Generator(device=self.device).manual_seed(seeds[name])
                           for name in ("fisher_labels", "fd_probes", "exploration")}
        self.version = 0
        self.retired_count = 0
        self.update_count = 0
        self.edges = torch.linspace(config.return_min, config.return_max, model.return_bins + 1,
                                    dtype=self.dtype, device=self.device)

    def design(self, batch: LearningBatch) -> pm.LikelihoodDesign:
        return pm.LikelihoodDesign(
            pm.return_bin_values(self.edges), batch.baselines.to(self.device, self.dtype),
            self.config.temperature, torch.eye(7, device=self.device, dtype=self.dtype) * self.config.action_variance,
        )

    def predict(self, params, batch: LearningBatch) -> dict:
        return predict_histories(self.model, params, batch.histories)

    def nll(self, params, batch: LearningBatch) -> Tensor:
        return pm.joint_nll(self.predict(params, batch), batch.labels.to(self.device),
                            batch.actions.to(self.device, self.dtype), self.design(batch))

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "parameters": {name: value.detach().cpu().clone() for name, value in self.params.items()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "diagonal": {name: value.detach().cpu().clone() for name, value in self.diagonal.items()},
            "anchor": {name: value.detach().cpu().clone() for name, value in self.anchor.items()},
            "rng": {name: generator.get_state().cpu().clone() for name, generator in self.generators.items()},
            "version": self.version, "retired_count": self.retired_count, "update_count": self.update_count,
            "banks": self.banks,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["schema_version"] != 1 or state["banks"] != self.banks:
            raise ValueError("learner checkpoint schema/parameter banks differ")
        if set(state["parameters"]) != set(self.params):
            raise ValueError("learner checkpoint parameters differ")
        with torch.no_grad():
            for name, value in self.params.items():
                restored = state["parameters"][name]
                if restored.shape != value.shape or not bool(torch.isfinite(restored).all()):
                    raise ValueError("invalid checkpoint parameter")
                value.copy_(restored.to(value))
        # PyTorch may reuse CPU optimizer tensors; a paired bootstrap must not
        # become a mutable alias of the first condition's Adam state.
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.diagonal = {name: value.to(self.params[name]).clone() for name, value in state["diagonal"].items()}
        self.anchor = {name: value.to(self.params[name]).clone() for name, value in state["anchor"].items()}
        for name, generator in self.generators.items():
            generator.set_state(state["rng"][name].cpu())
        self.version, self.retired_count, self.update_count = (
            state["version"], state["retired_count"], state["update_count"])
        self.model.encoder.publication = self.version
        self.model.encoder.clear_caches()

    def fit(self, *, new: LearningBatch | None, replay: LearningBatch | None,
            acquired_count: int, steps: int | None = None) -> dict[str, float]:
        if new is None and replay is None:
            raise ValueError("empty optimizer event")
        rollback = self.state_dict()
        start = time.monotonic()
        cfg = self.config
        stm_anchor = {name: self.params[name].detach().clone() for name in self.banks["stm"]}
        probes = pm.rademacher_probes(self.params, self.banks["stm"], cfg.fd_probes,
                    generator=self.generators["fd_probes"]) if replay is not None and cfg.fd_strength > 0 else None
        try:
            count = cfg.optimizer_steps if steps is None else steps
            if type(count) is not int or count < 1:
                raise ValueError("optimizer steps must be positive")
            for _ in range(count):
                self.optimizer.zero_grad(set_to_none=True)
                fd = None
                if probes is not None:
                    fd = pm.stm_fd_penalty(
                        lambda p: self.predict(p, replay), self.params, self.design(replay),
                        self.banks["stm"], probes,
                        # Records carry zero-based indices; the age formula uses
                        # acquired-observation counts, so the newest item is age 0.
                        pm.replay_ages(acquired_count, replay.arrivals.to(self.device) + 1, cfg.replay_capacity),
                        cfg.fd_step,
                    )
                losses = pm.regularized_objective(
                    self.params, ltm_names=self.banks["ltm"], stm_names=self.banks["stm"],
                    ltm_anchor=self.anchor, fisher_diagonal=self.diagonal, stm_anchor=stm_anchor,
                    new_nll=None if new is None else self.nll(self.params, new),
                    replay_nll=None if replay is None else self.nll(self.params, replay),
                    pi=cfg.pi, ridge_coefficient=cfg.ridge, numerical_floor=cfg.ridge_floor,
                    fd_penalty=fd, fd_strength=cfg.fd_strength,
                )
                losses["total"].backward()
                for parameter in self.params.values():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("nonfinite optimizer gradient")
                self.optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in self.params.values()):
                    raise FloatingPointError("nonfinite accepted parameter")
            self.update_count += count
            self.version += 1
            self.model.mark_published()
            metrics = {name: float(value.detach()) for name, value in losses.items()}
            metrics["update_seconds"] = time.monotonic() - start
            return metrics
        except BaseException:
            self.load_state_dict(rollback)
            raise

    def retire(self, batch: LearningBatch) -> dict[str, float]:
        rollback = self.state_dict()
        start = time.monotonic()
        try:
            ltm = {name: self.params[name] for name in self.banks["ltm"]}
            fisher = pm.model_fisher_diagonal(
                lambda p: self.predict(p, batch), ltm, self.design(batch), method="sampled",
                draws=self.config.fisher_draws, generator=self.generators["fisher_labels"],
            )
            self.diagonal, self.anchor, eta = pm.eviction_ema(
                self.diagonal, self.anchor, ltm, fisher, retired_count=len(batch.histories), pi=self.config.pi,
            )
            self.retired_count += len(batch.histories)
            return {"fisher_seconds": time.monotonic() - start, "retirement_eta": eta}
        except BaseException:
            self.load_state_dict(rollback)
            raise

    @torch.no_grad()
    def choose_action(self, history: dict, *, explore: bool = True) -> dict:
        start = time.monotonic()
        output, _, metadata = self.model.predict(history)
        mean = output["action_mean"][0]
        noise = torch.randn(mean.shape, device=self.device, dtype=mean.dtype,
                            generator=self.generators["exploration"]) if explore else torch.zeros_like(mean)
        z = mean + self.config.exploration_std * noise
        probabilities = torch.cat((z, z.new_zeros(1))).softmax(-1)
        baseline = (output["return_logits"].softmax(-1) * pm.return_bin_values(self.edges)).sum(-1)[0]
        return {"logits": z.cpu(), "probabilities": probabilities.cpu(), "baseline": float(baseline),
                "model_version": self.version, "inference_seconds": time.monotonic() - start,
                "metadata": metadata}
