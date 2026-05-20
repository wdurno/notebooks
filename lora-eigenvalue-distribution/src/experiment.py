from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import MNISTTSampler, SampleBatch, make_validation_sets
from .fisher import (
    OnlineDiagonalEWC,
    SpectralStats,
    empirical_fisher_eigenvalues,
    estimate_diagonal_fisher,
)
from .model import MNISTLoRAMLP


@dataclass
class ExperimentConfig:
    data_dir: str = "data"
    seed: int = 0
    device: str = "auto"
    n_initial: int = 10_000
    initial_epochs: int = 8
    initial_batch_size: int = 256
    initial_lr: float = 1e-3
    dt: float = 0.001
    t_stop: float = 0.5
    m: int = 20
    finetune_steps_per_t: int = 1
    finetune_lr: float = 5e-3
    ewc_lambda: float = 10.0
    ewc_rho: float = 0.01
    lora_rank: int = 2
    lora_alpha: float = 1.0
    hidden_sizes: tuple[int, int] = (128, 64)
    validation_per_digit: int = 200
    fisher_samples: int = 512
    initial_fisher_samples: int = 1_024


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    t_values: np.ndarray
    accuracy_9: np.ndarray
    accuracy_all: np.ndarray
    accuracy_non9: np.ndarray
    losses: np.ndarray
    lora_eigenvalues: np.ndarray
    full_eigenvalues: np.ndarray
    lora_stats: dict[str, float]
    full_stats: dict[str, float]
    ewc_diag_weights: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "t_values": self.t_values,
            "accuracy_9": self.accuracy_9,
            "accuracy_all": self.accuracy_all,
            "accuracy_non9": self.accuracy_non9,
            "losses": self.losses,
            "lora_eigenvalues": self.lora_eigenvalues,
            "full_eigenvalues": self.full_eigenvalues,
            "lora_stats": self.lora_stats,
            "full_stats": self.full_stats,
            "ewc_diag_weights": self.ewc_diag_weights,
        }


class Experiment:
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.device = self._resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.train_sampler = MNISTTSampler(
            self.config.data_dir,
            train=True,
            download=True,
            device=self.device,
            seed=self.config.seed,
        )
        self.validation = make_validation_sets(
            self.config.data_dir,
            n_per_digit=self.config.validation_per_digit,
            device=self.device,
            seed=self.config.seed + 1,
        )
        self.model = MNISTLoRAMLP(
            hidden_sizes=self.config.hidden_sizes,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        ).to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return requested

    def run_experiment(self) -> ExperimentResult:
        self._fit_initial_model()

        self.model.set_lora_enabled(True)
        self.model.freeze_base()
        lora_named = list(self.model.named_lora_parameters())
        lora_params = [param for _, param in lora_named]

        init_batch = self.train_sampler.sample(self.config.n_initial, 0.0, device=self.device)
        ewc = OnlineDiagonalEWC(lora_named, rho=self.config.ewc_rho, lambda_=self.config.ewc_lambda)
        init_diag = estimate_diagonal_fisher(
            self.model,
            init_batch.x,
            init_batch.y,
            lora_named,
            max_samples=self.config.initial_fisher_samples,
        )
        ewc.update_from_squared_grads(init_diag)
        ewc.refresh_reference(lora_named)

        optimizer = torch.optim.AdamW(lora_params, lr=self.config.finetune_lr)
        t_values: list[float] = []
        accuracy_9: list[float] = []
        accuracy_all: list[float] = []
        accuracy_non9: list[float] = []
        losses: list[float] = []

        t = 0.0
        while t <= self.config.t_stop:
            t += self.config.dt
            batch = self.train_sampler.sample(self.config.m, min(t, 1.0), device=self.device)
            loss_value = self._finetune_step(batch, optimizer, ewc, lora_named)

            diag = estimate_diagonal_fisher(self.model, batch.x, batch.y, lora_named)
            ewc.update_from_squared_grads(diag)
            ewc.refresh_reference(lora_named)

            metrics = self._accuracy_metrics()
            t_values.append(t)
            accuracy_9.append(metrics["nine"])
            accuracy_all.append(metrics["all"])
            accuracy_non9.append(metrics["non9"])
            losses.append(loss_value)

        final_batch = self.train_sampler.sample(self.config.fisher_samples, min(t, 1.0), device=self.device)
        lora_eigs = empirical_fisher_eigenvalues(self.model, final_batch.x, final_batch.y, lora_params)
        full_eigs = self._full_model_fisher_eigenvalues(final_batch)
        lora_stats = SpectralStats.from_eigenvalues(lora_eigs)
        full_stats = SpectralStats.from_eigenvalues(full_eigs)

        return ExperimentResult(
            config=self.config,
            t_values=np.asarray(t_values),
            accuracy_9=np.asarray(accuracy_9),
            accuracy_all=np.asarray(accuracy_all),
            accuracy_non9=np.asarray(accuracy_non9),
            losses=np.asarray(losses),
            lora_eigenvalues=lora_stats.eigenvalues.numpy(),
            full_eigenvalues=full_stats.eigenvalues.numpy(),
            lora_stats=self._stats_dict(lora_stats),
            full_stats=self._stats_dict(full_stats),
            ewc_diag_weights=ewc.flat_fisher().numpy(),
        )

    def _fit_initial_model(self) -> None:
        self.model.set_lora_enabled(False)
        self.model.freeze_lora()
        optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config.initial_lr,
        )
        loader = self.train_sampler.loader(
            self.config.n_initial,
            0.0,
            self.config.initial_batch_size,
            shuffle=True,
        )

        self.model.train()
        for _ in range(self.config.initial_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()

    def _finetune_step(
        self,
        batch: SampleBatch,
        optimizer: torch.optim.Optimizer,
        ewc: OnlineDiagonalEWC,
        lora_named: list[tuple[str, torch.nn.Parameter]],
    ) -> float:
        self.model.train()
        last_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.config.finetune_steps_per_t):
            optimizer.zero_grad(set_to_none=True)
            ce_loss = F.cross_entropy(self.model(batch.x), batch.y)
            loss = ce_loss + ewc.penalty(lora_named)
            loss.backward()
            optimizer.step()
            last_loss = ce_loss.detach()
        return float(last_loss.item())

    @torch.no_grad()
    def _accuracy_metrics(self) -> dict[str, float]:
        self.model.eval()
        return {
            name: self._accuracy(batch.x, batch.y)
            for name, batch in self.validation.items()
        }

    def _accuracy(self, x: Tensor, y: Tensor) -> float:
        pred = self.model(x).argmax(dim=1)
        return float((pred == y).float().mean().item())

    def _full_model_fisher_eigenvalues(self, batch: SampleBatch) -> Tensor:
        old_requires_grad = [param.requires_grad for param in self.model.parameters()]
        try:
            for param in self.model.parameters():
                param.requires_grad_(True)
            return empirical_fisher_eigenvalues(
                self.model,
                batch.x,
                batch.y,
                [param for param in self.model.parameters()],
            )
        finally:
            for param, requires_grad in zip(self.model.parameters(), old_requires_grad, strict=True):
                param.requires_grad_(requires_grad)

    @staticmethod
    def _stats_dict(stats: SpectralStats) -> dict[str, float]:
        return {
            "trace": stats.trace,
            "condition_number": stats.condition_number,
            "effective_rank": stats.effective_rank,
        }
