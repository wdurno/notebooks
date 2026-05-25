from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch

from src.data.io import JsonlExample
from src.lanczos import l_lanczos
from src.training import SupervisedDataset


@dataclass
class EWCState:
    theta_star: torch.Tensor
    low_rank: torch.Tensor
    diagonal: torch.Tensor
    param_shapes: list[tuple[str, torch.Size, int]]
    fisher_wall_seconds: float
    ewc_n0: int
    ewc_rank: int

    @property
    def effective_rank(self) -> int:
        return self.low_rank.shape[1]


def trainable_named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def flatten_trainable_parameters(model) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    chunks = []
    shapes: list[tuple[str, torch.Size, int]] = []
    for name, param in trainable_named_parameters(model):
        flat = param.detach().reshape(-1)
        chunks.append(flat)
        shapes.append((name, param.shape, flat.numel()))
    if not chunks:
        raise ValueError("Model has no trainable parameters.")
    return torch.cat(chunks), shapes


def flatten_live_trainable_parameters(model) -> torch.Tensor:
    chunks = [param.reshape(-1) for _name, param in trainable_named_parameters(model)]
    if not chunks:
        raise ValueError("Model has no trainable parameters.")
    return torch.cat(chunks)


def flatten_trainable_grads(model) -> torch.Tensor:
    chunks = []
    for _name, param in trainable_named_parameters(model):
        if param.grad is None:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    return torch.cat(chunks)


def clear_grads(model) -> None:
    for param in model.parameters():
        param.grad = None


def gradient_generator(
    model,
    tokenizer,
    examples: list[JsonlExample],
    max_seq_length: int,
) -> Iterator[torch.Tensor]:
    from src.modeling import get_primary_device

    device = get_primary_device(model)
    dataset = SupervisedDataset(examples, tokenizer, max_seq_length)
    for item in dataset:
        clear_grads(model)
        batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        yield flatten_trainable_grads(model).detach()
    clear_grads(model)


def estimate_ewc(
    model,
    tokenizer,
    examples: list[JsonlExample],
    ewc_n0: int,
    ewc_rank: int,
    max_seq_length: int,
    min_off_diag: float = 1e-12,
) -> EWCState:
    model.train()
    start = time.time()
    theta_star, shapes = flatten_trainable_parameters(model)
    device = theta_star.device
    selected = examples[:ewc_n0]
    p = theta_star.numel()

    if ewc_n0 <= 0 or not selected:
        low_rank = torch.zeros((p, 0), device=device)
        diagonal = torch.zeros((p, 1), device=device)
        return EWCState(theta_star, low_rank, diagonal, shapes, time.time() - start, 0, ewc_rank)

    def get_grad_generator():
        return lambda: gradient_generator(model, tokenizer, selected, max_seq_length)

    if ewc_rank > 0:
        low_rank, diagonal = l_lanczos(
            get_grad_generator=get_grad_generator,
            r=ewc_rank,
            p=p,
            device=device,
            calc_diag=True,
            disable_tqdm=True,
            min_off_diag=min_off_diag,
            normalize=True,
        )
    else:
        low_rank = torch.zeros((p, 0), device=device)
        diagonal = torch.zeros((p, 1), device=device)
        n = 0
        for grad in gradient_generator(model, tokenizer, selected, max_seq_length):
            grad = grad.to(device).reshape(-1, 1)
            diagonal += grad * grad
            n += 1
        diagonal = diagonal / max(n, 1)

    return EWCState(
        theta_star=theta_star.detach(),
        low_rank=low_rank.detach(),
        diagonal=diagonal.detach(),
        param_shapes=shapes,
        fisher_wall_seconds=time.time() - start,
        ewc_n0=len(selected),
        ewc_rank=ewc_rank,
    )


def ewc_weighted_drift(model, state: EWCState) -> torch.Tensor:
    theta = flatten_live_trainable_parameters(model)
    delta = theta - state.theta_star.to(theta.device)
    drift = torch.sum(state.diagonal.to(theta.device).reshape(-1) * delta.pow(2))
    if state.low_rank.numel() > 0:
        projected = state.low_rank.to(theta.device).T.matmul(delta.reshape(-1, 1))
        drift = drift + torch.sum(projected.pow(2))
    return drift


def make_ewc_penalty(model, state: EWCState):
    def penalty() -> torch.Tensor:
        return ewc_weighted_drift(model, state)

    return penalty
