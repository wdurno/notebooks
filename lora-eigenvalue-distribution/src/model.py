from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 2,
        alpha: float = 1.0,
        bias: bool = True,
        lora_std: float = 0.01,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        self.lora_enabled = rank > 0

        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.normal_(self.lora_a, mean=0.0, std=lora_std)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self.rank > 0 and self.lora_enabled:
            lora = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
            out = out + lora
        return out


class MNISTLoRAMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, int] = (128, 64),
        rank: int = 2,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Flatten(),
            LoRALinear(28 * 28, h1, rank=rank, alpha=alpha),
            nn.ReLU(),
            LoRALinear(h1, h2, rank=rank, alpha=alpha),
            nn.ReLU(),
            LoRALinear(h2, 10, rank=rank, alpha=alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def set_lora_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_enabled = enabled and module.rank > 0

    def freeze_lora(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = not name.endswith(("lora_a", "lora_b"))

    def freeze_base(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = name.endswith(("lora_a", "lora_b"))

    def train_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters():
            if name.endswith(("lora_a", "lora_b")):
                yield param

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if name.endswith(("lora_a", "lora_b")):
                yield name, param

    def reset_lora(self) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear) and module.rank > 0:
                nn.init.normal_(module.lora_a, mean=0.0, std=0.01)
                nn.init.zeros_(module.lora_b)


def count_parameters(parameters: Iterator[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)
