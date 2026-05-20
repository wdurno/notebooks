from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class SampleBatch:
    x: Tensor
    y: Tensor


class MNISTTSampler:
    """Sample MNIST with P(y=9)=t and P(y=k)=(1-t)/9 for k in 0..8."""

    def __init__(
        self,
        root: str | Path = "data",
        *,
        train: bool = True,
        download: bool = True,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root=str(root), train=train, download=download, transform=transform)

        xs = dataset.data.float().div_(255.0).unsqueeze(1)
        ys = dataset.targets.long()

        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

        self.by_digit: dict[int, Tensor] = {}
        for digit in range(10):
            self.by_digit[digit] = torch.nonzero(ys == digit, as_tuple=False).flatten()

        self.xs = xs
        self.ys = ys

    def sample(self, n: int, t: float, *, device: torch.device | str | None = None) -> SampleBatch:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t must be in [0, 1], got {t}")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        probs = torch.full((10,), (1.0 - t) / 9.0)
        probs[9] = t
        digits = torch.multinomial(probs, n, replacement=True, generator=self.generator)

        idx = torch.empty(n, dtype=torch.long)
        for digit in range(10):
            mask = digits == digit
            count = int(mask.sum().item())
            if count == 0:
                continue
            pool = self.by_digit[digit]
            chosen = torch.randint(len(pool), (count,), generator=self.generator)
            idx[mask] = pool[chosen]

        target_device = torch.device(device) if device is not None else self.device
        return SampleBatch(self.xs[idx].to(target_device), self.ys[idx].to(target_device))

    def loader(self, n: int, t: float, batch_size: int, *, shuffle: bool = True) -> DataLoader:
        batch = self.sample(n, t, device="cpu")
        dataset = TensorDataset(batch.x, batch.y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_validation_sets(
    root: str | Path = "data",
    *,
    n_per_digit: int = 500,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, SampleBatch]:
    sampler = MNISTTSampler(root=root, train=False, download=True, device=device, seed=seed)
    batches: dict[str, SampleBatch] = {}
    per_digit_x: list[Tensor] = []
    per_digit_y: list[Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for digit in range(10):
        pool = sampler.by_digit[digit]
        count = min(n_per_digit, len(pool))
        chosen = pool[torch.randperm(len(pool), generator=generator)[:count]]
        per_digit_x.append(sampler.xs[chosen])
        per_digit_y.append(sampler.ys[chosen])

    x_all = torch.cat(per_digit_x).to(device)
    y_all = torch.cat(per_digit_y).to(device)
    batches["all"] = SampleBatch(x_all, y_all)

    batches["nine"] = SampleBatch(per_digit_x[9].to(device), per_digit_y[9].to(device))
    x_non9 = torch.cat(per_digit_x[:9]).to(device)
    y_non9 = torch.cat(per_digit_y[:9]).to(device)
    batches["non9"] = SampleBatch(x_non9, y_non9)
    return batches
