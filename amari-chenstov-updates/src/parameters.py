"""Stable flattening and restoration of trainable parameter pytrees."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class ParameterLayoutError(ValueError):
    """Raised when tensors do not match a recorded parameter layout."""


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int
    dtype: str

    @property
    def numel(self) -> int:
        return self.stop - self.start

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "start": self.start,
            "stop": self.stop,
            "dtype": self.dtype,
        }


@dataclasses.dataclass(frozen=True)
class ParameterLayout:
    specs: tuple[ParameterSpec, ...]
    total_numel: int

    @classmethod
    def from_named_parameters(
        cls,
        named_parameters: Sequence[tuple[str, Tensor]],
    ) -> "ParameterLayout":
        specs = []
        offset = 0
        seen_names = set()
        for name, parameter in named_parameters:
            if name in seen_names:
                raise ParameterLayoutError(f"duplicate parameter name: {name}")
            seen_names.add(name)
            numel = parameter.numel()
            specs.append(
                ParameterSpec(
                    name=name,
                    shape=tuple(parameter.shape),
                    start=offset,
                    stop=offset + numel,
                    dtype=str(parameter.dtype),
                )
            )
            offset += numel

        if not specs:
            raise ParameterLayoutError("layout requires at least one parameter")
        return cls(specs=tuple(specs), total_numel=offset)

    @classmethod
    def from_module(cls, module: nn.Module) -> "ParameterLayout":
        return cls.from_named_parameters(
            tuple(
                (name, parameter)
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def validate_named_tensors(self, tensors: Mapping[str, Tensor]) -> None:
        supplied = set(tensors)
        expected = set(self.names)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ParameterLayoutError(
                f"named tensors do not match layout: {', '.join(details)}"
            )

        for spec in self.specs:
            tensor = tensors[spec.name]
            if tuple(tensor.shape) != spec.shape:
                raise ParameterLayoutError(
                    f"shape mismatch for {spec.name}: expected {spec.shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            if str(tensor.dtype) != spec.dtype:
                raise ParameterLayoutError(
                    f"dtype mismatch for {spec.name}: expected {spec.dtype}, "
                    f"got {tensor.dtype}"
                )

    def validate_module(self, module: nn.Module) -> None:
        current = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self.validate_named_tensors(current)

    def flatten_named(self, tensors: Mapping[str, Tensor]) -> Tensor:
        self.validate_named_tensors(tensors)
        pieces = [tensors[spec.name].reshape(-1) for spec in self.specs]
        try:
            return torch.cat(pieces)
        except RuntimeError as exc:
            raise ParameterLayoutError(
                "parameters must share a compatible dtype and device"
            ) from exc

    def flatten_module(self, module: nn.Module, *, detach: bool = False) -> Tensor:
        named = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        flat = self.flatten_named(named)
        return flat.detach().clone() if detach else flat

    def flatten_batched_named(self, tensors: Mapping[str, Tensor]) -> Tensor:
        supplied = set(tensors)
        expected = set(self.names)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ParameterLayoutError(
                f"named tensors do not match layout: {', '.join(details)}"
            )

        batch_size = None
        pieces = []
        for spec in self.specs:
            tensor = tensors[spec.name]
            expected_rank = len(spec.shape) + 1
            if tensor.ndim != expected_rank or tuple(tensor.shape[1:]) != spec.shape:
                raise ParameterLayoutError(
                    f"batched shape mismatch for {spec.name}: expected "
                    f"(batch, {spec.shape}), got {tuple(tensor.shape)}"
                )
            if str(tensor.dtype) != spec.dtype:
                raise ParameterLayoutError(
                    f"dtype mismatch for {spec.name}: expected {spec.dtype}, "
                    f"got {tensor.dtype}"
                )
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ParameterLayoutError("batched tensors have unequal batch sizes")
            pieces.append(tensor.reshape(tensor.shape[0], -1))

        try:
            return torch.cat(pieces, dim=1)
        except RuntimeError as exc:
            raise ParameterLayoutError(
                "parameters must share a compatible dtype and device"
            ) from exc

    def unflatten_named(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.ndim != 1 or vector.numel() != self.total_numel:
            raise ParameterLayoutError(
                f"expected vector of shape ({self.total_numel},), "
                f"got {tuple(vector.shape)}"
            )
        return {
            spec.name: vector[spec.start : spec.stop].reshape(spec.shape)
            for spec in self.specs
        }

    def copy_vector_to_module(self, module: nn.Module, vector: Tensor) -> None:
        """Copy a flat parameter vector into a validated module in place."""

        self.validate_module(module)
        current = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        first = current[self.specs[0].name]
        if vector.dtype != first.dtype or vector.device != first.device:
            raise ParameterLayoutError(
                "parameter vector must share the module dtype and device"
            )
        restored = self.unflatten_named(vector)
        with torch.no_grad():
            for name in self.names:
                current[name].copy_(restored[name])

    def metadata(self) -> dict[str, Any]:
        return {
            "total_numel": self.total_numel,
            "parameters": [spec.to_mapping() for spec in self.specs],
        }

    def assert_metadata(self, metadata: Mapping[str, Any]) -> None:
        if metadata != self.metadata():
            raise ParameterLayoutError("parameter-layout metadata does not match")
