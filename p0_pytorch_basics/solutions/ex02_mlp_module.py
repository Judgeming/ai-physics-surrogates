"""Exercise 2 (reference solution): a configurable MLP as an nn.Module."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """Fully connected network: in_dim -> hidden... -> out_dim, no output activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        layers: list[nn.Module] = []
        width = in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalars in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP(4, 1, hidden=(64, 64))
    print(model)
    print("trainable parameters:", count_parameters(model))
