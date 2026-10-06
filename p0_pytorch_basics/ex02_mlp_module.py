"""Exercise 2: a configurable MLP as an nn.Module (~0.5-1 h).

Check your work:  uv run pytest -k ex02
Run it:           uv run python -m p0_pytorch_basics.ex02_mlp_module
"""

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
    """Fully connected network: in_dim -> hidden... -> out_dim, no output activation.

    Example: MLP(4, 1, hidden=(64, 64)) is Linear(4,64) GELU Linear(64,64) GELU Linear(64,1).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        # TODO:
        # - raise ValueError for an activation name not in ACTIVATIONS
        # - build a list of layers: Linear + activation per hidden width, then a final Linear
        # - store it as self.net = nn.Sequential(*layers)
        #   (assigning a module to an attribute is what registers its parameters)
        if activation not in ACTIVATIONS:
            raise ValueError
        layers: list[nn.Module] = []
        width = in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential (*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number ofarameter trainable scalars. Hint: model.ps(), p.numel(), p.requires_grad."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


if __name__ == "__main__":
    model = MLP(4, 1, hidden=(64, 64))
    print(model)
    print("trainable parameters:", count_parameters(model))  # work out 4545 by hand first
