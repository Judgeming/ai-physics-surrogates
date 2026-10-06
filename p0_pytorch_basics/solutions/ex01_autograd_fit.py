"""Exercise 1 (reference solution): inverse heat problem with autograd.

Recover the heat generation q and convection coefficient h of a slab from
noisy thermocouple readings. Parameters are optimized in log space so they
stay positive and have comparable scales.
"""

from __future__ import annotations

import math

import torch

from ..physics import temperature

TRUE_Q = 2.0e6  # W/m^3
TRUE_H = 800.0  # W/(m^2 K)
K = 5.0  # W/(m K), known material property


def make_measurements(
    n: int = 41, noise_std: float = 0.1, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy temperature readings at n evenly spaced positions across the slab."""
    gen = torch.Generator().manual_seed(seed)
    xi = torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
    t_true = temperature(xi, TRUE_Q, K, TRUE_H)
    noise = noise_std * torch.randn(n, generator=gen, dtype=torch.float64)
    return xi, t_true + noise


def fit_manual(
    xi: torch.Tensor,
    t_meas: torch.Tensor,
    k: float = K,
    q0: float = 5.0e5,
    h0: float = 3000.0,
    lr: float = 2.0e-3,
    steps: int = 5000,
) -> tuple[float, float]:
    """Plain gradient descent: autograd computes gradients, you apply the update."""
    log_q = torch.tensor(math.log(q0), dtype=torch.float64, requires_grad=True)
    log_h = torch.tensor(math.log(h0), dtype=torch.float64, requires_grad=True)
    for _ in range(steps):
        pred = temperature(xi, log_q.exp(), k, log_h.exp())
        loss = ((pred - t_meas) ** 2).mean()
        loss.backward()
        with torch.no_grad():  # the update itself must not be tracked
            log_q -= lr * log_q.grad
            log_h -= lr * log_h.grad
        log_q.grad.zero_()  # gradients accumulate unless cleared
        log_h.grad.zero_()
    return log_q.exp().item(), log_h.exp().item()


def fit_adam(
    xi: torch.Tensor,
    t_meas: torch.Tensor,
    k: float = K,
    q0: float = 5.0e5,
    h0: float = 3000.0,
    lr: float = 0.05,
    steps: int = 1500,
) -> tuple[float, float]:
    """Same fit with torch.optim.Adam doing the update step."""
    log_q = torch.tensor(math.log(q0), dtype=torch.float64, requires_grad=True)
    log_h = torch.tensor(math.log(h0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_q, log_h], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        pred = temperature(xi, log_q.exp(), k, log_h.exp())
        loss = ((pred - t_meas) ** 2).mean()
        loss.backward()
        optimizer.step()
    return log_q.exp().item(), log_h.exp().item()


if __name__ == "__main__":
    xi, t_meas = make_measurements()
    for name, fit in (("manual GD", fit_manual), ("Adam", fit_adam)):
        q, h = fit(xi, t_meas)
        print(
            f"{name:>9}: q = {q:.4g} W/m^3 ({q / TRUE_Q - 1:+.2%}), "
            f"h = {h:.4g} W/m^2K ({h / TRUE_H - 1:+.2%})"
        )
