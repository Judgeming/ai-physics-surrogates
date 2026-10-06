"""Exercise 1: inverse heat problem with autograd (~1-1.5 h).

Recover the heat generation q and convection coefficient h of a slab from
noisy thermocouple readings (k is known). You already know tensors and
autograd from BoTorch; the point here is the raw optimization loop that every
later training loop is built on.

Check your work:  uv run pytest -k ex01
Run it:           uv run python -m p0_pytorch_basics.ex01_autograd_fit
"""

from __future__ import annotations

import math  # noqa: F401  (you will need math.log)

import torch

from .physics import temperature

TRUE_Q = 2.0e6  # W/m^3
TRUE_H = 800.0  # W/(m^2 K)
K = 5.0  # W/(m K), known material property


def make_measurements(
    n: int = 41, noise_std: float = 0.1, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy temperature readings at n evenly spaced positions (provided)."""
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
    """Plain gradient descent, no torch.optim. Returns (q, h) as Python floats.

    TODO:
    
    1. Create two scalar float64 leaf tensors log_q, log_h (initialized at log(q0),
       log(h0)) with requires_grad=True. Optimizing in log space keeps q, h > 0
       and puts both parameters on a similar scale.
    2. Loop `steps` times:
         - predict with temperature(xi, log_q.exp(), k, log_h.exp())
         - loss = mean squared error against t_meas
         - loss.backward()
         - update each parameter in-place inside `with torch.no_grad():`
         - zero both .grad tensors (they accumulate otherwise)
    3. Return (log_q.exp().item(), log_h.exp().item()).

    Try lr=4e-3 once and watch what happens, then explain why.
    """
    log_q = torch.tensor(math.log(q0), dtype=torch.float64, requires_grad=True)
    log_h = torch.tensor(math.log(h0), dtype=torch.float64, requires_grad=True)
    for step in range(steps):
        prediction = temperature(xi, log_q.exp(), k, log_h.exp())
        loss = ((prediction - t_meas) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            log_q -= lr * log_q.grad
            log_h -= lr * log_h.grad
        log_q.grad.zero_()
        log_h.grad.zero_()
    return (log_q.exp().item(), log_h.exp().item())


def fit_adam(
    xi: torch.Tensor,
    t_meas: torch.Tensor,
    k: float = K,
    q0: float = 5.0e5,
    h0: float = 3000.0,
    lr: float = 0.05,
    steps: int = 1500,
) -> tuple[float, float]:
    """Same fit, but torch.optim.Adam performs the update.

    TODO: same parameters as fit_manual; build torch.optim.Adam([log_q, log_h], lr=lr);
    each step: optimizer.zero_grad() -> forward -> loss.backward() -> optimizer.step().

    Question to answer for yourself: both methods land ~2% away from the true
    values. Is that an optimizer problem or something else?
    """
    log_q = torch.tensor(math.log(q0), dtype=torch.float64, requires_grad=True)
    log_h = torch.tensor(math.log(h0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_q, log_h], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        prediction = temperature(xi, log_q.exp(), k, log_h.exp())
        loss = ((prediction - t_meas) ** 2).mean()
        loss.backward()
        optimizer.step()
    return (log_q.exp().item(), log_h.exp().item())


if __name__ == "__main__":
    xi, t_meas = make_measurements()
    for name, fit in (("manual GD", fit_manual), ("Adam", fit_adam)):
        q, h = fit(xi, t_meas)
        print(
            f"{name:>9}: q = {q:.4g} W/m^3 ({q / TRUE_Q - 1:+.2%}), "
            f"h = {h:.4g} W/m^2K ({h / TRUE_H - 1:+.2%})"
        )
