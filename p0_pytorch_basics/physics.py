"""Analytical model used throughout P0 (provided, not an exercise).

A 1D slab of thickness L with uniform volumetric heat generation q, thermal
conductivity k, and convective cooling (coefficient h, coolant temperature
T_INF) on both faces. With x = xi * L and xi in [-0.5, 0.5], the steady-state
temperature is

    T(xi) = q / (2k) * (L^2/4 - x^2) + q L / (2h) + T_INF

The functions accept NumPy arrays or torch tensors (plain arithmetic only), so
the same code serves both the data generator and the autograd exercise.
"""

from __future__ import annotations

import numpy as np

L = 0.01  # slab thickness [m]
T_INF = 25.0  # coolant temperature [deg C]

# Training ranges, sampled log-uniformly.
Q_RANGE = (1.0e5, 5.0e6)  # volumetric heat generation [W/m^3]
K_RANGE = (5.0, 200.0)  # thermal conductivity [W/(m K)]
H_RANGE = (200.0, 5000.0)  # convection coefficient [W/(m^2 K)]


def temperature(xi, q, k, h, length: float = L, t_inf: float = T_INF):
    """Steady temperature [deg C] at normalized position xi."""
    x = xi * length
    return q / (2 * k) * (length**2 / 4 - x**2) + q * length / (2 * h) + t_inf


def biot(h, k, length: float = L):
    """Biot number Bi = h L / k (surface convection vs. internal conduction)."""
    return h * length / k


def sample_params(
    n: int,
    rng: np.random.Generator,
    q_range: tuple[float, float] = Q_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    h_range: tuple[float, float] = H_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw n (q, k, h) triples, each log-uniform over its range."""

    def log_uniform(lo: float, hi: float) -> np.ndarray:
        return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)

    return log_uniform(*q_range), log_uniform(*k_range), log_uniform(*h_range)
