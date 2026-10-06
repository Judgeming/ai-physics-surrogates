"""Exercise 5: out-of-distribution tests and dimensional analysis (~1.5-2 h).

Nondimensionalize the slab problem:
    theta = (T - T_INF) / (q L^2 / k) = (1/4 - xi^2) / 2 + 1 / (2 Bi),   Bi = h L / k
Four inputs (xi, q, k, h) collapse to two (xi, Bi), and q drops out entirely.
You will train a 4-input "dimensional" MLP and a 2-input "dimensionless" MLP at
several training sizes and compare them in-distribution and out-of-distribution.

Before running, write down your prediction for each test set. Then check it.

Check your work:  uv run pytest -k ex05
Run it:           uv run python -m p0_pytorch_basics.ex05_ood_dimensionless   (~1 min on CPU)
"""

from __future__ import annotations

import copy  # noqa: F401
import math  # noqa: F401
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401
from torch.utils.data import DataLoader, TensorDataset  # noqa: F401

from .ex02_mlp_module import MLP
from .ex03_dataset import Normalizer, raw_features, raw_target
from .physics import H_RANGE, K_RANGE, L, Q_RANGE, T_INF, biot, sample_params, temperature  # noqa: F401

Split = dict[str, np.ndarray]


def make_split(
    n_configs: int,
    points_per_config: int = 32,
    seed: int = 0,
    q_range: tuple[float, float] = Q_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    h_range: tuple[float, float] = H_RANGE,
) -> Split:
    """Flattened raw inputs {"xi", "q", "k", "h"} for n_configs random slabs (provided)."""
    rng = np.random.default_rng(seed)
    q, k, h = sample_params(n_configs, rng, q_range, k_range, h_range)
    xi = rng.uniform(-0.5, 0.5, size=(n_configs, points_per_config))
    rep = lambda a: np.repeat(a, points_per_config)  # noqa: E731
    return {"xi": xi.ravel(), "q": rep(q), "k": rep(k), "h": rep(h)}


def dimensionless_features(xi, q, k, h) -> np.ndarray:
    """(N, 2) array [xi, log10 Bi]. q is accepted for a uniform signature but not used."""
    raise NotImplementedError("TODO: exercise 5 (dimensionless_features)")


def dimensionless_target(xi, q, k, h) -> np.ndarray:
    """(N, 1) array of log10 theta, theta = (temperature(...) - T_INF) / (q L^2 / k)."""
    raise NotImplementedError("TODO: exercise 5 (dimensionless_target)")


def train_on_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: Sequence[int] = (64, 64, 64),
    epochs: int = 300,
    lr: float = 3e-3,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[MLP, Normalizer, Normalizer]:
    """A compact training loop on in-memory arrays. Write this one WITHOUT looking at ex04.

    - seed torch; convert arrays to float32 tensors; fit Normalizers on the training arrays
    - DataLoader(TensorDataset(normalized x, normalized y), shuffle=True, seeded generator)
    - MLP(in_dim from x, out_dim from y) on `device`; AdamW; CosineAnnealingLR(T_max=epochs)
    - after each epoch compute the val loss; keep a deepcopy of the best state_dict
    - load the best weights and return (model on CPU in eval mode, x_norm, y_norm)
    """
    raise NotImplementedError("TODO: exercise 5 (train_on_arrays)")


@torch.no_grad()
def predict_delta_t(
    model: MLP,
    x_norm: Normalizer,
    y_norm: Normalizer,
    split: Split,
    dimensionless: bool,
) -> np.ndarray:
    """Predicted temperature rise [deg C] (1-D float64 array) for either feature set.

    Dimensional model: dT = 10 ** output.
    Dimensionless model: theta = 10 ** output, then dT = theta * q L^2 / k.
    """
    raise NotImplementedError("TODO: exercise 5 (predict_delta_t)")


# ---------------------------------------------------------------- provided below


def rel_l2(pred: np.ndarray, split: Split) -> float:
    true = temperature(**split) - T_INF
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def run_comparison(
    n_train_list: Sequence[int] = (25, 100, 400),
    epochs: int = 300,
    points_per_config: int = 32,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> dict[str, dict[int, dict[str, float]]]:
    """Relative L2 error of the temperature rise for both models on three test sets."""
    tests = {
        "in-dist": make_split(200, points_per_config, seed=100),
        "OOD q (x10)": make_split(200, points_per_config, seed=101, q_range=(5.0e6, 5.0e7)),
        "OOD h (x4)": make_split(200, points_per_config, seed=102, h_range=(5000.0, 20000.0)),
    }
    val = make_split(50, points_per_config, seed=99)
    results: dict[str, dict[int, dict[str, float]]] = {"dimensional": {}, "dimensionless": {}}
    for n_train in n_train_list:
        train = make_split(n_train, points_per_config, seed=n_train)
        for name, dimless in (("dimensional", False), ("dimensionless", True)):
            feats = dimensionless_features if dimless else raw_features
            target = dimensionless_target if dimless else raw_target
            model, xn, yn = train_on_arrays(
                feats(**train), target(**train), feats(**val), target(**val), epochs=epochs, device=device
            )
            results[name][n_train] = {
                test_name: rel_l2(predict_delta_t(model, xn, yn, split, dimless), split)
                for test_name, split in tests.items()
            }
            row = ", ".join(f"{k}: {v:.2%}" for k, v in results[name][n_train].items())
            log(f"n_train={n_train:4d} {name:>13} | {row}")
    return results


def plot_results(results: dict[str, dict[int, dict[str, float]]], out_png: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_names = list(next(iter(results["dimensional"].values())).keys())
    fig, axes = plt.subplots(1, len(test_names), figsize=(4.2 * len(test_names), 3.4), sharey=True)
    for ax, test_name in zip(axes, test_names):
        for name, style in (("dimensional", "o-"), ("dimensionless", "s--")):
            ns = sorted(results[name])
            ax.plot(ns, [results[name][n][test_name] for n in ns], style, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(test_name)
        ax.set_xlabel("training slabs")
    axes[0].set_ylabel("relative L2 error of T - T_inf")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    out = Path("runs/p0_ood")
    out.mkdir(parents=True, exist_ok=True)
    results = run_comparison()
    plot_results(results, out / "data_efficiency.png")
    print(f"plot: {out / 'data_efficiency.png'}")
