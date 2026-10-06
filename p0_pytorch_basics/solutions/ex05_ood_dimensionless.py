"""Exercise 5 (reference solution): out-of-distribution tests and dimensional analysis.

Nondimensionalizing the slab problem gives
    theta = (T - T_INF) / (q L^2 / k) = (1/4 - xi^2) / 2 + 1 / (2 Bi),  Bi = h L / k
so the four inputs (xi, q, k, h) collapse to two (xi, Bi), and q drops out.
This compares a 4-input "dimensional" MLP against a 2-input "dimensionless" MLP
on in-distribution and out-of-distribution test sets at several training sizes.

Run:
    uv run python -m p0_pytorch_basics.solutions.ex05_ood_dimensionless
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..physics import H_RANGE, K_RANGE, L, Q_RANGE, T_INF, biot, sample_params, temperature
from .ex02_mlp_module import MLP
from .ex03_dataset import Normalizer, raw_features, raw_target

Split = dict[str, np.ndarray]


def make_split(
    n_configs: int,
    points_per_config: int = 32,
    seed: int = 0,
    q_range: tuple[float, float] = Q_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    h_range: tuple[float, float] = H_RANGE,
) -> Split:
    """Flattened raw inputs (xi, q, k, h) for n_configs random slabs."""
    rng = np.random.default_rng(seed)
    q, k, h = sample_params(n_configs, rng, q_range, k_range, h_range)
    xi = rng.uniform(-0.5, 0.5, size=(n_configs, points_per_config))
    rep = lambda a: np.repeat(a, points_per_config)  # noqa: E731
    return {"xi": xi.ravel(), "q": rep(q), "k": rep(k), "h": rep(h)}


def dimensionless_features(xi, q, k, h) -> np.ndarray:
    """(N, 2) array [xi, log10 Bi]. q is deliberately absent."""
    xi, k, h = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (xi, k, h)))
    return np.stack([xi, np.log10(biot(h, k))], axis=-1).reshape(-1, 2)


def dimensionless_target(xi, q, k, h) -> np.ndarray:
    """(N, 1) array of log10 theta with theta = (T - T_INF) / (q L^2 / k)."""
    xi, q, k, h = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (xi, q, k, h)))
    theta = (temperature(xi, q, k, h) - T_INF) / (q * L**2 / k)
    return np.log10(theta).reshape(-1, 1)


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
    """Compact training loop on in-memory arrays; keeps the weights with the best val loss."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    x_norm, y_norm = Normalizer().fit(xt), Normalizer().fit(yt)
    loader = DataLoader(
        TensorDataset(x_norm.transform(xt), y_norm.transform(yt)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xv = x_norm.transform(torch.as_tensor(x_val, dtype=torch.float32)).to(device)
    yv = y_norm.transform(torch.as_tensor(y_val, dtype=torch.float32)).to(device)

    model = MLP(xt.shape[1], yt.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val, best_state = math.inf, copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(xv), yv).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(), x_norm, y_norm


@torch.no_grad()
def predict_delta_t(
    model: MLP,
    x_norm: Normalizer,
    y_norm: Normalizer,
    split: Split,
    dimensionless: bool,
) -> np.ndarray:
    """Predicted temperature rise [deg C] for a split, for either feature set."""
    feats_fn = dimensionless_features if dimensionless else raw_features
    x = torch.as_tensor(feats_fn(**split), dtype=torch.float32)
    log_out = y_norm.inverse_transform(model(x_norm.transform(x))).squeeze(-1).double().numpy()
    if dimensionless:
        return 10.0**log_out * split["q"] * L**2 / split["k"]
    return 10.0**log_out


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
