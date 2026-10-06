"""Exercise 4: a complete, reproducible training script (~2-3 h).

Uses your MLP (ex02) and data pipeline (ex03). The CLI, plotting and config are
provided; you write the parts every training script has.

Check your work:  uv run pytest -k ex04
Run it:           uv run python -m p0_pytorch_basics.ex04_train
                  uv run python -m p0_pytorch_basics.ex04_train --device cpu --hidden 128,128
Watch it:         uv run tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json  # noqa: F401
import math  # noqa: F401
import random  # noqa: F401
from dataclasses import asdict, dataclass, fields  # noqa: F401
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .ex02_mlp_module import MLP
from .ex03_dataset import Normalizer, make_loaders, raw_features  # noqa: F401
from .physics import T_INF, temperature  # noqa: F401


@dataclass
class TrainConfig:
    n_train_configs: int = 400
    n_val_configs: int = 100
    points_per_config: int = 32
    batch_size: int = 256
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "gelu"
    lr: float = 3e-3
    weight_decay: float = 0.0
    max_epochs: int = 300
    patience: int = 40
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs/p0_slab"
    tensorboard: bool = True


def set_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and torch."""
    raise NotImplementedError("TODO: exercise 4 (set_seed)")


def pick_device(preference: str = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu; anything else -> torch.device(preference)."""
    raise NotImplementedError("TODO: exercise 4 (pick_device)")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over the training data; return the sample-weighted mean loss.

    The five lines to know by heart: model.train(); then per batch move to device,
    optimizer.zero_grad(), loss = loss_fn(model(x), y), loss.backward(), optimizer.step().
    Accumulate loss.item() * batch_size (not the tensor, which keeps the graph alive).
    """
    raise NotImplementedError("TODO: exercise 4 (train_one_epoch)")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    y_norm: Normalizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Return {"loss", "rel_l2", "max_abs_err_C"}.

    - model.eval() first (matters once you add dropout/batch-norm; good habit now)
    - "loss": sample-weighted mean loss in normalized space
    - convert predictions and targets back to the temperature rise in deg C:
      dT = 10 ** y_norm.inverse_transform(...)
    - "rel_l2": ||dT_pred - dT_true|| / ||dT_true|| over the whole loader
    - "max_abs_err_C": worst-case absolute error in deg C (engineers care about this one)
    """
    raise NotImplementedError("TODO: exercise 4 (evaluate)")


def fit(cfg: TrainConfig) -> dict[str, Any]:
    """Train with early stopping and return
    {"best_epoch", "best_val" (history row of the best epoch), "checkpoint" (path str), "history" (list of dicts)}.

    Outline:
    1. set_seed, pick_device, create cfg.out_dir
    2. loaders + normalizers from make_loaders; MLP(4, 1, ...) moved to the device
    3. optimizer = torch.optim.AdamW(...); scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10);
       loss_fn = nn.MSELoss()
    4. if cfg.tensorboard: SummaryWriter(log_dir=out_dir/"tb") (import it inside the if)
    5. per epoch: train_one_epoch -> evaluate -> scheduler.step(val loss) -> append a history row
       {"epoch", "train_loss", "lr", **val} -> log scalars
    6. if val loss improved: torch.save({"model_state", "config": asdict(cfg), "x_norm": x_norm.state_dict(),
       "y_norm": y_norm.state_dict(), "epoch", "val"}, out_dir/"best.pt")
       elif no improvement for cfg.patience epochs: stop
    7. close the writer, write history.json, return the dict above
    """
    raise NotImplementedError("TODO: exercise 4 (fit)")


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MLP, Normalizer, Normalizer, TrainConfig]:
    """Rebuild (model, x_norm, y_norm, cfg) from a checkpoint.

    Hints: torch.load(path, map_location=device, weights_only=True); the saved config's
    "hidden" may come back as a list, convert it to a tuple; model.load_state_dict(...);
    return the model in eval mode.
    """
    raise NotImplementedError("TODO: exercise 4 (load_checkpoint)")


@torch.no_grad()
def predict_temperature(model: nn.Module, x_norm: Normalizer, y_norm: Normalizer, xi, q, k, h) -> np.ndarray:
    """Surrogate temperature [deg C] as a 1-D NumPy array for broadcastable inputs.

    raw_features -> float32 tensor on the model's device -> x_norm.transform -> model
    -> y_norm.inverse_transform -> 10 ** (...) + T_INF -> .cpu().numpy()
    """
    raise NotImplementedError("TODO: exercise 4 (predict_temperature)")


# ---------------------------------------------------------------- provided below


def plot_predictions(ckpt_path: str | Path, out_png: str | Path, n_slabs: int = 4, seed: int = 123) -> None:
    """Overlay surrogate and analytical profiles for a few random slabs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .physics import sample_params

    model, x_norm, y_norm, _ = load_checkpoint(ckpt_path)
    q, k, h = sample_params(n_slabs, np.random.default_rng(seed))
    xi = np.linspace(-0.5, 0.5, 101)
    fig, axes = plt.subplots(1, n_slabs, figsize=(4 * n_slabs, 3.2))
    for ax, qi, ki, hi in zip(axes, q, k, h):
        ax.plot(xi, temperature(xi, qi, ki, hi), "k-", label="analytical")
        ax.plot(xi, predict_temperature(model, x_norm, y_norm, xi, qi, ki, hi), "r--", label="MLP")
        ax.set_title(f"q={qi:.2g}, k={ki:.3g}, h={hi:.3g}", fontsize=9)
        ax.set_xlabel("x / L")
    axes[0].set_ylabel("T [deg C]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in fields(TrainConfig):
        value = getattr(defaults, f.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in {"1", "true", "yes"}, default=value)
        elif isinstance(value, tuple):
            parser.add_argument(f"--{f.name}", type=lambda s: tuple(int(v) for v in s.split(",")), default=value)
        else:
            parser.add_argument(f"--{f.name}", type=type(value), default=value)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    result = fit(cfg)
    best = result["best_val"]
    print(
        f"best epoch {result['best_epoch']}: rel L2 = {best['rel_l2']:.3%}, "
        f"max |error| = {best['max_abs_err_C']:.3f} deg C"
    )
    png = Path(cfg.out_dir) / "predictions.png"
    plot_predictions(result["checkpoint"], png)
    print(f"checkpoint: {result['checkpoint']}\nplot: {png}")


if __name__ == "__main__":
    main()
