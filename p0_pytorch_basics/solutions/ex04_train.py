"""Exercise 4 (reference solution): a complete, reproducible training script.

Run:
    uv run python -m p0_pytorch_basics.solutions.ex04_train
    uv run tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..physics import T_INF, temperature
from .ex02_mlp_module import MLP
from .ex03_dataset import Normalizer, make_loaders, raw_features


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def pick_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over the training data; returns the sample-weighted mean loss."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    y_norm: Normalizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Loss in normalized space plus engineering metrics on the temperature rise [deg C]."""
    model.eval()
    total, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += loss_fn(pred, y).item() * x.shape[0]
        preds.append(pred.cpu())
        targets.append(y.cpu())
    pred, target = torch.cat(preds), torch.cat(targets)
    dt_pred = 10.0 ** y_norm.inverse_transform(pred)
    dt_true = 10.0 ** y_norm.inverse_transform(target)
    err = dt_pred - dt_true
    return {
        "loss": total / len(target),
        "rel_l2": (torch.linalg.norm(err) / torch.linalg.norm(dt_true)).item(),
        "max_abs_err_C": err.abs().max().item(),
    }


def fit(cfg: TrainConfig) -> dict[str, Any]:
    """Train with early stopping; saves the best checkpoint and the history to cfg.out_dir."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, (x_norm, y_norm) = make_loaders(
        cfg.n_train_configs, cfg.n_val_configs, cfg.points_per_config, cfg.batch_size, cfg.seed
    )
    model = MLP(4, 1, hidden=cfg.hidden, activation=cfg.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out / "tb"))

    ckpt_path = out / "best.pt"
    best_loss, best_epoch, history = math.inf, -1, []
    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = evaluate(model, val_loader, y_norm, loss_fn, device)
        scheduler.step(val["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "lr": lr_now, **val})
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val["loss"], epoch)
            writer.add_scalar("val/rel_l2", val["rel_l2"], epoch)
            writer.add_scalar("val/max_abs_err_C", val["max_abs_err_C"], epoch)
            writer.add_scalar("lr", lr_now, epoch)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "x_norm": x_norm.state_dict(),
                    "y_norm": y_norm.state_dict(),
                    "epoch": epoch,
                    "val": val,
                },
                ckpt_path,
            )
        elif epoch - best_epoch >= cfg.patience:
            break
    if writer is not None:
        writer.close()
    (out / "history.json").write_text(json.dumps(history, indent=1))
    return {"best_epoch": best_epoch, "best_val": history[best_epoch], "checkpoint": str(ckpt_path), "history": history}


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MLP, Normalizer, Normalizer, TrainConfig]:
    """Rebuild the model and normalizers exactly as they were at the best epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    raw = dict(ckpt["config"])
    raw["hidden"] = tuple(raw["hidden"])
    cfg = TrainConfig(**raw)
    model = MLP(4, 1, hidden=cfg.hidden, activation=cfg.activation)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, Normalizer.from_state_dict(ckpt["x_norm"]), Normalizer.from_state_dict(ckpt["y_norm"]), cfg


@torch.no_grad()
def predict_temperature(model: nn.Module, x_norm: Normalizer, y_norm: Normalizer, xi, q, k, h) -> np.ndarray:
    """Surrogate temperature [deg C] for broadcastable inputs."""
    device = next(model.parameters()).device
    feats = torch.as_tensor(raw_features(xi, q, k, h), dtype=torch.float32, device=device)
    log_dt = y_norm.inverse_transform(model(x_norm.transform(feats))).cpu()
    return (10.0**log_dt).squeeze(-1).numpy() + T_INF


def plot_predictions(ckpt_path: str | Path, out_png: str | Path, n_slabs: int = 4, seed: int = 123) -> None:
    """Overlay surrogate and analytical profiles for a few random slabs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..physics import sample_params

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
