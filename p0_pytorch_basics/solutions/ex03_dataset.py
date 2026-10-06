"""Exercise 3 (reference solution): Dataset, normalization, and DataLoaders.

Each sample is one (position, slab configuration) pair:
    features = [xi, log10 q, log10 k, log10 h]
    target   = log10(T - T_INF)
The temperature rise spans ~0.1 to ~150 deg C across the training ranges, so
the target is predicted in log space and both sides are standardized.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..physics import H_RANGE, K_RANGE, Q_RANGE, T_INF, sample_params, temperature

FEATURE_NAMES = ("xi", "log10_q", "log10_k", "log10_h")


def raw_features(xi, q, k, h) -> np.ndarray:
    """Stack broadcast inputs into an (N, 4) feature array."""
    xi, q, k, h = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (xi, q, k, h)))
    return np.stack([xi, np.log10(q), np.log10(k), np.log10(h)], axis=-1).reshape(-1, 4)


def raw_target(xi, q, k, h) -> np.ndarray:
    """(N, 1) array of log10 temperature rise above the coolant."""
    xi, q, k, h = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (xi, q, k, h)))
    return np.log10(temperature(xi, q, k, h) - T_INF).reshape(-1, 1)


class Normalizer:
    """Per-column standardization z = (x - mean) / std, fitted on training data only."""

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> None:
        self.mean = mean
        self.std = std

    def fit(self, x: torch.Tensor) -> Normalizer:
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> Normalizer:
        return cls(mean=state["mean"], std=state["std"])


class SlabDataset(Dataset):
    """n_configs random slabs, each sampled at points_per_config random positions."""

    def __init__(
        self,
        n_configs: int,
        points_per_config: int = 32,
        seed: int = 0,
        x_norm: Normalizer | None = None,
        y_norm: Normalizer | None = None,
        q_range: tuple[float, float] = Q_RANGE,
        k_range: tuple[float, float] = K_RANGE,
        h_range: tuple[float, float] = H_RANGE,
    ) -> None:
        rng = np.random.default_rng(seed)
        q, k, h = sample_params(n_configs, rng, q_range, k_range, h_range)
        xi = rng.uniform(-0.5, 0.5, size=(n_configs, points_per_config))
        qq, kk, hh = (a[:, None] for a in (q, k, h))  # broadcast against xi
        x = torch.as_tensor(raw_features(xi, qq, kk, hh), dtype=torch.float32)
        y = torch.as_tensor(raw_target(xi, qq, kk, hh), dtype=torch.float32)
        self.x_norm = x_norm if x_norm is not None else Normalizer().fit(x)
        self.y_norm = y_norm if y_norm is not None else Normalizer().fit(y)
        self.x = self.x_norm.transform(x)
        self.y = self.y_norm.transform(y)
        self.params = np.stack([q, k, h], axis=1)  # kept for plotting

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def make_loaders(
    n_train_configs: int,
    n_val_configs: int,
    points_per_config: int = 32,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, tuple[Normalizer, Normalizer]]:
    """Train/val loaders; validation reuses the training normalizers (no leakage)."""
    train_ds = SlabDataset(n_train_configs, points_per_config, seed=seed)
    val_ds = SlabDataset(
        n_val_configs,
        points_per_config,
        seed=seed + 1,
        x_norm=train_ds.x_norm,
        y_norm=train_ds.y_norm,
    )
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, (train_ds.x_norm, train_ds.y_norm)


if __name__ == "__main__":
    train_loader, val_loader, _ = make_loaders(400, 100)
    x, y = next(iter(train_loader))
    print("batch:", tuple(x.shape), tuple(y.shape))
    print("feature mean/std:", x.mean(0).numpy().round(2), x.std(0).numpy().round(2))
