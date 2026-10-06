"""Exercise 3: Dataset, normalization, and DataLoaders (~1-1.5 h).

Each sample is one (position, slab configuration) pair:
    features = [xi, log10 q, log10 k, log10 h]
    target   = log10(T - T_INF)
The temperature rise spans ~0.1 to ~150 deg C across the training ranges, so
the target is learned in log space and both sides are standardized.

Check your work:  uv run pytest -k ex03
Run it:           uv run python -m p0_pytorch_basics.ex03_dataset
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .physics import H_RANGE, K_RANGE, Q_RANGE, T_INF, sample_params, temperature  # noqa: F401

FEATURE_NAMES = ("xi", "log10_q", "log10_k", "log10_h")


def raw_features(xi, q, k, h) -> np.ndarray:
    """Stack broadcast inputs into an (N, 4) float64 array [xi, log10 q, log10 k, log10 h].

    Inputs may be scalars or arrays of broadcastable shapes. Hint: np.broadcast_arrays,
    np.stack(..., axis=-1), then reshape(-1, 4).
    """
    xi, log10_q, log10_k, log10_h = np.broad
    raise NotImplementedError("TODO: exercise 3 (raw_features)")


def raw_target(xi, q, k, h) -> np.ndarray:
    """(N, 1) array of log10(temperature(...) - T_INF)."""
    raise NotImplementedError("TODO: exercise 3 (raw_target)")


class Normalizer:
    """Per-column standardization z = (x - mean) / std.

    Fit it on training data only; validation and test data reuse the training
    statistics. Store mean/std as tensors so the object can go into a checkpoint.
    """

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> None:
        self.mean = mean
        self.std = std

    def fit(self, x: torch.Tensor) -> Normalizer:
        """Column mean/std over dim 0 (clamp std to >= 1e-8); return self for chaining."""
        raise NotImplementedError("TODO: exercise 3 (Normalizer.fit)")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Hint: move mean/std to x.device so this also works on GPU tensors."""
        raise NotImplementedError("TODO: exercise 3 (Normalizer.transform)")

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: exercise 3 (Normalizer.inverse_transform)")

    def state_dict(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("TODO: exercise 3 (Normalizer.state_dict)")

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> Normalizer:
        raise NotImplementedError("TODO: exercise 3 (Normalizer.from_state_dict)")


class SlabDataset(Dataset):
    """n_configs random slabs, each sampled at points_per_config random positions.

    TODO in __init__:
    1. rng = np.random.default_rng(seed); draw (q, k, h) with sample_params(...).
    2. xi = rng.uniform(-0.5, 0.5, size=(n_configs, points_per_config)).
    3. Build float32 tensors x (N, 4) and y (N, 1) with raw_features / raw_target
       (give q, k, h a trailing axis so they broadcast against xi).
    4. Use the normalizers passed in, or fit new ones on this data if None.
       Keep them as self.x_norm / self.y_norm.
    5. Store the normalized tensors as self.x / self.y, and
       self.params = np.stack([q, k, h], axis=1) for plotting later.
    """

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
        raise NotImplementedError("TODO: exercise 3 (SlabDataset.__init__)")

    def __len__(self) -> int:
        raise NotImplementedError("TODO: exercise 3 (SlabDataset.__len__)")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: exercise 3 (SlabDataset.__getitem__)")


def make_loaders(
    n_train_configs: int,
    n_val_configs: int,
    points_per_config: int = 32,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, tuple[Normalizer, Normalizer]]:
    """Train/val loaders plus the training normalizers.

    TODO:
    - train dataset with `seed`; val dataset with `seed + 1` that REUSES the
      training normalizers (fitting new ones on validation data is leakage)
    - train loader shuffles (pass generator=torch.Generator().manual_seed(seed) for
      reproducibility); val loader does not
    """
    raise NotImplementedError("TODO: exercise 3 (make_loaders)")


if __name__ == "__main__":
    train_loader, val_loader, _ = make_loaders(400, 100)
    x, y = next(iter(train_loader))
    print("batch:", tuple(x.shape), tuple(y.shape))
    print("feature mean/std:", x.mean(0).numpy().round(2), x.std(0).numpy().round(2))
