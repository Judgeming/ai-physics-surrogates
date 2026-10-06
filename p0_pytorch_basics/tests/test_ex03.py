import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from p0_pytorch_basics.physics import T_INF, temperature


def test_raw_features_and_target(load):
    ex = load("ex03_dataset")
    feats = ex.raw_features(0.0, 1e6, 10.0, 1000.0)
    np.testing.assert_allclose(feats, [[0.0, 6.0, 1.0, 3.0]])
    xi = np.array([-0.5, 0.0, 0.25])
    target = ex.raw_target(xi, 1e6, 10.0, 1000.0)
    assert target.shape == (3, 1)
    np.testing.assert_allclose(target[:, 0], np.log10(temperature(xi, 1e6, 10.0, 1000.0) - T_INF))


def test_normalizer_roundtrip_and_state(load):
    ex = load("ex03_dataset")
    x = torch.randn(500, 3, dtype=torch.float64) * torch.tensor([1.0, 10.0, 0.1], dtype=torch.float64)
    x = x + torch.tensor([5.0, -2.0, 100.0], dtype=torch.float64)
    norm = ex.Normalizer().fit(x)
    z = norm.transform(x)
    assert torch.allclose(z.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(z.std(0), torch.ones(3, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(norm.inverse_transform(z), x)
    restored = ex.Normalizer.from_state_dict(norm.state_dict())
    assert torch.allclose(restored.transform(x), z)


def test_dataset_items_and_determinism(load):
    ex = load("ex03_dataset")
    ds = ex.SlabDataset(10, points_per_config=8, seed=0)
    assert len(ds) == 80
    x, y = ds[0]
    assert x.shape == (4,) and y.shape == (1,)
    assert x.dtype == torch.float32
    assert torch.equal(ex.SlabDataset(10, 8, seed=0).x, ds.x)
    assert not torch.equal(ex.SlabDataset(10, 8, seed=1).x, ds.x)


def test_loaders_reuse_training_statistics(load):
    ex = load("ex03_dataset")
    train_loader, val_loader, (x_norm, y_norm) = ex.make_loaders(20, 5, points_per_config=8, batch_size=64)
    assert len(train_loader.dataset) == 160 and len(val_loader.dataset) == 40
    assert train_loader.dataset.x_norm is x_norm and val_loader.dataset.x_norm is x_norm
    assert val_loader.dataset.y_norm is y_norm
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert not isinstance(train_loader.sampler, SequentialSampler)
    train_x = train_loader.dataset.x
    assert torch.allclose(train_x.mean(0), torch.zeros(4), atol=1e-4)
