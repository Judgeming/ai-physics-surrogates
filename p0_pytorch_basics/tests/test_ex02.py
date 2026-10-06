import pytest
import torch
from torch import nn


def test_forward_shape(load):
    ex = load("ex02_mlp_module")
    model = ex.MLP(4, 1, hidden=(32, 32))
    assert isinstance(model, nn.Module)
    assert model(torch.randn(10, 4)).shape == (10, 1)


def test_parameter_count(load):
    ex = load("ex02_mlp_module")
    # (4*64 + 64) + (64*64 + 64) + (64*1 + 1)
    assert ex.count_parameters(ex.MLP(4, 1, hidden=(64, 64))) == 4545


def test_activation_choice(load):
    ex = load("ex02_mlp_module")
    model = ex.MLP(4, 1, hidden=(8,), activation="tanh")
    assert any(isinstance(m, nn.Tanh) for m in model.modules())
    with pytest.raises(ValueError):
        ex.MLP(4, 1, activation="not-an-activation")


def test_gradients_reach_every_parameter(load):
    ex = load("ex02_mlp_module")
    model = ex.MLP(3, 2, hidden=(16, 16))
    model(torch.randn(5, 3)).pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name
