from pathlib import Path

import numpy as np
import torch

from p0_pytorch_basics.physics import temperature


def _quick_fit(ex, out_dir):
    cfg = ex.TrainConfig(
        n_train_configs=200, n_val_configs=50, max_epochs=120, device="cpu", tensorboard=False, out_dir=str(out_dir)
    )
    return ex.fit(cfg)


def test_pick_device(load):
    ex = load("ex04_train")
    assert ex.pick_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert ex.pick_device("auto").type == expected


def test_training_reaches_2_percent_and_checkpoint_roundtrips(load, tmp_path):
    ex = load("ex04_train")
    result = _quick_fit(ex, tmp_path)
    assert result["best_val"]["rel_l2"] < 0.02
    assert Path(result["checkpoint"]).exists()
    assert (tmp_path / "history.json").exists()

    model, x_norm, y_norm, cfg = ex.load_checkpoint(result["checkpoint"])
    assert cfg.n_train_configs == 200
    xi = np.linspace(-0.5, 0.5, 11)
    pred = ex.predict_temperature(model, x_norm, y_norm, xi, 1e6, 20.0, 1000.0)
    true = temperature(xi, 1e6, 20.0, 1000.0)
    assert pred.shape == (11,)
    np.testing.assert_allclose(pred, true, atol=0.5)  # deg C; true rise here is ~5.6 deg C
