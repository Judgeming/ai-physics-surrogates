import numpy as np

from p0_pytorch_basics.physics import L


def test_dimensionless_target_matches_closed_form(load):
    ex = load("ex05_ood_dimensionless")
    xi, q, k, h = np.array([-0.3, 0.0, 0.4]), 2e6, 20.0, 1500.0
    bi = h * L / k
    theta = (0.25 - xi**2) / 2 + 1 / (2 * bi)
    np.testing.assert_allclose(ex.dimensionless_target(xi, q, k, h)[:, 0], np.log10(theta))
    np.testing.assert_allclose(ex.dimensionless_features(xi, q, k, h)[:, 1], np.log10(bi))


def test_features_do_not_depend_on_q(load):
    ex = load("ex05_ood_dimensionless")
    xi = np.linspace(-0.5, 0.5, 5)
    a = ex.dimensionless_features(xi, 1e5, 20.0, 1500.0)
    b = ex.dimensionless_features(xi, 1e7, 20.0, 1500.0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(
        ex.dimensionless_target(xi, 1e5, 20.0, 1500.0), ex.dimensionless_target(xi, 1e7, 20.0, 1500.0), rtol=1e-12
    )


def test_compact_training_loop_learns_and_is_q_invariant(load):
    ex = load("ex05_ood_dimensionless")
    train, val, test = ex.make_split(60, 16, seed=0), ex.make_split(20, 16, seed=1), ex.make_split(30, 16, seed=2)
    feats, target = ex.dimensionless_features, ex.dimensionless_target
    model, xn, yn = ex.train_on_arrays(feats(**train), target(**train), feats(**val), target(**val), epochs=80)
    err = ex.rel_l2(ex.predict_delta_t(model, xn, yn, test, dimensionless=True), test)
    assert err < 0.05
    hot = {**test, "q": test["q"] * 10}
    err_hot = ex.rel_l2(ex.predict_delta_t(model, xn, yn, hot, dimensionless=True), hot)
    assert abs(err_hot - err) < 1e-9
