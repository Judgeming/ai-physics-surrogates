import pytest


@pytest.mark.parametrize("fit_name", ["fit_manual", "fit_adam"])
def test_recovers_parameters_within_5_percent(load, fit_name):
    ex = load("ex01_autograd_fit")
    xi, t_meas = ex.make_measurements()
    q, h = getattr(ex, fit_name)(xi, t_meas)
    assert abs(q / ex.TRUE_Q - 1) < 0.05
    assert abs(h / ex.TRUE_H - 1) < 0.05


def test_both_optimizers_reach_the_same_least_squares_optimum(load):
    ex = load("ex01_autograd_fit")
    xi, t_meas = ex.make_measurements()
    (q1, h1), (q2, h2) = ex.fit_manual(xi, t_meas), ex.fit_adam(xi, t_meas)
    assert q1 == pytest.approx(q2, rel=5e-3)
    assert h1 == pytest.approx(h2, rel=5e-3)
