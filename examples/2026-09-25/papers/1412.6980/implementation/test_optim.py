import numpy as np

from optim import Adagrad, Adam, NesterovSGD


def test_adam_first_step_is_alpha_times_sign():
    # With bias correction, the first update has magnitude ~alpha regardless of gradient scale.
    theta = np.zeros(3)
    out = Adam(alpha=0.1).step(theta, np.array([1e-3, -5.0, 200.0]))
    assert np.allclose(out, -0.1 * np.sign([1e-3, -5.0, 200.0]), atol=1e-4)


def test_optimizers_minimise_quadratic():
    for opt in (Adam(0.1), Adagrad(0.5), NesterovSGD(0.05)):
        theta = np.array([3.0, -2.0])
        for _ in range(500):
            at = opt.lookahead(theta) if hasattr(opt, "lookahead") else theta
            theta = opt.step(theta, 2 * at)
        assert np.linalg.norm(theta) < 0.05, type(opt).__name__
