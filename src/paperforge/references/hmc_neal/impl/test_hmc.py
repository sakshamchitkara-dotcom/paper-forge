import numpy as np

from hmc import gaussian_target, hmc_step, leapfrog


def test_leapfrog_is_reversible_and_nearly_conserves_energy():
    U, g = gaussian_target(np.array([1.0, 2.0]))
    q0, p0 = np.array([0.3, -1.0]), np.array([0.5, 0.2])
    q1, p1 = leapfrog(q0, p0, g, 0.05, 40)
    q2, p2 = leapfrog(q1, p1, g, 0.05, 40)
    assert np.allclose(q2, q0) and np.allclose(p2, p0)  # momentum flip makes it an involution
    H = lambda q, p: U(q) + p @ p / 2
    assert abs(H(q1, p1) - H(q0, p0)) < 1e-2


def test_hmc_samples_standard_normal():
    U, g = gaussian_target(np.ones(3))
    rng = np.random.default_rng(1)
    q, xs = np.zeros(3), []
    for _ in range(3000):
        q, _ = hmc_step(q, U, g, 0.2, 10, rng)
        xs.append(q)
    xs = np.array(xs)
    assert np.all(abs(xs.mean(0)) < 0.1) and np.all(abs(xs.std(0) - 1) < 0.1)
