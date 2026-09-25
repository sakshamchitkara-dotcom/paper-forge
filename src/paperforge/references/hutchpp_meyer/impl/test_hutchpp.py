import numpy as np

from hutchpp import hutchinson, hutchpp, na_hutchpp, power_law_matrix, subspace_projection


def test_exact_when_rank_fits_in_the_sketch():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((200, 5))
    A = B @ B.T  # rank 5: Q from 10 queries spans range(A), residual is zero
    tr = np.trace(A)
    assert abs(hutchpp(lambda X: A @ X, 200, 30, rng) - tr) < 1e-8 * tr
    assert abs(subspace_projection(lambda X: A @ X, 200, 20, rng) - tr) < 1e-8 * tr
    assert abs(na_hutchpp(lambda X: A @ X, 200, 40, rng) - tr) < 1e-6 * tr


def test_estimators_unbiased_and_use_m_queries():
    rng = np.random.default_rng(1)
    A, lam = power_law_matrix(100, 1.0, rng)
    for f in (hutchinson, hutchpp, na_hutchpp):
        used = []
        mv = lambda X: used.append(X.shape[1]) or A @ X
        est = np.mean([f(mv, 100, 24, rng) for _ in range(300)])
        assert abs(est - lam.sum()) < 0.05 * lam.sum()
        assert sum(used) == 24 * 300
