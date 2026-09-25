import numpy as np

from rsvd import range_finder, rsvd


def test_exact_for_low_rank_matrix():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 5)) @ rng.standard_normal((5, 40))
    U, s, Vt = rsvd(A, k=5, p=5, rng=rng)
    assert np.allclose(U @ np.diag(s) @ Vt, A, atol=1e-8)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False)[:5])


def test_range_finder_orthonormal_with_power_iterations():
    rng = np.random.default_rng(1)
    Q = range_finder(rng.standard_normal((50, 30)), 8, q=2, rng=rng)
    assert Q.shape == (50, 8) and np.allclose(Q.T @ Q, np.eye(8))
