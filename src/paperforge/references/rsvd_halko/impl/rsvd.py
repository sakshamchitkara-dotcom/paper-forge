"""Randomized range finder and SVD (Halko, Martinsson & Tropp, arXiv:0909.4061).

Hand-written reference implementation for paper-forge, from Algorithms 4.1, 4.4 and 5.1.
"""

import numpy as np


def range_finder(A, ell, q=0, rng=None):
    """Orthonormal Q (m x ell) approximating range(A); q = power iterations (Alg 4.1 / 4.4)."""
    rng = rng or np.random.default_rng()
    omega = rng.standard_normal((A.shape[1], ell))
    Q, _ = np.linalg.qr(A @ omega)
    for _ in range(q):  # Algorithm 4.4: orthonormalise between every application
        W, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ W)
    return Q


def rsvd(A, k, p=5, q=0, rng=None):
    """Rank-k approximate SVD via Algorithm 5.1 (direct SVD of B = Q^T A)."""
    Q = range_finder(A, k + p, q, rng)
    Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def thm1_1_bound(sigma, k, p, m, n):
    return (1 + 4 * np.sqrt(k + p) / (p - 1) * np.sqrt(min(m, n))) * sigma[k]


def thm10_6_bound(sigma, k, p):
    return (1 + np.sqrt(k / (p - 1))) * sigma[k] + np.e * np.sqrt(k + p) / p * np.sqrt(np.sum(sigma[k:] ** 2))


def eq1_9_bound(sigma, k, p, m, n):
    return (1 + 11 * np.sqrt(k + p) * np.sqrt(min(m, n))) * sigma[k]
