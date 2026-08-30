"""Hutch++ and baselines for stochastic trace estimation (Meyer, Musco, Musco & Woodruff,
arXiv:2010.09649).

Hand-written reference implementation for paper-forge, from Algorithms 1 and 2 and the
method list in Sec 6. Every estimator sees A only through `matvec(X) = A @ X` and uses
exactly `m` matrix-vector products.
"""

import numpy as np


def signs(rng, d, k):
    """d x k matrix of i.i.d. +-1 entries."""
    return rng.choice([-1.0, 1.0], size=(d, k))


def hutchinson(matvec, d, m, rng):
    """Hutchinson's estimator: (1/m) tr(G^T A G) with +-1 vectors."""
    G = signs(rng, d, m)
    return float(np.sum(G * matvec(G)) / m)


def hutchpp(matvec, d, m, rng):
    """Algorithm 1: tr(Q^T A Q) + (3/m) tr(G^T (I-QQ^T) A (I-QQ^T) G), m/3 queries per step."""
    k = m // 3
    S, G = signs(rng, d, k), signs(rng, d, m - 2 * k)
    Q, _ = np.linalg.qr(matvec(S))
    G_perp = G - Q @ (Q.T @ G)
    return float(np.trace(Q.T @ matvec(Q)) + np.sum(G_perp * matvec(G_perp)) / G.shape[1])


def na_hutchpp(matvec, d, m, rng, c1=0.25, c2=0.5):
    """Algorithm 2 (non-adaptive): tr((S^T Z)^+ (W^T Z)) + (1/(c3 m)) [tr(G^T A G) - tr(G^T Z (S^T Z)^+ W^T G)]."""
    k1, k2 = int(c1 * m), int(c2 * m)
    S, R, G = signs(rng, d, k1), signs(rng, d, k2), signs(rng, d, m - k1 - k2)
    Z, W = matvec(R), matvec(S)  # all queries fixed up front: one batch in principle
    AG = matvec(G)
    P = np.linalg.pinv(S.T @ Z)
    return float(np.trace(P @ (W.T @ Z)) + (np.sum(G * AG) - np.trace(G.T @ Z @ P @ (W.T @ G))) / G.shape[1])


def subspace_projection(matvec, d, m, rng):
    """[SAI17] with q = 1 as in Sec 6: Q = orth(A S), return tr(Q^T A Q); k(q+1) = m queries."""
    S = signs(rng, d, m // 2)
    Q, _ = np.linalg.qr(matvec(S))
    return float(np.trace(Q.T @ matvec(Q)))


def power_law_matrix(d, c, rng):
    """A = Q^T diag(i^-c) Q with Q a random orthogonal matrix (Sec 6.1). Returns (A, eigenvalues)."""
    lam = np.arange(1, d + 1, dtype=float) ** -c
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q.T * lam) @ Q, lam
