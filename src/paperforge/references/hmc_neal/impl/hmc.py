"""Hamiltonian Monte Carlo and random-walk Metropolis (Neal, 2011, arXiv:1206.1901).

Hand-written reference implementation for paper-forge, from Sections 2.3 and 3.2.
"""

import numpy as np


def leapfrog(q, p, grad_U, eps, L):
    """L leapfrog steps (Neal eq. 2.28-2.30) for H(q, p) = U(q) + p.p/2."""
    q, p = q.copy(), p.copy()
    p -= eps * grad_U(q) / 2
    for i in range(L):
        q += eps * p
        if i != L - 1:
            p -= eps * grad_U(q)
    p -= eps * grad_U(q) / 2
    return q, -p  # negate momentum so the proposal is symmetric


def hmc_step(q, U, grad_U, eps, L, rng):
    """One HMC iteration (Neal Sec 3.2). Returns (new_q, accepted)."""
    p = rng.standard_normal(q.shape)
    q_new, p_new = leapfrog(q, p, grad_U, eps, L)
    current_H = U(q) + p @ p / 2
    proposed_H = U(q_new) + p_new @ p_new / 2
    if np.log(rng.uniform()) < current_H - proposed_H:
        return q_new, True
    return q, False


def rwm_step(q, U, sd, rng):
    """One random-walk Metropolis update with isotropic Gaussian proposal."""
    q_new = q + sd * rng.standard_normal(q.shape)
    if np.log(rng.uniform()) < U(q) - U(q_new):
        return q_new, True
    return q, False


def gaussian_target(sds):
    """Independent zero-mean Gaussian with the given standard deviations."""
    prec = 1.0 / np.asarray(sds) ** 2
    return (lambda q: 0.5 * np.sum(prec * q * q)), (lambda q: prec * q)
