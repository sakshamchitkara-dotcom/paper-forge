"""Check the average-case error bounds of Halko et al. (2009) on synthetic matrices."""

import json

import numpy as np

from rsvd import eq1_9_bound, range_finder, thm1_1_bound, thm10_6_bound

rng = np.random.default_rng(0)
m, n, k, p, trials = 400, 300, 10, 5, 200
U, _ = np.linalg.qr(rng.standard_normal((m, n)))
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
sigma = 1.0 / np.sqrt(np.arange(1, n + 1))
A = U @ np.diag(sigma) @ V.T


def err(Q):
    return np.linalg.norm(A - Q @ (Q.T @ A), 2)


e0 = np.array([err(range_finder(A, k + p, 0, rng)) for _ in range(trials)])
e1 = np.array([err(range_finder(A, k + p, 1, rng)) for _ in range(trials)])
metrics = {
    "sigma_k_plus_1": float(sigma[k]),
    "mean_error_q0": float(e0.mean()),
    "mean_error_q1": float(e1.mean()),
    "mean_error_over_thm1_1_bound": float(e0.mean() / thm1_1_bound(sigma, k, p, m, n)),
    "mean_error_over_thm10_6_bound": float(e0.mean() / thm10_6_bound(sigma, k, p)),
    "max_error_over_eq1_9_bound": float(e0.max() / eq1_9_bound(sigma, k, p, m, n)),
    "power_iteration_reduces_error": bool(e1.mean() < e0.mean()),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
