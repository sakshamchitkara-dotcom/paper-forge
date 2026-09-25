"""Reproduce Neal (2011) Sec 3.3: HMC vs random-walk Metropolis on a 100-d Gaussian."""

import json

import numpy as np

from hmc import gaussian_target, hmc_step, rwm_step

rng = np.random.default_rng(0)
sds = np.arange(1, 101) / 100.0  # 0.01, 0.02, ..., 1.00
U, grad_U = gaussian_target(sds)
N_ITER, L = 1000, 150

q = rng.standard_normal(100) * sds
hmc_samples, hmc_acc = [], 0
for _ in range(N_ITER):
    q, acc = hmc_step(q, U, grad_U, rng.uniform(0.0104, 0.0156), L, rng)
    hmc_acc += acc
    hmc_samples.append(q)

q = rng.standard_normal(100) * sds
rwm_samples, rwm_acc = [], 0
for _ in range(N_ITER):
    for _ in range(L):  # 150 RWM updates per iteration = equal computation (Sec 3.3)
        q, acc = rwm_step(q, U, rng.uniform(0.0176, 0.0264), rng)
        rwm_acc += acc
    rwm_samples.append(q)

hmc_samples, rwm_samples = np.array(hmc_samples), np.array(rwm_samples)
hmc_sd_err = abs(hmc_samples[:, -1].std() - 1.0)
rwm_sd_err = abs(rwm_samples[:, -1].std() - 1.0)
metrics = {
    "hmc_rejection_rate": 1 - hmc_acc / N_ITER,
    "rwm_rejection_rate": 1 - rwm_acc / (N_ITER * L),
    "hmc_sd_estimate_largest": float(hmc_samples[:, -1].std()),
    "rwm_sd_estimate_largest": float(rwm_samples[:, -1].std()),
    "hmc_better_std_estimate_for_largest_sd": bool(hmc_sd_err < rwm_sd_err),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
