"""Reproduce Meyer et al. (2021) Sec 6.1: trace estimators on power-law spectra."""

import json

import numpy as np

from hutchpp import hutchinson, hutchpp, na_hutchpp, power_law_matrix, subspace_projection

rng = np.random.default_rng(0)
D, TRIALS, MS, CS = 1000, 100, [12, 24, 48, 96, 192, 384], [2.0, 1.5, 1.0, 0.5]
METHODS = {"hutchinson": hutchinson, "hutchpp": hutchpp, "na_hutchpp": na_hutchpp,
           "subspace": subspace_projection}


def fro_over_trace(d, c):
    lam = np.arange(1, d + 1, dtype=float) ** -c
    return float(np.sqrt(np.sum(lam ** 2)) / np.sum(lam))


def slope(errs):
    return float(np.polyfit(np.log(MS), np.log(errs), 1)[0])


med = {}  # (method, c) -> median relative error per m
for c in CS:
    A, lam = power_law_matrix(D, c, rng)
    tr = lam.sum()
    for name, f in METHODS.items():
        med[name, c] = [float(np.median([abs(f(lambda X: A @ X, D, m, rng) - tr) / tr for _ in range(TRIALS)]))
                        for m in MS]

last = {k: v[-1] for k, v in med.items()}  # comparisons at the largest m
metrics = {
    "fro_over_trace_c2": fro_over_trace(5000, 2.0),
    "fro_over_trace_c05": fro_over_trace(5000, 0.5),
    "hutchinson_loglog_slope_c05": slope(med["hutchinson", 0.5]),
    "hutchpp_loglog_slope_c2": slope(med["hutchpp", 2.0]),
    "hutchpp_beats_hutchinson_c2": bool(last["hutchpp", 2.0] < last["hutchinson", 2.0]),
    "subspace_beats_hutchinson_only_at_c2": bool(
        last["subspace", 2.0] < last["hutchinson", 2.0]
        and all(last["subspace", c] >= last["hutchinson", c] for c in CS if c != 2.0)),
    "hutchpp_beats_na_hutchpp_all_c": bool(all(last["hutchpp", c] < last["na_hutchpp", c] for c in CS)),
    "m_grid": MS,
    "median_rel_error": {f"{n}_c{c}": v for (n, c), v in med.items()},
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
