# Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions

**Paper:** [https://arxiv.org/abs/0909.4061](https://arxiv.org/abs/0909.4061)
**Cite:** Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp (2009). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. arXiv:0909.4061. https://arxiv.org/abs/0909.4061

**Status: Reproduced** — hand-written reference implementation (scripted backend).
Tests passed: True · sandbox: docker · runtime: 4.147 s · attempts: 1


## Claimed vs reproduced

| metric | kind | claimed | reproduced | tolerance | verdict | source in paper |
|---|---|---|---|---|---|---|
| `mean_error_over_thm1_1_bound` | at_most | 1 | 0.02759 | — | **match** | Theorem 1.1, eq. (1.8): E||A - QQ*A|| <= [1 + 4 sqrt(k+p)/(p-1) sqrt(min(m,n))] sigma_{k+1} |
| `mean_error_over_thm10_6_bound` | at_most | 1 | 0.1222 | — | **match** | Theorem 10.6 (average spectral error): E||(I-P_Y)A|| <= (1 + sqrt(k/(p-1))) sigma_{k+1} + e sqrt(k+p)/p (sum_{j>k} sigma_j^2)^{1/2} |
| `max_error_over_eq1_9_bound` | at_most | 1 | 0.003241 | — | **match** | eq. (1.9): ||A - QQ*A|| <= [1 + 11 sqrt(k+p) sqrt(min(m,n))] sigma_{k+1} with probability >= 1 - 6 p^-p |
| `power_iteration_reduces_error` | qualitative | True | True | — | **match** | Sec 4.5 / 10.4: the power scheme (Algorithm 4.3) with q >= 1 reduces the error when singular values decay slowly |

## Fidelity notes (deviations from the paper)

- Test matrices are synthetic (the paper's numerical section uses other matrices): A = U diag(s) V^T, 400 x 300, random orthogonal U, V, with slowly decaying singular values s_j = 1/j^0.5 so that the bounds are not trivially loose.
- k = 10, p = 5 (the paper's suggested oversampling), Gaussian test matrix, 200 independent trials; "mean error" is the Monte-Carlo estimate of the expectation in Theorems 1.1 / 10.6, so it carries sampling noise (the ratios are far from 1, so this does not affect the verdict).
- Spectral norms are computed exactly with numpy (the paper's posterior estimator, Sec 4.3, is not used).
- Power iteration uses Algorithm 4.4 (re-orthonormalised subspace iteration) rather than plain Algorithm 4.3, as the paper recommends for floating-point stability.

## Notes

- Hand-written reference implementation by the paper-forge authors, written from Algorithms 4.1, 4.3 and 5.1 and Theorems 1.1 / 10.6. No code from the paper or any other implementation was copied.
- The status above is computed by paper-forge from `implementation/metrics.json` written by the
  sandboxed run; see `implementation/run.log` for the full output.
- The paper remains the authority. This implementation was written from the paper's description;
  no code from the authors was copied. Check the paper's license on arXiv before reusing its text or figures.

## Files

- `implementation/` — code, tests (`test_*.py`) and `experiment.py`
- `results.json` — machine-readable results
