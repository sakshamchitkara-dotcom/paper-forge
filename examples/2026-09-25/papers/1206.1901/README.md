# MCMC using Hamiltonian dynamics

**Paper:** [https://arxiv.org/abs/1206.1901](https://arxiv.org/abs/1206.1901)
**Cite:** Radford M. Neal (2012). MCMC using Hamiltonian dynamics. arXiv:1206.1901. https://arxiv.org/abs/1206.1901

**Status: Reproduced** — hand-written reference implementation (scripted backend).
Tests passed: True · sandbox: docker · runtime: 1.157 s · attempts: 1


## Claimed vs reproduced

| metric | kind | claimed | reproduced | tolerance | verdict | source in paper |
|---|---|---|---|---|---|---|
| `hmc_rejection_rate` | numeric | 0.13 | 0.148 | 0.03 | **match** | Sec 3.3, 100-dimensional Gaussian: 'The rejection rate was 0.13 for HMC' |
| `rwm_rejection_rate` | numeric | 0.75 | 0.7502 | 0.03 | **match** | Sec 3.3: '... and 0.75 for random-walk Metropolis' |
| `hmc_better_std_estimate_for_largest_sd` | qualitative | True | True | — | **match** | Sec 3.3 / Figure 7: HMC estimates of the largest-sd coordinate are far more accurate than random-walk Metropolis at equal computation |

## Fidelity notes (deviations from the paper)

- Target, kinetic energy K(p) = p^T p / 2, trajectory length L = 150, stepsize eps ~ U(0.0104, 0.0156) and random-walk proposal sd ~ U(0.0176, 0.0264) follow Sec 3.3 exactly.
- Neal ran 1000 HMC iterations and gave random-walk Metropolis 150 updates per iteration to equalise computation; we do the same (1000 x 150 RWM proposals).
- Initial state: Neal does not state it; we start from a draw of the target distribution's scale (q = 0 would also work). Rejection rates are insensitive to this after burn-in; we discard no samples, as in the paper's plots.
- Random seed is fixed (0). Measured across seeds 0-5, the HMC rejection rate ranged 0.119-0.148 (binomial sd over 1000 iterations is about 0.011) and the RWM rate 0.7497-0.7507; the 0.03 tolerance is roughly 3 sd.
- The qualitative claim is checked by comparing the relative error of the estimated standard deviation of the coordinate with sd = 1.00.

## Notes

- Hand-written reference implementation by the paper-forge authors, written from the paper's description (Sections 2.3, 3.2, 3.3). No code from the paper or any other implementation was copied.
- The status above is computed by paper-forge from `implementation/metrics.json` written by the
  sandboxed run; see `implementation/run.log` for the full output.
- The paper remains the authority. This implementation was written from the paper's description;
  no code from the authors was copied. Check the paper's license on arXiv before reusing its text or figures.

## Files

- `implementation/` — code, tests (`test_*.py`) and `experiment.py`
- `results.json` — machine-readable results
