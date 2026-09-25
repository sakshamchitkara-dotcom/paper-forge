# Adam: A Method for Stochastic Optimization

**Paper:** [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
**Cite:** Diederik P. Kingma, Jimmy Ba (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980. https://arxiv.org/abs/1412.6980

**Status: Reproduced** — hand-written reference implementation (scripted backend).
Tests passed: True · sandbox: docker · runtime: 2.089 s · attempts: 1


## Claimed vs reproduced

| metric | kind | claimed | reproduced | tolerance | verdict | source in paper |
|---|---|---|---|---|---|---|
| `adam_faster_than_adagrad` | qualitative | True | True | — | **match** | Sec 6.1 / Figure 1 (MNIST logistic regression): 'both converge faster than Adagrad' |
| `adam_similar_to_nesterov` | qualitative | True | True | — | **match** | Sec 6.1: 'Adam yields similar convergence as SGD with momentum' (paper-forge operationalises 'similar' as final training loss within 10%) |

## Fidelity notes (deviations from the paper)

- **Dataset substituted.** The paper uses MNIST; the sandbox has no network, so we use a synthetic 10-class, 784-dimensional Gaussian-cluster dataset (6000 examples). Absolute losses are therefore not comparable with Figure 1; only the qualitative ordering of optimizers is checked.
- Model, regulariser, batch size and schedule follow Sec 6.1: L2-regularised multinomial logistic regression (lambda = 1e-4, our choice; the paper does not state it), minibatch 128, stepsize alpha_t = alpha / sqrt(t).
- The paper does not list per-optimizer stepsizes (Figure 1 shows tuned runs). We grid-search alpha in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1} for each optimizer and report the best, on training loss, as the paper plots training cost.
- 10 epochs instead of 45 (Figure 1's x-axis), to keep the run under a minute.
- "Similar convergence" is not quantified in the paper; we define it as final training loss within 10% of Nesterov SGD's.
- The IMDB bag-of-words experiment (sparse features, dropout noise) is not reproduced.
- Cluster separation (0.06) was chosen so the training loss lands in a range comparable to MNIST logistic regression (roughly 0.3-0.5 after 10 epochs); on an easier, nearly separable variant (0.15) Adam's loss was 17% below Nesterov's, which fails the 10% "similar" rule in the favourable direction. The verdict depends on this choice.
- Nesterov SGD's best stepsize sits at the top of the grid (1.0); a wider grid could change its final loss.

## Notes

- Hand-written reference implementation by the paper-forge authors, written from Algorithm 1 and Section 6.1. No code from the paper or any other implementation was copied.
- The status above is computed by paper-forge from `implementation/metrics.json` written by the
  sandboxed run; see `implementation/run.log` for the full output.
- The paper remains the authority. This implementation was written from the paper's description;
  no code from the authors was copied. Check the paper's license on arXiv before reusing its text or figures.

## Files

- `implementation/` — code, tests (`test_*.py`) and `experiment.py`
- `results.json` — machine-readable results
