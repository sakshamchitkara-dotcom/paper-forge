# Changelog

## 0.2.0 — 2026-09-25

### Added
- arXiv listing fallback: when `export.arxiv.org/api/query` still fails after its retries
  (e.g. the intermittent 406s), screening reads `rss.arxiv.org/atom/<category>` through the
  same rate-limited client and keeps only first announcements (new/cross, v1).
- Hand-written reference for Meyer, Musco, Musco & Woodruff, *Hutch++: Optimal Stochastic
  Trace Estimation* (arXiv:2010.09649): Hutchinson, Hutch++, NA-Hutch++ and Subspace
  Projection on the Sec 6.1 power-law spectra, seven cited claims, fidelity notes. A docker
  sandbox run matched all seven.

### Changed
- The User-Agent carries the package version and, if `FORGE_CONTACT` is set, a `mailto:`
  contact. The daily workflow passes the `FORGE_CONTACT` repository variable through.
- Retries on 429/503 wait at least the server's numeric `Retry-After` (capped at 120 s).

## 0.1.0 — 2026-09-25

- First release: arXiv + Hugging Face screening with a transparent rubric, heuristic or
  Claude analysis, scripted (hand-written references for Neal, Halko et al., Kingma & Ba) and
  Claude reimplementation backends, docker/subprocess sandbox, daily report, leaderboard and
  GitHub Pages site.
