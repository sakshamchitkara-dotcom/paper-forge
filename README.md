# paper-forge

A daily research screener. It pulls new arXiv papers, scores how practical each one is to
reproduce on a laptop, and for papers it picks, builds a small working implementation. It
runs that implementation in a sandbox and writes a report that compares what the paper
claimed with what the run actually produced.

```
arXiv API ─┐                       ┌─ heuristic rubric (always)      ┌─ scripted: hand-written references (offline)
HF daily ──┼─ dedupe (sqlite) ─ score ─ full text ─ analysis ────────┼─ claude: agent loop writes repo + tests
S2 (opt) ──┘                       └─ Claude JSON analysis (if key)  └─→ sandbox (docker --network none | subprocess)
                                                                          → results.json → daily report / leaderboard / site
```

## Integrity rules (enforced in code)

- **Only a run can mark a reproduction as successful.** `results.py` decides the status from the
  `metrics.json` that the sandboxed `experiment.py` wrote. A crash, a timeout or missing metrics
  means `failed`. Failing unit tests cap the status at `partial`. A stale metrics file from an
  earlier attempt is deleted before each run.
- Statuses: `reproduced`, `partial`, `not_reproduced`, `unverified` (it ran, but there was
  nothing to compare against), and `failed`.
- Every paper page includes the citation, the arXiv link, the backend label (hand-written or
  generated), the claimed-vs-reproduced table with the paper section each claim comes from, and
  **fidelity notes** that list each deviation from the paper.
- Implementations are written from the paper's description. The Claude backend is told not to
  copy the authors' code. Check the paper's license on arXiv before reusing its text or figures.

## Install

```bash
pip install -e ".[claude,dev]"     # Python 3.10+
docker build -t paper-forge-sandbox:latest docker/   # optional: stronger sandbox
```

## Usage

```bash
forge screen --categories cs.LG,stat.ML,cs.IR --max-results 40 --top-k 5   # fetch + score + analyze
forge references                                   # list the hand-written reference implementations
forge reimplement 1206.1901 --backend scripted     # run one reference in the sandbox
forge reimplement 2609.12345 --backend claude      # generate + run (needs ANTHROPIC_API_KEY; costs tokens)
forge daily                                        # screen + rebuild site/  (screen-only)
forge daily --reimplement --backend scripted       # + run the reference demo
forge report                                       # rebuild site/ from data/forge.db
```

Configuration lives in `forge.toml`. Anything you leave out falls back to the defaults in
`src/paperforge/config.py`. Outputs:

```
site/index.html, leaderboard.{html,md}, daily/<day>.{html,md}
site/papers/<arxiv id>/{README.md,index.html,results.json,implementation/{*.py,metrics.json,run.log}}
data/forge.db       # sqlite: dedupe across days + every run
```

## Sources

| source | used for | notes |
|---|---|---|
| `export.arxiv.org/api/query` | new submissions per category, lookups by id | at least 3 s between calls (arXiv's guidance); exponential backoff on 406/429/5xx |
| `huggingface.co/api/daily_papers` | upvotes, `githubRepo` code links, extra candidates | open endpoint, no key |
| Semantic Scholar batch API | citation counts (velocity) | off by default; unauthenticated calls get 429s quickly. Set `SEMANTIC_SCHOLAR_API_KEY` |
| Papers with Code | — | its API is gone (paperswithcode.com redirects to HF trending). Code links come from HF and from GitHub URLs in the abstract/comment |
| arXiv HTML / PDF | full text for the top-k | LaTeXML HTML first (equations kept as LaTeX), pypdf fallback |

## Screening rubric

Each criterion scores from 0 to 1 and records the exact text that triggered it. The total score
is the weighted mean × 100. Weights are set in `screening.weights`.

| criterion | weight | raises the score | lowers the score |
|---|---:|---|---|
| compute | 3.0 | closed-form, convex, toy/synthetic, CPU, numpy | billion-scale, GPU-hours/A100/H100/TPU, pretraining, LLM, diffusion |
| data | 2.0 | MNIST/CIFAR-10/UCI/synthetic/simulated | ImageNet/Common Crawl/web-scale, proprietary/in-house data |
| clarity | 2.0 | "Algorithm N"/pseudocode, update rule, theorem/bound, "simple" | survey/position paper, agentic frameworks (title/abstract only) |
| code | 1.0 | public GitHub repo (HF or abstract) | — |
| interest | 1.5 | configured keywords, HF upvotes, citations per month | — |

The top-k papers also get a structured analysis: method summary, key equations, datasets,
claimed metrics, compute estimate, laptop feasibility and a minimal reproduction plan. If
`ANTHROPIC_API_KEY` is set, `claude-opus-5-5` fills a JSON schema through structured outputs,
with effort set explicitly. Without a key, or when the API refuses or errors, regex heuristics
fill the same schema. The report shows which source produced each analysis.

## Reimplementation backends

**scripted (offline, no key).** Four *hand-written reference implementations* of real, classic
papers, written by the paper-forge authors and labeled that way everywhere. Their claims were
transcribed from the paper PDFs, and each one cites its section:

| paper | what is checked |
|---|---|
| Neal, *MCMC using Hamiltonian dynamics* (arXiv:1206.1901) | Sec 3.3: HMC rejection rate 0.13 and random-walk Metropolis 0.75 on the 100-d Gaussian; HMC gives the better sd estimate |
| Halko, Martinsson & Tropp, *Finding structure with randomness* (arXiv:0909.4061) | the Theorem 1.1 and Theorem 10.6 average-error bounds, the (1.9) tail bound, and the power scheme reducing the error |
| Kingma & Ba, *Adam* (arXiv:1412.6980) | Sec 6.1 qualitative claims: Adam beats Adagrad and is similar to Nesterov SGD (synthetic stand-in for MNIST, disclosed) |
| Meyer, Musco, Musco & Woodruff, *Hutch++* (arXiv:2010.09649) | Sec 6.1: the ‖A‖_F/tr(A) values 0.63 and 0.02, Hutchinson's m^-1/2 error rate, and the Figure 1 orderings of Hutch++, NA-Hutch++, Subspace Projection and Hutchinson (d = 1000 instead of 5000, disclosed) |

**claude (opt-in, costs tokens).** Claude receives the analysis and returns a complete file set
(method module, `test_*.py`, `experiment.py`) through a JSON schema. Files are written only if
their paths stay inside the working directory. paper-forge runs pytest and the experiment in the
sandbox, then sends failures and missing metric keys back to Claude. It retries up to
`max_iterations` times. Claimed values come from the analysis and are compared with a 10%
relative tolerance by default. The fidelity notes say so.

**Sandbox.** `docker` runs `--network none` with CPU, memory and pids caps, and with BLAS threads
pinned to the CPU quota. `subprocess` is the fallback: the same interpreter with a scrubbed
environment (no API keys), a sitecustomize guard that blocks sockets, CPU-time and file-size
rlimits, and a process-group kill on timeout. The subprocess guard prevents accidents but is
**not** a security boundary. Use docker for untrusted code. `auto` picks docker when the image
exists.

## Scheduling

- `.github/workflows/daily.yml` runs on weekday mornings. It screens, keeps the sqlite dedupe
  state in the Actions cache, and deploys `site/` to GitHub Pages. Reimplementation is off by
  default. Turn it on with the workflow input or the `FORGE_REIMPLEMENT=true` repo variable,
  and add an `ANTHROPIC_API_KEY` secret.
- `.github/workflows/ci.yml` runs the tests on Python 3.10 and 3.12.

## Real output (2026-09-25)

A real screen of cs.LG, stat.ML and cs.IR (40 per category, plus Hugging Face daily papers),
with no Anthropic key set, so the analyses are heuristic:

```
$ forge -c demo.toml daily --reimplement --backend scripted --day 2026-09-25
screened 112 new papers
reproduced      1412.6980    scripted  Adam: A Method for Stochastic Optimization
reproduced      1206.1901    scripted  MCMC using Hamiltonian dynamics
reproduced      0909.4061    scripted  Finding structure with randomness: Probabilistic algorithms for constr
report: site/daily/2026-09-25.html

$ forge -c demo.toml screen --day 2026-09-26        # same listings again: dedupe
0 new papers screened for 2026-09-26
```

(`demo.toml` sets `categories = ["cs.LG","stat.ML","cs.IR"]`, `max_results_per_category = 40`,
`top_k = 5`. An earlier `-v` run showed 40 + 30 + 30 papers from the three categories and 12
more added from the HF daily list. The whole screen took 16-25 s.)

Measured in the docker sandbox. The three matches here are for the hand-written references, not
for new papers:

| reference | claimed | reproduced |
|---|---|---|
| HMC rejection rate (Neal §3.3) | 0.13 | 0.148 (tolerance 0.03; seeds 0-5 gave 0.119-0.148) |
| RWM rejection rate (Neal §3.3) | 0.75 | 0.7502 |
| rSVD mean error / Thm 10.6 bound | ≤ 1 | 0.122 |
| rSVD mean error / Thm 1.1 bound | ≤ 1 | 0.028 |
| Adam final loss vs Adagrad / Nesterov | faster / similar | 0.480 vs 1.053 / 0.445 |
| Hutch++ ‖A‖_F/tr(A), c = 2 / c = 0.5 (Meyer §6.1) | 0.63 / 0.02 | 0.6325 / 0.0215 |
| Hutchinson log-log error slope, c = 0.5 | ≈ -0.5 | -0.521 (d = 1000, 100 trials; measured 2026-09-25 in docker, 41 s) |

The full generated site is in [`examples/2026-09-25/`](examples/2026-09-25/). Screenshots:
[daily report](examples/daily-report-screenshot.png), [paper page](examples/paper-page-screenshot.png).

## Development

```bash
pytest -q        # fixture Atom feed, rubric, store, sandbox (incl. timeout/no-network), reimpl loop, reports
```

MIT licensed. The papers belong to their authors. Always cite the original work.
