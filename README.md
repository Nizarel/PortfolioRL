# PortfolioRL: Reinforcement Learning for Dynamic Portfolio Rebalancing

A Deep Q-Network agent that learns to reallocate a four-asset portfolio week by
week, evaluated against nine rule-based benchmarks under identical transaction
costs, with the statistical machinery needed to say whether any difference is
real.

> This is a research and backtesting project for a graduate reinforcement
> learning course. **It is not investment advice and does not guarantee future
> performance.**

---

## What this project actually claims

Most applied-RL trading write-ups report a single equity curve from a single
seed on a single split and call it a result. This one is built so that the
opposite is possible: the test window is opened exactly once, the ablation runs
across matched seeds, every Sharpe difference gets a block-bootstrap confidence
interval, and the headline Sharpe is deflated by the number of configurations
that were searched to find it.

If the honest answer turns out to be "not demonstrated", the project is set up
to say so.

---

## Problem formulation

The task is a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

| Component | Definition |
|---|---|
| **State** | 31 dimensions: 23 market features (per-asset returns, 20- and 60-day volatility, 63-day momentum, 50/200-day moving-average ratio, cross-asset signals) plus 8 portfolio-state features (current weights, drawdown, portfolio volatility, days held, previous action) |
| **Action** | 6 discrete allocations, from 100% cash-like to 80/20 equity-heavy |
| **Reward** | $100 \cdot [\log(1 + r^{\text{net}}_t) - \lambda_1 \tau_t - \lambda_2 \sigma_t - \lambda_3 \max(0, \Delta \text{DD}_t)]$ |
| **Transition** | Five trading days of daily-compounded returns after each decision |
| **Episode** | Training: random 52-decision windows. Evaluation: one deterministic pass over the split |

**Decisions are weekly; accounting is daily.** Weekly decisions keep turnover and
costs realistic and give ~1,100 decision points rather than 5,300 nearly
identical ones. Daily accounting means volatility, drawdown and the wealth path
are measured at the frequency they actually occur.

**Assets:** SPY (US equities), TLT (long Treasuries), GLD (gold), SHY (short
Treasuries). SHY rather than BIL because BIL's May 2007 inception post-dates the
start of the sample.

**Sample:** 2004-11-18 to 2025-12-31, set by GLD's inception — the first date on
which all four assets exist.

| Split | Window | Days | Role |
|---|---|---|---|
| Train | 2004-11-18 to 2017-12-31 | 3,102 | Weight updates only. Contains 2008. |
| Validation | 2018-01-01 to 2020-12-31 | 756 | Hyperparameters and checkpoint selection. Contains the COVID crash. |
| Test | 2021-01-01 to 2025-12-31 | 1,255 | Opened once, in notebook 05. Contains the 2022 stock-and-bond drawdown. |

---

## The agent

A custom PyTorch DQN (~12,800 parameters, 31 → 128 → 64 → 6), not
Stable-Baselines3 — SB3's `DQN` is deliberately vanilla and cannot express the
ablation below without rewriting its loss.

The ablation is a clean 2×2 factorial, so each cell differs from its neighbours
by exactly one mechanism:

| Variant | Double Q-learning | Duelling heads |
|---|---|---|
| Vanilla DQN | no | no |
| Double DQN | yes | no |
| Duelling DQN | no | yes |
| Double + Duelling DQN | yes | yes |

Prioritized experience replay was in the original plan and was **dropped**: in
this environment, high TD error concentrates in genuinely high-variance market
episodes rather than in informative-but-rare ones, so prioritizing them
oversamples exactly the periods where the reward signal is least reliable.

**Stabilization:** uniform experience replay, a target network synced every 500
steps, Huber loss (weekly financial rewards are heavy-tailed), gradient-norm
clipping, and ε-greedy exploration annealed linearly from 1.0 to 0.05 over the
first 30% of training.

---

## The benchmarks

All nine execute inside the *same* environment as the agent and pay the *same*
5 bps per unit of turnover. A comparison against a frictionless buy-and-hold
curve would flatter the agent by construction.

- **Rebalanced:** 60/40, equal weight, equity-heavy 80/20, all cash (SHY)
- **Buy and hold:** 100% SPY, 60/40 held
- **Adaptive:** volatility targeting (10% annual), trend following (Faber 2007, MA50/MA200)
- **Floor:** random allocation, reported as a distribution over 30 seeds

Naive inverse-volatility risk parity was implemented and then rejected: with SHY
in the universe its volatility is roughly a tenth of the others', so the rule
degenerates to a near-100% cash position and merely duplicates the all-cash
benchmark.

---

## Statistical treatment

Three hazards stand between "the Sharpe is higher" and "the agent is better".

1. **Seed variance** — every variant is trained across matched seeds (seed *k* of
   one variant sees the same initialisation and episode draws as seed *k* of
   every other), making the comparison paired. Tested with both a paired *t*-test
   and a Wilcoxon signed-rank test.
2. **Sampling noise** — daily returns are serially correlated and heavy-tailed, so
   confidence intervals come from a **stationary block bootstrap** (Politis &
   Romano, 1994) with geometrically distributed blocks, resampled with a shared
   index draw so the comparison stays paired. Lo's (2002) analytic standard error
   is reported alongside as the optimistic bound.
3. **Multiple testing** — Holm–Bonferroni across benchmarks; the **Deflated Sharpe
   Ratio** (Bailey & López de Prado, 2014) across the searched configurations.
   The **minimum track record length** answers directly whether five years of test
   data is even long enough to detect the observed edge.

Robustness is checked two further ways: retraining at 0/5/10/20 bps (retraining,
not re-scoring — the question is whether the approach survives higher costs, not
what this policy would have earned paying more), and walk-forward retraining with
a two-year validation buffer before each test year.

---

## Repository layout

```
src/portfoliorl/
  config.py         paths, splits, action set, all hyperparameter defaults
  data.py           yfinance download and caching
  features.py       31-dim observation, train-only scaler, dataset splits
  env.py            Gymnasium environment, portfolio accounting, run_policy
  metrics.py        16-column performance scorecard
  benchmarks.py     the nine rule-based comparators
  agent.py          replay buffer, Q-network, DQN/Double/duelling agent
  train.py          training loop, validation-based checkpoint selection
  tuning.py         coarse grid + Optuna TPE with median pruning
  significance.py   block bootstrap, PSR, DSR, Holm-Bonferroni
  experiments.py    ablation, cost sweep, walk-forward, result caching
  plots.py          house style, colour registry, figure and table helpers

notebooks/
  00_run_all.ipynb           pipeline map, provenance, reproducibility check
  01_data_eda.ipynb          the data is real, clean, and contains the regimes claimed
  02_env_benchmarks.ipynb    the environment is correct; the benchmarks are hard
  03_dqn_training.ipynb      stable training, plus two design decisions measured
  04_tuning.ipynb            the hyperparameters were searched, not guessed
  05_results_ablation.ipynb  the result, the ablation, and whether it is significant

tests/       112 tests, no network access required
artifacts/   figures, results, models, logs (committed)
Docs/        design document and implementation plan
```

---

## Running it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-build-isolation
python -m ipykernel install --user --name portfoliorl --display-name "Python (PortfolioRL)"

python -m pytest tests/ -q

jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=14400 `
    notebooks/01_data_eda.ipynb notebooks/02_env_benchmarks.ipynb `
    notebooks/03_dqn_training.ipynb notebooks/04_tuning.ipynb `
    notebooks/05_results_ablation.ipynb notebooks/00_run_all.ipynb
```

Notebook 01 is the only one that touches the network. Every experiment in
notebook 05 caches to `artifacts/results/`; set `FORCE = True` in its setup cell
to discard the cache and retrain.

### Compute

**CPU only, and deliberately so.** The Q-network has ~12,800 parameters — at that
size, CUDA kernel-launch overhead exceeds the arithmetic, and the real bottleneck
is stepping the Python environment one week at a time, which a GPU does not
accelerate. A 120,000-step training run takes about five minutes on a laptop CPU.
Moving it to a GPU would make it slower.

Full pipeline from scratch: roughly three hours, dominated by notebook 05's 51
training runs.

---

## Status

- [x] Data pipeline, feature engineering, EDA
- [x] Gymnasium environment with golden tests
- [x] Metrics and nine benchmarks
- [x] DQN agent, training loop, 2×2 ablation
- [x] Two-stage hyperparameter search (grid + Optuna TPE)
- [x] Significance testing and experiment orchestration
- [x] Notebooks 01–03 executed end to end
- [x] Design document updated to match the implementation
- [ ] Notebooks 04, 05 and 00 executed end to end
- [ ] Final report and submission package

---

## Key references

Mnih et al. (2015) *Human-level control through deep reinforcement learning* ·
van Hasselt, Guez & Silver (2016) *Deep RL with double Q-learning* ·
Wang et al. (2016) *Duelling network architectures* ·
Akiba et al. (2019) *Optuna* ·
Bergstra & Bengio (2012) *Random search for hyper-parameter optimization* ·
Politis & Romano (1994) *The stationary bootstrap* ·
Lo (2002) *The statistics of Sharpe ratios* ·
Bailey & López de Prado (2014) *The deflated Sharpe ratio* ·
Faber (2007) *A quantitative approach to tactical asset allocation* ·
Sutton & Barto (2018) *Reinforcement Learning: An Introduction*

Full list in [Docs/PortfolioRL_ProjectAssignment3.md](Docs/PortfolioRL_ProjectAssignment3.md).
