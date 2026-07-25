# PortfolioRL — Implementation Plan

**Project:** PortfolioRL: Reinforcement Learning for Dynamic Portfolio Rebalancing
**Course:** Reinforcement Learning, Penn State University
**Deliverable targeted:** Final Project Assignment (Deliverable 5)
**Plan date:** 2026-07-25
**Status:** Plan approved — implementation not yet started

---

## Table of Contents

1. [Purpose of This Document](#1-purpose-of-this-document)
2. [Deliverable and Rubric Mapping](#2-deliverable-and-rubric-mapping)
3. [Design Review of the Assignment 3 Proposal](#3-design-review-of-the-assignment-3-proposal)
   3.1 [Correctness Issues to Fix Before Coding](#31-correctness-issues-to-fix-before-coding)
   3.2 [Rigor Upgrades](#32-rigor-upgrades)
   3.3 [Internal Consistency Defects in the Document](#33-internal-consistency-defects-in-the-document)
4. [Approved Design Decisions](#4-approved-design-decisions)
5. [Verified Technical Environment](#5-verified-technical-environment)
6. [Repository and Artifact Layout](#6-repository-and-artifact-layout)
7. [Implementation Phases](#7-implementation-phases)
   7.0 [Phase 0 — Scaffold](#phase-0--scaffold)
   7.1 [Phase 1 — Data and Features](#phase-1--data-and-features)
   7.2 [Phase 2 — Environment](#phase-2--environment)
   7.3 [Phase 3 — Benchmark Policies](#phase-3--benchmark-policies)
   7.4 [Phase 4 — Metrics](#phase-4--metrics)
   7.5 [Phase 5 — DQN Agent](#phase-5--dqn-agent)
   7.6 [Phase 6 — Hyperparameter Tuning](#phase-6--hyperparameter-tuning)
   7.7 [Phase 7 — Final Runs, Seeds, Ablation](#phase-7--final-runs-seeds-ablation)
   7.8 [Phase 8 — Notebooks](#phase-8--notebooks)
   7.9 [Phase 9 — Report, Documentation, Packaging](#phase-9--report-documentation-packaging)
   7.10 [Phase 10 — Video Presentation](#phase-10--video-presentation)
8. [Test Strategy](#8-test-strategy)
9. [Risk Register](#9-risk-register)
10. [Definition of Done](#10-definition-of-done)
11. [Additional References to Add](#11-additional-references-to-add)

---

## 1. Purpose of This Document

This plan converts the Assignment 3 design document
([PortfolioRL_ProjectAssignment3.md](PortfolioRL_ProjectAssignment3.md)) into an
executable engineering plan for the final deliverable. It records:

- a detailed technical review of the proposal, with the specific defects that would
  produce wrong or misleading results if implemented as written;
- the design decisions that were approved to resolve those defects;
- verified facts about the development environment that constrain the tooling choice;
- a phase-by-phase build order with concrete deliverables and acceptance tests.

Every deviation from the Assignment 3 document is listed explicitly in
[Section 4](#4-approved-design-decisions) so it can be justified in the final report.

---

## 2. Deliverable and Rubric Mapping

The Final Project Assignment requires five submissions, graded out of 15 points.

| # | Assignment requirement | Rubric criterion | Points | Produced by |
|---|---|---|---|---|
| 1 | Discuss results for the proposed RL problem; report all performance measures obtained | Results | 7 | [Phase 7](#phase-7--final-runs-seeds-ablation), [Phase 9](#phase-9--report-documentation-packaging) |
| 2 | Submit final Python script as a Jupyter file; place notebooks + dataset/artifacts in a folder and zip it | (supports Results / Documentation) | — | [Phase 8](#phase-8--notebooks), [Phase 9](#phase-9--report-documentation-packaging) |
| 3 | Discuss experience working on this RL project | Experience Statement | 1 | [Phase 9](#phase-9--report-documentation-packaging) |
| 4 | 20-minute (max) video: problem, algorithm walkthrough, metrics, script walkthrough, results and conclusions | Video Presentation | 4 | [Phase 10](#phase-10--video-presentation) |
| 5 | Upload report, zipped folder, video to Canvas | Documentation | 3 | [Phase 9](#phase-9--report-documentation-packaging) |

**Implication for build order.** "Results" is worth 7 of 15 points and explicitly asks
for *all* the performance measures promised in Section 4 of the proposal. The metrics
module ([Phase 4](#phase-4--metrics)) is therefore a first-class deliverable, not an
afterthought, and every metric listed in proposal Table 10 must appear in the final
scorecard for every strategy.

---

## 3. Design Review of the Assignment 3 Proposal

**Overall assessment.** The MDP formulation is correct, the DQN justification is sound
and well-argued, the treatment of look-ahead bias and walk-forward validation is better
than typical for a course project, and the evaluation framework is comprehensive. The
issues below are implementation-level defects — things that are defensible on paper but
that produce a broken or misleading agent when coded literally.

### 3.1 Correctness Issues to Fix Before Coding

#### R1 — The reward double-counts persistent risk

*Location:* Section 2.5.

The proposed reward is

```
Reward_t = portfolio_return_t − λ₁·turnover_t − λ₂·volatility_t − λ₃·drawdown_t
```

`drawdown_t` is a *level*, not an event. If the portfolio enters a 20% drawdown and stays
there for 100 steps, the agent is charged `λ₃ × 0.20` one hundred times for a single loss
that it already paid for through the return term. The learned behaviour is pathological:
once in a drawdown the agent's best response is to move to cash and stay there, because
recovering requires accepting risk while the standing penalty continues to accrue.

**Fix.** Penalise the *increment* in drawdown:

```
dd_penalty_t = max(0, DD_t − DD_{t−1})
```

This charges the agent once, at the moment new losses are made, and leaves recovery
unpenalised. Volatility may remain a level, since it is a genuine per-step risk exposure
rather than a cumulative one.

#### R2 — Arithmetic returns are not time-additive

*Location:* Section 2.5, Section 3.1.

The undiscounted sum of arithmetic returns is not terminal wealth:
`Σ r_t ≠ ∏(1 + r_t) − 1`. With `γ → 1` the RL objective therefore optimises a quantity
that is only approximately related to the business objective.

**Fix.** Use `log(1 + r_p,t)`. Log returns are additive, so `Σ reward_t` equals terminal
log wealth exactly. The RL objective then *is* the stated business objective rather than
a proxy for it, which is a point worth making explicitly in the final report.

#### R3 — Reward scale silently kills the gradient

*Location:* Section 3.5.

Daily portfolio log returns are O(10⁻³). Squaring them in the TD loss gives O(10⁻⁶).
At `learning_rate = 1e-3` the network barely moves, training curves look flat, and the
usual (wrong) conclusion is that "DQN does not work on this problem."

**Fix.** Multiply the reward by 100 so it is expressed in percentage points. Record
`reward_scale` as an environment constant and state it in the report — it changes the
numerical scale of Q-values and TD loss, so the training diagnostics must be interpreted
against it.

#### R4 — Turnover must be measured against *drifted* weights

*Location:* Section 4.5.

Between two rebalancing dates the weights drift with realised returns:

```
w⁻_{t+1} = ( w_t ⊙ (1 + r_{t+1}) ) / ( 1 + w_t · r_{t+1} )
```

Computing `turnover = ½ Σ|w_target,t − w_target,t−1|` compares the new target with the
previous *target*, which reports zero turnover whenever the agent repeats an action —
even though real drift has to be traded back. Conversely it reports large turnover on an
action change that the market had already partly performed.

**Fix.** `turnover_t = ½ Σᵢ |w_target,t,i − w⁻_t,i|`. This is the single most commonly
omitted detail in student portfolio backtests and directly affects the transaction-cost
results in Section 4.5.

#### R5 — Transaction cost must reduce wealth, not only reward

*Location:* Sections 2.5 and 4.2.

The proposal places turnover in the reward but does not state that the cost is deducted
from portfolio value. If it is not, "net of costs" performance metrics and the training
reward describe two different portfolios, and the gross-vs-net comparison promised in
Section 4.2 is not well defined.

**Fix.** Deduct `cost_t = turnover_t × fee_rate` from portfolio value inside the
environment's `step()`. The reward's turnover term and the reported cost drag then refer
to the same quantity.

#### R6 — One historical path plus full-window episodes equals memorisation

*Location:* Section 2.6.

A single pass over 2010–2019 at daily frequency is ~2,500 transitions. DQN typically needs
10⁵–10⁶ steps, i.e. 40–400 identical replays of one price path. The agent will memorise
the realised sequence — "in March 2020 go to cash" — rather than learn a state-conditional
policy. Validation performance will then look arbitrary rather than informative.

**Fix (highest-impact change in this plan).** Use **random-start, fixed-length
sub-episodes**: on `reset()`, sample a random start index inside the training window and
run for a fixed number of decisions (~52 weekly decisions ≈ 1 year). One price path
becomes a large family of overlapping trajectories with varied starting regimes and
varied initial drawdown states. Evaluation episodes remain deterministic single passes
over the full validation or test window.

#### R7 — Daily rebalancing conflicts with both cost realism and the discount factor

*Location:* Sections 2.6 and 3.5.

Two independent problems. First, daily rebalancing across six allocations incurs
continuous turnover; at any realistic fee the cost drag dominates the signal. Second,
`γ = 0.99` at daily frequency gives an effective horizon of ~100 trading days, which is
too short for an objective whose stated purpose is drawdown control over market regimes.

**Fix.** Decide **weekly** (every 5 trading days); compute portfolio value and every
reported metric from **daily** returns. At weekly frequency `γ = 0.99` corresponds to an
effective horizon of ~2 years, which matches the problem.

#### R8 — The state omits a quantity the reward depends on (Markov violation)

*Location:* Section 2.3 versus Section 2.5.

The reward depends on realised *portfolio* volatility and on drawdown. Current weights and
drawdown are in the state, but only *per-asset* volatility is — the agent cannot infer its
own portfolio volatility without it.

**Fix.** Add rolling portfolio volatility and time-since-peak (drawdown duration) to the
state vector. Both are cheap and both are required for the MDP to be well posed.

#### R9 — Test window too short, and training never sees a bear market

*Location:* Section 3.4.

The proposed split trains on 2010–2019 — a period containing no severe equity bear market
and no rising-rate regime — and tests on 2022–2024, roughly 750 daily / 150 weekly steps.
Sharpe ratio estimates from 150 observations have very wide confidence intervals, and a
policy that has never observed a drawdown regime is being asked to handle one out of
sample.

**Fix.** Extend the start date to GLD's inception. SPY (1993), TLT (2002-07) and SHY
(2002-07) all pre-date GLD (2004-11-18), so 2004-11-18 is the common start.

#### R10 — Undefined risk-free rate and MAR

*Location:* Sections 4.3 and 4.4.

Sharpe uses `r_f` and Sortino uses `MAR`; neither is defined, so the reported numbers are
not reproducible.

**Fix.** Define `r_f` as the 13-week U.S. Treasury bill (`^IRX`, converted from an
annualised discount rate to a daily rate) and `MAR = 0`. Also note in the report that the
`√252` annualisation of Sharpe assumes i.i.d. returns (Lo, 2002), which financial returns
violate.

#### R11 — Benchmarks must pay the same transaction costs

*Location:* Section 4.8.

Equal-weight and 60/40 "periodically rebalanced" benchmarks have real turnover;
buy-and-hold pays only the entry trade. If the RL agent pays costs and the benchmarks do
not, the comparison is meaningless in the RL agent's disfavour.

**Fix.** Run every benchmark through the *same* environment object with the *same* fee
model, implemented as fixed policies. This also guarantees identical calendar, identical
starting capital and identical accounting.

### 3.2 Rigor Upgrades

These are not corrections; they are additions that materially strengthen the "Results"
criterion.

| ID | Upgrade | Rationale |
|---|---|---|
| U1 | **Causal feature construction** — every rolling feature computed as `.rolling(w).agg().shift(1)`; action at *t* applies to the return over *t→t+1*; the 200-day warm-up is trimmed *before* the training window begins. | Removes look-ahead bias at the row level, not just at the split level. Section 3.4 addresses splits but not row alignment. |
| U2 | **Block bootstrap confidence intervals** (moving-block or stationary bootstrap) rather than i.i.d. bootstrap. | Section 4.8 asks for bootstrap CIs. Financial returns are autocorrelated and heteroskedastic; i.i.d. resampling destroys that structure and produces intervals that are far too narrow. |
| U3 | **Probabilistic and Deflated Sharpe Ratio** (Bailey & López de Prado, 2014). | ~30 Optuna trials are run and the best validation Sharpe is selected. The selected Sharpe is therefore a maximum over many trials and is upward-biased. DSR adjusts for the number of trials, non-normality and sample length. This pre-empts the most obvious methodological criticism of the whole project. |
| U4 | **Random-action policy as a fifth benchmark.** | Distinguishes "the agent learned something" from "the action menu is well designed." Without it, a mediocre agent can be flattered by a favourable action set. |
| U5 | **Allocation-over-time plot overlaid on SPY drawdown**, plus an action-frequency table. | Section 4.5 claims the analysis will reveal "whether the agent learns meaningful regime-dependent behavior." This is the figure that demonstrates it rather than asserting it. |
| U6 | **Transaction-cost sensitivity sweep** (0 / 5 / 10 / 20 bps) on the final policy. | Turns a single fragile number into a robustness result, and directly supports the "practicality" discussion in Section 4.5. |
| U7 | **Frozen on-disk data cache** (parquet + CSV committed to the artifacts folder). | `yfinance` results change over time (restatements, adjustment revisions, API changes). A graded artifact must reproduce exactly; the notebook loads the cache and only re-downloads on explicit request. |

### 3.3 Internal Consistency Defects in the Document

| ID | Defect | Location | Resolution |
|---|---|---|---|
| D1 | Section 3.3 proposes DQN plus Double DQN. Sections 4.6 and 4.7 promise results for "DQN, Double DQN, Dueling DQN, and prioritized-replay variants." | §3.3 vs §4.6/§4.7 | Implement DQN, Double DQN, Dueling DQN and Double+Dueling as a four-row ablation. Drop prioritized replay and remove it from §4.6/§4.7. |
| D2 | Table numbering jumps from Table 6 to Table 10; Tables 7–9 do not exist. | §4.7 | Renumber. |
| D3 | Tables 8 and 9 are referenced nowhere but the numbering implies they existed in an earlier draft. | §4 | Confirm no dangling cross-references remain after renumbering. |
| D4 | Section 4.7 promises an "information ratio" without naming the benchmark it is measured against. | §4.7 | Report IR against each benchmark separately; state that IR is benchmark-relative. |
| D5 | Section 2.4 lists "SHY or BIL"; BIL's inception (2007-05) is later than the new start date. | §2.4, §3.4, Table 1 | Fix the universe to SHY. |

---

## 4. Approved Design Decisions

These decisions were confirmed and supersede the corresponding statements in the
Assignment 3 document. Each must be restated and justified in the final report.

| # | Decision | Supersedes | Justification |
|---|---|---|---|
| A1 | **Weekly rebalancing decisions; daily return accounting and daily-frequency metrics.** | §2.6 (implied daily) | R7 — cost realism and `γ` horizon. |
| A2 | **Sample period 2004-11-18 → 2025-12-31. Train 2004-11-18–2017-12-31; validation 2018-01-01–2020-12-31; test 2021-01-01–2025-12-31.** | §3.4 (2010–2019 / 2020–2021 / 2022–2024) | R9 — training now contains the 2008 GFC; validation contains the 2018 Q4 selloff, the 2020 COVID crash and its recovery; test contains the 2022 simultaneous stock-and-bond drawdown. |
| A3 | **Algorithm scope: DQN, Double DQN, Dueling DQN, Double+Dueling. Prioritized experience replay dropped.** | §3.3 / §4.6 conflict (D1) | Dueling is a low-cost architectural change (a value/advantage head) and yields a clean four-row ablation. PER adds a sum-tree and importance-sampling corrections for little expected benefit in a small, dense-reward, six-action problem. |
| A4 | **Code delivered as an importable `src/portfoliorl/` package plus five story notebooks and an orchestration notebook.** | Assignment allows "Jupyter file(s)" | Logic in the package is unit-testable and reviewable; notebooks stay readable and narrative. `00_run_all.ipynb` gives the grader a single end-to-end entry point. |
| A5 | **Reward:** `100 × [ log(1 + r_p) − λ₁·turnover − λ₂·σ_p − λ₃·max(0, ΔDD) ]`, with cost also deducted from wealth. | §2.5 | R1, R2, R3, R5. |
| A6 | **Training episodes are random-start, fixed-length (52 decisions). Evaluation episodes are deterministic full passes.** | §2.6 | R6. |
| A7 | **Custom PyTorch DQN implementation**, with Stable-Baselines3 used only as an optional sanity cross-check on the vanilla variant. | — | SB3's DQN documentation states it "provides only vanilla Deep Q-Learning and has no extensions such as Double-DQN, Dueling-DQN and Prioritized Experience Replay." The ablation in A3 therefore cannot be built on SB3. |
| A8 | **The Assignment 3 document will be updated** to reflect R1–R11, D1–D5 and A1–A7. | — | Keeps the design document and the implementation consistent, so the final report does not have to explain two different designs. |

### 4.1 Resulting MDP Specification

| Component | Specification |
|---|---|
| **Decision frequency** | Every 5 trading days (weekly) |
| **Assets** | SPY (equities), TLT (long Treasuries), GLD (gold), SHY (short Treasuries / cash proxy) |
| **Actions** | 6 discrete target allocations (proposal Table 4, unchanged) |
| **State** | Per asset: last-period return, 20d volatility, 60d volatility, 63d momentum, price/MA50, MA50/MA200. Portfolio: current drifted weights (4), rolling portfolio volatility, current drawdown, drawdown duration. All standardised with train-window statistics only. |
| **Reward** | `100 × [ log(1+r_p) − λ₁·turnover − λ₂·σ_p − λ₃·max(0, ΔDD) ]` |
| **Transition** | Apply target weights, charge cost against wealth, compound 5 daily returns, drift weights, recompute risk state |
| **Episode (train)** | 52 consecutive decisions from a random start index in the training window |
| **Episode (eval)** | One deterministic pass over the validation or test window |
| **Discount** | `γ = 0.99` weekly (≈ 2-year effective horizon), tuned in [0.90, 0.999] |
| **Costs** | 5 bps of turnover, one-way; sensitivity at 0 / 5 / 10 / 20 bps |

---

## 5. Verified Technical Environment

Checked on the development machine on 2026-07-25. These facts constrain the tooling and
are recorded here because one of them would otherwise block the project.

| Item | Finding |
|---|---|
| Platform | Windows on **ARM64** |
| Python | 3.12.10 (ARM64). A 3.13 interpreter is also present but must **not** be used — see below. |
| **PyTorch** | `pip install torch` **fails** on `win_arm64` from PyPI ("No matching distribution found"). It installs correctly from the official index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, resolving `torch-2.13.0+cpu-cp312-cp312-win_arm64.whl`. CPU-only; no CUDA on ARM. Requires Python 3.12. |
| Other wheels (all available for cp312-win_arm64) | `gymnasium 1.3.0`, `yfinance 1.5.2`, `pandas 3.0.5`, `numpy 2.5.1`, `matplotlib 3.11.1`, `optuna 4.9.0`, `scipy 1.18.0` |
| Gymnasium API | `step()` returns the 5-tuple `(obs, reward, terminated, truncated, info)`; `reset(seed=None, options=None)` must call `super().reset(seed=seed)` first; `gymnasium.utils.env_checker.check_env` is available for validation. |
| Stable-Baselines3 DQN | Vanilla only — no Double, Dueling or PER. Confirms decision A7. |
| Compute budget | An MLP DQN over ~10⁵ steps on CPU runs in minutes. Roughly 30 pruned Optuna trials plus 8 final seeded runs is well under an hour of total compute. No GPU or cloud compute required. |

**Environment setup commands** (recorded so the report's reproducibility section is exact):

```powershell
python -m venv .venv                      # must be the 3.12 interpreter
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 6. Repository and Artifact Layout

```
PortfolioRL/
├─ README.md                        project overview + reproduction instructions
├─ requirements.txt                 pinned dependencies (torch install note included)
├─ .gitignore
├─ Docs/
│  ├─ PortfolioRL_ProjectAssignment3.md    design document (to be updated per A8)
│  ├─ PortfolioRL_ImplementationPlan.md    this file
│  ├─ PortfolioRL_FinalReport.md           final report: results + experience statement
│  ├─ video_outline.md                     timed 20-minute presentation script
│  └─ figures/                             figures referenced by the report
├─ data/
│  ├─ raw/          adjusted prices as downloaded (parquet + csv), frozen
│  └─ processed/    aligned returns, causal features, split index, scaler stats
├─ src/portfoliorl/
│  ├─ __init__.py
│  ├─ config.py       frozen dataclasses: DataConfig, EnvConfig, AgentConfig
│  ├─ utils.py        seeding, JSON run bookkeeping
│  ├─ data.py         download, cache, align, quality report
│  ├─ features.py     causal feature engineering, splits, train-only scaler
│  ├─ env.py          PortfolioEnv(gymnasium.Env)
│  ├─ benchmarks.py   fixed policies run through the same env
│  ├─ metrics.py      full scorecard + block bootstrap + PSR/DSR
│  ├─ agent.py        replay buffer, Q-network (dueling optional), DQNAgent
│  ├─ train.py        training loop, eval callback, checkpointing
│  ├─ tuning.py       Optuna objective and study driver
│  └─ plots.py        every figure used in the report
├─ notebooks/
│  ├─ 00_run_all.ipynb           end-to-end orchestration
│  ├─ 01_data_eda.ipynb          data, features, splits, EDA
│  ├─ 02_env_benchmarks.ipynb    environment walkthrough + benchmark scorecard
│  ├─ 03_dqn_training.ipynb      agent, training diagnostics
│  ├─ 04_tuning.ipynb            coarse search + Optuna study
│  └─ 05_results_ablation.ipynb  final results, ablation, significance, figures
├─ tests/
│  ├─ test_features.py    causality / no-leakage assertions
│  ├─ test_env.py         environment invariants
│  └─ test_metrics.py     closed-form metric checks
├─ artifacts/
│  ├─ models/       best checkpoints per variant per seed
│  ├─ results/      scorecard CSVs, per-step trajectories, tuning results
│  ├─ figures/      generated PNGs
│  ├─ logs/         training logs (JSONL)
│  └─ optuna.db     persisted study
└─ package_submission.py   builds the zipped submission folder
```

---

## 7. Implementation Phases

Phases are ordered so that each one is verifiable before the next depends on it. The
critical path is P1 → P2 → P3/P4 → P5 → P6 → P7.

### Phase 0 — Scaffold

**Goal.** A reproducible environment and a single source of truth for configuration.

**Tasks**
1. Create the `.venv` (Python 3.12) and install dependencies per [Section 5](#5-verified-technical-environment).
2. `requirements.txt` including the ARM64 torch installation note.
3. `src/portfoliorl/config.py` — frozen dataclasses `DataConfig`, `EnvConfig`,
   `AgentConfig`, `ProjectConfig`; the discrete action matrix and action labels;
   canonical output paths and `ensure_dirs()`.
4. `src/portfoliorl/utils.py` — `set_global_seed()` (Python, NumPy, torch),
   JSON run bookkeeping.
5. `.gitignore` for regenerated artifacts.

**Acceptance.** `import portfoliorl; import torch, gymnasium, yfinance, optuna` succeeds
in the venv; `ensure_dirs()` creates the full tree.

**Status.** Partially complete — `requirements.txt`, `.gitignore`, `config.py`,
`utils.py` and the venv have been created.

---

### Phase 1 — Data and Features

**Goal.** A frozen, leakage-free, fully documented feature matrix.

**Tasks**
1. `data.py::download_prices()` — fetch adjusted daily closes for SPY, TLT, GLD, SHY and
   `^IRX` over 2004-11-01 → 2025-12-31 via `yfinance`. Write to `data/raw/` as parquet
   **and** CSV. All downstream code reads the cache; re-download only on
   `force_refresh=True` (U7).
2. `data.py::quality_report()` — per-ticker first/last observation, missing-day counts,
   calendar-alignment diagnostics, zero/duplicate-price checks, extreme-return flags.
   Emitted as a markdown table for the report.
3. Align all tickers to the intersection of trading calendars; document any dropped rows.
4. Convert adjusted closes to daily **log** returns.
5. `features.py::build_features()` — per asset: previous-period return, 20d and 60d
   rolling volatility, 63d momentum, price/MA50 ratio, MA50/MA200 ratio. **Every feature
   ends in `.shift(1)`** so the feature at *t* uses only information available at the
   close of *t* (U1).
6. Trim the 200-day warm-up **before** `train_start`, so the training window is not
   shortened by NaN removal.
7. `features.py::chronological_split()` — return boolean index masks for train /
   validation / test per decision A2. Assert the masks are contiguous, non-overlapping
   and exhaustive.
8. `features.py::fit_scaler()` / `apply_scaler()` — z-score statistics computed on the
   **training window only**, persisted to `data/processed/scaler.json`, applied unchanged
   to validation and test.
9. Convert `^IRX` (annualised discount rate, percent) to a daily risk-free rate series
   for the Sharpe/Sortino calculations (R10).

**Deliverables.** `data/raw/prices.parquet`, `data/raw/prices.csv`,
`data/processed/features.parquet`, `data/processed/splits.json`,
`data/processed/scaler.json`, `Docs/figures/` EDA plots (cumulative growth of each ETF,
rolling correlation matrix, rolling volatility, drawdown of each asset, split boundaries
annotated on the SPY price series).

**Acceptance.** `tests/test_features.py` asserts (a) no feature column correlates with
its own contemporaneous return at lag 0 by construction, (b) shifting the price series
forward by one day changes no feature value at *t*, (c) scaler statistics computed on
train differ from those on the full sample, confirming the scaler was not fit on the
whole set, (d) no NaNs remain after warm-up trimming.

---

### Phase 2 — Environment

**Goal.** A `gymnasium.Env` whose accounting is provably correct.

**`PortfolioEnv` contract**

| Element | Definition |
|---|---|
| `observation_space` | `Box(-inf, inf, (n_features,), float32)` — scaled market features concatenated with unscaled portfolio state (weights, portfolio vol, drawdown, drawdown duration) |
| `action_space` | `Discrete(6)` |
| `reset(seed, options)` | Calls `super().reset(seed=seed)`. Training mode samples a random start index; evaluation mode starts at the window start. Initial weights = 100% SHY, initial value = 100,000. |
| `step(action)` | 1. Look up target weights. 2. `turnover = ½Σ|w_target − w_drift|` (R4). 3. `cost = turnover × fee`; deduct from wealth (R5). 4. Compound the next 5 daily returns; record each day for metrics. 5. Drift weights (R4 formula). 6. Update peak, drawdown, drawdown duration, rolling portfolio vol. 7. Compute reward per A5. 8. `terminated` on window exhaustion; `truncated` on episode-length limit. |
| `info` | Every reward component separately (`log_return`, `turnover`, `cost`, `portfolio_vol`, `dd_increment`), plus `portfolio_value`, `weights`, `date`, `daily_returns` — so the reward can be audited term by term and the metrics module can consume daily data. |

**Tasks**
1. Implement `env.py` per the contract.
2. `env.py::run_policy(env, policy_fn)` — deterministic rollout helper returning a tidy
   daily DataFrame (`date, portfolio_value, daily_return, weights…, turnover, cost, action`).
   Used identically by benchmarks and by the trained agent, guaranteeing a fair comparison.
3. Validate with `gymnasium.utils.env_checker.check_env`.

**Acceptance (`tests/test_env.py`) — these tests are the backbone of the project's credibility**
- **Golden test:** with `transaction_cost_bps = 0`, an agent that always plays action *k*
  reproduces the equity curve of that static allocation computed independently from the
  price data, to floating-point tolerance. This single test validates drift, compounding
  and accounting simultaneously.
- With `transaction_cost_bps = 0` and a constant action, turnover after the first step
  equals the pure drift amount, and is exactly zero for the single-asset action (100% SHY).
- Weights sum to 1 at every step; portfolio value is strictly positive.
- Costs are monotonically non-decreasing in the fee parameter; net value ≤ gross value.
- Drawdown is in [0, 1); a new high resets drawdown to 0 and duration to 0.
- Reward equals the sum of its `info` components times `reward_scale`.
- `reset(seed=s)` twice yields identical episodes; different seeds yield different
  start indices in training mode and identical start indices in evaluation mode.
- Episodes never index past the end of their split window.

---

### Phase 3 — Benchmark Policies

**Goal.** Five reference strategies computed under conditions identical to the agent's.

| Benchmark | Implementation |
|---|---|
| Buy-and-hold SPY | Action fixed to 100% SPY (added as an evaluation-only allocation); pays only the entry cost |
| Equal weight | Action 4, rebalanced at every decision |
| Fixed 60/40 | Action 1, rebalanced at every decision |
| Calendar (monthly) rebalanced | Target = equal weight, but rebalance only on the first decision of each month; drift in between |
| Random policy (U4) | Uniform random action, averaged over the same number of seeds as the agent |

**Tasks.** Implement each as a `policy_fn` and run through `run_policy` on the test
window (and on validation, for context). Persist per-strategy daily trajectories to
`artifacts/results/benchmarks_test.csv`.

**Acceptance.** Buy-and-hold SPY's cumulative return matches an independent calculation
from the raw price series to within the modelled entry cost. Monthly-rebalanced turnover
is strictly lower than every-decision-rebalanced turnover for the same target weights.

---

### Phase 4 — Metrics

**Goal.** Every metric in proposal Table 10, computed once and reused everywhere.

**Tasks**
1. `metrics.py` implementing, from a daily return series:
   - Return: cumulative return, CAGR — reported gross **and** net.
   - Risk: annualised volatility, maximum drawdown, downside deviation, longest drawdown
     duration, 95% historical VaR and CVaR.
   - Risk-adjusted: Sharpe (against the `^IRX` daily rate), Sortino (MAR = 0), Calmar.
   - Trading: average turnover per decision, annualised turnover, total cost drag,
     action-frequency distribution, time-in-allocation.
   - Relative (per benchmark): excess CAGR, tracking error, information ratio, up/down
     capture ratio.
2. `metrics.scorecard(returns_by_strategy) -> DataFrame` producing the report table.
3. `metrics.block_bootstrap_ci()` — moving-block bootstrap over daily returns, block
   length ≈ 21 days, 10,000 resamples, for Sharpe and CAGR (U2).
4. `metrics.probabilistic_sharpe_ratio()` and `metrics.deflated_sharpe_ratio()`,
   the latter taking the number of Optuna trials as the multiple-testing input (U3).

**Acceptance (`tests/test_metrics.py`).** Constant-return series give
`volatility = 0`, `max_drawdown = 0`, and CAGR equal to the closed-form value. A series
with a single known 30% peak-to-trough decline yields `max_drawdown = 0.30`. Sharpe of a
series shifted by a constant matches the analytic shift. A strategy compared against
itself yields tracking error 0 and undefined/NaN information ratio (handled explicitly).
PSR of a Sharpe equal to the benchmark Sharpe is 0.5.

---

### Phase 5 — DQN Agent

**Goal.** A correct, instrumented DQN with Double and Dueling as toggles.

**Tasks**
1. `agent.py::ReplayBuffer` — pre-allocated NumPy circular buffer of
   `(s, a, r, s', done)`; uniform sampling.
2. `agent.py::QNetwork` — MLP, configurable hidden sizes, ReLU. `dueling=True` splits the
   final layer into a scalar value stream and a 6-way advantage stream, recombined as
   `Q = V + (A − mean(A))`.
3. `agent.py::DQNAgent` — ε-greedy action selection with linear decay over
   `eps_decay_fraction` of total steps; Huber (smooth L1) loss; gradient-norm clipping;
   hard target-network update every `target_update_interval` steps (soft update via `tau`
   supported as a tuning option). `double_dqn=True` selects `argmax` with the online
   network and evaluates with the target network.
4. `train.py::train()` — the main loop; logs per-step to JSONL: TD loss, mean and
   standard deviation of Q-values, ε, episodic reward, action counts, gradient norm.
5. `train.py::EvalCallback` — every `eval_every` steps, run a deterministic
   (`ε = 0`) full pass over the **validation** window, compute Sharpe, and checkpoint the
   best-so-far model. The test window is never touched here.

**Acceptance.**
- Overfit test: on a 200-step toy window the agent's training reward strictly improves
  and the greedy policy converges to a stable action.
- The `double_dqn=True` path produces target values that are ≤ the `double_dqn=False`
  targets on average across a fixed batch, demonstrating overestimation reduction.
- The dueling network's parameter count and output shape match expectations; with
  `dueling=True` and a constant advantage stream, `Q = V` for all actions.
- Setting the seed reproduces the training curve exactly.
- Q-values do not diverge: `|Q|` stays within a plausible bound given `reward_scale`
  and `γ` (`|Q| ≲ reward_scale × max|r| / (1 − γ)`).

---

### Phase 6 — Hyperparameter Tuning

**Goal.** A defensible, documented configuration selected without touching the test set.

**Stage 1 — coarse grid** over the three parameters expected to dominate, using short
runs (~30k steps) to eliminate clearly bad regions:

| Parameter | Grid |
|---|---|
| `learning_rate` | 1e-4, 3e-4, 1e-3 |
| `gamma` | 0.95, 0.99, 0.995 |
| `hidden_sizes` | (64,), (128, 64), (256, 128) |

**Stage 2 — Optuna** (TPE sampler + MedianPruner), ~30 trials, over the narrowed ranges
plus the remaining parameters:

| Parameter | Range |
|---|---|
| `learning_rate` | log-uniform, narrowed by Stage 1 |
| `gamma` | narrowed by Stage 1 |
| `buffer_size` | 10,000 – 100,000 |
| `batch_size` | {32, 64, 128} |
| `target_update_interval` | 250 – 2,500 |
| `eps_decay_fraction` | 0.1 – 0.5 |
| `hidden_sizes` | narrowed by Stage 1 |
| `lambda_turnover` (λ₁) | 0.0 – 1.0 |
| `lambda_volatility` (λ₂) | 0.0 – 1.0 |
| `lambda_drawdown` (λ₃) | 0.0 – 2.0 |

**Objective.** Validation-window Sharpe ratio, per proposal Section 3.5. Maximum drawdown
on validation is the documented tie-breaker between configurations within 0.05 Sharpe of
each other. Each trial is averaged over 2 seeds to reduce selection noise.

**Tasks.** `tuning.py` with the Optuna objective, an SQLite-backed study at
`artifacts/optuna.db` (resumable and submittable as an artifact), and Optuna's
optimisation-history, parallel-coordinate and parameter-importance plots exported to
`artifacts/figures/`.

**Acceptance.** The study is resumable after interruption. The number of completed trials
is recorded and passed to the Deflated Sharpe calculation (U3). The selected configuration
is written to `artifacts/results/best_config.json` alongside the validation metrics that
justified it.

---

### Phase 7 — Final Runs, Seeds, Ablation

**Goal.** The evidence base for the "Results" criterion (7 of 15 points).

**Tasks**
1. Retrain the selected configuration for each of the four variants — DQN, Double DQN,
   Dueling DQN, Double+Dueling — with **8 random seeds** each (32 runs total).
2. Evaluate each run **once** on the test window using the checkpoint chosen by validation
   Sharpe. The test window is opened exactly once, at this point, per proposal Section 4.8.
3. Produce the full scorecard for all four variants and all five benchmarks.
4. Report mean, median, standard deviation, min and max across seeds for every headline
   metric — not the best run (proposal Section 4.8, item 3).
5. Block-bootstrap 95% CIs for Sharpe and CAGR; PSR and DSR for the best variant.
6. Transaction-cost sensitivity sweep at 0 / 5 / 10 / 20 bps (U6).
7. Walk-forward robustness check: re-fit on an expanding window and evaluate on the
   following year, for each year of the test period.

**Figures**
- Cumulative net equity curves: best RL variant (median seed) versus all benchmarks.
- Underwater (drawdown) plot for the same set.
- Allocation-over-time stacked area for the RL agent, overlaid on SPY drawdown (U5).
- Action-frequency bar chart, RL agent versus random policy.
- Training diagnostics: episodic reward, TD loss, mean Q, ε — mean ± band across seeds.
- Seed-dispersion box plots for Sharpe, CAGR and max drawdown, by variant.
- Cost-sensitivity curve: net Sharpe versus fee in bps.
- Optuna parameter-importance plot.

**Acceptance.** `artifacts/results/final_scorecard.csv` contains every metric from
proposal Table 10 for every strategy, with seed dispersion. Test-set evaluation appears
exactly once in the code path and is asserted as such.

---

### Phase 8 — Notebooks

**Goal.** The submitted Jupyter artifact — a readable narrative over tested code.

| Notebook | Contents |
|---|---|
| `01_data_eda.ipynb` | Universe rationale, download/cache, quality report, causal feature construction with a worked leakage demonstration, split visualisation, EDA figures |
| `02_env_benchmarks.ipynb` | Environment design walkthrough, `check_env`, the golden constant-action test executed live, benchmark scorecard |
| `03_dqn_training.ipynb` | Network and replay buffer, one full training run with live diagnostics, Double/Dueling explained and demonstrated |
| `04_tuning.ipynb` | Coarse grid results, Optuna study, importance plots, selected configuration with justification |
| `05_results_ablation.ipynb` | Test-set evaluation, full scorecard, ablation, significance testing, every report figure, interpretation against proposal Section 4.9 |
| `00_run_all.ipynb` | Executes the pipeline end to end from the cached data; the grader's single entry point |

**Convention.** Notebooks contain narrative, orchestration and figures. Algorithms live
in `src/portfoliorl/`. Every notebook is executed top-to-bottom and committed **with
outputs** so results are visible without re-running.

**Acceptance.** `00_run_all.ipynb` runs clean from a fresh kernel using only the cached
data, with no network access required.

---

### Phase 9 — Report, Documentation, Packaging

**Goal.** Satisfy the Results (7), Experience Statement (1) and Documentation (3) criteria.

**Tasks**
1. Update `Docs/PortfolioRL_ProjectAssignment3.md` per decision A8 — apply R1–R11,
   fix D1–D5, renumber tables, add the new references.
2. Write `Docs/PortfolioRL_FinalReport.md`:
   - Section 5, Results — full scorecard, all figures, seed dispersion, significance
     testing, ablation, cost sensitivity.
   - Section 6, Discussion — regime behaviour, cost/turnover trade-off, where the agent
     beats the benchmarks and where it does not, applying the decision criteria of
     proposal Section 4.9 explicitly (the preferred algorithm is *not* simply the
     highest-return one).
   - Section 7, Limitations — single historical price path; overlapping training
     episodes are not independent samples; no slippage, market-impact, tax or borrowing
     model; four-asset ETF universe with survivorship-free but limited history; discrete
     action menu constrains attainable allocations; results are period-specific.
   - Section 8, Experience Statement — a candid account of what was hard: the reward
     double-counting bug and how it was diagnosed, the reward-scaling gradient problem,
     the drift-adjusted turnover subtlety, the episode-design change that made learning
     work, the Windows-ARM64 PyTorch installation obstacle, and what would be done
     differently (continuous actions with PPO/DDPG, walk-forward retraining).
   - Section 9, Future Work — continuous action space, regime-conditioned reward,
     larger universe, distributional RL.
   - Section 10, Reproducibility — exact environment setup, seeds, run commands.
3. Rewrite `README.md` — overview, results headline, repository map, setup, how to
   reproduce, artifact inventory, scope disclaimer.
4. `package_submission.py` — assemble `submission/` containing notebooks (with outputs),
   `src/`, `tests/`, `data/` (frozen cache), `artifacts/results/`, `artifacts/figures/`,
   best model checkpoints, `requirements.txt`, `README.md`, and both documents; then zip.

**Acceptance.** The zip unpacks to a folder in which `00_run_all.ipynb` executes offline.
Every metric named in proposal Table 10 appears in the report. The experience statement is
specific and technical rather than generic.

---

### Phase 10 — Video Presentation

**Goal.** A 20-minute maximum recording covering the five points the assignment lists.

**Timed outline** (`Docs/video_outline.md`), 19 minutes leaving buffer:

| Time | Segment | Content |
|---|---|---|
| 0:00–1:30 | Problem and business goal | Why static 60/40 and calendar rebalancing under-adapt; what adaptive allocation is worth |
| 1:30–4:30 | RL formulation | MDP tuple; state, six-action menu, reward with its three penalties; why RL rather than supervised prediction |
| 4:30–8:00 | Algorithm walkthrough | Bellman optimality → tabular Q-learning → why the continuous state forces function approximation → DQN; replay buffer and target network; Double and Dueling extensions |
| 8:00–10:30 | Evaluation metrics | The balanced scorecard; why Sharpe rather than raw return is the selection criterion; chronological splitting and the single test-set rule |
| 10:30–14:30 | Code walkthrough | `env.step()` line by line — drift, turnover, cost, reward terms; the golden constant-action test; the training loop and eval callback |
| 14:30–17:30 | Results | Scorecard versus benchmarks, equity and drawdown curves, allocation-over-time regime behaviour, ablation, seed dispersion, cost sensitivity |
| 17:30–19:00 | Conclusions | What worked, what did not, limitations, future work |

**Tasks.** Write the outline with speaker notes, export the figures needed as slides,
rehearse against the clock, record.

---

## 8. Test Strategy

`pytest` suite run before every results-generating notebook execution.

| File | Focus | Key assertions |
|---|---|---|
| `test_features.py` | No look-ahead | Feature at *t* is invariant to future prices; scaler fit on train only; no residual NaNs; splits contiguous and non-overlapping |
| `test_env.py` | Accounting correctness | Golden constant-action equity-curve test; weights sum to 1; positive wealth; drift-adjusted turnover; cost monotonicity in fee; drawdown bounds and reset; reward equals sum of `info` components; seed determinism; no index overrun |
| `test_metrics.py` | Metric correctness | Closed-form checks on constructed series; known-drawdown series; self-comparison edge cases; PSR = 0.5 at the benchmark Sharpe |
| `test_agent.py` | Learning machinery | Overfit-a-toy-window test; Double DQN target ≤ vanilla target on average; dueling recombination identity; seed reproducibility; Q-value bound |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Agent fails to beat benchmarks out of sample | High | Medium | This is a legitimate scientific result, not a project failure. The report is structured to present it honestly with significance testing (U2, U3). The rubric rewards reporting *all* performance measures and interpreting them, not beating the market. Plan the discussion for this outcome from the start. |
| Overfitting to the single historical path | High | High | Random-start episodes (A6), multi-seed averaging, strict validation-only model selection, walk-forward robustness check, DSR adjustment for trial count |
| Reward-shaping weights dominate the return signal | Medium | Medium | λ₁–λ₃ tuned jointly with RL hyperparameters (proposal Table 6); report the reward-component decomposition from `info` so their relative contributions are visible |
| Training instability / diverging Q-values | Medium | Medium | Huber loss, gradient clipping, target network, reward scaling (R3), explicit Q-magnitude diagnostic in the acceptance criteria |
| `yfinance` API change or data revision mid-project | Medium | High | Frozen on-disk cache committed as an artifact (U7); network access needed only once |
| Torch unavailable / ARM64 build issues | Resolved | High | Verified working via the official CPU index; commands recorded in [Section 5](#5-verified-technical-environment) |
| Scope creep from the four-variant ablation | Medium | Low | PER already dropped (A3); variants share one training loop behind two boolean flags |
| Test-set leakage through repeated evaluation | Low | Critical | Test evaluation exists at exactly one point in the code path (Phase 7 step 2) and is asserted; all tuning uses the validation window only |

---

## 10. Definition of Done

- [ ] All `pytest` suites pass, including the golden constant-action environment test.
- [ ] `00_run_all.ipynb` executes clean from a fresh kernel, offline, on the cached data.
- [ ] Every metric in proposal Table 10 is reported for all four RL variants and all five
      benchmarks, with mean/median/std/min/max across 8 seeds.
- [ ] Test-window evaluation occurs exactly once in the code path.
- [ ] Block-bootstrap CIs, PSR and DSR reported for the headline result.
- [ ] All eight report figures generated and referenced in the text.
- [ ] `Docs/PortfolioRL_ProjectAssignment3.md` updated per A8 with tables renumbered.
- [ ] `Docs/PortfolioRL_FinalReport.md` complete, including the Limitations and
      Experience Statement sections.
- [ ] `README.md` documents exact reproduction steps including the ARM64 torch command.
- [ ] `submission.zip` builds and unpacks to a working folder.
- [ ] `Docs/video_outline.md` written and the video recorded within 20 minutes.

---

## 11. Additional References to Add

To be appended to Section 5 of the design document, in support of the rigor upgrades:

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for
  selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio
  Management*, 40(5), 94–107. — supports U3.
- Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2016). Deep direct reinforcement
  learning for financial signal representation and trading. *IEEE Transactions on Neural
  Networks and Learning Systems*, 28(3), 653–664. — related work on RL for trading.
- Fischer, T. G. (2018). *Reinforcement learning in financial markets — a survey*
  (FAU Discussion Papers in Economics No. 12/2018). — positions the project in the
  literature.
- Liu, X.-Y., Yang, H., Gao, J., & Wang, C. D. (2021). FinRL: Deep reinforcement learning
  framework to automate trading in quantitative finance. *Proceedings of the Second ACM
  International Conference on AI in Finance.* — comparable open-source framework.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4),
  36–52. — supports the √252 annualisation caveat (R10).
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the
  American Statistical Association*, 89(428), 1303–1313. — supports U2.

---

## Appendix A — Change Log Against Assignment 3

| Assignment 3 statement | Revised statement | Reference |
|---|---|---|
| Reward penalises drawdown and volatility levels | Reward penalises the drawdown *increment*; volatility remains a level | R1 |
| Reward uses arithmetic portfolio return | Reward uses log return, scaled ×100 | R2, R3 |
| Turnover = change in target weights | Turnover = target weights minus *drifted* weights | R4 |
| Cost appears in the reward | Cost also deducted from portfolio value | R5 |
| Episode = full backtesting period | Training episodes = random-start, 52-decision sub-episodes | R6 |
| Daily or weekly rebalancing steps | Weekly decisions; daily return accounting | R7 (A1) |
| State omits portfolio volatility | State adds portfolio volatility and drawdown duration | R8 |
| 2010–2019 / 2020–2021 / 2022–2024 | 2004-11-18–2017 / 2018–2020 / 2021–2025 | R9 (A2) |
| `r_f` and MAR unspecified | `r_f` = `^IRX` daily; MAR = 0 | R10 |
| Benchmark cost treatment unstated | All benchmarks run through the same env and fee model | R11 |
| §3.3 DQN + Double; §4.6 also Dueling + PER | DQN + Double + Dueling + Double&Dueling; PER dropped | D1 (A3) |
| Tables 6 → 10 | Tables renumbered contiguously | D2, D3 |
| Information ratio, benchmark unnamed | IR reported against each benchmark separately | D4 |
| "SHY or BIL" | SHY (BIL post-dates the new start date) | D5 |
