# Project Assignment 4 — Progress Report

## *PortfolioRL: Reinforcement Learning for Dynamic Portfolio Rebalancing*

> **Scope.** This document reports **progress**: what was implemented, what results came
> out, and what remains. The problem formulation, MDP specification, algorithm justification
> and metric definitions were submitted as Assignment 3 and are unchanged; Section 1 recalls
> them briefly, and `Docs/PortfolioRL_ProjectAssignment3.md` holds the full treatment. Every
> number below is reproduced from a committed artefact, named next to the table it appears in.

---

## 1. Introduction

### What the project asks

A long-term investor holding several asset classes must periodically determine the portfolio
weights to hold. Conventional benchmark policies, such as a fixed 60/40 equity-bond mix,
apply the same allocation rule regardless of prevailing market conditions. **PortfolioRL
examines whether a reinforcement learning agent that observes market conditions and adjusts
its allocation can improve on these benchmark policies after transaction costs are charged.**

The problem is posed as a Markov Decision Process and solved with a Deep Q-Network:

| Element | Specification |
|---|---|
| Universe | SPY (US equity), TLT (long Treasuries), GLD (gold), SHY (cash proxy) |
| State | 31 dimensions — 23 market features (momentum, volatility, drawdown, yield-curve level and slope) + 8 portfolio features (current weights, drift, time since rebalance) |
| Action | 6 discrete target allocations, from defensive to equity-heavy |
| Reward | Differential Sharpe ratio, scaled ×100, penalised by realised drawdown and by the cost of trading |
| Cadence | Weekly decisions, daily mark-to-market accounting |
| Agent | DQN with a 31→128→64→6 Q-network (~12,800 parameters), replay buffer, target network, ε-greedy exploration |
| Cost model | 10 bps per unit turnover, charged on every trade |

The full derivation — why an MDP, why value-based RL rather than policy-gradient, why the
differential Sharpe reward, and how each metric is defined — was submitted as Assignment 3
and is not repeated here.

### Where the project stands

Everything planned in Assignment 3 has been built and run. The table below maps each stage
of the pipeline to the artefact that proves it executed.

| Stage | Status | Evidence |
|---|---|---|
| Data acquisition and feature engineering | Complete | `data/processed/`, notebook 01 |
| Environment, cost model, benchmark suite | Complete | notebook 02, `02_benchmark_scorecard_test.csv` |
| DQN training and reward-scale study | Complete | notebook 03, `03_*.csv` |
| Hyperparameter search (grid + Optuna) | Complete | notebook 04, `04_*.csv/json` |
| Evaluation, ablation, significance, robustness | Complete | notebook 05, `05_*.csv/json` |
| Test split opened | Once, at the end | `05_summary.json` |

### The headline finding

The principal positive result comes from the walk-forward analysis. When the agent is
retrained annually, it attains a mean raw Sharpe ratio of approximately **1.00**, compared
with 0.387 for the train-once policy, and exceeds weekly rebalanced 60/40 in four of five
annual folds. This includes 2022, when the agent returned -17.1% versus -23.4% for 60/40.
Because this analysis was conducted after the held-out test period was examined, it is
exploratory; nevertheless, it motivates periodic refitting as the primary next experiment.

Under the primary train-once evaluation protocol, the agent does not outperform the static
benchmarks: its excess Sharpe ratio is 0.141, compared with 0.380 for 60/40 buy-and-hold and
0.700 for SPY. No pairwise comparison remains significant after Holm correction. In addition,
the Deflated Sharpe Ratio is **0.238**, indicating that the selected configuration does not
provide sufficient evidence of skill after accounting for the 72 configurations considered
during development. The validation Sharpe of 1.987 reported in Section 5 therefore did not
generalise to the held-out test period.

Three diagnostics clarify the limits of the primary evaluation:

1. **Seed variance dominates the estimated effects.** The within-cell seed standard deviation
   is 0.131 Sharpe, larger than the tuning effect (+0.052) and the training-budget effect
   (−0.065) combined. Single-seed comparisons are therefore insufficient for estimating
   these effects reliably.
2. **Validation performance does not predict test performance.** The correlation is
   **−0.365** in this experiment, indicating that the observed validation ranking did not
   provide a reliable basis for selecting a final configuration.
3. **The test sample is short relative to the estimated track-record requirement.**
   Establishing this Sharpe ratio at 95% confidence requires approximately 4,500 days,
   whereas the test split contains 1,254 days.

 Taken together, these findings indicate that regime change, seed sensitivity, and model
 selection are important limitations of the current evaluation. Section 8 maps each limitation
 to a concrete follow-up experiment, with the objective of improving performance against the
 diversified static benchmarks.

Sections 3–7 present the evidence for each of these claims; Section 8 sets out what remains
before the final submission.

---

## 2. Implementation status

What remains is writing and presentation, not computation.

### Packages and scripts

| Layer | Contents |
|---|---|
| **Stack** | Python 3.12, PyTorch 2.13 (CPU), Gymnasium, NumPy / pandas / SciPy, Optuna, yfinance, Matplotlib |
| **`src/portfoliorl/`** | `config` · `data` · `features` · `env` · `metrics` · `benchmarks` · `agent` · `train` · `tuning` · `significance` · `experiments` · `plots` — 12 modules |
| **`notebooks/`** | `00_run_all` (provenance) · `01_data_eda` · `02_env_benchmarks` · `03_dqn_training` · `04_tuning` · `05_results_ablation` |
| **`tests/`** | 8 modules, **135 passing tests**, including golden tests that lock the environment's accounting invariants |
| **`tools/`** | `package_submission.py` — allow-list archive builder |

All algorithmic logic resides in `src/portfoliorl/`; the notebooks execute, document, and
visualise the package-level implementation.

### Dataset

Four ETFs from yfinance — SPY, TLT, GLD, SHY — daily, **2004-11-18 to 2025-12-31** (GLD's
inception is the first date on which all four exist). Cached to `data/processed/` as CSV
with the fitted scaler persisted alongside.

| Split | Window | Days | Role |
|---|---|---|---|
| Train | 2004-11-18 → 2017-12-31 | 3,102 | Weight updates only. Contains 2008. |
| Validation | 2018-01-01 → 2020-12-31 | 756 | Hyperparameters and checkpoint selection. Contains COVID. |
| Test | 2021-01-01 → 2025-12-31 | 1,255 | **Opened once**, in notebook 05. Contains the 2022 rate shock. |

The scaler is fit on training data only. A 120,000-step run takes **~5 minutes on CPU**; the
full pipeline takes about three hours, dominated by notebook 05's 51 training runs.

---

## 3. Results obtained

Headline model: the tuned configuration (Double + duelling, lr 6.71e-4, γ = 0.983,
λ_drawdown = 0.179), 120,000 steps, seed 0, checkpoint selected on validation Sharpe.

![Out-of-sample growth of $100,000, 2021-2025](../artifacts/figures/05_01_equity_curves.png)

***Figure 1.*** *Out-of-sample equity curves. All ten strategies execute in the same
environment and pay the same 5 bps per unit of turnover. The RL agent is the heavy black
line — eighth of ten.* (`05_01_equity_curves.png`)

**Test-split scorecard, excess Sharpe basis** (13-week T-bill, mean 3.21% over the window).
Source: `artifacts/results/05_test_scorecard.csv`.

| Strategy | Sharpe | CAGR | Vol | Max DD | Final wealth | Ann. turnover |
|---|---|---|---|---|---|---|
| 100% SPY | 0.700 | 14.71% | 17.10% | 24.50% | $197,887 | 0.20× |
| Trend following | 0.567 | 9.40% | 11.30% | 15.25% | $155,503 | 1.13× |
| Equity-heavy 80/20 | 0.520 | 10.09% | 14.25% | 25.60% | $161,035 | 0.38× |
| Equal weight | 0.399 | 6.35% | 8.24% | 18.61% | $135,561 | 0.44× |
| 60/40 buy & hold | 0.380 | 7.71% | 13.46% | 26.99% | $144,586 | 0.20× |
| Volatility target | 0.236 | 4.96% | 8.41% | 19.48% | $122,444 | 7.73× |
| 60/40 rebalanced | 0.230 | 5.45% | 12.44% | 27.33% | $129,889 | 0.47× |
| **RL (DQN)** | **0.141** | **4.29%** | **13.03%** | **26.45%** | **$117,598** | **9.43×** |
| Random (30 seeds) | −0.047 | 2.23% | 10.32% | 27.25% | $99,493 | 23.05× |
| All cash (SHY) | −0.812 | 1.63% | 1.95% | 5.71% | $108,397 | 0.00× |

Also recorded: hit rate 52.5%, Sortino 0.558, Calmar 0.162, daily 95% CVaR 1.94%, longest
drawdown 885 days, action entropy 1.465 nats of a possible 1.792 — so the policy uses its
action set rather than collapsing onto one allocation.

The train-once agent exceeds the random policy and cash on the excess-Sharpe basis, but trails
the remaining static strategies. It assumes volatility similar to 60/40 buy-and-hold while
delivering a lower return. Two observations help explain this result:

1. **Cost drag is real but secondary.** 9.43× annual turnover consumed 4.69% of terminal
   wealth (≈ 0.94%/yr), roughly 45% of the 2.06-point CAGR gap to equal weight. Adding it
   wealth (approximately 0.94% per year), roughly 45% of the 2.06-point CAGR gap to equal
   weight. Even without this cost, the agent remains below every static allocation. Allocation
   selection is therefore the primary observed limitation, rather than transaction cost alone.
2. **Limited defensive reallocation.** Maximum drawdown was 26.45%, within one percentage
   point of 60/40 buy-and-hold. The allocation trace (`05_04_allocation_over_time.png`) also
   shows an equity allocation near 80% through the 2022 drawdown.

---

## 4. Is the result distinguishable from noise?

![Block-bootstrap significance](../artifacts/figures/05_10_bootstrap_significance.png)

***Figure 2.*** *Stationary block bootstrap (Politis & Romano, 1994), 2,000 resamples, mean
block 10 days, paired via a shared index draw. Every interval straddles zero; nothing is
significant after Holm–Bonferroni.* (`05_10_bootstrap_significance.png`)

The smallest unadjusted *p*-value is for the comparison with 100% SPY (Δ = −0.501,
*p* = 0.053 unadjusted; 0.473 after Holm adjustment). Its negative point estimate favours
SPY, but it is not statistically significant at the specified family-wise error rate. Source:
`artifacts/results/05_significance.csv`.

> **Reading note.** `significance.csv` is on the *raw* basis, so its all-cash row shows the
> agent losing to cash (raw cash Sharpe +0.836, because the raw ratio does not subtract the
> risk-free rate that cash *is*). On the excess basis of Section 3 the agent beats cash by
> 0.95. The two bases are never comparable, so every Sharpe in this report states which one
> it is on.

| Diagnostic | Value | Reading |
|---|---|---|
| Test Sharpe — excess / raw | 0.141 / 0.387 | The excess basis is used for the Section 3 scorecard; inference statistics use raw total returns. |
| Lo (2002) analytic SE (raw) | 0.448 | The estimate is smaller than its serial-correlation-adjusted standard error. |
| Probabilistic Sharpe vs 0 (raw) | 0.807 | Before correcting for the search process, there is 80.7% probability that the raw Sharpe exceeds zero. |
| Configurations evaluated | 72 | 18 grid + 30 Optuna + 24 ablation |
| **Expected max Sharpe from noise** | **0.705** | The expected maximum under the multiple-testing adjustment. |
| **Deflated Sharpe Ratio** | **0.238** | After accounting for search intensity, the observed raw Sharpe is below the adjusted benchmark. |
| Minimum track record | 4,500 days | Needed to establish the effect at 95% |
| Test split length | 1,254 days | **3.6× short** |

Source: `artifacts/results/05_summary.json` → `inference`.

> **Metric-basis reconciliation.** The scorecard reports excess Sharpe ratios, whereas the
> serial-correlation and multiple-testing diagnostics are computed from raw total returns.
> The report therefore labels the basis for every Sharpe ratio and does not use the previously
> reported PSR comparison with 60/40, which mixed the two bases. The final notebook run will
> persist consistently labelled raw and excess values for both the agent and each benchmark.

---

## 5. Ablation and hyperparameter tuning

![Seed dispersion across the 2x2 ablation](../artifacts/figures/05_06_seed_dispersion.png)

***Figure 3.*** *Six matched seeds per variant — seed k of one variant sees the same
initialisation and episode draws as seed k of every other, so the comparison is paired. The
within-variant spread swamps every between-variant gap.* (`05_06_seed_dispersion.png`)

| Variant | Test Sharpe (raw, 6 seeds) | vs Double+Duelling: Cohen's *d* | *p* |
|---|---|---|---|
| Double DQN | 0.628 ± 0.173 | −0.29 | 0.505 |
| Duelling DQN | 0.604 ± 0.207 | −0.19 | 0.669 |
| Double + Duelling | 0.554 ± 0.133 | — | — |
| Vanilla DQN | 0.519 ± 0.256 | +0.10 | 0.821 |

Sources: `05_ablation.csv`, `05_ablation_paired_tests.csv`. No pairwise contrast is
statistically significant and every |*d*| is below 0.3. Within this sample, the four variants
are not distinguishable. With approximately 620 training decisions per pass and six actions,
the experiment provides no evidence that overestimation-bias correction is the limiting
factor.

![Tuning versus budget](../artifacts/figures/05_08_tuning_vs_budget.png)

***Figure 4.*** *Left: a 2×2 factorial over {default, tuned} × {60k, 120k steps}, three
matched seeds per cell. Right: validation Sharpe against test Sharpe for all twelve runs.*
(`05_08_tuning_vs_budget.png`)

The two-stage search (18-point grid, then 30 Optuna TPE trials with median pruning — 19
pruned, 11 completed) improved validation performance: refitting at full budget increased
validation Sharpe from 0.992 to **1.677** and reduced validation drawdown from 19.5% to
11.7%. These gains did not transfer to the test period.

| Effect (excess Sharpe) | Magnitude |
|---|---|
| Tuning (tuned − default) | +0.052 |
| Budget (120k − 60k) | −0.065 |
| Interaction | −0.076 |
| **Mean within-cell seed sd** | **0.131** |
| **Validation/test Sharpe correlation** | **−0.365** |

Each estimated main effect is smaller than the within-cell seed standard deviation. The
120,000-step condition has a lower mean than the 60,000-step condition in this experiment,
and validation and test Sharpe ratios are negatively correlated. These results suggest that
the selected headline seed (raw Sharpe 0.387) is not representative of the pooled ablation
means (0.52-0.63).

---

## 6. Robustness

![Walk-forward retraining](../artifacts/figures/05_12_walk_forward.png)

***Figure 5.*** *Five annual test folds, retrained before each with a two-year validation
buffer, three seeds per fold. Error bars are seed standard deviations; blue is
weekly-rebalanced 60/40 over the same fold.* (`05_12_walk_forward.png`)

| Test year | RL Sharpe (raw) | RL CAGR | Max DD |
|---|---|---|---|
| 2021 | 1.959 ± 0.666 | 18.24% | 6.01% |
| 2022 | −0.994 ± 0.305 | −17.12% | 23.96% |
| 2023 | 1.177 ± 0.378 | 12.63% | 10.70% |
| 2024 | 1.357 ± 0.365 | 14.37% | 5.56% |
| 2025 | 1.505 ± 0.383 | 16.76% | 9.14% |
| **Mean** | **≈ 1.00** | ≈ 8.98% | 11.07% |

The walk-forward analysis is the most decision-relevant robustness result. With annual
retraining, the agent averages a raw Sharpe near 1.00 versus 0.387 for the train-once model,
and exceeds weekly rebalanced 60/40 in four of five folds. In 2022, it returned -17.1%
compared with -23.4% for 60/40. These results are consistent with the hypothesis that a
frozen 2017 training cut-off is vulnerable to subsequent regime change; they do not establish
causality or a deployable performance claim.

Two limitations remain. The 2022 result is still an absolute loss of 17%, and turnover was
lowest in that year (0.155), indicating limited defensive reallocation. In addition, this
analysis was performed after the original test split was opened. It is therefore exploratory
and requires confirmation on an unseen holdout period.

**Cost sweep** (retrained, not re-scored, at 0/5/10/20 bps, three seeds each;
`05_cost_sweep.csv`): turnover falls monotonically from 0.284 to 0.184 per decision as costs
rise, indicating that the policy responds to the transaction-cost signal. Sharpe is
non-monotonic (0.520 → 0.739 → 0.627 → 0.537 raw), and the three-seed design does not support
a precise conclusion about the performance-cost relationship. The policy remains operational
at 20 bps, but the sweep supports no stronger robustness claim.

---

## 7. Discussion — what the evidence supports

| Claim | Verdict |
|---|---|
| The agent beats the rule-based benchmarks on risk-adjusted return | **No.** Eighth of ten on excess Sharpe. |
| Any benchmark difference is statistically significant | **No.** Nothing survives Holm–Bonferroni. |
| The headline Sharpe survives correction for search intensity | **No.** DSR 0.238; noise expectation 0.705 > observed 0.387. |
| The test window is long enough to detect the effect | **No.** 1,254 days against a 4,500-day requirement. |
| Double Q-learning / duelling heads help here | **Not demonstrated.** All \|*d*\| < 0.3, all *p* > 0.5. |
| Tuning transfers from validation to test | **No.** Correlation −0.365. |
| The agent responds economically to transaction costs | **Yes.** Turnover 0.284 → 0.184 as costs rise 0 → 20 bps. |
| Periodic retraining beats train-once | **Suggestive, exploratory.** Mean raw Sharpe ≈ 1.00 vs 0.387. |

The evidence does not support a claim that the train-once agent is superior to the benchmark
strategies. The contribution of this work is a complete, tested, and reproducible RL
evaluation pipeline that records the sources of uncertainty rather than treating a favourable
validation result as conclusive. The observed performance gap is associated with three
testable mechanisms: allocation choices that were poorly defensive during 2022, transaction
costs arising from high turnover, and a validation signal that did not transfer to the test
period. The walk-forward analysis further suggests that the fixed training cut-off may be
important.

These mechanisms motivate the follow-up experiments in Section 8. They are hypotheses to be
tested on a new, prespecified evaluation protocol; they are not claims of achieved
outperformance.

---

## 8. Remaining work

### Deliverables

| Item | Weight | Status |
|---|---|---|
| Results discussion, all performance measures | 7 pts | Data complete; 10-section draft at `Docs/PortfolioRL_FinalReport.md` |
| Experience statement | 1 pt | Draft exists as §8 of the final report; needs expanding |
| Video presentation (≤ 20 min) | 4 pts | Not started — slides to be built from the 52 committed figures |
| Documentation and zipped submission | 3 pts | `package_submission.py` builds and verifies it |

### Corrections before the final submission

1. **Persist Sharpe ratios on a consistent basis** in the `inference` block of `_build_nb05.py`
   (see Section 4). The scorecard uses excess returns while the inference statistics use raw
   returns. The final notebook run will store both bases explicitly for every strategy.
2. The legend in `05_04_allocation_over_time.png` overlaps its title and needs a layout fix.

### Improvements targeted at beating the static benchmarks

Each diagnostic in this report points to a specific improvement experiment. The interventions
are prioritised by their direct connection to observed limitations. Their effect on benchmark
performance remains an empirical question and will be evaluated on data not used for model
selection.

| # | Proposed change | Diagnostic addressed | Evaluation criterion | Scope |
|---|---|---|---|---|
| 1 | **Evaluate periodic retraining** rather than a single train-once policy | Regime staleness (§6) | Confirm the walk-forward result on a new, prespecified holdout | Exploratory result exists; confirmation is required |
| 2 | **Ensemble independently seeded agents** by averaging Q-values or voting on actions | Seed sd 0.131 exceeds the estimated effects (§5) | Compare ensemble mean and dispersion with single-seed policies | Feasible as a follow-up experiment |
| 3 | **Use median-across-seeds or purged cross-fold model selection** rather than maximum validation Sharpe | Validation/test correlation of −0.365 (§5) | Assess whether selection stability and out-of-sample ranking improve | Feasible as a follow-up experiment |
| 4 | **Reduce turnover** with a no-trade band, switching penalty, or lower decision frequency | 9.43× annual turnover and 4.69% terminal-wealth cost (§3) | Measure turnover, net return, and drawdown jointly | Feasible with an environment revision and retraining |
| 5 | **Extend the state and risk specification** with additional regime features and a state-dependent drawdown penalty | Limited defensive reallocation during 2022 (§3) | Evaluate whether allocations and drawdown behaviour improve in adverse regimes | Beyond the current two-week window |
| 6 | **Broaden the evaluation** through a new holdout period or a second asset universe | 1,254 days versus a 4,500-day estimated requirement (§4) | Re-estimate uncertainty on additional independent observations | Data- and time-dependent |

The appropriate short-term objective is to determine whether periodic retraining, more stable
model selection, and lower turnover can improve on the diversified benchmarks, particularly
60/40 and equal weight. The evidence does not justify forecasting a specific Sharpe ratio or
claiming that these changes will outperform most benchmarks. SPY's 0.700 excess Sharpe over
this predominantly favourable equity period remains a demanding reference point for an
unlevered, diversified allocator.

Any experiment conducted after the original test split was opened will be identified as
exploratory until it is confirmed on data not used during development.

**Schedule.** Days 1–5 finalise the report against the CSVs · days 6–8 experience statement,
documentation, archive verification · days 9–12 slide deck and recording · days 13–14 buffer.

---

## 9. Artefacts submitted with this report

| Group | Contents |
|---|---|
| Notebooks | 6 `.ipynb`, all executed, all committed with outputs |
| Source | 12 modules in `src/portfoliorl/` |
| Tests | 8 modules, 135 passing tests |
| Results | 17 CSV/JSON tables in `artifacts/results/`, plus per-seed learning curves in `curves/` |
| Figures | 52 PNGs in `artifacts/figures/` |
| Models | 4 checkpoints, including `05_headline_seed0.pt` |
| Data | Processed dataset, persisted scaler, download manifest |
| Documents | This report, `PortfolioRL_ProjectAssignment3.md`, `PortfolioRL_FinalReport.md` (draft), `PortfolioRL_ImplementationPlan.md`, `README.md` |

---

## References

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *The Journal of
Portfolio Management*, 40(5), 94–107. · Faber, M. T. (2007). A quantitative approach to
tactical asset allocation. *The Journal of Wealth Management*, 9(4), 69–79. · Lo, A. W.
(2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52. ·
Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*,
518(7540), 529–533. · Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
*JASA*, 89(428), 1303–1313. · van Hasselt, H., Guez, A., & Silver, D. (2016). Deep
reinforcement learning with double Q-learning. *AAAI*, 30(1). · Wang, Z., et al. (2016).
Duelling network architectures for deep reinforcement learning. *ICML*.

*Full reference list in `Docs/PortfolioRL_ProjectAssignment3.md` §6.*
