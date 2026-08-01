# Dynamic Multi-Asset Portfolio Rebalancing with Deep Q-Networks

**Final project report — Reinforcement Learning, Penn State University**

> **Not investment advice.** This is a course project on a five-year out-of-sample
> window. Nothing here is a recommendation to trade, and the central conclusion is
> that the results are statistically indistinguishable from chance.

---

## Abstract

I trained a Double+Duelling DQN to choose weekly among six fixed allocations of
SPY, TLT, GLD and SHY, with daily portfolio accounting and a 5 bps transaction
cost charged against both wealth and reward. On a held-out 2021–2025 test window
the agent earned an excess Sharpe of **0.141** against **0.700** for buy-and-hold
SPY and **0.230** for a rebalanced 60/40 portfolio, finishing eighth of ten
strategies.

The more useful finding is what the statistics say about that number. A
stationary block bootstrap cannot reject equality between the agent and *any* of
the nine benchmarks after multiplicity correction — including the random-allocation
floor. Across 72 configurations evaluated during development, the expected maximum
Sharpe attributable to noise alone is **0.705**, and the minimum track record
length needed to establish skill is **4,500 trading days** against the **1,254**
available. A four-cell factorial further shows that neither the tuned
hyperparameters nor the training budget explains the headline number: **seed
variance dominates every effect I set out to measure**, with a single
configuration spanning raw Sharpe 0.317 to 0.770 across three seeds.

The honest conclusion is not "RL underperforms." It is that a five-year test
window carries far too little statistical power to support any claim about this
agent, and that reporting a single-seed headline — as I initially did — was itself
the methodological error.

---

## 1. What was built

Six notebooks, a `src/portfoliorl/` package, and 135 tests.

| Notebook | Purpose | Figures |
|---|---|---|
| `00_run_all` | Pipeline map, provenance, reproducibility assertion, manifest | 2 |
| `01_data_eda` | Data, features, leakage protocol | 14 |
| `02_env_benchmarks` | MDP, environment golden tests, nine benchmarks | 8 |
| `03_dqn_training` | Agent architecture, training dynamics | 10 |
| `04_tuning` | Coarse grid then TPE search | 7 |
| `05_results_ablation` | Test evaluation, ablation, significance, robustness | 13 |

**Problem formulation.** State is 31-dimensional (23 market features + 8 portfolio
features). Action is one of six allocations. Decisions are weekly; accounting is
daily. Reward is

$$r_t = 100\left[\log(1 + R_t^{\text{net}}) - \lambda_1 \tau_t - \lambda_2 \sigma_t - \lambda_3 \max(0, \Delta \text{DD}_t)\right]$$

with turnover measured against *drifted* weights, so the agent is not charged for
drift it did not cause. Full derivation in `Docs/PortfolioRL_ProjectAssignment3.md`.

**Data.** SPY/TLT/GLD/SHY daily closes, 2004-11-18 to 2025-12-31 (5,313 rows, zero
missing), split train 3,102 / validation 756 / test 1,255 days. The start date is
GLD's inception. The test window was opened exactly once, in notebook 05.

**A convention that matters.** The scorecard reports Sharpe **net of T-bills**;
the `experiments` module reports it against a **zero rate**. The same agent scores
0.141 excess and 0.387 raw. I state which basis is in use everywhere below,
because conflating them is an easy and consequential mistake — I made it once
mid-project and it inverted a conclusion.

---

## 2. Headline result

Test split, 2021-01-04 to 2025-12-31, excess Sharpe basis
(`05_01_equity_curves.png`, `05_02_underwater.png`).

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
| Random allocation | −0.047 | 2.23% | 10.32% | 27.25% | $99,493 | 23.05× |
| All cash (SHY) | −0.812 | 1.63% | 1.95% | 5.71% | $108,397 | 0.00× |

The agent beats only the random floor and cash. It also carries the **second-highest
turnover in the table at 9.43× per year**, against 0.47× for the rebalanced 60/40 it
fails to beat.

### Turnover is not the whole story

Total transaction costs consumed 4.69% of terminal wealth over five years — roughly
0.94% per year. Adding that back lifts the agent's CAGR from 4.29% to about 5.2%,
still less than half of SPY's 14.71%. **Churn is a real defect but a secondary one;
the dominant problem is allocation choice, not execution drag**
(`05_13_return_waterfall.png`).

The transaction-cost sweep supports this. Raising costs from 0 to 20 bps cuts mean
turnover from 0.284 to 0.184 — the turnover penalty is doing its job — but test
Sharpe moves non-monotonically (0.520 → 0.739 → 0.627 → 0.537 raw, seed SDs of
0.09–0.42). The cost level is not what determines performance
(`05_11_cost_sensitivity.png`).

---

## 3. The central finding: seed variance dominates

The headline agent is one seed. That turns out to matter more than every design
choice I studied.

**Four-cell factorial**, crossing tuned-versus-default hyperparameters with a
60k-versus-120k step budget, three matched seeds each, raw Sharpe
(`05_08_tuning_vs_budget.png`):

| | 60k steps | 120k steps |
|---|---|---|
| Default | 0.517 ± 0.151 | 0.461 ± 0.178 |
| Tuned | 0.537 ± 0.085 | 0.495 ± 0.240 |

Isolated effects: budget −0.056 (default) and −0.042 (tuned); tuning +0.021 (at
60k) and +0.034 (at 120k). **Every effect is an order of magnitude smaller than the
seed standard deviation.** A single configuration — tuned at 120k — spans 0.328,
0.387 and 0.770 across seeds 0, 1 and 2.

This refutes two hypotheses I held in sequence. It is not validation-set
overfitting from the hyperparameter search, and it is not overtraining from the
doubled budget. Seed 0 was simply an unlucky draw, and the headline inherited it.

**Architecture ablation**, 2×2 factorial over Double and Duelling, six seeds, 60k
steps, excess Sharpe (`05_06_seed_dispersion.png`, `05_07_ablation_bars.png`):

| Variant | Excess Sharpe | Range across seeds |
|---|---|---|
| Double DQN | 0.338 ± 0.197 | −0.001 to 0.508 |
| Duelling DQN | 0.312 ± 0.232 | −0.096 to 0.591 |
| Double+Duelling | 0.276 ± 0.113 | 0.103 to 0.388 |
| Vanilla DQN | 0.250 ± 0.258 | −0.068 to 0.549 |

Paired tests against Double+Duelling: every Holm-adjusted p-value is **1.00**, all
|Cohen's d| ≤ 0.29, n = 6. **No detectable effect from either algorithmic
refinement.** Given seed ranges that routinely straddle zero, this is unsurprising —
the experiment is badly underpowered to detect effects of the size these
refinements plausibly produce.

Note that these ablation means (0.25–0.34 excess) bracket the rebalanced 60/40
benchmark at 0.230 and sit below equal weight at 0.399. The headline's 0.141 is at
the low end of this distribution but well within one standard deviation of it.

---

## 4. Statistical inference

Three hazards were pre-committed in the design document and all three bite.

**Stationary block bootstrap**, 2,000 resamples with a mean block length of 10
days, RL versus each benchmark (`05_10_bootstrap_significance.png`):

| Comparison | ΔSharpe | 95% CI | p | Holm-adj. |
|---|---|---|---|---|
| vs 100% SPY | −0.501 | [−1.020, −0.006] | 0.053 | 0.473 |
| vs Trend following | −0.464 | [−1.270, 0.292] | 0.242 | 1.000 |
| vs Equal weight | −0.401 | [−0.917, 0.132] | 0.146 | 1.000 |
| vs 60/40 rebalanced | −0.101 | [−0.507, 0.290] | 0.611 | 1.000 |
| vs Random allocation | +0.122 | [−0.518, 0.696] | 0.714 | 1.000 |

**Not one comparison survives multiplicity correction.** The agent is not
statistically distinguishable from buy-and-hold SPY, and equally not
distinguishable from a random allocator. Both statements follow from the same
fact: the test window is too short.

**Deflated Sharpe and track record length.** Across 72 configurations evaluated
during development (18 grid points, 30 search trials, 24 ablation runs), with trial
Sharpe variance 0.085, the expected maximum Sharpe from noise alone is **0.705** —
five times the agent's realised 0.141. Deflated Sharpe is **0.238**; probabilistic
Sharpe against zero is 0.807 but against the 60/40 benchmark only 0.638. Lo's
standard error for a Sharpe estimated on 1,254 daily observations is **0.448**, so
the headline sits well inside one standard error of zero.

Minimum track record length to establish a positive Sharpe at 95% confidence:
**4,500 trading days ≈ 18 years**. I have 1,254.

**This is the report's real result.** The evaluation design cannot support a claim
about this agent in either direction, and the deflated Sharpe machinery predicted
that before the test window was opened.

---

## 5. Robustness: walk-forward

Expanding-window retraining, one model per test year, two validation years
buffered between train and test, three seeds, raw Sharpe
(`05_12_walk_forward.png`):

| Test year | Sharpe | CAGR | Max DD |
|---|---|---|---|
| 2021 | +1.959 ± 0.666 | 18.2% | 6.0% |
| 2022 | −0.994 ± 0.305 | −17.1% | 24.0% |
| 2023 | +1.177 ± 0.378 | 12.6% | 10.7% |
| 2024 | +1.357 ± 0.365 | 14.4% | 5.6% |
| 2025 | +1.505 ± 0.383 | 16.8% | 9.1% |

Four of five years are strongly positive; 2022 is catastrophic. That single year —
the simultaneous stock and bond drawdown — is where the static 2021-trained
headline agent does most of its damage, and it is the one regime absent from the
2004–2017 training data. Periodically retrained agents recover afterwards; the
single fixed agent does not.

I want to be careful not to overclaim here. A mean of yearly Sharpes is not the
Sharpe of the concatenated series, each fold has three seeds, and the folds share
training data. This is suggestive of retraining cadence mattering, not evidence
of it.

---

## 6. What the policy actually does

`05_04_allocation_over_time.png` and `05_05_action_usage.png`. Action entropy is
1.465 nats against a 1.792 maximum, so the agent uses most of its action set rather
than collapsing onto one allocation — a failure mode that killed several early
prototypes and that the naive inverse-volatility risk-parity benchmark exhibits
(it degenerates to ~100% cash, which is why it was cut).

The agent switches allocation far more often than any sensible weekly rebalancer,
which is the source of the 9.43× turnover. Nothing in the reward penalises
*switching frequency* directly — only turnover magnitude — and with six discrete
allocations, a single action change can move a large fraction of the portfolio.

---

## 7. Limitations

1. **The test window is too short**, by a factor of about 3.6 against the minimum
   track record length. Every claim in this report inherits that.
2. **Six discrete allocations** is a coarse action space. Real rebalancing is
   continuous, and discretisation forces large jumps that inflate turnover.
3. **Four assets, one market regime family.** 2021–2025 contains exactly one major
   stock/bond drawdown, and the agent never saw one in training.
4. **Costs are modelled, not measured.** A flat 5 bps with no slippage, market
   impact or spread dynamics flatters any high-turnover strategy.
5. **Single-seed headline.** Corrected in analysis, but the framing error is
   instructive: with seed SDs of 0.09–0.26, one run conveys almost nothing.
6. **No transaction-frequency penalty**, only a magnitude penalty — a likely
   contributor to the churn.

---

## 8. What I would do differently

- Report a **seed distribution, never a point estimate**. Every headline should be
  a median across ≥10 seeds with an interquartile range.
- Choose the test window by **required statistical power**, computed before any
  training, rather than by convenience.
- Add a **switching cost** to the reward, distinct from turnover magnitude.
- Prefer **walk-forward as the primary protocol** over a single static split, given
  what section 5 suggests about regime staleness.
- Treat the **deflated Sharpe as a stopping criterion**: once expected max noise
  Sharpe exceeds the plausible signal, more search is actively harmful.

---

## 9. Reproducibility

Deterministic given a seed — verified by training the same configuration twice in
one process and obtaining identical validation and test Sharpes to six decimal
places. The seeded TPE sampler reproduces exactly: a full re-run of notebook 04
five days later returned the same 30 trials (19 pruned, 11 complete), the same best
trial 28, and the same best value 1.98728.

Notebook 00 asserts end-to-end reproducibility by reloading the committed
checkpoint, re-evaluating it, and comparing against the committed scorecard. All
four metrics match to machine precision (differences of 0.0 to 5.6e-17), and the
notebook prints both the raw and excess Sharpe so the two bases can never be
silently confused again.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook --inplace notebooks/00_run_all.ipynb
```

Every expensive stage caches to `artifacts/` and resumes from a per-run journal, so
an interrupted run costs at most one training run rather than the whole experiment.

---

## 10. Conclusion

A Double+Duelling DQN with a realistic cost model, a tuned configuration and a
carefully leak-proofed evaluation protocol produced an out-of-sample excess Sharpe
of 0.141, below every benchmark except a random allocator and cash.

That number should not be read as a measurement of the method. The bootstrap
cannot separate it from any benchmark, the deflated Sharpe places it far below what
72 configurations of pure noise would be expected to produce, and a factorial
experiment shows seed choice swamping both the hyperparameters and the training
budget. The evaluation design — five years, one seed, 72 configurations — was never
capable of resolving an effect this size.

The project's genuine contribution is the apparatus that establishes this: matched
seeds, paired tests with Holm correction, stationary block bootstrap, deflated
Sharpe, and minimum track record length, all computed rather than asserted. That
machinery is what turned a weak result into a defensible one, and it is what I
would keep.

---

## Figure index

Data and features: `01_01`–`01_14` · Environment and benchmarks: `02_01`–`02_08` ·
Training: `03_01`–`03_10` · Tuning: `04_01`–`04_07` · Results: `05_01`–`05_13` ·
Pipeline and contact sheet: `00_01`, `00_02`. All under `artifacts/figures/`.

## References

- Lo, A. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*.
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*.
- Politis, D. & Romano, J. (1994). The Stationary Bootstrap. *JASA*.
- van Hasselt, H., Guez, A. & Silver, D. (2016). Deep RL with Double Q-learning. *AAAI*.
- Wang, Z. et al. (2016). Duelling Network Architectures for Deep RL. *ICML*.
- Mnih, V. et al. (2015). Human-level control through deep RL. *Nature*.
- Faber, M. (2007). A Quantitative Approach to Tactical Asset Allocation. *Journal of Wealth Management*.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*.
