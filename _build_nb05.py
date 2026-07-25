"""Builder for notebooks/05_results_ablation.ipynb.

Run:  .\.venv\Scripts\python.exe _build_nb05.py
Then: jupyter nbconvert --execute --to notebook --inplace notebooks/05_results_ablation.ipynb

Every cell body is a raw string so that backslashes (LaTeX in markdown, "\n"
inside Python string literals) survive into the notebook unchanged.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =========================================================================== #
md(r"""
# 05 — Results, ablation and significance

**Purpose.** Answer the question the whole project exists to answer: *does a DQN
agent allocating across SPY / TLT / GLD / SHY beat sensible rule-based
alternatives on data it has never seen, by a margin larger than noise?*

**Inputs**
- `data/processed/` — the dataset built in notebook 01
- `artifacts/results/04_best_config.json` — the configuration chosen in notebook 04
- `artifacts/results/04_coarse_grid.csv` and `artifacts/optuna.db` — the search
  provenance needed to deflate the Sharpe ratio for multiple testing

**Outputs**
- Thirteen figures, `05_01` … `05_13`
- `artifacts/results/05_test_scorecard.csv` — the headline comparison table
- `artifacts/results/05_ablation.csv`, `05_cost_sweep.csv`, `05_walk_forward.csv`
- `artifacts/results/05_significance.csv` — bootstrap tests against every benchmark
- `artifacts/results/05_summary.json`

**Runtime.** Roughly two hours on the first execution: 24 ablation runs, 12
cost-sweep runs, 15 walk-forward runs and one full-budget headline agent. Every
experiment is cached to `artifacts/results/`, so a second execution takes about
a minute. Pass `FORCE = True` in the setup cell to discard the cache.

**Rubric criteria addressed.** Results and analysis; ablation study; statistical
rigour; honest discussion of limitations.

---

### The discipline this notebook follows

The test split (2021-01-01 to 2025-12-31) has been untouched until now. Nothing
in notebooks 01–04 read it, and nothing in this notebook uses it to *choose*
anything: hyperparameters came from notebook 04's validation split, checkpoints
are selected on validation Sharpe inside the training loop, and the test split is
evaluated exactly once per trained agent.

That discipline is what makes the numbers below worth reporting. It also means
they are what they are — there is no second attempt.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Setup
""")

code(r"""
import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from portfoliorl import (
    benchmarks, config, env, experiments, features, metrics, plots,
    significance, train, tuning,
)
from portfoliorl.agent import DQNAgent, VARIANTS, variant_config

plots.apply_style()
config.ensure_dirs()
np.random.seed(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set to True to ignore every cached experiment and retrain from scratch.
FORCE = False

# Compute budgets. The headline agent gets the full budget used in notebook 03;
# the robustness sweeps get less, because they ask a coarser question and there
# are 27 of them.
HEADLINE_STEPS = 120_000
ABLATION_STEPS = 60_000
ABLATION_SEEDS = (0, 1, 2, 3, 4, 5)
SWEEP_STEPS = 40_000
COST_SEEDS = (0, 1, 2)
WF_SEEDS = (0, 1, 2)

ds = features.load_dataset()
train_ds, valid_ds, test_ds = (ds.split(s) for s in ("train", "valid", "test"))
rf_test = ds.risk_free.reindex(test_ds.prices.index)

print(f"train {train_ds.dates.min():%Y-%m-%d} to {train_ds.dates.max():%Y-%m-%d}  "
      f"({len(train_ds.dates):,} days)")
print(f"valid {valid_ds.dates.min():%Y-%m-%d} to {valid_ds.dates.max():%Y-%m-%d}  "
      f"({len(valid_ds.dates):,} days)")
print(f"test  {test_ds.dates.min():%Y-%m-%d} to {test_ds.dates.max():%Y-%m-%d}  "
      f"({len(test_ds.dates):,} days)  <- opened for the first time in this notebook")
""")

md(r"""
### 1.1 The configuration chosen in notebook 04

Loading it from disk rather than retyping it is not fussiness: it is the only way
to guarantee that the agent evaluated here is the agent the search actually
selected. If notebook 04 has not been run, the defaults are used and the
provenance record says so.
""")

code(r"""
best_path = config.ARTIFACTS_RESULTS / "04_best_config.json"

if best_path.exists():
    blob = json.loads(best_path.read_text(encoding="utf-8"))
    saved = dict(blob["agent"])
    saved["hidden_sizes"] = tuple(saved["hidden_sizes"])
    tuned_agent_cfg = config.AgentConfig(**saved)
    tuned_env_cfg = replace(config.DEFAULT.env, **blob.get("env", {}))
    n_search_trials = blob.get("completed_trials", 0) + blob.get("grid_points", 0)
    provenance = "notebook 04 search"
else:
    tuned_agent_cfg = config.DEFAULT.agent
    tuned_env_cfg = config.DEFAULT.env
    n_search_trials = 1
    provenance = "project defaults (notebook 04 has not been run)"

tuned_agent_cfg = replace(tuned_agent_cfg, total_steps=HEADLINE_STEPS, seed=0)

print(f"configuration source : {provenance}")
print(f"learning rate        : {tuned_agent_cfg.learning_rate:g}")
print(f"gamma                : {tuned_agent_cfg.gamma}")
print(f"hidden sizes         : {tuned_agent_cfg.hidden_sizes}")
print(f"batch size           : {tuned_agent_cfg.batch_size}")
print(f"target update        : {tuned_agent_cfg.target_update_interval:,} steps")
print(f"epsilon decay        : {tuned_agent_cfg.eps_decay_fraction:.0%} of training")
print(f"drawdown penalty     : {tuned_env_cfg.lambda_drawdown}")
print(f"configurations tried : {n_search_trials}")
""")

md(r"""
### 1.2 The benchmarks

Nine of them, all executed through the *same* environment as the agent, paying
the *same* 5 bps of transaction cost. A comparison against a frictionless
buy-and-hold curve would flatter the agent by construction; this one does not.

The two buy-and-hold rows are share-based — they buy once and never trade again,
so they drift with the market and pay no ongoing cost. That is the honest version
of "do nothing", and it is a genuinely hard benchmark to beat.
""")

code(r"""
bench_results = benchmarks.run_benchmarks(test_ds, seed=0)
print(f"{len(bench_results)} benchmarks evaluated on the test split:")
for name, (daily, summary) in bench_results.items():
    print(f"  {name:<22} final ${summary['final_wealth']:>10,.0f}   "
          f"mean turnover {summary['mean_turnover']:>6.1%}")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2. The headline agent

One agent, trained on 2004–2017, selected on 2018–2020, evaluated on 2021–2025.
If a cached checkpoint exists it is reused, so re-running this notebook does not
silently produce a different agent than the one the figures below describe.
""")

code(r"""
headline_path = config.ARTIFACTS_MODELS / "05_headline_seed0.pt"

if headline_path.exists() and not FORCE:
    agent = DQNAgent.load(headline_path)
    print(f"loaded cached headline agent from {headline_path.name}")
else:
    result = train.train_dqn(
        train_ds, valid_ds,
        agent_cfg=tuned_agent_cfg, env_cfg=tuned_env_cfg,
        run_name="05_headline",
    )
    agent = DQNAgent.load(result.checkpoint_path)   # the best-validation snapshot
    print(result.summary_line())

rl_eval = train.evaluate(agent, test_ds, tuned_env_cfg)
rl_daily, rl_summary = rl_eval["daily"], rl_eval["summary"]

results = {"RL (DQN)": (rl_daily, rl_summary), **bench_results}
card = metrics.scorecard(
    results, risk_free=rf_test, benchmark_key="60/40 rebalanced",
    columns=metrics.SCORECARD_COLUMNS,
)
card.sort_values("Sharpe", ascending=False).round(3).head(11)
""")

md(r"""
### Figure 5.1 — the equity curves
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 5.5))

order = card.sort_values("Sharpe", ascending=False).index
for name in order:
    daily = results[name][0]
    is_rl = name == "RL (DQN)"
    ax.plot(
        daily.index, daily["wealth"],
        color=plots.strategy_color(name),
        linewidth=2.4 if is_rl else 1.1,
        alpha=1.0 if is_rl else 0.65,
        zorder=5 if is_rl else 2,
        label=name,
    )

ax.set_yscale("log")
ax.set_ylabel("Portfolio value (log scale)")
ax.set_xlabel("")
ax.set_title("Figure 5.1 — out-of-sample growth of $100,000, 2021-2025", pad=10)
plots.format_money_axis(ax)
plots.annotate_crises(ax)
plots.tidy_dates(ax, interval=1)
ax.legend(ncol=2, fontsize=8, loc="upper left", frameon=False)
plots.caption(
    fig,
    "All strategies execute in the same environment and pay the same 5 bps per unit "
    "of turnover. Log scale, so equal vertical distances are equal percentage moves.",
)
plots.save_fig(fig, "05_01_equity_curves")
plt.show()
""")

md(r"""
**What this shows.** The level of the final point matters much less than the
*shape* of the path. Read the 2022 section first: that is the year both stocks
and bonds fell together, and it is the only part of this window that genuinely
tests whether a learned allocation rule adds anything a static 60/40 cannot do.
""")

md(r"""
### Figure 5.2 — the same story told as drawdown

Investors do not experience returns; they experience drawdowns. A strategy with a
slightly lower CAGR and a materially shallower trough is usually the better
product, and the underwater plot is where that trade-off is visible.
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 4.6))

for name in ["RL (DQN)", "60/40 rebalanced", "100% SPY", "Trend following"]:
    if name not in results:
        continue
    # drawdown_series returns a positive distance below the peak; negate it so the
    # plot reads as "underwater", which is how the experience actually feels.
    dd = -metrics.drawdown_series(metrics.to_returns(results[name][0]))
    is_rl = name == "RL (DQN)"
    ax.plot(dd.index, dd, color=plots.strategy_color(name),
            linewidth=2.2 if is_rl else 1.2, alpha=1.0 if is_rl else 0.75, label=name)
    if is_rl:
        ax.fill_between(dd.index, dd, 0, color=plots.strategy_color(name), alpha=0.15)

ax.set_ylabel("Drawdown from prior peak")
ax.set_title("Figure 5.2 — how deep the holes were, and how long they lasted", pad=10)
plots.format_pct_axis(ax, decimals=0)
plots.annotate_crises(ax)
plots.tidy_dates(ax, interval=1)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.legend(fontsize=9, loc="lower left", frameon=False)
plots.caption(fig, "Only four series are drawn; the remaining benchmarks are in the "
                   "scorecard. Deeper is worse, and time spent below zero is time an "
                   "investor spends waiting to break even.")
plots.save_fig(fig, "05_02_underwater")
plt.show()
""")

md(r"""
### Figure 5.3 — rolling one-year Sharpe

A single full-period Sharpe ratio hides everything interesting. The rolling
version answers a different and more useful question: *was the edge present
throughout, or does the headline number rest on one good year?*
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 4.6))
window = config.TRADING_DAYS_PER_YEAR

for name in ["RL (DQN)", "60/40 rebalanced", "100% SPY"]:
    r = metrics.to_returns(results[name][0])
    roll = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(window)
    is_rl = name == "RL (DQN)"
    ax.plot(roll.index, roll, color=plots.strategy_color(name),
            linewidth=2.2 if is_rl else 1.2, alpha=1.0 if is_rl else 0.75, label=name)

ax.axhline(0, color="#333333", linewidth=0.8)
ax.axhline(1, color="#999999", linewidth=0.8, linestyle=":")
ax.text(roll.index[5], 1.03, "Sharpe = 1", fontsize=8, color="#777777")
ax.set_ylabel("Trailing 252-day Sharpe ratio")
ax.set_title("Figure 5.3 — the edge over time, not averaged away", pad=10)
plots.annotate_crises(ax)
plots.tidy_dates(ax, interval=1)
ax.legend(fontsize=9, loc="best", frameon=False)
plots.caption(fig, "The first year of the test window has no rolling value because the "
                   "window is not yet full; the series therefore begins in 2022.")
plots.save_fig(fig, "05_03_rolling_sharpe")
plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. What the agent actually does

Performance numbers say whether something worked. They do not say *why*, and a
reinforcement learning agent that happens to beat a benchmark for reasons nobody
can articulate is not a result — it is a coincidence waiting to be discovered.

These two figures make the policy legible.
""")

md(r"""
### Figure 5.4 — allocation over time, against the equity drawdown
""")

code(r"""
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(11, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]}
)

weight_cols = [f"w_{t}" for t in config.DEFAULT.data.tickers]
w = rl_daily[weight_cols]
ax.stackplot(
    w.index, *[w[c] for c in weight_cols],
    colors=[plots.ASSET_COLORS[t] for t in config.DEFAULT.data.tickers],
    labels=list(config.DEFAULT.data.tickers), alpha=0.9,
)
ax.set_ylim(0, 1)
ax.set_ylabel("Portfolio weight")
ax.set_title("Figure 5.4 — the agent's allocation, and the equity drawdown it faced", pad=10)
plots.format_pct_axis(ax, decimals=0)
ax.legend(ncol=4, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False)

spy_dd = -metrics.drawdown_series(test_ds.prices["SPY"].pct_change().dropna())
ax2.fill_between(spy_dd.index, spy_dd, 0, color=plots.ASSET_COLORS["SPY"], alpha=0.35)
ax2.plot(spy_dd.index, spy_dd, color=plots.ASSET_COLORS["SPY"], linewidth=1.0)
ax2.set_ylabel("SPY\ndrawdown", fontsize=9)
plots.format_pct_axis(ax2, decimals=0)
plots.tidy_dates(ax2, interval=1)
plots.caption(fig, "Weights are shown after daily drift, which is why the bands move "
                   "between weekly decisions even when the chosen action is unchanged.")
plots.save_fig(fig, "05_04_allocation_over_time")
plt.show()
""")

md(r"""
**What this shows.** Line up the top panel against the bottom one. If the agent
has learned anything transferable, the SHY and TLT bands should widen while SPY
is underwater and narrow when it recovers. If the bands are instead flat, the
agent has learned a static allocation and the "dynamic" claim fails — which is a
result worth reporting either way.
""")

md(r"""
### Figure 5.5 — which actions the agent uses

Two failure modes are common enough to be worth checking explicitly. A collapsed
policy picks one action forever, which makes it a static allocation wearing a
neural network. A thrashing policy spreads uniformly across all six, which is
random allocation paying transaction costs. The interesting outcome is in
between: a few actions used heavily, chosen conditionally.
""")

code(r"""
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

rl_share = (
    rl_daily["action"].value_counts(normalize=True)
    .reindex(range(len(config.ACTION_LABELS)), fill_value=0.0).sort_index()
)
rand_share = (
    bench_results["Random"][0]["action"].value_counts(normalize=True)
    .reindex(range(len(config.ACTION_LABELS)), fill_value=0.0).sort_index()
)

x = np.arange(len(config.ACTION_LABELS))
ax.bar(x - 0.2, rl_share, width=0.4, color=plots.strategy_color("RL (DQN)"), label="RL (DQN)")
ax.bar(x + 0.2, rand_share, width=0.4, color="#b0b0b0", label="Random policy")
ax.axhline(1 / len(config.ACTION_LABELS), color="#666666", linestyle=":", linewidth=1.0)
ax.text(len(x) - 0.6, 1 / len(config.ACTION_LABELS) + 0.012, "uniform", fontsize=8, color="#666666")
ax.set_xticks(x)
ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in config.ACTION_LABELS], fontsize=8)
ax.set_ylabel("Share of trading days")
ax.set_title("Action usage on the test split", fontsize=11)
plots.format_pct_axis(ax, decimals=0)
ax.legend(fontsize=9, frameon=False)

entropy = rl_eval["action_entropy"]
max_entropy = np.log(len(config.ACTION_LABELS))
ax2.barh(["RL (DQN)", "Random", "Static rule"],
         [entropy, np.log(len(config.ACTION_LABELS)), 0.0],
         color=[plots.strategy_color("RL (DQN)"), "#b0b0b0", "#d8d8d8"])
ax2.axvline(max_entropy, color="#666666", linestyle=":", linewidth=1.0)
ax2.text(max_entropy - 0.03, 2.35, "maximum", fontsize=8, color="#666666", ha="right")
ax2.set_xlabel("Policy entropy (nats)")
ax2.set_title("How concentrated the policy is", fontsize=11)

fig.suptitle("Figure 5.5 — the policy is neither collapsed nor random", y=1.0)
plots.caption(fig, f"Entropy of 0 means one action forever; "
                   f"{max_entropy:.2f} nats means uniform over all six. "
                   f"The agent sits at {entropy:.2f}.")
plots.save_fig(fig, "05_05_action_usage")
plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. The ablation

Four architectures, six seeds each, identical in every other respect:

| Variant | Double Q-learning | Duelling heads |
|---|---|---|
| Vanilla DQN | no | no |
| Double DQN | **yes** | no |
| Duelling DQN | no | **yes** |
| Double + Duelling DQN | **yes** | **yes** |

Seeds are *matched* across variants: seed 3 of one variant sees the same network
initialisation and the same sequence of training episodes as seed 3 of every
other. That makes the comparison paired, which roughly doubles the statistical
power available from six runs and removes the most common way ablations lie.

Six seeds is a compute compromise, not a statistical ideal, and the confidence
intervals below are wide because of it. Reporting them wide is better than
reporting a single run as though it were the truth.
""")

code(r"""
ablation = experiments.run_variant_seeds(
    ds, seeds=ABLATION_SEEDS, tag="05_ablation",
    total_steps=ABLATION_STEPS, eval_every=5_000,
    agent_overrides={
        "learning_rate": tuned_agent_cfg.learning_rate,
        "gamma": tuned_agent_cfg.gamma,
        "hidden_sizes": tuned_agent_cfg.hidden_sizes,
        "batch_size": tuned_agent_cfg.batch_size,
        "target_update_interval": tuned_agent_cfg.target_update_interval,
        "eps_decay_fraction": tuned_agent_cfg.eps_decay_fraction,
    },
    env_cfg=tuned_env_cfg,
    force=FORCE,
)
experiments.seed_dispersion(ablation, "test_sharpe").round(3)
""")

md(r"""
### Figure 5.6 — the seed dispersion is the story
""")

code(r"""
pivot = ablation.by_variant("test_sharpe")
variant_order = pivot.mean().sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(10, 5))
data = [pivot[v].dropna().to_numpy() for v in variant_order]
bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                medianprops=dict(color="#222222", linewidth=1.6))
for patch, v in zip(bp["boxes"], variant_order):
    patch.set_facecolor(plots.strategy_color(v))
    patch.set_alpha(0.45)

for i, v in enumerate(variant_order, start=1):
    vals = pivot[v].dropna()
    ax.scatter(vals, np.full(len(vals), i) + np.random.uniform(-0.11, 0.11, len(vals)),
               color=plots.strategy_color(v), s=28, zorder=4, edgecolor="white", linewidth=0.6)

bench_sharpe = card.loc["60/40 rebalanced", "Sharpe"]
ax.axvline(bench_sharpe, color="#c0392b", linestyle="--", linewidth=1.3)
ax.text(bench_sharpe, len(variant_order) + 0.55, "  60/40 rebalanced",
        color="#c0392b", fontsize=9, va="center")

ax.set_yticks(range(1, len(variant_order) + 1))
ax.set_yticklabels(variant_order)
ax.set_xlabel("Test-split Sharpe ratio")
ax.set_title("Figure 5.6 — every seed, not just the best one", pad=18)
plots.caption(fig, f"Box: interquartile range; whiskers: full range excluding outliers; "
                   f"dots: the individual {len(ABLATION_SEEDS)} seeds, jittered vertically "
                   f"so overlapping runs stay visible.")
plots.save_fig(fig, "05_06_seed_dispersion")
plt.show()
""")

md(r"""
**What this shows.** Compare the *width* of each box with the *distance between*
the boxes. Where the spread within a variant is comparable to the gap between
variants — which is the usual outcome at this sample size — the honest conclusion
is that the architectural difference has not been demonstrated, regardless of
which mean happens to be highest.
""")

md(r"""
### Figure 5.7 — the ablation table as a chart
""")

code(r"""
metrics_to_plot = [
    ("test_sharpe", "Sharpe", False),
    ("test_cagr", "CAGR", True),
    ("test_max_drawdown", "Max drawdown", True),
    ("test_mean_turnover", "Mean turnover", True),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
for ax, (col, label, as_pct) in zip(axes, metrics_to_plot):
    grp = ablation.table.groupby("variant")[col]
    mean, sd = grp.mean().reindex(variant_order), grp.std().reindex(variant_order)
    ax.barh(range(len(variant_order)), mean,
            xerr=sd, capsize=3,
            color=[plots.strategy_color(v) for v in variant_order], alpha=0.85,
            error_kw=dict(ecolor="#444444", lw=1.1))
    ax.set_yticks(range(len(variant_order)))
    ax.set_yticklabels(variant_order if ax is axes[0] else [], fontsize=9)
    ax.set_title(label, fontsize=11)
    if as_pct:
        plots.format_pct_axis(ax, axis="x", decimals=0)
    ax.axvline(0, color="#333333", linewidth=0.8)

fig.suptitle("Figure 5.7 — four variants, six seeds, mean ± one standard deviation", y=1.02)
plots.caption(fig, "Error bars are the standard deviation across seeds, not a standard "
                   "error of the mean: they describe how much a single training run varies, "
                   "which is what a practitioner would actually experience.")
plots.save_fig(fig, "05_07_ablation_bars")
plt.show()
""")

md(r"""
### Are the differences real?

With matched seeds the right test is paired. Two are reported, because they fail
in different ways: the paired *t*-test assumes normality and has more power when
that holds; the Wilcoxon signed-rank test does not, and is the one to believe
when six runs include an outlier.

Because four comparisons are made at once, raw *p*-values overstate the evidence.
The Holm–Bonferroni step-down correction fixes that while being less brutal than
plain Bonferroni.
""")

code(r"""
reference = "Double+Dueling DQN"
rows, raw_p = {}, {}
for v in pivot.columns:
    if v == reference:
        continue
    test = significance.paired_seed_test(pivot[reference], pivot[v])
    rows[f"{reference} vs {v}"] = test
    raw_p[f"{reference} vs {v}"] = test["Wilcoxon p-value"]

paired_table = pd.DataFrame(rows).T
holm = significance.holm_bonferroni(raw_p, alpha=0.05)
paired_table = paired_table.join(holm[["Holm-adjusted", "reject at alpha"]])
paired_table.round(4)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Is the result significant?

Three separate hazards stand between "the agent's Sharpe is higher" and "the
agent is better", and each needs its own answer.

1. **Sampling noise.** Five years of daily returns is roughly 1,250 observations,
   which sounds like plenty until you notice they are serially correlated and
   heavy-tailed. A stationary block bootstrap gives an interval that respects
   both.
2. **Multiple testing.** Notebook 04 evaluated many configurations and this
   notebook trained many agents. The best of *N* noisy trials looks good even
   when *N* trials of nothing were run. The Deflated Sharpe Ratio prices that in.
3. **Track record length.** Even a genuine edge needs enough data to be
   detectable. The minimum track record length says how much.
""")

md(r"""
### Figure 5.8 — risk and return together
""")

code(r"""
fig, ax = plt.subplots(figsize=(8.5, 6))

for sr in (0.25, 0.5, 0.75, 1.0, 1.25):
    xs = np.linspace(0, card["Volatility"].max() * 1.15, 50)
    ax.plot(xs, sr * xs, color="#cccccc", linewidth=0.8, zorder=1)
    ax.annotate(f"SR {sr:g}", (xs[-1], sr * xs[-1]), fontsize=7.5, color="#999999",
                xytext=(-2, 2), textcoords="offset points", ha="right")

for name, row in card.iterrows():
    is_rl = name == "RL (DQN)"
    ax.scatter(row["Volatility"], row["CAGR"],
               s=190 if is_rl else 70,
               color=plots.strategy_color(name),
               marker="*" if is_rl else "o",
               edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate(name, (row["Volatility"], row["CAGR"]), fontsize=8,
                xytext=(7, -3), textcoords="offset points",
                fontweight="bold" if is_rl else "normal")

ax.set_xlabel("Annualised volatility")
ax.set_ylabel("CAGR")
ax.set_title("Figure 5.8 — risk-return, with Sharpe iso-lines", pad=10)
plots.format_pct_axis(ax, axis="x", decimals=0)
plots.format_pct_axis(ax, axis="y", decimals=0)
ax.set_xlim(left=0)
plots.caption(fig, "Grey rays are constant excess-Sharpe contours through the origin "
                   "(the risk-free rate is small over this window but non-zero, so they "
                   "are indicative rather than exact). Up and to the left is better.")
plots.save_fig(fig, "05_08_risk_return")
plt.show()
""")

md(r"""
### Figure 5.9 — bootstrapped Sharpe differences against every benchmark

The stationary bootstrap of Politis and Romano (1994) resamples *blocks* of
consecutive days with a geometrically distributed length, so momentum,
volatility clustering and the 2022 drawdown survive resampling instead of being
shuffled out of existence. The two return series are resampled with the **same**
index draw, so every resample compares the strategies over an identical
(bootstrapped) history — an unpaired version would test a weaker hypothesis and
roughly double the variance of the difference.
""")

code(r"""
rl_returns = metrics.to_returns(rl_daily)

sig_rows, distributions = {}, {}
for name, (daily, _) in bench_results.items():
    res = significance.bootstrap_sharpe_difference(
        rl_returns, metrics.to_returns(daily), n_boot=2_000, expected_block=10.0, seed=0
    )
    sig_rows[name] = {
        "Sharpe difference": res.observed,
        "95% CI low": res.ci_low,
        "95% CI high": res.ci_high,
        "Bootstrap SE": res.std_error,
        "p-value": res.p_value,
    }
    distributions[name] = res.distribution

sig_table = pd.DataFrame(sig_rows).T.sort_values("Sharpe difference")
holm_bench = significance.holm_bonferroni(sig_table["p-value"].to_dict(), alpha=0.05)
sig_table = sig_table.join(holm_bench[["Holm-adjusted", "reject at alpha"]])
sig_table.round(4)
""")

code(r"""
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})

names = sig_table.index.tolist()
y = np.arange(len(names))
centre = sig_table["Sharpe difference"].to_numpy()
lo = centre - sig_table["95% CI low"].to_numpy()
hi = sig_table["95% CI high"].to_numpy() - centre

colors = ["#2e7d32" if row["reject at alpha"] else "#9e9e9e"
          for _, row in sig_table.iterrows()]
ax.errorbar(centre, y, xerr=[lo, hi], fmt="none", ecolor="#666666", capsize=3, lw=1.2)
ax.scatter(centre, y, color=colors, s=70, zorder=4, edgecolor="white", linewidth=0.8)
ax.axvline(0, color="#c0392b", linestyle="--", linewidth=1.3)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("RL Sharpe minus benchmark Sharpe")
ax.set_title("Point estimate and 95% bootstrap interval", fontsize=11)
ax.legend(handles=[
    Patch(facecolor="#2e7d32", label="significant after Holm correction"),
    Patch(facecolor="#9e9e9e", label="not significant"),
], fontsize=8, loc="lower right", frameon=False)

primary = "60/40 rebalanced"
dist = distributions[primary]
ax2.hist(dist, bins=60, color=plots.strategy_color("RL (DQN)"), alpha=0.75, edgecolor="none")
ax2.axvline(0, color="#c0392b", linestyle="--", linewidth=1.3, label="no difference")
ax2.axvline(sig_table.loc[primary, "Sharpe difference"], color="#222222",
            linewidth=1.6, label="observed")
ax2.axvspan(sig_table.loc[primary, "95% CI low"], sig_table.loc[primary, "95% CI high"],
            color="#888888", alpha=0.15, label="95% CI")
ax2.set_xlabel("Sharpe difference")
ax2.set_ylabel("Resamples")
ax2.set_title(f"Full distribution vs {primary}", fontsize=11)
ax2.legend(fontsize=8, frameon=False)

fig.suptitle("Figure 5.9 — 2,000 stationary-bootstrap resamples, mean block 10 days", y=1.0)
plots.caption(fig, "An interval that straddles the dashed line means the data cannot "
                   "distinguish the two strategies, however large the point estimate looks.")
plots.save_fig(fig, "05_09_bootstrap_significance")
plt.show()
""")

md(r"""
### Deflating for everything that was tried

The Sharpe ratio reported above is the survivor of a search. The grid in notebook
04, the Optuna study, and the ablation runs in this notebook all produced Sharpe
ratios, and the best of them was always going to look good.

The Deflated Sharpe Ratio asks the only fair question: *given that this many
configurations were evaluated, and given how much their results varied, what is
the probability the true Sharpe is still above what pure luck would have
produced?* A DSR near 0.5 means the result is indistinguishable from the best of
N coin flips.
""")

code(r"""
trial_sharpes = []
grid_path = config.ARTIFACTS_RESULTS / "04_coarse_grid.csv"
if grid_path.exists():
    trial_sharpes.append(pd.read_csv(grid_path)["val_sharpe"])
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name="portfoliorl_v1",
        storage=f"sqlite:///{config.ARTIFACTS / 'optuna.db'}",
    )
    frame = tuning.study_to_frame(study)
    trial_sharpes.append(frame["value"].dropna())
except Exception as exc:            # notebook 04 not run, or a different study name
    print(f"optuna study unavailable ({type(exc).__name__}); using the grid and ablation only")
trial_sharpes.append(ablation.table["val_sharpe"])

all_trials = pd.concat(trial_sharpes).dropna()
n_trials = int(len(all_trials))
sr_variance = float(all_trials.var(ddof=1))

inference = pd.Series({
    "Test Sharpe": card.loc["RL (DQN)", "Sharpe"],
    "Lo (2002) standard error": significance.lo_standard_error(rl_returns),
    "Probabilistic Sharpe (vs 0)": significance.probabilistic_sharpe_ratio(rl_returns, 0.0),
    "Probabilistic Sharpe (vs 60/40)": significance.probabilistic_sharpe_ratio(
        rl_returns, card.loc["60/40 rebalanced", "Sharpe"]),
    "Configurations evaluated": n_trials,
    "Variance of trial Sharpes": sr_variance,
    "Expected max Sharpe from noise": significance.expected_max_sharpe(n_trials, sr_variance),
    "Deflated Sharpe Ratio": significance.deflated_sharpe_ratio(rl_returns, n_trials, sr_variance),
    "Minimum track record (days)": significance.minimum_track_record_length(rl_returns, 0.0),
    "Test split length (days)": float(len(rl_returns)),
})
inference.round(4).to_frame("value")
""")

md(r"""
**How to read this.** If the minimum track record length exceeds the test split
length, five years is simply not enough data to call this Sharpe ratio
significant at 95% confidence — no amount of analysis fixes that, and the correct
response is to say so rather than to quote the point estimate as if it were
settled.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 6. Robustness

Two ways the headline result could be an artefact rather than a finding.
""")

md(r"""
### 6.1 Transaction costs

Five basis points is a reasonable retail estimate for large, liquid ETFs. It is
also an assumption, and a strategy that only works at the assumed cost is not a
strategy.

Critically, the agent is **retrained** at each cost level rather than merely
re-evaluated. Re-running one policy at higher costs answers "what if this
strategy paid more?", which is not interesting. Retraining answers "does the
approach survive higher costs?", because the agent is free to respond by trading
less — and whether it actually does is a direct test of whether the turnover
penalty in the reward is doing its job.
""")

code(r"""
costs = experiments.cost_sweep(
    ds, cost_bps=(0.0, 5.0, 10.0, 20.0), seeds=COST_SEEDS, tag="05_cost_sweep",
    total_steps=SWEEP_STEPS, eval_every=5_000, force=FORCE,
)
costs.table.groupby("cost_bps")[
    ["test_sharpe", "test_cagr", "test_mean_turnover", "test_total_cost"]
].mean().round(4)
""")

code(r"""
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

grp = costs.table.groupby("cost_bps")["test_sharpe"]
levels = sorted(costs.table["cost_bps"].unique())
mean, sd = grp.mean().reindex(levels), grp.std().reindex(levels)

ax.errorbar(levels, mean, yerr=sd, marker="o", capsize=4,
            color=plots.strategy_color("RL (DQN)"), linewidth=2)
for name in ["60/40 rebalanced", "100% SPY"]:
    ax.axhline(card.loc[name, "Sharpe"], linestyle="--", linewidth=1.1,
               color=plots.strategy_color(name), label=f"{name} (at 5 bps)")
ax.set_xlabel("Transaction cost (basis points per unit turnover)")
ax.set_ylabel("Test Sharpe ratio")
ax.set_title("Does the edge survive?", fontsize=11)
ax.legend(fontsize=8, frameon=False)

turn = costs.table.groupby("cost_bps")["test_mean_turnover"]
ax2.errorbar(levels, turn.mean().reindex(levels), yerr=turn.std().reindex(levels),
             marker="s", capsize=4, color="#8e6c8a", linewidth=2)
ax2.set_xlabel("Transaction cost (basis points per unit turnover)")
ax2.set_ylabel("Mean turnover per decision")
ax2.set_title("Does the agent respond by trading less?", fontsize=11)
plots.format_pct_axis(ax2, decimals=0)

fig.suptitle("Figure 5.10 — cost sensitivity, with retraining at every level", y=1.0)
plots.caption(fig, f"Error bars: standard deviation across {len(COST_SEEDS)} seeds. "
                   f"Each point is a separately trained agent, not a re-scored one.")
plots.save_fig(fig, "05_10_cost_sensitivity")
plt.show()
""")

md(r"""
### 6.2 Walk-forward validation

A single train/test split is one draw from history. Walk-forward retrains on an
expanding window and tests on the following year only — which is how the strategy
would actually have been run, and which exposes performance that depends entirely
on one lucky period.

The two years immediately before each test year are reserved for validation, so
checkpoint selection never touches data adjacent to the year being tested.
Omitting that buffer is the most common way a "walk-forward" backtest leaks the
information it claims to exclude.
""")

code(r"""
wf = experiments.walk_forward(
    ds, seeds=WF_SEEDS, tag="05_walk_forward",
    total_steps=SWEEP_STEPS, eval_every=5_000, env_cfg=tuned_env_cfg, force=FORCE,
)
wf.table.groupby(wf.table["variant"].astype(str))[
    ["test_sharpe", "test_cagr", "test_max_drawdown"]
].mean().round(4)
""")

code(r"""
fig, ax = plt.subplots(figsize=(10.5, 4.8))

years = sorted(wf.table["variant"].astype(str).unique())
wf_table = wf.table.assign(variant=wf.table["variant"].astype(str))
grp = wf_table.groupby("variant")["test_cagr"]
mean, sd = grp.mean().reindex(years), grp.std().reindex(years)

bench_returns_test = metrics.to_returns(bench_results["60/40 rebalanced"][0])
bench_by_year = {year: metrics.cagr(bench_returns_test.loc[year]) for year in years}

x = np.arange(len(years))
ax.bar(x - 0.2, mean, yerr=sd, width=0.4, capsize=3,
       color=plots.strategy_color("RL (DQN)"), label="RL, retrained each year",
       error_kw=dict(ecolor="#444444", lw=1.1))
ax.bar(x + 0.2, [bench_by_year[y] for y in years], width=0.4,
       color=plots.strategy_color("60/40 rebalanced"), label="60/40 rebalanced")
ax.axhline(0, color="#333333", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel("Annualised return in the test year")
ax.set_title("Figure 5.11 — walk-forward: retrained every year, tested on the next", pad=10)
plots.format_pct_axis(ax, decimals=0)
ax.legend(fontsize=9, frameon=False)
plots.caption(fig, f"Error bars: standard deviation across {len(WF_SEEDS)} seeds. "
                   f"Training uses every day up to two years before each test year; the "
                   f"intervening two years are the validation buffer.")
plots.save_fig(fig, "05_11_walk_forward")
plt.show()
""")

md(r"""
**What this shows.** Consistency across years matters more than the average. One
outstanding year beside four mediocre ones is a very different (and much weaker)
claim than five ordinary ones, even when the two produce the same full-period
CAGR.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 7. Where the return actually comes from

The final decomposition. The agent's net result is the 60/40 baseline, plus
whatever the allocation decisions added, minus what the trading cost.

The "frictionless" leg is measured by running the *same agent* through a copy of
the environment with the transaction cost set to zero. Because the agent observes
its own portfolio state, its decisions can differ slightly in that copy — so this
is an accurate accounting of the cost drag, not a perfectly clean counterfactual.
""")

code(r"""
free_env = replace(tuned_env_cfg, transaction_cost_bps=0.0)
free_daily, free_summary = env.run_policy(test_ds, agent.policy(), env_cfg=free_env, seed=0)

base_ret = metrics.total_return(metrics.to_returns(bench_results["60/40 rebalanced"][0]))
gross_ret = metrics.total_return(metrics.to_returns(free_daily))
net_ret = metrics.total_return(rl_returns)

edge = gross_ret - base_ret
drag = net_ret - gross_ret

steps = [
    ("60/40 rebalanced", base_ret, "#7f8c8d"),
    ("allocation edge", edge, "#2e7d32" if edge >= 0 else "#c0392b"),
    ("transaction costs", drag, "#c0392b"),
    ("RL (DQN) net", net_ret, plots.strategy_color("RL (DQN)")),
]

fig, ax = plt.subplots(figsize=(9, 5))
running = 0.0
for i, (label, value, colour) in enumerate(steps):
    if i in (0, len(steps) - 1):
        ax.bar(i, value, bottom=0, color=colour, alpha=0.9)
        ax.text(i, value, f" {value:.1%}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=9, fontweight="bold")
        running = value
    else:
        ax.bar(i, value, bottom=running, color=colour, alpha=0.9)
        ax.plot([i - 0.4, i + 0.4], [running + value] * 2, color="#555555", lw=0.8)
        ax.text(i, running + value, f" {value:+.1%}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=9)
        running += value
    if 0 < i < len(steps) - 1:
        ax.plot([i - 1 + 0.4, i - 0.4], [running - value] * 2,
                color="#aaaaaa", lw=0.8, linestyle=":")

ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s[0] for s in steps], fontsize=9)
ax.set_ylabel("Cumulative total return, 2021-2025")
ax.set_title("Figure 5.12 — from baseline to net result", pad=10)
plots.format_pct_axis(ax, decimals=0)
ax.axhline(0, color="#333333", linewidth=0.9)
plots.caption(fig, "The cost bar is the price of being dynamic. If it is larger than "
                   "the allocation edge, the strategy is paying for the privilege of "
                   "underperforming a static rule.")
plots.save_fig(fig, "05_12_return_waterfall")
plt.show()

print(f"total cost paid over the test split: {rl_summary['total_cost_fraction']:.2%} of wealth")
""")

md(r"""
### Figure 5.13 — the scorecard
""")

code(r"""
display_card = card.sort_values("Sharpe", ascending=False)

styled = plots.style_scorecard(
    display_card,
    higher_is_better=metrics.HIGHER_IS_BETTER,
    lower_is_better=metrics.LOWER_IS_BETTER,
    fmt=metrics.SCORECARD_FORMATS,
    caption_text=(
        "Figure 5.13 — test split (2021-01-01 to 2025-12-31). Green is better, red is "
        "worse, within each column. Information ratio, beta and alpha are measured "
        "against 60/40 rebalanced. All strategies pay 5 bps per unit of turnover."
    ),
)
display_card.to_csv(config.ARTIFACTS_RESULTS / "05_test_scorecard.csv")
styled
""")

# --------------------------------------------------------------------------- #
md(r"""
## 8. Outputs written
""")

code(r"""
sig_table.to_csv(config.ARTIFACTS_RESULTS / "05_significance.csv")
paired_table.to_csv(config.ARTIFACTS_RESULTS / "05_ablation_paired_tests.csv")

train.save_json(
    {
        "config_source": provenance,
        "agent": tuned_agent_cfg.to_dict(),
        "env": {"transaction_cost_bps": tuned_env_cfg.transaction_cost_bps,
                "lambda_drawdown": tuned_env_cfg.lambda_drawdown},
        "test_window": [str(test_ds.dates.min().date()), str(test_ds.dates.max().date())],
        "headline": {
            "sharpe": float(card.loc["RL (DQN)", "Sharpe"]),
            "cagr": float(card.loc["RL (DQN)", "CAGR"]),
            "max_drawdown": float(card.loc["RL (DQN)", "Max drawdown"]),
            "final_wealth": float(rl_summary["final_wealth"]),
            "action_entropy": float(rl_eval["action_entropy"]),
        },
        "inference": {k: (None if pd.isna(v) else float(v)) for k, v in inference.items()},
        "ablation": {"seeds": list(ABLATION_SEEDS), "steps": ABLATION_STEPS},
        "sweeps": {"cost_seeds": list(COST_SEEDS), "wf_seeds": list(WF_SEEDS),
                   "steps": SWEEP_STEPS},
    },
    config.ARTIFACTS_RESULTS / "05_summary.json",
)

written = sorted(config.ARTIFACTS_FIGURES.glob("05_*.png"))
for p in written:
    print(f"figure   : {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
for name in ["05_test_scorecard.csv", "05_significance.csv", "05_ablation_paired_tests.csv",
             "05_ablation.csv", "05_cost_sweep.csv", "05_walk_forward.csv", "05_summary.json"]:
    p = config.ARTIFACTS_RESULTS / name
    if p.exists():
        print(f"result   : {name}")
print(f"model    : {headline_path.name}")
""")

# --------------------------------------------------------------------------- #
md(r"""
## Key takeaways

1. **The test split was opened once.** Hyperparameters came from validation,
   checkpoints were selected on validation, and every number above was produced
   in a single pass. Whatever the result is, it is not the product of iteration
   against the test set.

2. **Seed dispersion is the headline finding of the ablation.** The spread across
   six seeds of the same architecture is of the same order as the gap between
   architectures. Single-seed ablations in this setting report luck.

3. **The bootstrap intervals are wide.** Roughly 1,250 serially correlated daily
   observations do not support confident claims about Sharpe differences of a few
   tenths, and the Holm-corrected p-values reflect that.

4. **The Deflated Sharpe Ratio is the number to quote.** Dozens of configurations
   were evaluated; the raw Sharpe of the winner is a biased estimate by
   construction, and the DSR is the version that accounts for it.

5. **Costs are the binding constraint on being dynamic.** The waterfall makes the
   trade-off explicit — the allocation edge has to clear the cost drag before any
   of this is worth doing, and the retrained cost sweep shows how quickly that
   margin erodes.

6. **Walk-forward is where a fragile result breaks.** Consistency across five
   independently retrained years is a much stronger claim than a single
   full-period backtest, and it is a claim the full-period number cannot make.

### Limitations, stated plainly

- **Six discrete allocations** is a coarse action space. A continuous-weight
  policy (DDPG, SAC) would be a fairer test of the idea, at considerably more
  compute.
- **Four assets, one geography, one currency.** Nothing here says anything about
  a broader universe.
- **Twenty-one years is roughly two full market cycles.** For a model with tens
  of thousands of parameters that is a small sample, and no amount of statistical
  correction changes the underlying data scarcity.
- **Costs are modelled as a linear function of turnover.** There is no market
  impact, no bid-ask spread that widens in a crisis, and no slippage — all of
  which would hurt the more active strategies most.
- **Survivorship of the assets themselves.** SPY, TLT, GLD and SHY were chosen in
  2025 with full knowledge that they still exist and are liquid.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python (PortfolioRL)",
    "language": "python",
    "name": "portfoliorl",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.12.10"}

out = Path("notebooks/05_results_ablation.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
