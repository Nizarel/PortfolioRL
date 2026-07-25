"""Tests for the metric definitions and the benchmark suite.

Metrics are checked against hand-computable cases rather than against a second
implementation: a constant-return series has an exactly known CAGR, Sharpe and
drawdown, so any sign error, annualisation mistake or off-by-one in the
compounding shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfoliorl import benchmarks, config, env, metrics


@pytest.fixture()
def flat_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=metrics.ANNUAL * 2, name="date")


# --------------------------------------------------------------------------- #
# Closed-form metric checks
# --------------------------------------------------------------------------- #
def test_constant_growth_has_known_cagr_and_zero_risk(flat_index):
    """A riskless 0.05%/day series: CAGR = 1.0005^252 - 1, vol = 0, no drawdown."""
    returns = pd.Series(0.0005, index=flat_index)

    assert metrics.cagr(returns) == pytest.approx(1.0005**metrics.ANNUAL - 1.0, rel=1e-10)
    assert metrics.annual_volatility(returns) == pytest.approx(0.0, abs=1e-15)
    assert metrics.max_drawdown(returns) == pytest.approx(0.0, abs=1e-15)
    assert metrics.longest_drawdown(returns) == 0
    assert metrics.hit_rate(returns) == 1.0
    assert np.isnan(metrics.sharpe_ratio(returns)), "Zero variance leaves Sharpe undefined"


def test_sharpe_matches_its_definition(flat_index):
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0004, 0.01, len(flat_index)), index=flat_index)

    expected = np.sqrt(252) * returns.mean() / returns.std(ddof=1)
    assert metrics.sharpe_ratio(returns) == pytest.approx(expected, rel=1e-12)

    # A constant risk-free rate must shift the numerator only.
    rf = 0.0001
    excess = returns - rf
    expected_excess = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    assert metrics.sharpe_ratio(returns, rf) == pytest.approx(expected_excess, rel=1e-12)
    assert metrics.sharpe_ratio(returns, rf) < metrics.sharpe_ratio(returns)


def test_sortino_exceeds_sharpe_for_right_skewed_returns(flat_index):
    """Sortino ignores upside dispersion, so it should reward positive skew."""
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0004, 0.008, len(flat_index)), index=flat_index)
    returns.iloc[::40] += 0.05  # occasional large gains, no extra downside

    assert metrics.sortino_ratio(returns) > metrics.sharpe_ratio(returns)


def test_drawdown_recovers_a_hand_built_path():
    """Wealth 100 -> 120 -> 90 -> 150: peak 120, trough 90, max drawdown 25%."""
    wealth = np.array([100.0, 120.0, 90.0, 150.0])
    returns = pd.Series(np.concatenate([[0.0], wealth[1:] / wealth[:-1] - 1.0]))

    assert metrics.max_drawdown(returns) == pytest.approx(0.25, rel=1e-12)
    assert metrics.longest_drawdown(returns) == 1
    np.testing.assert_allclose(
        metrics.drawdown_series(returns).to_numpy(), [0.0, 0.0, 0.25, 0.0], atol=1e-12
    )


def test_var_and_cvar_are_positive_losses_and_ordered(flat_index):
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0, 0.01, len(flat_index)), index=flat_index)

    var = metrics.value_at_risk(returns, 0.95)
    cvar = metrics.conditional_value_at_risk(returns, 0.95)
    assert var > 0 and cvar > 0
    assert cvar > var, "Expected shortfall must exceed the quantile it conditions on"


def test_information_ratio_is_zero_against_itself(flat_index):
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.0003, 0.01, len(flat_index)), index=flat_index)
    assert np.isnan(metrics.information_ratio(returns, returns))


def test_beta_of_a_levered_copy_is_the_leverage(flat_index):
    rng = np.random.default_rng(9)
    market = pd.Series(rng.normal(0.0003, 0.01, len(flat_index)), index=flat_index)
    levered = 1.5 * market

    beta, alpha = metrics.beta_alpha(levered, market)
    assert beta == pytest.approx(1.5, rel=1e-10)
    assert alpha == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Summary plumbing
# --------------------------------------------------------------------------- #
def test_summary_reports_the_environments_own_terminal_wealth(train_dataset):
    daily, summary = env.run_policy(train_dataset, benchmarks.constant_action(1))
    row = metrics.performance_summary(daily, summary=summary, name="60/40")

    assert row["Final wealth"] == pytest.approx(summary["final_wealth"], rel=1e-12)
    assert row["Total cost"] == pytest.approx(summary["total_cost_fraction"], rel=1e-12)
    assert row["Ann. turnover"] > 0


def test_scorecard_stacks_strategies_and_names_its_benchmark(train_dataset):
    results = {
        "60/40": env.run_policy(train_dataset, benchmarks.constant_action(1)),
        "Equal weight": env.run_policy(train_dataset, benchmarks.constant_action(4)),
    }
    table = metrics.scorecard(results, benchmark_key="60/40")

    assert list(table.index) == ["60/40", "Equal weight"]
    assert pd.isna(table.loc["60/40", "Information ratio"]), "Benchmark has no IR vs itself"
    assert np.isfinite(table.loc["Equal weight", "Information ratio"])

    display = metrics.format_scorecard(table[list(metrics.SCORECARD_COLUMNS)])
    assert display.loc["60/40", "CAGR"].endswith("%")


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def test_every_benchmark_runs_and_produces_a_full_price_path(train_dataset):
    results = benchmarks.run_benchmarks(train_dataset, seed=0)

    expected = set(benchmarks.STATIC_BENCHMARKS) | set(benchmarks.HOLD_BENCHMARKS)
    expected |= {"Volatility target", "Trend following", "Random"}
    assert set(results) == expected

    for name, (daily, summary) in results.items():
        assert len(daily) > 100, name
        assert (daily["wealth"] > 0).all(), name
        assert np.isfinite(summary["final_wealth"]), name


def test_buy_and_hold_pays_one_entry_cost_and_nothing_more(train_dataset):
    cfg = config.DEFAULT.env
    daily, summary = benchmarks.buy_and_hold(train_dataset, {"SPY": 1.0})

    start = np.asarray(config.ACTION_ALLOCATIONS[cfg.initial_action])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    expected_turnover = 0.5 * np.abs(target - start).sum()
    expected_cost = 2.0 * expected_turnover * cfg.transaction_cost_bps / 1e4

    assert summary["n_decisions"] == 1
    assert summary["total_cost_fraction"] == pytest.approx(expected_cost, rel=1e-12)

    # The wealth path must track SPY exactly, net of that single cost.
    spy = train_dataset.prices["SPY"]
    expected_final = cfg.initial_value * (1 - expected_cost) * spy.iloc[-1] / spy.iloc[0]
    assert summary["final_wealth"] == pytest.approx(expected_final, rel=1e-12)
    assert daily["w_SPY"].round(10).eq(1.0).all()


def test_buy_and_hold_rejects_weights_that_do_not_sum_to_one(train_dataset):
    with pytest.raises(ValueError, match="sum to 1"):
        benchmarks.buy_and_hold(train_dataset, {"SPY": 0.6, "TLT": 0.3})


def test_rebalancing_and_holding_differ(train_dataset):
    """If these two agreed, the rebalancing machinery would be doing nothing."""
    held, _ = benchmarks.buy_and_hold(train_dataset, {"SPY": 0.6, "TLT": 0.4})
    rebalanced, _ = env.run_policy(train_dataset, benchmarks.constant_action(1))

    common = held.index.intersection(rebalanced.index)
    assert len(common) > 100
    difference = (held.loc[common, "wealth"] - rebalanced.loc[common, "wealth"]).abs()
    assert difference.max() > 1.0


def test_volatility_target_policy_derisks_when_markets_get_choppy(train_dataset):
    """The whole point of the rule: less equity when estimated volatility is high.

    Driven with synthetic observations rather than the price fixture, whose
    per-asset volatilities are constant by construction and so contain no
    regimes for the rule to react to.
    """
    scaler = train_dataset.scaler
    policy = benchmarks.volatility_target_policy(train_dataset, target_vol=0.10)

    def observation(vols: dict[str, float]) -> np.ndarray:
        raw = scaler.mean_.copy()  # every other feature at its training mean
        for ticker, value in vols.items():
            raw[f"{ticker}_vol20"] = value
        scaled = (raw - scaler.mean_) / scaler.std_
        return np.concatenate([scaled.to_numpy(), np.zeros(8)])

    calm = policy(observation({"SPY": 0.08, "TLT": 0.06, "GLD": 0.07, "SHY": 0.01}))
    normal = policy(observation({"SPY": 0.16, "TLT": 0.13, "GLD": 0.15, "SHY": 0.02}))
    stress = policy(observation({"SPY": 0.45, "TLT": 0.25, "GLD": 0.30, "SHY": 0.03}))

    equity = np.asarray(config.ACTION_ALLOCATIONS)[:, 0]
    assert equity[calm] > equity[normal] > equity[stress]
    assert stress == 0, "A volatility shock should push the rule all the way to cash"


def test_trend_following_policy_switches_on_the_moving_average_signal(train_dataset):
    policy = benchmarks.trend_following_policy(train_dataset, risk_on=3, risk_off=0)
    _, summary = env.run_policy(train_dataset, policy)
    chosen = set(summary["decisions"]["action"].unique())

    assert chosen == {0, 3}, "The rule is binary by construction"


def test_random_policy_is_seed_reproducible_and_actually_random(train_dataset):
    first = env.run_policy(train_dataset, benchmarks.random_policy(0), seed=0)[1]
    same = env.run_policy(train_dataset, benchmarks.random_policy(0), seed=0)[1]
    other = env.run_policy(train_dataset, benchmarks.random_policy(1), seed=0)[1]

    assert first["final_wealth"] == same["final_wealth"]
    assert first["final_wealth"] != other["final_wealth"]
    assert first["decisions"]["action"].nunique() == len(config.ACTION_ALLOCATIONS)
    assert first["mean_turnover"] > 0.1, "Random trading should churn the portfolio"


def test_random_floor_spread_is_reported(train_dataset):
    table = benchmarks.average_random_floor(train_dataset, n_seeds=5)
    assert len(table) == 5
    assert table["Sharpe"].std() > 0, "Different seeds must give different outcomes"
