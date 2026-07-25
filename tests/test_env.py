"""Correctness tests for the portfolio environment.

The environment is where a subtle bug produces *better* results, so these tests
check it against independently derived answers rather than against itself.

The two golden tests are the important ones:

``test_golden_single_asset_matches_closed_form``
    Holding 100% of one asset must reproduce that asset's own price ratio,
    times a one-off entry cost. Computed straight from the price series, with
    no reference to the environment's arithmetic.

``test_golden_buy_and_hold_matches_share_accounting``
    A single long decision must reproduce a share-based buy-and-hold
    calculation (buy N_i shares, never trade, mark to market). This exercises
    weight drift and daily compounding together, and is derived from a
    completely different formulation than the one the environment implements.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from portfoliorl import config, env


# --------------------------------------------------------------------------- #
# Gymnasium API conformance
# --------------------------------------------------------------------------- #
def test_conforms_to_the_gymnasium_api(train_dataset):
    """`check_env` verifies the 5-tuple step signature, seeding and spaces."""
    environment = env.PortfolioEnv(train_dataset, mode="train")
    check_env(environment, skip_render_check=True)


def test_observation_shape_and_finiteness(train_dataset):
    environment = env.PortfolioEnv(train_dataset, mode="eval")
    obs, info = environment.reset(seed=0)

    assert obs.shape == (31,), "The report quotes a 31-dimensional observation"
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert environment.observation_space.contains(obs)

    for action in range(environment.action_space.n):
        obs, reward, terminated, truncated, info = environment.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert not terminated, "Endings are time limits, so they are truncations"
        if truncated:
            break


# --------------------------------------------------------------------------- #
# Golden test 1: single-asset closed form
# --------------------------------------------------------------------------- #
def test_golden_single_asset_matches_closed_form(train_dataset):
    """100% SHY, held throughout, must equal SHY's own price ratio net of entry cost.

    Because the episode starts already holding SHY (``initial_action = 0``) and
    the agent keeps choosing it, turnover is zero at every decision and the
    wealth path collapses to a quantity computable from the price series alone.
    """
    cfg = config.DEFAULT.env
    daily, summary = env.run_policy(train_dataset, policy=lambda _obs: 0, env_cfg=cfg)

    prices = train_dataset.prices["SHY"]
    first, last = daily.index[0], daily.index[-1]

    # The environment applies returns starting one day after the reset index, so
    # the reference ratio starts from the day *before* the first recorded day.
    start_price = prices.loc[: first].iloc[-2]
    expected = cfg.initial_value * (prices.loc[last] / start_price)

    assert summary["final_wealth"] == pytest.approx(expected, rel=1e-12)
    assert summary["mean_turnover"] == pytest.approx(0.0, abs=1e-15)
    assert summary["total_cost_fraction"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# Golden test 2: share-based buy-and-hold
# --------------------------------------------------------------------------- #
def test_golden_buy_and_hold_matches_share_accounting(train_dataset):
    """One long decision must equal 'buy N_i shares and never trade again'.

    Written from the share formulation -- ``W_T = sum_i N_i P_i(T)`` -- which
    shares no code path with the environment's weight-drift recursion. If the
    drift formula, the daily compounding or the cost deduction were wrong, these
    two numbers would not agree to machine precision.
    """
    prices = train_dataset.prices
    n_days = len(prices)

    # A single decision spanning the entire split: no rebalancing after entry.
    cfg = dataclasses.replace(
        config.DEFAULT.env,
        steps_per_decision=n_days - 1,
        episode_length=1,
        initial_action=0,  # start in 100% SHY, so entering action 4 costs money
    )
    target_action = 4  # equal weight across all four assets

    daily, summary = env.run_policy(train_dataset, policy=lambda _obs: target_action, env_cfg=cfg)

    # --- Independent reconstruction ---------------------------------------- #
    weights_before = np.asarray(config.ACTION_ALLOCATIONS[cfg.initial_action])
    weights_after = np.asarray(config.ACTION_ALLOCATIONS[target_action])

    turnover = 0.5 * np.abs(weights_after - weights_before).sum()
    cost_fraction = 2.0 * turnover * cfg.transaction_cost_bps / 1e4
    wealth_after_cost = cfg.initial_value * (1.0 - cost_fraction)

    start_prices = prices.iloc[0].to_numpy()
    end_prices = prices.loc[daily.index[-1]].to_numpy()
    shares = wealth_after_cost * weights_after / start_prices  # buy once
    expected = float(shares @ end_prices)                      # mark to market

    assert summary["final_wealth"] == pytest.approx(expected, rel=1e-10)
    assert summary["total_cost_fraction"] == pytest.approx(cost_fraction, rel=1e-12)


# --------------------------------------------------------------------------- #
# Reward construction
# --------------------------------------------------------------------------- #
def test_return_term_sums_to_terminal_log_wealth(train_dataset):
    """With penalties switched off, sum(reward) / scale == log(W_T / W_0) exactly.

    This is the whole reason for using log returns. If it holds, maximising
    undiscounted episode return is *identical* to maximising terminal wealth --
    the RL objective is the business objective, not a proxy for it.
    """
    cfg = dataclasses.replace(
        config.DEFAULT.env,
        lambda_turnover=0.0,
        lambda_volatility=0.0,
        lambda_drawdown=0.0,
    )
    # An action sequence that actually trades, so costs are exercised too.
    counter = {"i": 0}

    def rotating_policy(_obs):
        counter["i"] += 1
        return counter["i"] % 6

    _, summary = env.run_policy(train_dataset, rotating_policy, env_cfg=cfg)

    expected = np.log(summary["final_wealth"] / cfg.initial_value)
    assert summary["total_reward"] / cfg.reward_scale == pytest.approx(expected, rel=1e-10)


def test_penalties_are_non_positive_and_decomposition_adds_up(train_dataset):
    cfg = config.DEFAULT.env
    _, summary = env.run_policy(train_dataset, lambda _obs: 2, env_cfg=cfg)
    decisions = summary["decisions"]

    for column in ("reward_turnover_penalty", "reward_vol_penalty", "reward_dd_penalty"):
        assert (decisions[column] <= 1e-12).all(), f"{column} must never be a bonus"

    recomposed = (
        decisions["reward_return"]
        + decisions["reward_turnover_penalty"]
        + decisions["reward_vol_penalty"]
        + decisions["reward_dd_penalty"]
    )
    np.testing.assert_allclose(recomposed.to_numpy(), decisions["reward"].to_numpy(), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Transaction costs
# --------------------------------------------------------------------------- #
def test_turnover_is_measured_against_drifted_weights(train_dataset):
    """A constant multi-asset action must still incur turnover after the first trade.

    If turnover were computed target-vs-previous-target it would be exactly zero
    from the second decision onwards, and the cost of maintaining a fixed
    allocation would vanish. That is the single most common way a rebalancing
    backtest flatters itself.
    """
    _, summary = env.run_policy(train_dataset, lambda _obs: 4)  # equal weight, held
    decisions = summary["decisions"]

    later = decisions["turnover"].iloc[1:]
    assert (later > 0).mean() > 0.95, "Drift should force a trade almost every week"
    assert later.max() < 0.10, "Weekly drift-driven turnover should be small, not wholesale"


def test_higher_costs_strictly_reduce_terminal_wealth(train_dataset):
    """Monotonicity in the fee level -- the basis of the cost sweep in notebook 05."""
    wealth = []
    for bps in (0.0, 5.0, 10.0, 20.0):
        cfg = dataclasses.replace(config.DEFAULT.env, transaction_cost_bps=bps)
        _, summary = env.run_policy(train_dataset, lambda _obs: 4, env_cfg=cfg)
        wealth.append(summary["final_wealth"])

    assert wealth == sorted(wealth, reverse=True)
    assert wealth[0] > wealth[-1]


def test_zero_cost_zero_turnover_policy_pays_nothing(train_dataset):
    cfg = dataclasses.replace(config.DEFAULT.env, transaction_cost_bps=0.0)
    _, summary = env.run_policy(train_dataset, lambda _obs: 3, env_cfg=cfg)
    assert summary["total_cost_fraction"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# Episode construction
# --------------------------------------------------------------------------- #
def test_training_episodes_have_the_configured_length(train_dataset):
    cfg = config.DEFAULT.env
    environment = env.PortfolioEnv(train_dataset, cfg, mode="train")
    environment.reset(seed=0)

    steps = 0
    while True:
        _, _, _, truncated, _ = environment.step(1)
        steps += 1
        if truncated:
            break
    assert steps == cfg.episode_length


def test_random_starts_actually_vary(train_dataset):
    """Otherwise the agent replays one trajectory and memorises it."""
    environment = env.PortfolioEnv(train_dataset, mode="train")
    starts = {environment.reset(seed=seed)[1]["start_date"] for seed in range(30)}
    assert len(starts) > 25, "Start dates should be almost all distinct"


def test_eval_mode_is_deterministic_and_covers_the_whole_split(train_dataset):
    environment = env.PortfolioEnv(train_dataset, mode="eval")
    first, _ = environment.reset(seed=0)
    second, _ = environment.reset(seed=999)
    np.testing.assert_array_equal(first, second)

    daily_a, summary_a = env.run_policy(train_dataset, lambda _obs: 1, seed=0)
    daily_b, summary_b = env.run_policy(train_dataset, lambda _obs: 1, seed=123)
    assert summary_a["final_wealth"] == summary_b["final_wealth"]

    span = config.DEFAULT.env.steps_per_decision
    assert len(daily_a) >= len(train_dataset.prices) - span - 1


def test_wealth_never_becomes_non_finite(train_dataset):
    rng = np.random.default_rng(0)
    daily, summary = env.run_policy(
        train_dataset, lambda _obs: int(rng.integers(0, 6)), seed=0
    )
    assert np.isfinite(daily["wealth"]).all()
    assert (daily["wealth"] > 0).all()
    assert np.isfinite(summary["total_reward"])


def test_weights_always_sum_to_one(train_dataset):
    daily, _ = env.run_policy(train_dataset, lambda _obs: 2)
    weight_cols = [c for c in daily.columns if c.startswith("w_")]
    np.testing.assert_allclose(daily[weight_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-12)
