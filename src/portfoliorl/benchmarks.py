"""Benchmark strategies.

A reinforcement-learning result is only meaningful relative to what a simple
rule would have achieved on the *same* data, paying the *same* costs, at the
*same* decision frequency. Every benchmark here is therefore either

* a :data:`portfoliorl.env.Policy` executed through :func:`env.run_policy`, so it
  goes through exactly the same accounting as the agent, or
* a share-based buy-and-hold path built with the identical entry-cost model
  (:func:`buy_and_hold`), for the two references that cannot be expressed in the
  discrete action set.

The set spans three levels of difficulty:

``Random``
    Uniform over the six actions. This is the *floor*: it churns the portfolio
    and pays costs without any information. Beating a well-chosen static
    allocation is the real test, but failing to beat Random would mean the agent
    has learned nothing at all.
``Static allocations``
    60/40, equal weight, all-equity, all-cash. These are what an investor could
    actually do with no model whatsoever, so they are the honest bar.
``Adaptive rules``
    Volatility targeting and trend following. Both react to the same features
    the agent sees. If the agent cannot beat these, its advantage is not coming
    from the state information it was given, and the network is not earning its
    keep.

A note on what was *not* used: a naive inverse-volatility ("risk parity") rule
degenerates here. SHY's annualised volatility is about 1.5% against 15-19% for
the risky assets, so ``1/sigma`` weights put roughly 80% in SHY and the nearest
discrete allocation is always 100% cash. It would have been indistinguishable
from the all-cash benchmark while looking deceptively strong on risk-adjusted
measures. Volatility *targeting* is the meaningful discrete analogue.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import config, env, features

Policy = env.Policy


# --------------------------------------------------------------------------- #
# Policy constructors
# --------------------------------------------------------------------------- #
def constant_action(action: int) -> Policy:
    """Always choose the same allocation.

    Note this is *not* buy-and-hold: the environment rebalances back to the
    target every week, so the strategy still pays the cost of correcting drift.
    """

    def policy(_obs: np.ndarray) -> int:
        return action

    policy.__name__ = f"constant_{action}"
    return policy


def random_policy(seed: int = 0, n_actions: int | None = None) -> Policy:
    """Uniformly random allocation each week -- the information-free floor."""
    rng = np.random.default_rng(seed)
    n = n_actions if n_actions is not None else len(config.ACTION_ALLOCATIONS)

    def policy(_obs: np.ndarray) -> int:
        return int(rng.integers(0, n))

    policy.__name__ = "random"
    return policy


def _raw_feature_reader(
    dataset: features.Dataset, names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate named features in the observation and return the de-standardising terms.

    Benchmark policies see exactly what the agent sees -- a scaled observation
    vector -- so to use a feature in its natural units they must invert the
    scaler themselves. Doing that here keeps every policy on the same footing.
    """
    scaler = dataset.scaler
    if scaler is None or scaler.mean_ is None or scaler.std_ is None:
        raise ValueError("Adaptive benchmark policies need a fitted scaler on the dataset")

    columns = list(scaler.columns_)
    idx = np.array([columns.index(name) for name in names])
    return idx, scaler.mean_.to_numpy()[idx], scaler.std_.to_numpy()[idx]


def volatility_target_policy(
    dataset: features.Dataset,
    target_vol: float = 0.10,
    cfg: config.DataConfig | None = None,
) -> Policy:
    """Hold the allocation whose expected volatility is closest to ``target_vol``.

    Volatility targeting is the standard risk-based rule and, unlike naive
    inverse-volatility weighting, it maps sensibly onto a discrete menu: when
    markets get choppier every allocation's estimated volatility rises, so the
    rule slides down the risk dial towards cash and back up again when calm
    returns.

    The volatility of a candidate allocation is estimated as
    ``sum_i w_i sigma_i`` -- the perfectly-correlated upper bound -- because the
    observation carries per-asset volatilities but not a full covariance matrix.
    The target is calibrated against that same proxy, so the rule is internally
    consistent; the approximation simply makes it a little more defensive than a
    covariance-aware version would be.
    """
    cfg = cfg or config.DEFAULT.data
    idx, mean, std = _raw_feature_reader(dataset, [f"{t}_vol20" for t in cfg.tickers])
    allocations = np.asarray(config.ACTION_ALLOCATIONS, dtype=float)

    def policy(obs: np.ndarray) -> int:
        sigma = np.asarray(obs, dtype=float)[idx] * std + mean  # annualised, raw units
        sigma = np.maximum(sigma, 1e-6)
        candidate_vol = allocations @ sigma
        return int(np.argmin(np.abs(candidate_vol - target_vol)))

    policy.__name__ = "volatility_target"
    return policy


def trend_following_policy(
    dataset: features.Dataset,
    risk_on: int = 3,
    risk_off: int = 0,
    signal_ticker: str = "SPY",
) -> Policy:
    """Classic time-series momentum: hold equities while the trend is up, else cash.

    This is the Faber (2007) moving-average rule expressed in the project's
    action space -- go equity-heavy when the 50-day average is above the 200-day
    average, retreat to cash when it is not. It is deliberately the crudest
    possible market-timing rule, which makes it a fair test of whether the agent
    has learned anything a single threshold could not.
    """
    idx, mean, std = _raw_feature_reader(dataset, [f"{signal_ticker}_ma_ratio"])

    def policy(obs: np.ndarray) -> int:
        ma_ratio = float(np.asarray(obs, dtype=float)[idx][0] * std[0] + mean[0])
        return risk_on if ma_ratio > 0.0 else risk_off

    policy.__name__ = "trend_following"
    return policy


# --------------------------------------------------------------------------- #
# Share-based buy-and-hold
# --------------------------------------------------------------------------- #
def buy_and_hold(
    dataset: features.Dataset,
    weights: Mapping[str, float] | np.ndarray,
    env_cfg: config.EnvConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Buy once, never trade again, mark to market daily.

    Some references -- 100% SPY, or a 60/40 that is genuinely never rebalanced --
    are not reachable as a repeated discrete action, so they are simulated here
    directly. The entry cost uses the same model as the environment (turnover
    against the starting allocation, ``2 x turnover x bps``), so the comparison
    remains like-for-like.

    Returns the same ``(daily, summary)`` pair as :func:`env.run_policy` so it
    can be dropped straight into a scorecard.
    """
    cfg = env_cfg or config.DEFAULT.env
    prices = dataset.prices
    tickers = list(prices.columns)

    if isinstance(weights, Mapping):
        target = np.array([float(weights.get(t, 0.0)) for t in tickers])
    else:
        target = np.asarray(weights, dtype=float)
    if not np.isclose(target.sum(), 1.0):
        raise ValueError(f"Buy-and-hold weights must sum to 1, got {target.sum():.6f}")

    start = np.asarray(config.ACTION_ALLOCATIONS[cfg.initial_action], dtype=float)
    turnover = 0.5 * float(np.abs(target - start).sum())
    cost_fraction = 2.0 * turnover * cfg.transaction_cost_bps / 1e4

    invested = cfg.initial_value * (1.0 - cost_fraction)
    shares = invested * target / prices.iloc[0].to_numpy()
    wealth = pd.Series(prices.to_numpy() @ shares, index=prices.index, name="wealth")

    # Drifting weights, for the allocation plots.
    values = prices.to_numpy() * shares
    weight_frame = pd.DataFrame(
        values / values.sum(axis=1, keepdims=True),
        index=prices.index,
        columns=[f"w_{t}" for t in tickers],
    )

    returns = wealth.pct_change().fillna(0.0)
    daily = pd.DataFrame(
        {
            "wealth": wealth,
            "return": returns,
            "drawdown": 1.0 - wealth / wealth.cummax(),
            "action": -1,  # not a discrete action
        }
    ).join(weight_frame)

    decisions = pd.DataFrame(
        {
            "action": [-1],
            "turnover": [turnover],
            "cost_fraction": [cost_fraction],
            "reward": [np.nan],
            "reward_return": [np.nan],
            "reward_turnover_penalty": [np.nan],
            "reward_vol_penalty": [np.nan],
            "reward_dd_penalty": [np.nan],
        },
        index=pd.Index([prices.index[0]], name="date"),
    )

    summary = {
        "final_wealth": float(wealth.iloc[-1]),
        "total_reward": float("nan"),
        "n_decisions": 1,
        "n_days": int(len(daily)),
        "total_cost_fraction": cost_fraction,
        "mean_turnover": turnover,
        "decisions": decisions,
    }
    return daily, summary


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: Benchmarks executed through the environment, as ``name -> action index``.
STATIC_BENCHMARKS: dict[str, int] = {
    "60/40 rebalanced": 1,
    "Equal weight": 4,
    "Equity-heavy 80/20": 3,
    "All cash (SHY)": 0,
}

#: Benchmarks simulated as share-based buy-and-hold, as ``name -> weights``.
HOLD_BENCHMARKS: dict[str, dict[str, float]] = {
    "100% SPY": {"SPY": 1.0},
    "60/40 buy & hold": {"SPY": 0.6, "TLT": 0.4},
}


def build_policies(
    dataset: features.Dataset,
    seed: int = 0,
    cfg: config.DataConfig | None = None,
    n_actions: int | None = None,
) -> dict[str, Policy]:
    """All environment-executed benchmark policies, in report order.

    ``n_actions`` must match the menu the agent is using: the random floor is
    only a fair comparator if it draws from the same set of allocations.
    """
    policies: dict[str, Policy] = {
        name: constant_action(action) for name, action in STATIC_BENCHMARKS.items()
    }
    policies["Volatility target"] = volatility_target_policy(dataset, cfg=cfg)
    policies["Trend following"] = trend_following_policy(dataset)
    policies["Random"] = random_policy(seed, n_actions=n_actions)
    return policies


def run_benchmarks(
    dataset: features.Dataset,
    env_cfg: config.EnvConfig | None = None,
    seed: int = 0,
    cfg: config.DataConfig | None = None,
    include_hold: bool = True,
) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Evaluate every benchmark on one split and return ``name -> (daily, summary)``."""
    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    menu = (env_cfg or config.DEFAULT.env).n_actions

    for name, policy in build_policies(dataset, seed=seed, cfg=cfg, n_actions=menu).items():
        results[name] = env.run_policy(dataset, policy, env_cfg=env_cfg, seed=seed)

    if include_hold:
        for name, weights in HOLD_BENCHMARKS.items():
            results[name] = buy_and_hold(dataset, weights, env_cfg=env_cfg)

    return results


def average_random_floor(
    dataset: features.Dataset,
    env_cfg: config.EnvConfig | None = None,
    n_seeds: int = 30,
) -> pd.DataFrame:
    """Distribution of outcomes for the random policy across many seeds.

    A single random run can look impressive or terrible by luck. Reporting the
    spread turns "the agent beat random" into a statement with a stated
    confidence rather than an anecdote.
    """
    from . import metrics  # local import keeps the module import graph shallow

    rows = []
    for seed in range(n_seeds):
        daily, summary = env.run_policy(
            dataset, random_policy(seed), env_cfg=env_cfg, seed=seed
        )
        row = metrics.performance_summary(daily, summary=summary, name=f"seed {seed}")
        rows.append(row)
    return pd.DataFrame(rows)
