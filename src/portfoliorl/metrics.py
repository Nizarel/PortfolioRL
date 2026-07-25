"""Performance and risk metrics.

Every number reported in the project passes through this module, so that the RL
agent and the benchmarks are scored by identical code. The definitions follow
standard practice; where a choice had to be made it is stated explicitly in the
docstring rather than buried in the implementation.

Conventions
-----------
* Returns are **daily simple** returns, matching the environment's accounting.
* Annualisation uses 252 trading days.
* The Sharpe ratio uses the daily excess return over the ``^IRX`` short rate,
  annualised as ``sqrt(252) * mean / std``. This is the usual convention, but
  it assumes i.i.d. returns; Lo (2002) shows autocorrelation biases it, which is
  why the significance testing in notebook 05 uses a block bootstrap rather than
  relying on this point estimate alone.
* The Sortino ratio uses a minimum acceptable return (MAR) of zero and divides
  by downside deviation computed over *all* observations (the "full" convention
  of Sortino & Price 1994), not only the losing days -- dividing by the count of
  losing days rewards strategies that trade rarely.
* Drawdown is measured on the wealth path, so it includes transaction costs.

References
----------
Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*.
Sortino, F. & Price, L. (1994). Performance measurement in a downside risk
framework. *Journal of Investing*.
Magdon-Ismail, M. & Atiya, A. (2004). Maximum drawdown. *Risk*.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config

ANNUAL = config.TRADING_DAYS_PER_YEAR

#: Column order used for every scorecard in the report.
SCORECARD_COLUMNS: tuple[str, ...] = (
    "Final wealth",
    "Total return",
    "CAGR",
    "Volatility",
    "Sharpe",
    "Sortino",
    "Max drawdown",
    "Calmar",
    "Longest drawdown (days)",
    "VaR 95% (daily)",
    "CVaR 95% (daily)",
    "Hit rate",
    "Skew",
    "Excess kurtosis",
    "Ann. turnover",
    "Total cost",
)

#: Metrics where a larger value is better -- used to colour the styled tables.
HIGHER_IS_BETTER: tuple[str, ...] = (
    "Final wealth",
    "Total return",
    "CAGR",
    "Sharpe",
    "Sortino",
    "Calmar",
    "Hit rate",
    "Information ratio",
)

#: Metrics where a smaller value is better.
LOWER_IS_BETTER: tuple[str, ...] = (
    "Volatility",
    "Max drawdown",
    "Longest drawdown (days)",
    "Ann. turnover",
    "Total cost",
)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def to_returns(daily: pd.DataFrame | pd.Series) -> pd.Series:
    """Extract the daily return series from an episode frame (or pass a Series through)."""
    if isinstance(daily, pd.Series):
        return daily.astype(float)
    if "return" in daily.columns:
        return daily["return"].astype(float)
    raise KeyError("Expected a 'return' column produced by PortfolioEnv.to_frame()")


def wealth_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a return series into a wealth path."""
    return initial * (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    """Geometric annual growth rate.

    The exponent uses the number of trading days rather than calendar time so
    that it stays consistent with the daily accounting elsewhere.
    """
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (ANNUAL / n) - 1.0


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(ANNUAL))


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series | float = 0.0) -> float:
    """Annualised Sharpe ratio on daily excess returns."""
    excess = _excess(returns, risk_free)
    sigma = excess.std(ddof=1)
    if not np.isfinite(sigma) or sigma < 1e-15:
        return float("nan")
    return float(np.sqrt(ANNUAL) * excess.mean() / sigma)


def sortino_ratio(returns: pd.Series, mar: float = 0.0) -> float:
    """Annualised Sortino ratio with a zero minimum acceptable return."""
    shortfall = np.minimum(returns - mar, 0.0)
    downside = np.sqrt(np.mean(np.square(shortfall)))
    if not np.isfinite(downside) or downside < 1e-15:
        return float("nan")
    return float(np.sqrt(ANNUAL) * (returns.mean() - mar) / downside)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional distance below the running peak of the wealth curve."""
    wealth = wealth_curve(returns)
    return 1.0 - wealth / wealth.cummax()


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).max())


def longest_drawdown(returns: pd.Series) -> int:
    """Length in trading days of the longest stretch spent below a prior peak.

    Depth alone understates the experience of holding a strategy: a 20% drawdown
    recovered in a month is very different from a 20% drawdown that lasts three
    years, and only the second one causes investors to capitulate.
    """
    underwater = drawdown_series(returns) > 1e-12
    longest = current = 0
    for flag in underwater.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd < 1e-12:
        return float("nan")
    return float(cagr(returns) / mdd)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical daily VaR, reported as a positive loss magnitude."""
    return float(-np.quantile(returns, 1.0 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Mean loss on the worst ``1 - level`` of days (expected shortfall)."""
    threshold = np.quantile(returns, 1.0 - level)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualised active return divided by tracking error.

    The benchmark must be named wherever this is reported -- an information
    ratio without a stated benchmark is meaningless.
    """
    active = (returns - benchmark.reindex(returns.index)).dropna()
    tracking_error = active.std(ddof=1)
    if not np.isfinite(tracking_error) or tracking_error < 1e-15:
        return float("nan")
    return float(np.sqrt(ANNUAL) * active.mean() / tracking_error)


def beta_alpha(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """OLS beta and annualised alpha against a benchmark return series."""
    joined = pd.concat([returns, benchmark.reindex(returns.index)], axis=1).dropna()
    if len(joined) < 3:
        return float("nan"), float("nan")
    y = joined.iloc[:, 0].to_numpy()
    x = joined.iloc[:, 1].to_numpy()
    variance = x.var(ddof=1)
    if variance < 1e-18:
        return float("nan"), float("nan")
    beta = float(np.cov(y, x, ddof=1)[0, 1] / variance)
    alpha = float((y.mean() - beta * x.mean()) * ANNUAL)
    return beta, alpha


def _excess(returns: pd.Series, risk_free: pd.Series | float) -> pd.Series:
    if isinstance(risk_free, pd.Series):
        return (returns - risk_free.reindex(returns.index).fillna(0.0)).dropna()
    return returns - float(risk_free)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def performance_summary(
    daily: pd.DataFrame | pd.Series,
    *,
    risk_free: pd.Series | float = 0.0,
    benchmark: pd.Series | None = None,
    summary: Mapping[str, object] | None = None,
    name: str = "strategy",
) -> pd.Series:
    """Full metric set for one strategy, ready to be stacked into a scorecard.

    Parameters
    ----------
    daily
        Episode frame from :meth:`PortfolioEnv.to_frame`, or a return Series.
    risk_free
        Daily risk-free rate for the Sharpe ratio.
    benchmark
        Optional daily return series for the information ratio, beta and alpha.
    summary
        Optional dict from :func:`portfoliorl.env.run_policy`, used to report
        turnover and realised transaction costs.
    """
    returns = to_returns(daily)
    n_years = max(len(returns) / ANNUAL, 1e-9)

    if isinstance(daily, pd.DataFrame) and "wealth" in daily.columns:
        final_wealth = float(daily["wealth"].iloc[-1])
    else:
        final_wealth = float(wealth_curve(returns, config.DEFAULT.env.initial_value).iloc[-1])

    values: dict[str, float] = {
        "Final wealth": final_wealth,
        "Total return": total_return(returns),
        "CAGR": cagr(returns),
        "Volatility": annual_volatility(returns),
        "Sharpe": sharpe_ratio(returns, risk_free),
        "Sortino": sortino_ratio(returns),
        "Max drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Longest drawdown (days)": float(longest_drawdown(returns)),
        "VaR 95% (daily)": value_at_risk(returns),
        "CVaR 95% (daily)": conditional_value_at_risk(returns),
        "Hit rate": hit_rate(returns),
        "Skew": float(returns.skew()),
        "Excess kurtosis": float(returns.kurtosis()),
    }

    if summary is not None:
        decisions = summary.get("decisions")
        turnover_total = float(decisions["turnover"].sum()) if decisions is not None else np.nan
        values["Ann. turnover"] = turnover_total / n_years
        values["Total cost"] = float(summary.get("total_cost_fraction", np.nan))
    else:
        values["Ann. turnover"] = np.nan
        values["Total cost"] = np.nan

    if benchmark is not None:
        values["Information ratio"] = information_ratio(returns, benchmark)
        beta, alpha = beta_alpha(returns, benchmark)
        values["Beta"] = beta
        values["Alpha"] = alpha

    return pd.Series(values, name=name)


def scorecard(
    results: Mapping[str, tuple[pd.DataFrame, Mapping[str, object]]],
    *,
    risk_free: pd.Series | float = 0.0,
    benchmark_key: str | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Stack per-strategy summaries into one comparison table.

    Parameters
    ----------
    results
        Mapping of strategy name to the ``(daily, summary)`` pair returned by
        :func:`portfoliorl.env.run_policy`.
    benchmark_key
        Name of the strategy in ``results`` to use as the reference for the
        information ratio, beta and alpha. Naming it here is what keeps the
        benchmark explicit in the report.
    """
    benchmark_returns = None
    if benchmark_key is not None:
        benchmark_returns = to_returns(results[benchmark_key][0])

    rows = []
    for name, (daily, summary) in results.items():
        reference = None if name == benchmark_key else benchmark_returns
        rows.append(
            performance_summary(
                daily,
                risk_free=risk_free,
                benchmark=reference,
                summary=summary,
                name=name,
            )
        )

    table = pd.DataFrame(rows)
    if columns is not None:
        table = table[list(columns)]
    return table


#: Display formats used when rendering a scorecard for the report.
SCORECARD_FORMATS: dict[str, str] = {
    "Final wealth": "${:,.0f}",
    "Total return": "{:.1%}",
    "CAGR": "{:.2%}",
    "Volatility": "{:.2%}",
    "Sharpe": "{:.3f}",
    "Sortino": "{:.3f}",
    "Max drawdown": "{:.2%}",
    "Calmar": "{:.3f}",
    "Longest drawdown (days)": "{:,.0f}",
    "VaR 95% (daily)": "{:.2%}",
    "CVaR 95% (daily)": "{:.2%}",
    "Hit rate": "{:.1%}",
    "Skew": "{:.2f}",
    "Excess kurtosis": "{:.2f}",
    "Ann. turnover": "{:.2f}",
    "Total cost": "{:.2%}",
    "Information ratio": "{:.3f}",
    "Beta": "{:.2f}",
    "Alpha": "{:.2%}",
}


def format_scorecard(table: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy with per-column formatting applied."""
    out = table.copy()
    for column in out.columns:
        fmt = SCORECARD_FORMATS.get(column, "{:.3f}")
        out[column] = out[column].map(lambda v, f=fmt: "—" if pd.isna(v) else f.format(v))
    return out
