"""Statistical significance for backtests, where the usual t-test does not apply.

The problem
-----------
"Strategy A has a higher Sharpe ratio than strategy B" is an estimate, not a
fact.  Three things make the naive comparison unsafe on financial data:

1. **Autocorrelation and heteroskedasticity.**  Daily returns are not i.i.d.
   Volatility clusters, so the effective sample size is far smaller than the
   number of observations, and the classic standard error
   :math:`\\sigma_{SR} = \\sqrt{(1 + SR^2/2)/T}` is optimistic.
2. **Non-normality.**  Returns are negatively skewed and fat-tailed.  A Sharpe
   ratio is a ratio of moments, so its sampling distribution inherits those
   higher moments -- a point made precisely by Bailey & Lopez de Prado's
   Probabilistic Sharpe Ratio.
3. **Selection.**  If 48 configurations were evaluated and the best reported,
   the reported Sharpe is the *maximum* of 48 noisy draws.  Its expectation is
   strictly greater than the true Sharpe of the best configuration, even if
   every configuration is worthless.

Tools provided
--------------
* :func:`stationary_bootstrap` -- Politis & Romano (1994) resampling with
  geometrically distributed block lengths, which preserves short-range serial
  dependence while remaining stationary at the block boundaries.  Fixed-length
  block bootstraps introduce artefacts at the joins; the geometric length
  removes them.
* :func:`bootstrap_sharpe_difference` -- the confidence interval and p-value
  actually reported in notebook 05.
* :func:`probabilistic_sharpe_ratio` -- the probability that the true Sharpe
  exceeds a benchmark, adjusted for skew and kurtosis.
* :func:`deflated_sharpe_ratio` -- the same, with the benchmark raised to the
  Sharpe one would expect from the *best of N* trials of pure noise.

References
----------
Politis & Romano (1994), "The Stationary Bootstrap", *JASA* 89.  Lo (2002),
"The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58.  Bailey &
Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", *Journal of Risk*
15; and (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio Management*
40.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config

ANNUAL = config.TRADING_DAYS_PER_YEAR
EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Block bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int, expected_block: float, size: int | None = None, rng=None
) -> np.ndarray:
    """Index array for one stationary-bootstrap resample.

    Blocks start at a uniformly random position and continue with probability
    :math:`1 - 1/L` at each step, giving geometrically distributed block lengths
    with mean ``expected_block``.  Indices wrap around the end of the sample,
    which is what makes the resampled series stationary.

    ``expected_block`` should be long enough to span the dependence being
    preserved.  For daily equity returns, volatility clustering operates over
    roughly a fortnight, so 10-20 trading days is the usual choice.
    """
    rng = rng or np.random.default_rng()
    size = size or n
    p = 1.0 / expected_block

    idx = np.empty(size, dtype=np.int64)
    current = int(rng.integers(n))
    for t in range(size):
        idx[t] = current
        if rng.random() < p:
            current = int(rng.integers(n))       # start a new block
        else:
            current = (current + 1) % n          # continue, wrapping around
    return idx


def stationary_bootstrap(
    data: np.ndarray | pd.Series | pd.DataFrame,
    n_boot: int = 1_000,
    expected_block: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate ``n_boot`` resamples, preserving cross-sectional alignment.

    When ``data`` is 2-D the *same* index array is applied to every column, so
    a paired comparison of two strategies keeps them synchronised: both are
    resampled over the same market days, which is the point of a paired test.
    """
    arr = np.asarray(data, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[:, None]

    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    out = np.empty((n_boot, n, arr.shape[1]))
    for b in range(n_boot):
        out[b] = arr[stationary_bootstrap_indices(n, expected_block, n, rng)]
    return out[:, :, 0] if single else out


# --------------------------------------------------------------------------- #
# Sharpe ratio inference
# --------------------------------------------------------------------------- #
def _sharpe(x: np.ndarray, axis: int = -1) -> np.ndarray:
    sd = np.std(x, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.mean(x, axis=axis) / sd * np.sqrt(ANNUAL), np.nan)


def lo_standard_error(returns: np.ndarray | pd.Series) -> float:
    """Lo (2002) i.i.d. standard error of an annualised Sharpe ratio.

    Reported alongside the bootstrap interval so a reader can see how much the
    serial dependence in the data actually costs.  It is the *optimistic* bound.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("nan")
    sr = float(_sharpe(r))
    return float(np.sqrt((1.0 + 0.5 * (sr / np.sqrt(ANNUAL)) ** 2) / n) * np.sqrt(ANNUAL))


@dataclass
class BootstrapResult:
    """Outcome of a paired bootstrap comparison of two return streams."""

    observed: float
    mean: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    expected_block: float
    distribution: np.ndarray

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Observed difference": self.observed,
                "Bootstrap mean": self.mean,
                "Bootstrap SE": self.std_error,
                "95% CI low": self.ci_low,
                "95% CI high": self.ci_high,
                "p-value": self.p_value,
                "Resamples": self.n_boot,
            }
        )


def bootstrap_sharpe_difference(
    strategy: np.ndarray | pd.Series,
    benchmark: np.ndarray | pd.Series,
    n_boot: int = 2_000,
    expected_block: float = 10.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired stationary-bootstrap test of :math:`SR_{strategy} - SR_{benchmark}`.

    The two series are resampled with the **same** index draw, so every
    resample compares the strategies over an identical (bootstrapped) market
    history.  Comparing independently resampled series would test a different
    and much weaker hypothesis, and would inflate the variance of the
    difference by roughly a factor of two.

    The p-value is two-sided and computed from the bootstrap distribution
    recentred on zero -- the standard percentile-t-free approach when the null
    is "no difference".
    """
    a = pd.Series(strategy).astype(float)
    b = pd.Series(benchmark).astype(float)
    if isinstance(strategy, pd.Series) and isinstance(benchmark, pd.Series):
        # Align on dates so a shorter series cannot be silently compared
        # against a different window.
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        a, b = joined.iloc[:, 0], joined.iloc[:, 1]

    paired = np.column_stack([a.to_numpy(), b.to_numpy()])
    observed = float(_sharpe(paired[:, 0]) - _sharpe(paired[:, 1]))

    samples = stationary_bootstrap(paired, n_boot, expected_block, seed)
    diffs = _sharpe(samples[:, :, 0], axis=1) - _sharpe(samples[:, :, 1], axis=1)
    diffs = diffs[np.isfinite(diffs)]

    centred = diffs - diffs.mean()
    p_value = float(np.mean(np.abs(centred) >= abs(observed)))

    return BootstrapResult(
        observed=observed,
        mean=float(diffs.mean()),
        std_error=float(diffs.std(ddof=1)),
        ci_low=float(np.quantile(diffs, alpha / 2)),
        ci_high=float(np.quantile(diffs, 1 - alpha / 2)),
        p_value=p_value,
        n_boot=len(diffs),
        expected_block=expected_block,
        distribution=diffs,
    )


# --------------------------------------------------------------------------- #
# Probabilistic and Deflated Sharpe
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    returns: np.ndarray | pd.Series, benchmark_sr: float = 0.0
) -> float:
    r"""Probability that the true Sharpe ratio exceeds ``benchmark_sr``.

    .. math::
       \widehat{PSR}(SR^{*}) = \Phi\!\left(
         \frac{(\widehat{SR} - SR^{*})\sqrt{T - 1}}
              {\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}
       \right)

    where :math:`\gamma_3` is skewness and :math:`\gamma_4` is kurtosis (not
    excess).  The denominator is the correction that matters: **negative skew
    and fat tails both reduce** the PSR for a given Sharpe, which is exactly the
    right behaviour for a strategy whose losses arrive in clusters.

    ``benchmark_sr`` is annualised, as is the estimated Sharpe.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or np.std(r, ddof=1) == 0:
        return float("nan")

    sr_daily = np.mean(r) / np.std(r, ddof=1)
    target_daily = benchmark_sr / np.sqrt(ANNUAL)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))

    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily**2))
    z = (sr_daily - target_daily) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    r"""Sharpe you would expect from the *best of N* strategies of zero skill.

    .. math::
       E[\max SR] \approx \sqrt{V}\left[(1-\gamma)\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
         + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]

    with :math:`\gamma` the Euler-Mascheroni constant and :math:`V` the variance
    of Sharpe ratios *across* trials.  This is the benchmark the Deflated Sharpe
    Ratio tests against, and it is the number that makes the multiple-testing
    problem concrete: with 48 trials and a cross-trial Sharpe standard deviation
    of 0.3, pure noise is expected to produce a best Sharpe of roughly 0.7.
    """
    if n_trials < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    n = float(n_trials)
    term = (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n)
    term += EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n * np.e))
    return float(np.sqrt(sr_variance) * term)


def deflated_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    n_trials: int,
    sr_variance: float,
) -> float:
    """Probabilistic Sharpe Ratio against the expected maximum of ``n_trials``.

    This is the single most important number in the results notebook.  A
    strategy can have a perfectly respectable Sharpe and a DSR near 0.5,
    meaning: *given how many configurations were tried, this result is what
    noise would have produced anyway.*

    ``sr_variance`` should be the observed variance of annualised Sharpe ratios
    across the trials that were run -- not an assumption.  This project has it
    directly, from the Optuna study and the multi-seed runs.
    """
    threshold = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=threshold)


def minimum_track_record_length(
    returns: np.ndarray | pd.Series, benchmark_sr: float = 0.0, confidence: float = 0.95
) -> float:
    """How many observations would be needed to call the Sharpe significant.

    Answers the reviewer's question "is five years of test data enough?" with a
    number rather than an opinion.  Returns ``inf`` when the observed Sharpe
    does not exceed the benchmark at all.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return float("nan")

    sr_daily = np.mean(r) / np.std(r, ddof=1)
    target_daily = benchmark_sr / np.sqrt(ANNUAL)
    if sr_daily <= target_daily:
        return float("inf")

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))
    numerator = 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily**2
    z = stats.norm.ppf(confidence)
    return float(1.0 + numerator * (z / (sr_daily - target_daily)) ** 2)


# --------------------------------------------------------------------------- #
# Multiple comparisons across seeds
# --------------------------------------------------------------------------- #
def paired_seed_test(
    a: np.ndarray | pd.Series, b: np.ndarray | pd.Series
) -> pd.Series:
    """Paired comparison of two variants across matched seeds.

    Reports both the parametric paired t-test and the non-parametric Wilcoxon
    signed-rank test.  With eight seeds, normality is unverifiable, so the
    Wilcoxon result is the one to trust when they disagree.  Cohen's *d* is
    included because with a sample of eight the effect size is more informative
    than the p-value.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:      # raised when all differences are zero
        w_stat, w_p = np.nan, 1.0

    sd = diff.std(ddof=1)
    return pd.Series(
        {
            "n seeds": n,
            "mean difference": diff.mean(),
            "sd of difference": sd,
            "Cohen's d": diff.mean() / sd if sd > 0 else np.nan,
            "t statistic": t_stat,
            "t p-value": t_p,
            "Wilcoxon statistic": w_stat,
            "Wilcoxon p-value": w_p,
        }
    )


def holm_bonferroni(p_values: pd.Series | dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction for a family of comparisons.

    Comparing the agent against nine benchmarks means nine chances to find a
    "significant" difference by luck.  Holm controls the family-wise error rate
    while being uniformly more powerful than plain Bonferroni, and it needs no
    independence assumption -- which matters here, since the benchmarks are
    heavily correlated with each other.
    """
    s = pd.Series(p_values).dropna().sort_values()
    m = len(s)
    adjusted, running = [], 0.0
    for i, p in enumerate(s.to_numpy()):
        running = max(running, (m - i) * p)
        adjusted.append(min(1.0, running))
    return pd.DataFrame(
        {"p-value": s, "Holm-adjusted": adjusted, "reject at alpha": np.array(adjusted) < alpha},
        index=s.index,
    )
