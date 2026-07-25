"""Tests for the significance machinery.

Statistical code is unusually easy to get subtly wrong and unusually hard to
notice, because a plausible p-value comes out either way.  These tests check
the properties that must hold by construction: a test comparing a series to
itself must find nothing, a bootstrap must preserve serial dependence, the
Probabilistic Sharpe Ratio must fall when skew turns negative, and the Deflated
Sharpe Ratio must fall as the number of trials rises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfoliorl import config, significance as sig


ANNUAL = config.TRADING_DAYS_PER_YEAR


@pytest.fixture
def iid_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0004, 0.01, 2_000)


@pytest.fixture
def ar1_returns():
    """Serially correlated returns: the case an i.i.d. bootstrap would mishandle."""
    rng = np.random.default_rng(7)
    eps = rng.normal(0.0004, 0.01, 3_000)
    x = np.zeros_like(eps)
    for t in range(1, len(x)):
        x[t] = 0.35 * x[t - 1] + eps[t]
    return x


# --------------------------------------------------------------------------- #
# Bootstrap mechanics
# --------------------------------------------------------------------------- #
def test_bootstrap_indices_stay_in_range_and_are_the_right_length():
    rng = np.random.default_rng(0)
    idx = sig.stationary_bootstrap_indices(500, expected_block=10, size=500, rng=rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_block_length_is_geometric_with_the_requested_mean():
    rng = np.random.default_rng(1)
    idx = sig.stationary_bootstrap_indices(10_000, expected_block=20, size=200_000, rng=rng)
    # A new block starts wherever the index is not the previous one plus one.
    starts = np.flatnonzero(np.diff(idx) != 1) + 1
    lengths = np.diff(np.concatenate([[0], starts, [len(idx)]]))
    assert 17 < lengths.mean() < 23


def test_bootstrap_preserves_serial_correlation(ar1_returns):
    """The whole reason for a *block* bootstrap: an i.i.d. resample would destroy
    the autocorrelation and understate the uncertainty of the Sharpe ratio."""
    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    original = lag1(ar1_returns)
    blocks = sig.stationary_bootstrap(ar1_returns, n_boot=40, expected_block=25, seed=0)
    block_rho = np.mean([lag1(b) for b in blocks])

    rng = np.random.default_rng(0)
    iid_rho = np.mean(
        [lag1(rng.choice(ar1_returns, size=len(ar1_returns), replace=True)) for _ in range(40)]
    )

    assert original > 0.25
    assert block_rho > 0.7 * original       # most of the dependence survives
    assert abs(iid_rho) < 0.05              # the naive bootstrap destroys it


def test_paired_bootstrap_resamples_both_series_together():
    """If the two columns were resampled independently, a series compared with a
    copy of itself would show a non-zero difference in some resamples."""
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0.0005, 0.01, 800))
    res = sig.bootstrap_sharpe_difference(x, x.copy(), n_boot=200, seed=0)
    assert res.observed == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(res.distribution, 0.0, atol=1e-10)
    assert res.p_value >= 0.99


def test_bootstrap_aligns_two_series_on_their_shared_dates():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.0006, 0.01, 600), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.01, 400), index=idx[:400])
    res = sig.bootstrap_sharpe_difference(a, b, n_boot=100, seed=0)
    assert np.isfinite(res.observed)
    # Only the overlapping window is used, so the observed difference must match
    # a manual computation on that window.
    manual = sig._sharpe(a.loc[b.index].to_numpy()) - sig._sharpe(b.to_numpy())
    assert res.observed == pytest.approx(float(manual), rel=1e-12)


def test_a_genuinely_better_strategy_is_detected():
    rng = np.random.default_rng(11)
    n = 3_000
    weak = rng.normal(0.0001, 0.01, n)
    strong = weak + rng.normal(0.0006, 0.002, n)
    res = sig.bootstrap_sharpe_difference(strong, weak, n_boot=500, seed=0)
    assert res.observed > 0
    assert res.ci_low > 0            # the interval excludes "no difference"
    assert res.significant(0.05)


def test_two_identical_processes_are_not_declared_different():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0004, 0.01, 2_000)
    b = rng.normal(0.0004, 0.01, 2_000)
    res = sig.bootstrap_sharpe_difference(a, b, n_boot=500, seed=1)
    assert not res.significant(0.05)
    assert res.ci_low < 0 < res.ci_high


def test_bootstrap_result_summary_is_reportable(iid_returns):
    rng = np.random.default_rng(13)
    other = rng.normal(0.0002, 0.01, len(iid_returns))
    res = sig.bootstrap_sharpe_difference(iid_returns, other, n_boot=200, seed=0)
    s = res.summary()
    assert {"Observed difference", "95% CI low", "95% CI high", "p-value"} <= set(s.index)
    assert s["95% CI low"] < s["95% CI high"]


# --------------------------------------------------------------------------- #
# Lo standard error
# --------------------------------------------------------------------------- #
def test_lo_standard_error_shrinks_with_sample_size():
    rng = np.random.default_rng(5)
    short = sig.lo_standard_error(rng.normal(0.0004, 0.01, 250))
    long = sig.lo_standard_error(rng.normal(0.0004, 0.01, 4_000))
    assert short > long > 0
    # Roughly the 1/sqrt(T) scaling the formula implies.
    assert short / long == pytest.approx(4.0, rel=0.35)


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------- #
def test_psr_is_high_for_a_clearly_positive_sharpe(iid_returns):
    psr = sig.probabilistic_sharpe_ratio(iid_returns, benchmark_sr=0.0)
    assert 0.9 < psr <= 1.0


def test_psr_falls_as_the_benchmark_rises(iid_returns):
    values = [sig.probabilistic_sharpe_ratio(iid_returns, b) for b in (0.0, 0.5, 1.0, 2.0)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] < values[0]


def test_negative_skew_reduces_the_psr():
    """Two series with the same mean and standard deviation but different skew
    must not receive the same confidence -- that is the entire point of the PSR."""
    rng = np.random.default_rng(9)
    n = 3_000
    symmetric = rng.normal(0, 1, n)
    left_skewed = -stats.skewnorm.rvs(a=6, size=n, random_state=9)

    def standardise(x):
        return 0.0005 + 0.01 * (x - x.mean()) / x.std(ddof=1)

    a, b = standardise(symmetric), standardise(left_skewed)
    assert stats.skew(b) < -0.5
    assert sig.probabilistic_sharpe_ratio(a) > sig.probabilistic_sharpe_ratio(b)


def test_psr_is_nan_for_a_degenerate_series():
    assert np.isnan(sig.probabilistic_sharpe_ratio(np.zeros(100)))
    assert np.isnan(sig.probabilistic_sharpe_ratio(np.array([0.01])))


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_grows_with_the_number_of_trials():
    values = [sig.expected_max_sharpe(n, 0.25) for n in (2, 10, 50, 500)]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert sig.expected_max_sharpe(1, 0.25) == 0.0
    assert sig.expected_max_sharpe(50, 0.0) == 0.0


def test_expected_max_sharpe_matches_a_monte_carlo_check():
    """Sanity-check the closed form against simulation, because an analytic
    formula copied from a paper is exactly the kind of thing that gets
    transcribed wrong."""
    rng = np.random.default_rng(0)
    var = 0.09
    n_trials = 40
    draws = rng.normal(0.0, np.sqrt(var), size=(20_000, n_trials))
    empirical = draws.max(axis=1).mean()
    assert sig.expected_max_sharpe(n_trials, var) == pytest.approx(empirical, rel=0.1)


def test_deflated_sharpe_falls_as_more_configurations_are_tried(iid_returns):
    few = sig.deflated_sharpe_ratio(iid_returns, n_trials=2, sr_variance=0.09)
    many = sig.deflated_sharpe_ratio(iid_returns, n_trials=500, sr_variance=0.09)
    assert few > many
    assert 0.0 <= many <= 1.0


def test_deflated_sharpe_punishes_a_wide_spread_of_trial_results(iid_returns):
    tight = sig.deflated_sharpe_ratio(iid_returns, n_trials=48, sr_variance=0.01)
    wide = sig.deflated_sharpe_ratio(iid_returns, n_trials=48, sr_variance=0.50)
    assert tight > wide


# --------------------------------------------------------------------------- #
# Minimum track record length
# --------------------------------------------------------------------------- #
def test_minimum_track_record_length_is_finite_for_a_good_strategy(iid_returns):
    mtrl = sig.minimum_track_record_length(iid_returns, benchmark_sr=0.0)
    assert 0 < mtrl < len(iid_returns)


def test_minimum_track_record_length_is_infinite_when_there_is_no_edge():
    rng = np.random.default_rng(6)
    flat = rng.normal(0.0, 0.01, 1_000)
    assert sig.minimum_track_record_length(flat, benchmark_sr=1.0) == np.inf


def test_minimum_track_record_length_grows_with_the_hurdle(iid_returns):
    low = sig.minimum_track_record_length(iid_returns, benchmark_sr=0.0)
    high = sig.minimum_track_record_length(iid_returns, benchmark_sr=0.5)
    assert high > low


# --------------------------------------------------------------------------- #
# Seed-level comparisons
# --------------------------------------------------------------------------- #
def test_paired_seed_test_reports_both_parametric_and_rank_based_results():
    a = np.array([1.10, 1.05, 1.20, 0.95, 1.15, 1.08, 1.12, 1.00])
    b = a - 0.12
    out = sig.paired_seed_test(a, b)
    assert out["n seeds"] == 8
    assert out["mean difference"] == pytest.approx(0.12)
    assert out["t p-value"] < 0.01
    assert out["Wilcoxon p-value"] < 0.05
    assert out["Cohen's d"] > 3


def test_paired_seed_test_survives_identical_inputs():
    a = np.array([1.0, 1.1, 1.2, 0.9])
    out = sig.paired_seed_test(a, a)
    assert out["mean difference"] == 0.0
    assert out["Wilcoxon p-value"] == 1.0


def test_holm_bonferroni_is_stricter_than_the_raw_p_values():
    raw = {"vs 60/40": 0.01, "vs SPY": 0.03, "vs equal weight": 0.04, "vs cash": 0.20}
    out = sig.holm_bonferroni(raw, alpha=0.05)
    assert (out["Holm-adjusted"] >= out["p-value"] - 1e-12).all()
    assert out["Holm-adjusted"].is_monotonic_increasing   # step-down ordering
    assert out.loc["vs 60/40", "reject at alpha"]
    assert not out.loc["vs cash", "reject at alpha"]


def test_holm_bonferroni_matches_the_hand_computed_values():
    out = sig.holm_bonferroni({"a": 0.01, "b": 0.04}, alpha=0.05)
    assert out.loc["a", "Holm-adjusted"] == pytest.approx(0.02)   # 2 * 0.01
    assert out.loc["b", "Holm-adjusted"] == pytest.approx(0.04)   # 1 * 0.04
