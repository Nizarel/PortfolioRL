"""Tests for causal feature construction and train-only standardisation.

The causality test is the important one: a look-ahead bug is silent, produces
better-looking results, and invalidates every number in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfoliorl import config, data, features


@pytest.fixture(scope="module")
def synthetic_prices() -> pd.DataFrame:
    """A reproducible 4-asset geometric random walk on business days."""
    rng = np.random.default_rng(12345)
    dates = pd.bdate_range("2004-11-01", "2025-12-31", name="date")
    tickers = list(config.DEFAULT.data.tickers)
    steps = rng.normal(loc=0.0003, scale=0.01, size=(len(dates), len(tickers)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)
    return prices


@pytest.fixture(scope="module")
def synthetic_rf(synthetic_prices: pd.DataFrame) -> pd.Series:
    """A slowly drifting daily risk-free rate on the same calendar."""
    annual = pd.Series(
        np.linspace(0.01, 0.05, len(synthetic_prices)), index=synthetic_prices.index
    )
    return annual / config.TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
def test_features_do_not_use_same_day_or_future_prices(synthetic_prices, synthetic_rf):
    """Perturbing the price on day k must not change any feature row up to k.

    Feature row ``k`` is meant to be knowable before day ``k`` trades.  If the
    ``.shift(1)`` in ``build_market_features`` were removed, this test fails on
    row ``k`` immediately.
    """
    baseline = features.build_market_features(synthetic_prices, synthetic_rf)

    k = 3000  # deep enough that every look-back window is fully populated
    perturbed_prices = synthetic_prices.copy()
    perturbed_prices.iloc[k] *= 1.25  # a large, unmistakable shock

    perturbed = features.build_market_features(perturbed_prices, synthetic_rf)

    pd.testing.assert_frame_equal(baseline.iloc[: k + 1], perturbed.iloc[: k + 1])

    # Sanity check the test itself: the very next row *must* react, otherwise we
    # would be proving nothing (e.g. if the shock had been silently dropped).
    assert not np.allclose(
        baseline.iloc[k + 1].to_numpy(), perturbed.iloc[k + 1].to_numpy(), equal_nan=True
    )


def test_feature_row_matches_manual_lagged_computation(synthetic_prices, synthetic_rf):
    """Cross-check one feature against an independently written formula."""
    cfg = config.DEFAULT.data
    built = features.build_market_features(synthetic_prices, synthetic_rf)

    k = 2500
    # 20-day annualised vol as of the *previous* day, computed by hand.
    window = synthetic_prices["SPY"].pct_change().iloc[k - cfg.vol_window : k]
    expected = window.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)

    assert built["SPY_vol20"].iloc[k] == pytest.approx(expected, rel=1e-10)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def test_dataset_has_expected_shape_and_no_nans(synthetic_prices, synthetic_rf):
    ds = features.build_dataset(synthetic_prices, synthetic_rf)

    n_assets = config.DEFAULT.data.n_assets
    # 5 per-asset features, plus SPY/TLT correlation, plus 2 rate features.
    assert ds.n_market_features == 5 * n_assets + 3
    assert ds.obs_dim == ds.n_market_features + features.N_PORTFOLIO_FEATURES
    assert ds.obs_dim == 31, "Observation dimension is quoted as 31 in the report"

    assert not ds.features.isna().any().any()
    assert not ds.prices.isna().any().any()
    assert len(ds.features) == len(ds.prices) == len(ds.returns) == len(ds.risk_free)


def test_warmup_is_trimmed_before_the_training_window(synthetic_prices, synthetic_rf):
    """No NaN-driven silent shortening of the training period."""
    cfg = config.DEFAULT.data
    ds = features.build_dataset(synthetic_prices, synthetic_rf)
    train = ds.split("train")

    assert len(train.features) > 2000, "Training window unexpectedly short"
    assert train.features.index.min() >= pd.Timestamp(cfg.train_start)
    assert train.features.index.max() <= pd.Timestamp(cfg.train_end)


def test_splits_are_chronological_and_disjoint(synthetic_prices, synthetic_rf):
    ds = features.build_dataset(synthetic_prices, synthetic_rf)
    train, valid, test = ds.split("train"), ds.split("valid"), ds.split("test")

    assert train.dates.max() < valid.dates.min()
    assert valid.dates.max() < test.dates.min()
    assert len(train.dates.intersection(test.dates)) == 0


# --------------------------------------------------------------------------- #
# Scaler
# --------------------------------------------------------------------------- #
def test_scaler_is_fitted_on_training_rows_only(synthetic_prices, synthetic_rf):
    """The scaler's statistics must equal the train-window statistics exactly."""
    cfg = config.DEFAULT.data
    ds = features.build_dataset(synthetic_prices, synthetic_rf)

    train_raw = ds.features_raw.loc[cfg.train_start : cfg.train_end]
    pd.testing.assert_series_equal(ds.scaler.mean_, train_raw.mean(), check_names=False)

    # Standardised training features are therefore centred; the test split is
    # NOT expected to be, and that asymmetry is the point.
    train_scaled = ds.features.loc[cfg.train_start : cfg.train_end]
    assert np.abs(train_scaled.mean()).max() < 0.05


def test_scaler_clips_extreme_values(synthetic_prices, synthetic_rf):
    ds = features.build_dataset(synthetic_prices, synthetic_rf)
    assert ds.features.to_numpy().max() <= ds.scaler.clip + 1e-9
    assert ds.features.to_numpy().min() >= -ds.scaler.clip - 1e-9


def test_scaler_round_trips_through_json(synthetic_prices, synthetic_rf):
    """Serialisation matters: evaluation must reuse the training scaler."""
    ds = features.build_dataset(synthetic_prices, synthetic_rf)
    restored = features.FeatureScaler.from_dict(ds.scaler.to_dict())
    pd.testing.assert_frame_equal(
        restored.transform(ds.features_raw), ds.scaler.transform(ds.features_raw)
    )


# --------------------------------------------------------------------------- #
# Split helpers
# --------------------------------------------------------------------------- #
def test_split_labels_cover_every_configured_window():
    cfg = config.DEFAULT.data
    index = pd.bdate_range(cfg.train_start, cfg.test_end)
    labels = data.split_series(index, cfg)
    assert set(labels.unique()) <= {"train", "valid", "test", "unused"}
    assert (labels == "train").sum() > 0
    assert (labels == "valid").sum() > 0
    assert (labels == "test").sum() > 0
