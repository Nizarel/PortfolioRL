"""Shared fixtures.

A synthetic price path is used rather than the downloaded cache so that the test
suite runs without network access, without depending on which snapshot happens
to be on disk, and fast enough to run on every edit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfoliorl import config, features


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Four correlated geometric random walks on a business-day calendar."""
    rng = np.random.default_rng(20240501)
    dates = pd.bdate_range("2004-11-01", "2025-12-31", name="date")
    tickers = list(config.DEFAULT.data.tickers)

    # Different drifts and volatilities per asset so that weights genuinely
    # drift apart between rebalances -- a test on four identical series would
    # never exercise the drift logic.
    drift = np.array([0.00035, 0.00015, 0.00025, 0.00005])
    vol = np.array([0.011, 0.008, 0.010, 0.001])
    steps = rng.normal(size=(len(dates), len(tickers))) * vol + drift

    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)
    return prices


@pytest.fixture(scope="session")
def synthetic_rf(synthetic_prices: pd.DataFrame) -> pd.Series:
    annual = pd.Series(
        np.linspace(0.01, 0.05, len(synthetic_prices)), index=synthetic_prices.index
    )
    return annual / config.TRADING_DAYS_PER_YEAR


@pytest.fixture(scope="session")
def dataset(synthetic_prices, synthetic_rf) -> features.Dataset:
    return features.build_dataset(synthetic_prices, synthetic_rf)


@pytest.fixture(scope="session")
def train_dataset(dataset) -> features.Dataset:
    return dataset.split("train")
