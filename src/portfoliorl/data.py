"""Market data acquisition, on-disk caching and quality reporting.

Design notes
------------
**Frozen cache.**  ``yfinance`` hits a live endpoint whose history is silently
revised (dividend restatements, symbol changes, occasional bad ticks).  A
project whose results cannot be reproduced next month is not a scientific
result, so every download is written to ``data/raw/`` together with a manifest
recording *when* it was fetched and *which* configuration produced it.  All
downstream code reads the cache, never the network.

**No Parquet.**  The cache is CSV.  Neither ``pyarrow`` nor ``fastparquet``
ships a wheel for this machine's platform (Windows/ARM64), and for a few
thousand daily rows the size difference is irrelevant.  CSV is also diffable,
which makes an accidental change to the frozen dataset visible in ``git``.

**yfinance gotchas that this module deliberately handles.**

1. ``download(..., auto_adjust=True)`` is the *default* in modern yfinance.
   The returned ``Close`` column is therefore already adjusted for splits and
   dividends, and there is **no** ``Adj Close`` column.  Code copied from older
   tutorials that reaches for ``Adj Close`` will raise ``KeyError`` here.
2. Multi-ticker downloads return a two-level column index
   ``(Price, Ticker)``.  We flatten it to a plain ticker-per-column frame.
3. The ``end`` argument is **exclusive**.  We add one day so that the configured
   end date is actually included.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import load_json, save_json

PRICES_CSV = config.DATA_RAW / "prices_close.csv"
VOLUME_CSV = config.DATA_RAW / "volume.csv"
RISK_FREE_CSV = config.DATA_RAW / "risk_free.csv"
MANIFEST_JSON = config.DATA_RAW / "download_manifest.json"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around ``yfinance.download`` with the project's conventions.

    Imported lazily: the rest of the pipeline must remain importable (and
    testable) on a machine with no network stack or no yfinance installed.
    """
    import yfinance as yf

    # `end` is exclusive in yfinance, so push it out by one day to include it.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,  # explicit even though it is the default -- see module docstring
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers} over {start}..{end}")
    return raw


def _flatten(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Extract one price field from yfinance's ``(Price, Ticker)`` column index."""
    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw[field].copy()
    else:
        # Single-ticker downloads come back with flat columns.
        frame = raw[[field]].copy()
        frame.columns = tickers
    # Preserve the configured universe order rather than yfinance's alphabetical
    # ordering, because ACTION_ALLOCATIONS rows are positional.
    return frame.reindex(columns=tickers)


def download_market_data(
    cfg: config.DataConfig | None = None, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (or load from cache) adjusted closes and volumes for the universe.

    Returns
    -------
    (prices, volumes)
        Both indexed by trading date, one column per ticker in ``cfg.tickers``
        order.
    """
    cfg = cfg or config.DEFAULT.data
    config.ensure_dirs()

    if PRICES_CSV.exists() and not force:
        prices = _read_cached(PRICES_CSV)
        volumes = _read_cached(VOLUME_CSV) if VOLUME_CSV.exists() else pd.DataFrame(index=prices.index)
        return prices, volumes

    tickers = list(cfg.tickers)
    raw = _yf_download(tickers, cfg.start, cfg.end)

    prices = _flatten(raw, "Close", tickers)
    volumes = _flatten(raw, "Volume", tickers)

    prices.index.name = "date"
    volumes.index.name = "date"
    prices.to_csv(PRICES_CSV)
    volumes.to_csv(VOLUME_CSV)
    _write_manifest(cfg, prices)
    return prices, volumes


def download_risk_free(cfg: config.DataConfig | None = None, force: bool = False) -> pd.Series:
    """Fetch (or load) the daily risk-free rate as a *decimal daily* rate.

    ``^IRX`` quotes the 13-week Treasury bill discount rate as an annualised
    percentage (e.g. ``5.32`` means 5.32%/yr).  The Sharpe and Sortino ratios in
    this project are computed on daily excess returns, so the series is
    converted to a daily simple rate by dividing by 100 and by the number of
    trading days per year.  Gaps (the index does not print on every equity
    trading day) are forward-filled, which is the correct treatment for a rate
    that is constant between quotes.
    """
    cfg = cfg or config.DEFAULT.data
    config.ensure_dirs()

    if RISK_FREE_CSV.exists() and not force:
        return _read_cached(RISK_FREE_CSV).iloc[:, 0]

    raw = _yf_download([cfg.risk_free_ticker], cfg.start, cfg.end)
    annual_pct = _flatten(raw, "Close", [cfg.risk_free_ticker]).iloc[:, 0]

    daily = (annual_pct / 100.0) / config.TRADING_DAYS_PER_YEAR
    daily.name = "rf_daily"
    daily.index.name = "date"
    daily.to_frame().to_csv(RISK_FREE_CSV)
    return daily


def _read_cached(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _write_manifest(cfg: config.DataConfig, prices: pd.DataFrame) -> None:
    """Record provenance so a stale or mismatched cache is detectable."""
    save_json(
        {
            "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "tickers": list(cfg.tickers),
            "requested_start": cfg.start,
            "requested_end": cfg.end,
            "actual_first_date": str(prices.index.min().date()),
            "actual_last_date": str(prices.index.max().date()),
            "n_rows": int(len(prices)),
            "yfinance_auto_adjust": True,
        },
        MANIFEST_JSON,
    )


def cache_manifest() -> dict | None:
    """Return the download manifest, or ``None`` if nothing has been cached."""
    return load_json(MANIFEST_JSON) if MANIFEST_JSON.exists() else None


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def quality_report(prices: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ticker summarising coverage and data-integrity checks.

    The checks are chosen to catch the specific ways free market data goes
    wrong: a symbol that did not exist yet (``first_date``), stale quotes that
    would fake a zero-volatility day (``n_zero_returns``), and bad ticks that
    would hand the agent an impossible reward (``n_extreme_moves``).
    """
    rows = []
    returns = prices.pct_change()
    for ticker in prices.columns:
        px = prices[ticker].dropna()
        ret = returns[ticker].dropna()
        row = {
            "first_date": px.index.min().date(),
            "last_date": px.index.max().date(),
            "n_obs": len(px),
            "n_missing": int(prices[ticker].isna().sum()),
            "pct_missing": float(prices[ticker].isna().mean()),
            # A run of identical closes means the feed went stale; it would show
            # up to the agent as a risk-free asset.
            "n_zero_returns": int((ret == 0).sum()),
            # >20% in a day is possible for equities in a crash but is far more
            # often a split that was not adjusted or a bad print.
            "n_extreme_moves": int((ret.abs() > 0.20).sum()),
            "max_abs_move": float(ret.abs().max()),
            "ann_vol": float(ret.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
        }
        if volumes is not None and ticker in volumes.columns:
            row["median_volume"] = float(volumes[ticker].median())
        rows.append(row)
    return pd.DataFrame(rows, index=prices.columns).rename_axis("ticker")


def align_and_trim(
    prices: pd.DataFrame, risk_free: pd.Series, cfg: config.DataConfig | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict everything to dates where the *entire* universe is tradable.

    An episode in which one asset has no price is not a portfolio problem, so
    rows before the last inception date are dropped outright rather than being
    imputed.  The risk-free series is forward-filled onto the equity calendar
    because a T-bill rate is a step function between quotes.
    """
    cfg = cfg or config.DEFAULT.data
    prices = prices.dropna(how="any").sort_index()
    prices = prices.loc[str(cfg.start) : str(cfg.end)]

    rf = risk_free.sort_index().reindex(prices.index).ffill()
    # A leading NaN can survive the ffill if the rate series starts late; a
    # short-rate of zero is a safer default than dropping tradable days.
    rf = rf.fillna(0.0)
    return prices, rf


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_bounds(cfg: config.DataConfig | None = None) -> dict[str, tuple[str, str]]:
    cfg = cfg or config.DEFAULT.data
    return {
        "train": (cfg.train_start, cfg.train_end),
        "valid": (cfg.valid_start, cfg.valid_end),
        "test": (cfg.test_start, cfg.test_end),
    }


def slice_split(
    frame: pd.DataFrame | pd.Series, split: str, cfg: config.DataConfig | None = None
) -> pd.DataFrame | pd.Series:
    """Return the rows of ``frame`` belonging to ``split``.

    Splits are strictly chronological and non-overlapping.  Random or k-fold
    splitting would leak future information into training through the
    autocorrelation of volatility, which is exactly the signal the agent trades
    on -- it would inflate every reported metric.
    """
    start, end = split_bounds(cfg)[split]
    return frame.loc[start:end]


def split_series(index: pd.DatetimeIndex, cfg: config.DataConfig | None = None) -> pd.Series:
    """Label each date with its split, for plotting and sanity checks."""
    bounds = split_bounds(cfg)
    labels = pd.Series("unused", index=index, dtype="object")
    for name, (start, end) in bounds.items():
        labels.loc[start:end] = name
    return labels
