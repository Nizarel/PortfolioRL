"""Causal feature engineering and train-only standardisation.

The single most important property of this module is **causality**: the feature
row stamped at date *t* is computed exclusively from information available
strictly *before* *t*.  Every rolling statistic is therefore followed by
``.shift(1)``.

Why this matters more here than in ordinary supervised learning: a rolling
20-day volatility computed *through* day *t* contains day *t*'s return.  The
agent would observe today's realised move, then be paid for a position it is
pretending to have taken before seeing it.  Backtests built this way produce
spectacular, entirely fictional Sharpe ratios.  Notebook 01 demonstrates the
size of the distortion rather than merely asserting it.

The observation the agent finally receives is 31-dimensional:

===============================  ====  ==========================================
Block                            Dim   Source
===============================  ====  ==========================================
Per-asset market features        20    this module (4 assets x 5 features)
SPY/TLT rolling correlation       1    this module
Risk-free level and 20d change    2    this module
Current portfolio weights         4    environment (drifted, post-return)
Portfolio rolling volatility      1    environment
Current drawdown                  1    environment
Drawdown duration                 1    environment
Fraction of episode elapsed       1    environment
===============================  ====  ==========================================

The last eight are appended by :mod:`portfoliorl.env` because they depend on the
agent's own trajectory, not on the market alone.  Including them is not
cosmetic: the reward penalises portfolio volatility and drawdown, so without
them the problem is not Markov and the value function is unlearnable in
principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

#: Number of observation dimensions contributed by the environment (portfolio
#: state), as opposed to the market.  Kept here so the two modules cannot drift
#: apart silently.
N_PORTFOLIO_FEATURES = 8


# --------------------------------------------------------------------------- #
# Market features
# --------------------------------------------------------------------------- #
def build_market_features(
    prices: pd.DataFrame,
    risk_free: pd.Series,
    cfg: config.DataConfig | None = None,
) -> pd.DataFrame:
    """Build the causal market-feature matrix.

    Every column is shifted by one trading day, so row *t* is knowable at the
    open of day *t*.  Rows shorter than the longest look-back window
    (``cfg.warmup_days``) contain NaNs and are dropped by the caller via
    :func:`trim_warmup`.
    """
    cfg = cfg or config.DEFAULT.data
    returns = prices.pct_change()
    out: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        px = prices[ticker]
        ret = returns[ticker]

        # 1. Short-horizon reversal / drift: last week's cumulative return.
        out[f"{ticker}_ret5"] = px.pct_change(5)

        # 2. Intermediate momentum (~3 months).  The classic cross-sectional
        #    and time-series momentum horizon; this is the feature that should
        #    drive rotation between equities, bonds and gold.
        out[f"{ticker}_mom63"] = px.pct_change(cfg.momentum_window)

        # 3./4. Realised volatility at two horizons, annualised.  Two horizons
        #    rather than one so the agent can see whether risk is *rising* --
        #    a short window above a long window is a regime-change signal.
        out[f"{ticker}_vol20"] = ret.rolling(cfg.vol_window).std() * np.sqrt(
            config.TRADING_DAYS_PER_YEAR
        )
        out[f"{ticker}_vol60"] = ret.rolling(cfg.vol_window_long).std() * np.sqrt(
            config.TRADING_DAYS_PER_YEAR
        )

        # 5. Trend state: fast moving average relative to slow.  Expressed as a
        #    ratio minus one so that zero means "no trend" and the feature is
        #    scale-free across assets with very different price levels.
        ma_fast = px.rolling(cfg.ma_fast).mean()
        ma_slow = px.rolling(cfg.ma_slow).mean()
        out[f"{ticker}_ma_ratio"] = ma_fast / ma_slow - 1.0

    # Cross-asset structure.  The whole premise of a 60/40 portfolio is that
    # equities and bonds are negatively correlated; that relationship inverted
    # in 2022 and destroyed the diversification benefit.  The agent cannot
    # rotate away from a broken hedge unless it can see the hedge breaking.
    if "SPY" in prices.columns and "TLT" in prices.columns:
        out["spy_tlt_corr60"] = (
            returns["SPY"].rolling(cfg.vol_window_long).corr(returns["TLT"])
        )

    # Rate environment.  The level sets the opportunity cost of holding cash;
    # the change is what actually repriced bonds in 2022.
    rf = risk_free.reindex(prices.index).ffill()
    out["rf_level"] = rf * config.TRADING_DAYS_PER_YEAR  # back to annualised decimal
    out["rf_chg20"] = out["rf_level"].diff(20)

    features = pd.DataFrame(out, index=prices.index)

    # THE causality guarantee.  Everything above may touch row t; the shift
    # makes row t contain only information from t-1 and earlier.
    features = features.shift(1)
    features.index.name = "date"
    return features


def trim_warmup(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: config.DataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the leading rows whose long look-backs are not yet defined.

    Trimming happens *before* the train window starts, not by dropping NaNs
    inside it, so the training period itself is never silently shortened.
    """
    cfg = cfg or config.DEFAULT.data
    valid = features.dropna(how="any")
    if valid.empty:
        raise ValueError(
            "No fully-populated feature rows. The sample is shorter than the "
            f"longest look-back window ({cfg.warmup_days} trading days)."
        )
    first = valid.index[0]
    return features.loc[first:], prices.loc[first:]


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret5": "Trailing 1-week total return",
    "mom63": "Trailing 3-month total return (momentum)",
    "vol20": "Annualised realised volatility, 20-day window",
    "vol60": "Annualised realised volatility, 60-day window",
    "ma_ratio": "50-day / 200-day moving-average ratio, minus 1 (trend state)",
    "spy_tlt_corr60": "60-day rolling correlation between SPY and TLT returns",
    "rf_level": "Annualised 13-week T-bill rate",
    "rf_chg20": "20-day change in the annualised T-bill rate",
}


def describe_features(columns: pd.Index) -> pd.DataFrame:
    """Human-readable table of what each feature column means (for notebook 01)."""
    rows = []
    for col in columns:
        suffix = col.split("_", 1)[1] if "_" in col and col.split("_", 1)[0].isupper() else col
        rows.append(
            {
                "feature": col,
                "asset": col.split("_", 1)[0] if col.split("_", 1)[0].isupper() else "-",
                "description": FEATURE_DESCRIPTIONS.get(suffix, ""),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
@dataclass
class FeatureScaler:
    """Z-score scaler fitted on the training window only.

    Fitting on the full sample would leak the mean and variance of the test
    period into training -- a subtle but real form of look-ahead that is
    responsible for a large share of irreproducible results in the quantitative
    finance literature.

    Outputs are clipped to +/- ``clip`` standard deviations.  March 2020 produced
    volatility readings roughly eight training-sigmas from the mean; left
    unclipped, a single such row dominates the first layer's activations and
    destabilises early training.  Clipping preserves the ordering ("this is
    extreme") while bounding the magnitude.
    """

    clip: float = 5.0
    mean_: pd.Series | None = field(default=None, repr=False)
    std_: pd.Series | None = field(default=None, repr=False)
    columns_: list[str] = field(default_factory=list, repr=False)

    def fit(self, train_features: pd.DataFrame) -> "FeatureScaler":
        self.columns_ = list(train_features.columns)
        self.mean_ = train_features.mean()
        std = train_features.std()
        # Guard against a constant column producing a divide-by-zero.
        self.std_ = std.where(std > 1e-12, 1.0)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        scaled = (features[self.columns_] - self.mean_) / self.std_
        return scaled.clip(-self.clip, self.clip)

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_features).transform(train_features)

    def to_dict(self) -> dict:
        return {
            "clip": self.clip,
            "columns": self.columns_,
            "mean": {k: float(v) for k, v in self.mean_.items()},  # type: ignore[union-attr]
            "std": {k: float(v) for k, v in self.std_.items()},  # type: ignore[union-attr]
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureScaler":
        scaler = cls(clip=payload["clip"])
        scaler.columns_ = list(payload["columns"])
        scaler.mean_ = pd.Series(payload["mean"])
        scaler.std_ = pd.Series(payload["std"])
        return scaler


# --------------------------------------------------------------------------- #
# End-to-end dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Everything the environment needs, aligned on one date index.

    ``features`` are standardised; ``features_raw`` are kept for the exploratory
    plots in notebook 01, where the point is to show the economic quantity
    (a 32% annualised volatility) rather than a z-score.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    features: pd.DataFrame
    features_raw: pd.DataFrame
    risk_free: pd.Series
    scaler: FeatureScaler

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    @property
    def n_market_features(self) -> int:
        return self.features.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.n_market_features + N_PORTFOLIO_FEATURES

    def env_obs_dim(self, cfg: config.EnvConfig | None = None) -> int:
        """Observation width actually produced by :class:`PortfolioEnv`.

        ``obs_dim`` reports the base layout; the previous-action feature is
        optional and only present when the env config asks for it.
        """
        cfg = cfg or config.DEFAULT.env
        return self.obs_dim + int(cfg.include_prev_action)

    def split(self, name: str, cfg: config.DataConfig | None = None) -> "Dataset":
        """Return a new :class:`Dataset` restricted to one chronological split."""
        from .data import split_bounds

        start, end = split_bounds(cfg)[name]
        return Dataset(
            prices=self.prices.loc[start:end],
            returns=self.returns.loc[start:end],
            features=self.features.loc[start:end],
            features_raw=self.features_raw.loc[start:end],
            risk_free=self.risk_free.loc[start:end],
            scaler=self.scaler,
        )


def build_dataset(
    prices: pd.DataFrame,
    risk_free: pd.Series,
    cfg: config.DataConfig | None = None,
) -> Dataset:
    """Assemble the full modelling dataset from cached prices.

    Order of operations is deliberate and is the part most often got wrong:

    1. build causal features (shift by one day),
    2. trim the look-back warm-up **before** the train window opens,
    3. fit the scaler on the **training rows only**,
    4. apply that fitted scaler to every split.
    """
    cfg = cfg or config.DEFAULT.data

    features_raw = build_market_features(prices, risk_free, cfg)
    features_raw, prices = trim_warmup(features_raw, prices, cfg)
    features_raw = features_raw.dropna(how="any")
    prices = prices.loc[features_raw.index]

    returns = prices.pct_change().fillna(0.0)
    rf = risk_free.reindex(prices.index).ffill().fillna(0.0)

    train_rows = features_raw.loc[cfg.train_start : cfg.train_end]
    if train_rows.empty:
        raise ValueError("Training window contains no rows after warm-up trimming.")

    scaler = FeatureScaler().fit(train_rows)
    features = scaler.transform(features_raw)

    return Dataset(
        prices=prices,
        returns=returns,
        features=features,
        features_raw=features_raw,
        risk_free=rf,
        scaler=scaler,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
#: Notebooks 02-05 must not silently rebuild the dataset with a different
#: configuration than notebook 01 used.  Persisting the processed frames (and
#: the fitted scaler) makes the pipeline a true DAG: 01 produces, the rest
#: consume.
_PROCESSED_FILES = {
    "prices": "prices.csv",
    "returns": "returns.csv",
    "features": "features_scaled.csv",
    "features_raw": "features_raw.csv",
    "risk_free": "risk_free.csv",
}
_SCALER_FILE = "scaler.json"


def save_dataset(ds: Dataset, folder=None):
    """Write every component of ``ds`` to ``data/processed/``."""
    from .utils import save_json

    folder = folder or config.DATA_PROCESSED
    folder.mkdir(parents=True, exist_ok=True)
    ds.prices.to_csv(folder / _PROCESSED_FILES["prices"])
    ds.returns.to_csv(folder / _PROCESSED_FILES["returns"])
    ds.features.to_csv(folder / _PROCESSED_FILES["features"])
    ds.features_raw.to_csv(folder / _PROCESSED_FILES["features_raw"])
    ds.risk_free.to_frame("rf_daily").to_csv(folder / _PROCESSED_FILES["risk_free"])
    save_json(ds.scaler.to_dict(), folder / _SCALER_FILE)
    return folder


def load_dataset(folder=None) -> Dataset:
    """Rebuild a :class:`Dataset` from ``data/processed/`` without recomputing."""
    from .utils import load_json

    folder = folder or config.DATA_PROCESSED

    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(folder / _PROCESSED_FILES[name], index_col=0, parse_dates=True)

    return Dataset(
        prices=_read("prices"),
        returns=_read("returns"),
        features=_read("features"),
        features_raw=_read("features_raw"),
        risk_free=_read("risk_free").iloc[:, 0],
        scaler=FeatureScaler.from_dict(load_json(folder / _SCALER_FILE)),
    )


def processed_dataset_exists(folder=None) -> bool:
    folder = folder or config.DATA_PROCESSED
    return all((folder / f).exists() for f in _PROCESSED_FILES.values()) and (
        folder / _SCALER_FILE
    ).exists()

