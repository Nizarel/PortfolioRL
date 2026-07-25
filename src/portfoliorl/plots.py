"""Shared plotting style and figure-export helpers.

Every figure in the project is produced through this module so that the
notebooks, the written report and the video slides all share one visual
language, and so that every figure exists on disk as a PNG that can be embedded
elsewhere without being redrawn (and therefore without any risk of the report
showing a different run than the notebook).

Usage pattern in a notebook::

    from portfoliorl import plots
    plots.apply_style()

    fig, ax = plt.subplots()
    ...
    plots.save_fig(fig, "01_asset_growth")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import config

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
#: One fixed colour per asset, used in *every* figure.  Readers learn the
#: mapping once (SPY = blue equities, TLT = red bonds, GLD = gold, SHY = grey
#: cash) and never have to re-read a legend.
ASSET_COLORS: dict[str, str] = {
    "SPY": "#1f77b4",
    "TLT": "#d62728",
    "GLD": "#e6a817",
    "SHY": "#7f7f7f",
}

#: Background tints for the chronological splits.  Deliberately pale so that
#: they sit behind the data without competing with it.
SPLIT_COLORS: dict[str, str] = {
    "train": "#4c72b0",
    "valid": "#dd8452",
    "test": "#55a868",
}

#: Strategy colours.  The RL agent is always black so it stands out against the
#: benchmark family regardless of how many benchmarks are on the chart.
#: Keys match the names used in `benchmarks.STATIC_BENCHMARKS`,
#: `benchmarks.HOLD_BENCHMARKS` and the ablation labels, so any results mapping
#: can be coloured by lookup without a translation table.
STRATEGY_COLORS: dict[str, str] = {
    # Learned policies -- always black or near-black.
    "RL (DQN)": "#000000",
    "Double+Dueling DQN": "#000000",
    "Double DQN": "#3b3b6d",
    "Dueling DQN": "#6d3b5c",
    "Vanilla DQN": "#8c8c8c",
    # Rule-based benchmarks.
    "60/40 rebalanced": "#1f77b4",
    "60/40 buy & hold": "#7fb3d5",
    "Equal weight": "#2ca02c",
    "Equity-heavy 80/20": "#9467bd",
    "100% SPY": "#8c564b",
    "All cash (SHY)": "#7f7f7f",
    "Volatility target": "#17becf",
    "Trend following": "#dd8452",
    "Random": "#c44e52",
}


def strategy_color(name: str, fallback: str = "#555555") -> str:
    """Colour for a strategy, falling back gracefully for ad-hoc labels."""
    return STRATEGY_COLORS.get(name, fallback)

#: Highlight windows used to annotate market regimes on time-series charts.
#: These are the three stress episodes the agent is expected to react to.
CRISIS_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2007-10-09", "2009-03-09", "GFC"),
    ("2020-02-19", "2020-03-23", "COVID"),
    ("2022-01-03", "2022-10-12", "2022 rate shock"),
)


# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    """Install the project-wide Matplotlib defaults.

    Called once at the top of every notebook.  Keeping this in code rather than
    in a ``.mplstyle`` file means the settings are visible and explainable in
    the notebook narrative.
    """
    mpl.rcParams.update(
        {
            # Figure geometry: wide enough for a time series to be legible on a
            # slide, and exported at the same dpi it is displayed at.
            "figure.figsize": (11, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": config.FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # Typography large enough to survive being shrunk into a report.
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Light grid, no top/right spines: keeps the ink-to-data ratio high.
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "date.autoformatter.year": "%Y",
        }
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def save_fig(fig: Figure, name: str, subdir: str | None = None) -> Path:
    """Write ``fig`` to ``artifacts/figures/<name>.png`` and return the path.

    Parameters
    ----------
    fig
        The figure to export.
    name
        File stem *without* extension.  Use the ``NN_short_description``
        convention (notebook number first) so the figures directory sorts into
        the order the report presents them.
    subdir
        Optional sub-folder, used for bulk diagnostic figures (e.g. per-seed
        training curves) that should not clutter the main figure list.
    """
    out_dir = config.ARTIFACTS_FIGURES if subdir is None else config.ARTIFACTS_FIGURES / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- #
# Axis decorations
# --------------------------------------------------------------------------- #
def shade_splits(
    ax: plt.Axes,
    cfg: config.DataConfig | None = None,
    label: bool = True,
    alpha: float = 0.07,
    label_y: float = 0.015,
) -> None:
    """Tint the train / validation / test date ranges behind a time series.

    Present on nearly every time-series chart in the project: it makes the
    chronological, non-shuffled split protocol visually obvious and removes any
    doubt about which data the agent was fitted on.

    Labels sit at the *bottom* of the axes by default because the legend on a
    cumulative-growth chart almost always lives in the upper left.
    """
    cfg = cfg or config.DEFAULT.data
    spans = (
        ("train", cfg.train_start, cfg.train_end),
        ("valid", cfg.valid_start, cfg.valid_end),
        ("test", cfg.test_start, cfg.test_end),
    )
    for split, start, end in spans:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=SPLIT_COLORS[split],
            alpha=alpha,
            zorder=0,
            lw=0,
        )
        if label:
            # Axes-relative y so the label stays put regardless of the data range.
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(
                mid,
                label_y,
                split.upper(),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=SPLIT_COLORS[split],
                fontweight="bold",
                alpha=0.9,
            )


def annotate_crises(
    ax: plt.Axes,
    windows: Iterable[tuple[str, str, str]] | None = None,
    alpha: float = 0.12,
    label: bool = True,
    label_y: float = 0.11,
) -> None:
    """Shade the GFC / COVID / 2022 stress windows and name them.

    Used wherever the point of the chart is "look how the assets behave when
    conditions change" -- which is the entire premise of a regime-switching
    allocation policy.
    """
    for start, end, name in windows or CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="firebrick", alpha=alpha, lw=0)
        if label:
            ax.text(
                pd.Timestamp(start),
                label_y,
                f" {name}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=8,
                color="firebrick",
            )


def format_pct_axis(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    """Render a fractional axis (0.05) as a percentage (5%)."""
    fmt = PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def format_money_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Render a currency axis compactly ($100k, $1.2M)."""

    def _fmt(v: float, _pos: int) -> str:
        if abs(v) >= 1e6:
            return f"${v / 1e6:,.1f}M"
        if abs(v) >= 1e3:
            return f"${v / 1e3:,.0f}k"
        return f"${v:,.0f}"

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_fmt))


def tidy_dates(ax: plt.Axes, interval: int = 2) -> None:
    """Use year ticks on a long daily time series instead of dense date labels."""
    ax.xaxis.set_major_locator(mdates.YearLocator(interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")


def caption(fig: Figure, text: str) -> None:
    """Attach a small source/method note under a figure.

    Every exported figure carries its own provenance so that a reader who sees
    only the PNG (in the report or a slide) still knows what produced it.
    """
    fig.text(0.0, -0.04, text, fontsize=8, color="0.35", ha="left", va="top", wrap=True)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def style_scorecard(
    df: pd.DataFrame,
    higher_is_better: Sequence[str] = (),
    lower_is_better: Sequence[str] = (),
    fmt: str | Mapping[str, str] = "{:,.3f}",
    caption_text: str | None = None,
):
    """Return a colour-graded :class:`pandas.io.formats.style.Styler`.

    Performance tables are the primary evidence for the results section, so they
    get the same care as the charts: green means good, red means bad, and the
    direction is stated explicitly per column rather than assumed (max drawdown
    and volatility are *lower is better*, Sharpe and CAGR are *higher*).

    ``fmt`` may be a single format string or a column-to-format mapping such as
    :data:`portfoliorl.metrics.SCORECARD_FORMATS`; the mapping is filtered to the
    columns actually present so a subset of the scorecard can be shown.
    """
    if isinstance(fmt, Mapping):
        fmt = {k: v for k, v in fmt.items() if k in df.columns}

    styler = df.style.format(fmt, na_rep="--")
    present = set(df.columns)
    higher = [c for c in higher_is_better if c in present]
    lower = [c for c in lower_is_better if c in present]
    if higher:
        styler = styler.background_gradient(cmap="RdYlGn", subset=higher, axis=0)
    if lower:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=lower, axis=0)
    if caption_text:
        styler = styler.set_caption(caption_text)
    return styler
