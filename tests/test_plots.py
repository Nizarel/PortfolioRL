"""Tests for the shared plotting helpers.

Figures are the primary output of this project, and a plotting helper that
quietly distorts an axis is worse than one that crashes -- the chart still
renders, still looks plausible, and misleads. These tests pin the behaviour
that is easy to get silently wrong.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfoliorl import plots


@pytest.fixture
def test_split_axes():
    """An axes containing only 2021-2025 data, as in the results notebook."""
    idx = pd.bdate_range("2021-01-01", "2025-12-31")
    fig, ax = plt.subplots()
    ax.plot(idx, np.linspace(1.0, 1.6, len(idx)))
    yield ax
    plt.close(fig)


def _xlim_as_timestamps(ax) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = ax.get_xlim()
    return (
        pd.Timestamp(mdates.num2date(lo)).tz_localize(None),
        pd.Timestamp(mdates.num2date(hi)).tz_localize(None),
    )


class TestAnnotateCrises:
    def test_does_not_stretch_the_axis_to_reach_an_earlier_crisis(self, test_split_axes):
        """The regression this was written for.

        ``axvspan`` participates in autoscaling, so shading the 2007-2009 GFC on
        a chart whose data starts in 2021 used to drag the left limit back to
        2007 and squash five years of curves into the right-hand quarter of the
        frame. The chart still rendered, which is exactly why it survived review.
        """
        before = test_split_axes.get_xlim()
        plots.annotate_crises(test_split_axes)
        assert test_split_axes.get_xlim() == before

    def test_skips_windows_that_lie_entirely_outside_the_plotted_range(self, test_split_axes):
        n_before = len(test_split_axes.patches)
        plots.annotate_crises(
            test_split_axes,
            windows=(("2008-09-01", "2009-03-31", "GFC"),),
        )
        assert len(test_split_axes.patches) == n_before

    def test_shades_a_window_that_falls_inside_the_range(self, test_split_axes):
        n_before = len(test_split_axes.patches)
        plots.annotate_crises(
            test_split_axes,
            windows=(("2022-01-01", "2022-10-31", "2022 selloff"),),
        )
        assert len(test_split_axes.patches) == n_before + 1

    def test_clips_a_window_that_straddles_the_edge(self, test_split_axes):
        """A window starting before the data should be cut at the axis limit."""
        lo, _ = _xlim_as_timestamps(test_split_axes)
        plots.annotate_crises(
            test_split_axes,
            windows=(("2020-02-01", "2021-06-30", "COVID"),),
            label=False,
        )
        span = test_split_axes.patches[-1]
        left = pd.Timestamp(mdates.num2date(span.get_x())).tz_localize(None)
        assert left >= lo - pd.Timedelta(days=1)

    def test_drops_the_label_for_a_sliver_too_narrow_to_hold_text(self, test_split_axes):
        """Clipping can leave a window too thin for its name to fit.

        The axis lower limit sits before the first data point because matplotlib
        adds a margin, so the window is built relative to the realised limit
        rather than to a hard-coded date.
        """
        lo, _ = _xlim_as_timestamps(test_split_axes)
        n_texts = len(test_split_axes.texts)
        plots.annotate_crises(
            test_split_axes,
            windows=((
                (lo - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
                (lo + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                "barely visible",
            ),),
        )
        assert len(test_split_axes.patches) > 0
        assert len(test_split_axes.texts) == n_texts

    def test_labels_a_window_wide_enough_to_read(self, test_split_axes):
        n_texts = len(test_split_axes.texts)
        plots.annotate_crises(
            test_split_axes,
            windows=(("2022-01-01", "2022-10-31", "2022 selloff"),),
        )
        assert len(test_split_axes.texts) == n_texts + 1

    def test_empty_axes_still_shades_every_window(self):
        """With nothing plotted there is no range to clip against."""
        fig, ax = plt.subplots()
        plots.annotate_crises(ax)
        assert len(ax.patches) == len(plots.CRISIS_WINDOWS)
        plt.close(fig)

    def test_full_history_axes_keeps_all_windows(self):
        idx = pd.bdate_range("2004-11-18", "2025-12-31")
        fig, ax = plt.subplots()
        ax.plot(idx, np.linspace(1.0, 5.0, len(idx)))
        plots.annotate_crises(ax)
        assert len(ax.patches) == len(plots.CRISIS_WINDOWS)
        plt.close(fig)


class TestSaveFig:
    def test_writes_a_png_and_returns_its_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plots.config, "ARTIFACTS_FIGURES", tmp_path)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = plots.save_fig(fig, "unit_test_figure")
        plt.close(fig)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_subdir_is_created_on_demand(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plots.config, "ARTIFACTS_FIGURES", tmp_path)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = plots.save_fig(fig, "nested", subdir="appendix")
        plt.close(fig)
        assert path.parent.name == "appendix"
        assert path.exists()


class TestStrategyColour:
    def test_known_strategy_gets_its_registered_colour(self):
        name = next(iter(plots.STRATEGY_COLORS))
        assert plots.strategy_color(name) == plots.STRATEGY_COLORS[name]

    def test_unknown_strategy_falls_back_rather_than_raising(self):
        """New benchmarks get added faster than the colour registry does."""
        assert plots.strategy_color("Strategy That Does Not Exist") == "#555555"


class TestStyleScorecard:
    def test_a_format_mapping_may_name_columns_that_are_absent(self):
        """Notebooks pass ``metrics.SCORECARD_FORMATS`` against trimmed tables."""
        frame = pd.DataFrame({"Sharpe": [1.2, 0.8], "CAGR": [0.10, 0.05]},
                             index=["RL", "60/40"])
        styler = plots.style_scorecard(
            frame,
            higher_is_better=("Sharpe", "CAGR"),
            fmt={"Sharpe": "{:.2f}", "CAGR": "{:.1%}", "Missing column": "{:.3f}"},
        )
        assert styler.to_html()

    def test_highlighting_does_not_mutate_the_input_frame(self):
        frame = pd.DataFrame({"Sharpe": [1.2, 0.8]}, index=["RL", "60/40"])
        before = frame.copy()
        plots.style_scorecard(frame, higher_is_better=("Sharpe",))
        pd.testing.assert_frame_equal(frame, before)
