"""Tests for experiment orchestration.

Real experiment runs take tens of minutes, so these use tiny budgets and the
synthetic dataset.  What is being verified is the *scaffolding*: that seeds are
matched across variants, that the cache round-trips, that the walk-forward folds
cannot leak future data into training, and that the slicing helper does not
silently re-fit the feature scaler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfoliorl import config, experiments


@pytest.fixture(autouse=True)
def artifacts_in_tmp(tmp_path, monkeypatch):
    """Redirect every artefact path so tests never touch the real project output."""
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path / "models")
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path / "logs")
    monkeypatch.setattr(config, "ARTIFACTS_FIGURES", tmp_path / "figures")
    monkeypatch.setattr(experiments, "CURVE_DIR", tmp_path / "results" / "curves")
    for p in ("results", "models", "logs", "figures"):
        (tmp_path / p).mkdir(parents=True, exist_ok=True)
    return tmp_path


TINY = dict(total_steps=400, eval_every=200)
TWO_VARIANTS = {
    "Vanilla DQN": {"double": False, "dueling": False},
    "Double+Dueling DQN": {"double": True, "dueling": True},
}


# --------------------------------------------------------------------------- #
# Variant / seed sweep
# --------------------------------------------------------------------------- #
def test_variant_sweep_produces_one_row_per_variant_and_seed(dataset):
    res = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0, 1), tag="unit_ablation",
        progress=None, **TINY,
    )
    assert len(res.table) == 4
    assert set(res.table["variant"]) == set(TWO_VARIANTS)
    assert sorted(res.table["seed"].unique()) == [0, 1]
    assert len(res.curves) == 4


def test_both_validation_and_test_metrics_are_recorded(dataset):
    res = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0,), tag="unit_metrics",
        progress=None, **TINY,
    )
    cols = set(res.table.columns)
    for prefix in ("val", "test"):
        assert {f"{prefix}_sharpe", f"{prefix}_cagr", f"{prefix}_max_drawdown"} <= cols
    assert res.table["test_final_wealth"].gt(0).all()
    assert res.table["selected_step"].ge(0).all()


def test_matched_seeds_give_identical_initialisation_across_variants(dataset):
    """Variants must be compared on the same seeds so the test can be paired --
    otherwise half of the measured 'effect' is just different random draws."""
    res = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0, 1, 2), tag="unit_paired",
        progress=None, **TINY,
    )
    pivot = res.by_variant("test_sharpe")
    assert list(pivot.index) == [0, 1, 2]
    assert set(pivot.columns) == set(TWO_VARIANTS)
    assert pivot.notna().all().all()


def test_results_round_trip_through_the_cache(dataset):
    first = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0,), tag="unit_cache",
        progress=None, **TINY,
    )
    assert experiments.ExperimentResults.exists("unit_cache")

    # A second call with force=False must not retrain -- it should be near
    # instant and return byte-identical numbers.
    second = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0,), tag="unit_cache",
        progress=None, **TINY,
    )
    pd.testing.assert_frame_equal(
        first.table.reset_index(drop=True), second.table.reset_index(drop=True)
    )
    assert set(first.curves) == set(second.curves)
    for key in first.curves:
        assert len(first.curves[key]) == len(second.curves[key])


def test_seed_dispersion_reports_the_spread_not_just_the_mean(dataset):
    res = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0, 1, 2), tag="unit_spread",
        progress=None, **TINY,
    )
    disp = experiments.seed_dispersion(res)
    assert set(disp.columns) >= {"mean", "std", "min", "max", "range", "n"}
    assert (disp["n"] == 3).all()
    assert (disp["max"] >= disp["min"]).all()
    assert disp["mean"].is_monotonic_decreasing


def test_median_seed_curve_is_a_real_run_not_an_average(dataset):
    res = experiments.run_variant_seeds(
        dataset, variants=TWO_VARIANTS, seeds=(0, 1, 2), tag="unit_median",
        progress=None, **TINY,
    )
    curve = experiments.median_seed_curve(res, "Double+Dueling DQN")
    matches = [
        k for k, v in res.curves.items()
        if k.startswith("Double+Dueling DQN|") and np.allclose(
            v["wealth"].to_numpy(), curve["wealth"].to_numpy()
        )
    ]
    assert len(matches) == 1, "the plotted curve must be one of the actual runs"


# --------------------------------------------------------------------------- #
# Cost sweep
# --------------------------------------------------------------------------- #
def test_cost_sweep_covers_every_requested_level(dataset):
    res = experiments.cost_sweep(
        dataset, cost_bps=(0.0, 20.0), seeds=(0,), tag="unit_cost",
        progress=None, **TINY,
    )
    assert set(res.table["cost_bps"]) == {0.0, 20.0}
    assert res.table.loc[res.table["cost_bps"] == 0.0, "test_total_cost"].iloc[0] == 0.0
    assert res.table.loc[res.table["cost_bps"] == 20.0, "test_total_cost"].iloc[0] > 0.0


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def test_walk_forward_folds_never_train_on_or_after_the_test_year():
    folds = experiments.walk_forward_folds(
        None, first_test_year=2021, last_test_year=2025, valid_years=2
    )
    assert [f["fold"] for f in folds] == [2021, 2022, 2023, 2024, 2025]
    for f in folds:
        train_end = pd.Timestamp(f["train_end"])
        valid_start = pd.Timestamp(f["valid_start"])
        test_start = pd.Timestamp(f["test_start"])
        assert train_end < valid_start < test_start
        # The validation buffer must sit entirely before the test year.
        assert pd.Timestamp(f["valid_end"]) < test_start
        # And training must stop a full valid_years before the test year.
        assert train_end.year == f["fold"] - 3


def test_walk_forward_training_window_expands():
    folds = experiments.walk_forward_folds(None, first_test_year=2021, last_test_year=2024)
    ends = [pd.Timestamp(f["train_end"]) for f in folds]
    assert all(a < b for a, b in zip(ends, ends[1:]))


def test_slice_keeps_the_original_scaler(dataset):
    """Re-fitting the scaler per fold would leak that fold's distribution into
    its own features -- a subtle and very common backtest bug."""
    sliced = experiments._slice(dataset, "2010-01-01", "2012-12-31")
    assert sliced.scaler is dataset.scaler
    assert sliced.dates.min() >= pd.Timestamp("2010-01-01")
    assert sliced.dates.max() <= pd.Timestamp("2012-12-31")
    assert sliced.obs_dim == dataset.obs_dim
    assert len(sliced.prices) == len(sliced.features) == len(sliced.returns)


def test_walk_forward_runs_and_labels_folds_by_year(dataset):
    res = experiments.walk_forward(
        dataset, seeds=(0,), tag="unit_wf", first_test_year=2021, last_test_year=2022,
        progress=None, **TINY,
    )
    assert list(res.table["variant"]) == ["2021", "2022"]
    assert res.table["train_days"].is_monotonic_increasing
    assert res.table["test_days"].gt(0).all()
    assert np.isfinite(res.table["test_sharpe"]).all()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_manifest_records_size_and_timestamp(tmp_path):
    f = config.ARTIFACTS_RESULTS / "example.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    missing = config.ARTIFACTS_RESULTS / "not_there.csv"

    import json

    path = experiments.save_manifest([f, missing])
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["bytes"] == f.stat().st_size
    assert "modified" in entries[0]
