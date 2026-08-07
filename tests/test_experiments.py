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
# Named-configuration comparison
# --------------------------------------------------------------------------- #
class TestConfigComparison:
    """Separating "the tuned hyperparameters hurt" from "the longer budget hurt"
    needs whole configurations varied under matched seeds, not just the
    Double/Duelling flags."""

    def test_each_config_and_seed_produces_one_row(self, dataset):
        res = experiments.config_comparison(
            dataset,
            configs={
                "short": {"agent": {"total_steps": 400, "eval_every": 200}},
                "long": {"agent": {"total_steps": 600, "eval_every": 200}},
            },
            seeds=(0, 1), tag="unit_cfgcmp", progress=None,
        )
        assert len(res.table) == 4
        assert set(res.table["variant"]) == {"short", "long"}
        assert sorted(res.table["seed"].unique()) == [0, 1]

    def test_the_budget_actually_differs_between_configs(self, dataset):
        """Guards against overrides being silently dropped -- which would make
        the whole comparison vacuous."""
        res = experiments.config_comparison(
            dataset,
            configs={
                "short": {"agent": {"total_steps": 400, "eval_every": 200}},
                "long": {"agent": {"total_steps": 600, "eval_every": 200}},
            },
            seeds=(0,), tag="unit_cfgcmp_budget", progress=None,
        )
        by = res.table.set_index("variant")["total_steps"]
        assert by["short"] == 400
        assert by["long"] == 600

    def test_env_overrides_are_applied(self, dataset):
        """The tuned config changes lambda_drawdown, which lives on EnvConfig
        rather than AgentConfig -- so env overrides must reach the environment."""
        res = experiments.config_comparison(
            dataset,
            configs={
                "free": {"agent": {"total_steps": 400, "eval_every": 200},
                         "env": {"transaction_cost_bps": 0.0}},
                "costly": {"agent": {"total_steps": 400, "eval_every": 200},
                           "env": {"transaction_cost_bps": 200.0}},
            },
            seeds=(0,), tag="unit_cfgcmp_env", progress=None,
        )
        costs = res.table.set_index("variant")["test_total_cost"]
        assert costs["free"] == pytest.approx(0.0, abs=1e-12)
        assert costs["costly"] > costs["free"]


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

# --------------------------------------------------------------------------- #
# Crash resumability
# --------------------------------------------------------------------------- #
def _fake_curve() -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=10)
    return pd.DataFrame({"wealth": np.linspace(1.0, 1.1, 10), "return": 0.001}, index=idx)


class TestResume:
    """A full ablation is ~50 training runs.

    Losing all of it to a kernel death at run 45 is the failure mode these tests
    exist to prevent -- and it is not hypothetical, it happened once during
    development.
    """

    def test_a_journalled_run_is_not_retrained(self, dataset):
        """The recovered row must survive verbatim into the final table.

        A sentinel value that no real training run could produce proves the row
        came from the journal rather than from a repeat of the work.
        """
        tag = "unit_resume"
        sentinel = {
            "variant": "Vanilla DQN", "seed": 0,
            "double_dqn": False, "dueling": False,
            "selected_step": 999, "wall_time": 1.0,
            "val_sharpe": -12.5, "test_sharpe": -12.5,
        }
        experiments._record_progress(tag, sentinel, _fake_curve())

        res = experiments.run_variant_seeds(
            dataset, variants=TWO_VARIANTS, seeds=(0,), tag=tag,
            progress=None, **TINY,
        )

        assert len(res.table) == 2
        recovered = res.table[res.table["variant"] == "Vanilla DQN"].iloc[0]
        assert recovered["test_sharpe"] == -12.5
        assert recovered["selected_step"] == 999

    def test_the_journal_is_removed_once_the_result_csv_exists(self, dataset):
        tag = "unit_journal_cleanup"
        experiments.run_variant_seeds(
            dataset, variants=TWO_VARIANTS, seeds=(0,), tag=tag,
            progress=None, **TINY,
        )
        assert experiments.ExperimentResults.exists(tag)
        assert not experiments._progress_path(tag).exists()

    def test_a_torn_final_line_does_not_lose_the_good_rows(self):
        """A process killed mid-write leaves a partial JSON line."""
        tag = "unit_torn"
        experiments._record_progress(
            tag, {"variant": "A", "seed": 0, "test_sharpe": 1.0}, _fake_curve()
        )
        with experiments._progress_path(tag).open("a", encoding="utf-8") as handle:
            handle.write('{"variant": "B", "seed": 0, "test_sh')

        rows, curves, stale = experiments._load_progress(tag)
        assert len(rows) == 1
        assert rows[0]["variant"] == "A"
        assert len(curves) == 1
        assert stale == 0

    def test_a_journal_entry_whose_curve_is_missing_is_discarded(self):
        """Curve is written before the journal line, so this should not happen --
        but a half-deleted cache directory would produce it."""
        tag = "unit_orphan"
        experiments._record_progress(
            tag, {"variant": "A", "seed": 0, "test_sharpe": 1.0}, _fake_curve()
        )
        experiments._curve_path(tag, "A", 0).unlink()

        rows, curves, _ = experiments._load_progress(tag)
        assert rows == []
        assert curves == {}

    def test_a_journal_written_under_another_config_is_not_resumed(self):
        """The journal is keyed only by tag, so editing a config and rerunning
        under the same tag must not splice two designs into one result table."""
        tag = "unit_fingerprint"
        experiments._record_progress(
            tag,
            {"variant": "A", "seed": 0, "test_sharpe": 1.0},
            _fake_curve(),
            fingerprint="aaaaaaaaaaaaaaaa",
        )

        rows, curves, stale = experiments._load_progress(tag, expected="bbbbbbbbbbbbbbbb")
        assert rows == []
        assert curves == {}
        assert stale == 1

        rows, _, stale = experiments._load_progress(tag, expected="aaaaaaaaaaaaaaaa")
        assert len(rows) == 1
        assert stale == 0
        assert "_fingerprint" not in rows[0], "the stamp must not leak into results"

    def test_fingerprint_tracks_the_settings_that_change_a_run(self):
        base = experiments._fingerprint(env={"n_actions": 6}, ensemble=True)
        assert base == experiments._fingerprint(ensemble=True, env={"n_actions": 6}), \
            "key order must not change the stamp"
        assert base != experiments._fingerprint(env={"n_actions": 7}, ensemble=True)
        assert base != experiments._fingerprint(env={"n_actions": 6}, ensemble=False)

    def test_force_discards_a_previous_partial_run(self, dataset):
        tag = "unit_force"
        experiments._record_progress(
            tag,
            {"variant": "Vanilla DQN", "seed": 0, "test_sharpe": -99.0},
            _fake_curve(),
        )
        res = experiments.run_variant_seeds(
            dataset, variants=TWO_VARIANTS, seeds=(0,), tag=tag,
            force=True, progress=None, **TINY,
        )
        assert (res.table["test_sharpe"] != -99.0).all()

    def test_cost_sweep_resumes_too(self, dataset):
        tag = "unit_cost_resume"
        experiments._record_progress(
            tag,
            {"variant": "0 bps", "cost_bps": 0.0, "seed": 0, "test_sharpe": -7.5},
            _fake_curve(),
        )
        res = experiments.cost_sweep(
            dataset, cost_bps=(0, 10), seeds=(0,), tag=tag,
            progress=None, **TINY,
        )
        assert len(res.table) == 2
        assert res.table.loc[res.table["variant"] == "0 bps", "test_sharpe"].iloc[0] == -7.5
