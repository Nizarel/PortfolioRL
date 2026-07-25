"""Tests for the hyperparameter search module.

The search itself is expensive, so these tests exercise the *plumbing* -- grid
expansion, config materialisation, the Pareto helper, and that an Optuna study
with a stub objective runs, prunes and persists -- rather than running real
training.  Where training is needed it is a two-point grid at a few hundred
steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfoliorl import config, tuning


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def test_grid_expands_to_the_full_cartesian_product():
    grid = {"learning_rate": (1e-3, 3e-3), "gamma": (0.95, 0.99), "batch_size": (32,)}
    points = tuning.grid_points(grid)
    assert len(points) == 4
    assert all(set(p) == set(grid) for p in points)
    assert len({tuple(sorted(p.items())) for p in points}) == 4


def test_default_grid_is_small_enough_to_actually_run():
    points = tuning.grid_points()
    assert len(points) == 18  # 3 learning rates x 2 gammas x 3 network sizes
    assert all("learning_rate" in p and "hidden_sizes" in p for p in points)


def test_run_grid_returns_one_sorted_row_per_point(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path)

    table = tuning.run_grid(
        dataset.split("train"),
        dataset.split("valid"),
        grid={"learning_rate": (1e-3, 3e-3)},
        base=config.AgentConfig(hidden_sizes=(16, 8), learning_starts=64, batch_size=16),
        total_steps=300,
        eval_every=150,
        progress=None,
    )
    assert len(table) == 2
    assert {"learning_rate", "val_sharpe", "val_cagr", "seconds"} <= set(table.columns)
    # Sorted best-first, so a reader can take .iloc[0] without re-sorting.
    assert table["val_sharpe"].is_monotonic_decreasing
    assert table["seconds"].gt(0).all()


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def test_suggest_returns_values_inside_the_documented_ranges():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    for _ in range(15):
        trial = study.ask()
        agent_over, env_over = tuning.suggest(trial)
        assert 1e-4 <= agent_over["learning_rate"] <= 5e-3
        assert 0.90 <= agent_over["gamma"] <= 0.995
        assert agent_over["hidden_sizes"] in tuning.HIDDEN_CHOICES.values()
        assert agent_over["batch_size"] in (32, 64, 128)
        assert agent_over["target_update_interval"] in (250, 500, 1000)
        assert 0.1 <= agent_over["eps_decay_fraction"] <= 0.5
        assert 0.0 <= env_over["lambda_drawdown"] <= 1.0
        study.tell(trial, 0.0)


def test_every_search_dimension_is_documented():
    """A hyperparameter that is tuned but undocumented is an undisclosed degree
    of freedom, which is exactly what the reporting is meant to prevent."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    tuning.suggest(trial)
    assert set(trial.params) <= set(tuning.SEARCH_SPACE_DOC)
    assert set(trial.params) == set(tuning.SEARCH_SPACE_DOC)


# --------------------------------------------------------------------------- #
# Study handling
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_study():
    """A cheap study over the real search space with a synthetic objective."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        agent_over, env_over = tuning.suggest(trial)
        # Deterministic, unimodal in learning rate so the best trial is knowable.
        trial.set_user_attr("val_max_drawdown", 0.1 + agent_over["gamma"] / 10)
        trial.set_user_attr("val_cagr", 0.05)
        return -abs(np.log10(agent_over["learning_rate"]) + 3.0)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=12)
    return study


def test_study_to_frame_flattens_params_and_user_attributes(stub_study):
    df = tuning.study_to_frame(stub_study)
    assert len(df) == 12
    assert {"number", "state", "value"} <= set(df.columns)
    assert "param_learning_rate" in df.columns
    assert "val_max_drawdown" in df.columns


def test_best_configs_materialises_concrete_config_objects(stub_study):
    agent_cfg, env_cfg = tuning.best_configs(stub_study)
    assert isinstance(agent_cfg, config.AgentConfig)
    assert isinstance(env_cfg, config.EnvConfig)
    assert isinstance(agent_cfg.hidden_sizes, tuple)
    assert agent_cfg.learning_rate == stub_study.best_params["learning_rate"]
    assert env_cfg.lambda_drawdown == stub_study.best_params["lambda_drawdown"]
    # Untuned fields must be inherited unchanged from the base config.
    assert agent_cfg.total_steps == config.DEFAULT.agent.total_steps
    assert env_cfg.transaction_cost_bps == config.DEFAULT.env.transaction_cost_bps


def test_best_configs_accepts_overrides(stub_study):
    agent_cfg, _ = tuning.best_configs(stub_study, total_steps=1234, seed=9)
    assert agent_cfg.total_steps == 1234 and agent_cfg.seed == 9


def test_summary_records_how_many_configurations_were_tried(stub_study, tmp_path):
    """The trial count feeds the Deflated Sharpe Ratio in notebook 05; omitting
    it is how tuned backtests overstate their evidence."""
    import json

    path = tuning.save_study_summary(stub_study, tmp_path / "summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 12
    assert payload["best_value"] == pytest.approx(stub_study.best_value)
    assert set(payload["search_space"]) == set(tuning.SEARCH_SPACE_DOC)
    assert set(payload["best_params"]) <= set(tuning.SEARCH_SPACE_DOC)


def test_pruner_is_configured_with_a_warm_up(dataset, tmp_path, monkeypatch):
    """Pruning on the first evaluation would discard slow starters, so a warm-up
    is mandatory -- this pins that the study is built with one."""
    pytest.importorskip("optuna")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_RESULTS", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_FIGURES", tmp_path)

    study = tuning.run_optuna(
        dataset.split("train"),
        dataset.split("valid"),
        n_trials=1,
        study_name="unit",
        storage=f"sqlite:///{(tmp_path / 'unit.db').as_posix()}",
        total_steps=300,
        eval_every=150,
        n_startup_trials=5,
        n_warmup_steps=10_000,
        progress=None,
    )
    assert len(study.trials) == 1
    assert study.pruner._n_warmup_steps == 10_000
    assert study.direction.name == "MAXIMIZE"


# --------------------------------------------------------------------------- #
# Pareto helper
# --------------------------------------------------------------------------- #
def test_pareto_front_keeps_only_non_dominated_points():
    df = pd.DataFrame(
        {
            "val_max_drawdown": [0.10, 0.15, 0.20, 0.25, 0.12],
            "value": [0.50, 0.40, 0.90, 0.80, 0.30],
        }
    )
    front = tuning.pareto_front(df)
    # Sorted by drawdown: 0.10/0.50 -> 0.12/0.30 (dominated) -> 0.15/0.40
    # (dominated) -> 0.20/0.90 (kept) -> 0.25/0.80 (dominated).
    assert list(front["val_max_drawdown"]) == [0.10, 0.20]
    assert list(front["value"]) == [0.50, 0.90]
    assert front["value"].is_monotonic_increasing


def test_pareto_front_ignores_failed_trials():
    df = pd.DataFrame(
        {"val_max_drawdown": [0.1, np.nan, 0.2], "value": [0.5, 0.9, np.nan]}
    )
    front = tuning.pareto_front(df)
    assert len(front) == 1
    assert front["value"].iloc[0] == 0.5
