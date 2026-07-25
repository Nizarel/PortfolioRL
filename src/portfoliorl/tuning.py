"""Hyperparameter search: a coarse grid first, then Bayesian optimisation.

Why two stages
--------------
The two stages answer different questions and doing only one of them is a
common mistake.

A **coarse grid** answers *"is the response surface smooth, and roughly where is
the good region?"*  Its value is diagnostic: a grid whose cells are all within
noise of each other tells you the hyperparameter does not matter, which is worth
knowing before spending a hundred trials refining it.  A grid is also trivially
parallel, exhaustive within its own axes, and impossible to get subtly wrong.

**Optuna's TPE sampler** then answers *"what is the best configuration?"*  Tree-
structured Parzen Estimation (Bergstra et al., 2011) models
:math:`p(\\text{hyperparameters} \\mid \\text{good result})` and
:math:`p(\\text{hyperparameters} \\mid \\text{bad result})` separately and samples
where their ratio is largest.  Grid search wastes evaluations because most
hyperparameters do not matter and a grid spends the same effort on all of them;
random search fixes that but ignores everything it has already learned.

Pruning
-------
``MedianPruner`` stops a trial whose intermediate validation Sharpe is below the
median of previous trials at the same step.  With ~5 minutes per full run, this
is the difference between a search that fits in a coffee break and one that does
not.  A warm-up of several evaluations is enforced first, because early
validation Sharpe is dominated by exploration noise and pruning on it would
discard configurations that were merely slow to start.

What is being optimised
-----------------------
**Validation Sharpe**, on 2018-2020, exactly as in ``train.py``.  The test split
is not touched by anything in this module.  This matters more here than
anywhere else in the project: hyperparameter search is a *multiple-comparisons
machine*, and running it against test data would produce a configuration
selected for its luck on the very sample used to report results.  The number of
configurations tried is recorded so that notebook 05 can deflate the final
Sharpe ratio accordingly (Bailey & Lopez de Prado, 2014).

References
----------
Bergstra, Bardenet, Bengio & Kegl (2011), "Algorithms for Hyper-Parameter
Optimization", *NeurIPS*.  Akiba et al. (2019), "Optuna: A Next-generation
Hyperparameter Optimization Framework", *KDD*.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config, train
from .agent import variant_config
from .features import Dataset


# --------------------------------------------------------------------------- #
# Stage 1: coarse grid
# --------------------------------------------------------------------------- #
COARSE_GRID: dict[str, Sequence[Any]] = {
    "learning_rate": (3e-4, 1e-3, 3e-3),
    "gamma": (0.95, 0.99),
    "hidden_sizes": ((64, 32), (128, 64), (256, 128)),
}


def grid_points(grid: dict[str, Sequence[Any]] | None = None) -> list[dict[str, Any]]:
    """Expand a grid specification into the list of configurations to run."""
    grid = grid or COARSE_GRID
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def run_grid(
    train_ds: Dataset,
    valid_ds: Dataset,
    grid: dict[str, Sequence[Any]] | None = None,
    *,
    base: config.AgentConfig | None = None,
    total_steps: int = 40_000,
    eval_every: int = 5_000,
    seed: int = 0,
    env_cfg: config.EnvConfig | None = None,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Train one short run per grid point; return a tidy results table.

    ``total_steps`` is deliberately shorter than the final training budget.  The
    grid is a *map* of the response surface, not the final fit, and a coarse map
    at a third of the cost is a better use of the compute than a precise map of
    a region we are about to refine anyway.
    """
    base = base or config.DEFAULT.agent
    points = grid_points(grid)
    rows: list[dict[str, Any]] = []

    for i, point in enumerate(points, 1):
        cfg = replace(base, total_steps=total_steps, eval_every=eval_every,
                      seed=seed, **point)
        t0 = time.perf_counter()
        result = train.train_dqn(
            train_ds, valid_ds,
            agent_cfg=cfg, env_cfg=env_cfg,
            run_name=f"grid{i:02d}",
            save_checkpoints=False, write_log=False, progress=None,
        )
        row: dict[str, Any] = dict(point)
        row["hidden_sizes"] = str(point.get("hidden_sizes", base.hidden_sizes))
        row.update(
            val_sharpe=result.best_val_sharpe,
            val_cagr=result.best.get("val_cagr", np.nan),
            val_max_drawdown=result.best.get("val_max_drawdown", np.nan),
            val_action_entropy=result.best.get("val_action_entropy", np.nan),
            best_step=result.best.get("step", 0),
            seconds=round(time.perf_counter() - t0, 1),
        )
        rows.append(row)
        if progress is not None:
            desc = ", ".join(f"{k}={row[k]}" for k in point)
            progress(f"  [{i:>2}/{len(points)}] {desc:<52} "
                     f"val Sharpe {row['val_sharpe']:+.3f}  ({row['seconds']:.0f}s)")

    return pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 2: Optuna
# --------------------------------------------------------------------------- #
SEARCH_SPACE_DOC = {
    "learning_rate": "log-uniform 1e-4 to 5e-3 -- the single most influential knob",
    "gamma": "0.90 to 0.995 -- effective horizon 10 to 200 weeks",
    "hidden_sizes": "3 discrete widths -- capacity, bounded by 620 training decisions",
    "batch_size": "32/64/128 -- gradient-noise scale",
    "target_update_interval": "250/500/1000 -- how stale the bootstrap target is",
    "eps_decay_fraction": "0.1 to 0.5 -- how long exploration lasts",
    "lambda_drawdown": "0.0 to 1.0 -- how strongly new drawdowns are punished",
}

HIDDEN_CHOICES = {"64,32": (64, 32), "128,64": (128, 64), "256,128": (256, 128)}


def suggest(trial) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw one configuration from the search space.

    Returns ``(agent_overrides, env_overrides)``.  ``lambda_drawdown`` is an
    *environment* parameter, not an agent one: it changes the objective the
    agent is optimising, so tuning it is tuning the problem definition.  That is
    defensible only because it is tuned on validation data and reported openly
    -- it is included precisely because a reviewer would otherwise ask whether
    the penalty weights were quietly chosen to flatter the result.
    """
    agent_overrides = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.995),
        "hidden_sizes": HIDDEN_CHOICES[
            trial.suggest_categorical("hidden_sizes", list(HIDDEN_CHOICES))
        ],
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [250, 500, 1000]
        ),
        "eps_decay_fraction": trial.suggest_float("eps_decay_fraction", 0.1, 0.5),
    }
    env_overrides = {
        "lambda_drawdown": trial.suggest_float("lambda_drawdown", 0.0, 1.0),
    }
    return agent_overrides, env_overrides


def make_objective(
    train_ds: Dataset,
    valid_ds: Dataset,
    *,
    base_agent: config.AgentConfig | None = None,
    base_env: config.EnvConfig | None = None,
    total_steps: int = 40_000,
    eval_every: int = 5_000,
    seed: int = 0,
) -> Callable[[Any], float]:
    """Build the Optuna objective: maximise validation Sharpe."""
    import optuna

    base_agent = base_agent or config.DEFAULT.agent
    base_env = base_env or config.DEFAULT.env

    def objective(trial) -> float:
        agent_overrides, env_overrides = suggest(trial)
        agent_cfg = replace(base_agent, total_steps=total_steps,
                            eval_every=eval_every, seed=seed, **agent_overrides)
        env_cfg = replace(base_env, **env_overrides)

        def report(step: int, sharpe: float) -> None:
            trial.report(sharpe, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = train.train_dqn(
            train_ds, valid_ds,
            agent_cfg=agent_cfg, env_cfg=env_cfg,
            run_name=f"optuna_t{trial.number:03d}",
            save_checkpoints=False, write_log=False, progress=None,
            pruning_callback=report,
        )
        # Stash the secondary metrics so notebook 04 can plot the risk-return
        # trade-off across trials, not just the scalar objective.
        for key in ("val_cagr", "val_max_drawdown", "val_action_entropy",
                    "val_mean_turnover", "step"):
            trial.set_user_attr(key, result.best.get(key, float("nan")))
        trial.set_user_attr("seconds", round(result.wall_time, 1))
        return result.best_val_sharpe

    return objective


def run_optuna(
    train_ds: Dataset,
    valid_ds: Dataset,
    *,
    n_trials: int = 40,
    study_name: str = "portfoliorl",
    storage: str | None = None,
    seed: int = 0,
    total_steps: int = 40_000,
    eval_every: int = 5_000,
    n_startup_trials: int = 10,
    n_warmup_steps: int = 15_000,
    progress: Callable[[str], None] | None = print,
):
    """Run a TPE study with median pruning and return the :class:`optuna.Study`.

    The study is persisted to SQLite under ``artifacts/`` so a search can be
    resumed, and so notebook 04 can be re-run for plotting without repeating the
    optimisation.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config.ensure_dirs()
    if storage is None:
        storage = f"sqlite:///{(config.ARTIFACTS / 'optuna.db').as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        # Seeded sampler: TPE is stochastic, and an unseeded search is not a
        # reproducible experiment.
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        ),
    )

    objective = make_objective(
        train_ds, valid_ds, total_steps=total_steps, eval_every=eval_every, seed=seed
    )

    def _callback(study_, trial) -> None:
        if progress is None:
            return
        state = trial.state.name.lower()
        value = "pruned" if trial.value is None else f"{trial.value:+.3f}"
        progress(
            f"  trial {trial.number:>3} {state:<9} Sharpe {value:>7}   "
            f"best so far {study_.best_value:+.3f}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[_callback],
                   gc_after_trial=True)
    return study


def study_to_frame(study) -> pd.DataFrame:
    """Flatten a study into a tidy DataFrame (params + user attributes)."""
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": t.state.name, "value": t.value}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update(t.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def best_configs(
    study,
    base_agent: config.AgentConfig | None = None,
    base_env: config.EnvConfig | None = None,
    **agent_overrides,
) -> tuple[config.AgentConfig, config.EnvConfig]:
    """Materialise the study's best trial as concrete config objects."""
    base_agent = base_agent or config.DEFAULT.agent
    base_env = base_env or config.DEFAULT.env
    p = dict(study.best_params)

    env_cfg = replace(base_env, lambda_drawdown=p.pop("lambda_drawdown", base_env.lambda_drawdown))
    if "hidden_sizes" in p:
        p["hidden_sizes"] = HIDDEN_CHOICES[p["hidden_sizes"]]
    agent_cfg = replace(base_agent, **p, **agent_overrides)
    return agent_cfg, env_cfg


def save_study_summary(study, path: str | Path | None = None) -> Path:
    """Write the search's provenance: best trial, trial count, search space.

    The **number of configurations evaluated** is recorded because it is an
    input to the Deflated Sharpe Ratio in notebook 05.  Reporting a tuned
    strategy's Sharpe without also reporting how many were tried is the most
    common way backtests overstate their evidence.
    """
    path = Path(path or config.ARTIFACTS_RESULTS / "04_optuna_summary.json")
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    payload = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "trial_states": states,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "search_space": SEARCH_SPACE_DOC,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def pareto_front(
    df: pd.DataFrame, x: str = "val_max_drawdown", y: str = "value"
) -> pd.DataFrame:
    """Non-dominated trials: maximal ``y`` for minimal ``x``.

    Used in notebook 04 to show that the highest-Sharpe configuration is not
    automatically the one an investor would choose -- several trials give up a
    little Sharpe for a materially shallower drawdown.
    """
    sub = df.dropna(subset=[x, y]).sort_values(x)
    keep, best = [], -np.inf
    for _, row in sub.iterrows():
        if row[y] > best:
            keep.append(row)
            best = row[y]
    return pd.DataFrame(keep)
