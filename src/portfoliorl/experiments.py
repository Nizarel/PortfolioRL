"""Experiment orchestration: the ablation, the seed sweep, and the robustness checks.

Everything expensive lives here rather than in a notebook, for three reasons:
it can be unit-tested, it can be cached, and a notebook that re-runs in ten
seconds from cache is a notebook people actually re-run.

The four experiments
--------------------
**1. Variant ablation across seeds.**  Four architectures (vanilla, Double,
duelling, both) times several seeds.  A single-seed ablation is not an
ablation: DQN's seed-to-seed spread on this problem is comparable to the effect
being measured, so a one-seed comparison mostly reports which run got lucky.
Seeds are matched across variants -- variant A seed 3 and variant B seed 3 face
the same initialisation and the same episode draws -- which makes the comparison
paired and materially more powerful.

**2. Transaction-cost sensitivity.**  The agent is *retrained* at each cost
level, not merely re-evaluated.  Re-evaluating a single policy under higher
costs answers "what if this strategy paid more?", which is not the interesting
question.  Retraining answers "does the approach survive higher costs?", because
the agent can respond by trading less.

**3. Walk-forward validation.**  A single train/test split is one draw.
Walk-forward retrains on an expanding window and tests on the following year,
which is how the strategy would actually have been run, and it exposes
performance that depends entirely on one lucky period.

**4. Selection integrity.**  Every experiment selects its checkpoint on the
validation split.  The test split is evaluated once per run, at the end, and is
never used to choose anything.

Caching
-------
Results are written to ``artifacts/results/`` as CSV plus a per-run daily curve.
Re-running a notebook loads the cache unless ``force=True``.  The cache key
includes the variant, the seed and the run tag, so adding a seed does not
invalidate the existing ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config, metrics, train
from .agent import VARIANTS, ensemble_q_values, hysteresis_policy, variant_config
from .features import Dataset


CURVE_DIR = config.ARTIFACTS_RESULTS / "curves"


def _curve_path(tag: str, label: str, seed: int) -> Path:
    safe = label.replace(" ", "_").replace("/", "-").replace("+", "plus")
    return CURVE_DIR / f"{tag}__{safe}__seed{seed}.csv"


# --------------------------------------------------------------------------- #
# Crash resumability
#
# A full ablation is ~50 training runs and over an hour of CPU. Holding all of
# it in memory until the final ``save()`` means a kernel death at run 45 costs
# everything. Each finished run is therefore appended to a progress journal the
# moment it completes, and a restarted run skips whatever the journal already
# contains. The journal is deleted once the real result CSV is written, so it
# only ever exists while a run is in flight or after one has died.
# --------------------------------------------------------------------------- #
def _progress_path(tag: str) -> Path:
    return CURVE_DIR / f"{tag}.progress.jsonl"


def _load_progress(tag: str) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Recover rows and curves from a run that did not finish."""
    path = _progress_path(tag)
    if not path.exists():
        return [], {}

    rows: list[dict[str, Any]] = []
    curves: dict[str, pd.DataFrame] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # a torn final line means the process died mid-write; everything
            # before it is still good
            break
        curve_path = _curve_path(tag, str(row["variant"]), int(row["seed"]))
        if not curve_path.exists():
            continue
        rows.append(row)
        curves[f"{row['variant']}|{row['seed']}"] = pd.read_csv(
            curve_path, index_col=0, parse_dates=True
        )
    return rows, curves


def _record_progress(tag: str, row: Mapping[str, Any], curve: pd.DataFrame) -> None:
    """Persist one finished run. Curve first, so the journal never claims a
    result whose curve is missing."""
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(_curve_path(tag, str(row["variant"]), int(row["seed"])))
    with _progress_path(tag).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), default=str) + "\n")


def _clear_progress(tag: str) -> None:
    _progress_path(tag).unlink(missing_ok=True)


def _resume(
    tag: str, force: bool, progress: Callable[[str], None] | None
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], set[tuple[str, int]]]:
    if force:
        _clear_progress(tag)
        return [], {}, set()
    rows, curves = _load_progress(tag)
    if rows and progress:
        progress(f"resuming {tag}: {len(rows)} run(s) recovered from a previous attempt")
    return rows, curves, {(str(r["variant"]), int(r["seed"])) for r in rows}


@dataclass
class ExperimentResults:
    """One tidy table plus the daily wealth curve behind every row."""

    tag: str
    table: pd.DataFrame
    curves: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return config.ARTIFACTS_RESULTS / f"{self.tag}.csv"

    def save(self) -> Path:
        CURVE_DIR.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(self.path, index=False)
        for key, curve in self.curves.items():
            label, seed = key.rsplit("|", 1)
            curve.to_csv(_curve_path(self.tag, label, int(seed)))
        return self.path

    @classmethod
    def load(cls, tag: str) -> "ExperimentResults":
        table = pd.read_csv(config.ARTIFACTS_RESULTS / f"{tag}.csv")
        curves: dict[str, pd.DataFrame] = {}
        for _, row in table.iterrows():
            path = _curve_path(tag, str(row["variant"]), int(row["seed"]))
            if path.exists():
                curves[f"{row['variant']}|{row['seed']}"] = pd.read_csv(
                    path, index_col=0, parse_dates=True
                )
        return cls(tag=tag, table=table, curves=curves)

    @classmethod
    def exists(cls, tag: str) -> bool:
        return (config.ARTIFACTS_RESULTS / f"{tag}.csv").exists()

    def curve(self, variant: str, seed: int) -> pd.DataFrame:
        return self.curves[f"{variant}|{seed}"]

    def by_variant(self, metric: str = "test_sharpe") -> pd.DataFrame:
        """Seeds as rows, variants as columns -- the shape both the box plot and
        the paired significance test need."""
        return self.table.pivot(index="seed", columns="variant", values=metric)

    def summary(self, metrics_: Sequence[str] = ()) -> pd.DataFrame:
        """Mean and standard deviation across seeds, per variant."""
        metrics_ = metrics_ or (
            "test_sharpe", "test_cagr", "test_max_drawdown", "val_sharpe"
        )
        agg = self.table.groupby("variant")[list(metrics_)].agg(["mean", "std"])
        return agg.sort_values(("test_sharpe", "mean"), ascending=False)


# --------------------------------------------------------------------------- #
# Core runner
# --------------------------------------------------------------------------- #
def _evaluate_split(
    agent, split_ds: Dataset, env_cfg, prefix: str, policy=None
) -> tuple[dict[str, Any], pd.DataFrame]:
    res = train.evaluate(agent, split_ds, env_cfg, policy=policy)
    perf = res["performance"]
    return {
        f"{prefix}_sharpe": perf["Sharpe"],
        f"{prefix}_sortino": perf["Sortino"],
        f"{prefix}_cagr": perf["CAGR"],
        f"{prefix}_volatility": perf["Volatility"],
        f"{prefix}_max_drawdown": perf["Max drawdown"],
        f"{prefix}_calmar": perf["Calmar"],
        f"{prefix}_final_wealth": res["final_wealth"],
        f"{prefix}_action_entropy": res["action_entropy"],
        f"{prefix}_mean_turnover": res["summary"]["mean_turnover"],
        f"{prefix}_total_cost": res["summary"]["total_cost_fraction"],
    }, res["daily"]


def run_variant_seeds(
    dataset: Dataset,
    *,
    variants: Mapping[str, Mapping[str, bool]] | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
    tag: str = "05_ablation",
    total_steps: int = 60_000,
    eval_every: int = 5_000,
    agent_overrides: Mapping[str, Any] | None = None,
    env_cfg: config.EnvConfig | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> ExperimentResults:
    """Train every (variant, seed) pair and evaluate on validation *and* test.

    The test evaluation is recorded here but is not used for any selection --
    the checkpoint is chosen inside ``train_dqn`` on validation Sharpe.  Writing
    both into the same table is deliberate: notebook 05 plots validation against
    test to show how much of the validation ranking survives, which is a far more
    informative picture than test numbers alone.
    """
    if not force and ExperimentResults.exists(tag):
        if progress:
            progress(f"loading cached results from {tag}.csv (pass force=True to rerun)")
        return ExperimentResults.load(tag)

    variants = variants or VARIANTS
    env_cfg = env_cfg or config.DEFAULT.env
    overrides = dict(agent_overrides or {})

    train_ds, valid_ds, test_ds = (dataset.split(s) for s in ("train", "valid", "test"))

    rows, curves, already_done = _resume(tag, force, progress)
    total = len(variants) * len(seeds)
    started = time.perf_counter()
    completed_here = 0

    for i, (label, flags) in enumerate(variants.items()):
        for j, seed in enumerate(seeds):
            if (label, seed) in already_done:
                continue
            cfg_i = variant_config(
                double=flags["double"], dueling=flags["dueling"],
                total_steps=total_steps, eval_every=eval_every, seed=seed, **overrides
            )
            result = train.train_dqn(
                train_ds, valid_ds, agent_cfg=cfg_i, env_cfg=env_cfg,
                run_name=f"{tag}_{label.replace(' ', '_')}",
                save_checkpoints=False, write_log=False, progress=None,
            )
            row: dict[str, Any] = {
                "variant": label, "seed": seed,
                "double_dqn": flags["double"], "dueling": flags["dueling"],
                "selected_step": result.best.get("step", 0),
                "wall_time": round(result.wall_time, 1),
            }
            val_metrics, _ = _evaluate_split(result.agent, valid_ds, env_cfg, "val")
            test_metrics, test_daily = _evaluate_split(result.agent, test_ds, env_cfg, "test")
            row.update(val_metrics)
            row.update(test_metrics)
            rows.append(row)
            curves[f"{label}|{seed}"] = test_daily
            _record_progress(tag, row, test_daily)
            completed_here += 1

            if progress:
                done = len(rows)
                eta = (time.perf_counter() - started) / completed_here * (total - done)
                progress(
                    f"  [{done:>2}/{total}] {label:<20} seed {seed}  "
                    f"val {row['val_sharpe']:+.2f}  test {row['test_sharpe']:+.2f}  "
                    f"(eta {eta / 60:.0f}m)"
                )

    out = ExperimentResults(tag=tag, table=pd.DataFrame(rows), curves=curves)
    out.save()
    _clear_progress(tag)
    return out


# --------------------------------------------------------------------------- #
# Named-configuration comparison
# --------------------------------------------------------------------------- #
def config_comparison(
    dataset: Dataset,
    *,
    configs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    seeds: Sequence[int] = (0, 1, 2),
    tag: str = "05_config_comparison",
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> ExperimentResults:
    """Train several *named* whole configurations under matched seeds.

    ``run_variant_seeds`` only varies the Double/Duelling flags, which is the
    wrong tool for the question raised by the headline result: the tuned agent
    scored far below the untuned ablation agents, but it differed from them in
    *two* ways at once -- tuned hyperparameters and a doubled training budget.
    Those effects are confounded and cannot be separated by re-reading the
    existing tables.

    Each entry maps a label to ``{"agent": {...}, "env": {...}}``, either of
    which may be omitted.  Seeds are matched across configurations so the
    comparison is paired, exactly as in the variant ablation.
    """
    from dataclasses import replace

    if not force and ExperimentResults.exists(tag):
        if progress:
            progress(f"loading cached results from {tag}.csv")
        return ExperimentResults.load(tag)

    train_ds, valid_ds, test_ds = (dataset.split(s) for s in ("train", "valid", "test"))
    rows, curves, already_done = _resume(tag, force, progress)
    total = len(configs) * len(seeds)

    for label, spec in configs.items():
        agent_overrides = dict(spec.get("agent", {}))
        env_overrides = dict(spec.get("env", {}))
        env_cfg = replace(config.DEFAULT.env, **env_overrides) if env_overrides else None

        for seed in seeds:
            if (label, seed) in already_done:
                continue
            cfg_i = variant_config(
                double=agent_overrides.pop("double", True),
                dueling=agent_overrides.pop("dueling", True),
                seed=seed,
                **agent_overrides,
            )
            result = train.train_dqn(
                train_ds, valid_ds, agent_cfg=cfg_i, env_cfg=env_cfg,
                run_name=f"{tag}_{label.replace(' ', '_')}",
                save_checkpoints=False, write_log=False, progress=None,
            )
            val_metrics, _ = _evaluate_split(result.agent, valid_ds, env_cfg, "val")
            test_metrics, test_daily = _evaluate_split(result.agent, test_ds, env_cfg, "test")
            row = {
                "variant": label, "seed": seed,
                "total_steps": cfg_i.total_steps,
                "selected_step": result.best.get("step", 0),
                "wall_time": round(result.wall_time, 1),
                **val_metrics, **test_metrics,
            }
            rows.append(row)
            curves[f"{label}|{seed}"] = test_daily
            _record_progress(tag, row, test_daily)
            if progress:
                progress(f"  [{len(rows):>2}/{total}] {label:<28} seed {seed}  "
                         f"val {row['val_sharpe']:+.2f}  test {row['test_sharpe']:+.2f}")

    out = ExperimentResults(tag=tag, table=pd.DataFrame(rows), curves=curves)
    out.save()
    _clear_progress(tag)
    return out


# --------------------------------------------------------------------------- #
# Transaction-cost sensitivity
# --------------------------------------------------------------------------- #
def cost_sweep(
    dataset: Dataset,
    *,
    cost_bps: Sequence[float] = (0.0, 5.0, 10.0, 20.0),
    seeds: Sequence[int] = (0, 1, 2),
    tag: str = "05_cost_sweep",
    total_steps: int = 60_000,
    eval_every: int = 5_000,
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> ExperimentResults:
    """Retrain and re-evaluate at several transaction-cost levels.

    A strategy that only works at zero cost is a strategy that does not work.
    Retraining at each level lets the agent respond to the cost -- if turnover
    falls as costs rise, the reward function is doing its job; if it does not,
    the turnover penalty is mis-specified.
    """
    from dataclasses import replace

    if not force and ExperimentResults.exists(tag):
        if progress:
            progress(f"loading cached results from {tag}.csv")
        return ExperimentResults.load(tag)

    train_ds, valid_ds, test_ds = (dataset.split(s) for s in ("train", "valid", "test"))
    rows, curves, already_done = _resume(tag, force, progress)

    for bps in cost_bps:
        env_cfg = replace(config.DEFAULT.env, transaction_cost_bps=bps)
        label = f"{bps:.0f} bps"
        for seed in seeds:
            if (label, seed) in already_done:
                continue
            cfg_i = variant_config(double=True, dueling=True, total_steps=total_steps,
                                   eval_every=eval_every, seed=seed)
            result = train.train_dqn(
                train_ds, valid_ds, agent_cfg=cfg_i, env_cfg=env_cfg,
                run_name=f"{tag}_{int(bps)}bps",
                save_checkpoints=False, write_log=False, progress=None,
            )
            test_metrics, test_daily = _evaluate_split(result.agent, test_ds, env_cfg, "test")
            row = {"variant": label, "cost_bps": bps, "seed": seed, **test_metrics}
            rows.append(row)
            curves[f"{label}|{seed}"] = test_daily
            _record_progress(tag, row, test_daily)
            if progress:
                progress(f"  {label:>7}  seed {seed}  test Sharpe "
                         f"{test_metrics['test_sharpe']:+.2f}  turnover "
                         f"{test_metrics['test_mean_turnover']:.1%}")

    out = ExperimentResults(tag=tag, table=pd.DataFrame(rows), curves=curves)
    out.save()
    _clear_progress(tag)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward_folds(
    dataset: Dataset,
    *,
    first_test_year: int = 2021,
    last_test_year: int = 2025,
    valid_years: int = 2,
) -> list[dict[str, Any]]:
    """Expanding-window folds: train on everything up to year Y, test on year Y.

    The two years immediately before the test year are held out for validation,
    so checkpoint selection never touches data adjacent to (or after) the test
    year.  Without that buffer the "walk forward" would leak the very
    information it claims to exclude.
    """
    folds = []
    for year in range(first_test_year, last_test_year + 1):
        folds.append(
            {
                "fold": year,
                "train_end": f"{year - valid_years - 1}-12-31",
                "valid_start": f"{year - valid_years}-01-01",
                "valid_end": f"{year - 1}-12-31",
                "test_start": f"{year}-01-01",
                "test_end": f"{year}-12-31",
            }
        )
    return folds


def _slice(dataset: Dataset, start: str | None, end: str | None) -> Dataset:
    """Restrict a Dataset to a date window without re-fitting the scaler."""
    from .features import Dataset as DS

    mask = pd.Series(True, index=dataset.prices.index)
    if start:
        mask &= dataset.prices.index >= pd.Timestamp(start)
    if end:
        mask &= dataset.prices.index <= pd.Timestamp(end)
    idx = dataset.prices.index[mask.to_numpy()]
    return DS(
        prices=dataset.prices.loc[idx],
        returns=dataset.returns.loc[idx],
        features=dataset.features.loc[idx],
        features_raw=dataset.features_raw.loc[idx],
        risk_free=dataset.risk_free.loc[idx],
        scaler=dataset.scaler,
    )


def walk_forward(
    dataset: Dataset,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    tag: str = "05_walk_forward",
    total_steps: int = 60_000,
    eval_every: int = 5_000,
    agent_overrides: Mapping[str, Any] | None = None,
    env_cfg: config.EnvConfig | None = None,
    ensemble: bool = False,
    hysteresis_margin: float = 0.0,
    force: bool = False,
    progress: Callable[[str], None] | None = print,
    **fold_kwargs,
) -> ExperimentResults:
    """Retrain once per fold and evaluate on the following year only.

    ``ensemble`` adds one extra row per fold, with ``seed = -1``, holding the
    policy that acts on the *mean* Q-values of every seed trained for that fold.
    Seed dispersion is the largest effect in this project, so the ensemble is
    reported as its own strategy rather than as a summary statistic of the
    individual seeds.
    """
    if not force and ExperimentResults.exists(tag):
        if progress:
            progress(f"loading cached results from {tag}.csv")
        return ExperimentResults.load(tag)

    env_cfg = env_cfg or config.DEFAULT.env
    overrides = dict(agent_overrides or {})
    rows, curves, already_done = _resume(tag, force, progress)

    def _policy_for(agents):
        qfn = ensemble_q_values(agents)
        if hysteresis_margin > 0.0:
            return hysteresis_policy(
                qfn, margin=hysteresis_margin, initial_action=env_cfg.initial_action
            )
        return lambda obs: int(np.argmax(qfn(obs)))

    for fold in walk_forward_folds(dataset, **fold_kwargs):
        label = str(fold["fold"])
        wanted = list(seeds) + ([-1] if ensemble else [])
        if all((label, seed) in already_done for seed in wanted):
            continue

        tr = _slice(dataset, None, fold["train_end"])
        va = _slice(dataset, fold["valid_start"], fold["valid_end"])
        te = _slice(dataset, fold["test_start"], fold["test_end"])
        fold_agents = []

        for seed in seeds:
            done = (label, seed) in already_done
            # An ensemble needs every seed's weights in memory, so a resumed run
            # must retrain even the folds it has already scored individually.
            if done and not ensemble:
                continue
            cfg_i = variant_config(double=True, dueling=True, total_steps=total_steps,
                                   eval_every=eval_every, seed=seed, **overrides)
            result = train.train_dqn(
                tr, va, agent_cfg=cfg_i, env_cfg=env_cfg,
                run_name=f"{tag}_{fold['fold']}",
                save_checkpoints=False, write_log=False, progress=None,
            )
            fold_agents.append(result.agent)
            if done:
                continue
            test_metrics, test_daily = _evaluate_split(
                result.agent, te, env_cfg, "test", policy=_policy_for([result.agent])
            )
            row = {
                "variant": label, "seed": seed,
                "train_days": len(tr.dates), "test_days": len(te.dates),
                **{k: v for k, v in fold.items() if k != "fold"},
                **test_metrics,
            }
            rows.append(row)
            curves[f"{label}|{seed}"] = test_daily
            _record_progress(tag, row, test_daily)
            if progress:
                progress(f"  fold {label}  seed {seed}  "
                         f"train to {fold['train_end']}  "
                         f"test Sharpe {test_metrics['test_sharpe']:+.2f}")

        if ensemble and (label, -1) not in already_done:
            test_metrics, test_daily = _evaluate_split(
                fold_agents[0], te, env_cfg, "test", policy=_policy_for(fold_agents)
            )
            row = {
                "variant": label, "seed": -1,
                "train_days": len(tr.dates), "test_days": len(te.dates),
                **{k: v for k, v in fold.items() if k != "fold"},
                **test_metrics,
            }
            rows.append(row)
            curves[f"{label}|{-1}"] = test_daily
            _record_progress(tag, row, test_daily)
            if progress:
                progress(f"  fold {label}  ensemble  "
                         f"test Sharpe {test_metrics['test_sharpe']:+.2f}")

    out = ExperimentResults(tag=tag, table=pd.DataFrame(rows), curves=curves)
    out.save()
    _clear_progress(tag)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward concatenation
#
# A mean of per-year Sharpes is not the Sharpe an investor would have realised:
# it weights a calm year and a crisis year equally and hides the compounding
# between them. Chaining the fold curves into one continuous out-of-sample
# series is the only figure that answers "what would holding this strategy from
# 2021 to 2025 actually have done", and it is the number the benchmarks are
# quoted on.
# --------------------------------------------------------------------------- #
DECISION_SPAN = config.DEFAULT.env.steps_per_decision


def concat_walk_forward_curve(
    results: ExperimentResults, seed: int, initial_value: float | None = None
) -> pd.DataFrame:
    """Chain one seed's per-fold test years into a single continuous curve."""
    initial_value = (
        config.DEFAULT.env.initial_value if initial_value is None else initial_value
    )
    sub = results.table[results.table["seed"] == seed]
    if sub.empty:
        raise KeyError(f"no walk-forward rows for seed {seed}")

    folds = sorted(str(v) for v in sub["variant"].unique())
    pieces = [results.curve(fold, seed) for fold in folds]
    joined = pd.concat(pieces).sort_index()

    # Wealth is rebuilt by compounding, because each fold's own curve restarts
    # at the initial value.
    wealth = initial_value * (1.0 + joined["return"]).cumprod()
    out = joined.copy()
    out["wealth"] = wealth
    out["drawdown"] = 1.0 - wealth / wealth.cummax()
    return out


def concat_walk_forward(
    results: ExperimentResults,
    *,
    risk_free: pd.Series | float = 0.0,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Score every seed's chained out-of-sample curve; one row per seed."""
    seeds = seeds if seeds is not None else sorted(results.table["seed"].unique())
    rows = []
    for seed in seeds:
        seed = int(seed)
        curve = concat_walk_forward_curve(results, seed)
        row = metrics.performance_summary(
            curve, risk_free=risk_free, name=f"seed {seed}"
        )
        sub = results.table[results.table["seed"] == seed]
        n_years = max(len(curve) / metrics.ANNUAL, 1e-9)
        decisions = (sub["test_days"] // DECISION_SPAN) if "test_days" in sub else 0
        row["Ann. turnover"] = float((sub["test_mean_turnover"] * decisions).sum()) / n_years
        row["Total cost"] = float(sub["test_total_cost"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation helpers used by notebook 05
# --------------------------------------------------------------------------- #
def median_seed_curve(results: ExperimentResults, variant: str) -> pd.DataFrame:
    """The curve of the *median-Sharpe* seed, not the mean of the curves.

    Averaging wealth paths across seeds produces a curve no single run ever
    experienced, with an artificially smooth shape and a flattered drawdown.
    Picking the median run keeps the plotted path real.
    """
    sub = results.table[results.table["variant"] == variant]
    ordered = sub.sort_values("test_sharpe")
    median_seed = int(ordered.iloc[len(ordered) // 2]["seed"])
    return results.curve(variant, median_seed)


def seed_dispersion(results: ExperimentResults, metric: str = "test_sharpe") -> pd.DataFrame:
    """Spread of a metric across seeds -- the honest error bar for the ablation."""
    grp = results.table.groupby("variant")[metric]
    out = pd.DataFrame(
        {
            "mean": grp.mean(), "std": grp.std(), "min": grp.min(),
            "median": grp.median(), "max": grp.max(), "n": grp.count(),
        }
    )
    out["range"] = out["max"] - out["min"]
    return out.sort_values("mean", ascending=False)


def save_manifest(paths: Iterable[Path], path: str | Path | None = None) -> Path:
    """Record every artefact with its size and modification time.

    Used by notebook 00 to prove that the committed outputs correspond to the
    committed code rather than to an earlier run.
    """
    path = Path(path or config.ARTIFACTS_RESULTS / "manifest.json")
    entries = []
    for p in sorted(paths):
        p = Path(p)
        if not p.exists():
            continue
        try:
            name = str(p.relative_to(config.PROJECT_ROOT))
        except ValueError:
            name = str(p)
        entries.append(
            {
                "path": name,
                "bytes": p.stat().st_size,
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path
