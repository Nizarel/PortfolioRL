"""Phase 4 confirmatory walk-forward. Runs ONCE, on 2021-2025.

Everything this script does is decided in advance. The configuration comes from
the Phase 3 screening ladder, which never saw these years, and is written to
``06_frozen_config.json`` before any training starts so the frozen values are
auditable against the results.

If a change is made to FROZEN_* after seeing the output of this script, the
2021-2025 window has become a validation set and the reported number is no
longer out of sample. Re-run with a fresh tag and say so in the report.

Run:  .\\.venv\\Scripts\\python.exe tools\\run_confirmatory.py
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

from portfoliorl import benchmarks, config, experiments, features, metrics, significance

TAG = "06_confirmatory"
SEEDS = tuple(range(10))
TOTAL_STEPS = 60_000
FIRST_YEAR, LAST_YEAR = 2021, 2025

# --- Frozen by Phase 3 screening ------------------------------------------ #
FROZEN_ENV = dataclasses.replace(config.DEFAULT.env)
FROZEN_AGENT_OVERRIDES: dict = {"selection": "smoothed", "select_window": 3}
FROZEN_ENSEMBLE = True
FROZEN_HYSTERESIS = 0.0

#: Benchmarks the agent is claimed to beat. Everything else is reported but not
#: tested, which keeps the multiple-comparison correction honest about its size.
TESTED_AGAINST = ("60/40 rebalanced", "Equal weight")


def _freeze() -> dict:
    frozen = {
        "tag": TAG,
        "seeds": list(SEEDS),
        "total_steps": TOTAL_STEPS,
        "test_years": [FIRST_YEAR, LAST_YEAR],
        "env": dataclasses.asdict(FROZEN_ENV),
        "agent_overrides": FROZEN_AGENT_OVERRIDES,
        "ensemble": FROZEN_ENSEMBLE,
        "hysteresis_margin": FROZEN_HYSTERESIS,
        "tested_against": list(TESTED_AGAINST),
    }
    path = config.ARTIFACTS_RESULTS / "06_frozen_config.json"
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"froze configuration to {path}")
    return frozen


def main() -> None:
    _freeze()

    prices = pd.read_csv(config.DATA_PROCESSED / "prices.csv", index_col=0, parse_dates=True)
    rf = pd.read_csv(
        config.DATA_PROCESSED / "risk_free.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    dataset = features.build_dataset(prices, rf)

    window = slice(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31")
    rf_test = rf.loc[window]

    results = experiments.walk_forward(
        dataset,
        seeds=SEEDS,
        tag=TAG,
        total_steps=TOTAL_STEPS,
        first_test_year=FIRST_YEAR,
        last_test_year=LAST_YEAR,
        env_cfg=FROZEN_ENV,
        agent_overrides=FROZEN_AGENT_OVERRIDES,
        ensemble=FROZEN_ENSEMBLE,
        hysteresis_margin=FROZEN_HYSTERESIS,
    )

    # --- Per-seed chained out-of-sample performance ----------------------- #
    per_seed = experiments.concat_walk_forward(results, risk_free=rf_test)
    per_seed.to_csv(config.ARTIFACTS_RESULTS / "06_confirmatory_per_seed.csv")
    individual = per_seed.drop(index="seed -1", errors="ignore")

    print("\n--- per-seed chained 2021-2025 (excess basis) ---")
    print(individual[["Sharpe", "CAGR", "Max drawdown", "Ann. turnover"]].to_string())
    print(f"\nmedian Sharpe {individual['Sharpe'].median():.3f}   "
          f"IQR {individual['Sharpe'].quantile(.25):.3f}-{individual['Sharpe'].quantile(.75):.3f}   "
          f"worst {individual['Sharpe'].min():.3f}")

    # --- Scorecard against every benchmark -------------------------------- #
    test_ds = dataset.split("test")
    bench = benchmarks.run_benchmarks(test_ds, env_cfg=FROZEN_ENV)
    board = metrics.scorecard(bench, risk_free=rf_test, columns=list(metrics.SCORECARD_COLUMNS))

    agent_label = "RL (DQN, walk-forward ensemble)" if FROZEN_ENSEMBLE else "RL (DQN, walk-forward)"
    agent_key = "seed -1" if "seed -1" in per_seed.index else individual["Sharpe"].idxmin()
    agent_row = per_seed.loc[agent_key].rename(agent_label)
    board = pd.concat([board, agent_row.to_frame().T])
    board = board.sort_values("Sharpe", ascending=False)
    board.to_csv(config.ARTIFACTS_RESULTS / "06_confirmatory_scorecard.csv")

    print("\n--- scorecard, chained 2021-2025 (excess basis) ---")
    print(board[["Sharpe", "CAGR", "Volatility", "Max drawdown", "Ann. turnover"]].to_string())

    # --- Significance ----------------------------------------------------- #
    agent_curve = experiments.concat_walk_forward_curve(results, seed=int(agent_key.split()[-1]))
    agent_returns = metrics.to_returns(agent_curve)

    rows = []
    for name in TESTED_AGAINST:
        bench_returns = metrics.to_returns(bench[name][0])
        joined = pd.concat(
            [agent_returns.rename("a"), bench_returns.rename("b")], axis=1, join="inner"
        ).dropna()
        res = significance.bootstrap_sharpe_difference(
            joined["a"], joined["b"], n_boot=2000, expected_block=10.0, seed=0
        )
        rows.append({
            "benchmark": name,
            "sharpe_difference": res.observed,
            "ci_low": res.ci_low,
            "ci_high": res.ci_high,
            "p_value": res.p_value,
            "n_days": len(joined),
        })

    sig = pd.DataFrame(rows).set_index("benchmark")
    adjusted = significance.holm_bonferroni(sig["p_value"])
    sig = sig.join(adjusted[["Holm-adjusted", "reject at alpha"]])
    sig.to_csv(config.ARTIFACTS_RESULTS / "06_confirmatory_significance.csv")

    print("\n--- paired stationary bootstrap vs the tested benchmarks ---")
    print(sig.to_string())

    summary = {
        "median_sharpe": float(individual["Sharpe"].median()),
        "worst_seed_sharpe": float(individual["Sharpe"].min()),
        "ensemble_sharpe": float(per_seed.loc[agent_key, "Sharpe"]),
        "beats_60_40": bool(per_seed.loc[agent_key, "Sharpe"] > board.loc["60/40 rebalanced", "Sharpe"]),
        "beats_equal_weight": bool(per_seed.loc[agent_key, "Sharpe"] > board.loc["Equal weight", "Sharpe"]),
        "n_seeds": len(individual),
    }
    (config.ARTIFACTS_RESULTS / "06_confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
