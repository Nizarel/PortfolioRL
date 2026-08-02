"""Phase 3 screening ladder.

Every candidate change is screened on test years 2018-2020, which are *not* the
reporting window. The 2021-2025 test split has already been opened once; using
it to choose between configurations would turn it into a validation set and make
the final number meaningless.

Each rung adds one change to the rung above it. A change is kept only if it
moves the concatenated Sharpe by more than the seed-to-seed spread, because the
existing ablation showed every architectural effect in this project to be an
order of magnitude smaller than seed noise.

Run:  .\\.venv\\Scripts\\python.exe tools\\run_screening.py
"""

from __future__ import annotations

import dataclasses
import json
import time

import pandas as pd

from portfoliorl import config, experiments, features

SEEDS = (0, 1, 2)
TOTAL_STEPS = 40_000
FIRST_YEAR, LAST_YEAR = 2018, 2020

BASE_ENV = config.DEFAULT.env

# The proposal's original 6-action menu, so rung S0 is the honest starting point.
PROPOSAL_MENU = dataclasses.replace(BASE_ENV, n_actions=6)

LADDER: dict[str, dict] = {
    "S0 baseline": dict(
        env_cfg=PROPOSAL_MENU,
    ),
    "S1 smoothed+ensemble": dict(
        env_cfg=PROPOSAL_MENU,
        agent_overrides={"selection": "smoothed", "select_window": 3},
        ensemble=True,
    ),
    "S2 +full equity action": dict(
        env_cfg=BASE_ENV,
        agent_overrides={"selection": "smoothed", "select_window": 3},
        ensemble=True,
    ),
    "S3 +switch penalty": dict(
        env_cfg=dataclasses.replace(BASE_ENV, lambda_switch=0.001, include_prev_action=True),
        agent_overrides={"selection": "smoothed", "select_window": 3},
        ensemble=True,
    ),
    "S4 +stress sampling": dict(
        env_cfg=dataclasses.replace(
            BASE_ENV,
            lambda_switch=0.001,
            include_prev_action=True,
            stress_sampling_fraction=0.25,
        ),
        agent_overrides={"selection": "smoothed", "select_window": 3},
        ensemble=True,
    ),
    "S5 +longer episodes": dict(
        env_cfg=dataclasses.replace(
            BASE_ENV,
            lambda_switch=0.001,
            include_prev_action=True,
            stress_sampling_fraction=0.25,
            episode_length=104,
        ),
        agent_overrides={"selection": "smoothed", "select_window": 3},
        ensemble=True,
    ),
}


def main() -> None:
    prices = pd.read_csv(config.DATA_PROCESSED / "prices.csv", index_col=0, parse_dates=True)
    rf = pd.read_csv(
        config.DATA_PROCESSED / "risk_free.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    dataset = features.build_dataset(prices, rf)

    # Screening years are scored on their own risk-free rate, on the same
    # excess basis the final scorecard uses.
    screen_rf = rf.loc[f"{FIRST_YEAR}-01-01" : f"{LAST_YEAR}-12-31"]

    rows = []
    for name, spec in LADDER.items():
        started = time.time()
        print(f"\n=== {name} ===", flush=True)
        tag = "06_screen_" + name.split()[0].lower()
        res = experiments.walk_forward(
            dataset,
            seeds=SEEDS,
            tag=tag,
            total_steps=TOTAL_STEPS,
            first_test_year=FIRST_YEAR,
            last_test_year=LAST_YEAR,
            **spec,
        )
        scored = experiments.concat_walk_forward(res, risk_free=screen_rf)

        individual = scored.drop(index="seed -1", errors="ignore")
        row = {
            "rung": name,
            "sharpe_median": float(individual["Sharpe"].median()),
            "sharpe_min": float(individual["Sharpe"].min()),
            "sharpe_max": float(individual["Sharpe"].max()),
            "sharpe_spread": float(individual["Sharpe"].max() - individual["Sharpe"].min()),
            "cagr_median": float(individual["CAGR"].median()),
            "turnover_median": float(individual["Ann. turnover"].median()),
            "minutes": (time.time() - started) / 60.0,
        }
        if "seed -1" in scored.index:
            ens = scored.loc["seed -1"]
            row["sharpe_ensemble"] = float(ens["Sharpe"])
            row["cagr_ensemble"] = float(ens["CAGR"])
            row["turnover_ensemble"] = float(ens["Ann. turnover"])
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in row.items()}, indent=2), flush=True)

    table = pd.DataFrame(rows)
    out = config.ARTIFACTS_RESULTS / "06_screening.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
