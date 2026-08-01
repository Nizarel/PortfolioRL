r"""Run the tuned-vs-default 2x2 ahead of notebook 05 so the notebook hits cache.

The headline agent differs from the ablation agents in two ways at once -- tuned
hyperparameters and a doubled training budget -- so the existing tables cannot
say which one caused the collapse in test Sharpe.  This crosses the two factors
under matched seeds.  Notebook 05 calls ``config_comparison`` with the same tag
and reads the result back.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from portfoliorl import config, experiments, features

TAG = "05_config_comparison"
SEEDS = (0, 1, 2)


def tuned_overrides() -> tuple[dict, dict]:
    """Split the search's winning parameters into agent and environment halves."""
    best = json.load(open(config.ARTIFACTS_RESULTS / "04_best_config.json"))
    agent = dict(best["agent"])
    env = dict(best.get("env", {}))

    # config_comparison names the architecture flags "double"/"dueling"
    agent["double"] = agent.pop("double_dqn", True)
    agent["dueling"] = agent.pop("dueling", True)
    agent["hidden_sizes"] = tuple(agent["hidden_sizes"])

    # the budget and seed are the factors being crossed, not part of the config
    for key in ("total_steps", "eval_every", "seed"):
        agent.pop(key, None)
    return agent, env


def main() -> None:
    dataset = features.load_dataset()
    agent_overrides, env_overrides = tuned_overrides()
    print("tuned agent overrides:", agent_overrides)
    print("tuned env overrides:  ", env_overrides)

    configs = {}
    for budget in (60_000, 120_000):
        configs[f"default @ {budget // 1000}k"] = {
            "agent": {"total_steps": budget, "eval_every": 5_000},
        }
        configs[f"tuned @ {budget // 1000}k"] = {
            "agent": {**agent_overrides, "total_steps": budget, "eval_every": 5_000},
            "env": env_overrides,
        }

    started = time.perf_counter()
    res = experiments.config_comparison(
        dataset, configs=configs, seeds=SEEDS, tag=TAG,
    )
    print(f"\ndone in {(time.perf_counter() - started) / 60:.1f} minutes")
    print(res.summary(("test_sharpe", "test_cagr", "test_max_drawdown", "val_sharpe")))


if __name__ == "__main__":
    main()
