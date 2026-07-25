"""Gymnasium environment: a transaction-cost-aware portfolio simulator.

The environment is the part of this project most likely to be wrong in a way
that produces *better* results, so it is written to be verifiable rather than
merely plausible.  ``tests/test_env.py`` contains a golden test that replays a
constant action through the environment and compares the resulting wealth path,
step by step, against an independently written static-allocation calculation.

Mechanics of one step
---------------------
A step is one **rebalancing decision** and spans ``steps_per_decision`` trading
days (5 = weekly).  Within a step:

1. The agent's chosen action names a **target** weight vector.
2. Turnover is measured against the **drifted** weights, not the previous
   target.  Between decisions, weights move on their own as assets appreciate
   at different rates:

   .. math::
      w^-_{t+1} = \\frac{w_t \\odot (1 + r_{t+1})}{1 + w_t \\cdot r_{t+1}}

   Ignoring this drift and comparing target-to-previous-target systematically
   *understates* turnover, and therefore understates cost, which is a
   self-serving error in a paper about cost-aware rebalancing.
3. The trading cost is deducted from **wealth**, not merely subtracted from the
   reward.  A cost that only appears in the reward is a cost the equity curve
   never pays, and the reported CAGR would be fictitious.
4. Daily returns are then applied one day at a time, so every reported metric
   (volatility, drawdown, Sharpe) is computed at daily frequency even though
   decisions are weekly.

Reward
------
.. math::
   R_t = 100 \\times \\Big[ \\log(1 + r^{net}_t)
        - \\lambda_1 \\cdot \\text{turnover}_t
        - \\lambda_2 \\cdot \\sigma^{20d}_t
        - \\lambda_3 \\cdot \\max(0, DD_t - DD_{t-1}) \\Big]

* **Log returns** because they are time-additive: the undiscounted sum of the
  return term over an episode equals terminal log wealth exactly.  The RL
  objective therefore *is* the business objective rather than a proxy for it.
  Arithmetic returns do not have this property -- +50% followed by -50% sums to
  zero while actually losing 25%.
* :math:`r^{net}` is already net of trading cost, so :math:`\\lambda_1` is an
  *additional* behavioural aversion to churn, not a second charge for the same
  thing.
* The **drawdown increment** rather than the level: see ``config.EnvConfig``.
* The ``x 100`` scaling expresses rewards in percentage points.  Raw weekly log
  returns are O(1e-3), which drives the squared TD error to O(1e-6) and leaves
  effectively no gradient signal.  Notebook 03 demonstrates this directly.

Episodes
--------
Training uses **random-start, fixed-length sub-episodes** (52 decisions ~ 1
year).  With a single historical price path, full-window episodes mean the
agent replays the identical 2,500-step trajectory hundreds of times and simply
memorises it.  Random starts turn one path into many thousands of overlapping
trajectories and are the single highest-impact design decision in the project.

Evaluation uses a **deterministic full pass** over the split, starting at the
first date and running to the last, which is what a backtest actually is.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from . import config
from .features import Dataset, N_PORTFOLIO_FEATURES


class PortfolioEnv(gym.Env):
    """Discrete-action, weekly-rebalancing portfolio allocation environment.

    Parameters
    ----------
    dataset
        A (usually split-restricted) :class:`~portfoliorl.features.Dataset`.
    env_cfg
        Simulation and reward parameters.
    mode
        ``"train"`` draws random-start sub-episodes; ``"eval"`` performs one
        deterministic pass over the whole split.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: Dataset,
        env_cfg: config.EnvConfig | None = None,
        mode: str = "train",
    ) -> None:
        super().__init__()
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

        self.cfg = env_cfg or config.DEFAULT.env
        self.mode = mode

        # Cache the data as contiguous float arrays.  Indexing a NumPy array is
        # ~50x faster than .iloc on a DataFrame, and the environment is stepped
        # millions of times -- this is the difference between a 2-minute and a
        # 90-minute training run.
        self._dates = dataset.prices.index
        self._returns = dataset.returns.to_numpy(dtype=np.float64)
        self._features = dataset.features.to_numpy(dtype=np.float64)
        self._n_days = len(self._returns)
        self._n_assets = self._returns.shape[1]
        self._asset_names = list(dataset.prices.columns)

        self._allocations = np.asarray(config.ACTION_ALLOCATIONS, dtype=np.float64)
        if self._allocations.shape[1] != self._n_assets:
            raise ValueError(
                f"ACTION_ALLOCATIONS has {self._allocations.shape[1]} columns but the "
                f"dataset has {self._n_assets} assets"
            )
        if not np.allclose(self._allocations.sum(axis=1), 1.0):
            raise ValueError("Every row of ACTION_ALLOCATIONS must sum to 1")

        self.action_space = spaces.Discrete(len(self._allocations))
        obs_dim = self._features.shape[1] + N_PORTFOLIO_FEATURES
        # Finite bounds rather than +/-inf: market features are clipped by the
        # FeatureScaler and every portfolio-state component is bounded by
        # construction (weights in [0, 1], drawdown in [0, 1], duration capped
        # at 3 years).  Declaring honest bounds lets `check_env` verify them and
        # documents the expected input range of the Q-network.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self._max_decisions = self._n_days // self.cfg.steps_per_decision
        if self._max_decisions < 1:
            raise ValueError(
                f"Split of {self._n_days} days is shorter than one decision period "
                f"of {self.cfg.steps_per_decision} days"
            )

        # Per-episode state, initialised properly in reset().
        self._idx = 0
        self._start_idx = 0
        self._decision = 0
        self._n_decisions = 0
        self._wealth = self.cfg.initial_value
        self._peak = self.cfg.initial_value
        self._weights = self._allocations[self.cfg.initial_action].copy()
        self._daily_returns: list[float] = []
        self._dd_days = 0
        self._prev_dd = 0.0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        # Must be called first: it seeds self.np_random, which is what the
        # random-start sampler below draws from.  Skipping it is the most common
        # cause of a "reproducible" experiment that is not.
        super().reset(seed=seed)

        span = self.cfg.steps_per_decision
        if self.mode == "train" and self.cfg.random_start_episodes:
            self._n_decisions = min(self.cfg.episode_length, self._max_decisions)
            last_start = self._n_days - self._n_decisions * span
            # Sample the *day* rather than the decision index so episodes are not
            # all phase-aligned to the same weekday.
            self._start_idx = int(self.np_random.integers(0, max(last_start, 1)))
        else:
            self._n_decisions = self._max_decisions
            self._start_idx = 0

        self._idx = self._start_idx
        self._decision = 0
        self._wealth = self.cfg.initial_value
        self._peak = self.cfg.initial_value
        self._weights = self._allocations[self.cfg.initial_action].copy()
        self._daily_returns = []
        self._dd_days = 0
        self._prev_dd = 0.0
        self.history = []

        return self._observation(), {"start_date": self._dates[self._start_idx]}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        target = self._allocations[action]
        wealth_open = self._wealth

        # --- 1. Trade from the DRIFTED weights to the target ------------- #
        # turnover is the conventional one-way measure: 0.5 * sum |dw|.
        # Total traded notional is sum |dw| = 2 * turnover, and the one-way cost
        # rate applies to that whole notional.
        delta = np.abs(target - self._weights)
        turnover = 0.5 * float(delta.sum())
        cost_rate = self.cfg.transaction_cost_bps / 1e4
        cost_fraction = 2.0 * turnover * cost_rate

        self._wealth *= 1.0 - cost_fraction  # cost paid out of WEALTH, not just reward
        self._weights = target.copy()

        # --- 2. Hold for `steps_per_decision` trading days ---------------- #
        span = self.cfg.steps_per_decision
        dd_before = self._prev_dd
        days_held = 0

        for _ in range(span):
            if self._idx >= self._n_days - 1:
                break
            self._idx += 1
            daily = self._returns[self._idx]

            port_return = float(self._weights @ daily)
            self._wealth *= 1.0 + port_return
            self._weights = self._drift(self._weights, daily)

            self._daily_returns.append(port_return)
            self._peak = max(self._peak, self._wealth)
            drawdown = 1.0 - self._wealth / self._peak
            self._dd_days = 0 if drawdown <= 1e-12 else self._dd_days + 1

            self.history.append(
                {
                    "date": self._dates[self._idx],
                    "wealth": self._wealth,
                    "return": port_return,
                    "drawdown": drawdown,
                    "action": action,
                    **{f"w_{name}": w for name, w in zip(self._asset_names, self._weights)},
                }
            )
            days_held += 1

        current_dd = 1.0 - self._wealth / self._peak

        # --- 3. Reward --------------------------------------------------- #
        net_return = self._wealth / wealth_open - 1.0
        return_term = (
            float(np.log1p(net_return)) if self.cfg.use_log_return else float(net_return)
        )
        vol_term = self._portfolio_vol()
        dd_increment = max(0.0, current_dd - dd_before)

        reward = self.cfg.reward_scale * (
            return_term
            - self.cfg.lambda_turnover * turnover
            - self.cfg.lambda_volatility * vol_term
            - self.cfg.lambda_drawdown * dd_increment
        )

        self._prev_dd = current_dd
        self._decision += 1

        # Both endings are *time limits* imposed by the finite data sample, not
        # a genuine absorbing state, so both are reported as truncation.  This
        # matters for learning: the target network must still bootstrap from the
        # final observation rather than treating it as terminal with V = 0.
        out_of_data = self._idx >= self._n_days - 1
        truncated = bool(self._decision >= self._n_decisions or out_of_data or days_held == 0)

        info = {
            "date": self._dates[self._idx],
            "wealth": self._wealth,
            "net_return": net_return,
            "turnover": turnover,
            "cost_fraction": cost_fraction,
            "drawdown": current_dd,
            "dd_increment": dd_increment,
            "portfolio_vol": vol_term,
            "days_held": days_held,
            # Reward decomposition: consumed by the stacked-bar diagnostic in
            # notebook 02 that shows which term is actually driving behaviour.
            "reward_return": self.cfg.reward_scale * return_term,
            "reward_turnover_penalty": -self.cfg.reward_scale
            * self.cfg.lambda_turnover
            * turnover,
            "reward_vol_penalty": -self.cfg.reward_scale * self.cfg.lambda_volatility * vol_term,
            "reward_dd_penalty": -self.cfg.reward_scale * self.cfg.lambda_drawdown * dd_increment,
            "weights": self._weights.copy(),
        }

        return self._observation(), float(reward), False, truncated, info

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _drift(weights: np.ndarray, daily_returns: np.ndarray) -> np.ndarray:
        """Update weights for one day of price movement with no trading.

        This is the mechanical fact that makes a "buy and hold" portfolio drift
        away from its stated allocation, and therefore the reason rebalancing
        costs money.
        """
        grown = weights * (1.0 + daily_returns)
        total = grown.sum()
        if total <= 0:
            # Total wipe-out is not reachable with these ETFs, but guard against
            # dividing by zero rather than emitting a silent NaN into the state.
            return weights
        return grown / total

    def _portfolio_vol(self) -> float:
        """Rolling std of *daily* portfolio returns (not annualised).

        Deliberately left in daily units: see the calibration note on
        ``EnvConfig.lambda_volatility``.
        """
        window = self._daily_returns[-self.cfg.portfolio_vol_window :]
        if len(window) < 2:
            return 0.0
        return float(np.std(window, ddof=1))

    def _observation(self) -> np.ndarray:
        """Concatenate market features with the agent's own portfolio state.

        The eight portfolio components are already O(1) by construction, so they
        bypass the feature scaler: weights are fractions, daily volatility is
        multiplied by 100 to sit near unity, drawdown is a fraction, and duration
        is expressed in years.
        """
        market = self._features[self._idx]

        drawdown = 1.0 - self._wealth / self._peak
        portfolio = np.array(
            [
                *self._weights,
                self._portfolio_vol() * 100.0,
                drawdown,
                min(self._dd_days / config.TRADING_DAYS_PER_YEAR, 3.0),
                self._decision / max(self._n_decisions, 1),
            ],
            dtype=np.float64,
        )
        obs = np.concatenate([market, portfolio])
        # Guard the declared observation-space bounds.  Only a pathological
        # volatility spike could reach the limit, but an out-of-bounds
        # observation would silently violate the space contract.
        np.clip(obs, -10.0, 10.0, out=obs)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def to_frame(self) -> pd.DataFrame:
        """Daily record of the episode just played.

        Daily rather than per-decision because every performance metric in the
        report (Sharpe, Sortino, maximum drawdown) is defined on daily returns.
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history).set_index("date")

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    @property
    def wealth(self) -> float:
        return self._wealth


# --------------------------------------------------------------------------- #
# Policy roll-out
# --------------------------------------------------------------------------- #
Policy = Callable[[np.ndarray], int]


def run_policy(
    dataset: Dataset,
    policy: Policy,
    env_cfg: config.EnvConfig | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Play ``policy`` through one deterministic full pass of ``dataset``.

    Every strategy in this project -- the RL agent, all rule-based benchmarks
    and the random-policy floor -- is evaluated through this one function, so
    they are guaranteed to face identical prices, identical rebalancing
    frequency and identical transaction costs.  A benchmark evaluated outside
    the environment would silently trade for free.
    """
    env = PortfolioEnv(dataset, env_cfg, mode="eval")
    obs, _ = env.reset(seed=seed)

    rewards: list[float] = []
    decisions: list[dict[str, Any]] = []

    while True:
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        decisions.append(
            {
                "date": info["date"],
                "action": action,
                "turnover": info["turnover"],
                "cost_fraction": info["cost_fraction"],
                "reward": reward,
                "reward_return": info["reward_return"],
                "reward_turnover_penalty": info["reward_turnover_penalty"],
                "reward_vol_penalty": info["reward_vol_penalty"],
                "reward_dd_penalty": info["reward_dd_penalty"],
            }
        )
        if terminated or truncated:
            break

    daily = env.to_frame()
    summary = {
        "final_wealth": env.wealth,
        "total_reward": float(np.sum(rewards)),
        "n_decisions": len(rewards),
        "n_days": len(daily),
        "total_cost_fraction": float(sum(d["cost_fraction"] for d in decisions)),
        "mean_turnover": float(np.mean([d["turnover"] for d in decisions])),
        "decisions": pd.DataFrame(decisions).set_index("date"),
    }
    return daily, summary
