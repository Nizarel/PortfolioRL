"""Training loop: run the agent through the market, evaluate, checkpoint.

Three design decisions in this file matter more than the rest of the code.

**1. Truncation is not termination.**  Training episodes end after 52 decisions
because that is the sub-episode budget, not because the world ended.  The
bootstrap target must still be applied, so the ``done`` flag written into the
replay buffer is ``terminated`` (always ``False`` here), never ``truncated``.
Conflating them teaches the agent that all value vanishes every 52 weeks, which
makes it myopically greedy near the (arbitrary) episode boundary.  This is the
single most common silent bug in time-limited RL and is why Gymnasium split the
two flags apart in the first place.

**2. Model selection happens on the validation split, never the test split.**
Checkpoints are scored by validation Sharpe.  The test split (2021-2025) is
touched exactly once, in notebook 05, after everything is frozen.  Selecting on
test would make every number in the report an in-sample number, which is
precisely the failure mode that makes most published backtests unreproducible.

**3. Validation Sharpe, not validation reward.**  Reward includes the shaping
penalties, which are our modelling choices; Sharpe is the quantity an investor
actually cares about.  Selecting on reward would reward an agent for being good
at our reward function rather than good at investing.

Everything is logged to a JSONL file under ``artifacts/logs/`` so a run can be
replotted without re-training.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config, metrics
from .agent import DQNAgent, epsilon_by_step, save_json
from .env import PortfolioEnv, run_policy
from .features import Dataset

# ``save_json`` is re-exported so notebooks can write run metadata without
# needing a second import from agent.py.
__all__ = ["TrainResult", "evaluate", "train_dqn", "load_log", "smooth", "save_json"]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Everything notebook 03 needs to tell the story of one training run."""

    agent: DQNAgent
    episodes: pd.DataFrame          # one row per training episode
    updates: pd.DataFrame           # one row per logged gradient step
    evaluations: pd.DataFrame       # one row per validation evaluation
    best: dict[str, Any]            # metrics of the selected checkpoint
    config: dict[str, Any]
    wall_time: float
    log_path: Path | None = None
    checkpoint_path: Path | None = None
    history: dict[str, Any] = field(default_factory=dict)

    @property
    def best_val_sharpe(self) -> float:
        return float(self.best.get("val_sharpe", np.nan))

    def summary_line(self) -> str:
        return (
            f"{self.config.get('run_name', 'run')}: "
            f"best val Sharpe {self.best_val_sharpe:.3f} "
            f"at step {self.best.get('step', 0):,} "
            f"({self.wall_time:.0f}s, {len(self.episodes)} episodes)"
        )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    agent: DQNAgent,
    dataset: Dataset,
    env_cfg: config.EnvConfig | None = None,
    risk_free: float | pd.Series = 0.0,
) -> dict[str, Any]:
    """Deterministic greedy pass over ``dataset``; returns headline metrics.

    Uses the same :func:`~portfoliorl.env.run_policy` as every benchmark, so the
    agent is charged the same transaction costs on the same dates.
    """
    daily, summary = run_policy(dataset, agent.policy(), env_cfg=env_cfg)
    perf = metrics.performance_summary(
        daily, risk_free=risk_free, summary=summary, name="agent"
    )
    actions = summary["decisions"]["action"].to_numpy()
    counts = np.bincount(actions, minlength=len(config.ACTION_ALLOCATIONS))
    share = counts / max(1, counts.sum())
    # Entropy of the action distribution, in nats.  A collapsed policy (one
    # action always) scores 0; a uniform policy scores log(6) = 1.79.  This is
    # the cheapest early-warning signal for a degenerate agent.
    nz = share[share > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "final_wealth": float(summary["final_wealth"]),
        "total_reward": float(summary["total_reward"]),
        "sharpe": float(perf["Sharpe"]),
        "cagr": float(perf["CAGR"]),
        "max_drawdown": float(perf["Max drawdown"]),
        "volatility": float(perf["Volatility"]),
        "mean_turnover": float(summary["mean_turnover"]),
        "action_entropy": entropy,
        "action_share": share.tolist(),
        "daily": daily,
        "summary": summary,
        "performance": perf,
    }


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_dqn(
    train_dataset: Dataset,
    valid_dataset: Dataset | None = None,
    agent_cfg: config.AgentConfig | None = None,
    env_cfg: config.EnvConfig | None = None,
    *,
    run_name: str = "dqn",
    log_every: int = 200,
    save_checkpoints: bool = True,
    write_log: bool = True,
    progress: Callable[[str], None] | None = print,
    risk_free: float | pd.Series = 0.0,
    pruning_callback: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Train a DQN agent and select the checkpoint with the best validation Sharpe.

    Parameters
    ----------
    train_dataset, valid_dataset
        Split-restricted datasets.  If ``valid_dataset`` is ``None`` no model
        selection is performed and the final weights are returned.
    agent_cfg, env_cfg
        Hyperparameters.  ``agent_cfg.seed`` seeds PyTorch, the replay buffer,
        the exploration RNG and the environment's episode sampler.
    run_name
        Used for the log filename and the checkpoint filename.
    log_every
        Gradient-step diagnostics are averaged over this many updates before
        being recorded, which keeps the history small enough to plot.
    pruning_callback
        Called as ``(step, val_sharpe)`` after each evaluation.  Optuna's
        pruner is wired in here in ``tuning.py``; it raises to abort a run.
    """
    agent_cfg = agent_cfg or config.DEFAULT.agent
    env_cfg = env_cfg or config.DEFAULT.env
    config.ensure_dirs()

    env = PortfolioEnv(train_dataset, env_cfg, mode="train")
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        cfg=agent_cfg,
    )

    log_path = config.ARTIFACTS_LOGS / f"{run_name}_seed{agent_cfg.seed}.jsonl"
    log_file = log_path.open("w", encoding="utf-8") if write_log else None
    ckpt_path = config.ARTIFACTS_MODELS / f"{run_name}_seed{agent_cfg.seed}.pt"

    episodes: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_sharpe": -np.inf, "step": 0}

    window: list[Any] = []          # UpdateStats awaiting aggregation
    ep_reward = 0.0
    ep_actions: list[int] = []
    ep_index = 0
    t0 = time.perf_counter()

    obs, _ = env.reset(seed=agent_cfg.seed)

    for step in range(1, agent_cfg.total_steps + 1):
        eps = epsilon_by_step(step, agent_cfg)
        action = agent.act(obs, epsilon=eps)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # See module docstring: store *termination*, not truncation.
        agent.buffer.add(obs, action, reward, next_obs, terminated)

        ep_reward += reward
        ep_actions.append(action)
        obs = next_obs

        if step % agent_cfg.train_freq == 0:
            stats = agent.update()
            if stats is not None:
                window.append(stats)

        if terminated or truncated:
            ep_index += 1
            wealth = env.wealth
            counts = np.bincount(ep_actions, minlength=agent.n_actions)
            share = counts / counts.sum()
            nz = share[share > 0]
            episodes.append(
                {
                    "episode": ep_index,
                    "step": step,
                    "reward": ep_reward,
                    "final_wealth": wealth,
                    "n_decisions": len(ep_actions),
                    "epsilon": eps,
                    "action_entropy": float(-(nz * np.log(nz)).sum()),
                    "start_date": info["date"],
                }
            )
            ep_reward = 0.0
            ep_actions = []
            obs, _ = env.reset()

        if window and step % log_every == 0:
            row = {
                "step": step,
                "epsilon": eps,
                "buffer": len(agent.buffer),
                "loss": float(np.mean([s.loss for s in window])),
                "q_mean": float(np.mean([s.q_mean for s in window])),
                "q_max": float(np.mean([s.q_max for s in window])),
                "target_mean": float(np.mean([s.target_mean for s in window])),
                "td_error_abs": float(np.mean([s.td_error_abs for s in window])),
                "grad_norm": float(np.mean([s.grad_norm for s in window])),
            }
            updates.append(row)
            if log_file is not None:
                log_file.write(json.dumps({"kind": "update", **row}) + "\n")
            window = []

        if valid_dataset is not None and step % agent_cfg.eval_every == 0:
            ev = evaluate(agent, valid_dataset, env_cfg, risk_free=risk_free)
            row = {
                "step": step,
                "val_sharpe": ev["sharpe"],
                "val_cagr": ev["cagr"],
                "val_max_drawdown": ev["max_drawdown"],
                "val_final_wealth": ev["final_wealth"],
                "val_action_entropy": ev["action_entropy"],
                "val_mean_turnover": ev["mean_turnover"],
                "action_share": ev["action_share"],
            }
            evaluations.append(row)
            if log_file is not None:
                log_file.write(json.dumps({"kind": "eval", **row}) + "\n")

            if ev["sharpe"] > best["val_sharpe"]:
                best = {k: v for k, v in row.items()}
                best["episode"] = ep_index
                if save_checkpoints:
                    agent.save(ckpt_path)
                    best["checkpoint"] = str(ckpt_path)
                # Snapshot the weights in memory too, so a caller that did not
                # ask for checkpoints still gets the *selected* agent back.
                best["_state"] = {
                    k: v.detach().clone() for k, v in agent.online.state_dict().items()
                }

            if progress is not None:
                progress(
                    f"  step {step:>7,} | eps {eps:0.3f} | "
                    f"val Sharpe {ev['sharpe']:+0.3f} | "
                    f"val CAGR {ev['cagr']:+0.2%} | "
                    f"maxDD {ev['max_drawdown']:0.2%} | "
                    f"entropy {ev['action_entropy']:0.2f}"
                )
            if pruning_callback is not None:
                pruning_callback(step, ev["sharpe"])

    wall = time.perf_counter() - t0
    if log_file is not None:
        log_file.close()

    # Restore the selected weights so the returned agent *is* the chosen model.
    state = best.pop("_state", None)
    if state is not None:
        agent.online.load_state_dict(state)
        agent.sync_target()

    if progress is not None and valid_dataset is not None:
        progress(
            f"  selected step {best.get('step', 0):,} "
            f"(val Sharpe {best.get('val_sharpe', float('nan')):.3f}) "
            f"in {wall:.0f}s"
        )

    return TrainResult(
        agent=agent,
        episodes=pd.DataFrame(episodes),
        updates=pd.DataFrame(updates),
        evaluations=pd.DataFrame(evaluations),
        best=best,
        config={
            "run_name": run_name,
            "agent": agent_cfg.to_dict(),
            "env": asdict(env_cfg),
        },
        wall_time=wall,
        log_path=log_path if write_log else None,
        checkpoint_path=ckpt_path if save_checkpoints else None,
    )


# --------------------------------------------------------------------------- #
# Log replay
# --------------------------------------------------------------------------- #
def load_log(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read a JSONL training log back into DataFrames keyed by record kind."""
    rows: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            rows.setdefault(rec.pop("kind"), []).append(rec)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def smooth(series: pd.Series | np.ndarray, window: int = 25) -> np.ndarray:
    """Centred moving average used for the noisy learning curves in NB03."""
    s = pd.Series(np.asarray(series, dtype=float))
    return s.rolling(window, min_periods=max(1, window // 4), center=True).mean().to_numpy()
