"""Central configuration for the PortfolioRL project.

Every tunable constant lives here so that a run is fully described by a small
number of frozen dataclasses.  This makes experiments reproducible and lets the
notebooks report exactly which configuration produced a given artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS_MODELS = ARTIFACTS / "models"
ARTIFACTS_RESULTS = ARTIFACTS / "results"
ARTIFACTS_FIGURES = ARTIFACTS / "figures"
ARTIFACTS_LOGS = ARTIFACTS / "logs"

TRADING_DAYS_PER_YEAR = 252

#: Resolution used for every exported figure.  150 dpi is large enough to stay
#: crisp in the written report and in 1080p video slides without bloating the
#: repository.
FIGURE_DPI = 150


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline."""
    for d in (
        DATA_RAW,
        DATA_PROCESSED,
        ARTIFACTS,
        ARTIFACTS_MODELS,
        ARTIFACTS_RESULTS,
        ARTIFACTS_FIGURES,
        ARTIFACTS_LOGS,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    """Asset universe, sample period and chronological splits.

    The start date is driven by GLD's inception (2004-11-18); SPY, TLT and SHY
    all pre-date it.  Starting here puts the 2008 global financial crisis inside
    the training window, which matters because a policy that has never seen a
    bear market cannot be expected to behave sensibly in one.
    """

    tickers: tuple[str, ...] = ("SPY", "TLT", "GLD", "SHY")
    risk_free_ticker: str = "^IRX"  # 13-week T-bill discount rate (annualised, %)

    start: str = "2004-11-01"
    end: str = "2025-12-31"

    train_start: str = "2004-11-18"
    train_end: str = "2017-12-31"
    valid_start: str = "2018-01-01"
    valid_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2025-12-31"

    # Feature look-back windows (trading days).
    vol_window: int = 20
    vol_window_long: int = 60
    momentum_window: int = 63
    ma_fast: int = 50
    ma_slow: int = 200

    @property
    def warmup_days(self) -> int:
        """Longest look-back; rows before this are dropped as feature warm-up."""
        return max(
            self.vol_window,
            self.vol_window_long,
            self.momentum_window,
            self.ma_slow,
        )

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: Discrete allocation menu.  Rows must sum to 1 and align with DataConfig.tickers
#: order (SPY, TLT, GLD, SHY).  Rows 0-5 mirror Table 4 of the project proposal.
#:
#: Row 6 is a documented departure from the proposal.  The original menu topped
#: out at 80/20, so 100% SPY -- the single best-performing benchmark over the
#: test window -- was not reachable by *any* policy.  The agent was being asked
#: to beat an allocation it was structurally forbidden from holding.  New rows
#: are appended, never inserted, so the integer indices used by
#: ``benchmarks.STATIC_BENCHMARKS`` stay valid.
ACTION_ALLOCATIONS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 0.00, 1.00),  # 0 defensive / cash-like
    (0.60, 0.40, 0.00, 0.00),  # 1 traditional balanced 60/40
    (0.40, 0.40, 0.20, 0.00),  # 2 diversified balanced
    (0.80, 0.20, 0.00, 0.00),  # 3 equity-heavy
    (0.25, 0.25, 0.25, 0.25),  # 4 equal weight
    (0.20, 0.60, 0.20, 0.00),  # 5 defensive bond-heavy
    (1.00, 0.00, 0.00, 0.00),  # 6 full equity
)

ACTION_LABELS: tuple[str, ...] = (
    "Cash (100% SHY)",
    "Balanced 60/40",
    "Diversified 40/40/20",
    "Equity-heavy 80/20",
    "Equal weight",
    "Bond-heavy 20/60/20",
    "Full equity (100% SPY)",
)


@dataclass(frozen=True)
class EnvConfig:
    """Market-simulation and reward parameters.

    Design decisions that differ from the original proposal, and why:

    * ``steps_per_decision = 5`` -- the agent rebalances weekly, but portfolio
      value and every reported metric are still computed from *daily* returns.
      Daily rebalancing is dominated by transaction-cost drag and, at
      ``gamma = 0.99``, gives an effective horizon of only ~100 trading days,
      which is far too short for a drawdown-aware objective.
    * ``use_log_return = True`` -- log returns are time-additive, so the
      undiscounted sum of rewards equals terminal log wealth.  The RL objective
      then *is* the business objective rather than a proxy for it.
    * ``lambda_drawdown`` multiplies the *increment* in drawdown, not its level.
      Penalising the level charges the agent repeatedly for a single loss and
      teaches it to freeze instead of recover.
    * ``reward_scale = 100`` -- raw weekly log returns are O(1e-3), which drives
      the TD loss to O(1e-6) and effectively zeroes the gradient signal.
      Rewards are expressed in percentage points instead.
    """

    steps_per_decision: int = 5  # trading days per rebalancing decision
    initial_value: float = 100_000.0

    transaction_cost_bps: float = 5.0  # one-way cost, basis points of traded notional

    #: Index into ACTION_ALLOCATIONS describing the holding at episode start.
    #: 0 = 100% SHY.  Starting in cash means every strategy, including the
    #: benchmarks, pays an identical entry cost to establish its position -- so
    #: no strategy gets a free ride into the market.
    initial_action: int = 0

    #: Truncate the allocation menu to its first ``n_actions`` rows.  The full
    #: equity row was *appended* at index 6, so ``n_actions = 6`` reproduces the
    #: proposal's Table 4 menu exactly -- which is what makes "does the extra
    #: action help?" an answerable question rather than a claim.
    #: ``None`` uses the whole menu.
    n_actions: int | None = None

    use_log_return: bool = True
    reward_scale: float = 100.0

    # ----------------------------------------------------------------- #
    # Reward penalty weights
    # ----------------------------------------------------------------- #
    # These are only meaningful relative to the size of the term they compete
    # with.  Over one weekly decision the log-return term is O(1.5e-3)
    # (~8%/yr / 52), so each penalty is calibrated to be of comparable order.
    # Getting this wrong is not a matter of taste: a penalty 100x the return
    # term produces an agent that sits in cash forever and reports a flat
    # equity curve.  Optuna searches all three in notebook 04.
    #
    #: Applied to turnover (= 0.5 * sum |dw|).  The *explicit* trading cost is
    #: already deducted from wealth; this is an additional behavioural aversion
    #: covering slippage and market impact.  0.002 charges an extra 20 bps per
    #: unit of turnover, i.e. roughly 4x the modelled commission.
    lambda_turnover: float = 0.002
    #: Applied to the 20-day rolling standard deviation of *daily* portfolio
    #: returns (NOT annualised).  Daily sigma is O(1e-2), so 0.10 contributes
    #: O(1e-3) -- the same order as the return term.  Annualising first would
    #: make this term ~16x larger and dominate the objective.
    lambda_volatility: float = 0.10
    #: Applied to the *increment* in drawdown over the decision period,
    #: max(0, DD_t - DD_{t-1}).  Penalising the drawdown *level* would charge
    #: the agent repeatedly for a single loss and teach it to freeze rather
    #: than recover.  At 0.50 a bad week hurts 1.5x as much as its raw loss,
    #: which encodes loss aversion without paralysing the policy.
    lambda_drawdown: float = 0.50

    #: Applied to a plain 0/1 indicator that the allocation *changed* this
    #: decision, independent of how large the change was.  ``lambda_turnover``
    #: is proportional to trade size, so it barely discourages the many tiny
    #: reallocations that generate most of the agent's ~9x annual turnover.
    #: A fixed charge per switch is what makes the policy commit to a holding.
    #:
    #: Calibration: this multiplies an indicator of magnitude 1, not a quantity
    #: of order 1e-2 like the other three penalties, so it needs a weight two
    #: orders of magnitude smaller.  The weekly log-return term is O(1.5e-3),
    #: so 0.001 charges roughly two-thirds of a typical week's return per
    #: switch.  Values like 0.5 -- reasonable for ``lambda_drawdown`` -- would
    #: be ~300x the return term and freeze the policy outright.
    #: Default 0.0 keeps the proposal's reward exactly reproducible.
    lambda_switch: float = 0.0

    #: Look-back for the portfolio-volatility state feature and reward penalty.
    portfolio_vol_window: int = 20

    # Episode construction (training only).  Random-start sub-episodes turn one
    # historical price path into many overlapping trajectories; without this the
    # agent simply memorises the single realised path.
    random_start_episodes: bool = True
    episode_length: int = 52  # decisions per episode (~1 year at weekly steps)

    #: Fraction of training episodes forced to start in a *stress* regime
    #: (equity momentum negative and the stock/bond correlation unusually high
    #: -- the 2022 configuration, where bonds stop hedging equities).
    #:
    #: Measured on the 2004-2017 training split, the 60-day SPY/TLT correlation
    #: is negative on 92% of days (median -0.46).  A regime where equities fall
    #: while that correlation turns *positive* covers 28 days, 1% of the split.
    #: Uniform sampling therefore almost never shows the agent the regime that
    #: produced its worst walk-forward year.  0.0 restores uniform sampling.
    stress_sampling_fraction: float = 0.0

    #: Quantile of the in-split correlation distribution above which a day
    #: counts as stressed.  An absolute "correlation > 0" threshold is the
    #: cleaner story but selects too few days to train on, so the threshold is
    #: defined relative to what the split actually contains: 0.70 keeps the
    #: most bond-unhelpful third of equity-drawdown days (92 days in train).
    stress_corr_quantile: float = 0.70

    #: Append the previously held action to the observation.  Without it the
    #: state is not Markov once a switching penalty exists: the agent cannot
    #: tell whether keeping its allocation is free or costly.
    include_prev_action: bool = False


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentConfig:
    """DQN hyperparameters.  Defaults are the coarse-search starting point."""

    double_dqn: bool = True
    dueling: bool = True

    hidden_sizes: tuple[int, ...] = (128, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99

    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 1  # gradient steps per environment step
    target_update_interval: int = 500
    tau: float = 1.0  # 1.0 = hard target update

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.30  # fraction of total steps spent decaying

    max_grad_norm: float = 10.0
    huber_delta: float = 1.0

    total_steps: int = 120_000
    eval_every: int = 5_000

    #: Checkpoint selection rule. ``"best"`` takes the single highest validation
    #: Sharpe; ``"smoothed"`` takes the highest rolling mean over
    #: ``select_window`` consecutive evaluations. With ~24 evaluations per run,
    #: picking the single best is largely picking the luckiest -- notebook 05
    #: measures a validation-to-test rank correlation of -0.365.
    selection: str = "best"
    select_window: int = 3

    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


DEFAULT = ProjectConfig()
