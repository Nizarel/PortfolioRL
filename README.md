# PortfolioRL: Reinforcement Learning for Dynamic Portfolio Rebalancing

Reinforcement learning agent that learns to dynamically rebalance a multi-asset portfolio, benchmarked against traditional rule-based allocation strategies.

## Overview

Traditional portfolio strategies — fixed 60/40 allocation, equal-weight, monthly rebalancing — are static and don't adapt to changing market regimes (volatility spikes, drawdowns, rate changes, growth-vs-defensive rotations). PortfolioRL trains a Deep Q-Network (DQN) agent to observe market signals and choose a portfolio allocation at each rebalancing step, with the objective of maximizing risk-adjusted returns while controlling transaction costs and downside risk.

This is a research and backtesting project. **It is not investment advice and does not guarantee future performance.**

## Problem Formulation

The task is modeled as a Markov Decision Process (S, A, P, R, γ):

| Component | Description |
|---|---|
| **State** | Recent returns, rolling volatility, momentum, moving-average ratios, current portfolio weights, current drawdown |
| **Action** | One of 6 discrete portfolio allocations (defensive → equity-heavy) |
| **Reward** | Portfolio return, penalized for turnover, volatility, and drawdown |
| **Environment** | Historical market simulation built from daily price data |

**Asset universe:** SPY (equities), TLT (long bonds), GLD (gold), SHY/BIL (cash proxy)

**Benchmarks:** buy-and-hold, equal-weight, fixed 60/40, calendar (monthly) rebalancing

## Approach

- **Algorithm:** Deep Q-Network (DQN), with Double DQN considered as a refinement to reduce overestimation bias
- **Stabilization:** experience replay buffer + periodically-updated target network
- **Exploration:** ε-greedy, decaying from ~1.0 to a floor of ~0.05
- **Data:** daily adjusted close prices via `yfinance`, split chronologically (train / validation / test — no shuffling, to avoid look-ahead bias)
- **Tuning:** coarse search → refined random/Bayesian search (e.g. Optuna), selected by validation Sharpe ratio with max drawdown as a tie-breaker

## Evaluation

Performance is judged on a balanced scorecard rather than raw return, including:

- **Return:** cumulative return, CAGR
- **Risk:** annualized volatility, max drawdown, downside deviation
- **Risk-adjusted:** Sharpe, Sortino, Calmar ratios
- **Trading:** turnover, transaction-cost drag
- **RL diagnostics:** episodic reward, TD loss, Q-value stability, exploration behavior, seed variance
- **Relative:** excess return vs. each benchmark, information ratio

Final model selection favors the strongest overall balance of these metrics — not the single highest return — evaluated once on a held-out test window using multiple random seeds.

## Project Status

This repository currently contains the project design document (Assignment 3): problem formulation, MDP/DQN mathematical foundations, algorithm justification, dataset plan, hyperparameter tuning plan, and the full performance evaluation framework.

- [x] Business problem & research goal
- [x] RL problem formulation (agent, environment, state, action, reward)
- [x] Mathematical foundations & algorithm selection (DQN)
- [x] Hyperparameter tuning plan
- [x] Performance evaluation metrics & protocol
- [ ] Environment implementation
- [ ] Agent training
- [ ] Benchmark comparison & results
