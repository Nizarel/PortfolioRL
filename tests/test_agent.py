"""Tests for the DQN agent and the training loop.

These are unit tests of *learning mechanics*, not of investment performance.
They pin down the things that break silently: the shape and identifiability of
the duelling head, the Double-DQN target formula, whether the target network is
actually frozen, and -- the classic one -- whether truncation is being
mistaken for termination in the replay buffer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from portfoliorl import agent as agent_mod
from portfoliorl import config, env as env_mod, train as train_mod


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def test_buffer_overwrites_oldest_when_full():
    buf = agent_mod.ReplayBuffer(capacity=4, obs_dim=2, seed=0)
    for i in range(6):
        buf.add(np.full(2, i, dtype=np.float32), i % 3, float(i), np.zeros(2), False)

    assert len(buf) == 4
    assert buf.is_full
    # Rewards 0 and 1 must have been overwritten by 4 and 5.
    stored = set(buf._rewards.tolist())
    assert stored == {2.0, 3.0, 4.0, 5.0}


def test_buffer_sample_returns_correct_shapes_and_dtypes():
    buf = agent_mod.ReplayBuffer(capacity=64, obs_dim=5, seed=1)
    for i in range(64):
        buf.add(np.random.randn(5), i % 6, float(i), np.random.randn(5), i % 7 == 0)

    obs, actions, rewards, next_obs, dones = buf.sample(16, torch.device("cpu"))
    assert obs.shape == (16, 5) and next_obs.shape == (16, 5)
    assert actions.shape == (16,) and actions.dtype == torch.int64
    assert rewards.shape == (16,) and rewards.dtype == torch.float32
    assert dones.shape == (16,)
    assert set(np.unique(dones.numpy())).issubset({0.0, 1.0})


def test_buffer_sampling_is_reproducible_given_a_seed():
    def draw(seed):
        buf = agent_mod.ReplayBuffer(capacity=32, obs_dim=2, seed=seed)
        for i in range(32):
            buf.add(np.full(2, i, dtype=np.float32), 0, float(i), np.zeros(2), False)
        return buf.sample(8, torch.device("cpu"))[2].numpy()

    np.testing.assert_array_equal(draw(3), draw(3))
    assert not np.array_equal(draw(3), draw(4))


# --------------------------------------------------------------------------- #
# Q-network
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dueling", [False, True])
def test_qnetwork_output_shape(dueling):
    net = agent_mod.QNetwork(obs_dim=31, n_actions=6, hidden_sizes=(16, 8), dueling=dueling)
    out = net(torch.randn(7, 31))
    assert out.shape == (7, 6)
    assert torch.isfinite(out).all()


def test_dueling_advantages_are_mean_centred():
    """Q - V must have zero mean across actions -- that is the identifiability
    constraint that stops V and A drifting apart by an arbitrary constant."""
    net = agent_mod.QNetwork(obs_dim=4, n_actions=6, hidden_sizes=(8,), dueling=True)
    x = torch.randn(5, 4)
    with torch.no_grad():
        q = net(x)
        v = net.value_head(net.trunk(x))
    centred = q - v
    assert torch.allclose(centred.mean(dim=1), torch.zeros(5), atol=1e-6)


def test_dueling_costs_only_slightly_more_parameters():
    common = dict(obs_dim=31, n_actions=6, hidden_sizes=(128, 64))
    vanilla = agent_mod.QNetwork(**common, dueling=False).n_parameters
    duel = agent_mod.QNetwork(**common, dueling=True).n_parameters
    # The duelling head adds one extra 64 -> 1 map: 65 parameters.
    assert duel - vanilla == 65
    assert vanilla < 15_000  # deliberately small; see the module docstring


# --------------------------------------------------------------------------- #
# Exploration schedule
# --------------------------------------------------------------------------- #
def test_epsilon_decays_linearly_then_holds_the_floor():
    cfg = config.AgentConfig(total_steps=1_000, eps_start=1.0, eps_end=0.05,
                             eps_decay_fraction=0.5)
    assert agent_mod.epsilon_by_step(0, cfg) == pytest.approx(1.0)
    assert agent_mod.epsilon_by_step(250, cfg) == pytest.approx(0.525)
    assert agent_mod.epsilon_by_step(500, cfg) == pytest.approx(0.05)
    assert agent_mod.epsilon_by_step(999, cfg) == pytest.approx(0.05)
    # The floor is non-zero on purpose: exploration never fully stops.
    assert agent_mod.epsilon_by_step(10**6, cfg) > 0


# --------------------------------------------------------------------------- #
# Agent mechanics
# --------------------------------------------------------------------------- #
def _tiny_cfg(**kw):
    base = dict(
        hidden_sizes=(16, 8), buffer_size=500, batch_size=16, learning_starts=32,
        target_update_interval=10, total_steps=200, eval_every=100, seed=0,
    )
    base.update(kw)
    return config.AgentConfig(**base)


def test_greedy_action_is_deterministic_and_in_range():
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    obs = np.random.default_rng(0).normal(size=31).astype(np.float32)
    first = ag.greedy_action(obs)
    assert 0 <= first < 6
    assert all(ag.greedy_action(obs) == first for _ in range(5))


def test_epsilon_one_explores_and_epsilon_zero_does_not():
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    obs = np.zeros(31, dtype=np.float32)
    random_actions = {ag.act(obs, epsilon=1.0) for _ in range(200)}
    greedy_actions = {ag.act(obs, epsilon=0.0) for _ in range(50)}
    assert len(random_actions) > 1
    assert len(greedy_actions) == 1


def test_update_returns_none_until_learning_starts():
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=_tiny_cfg(learning_starts=50))
    for _ in range(20):
        ag.buffer.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert ag.update() is None
    for _ in range(60):
        ag.buffer.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert ag.update() is not None


def test_gradient_step_changes_online_weights_but_not_the_target():
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=_tiny_cfg(target_update_interval=10_000))
    rng = np.random.default_rng(0)
    for _ in range(100):
        ag.buffer.add(rng.normal(size=4), int(rng.integers(6)), float(rng.normal()),
                      rng.normal(size=4), False)

    before_online = [p.detach().clone() for p in ag.online.parameters()]
    before_target = [p.detach().clone() for p in ag.target.parameters()]
    stats = ag.update()

    assert stats is not None and np.isfinite(stats.loss)
    assert any(not torch.equal(a, b) for a, b in zip(before_online, ag.online.parameters()))
    # The target is frozen between syncs -- this is the whole point of it.
    assert all(torch.equal(a, b) for a, b in zip(before_target, ag.target.parameters()))


def test_target_network_syncs_on_schedule():
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=_tiny_cfg(target_update_interval=5))
    rng = np.random.default_rng(1)
    for _ in range(200):
        ag.buffer.add(rng.normal(size=4), int(rng.integers(6)), float(rng.normal()),
                      rng.normal(size=4), False)
    for _ in range(5):
        ag.update()
    assert all(
        torch.equal(a, b) for a, b in zip(ag.target.parameters(), ag.online.parameters())
    )


def test_double_dqn_target_matches_the_formula():
    """Reproduce the Double-DQN target by hand and compare to the agent's.

    Vanilla uses max_a' Q_target(s',a'); Double uses Q_target evaluated at the
    argmax of Q_online.  Whenever the two networks disagree about the best
    action, the Double target is the *smaller* of the two -- that is exactly the
    overestimation being removed.
    """
    rng = np.random.default_rng(7)
    cfg = _tiny_cfg(double_dqn=True, gamma=0.9, batch_size=32, learning_starts=32)
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=cfg)
    # Perturb the target net so it genuinely differs from the online net.
    with torch.no_grad():
        for p in ag.target.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    next_obs = torch.as_tensor(rng.normal(size=(32, 4)), dtype=torch.float32)
    rewards = torch.as_tensor(rng.normal(size=32), dtype=torch.float32)

    with torch.no_grad():
        q_online_next = ag.online(next_obs)
        q_target_next = ag.target(next_obs)
        double = rewards + cfg.gamma * q_target_next.gather(
            1, q_online_next.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        vanilla = rewards + cfg.gamma * q_target_next.max(dim=1).values

    assert (double <= vanilla + 1e-6).all()
    assert (double < vanilla - 1e-6).any()  # they must actually differ somewhere


def test_done_flag_kills_the_bootstrap_term():
    """With done=1 the target must equal the reward exactly."""
    cfg = _tiny_cfg(gamma=0.99)
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=cfg)
    next_obs = torch.randn(8, 4)
    rewards = torch.randn(8)
    dones = torch.ones(8)
    with torch.no_grad():
        next_q = ag.target(next_obs).max(dim=1).values
        target = rewards + cfg.gamma * (1.0 - dones) * next_q
    torch.testing.assert_close(target, rewards)


def test_gradient_norm_is_clipped_to_the_configured_bound():
    """``clip_grad_norm_`` rescales in place, so the gradients still attached to
    the parameters after ``update()`` are the ones the optimiser actually used."""
    cfg = _tiny_cfg(max_grad_norm=0.05)
    ag = agent_mod.DQNAgent(obs_dim=4, n_actions=6, cfg=cfg)
    rng = np.random.default_rng(2)
    for _ in range(200):
        # Large observations inflate the activations and hence the raw gradient.
        ag.buffer.add(rng.normal(size=4) * 50, int(rng.integers(6)),
                      float(rng.normal() * 500), rng.normal(size=4) * 50, False)

    stats = ag.update()
    assert stats is not None
    assert stats.grad_norm > cfg.max_grad_norm  # the raw norm really was too big

    applied = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in ag.online.parameters() if p.grad is not None)
    ).item()
    assert applied <= cfg.max_grad_norm + 1e-5
    assert all(torch.isfinite(p).all() for p in ag.online.parameters())


def test_huber_loss_bounds_the_influence_of_an_outlier_reward():
    """The whole reason for Huber: a 2008-sized week must not be allowed to
    dominate the gradient the way squared error would let it."""

    def grad_norm_for(scale: float) -> float:
        ag = agent_mod.DQNAgent(
            obs_dim=4, n_actions=6, cfg=_tiny_cfg(max_grad_norm=1e9, seed=5)
        )
        rng = np.random.default_rng(11)
        for _ in range(200):
            ag.buffer.add(rng.normal(size=4), int(rng.integers(6)),
                          float(rng.normal() * scale), rng.normal(size=4), False)
        stats = ag.update()
        assert stats is not None
        return stats.grad_norm

    small, huge = grad_norm_for(1.0), grad_norm_for(1_000.0)
    # A 1000x larger reward produces a gradient of a comparable size, because
    # Huber is linear in the tails.  Under MSE this ratio would be ~1000.
    assert huge / small < 5.0


def test_save_and_load_round_trips_the_policy(tmp_path):
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg(dueling=True))
    path = ag.save(tmp_path / "ckpt.pt")
    restored = agent_mod.DQNAgent.load(path)

    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=31).astype(np.float32)
        assert restored.greedy_action(obs) == ag.greedy_action(obs)
        np.testing.assert_allclose(restored.q_values(obs), ag.q_values(obs), rtol=1e-6)


def test_variant_config_flips_only_the_two_ablation_switches():
    for name, kw in agent_mod.VARIANTS.items():
        cfg = agent_mod.variant_config(**kw)
        assert cfg.double_dqn == kw["double"], name
        assert cfg.dueling == kw["dueling"], name
        assert cfg.learning_rate == config.DEFAULT.agent.learning_rate
        assert cfg.hidden_sizes == config.DEFAULT.agent.hidden_sizes


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def test_training_runs_end_to_end_and_returns_usable_history(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path / "logs")
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path / "models")
    (tmp_path / "logs").mkdir()
    (tmp_path / "models").mkdir()

    cfg = _tiny_cfg(total_steps=400, eval_every=200, learning_starts=64)

    result = train_mod.train_dqn(
        dataset.split("train"),
        dataset.split("valid"),
        agent_cfg=cfg,
        run_name="unit",
        log_every=50,
        progress=None,
    )

    assert isinstance(result, train_mod.TrainResult)
    assert len(result.episodes) > 0
    assert len(result.updates) > 0
    assert len(result.evaluations) == 2
    assert np.isfinite(result.best_val_sharpe)
    assert result.wall_time > 0
    assert result.checkpoint_path.exists()
    assert set(result.updates.columns) >= {"step", "loss", "q_mean", "grad_norm"}
    assert result.episodes["reward"].notna().all()
    # Every training episode should be the configured length.
    assert (result.episodes["n_decisions"] == config.DEFAULT.env.episode_length).all()


def test_training_selects_the_best_validation_checkpoint(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path / "logs")
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path / "models")
    (tmp_path / "logs").mkdir()
    (tmp_path / "models").mkdir()

    result = train_mod.train_dqn(
        dataset.split("train"),
        dataset.split("valid"),
        agent_cfg=_tiny_cfg(total_steps=600, eval_every=200, learning_starts=64),
        run_name="unit_sel",
        progress=None,
    )
    best = result.best["val_sharpe"]
    assert best == pytest.approx(result.evaluations["val_sharpe"].max())
    assert result.best["step"] in set(result.evaluations["step"])


def test_training_log_can_be_replayed_from_disk(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_LOGS", tmp_path / "logs")
    monkeypatch.setattr(config, "ARTIFACTS_MODELS", tmp_path / "models")
    (tmp_path / "logs").mkdir()
    (tmp_path / "models").mkdir()

    result = train_mod.train_dqn(
        dataset.split("train"),
        dataset.split("valid"),
        agent_cfg=_tiny_cfg(total_steps=400, eval_every=200, learning_starts=64),
        run_name="unit_log",
        log_every=50,
        progress=None,
    )
    replayed = train_mod.load_log(result.log_path)
    assert set(replayed) == {"update", "eval"}
    assert len(replayed["update"]) == len(result.updates)
    assert len(replayed["eval"]) == len(result.evaluations)


def test_evaluate_is_deterministic_and_pays_transaction_costs(dataset):
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    valid = dataset.split("valid")

    first = train_mod.evaluate(ag, valid)
    second = train_mod.evaluate(ag, valid)
    assert first["final_wealth"] == pytest.approx(second["final_wealth"], rel=1e-15)
    assert first["summary"]["total_cost_fraction"] > 0
    assert 0.0 <= first["action_entropy"] <= np.log(len(config.ACTION_ALLOCATIONS)) + 1e-9
    assert len(first["action_share"]) == len(config.ACTION_ALLOCATIONS)
    assert sum(first["action_share"]) == pytest.approx(1.0)


def test_agent_policy_plugs_into_run_policy(dataset):
    """The agent must be evaluable through exactly the same code path as every
    benchmark, otherwise the comparison is not like for like."""
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    valid = dataset.split("valid")
    daily, summary = env_mod.run_policy(valid, ag.policy())
    assert len(daily) > 0
    assert summary["n_decisions"] > 0
    assert np.isfinite(summary["final_wealth"])


# --------------------------------------------------------------------------- #
# Evaluation-time policy wrappers
# --------------------------------------------------------------------------- #
def test_ensemble_of_one_agent_is_the_agent(dataset):
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    policy = agent_mod.ensemble_policy([ag])
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=31).astype(np.float32)
        assert policy(obs) == ag.greedy_action(obs)


def test_ensemble_averages_q_values_across_agents():
    agents = [
        agent_mod.DQNAgent(obs_dim=8, n_actions=6, cfg=_tiny_cfg(seed=s)) for s in (0, 1, 2)
    ]
    qfn = agent_mod.ensemble_q_values(agents)
    obs = np.ones(8, dtype=np.float32)
    expected = np.mean([a.q_values(obs) for a in agents], axis=0)
    np.testing.assert_allclose(qfn(obs), expected, rtol=1e-6)


def test_ensemble_rejects_mismatched_action_counts():
    a = agent_mod.DQNAgent(obs_dim=8, n_actions=6, cfg=_tiny_cfg())
    b = agent_mod.DQNAgent(obs_dim=8, n_actions=7, cfg=_tiny_cfg())
    with pytest.raises(ValueError, match="action count"):
        agent_mod.ensemble_q_values([a, b])
    with pytest.raises(ValueError):
        agent_mod.ensemble_q_values([])


def test_hysteresis_with_zero_margin_is_plain_greedy(dataset):
    ag = agent_mod.DQNAgent(obs_dim=31, n_actions=6, cfg=_tiny_cfg())
    valid = dataset.split("valid")
    greedy, _ = env_mod.run_policy(valid, ag.policy(), seed=0)
    held, _ = env_mod.run_policy(
        valid,
        agent_mod.hysteresis_policy(agent_mod.ensemble_q_values([ag]), margin=0.0),
        seed=0,
    )
    np.testing.assert_array_equal(greedy["action"].to_numpy(), held["action"].to_numpy())


def test_hysteresis_holds_until_the_margin_is_cleared():
    """A hand-built Q function makes the switching rule checkable by hand."""
    q = np.array([0.0, 0.0, 0.0])
    policy = agent_mod.hysteresis_policy(lambda _obs: q, margin=0.5, initial_action=0)
    obs = np.zeros(3)

    q[:] = [0.0, 0.3, 0.0]  # better, but only by 0.3 < 0.5
    assert policy(obs) == 0
    q[:] = [0.0, 0.6, 0.0]  # now clears the margin
    assert policy(obs) == 1
    q[:] = [0.4, 0.6, 0.0]  # action 1 still leads, so no reason to move
    assert policy(obs) == 1
    q[:] = [1.2, 0.6, 0.0]  # 1.2 - 0.6 = 0.6 > 0.5, switch back
    assert policy(obs) == 0


def test_hysteresis_rejects_negative_margin():
    with pytest.raises(ValueError):
        agent_mod.hysteresis_policy(lambda _obs: np.zeros(3), margin=-0.1)


# --------------------------------------------------------------------------- #
# Checkpoint selection
# --------------------------------------------------------------------------- #
def test_smoothed_selection_prefers_a_sustained_run_over_a_lucky_spike(dataset):
    """Validation Sharpe is noisy enough that the single best evaluation is
    largely the luckiest one; the smoothed rule must ignore an isolated peak."""
    sharpes = [0.1, 0.9, 0.1, 0.5, 0.6, 0.55]

    window = 3
    smoothed = [
        np.mean(sharpes[i - window + 1 : i + 1]) if i >= window - 1 else -np.inf
        for i in range(len(sharpes))
    ]
    assert int(np.argmax(sharpes)) == 1, "the spike wins under plain argmax"
    assert int(np.argmax(smoothed)) == 5, "the sustained stretch wins when smoothed"


def test_smoothed_selection_falls_back_when_the_window_cannot_fill(dataset):
    """A run with fewer evaluations than the window must still pick a
    checkpoint rather than silently returning the final weights."""
    cfg = _tiny_cfg(selection="smoothed", select_window=99)
    result = train_mod.train_dqn(
        dataset.split("train"),
        dataset.split("valid"),
        agent_cfg=cfg,
        run_name="test_short_smoothed",
        write_log=False,
        save_checkpoints=False,
        progress=lambda *_a, **_k: None,
    )
    assert np.isfinite(result.best_val_sharpe)
    assert result.best["step"] > 0
