"""Deep Q-Network agent: replay buffer, Q-network, and the learning update.

Why a hand-written DQN rather than Stable-Baselines3
----------------------------------------------------
SB3's ``DQN`` is deliberately *vanilla*: it implements neither Double Q-learning
nor the duelling architecture, and its documentation states that these
extensions are out of scope.  The ablation this project promises (vanilla vs.
Double vs. Duelling vs. both) is therefore impossible to run inside SB3 without
subclassing so much of it that little of the library remains.  Writing ~300
lines of PyTorch instead makes every ablation a single boolean flag and, more
importantly, makes the *reason* each extension helps visible in the code.

The three ideas, and what each fixes
------------------------------------
**1. Experience replay** (Mnih et al., 2015).  Consecutive market observations
are enormously correlated -- Monday's features are nearly Tuesday's -- and
training a network on a correlated stream makes the gradient estimates
statistically dependent and the optimisation unstable.  Replay stores
transitions and samples them uniformly at random, which decorrelates the batch
and lets each transition be reused many times (important here: the whole
training split contains only ~620 decisions).

**2. A target network** (Mnih et al., 2015).  The regression target
:math:`r + \\gamma \\max_{a'} Q(s', a')` is computed from the very network being
updated, so every gradient step moves the target as well.  Freezing a copy of
the weights for ``target_update_interval`` steps turns a moving-target problem
into a sequence of ordinary supervised regressions.

**3. Double Q-learning** (van Hasselt, Guez & Silver, 2016).  The
:math:`\\max` operator uses the same values both to *select* and to *evaluate*
the greedy action, so any positive noise in an action's estimate is
preferentially selected -- a systematic overestimation bias.  Double DQN
decouples them: the **online** network picks the argmax, the **target** network
scores it.

.. math::
   y^{Double}_t = r_t + \\gamma\\, Q_{\\theta^-}\\!\\big(s_{t+1},\\;
       \\arg\\max_{a'} Q_{\\theta}(s_{t+1}, a')\\big)

Overestimation is not a cosmetic concern in finance: an agent that
systematically overvalues the equity-heavy action will hold it through
drawdowns.

**4. Duelling architecture** (Wang et al., 2016).  The trunk splits into a
state-value stream :math:`V(s)` and an advantage stream :math:`A(s,a)`,
recombined as

.. math::
   Q(s,a) = V(s) + \\Big(A(s,a) - \\tfrac{1}{|\\mathcal{A}|}
            \\sum_{a'} A(s,a')\\Big)

The mean-subtraction is what makes the decomposition identifiable (otherwise
any constant could be shifted between the streams).  This matters here because
of the structure of the problem: in most weeks *every* allocation earns roughly
the market's return, so the six action-values are dominated by a common term
that depends only on the state.  Duelling learns that common term once, from
every transition, instead of six times independently.

Loss
----
Huber (smooth L1) rather than MSE.  Weekly rewards are heavy-tailed -- a single
2008 week is many standard deviations from typical -- and squared error would
let those weeks dominate the gradient.  Huber is quadratic near zero (so it
keeps MSE's well-behaved gradients for the bulk of the data) and linear in the
tails (so an outlier contributes a bounded gradient).

References
----------
Mnih et al. (2015), "Human-level control through deep reinforcement learning",
*Nature* 518.  van Hasselt, Guez & Silver (2016), "Deep Reinforcement Learning
with Double Q-learning", *AAAI*.  Wang et al. (2016), "Duelling Network
Architectures for Deep Reinforcement Learning", *ICML*.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions, stored as NumPy arrays.

    Storing ``float32`` NumPy arrays and converting to tensors only at sample
    time uses roughly a third of the memory of keeping tensors, and the
    conversion cost is negligible next to the forward/backward pass.

    Parameters
    ----------
    capacity
        Maximum number of transitions retained.  Once full, the oldest
        transition is overwritten.
    obs_dim
        Length of the observation vector.
    seed
        Seed for the sampling RNG.  Kept separate from the environment and
        exploration RNGs so that changing one does not silently change another.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest if at capacity.

        ``done`` must be the **termination** flag, never truncation.  Our
        episodes end because the sub-episode budget ran out, not because the
        world ended: the market keeps trading on Monday.  Bootstrapping is
        therefore still valid at the end of an episode, and passing
        ``done=True`` there would teach the agent that wealth stops mattering
        every 52 weeks.  See ``train.py`` for where this distinction is applied.
        """
        i = self._pos
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Draw a uniform random batch and return it as tensors on ``device``."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731
        return (
            to(self._obs),
            to(self._actions),
            to(self._rewards),
            to(self._next_obs),
            to(self._dones),
        )


# --------------------------------------------------------------------------- #
# Q-network
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """MLP Q-network with an optional duelling head.

    The trunk is deliberately small (default 31 -> 128 -> 64, ~12.5k
    parameters).  With roughly 620 weekly decisions in the training split, a
    large network would memorise the 2008 crisis rather than learn a rule about
    volatility, and the whole project would become an exercise in overfitting a
    single price path.  Small networks are also why this project runs on a CPU
    in minutes.

    Parameters
    ----------
    obs_dim, n_actions
        Input and output widths.
    hidden_sizes
        Widths of the shared trunk.
    dueling
        If ``True``, split into value and advantage streams and recombine with
        mean-subtracted advantages.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.dueling = dueling

        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)

        if dueling:
            # Both streams are single linear maps off the shared trunk: the
            # representation is shared, only the read-out differs.
            self.value_head = nn.Linear(last, 1)
            self.advantage_head = nn.Linear(last, n_actions)
        else:
            self.q_head = nn.Linear(last, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        if not self.dueling:
            return self.q_head(z)
        value = self.value_head(z)                       # (B, 1)
        advantage = self.advantage_head(z)               # (B, A)
        # Subtracting the mean advantage fixes the additive degeneracy between
        # V and A.  Using the mean rather than the max (also valid) is the
        # choice made in Wang et al. (2016) because it is more stable in
        # practice: the max makes the target depend on a single noisy unit.
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Exploration schedule
# --------------------------------------------------------------------------- #
def epsilon_by_step(step: int, cfg: config.AgentConfig) -> float:
    """Linearly annealed epsilon-greedy exploration rate.

    Epsilon falls from ``eps_start`` to ``eps_end`` over the first
    ``eps_decay_fraction`` of training and is then held flat.  The floor is
    kept non-zero on purpose: with a single historical price path, a fully
    greedy agent stops visiting the actions it has decided are bad and can
    never discover that they are good in a regime it has not yet seen.
    """
    decay_steps = max(1, int(cfg.eps_decay_fraction * cfg.total_steps))
    frac = min(1.0, step / decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class UpdateStats:
    """Diagnostics returned by one gradient step (all plotted in notebook 03)."""

    loss: float
    q_mean: float
    q_max: float
    target_mean: float
    td_error_abs: float
    grad_norm: float


class DQNAgent:
    """Double / duelling DQN with a target network and uniform replay.

    Parameters
    ----------
    obs_dim, n_actions
        Problem dimensions, normally taken from the environment's spaces.
    cfg
        Hyperparameters.  ``cfg.double_dqn`` and ``cfg.dueling`` are the two
        ablation switches.
    device
        Defaults to CPU.  This network is far too small for a GPU to help --
        kernel-launch overhead exceeds the arithmetic -- so CPU is not a
        limitation, it is the correct choice.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: config.AgentConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or config.DEFAULT.agent
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device)

        torch.manual_seed(self.cfg.seed)

        self.online = QNetwork(
            obs_dim, n_actions, self.cfg.hidden_sizes, self.cfg.dueling
        ).to(self.device)
        self.target = QNetwork(
            obs_dim, n_actions, self.cfg.hidden_sizes, self.cfg.dueling
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()  # never trained directly; only ever copied into

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.cfg.learning_rate
        )
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed=self.cfg.seed)

        # Exploration RNG kept separate from the buffer and environment RNGs so
        # each source of randomness can be varied independently in the ablation.
        self._rng = np.random.default_rng(self.cfg.seed + 10_000)
        self.n_updates = 0

    # -- acting ------------------------------------------------------------ #
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        return self.greedy_action(obs)

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        """Deterministic argmax action -- the policy used for all evaluation."""
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(t).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for one observation, as a NumPy array (for diagnostics)."""
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(t).squeeze(0).cpu().numpy()

    def policy(self):
        """Return a greedy ``env.Policy`` callable for use with ``run_policy``."""
        return lambda obs: self.greedy_action(obs)

    # -- learning ---------------------------------------------------------- #
    def update(self) -> UpdateStats | None:
        """Perform one gradient step; returns ``None`` if the buffer is too small."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.cfg.batch_size, self.device
        )

        # Q(s, a) for the actions actually taken.
        q_all = self.online(obs)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Online net selects, target net evaluates.
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        # Huber loss: quadratic within +/- delta, linear beyond.
        loss = F.smooth_l1_loss(q_taken, target, beta=self.cfg.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clipping bounds the damage a single anomalous batch (a 2008 week
        # sampled several times) can do to the weights.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()

        self.n_updates += 1
        if self.n_updates % self.cfg.target_update_interval == 0:
            self.sync_target()

        with torch.no_grad():
            td_abs = (q_taken - target).abs().mean().item()

        return UpdateStats(
            loss=float(loss.item()),
            q_mean=float(q_all.mean().item()),
            q_max=float(q_all.max().item()),
            target_mean=float(target.mean().item()),
            td_error_abs=float(td_abs),
            grad_norm=float(grad_norm),
        )

    def sync_target(self) -> None:
        """Copy (``tau=1``) or Polyak-average (``tau<1``) online -> target."""
        tau = self.cfg.tau
        if tau >= 1.0:
            self.target.load_state_dict(self.online.state_dict())
            return
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1.0 - tau).add_(tau * op)

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        """Write weights plus the config needed to reconstruct the network.

        Written to a temporary file and then atomically moved into place.  On
        Windows a checkpoint that was just written is frequently still held open
        by the virus scanner, and re-opening the same path a few seconds later
        (this training loop checkpoints ~24 times) fails with
        ``error code: 32``.  Writing a fresh temporary name each time sidesteps
        the contention, and the ``os.replace`` is retried briefly in case the
        destination itself is locked.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "agent_cfg": self.cfg.to_dict(),
            "n_updates": self.n_updates,
        }
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{self.n_updates}.tmp")
        torch.save(blob, tmp)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: config.AgentConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> "DQNAgent":
        """Rebuild an agent from a checkpoint written by :meth:`save`."""
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        if cfg is None:
            saved = dict(blob["agent_cfg"])
            saved["hidden_sizes"] = tuple(saved["hidden_sizes"])
            cfg = config.AgentConfig(**saved)
        agent = cls(blob["obs_dim"], blob["n_actions"], cfg=cfg, device=device)
        agent.online.load_state_dict(blob["online"])
        agent.target.load_state_dict(blob["target"])
        agent.n_updates = int(blob.get("n_updates", 0))
        return agent


def variant_config(
    base: config.AgentConfig | None = None, *, double: bool, dueling: bool, **overrides
) -> config.AgentConfig:
    """Build one row of the ablation grid.

    ``variant_config(double=False, dueling=False)`` is vanilla DQN;
    ``double=True, dueling=True`` is the headline agent.  Everything else is
    held fixed, which is what makes the four runs comparable.
    """
    from dataclasses import replace

    base = base or config.DEFAULT.agent
    return replace(base, double_dqn=double, dueling=dueling, **overrides)


VARIANTS: dict[str, dict[str, bool]] = {
    "Vanilla DQN": {"double": False, "dueling": False},
    "Double DQN": {"double": True, "dueling": False},
    "Dueling DQN": {"double": False, "dueling": True},
    "Double+Dueling DQN": {"double": True, "dueling": True},
}


def describe_agent(agent: DQNAgent) -> str:
    """One-line human-readable description, used in notebook output cells."""
    bits = []
    bits.append("Double" if agent.cfg.double_dqn else "Vanilla")
    if agent.cfg.dueling:
        bits.append("duelling")
    arch = " -> ".join(
        [str(agent.obs_dim), *map(str, agent.cfg.hidden_sizes), str(agent.n_actions)]
    )
    return (
        f"{' + '.join(bits)} DQN | {arch} | "
        f"{agent.online.n_parameters:,} parameters | device={agent.device}"
    )


def save_json(obj, path: str | Path) -> Path:
    """Small helper used by ``train.py`` for run metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
